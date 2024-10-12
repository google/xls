// Copyright 2024 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cassert>
#include <cstdint>
#include <string>
#include <string_view>
#include <utility>

// Some of these need the keep IWYU pragma as they are required by *.inc files

#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/TypeRange.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "mlir/include/mlir/Transforms/FoldUtils.h"
#include "mlir/include/mlir/Transforms/RegionUtils.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {

#define GEN_PASS_DEF_LOWERCOUNTEDFORPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"  // IWYU pragma: keep

namespace {
using ::llvm::SmallVector;

namespace fixed {
// TODO(jmolloy): This is a copy of the one in SCF utils. But that version is
// hardcoded to assume the region comes from a FuncOp, whereas this one just
// looks for the SymbolOpInterface, so it works with EprocOps too.
FailureOr<func::FuncOp> outlineSingleBlockRegion(RewriterBase &rewriter,
                                                 Location loc, Region &region,
                                                 StringRef funcName,
                                                 func::CallOp *callOp) {
  assert(!funcName.empty() && "funcName cannot be empty");
  if (!region.hasOneBlock()) {
    return failure();
  }

  Block *originalBlock = &region.front();
  Operation *originalTerminator = originalBlock->getTerminator();

  // Outline before current function.
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(region.getParentOfType<SymbolOpInterface>());

  SetVector<Value> captures;
  mlir::getUsedValuesDefinedAbove(region, captures);

  ValueRange outlinedValues(captures.getArrayRef());
  SmallVector<Type> outlinedFuncArgTypes;
  SmallVector<Location> outlinedFuncArgLocs;
  // Region's arguments are exactly the first block's arguments as per
  // Region::getArguments().
  // Func's arguments are cat(regions's arguments, captures arguments).
  for (BlockArgument arg : region.getArguments()) {
    outlinedFuncArgTypes.push_back(arg.getType());
    outlinedFuncArgLocs.push_back(arg.getLoc());
  }
  for (Value value : outlinedValues) {
    outlinedFuncArgTypes.push_back(value.getType());
    outlinedFuncArgLocs.push_back(value.getLoc());
  }
  FunctionType outlinedFuncType =
      FunctionType::get(rewriter.getContext(), outlinedFuncArgTypes,
                        originalTerminator->getOperandTypes());
  auto outlinedFunc =
      rewriter.create<func::FuncOp>(loc, funcName, outlinedFuncType);
  Block *outlinedFuncBody = outlinedFunc.addEntryBlock();

  // Merge blocks while replacing the original block operands.
  // Warning: `mergeBlocks` erases the original block, reconstruct it later.
  int64_t numOriginalBlockArguments = originalBlock->getNumArguments();
  auto outlinedFuncBlockArgs = outlinedFuncBody->getArguments();
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(outlinedFuncBody);
    rewriter.mergeBlocks(
        originalBlock, outlinedFuncBody,
        outlinedFuncBlockArgs.take_front(numOriginalBlockArguments));
    // Explicitly set up a new ReturnOp terminator.
    rewriter.setInsertionPointToEnd(outlinedFuncBody);
    rewriter.create<func::ReturnOp>(loc, originalTerminator->getResultTypes(),
                                    originalTerminator->getOperands());
  }

  // Reconstruct the block that was deleted and add a
  // terminator(call_results).
  Block *newBlock = rewriter.createBlock(
      &region, region.begin(),
      TypeRange{outlinedFuncArgTypes}.take_front(numOriginalBlockArguments),
      ArrayRef<Location>(outlinedFuncArgLocs)
          .take_front(numOriginalBlockArguments));
  {
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(newBlock);
    SmallVector<Value> callValues;
    llvm::append_range(callValues, newBlock->getArguments());
    llvm::append_range(callValues, outlinedValues);
    auto call = rewriter.create<func::CallOp>(loc, outlinedFunc, callValues);
    if (callOp) {
      *callOp = call;
    }

    // `originalTerminator` was moved to `outlinedFuncBody` and is still valid.
    // Clone `originalTerminator` to take the callOp results then erase it from
    // `outlinedFuncBody`.
    IRMapping bvm;
    bvm.map(originalTerminator->getOperands(), call->getResults());
    rewriter.clone(*originalTerminator, bvm);
    rewriter.eraseOp(originalTerminator);
  }

  // Lastly, explicit RAUW outlinedValues, only for uses within `outlinedFunc`.
  // Clone the `arith::ConstantIndexOp` at the start of `outlinedFuncBody`.
  for (auto it : llvm::zip(outlinedValues, outlinedFuncBlockArgs.take_back(
                                               outlinedValues.size()))) {
    Value orig = std::get<0>(it);
    Value repl = std::get<1>(it);
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToStart(outlinedFuncBody);
      if (Operation *cst = orig.getDefiningOp<arith::ConstantIndexOp>()) {
        IRMapping bvm;
        repl = rewriter.clone(*cst, bvm)->getResult(0);
      }
    }
    orig.replaceUsesWithIf(repl, [&](OpOperand &opOperand) {
      return outlinedFunc->isProperAncestor(opOperand.getOwner());
    });
  }

  return outlinedFunc;
}
}  // namespace fixed

// Attribute name for the preferred name of the function.
constexpr std::string_view kPreferredNameAttr = "_preferred_name";

// Rewrites multiple loop-carried values to be a single Tuple. After this
// rewrite there is only a single loop-carried value.
class TuplifyRewrite : public OpConversionPattern<ForOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ForOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInits().size() == 1) {
      return failure();
    }
    // Tuple up the inits.
    auto tupleOp = rewriter.create<TupleOp>(op.getLoc(), op.getInits());

    auto forOp = rewriter.create<ForOp>(
        op->getLoc(), tupleOp.getType(), tupleOp->getResults(),
        op.getInvariants(), op.getTripCountAttr());
    forOp->setAttr(kPreferredNameAttr, op->getAttr(kPreferredNameAttr));

    SmallVector<Value> results;
    for (auto [i, result] : llvm::enumerate(op->getResults())) {
      results.push_back(rewriter.create<TupleIndexOp>(
          op->getLoc(), result.getType(), forOp.getResult(0), i));
    }

    // Modify the body to use the tuple.
    mlir::IRMapping mapping;
    auto oldArguments = op.getBody().getArguments();
    Block *block = &forOp.getBody().emplaceBlock();
    rewriter.setInsertionPointToStart(block);
    mapping.map(
        oldArguments.front(),
        block->addArgument(oldArguments.front().getType(), op.getLoc()));

    BlockArgument newCarry = block->addArgument(tupleOp.getType(), op.getLoc());
    for (auto [i, oldArg] :
         llvm::enumerate(oldArguments.slice(1, op.getInits().size()))) {
      auto tupleIndexOp = rewriter.create<TupleIndexOp>(
          op->getLoc(), oldArg.getType(), newCarry, i);
      mapping.map(oldArg, tupleIndexOp);
    }
    for (auto &invariant : oldArguments.take_back(op.getInvariants().size())) {
      mapping.map(invariant,
                  block->addArgument(invariant.getType(), op.getLoc()));
    }
    rewriter.cloneRegionBefore(op.getBodyRegion(), forOp.getBodyRegion(),
                               forOp.getBodyRegion().end(), mapping);
    assert(forOp.getBodyRegion().getBlocks().size() == 2 &&
           "Expected two blocks in the body region");
    assert(forOp.getBodyRegion().getBlocks().back().getNumArguments() == 0 &&
           "Expected no arguments in the second block");
    rewriter.mergeBlocks(&forOp.getBody().back(), &forOp.getBody().front());

    rewriter.setInsertionPoint(block->getTerminator());
    auto tupleOpForYield = rewriter.create<TupleOp>(
        op->getLoc(), block->getTerminator()->getOperands());
    block->getTerminator()->setOperands(tupleOpForYield->getResults());

    rewriter.replaceOp(op, results);
    return success();
  }
};

StringAttr createUniqueName(MLIRContext &context, SymbolTable &symbolTable,
                            DenseSet<StringRef> &addedSymbols,
                            StringRef prefix) {
  if (symbolTable.lookup(prefix) == nullptr && !addedSymbols.contains(prefix)) {
    addedSymbols.insert(prefix);
    return StringAttr::get(&context, prefix);
  }

  unsigned uniquingCounter = 0;
  llvm::SmallString<128> name = SymbolTable::generateSymbolName<128>(
      prefix,
      [&](llvm::StringRef candidate) {
        return symbolTable.lookup(candidate) ||
               addedSymbols.contains(candidate.str());
      },
      uniquingCounter);
  auto result = StringAttr::get(&context, name);
  addedSymbols.insert(result);
  return result;
}

class ForToCountedForRewrite : public OpConversionPattern<ForOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ForOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (adaptor.getInits().size() != 1) {
      // Needs to be tuplified first.
      return failure();
    }

    // Fetch the preferred name; we did a single pass over the module and
    // stashed the result in an attribute. This is important because the
    // rewriter will defer creation of the new FuncOp, so the symbol table will
    // not be updated until after rewrites have completed (meaning
    // createUniqueName would always return the same value in the same rewrite
    // cycle causing clashes).
    std::string name = cast<StringAttr>(op->getAttr(kPreferredNameAttr)).str();

    mlir::func::CallOp callOp;
    auto func = fixed::outlineSingleBlockRegion(
        rewriter, op->getLoc(), op.getBodyRegion(), name, &callOp);
    if (failed(func)) {
      return rewriter.notifyMatchFailure(op, "Failed to outline body");
    }
    (*func)->setAttr("xls", rewriter.getBoolAttr(true));
    (*func).setPrivate();
    rewriter.replaceOpWithNewOp<CountedForOp>(
        op, op->getResultTypes(), adaptor.getInits().front(),
        adaptor.getInvariants(), adaptor.getTripCountAttr(),
        SymbolRefAttr::get(*func),
        /*stride=*/rewriter.getI64IntegerAttr(1));
    return success();
  }
};

class LowerCountedForPass
    : public impl::LowerCountedForPassBase<LowerCountedForPass> {
 private:
  void runOnOperation() override {
    // See comment in ForToCountedForRewrite for why we do this.
    DenseSet<StringRef> addedSymbols;
    SymbolTable symbolTable(getOperation());
    getOperation().walk([&](ForOp op) {
      op->setAttr(kPreferredNameAttr,
                  createUniqueName(getContext(), symbolTable, addedSymbols,
                                   "for_body"));
    });

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
    target.addIllegalOp<ForOp>();
    RewritePatternSet patterns(&getContext());
    patterns.add<TuplifyRewrite, ForToCountedForRewrite>(&getContext());
    if (failed(mlir::applyPartialConversion(getOperation(), target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::xls
