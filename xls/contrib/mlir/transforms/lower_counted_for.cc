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
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/Support/LogicalResult.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/SymbolTable.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/ValueRange.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"
#include "mlir/include/mlir/Pass/PassRegistry.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Transforms/DialectConversion.h"
#include "mlir/include/mlir/Transforms/FoldUtils.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {

#define GEN_PASS_DEF_LOWERCOUNTEDFORPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

using ::llvm::SmallVector;

namespace {

// Attribute name for the preferred name of the function.
constexpr std::string_view kPreferredNameAttr = "_preferred_name";

// Rewrites multiple loop-carried values to be a single Tuple. After this
// rewrite there is only a single loop-carried value.
class TuplifyRewrite : public OpConversionPattern<ForOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ForOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
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
    Block* block = &forOp.getBody().emplaceBlock();
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
    for (auto& invariant : oldArguments.take_back(op.getInvariants().size())) {
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

std::string createUniqueName(Operation* op, std::string prefix) {
  // TODO(jpienaar): This could be made more efficient. Current approach does
  // work that could be cached and reused.
  mlir::Operation* symbolTableOp =
      op->getParentWithTrait<mlir::OpTrait::SymbolTable>();
  if (mlir::SymbolTable::lookupSymbolIn(symbolTableOp, prefix) == nullptr) {
    return prefix;
  }

  unsigned uniquingCounter = 0;
  llvm::SmallString<128> name = SymbolTable::generateSymbolName<128>(
      prefix,
      [&](llvm::StringRef candidate) {
        return mlir::SymbolTable::lookupSymbolIn(symbolTableOp, candidate) !=
               nullptr;
      },
      uniquingCounter);
  return std::string(name.str());
}

class ForToCountedForRewrite : public OpConversionPattern<ForOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ForOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
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
    std::string preferredName =
        cast<StringAttr>(op->getAttr(kPreferredNameAttr)).str();
    std::string name = createUniqueName(op, preferredName);

    mlir::func::CallOp callOp;
    auto func = mlir::outlineSingleBlockRegion(
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
    getOperation().walk([&](ForOp op) {
      op->setAttr(
          kPreferredNameAttr,
          StringAttr::get(op->getContext(), createUniqueName(op, "for_body")));
    });

    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
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
