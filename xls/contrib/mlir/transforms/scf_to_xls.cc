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

#include <algorithm>
#include <cstdint>
#include <utility>

#include "llvm/include/llvm/ADT/STLExtras.h"
#include "mlir/include/mlir/Support/WalkResult.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/Support/LogicalResult.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/Matchers.h"
#include "mlir/include/mlir/IR/OpDefinition.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
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

#define GEN_PASS_DEF_SCFTOXLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

using ::llvm::SmallVector;
using ::mlir::scf::IfOp;
using ::mlir::scf::IndexSwitchOp;

LogicalResult rewriteFor(mlir::scf::ForOp op, mlir::PatternRewriter& rewriter) {
  if (auto step = op.getConstantStep(); !step.has_value() || !step->isOne()) {
    if (step.has_value()) {
      return rewriter.notifyMatchFailure(op, "non-one step");
    }
    return rewriter.notifyMatchFailure(op, "non-constant step");
  }
  llvm::APInt lower_bound_attr;
  if (!matchPattern(op.getLowerBound(), m_ConstantInt(&lower_bound_attr))) {
    return rewriter.notifyMatchFailure(op, "non-constant lower bound");
  }

  llvm::APInt upper_bound_attr;
  if (!matchPattern(op.getUpperBound(), m_ConstantInt(&upper_bound_attr))) {
    return rewriter.notifyMatchFailure(op, "non-constant upper bound");
  }

  llvm::APInt trip_count = upper_bound_attr - lower_bound_attr;

  auto invariant_args = mlir::makeRegionIsolatedFromAbove(
      rewriter, op.getBodyRegion(), [](Operation* op) -> bool {
        // Sink anything that can be constant folded.
        llvm::SmallVector<OpFoldResult> results;
        return succeeded(op->fold(results)) && results.size() == 1 &&
               isa<Attribute>(results[0]);
      });

  auto xls_for = ForOp::create(
      rewriter, op->getLoc(), op.getResultTypes(), op.getInitArgs(),
      invariant_args, rewriter.getI64IntegerAttr(trip_count.getLimitedValue()));
  rewriter.inlineRegionBefore(op.getBodyRegion(), xls_for.getBody(),
                              xls_for.getBody().end());
  Operation* terminator = xls_for.getBody().front().getTerminator();
  rewriter.setInsertionPoint(terminator);
  rewriter.replaceOpWithNewOp<YieldOp>(terminator, terminator->getOperands());

  rewriter.replaceOp(op, xls_for);
  return success();
}

class ScfIfOpRewrite : public OpConversionPattern<IfOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IfOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    Block* block = op->getBlock();
    Block* tailBlock = rewriter.splitBlock(block, op->getIterator());
    Operation* thenYield = op.thenYield();
    Operation* elseYield = op.elseBlock() ? op.elseYield() : nullptr;
    rewriter.mergeBlocks(&op.getThenRegion().front(), block);
    if (elseYield) {
      rewriter.mergeBlocks(&op.getElseRegion().front(), block);
    }
    rewriter.mergeBlocks(tailBlock, block);

    if (elseYield) {
      SmallVector<Value> yields;
      for (auto [thenOp, elseOp] :
           llvm::zip(thenYield->getOperands(), elseYield->getOperands())) {
        // SelOp treats condition as a 1-bit index, so zero will return the
        // first variadic operand and one till return the otherwise.
        yields.push_back(SelOp::create(rewriter, op.getLoc(), op.getCondition(),
                                       /*otherwise=*/thenOp,
                                       ValueRange{elseOp}));
      }
      rewriter.replaceOp(op, yields);
    }
    rewriter.eraseOp(thenYield);
    if (elseYield) {
      rewriter.eraseOp(elseYield);
    } else {
      rewriter.eraseOp(op);
    }

    return success();
  }
};

class ScfIndexSwitchOpRewrite : public OpConversionPattern<IndexSwitchOp> {
 public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IndexSwitchOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    (void)adaptor;
    Block* block = op->getBlock();
    Block* tailBlock = rewriter.splitBlock(block, op->getIterator());

    SmallVector<Operation*> yieldOps;
    for (Region& region : op.getCaseRegions()) {
      yieldOps.push_back(region.front().getTerminator());
      rewriter.mergeBlocks(&region.front(), block);
    }
    Operation* defaultYield = op.getDefaultRegion().front().getTerminator();
    rewriter.mergeBlocks(&op.getDefaultRegion().front(), block);
    rewriter.mergeBlocks(tailBlock, block);

    // The logic for how to create an N-way match is taken from DSLX:
    // function_builder.cc:MatchTrue.
    SmallVector<Value> selectors;
    for (int64_t caseValue : op.getCases()) {
      selectors.push_back(EqOp::create(
          rewriter, op.getLoc(), op.getOperand(),
          ConstantScalarOp::create(rewriter, op.getLoc(),
                                   rewriter.getIndexType(),
                                   rewriter.getIndexAttr(caseValue))));
    }
    // Reverse the order of the bits because bit index and indexing of concat
    // elements are reversed.
    std::reverse(selectors.begin(), selectors.end());
    Value concat = ConcatOp::create(rewriter, op.getLoc(),
                                    rewriter.getIntegerType(selectors.size()),
                                    ValueRange(selectors));

    int numOperands = yieldOps.empty() ? 0 : yieldOps[0]->getNumOperands();
    SmallVector<Value> results;
    for (int i = 0; i < numOperands; ++i) {
      SmallVector<Value> operands;
      for (Operation* yieldOp : yieldOps) {
        operands.push_back(yieldOp->getOperand(i));
      }
      results.push_back(
          PrioritySelOp::create(rewriter, op.getLoc(), operands[0].getType(),
                                concat, operands, defaultYield->getOperand(i)));
    }
    for (Operation* yieldOp : yieldOps) {
      rewriter.eraseOp(yieldOp);
    }
    rewriter.eraseOp(defaultYield);
    rewriter.replaceOp(op, results);
    return success();
  }
};

// Rewrites predicates of predicatable operations to be the conjunction of
// all predicates in the region. This makes it safe to if-convert any parent
// IfOps / IndexSwitchOps.
class PredicateRewriter {
 public:
  LogicalResult rewrite(Operation* op) {
    op->walk([&](Operation* op, WalkStage stage) {
      if (auto predOp = dyn_cast<PredicatableOpInterface>(op);
          predOp && stage.isBeforeAllRegions()) {
        visitPredicatableOp(predOp);
      }
      if (!stage.isBeforeAllRegions()) {
        exitRegion(op);
      }
      if (!stage.isAfterAllRegions()) {
        enterRegion(op, stage.getNextRegion());
      }
    });
    return success();
  }

 private:
  Value addCondition(Value condition, Operation* at) {
    if (conditionStack.empty()) {
      conditionStack.push_back(condition);
    } else {
      OpBuilder b(at);
      conditionStack.push_back(
          AndOp::create(b, at->getLoc(), conditionStack.back().getType(),
                        ValueRange{conditionStack.back(), condition}));
    }
    return conditionStack.back();
  }

  Value negateCondition(Operation* op, Value condition) {
    OpBuilder builder(op);
    return NotOp::create(builder, op->getLoc(), condition);
  }

  void popCondition() { conditionStack.pop_back(); }

  void enterRegion(Operation* op, int regionIndex) {
    if (auto ifOp = dyn_cast<IfOp>(op)) {
      if (regionIndex == 0) {
        addCondition(ifOp.getCondition(), op);
      } else {
        addCondition(negateCondition(op, ifOp.getCondition()), op);
      }
    } else if (auto indexSwitchOp = dyn_cast<IndexSwitchOp>(op)) {
      mlir::ImplicitLocOpBuilder builder(op->getLoc(), op);
      if (regionIndex == 0) {
        // Default region.
        SmallVector<Value> cases;
        for (int64_t caseValue : indexSwitchOp.getCases()) {
          cases.push_back(EqOp::create(builder, builder.getI1Type(),
                                       indexSwitchOp.getOperand(),
                                       indexTypedScalar(builder, caseValue)));
        }
        addCondition(
            NotOp::create(builder, OrOp::create(builder, builder.getI1Type(),
                                                ValueRange(cases))),
            op);
      } else {
        Value condition = EqOp::create(
            builder, builder.getI1Type(), indexSwitchOp.getOperand(),
            indexTypedScalar(builder,
                             indexSwitchOp.getCases()[regionIndex - 1]));
        addCondition(condition, op);
      }
    }
  }

  void exitRegion(Operation* op) {
    if (isa<IfOp, IndexSwitchOp>(op)) {
      popCondition();
    }
  }

  void visitPredicatableOp(PredicatableOpInterface op) {
    if (conditionStack.empty()) {
      return;
    }
    if (!op.getCondition()) {
      op.setCondition(conditionStack.back());
    } else {
      op.setCondition(addCondition(op.getCondition(), op));
      popCondition();
    }
  }

  Value indexTypedScalar(mlir::ImplicitLocOpBuilder builder, int64_t value) {
    return ConstantScalarOp::create(builder, builder.getIndexType(),
                                    builder.getIndexAttr(value));
  }

  SmallVector<Value> conditionStack;
};

class ScfToXlsConversionPass
    : public impl::ScfToXlsPassBase<ScfToXlsConversionPass> {
 public:
  void runOnOperation() override {
    getOperation()->walk([&](XlsRegionOpInterface interface) {
      if (interface.isSupportedRegion()) {
        runOnOperation(interface);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
  }

  void runOnOperation(XlsRegionOpInterface interface) {
    // Rewrite nested predicates first.
    PredicateRewriter predicateRewriter;
    if (failed(predicateRewriter.rewrite(interface))) {
      signalPassFailure();
    }

    // Start by converting IfsOp to SelOps.
    ConversionTarget target(getContext());
    target.markUnknownOpDynamicallyLegal([](Operation*) { return true; });
    target.addIllegalOp<IfOp, IndexSwitchOp>();
    RewritePatternSet patterns(&getContext());
    patterns.add<ScfIfOpRewrite, ScfIndexSwitchOpRewrite>(&getContext());
    if (failed(mlir::applyPartialConversion(interface, target,
                                            std::move(patterns)))) {
      signalPassFailure();
    }

    // Then convert ForOps to CountedForOps. These need to be done in postorder
    // to ensure that isolating regions from above works transitively.
    interface->walk([&](mlir::scf::ForOp op) {
      mlir::PatternRewriter rewriter(op->getContext());
      rewriter.setInsertionPoint(op);
      LogicalResult result = rewriteFor(op, rewriter);
      return failed(result) ? WalkResult::interrupt() : WalkResult::advance();
    });
  }
};

}  // namespace
}  // namespace mlir::xls
