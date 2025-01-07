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

#include <cstdint>
#include <utility>

#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xls/contrib/mlir/IR/xls_ops.h"
#include "xls/contrib/mlir/transforms/passes.h"  // IWYU pragma: keep

namespace mlir::xls {

#define GEN_PASS_DEF_EXPANDMACROOPSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

class ArrayUpdateSliceOpRewrite : public OpRewritePattern<ArrayUpdateSliceOp> {
 public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ArrayUpdateSliceOp op,
                                PatternRewriter& rewriter) const override {
    Value v = op.getArray();
    Type elementType = op.getType().getElementType();
    for (int64_t i = 0, e = op.getWidth(); i < e; ++i) {
      Value extracted = rewriter.create<ArrayIndexStaticOp>(
          op.getLoc(), elementType, op.getSlice(),
          rewriter.getI64IntegerAttr(i));
      Value index = rewriter.create<AddOp>(
          op.getLoc(), op.getStart(),
          rewriter.create<ConstantScalarOp>(op.getLoc(), rewriter.getI32Type(),
                                            rewriter.getI32IntegerAttr(i)));
      v = rewriter.create<ArrayUpdateOp>(op.getLoc(), v.getType(), v, extracted,
                                         index);
    }
    rewriter.replaceOp(op, v);
    return success();
  }
};

class ExpandMacroOpsPass
    : public impl::ExpandMacroOpsPassBase<ExpandMacroOpsPass> {
 public:
  void runOnOperation() override {
    getOperation()->walk([&](Operation* op) {
      if (auto interface = dyn_cast<XlsRegionOpInterface>(op)) {
        if (interface.isSupportedRegion()) {
          runOnOperation(interface);
          return WalkResult::skip();
        }
      }
      return WalkResult::advance();
    });
  }

  void runOnOperation(XlsRegionOpInterface operation) {
    RewritePatternSet patterns(&getContext());
    patterns.add<ArrayUpdateSliceOpRewrite>(&getContext());
    if (failed(mlir::applyPatternsGreedily(operation, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace mlir::xls
