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

#include "xls/contrib/mlir/transforms/math_to_xls.h"

#include <utility>

#include "mlir/include/mlir/Support/WalkResult.h"
#include "llvm/include/llvm/Support/LogicalResult.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/Builders.h"  // IWYU pragma: keep
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/PatternMatch.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xls/contrib/mlir/IR/xls_ops.h"

namespace mlir::xls {

#define GEN_PASS_DEF_MATHTOXLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {

using ::llvm::SmallVector;

#include "xls/contrib/mlir/transforms/math_to_xls_patterns.inc"

class MathToXlsPass : public impl::MathToXlsPassBase<MathToXlsPass> {
 public:
  void runOnOperation() override;
  LogicalResult initialize(MLIRContext *context) override;

 private:
  FrozenRewritePatternSet patterns;
};

LogicalResult MathToXlsPass::initialize(MLIRContext *context) {
  RewritePatternSet p(context);
  populateMathToXlsConversionPatterns(p);
  patterns = FrozenRewritePatternSet(std::move(p));
  return success();
}

void MathToXlsPass::runOnOperation() {
  auto result = getOperation()->walk([&](Operation *op) {
    if (auto interface = dyn_cast<XlsRegionOpInterface>(op)) {
      if (interface.isSupportedRegion()) {
        if (failed(applyPatternsGreedily(op, patterns))) {
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::skip();
  });
  if (result.wasInterrupted()) {
    signalPassFailure();
  }
}

}  // namespace

void populateMathToXlsConversionPatterns(RewritePatternSet &patterns) {
  populateWithGenerated(patterns);
}

}  // namespace mlir::xls
