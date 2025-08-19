// Copyright 2025 The XLS Authors
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

#include <memory>

#include "absl/status/status.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OwningOpRef.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "xls/contrib/mlir/IR/xls_ops.h"  // IWYU pragma: keep
#include "xls/contrib/mlir/tools/xls_translate/xls_translate_from_mlir.h"
#include "xls/contrib/mlir/tools/xls_translate/xls_translate_to_mlir.h"
#include "xls/tools/opt.h"

#define DEBUG_TYPE "optimize-using-xls"

namespace mlir::xls {

#define GEN_PASS_DEF_OPTIMIZEUSINGXLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {
struct OptimizeUsingXlsPass
    : public mlir::xls::impl::OptimizeUsingXlsPassBase<OptimizeUsingXlsPass> {
  void runOnOperation() override;

  LogicalResult initialize(MLIRContext* context) override {
    dslx_cache_ = std::make_shared<DslxPackageCache>();
    return success();
  }

  // This is a shared ptr as this class needs to be copied.
  std::shared_ptr<DslxPackageCache> dslx_cache_;
};
}  // namespace

void OptimizeUsingXlsPass::runOnOperation() {
  ModuleOp module = getOperation();

  FailureOr<std::unique_ptr<::xls::Package>> package =
      mlirXlsToXls(module,
                   /*dslx_search_path=*/"", *dslx_cache_);
  if (failed(package)) {
    return signalPassFailure();
  }

  ::xls::tools::OptOptions opt_options;
  opt_options.top = module.getName().value_or("_package");
  absl::Status status =
      ::xls::tools::OptimizeIrForTop(package->get(), opt_options);
  if (!status.ok()) {
    module.emitError("failed to optimize IR: ") << status.ToString();
    return signalPassFailure();
  }
  OwningOpRef<Operation*> new_module_op =
      XlsToMlirXlsTranslate(*package.value(), &getContext());
  if (!new_module_op) {
    module.emitError("failed to translate optimized XLS IR back to MLIR");
    return signalPassFailure();
  }
  module.getBodyRegion().takeBody(
      cast<ModuleOp>(*new_module_op).getBodyRegion());
}

}  // namespace mlir::xls
