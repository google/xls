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
#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "llvm/include/llvm/ADT/STLExtras.h"
#include "llvm/include/llvm/Support/Casting.h"
#include "llvm/include/llvm/Support/DebugLog.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/IR/Builders.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/BuiltinTypes.h"
#include "mlir/include/mlir/IR/Diagnostics.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/OwningOpRef.h"
#include "mlir/include/mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Support/LLVM.h"
#include "google/protobuf/text_format.h"
#include "xls/contrib/mlir/IR/xls_ops.h"  // IWYU pragma: keep  // IWYU pragma: keep
#include "xls/contrib/mlir/tools/xls_translate/xls_translate_from_mlir.h"
#include "xls/contrib/mlir/tools/xls_translate/xls_translate_to_mlir.h"
#include "xls/contrib/mlir/transforms/passes.h"  // IWYU pragma: keep
#include "xls/ir/clone_package.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/tools/opt.h"

#define DEBUG_TYPE "optimize-using-xls"

namespace mlir::xls {

#define GEN_PASS_DEF_OPTIMIZEUSINGXLSPASS
#include "xls/contrib/mlir/transforms/passes.h.inc"

namespace {
struct OptimizeUsingXlsPass
    : public mlir::xls::impl::OptimizeUsingXlsPassBase<OptimizeUsingXlsPass> {
  using OptimizeUsingXlsPassBase::OptimizeUsingXlsPassBase;
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

  if (failed(optimizeUsingXls(module, *dslx_cache_, xls_pipeline))) {
    return signalPassFailure();
  }
}

LogicalResult optimizeUsingXls(ModuleOp module, DslxPackageCache& dslx_cache,
                               std::optional<std::string> xls_pipeline,
                               ArrayRef<StringRef> tops) {
  FailureOr<std::unique_ptr<::xls::Package>> package =
      mlirXlsToXls(module, /*dslx_search_path=*/"", dslx_cache);
  if (failed(package)) {
    return failure();
  }

  ::xls::tools::OptOptions opt_options;
  if (xls_pipeline.has_value()) {
    ::xls::PassPipelineProto pass_pipeline;
    if (!google::protobuf::TextFormat::ParseFromString(*xls_pipeline, &pass_pipeline)) {
      return mlir::emitError(module.getLoc())
             << "invalid pass pipeline: " << *xls_pipeline;
    }
    opt_options.pass_pipeline = pass_pipeline;
  }

  auto optimizeForTop =
      [&](::xls::Package* pkg,
          StringRef top) -> FailureOr<OwningOpRef<Operation*>> {
    opt_options.top = top.str();

    if (!xls_pipeline.has_value() || !xls_pipeline->empty()) {
      LDBG() << "Optimizing IR for top: '" << opt_options.top << "' using \n\t"
             << xls_pipeline.value_or("default pipeline");

      absl::Status status = ::xls::tools::OptimizeIrForTop(pkg, opt_options);
      if (!status.ok()) {
        return module.emitError("failed to optimize IR: ") << status.ToString();
      }
    }

    OwningOpRef<Operation*> new_module_op =
        XlsToMlirXlsTranslate(*pkg, module.getContext());
    if (!new_module_op) {
      return module.emitError(
          "failed to translate optimized XLS IR back to MLIR");
    }
    return new_module_op;
  };

  auto updateFunction = [&](OwningOpRef<Operation*>& new_module_op,
                            StringRef top) -> LogicalResult {
    ModuleOp new_module = cast<ModuleOp>(new_module_op.get());
    auto optimized_func = new_module.lookupSymbol<mlir::func::FuncOp>(top);
    if (!optimized_func) {
      return module.emitError("could not find optimized func ") << top;
    }
    auto original_func = module.lookupSymbol<mlir::func::FuncOp>(top);
    // XLS optimization may change both input and output types (e.g. float
    // decomposition into tuples, or packing multiple returns into a single
    // tuple). We accept the optimized types unconditionally; callers are
    // responsible for bridging any type mismatches at their boundaries.
    original_func.getBody().takeBody(optimized_func.getBody());
    original_func.setFunctionType(optimized_func.getFunctionType());
    return success();
  };

  if (tops.empty()) {
    std::string default_top = module.getName().value_or("_package").str();
    auto new_module_op = optimizeForTop((*package).get(), default_top);
    if (failed(new_module_op)) {
      return failure();
    }
    // If no explicit tops were given, preserve the original
    // behavior of replacing the entire module body.
    module.getBodyRegion().takeBody(
        cast<ModuleOp>(new_module_op->get()).getBodyRegion());
  } else {
    for (StringRef top : tops.drop_back()) {
      auto pkg_clone_status = ::xls::ClonePackage(package->get());
      if (!pkg_clone_status.ok()) {
        return module.emitError("failed to clone package: ")
               << pkg_clone_status.status().ToString();
      }
      std::unique_ptr<::xls::Package> pkg_clone =
          std::move(pkg_clone_status).value();

      auto new_module_op = optimizeForTop(pkg_clone.get(), top);
      if (failed(new_module_op) ||
          failed(updateFunction(*new_module_op, top))) {
        return failure();
      }
    }
    // Optimize the last top using the original package rather than a clone.
    auto new_module_op = optimizeForTop((*package).get(), tops.back());
    if (failed(new_module_op)) {
      return failure();
    }
    if (failed(updateFunction(*new_module_op, tops.back()))) {
      return failure();
    }
  }

  return success();
}

}  // namespace mlir::xls
