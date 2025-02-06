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

#include <string>

#include "llvm/include/llvm/ADT/APFloat.h"
#include "llvm/include/llvm/ADT/StringExtras.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/ADT/Twine.h"
#include "llvm/include/llvm/Support/CommandLine.h"
#include "llvm/include/llvm/Support/LogicalResult.h"
#include "llvm/include/llvm/Support/SourceMgr.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/BuiltinOps.h"
#include "mlir/include/mlir/IR/MLIRContext.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"
#include "xls/codegen/xls_metrics.pb.h"
#include "xls/contrib/mlir/IR/register.h"
#include "xls/contrib/mlir/tools/xls_translate/xls_stitch.h"
#include "xls/contrib/mlir/tools/xls_translate/xls_translate_from_mlir.h"
#include "xls/contrib/mlir/tools/xls_translate/xls_translate_to_mlir.h"
#include "xls/public/c_api.h"

namespace mlir::xls {
namespace {

// NOLINTNEXTLINE
llvm::cl::opt<std::string> mainFunction("main-function",
                                        llvm::cl::desc("Main function"),
                                        llvm::cl::init(""));

// NOLINTNEXTLINE
llvm::cl::opt<bool> optimizeIr(
    "optimize-ir", llvm::cl::desc("Whether to optimize IR post translation"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<std::string> dslxSearchPath(
    "dslx-search-path", llvm::cl::desc("Search path for DSLX files"),
    llvm::cl::init(""));

// NOLINTNEXTLINE
llvm::cl::opt<bool> privatizeAndDceFunctions(
    "privatize-and-dce-functions",
    llvm::cl::desc("Whether to privatize all non-top functions and run "
                   "SymbolDCE first"),
    llvm::cl::init(false));

// NOLINTNEXTLINE
llvm::cl::opt<bool> dumpCodegenMetrics(
    "dump-codegen-metrics",
    llvm::cl::desc("Whether to dump XLS codegen metric"),
    llvm::cl::init(false));

void registerInputDialects(DialectRegistry& registry) {
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect>();
  registerXlsDialect(registry);
}

static void printCodegenMetrics(
    const ::xls::Package& package,
    const ::xls::verilog::BlockMetricsProto& metrics) {
  llvm::errs() << "Generated XLS metrics:\n";
  llvm::errs() << "  xls_node count: " << package.GetFunctionNodeCount()
               << "\n";
  llvm::errs() << "  xls_flop count: " << metrics.flop_count() << "\n";
  llvm::errs() << "  xls_max_input_to_reg_delay_ps: "
               << metrics.max_input_to_reg_delay_ps() << "\n";
  llvm::errs() << "  xls_max_reg_to_reg_delay_ps: "
               << metrics.max_reg_to_reg_delay_ps() << "\n";
  llvm::errs() << "  xls_max_reg_to_output_delay_ps: "
               << metrics.max_reg_to_output_delay_ps() << "\n";
  llvm::errs() << "  xls_max_feedthrough_path_delay_ps: "
               << metrics.max_feedthrough_path_delay_ps() << "\n";
}

LogicalResult mlirXlsToXlsTranslate(Operation* op, llvm::raw_ostream& output) {
  MlirXlsToXlsTranslateOptions options;
  options.main_function = mainFunction;
  options.privatize_and_dce_functions = privatizeAndDceFunctions;
  options.optimize_ir = optimizeIr;
  options.dslx_search_path = dslxSearchPath;
  return MlirXlsToXlsTranslate(op, output, options);
}

OwningOpRef<Operation*> xlsToMlirXlsTranslate(llvm::SourceMgr& mgr,
                                              MLIRContext* ctx) {
  return XlsToMlirXlsTranslate(mgr, ctx);
}

LogicalResult mlirXlsToVerilogTranslate(Operation* op,
                                        llvm::raw_ostream& output) {
  MlirXlsToXlsTranslateOptions options;
  options.main_function = mainFunction;
  options.privatize_and_dce_functions = privatizeAndDceFunctions;
  options.optimize_ir = optimizeIr;
  options.dslx_search_path = dslxSearchPath;
  options.generate_verilog = true;
  if (dumpCodegenMetrics) {
    return MlirXlsToXlsTranslate(op, output, options, printCodegenMetrics);
  }
  return MlirXlsToXlsTranslate(op, output, options);
}

LogicalResult mlirXlsStitch(Operation* op, llvm::raw_ostream& output) {
  XlsStitchOptions options;
  return XlsStitch(cast<ModuleOp>(op), output, options);
}

TranslateFromMLIRRegistration mlirXlsToXlsTranslateRegistration(
    "mlir-xls-to-xls", "convert from MLIR XLS dialect to XLS",
    mlirXlsToXlsTranslate, registerInputDialects);

TranslateToMLIRRegistration XlsToMlirXlsTranslateRegistration(
    "xls-to-mlir-xls", "convert from XLS to MLIR XLS dialect",
    xlsToMlirXlsTranslate, registerInputDialects);

TranslateFromMLIRRegistration mlirXlsToVerilogTranslateRegistration(
    "mlir-xls-to-verilog", "convert from MLIR XLS dialect to Verilog",
    mlirXlsToVerilogTranslate, registerInputDialects);

TranslateFromMLIRRegistration mlirXlsStitchRegistration(
    "mlir-xls-stitch", "stitch together XLS modules", mlirXlsStitch,
    registerInputDialects);

}  // namespace
}  // namespace mlir::xls

int main(int argc, char** argv) {
  // We allow ABSL flags to be passed to this binary after a double-dash:
  // xls_translate ... -- --alsologtostderr
  char** mlir_argv = argv;
  char** absl_argv = argv;
  int mlir_argc = argc, absl_argc = 1;
  for (int i = 0; i < argc; ++i) {
    if (std::string(argv[i]) == std::string("--")) {
      // -- found; split into MLIR and ABSL args.
      absl_argv = &argv[i];  // -- becomes argv[0] for absl.
      mlir_argc = i;
      absl_argc = argc - i;
      break;
    }
  }
  xls_init_xls("Initializing XLS", absl_argc, absl_argv);

  return failed(
      mlir::mlirTranslateMain(mlir_argc, mlir_argv, "XLS translator\n"));
}
