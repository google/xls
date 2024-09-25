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
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "llvm/include/llvm/ADT/APFloat.h"
#include "llvm/include/llvm/ADT/StringExtras.h"
#include "llvm/include/llvm/ADT/StringRef.h"
#include "llvm/include/llvm/ADT/Twine.h"
#include "llvm/include/llvm/Support/CommandLine.h"
#include "llvm/include/llvm/Support/LogicalResult.h"
#include "llvm/include/llvm/Support/raw_ostream.h"
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"
#include "mlir/include/mlir/IR/Value.h"
#include "mlir/include/mlir/IR/Visitors.h"
#include "mlir/include/mlir/InitAllDialects.h"
#include "mlir/include/mlir/InitAllExtensions.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Tools/mlir-translate/MlirTranslateMain.h"
#include "mlir/include/mlir/Tools/mlir-translate/Translation.h"
#include "xls/contrib/mlir/IR/register.h"
#include "xls/contrib/mlir/tools/xls_translate/xls_translate.h"
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

void registerInputDialects(DialectRegistry& registry) {
  // TODO(jpienaar): Registering all as start/prototyping.
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  registerXlsDialect(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::tensor::TensorDialect>();
}

LogicalResult mlirXlsToXlsTranslate(Operation* op, llvm::raw_ostream& output) {
  MlirXlsToXlsTranslateOptions options;
  options.main_function = mainFunction;
  options.optimize_ir = optimizeIr;
  options.dslx_search_path = dslxSearchPath;
  return MlirXlsToXlsTranslate(op, output, options);
}

LogicalResult mlirXlsToVerilogTranslate(Operation* op,
                                        llvm::raw_ostream& output) {
  MlirXlsToXlsTranslateOptions options;
  options.main_function = mainFunction;
  options.optimize_ir = optimizeIr;
  options.dslx_search_path = dslxSearchPath;
  options.generate_verilog = true;
  return MlirXlsToXlsTranslate(op, output, options);
}

TranslateFromMLIRRegistration mlirXlsToXlsTranslateRegistration(
    "mlir-xls-to-xls", "convert from MLIR XLS dialect to XLS",
    mlirXlsToXlsTranslate, registerInputDialects);

TranslateFromMLIRRegistration mlirXlsToVerilogTranslateRegistration(
    "mlir-xls-to-verilog", "convert from MLIR XLS dialect to Verilog",
    mlirXlsToVerilogTranslate, registerInputDialects);

}  // namespace
}  // namespace mlir::xls

int main(int argc, char** argv) {
  xls_init_xls("Initializing XLS", 1, argv);
  return failed(mlir::mlirTranslateMain(argc, argv, "XLS translator\n"));
}
