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

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/include/mlir/Support/LogicalResult.h"
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/include/mlir/Dialect/Math/IR/Math.h"
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/include/mlir/IR/Dialect.h"  // IWYU pragma: keep
#include "mlir/include/mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"
#include "xls/contrib/mlir/IR/register.h"

int main(int argc, char** argv) {
  mlir::DialectRegistry registry;
  mlir::xls::registerXlsDialect(registry);
  registry.insert<mlir::arith::ArithDialect, mlir::func::FuncDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::math::MathDialect>();
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
