#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"
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
