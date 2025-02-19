// RUN: xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s

func.func @float_constant() -> bf16 {
  // CHECK: literal(value=(1, 138, 24)
  %0 = arith.constant -2432.0 : bf16
  return %0 : bf16
}
