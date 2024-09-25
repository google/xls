// RUN: xls/contrib/mlir/xls_opt -math-to-xls -canonicalize %s 2>&1 | FileCheck %s

// CHECK-LABEL: exp2
// CHECK: xls.call
func.func @exp2(%arg0: f32) -> f32 attributes { "xls" = true } {
  %0 = math.exp2 %arg0 : f32
  return %0 : f32
}

// CHECK-LABEL: vexp2
// TODO(jpienaar): While this test would pass, the final generated program is
// not correct as it would need to be a vector call.
// CHECK: xls.call
func.func @vexp2(%arg0: tensor<10xf32>) -> tensor<10xf32> attributes { "xls" = true } {
  %0 = math.exp2 %arg0 : tensor<10xf32>
  return %0 : tensor<10xf32>
}
