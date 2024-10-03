// RUN: xls/contrib/mlir/xls_opt -arith-to-xls -canonicalize %s 2>&1 | FileCheck %s

// CHECK-LABEL: @constants
// CHECK-DAG: arith.constant 1
// CHECK-DAG: arith.constant dense
// CHECK-DAG: xls.constant_tensor
// CHECK-DAG: xls.constant_scalar
// CHECK-DAG: xls.constant_scalar
// CHECK-DAG: xls.constant_tensor
// CHECK-NEXT: return
func.func @constants() -> (tensor<2xindex>, index, tensor<2xi32>, i32, bf16, tensor<2xbf16>) attributes { "xls" = true } {
  %0 = arith.constant dense<42> : tensor<2xindex>
  %1 = arith.constant 1 : index
  %2 = arith.constant dense<1337> : tensor<2xi32>
  %3 = arith.constant 123 : i32
  %4 = arith.constant 1.0 : bf16
  %5 = arith.constant dense<1.0> : tensor<2xbf16>
  return %0, %1, %2, %3, %4, %5 : tensor<2xindex>, index, tensor<2xi32>, i32, bf16, tensor<2xbf16>
}

// CHECK-LABEL: extract_and_addi
// CHECK: xls.add
func.func @extract_and_addi(%arg0: tensor<3x3xi32>, %arg1: i32) -> i32 attributes { "xls" = true } {
  %0 = arith.constant 1 : index
  %1 = arith.constant 2 : index
  %2 = tensor.extract %arg0[%0, %1] : tensor<3x3xi32>
  %3 = arith.addi %2, %2 : i32
  return %3 : i32
}

// CHECK-LABEL: cmp
// CHECK: xls.eq
func.func @cmp(%arg0: tensor<3x3xi32>, %arg1: tensor<3x3xi32>) -> tensor<3x3xi1> attributes { "xls" = true } {
  %0 = arith.cmpi eq, %arg0, %arg1 : tensor<3x3xi32>
  return %0 : tensor<3x3xi1>
}

// CHECK-LABEL: shra
// CHECK: xls.shra
func.func @shra(%arg0: tensor<3x3xi32>, %arg1: tensor<3x3xi32>) -> tensor<3x3xi32> attributes { "xls" = true } {
  %0 = arith.shrsi %arg0, %arg1 : tensor<3x3xi32>
  return %0 : tensor<3x3xi32>
}

// CHECK-LABEL: and
// CHECK: xls.and
func.func @and(%arg0: tensor<3x3xi32>, %arg1: tensor<3x3xi32>) -> tensor<3x3xi32> attributes { "xls" = true } {
  %0 = arith.andi %arg0, %arg1 : tensor<3x3xi32>
  return %0 : tensor<3x3xi32>
}

// CHECK-LABEL: addf
// CHECK: call_dslx
// CHECK-SAME: "add"
func.func @addf(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> attributes { "xls" = true } {
  %0 = arith.addf %arg0, %arg1 : tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: subf
// CHECK: call_dslx
// CHECK-SAME: "sub"
func.func @subf(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> attributes { "xls" = true } {
  %0 = arith.subf %arg0, %arg1 : tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: mulf
// CHECK: call_dslx
// CHECK-SAME: "mul"
func.func @mulf(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> attributes { "xls" = true } {
  %0 = arith.mulf %arg0, %arg1 : tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// TODO(jmolloy): Should be calling "div" but div doesn't exist yet.
// CHECK-LABEL: divf
// CHECK: call_dslx
// CHECK-SAME: "add"
func.func @divf(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> attributes { "xls" = true } {
  %0 = arith.divf %arg0, %arg1 : tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: maxf
// CHECK: call_dslx
// CHECK-SAME: "gt_2"
// CHECK: xls.sel
func.func @maxf(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> attributes { "xls" = true } {
  %0 = arith.maximumf %arg0, %arg1 : tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: minf
// CHECK: call_dslx
// CHECK-SAME: "lt_2"
// CHECK: xls.sel
func.func @minf(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> attributes { "xls" = true } {
  %0 = arith.minimumf %arg0, %arg1 : tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: @ext(
// CHECK: xls.sign_ext
// CHECK: xls.zero_ext
func.func @ext(%arg0: i32) -> (i64, i64) attributes { "xls" = true } {
  %0 = arith.extsi %arg0 : i32 to i64
  %1 = arith.extui %arg0 : i32 to i64
  return %0, %1 : i64, i64
}

// CHECK-LABEL: @extf(
// CHECK: call_dslx
// CHECK-SAME: "ext"
func.func @extf(%arg0: bf16) -> f32 attributes { "xls" = true } {
  %0 = arith.extf %arg0 : bf16 to f32
  return %0 : f32
}

// CHECK-LABEL: @truncf(
// CHECK: call_dslx
// CHECK-SAME: "trunc"
func.func @truncf(%arg0: f32) -> bf16 attributes { "xls" = true } {
  %0 = arith.truncf %arg0 : f32 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @trunci(
// CHECK: xls.bit_slice %{{.*}} {start = 0 : i64, width = 16 : i64} : (i32) -> i16
func.func @trunci(%arg0: i32) -> i16 attributes { "xls" = true } {
  %0 = arith.trunci %arg0 : i32 to i16
  return %0 : i16
}
