// RUN: xls_opt -arith-to-xls -canonicalize %s 2>&1 | FileCheck %s

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

// CHECK-LABEL: maxf
// CHECK: call_dslx
// CHECK-SAME: "gt_2"
// CHECK-SAME: -> tensor<3x3xi1>
// CHECK: xls.sel %[[_:.*]] in [%arg1, %arg0]
// CHECK-SAME: -> tensor<3x3xf32>
func.func @maxf(%arg0: tensor<3x3xf32>, %arg1: tensor<3x3xf32>) -> tensor<3x3xf32> attributes { "xls" = true } {
  %0 = arith.maximumf %arg0, %arg1 : tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
}

// CHECK-LABEL: minf
// CHECK: call_dslx
// CHECK-SAME: "lt_2"
// CHECK: xls.sel %[[_:.*]] in [%arg1, %arg0]
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
func.func @extf(%arg0: tensor<3x3xbf16>) -> tensor<3x3xf32> attributes { "xls" = true } {
  %0 = arith.extf %arg0 : tensor<3x3xbf16> to tensor<3x3xf32>
  return %0 : tensor<3x3xf32>
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

// CHECK-LABEL: @sitofp32
// CHECK: xls.call_dslx
// CHECK-SAME: from_int32
// CHECK-SAME: (i32) -> f32
func.func @sitofp32(%arg0: i32) -> f32 attributes { "xls" = true } {
  %0 = arith.sitofp %arg0 : i32 to f32
  return %0 : f32
}

// CHECK-LABEL: @sitofp16
// CHECK: xls.call_dslx
// CHECK-SAME: from_int8
// CHECK-SAME: (i8) -> bf16
func.func @sitofp16(%arg0: i8) -> bf16 attributes { "xls" = true } {
  %0 = arith.sitofp %arg0 : i8 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @si32tofp16
// CHECK: xls.call_dslx
// CHECK-SAME: from_int32
// CHECK: xls.call_dslx
// CHECK-SAME: from_float32
func.func @si32tofp16(%arg0: i32) -> bf16 attributes { "xls" = true } {
  %0 = arith.sitofp %arg0 : i32 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @uitofp32
// CHECK: xls.call_dslx
// CHECK-SAME: from_uint32
// CHECK-SAME: (i32) -> f32
func.func @uitofp32(%arg0: i32) -> f32 attributes { "xls" = true } {
  %0 = arith.uitofp %arg0 : i32 to f32
  return %0 : f32
}

// CHECK-LABEL: @uitofp16
// CHECK: xls.call_dslx
// CHECK-SAME: from_uint8
// CHECK-SAME: (i8) -> bf16
func.func @uitofp16(%arg0: i8) -> bf16 attributes { "xls" = true } {
  %0 = arith.uitofp %arg0 : i8 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @ui32tofp16
// CHECK: xls.call_dslx
// CHECK-SAME: from_uint32
// CHECK: xls.call_dslx
// CHECK-SAME: from_float32
func.func @ui32tofp16(%arg0: i32) -> bf16 attributes { "xls" = true } {
  %0 = arith.uitofp %arg0 : i32 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @eq
// CHECK:  xls.call_dslx
// CHECK-SAME: eq_2
// CHECK-SAME: (f32, f32) -> i1
func.func @eq(%arg0: f32, %arg1: f32) -> i1 attributes { "xls" = true } {
  %0 = arith.cmpf oeq, %arg0, %arg1 : f32
  return %0 : i1
}

// CHECK-LABEL: @ne
// CHECK: xls.call_dslx
// CHECK-SAME: eq_2
// CHECK-SAME: (f32, f32) -> i1
// CHECK-NEXT: xls.xor
func.func @ne(%arg0: f32, %arg1: f32) -> i1 attributes { "xls" = true } {
  %0 = arith.cmpf une, %arg0, %arg1 : f32
  return %0 : i1
}

// CHECK-LABEL: @uno
// CHECK: xls.call_dslx
// CHECK-SAME: add
// CHECK: xls.call_dslx
// CHECK-SAME: is_nan
func.func @uno(%arg0: f32, %arg1: f32) -> i1 attributes { "xls" = true } {
  %0 = arith.cmpf uno, %arg0, %arg1 : f32
  return %0 : i1
}

// CHECK-LABEL: @fptosi32
// CHECK: xls.call_dslx
// CHECK-SAME: to_int32
// CHECK-SAME: (f32) -> i32
func.func @fptosi32(%arg0: f32) -> i32 attributes { "xls" = true } {
  %0 = arith.fptosi %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: @fptosi16
// CHECK: xls.call_dslx
// CHECK-SAME: to_int16
// CHECK-SAME: (bf16) -> i16
func.func @fptosi16(%arg0: bf16) -> i16 attributes { "xls" = true } {
  %0 = arith.fptosi %arg0 : bf16 to i16
  return %0 : i16
}

// CHECK-LABEL: @fptosi8
// CHECK: xls.call_dslx
// CHECK-SAME: to_int16
// CHECK-SAME: (bf16) -> i16
// CHECK-NEXT: xls.bit_slice %0 {start = 0 : i64, width = 8 : i64} : (i16) -> i8
func.func @fptosi8(%arg0: bf16) -> i8 attributes { "xls" = true } {
  %0 = arith.fptosi %arg0 : bf16 to i8
  return %0 : i8
}

// CHECK-LABEL: @fptoui32
// CHECK: xls.call_dslx
// CHECK-SAME: to_uint32
// CHECK-SAME: (f32) -> i32
func.func @fptoui32(%arg0: f32) -> i32 attributes { "xls" = true } {
  %0 = arith.fptoui %arg0 : f32 to i32
  return %0 : i32
}

// CHECK-LABEL: @fptoui16
// CHECK: xls.call_dslx
// CHECK-SAME: to_uint16
// CHECK-SAME: (bf16) -> i16
func.func @fptoui16(%arg0: bf16) -> i16 attributes { "xls" = true } {
  %0 = arith.fptoui %arg0 : bf16 to i16
  return %0 : i16
}

// CHECK-LABEL: @fptoui8
// CHECK: xls.call_dslx
// CHECK-SAME: to_uint16
// CHECK-SAME: (bf16) -> i16
// CHECK-NEXT: xls.bit_slice %0 {start = 0 : i64, width = 8 : i64} : (i16) -> i8
func.func @fptoui8(%arg0: bf16) -> i8 attributes { "xls" = true } {
  %0 = arith.fptoui %arg0 : bf16 to i8
  return %0 : i8
}

// CHECK-LABEL: @fptoui64
// CHECK: xls.call_dslx
// CHECK-SAME: to_uint32
// CHECK-SAME: (f32) -> i32
// CHECK-NEXT: xls.zero_ext %0 : (i32) -> i64
func.func @fptoui64(%arg0: f32) -> i64 attributes { "xls" = true } {
  %0 = arith.fptoui %arg0 : f32 to i64
  return %0 : i64
}

// CHECK-LABEL: @fp16toui64
// CHECK: xls.call_dslx
// CHECK-SAME: to_uint16
// CHECK-SAME: (bf16) -> i16
// CHECK-NEXT: xls.zero_ext %0 : (i16) -> i64
func.func @fp16toui64(%arg0: bf16) -> i64 attributes { "xls" = true } {
  %0 = arith.fptoui %arg0 : bf16 to i64
  return %0 : i64
}

// CHECK-LABEL: @ui64tofp16
// CHECK: xls.bit_slice
// CHECK-NEXT: xls.call_dslx
// CHECK-SAME: from_uint32
// CHECK-NEXT: xls.call_dslx
// CHECK-SAME: from_float32
// CHECK-SAME: (f32) -> bf16
func.func @ui64tofp16(%arg0: i64) -> bf16 attributes { "xls" = true } {
  %0 = arith.uitofp %arg0 : i64 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @si4tofp16
// CHECK: xls.call_dslx
// CHECK-SAME: from_int32
// CHECK-NEXT: from_float32
func.func @si4tofp16(%arg0: i4) -> bf16 attributes { "xls" = true } {
  %0 = arith.sitofp %arg0 : i4 to bf16
  return %0 : bf16
}

// CHECK-LABEL: @fp16tosi4
// CHECK: xls.call_dslx
// CHECK-SAME: to_int16
// CHECK-NEXT: xls.bit_slice %0 {start = 0 : i64, width = 4 : i64} : (i16) -> i4
func.func @fp16tosi4(%arg0: bf16) -> i4 attributes { "xls" = true } {
  %0 = arith.fptosi %arg0 : bf16 to i4
  return %0 : i4
}

// CHECK-LABEL: @fp32toui4
// CHECK: xls.call_dslx
// CHECK-SAME: to_uint32
// CHECK-NEXT: xls.bit_slice %0 {start = 0 : i64, width = 4 : i64} : (i32) -> i4
func.func @fp32toui4(%arg0: f32) -> i4 attributes { "xls" = true } {
  %0 = arith.fptoui %arg0 : f32 to i4
  return %0 : i4
}

// CHECK-LABEL: @fptosi64
// CHECK: xls.call_dslx
// CHECK-SAME: to_int32
// CHECK-SAME: (f32) -> i32
// CHECK-NEXT: xls.sign_ext %0 : (i32) -> i64
func.func @fptosi64(%arg0: f32) -> i64 attributes { "xls" = true } {
  %0 = arith.fptosi %arg0 : f32 to i64
  return %0 : i64
}

// CHECK-LABEL: @negate
// CHECK: xls.call_dslx
// CHECK-SAME: neg
func.func @negate(%arg0: bf16) -> bf16 attributes { "xls" = true } {
  %0 = arith.negf %arg0 : bf16
  return %0 : bf16
}

// Check that we can convert an sproc's next region too.
// CHECK-LABEL: @sproc
// CHECK: next (
// CHECK-NEXT: %[[X:.*]] = xls.add
// CHECK-NEXT: xls.yield %[[X]] : i32
xls.sproc @sproc(%arg0: !xls.schan<i32, in>) top {
  spawns {
    xls.yield
  }
  next(%state: i32) zeroinitializer {
    %0 = arith.addi %state, %state : i32
    xls.yield %0 : i32
  }
}
