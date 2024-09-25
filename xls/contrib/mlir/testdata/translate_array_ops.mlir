// RUN: xls/contrib/mlir/xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s --dump-input-filter=all --check-prefix=XLS

func.func private @ggenerated(%arg1: !xls.array<2 x i4>) -> !xls.array<2 x i4> {
      %1 = "xls.constant_scalar"() <{value = 3 : index}> : () -> i32
      %2 = "xls.constant_scalar"() <{value = 2 : index}> : () -> i32
      %3 = "xls.constant_scalar"() <{value = -2 : i4}> : () -> i4
      %4 = "xls.constant_scalar"() <{value = -7 : i4}> : () -> i4
      %5 = "xls.constant_scalar"() <{value = -3 : i4}> : () -> i4
      %6 = "xls.array_zero"() : () -> !xls.array<2 x i4>
      %7 = "xls.array_index_static"(%arg1) <{index = 2 : i64}> : (!xls.array<2 x i4>) -> i4
      %8 = "xls.array_index_static"(%arg1) <{index = 3 : i64}> : (!xls.array<2 x i4>) -> i4
      %9 = xls.add %7, %8 : i4
      %10 = xls.umul %9, %5 : i4
      %11 = xls.umul %7, %4 : i4
      %12 = xls.umul %8, %3 : i4
      %13 = xls.add %12, %11 : i4
      %14 = "xls.array_update"(%6, %10, %2) : (!xls.array<2 x i4>, i4, i32) -> !xls.array<2 x i4>
      %15 = "xls.array_update"(%14, %13, %1) : (!xls.array<2 x i4>, i4, i32) -> !xls.array<2 x i4>
  return %15 : !xls.array<2 x i4>
}

// XLS-LABEL: fn array
// XLS-SAME: ([[ARG0:.*]]: bits[32]{{.*}}, [[ARG1:.*]]: bits[32]{{.*}}) -> bits[32][2]
// XLS: ret {{.*}}: bits[32][2] = array([[ARG0]], [[ARG1]]
func.func private @array(%arg0: i32 loc("a"), %arg1: i32 loc("b")) -> !xls.array<2 x i32> {
  %0 = xls.array %arg0, %arg1 : (i32, i32) -> !xls.array<2 x i32> loc("output")
  return %0 : !xls.array<2 x i32>
}

// XLS-LABEL: array_index
func.func private @array_index(%arg0: !xls.array<4 x i8>, %arg1: i32) -> i8 {
  // XLA: = array_index
  %0 = "xls.array_index"(%arg0, %arg1) : (!xls.array<4 x i8>, i32) -> i8
  return %0 : i8
}

// XLS-LABEL: array_slice
func.func private @array_slice(%arg0: !xls.array<4 x i8>, %arg1: i32) -> !xls.array<1 x i8> {
  // XLS: = array_slice
  %0 = "xls.array_slice"(%arg0, %arg1) { width = 1 : i64 } : (!xls.array<4 x i8>, i32) -> !xls.array<1 x i8>
  return %0 : !xls.array<1 x i8>
}

// XLS-LABEL: array_update
func.func private @array_update(%arg0: !xls.array<4 x i8>, %arg1: i8, %arg2: i32) -> !xls.array<4 x i8> {
  // XLS: = array_update
  %0 = "xls.array_update"(%arg0, %arg1, %arg2) : (!xls.array<4 x i8>, i8, i32) -> !xls.array<4 x i8>
  return %0 : !xls.array<4 x i8>
}

// XLS-LABEL: array_zero
func.func private @array_zero() -> !xls.array<4 x f32> {
  // XLS: (bits[1], bits[8], bits[23])[4] = literal(value=[(0, 0, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)]
  %0 = "xls.array_zero"() : () -> !xls.array<4 x f32>
  return %0 : !xls.array<4 x f32>
}

// XLS-LABEL: array_concat
func.func private @array_concat(%arg0: !xls.array<2 x i32>, %arg1: !xls.array<2 x i32>) -> !xls.array<4 x i32> {
  // XLS: bits[32][4] = array_concat
  %0 = xls.array_concat %arg0, %arg1 : (!xls.array<2 x i32>, !xls.array<2 x i32>) -> !xls.array<4 x i32>
  return %0 : !xls.array<4 x i32>
}

// Dummy to provide a top level function for XLS.
func.func @topFn(%arg0: i32) -> i32 {
  return %arg0 : i32
}
