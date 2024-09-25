// RUN: xls/contrib/mlir/xls_opt -expand-macro-ops %s 2>&1 | FileCheck %s

// CHECK-LABEL: @array_update_slice
func.func @array_update_slice(%arg0: !xls.array<4 x i32>, %arg1: !xls.array<2 x i32>, %arg2: i32) -> !xls.array<4 x i32> attributes { "xls" = true } {
// CHECK-NEXT:   %0 = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:   %1 = "xls.constant_scalar"() <{value = 0 : i32}> : () -> i32
// CHECK-NEXT:   %2 = "xls.array_index_static"(%arg1) <{index = 0 : i64}> : (!xls.array<2 x i32>) -> i32
// CHECK-NEXT:   %3 = xls.add %arg2, %1 : i32
// CHECK-NEXT:   %4 = "xls.array_update"(%arg0, %2, %3) : (!xls.array<4 x i32>, i32, i32) -> !xls.array<4 x i32>
// CHECK-NEXT:   %5 = "xls.array_index_static"(%arg1) <{index = 1 : i64}> : (!xls.array<2 x i32>) -> i32
// CHECK-NEXT:   %6 = xls.add %arg2, %0 : i32
// CHECK-NEXT:   %7 = "xls.array_update"(%4, %5, %6) : (!xls.array<4 x i32>, i32, i32) -> !xls.array<4 x i32>
// CHECK-NEXT:   return %7 : !xls.array<4 x i32>
  %0 = xls.array_update_slice %arg1 into %arg0[%arg2 +: 2] : !xls.array<4 x i32>
  return %0 : !xls.array<4 x i32>
}
