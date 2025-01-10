// RUN: xls_opt -canonicalize %s 2>&1 | FileCheck %s

// CHECK-LABEL: func @umul_to_shll
func.func @umul_to_shll(%arg0: i32) -> i32 {
  // CHECK: %[[VAL:.*]] = "xls.constant_scalar"() <{value = 2 : index}> : () -> index
  // CHECK: xls.shll %arg0, %[[VAL]]
  %cst = "xls.constant_scalar"() { value = 4 : i7 } : () -> i32
  %0 = xls.umul %arg0, %cst : i32
  return %0 : i32
}

// CHECK-LABEL: func.func @tuple_index_folding
// CHECK-SAME:  (%[[ARG:.*]]: i32)
func.func @tuple_index_folding(%v1: i32) -> i32 {
// CHECK-NEXT:      return %[[ARG]]
  %tuple = "xls.tuple"(%v1, %v1) : (i32, i32) -> tuple<i32, i32>
  %member1 = "xls.tuple_index"(%tuple) <{index = 1}> : (tuple<i32, i32>) -> i32
  return %member1 : i32
}
