// RUN: xls_opt -index-type-conversion=index-bitwidth=32 %s 2>&1 | FileCheck %s --check-prefix=INDEX32
// RUN: xls_opt -index-type-conversion=index-bitwidth=64 %s 2>&1 | FileCheck %s --check-prefix=INDEX64

// INDEX32-LABEL: func.func @i64_to_index
// INDEX64-LABEL: func.func @i64_to_index
// INDEX32-SAME:  %[[lhs:.*]]: i64,
// INDEX64-SAME:  %[[lhs:.*]]: i64,
// INDEX32-SAME:  %[[rhs:.*]]: i64) -> i64 attributes {xls = true} {
// INDEX64-SAME:  %[[rhs:.*]]: i64) -> i64 attributes {xls = true} {
func.func @i64_to_index(%lhs : i64, %rhs : i64) -> i64 attributes {xls = true} {
  // INDEX32:           %[[lhs_i32:.*]] = xls.bit_slice %[[lhs]] {start = 0 : i64, width = 32 : i64} : (i64) -> i32
  %lhs_index = arith.index_cast %lhs : i64 to index
  // INDEX32:           %[[rhs_i32:.*]] = xls.bit_slice %[[rhs]] {start = 0 : i64, width = 32 : i64} : (i64) -> i32
  %rhs_index = arith.index_cast %rhs : i64 to index

  // INDEX32:           %[[add:.*]] = xls.add %[[lhs_i32]], %[[rhs_i32]] : i32
  // INDEX64:           %[[add:.*]] = xls.add %[[lhs]], %[[rhs]] : i64
  %ret = xls.add %lhs_index, %rhs_index : index

  // INDEX32:           %[[ret:.*]] = xls.sign_ext %[[add]] : (i32) -> i64
  %ret_int = arith.index_cast %ret: index to i64

  // INDEX32:           return %[[ret]] : i64
  // INDEX64:           return %[[add]] : i64
  return %ret_int : i64
}

// INDEX32-LABEL:   func.func @i32_to_index(
// INDEX64-LABEL:   func.func @i32_to_index(
// INDEX32-SAME:    %[[lhs:.*]]: i32,
// INDEX32-SAME:    %[[rhs:.*]]: i32) -> i32 attributes {xls = true} {
// INDEX64-SAME:    %[[lhs:.*]]: i32,
// INDEX64-SAME:    %[[rhs:.*]]: i32) -> i32 attributes {xls = true} {
func.func @i32_to_index(%lhs : i32, %rhs : i32) -> i32 attributes {xls = true} {
  // INDEX64:           %[[lhs_index:.*]] = xls.sign_ext %[[lhs]] : (i32) -> i64
  // INDEX64:           %[[rhs_index:.*]] = xls.sign_ext %[[rhs]] : (i32) -> i64
  %lhs_index = arith.index_cast %lhs : i32 to index
  %rhs_index = arith.index_cast %rhs : i32 to index

  // INDEX32:           %[[add:.*]] = xls.add %[[lhs]], %[[rhs]] : i32
  // INDEX64:           %[[add:.*]] = xls.add %[[lhs_index]], %[[rhs_index]] : i64
  %add = xls.add %lhs_index, %rhs_index : index

  // INDEX64:           %[[ret:.*]] = xls.bit_slice %[[add]] {start = 0 : i64, width = 32 : i64} : (i64) -> i32
  %ret = arith.index_castui %add: index to i32

  // INDEX32:           return %[[add]] : i32
  // INDEX64:           return %[[ret]] : i32
  return %ret : i32
}

// INDEX32-LABEL:   func.func @constant_index(
// INDEX64-LABEL:   func.func @constant_index(
// INDEX32-SAME:    %[[array:.*]]: !xls.array<2 x i32>) -> i32 attributes {xls = true} {
// INDEX64-SAME:    %[[array:.*]]: !xls.array<2 x i32>) -> i32 attributes {xls = true} {
func.func @constant_index(%array : !xls.array<2 x i32>) -> i32 attributes {xls = true} {
  // INDEX32:           %[[index:.*]] = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
  // INDEX64:           %[[index:.*]] = "xls.constant_scalar"() <{value = 1 : i64}> : () -> i64
  %cst = arith.constant 1 : index
  // INDEX32:           "xls.constant_scalar"() <{value = 21 : i32}> : () -> i32
  // INDEX64:           "xls.constant_scalar"() <{value = 21 : i64}> : () -> i64
  %2 = "xls.constant_scalar"() <{value = 21 : index}> : () -> index

  // INDEX32:           %[[ret:.*]] = "xls.array_index"(%[[array]], %[[index]]) : (!xls.array<2 x i32>, i32) -> i32
  // INDEX64:           %[[ret:.*]] = "xls.array_index"(%[[array]], %[[index]]) : (!xls.array<2 x i32>, i64) -> i32
  %ret = "xls.array_index"(%array, %cst) : (!xls.array<2 x i32>, index) -> i32

  // INDEX32:           return %[[ret]] : i32
  // INDEX64:           return %[[ret]] : i32
  return %ret : i32
}

// INDEX64-LABEL:   func.func @array() -> i32 attributes {xls = true} {
// INDEX32-LABEL:   func.func @array() -> i32 attributes {xls = true} {
func.func @array() -> i32 attributes {xls = true} {
  // INDEX32:           %[[cst:.*]] = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
  // INDEX64:           %[[cst:.*]] = "xls.constant_scalar"() <{value = 1 : i64}> : () -> i64
  %cst = arith.constant 1 : index

  // INDEX32:           %[[array:.*]] = xls.array %[[cst]], %[[cst]] : (i32, i32) -> !xls.array<2 x i32>
  // INDEX64:           %[[array:.*]] = xls.array %[[cst]], %[[cst]] : (i64, i64) -> !xls.array<2 x i64>
  %array = "xls.array"(%cst, %cst) : (index, index) -> !xls.array<2 x index>

  // INDEX32:           %[[ele:.*]] = "xls.array_index"(%[[array]], %[[cst]]) : (!xls.array<2 x i32>, i32) -> i32
  // INDEX64:           %[[ele:.*]] = "xls.array_index"(%[[array]], %[[cst]]) : (!xls.array<2 x i64>, i64) -> i64
  %ret = "xls.array_index"(%array, %cst) : (!xls.array<2 x index>, index) -> index

  // INDEX64:           %[[ret:.*]] = xls.bit_slice %[[ele]] {start = 0 : i64, width = 32 : i64} : (i64) -> i32
  %ret_i32 = arith.index_cast %ret: index to i32

  // INDEX32:           return %[[ele]] : i32
  // INDEX64:           return %[[ret]] : i32
  return %ret_i32 : i32
}

// INDEX32-LABEL:   func.func @array_nested() -> i32 attributes {xls = true} {
// INDEX64-LABEL:   func.func @array_nested() -> i32 attributes {xls = true} {
func.func @array_nested() -> i32 attributes {xls = true} {
  // INDEX32:           %[[cst:.*]] = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
  // INDEX64:           %[[cst:.*]] = "xls.constant_scalar"() <{value = 1 : i64}> : () -> i64
  %cst = arith.constant 1 : index

  // INDEX32:           %[[array:.*]] = xls.array %[[cst]], %[[cst]] : (i32, i32) -> !xls.array<2 x i32>
  // INDEX64:           %[[array:.*]] = xls.array %[[cst]], %[[cst]] : (i64, i64) -> !xls.array<2 x i64>
  %array = "xls.array"(%cst, %cst) : (index, index) -> !xls.array<2 x index>

  // INDEX32:           %[[array_2d:.*]] = xls.array %[[array]], %[[array]] : (!xls.array<2 x i32>, !xls.array<2 x i32>) -> !xls.array<2 x !xls.array<2 x i32>>
  // INDEX64:           %[[array_2d:.*]] = xls.array %[[array]], %[[array]] : (!xls.array<2 x i64>, !xls.array<2 x i64>) -> !xls.array<2 x !xls.array<2 x i64>>
  %array_2d= "xls.array"(%array, %array) : (!xls.array<2 x index>, !xls.array<2 x index>) -> !xls.array<2 x !xls.array<2 x index>>

  // INDEX32:           %[[ele:.*]] = "xls.array_index"(%[[array_2d]], %[[cst]]) : (!xls.array<2 x !xls.array<2 x i32>>, i32) -> !xls.array<2 x i32>
  // INDEX64:           %[[ele:.*]] = "xls.array_index"(%[[array_2d]], %[[cst]]) : (!xls.array<2 x !xls.array<2 x i64>>, i64) -> !xls.array<2 x i64>
  %array_1 = "xls.array_index"(%array_2d, %cst) : (!xls.array<2 x !xls.array<2 x index>>, index) -> !xls.array<2 x index>

  // INDEX32:           %[[ret:.*]] = "xls.array_index"(%[[ele]], %[[cst]]) : (!xls.array<2 x i32>, i32) -> i32
  // INDEX64:           %[[ret:.*]] = "xls.array_index"(%[[ele]], %[[cst]]) : (!xls.array<2 x i64>, i64) -> i64
  %ret = "xls.array_index"(%array_1, %cst) : (!xls.array<2 x index>, index) -> index

  // INDEX64:           %[[ret_i32:.*]] = xls.bit_slice %[[ret]] {start = 0 : i64, width = 32 : i64} : (i64) -> i32
  %ret_i32 = arith.index_cast %ret: index to i32

  // INDEX32:           return %[[ret]] : i32
  // INDEX64            return %[[ret_i32]] : i32
  return %ret_i32 : i32
}


// INDEX32-LABEL:   func.func @tuple(
// INDEX32-SAME:    %[[int:.*]]: i64) -> i32 attributes {xls = true} {
// INDEX64-LABEL:   func.func @tuple(
// INDEX64-SAME:    %[[int:.*]]: i64) -> i32 attributes {xls = true} {
func.func @tuple(%i : i64) -> i32 attributes {xls = true} {
  // INDEX32:           %[[index:.*]] = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
  // INDEX64:           %[[index:.*]] = "xls.constant_scalar"() <{value = 1 : i64}> : () -> i64
  %cst = arith.constant 1 : index

  // INDEX32:           %[[tuple:.*]] = "xls.tuple"(%[[index]], %[[int]]) : (i32, i64) -> tuple<i32, i64>
  // INDEX64:           %[[tuple:.*]] = "xls.tuple"(%[[index]], %[[int]]) : (i64, i64) -> tuple<i64, i64>
  %tuple = "xls.tuple"(%cst, %i) : (index, i64) -> tuple<index, i64>

  // INDEX32:           %[[ele:.*]] = "xls.tuple_index"(%[[tuple]]) <{index = 0 : i64}> : (tuple<i32, i64>) -> i32
  // INDEX64:           %[[ele:.*]] = "xls.tuple_index"(%[[tuple]]) <{index = 0 : i64}> : (tuple<i64, i64>) -> i64
  %ret = "xls.tuple_index"(%tuple) { index = 0 : i64 } : (tuple<index, i64> ) -> index

  // INDEX64:           %[[ret:.*]] = xls.bit_slice %[[ele]] {start = 0 : i64, width = 32 : i64} : (i64) -> i32
  %ret_i32 = arith.index_cast %ret: index to i32

  // INDEX32:           return %[[ele]] : i32
  // INDEX64:           return %[[ret]] : i32
  return %ret_i32 : i32
}

// INDEX32-LABEL:   func.func @forloop(
// INDEX64-LABEL:   func.func @forloop(
func.func @forloop(%arg0: i32, %arg1: i8, %arg2: i9) -> i32 attributes {xls = true} {
  %0 = xls.for inits(%arg0) invariants(%arg1, %arg2) {
    ^bb0(%arg3: index, %arg4: i32, %arg5: i8, %arg6: i9):
    // INDEX32:         %indvar: i32
    // INDEX64:         %indvar: i64
    // INDEX64-NEXT:    xls.bit_slice %indvar {start = 0 : i64, width = 32 : i64} : (i64) -> i32
    %i = arith.index_cast %arg3 : index to i32
    xls.yield %i : i32
  } { trip_count = 6 : i64 } : (i32, i8, i9) -> i32
  return %0 : i32
}

// INDEX32-LABEL:   xls.chan @mychan : i32
// INDEX64-LABEL:   xls.chan @mychan : i64
xls.chan @mychan : index

