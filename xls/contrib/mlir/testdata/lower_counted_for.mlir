// RUN: xls/contrib/mlir/xls_opt -xls-lower-for-ops %s 2>&1 | FileCheck %s

// CHECK-LABEL:   func.func private @for_body(
// CHECK-SAME:         %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_4:.*]] = arith.addi %[[VAL_1]], %[[VAL_2]] : i32
// CHECK:           return %[[VAL_4]] : i32

// CHECK-LABEL:   func.func @reduce(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32)
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1024 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = "xls.counted_for"(%[[VAL_4]], %[[VAL_0]]) <{stride = 1 : i64, to_apply = @for_body, trip_count = 1024 : i64}> : (i32, i32) -> i32
// CHECK:           return %[[VAL_5]] : i32
func.func @reduce(%arg0: i32) -> i32 attributes {xls = true} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = xls.for inits(%c0_i32) invariants(%arg0) {
  ^bb0(%indvar: i32, %carry: i32, %invariant: i32):
    %1 = arith.index_cast %indvar : i32 to index
    %2 = arith.addi %carry, %invariant : i32
    xls.yield %2 : i32
  } {trip_count = 1024 : i64} : (i32, i32) -> i32
  return %0 : i32
}

// CHECK-LABEL:   func.func private @for_body_0(
// CHECK-SAME:        %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: tuple<i32, i32>, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32) -> tuple<i32, i32>
// CHECK-DAG:       %[[VAL_4:.*]] = "xls.tuple_index"(%[[VAL_1]]) <{index = 0 : i64}> : (tuple<i32, i32>) -> i32
// CHECK-DAG:       %[[VAL_5:.*]] = "xls.tuple_index"(%[[VAL_1]]) <{index = 1 : i64}> : (tuple<i32, i32>) -> i32
// CHECK:           %[[VAL_6:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_7:.*]] = arith.muli %[[VAL_4]], %[[VAL_2]] : i32
// CHECK:           %[[VAL_8:.*]] = arith.addi %[[VAL_7]], %[[VAL_3]] : i32
// CHECK:           %[[VAL_9:.*]] = "xls.tuple"(%[[VAL_8]], %[[VAL_5]]) : (i32, i32) -> tuple<i32, i32>
// CHECK:           return %[[VAL_9]] : tuple<i32, i32>

// CHECK-LABEL:   func.func @reduce_arity_2(
// CHECK-SAME:        %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32) -> i32
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1024 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_6:.*]] = "xls.tuple"(%[[VAL_5]], %[[VAL_5]]) : (i32, i32) -> tuple<i32, i32>
// CHECK:           %[[VAL_7:.*]] = "xls.counted_for"(%[[VAL_6]], %[[VAL_1]], %[[VAL_0]]) <{stride = 1 : i64, to_apply = @for_body_0, trip_count = 1024 : i64}> : (tuple<i32, i32>, i32, i32) -> tuple<i32, i32>
// CHECK:           %[[VAL_8:.*]] = "xls.tuple_index"(%[[VAL_7]]) <{index = 0 : i64}> : (tuple<i32, i32>) -> i32
// TODO(jpienaar): We could avoid generating the tuple indexing below.
// CHECK:           %[[VAL_9:.*]] = "xls.tuple_index"(%[[VAL_7]]) <{index = 1 : i64}> : (tuple<i32, i32>) -> i32
// CHECK:           return %[[VAL_8]] : i32
func.func @reduce_arity_2(%arg0: i32, %arg1: i32) -> i32 attributes {xls = true} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0:2 = xls.for inits(%c0_i32, %c0_i32) invariants(%arg1, %arg0) {
  ^bb0(%indvar: i32, %carry: i32, %carry_0: i32, %invariant: i32, %invariant_1: i32):
    %1 = arith.index_cast %indvar : i32 to index
    %2 = arith.muli %carry, %invariant : i32
    %3 = arith.addi %2, %invariant_1 : i32
    xls.yield %3, %carry_0 : i32, i32
  } {trip_count = 1024 : i64} : (i32, i32, i32, i32) -> (i32, i32)
  return %0#0 : i32
}

// CHECK-LABEL:   func.func private @for_body_3(
// CHECK-SAME:      %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32, %[[VAL_3:.*]]: i32) -> i32
// CHECK:           %[[VAL_4:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_5:.*]] = arith.addi %[[VAL_2]], %[[VAL_3]] : i32
// CHECK:           return %[[VAL_5]] : i32

// CHECK-LABEL:   func.func private @for_body_2(
// CHECK-SAME:        %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) -> i32
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_4:.*]] = "xls.counted_for"(%[[VAL_1]], %[[VAL_1]], %[[VAL_2]]) <{stride = 1 : i64, to_apply = @for_body_3, trip_count = 1024 : i64}> : (i32, i32, i32) -> i32
// CHECK:           return %[[VAL_4]] : i32

// CHECK-LABEL:   func.func private @for_body_1(
// CHECK-SAME:        %[[VAL_0:.*]]: i32, %[[VAL_1:.*]]: i32, %[[VAL_2:.*]]: i32) -> i32
// CHECK:           %[[VAL_3:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:           %[[VAL_4:.*]] = "xls.counted_for"(%[[VAL_1]], %[[VAL_2]]) <{stride = 1 : i64, to_apply = @for_body_2, trip_count = 1024 : i64}> : (i32, i32) -> i32
// CHECK:           return %[[VAL_4]] : i32

// CHECK-LABEL:   func.func @triple_nest(
// CHECK-SAME:                           %[[VAL_0:.*]]: i32) -> i32
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1024 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = "xls.counted_for"(%[[VAL_4]], %[[VAL_0]]) <{stride = 1 : i64, to_apply = @for_body_1, trip_count = 1024 : i64}> : (i32, i32) -> i32
// CHECK:           return %[[VAL_5]] : i32
func.func @triple_nest(%arg0: i32) -> i32 attributes {xls = true} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = xls.for inits(%c0_i32) invariants(%arg0) {
  ^bb0(%indvar: i32, %carry: i32, %invariant: i32):
    %1 = arith.index_cast %indvar : i32 to index
    %2 = xls.for inits(%carry) invariants(%invariant) {
    ^bb0(%indvar_0: i32, %carry_1: i32, %invariant_2: i32):
      %3 = arith.index_cast %indvar_0 : i32 to index
      %4 = xls.for inits(%carry_1) invariants(%carry_1, %invariant_2) {
      ^bb0(%indvar_3: i32, %carry_4: i32, %invariant_5: i32, %invariant_6: i32):
        %5 = arith.index_cast %indvar_3 : i32 to index
        %6 = arith.addi %invariant_5, %invariant_6 : i32
        xls.yield %6 : i32
      } {trip_count = 1024 : i64} : (i32, i32, i32) -> i32
      xls.yield %4 : i32
    } {trip_count = 1024 : i64} : (i32, i32) -> i32
    xls.yield %2 : i32
  } {trip_count = 1024 : i64} : (i32, i32) -> i32
  return %0 : i32
}

// CHECK-LABEL:   xls.eproc @proc_reduce(
// CHECK-SAME:                      %[[VAL_0:.*]]: i32)
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1024 : index
// CHECK-DAG:       %[[VAL_3:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : i32
// CHECK:           %[[VAL_5:.*]] = "xls.counted_for"(%[[VAL_4]], %[[VAL_0]]) <{stride = 1 : i64, to_apply = @for_body_4, trip_count = 1024 : i64}> : (i32, i32) -> i32
// CHECK:           xls.yield %[[VAL_5]] : i32
xls.eproc @proc_reduce(%arg0: i32) zeroinitializer  {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = xls.for inits(%c0_i32) invariants(%arg0) {
  ^bb0(%indvar: i32, %carry: i32, %invariant: i32):
    %1 = arith.index_cast %indvar : i32 to index
    %2 = arith.addi %carry, %invariant : i32
    xls.yield %2 : i32
  } {trip_count = 1024 : i64} : (i32, i32) -> i32
  xls.yield %0 : i32
}
