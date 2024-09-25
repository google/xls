// RUN: xls/contrib/mlir/xls_opt -xls-lower-for-ops %s 2>&1 | FileCheck %s

// CHECK:       func.func @for_body(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    %1 = arith.addi %arg1, %arg2 : i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @reduce(%arg0: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1024 = arith.constant 1024 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = "xls.counted_for"(%c0_i32, %arg0) <{stride = 1 : i64, to_apply = @for_body, trip_count = 1024 : i64}> : (i32, i32) -> i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
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

// CHECK-LABEL: func.func @for_body_0(%arg0: i32, %arg1: tuple<i32, i32>, %arg2: i32, %arg3: i32) -> tuple<i32, i32> attributes {xls = true} {
// CHECK-NEXT:    %0 = "xls.tuple_index"(%arg1) <{index = 0 : i64}> : (tuple<i32, i32>) -> i32
// CHECK-NEXT:    %1 = "xls.tuple_index"(%arg1) <{index = 1 : i64}> : (tuple<i32, i32>) -> i32
// CHECK-NEXT:    %2 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    %3 = arith.muli %0, %arg2 : i32
// CHECK-NEXT:    %4 = arith.addi %3, %arg3 : i32
// CHECK-NEXT:    %5 = "xls.tuple"(%4, %1) : (i32, i32) -> tuple<i32, i32>
// CHECK-NEXT:    return %5 : tuple<i32, i32>
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @reduce_arity_2(%arg0: i32, %arg1: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1024 = arith.constant 1024 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = "xls.tuple"(%c0_i32, %c0_i32) : (i32, i32) -> tuple<i32, i32>
// CHECK-NEXT:    %1 = "xls.counted_for"(%0, %arg1, %arg0) <{stride = 1 : i64, to_apply = @for_body_0, trip_count = 1024 : i64}> : (tuple<i32, i32>, i32, i32) -> tuple<i32, i32>
// CHECK-NEXT:    %2 = "xls.tuple_index"(%1) <{index = 0 : i64}> : (tuple<i32, i32>) -> i32
// CHECK-NEXT:    %3 = "xls.tuple_index"(%1) <{index = 1 : i64}> : (tuple<i32, i32>) -> i32
// CHECK-NEXT:    return %2 : i32
// CHECK-NEXT:  }
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

// CHECK-LABEL: func.func @for_body_3(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    %1 = arith.addi %arg2, %arg3 : i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @for_body_2(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    %1 = "xls.counted_for"(%arg1, %arg1, %arg2) <{stride = 1 : i64, to_apply = @for_body_3, trip_count = 1024 : i64}> : (i32, i32, i32) -> i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @for_body_1(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %0 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:    %1 = "xls.counted_for"(%arg1, %arg2) <{stride = 1 : i64, to_apply = @for_body_2, trip_count = 1024 : i64}> : (i32, i32) -> i32
// CHECK-NEXT:    return %1 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  func.func @triple_nest(%arg0: i32) -> i32 attributes {xls = true} {
// CHECK-NEXT:    %c0 = arith.constant 0 : index
// CHECK-NEXT:    %c1024 = arith.constant 1024 : index
// CHECK-NEXT:    %c1 = arith.constant 1 : index
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %0 = "xls.counted_for"(%c0_i32, %arg0) <{stride = 1 : i64, to_apply = @for_body_1, trip_count = 1024 : i64}> : (i32, i32) -> i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
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
