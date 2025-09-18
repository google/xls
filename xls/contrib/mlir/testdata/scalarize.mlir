// RUN: xls_opt -scalarize -canonicalize -cse %s 2>&1 | FileCheck %s

// CHECK-LABEL: @signature
// CHECK-NEXT: {{constant_scalar.*value = 6}}
// CHECK-NEXT: {{constant_scalar.*value = 7}}
// CHECK-NEXT: array_index_static
// CHECK-NEXT: add
// CHECK-NEXT: array_index_static
// CHECK-NEXT: add
// CHECK-NEXT: array
func.func @signature(%arg0: tensor<2xi32>) -> tensor<2xi32> attributes { "xls" = true } {
  %0 = "xls.constant_tensor"() { value = dense<[6, 7]> : tensor<2xi32> } : () -> tensor<2xi32>
  %1 = xls.add %arg0, %0 : tensor<2xi32>
  return %1 : tensor<2xi32>
}

// CHECK-LABEL: @noarg
// CHECK-NEXT: {{constant_scalar.*value = 12}}
// CHECK-NEXT: {{constant_scalar.*value = 14}}
// CHECK-NEXT: array
func.func @noarg() -> tensor<2xi32> attributes { "xls" = true } {
  %0 = "xls.constant_tensor"() { value = dense<[6, 7]> : tensor<2xi32> } : () -> tensor<2xi32>
  %1 = xls.add %0, %0 : tensor<2xi32>
  return %1 : tensor<2xi32>
}

// CHECK-LABEL: @empty
// CHECK: return %arg0 : !xls.array<2xi32>
func.func @empty(%arg0: tensor<2xi32>, %arg1: i32) -> tensor<2xi32> attributes { "xls" = true } {
  return %arg0 : tensor<2xi32>
}

// CHECK-LABEL: @tensor_insert
// CHECK: constant
// CHECK-SAME: 13
// CHECK-NEXT: array_update
// CHECK-SAME: xls.array<14xi32>
func.func @tensor_insert(%arg0: tensor<2x7xi32>, %arg1: i32) -> tensor<2x7xi32> attributes { "xls" = true } {
  %0 = arith.constant 1 : index
  %1 = arith.constant 6 : index
  %2 = tensor.insert %arg1 into %arg0[%0, %1] : tensor<2x7xi32>
  return %2 : tensor<2x7xi32>
}

// CHECK-LABEL: @tensor_concat
// CHECK: xls.array_concat
// CHECK-SAME: xls.array<28xf32>
func.func @tensor_concat(%arg0: tensor<2x7xf32>, %arg1: tensor<2x7xf32>) -> tensor<4x7xf32> attributes { "xls" = true } {
  %0 = tensor.concat dim(0) %arg0, %arg1 : (tensor<2x7xf32>, tensor<2x7xf32>) -> tensor<4x7xf32>
  return %0 : tensor<4x7xf32>
}

// CHECK-LABEL: @tensor_extract_element
// CHECK: array_index_static
// CHECK-SAME: index = 9
func.func @tensor_extract_element(%arg0: tensor<3x7xi32>, %arg1: i32) -> i32 attributes { "xls" = true } {
  %0 = arith.constant 1 : index
  %1 = arith.constant 2 : index
  %2 = tensor.extract %arg0[%0, %1] : tensor<3x7xi32>
  return %2 : i32
}

// CHECK-LABEL: @tensor_extract_single_slice_1d_unit
// CHECK-NOT: xls.array_index
// CHECK: xls.array_slice
// CHECK-NEXT: return
func.func @tensor_extract_single_slice_1d_unit(%arg0: tensor<3xi32>, %arg1: index) -> tensor<i32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[%arg1] [1] [1] : tensor<3xi32> to tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @tensor_extract_single_slice_1d_subset
// CHECK-NOT: xls.array_index
// CHECK: xls.array_slice
// CHECK-NEXT: return
func.func @tensor_extract_single_slice_1d_subset(%arg0: tensor<3xi32>, %arg1: index) -> tensor<2xi32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[%arg1] [2] [1] : tensor<3xi32> to tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: @tensor_extract_single_slice_1d_all
// CHECK-NOT: xls.array_index
// CHECK: xls.array_slice
// CHECK-NEXT: return
func.func @tensor_extract_single_slice_1d_all(%arg0: tensor<3xi32>, %arg1: index) -> tensor<3xi32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[%arg1] [3] [1] : tensor<3xi32> to tensor<3xi32>
  return %0 : tensor<3xi32>
}

// CHECK-LABEL: @tensor_extract_single_slice_2d_scalar
// CHECK-NOT: xls.array_index
// CHECK: xls.array_slice
// CHECK-NEXT: return
func.func @tensor_extract_single_slice_2d_scalar(%arg0: tensor<3x3xi32>, %arg1: index) -> tensor<i32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[0, %arg1] [1, 1] [1, 1] : tensor<3x3xi32> to tensor<i32>
  return %0 : tensor<i32>
}

// CHECK-LABEL: @tensor_extract_single_slice_2d_unit
// CHECK-NOT: xls.array_index
// CHECK: xls.array_slice
// CHECK-NEXT: return
func.func @tensor_extract_single_slice_2d_unit(%arg0: tensor<3x3xi32>, %arg1: index) -> tensor<3xi32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[0, %arg1] [1, 3] [1, 1] : tensor<3x3xi32> to tensor<3xi32>
  return %0 : tensor<3xi32>
}

// CHECK-LABEL: @tensor_extract_single_slice_2d_subset
// CHECK-NOT: xls.array_index
// CHECK: xls.array_slice
// CHECK-NEXT: return
func.func @tensor_extract_single_slice_2d_subset(%arg0: tensor<3x3xi32>, %arg1: index) -> tensor<2x3xi32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[0, %arg1] [2, 3] [1, 1] : tensor<3x3xi32> to tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: @tensor_extract_single_slice_2d_subset_leading_unit
// CHECK-NOT: xls.array_index
// CHECK: xls.array_slice
// CHECK-NEXT: return
func.func @tensor_extract_single_slice_2d_subset_leading_unit(%arg0: tensor<1x1x3x3x1x1x1xi32>, %arg1: index) -> tensor<2x3xi32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[0, 0, 0, %arg1, 0, 0, 0] [1, 1, 2, 3, 1, 1, 1] [1, 1, 1, 1, 1, 1, 1] : tensor<1x1x3x3x1x1x1xi32> to tensor<2x3xi32>
  return %0 : tensor<2x3xi32>
}

// CHECK-LABEL: @tensor_extract_single_slice_2d_all
// CHECK-NOT: xls.array_index
// CHECK: xls.array_slice
// CHECK-NEXT: return
func.func @tensor_extract_single_slice_2d_all(%arg0: tensor<3x3xi32>, %arg1: index) -> tensor<3x3xi32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[0, %arg1] [3, 3] [1, 1] : tensor<3x3xi32> to tensor<3x3xi32>
  return %0 : tensor<3x3xi32>
}

// CHECK-LABEL:   func.func @tensor_extract_non_leading_slice(
// CHECK-SAME:                                                %[[ARG_0:.*]]: !xls.array<24xi32>,
// CHECK-SAME:                                                %[[ARG_1:.*]]: index) -> !xls.array<12xi32> attributes {xls = true} {
// CHECK-DAG:       %[[C16:.*]] = "xls.constant_scalar"() <{value = 16 : index}> : () -> index
// CHECK-DAG:       %[[C8:.*]] = "xls.constant_scalar"() <{value = 8 : index}> : () -> index
// CHECK-DAG:       %[[C4:.*]] = "xls.constant_scalar"() <{value = 4 : index}> : () -> index
// CHECK-DAG:       %[[C0:.*]] = "xls.constant_scalar"() <{value = 0 : index}> : () -> index
// CHECK-DAG:       %[[VAL_6:.*]] = arith.constant 2 : index
// CHECK-DAG:       %[[VAL_7:.*]] = xls.umul %[[ARG_1]], %[[VAL_6]] : index
// CHECK-DAG:       %[[VAL_8:.*]] = xls.add %[[C0]], %[[VAL_7]] : index
// CHECK-DAG:       %[[VAL_9:.*]] = xls.add %[[VAL_8]], %[[C0]] : index
// CHECK-DAG:       %[[VAL_10:.*]] = xls.add %[[VAL_9]], %[[C0]] : index
// CHECK-DAG:       %[[VAL_11:.*]] = "xls.array_slice"(%[[ARG_0]], %[[VAL_10]]) <{width = 2 : i64}> : (!xls.array<24xi32>, index) -> !xls.array<2xi32>
// CHECK-DAG:       %[[VAL_12:.*]] = xls.add %[[VAL_8]], %[[C4]] : index
// CHECK-DAG:       %[[VAL_13:.*]] = xls.add %[[VAL_12]], %[[C0]] : index
// CHECK-DAG:       %[[VAL_14:.*]] = "xls.array_slice"(%[[ARG_0]], %[[VAL_13]]) <{width = 2 : i64}> : (!xls.array<24xi32>, index) -> !xls.array<2xi32>
// CHECK-DAG:       %[[VAL_15:.*]] = xls.array_concat %[[VAL_11]], %[[VAL_14]] : (!xls.array<2xi32>, !xls.array<2xi32>) -> !xls.array<4xi32>
// CHECK-DAG:       %[[VAL_16:.*]] = xls.add %[[VAL_9]], %[[C8]] : index
// CHECK-DAG:       %[[VAL_17:.*]] = "xls.array_slice"(%[[ARG_0]], %[[VAL_16]]) <{width = 2 : i64}> : (!xls.array<24xi32>, index) -> !xls.array<2xi32>
// CHECK-DAG:       %[[VAL_18:.*]] = xls.add %[[VAL_12]], %[[C8]] : index
// CHECK-DAG:       %[[VAL_19:.*]] = "xls.array_slice"(%[[ARG_0]], %[[VAL_18]]) <{width = 2 : i64}> : (!xls.array<24xi32>, index) -> !xls.array<2xi32>
// CHECK-DAG:       %[[VAL_20:.*]] = xls.array_concat %[[VAL_17]], %[[VAL_19]] : (!xls.array<2xi32>, !xls.array<2xi32>) -> !xls.array<4xi32>
// CHECK-DAG:       %[[VAL_21:.*]] = xls.add %[[VAL_9]], %[[C16]] : index
// CHECK-DAG:       %[[VAL_22:.*]] = "xls.array_slice"(%[[ARG_0]], %[[VAL_21]]) <{width = 2 : i64}> : (!xls.array<24xi32>, index) -> !xls.array<2xi32>
// CHECK-DAG:       %[[VAL_23:.*]] = xls.add %[[VAL_12]], %[[C16]] : index
// CHECK-DAG:       %[[VAL_24:.*]] = "xls.array_slice"(%[[ARG_0]], %[[VAL_23]]) <{width = 2 : i64}> : (!xls.array<24xi32>, index) -> !xls.array<2xi32>
// CHECK-DAG:       %[[VAL_25:.*]] = xls.array_concat %[[VAL_22]], %[[VAL_24]] : (!xls.array<2xi32>, !xls.array<2xi32>) -> !xls.array<4xi32>
// CHECK-DAG:       %[[VAL_26:.*]] = xls.array_concat %[[VAL_15]], %[[VAL_20]], %[[VAL_25]] : (!xls.array<4xi32>, !xls.array<4xi32>, !xls.array<4xi32>) -> !xls.array<12xi32>
// CHECK-DAG:       return %[[VAL_26]] : !xls.array<12xi32>
// CHECK-NEXT:    }
func.func @tensor_extract_non_leading_slice(%arg0: tensor<3x2x2x2xi32>, %arg1: index) ->  tensor<3x2x1x2xi32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[0, 0, %arg1, 0] [3, 2, 1, 2] [1, 1, 1, 1] : tensor<3x2x2x2xi32> to tensor<3x2x1x2xi32>
  return %0 :  tensor<3x2x1x2xi32>
}

// CHECK-LABEL:   func.func @tensor_extract_slice_unroll(
// CHECK-SAME:                                           %[[ARG_0:.*]]: !xls.array<9xi32>,
// CHECK-SAME:                                           %[[ARG_1:.*]]: index) -> !xls.array<4xi32> attributes {xls = true} {
// CHECK-DAG:       %[[C3:.*]] = "xls.constant_scalar"() <{value = 3 : index}> : () -> index
// CHECK-DAG:       %[[C0:.*]] = "xls.constant_scalar"() <{value = 0 : index}> : () -> index
// CHECK-DAG:       %[[VAL_4:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[VAL_5:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[VAL_6:.*]] = xls.umul %[[ARG_1]], %[[VAL_5]] : index
// CHECK-DAG:       %[[VAL_7:.*]] = xls.add %[[VAL_4]], %[[VAL_6]] : index
// CHECK-DAG:       %[[VAL_8:.*]] = xls.add %[[VAL_7]], %[[C0]] : index
// CHECK-DAG:       %[[VAL_9:.*]] = "xls.array_slice"(%[[ARG_0]], %[[VAL_8]]) <{width = 2 : i64}> : (!xls.array<9xi32>, index) -> !xls.array<2xi32>
// CHECK-DAG:       %[[VAL_10:.*]] = xls.add %[[VAL_7]], %[[C3]] : index
// CHECK-DAG:       %[[VAL_11:.*]] = "xls.array_slice"(%[[ARG_0]], %[[VAL_10]]) <{width = 2 : i64}> : (!xls.array<9xi32>, index) -> !xls.array<2xi32>
// CHECK-DAG:       %[[VAL_12:.*]] = xls.array_concat %[[VAL_9]], %[[VAL_11]] : (!xls.array<2xi32>, !xls.array<2xi32>) -> !xls.array<4xi32>
// CHECK-DAG:       return %[[VAL_12]] : !xls.array<4xi32>
// CHECK-NEXT:    }
func.func @tensor_extract_slice_unroll(%arg0: tensor<3x3xi32>, %arg1: index) -> tensor<2x2xi32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[0, %arg1] [2, 2] [1, 1] : tensor<3x3xi32> to tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// CHECK-LABEL: @for
// CHECK:         xls.for
// CHECK-NEXT:    ^bb0(%[[INDVAR:.*]]: i32, %[[CARRY:.*]]: i32,
// CHECK-SAME:          %[[INVARIANT:.*]]: !xls.array<2xi32>):
// CHECK-DAG:       %[[V1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[V2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[V3:.*]] = arith.index_cast %[[INDVAR]] : i32 to index
// CHECK-DAG:       %[[V4:.*]] = xls.umul %[[V3]], %[[V2]] : index
// CHECK-DAG:       %[[V5:.*]] = xls.add %[[V1]], %[[V4]] : index
// CHECK-DAG:       %[[V6:.*]] = "xls.array_index"(%[[INVARIANT]], %[[V5]]) : (!xls.array<2xi32>, index) -> i32
// CHECK-DAG:       %[[V7:.*]] = arith.addi %[[CARRY]], %[[V6]] : i32
// CHECK-NEXT:      xls.yield %[[V7]] : i32
// CHECK-NEXT:    } {trip_count = 1024 : i64} : (i32, !xls.array<2xi32>) -> i32
func.func @for(%arg0: tensor<2xi32>) -> i32 attributes {xls = true} {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %c1 = arith.constant 1 : index
  %c0_i32 = arith.constant 0 : i32
  %0 = xls.for inits(%c0_i32) invariants(%arg0) {
  ^bb0(%indvar: i32, %carry: i32, %invariant: tensor<2xi32>):
    %1 = arith.index_cast %indvar : i32 to index
    %2 = tensor.extract %invariant[%1] : tensor<2xi32>
    %3 = arith.addi %carry, %2 : i32
    xls.yield %3 : i32
  } {trip_count = 1024 : i64} : (i32, tensor<2xi32>) -> i32
  return %0 : i32
}


func.func private @callee(%arg0: i8) -> i8
// CHECK-LABEL: vectorized_call
// CHECK-DAG: xls.for inits(%[[_:.*]]) invariants(%arg0) {
// CHECK-DAG: ^bb0(%indvar: i32, %carry: !xls.array<2xi8>, %invariant: !xls.array<2xi8>):
// CHECK-DAG:    %[[ELT:.*]] = "xls.array_index"(%invariant, %indvar) : (!xls.array<2xi8>, i32) -> i8
// CHECK-DAG:    %[[CALL:.*]] = func.call @callee(%[[ELT]]) : (i8) -> i8
// CHECK-DAG:    %[[UPDATE:.*]] = "xls.array_update"(%carry, %[[CALL]], %indvar) : (!xls.array<2xi8>, i8, i32) -> !xls.array<2xi8>
// CHECK-DAG:    xls.yield %[[UPDATE]] : !xls.array<2xi8>
// CHECK-DAG: } {trip_count = 2 : i64} : (!xls.array<2xi8>, !xls.array<2xi8>) -> !xls.array<2xi8>
func.func @vectorized_call(%arg0: tensor<2xi8>) -> tensor<2xi8> attributes {xls = true} {
  %0 = xls.vectorized_call @callee(%arg0) : (tensor<2xi8>) -> tensor<2xi8>
  return %0 : tensor<2xi8>
}

// CHECK-LABEL: @tensor_empty
// CHECK-NEXT: "xls.array_zero"
func.func @tensor_empty() -> tensor<4xi32> attributes {xls = true} {
  %0 = tensor.empty() : tensor<4xi32>
  return %0 : tensor<4xi32>
}

// CHECK-LABEL: @comparison_ops
// CHECK: xls.sge %{{.*}}, %{{.*}} : (i32, i32) -> i1
// CHECK: xls.sge %{{.*}}, %{{.*}} : (i32, i32) -> i1
func.func @comparison_ops(%arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi1> attributes {xls = true} {
  %0 = xls.sge %arg1, %arg2 : (tensor<2xi32>, tensor<2xi32>) -> tensor<2xi1>
  return %0 : tensor<2xi1>
}

// CHECK-LABEL: @array_index
// CHECK: "xls.array_index"(%arg0, %arg1) : (!xls.array<2x!xls.array<2xi32>>, i32) -> !xls.array<2xi32>
func.func @array_index(%arg: !xls.array<2xtensor<2xi32>>, %idx: i32) -> tensor<2xi32> attributes {xls = true} {
  %0 = "xls.array_index"(%arg, %idx) : (!xls.array<2xtensor<2xi32>>, i32) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

xls.chan @mychan : tensor<3xi8>
xls.eproc @eproc(%arg: i32) zeroinitializer {
  %0 = "xls.constant_tensor"() { value = dense<[0, 1, 2]> : tensor<3xi8> } : () -> tensor<3xi8>
  %tkn1 = "xls.after_all"() : () -> !xls.token
  // %5 = xls.send %4, %3, @mychan : !xls.array<3xi8>
  %tkn2 = xls.send %tkn1, %0, @mychan : tensor<3xi8>
  // %tkn_out, %result = xls.blocking_receive %4, @mychan : !xls.array<3xi8>
  %tkn3, %val = xls.blocking_receive %tkn1, @mychan : tensor<3xi8>
  xls.yield %arg : i32
}

// CHECK-LABEL: @call_dslx
func.func @call_dslx(%arg0: tensor<4xi32>) -> tensor<4xf32> attributes {xls = true} {
// CHECK-DAG: xls.for inits(%[[_:.*]]) invariants(%arg0) {
// CHECK-DAG: ^bb0(%indvar: i32, %carry: !xls.array<4xf32>, %invariant: !xls.array<4xi32>):
// CHECK-DAG:   %[[ELT:.*]] = "xls.array_index"(%invariant, %indvar) : (!xls.array<4xi32>, i32) -> i32
// CHECK-DAG:   %[[CALL:.*]] = xls.call_dslx "foo.x" : "f"(%[[ELT]]) : (i32) -> f32
// CHECK-DAG:   %[[UPDATE:.*]] = "xls.array_update"(%carry, %[[CALL]], %indvar) : (!xls.array<4xf32>, f32, i32) -> !xls.array<4xf32>
// CHECK-DAG:   xls.yield %[[UPDATE]] : !xls.array<4xf32>
// CHECK-DAG: } {trip_count = 4 : i64} : (!xls.array<4xf32>, !xls.array<4xi32>) -> !xls.array<4xf32>
  %0 = xls.call_dslx "foo.x": "f"(%arg0) : (tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @call_dslx_with_splat
func.func @call_dslx_with_splat(%arg0: tensor<4xi32>, %arg1: i32) -> tensor<4xf32> attributes {xls = true} {
// CHECK-DAG: xls.for inits(%[[_:.*]]) invariants(%arg0, %arg1) {
// CHECK-DAG: ^bb0(%indvar: i32, %carry: !xls.array<4xf32>, %invariant: !xls.array<4xi32>, %invariant_0: i32):
// CHECK-DAG:   %[[ELT:.*]] = "xls.array_index"(%invariant, %indvar) : (!xls.array<4xi32>, i32) -> i32
// CHECK-DAG:   %[[CALL:.*]] = xls.call_dslx "foo.x" : "f"(%[[ELT]], %invariant_0) : (i32, i32) -> f32
// CHECK-DAG:   %[[UPDATE:.*]] = "xls.array_update"(%carry, %[[CALL]], %indvar) : (!xls.array<4xf32>, f32, i32) -> !xls.array<4xf32>
// CHECK-DAG:   xls.yield %[[UPDATE]] : !xls.array<4xf32>
// CHECK-DAG: } {trip_count = 4 : i64} : (!xls.array<4xf32>, !xls.array<4xi32>, i32) -> !xls.array<4xf32>
  %0 = xls.call_dslx "foo.x": "f"(%arg0, %arg1) : (tensor<4xi32>, i32) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @select
func.func @select(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>, %arg2: tensor<2xi1>) -> tensor<2xi32> attributes {xls = true} {
// CHECK-DAG: %[[ARG2_0:.*]] = "xls.array_index_static"(%arg2) <{index = 0 : i64}> : (!xls.array<2xi1>) -> i1
// CHECK-DAG: %[[ARG0_0:.*]] = "xls.array_index_static"(%arg0) <{index = 0 : i64}> : (!xls.array<2xi32>) -> i32
// CHECK-DAG: %[[ARG1_0:.*]] = "xls.array_index_static"(%arg1) <{index = 0 : i64}> : (!xls.array<2xi32>) -> i32
// CHECK-DAG: %[[SELECT_0:.*]] = xls.sel %[[ARG2_0]] in [%[[ARG1_0]]] else %[[ARG0_0]] : (i1, [i32], i32) -> i32
// CHECK-DAG: %[[ARG2_1:.*]] = "xls.array_index_static"(%arg2) <{index = 1 : i64}> : (!xls.array<2xi1>) -> i1
// CHECK-DAG: %[[ARG0_1:.*]] = "xls.array_index_static"(%arg0) <{index = 1 : i64}> : (!xls.array<2xi32>) -> i32
// CHECK-DAG: %[[ARG1_1:.*]] = "xls.array_index_static"(%arg1) <{index = 1 : i64}> : (!xls.array<2xi32>) -> i32
// CHECK-DAG: %[[SELECT_1:.*]] = xls.sel %[[ARG2_1]] in [%[[ARG1_1]]] else %[[ARG0_1]] : (i1, [i32], i32) -> i32
// CHECK-DAG: xls.array %[[SELECT_0]], %[[SELECT_1]] : (i32, i32) -> !xls.array<2xi32>
  %0 = xls.sel %arg2 in [%arg1] else %arg0 : (tensor<2xi1>, [tensor<2xi32>], tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: @priority_select
func.func @priority_select(%arg0: tensor<2xi16>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> attributes {xls = true} {
  // CHECK: xls.priority_sel
  // CHECK: xls.priority_sel
  %0 = "xls.priority_sel"(%arg0, %arg1, %arg2) : (tensor<2xi16>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

// CHECK-LABEL: @one_hot_select
func.func @one_hot_select(%arg0: tensor<2xi16>, %arg1: tensor<2xi32>, %arg2: tensor<2xi32>) -> tensor<2xi32> attributes {xls = true} {
  // CHECK: xls.one_hot_sel
  // CHECK: xls.one_hot_sel
  %0 = "xls.one_hot_sel"(%arg0, %arg1, %arg2) : (tensor<2xi16>, tensor<2xi32>, tensor<2xi32>) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}
