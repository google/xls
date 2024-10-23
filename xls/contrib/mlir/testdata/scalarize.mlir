// RUN: xls/contrib/mlir/xls_opt -scalarize -canonicalize %s 2>&1 | FileCheck %s

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
// CHECK: return %arg0 : !xls.array<2 x i32>
func.func @empty(%arg0: tensor<2xi32>, %arg1: i32) -> tensor<2xi32> attributes { "xls" = true } {
  return %arg0 : tensor<2xi32>
}

// CHECK-LABEL: @tensor_insert
// CHECK: constant
// CHECK-SAME: 13
// CHECK-NEXT: array_update
// CHECK-SAME: xls.array<14 x i32>
func.func @tensor_insert(%arg0: tensor<2x7xi32>, %arg1: i32) -> tensor<2x7xi32> attributes { "xls" = true } {
  %0 = arith.constant 1 : index
  %1 = arith.constant 6 : index
  %2 = tensor.insert %arg1 into %arg0[%0, %1] : tensor<2x7xi32>
  return %2 : tensor<2x7xi32>
}

// CHECK-LABEL: @tensor_concat
// CHECK: xls.array_concat
// CHECK-SAME: xls.array<28 x f32>
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

// CHECK-LABEL: @tensor_extract_slice_unroll
// CHECK: xls.array_index
// CHECK: xls.array_index
// CHECK: xls.array_index
// CHECK: xls.array_index
// CHECK-NOT: xls.array_index
// CHECK: xls.array {{.*}} -> !xls.array<4 x i32>
// CHECK: return %{{.*}} : !xls.array<4 x i32>
func.func @tensor_extract_slice_unroll(%arg0: tensor<3x3xi32>, %arg1: index) -> tensor<2x2xi32> attributes { "xls" = true } {
  %0 = tensor.extract_slice %arg0[0, %arg1] [2, 2] [1, 1] : tensor<3x3xi32> to tensor<2x2xi32>
  return %0 : tensor<2x2xi32>
}

// CHECK-LABEL: @for
// CHECK:         xls.for
// CHECK-NEXT:    ^bb0(%[[INDVAR:.*]]: i32, %[[CARRY:.*]]: i32,
// CHECK-SAME:          %[[INVARIANT:.*]]: !xls.array<2 x i32>):
// CHECK-DAG:       %[[V1:.*]] = arith.constant 0 : index
// CHECK-DAG:       %[[V2:.*]] = arith.constant 1 : index
// CHECK-DAG:       %[[V3:.*]] = arith.index_cast %[[INDVAR]] : i32 to index 
// CHECK-DAG:       %[[V4:.*]] = xls.umul %[[V3]], %[[V2]] : index
// CHECK-DAG:       %[[V5:.*]] = xls.add %[[V1]], %[[V4]] : index
// CHECK-DAG:       %[[V6:.*]] = "xls.array_index"(%[[INVARIANT]], %[[V5]]) : (!xls.array<2 x i32>, index) -> i32
// CHECK-DAG:       %[[V7:.*]] = arith.addi %[[CARRY]], %[[V6]] : i32
// CHECK-NEXT:      xls.yield %[[V7]] : i32
// CHECK-NEXT:    } {trip_count = 1024 : i64} : (i32, !xls.array<2 x i32>) -> i32
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
// CHECK: %0 = "xls.array_index_static"(%arg0) <{index = 0 : i64}> : (!xls.array<2 x i8>) -> i8
// CHECK: %1 = call @callee(%0) : (i8) -> i8
// CHECK: %2 = "xls.array_index_static"(%arg0) <{index = 1 : i64}> : (!xls.array<2 x i8>) -> i8
// CHECK: %3 = call @callee(%2) : (i8) -> i8
// CHECK: %4 = xls.array %1, %3 : (i8, i8) -> !xls.array<2 x i8>
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

// CHECK-LABEL: @array_index
// CHECK: "xls.array_index"(%arg0, %arg1) : (!xls.array<2 x !xls.array<2 x i32>>, i32) -> !xls.array<2 x i32>
func.func @array_index(%arg: !xls.array<2 x tensor<2xi32>>, %idx: i32) -> tensor<2xi32> attributes {xls = true} {
  %0 = "xls.array_index"(%arg, %idx) : (!xls.array<2 x tensor<2xi32>>, i32) -> tensor<2xi32>
  return %0 : tensor<2xi32>
}

xls.chan @mychan : tensor<3xi8>
xls.eproc @eproc(%arg: i32) zeroinitializer {
  %0 = "xls.constant_tensor"() { value = dense<[0, 1, 2]> : tensor<3xi8> } : () -> tensor<3xi8>
  %tkn1 = "xls.after_all"() : () -> !xls.token
  // %5 = xls.send %4, %3, @mychan : !xls.array<3 x i8>
  %tkn2 = xls.send %tkn1, %0, @mychan : tensor<3xi8>
  // %tkn_out, %result = xls.blocking_receive %4, @mychan : !xls.array<3 x i8>
  %tkn3, %val = xls.blocking_receive %tkn1, @mychan : tensor<3xi8>
  xls.yield %arg : i32
}

// CHECK-LABEL: @call_dslx
func.func @call_dslx(%arg0: tensor<4xi32>) -> tensor<4xf32> attributes {xls = true} {
  // CHECK: xls.call_dslx "foo.x" : "f"(%0) : (i32) -> f32
  // CHECK: xls.call_dslx "foo.x" : "f"
  // CHECK: xls.call_dslx "foo.x" : "f"
  // CHECK: xls.call_dslx "foo.x" : "f"
  %0 = xls.call_dslx "foo.x": "f"(%arg0) : (tensor<4xi32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: @call_dslx_with_splat
func.func @call_dslx_with_splat(%arg0: tensor<4xi32>, %arg1: i32) -> tensor<4xf32> attributes {xls = true} {
  // CHECK: xls.call_dslx "foo.x" : "f"(%0, %arg1) : (i32, i32) -> f32
  // CHECK: xls.call_dslx "foo.x" : "f"
  // CHECK: xls.call_dslx "foo.x" : "f"
  // CHECK: xls.call_dslx "foo.x" : "f"
  %0 = xls.call_dslx "foo.x": "f"(%arg0, %arg1) : (tensor<4xi32>, i32) -> tensor<4xf32>
  return %0 : tensor<4xf32>
}
