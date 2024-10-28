// RUN: xls/contrib/mlir/xls_opt -procify-loops -split-input-file %s 2>&1 | FileCheck %s

// The main rewrite is tested in convert_for_op_to_sproc_call.mlir and integration/procify.mlir.
// This file just tests that we apply to xls.unroll attributes.

// CHECK: xls.sproc @reduce_for_body
// CHECK: xls.sproc @reduce_for_controller
// CHECK: xls.sproc @reduce
xls.sproc @reduce() top attributes {boundary_channel_names = []} {
  spawns {
    xls.yield
  }
  next (%state: i32) zeroinitializer {
    %lb = arith.constant 0 : index
    %ub = arith.constant 1024 : index
    %step = arith.constant 1 : index
    // Initial sum set to 0.
    %sum_0 = arith.constant 0 : i32
    // iter_args binds initial values to the loop's region arguments.
    %sum = scf.for %iv = %lb to %ub step %step
        iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum_next = arith.addi %sum_iter, %state : i32
      // Yield current iteration sum to next iteration %sum_iter or to %sum
      // if final iteration.
      scf.yield %sum_next : i32
    } {xls.unroll = false}
    xls.yield %sum : i32
  }
}

// -----

// CHECK: xls.sproc @reduce_nested_for_body_for_body
// CHECK: xls.sproc @reduce_nested_for_body_for_controller
// CHECK: xls.sproc @reduce_nested_for_body
// CHECK: xls.sproc @reduce_nested_for_controller
// CHECK: xls.sproc @reduce_nested
xls.sproc @reduce_nested() top attributes {boundary_channel_names = []} {
  spawns {
    xls.yield
  }
  next (%state: i32) zeroinitializer {
    %addend = arith.constant 1 : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 1024 : index
    %step = arith.constant 1 : index
    // Initial sum set to 0.
    %sum_0 = arith.constant 0 : i32
    // iter_args binds initial values to the loop's region arguments.
    %sum = scf.for %iv = %lb to %ub step %step
        iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum2 = scf.for %iv2 = %lb to %ub step %step
          iter_args(%sum_iter2 = %sum_iter) -> (i32) {
        %sum3 = scf.for %iv3 = %lb to %ub step %step
            iter_args(%sum_iter3 = %sum_iter2) -> (i32) {
          %sum_next = arith.addi %sum_iter2, %addend : i32
          scf.yield %sum_next : i32
        } {xls.unroll = true}
        scf.yield %sum3 : i32
      } {xls.unroll = false}
      scf.yield %sum2 : i32
    } {xls.unroll = false}
    xls.yield %sum : i32
  }
}

// -----

// CHECK-NOT: xls.sproc
// CHECK: xls.sproc @reduce_coalescable_for_body_for_body
// CHECK: xls.sproc @reduce_coalescable_for_body_for_controller
// CHECK: xls.sproc @reduce_coalescable_for_body
// CHECK: xls.sproc @reduce_coalescable_for_controller
// CHECK: xls.sproc @reduce_coalescable
xls.sproc @reduce_coalescable() top attributes {boundary_channel_names = []} {
  spawns {
    xls.yield
  }
  next (%state: i32) zeroinitializer {
    %addend = arith.constant 1 : i32
    %lb = arith.constant 0 : index
    %ub = arith.constant 1024 : index
    %step = arith.constant 1 : index
    // Initial sum set to 0.
    %sum_0 = arith.constant 0 : i32
    // iter_args binds initial values to the loop's region arguments.
    %sum = scf.for %iv = %lb to %ub step %step
        iter_args(%sum_iter = %sum_0) -> (i32) {
      %sum2 = scf.for %iv2 = %lb to %ub step %step
          iter_args(%sum_iter2 = %sum_iter) -> (i32) {
        %sum3 = scf.for %iv3 = %lb to %ub step %step
            iter_args(%sum_iter3 = %sum_iter2) -> (i32) {
          %sum_next = arith.addi %sum_iter2, %addend : i32
          scf.yield %sum_next : i32
        } {xls.unroll = false}
        scf.yield %sum3 : i32
      } {xls.unroll = false}
      scf.yield %sum2 : i32
    } {xls.unroll = true}
    xls.yield %sum : i32
  }
}
