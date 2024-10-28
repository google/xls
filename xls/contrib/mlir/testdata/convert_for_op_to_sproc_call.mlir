// RUN: xls/contrib/mlir/xls_opt -test-convert-for-op-to-sproc -split-input-file -allow-unregistered-dialect %s | FileCheck %s

// CHECK:module @reduce_zeroes {
// CHECK:  xls.sproc @reduce_zeroes_for_body(%arg0: !xls.schan<index, in>, %arg1: !xls.schan<i32, in>, %arg2: !xls.schan<i32, out>) {
// CHECK:    spawns {
// CHECK:      xls.yield %arg0, %arg1, %arg2 : !xls.schan<index, in>, !xls.schan<i32, in>, !xls.schan<i32, out>
// CHECK:    }
// CHECK:    next (%arg0: !xls.schan<index, in>, %arg1: !xls.schan<i32, in>, %arg2: !xls.schan<i32, out>, %arg3: i32) zeroinitializer {
// CHECK:      %0 = xls.after_all  : !xls.token
// CHECK:      %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<index, in>) -> (!xls.token, index)
// CHECK:      %tkn_out_0, %result_1 = xls.sblocking_receive %0, %arg1 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:      %1 = arith.addi %result_1, %result_1 : i32
// CHECK:      %2 = xls.after_all  : !xls.token
// CHECK:      %3 = xls.ssend %2, %1, %arg2 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:      xls.yield %arg3 : i32
// CHECK:    }
// CHECK:  }
// CHECK:  xls.sproc @reduce_zeroes_for_controller(%arg0: !xls.schan<i32, in>, %arg1: !xls.schan<i32, out>) {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<index>("body_arg_0")
// CHECK:      %out_0, %in_1 = xls.schan<i32>("body_arg_1")
// CHECK:      %out_2, %in_3 = xls.schan<i32>("body_result_0")
// CHECK:      xls.spawn @reduce_zeroes_for_body(%in, %in_1, %out_2) : !xls.schan<index, in>, !xls.schan<i32, in>, !xls.schan<i32, out>
// CHECK:      xls.yield %arg0, %arg1, %out, %out_0, %in_3 : !xls.schan<i32, in>, !xls.schan<i32, out>, !xls.schan<index, out>, !xls.schan<i32, out>, !xls.schan<i32, in>
// CHECK:    }
// CHECK:    next (%arg0: !xls.schan<i32, in>, %arg1: !xls.schan<i32, out>, %arg2: !xls.schan<index, out>, %arg3: !xls.schan<i32, out>, %arg4: !xls.schan<i32, in>, %arg5: i32) zeroinitializer {
// CHECK:      %c0_i32 = arith.constant 0 : i32
// CHECK:      %c1_i32 = arith.constant 1 : i32
// CHECK:      %c2_i32 = arith.constant 2 : i32
// CHECK:      %0 = arith.cmpi eq, %arg5, %c0_i32 : i32
// CHECK:      %1 = scf.if %0 -> (i32) {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %tkn_out, %result = xls.sblocking_receive %4, %arg0 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:        scf.yield %result : i32
// CHECK:      } else {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %tkn_out, %result = xls.sblocking_receive %4, %arg4 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:        scf.yield %result : i32
// CHECK:      }
// CHECK:      %2 = arith.cmpi eq, %arg5, %c2_i32 : i32
// CHECK:      %3 = scf.if %2 -> (i32) {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %5 = xls.ssend %4, %1, %arg1 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:        scf.yield %c0_i32 : i32
// CHECK:      } else {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %5 = xls.ssend %4, %1, %arg3 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:        %6 = arith.index_cast %arg5 : i32 to index
// CHECK:        %7 = xls.after_all  : !xls.token
// CHECK:        %8 = xls.ssend %7, %6, %arg2 : (!xls.token, index, !xls.schan<index, out>) -> !xls.token
// CHECK:        %9 = arith.addi %arg5, %c1_i32 : i32
// CHECK:        scf.yield %9 : i32
// CHECK:      }
// CHECK:      xls.yield %3 : i32
// CHECK:    }
// CHECK:  }
// CHECK:  xls.sproc @reduce_zeroes() top attributes {boundary_channel_names = [], min_pipeline_stages = 2 : i64} {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<i32>("for_arg_0")
// CHECK:      %out_0, %in_1 = xls.schan<i32>("for_result_0")
// CHECK:      xls.spawn @reduce_zeroes_for_controller(%in, %out_0) : !xls.schan<i32, in>, !xls.schan<i32, out>
// CHECK:      xls.yield %out, %in_1 : !xls.schan<i32, out>, !xls.schan<i32, in>
// CHECK:    }
// CHECK:    next (%arg0: !xls.schan<i32, out>, %arg1: !xls.schan<i32, in>, %arg2: i32) zeroinitializer {
// CHECK:      %c0 = arith.constant 0 : index
// CHECK:      %c2 = arith.constant 2 : index
// CHECK:      %c1 = arith.constant 1 : index
// CHECK:      %c0_i32 = arith.constant 0 : i32
// CHECK:      %0 = xls.after_all  : !xls.token
// CHECK:      %1 = xls.ssend %0, %c0_i32, %arg0 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:      %2 = xls.after_all %1 : !xls.token
// CHECK:      %tkn_out, %result = xls.sblocking_receive %2, %arg1 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:      xls.yield %result : i32
// CHECK:    }
// CHECK:  }
// CHECK:}
module @reduce_zeroes {
  xls.sproc @reduce_zeroes() top attributes {boundary_channel_names = []} {
    spawns {
      xls.yield
    }
    next (%state: i32) zeroinitializer {
      %lb = arith.constant 0 : index
      %ub = arith.constant 2 : index
      %step = arith.constant 1 : index
      %sum_0 = arith.constant 0 : i32
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (i32) {
        %sum_next = arith.addi %sum_iter, %sum_iter : i32
        scf.yield %sum_next : i32
      }
      xls.yield %sum : i32
    }
  }
}

// -----

// CHECK:module @reduce_invariants {
// CHECK:  xls.sproc @reduce_invariants_for_body(%arg0: !xls.schan<index, in>, %arg1: !xls.schan<i32, in>, %arg2: !xls.schan<i32, in>, %arg3: !xls.schan<i32, out>) {
// CHECK:    spawns {
// CHECK:      xls.yield %arg0, %arg1, %arg2, %arg3 : !xls.schan<index, in>, !xls.schan<i32, in>, !xls.schan<i32, in>, !xls.schan<i32, out>
// CHECK:    }
// CHECK:    next (%arg0: !xls.schan<index, in>, %arg1: !xls.schan<i32, in>, %arg2: !xls.schan<i32, in>, %arg3: !xls.schan<i32, out>, %arg4: i32) zeroinitializer {
// CHECK:      %0 = xls.after_all  : !xls.token
// CHECK:      %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<index, in>) -> (!xls.token, index)
// CHECK:      %tkn_out_0, %result_1 = xls.sblocking_receive %0, %arg1 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:      %tkn_out_2, %result_3 = xls.sblocking_receive %0, %arg2 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:      %c1_i32 = arith.constant 1 : i32
// CHECK:      %1 = arith.addi %result_3, %result_1 : i32
// CHECK:      %2 = arith.addi %1, %c1_i32 : i32
// CHECK:      %3 = xls.after_all  : !xls.token
// CHECK:      %4 = xls.ssend %3, %2, %arg3 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:      xls.yield %arg4 : i32
// CHECK:    }
// CHECK:  }
// CHECK:  xls.sproc @reduce_invariants_for_controller(%arg0: !xls.schan<i32, in>, %arg1: !xls.schan<i32, in>, %arg2: !xls.schan<i32, out>) {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<index>("body_arg_0")
// CHECK:      %out_0, %in_1 = xls.schan<i32>("body_arg_1")
// CHECK:      %out_2, %in_3 = xls.schan<i32>("body_arg_2")
// CHECK:      %out_4, %in_5 = xls.schan<i32>("body_result_0")
// CHECK:      xls.spawn @reduce_invariants_for_body(%in, %in_1, %in_3, %out_4) : !xls.schan<index, in>, !xls.schan<i32, in>, !xls.schan<i32, in>, !xls.schan<i32, out>
// CHECK:      xls.yield %arg0, %arg1, %arg2, %out, %out_0, %out_2, %in_5 : !xls.schan<i32, in>, !xls.schan<i32, in>, !xls.schan<i32, out>, !xls.schan<index, out>, !xls.schan<i32, out>, !xls.schan<i32, out>, !xls.schan<i32, in>
// CHECK:    }
// CHECK:    next (%arg0: !xls.schan<i32, in>, %arg1: !xls.schan<i32, in>, %arg2: !xls.schan<i32, out>, %arg3: !xls.schan<index, out>, %arg4: !xls.schan<i32, out>, %arg5: !xls.schan<i32, out>, %arg6: !xls.schan<i32, in>, %arg7: i32, %arg8: i32) zeroinitializer {
// CHECK:      %c0_i32 = arith.constant 0 : i32
// CHECK:      %c1_i32 = arith.constant 1 : i32
// CHECK:      %c2_i32 = arith.constant 2 : i32
// CHECK:      %0 = arith.cmpi eq, %arg7, %c0_i32 : i32
// CHECK:      %1:2 = scf.if %0 -> (i32, i32) {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %tkn_out, %result = xls.sblocking_receive %4, %arg0 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:        %5 = xls.after_all  : !xls.token
// CHECK:        %tkn_out_0, %result_1 = xls.sblocking_receive %5, %arg1 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:        scf.yield %result, %result_1 : i32, i32
// CHECK:      } else {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %tkn_out, %result = xls.sblocking_receive %4, %arg6 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:        scf.yield %result, %arg8 : i32, i32
// CHECK:      }
// CHECK:      %2 = arith.cmpi eq, %arg7, %c2_i32 : i32
// CHECK:      %3 = scf.if %2 -> (i32) {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %5 = xls.ssend %4, %1#0, %arg2 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:        scf.yield %c0_i32 : i32
// CHECK:      } else {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %5 = xls.ssend %4, %1#0, %arg4 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:        %6 = xls.after_all  : !xls.token
// CHECK:        %7 = xls.ssend %6, %1#1, %arg5 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:        %8 = arith.index_cast %arg7 : i32 to index
// CHECK:        %9 = xls.after_all  : !xls.token
// CHECK:        %10 = xls.ssend %9, %8, %arg3 : (!xls.token, index, !xls.schan<index, out>) -> !xls.token
// CHECK:        %11 = arith.addi %arg7, %c1_i32 : i32
// CHECK:        scf.yield %11 : i32
// CHECK:      }
// CHECK:      xls.yield %3, %1#1 : i32, i32
// CHECK:    }
// CHECK:  }
// CHECK:  xls.sproc @reduce_invariants() top attributes {boundary_channel_names = [], min_pipeline_stages = 2 : i64} {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<i32>("for_arg_0")
// CHECK:      %out_0, %in_1 = xls.schan<i32>("for_arg_1")
// CHECK:      %out_2, %in_3 = xls.schan<i32>("for_result_0")
// CHECK:      xls.spawn @reduce_invariants_for_controller(%in, %in_1, %out_2) : !xls.schan<i32, in>, !xls.schan<i32, in>, !xls.schan<i32, out>
// CHECK:      xls.yield %out, %out_0, %in_3 : !xls.schan<i32, out>, !xls.schan<i32, out>, !xls.schan<i32, in>
// CHECK:    }
// CHECK:    next (%arg0: !xls.schan<i32, out>, %arg1: !xls.schan<i32, out>, %arg2: !xls.schan<i32, in>, %arg3: i32) zeroinitializer {
// CHECK:      %c0 = arith.constant 0 : index
// CHECK:      %c2 = arith.constant 2 : index
// CHECK:      %c1 = arith.constant 1 : index
// CHECK:      %c1_i32 = arith.constant 1 : i32
// CHECK:      %c0_i32 = arith.constant 0 : i32
// CHECK:      %0 = arith.addi %c0_i32, %c0_i32 : i32
// CHECK:      %1 = xls.after_all  : !xls.token
// CHECK:      %2 = xls.ssend %1, %c0_i32, %arg0 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:      %3 = xls.ssend %1, %0, %arg1 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:      %4 = xls.after_all %2, %3 : !xls.token
// CHECK:      %tkn_out, %result = xls.sblocking_receive %4, %arg2 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:      xls.yield %result : i32
// CHECK:    }
// CHECK:  }
// CHECK:}
module @reduce_invariants {
  xls.sproc @reduce_invariants() top attributes {boundary_channel_names = []} {
    spawns {
      xls.yield
    }
    next (%state: i32) zeroinitializer {
      %lb = arith.constant 0 : index
      %ub = arith.constant 2 : index
      %step = arith.constant 1 : index
      %step_i32 = arith.constant 1 : i32
      %sum_0 = arith.constant 0 : i32
      %sum_1 = arith.addi %sum_0, %sum_0 : i32
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (i32) {
        %sum_next = arith.addi %sum_1, %sum_iter : i32
        %sum_next2 = arith.addi %sum_next, %step_i32 : i32
        scf.yield %sum_next2 : i32
      }
      xls.yield %sum : i32
    }
  }
}

// -----

// CHECK:module @reduce_clone {
// CHECK:  xls.sproc @reduce_clone_for_body(%arg0: !xls.schan<index, in>, %arg1: !xls.schan<i32, in>, %arg2: !xls.schan<i32, out>) {
// CHECK:    spawns {
// CHECK:      xls.yield %arg0, %arg1, %arg2 : !xls.schan<index, in>, !xls.schan<i32, in>, !xls.schan<i32, out>
// CHECK:    }
// CHECK:    next (%arg0: !xls.schan<index, in>, %arg1: !xls.schan<i32, in>, %arg2: !xls.schan<i32, out>, %arg3: i32) zeroinitializer {
// CHECK:      %0 = xls.after_all  : !xls.token
// CHECK:      %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<index, in>) -> (!xls.token, index)
// CHECK:      %tkn_out_0, %result_1 = xls.sblocking_receive %0, %arg1 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:      %c0_i32 = arith.constant 0 : i32
// CHECK:      %1 = arith.addi %result_1, %c0_i32 : i32
// CHECK:      %2 = xls.after_all  : !xls.token
// CHECK:      %3 = xls.ssend %2, %1, %arg2 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:      xls.yield %arg3 : i32
// CHECK:    }
// CHECK:  }
// CHECK:  xls.sproc @reduce_clone_for_controller(%arg0: !xls.schan<i32, in>, %arg1: !xls.schan<i32, out>) {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<index>("body_arg_0")
// CHECK:      %out_0, %in_1 = xls.schan<i32>("body_arg_1")
// CHECK:      %out_2, %in_3 = xls.schan<i32>("body_result_0")
// CHECK:      xls.spawn @reduce_clone_for_body(%in, %in_1, %out_2) : !xls.schan<index, in>, !xls.schan<i32, in>, !xls.schan<i32, out>
// CHECK:      xls.yield %arg0, %arg1, %out, %out_0, %in_3 : !xls.schan<i32, in>, !xls.schan<i32, out>, !xls.schan<index, out>, !xls.schan<i32, out>, !xls.schan<i32, in>
// CHECK:    }
// CHECK:    next (%arg0: !xls.schan<i32, in>, %arg1: !xls.schan<i32, out>, %arg2: !xls.schan<index, out>, %arg3: !xls.schan<i32, out>, %arg4: !xls.schan<i32, in>, %arg5: i32) zeroinitializer {
// CHECK:      %c0_i32 = arith.constant 0 : i32
// CHECK:      %c1_i32 = arith.constant 1 : i32
// CHECK:      %c2_i32 = arith.constant 2 : i32
// CHECK:      %0 = arith.cmpi eq, %arg5, %c0_i32 : i32
// CHECK:      %1 = scf.if %0 -> (i32) {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %tkn_out, %result = xls.sblocking_receive %4, %arg0 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:        scf.yield %result : i32
// CHECK:      } else {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %tkn_out, %result = xls.sblocking_receive %4, %arg4 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:        scf.yield %result : i32
// CHECK:      }
// CHECK:      %2 = arith.cmpi eq, %arg5, %c2_i32 : i32
// CHECK:      %3 = scf.if %2 -> (i32) {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %5 = xls.ssend %4, %1, %arg1 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:        scf.yield %c0_i32 : i32
// CHECK:      } else {
// CHECK:        %4 = xls.after_all  : !xls.token
// CHECK:        %5 = xls.ssend %4, %1, %arg3 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:        %6 = arith.index_cast %arg5 : i32 to index
// CHECK:        %7 = xls.after_all  : !xls.token
// CHECK:        %8 = xls.ssend %7, %6, %arg2 : (!xls.token, index, !xls.schan<index, out>) -> !xls.token
// CHECK:        %9 = arith.addi %arg5, %c1_i32 : i32
// CHECK:        scf.yield %9 : i32
// CHECK:      }
// CHECK:      xls.yield %3 : i32
// CHECK:    }
// CHECK:  }
// CHECK:  xls.sproc @reduce_clone() top attributes {boundary_channel_names = [], min_pipeline_stages = 2 : i64} {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<i32>("for_arg_0")
// CHECK:      %out_0, %in_1 = xls.schan<i32>("for_result_0")
// CHECK:      xls.spawn @reduce_clone_for_controller(%in, %out_0) : !xls.schan<i32, in>, !xls.schan<i32, out>
// CHECK:      xls.yield %out, %in_1 : !xls.schan<i32, out>, !xls.schan<i32, in>
// CHECK:    }
// CHECK:    next (%arg0: !xls.schan<i32, out>, %arg1: !xls.schan<i32, in>, %arg2: i32) zeroinitializer {
// CHECK:      %c0 = arith.constant 0 : index
// CHECK:      %c2 = arith.constant 2 : index
// CHECK:      %c1 = arith.constant 1 : index
// CHECK:      %c0_i32 = arith.constant 0 : i32
// CHECK:      %0 = xls.after_all  : !xls.token
// CHECK:      %1 = xls.ssend %0, %c0_i32, %arg0 : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
// CHECK:      %2 = xls.after_all %1 : !xls.token
// CHECK:      %tkn_out, %result = xls.sblocking_receive %2, %arg1 : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
// CHECK:      xls.yield %result : i32
// CHECK:    }
// CHECK:  }
// CHECK:}
module @reduce_clone {
  xls.sproc @reduce_clone() top attributes {boundary_channel_names = []} {
    spawns {
      xls.yield
    }
    next (%state: i32) zeroinitializer {
      %lb = arith.constant 0 : index
      %ub = arith.constant 2 : index
      %step = arith.constant 1 : index
      %sum_0 = arith.constant 0 : i32
      %sum = scf.for %iv = %lb to %ub step %step
          iter_args(%sum_iter = %sum_0) -> (i32) {
        %sum_next = arith.addi %sum_iter, %sum_0 : i32
        scf.yield %sum_next : i32
      }
      xls.yield %sum : i32
    }
  }
}
