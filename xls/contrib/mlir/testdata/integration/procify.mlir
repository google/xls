// RUN: xls_opt --test-convert-for-op-to-sproc --xls-lower %s \
// RUN: | FileCheck --check-prefix=CHECK-MLIR %s

// RUN: xls_opt --test-convert-for-op-to-sproc --xls-lower %s \
// RUN: | xls_translate --mlir-xls-to-xls --main-function=reduce_2_0 \
// RUN: > %t

// RUN: xls_opt --test-convert-for-op-to-sproc --xls-lower %s \
// RUN: | xls_translate --mlir-xls-to-verilog \
// RUN:   --main-function=reduce_2_0 -- --delay_model=asap7 \
// RUN:   --generator=pipeline --pipeline_stages=2 --worst_case_throughput=2 --reset=rst \
// RUN: | FileCheck --check-prefix=CHECK-VERILOG %s

// RUN: codegen_main %t \
// RUN:   --output_verilog_path %t.v --generator=pipeline --pipeline_stages=2 \
// RUN:   --worst_case_throughput=2 --reset=rst --delay_model=asap7 \
// RUN: && FileCheck --check-prefix=CHECK-VERILOG %s < %t.v

// CHECK-VERILOG: module {{.+}}(
// CHECK-VERILOG: endmodule

// RUN: eval_proc_main  %t \
// RUN:   --backend=ir_interpreter --ticks=10 --show_trace --logtostderr \
// RUN: |& FileCheck --check-prefix=CHECK-RUNTIME %s

// CHECK-RUNTIME: sum_next: 0, 0
// CHECK-RUNTIME: sum_next: 1, 1
// CHECK-RUNTIME: sum_next: 2, 3
// CHECK-RUNTIME: sum_next: 0, 3
// CHECK-RUNTIME: sum_next: 1, 4
// CHECK-RUNTIME: sum_next: 2, 6

xls.sproc @reduce() top attributes {boundary_channel_names = []} {
  spawns {
    xls.yield
  }
  next (%state: i32) zeroinitializer {
    %lb = arith.constant 0 : index
    %ub = arith.constant 3 : index
    %step = arith.constant 1 : index
    %sum_0 = arith.constant 0 : i32
    %sum = scf.for %iv = %lb to %ub step %step
        iter_args(%sum_iter = %state) -> (i32) {
      %iv_i32 = arith.index_cast %iv : index to i32
      %sum_next = arith.addi %iv_i32, %sum_iter : i32
      %tkn = xls.after_all : !xls.token
      xls.trace %tkn, "sum_next: {}, {}"(%iv_i32, %sum_next) : i32, i32
      scf.yield %sum_next : i32
    }
    xls.yield %sum : i32
  }
}

// CHECK-MLIR:module {
// CHECK-MLIR:  xls.chan @for_arg_0 : i32
// CHECK-MLIR:  xls.chan @for_result_0 : i32
// CHECK-MLIR:  xls.chan @body_arg_0 : i32
// CHECK-MLIR:  xls.chan @body_arg_1 : i32
// CHECK-MLIR:  xls.chan @body_result_0 : i32
// CHECK-MLIR:  xls.eproc @reduce_for_body_0_2(%arg0: i32) zeroinitializer {
// CHECK-MLIR:    %0 = xls.after_all  : !xls.token
// CHECK-MLIR:    %tkn_out, %result = xls.blocking_receive %0, @body_arg_0 : i32
// CHECK-MLIR:    %tkn_out_0, %result_1 = xls.blocking_receive %0, @body_arg_1 : i32
// CHECK-MLIR:    %1 = xls.add %result, %result_1 : i32
// CHECK-MLIR:    %2 = xls.trace %0, "sum_next: {}, {}"(%result, %1) : i32, i32
// CHECK-MLIR:    %3 = xls.send %0, %1, @body_result_0 : i32
// CHECK-MLIR:    xls.yield %arg0 : i32
// CHECK-MLIR:  }
// CHECK-MLIR:  xls.eproc @reduce_for_controller_1_1(%arg0: i32) zeroinitializer {
// CHECK-MLIR:    %0 = "xls.constant_scalar"() <{value = 0 : i32}> : () -> i32
// CHECK-MLIR:    %1 = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
// CHECK-MLIR:    %2 = "xls.constant_scalar"() <{value = 3 : i32}> : () -> i32
// CHECK-MLIR:    %3 = xls.eq %arg0, %0 : (i32, i32) -> i1
// CHECK-MLIR:    %4 = xls.not %3 : i1
// CHECK-MLIR:    %5 = xls.after_all  : !xls.token
// CHECK-MLIR:    %tkn_out, %result = xls.blocking_receive %5, %3, @for_arg_0 : i32
// CHECK-MLIR:    %tkn_out_0, %result_1 = xls.blocking_receive %5, %4, @body_result_0 : i32
// CHECK-MLIR:    %6 = xls.sel %3 in [%result_1] else  %result : (i1, [i32], i32) -> i32
// CHECK-MLIR:    %7 = xls.eq %arg0, %2 : (i32, i32) -> i1
// CHECK-MLIR:    %8 = xls.not %7 : i1
// CHECK-MLIR:    %9 = xls.send %5, %6, %7, @for_result_0 : i32
// CHECK-MLIR:    %10 = xls.send %5, %6, %8, @body_arg_1 : i32
// CHECK-MLIR:    %11 = xls.send %5, %arg0, %8, @body_arg_0 : i32
// CHECK-MLIR:    %12 = xls.add %arg0, %1 : i32
// CHECK-MLIR:    %13 = xls.sel %7 in  [%12] else %0 : (i1, [i32], i32) -> i32
// CHECK-MLIR:    xls.yield %13 : i32
// CHECK-MLIR:  }
// CHECK-MLIR:  xls.eproc @reduce_2_0(%arg0: i32) zeroinitializer attributes {min_pipeline_stages = 2 : i64} {
// CHECK-MLIR:    %0 = xls.after_all  : !xls.token
// CHECK-MLIR:    %1 = xls.send %0, %arg0, @for_arg_0 : i32
// CHECK-MLIR:    %2 = xls.after_all %1 : !xls.token
// CHECK-MLIR:    %tkn_out, %result = xls.blocking_receive %2, @for_result_0 : i32
// CHECK-MLIR:    xls.yield %result : i32
// CHECK-MLIR:  }
// CHECK-MLIR:}
