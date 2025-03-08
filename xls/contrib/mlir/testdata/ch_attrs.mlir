// RUN: xls_translate --mlir-xls-to-xls %s > %t
// RUN: xls_translate --xls-to-mlir-xls %t | FileCheck %s

// Validate that channel/fifo attributes are correctly maintained during
// a round-trip conversion.

// CHECK-LABEL: xls.chan @chan0
// CHECK-NOT:     input_flop_kind
// CHECK-NOT:     output_flop_kind
// CHECK-SAME:    fifo_depth = 1
// CHECK-SAME:    bypass = true
// CHECK-SAME:    register_push_outputs = true
// CHECK-SAME:    register_pop_outputs = false
xls.chan @chan0 {fifo_config = #xls.fifo_config<fifo_depth = 1, bypass = true, register_push_outputs = true, register_pop_outputs = false>, send_supported = false} : i1

// CHECK-LABEL: xls.chan @chan1
// CHECK-NOT:     fifo_config
// CHECK-SAME:    input_flop_kind = #xls<flop_kind skid>
// CHECK-SAME:    output_flop_kind = #xls<flop_kind none>
xls.chan @chan1 {input_flop_kind = #xls<flop_kind skid>, output_flop_kind = #xls<flop_kind none>, send_supported = false} : i32

// CHECK-LABEL: xls.chan @chan2
// CHECK-NOT:    fifo_config
// CHECK-NOT:    input_flop_kind
// CHECK-NOT:    output_flop_kind
xls.chan @chan2 {send_supported = false} : i32

xls.eproc @foo() zeroinitializer {
  %tok = xls.after_all : !xls.token
  %tok0, %result0 = xls.blocking_receive %tok, @chan0 : i1
  %tok1, %result1 = xls.blocking_receive %tok, @chan1 : i32
  %tok2, %result2 = xls.blocking_receive %tok, @chan2 : i32
  xls.yield
}
