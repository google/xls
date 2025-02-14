// RUN: xls_translate --mlir-xls-to-xls %s -- 2>&1 | FileCheck %s

// CHECK-LABEL: chan ch_inp
// CHECK-SAME:    kind=streaming
// CHECK-SAME:    ops=receive_only
xls.chan @ch_inp {send_supported = false} : i32

// CHECK-LABEL: chan ch_out
// CHECK-SAME:    kind=streaming
// CHECK-SAME:    ops=send_only
// CHECK-SAME:    fifo_depth=1
// CHECK-SAME:    bypass=true
// CHECK-SAME:    register_push_outputs=true
// CHECK-SAME:    register_pop_outputs=false
xls.chan @ch_out {
  fifo_config = #xls.fifo_config<fifo_depth = 1, bypass = true, register_push_outputs = true, register_pop_outputs = false>,
  recv_supported = false
} : i32

// CHECK: top proc eproc
xls.eproc @eproc() zeroinitializer {
  %tok0 = xls.after_all : !xls.token
  %tok1, %val = xls.blocking_receive %tok0, @ch_inp : i32
  xls.send %tok1, %val, @ch_out : i32
  xls.yield
}

