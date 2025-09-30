// RUN: xls_opt -elaborate-procs -split-input-file %s 2>&1 | FileCheck %s
// CHECK-LABEL: xls.chan @req : i32
// CHECK-NEXT:  xls.chan @resp {fifo_config = #xls.fifo_config<fifo_depth = 1, bypass = true, register_push_outputs = true, register_pop_outputs = false>} : i32
// CHECK-NEXT:  xls.chan @rom1_req : i32
// CHECK-NEXT:  xls.chan @rom1_resp : i32
// CHECK-NEXT:  xls.eproc @rom(%arg0: i32) zeroinitializer discardable {
// CHECK-NEXT:    %0 = xls.after_all  : !xls.token
// CHECK-NEXT:    %tkn_out, %result = xls.blocking_receive %0, @rom_arg0 : i32
// CHECK-NEXT:    %1 = "xls.constant_scalar"() <{value = 1 : i32}> : () -> i32
// CHECK-NEXT:    %2 = xls.send %tkn_out, %1, @rom_arg1 : i32
// CHECK-NEXT:    xls.yield %arg0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  xls.chan @rom_arg0 : i32
// CHECK-NEXT:  xls.chan @rom_arg1 : i32
// CHECK-NEXT:  xls.instantiate_eproc @rom (@rom_arg0 as @rom1_req, @rom_arg1 as @rom1_resp)
// CHECK-NEXT:  xls.chan @rom2_req : i32
// CHECK-NEXT:  xls.chan @rom2_resp : i32
// CHECK-NEXT:  xls.instantiate_eproc @rom as "dram" (@rom_arg0 as @rom2_req, @rom_arg1 as @rom2_resp)
// CHECK-NEXT:  xls.eproc @proxy(%arg0: i32) zeroinitializer discardable attributes {min_pipeline_stages = 3 : i64} {
// CHECK-NEXT:    %0 = xls.after_all  : !xls.token
// CHECK-NEXT:    %tkn_out, %result = xls.blocking_receive %0, @proxy_arg0 : i32
// CHECK-NEXT:    %1 = xls.send %tkn_out, %result, @proxy_arg2 : i32
// CHECK-NEXT:    %tkn_out_0, %result_1 = xls.blocking_receive %1, @proxy_arg3 : i32
// CHECK-NEXT:    %2 = xls.send %tkn_out_0, %result_1, @proxy_arg1 : i32
// CHECK-NEXT:    xls.yield %arg0 : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  xls.chan @proxy_arg0 : i32
// CHECK-NEXT:  xls.chan @proxy_arg1 : i32
// CHECK-NEXT:  xls.chan @proxy_arg2 : i32
// CHECK-NEXT:  xls.chan @proxy_arg3 : i32
// CHECK-NEXT:  xls.instantiate_eproc @proxy (@proxy_arg0 as @req, @proxy_arg1 as @resp, @proxy_arg2 as @rom1_req, @proxy_arg3 as @rom1_resp)
// CHECK-NEXT:  xls.eproc @fetch(%arg0: i32) zeroinitializer discardable {
// CHECK-NEXT:    %0 = xls.after_all  : !xls.token
// CHECK-NEXT:    %1 = xls.send %0, %arg0, @fetch_arg0 : i32
// CHECK-NEXT:    %tkn_out, %result = xls.blocking_receive %1, @fetch_arg1 : i32
// CHECK-NEXT:    xls.yield %result : i32
// CHECK-NEXT:  }
// CHECK-NEXT:  xls.chan @fetch_arg0 : i32
// CHECK-NEXT:  xls.chan @fetch_arg1 : i32
// CHECK-NEXT:  xls.instantiate_eproc @fetch (@fetch_arg0 as @req, @fetch_arg1 as @resp)
// CHECK-NEXT:  xls.chan @boundary1 {fifo_config = #xls.fifo_config<fifo_depth = 1, bypass = true, register_push_outputs = true, register_pop_outputs = false>, input_flop_kind = #xls<flop_kind skid>, send_supported = false} : i32
// CHECK-NEXT:  xls.chan @boundary2 {recv_supported = false} : i32
// CHECK-NEXT:  xls.instantiate_eproc @rom (@rom_arg0 as @boundary1, @rom_arg1 as @boundary2)

xls.sproc @fetch() top {
  spawns {
    %req_out, %req_in = xls.schan<i32>("req")
    %resp_out, %resp_in = xls.schan<i32>("resp") attributes { fifo_config = #xls.fifo_config<fifo_depth = 1, bypass = true, register_push_outputs = true, register_pop_outputs = false> }
    xls.spawn @proxy(%req_in, %resp_out) : !xls.schan<i32, in>, !xls.schan<i32, out>
    xls.yield %req_out, %resp_in : !xls.schan<i32, out>, !xls.schan<i32, in>
  }
  next (%req: !xls.schan<i32, out>, %resp: !xls.schan<i32, in>, %state: i32) zeroinitializer {
    %tok1 = xls.after_all : !xls.token
    %tok2 = xls.ssend %tok1, %state, %req : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
    %tok3, %result = xls.sblocking_receive %tok2, %resp : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
    xls.yield %result : i32
  }
}

xls.sproc @proxy(%req: !xls.schan<i32, in>, %resp: !xls.schan<i32, out>) attributes {min_pipeline_stages = 3 : i64} {
  spawns {
    %rom1_req_out, %rom1_req_in = xls.schan<i32>("rom1_req")
    %rom1_resp_out, %rom1_resp_in = xls.schan<i32>("rom1_resp")
    xls.spawn @rom(%rom1_req_in, %rom1_resp_out) : !xls.schan<i32, in>, !xls.schan<i32, out>
    %rom2_req_out, %rom2_req_in = xls.schan<i32>("rom2_req")
    %rom2_resp_out, %rom2_resp_in = xls.schan<i32>("rom2_resp")
    xls.spawn @rom(%rom2_req_in, %rom2_resp_out) name_hint("dram") : !xls.schan<i32, in>, !xls.schan<i32, out>
    xls.yield %req, %resp, %rom1_req_out, %rom1_resp_in : !xls.schan<i32, in>, !xls.schan<i32, out>, !xls.schan<i32, out>, !xls.schan<i32, in>
  }
  next (%req: !xls.schan<i32, in>, %resp: !xls.schan<i32, out>, %rom_req: !xls.schan<i32, out>, %rom_resp: !xls.schan<i32, in>, %state: i32) zeroinitializer {
    %tok1 = xls.after_all : !xls.token
    %tok2, %result = xls.sblocking_receive %tok1, %req : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
    %tok3 = xls.ssend %tok2, %result, %rom_req : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
    %tok4, %result2 = xls.sblocking_receive %tok3, %rom_resp : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
    %tok5 = xls.ssend %tok4, %result2, %resp : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
    xls.yield %state : i32
  }
}

xls.sproc @rom(%req: !xls.schan<i32, in>, %resp: !xls.schan<i32, out>) top attributes {boundary_channels = [
  #xls.boundary_channel<name = "boundary1", fifo_config = #xls.fifo_config<fifo_depth = 1, bypass = true, register_push_outputs = true, register_pop_outputs = false>, input_flop_kind = #xls<flop_kind skid>>,
  #xls.boundary_channel<name = "boundary2">
]} {
  spawns {
    xls.yield %req, %resp : !xls.schan<i32, in>, !xls.schan<i32, out>
  }
  next (%req: !xls.schan<i32, in>, %resp: !xls.schan<i32, out>, %state: i32) zeroinitializer {
    %tok0 = xls.after_all : !xls.token
    %tok1, %address = xls.sblocking_receive %tok0, %req : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
    %one = "xls.constant_scalar"() {value = 1 : i32} : () -> i32
    %tok2 = xls.ssend %tok1, %one, %resp : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
    xls.yield %state : i32
  }
}

// -----

// CHECK-LABEL: xls.chan @req : i32
// CHECK-NEXT: xls.chan @resp : i32
// CHECK-NEXT: xls.instantiate_extern_eproc "external" as "e" ("req" as @req, "resp" as @resp)
xls.extern_sproc @external(req: !xls.schan<i32, in>, resp: !xls.schan<i32, out>)

// CHECK: xls.chan @main_arg0 : i32
// CHECK-NEXT: xls.chan @main_arg1 : i32
// CHECK: xls.instantiate_eproc @main (@main_arg0 as @req, @main_arg1 as @resp)
xls.sproc @main() top {
  spawns {
    %req_out, %req_in = xls.schan<i32>("req")
    %resp_out, %resp_in = xls.schan<i32>("resp")
    xls.spawn @external(%req_in, %resp_out) name_hint("e") : !xls.schan<i32, in>, !xls.schan<i32, out>
    xls.yield %req_out, %resp_in : !xls.schan<i32, out>, !xls.schan<i32, in>
  }
  next (%req: !xls.schan<i32, out>, %resp: !xls.schan<i32, in>, %state: i32) zeroinitializer {
    %tok = xls.after_all : !xls.token
    %tok2 = xls.ssend %tok, %state, %req : (!xls.token, i32, !xls.schan<i32, out>) -> !xls.token
    xls.yield %state : i32
  }
}


// -----

module {
  xls.sproc @"x['y']"(%arg0: !xls.schan<tensor<i32>, in>, %arg1: !xls.schan<tensor<i32>, out>, %arg2: !xls.schan<tensor<i32>, out>) {
    spawns {
      xls.yield %arg0, %arg1, %arg2 : !xls.schan<tensor<i32>, in>, !xls.schan<tensor<i32>, out>, !xls.schan<tensor<i32>, out>
    }
    next (%arg0: !xls.schan<tensor<i32>, in>, %arg1: !xls.schan<tensor<i32>, out>, %arg2: !xls.schan<tensor<i32>, out>, %arg3: index) zeroinitializer {
      %c1 = arith.constant 1 : index
      %0 = scf.index_switch %arg3 -> index
      case 1 {
        %1 = xls.after_all  : !xls.token
        %tkn_out, %result = xls.sblocking_receive %1, %arg0 : (!xls.token, !xls.schan<tensor<i32>, in>) -> (!xls.token, tensor<i32>)
        %2 = xls.after_all  : !xls.token
        %3 = xls.ssend %2, %result, %arg1 : (!xls.token, tensor<i32>, !xls.schan<tensor<i32>, out>) -> !xls.token
        %4 = xls.ssend %2, %result, %arg2 : (!xls.token, tensor<i32>, !xls.schan<tensor<i32>, out>) -> !xls.token
        scf.yield %c1 : index
      }
      default {
        %1 = xls.after_all  : !xls.token
        %tkn_out, %result = xls.sblocking_receive %1, %arg0 : (!xls.token, !xls.schan<tensor<i32>, in>) -> (!xls.token, tensor<i32>)
        %2 = xls.after_all  : !xls.token
        %3 = xls.ssend %2, %result, %arg1 : (!xls.token, tensor<i32>, !xls.schan<tensor<i32>, out>) -> !xls.token
        %4 = xls.ssend %2, %result, %arg2 : (!xls.token, tensor<i32>, !xls.schan<tensor<i32>, out>) -> !xls.token
        scf.yield %c1 : index
      }
      xls.yield %0 : index
    }
  }
  xls.sproc @some_wrapped_machine(%arg0: !xls.schan<tensor<i32>, in>, %arg1: !xls.schan<tensor<i32>, out>, %arg2: !xls.schan<tensor<i32>, out>) top attributes {
    boundary_channels = [#xls.boundary_channel<name = "x['y']">, #xls.boundary_channel<name = "x['y']1">, #xls.boundary_channel<name = "x['y']2">]
  } {
    spawns {
      %out, %in = xls.schan<tensor<i32>>("x['y']")
      %out_0, %in_1 = xls.schan<tensor<i32>>("x['y']")
      %out_2, %in_3 = xls.schan<tensor<i32>>("x['y']")
      xls.spawn @"x['y']"(%in, %out_0, %out_2) : !xls.schan<tensor<i32>, in>, !xls.schan<tensor<i32>, out>, !xls.schan<tensor<i32>, out>
      xls.yield %arg0, %arg1, %arg2, %out, %in_1, %in_3 : !xls.schan<tensor<i32>, in>, !xls.schan<tensor<i32>, out>, !xls.schan<tensor<i32>, out>, !xls.schan<tensor<i32>, out>, !xls.schan<tensor<i32>, in>, !xls.schan<tensor<i32>, in>
    }
    next (%arg0: !xls.schan<tensor<i32>, in>, %arg1: !xls.schan<tensor<i32>, out>, %arg2: !xls.schan<tensor<i32>, out>, %arg3: !xls.schan<tensor<i32>, out>, %arg4: !xls.schan<tensor<i32>, in>, %arg5: !xls.schan<tensor<i32>, in>) zeroinitializer {
      %0 = xls.after_all  : !xls.token
      %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<i32>, in>) -> (!xls.token, tensor<i32>)
      %1 = xls.after_all  : !xls.token
      %2 = xls.ssend %1, %result, %arg3 : (!xls.token, tensor<i32>, !xls.schan<tensor<i32>, out>) -> !xls.token
      %3 = xls.after_all  : !xls.token
      %tkn_out_0, %result_1 = xls.sblocking_receive %3, %arg4 : (!xls.token, !xls.schan<tensor<i32>, in>) -> (!xls.token, tensor<i32>)
      %tkn_out_2, %result_3 = xls.sblocking_receive %3, %arg5 : (!xls.token, !xls.schan<tensor<i32>, in>) -> (!xls.token, tensor<i32>)
      %4 = xls.after_all  : !xls.token
      %5 = xls.ssend %4, %result_1, %arg1 : (!xls.token, tensor<i32>, !xls.schan<tensor<i32>, out>) -> !xls.token
      %6 = xls.ssend %4, %result_3, %arg2 : (!xls.token, tensor<i32>, !xls.schan<tensor<i32>, out>) -> !xls.token
      xls.yield
    }
  }
}
