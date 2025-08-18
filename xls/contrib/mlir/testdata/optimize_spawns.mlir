// RUN: xls_opt -optimize-spawns %s 2>&1 | FileCheck %s

xls.sproc @fn(%arg0: !xls.schan<tensor<8xi32>, in>) {
  spawns {
    xls.yield
  }
  next(%arg1: i32) zeroinitializer {
    xls.yield %arg1 : i32
  }
}

xls.sproc @fn2(%arg0: !xls.schan<tensor<8xi32>, out>) {
  spawns {
    xls.yield
  }
  next(%arg1: i32) zeroinitializer {
    xls.yield %arg1 : i32
  }
}

// Consumes an argument and passes to a spawn.
// CHECK:  xls.sproc @consume_arg(%arg0: !xls.schan<tensor<8xi32>, in>) top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x")
// CHECK:      xls.spawn @fn(%arg0) : !xls.schan<tensor<8xi32>, in>
// CHECK:      xls.yield
// CHECK:    }
// CHECK:    next () zeroinitializer {
// CHECK:      xls.yield
// CHECK:    }
// CHECK:  }
xls.sproc @consume_arg(%arg0: !xls.schan<tensor<8xi32>, in>) top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn(%in) : !xls.schan<tensor<8xi32>, in>
    xls.yield %arg0, %out : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>, %arg1: !xls.schan<tensor<8xi32>, out>, %arg4: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<8xi32>, in>) -> (!xls.token, tensor<8xi32>)
    %1 = xls.after_all  : !xls.token
    %2 = xls.ssend %1, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    xls.yield %arg4 : i32
  }
}

// Produces a result from a spawn.
// CHECK:  xls.sproc @produce_result(%arg0: !xls.schan<tensor<8xi32>, out>) top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x")
// CHECK:      xls.spawn @fn2(%arg0) : !xls.schan<tensor<8xi32>, out>
// CHECK:      xls.yield
// CHECK:    }
// CHECK:    next () zeroinitializer {
// CHECK:      xls.yield
// CHECK:    }
// CHECK:  }
xls.sproc @produce_result(%arg0: !xls.schan<tensor<8xi32>, out>) top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
    xls.yield %in, %arg0 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>, %arg1: !xls.schan<tensor<8xi32>, out>, %arg2: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<8xi32>, in>) -> (!xls.token, tensor<8xi32>)
    %2 = xls.ssend %0, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    xls.yield %arg2 : i32
  }
}

// Contracts away the interior channel.
// CHECK:  xls.sproc @contract_away_interior_channel() top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x")
// CHECK:      xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
// CHECK:      xls.spawn @fn(%in) : !xls.schan<tensor<8xi32>, in>
// CHECK:      xls.yield
// CHECK:    }
// CHECK:    next () zeroinitializer {
// CHECK:      xls.yield
// CHECK:    }
// CHECK:  }
xls.sproc @contract_away_interior_channel() top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
    %out2, %in2 = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn(%in2) : !xls.schan<tensor<8xi32>, in>
    xls.yield %in, %out2 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>,%arg1: !xls.schan<tensor<8xi32>, out>, %arg2: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<8xi32>, in>) -> (!xls.token, tensor<8xi32>)
    %2 = xls.ssend %0, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    xls.yield %arg2 : i32
  }
}

// The result from this receive is used twice, so we can't contract away the
// interior channel.
// CHECK:  xls.sproc @receive_used_twice() top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x")
// CHECK:      xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
// CHECK:      %out_0, %in_1 = xls.schan<tensor<8xi32>>("x")
// CHECK:      xls.spawn @fn(%in_1) : !xls.schan<tensor<8xi32>, in>
// CHECK:      xls.yield %in, %out_0 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
// CHECK:    }
xls.sproc @receive_used_twice() top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
    %out2, %in2 = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn(%in2) : !xls.schan<tensor<8xi32>, in>
    xls.yield %in, %out2 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>,%arg1: !xls.schan<tensor<8xi32>, out>, %arg2: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<8xi32>, in>) -> (!xls.token, tensor<8xi32>)
    %2 = xls.ssend %0, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    %3 = xls.ssend %0, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    xls.yield %arg2 : i32
  }
}

// The send is predicated, so we can't contract away the interior channel.
// CHECK:  xls.sproc @send_predicated() top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x")
// CHECK:      xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
// CHECK:      %out_0, %in_1 = xls.schan<tensor<8xi32>>("x")
// CHECK:      xls.spawn @fn(%in_1) : !xls.schan<tensor<8xi32>, in>
// CHECK:      xls.yield %in, %out_0 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
// CHECK:    }
xls.sproc @send_predicated() top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
    %out2, %in2 = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn(%in2) : !xls.schan<tensor<8xi32>, in>
    xls.yield %in, %out2 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>,%arg1: !xls.schan<tensor<8xi32>, out>, %arg2: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<8xi32>, in>) -> (!xls.token, tensor<8xi32>)
    %true = arith.constant 1 : i1
    %2 = xls.ssend %0, %result, %arg1, %true : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>, i1) -> !xls.token
    xls.yield %arg2 : i32
  }
}

// The receive is predicated, so we can't contract away the interior channel.
// CHECK:  xls.sproc @recv_predicated() top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x")
// CHECK:      xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
// CHECK:      %out_0, %in_1 = xls.schan<tensor<8xi32>>("x")
// CHECK:      xls.spawn @fn(%in_1) : !xls.schan<tensor<8xi32>, in>
// CHECK:      xls.yield %in, %out_0 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
// CHECK:    }
xls.sproc @recv_predicated() top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
    %out2, %in2 = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn(%in2) : !xls.schan<tensor<8xi32>, in>
    xls.yield %in, %out2 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>,%arg1: !xls.schan<tensor<8xi32>, out>, %arg2: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %true = arith.constant 1 : i1
    %tkn_out, %result = xls.sblocking_receive %0, %arg0, %true : (!xls.token, !xls.schan<tensor<8xi32>, in>, i1) -> (!xls.token, tensor<8xi32>)
    %2 = xls.ssend %0, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    xls.yield %arg2 : i32
  }
}

// CHECK-LABEL: sproc @unused_args
// CHECK: yield
// CHECK: next () zeroinitializer {
// CHECK:   xls.yield
// CHECK: }
xls.sproc @unused_args(%arg0: !xls.schan<tensor<8xi32>, in>) {
  spawns {
    xls.yield %arg0 : !xls.schan<tensor<8xi32>, in>
  }
  next(%arg0: !xls.schan<tensor<8xi32>, in>, %arg1: i32) zeroinitializer {
    xls.yield %arg1 : i32
  }
}

#fifo = #xls.fifo_config<fifo_depth = 1, bypass = false, register_push_outputs = false, register_pop_outputs = false>

// Consumes an argument and passes to a spawn. The interior channel has a
// FifoConfig so can't be eliminated.
// CHECK:  xls.sproc @consume_arg_fifo(%arg0: !xls.schan<tensor<8xi32>, in>) top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x") attributes {fifo_config
// CHECK:      xls.spawn @fn(%in) : !xls.schan<tensor<8xi32>, in>
// CHECK:      xls.yield %arg0, %out
// CHECK:    }
xls.sproc @consume_arg_fifo(%arg0: !xls.schan<tensor<8xi32>, in>) top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x") attributes {fifo_config = #fifo}
    xls.spawn @fn(%in) : !xls.schan<tensor<8xi32>, in>
    xls.yield %arg0, %out : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>, %arg1: !xls.schan<tensor<8xi32>, out>, %arg4: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<8xi32>, in>) -> (!xls.token, tensor<8xi32>)
    %1 = xls.after_all  : !xls.token
    %2 = xls.ssend %1, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    xls.yield %arg4 : i32
  }
}

// Produces a result from a spawn. The interior channel has a FifoConfig so
// can't be eliminated.
// CHECK:  xls.sproc @produce_result_fifo(%arg0: !xls.schan<tensor<8xi32>, out>) top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x") attributes {fifo_config
// CHECK:      xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
// CHECK:      xls.yield %in, %arg0
// CHECK:    }
xls.sproc @produce_result_fifo(%arg0: !xls.schan<tensor<8xi32>, out>) top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x") attributes {fifo_config = #fifo}
    xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
    xls.yield %in, %arg0 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>, %arg1: !xls.schan<tensor<8xi32>, out>, %arg2: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<8xi32>, in>) -> (!xls.token, tensor<8xi32>)
    %2 = xls.ssend %0, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    xls.yield %arg2 : i32
  }
}

// Contracts away the interior channel. Both channels have FifoConfigs, so the
// FifoConfigs can be combined.
// CHECK:  xls.sproc @contract_away_interior_channel_fifo() top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x") attributes {fifo_config = #xls.fifo_config<fifo_depth = 2, bypass = false, register_push_outputs = false, register_pop_outputs = false>}
// CHECK:      xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
// CHECK:      xls.spawn @fn(%in) : !xls.schan<tensor<8xi32>, in>
// CHECK:      xls.yield
// CHECK:    }
// CHECK:    next () zeroinitializer {
// CHECK:      xls.yield
// CHECK:    }
// CHECK:  }
xls.sproc @contract_away_interior_channel_fifo() top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x") attributes {fifo_config = #fifo}
    xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
    %out2, %in2 = xls.schan<tensor<8xi32>>("x") attributes {fifo_config = #fifo}
    xls.spawn @fn(%in2) : !xls.schan<tensor<8xi32>, in>
    xls.yield %in, %out2 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>,%arg1: !xls.schan<tensor<8xi32>, out>, %arg2: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<8xi32>, in>) -> (!xls.token, tensor<8xi32>)
    %2 = xls.ssend %0, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    xls.yield %arg2 : i32
  }
}

// Contracts away the interior channel. One channel has a FifoConfig, the other
// does not, so the FifoConfig does not change.
// CHECK:  xls.sproc @contract_away_interior_channel_fifo2() top {
// CHECK:    spawns {
// CHECK:      %out, %in = xls.schan<tensor<8xi32>>("x") attributes {fifo_config = #xls.fifo_config<fifo_depth = 1, bypass = false, register_push_outputs = false, register_pop_outputs = false>}
// CHECK:      xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
// CHECK:      xls.spawn @fn(%in) : !xls.schan<tensor<8xi32>, in>
// CHECK:      xls.yield
// CHECK:    }
// CHECK:    next () zeroinitializer {
// CHECK:      xls.yield
// CHECK:    }
// CHECK:  }
xls.sproc @contract_away_interior_channel_fifo2() top {
  spawns {
    %out, %in = xls.schan<tensor<8xi32>>("x")
    xls.spawn @fn2(%out) : !xls.schan<tensor<8xi32>, out>
    %out2, %in2 = xls.schan<tensor<8xi32>>("x") attributes {fifo_config = #fifo}
    xls.spawn @fn(%in2) : !xls.schan<tensor<8xi32>, in>
    xls.yield %in, %out2 : !xls.schan<tensor<8xi32>, in>, !xls.schan<tensor<8xi32>, out>
  }
  next (%arg0: !xls.schan<tensor<8xi32>, in>,%arg1: !xls.schan<tensor<8xi32>, out>, %arg2: i32) zeroinitializer {
    %0 = xls.after_all  : !xls.token
    %tkn_out, %result = xls.sblocking_receive %0, %arg0 : (!xls.token, !xls.schan<tensor<8xi32>, in>) -> (!xls.token, tensor<8xi32>)
    %2 = xls.ssend %0, %result, %arg1 : (!xls.token, tensor<8xi32>, !xls.schan<tensor<8xi32>, out>) -> !xls.token
    xls.yield %arg2 : i32
  }
}
