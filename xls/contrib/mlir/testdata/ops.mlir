// RUN: xls/contrib/mlir/xls_opt %s --split-input-file --verify-diagnostics -- 2>&1 | FileCheck %s 

// CHECK-LABEL: func @identity
func.func @identity(%arg0: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.identity %arg0
  %0 = "xls.identity"(%arg0) : (tensor<32xi8>) -> tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @not
func.func @not(%arg0: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.not %arg0
  %0 = xls.not %arg0 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @and
func.func @and(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.and %arg0, %arg1
  %0 = xls.and %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @or
func.func @or(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.or %arg0, %arg1
  %0 = xls.or %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @xor
func.func @xor(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.xor %arg0, %arg1
  %0 = xls.xor %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @neg
func.func @neg(%arg0: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.neg %arg0
  %0 = xls.neg %arg0 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @add
func.func @add(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.add %arg0, %arg1
  %0 = xls.add %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @smul
func.func @smul(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.smul %arg0, %arg1
  %0 = xls.smul %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @umul
func.func @umul(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.umul %arg0, %arg1
  %0 = xls.umul %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @sdiv
func.func @sdiv(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.sdiv %arg0, %arg1
  %0 = xls.sdiv %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @smod
func.func @smod(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.smod %arg0, %arg1
  %0 = xls.smod %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @sub
func.func @sub(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.sub %arg0, %arg1
  %0 = xls.sub %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @udiv
func.func @udiv(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.udiv %arg0, %arg1
  %0 = xls.udiv %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @umod
func.func @umod(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.umod %arg0, %arg1
  %0 = xls.umod %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}


// CHECK-LABEL: func @eq
func.func @eq(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.eq %arg0, %arg1
  %0 = xls.eq %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @ne
func.func @ne(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.ne %arg0, %arg1
  %0 = xls.ne %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @slt
func.func @slt(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.slt %arg0, %arg1
  %0 = xls.slt %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @sle
func.func @sle(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.sle %arg0, %arg1
  %0 = xls.sle %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @sgt
func.func @sgt(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.sgt %arg0, %arg1
  %0 = xls.sgt %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @sge
func.func @sge(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.sge %arg0, %arg1
  %0 = xls.sge %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @ult
func.func @ult(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.ult %arg0, %arg1
  %0 = xls.ult %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @ule
func.func @ule(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.ule %arg0, %arg1
  %0 = xls.ule %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @ugt
func.func @ugt(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.ugt %arg0, %arg1
  %0 = xls.ugt %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @uge
func.func @uge(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi1> {
  // CHECK: xls.uge %arg0, %arg1
  %0 = xls.uge %arg0, %arg1 : (tensor<32xi8>, tensor<32xi8>) -> tensor<32xi1>
  return %0 : tensor<32xi1>
}

// CHECK-LABEL: func @shll
func.func @shll(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.shll %arg0, %arg1
  %0 = xls.shll %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @shra
func.func @shra(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.shra %arg0, %arg1
  %0 = xls.shra %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @shrl
func.func @shrl(%arg0: tensor<32xi8>, %arg1: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.shrl %arg0, %arg1
  %0 = xls.shrl %arg0, %arg1 : tensor<32xi8>
  return %0 : tensor<32xi8>
}

// CHECK-LABEL: func @zero_ext
func.func @zero_ext(%arg0: tensor<32xi8>) -> tensor<32xi32> {
  // CHECK: xls.zero_ext %arg0
  %0 = xls.zero_ext %arg0 : (tensor<32xi8>) ->tensor<32xi32>
  return %0 : tensor<32xi32>
}

// CHECK-LABEL: func @sign_ext
func.func @sign_ext(%arg0: tensor<32xi8>) -> tensor<32xi32> {
  // CHECK: xls.sign_ext %arg0
  %0 = xls.sign_ext %arg0 : (tensor<32xi8>) ->tensor<32xi32>
  return %0 : tensor<32xi32>
}

// CHECK-LABEL: func @tuple
func.func @tuple(%arg0: i32, %arg1: tensor<2xi8>) -> tuple<i32, tensor<2xi8>> {
  // CHECK: "xls.tuple"(%arg0, %arg1)
  %0 = "xls.tuple"(%arg0, %arg1) : (i32, tensor<2xi8>) -> tuple<i32, tensor<2xi8>>
  return %0 : tuple<i32, tensor<2xi8>>
}

// CHECK-LABEL: func @tuple_index
func.func @tuple_index(%arg0: tuple<i32, tensor<2xi8>>) -> i32 {
  // CHECK: "xls.tuple_index"(%arg0)
  // CHECK-SAME: index = 0 : i64
  %0 = "xls.tuple_index"(%arg0) { index = 0 : i64 } : (tuple<i32, tensor<2xi8>>) -> i32
  return %0 : i32
}

// CHECK-LABEL: func @bit_slice
func.func @bit_slice(%arg0: i32) -> i8 {
  // CHECK: xls.bit_slice
  %0 = xls.bit_slice %arg0 { start = 8 : i64, width = 8 : i64 } : (i32) -> i8
  return %0 : i8
}

// CHECK-LABEL: func @bit_slice_update
func.func @bit_slice_update(%arg0: i32, %arg1: i32, %arg2: i7) -> i32 {
  // CHECK: xls.bit_slice_update
  %0 = "xls.bit_slice_update"(%arg0, %arg1, %arg2) : (i32, i32, i7) -> i32
  return %0 : i32
}

// CHECK-LABEL: func @dynamic_bit_slice
func.func @dynamic_bit_slice(%arg0: i32, %arg1: i32) -> i7 {
  // CHECK: xls.dynamic_bit_slice
  %0 = "xls.dynamic_bit_slice"(%arg0, %arg1) { width = 7 : i64 } : (i32, i32) -> i7
  return %0 : i7
}

// CHECK-LABEL: func @concat
func.func @concat(%arg0: i32, %arg1: i32) -> i64 {
  // CHECK: xls.concat
  %0 = xls.concat %arg0, %arg1 : (i32, i32) -> i64
  return %0 : i64
}

// CHECK-LABEL: reverse
func.func @reverse(%arg0: i32) -> i32 {
  // CHECK: xls.reverse
  %0 = xls.reverse %arg0 : i32
  return %0 : i32
}

// CHECK-LABEL: decode
func.func @decode(%arg0: i4) -> i16 {
  // CHECK: xls.decode
  %0 = xls.decode %arg0 : (i4) -> i16
  return %0 : i16
}

// CHECK-LABEL: encode
func.func @encode(%arg0: i16) -> i4 {
  // CHECK: xls.encode
  %0 = xls.encode %arg0 : (i16) -> i4
  return %0 : i4
}

// CHECK-LABEL: one_hot
func.func @one_hot(%arg0: i4) -> i5 {
  // CHECK: xls.one_hot
  %0 = xls.one_hot %arg0 { lsb_prio = true } : (i4) -> i5
  return %0 : i5
}

// CHECK-LABEL: sel
func.func @sel(%arg0: i16, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK: xls.sel
  %0 = "xls.sel"(%arg0, %arg1, %arg1, %arg2) : (i16, i32, i32, i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: one_hot_sel
func.func @one_hot_sel(%arg0: i16, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK: xls.one_hot_sel
  %0 = "xls.one_hot_sel"(%arg0, %arg1, %arg2) : (i16, i32, i32) -> i32
  return %0 : i32
}

// CHECK-LABEL: priority_sel
func.func @priority_sel(%arg0: i16, %arg1: i32, %arg2: i32) -> i32 {
  // CHECK: xls.priority_sel
  %0 = "xls.priority_sel"(%arg0, %arg1, %arg2) : (i16, i32, i32) -> i32
  return %0 : i32
}

func.func @counted_for_apply(%arg0: i32, %arg1: i10, %arg2: i8, %arg3: i9) -> i32 {
  return %arg0 : i32
}

// CHECK-LABEL: counted_for
func.func @counted_for(%arg0: i10, %arg1: i8, %arg2: i9) -> i32 {
  // CHECK: xls.counted_for
  %0 = "xls.counted_for"(%arg0, %arg1, %arg2) { trip_count = 2 : i64, to_apply = @counted_for_apply } : (i10, i8, i9) -> i32
  return %0 : i32
}

// CHECK-LABEL: after_all
func.func @after_all(%arg0: !xls.token, %arg1: !xls.token) -> !xls.token {
  %0 = xls.after_all %arg0, %arg1 : !xls.token
  return %0 : !xls.token
}

// CHECK-LABEL: array
func.func @array(%arg0: i32, %arg1: i32) -> !xls.array<2 x i32> {
  // CHECK: xls.array
  %0 = xls.array %arg0, %arg1 : (i32, i32) -> !xls.array<2 x i32>
  return %0 : !xls.array<2 x i32>
}

// CHECK-LABEL: array_tensor
func.func @array_tensor(%arg0: tensor<2xi32>, %arg1: tensor<2xi32>) -> !xls.array<2 x tensor<2xi32>> {
  // CHECK: xls.array
  %0 = xls.array %arg0, %arg1 : (tensor<2xi32>, tensor<2xi32>) -> !xls.array<2 x tensor<2xi32>>
  return %0 : !xls.array<2 x tensor<2xi32>>
}

// CHECK-LABEL: array_index
func.func @array_index(%arg0: !xls.array<4 x i8>, %arg1: i32) -> i8 {
  // CHECK: xls.array_index
  %0 = "xls.array_index"(%arg0, %arg1) : (!xls.array<4 x i8>, i32) -> i8
  return %0 : i8
}

// CHECK-LABEL: array_slice
func.func @array_slice(%arg0: !xls.array<4 x i8>, %arg1: i32) -> !xls.array<1 x i8> {
  // CHECK: xls.array_slice
  %0 = "xls.array_slice"(%arg0, %arg1) { width = 1 : i64 } : (!xls.array<4 x i8>, i32) -> !xls.array<1 x i8>
  return %0 : !xls.array<1 x i8>
}

// CHECK-LABEL: array_update
func.func @array_update(%arg0: !xls.array<4 x i8>, %arg1: i8, %arg2: i32) -> !xls.array<4 x i8> {
  // CHECK: xls.array_update
  %0 = "xls.array_update"(%arg0, %arg1, %arg2) : (!xls.array<4 x i8>, i8, i32) -> !xls.array<4 x i8>
  return %0 : !xls.array<4 x i8>
}

// CHECK-LABEL: array_update_tensor
func.func @array_update_tensor(%arg0: !xls.array<4 x tensor<3xi8>>, %arg1: tensor<3xi8>, %arg2: tensor<3xi32>) -> !xls.array<4 x tensor<3xi8>> {
  // CHECK: xls.array_update
  %0 = "xls.array_update"(%arg0, %arg1, %arg2) : (!xls.array<4 x tensor<3xi8>>, tensor<3xi8>, tensor<3xi32>) -> !xls.array<4 x tensor<3xi8>>
  return %0 : !xls.array<4 x tensor<3xi8>>
}

// CHECK-LABEL: constant_tensor
func.func @constant_tensor() -> tensor<3xi8> {
  // CHECK: xls.constant_tensor
  %0 = "xls.constant_tensor"() { value = dense<[0, 1, 2]> : tensor<3xi8> } : () -> tensor<3xi8>
  return %0 : tensor<3xi8>
}

// CHECK-LABEL: constant_scalar
func.func @constant_scalar() -> i7 {
  // CHECK: xls.constant_scalar
  %0 = "xls.constant_scalar"() { value = 6 : i7 } : () -> i7
  return %0 : i7
}

// CHECK-LABEL: for
func.func @for(%arg0: i32, %arg1: i8, %arg2: i9) -> i32 {
  // CHECK: xls.for
  %0 = xls.for inits(%arg0) invariants(%arg1, %arg2) {
    ^bb0(%arg3: i32, %arg4: i32, %arg5: i8, %arg6: i9):
    xls.yield %arg3 : i32
  } { trip_count = 6 : i64 } : (i32, i8, i9) -> i32
  return %0 : i32
}

func.func private @callee(%arg0: i8) -> i8

// CHECK-LABEL: vectorized_call
func.func @vectorized_call(%arg0: tensor<32xi8>) -> tensor<32xi8> {
  // CHECK: xls.vectorized_call
  %0 = xls.vectorized_call @callee(%arg0) : (tensor<32xi8>) -> tensor<32xi8>
  return %0 : tensor<32xi8>
}

xls.chan @mychan : i32
xls.chan @vector_chan : tensor<32xi32>

// CHECK-LABEL: xls.eproc @eproc(%arg0: i32) zeroinitializer attributes {min_pipeline_stages = 2 : i64}
xls.eproc @eproc(%arg: i32) zeroinitializer attributes {min_pipeline_stages = 2 : i64} {
  xls.yield %arg : i32
}

// CHECK: xls.instantiate_eproc @eproc ()
xls.instantiate_eproc @eproc ()

// CHECK: xls.instantiate_eproc @eproc (@mychan as @vector_chan)
xls.instantiate_eproc @eproc (@mychan as @vector_chan)

// CHECK-LABEL: func @blocking_receive
func.func @blocking_receive(%arg0: !xls.token, %arg1: i1) -> i32 {
  // CHECK: xls.blocking_receive %arg0, %arg1, @mychan : i32
  %tkn_out, %result = xls.blocking_receive %arg0, %arg1, @mychan : i32
  return %result : i32
}

// CHECK-LABEL: func @blocking_receive_nopred
func.func @blocking_receive_nopred(%arg0: !xls.token) -> i32 {
  // CHECK: xls.blocking_receive %arg0, @mychan : i32
  %tkn_out, %result = xls.blocking_receive %arg0, @mychan : i32
  return %result : i32
}

// CHECK-LABEL: func @nonblocking_receive
func.func @nonblocking_receive(%arg0: !xls.token) -> tensor<32xi32> {
  // CHECK: %tkn_out, %result, %valid = xls.nonblocking_receive %arg0, @vector_chan : tensor<32xi32>
  %tkn_out, %result, %valid = xls.nonblocking_receive %arg0, @vector_chan : tensor<32xi32>
  return %result : tensor<32xi32>
}

// CHECK-LABEL: func @send
func.func @send(%arg0: !xls.token, %arg1: i32, %arg2: i1) -> !xls.token {
  // CHECK: xls.send %arg0, %arg1, %arg2, @mychan : i32
  %result = xls.send %arg0, %arg1, %arg2, @mychan : i32
  return %result : !xls.token
}

xls.sproc @sproc() {
  spawns {
    %0, %1 = xls.schan<i32>("mychan")
    xls.spawn @mytarget(%1, %0) : !xls.schan<i32, in>, !xls.schan<i32, out>
    xls.yield
  }
  next (%state: i32) zeroinitializer {
    xls.yield %state : i32
  }
}

// CHECK-LABEL: xls.sproc @mytarget(%arg0: !xls.schan<i32, in>, %arg1: !xls.schan<i32, out>) attributes {min_pipeline_stages = 2 : i64} {
xls.sproc @mytarget(%chan: !xls.schan<i32, in>, %chan2: !xls.schan<i32, out>) attributes {min_pipeline_stages = 2 : i64} {
  spawns {
    xls.yield
  }
  next (%state: i1) zeroinitializer {
    xls.yield %state : i1
  }
}

// CHECK-LABEL: xls.sproc @sproc2() top {
xls.sproc @sproc2() top {
  spawns {
    %0, %1 = xls.schan<i32>("mychan")
    xls.yield %1 : !xls.schan<i32, in>
  }
  next (%chan: !xls.schan<i32, in>, %state: i32) zeroinitializer {
    %tok = xls.after_all : !xls.token
    %tok2, %result = xls.sblocking_receive %tok, %chan : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
    xls.yield %state : i32
  }
}

// CHECK-LABEL: func @array_concat
func.func @array_concat(%arg0: !xls.array<2 x i32>, %arg1: !xls.array<2 x i32>) -> !xls.array<4 x i32> {
  // CHECK: xls.array_concat
  %0 = xls.array_concat %arg0, %arg1 : (!xls.array<2 x i32>, !xls.array<2 x i32>) -> !xls.array<4 x i32>
  return %0 : !xls.array<4 x i32>
}

// CHECK-LABEL: func @array_update_slice
func.func @array_update_slice(%arg0: !xls.array<4 x i32>, %arg1: !xls.array<2 x i32>, %arg2: i32, %arg3: i32) -> !xls.array<4 x i32> {
  // CHECK: xls.array_update_slice
  %0 = xls.array_update_slice %arg1 into %arg0[%arg2 +: 2] : !xls.array<4 x i32>
  return %0 : !xls.array<4 x i32>
}

// CHECK-LABEL: func @trace
func.func @trace(%arg0: i32, %tkn: !xls.token) -> !xls.token {
  // CHECK: xls.trace %arg1, "a {}"(%arg0) : i32 verbosity 1
  %0 = xls.trace %tkn, "a {}"(%arg0) : i32 verbosity 1
  return %0 : !xls.token
}

// CHECK-LABEL: func @trace_cond
func.func @trace_cond(%arg0: i32, %tkn: !xls.token, %cond: i1) -> !xls.token {
  // CHECK: xls.trace %arg1, %arg2, "a {}"
  %0 = xls.trace %tkn, %cond, "a {}"
  return %0 : !xls.token
}

// -----

// expected-error@+1 {{yielded state type does not match carried state type ('tuple<i7>' vs 'tuple<i32>'}}
xls.eproc @eproc(%arg: i32) zeroinitializer {
  %0 = "xls.constant_scalar"() { value = 6 : i7 } : () -> i7
  xls.yield %0 : i7
}

// -----

// expected-error@+1 {{op next expects 1 channels but spawns yields 0}}
xls.sproc @sproc() {
  spawns {
    xls.yield
  }
  next (%chan: !xls.schan<i32, in>, %state: i32) zeroinitializer {
    xls.yield %state : i32
  }
}

// -----

// expected-error@+1 {{op next expects channel of type '!xls.schan<i32, out>' but spawns yields channel of type '!xls.schan<i32, in>'}}
xls.sproc @sproc(%input: !xls.schan<i32, in>) {
  spawns {
    xls.yield %input : !xls.schan<i32, in>
  }
  next (%chan: !xls.schan<i32, out>, %state: i32) zeroinitializer {
    xls.yield %state : i32
  }
}

// -----

// expected-error@+1 {{op next expects 0 channels but spawns yields 1}}
xls.sproc @sproc(%input: !xls.schan<i32, in>) {
  spawns {
    xls.yield %input : !xls.schan<i32, in>
  }
  next (%state: i32) zeroinitializer {
    xls.yield %state : i32
  }
}

// -----

xls.sproc @sproc2() {
  spawns {
    %0, %1 = xls.schan<i32>("mychan")
    xls.yield %1 : !xls.schan<i32, in>
  }
  next (%chan: !xls.schan<i32, in>, %state: i32) zeroinitializer {
    %tok = xls.after_all : !xls.token
    %tok2, %result = xls.sblocking_receive %tok, %chan : (!xls.token, !xls.schan<i32, in>) -> (!xls.token, i32)
    // expected-error@+1 {{op channel is not an output channel}}
    %tok3 = xls.ssend %tok, %result, %chan : (!xls.token, i32, !xls.schan<i32, in>) -> !xls.token
    xls.yield %state : i32
  }
}

// -----

xls.eproc @eproc(%arg: i32) zeroinitializer {
  xls.yield %arg : i32
}

// expected-error@+1 {{'xls.instantiate_eproc' op '@unknown' does not reference a valid channel}}
xls.instantiate_eproc @eproc (@unknown as @unknown)
