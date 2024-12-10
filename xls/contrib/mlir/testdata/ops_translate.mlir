// RUN: xls_translate --mlir-xls-to-xls %s --main-function=identity
// This just currently verifies a successful translate. The main function
// doesn't matter as all are exported, one just set as top.

func.func @identity(%arg0: i8) -> i8 {
  %0 = "xls.identity"(%arg0) : (i8) -> i8
  return %0 : i8
}

func.func @not(%arg0: i8) -> i8 {
  %0 = xls.not %arg0 : i8
  return %0 : i8
}

func.func @and(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.and %arg0, %arg1 : i8
  return %0 : i8
}

func.func @or(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.or %arg0, %arg1 : i8
  return %0 : i8
}

func.func @xor(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.xor %arg0, %arg1 : i8
  return %0 : i8
}

func.func @neg(%arg0: i8) -> i8 {
  %0 = xls.neg %arg0 : i8
  return %0 : i8
}

func.func @add(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.add %arg0, %arg1 : i8
  return %0 : i8
}

func.func @smul(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.smul %arg0, %arg1 : i8
  return %0 : i8
}

func.func @umul(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.umul %arg0, %arg1 : i8
  return %0 : i8
}

func.func @sdiv(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.sdiv %arg0, %arg1 : i8
  return %0 : i8
}

func.func @smod(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.smod %arg0, %arg1 : i8
  return %0 : i8
}

func.func @sub(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.sub %arg0, %arg1 : i8
  return %0 : i8
}

func.func @udiv(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.udiv %arg0, %arg1 : i8
  return %0 : i8
}

func.func @umod(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.umod %arg0, %arg1 : i8
  return %0 : i8
}


func.func @eq(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.eq %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @ne(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.ne %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @slt(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.slt %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @sle(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.sle %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @sgt(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.sgt %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @sge(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.sge %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @ult(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.ult %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @ule(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.ule %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @ugt(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.ugt %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @uge(%arg0: i8, %arg1: i8) -> i1 {
  %0 = xls.uge %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

func.func @shll(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.shll %arg0, %arg1 : i8
  return %0 : i8
}

func.func @shra(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.shra %arg0, %arg1 : i8
  return %0 : i8
}

func.func @shrl(%arg0: i8, %arg1: i8) -> i8 {
  %0 = xls.shrl %arg0, %arg1 : i8
  return %0 : i8
}

func.func @zero_ext(%arg0: i8) -> i32 {
  %0 = xls.zero_ext %arg0 : (i8) ->i32
  return %0 : i32
}

func.func @sign_ext(%arg0: i8) -> i32 {
  %0 = xls.sign_ext %arg0 : (i8) ->i32
  return %0 : i32
}

func.func @tuple(%arg0: i32, %arg1: i16) -> tuple<i32, i16> {
  %0 = "xls.tuple"(%arg0, %arg1) : (i32, i16) -> tuple<i32, i16>
  return %0 : tuple<i32, i16>
}

func.func @tuple_index(%arg0: tuple<i32, i16>) -> i32 {
  %0 = "xls.tuple_index"(%arg0) { index = 0 : i64 } : (tuple<i32, i16>) -> i32
  return %0 : i32
}

func.func @bit_slice(%arg0: i32) -> i8 {
  %0 = xls.bit_slice %arg0 { start = 8 : i64, width = 8 : i64 } : (i32) -> i8
  return %0 : i8
}

func.func @bit_slice_update(%arg0: i32, %arg1: i32, %arg2: i7) -> i32 {
  %0 = "xls.bit_slice_update"(%arg0, %arg1, %arg2) : (i32, i32, i7) -> i32
  return %0 : i32
}

func.func @dynamic_bit_slice(%arg0: i32, %arg1: i32) -> i7 {
  %0 = "xls.dynamic_bit_slice"(%arg0, %arg1) { width = 7 : i64 } : (i32, i32) -> i7
  return %0 : i7
}

func.func @concat(%arg0: i32, %arg1: i32) -> i64 {
  %0 = xls.concat %arg0, %arg1 : (i32, i32) -> i64
  return %0 : i64
}

func.func @reverse(%arg0: i32) -> i32 {
  %0 = xls.reverse %arg0 : i32
  return %0 : i32
}

func.func @decode(%arg0: i4) -> i16 {
  %0 = xls.decode %arg0 : (i4) -> i16
  return %0 : i16
}

func.func @encode(%arg0: i16) -> i4 {
  %0 = xls.encode %arg0 : (i16) -> i4
  return %0 : i4
}

func.func @one_hot(%arg0: i4) -> i5 {
  %0 = xls.one_hot %arg0 { lsb_prio = true } : (i4) -> i5
  return %0 : i5
}

func.func @sel_nonpow2(%arg0: i2, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 {
  %0 = xls.sel %arg0 in [%arg1, %arg2, %arg3] else %arg1 : (i2, [i32, i32, i32], i32) -> i32
  return %0 : i32
}

func.func @sel_pow2(%arg0: i2, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> i32 {
  %0 = xls.sel %arg0 in [%arg1, %arg2, %arg3, %arg4] : (i2, [i32, i32, i32, i32]) -> i32
  return %0 : i32
}

func.func @one_hot_sel(%arg0: i2, %arg1: i32, %arg2: i32) -> i32 {
  %0 = "xls.one_hot_sel"(%arg0, %arg1, %arg2) : (i2, i32, i32) -> i32
  return %0 : i32
}

func.func @priority_sel(%arg0: i2, %arg1: i32, %arg2: i32) -> i32 {
  %0 = xls.priority_sel %arg0 in [%arg1, %arg2] else %arg2 : (i2, [i32, i32], i32) -> i32
  return %0 : i32
}

func.func @counted_for_apply(%arg0: i32, %arg1: i32, %arg2: i8, %arg3: i9) -> i32 {
  return %arg0 : i32
}

func.func @counted_for(%arg0: i32, %arg1: i8, %arg2: i9) -> i32 {
  %0 = "xls.counted_for"(%arg0, %arg1, %arg2) {
      trip_count = 2 : i64, to_apply = @counted_for_apply } : (i32, i8, i9) -> i32
  return %0 : i32
}

func.func @array(%arg0: i32, %arg1: i32) -> !xls.array<2 x i32> {
  %0 = xls.array %arg0, %arg1 : (i32, i32) -> !xls.array<2 x i32>
  return %0 : !xls.array<2 x i32>
}

func.func @array_tensor(%arg0: i64, %arg1: i64) -> !xls.array<2 x i64> {
  %0 = xls.array %arg0, %arg1 : (i64, i64) -> !xls.array<2 x i64>
  return %0 : !xls.array<2 x i64>
}

func.func @array_index(%arg0: !xls.array<4 x i8>, %arg1: i32) -> i8 {
  %0 = "xls.array_index"(%arg0, %arg1) : (!xls.array<4 x i8>, i32) -> i8
  return %0 : i8
}

func.func @array_slice(%arg0: !xls.array<4 x i8>, %arg1: i32) -> !xls.array<1 x i8> {
  %0 = "xls.array_slice"(%arg0, %arg1) { width = 1 : i64 } : (!xls.array<4 x i8>, i32) -> !xls.array<1 x i8>
  return %0 : !xls.array<1 x i8>
}

func.func @array_update(%arg0: !xls.array<4 x i8>, %arg1: i8, %arg2: i32) -> !xls.array<4 x i8> {
  %0 = "xls.array_update"(%arg0, %arg1, %arg2) : (!xls.array<4 x i8>, i8, i32) -> !xls.array<4 x i8>
  return %0 : !xls.array<4 x i8>
}

func.func @trace(%arg0: i32, %tkn: !xls.token) -> !xls.token {
  %0 = xls.trace %tkn, "a {}"(%arg0) : i32 verbosity 0
  return %0 : !xls.token
}

func.func @bitcast(%arg0: f32) -> i32 {
  %0 = arith.bitcast %arg0 : f32 to i32
  return %0 : i32
}

// TODO
// func.func @constant_tensor() -> tensor<3xi8> {
//   %0 = "xls.constant_tensor"() { value = dense<[0, 1, 2]> : tensor<3xi8> } : () -> tensor<3xi8>
//   return %0 : tensor<3xi8>
// }

// XLS-LABEL: constant_scalar
// XLS: bits[7] = literal(value=3
// XLS: bits[8] = literal(value=103
// XLS: bits[7] = literal(value=6
func.func @constant_scalar() -> i7 {
  %0 = "xls.constant_scalar"() { value = 6 : i7 } : () -> i7
  %1 = "xls.constant_scalar"() { value = 3 : i7 } : () -> i7
  %2 = "xls.constant_scalar"() { value = 103 : i8 } : () -> i8
  return %0 : i7
}

xls.chan @mychan : i32

// XLS-LABEL: proc eproc({{.*}: bits[32], {{.*}}: (bits[32], bits[1]), {{.*}}: bits[32]}, {{.*}}: bits[16])
// XLS:  next
xls.eproc @eproc(%arg0: i32, %arg1: tuple<i32, i1>, %arg2: i1, %arg3: bf16) zeroinitializer {
  %0 = "xls.constant_scalar"() { value = 6 : i32 } : () -> i32
  %tkn1 = "xls.after_all"() : () -> !xls.token
  %tkn_out, %result = xls.blocking_receive %tkn1, @mychan : i32
  %tkn2 = xls.send %tkn_out, %0, @mychan : i32
  %tkn_out2, %result2, %done = xls.nonblocking_receive %tkn2, %arg2, @mychan : i32
  xls.yield %arg0, %arg1, %arg2, %arg3 : i32, tuple<i32, i1>, i1, bf16
}
