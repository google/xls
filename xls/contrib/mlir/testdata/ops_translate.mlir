// RUN: xls_translate --mlir-xls-to-xls %s --main-function=identity > %t
// RUN: FileCheck --check-prefix=XLS %s < %t
// RUN: xls_translate --xls-to-mlir-xls %t | FileCheck --check-prefix=MLIR %s

// The above performs a round-trip conversion via XLS IR, and performs basic
// checking for proper conversion on the XLS and MLIR.
// The main function doesn't matter as all are exported, one just set as top.

// MLIR-LABEL: func.func @identity
func.func @identity(%arg0: i8) -> i8 {
  // MLIR: {{.*}} = xls.identity %{{.*}} : i8
  %0 = xls.identity %arg0 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @not
func.func @not(%arg0: i8) -> i8 {
  // MLIR: %{{.*}} = xls.not %{{.*}} : i8
  %0 = xls.not %arg0 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @and
func.func @and(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.and %{{.*}}, %{{.*}} : i8
  %0 = xls.and %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @nand
func.func @nand(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.nand %{{.*}}, %{{.*}} : i8
  %0 = xls.nand %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @or
func.func @or(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.or %{{.*}}, %{{.*}} : i8
  %0 = xls.or %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @nor
func.func @nor(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.nor %{{.*}}, %{{.*}} : i8
  %0 = xls.nor %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @xor
func.func @xor(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.xor %{{.*}}, %{{.*}} : i8
  %0 = xls.xor %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @and_reduce
func.func @and_reduce(%arg0: i8) -> i1 {
  // MLIR: %{{.*}} = xls.and_reduce %{{.*}} : (i8) -> i1
  %0 = xls.and_reduce %arg0 : (i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @or_reduce
func.func @or_reduce(%arg0: i8) -> i1 {
  // MLIR: %{{.*}} = xls.or_reduce %{{.*}} : (i8) -> i1
  %0 = xls.or_reduce %arg0 : (i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @xor_reduce
func.func @xor_reduce(%arg0: i8) -> i1 {
  // MLIR: %{{.*}} = xls.xor_reduce %{{.*}} : (i8) -> i1
  %0 = xls.xor_reduce %arg0 : (i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @neg
func.func @neg(%arg0: i8) -> i8 {
  // MLIR: %{{.*}} = xls.neg %{{.*}} : i8
  %0 = xls.neg %arg0 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @add
func.func @add(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.add %{{.*}}, %{{.*}} : i8
  %0 = xls.add %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @smul
func.func @smul(%arg0: i8, %arg1: i8) -> i16 {
  // MLIR: %{{.*}} = xls.smul %{{.*}}, %{{.*}} : (i8, i8) -> i16
  %0 = xls.smul %arg0, %arg1 : (i8, i8) -> i16
  return %0 : i16
}

// MLIR-LABEL: func.func @smulp
func.func @smulp(%arg0: i8, %arg1: i7) -> i9 {
  // MLIR: %{{.*}}, %{{.*}} = xls.smulp %{{.*}}, %{{.*}} : (i8, i7) -> (i9, i9)
  %0, %1 = xls.smulp %arg0, %arg1 : (i8, i7) -> (i9, i9)
  return %0 : i9
}

// MLIR-LABEL: func.func @umul
func.func @umul(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.umul %{{.*}}, %{{.*}} : i8
  %0 = xls.umul %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @umulp
func.func @umulp(%arg0: i8, %arg1: i7) -> i9 {
  // MLIR: %{{.*}}, %{{.*}} = xls.umulp %{{.*}}, %{{.*}} : (i8, i7) -> (i9, i9)
  %0, %1 = xls.umulp %arg0, %arg1 : (i8, i7) -> (i9, i9)
  return %0 : i9
}

// MLIR-LABEL: func.func @sdiv
func.func @sdiv(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.sdiv %{{.*}}, %{{.*}} : i8
  %0 = xls.sdiv %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @smod
func.func @smod(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.smod %{{.*}}, %{{.*}} : i8
  %0 = xls.smod %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @sub
func.func @sub(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.sub %{{.*}}, %{{.*}} : i8
  %0 = xls.sub %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @udiv
func.func @udiv(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.udiv %{{.*}}, %{{.*}} : i8
  %0 = xls.udiv %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @umod
func.func @umod(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.umod %{{.*}}, %{{.*}} : i8
  %0 = xls.umod %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @eq
func.func @eq(%arg0: tuple<i8, i8>, %arg1: tuple<i8, i8>) -> i1 {
  // MLIR: %{{.*}} = xls.eq %{{.*}}, %{{.*}} : (tuple<i8, i8>, tuple<i8, i8>) -> i1
  %0 = xls.eq %arg0, %arg1 : (tuple<i8, i8>, tuple<i8, i8>) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @ne
func.func @ne(%arg0: i8, %arg1: i8) -> i1 {
  // MLIR: %{{.*}} = xls.ne %{{.*}}, %{{.*}} : (i8, i8) -> i1
  %0 = xls.ne %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @slt
func.func @slt(%arg0: i8, %arg1: i8) -> i1 {
  // MLIR: %{{.*}} = xls.slt %{{.*}}, %{{.*}} : (i8, i8) -> i1
  %0 = xls.slt %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @sle
func.func @sle(%arg0: i8, %arg1: i8) -> i1 {
  // MLIR: %{{.*}} = xls.sle %{{.*}}, %{{.*}} : (i8, i8) -> i1
  %0 = xls.sle %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @sgt
func.func @sgt(%arg0: i8, %arg1: i8) -> i1 {
  // MLIR: %{{.*}} = xls.sgt %{{.*}}, %{{.*}} : (i8, i8) -> i1
  %0 = xls.sgt %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @sge
func.func @sge(%arg0: i8, %arg1: i8) -> i1 {
  // MLIR: %{{.*}} = xls.sge %{{.*}}, %{{.*}} : (i8, i8) -> i1
  %0 = xls.sge %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @ult
func.func @ult(%arg0: i8, %arg1: i8) -> i1 {
  // MLIR: %{{.*}} = xls.ult %{{.*}}, %{{.*}} : (i8, i8) -> i1
  %0 = xls.ult %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @ule
func.func @ule(%arg0: i8, %arg1: i8) -> i1 {
  // MLIR: %{{.*}} = xls.ule %{{.*}}, %{{.*}} : (i8, i8) -> i1
  %0 = xls.ule %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @ugt
func.func @ugt(%arg0: i8, %arg1: i8) -> i1 {
  // MLIR: %{{.*}} = xls.ugt %{{.*}}, %{{.*}} : (i8, i8) -> i1
  %0 = xls.ugt %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @uge
func.func @uge(%arg0: i8, %arg1: i8) -> i1 {
  // MLIR: %{{.*}} = xls.uge %{{.*}}, %{{.*}} : (i8, i8) -> i1
  %0 = xls.uge %arg0, %arg1 : (i8, i8) -> i1
  return %0 : i1
}

// MLIR-LABEL: func.func @shll
func.func @shll(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.shll %{{.*}}, %{{.*}} : i8
  %0 = xls.shll %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @shra
func.func @shra(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.shra %{{.*}}, %{{.*}} : i8
  %0 = xls.shra %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @shrl
func.func @shrl(%arg0: i8, %arg1: i8) -> i8 {
  // MLIR: %{{.*}} = xls.shrl %{{.*}}, %{{.*}} : i8
  %0 = xls.shrl %arg0, %arg1 : i8
  return %0 : i8
}

// MLIR-LABEL: func.func @zero_ext
func.func @zero_ext(%arg0: i8) -> i32 {
  // MLIR: %{{.*}} = xls.zero_ext %{{.*}} : (i8) -> i32
  %0 = xls.zero_ext %arg0 : (i8) -> i32
  return %0 : i32
}

// MLIR-LABEL: func.func @sign_ext
func.func @sign_ext(%arg0: i8) -> i32 {
  // MLIR: %{{.*}} = xls.sign_ext %{{.*}} : (i8) -> i32
  %0 = xls.sign_ext %arg0 : (i8) -> i32
  return %0 : i32
}

// MLIR-LABEL: func.func @tuple
func.func @tuple(%arg0: i32, %arg1: i16) -> tuple<i32, i16> {
  // MLIR: %{{.*}} = "xls.tuple"(%{{.*}}, %{{.*}}) : (i32, i16) -> tuple<i32, i16>
  %0 = "xls.tuple"(%arg0, %arg1) : (i32, i16) -> tuple<i32, i16>
  return %0 : tuple<i32, i16>
}

// MLIR-LABEL: func.func @tuple_index
func.func @tuple_index(%arg0: tuple<i32, i16>) -> i32 {
  // MLIR: %{{.*}} = "xls.tuple_index"(%{{.*}}) <{index = 0 : i64}> : (tuple<i32, i16>) -> i32
  %0 = "xls.tuple_index"(%arg0) { index = 0 : i64 } : (tuple<i32, i16>) -> i32
  return %0 : i32
}

// MLIR-LABEL: func.func @bit_slice
func.func @bit_slice(%arg0: i32) -> i8 {
  // MLIR: %{{.*}} = xls.bit_slice %{{.*}} {start = 8 : i64, width = 8 : i64} : (i32) -> i8
  %0 = xls.bit_slice %arg0 { start = 8 : i64, width = 8 : i64 } : (i32) -> i8
  return %0 : i8
}

// MLIR-LABEL: func.func @bit_slice_update
func.func @bit_slice_update(%arg0: i32, %arg1: i32, %arg2: i7) -> i32 {
  // MLIR: %{{.*}} = "xls.bit_slice_update"(%{{.*}}, %{{.*}}, %{{.*}}) : (i32, i32, i7) -> i32
  %0 = "xls.bit_slice_update"(%arg0, %arg1, %arg2) : (i32, i32, i7) -> i32
  return %0 : i32
}

// MLIR-LABEL: func.func @dynamic_bit_slice
func.func @dynamic_bit_slice(%arg0: i32, %arg1: i32) -> i7 {
  // MLIR: %{{.*}} = "xls.dynamic_bit_slice"(%{{.*}}, %{{.*}}) <{width = 7 : i64}> : (i32, i32) -> i7
  %0 = "xls.dynamic_bit_slice"(%arg0, %arg1) { width = 7 : i64 } : (i32, i32) -> i7
  return %0 : i7
}

// MLIR-LABEL: func.func @concat
func.func @concat(%arg0: i32, %arg1: i32) -> i64 {
  // MLIR: %{{.*}} = xls.concat %{{.*}}, %{{.*}} : (i32, i32) -> i64
  %0 = xls.concat %arg0, %arg1 : (i32, i32) -> i64
  return %0 : i64
}

// MLIR-LABEL: func.func @reverse
func.func @reverse(%arg0: i32) -> i32 {
  // MLIR: %{{.*}} = xls.reverse %{{.*}} : i32
  %0 = xls.reverse %arg0 : i32
  return %0 : i32
}

// MLIR-LABEL: func.func @decode
func.func @decode(%arg0: i4) -> i16 {
  // MLIR: %{{.*}} = xls.decode %{{.*}} {width = 16 : i64} : (i4) -> i16
  %0 = xls.decode %arg0 { width = 16 : i64 } : (i4) -> i16
  return %0 : i16
}

// MLIR-LABEL: func.func @encode
func.func @encode(%arg0: i16) -> i4 {
  // MLIR: %{{.*}} = xls.encode %{{.*}} : (i16) -> i4
  %0 = xls.encode %arg0 : (i16) -> i4
  return %0 : i4
}

// MLIR-LABEL: func.func @one_hot
func.func @one_hot(%arg0: i4) -> i5 {
  // MLIR: %{{.*}} = xls.one_hot %{{.*}} {lsb_prio = true} : (i4) -> i5
  %0 = xls.one_hot %arg0 { lsb_prio = true } : (i4) -> i5
  return %0 : i5
}

// MLIR-LABEL: func.func @sel_nonpow2
func.func @sel_nonpow2(%arg0: i2, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 {
  // MLIR: %{{.*}} = xls.sel %{{.*}} in [%{{.*}}, %{{.*}}, %{{.*}}] else %{{.*}} : (i2, [i32, i32, i32], i32) -> i32
  %0 = xls.sel %arg0 in [%arg1, %arg2, %arg3] else %arg1 : (i2, [i32, i32, i32], i32) -> i32
  return %0 : i32
}

// MLIR-LABEL: func.func @sel_pow2
func.func @sel_pow2(%arg0: i2, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32) -> i32 {
  // MLIR: %{{.*}} = xls.sel %{{.*}} in [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] : (i2, [i32, i32, i32, i32]) -> i32
  %0 = xls.sel %arg0 in [%arg1, %arg2, %arg3, %arg4] : (i2, [i32, i32, i32, i32]) -> i32
  return %0 : i32
}

// MLIR-LABEL: func.func @one_hot_sel
func.func @one_hot_sel(%arg0: i2, %arg1: i32, %arg2: i32) -> i32 {
  // MLIR: %{{.*}} = "xls.one_hot_sel"(%{{.*}}, %{{.*}}, %{{.*}}) : (i2, i32, i32) -> i32
  %0 = "xls.one_hot_sel"(%arg0, %arg1, %arg2) : (i2, i32, i32) -> i32
  return %0 : i32
}

// MLIR-LABEL: func.func @priority_sel
func.func @priority_sel(%arg0: i2, %arg1: i32, %arg2: i32) -> i32 {
  // MLIR: %{{.*}} = xls.priority_sel %{{.*}} in [%{{.*}}, %{{.*}}] else %{{.*}} : (i2, [i32, i32], i32) -> i32
  %0 = xls.priority_sel %arg0 in [%arg1, %arg2] else %arg2 : (i2, [i32, i32], i32) -> i32
  return %0 : i32
}

func.func @counted_for_apply(%arg0: i32, %arg1: i32, %arg2: i8, %arg3: i9) -> i32 {
  return %arg0 : i32
}

// MLIR-LABEL: func.func @counted_for
func.func @counted_for(%arg0: i32, %arg1: i8, %arg2: i9) -> i32 {
  // MLIR: %{{.*}} = "xls.counted_for"(%{{.*}}, %{{.*}}, %{{.*}}) <{stride = 1 : i64, to_apply = @counted_for_apply, trip_count = 2 : i64}> : (i32, i8, i9) -> i32
  %0 = "xls.counted_for"(%arg0, %arg1, %arg2) {stride = 1 : i64,
      trip_count = 2 : i64, to_apply = @counted_for_apply } : (i32, i8, i9) -> i32
  return %0 : i32
}

// MLIR-LABEL: func.func @array
func.func @array(%arg0: i32, %arg1: i32) -> !xls.array<2 x i32> {
  // MLIR: %{{.*}} = xls.array %{{.*}}, %{{.*}} : (i32, i32) -> !xls.array<2 x i32>
  %0 = xls.array %arg0, %arg1 : (i32, i32) -> !xls.array<2 x i32>
  return %0 : !xls.array<2 x i32>
}

// MLIR-LABEL: func.func @array_tensor
func.func @array_tensor(%arg0: i64, %arg1: i64) -> !xls.array<2 x i64> {
  // MLIR: %{{.*}} = xls.array %{{.*}}, %{{.*}} : (i64, i64) -> !xls.array<2 x i64>
  %0 = xls.array %arg0, %arg1 : (i64, i64) -> !xls.array<2 x i64>
  return %0 : !xls.array<2 x i64>
}

// MLIR-LABEL: func.func @array_index
func.func @array_index(%arg0: !xls.array<4 x i8>, %arg1: i32) -> i8 {
  // MLIR: %{{.*}} = "xls.array_index"(%{{.*}}, %{{.*}}) : (!xls.array<4 x i8>, i32) -> i8
  %0 = "xls.array_index"(%arg0, %arg1) : (!xls.array<4 x i8>, i32) -> i8
  return %0 : i8
}

// MLIR-LABEL: func.func @array_slice
func.func @array_slice(%arg0: !xls.array<4 x i8>, %arg1: i32) -> !xls.array<1 x i8> {
  // MLIR: %{{.*}} = "xls.array_slice"(%{{.*}}, %{{.*}}) <{width = 1 : i64}> : (!xls.array<4 x i8>, i32) -> !xls.array<1 x i8>
  %0 = "xls.array_slice"(%arg0, %arg1) { width = 1 : i64 } : (!xls.array<4 x i8>, i32) -> !xls.array<1 x i8>
  return %0 : !xls.array<1 x i8>
}

// MLIR-LABEL: func.func @array_update
func.func @array_update(%arg0: !xls.array<4 x tuple<i1, i2>>, %arg1: tuple<i1, i2>, %arg2: i32) -> !xls.array<4 x tuple<i1, i2>> {
  // MLIR: %{{.*}} = "xls.array_update"(%{{.*}}, %{{.*}}, %{{.*}}) : (!xls.array<4 x tuple<i1, i2>>, tuple<i1, i2>, i32) -> !xls.array<4 x tuple<i1, i2>>
  %0 = "xls.array_update"(%arg0, %arg1, %arg2) : (!xls.array<4 x tuple<i1, i2>>, tuple<i1, i2>, i32) -> !xls.array<4 x tuple<i1, i2>>
  return %0 : !xls.array<4 x tuple<i1, i2>>
}

// MLIR-LABEL: func.func @trace
func.func @trace(%arg0: i32, %tkn: !xls.token) -> !xls.token {
  // MLIR: %{{.*}} = xls.trace %{{.*}}, "a {}"(%{{.*}}) : i32 verbosity 1
  %0 = xls.trace %tkn, "a {}"(%arg0) : i32 verbosity 1
  return %0 : !xls.token
}

func.func @bitcast(%arg0: f32) -> i32 {
  %0 = arith.bitcast %arg0 : f32 to i32
  return %0 : i32
}

// MLIR-LABEL: func.func @gate
func.func @gate(%arg0: i32, %condition: i1) -> i32 {
  // MLIR: %{{.*}} = xls.gate %{{.*}}, %{{.*}} : (i1, i32) -> i32
  %0 = xls.gate %condition, %arg0 : (i1, i32) -> i32
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

// XLS-LABEL: complex_literal
func.func @complex_literal() -> tuple<tuple<i1, i2, tuple<i32, i32>>, !xls.array<2 x i3>> {
  // XLS: ret {{.*}}: ((bits[1], bits[2], (bits[32], bits[32])), bits[3][2]) = literal(value=((1, 2, (10, 0)), [4, 5]),
  %lit = xls.literal : tuple<tuple<i1, i2, tuple<i32, i32>>, !xls.array<2 x i3>> {
    %0 = "xls.constant_scalar"() <{value = true}> : () -> i1
    %1 = "xls.constant_scalar"() <{value = -2 : i2}> : () -> i2
    %2 = "xls.constant_scalar"() <{value = 10 : i32}> : () -> i32
    %3 = "xls.constant_scalar"() <{value = 0 : i32}> : () -> i32
    %4 = "xls.tuple"(%2, %3) : (i32, i32) -> tuple<i32, i32>
    %5 = "xls.tuple"(%0, %1, %4) : (i1, i2, tuple<i32, i32>) -> tuple<i1, i2, tuple<i32, i32>>
    %6 = "xls.constant_scalar"() <{value = -4 : i3}> : () -> i3
    %7 = "xls.constant_scalar"() <{value = -3 : i3}> : () -> i3
    %8 = xls.array %6, %7 : (i3, i3) -> !xls.array<2 x i3>
    %final = "xls.tuple"(%5, %8) : (tuple<i1, i2, tuple<i32, i32>>, !xls.array<2 x i3>) -> tuple<tuple<i1, i2, tuple<i32, i32>>, !xls.array<2 x i3>>
    xls.yield %final : tuple<tuple<i1, i2, tuple<i32, i32>>, !xls.array<2 x i3>>
  }
  return %lit : tuple<tuple<i1, i2, tuple<i32, i32>>, !xls.array<2 x i3>>
}

xls.chan @mychan : i32

// XLS-LABEL: proc eproc({{.*}}: bits[32], {{.*}}: (bits[32], bits[1]), {{.*}}: bits[1], {{.*}}: (bits[1], bits[8], bits[7]), init={
xls.eproc @eproc(%arg0: i32 loc("a"), %arg1: tuple<i32, i1> loc("b"),
    %arg2: i1 loc("pred"), %arg3: bf16 loc("fp")) zeroinitializer {
  // XLS: [[literal:[^ ]*]]: bits[32] = literal(value=6
  %0 = "xls.constant_scalar"() { value = 6 : i32 } : () -> i32
  %tkn1 = "xls.after_all"() : () -> !xls.token
  %tkn_out, %result = xls.blocking_receive %tkn1, @mychan : i32
  %tkn2 = xls.send %tkn_out, %0, @mychan : i32
  %tkn_out2, %result2, %done = xls.nonblocking_receive %tkn2, %arg2, @mychan : i32
  // XLS: [[not_pred:[^ ]*]]: bits[1] = not(
  %not_pred = xls.not %arg2 : i1
  // XLS: next_value(param=a, value=a, predicate=pred
  // XLS: next_value(param=a, value=[[literal]], predicate=[[not_pred]]
  %2 = xls.next_value [%arg2, %arg0], [%not_pred, %0] : (i32, i32) -> i32
  xls.yield %2, %arg1, %arg2, %arg3 : i32, tuple<i32, i1>, i1, bf16
}

// XLS-LABEL: proc eproc2({{.*}}: (bits[32], bits[1]), {{.*}}: bits[1], init={
xls.eproc @eproc2(%state: tuple<i32, i1> loc("state"), %pred: i1 loc("pred")) zeroinitializer {
  // XLS: [[literal1:[^ ]*]]: bits[32] = literal(value=6
  %lit1 = "xls.constant_scalar"() { value = 6 : i32 } : () -> i32
  // XLS: [[literal2:[^ ]*]]: bits[1] = literal(value=0
  %lit2 = "xls.constant_scalar"() { value = 0 : i1 } : () -> i1
  // XLS: [[new_tuple:[^ ]*]]: (bits[32], bits[1]) = tuple(
  %new_tuple = "xls.tuple"(%lit1, %lit2) : (i32, i1) -> tuple<i32, i1>

  // XLS: [[not_pred:[^ ]*]]: bits[1] = not(
  %not_pred = xls.not %pred : i1

  // XLS: next_value(param=state, value=state, predicate=pred
  // XLS: next_value(param=state, value=[[new_tuple]], predicate=[[not_pred]]
  %next_state = xls.next_value [%pred, %state], [%not_pred, %new_tuple] : (tuple<i32, i1>, tuple<i32, i1>) -> tuple<i32, i1>
  xls.yield %next_state, %pred : tuple<i32, i1>, i1
}

