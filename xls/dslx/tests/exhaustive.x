// Copyright 2026 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/// Exhaustive DSLX syntax showcase and editor stress test.
///
/// This standalone module exercises DSLX language constructs accepted by the
/// parser and typechecker alongside extreme editor cases (deep nesting, very
/// long identifiers, and long lines) for manual and automated testing of
/// syntax highlighting, outline, folding, linkification, and symbol navigation.
///
/// Note on comments: DSLX supports line comments only; /* block comment */
/// markers have no special meaning and are plain text inside a line comment.
//
// Multi-line comment block for comment-folding and UTF-8 rendering tests:
//   Line 1: Greek letters alpha/beta (α, β), arrow (→), infinity (∞).
//   Line 2: Nested punctuation () [] {} <> :: -> => .. ..= ... ++ +:
//   Line 3: Trailing note about `import` vs `use`: DSLX forbids mixing `import`
//           and `#![feature(use_syntax)]` (`use`) in the same module, so this
//           file exercises all `import` forms.

#![feature(explicit_state_access)]
#![feature(generics)]

// ============================================================================
// Imports (all `import` forms: stdlib, aliased stdlib, path, aliased path)
// ============================================================================

import std;
import std as stdlib_alias;
import xls.dslx.tests.mod_imported;
import xls.dslx.tests.mod_imported as imported_mod;

// ============================================================================
// Constants, Literals, Types, and Module-Level `const_assert!`
// ============================================================================

/// Public typed constant with an intentionally extreme identifier length.
pub const ANNOYING_LONG_IDENTIFIER_THAT_REALLY_SHOULD_NOT_BE_THAT_LONG: u32 = u32:0x1234_5678;

/// Private untyped and typed constants covering numeric, bool, char, and string literals.
const UNTYPED_CONST = u32:42;
pub const HEX_CONST: u32 = u32:0xDEAD_BEEF;
pub const BIN_CONST: u8 = u8:0b_1010_0101;
pub const NEG_CONST: s32 = s32:-42;
pub const BOOL_CONST: bool = true;
pub const MAX_U16: u16 = u16::MAX;
pub const ZERO_U16: u16 = u16::ZERO;
pub const WIDE_BITS: bits[12] = bits[12]:0xABC;
pub const EXPLICIT_UN: uN[10] = uN[10]:512;
pub const EXPLICIT_SN: sN[10] = sN[10]:-128;
pub const XN_SIGNED: xN[true][8] = xN[true][8]:-7;
pub const XN_UNSIGNED: xN[false][8] = xN[false][8]:7;
pub const CHAR_A: u8 = 'A';
pub const CHAR_ESCAPES: u8[5] = ['\n', '\r', '\t', '\\', '\''];
pub const CHAR_HEX: u8 = '\x7f';
pub const GREETING: u8[18] = "hello\nworld\t\"ok\"\\?";
pub const ELLIPSIS_ARRAY: u32[6] = [u32:1, u32:2, ...];

const_assert!(UNTYPED_CONST == u32:42);

const_assert!(ANNOYING_LONG_IDENTIFIER_THAT_REALLY_SHOULD_NOT_BE_THAT_LONG == u32:0x1234_5678);

const_assert!(MAX_U16 == u16:0xFFFF && ZERO_U16 == u16:0);

const_assert!(ELLIPSIS_ARRAY[5] == u32:2);

// ============================================================================
// Type Aliases (primitive, SV-attributed, tuple, multi-dim array, imported)
// ============================================================================

pub type Word = u32;

#[sv_type("exhaustive_byte_t")]
pub type SvByte = u8;

type WordPair = (Word, Word);
pub type Matrix2x4 = u8[4][2];
pub type ImportedPoint = mod_imported::Point;
pub type AliasedEnum = imported_mod::MyEnum;

// ============================================================================
// Enums (with SV attribute, explicit discriminants, and long identifiers)
// ============================================================================

#[sv_type("exhaustive_opcode_e")]
pub enum Opcode : u4 {
    NOP = 0,
    ADD = 1,
    SUB = 0b0010,
    XOR = 0x3,
    SHIFT = 4,
    CLAMP = 5,
    VERY_LONG_ENUM_VARIANT_NAME_FOR_EDITOR_OUTLINE_AND_SYMBOL_STRESS_TESTING = 15,
}

enum LocalTag : u2 {
    ALPHA = 0,
    BETA = 1,
    GAMMA = 2,
    DELTA = 3,
}

// ============================================================================
// Structs & `impl` Blocks (attributes, empty, parametric, generic, ToBits)
// ============================================================================

#[sv_type("exhaustive_header_s")]
#[derive(ToBits)]
pub struct Header { opcode: Opcode, tag: u8, flags: bool[2], length: Word }

pub struct EmptyMarker {}

pub struct Packet<N: u32, M: u32 = {N + u32:1}> { payload: uN[N], meta: uN[M], active: bool }

pub struct GenericBox<T: type = u32> { value: T }

pub struct RidiculouslyLongStructNameForTestingOutlineFoldingAndSymbolNavigation {
    extremely_verbose_primary_accumulator_field_name_for_stress_testing: u32,
    extremely_verbose_secondary_configuration_field_name_for_stress_testing: u16,
}

impl Header {
    const VERSION: u8 = u8:1;

    pub fn new(opcode: Opcode, tag: u8, flags: bool[2], length: Word) -> Self {
        Header { opcode, tag, flags, length }
    }

    pub fn checksum(self) -> Word {
        (self.opcode as Word) + (self.tag as Word) + self.length + (Header::VERSION as Word)
    }

    pub fn with_length(self: Self, length: Word) -> Self { Header { length, ..self } }
}

impl Packet<N, M> {
    const TOTAL_DATA_BITS: u32 = N + M;

    pub fn make(payload: uN[N], meta: uN[M], active: bool) -> Self {
        Packet<N, M> { payload, meta, active }
    }

    pub fn packed(self) -> uN[N + M] { self.meta ++ self.payload }
}

impl GenericBox<T> {
    pub fn wrap(value: T) -> Self { GenericBox<T> { value } }

    pub fn unwrap(self) -> T { self.value }
}

// ============================================================================
// Functions: Parametrics, Operators, Slicing, Destructuring, Control Flow
// ============================================================================

/// Generic identity with a default type parameter and explicit type annotation.
pub fn generic_identity<T: type = u32>(value: T) -> T { value }

/// Parametric function with derived parametric expression and tick identifiers.
pub fn parametric_widen<N: u32, M: u32 = {N + u32:4}>(x': uN[N]) -> uN[M] {
    const DELTA: u32 = M - N;
    const_assert!(DELTA >= u32:0);
    type LocalWide = uN[M];
    let widened': LocalWide = widening_cast<LocalWide>(x');
    widened' + LocalWide:1
}

/// Exercises arithmetic, bitwise, logical, comparison, unary, cast, concat, and slices.
pub fn operators_and_slices(a: u16, b: u16, signed_x: s16) -> (u16, u8, bool) {
    let arith: u16 = ((a + b) - (a % u16:7)) * u16:2 / u16:2;
    let bitwise: u16 = ((a & b) | (a ^ b)) ^ (!a);
    let shifted: u16 = (a << u16:2) >> u16:1;
    let neg_s: s16 = -signed_x;
    let cmp: bool = (a == b) || (a != b && a <= b) || (a < b) || (a > b) || (a >= b);

    let low_byte: u8 = a[0:8];
    let high_byte: u8 = a[8:];
    let prefix_nibble: u4 = a[:4];
    let width_slice: u4 = a[4+:u4];
    let rebuilt: u8 = prefix_nibble ++ width_slice;
    let updated_bits: u16 = bit_slice_update(arith, u4:4, rebuilt);

    let mix: u16 = updated_bits ^ bitwise ^ shifted ^ (neg_s as u16);
    (mix, low_byte ^ high_byte, cmp)
}

/// Exercises local declarations, tuple destructuring, rest-of-tuple, and index syntax.
pub fn destructuring_and_locals(pair: WordPair) -> Word {
    const OFFSET: Word = Word:3;
    type LocalTriple = (Word, (Word, Word, Word), bool);
    const_assert!(OFFSET == u32:3);

    let nested: LocalTriple = (pair.0, (pair.1, pair.0 + OFFSET, pair.1 + OFFSET), true);
    let (head, (mid, ..), flag) = nested;
    let (_, (_, last_mid, _), _) = nested;
    let boxed = GenericBox<Word>::wrap(head + mid + last_mid);
    let tag = if flag { LocalTag::ALPHA } else { LocalTag::BETA };
    match tag {
        LocalTag::ALPHA => boxed.unwrap(),
        _ => !boxed.unwrap(),
    }
}

/// Exercises `if` / `else if` / `else`, `const if`, `match`, and `const match`.
pub fn conditionals_and_matches<MODE: u32>(op: Opcode, x: u8, t: (u4, u4, u4)) -> u32 {
    let branch_val: u32 = if x == u8:0 {
        u32:10
    } else if x < u8:10 {
        u32:20
    } else {
        u32:30
    };

    let const_branch: u32 = const if MODE == u32:0 {
                                u16:100
                            } else if MODE == u32:1 {
                                u32:200
                            } else {
                                u8:50
                            } as
                            u32;

    let op_val: u32 = match op {
        Opcode::NOP => u32:0,
        Opcode::ADD | Opcode::SUB => u32:1,
        Opcode::XOR => u32:2,
        Opcode::SHIFT => u32:3,
        Opcode::CLAMP => u32:4,
        Opcode::VERY_LONG_ENUM_VARIANT_NAME_FOR_EDITOR_OUTLINE_AND_SYMBOL_STRESS_TESTING => u32:15,
    };

    let range_val: u32 = match x {
        u8:0..u8:4 => u32:1,
        u8:4..=u8:15 => u32:2,
        u8:16 | u8:32 | u8:64 => u32:3,
        _ => u32:4,
    };

    let tuple_val: u32 = match t {
        (u4:0, ..) => u32:10,
        (u4:1, a, _) => a as u32,
        (_, u4:2, c) => c as u32,
        _ => u32:99,
    };

    let const_matched: u32 = const match MODE {
        u32:0 => u32:1000,
        u32:1..u32:3 => u32:2000,
        _ => u32:3000,
    };

    branch_val + const_branch + op_val + range_val + tuple_val + const_matched
}

/// Exercises `for`, `const for`, `unroll_for!`, array indexing, lambdas, and `map`.
pub fn loops_and_lambdas(arr: u32[4]) -> (u32, u32, u32[4]) {
    let sum_range: u32 = for (i, acc): (u32, u32) in u32:0..u32:4 {
        acc + arr[i]
    }(u32:0);

    let sum_elems: u32 = for (elem, acc) in arr {
        acc + elem
    }(u32:0);

    let const_sum: u32 = const for (i, acc) in u32:0..u32:4 {
        acc + i
    }(u32:0);

    let unrolled: u32 = unroll_for! (i, acc): (u32, u32) in u32:0..u32:4 {
        acc + (arr[i] ^ i)
    }(u32:0);

    let scale: u32 = u32:2;
    let mapped: u32[4] = map(arr, |v| -> u32 { v * scale + const_sum });
    let nullary = (|| u32:5)();
    let binary = (|p: u32, q: u32| -> u32 { p + q })(sum_range, nullary);
    let implicit_body = map(u32:0..u32:4, |i| i + u32:1);

    (binary + sum_elems, unrolled + implicit_body[3], mapped)
}

/// Exercises standard library calls and built-in functions/macros.
pub fn builtins_showcase(x: u8, arr: u8[4]) -> u32 {
    assert!(x != u8:0xFF || true, "always_holds");
    assert_fmt!(bit_count<u8>() == u32:8, "bit_count_u8_mismatch");
    cover!("exhaustive_cover_point", x > u8:0);
    trace!(x);
    trace_fmt!("x = {:#x}", x);
    vtrace_fmt!(u32:1, "verbose x = {}", x);

    let safe_fail: u8 = if false { fail!("unreachable_branch", u8:0) } else { x };
    let z: u8 = zero!<u8>();
    let ones: u8 = all_ones!<u8>();
    let gated: u8 = gate!(x != u8:0, ones);
    let widened: u16 = widening_cast<u16>(safe_fail);
    let checked: u8 = checked_cast<u8>(widened);
    let leading: u8 = clz(x);
    let trailing: u8 = ctz(x);
    let log2_val: u32 = std::clog2(u32:16);
    let reductions: u3 = and_reduce(x) ++ or_reduce(x) ++ xor_reduce(x);
    let reversed: u8 = rev(x);
    let rev_arr: u8[4] = array_rev(arr);
    let sz: u32 = array_size(arr);
    let sliced: u8[2] = array_slice(arr, u32:1, [u8:0, u8:0]);
    let updated: u8[4] = update(arr, u32:0, x);
    let decoded: u8 = decode<u8>(x[0:3]);
    let encoded: u3 = encode(decoded);
    let oh: u9 = one_hot(x, true);
    let oh_sel: u8 = one_hot_sel(u4:0b0010, arr);
    let prio_sel: u8 = priority_sel(u4:0b0110, arr, u8:0);
    let sext: s16 = signex(x as s8, s16:0);
    let (um_lo, um_hi) = umulp(x, u8:3);
    let (sm_lo, sm_hi) = smulp(x as s8, s8:2);
    let rng: u8[4] = u8:0..u8:4;
    let zipped: (u8, u8)[4] = zip(arr, rng);
    let elem_cnt: u32 = element_count<u8[4]>();
    let lsb_bit: u1 = std::lsb(x);
    let pop: u8 = stdlib_alias::popcount(x);
    let imp_lsb: u1 = mod_imported::my_lsb(x[0:3]);
    let imp_const_lsb: u1 = imported_mod::my_lsb_uses_const(x[0:3]);

    (z as u32) + (gated as u32) + (checked as u32) + (leading as u32) + (trailing as u32) + log2_val +
    (reductions as u32) + (reversed as u32) + (rev_arr[0] as u32) + sz + (sliced[0] as u32) +
    (updated[0] as u32) + (encoded as u32) + (oh[0:8] as u32) + (oh_sel as u32) + (prio_sel as u32) +
    (sext as u32) + ((um_lo + um_hi) as u32) + ((sm_lo + sm_hi) as u32) + ((zipped[1]).0 as u32) +
    elem_cnt + (lsb_bit as u32) + (pop as u32) + (imp_lsb as u32) + (imp_const_lsb as u32)
}

// ============================================================================
// New-Style Procs (`proc` + `impl`, channels, state, spawn, test proc)
// ============================================================================

/// Parametric stateful worker proc exercising explicit state access and channel ops.
pub proc Accumulator<WIDTH: u32> {
    data_in: chan<uN[WIDTH]> in,
    data_out: chan<uN[WIDTH]> out,
    running_sum: uN[WIDTH],
}

impl Accumulator<WIDTH> {
    pub fn new(data_in: chan<uN[WIDTH]> in, data_out: chan<uN[WIDTH]> out) -> Self {
        Accumulator<WIDTH> { data_in, data_out, running_sum: zero!<uN[WIDTH]>() }
    }

    fn next(self) {
        let current = read(self.running_sum);
        let (tok, incoming) = recv(join(), self.data_in);
        let updated = current + incoming;
        send(tok, self.data_out, updated);
        write(self.running_sum, updated);
    }
}

#[test]
proc AccumulatorTest {
    test_terminator: chan<bool> out,
    to_acc: chan<u16> out,
    from_acc: chan<u16> in,
    aux_in: chan<u16> in,
    aux_out: chan<u16> out,
}

impl AccumulatorTest {
    fn new(test_terminator: chan<bool> out) -> Self {
        let (to_acc, acc_in) = chan<u16>("to_acc");
        let (acc_out, from_acc) = chan<u16, 1>("from_acc");
        let (aux_out, aux_in) = chan<u16, 1>("aux_loop");
        Accumulator<u32:16>::new(acc_in, acc_out).spawn();
        AccumulatorTest { test_terminator, to_acc, from_acc, aux_in, aux_out }
    }

    fn next(self) {
        let tok: token = send(join(), self.to_acc, u16:10);
        let (tok, first) = recv(tok, self.from_acc);
        assert_eq(first, u16:10);

        let tok = send_if(tok, self.to_acc, true, u16:5);
        let (tok, second) = recv_if(tok, self.from_acc, true, u16:0);
        assert_eq(second, u16:15);

        let (tok, nb_val, nb_valid) = recv_non_blocking(tok, self.aux_in, u16:77);
        assert_eq(nb_val, u16:77);
        assert_eq(nb_valid, false);

        let (tok, nb_if_val, nb_if_valid) = recv_if_non_blocking(tok, self.aux_in, false, u16:88);
        assert_eq(nb_if_val, u16:88);
        assert_eq(nb_if_valid, false);

        let tok = send_if(tok, self.aux_out, false, u16:99);
        send(tok, self.test_terminator, true);
    }
}

// ============================================================================
// Editor Stress Section: Deep Nesting, Extreme Identifiers, and Long Lines
// ============================================================================

/// Deeply nested blocks, conditionals, matches, loops, tuples, and expressions (12+ levels).
pub fn deeply_nested_stress(seed: u8) -> u32 {
    let outer = {
        if seed < u8:250 {
            match seed[0:2] {
                u2:0 | u2:1 => {
                    for (i, acc): (u32, u32) in u32:0..u32:2 {
                        let step = {
                            if (i & u32:1) == u32:0 {
                                match (seed[2:4], i[0:2]) {
                                    (u2:0, _) => {
                                        let inner_tuple = (
                                            (
                                                ((acc + u32:1), (acc + u32:2)),
                                                ((acc + u32:3), (acc + u32:4)),
                                            ), u32:5,
                                        );
                                        if inner_tuple.1 == u32:5 {
                                            match inner_tuple.0.0.0 {
                                                u32:0 => u32:10,
                                                _ => {
                                                    let deep_val =
                                                        { inner_tuple.0.0.1 + inner_tuple.0.1.0 };
                                                    deep_val + u32:1
                                                },
                                            }
                                        } else {
                                            u32:0
                                        }
                                    },
                                    _ => {
                                        let fallback = {
                                            if seed > u8:4 {
                                                acc + (seed as u32)
                                            } else {
                                                acc + u32:7
                                            }
                                            };
                                        fallback
                                    },
                                }
                            } else {
                                acc + u32:3
                            }
                            };
                        step
                    }(u32:0)
                    },
                _ => { if seed == u8:255 { u32:0 } else { (seed as u32) + u32:11 } },
            }
        } else {
            u32:999
        }
        };
    outer
}

/// Long identifier stress test across function, parameter, struct, and constant names.
pub fn ridiculously_long_function_name_to_stress_test_editor_outline_breadcrumb_and_symbol_panels
    (ridiculously_long_first_parameter_name_for_signature_wrapping_stress: u32,
     ridiculously_long_second_parameter_name_for_signature_wrapping_stress: u16)
    -> RidiculouslyLongStructNameForTestingOutlineFoldingAndSymbolNavigation {
    let combined_very_long_local_binding_identifier_for_hover_and_highlighting: u32 = ridiculously_long_first_parameter_name_for_signature_wrapping_stress +
                                                                                      ANNOYING_LONG_IDENTIFIER_THAT_REALLY_SHOULD_NOT_BE_THAT_LONG;
    RidiculouslyLongStructNameForTestingOutlineFoldingAndSymbolNavigation {
        extremely_verbose_primary_accumulator_field_name_for_stress_testing:
            combined_very_long_local_binding_identifier_for_hover_and_highlighting,
        extremely_verbose_secondary_configuration_field_name_for_stress_testing:
            ridiculously_long_second_parameter_name_for_signature_wrapping_stress,
    }
}

// ============================================================================
// Tests, Quickchecks, and Fuzz Tests
// ============================================================================

#[test]
fn test_imports_structs_and_impls() {
    let hdr = Header::new(Opcode::ADD, u8:7, [true, false], Word:16);
    let updated_hdr = hdr.with_length(Word:32);
    assert_eq(hdr.checksum(), u32:25);
    assert_eq(updated_hdr.checksum(), u32:41);
    assert_ne(hdr.to_bits(), updated_hdr.to_bits());

    let pkt = Packet<u32:8>::make(u8:0xAB, u9:0x155, true);
    assert_eq(Packet<u32:8>::TOTAL_DATA_BITS, u32:17);
    assert_eq(pkt.packed(), u17:0x155AB);

    let boxed = GenericBox<u16>::wrap(u16:99);
    assert_eq(boxed.unwrap(), u16:99);
    assert_eq(generic_identity(u32:123), u32:123);
    assert_eq(generic_identity<u8>(u8:5), u8:5);

    let pt = ImportedPoint { x: u32:3, y: u32:4 };
    assert_lt(pt.x, pt.y);
    assert_le(pt.x, pt.y);
    assert_gt(pt.y, pt.x);
    assert_ge(pt.y, pt.x);
    assert_eq(AliasedEnum::FOO as u8, u8:42);

    let _empty = EmptyMarker {};
    let _local_tag = LocalTag::GAMMA;
    let _mat: Matrix2x4 = [[u8:1, u8:2, u8:3, u8:4], [u8:5, u8:6, u8:7, u8:8]];
    let _sv: SvByte = SvByte:9;
    assert_eq(_sv, u8:9);
}

#[test]
fn test_expressions_control_flow_and_builtins() {
    assert_eq(parametric_widen<u32:8>(u8:9), u12:10);

    let (mix, byte_xor, cmp) = operators_and_slices(u16:0x1234, u16:0x5678, s16:-3);
    assert_ne(mix, u16:0);
    assert_eq(byte_xor, u8:0x26);
    assert_eq(cmp, true);

    assert_eq(destructuring_and_locals((Word:10, Word:20)), Word:43);
    assert_eq(conditionals_and_matches<u32:0>(Opcode::ADD, u8:5, (u4:1, u4:7, u4:0)), u32:1130);
    assert_eq(
        conditionals_and_matches<u32:1>(
            Opcode::VERY_LONG_ENUM_VARIANT_NAME_FOR_EDITOR_OUTLINE_AND_SYMBOL_STRESS_TESTING, u8:16,
            (u4:0, u4:1, u4:2)), u32:2258);

    let (sum_all, unrolled_sum, mapped) = loops_and_lambdas([u32:1, u32:2, u32:3, u32:4]);
    assert_eq(sum_all, u32:25);
    assert_eq(unrolled_sum, u32:16);
    assert_eq(mapped, [u32:8, u32:10, u32:12, u32:14]);

    let b_out = builtins_showcase(u8:0x2A, [u8:1, u8:2, u8:3, u8:4]);
    assert_gt(b_out, u32:0);
}

#[test]
fn test_stress_cases() {
    assert_eq(deeply_nested_stress(u8:0), u32:9);
    assert_eq(deeply_nested_stress(u8:12), u32:15);
    assert_eq(deeply_nested_stress(u8:255), u32:999);

    let long_struct =
        ridiculously_long_function_name_to_stress_test_editor_outline_breadcrumb_and_symbol_panels(
            u32:1, u16:2);
    assert_eq(
        long_struct.extremely_verbose_primary_accumulator_field_name_for_stress_testing,
        u32:0x1234_5679);
    assert_eq(
        long_struct.extremely_verbose_secondary_configuration_field_name_for_stress_testing, u16:2);
}

#[quickcheck]
fn quickcheck_xor_self_is_zero(x: u8) -> bool { (x ^ x) == u8:0 }

#[quickcheck(exhaustive)]
fn quickcheck_exhaustive_nibble_double_not(x: u4) -> bool { !(!x) == x }

#[quickcheck(test_count=16)]
fn quickcheck_count_add_commutes(a: u8, b: u8) -> bool { (a + b) == (b + a) }

#[fuzz_test]
fn fuzz_property_basic(x: u8) -> bool {
    (x ^ x) == u8:0
}

#[fuzz_test(domains=`u32:0..16, [u32:0, u32:1, u32:2]`)]
fn fuzz_property_with_domains(x: u32, y: u32) -> bool {
    (x + y) == (y + x)
}

// Long single-line function with many parameters.
// dslx-fmt::off
pub fn intentionally_unformatted_very_long_single_line_stress_function(param_01: u32, param_02: u32, param_03: u32, param_04: u32, param_05: u32, param_06: u32, param_07: u32, param_08: u32, param_09: u32, param_10: u32, param_11: u32, param_12: u32, param_13: u32, param_14: u32, param_15: u32, param_16: u32, param_17: u32, param_18: u32, param_19: u32, param_20: u32) -> u32 { (param_01 + param_02 + param_03 + param_04 + param_05 + param_06 + param_07 + param_08 + param_09 + param_10 + param_11 + param_12 + param_13 + param_14 + param_15 + param_16 + param_17 + param_18 + param_19 + param_20) ^ (ANNOYING_LONG_IDENTIFIER_THAT_REALLY_SHOULD_NOT_BE_THAT_LONG & u32:0x00FF_FFFF) }
// dslx-fmt::on

#[test]
fn test_long_line_stress() {
    let long_line_result = intentionally_unformatted_very_long_single_line_stress_function(
        u32:1, u32:2, u32:3, u32:4, u32:5, u32:6, u32:7, u32:8, u32:9, u32:10, u32:11, u32:12,
        u32:13, u32:14, u32:15, u32:16, u32:17, u32:18, u32:19, u32:20);
    assert_eq(long_line_result, u32:0x0034_56AA);
}
