// Copyright 2020 The XLS Authors
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
//
// DSLX implementation of an FMA (fused multiply-add) for any given APFloat
// configuration.
// This implementation uses (2 * (FRACTION + 1)) bits of precision for the
// multiply fraction and (3 * (FRACTION + 1)) for the add.
// The results have been tested (not exhaustively, of course! It's a 96-bit
// input space for binary32!) to be bitwise identical to those produced by
// glibc/libm 2.31 (for IEEE binary32 formats).
//
// The fundamentals of the multiply and add are the same as those in the
// standalone ops - the differences arise in the extra precision bits and the
// handling thereof (e.g., 72 vs. 24 bits for the add, for binary32).
import apfloat
import std

type APFloat = apfloat::APFloat;

// Simple utility struct for holding the result of the multiplication step.
struct Product<EXP_CARRY: u32, WIDE_FRACTION: u32> {
  sign: u1,
  bexp: uN[EXP_CARRY],
  fraction: uN[WIDE_FRACTION],
}

// Returns true if the given Product is infinite.
fn is_inf<EXP_CARRY: u32, WIDE_FRACTION: u32>(p: Product<EXP_CARRY, WIDE_FRACTION>) -> u1 {
  p.bexp == std::mask_bits<EXP_CARRY>() && p.fraction == uN[WIDE_FRACTION]:0
}

// Returns true if the given Product is NaN.
fn is_nan<EXP_CARRY: u32, WIDE_FRACTION: u32>(p: Product<EXP_CARRY, WIDE_FRACTION>) -> u1 {
  p.bexp == std::mask_bits<EXP_CARRY>() && p.fraction != uN[WIDE_FRACTION]:0
}

// The first step in FMA: multiply the first two operands, but skip rounding
// and truncation.
// Parametrics:
//   EXP_SZ: The bit width of the exponent of the current type.
//   FRACTION_SZ: The bit width of the fraction of the current type.
//   WIDE_FRACTION: 2x the full fraction size (i.e., including the usually
//    implicit leading "1"), necessary for correct precision.
//   EXP_CARRY: EXP_SZ plus one carry bit.
//   EXP_SIGN_CARRY: EXP_CARRY plus one sign bit.
// For an IEEE binary32 ("float"), these values would be 8, 23, 48, 9, and 10.
fn mul_no_round<EXP_SZ: u32, FRACTION_SZ: u32,
    WIDE_FRACTION: u32 = (FRACTION_SZ + u32:1) * u32:2,
    EXP_CARRY: u32 = EXP_SZ + u32:1,
    EXP_SIGN_CARRY: u32 = EXP_SZ + u32:2>(
        a: APFloat<EXP_SZ, FRACTION_SZ>, b: APFloat<EXP_SZ, FRACTION_SZ>) -> Product {
  // These steps are taken from apfloat_mul_2.x; look there for full comments.
  // Widen the fraction to full size and prepend the formerly-implicit "1".
  let a_fraction = (a.fraction as uN[WIDE_FRACTION]) | (uN[WIDE_FRACTION]:1 << FRACTION_SZ);
  let b_fraction = (b.fraction as uN[WIDE_FRACTION]) | (uN[WIDE_FRACTION]:1 << FRACTION_SZ);

  // Flush subnorms.
  let a_fraction = if a.bexp == uN[EXP_SZ]:0 { uN[WIDE_FRACTION]:0 } else { a_fraction };
  let b_fraction = if b.bexp == uN[EXP_SZ]:0 { uN[WIDE_FRACTION]:0 } else { b_fraction };
  let fraction = a_fraction * b_fraction;

  // Normalize - shift left one place if the top bit is 0.
  let fraction_shift = fraction[-1:] as uN[WIDE_FRACTION];
  let fraction = if fraction_shift == uN[WIDE_FRACTION]:0 { fraction << 1 } else { fraction };

  // e.g., for floats, 0xff -> 0x7f, A.K.A. 127, the exponent bias.
  let bias = std::mask_bits<EXP_SZ>() as sN[EXP_SIGN_CARRY] >> 1;
  let bexp = (a.bexp as sN[EXP_SIGN_CARRY]) + (b.bexp as sN[EXP_SIGN_CARRY]) -
      bias + (fraction_shift as sN[EXP_SIGN_CARRY]);
  let bexp =
      if a.bexp == bits[EXP_SZ]:0 || b.bexp == bits[EXP_SZ]:0 {
        sN[EXP_SIGN_CARRY]:0
      } else {
        bexp
      };

  // Note that we usually flush subnormals. Here, we preserve what we can for
  // compatability with reference implementations.
  // We only do this for the internal product - we otherwise don't handle
  // subnormal values (we flush them to 0).
  let is_subnormal = bexp <= sN[EXP_SIGN_CARRY]:0;
  let result_exp =
      if is_subnormal { uN[EXP_CARRY]:0 } else { bexp as uN[EXP_CARRY] };
  let sub_exp = std::abs(bexp) as uN[EXP_CARRY];
  let result_fraction = if is_subnormal { fraction >> sub_exp } else { fraction };

  // - Overflow infinites - saturate exp, clear fraction.
  let high_exp = std::mask_bits<EXP_CARRY>();
  let result_fraction = if result_exp < high_exp { result_fraction } else { uN[WIDE_FRACTION]:0 };
  let result_exp = if result_exp < high_exp { result_exp as uN[EXP_CARRY] } else { high_exp };

  // - Arg infinites. Any arg is infinite == result is infinite.
  let is_operand_inf = apfloat::is_inf(a) || apfloat::is_inf(b);
  let result_exp = if is_operand_inf { high_exp } else { result_exp };
  let result_fraction = if is_operand_inf {
    uN[WIDE_FRACTION]:0
  } else {
    result_fraction as uN[WIDE_FRACTION]
  };

  // - NaNs. NaN trumps infinities, so we handle it last.
  //   inf * 0 = NaN, i.e.,
  let has_0_arg = a.bexp == uN[EXP_SZ]:0 || b.bexp == uN[EXP_SZ]:0;
  let has_nan_arg = apfloat::is_nan(a) || apfloat::is_nan(b);
  let has_inf_arg = apfloat::is_inf(a) || apfloat::is_inf(b);
  let is_result_nan = has_nan_arg || (has_0_arg && has_inf_arg);
  let result_exp = if is_result_nan { high_exp } else { result_exp };
  let nan_fraction = uN[WIDE_FRACTION]:1 << (uN[WIDE_FRACTION]:1 - uN[WIDE_FRACTION]:1);
  let result_fraction = if is_result_nan { nan_fraction } else { result_fraction };

  let result_sign = a.sign != b.sign;
  let result_sign = if is_result_nan { u1:0 } else { result_sign };

  Product { sign: result_sign, bexp: result_exp, fraction: result_fraction }
}

// The main driver function.
// Many of the steps herein are fully described in the standalone adder or
// multiplier modules, but abridged comments are present here where useful.
//
// Parametrics:
//   EXP_SZ: The bit width of the exponent of the current type.
//   FRACTION_SZ: The bit width of the signficand of the current type.
//   EXP_CARRY: One greater than EXP_SZ, to hold a carry bit.
//   EXP_SIGN_CARRY: One greater than EXP_CARRY, to hold a sign bit.
//   WIDE_FRACTION: Fully-widened fraction to hold all rounding bits.
//   WIDE_FRACTION_CARRY: WIDE_FRACTION plus one carry bit.
//   WIDE_FRACTION_SIGN_CARRY: WIDE_FRACTION_CARRY plus one sign bit.
//   WIDE_FRACTION_LOW_BIT: Position of the LSB in the final fraction within a
//       WIDE_FRACTION element. All bits with lower index are for rounding.
//   WIDE_FRACTION_TOP_ROUNDING: One less than WIDE_FRACTION_LOW_BIT, in other words the
//       most-significant rounding bit.
pub fn fma<EXP_SZ: u32, FRACTION_SZ: u32,
    EXP_CARRY: u32 = EXP_SZ + u32:1,
    EXP_SIGN_CARRY: u32 = EXP_CARRY + u32:1,
    WIDE_FRACTION: u32 = (FRACTION_SZ + u32:1) * u32:3 + u32:1,
    WIDE_FRACTION_CARRY: u32 = WIDE_FRACTION + u32:1,
    WIDE_FRACTION_SIGN_CARRY: u32 = WIDE_FRACTION_CARRY + u32:1,
    WIDE_FRACTION_LOW_BIT: u32 = WIDE_FRACTION - FRACTION_SZ,
    WIDE_FRACTION_TOP_ROUNDING: u32 = WIDE_FRACTION_LOW_BIT - u32:1>(
        a: APFloat<EXP_SZ, FRACTION_SZ>, b: APFloat<EXP_SZ, FRACTION_SZ>,
        c: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
  let ab = mul_no_round<EXP_SZ, FRACTION_SZ>(a, b);

  let greater_exp = if ab.bexp > c.bexp as uN[EXP_CARRY] { ab.bexp }
                    else { c.bexp as uN[EXP_CARRY] };
  let greater_sign = if ab.bexp > c.bexp as uN[EXP_CARRY] { ab.sign } else { c.sign };

  // Make the implicit '1' explicit and flush subnormal "c" to 0 (already
  // done for ab inside mul_no_round()).
  let wide_c = c.fraction as uN[WIDE_FRACTION] | (uN[WIDE_FRACTION]:1 << FRACTION_SZ);
  let wide_c = if c.bexp == uN[EXP_SZ]:0 { uN[WIDE_FRACTION]:0 } else { wide_c };

  // Align AB and C so that the implicit '1' is in the MSB.
  // For binary32: so shift by 73-48 for AB, and 73-24 for C.
  let wide_ab =
      (ab.fraction as uN[WIDE_FRACTION]) << (WIDE_FRACTION - ((FRACTION_SZ + u32:1) * u32:2));
  let wide_c = wide_c << (WIDE_FRACTION - (FRACTION_SZ + u32:1));

  // Shift the operands into their correct positions.
  let rshift_ab = greater_exp - ab.bexp;
  let rshift_c = greater_exp - (c.bexp as uN[EXP_CARRY]);
  let shifted_ab = wide_ab >> rshift_ab;
  let shifted_c = wide_c >> rshift_c;

  // Calculate the sticky bits.
  let dropped_ab =
      wide_ab << ((WIDE_FRACTION as uN[EXP_CARRY] - rshift_ab) as uN[WIDE_FRACTION]);
  let dropped_c =
      wide_c << ((WIDE_FRACTION as uN[EXP_CARRY] - rshift_c) as uN[WIDE_FRACTION]);
  let dropped_c =
      if rshift_c >= (WIDE_FRACTION as uN[EXP_CARRY]) { wide_c } else { dropped_c };
  let sticky_ab = (dropped_ab != uN[WIDE_FRACTION]:0) as uN[WIDE_FRACTION];
  let sticky_c = (dropped_c != uN[WIDE_FRACTION]:0) as uN[WIDE_FRACTION];

  // Add the sticky bit and extend the operands with the sign and carry bits.
  let shifted_ab = (shifted_ab | sticky_ab) as sN[WIDE_FRACTION_SIGN_CARRY];
  let shifted_c = (shifted_c | sticky_c) as sN[WIDE_FRACTION_SIGN_CARRY];

  // Set the operands' signs.
  let shifted_ab = if ab.sign != greater_sign { -shifted_ab } else { shifted_ab };
  let shifted_c = if c.sign != greater_sign { -shifted_c } else { shifted_c };

  // Addition!
  let sum_fraction = shifted_ab + shifted_c;
  let fraction_is_zero = sum_fraction == sN[WIDE_FRACTION_SIGN_CARRY]:0;
  let result_sign = match (fraction_is_zero, sum_fraction < sN[WIDE_FRACTION_SIGN_CARRY]:0) {
    (true, _) => u1:0,
    (false, true) => !greater_sign,
    _ => greater_sign,
  };

  // Chop off the sign bit (after applying it, if necessary).
  let abs_fraction =
      (if sum_fraction < sN[WIDE_FRACTION_SIGN_CARRY]:0 { -sum_fraction }
       else { sum_fraction })
      as uN[WIDE_FRACTION_CARRY];

  // Normalize.
  let carry_bit = abs_fraction[-1:];
  let carry_fraction = (abs_fraction >> uN[WIDE_FRACTION_CARRY]:1) as uN[WIDE_FRACTION];
  let carry_fraction = carry_fraction | (abs_fraction[0:1] as uN[WIDE_FRACTION]);

  // If high bits were cancelled, shift the result back into the MSB (ignoring
  // the zeroed carry bit, which is handled above).
  let leading_zeroes = clz(abs_fraction);
  let cancel = leading_zeroes > uN[WIDE_FRACTION_CARRY]:1;
  let cancel_fraction =
      (abs_fraction << (leading_zeroes - uN[WIDE_FRACTION_CARRY]:1)) as uN[WIDE_FRACTION];
  let shifted_fraction = match(carry_bit, cancel) {
    (true, false) => carry_fraction,
    (false, true) => cancel_fraction,
    (false, false) => abs_fraction as uN[WIDE_FRACTION],
    _ => fail!("carry_and_cancel", uN[WIDE_FRACTION]:0)
  };

  // Similar to the rounding in apfloat_add_2, except that the fraction
  // starts at the bit below instead of bit 3.
  // For binary32, normal_chunk will be bits 0-48 (inclusive), stopping
  // immediately below the first bit in the final fraction.
  let normal_chunk = shifted_fraction[0:(WIDE_FRACTION_LOW_BIT - u32:1) as s32];
  let half_way_chunk =
      shifted_fraction[(WIDE_FRACTION_LOW_BIT - u32:2) as s32:(WIDE_FRACTION_LOW_BIT as s32)];
  let half_of_extra =
      uN[WIDE_FRACTION_TOP_ROUNDING]:1 << (WIDE_FRACTION_LOW_BIT - u32:2);
  let do_round_up =
      if (normal_chunk > half_of_extra) | (half_way_chunk == u2:0x3) { u1:1 }
      else { u1:0 };
  let rounded_fraction =
      if do_round_up {
        shifted_fraction as uN[WIDE_FRACTION_CARRY] +
            (uN[WIDE_FRACTION_CARRY]:1 << (WIDE_FRACTION_LOW_BIT - u32:1))
      } else {
        shifted_fraction as uN[WIDE_FRACTION_CARRY]
      };

  let rounding_carry = rounded_fraction[-1:];
  let result_fraction =
      (rounded_fraction >> ((WIDE_FRACTION_LOW_BIT - u32:1) as uN[WIDE_FRACTION_CARRY]))
      as uN[FRACTION_SZ];

  let bexp =
      greater_exp as sN[EXP_SIGN_CARRY] +
      rounding_carry as sN[EXP_SIGN_CARRY] + sN[EXP_SIGN_CARRY]:1 -
      leading_zeroes as sN[EXP_SIGN_CARRY];
  let bexp = if fraction_is_zero { sN[EXP_SIGN_CARRY]:0 } else { bexp };
  let bexp =
      if bexp < sN[EXP_SIGN_CARRY]:0 { uN[EXP_CARRY]:0 }
      else { (bexp as uN[EXP_CARRY]) };

  // Standard special case handling follows.

  // If the exponent underflowed, don't bother with denormals. Just flush to 0.
  let result_fraction = if bexp == uN[EXP_CARRY]:0 { uN[FRACTION_SZ]:0 }
                        else { result_fraction };

  // Handle exponent overflow infinities.
  let saturated_exp = std::mask_bits<EXP_SZ>() as uN[EXP_CARRY];
  let max_exp = std::mask_bits<EXP_SZ>();
  let result_fraction = if bexp < saturated_exp { result_fraction } else { uN[FRACTION_SZ]:0 };
  let result_exp = if bexp < saturated_exp { bexp as uN[EXP_SZ] } else { max_exp };

  // Handle arg infinities.
  let is_operand_inf = is_inf(ab) | apfloat::is_inf(c);
  let result_exp = if is_operand_inf { max_exp } else { result_exp };
  let result_fraction = if is_operand_inf { uN[FRACTION_SZ]:0 } else { result_fraction };
  // Result infinity is negative iff all infinite operands are neg.
  let has_pos_inf = (is_inf(ab) & (ab.sign == u1:0)) |
                    (apfloat::is_inf(c) & (c.sign == u1:0));
  let result_sign = if is_operand_inf { !has_pos_inf } else { result_sign };

  // Handle NaN; NaN trumps infinities, so we handle it last.
  // -inf + inf = NaN, i.e., if we have both positive and negative inf.
  let has_neg_inf =
      (is_inf(ab) & (ab.sign == u1:1)) |
      (apfloat::is_inf(c) & (c.sign == u1:1));
  let is_result_nan = is_nan(ab) |
      apfloat::is_nan(c) | (has_pos_inf & has_neg_inf);
  let result_exp = if is_result_nan { max_exp } else { result_exp };
  let result_fraction =
      if is_result_nan {
        uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:4)
      } else {
        result_fraction
      };
  let result_sign = if is_result_nan { u1:0 } else { result_sign };
  let is_result_inf = has_pos_inf | has_neg_inf;

  APFloat<EXP_SZ, FRACTION_SZ>{ sign: result_sign, bexp: result_exp as uN[EXP_SZ],
                           fraction: result_fraction as uN[FRACTION_SZ] }
}

import bfloat16
import float32
import float64
type BF16 = bfloat16::BF16;
type F32 = float32::F32;
type F64 = float64::F64;

#[test]
fn smoke() {
    let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23: 0 };
    let one_point_one = F32 { sign: u1:0, bexp: u8:127, fraction: u23: 0xccccd };
    let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23: 0x58cccd };
    let a = twenty_seven_point_one;
    let b = one_point_one;
    let c = zero;
    let expected = F32 { sign: u1:0, bexp: u8:0x83, fraction: u23:0x6e7ae2 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_f32() {
  let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23: 0 };
  let one_point_zero = F32 { sign: u1:0, bexp: u8:127, fraction: u23: 0 };
  let a = one_point_zero;
  let b = one_point_zero;
  let c = one_point_zero;
  let expected = F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_f64() {
  let zero = F64 { sign: u1:0, bexp: u11:0, fraction: u52: 0 };
  let one_point_zero = F64 { sign: u1:0, bexp: u11:1023, fraction: u52: 0 };
  let a = one_point_zero;
  let b = one_point_zero;
  let c = one_point_zero;
  let expected = F64 { sign: u1:0, bexp: u11:1024, fraction: u52:0 };
  let actual = fma<u32:11, u32:52>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_bf16() {
  let zero = BF16 { sign: u1:0, bexp: u8:0, fraction: u7: 0 };
  let one_point_zero = BF16 { sign: u1:0, bexp: u8:127, fraction: u7: 0 };
  let a = one_point_zero;
  let b = one_point_zero;
  let c = one_point_zero;
  let expected = BF16 { sign: u1:0, bexp: u8:128, fraction: u7:0 };
  let actual = fma<u32:8, u32:7>(a, b, c);
  assert_eq(expected, actual)
}

// Too complicated to be fully descriptive:
// (3250761 x -0.00542...) + 456.31...
// This set of tests will use the same inputs (or as close as is possible).
#[test]
fn manual_case_a_f32() {
  let a = F32 { sign: u1:0, bexp: u8:0x97, fraction: u23:0x565d43 };
  let b = F32 { sign: u1:1, bexp: u8:0x77, fraction: u23:0x319a49 };
  let c = F32 { sign: u1:0, bexp: u8:0x87, fraction: u23:0x642891 };
  let expected = F32 { sign: u1:1, bexp: u8:0x90, fraction: u23:0x144598 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn manual_case_a_f64() {
  let a = F64 { sign: u1:0, bexp: u11:0x417, fraction: u52:0x565d43 };
  let b = F64 { sign: u1:1, bexp: u11:0x3f7, fraction: u52:0x319a49 };
  let c = F64 { sign: u1:0, bexp: u11:0x407, fraction: u52:0x642891 };
  let expected = F64 { sign: u1:1, bexp: u11:0x40e, fraction: u52:0xfe000010f26c7 };
  let actual = fma<u32:11, u32:52>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn manual_case_a_bf16() {
  let a = BF16 { sign: u1:0, bexp: u8:0x97, fraction: u7:0x2b };
  let b = BF16 { sign: u1:1, bexp: u8:0x77, fraction: u7:0x18 };
  let c = BF16 { sign: u1:0, bexp: u8:0x87, fraction: u7:0x32 };
  let expected = BF16 { sign: u1:1, bexp: u8:0x8f, fraction: u7:0x4a };
  let actual = fma<u32:8, u32:7>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_zero() {
  let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23: 0 };
  let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23: 0x58cccd };
  let a = twenty_seven_point_one;
  let b = twenty_seven_point_one;
  let c = zero;
  let expected = F32 { sign: u1:0, bexp: u8:0x88, fraction: u23:0x379a3e };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one() {
  let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23: 0 };
  let one_point_zero = F32 { sign: u1:0, bexp: u8:127, fraction: u23: 0 };
  let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23: 0x58cccd };
  let a = twenty_seven_point_one;
  let b = twenty_seven_point_one;
  let c = one_point_zero;
  let expected = F32 { sign: u1:0, bexp: u8:0x88, fraction: u23:0x37da3e };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one_point_one() {
  let one_point_one = F32 { sign: u1:0, bexp: u8:127, fraction: u23: 0xccccd };
  let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23: 0x58cccd };
  let a = twenty_seven_point_one;
  let b = twenty_seven_point_one;
  let c = one_point_one;
  let expected = F32 { sign: u1:0, bexp: u8:0x88, fraction: u23:0x37e0a4 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_a() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x50, fraction: u23:0x1a8ddc };
  let b = F32 { sign: u1:0x1, bexp: u8:0xcb, fraction: u23:0xee7ac };
  let c = F32 { sign: u1:0x1, bexp: u8:0xb7, fraction: u23:0x609f18 };
  let expected = F32 { sign: u1:1, bexp: u8:0xb7, fraction: u23:0x609f18 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_b() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x23, fraction: u23:0x4d3a41 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x30, fraction: u23:0x35a901 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x96, fraction: u23:0x627c62 };
  let expected = F32 { sign: u1:0, bexp: u8:0x96, fraction: u23:0x627c62 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_c() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x71, fraction: u23:0x2f0932 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xe5, fraction: u23:0x416b76 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xcb, fraction: u23:0x5fd32a };
  let expected = F32 { sign: u1:1, bexp: u8:0xd8, fraction: u23:0x4386a };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_d() {
  let a = F32 { sign: u1:0x0, bexp: u8:0xac, fraction: u23:0x1d0d22 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xdb, fraction: u23:0x2fe688 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xa9, fraction: u23:0x2be1d2 };
  let expected = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x0 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_e() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x7b, fraction: u23:0x25e79f };
  let b = F32 { sign: u1:0x1, bexp: u8:0xff, fraction: u23:0x207370 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x39, fraction: u23:0x6bb348 };
  let expected = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x80000 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_f() {
  let a = F32 { sign: u1:0x1, bexp: u8:0xe0, fraction: u23:0x3cdaa8 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x96, fraction: u23:0x52549c };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1c, fraction: u23:0x21e0fd };
  let expected = F32 { sign: u1:0, bexp: u8:0xf8, fraction: u23:0x1b29c9 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_g() {
  let a = F32 { sign: u1:0x1, bexp: u8:0xc4, fraction: u23:0x73b59a };
  let b = F32 { sign: u1:0x0, bexp: u8:0xa6, fraction: u23:0x1631c0 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x29, fraction: u23:0x5b3d33 };
  let expected = F32 { sign: u1:1, bexp: u8:0xec, fraction: u23:0xefbc5 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_h() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x9b, fraction: u23:0x3f50d4 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x7b, fraction: u23:0x4beeb5 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x37, fraction: u23:0x6ad17c };
  let expected = F32 { sign: u1:1, bexp: u8:0x98, fraction: u23:0x18677d };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_i() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x66, fraction: u23:0x36e592 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xc8, fraction: u23:0x2b5bf1 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x52, fraction: u23:0x12900b };
  let expected = F32 { sign: u1:0, bexp: u8:0xaf, fraction: u23:0x74da11 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_j() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x88, fraction: u23:0x0f0e03 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xb9, fraction: u23:0x36006d };
  let c = F32 { sign: u1:0x1, bexp: u8:0xaa, fraction: u23:0x358b6b };
  let expected = F32 { sign: u1:0, bexp: u8:0xc2, fraction: u23:0x4b6865 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_k() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x29, fraction: u23:0x2fd76d };
  let b = F32 { sign: u1:0x1, bexp: u8:0xce, fraction: u23:0x63eded };
  let c = F32 { sign: u1:0x0, bexp: u8:0xfd, fraction: u23:0x21adee };
  let expected = F32 { sign: u1:0, bexp: u8:0xfd, fraction: u23:0x21adee };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_l() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x6a, fraction: u23:0x09c1b9 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x7c, fraction: u23:0x666a52 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x80, fraction: u23:0x626bcf };
  let expected = F32 { sign: u1:1, bexp: u8:0x80, fraction: u23:0x626bcf };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_m() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x70, fraction: u23:0x41e2db };
  let b = F32 { sign: u1:0x1, bexp: u8:0xd1, fraction: u23:0x013c17 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xb9, fraction: u23:0x30313f };
  let expected = F32 { sign: u1:0, bexp: u8:0xc2, fraction: u23:0x4419bf };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_n() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x33, fraction: u23:0x537374 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x40, fraction: u23:0x78fa62 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x09, fraction: u23:0x7cfb29 };
  let expected = F32 { sign: u1:1, bexp: u8:0x09, fraction: u23:0x7cfb36 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_o() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x94, fraction: u23:0x1aeb90 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x88, fraction: u23:0x1ab376 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x9d, fraction: u23:0x15dd1e };
  let expected = F32 { sign: u1:1, bexp: u8:0x9e, fraction: u23:0x288cde };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_p() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x88, fraction: u23:0x1ebb00 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xf6, fraction: u23:0x0950b6 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xfd, fraction: u23:0x6c314b };
  let expected = F32 { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x5e77d4 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_q() {
  let a = F32 { sign: u1:0x0, bexp: u8:0xda, fraction: u23:0x5b328f };
  let b = F32 { sign: u1:0x1, bexp: u8:0x74, fraction: u23:0x157da3 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1b, fraction: u23:0x6a3f25 };
  let expected = F32 { sign: u1:1, bexp: u8:0xd0, fraction: u23:0x000000 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_r() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x34, fraction: u23:0x4da000 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xf4, fraction: u23:0x4bc400 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x33, fraction: u23:0x54476d };
  let expected = F32 { sign: u1:1, bexp: u8:0xaa, fraction: u23:0x23ab4f };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_s() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x27, fraction: u23:0x732d83 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xbb, fraction: u23:0x4b2dcd };
  let c = F32 { sign: u1:0x0, bexp: u8:0x3a, fraction: u23:0x65e4bd };
  let expected = F32 { sign: u1:0, bexp: u8:0x64, fraction: u23:0x410099 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_t() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x17, fraction: u23:0x070770 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x86, fraction: u23:0x623b39 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1e, fraction: u23:0x6ea761 };
  let expected = F32 { sign: u1:1, bexp: u8:0x0c, fraction: u23:0x693bc0 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_u() {
  let a = F32 { sign: u1:0x0, bexp: u8:0xb1, fraction: u23:0x0c8800 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xc6, fraction: u23:0x2b3800 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x22, fraction: u23:0x00c677 };
  let expected = F32 { sign: u1:1, bexp: u8:0xf8, fraction: u23:0x3bfb2b };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_v() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x90, fraction: u23:0x04a800 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x1f, fraction: u23:0x099cb0 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x28, fraction: u23:0x4d6497 };
  let expected = F32 { sign: u1:1, bexp: u8:0x30, fraction: u23:0x0dd0cf };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_w() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x90, fraction: u23:0x0fdde1 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xa8, fraction: u23:0x663085 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x9b, fraction: u23:0x450d69 };
  let expected = F32 { sign: u1:0, bexp: u8:0xba, fraction: u23:0x015c9c };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_x() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x4c, fraction: u23:0x5ca821 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x87, fraction: u23:0x14808c };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1c, fraction: u23:0x585ccf };
  let expected = F32 { sign: u1:1, bexp: u8:0x55, fraction: u23:0x000000 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_y() {
  let a = F32 { sign: u1:0x0, bexp: u8:0xc5, fraction: u23:0x3a123b };
  let b = F32 { sign: u1:0x1, bexp: u8:0x3b, fraction: u23:0x7ee4d9 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x4f, fraction: u23:0x1d4ddc };
  let expected = F32 { sign: u1:1, bexp: u8:0x82, fraction: u23:0x39446d };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_z() {
  let a = F32 { sign: u1:0x1, bexp: u8:0xd4, fraction: u23:0x1b858b };
  let b = F32 { sign: u1:0x1, bexp: u8:0x9e, fraction: u23:0x59fa23 };
  let c = F32 { sign: u1:0x1, bexp: u8:0xb5, fraction: u23:0x3520e4 };
  let expected = F32 { sign: u1:0x0, bexp: u8:0xf4, fraction: u23:0x046c29 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_aa() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x9b, fraction: u23:0x3ac78d };
  let b = F32 { sign: u1:0x0, bexp: u8:0x3b, fraction: u23:0x542cbb };
  let c = F32 { sign: u1:0x1, bexp: u8:0x09, fraction: u23:0x0c609e };
  let expected = F32 { sign: u1:0x1, bexp: u8:0x58, fraction: u23:0x1acde3 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}
