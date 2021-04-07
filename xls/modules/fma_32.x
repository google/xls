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
// DSLX implementation of a 32-bit FMA (fused multiply-add).
// This implementation uses 48 (2 * (23 + 1) bits of precision for the
// multiply significand and and 72 bits (3 * (23 + 1)) for the add.
// The results have been tested (not exhaustively, of course! It's a 96-bit
// input space!) to be bitwise identical to those produced by glibc/libm 2.31.
//
// The fundamentals of the multiply and add are the same as those in the
// standalone ops - the differences arise in the extra precision bits and the
// handling thereof (e.g., 72 vs. 24 bits for the add).
import float32
import std

type F32 = float32::F32;

// Simple utility struct for holding the result of the multiplication step.
struct Product {
  sign: u1,
  bexp: u9,
  sfd: u48,
}

// Returns true if the given Product is infinity.
fn is_inf(p: Product) -> u1 {
  p.bexp == u9:0x1ff && p.sfd == u48:0
}

// Returns true if the given Product is NaNi.
fn is_nan(p: Product) -> u1 {
  p.bexp == u9:0x1ff && p.sfd != u48:0
}

// The first step in FMA: multiply the first two operands, but skip rounding
// and truncation.
fn mul_no_round(a: F32, b:F32) -> Product {
  // These steps are taken from apfloat_mul_2.x; look there for full comments.
  // u48: sfd_width * 2 + 2 implicit one bits.
  let a_sfd = (a.sfd as u48) | u48:0x800000;
  let b_sfd = (b.sfd as u48) | u48:0x800000;

  // Flush subnorms.
  let a_sfd = u48:0 if a.bexp == u8:0 else a_sfd;
  let b_sfd = u48:0 if b.bexp == u8:0 else b_sfd;
  let sfd = a_sfd * b_sfd;

  // Normalize - shift left one place if the top bit is 0.
  let sfd_shift = sfd[-1:] as u48;
  let sfd = sfd << 1 if sfd_shift == u48:0 else sfd;

  // s10 = 8-bit exponent + carry bit + 1 sign bit.
  let bias = s10:0x7f;
  let bexp = (a.bexp as s10) + (b.bexp as s10) - bias + (sfd_shift as s10);
  let bexp = s10:0 if a.bexp == u8:0 || b.bexp == u8:0 else bexp;

  // Note that we usually flush subnormals. Here, we preserve what we can for
  // compatability with reference implementations.
  // We only do this for the internal product - we otherwise don't handle
  // subnormal values (we flush them to 0).
  let is_subnormal = bexp <= s10:0;
  let result_exp = u9:0 if is_subnormal else bexp as u9;
  let sub_exp = std::abs(bexp) as u9;
  let result_sfd = sfd >> sub_exp if is_subnormal else sfd;

  // - Overflow infinites - saturate exp, clear sfd.
  let high_exp = std::mask_bits<u32:9>();
  let result_sfd = result_sfd if result_exp < high_exp else u48:0;
  let result_exp = result_exp as u9 if result_exp < high_exp else high_exp;

  // - Arg infinites. Any arg is infinite == result is infinite.
  let is_operand_inf = float32::is_inf(a) || float32::is_inf(b);
  let result_exp = high_exp if is_operand_inf else result_exp;
  let result_sfd = u48:0 if is_operand_inf else result_sfd as u48;

  // - NaNs. NaN trumps infinities, so we handle it last.
  //   inf * 0 = NaN, i.e.,
  let has_0_arg = a.bexp == u8:0 || b.bexp == u8:0;
  let has_nan_arg = float32::is_nan(a) || float32::is_nan(b);
  let has_inf_arg = float32::is_inf(a) || float32::is_inf(b);
  let is_result_nan = has_nan_arg || (has_0_arg && has_inf_arg);
  let result_exp = high_exp if is_result_nan else result_exp;
  let nan_sfd = u48:1 << (u48:48 - u48:1);
  let result_sfd = nan_sfd if is_result_nan else result_sfd;

  let result_sign = a.sign != b.sign;
  let result_sign = u1:0 if is_result_nan else result_sign;

  Product { sign: result_sign, bexp: result_exp, sfd: result_sfd }
}

// The main driver function.
// Many of the steps herein are fully described in the standalone adder or
// multiplier modules, but abridged comments are present here where useful.
fn fma_32(a: F32, b: F32, c: F32) -> F32 {
  let ab = mul_no_round(a, b);

  let greater_exp = ab.bexp if ab.bexp > c.bexp as u9 else c.bexp as u9;
  let greater_sign = ab.sign if ab.bexp > c.bexp as u9 else c.sign;

  // Make the implicit '1' explicit and flush subnormal "c" to 0 (already
  // done for ab inside mul_no_round()).
  let wide_c = c.sfd as uN[73] | uN[73]:0x800000;
  let wide_c = uN[73]:0 if c.bexp == u8:0 else wide_c;

  // Align AB and C so that the implicit '1' is in the MSB.
  let wide_ab = (ab.sfd as uN[73]) << 25;
  let wide_c = wide_c << 49;

  // Shift the operands into their correct positions.
  let rshift_ab = greater_exp - ab.bexp;
  let rshift_c = greater_exp - (c.bexp as u9);
  let shifted_ab = wide_ab >> rshift_ab;
  let shifted_c = wide_c >> rshift_c;

  // Calculate the sticky bits.
  // TODO(rspringer): 2021-04-05: This can be done more cleanly, but I'll get
  // to it when generalizing this code.
  let dropped_ab = wide_ab << ((u9:73 - rshift_ab) as uN[73]);
  let dropped_c = wide_c << ((u9:73 - rshift_c) as uN[73]);
  let dropped_c = wide_c if rshift_c >= u9:73 else dropped_c;
  let sticky_ab = (dropped_ab != uN[73]:0) as uN[73];
  let sticky_c = (dropped_c != uN[73]:0) as uN[73];

  // Add the sticky bit and extend the operands with the sign and carry bits.
  let shifted_ab = (shifted_ab | sticky_ab) as sN[75];
  let shifted_c = (shifted_c | sticky_c) as sN[75];

  // Set the operands' signs.
  let shifted_ab = -shifted_ab if ab.sign != greater_sign else shifted_ab;
  let shifted_c = -shifted_c if c.sign != greater_sign else shifted_c;

  // Addition!
  let sum_sfd = shifted_ab + shifted_c;
  let sfd_is_zero = sum_sfd == sN[75]:0;
  let result_sign = match (sfd_is_zero, sum_sfd < sN[75]:0) {
    (true, _) => u1:0,
    (false, true) => !greater_sign,
    _ => greater_sign,
  };

  // Chop off the sign bit (after applying it, if necessary).
  let abs_sfd = (-sum_sfd if sum_sfd < sN[75]:0 else sum_sfd) as uN[74];

  // Normalize.
  let carry_bit = abs_sfd[-1:];
  let carry_sfd = (abs_sfd >> uN[74]:1) as uN[73];
  let carry_sfd = carry_sfd | (abs_sfd[0:1] as uN[73]);

  // If high bits were cancelled, shift the result back into the MSB (ignoring
  // the zeroed carry bit, which is handled above).
  let leading_zeroes = clz(abs_sfd);
  let cancel = leading_zeroes > uN[74]:1;
  let cancel_sfd = (abs_sfd << (leading_zeroes - uN[74]:1)) as uN[73];
  let shifted_sfd = match(carry_bit, cancel) {
    (true, false) => carry_sfd,
    (false, true) => cancel_sfd,
    (false, false) => abs_sfd as uN[73],
    _ => fail!(uN[73]:0)
  };

  // Similar to the rounding in apfloat_add_2, except that the significand
  // starts at bit 49 (72-23) instead of bit 3.
  let normal_chunk = shifted_sfd[0:49];
  let half_way_chunk = shifted_sfd[48:50];
  let half_of_extra = u49:1 << 48;
  let do_round_up =
      u1:1 if (normal_chunk > half_of_extra) | (half_way_chunk == u2:0x3)
      else u1:0;
  let rounded_sfd =
      shifted_sfd as uN[74] + (uN[74]:1 << 49)
      if do_round_up else
        shifted_sfd as uN[74];

  let rounding_carry = rounded_sfd[-1:];
  let result_sfd = (rounded_sfd >> uN[74]:49) as u23;

  let bexp = greater_exp as s10 + rounding_carry as s10 +
      s10:1 - leading_zeroes as s10;
  let bexp = s10:0 if sfd_is_zero else bexp;
  let bexp = u9:0 if bexp < s10:0 else (bexp as u9);

  // Standard special case handling follows.

  // If the exponent underflowed, don't bother with denormals. Just flush to 0.
  let result_sfd = u23:0 if bexp == u9:0 else result_sfd;

  // Handle exponent overflow infinities.
  let saturated_exp = std::mask_bits<u32:8>() as u9;
  let max_exp = std::mask_bits<u32:8>();
  let result_sfd = result_sfd if bexp < saturated_exp else u23:0;
  let result_exp = bexp as u8 if bexp < saturated_exp else max_exp;

  // Handle arg infinities.
  let is_operand_inf = is_inf(ab) | float32::is_inf(c);
  let result_exp = max_exp if is_operand_inf else result_exp;
  let result_sfd = u23:0 if is_operand_inf else result_sfd;
  // Result infinity is negative iff all infinite operands are neg.
  let has_pos_inf = (is_inf(ab) & (ab.sign == u1:0)) |
                    (float32::is_inf(c) & (c.sign == u1:0));
  let result_sign = !has_pos_inf if is_operand_inf else result_sign;

  // Handle NaN; NaN trumps infinities, so we handle it last.
  // -inf + inf = NaN, i.e., if we have both positive and negative inf.
  let has_neg_inf =
      (is_inf(ab) & (ab.sign == u1:1)) |
      (float32::is_inf(c) & (c.sign == u1:1));
  let is_result_nan = is_nan(ab) |
      float32::is_nan(c) | (has_pos_inf & has_neg_inf);
  let result_exp = max_exp if is_result_nan else result_exp;
  let result_sfd = u23:0x80000 if is_result_nan else result_sfd;
  let result_sign = u1:0 if is_result_nan else result_sign;
  let is_result_inf = has_pos_inf | has_neg_inf;

  F32 { sign: result_sign, bexp: result_exp as u8, sfd: result_sfd as u23 }
}

#![test]
fn smoke() {
    let zero = F32 { sign: u1:0, bexp: u8:0, sfd: u23: 0 };
    let one_point_one = F32 { sign: u1:0, bexp: u8:127, sfd: u23: 0xccccd };
    let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, sfd: u23: 0x58cccd };
    let a = twenty_seven_point_one;
    let b = one_point_one;
    let c = zero;
    let expected = F32 { sign: u1:0, bexp: u8:0x83, sfd: u23:0x6e7ae2 };
    let actual = fma_32(a, b, c);
    assert_eq(expected, actual)
}

#![test]
fn one_x_one_plus_one() {
  let zero = F32 { sign: u1:0, bexp: u8:0, sfd: u23: 0 };
  let one_point_zero = F32 { sign: u1:0, bexp: u8:127, sfd: u23: 0 };
  let a = one_point_zero;
  let b = one_point_zero;
  let c = one_point_zero;
  let expected = F32 { sign: u1:0, bexp: u8:128, sfd: u23:0 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_zero() {
  let zero = F32 { sign: u1:0, bexp: u8:0, sfd: u23: 0 };
  let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, sfd: u23: 0x58cccd };
  let a = twenty_seven_point_one;
  let b = twenty_seven_point_one;
  let c = zero;
  let expected = F32 { sign: u1:0, bexp: u8:0x88, sfd: u23:0x379a3e };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one() {
  let zero = F32 { sign: u1:0, bexp: u8:0, sfd: u23: 0 };
  let one_point_zero = F32 { sign: u1:0, bexp: u8:127, sfd: u23: 0 };
  let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, sfd: u23: 0x58cccd };
  let a = twenty_seven_point_one;
  let b = twenty_seven_point_one;
  let c = one_point_zero;
  let expected = F32 { sign: u1:0, bexp: u8:0x88, sfd: u23:0x37da3e };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one_point_one() {
  let one_point_one = F32 { sign: u1:0, bexp: u8:127, sfd: u23: 0xccccd };
  let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, sfd: u23: 0x58cccd };
  let a = twenty_seven_point_one;
  let b = twenty_seven_point_one;
  let c = one_point_one;
  let expected = F32 { sign: u1:0, bexp: u8:0x88, sfd: u23:0x37e0a4 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_a() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x50, sfd: u23:0x1a8ddc };
  let b = F32 { sign: u1:0x1, bexp: u8:0xcb, sfd: u23:0xee7ac };
  let c = F32 { sign: u1:0x1, bexp: u8:0xb7, sfd: u23:0x609f18 };
  let expected = F32 { sign: u1:1, bexp: u8:0xb7, sfd: u23:0x609f18 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_b() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x23, sfd: u23:0x4d3a41 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x30, sfd: u23:0x35a901 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x96, sfd: u23:0x627c62 };
  let expected = F32 { sign: u1:0, bexp: u8:0x96, sfd: u23:0x627c62 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_c() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x71, sfd: u23:0x2f0932 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xe5, sfd: u23:0x416b76 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xcb, sfd: u23:0x5fd32a };
  let expected = F32 { sign: u1:1, bexp: u8:0xd8, sfd: u23:0x4386a };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_d() {
  let a = F32 { sign: u1:0x0, bexp: u8:0xac, sfd: u23:0x1d0d22 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xdb, sfd: u23:0x2fe688 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xa9, sfd: u23:0x2be1d2 };
  let expected = F32 { sign: u1:0, bexp: u8:0xff, sfd: u23:0x0 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_e() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x7b, sfd: u23:0x25e79f };
  let b = F32 { sign: u1:0x1, bexp: u8:0xff, sfd: u23:0x207370 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x39, sfd: u23:0x6bb348 };
  let expected = F32 { sign: u1:0, bexp: u8:0xff, sfd: u23:0x80000 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_f() {
  let a = F32 { sign: u1:0x1, bexp: u8:0xe0, sfd: u23:0x3cdaa8 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x96, sfd: u23:0x52549c };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1c, sfd: u23:0x21e0fd };
  let expected = F32 { sign: u1:0, bexp: u8:0xf8, sfd: u23:0x1b29c9 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_g() {
  let a = F32 { sign: u1:0x1, bexp: u8:0xc4, sfd: u23:0x73b59a };
  let b = F32 { sign: u1:0x0, bexp: u8:0xa6, sfd: u23:0x1631c0 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x29, sfd: u23:0x5b3d33 };
  let expected = F32 { sign: u1:1, bexp: u8:0xec, sfd: u23:0xefbc5 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_h() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x9b, sfd: u23:0x3f50d4 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x7b, sfd: u23:0x4beeb5 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x37, sfd: u23:0x6ad17c };
  let expected = F32 { sign: u1:1, bexp: u8:0x98, sfd: u23:0x18677d };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_i() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x66, sfd: u23:0x36e592 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xc8, sfd: u23:0x2b5bf1 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x52, sfd: u23:0x12900b };
  let expected = F32 { sign: u1:0, bexp: u8:0xaf, sfd: u23:0x74da11 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_j() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x88, sfd: u23:0x0f0e03 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xb9, sfd: u23:0x36006d };
  let c = F32 { sign: u1:0x1, bexp: u8:0xaa, sfd: u23:0x358b6b };
  let expected = F32 { sign: u1:0, bexp: u8:0xc2, sfd: u23:0x4b6865 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_k() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x29, sfd: u23:0x2fd76d };
  let b = F32 { sign: u1:0x1, bexp: u8:0xce, sfd: u23:0x63eded };
  let c = F32 { sign: u1:0x0, bexp: u8:0xfd, sfd: u23:0x21adee };
  let expected = F32 { sign: u1:0, bexp: u8:0xfd, sfd: u23:0x21adee };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_l() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x6a, sfd: u23:0x09c1b9 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x7c, sfd: u23:0x666a52 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x80, sfd: u23:0x626bcf };
  let expected = F32 { sign: u1:1, bexp: u8:0x80, sfd: u23:0x626bcf };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_m() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x70, sfd: u23:0x41e2db };
  let b = F32 { sign: u1:0x1, bexp: u8:0xd1, sfd: u23:0x013c17 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xb9, sfd: u23:0x30313f };
  let expected = F32 { sign: u1:0, bexp: u8:0xc2, sfd: u23:0x4419bf };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_n() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x33, sfd: u23:0x537374 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x40, sfd: u23:0x78fa62 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x09, sfd: u23:0x7cfb29 };
  let expected = F32 { sign: u1:1, bexp: u8:0x09, sfd: u23:0x7cfb36 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_o() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x94, sfd: u23:0x1aeb90 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x88, sfd: u23:0x1ab376 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x9d, sfd: u23:0x15dd1e };
  let expected = F32 { sign: u1:1, bexp: u8:0x9e, sfd: u23:0x288cde };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_p() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x88, sfd: u23:0x1ebb00 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xf6, sfd: u23:0x0950b6 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xfd, sfd: u23:0x6c314b };
  let expected = F32 { sign: u1:1, bexp: u8:0xfe, sfd: u23:0x5e77d4 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_q() {
  let a = F32 { sign: u1:0x0, bexp: u8:0xda, sfd: u23:0x5b328f };
  let b = F32 { sign: u1:0x1, bexp: u8:0x74, sfd: u23:0x157da3 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1b, sfd: u23:0x6a3f25 };
  let expected = F32 { sign: u1:1, bexp: u8:0xd0, sfd: u23:0x000000 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_r() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x34, sfd: u23:0x4da000 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xf4, sfd: u23:0x4bc400 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x33, sfd: u23:0x54476d };
  let expected = F32 { sign: u1:1, bexp: u8:0xaa, sfd: u23:0x23ab4f };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_s() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x27, sfd: u23:0x732d83 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xbb, sfd: u23:0x4b2dcd };
  let c = F32 { sign: u1:0x0, bexp: u8:0x3a, sfd: u23:0x65e4bd };
  let expected = F32 { sign: u1:0, bexp: u8:0x64, sfd: u23:0x410099 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_t() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x17, sfd: u23:0x070770 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x86, sfd: u23:0x623b39 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1e, sfd: u23:0x6ea761 };
  let expected = F32 { sign: u1:1, bexp: u8:0x0c, sfd: u23:0x693bc0 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_u() {
  let a = F32 { sign: u1:0x0, bexp: u8:0xb1, sfd: u23:0x0c8800 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xc6, sfd: u23:0x2b3800 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x22, sfd: u23:0x00c677 };
  let expected = F32 { sign: u1:1, bexp: u8:0xf8, sfd: u23:0x3bfb2b };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_v() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x90, sfd: u23:0x04a800 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x1f, sfd: u23:0x099cb0 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x28, sfd: u23:0x4d6497 };
  let expected = F32 { sign: u1:1, bexp: u8:0x30, sfd: u23:0x0dd0cf };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_w() {
  let a = F32 { sign: u1:0x0, bexp: u8:0x90, sfd: u23:0x0fdde1 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xa8, sfd: u23:0x663085 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x9b, sfd: u23:0x450d69 };
  let expected = F32 { sign: u1:0, bexp: u8:0xba, sfd: u23:0x015c9c };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_x() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x4c, sfd: u23:0x5ca821 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x87, sfd: u23:0x14808c };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1c, sfd: u23:0x585ccf };
  let expected = F32 { sign: u1:1, bexp: u8:0x55, sfd: u23:0x000000 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_y() {
  let a = F32 { sign: u1:0x0, bexp: u8:0xc5, sfd: u23:0x3a123b };
  let b = F32 { sign: u1:0x1, bexp: u8:0x3b, sfd: u23:0x7ee4d9 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x4f, sfd: u23:0x1d4ddc };
  let expected = F32 { sign: u1:1, bexp: u8:0x82, sfd: u23:0x39446d };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_z() {
  let a = F32 { sign: u1:0x1, bexp: u8:0xd4, sfd: u23:0x1b858b };
  let b = F32 { sign: u1:0x1, bexp: u8:0x9e, sfd: u23:0x59fa23 };
  let c = F32 { sign: u1:0x1, bexp: u8:0xb5, sfd: u23:0x3520e4 };
  let expected = F32 { sign: u1:0x0, bexp: u8:0xf4, sfd: u23:0x046c29 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}

#![test]
fn fail_case_aa() {
  let a = F32 { sign: u1:0x1, bexp: u8:0x9b, sfd: u23:0x3ac78d };
  let b = F32 { sign: u1:0x0, bexp: u8:0x3b, sfd: u23:0x542cbb };
  let c = F32 { sign: u1:0x1, bexp: u8:0x09, sfd: u23:0x0c609e };
  let expected = F32 { sign: u1:0x1, bexp: u8:0x58, sfd: u23:0x1acde3 };
  let actual = fma_32(a, b, c);
  assert_eq(expected, actual)
}
