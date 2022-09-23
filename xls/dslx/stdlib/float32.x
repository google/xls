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

// 32-bit floating point routines.
import apfloat

// TODO(rspringer): Make u32:8 and u32:23 symbolic constants. Currently, such
// constants don't propagate correctly and fail to resolve when in parametric
// specifications.
pub type F32 = apfloat::APFloat<u32:8, u32:23>;
pub type FloatTag = apfloat::APFloatTag;

pub type TaggedF32 = (FloatTag, F32);

pub fn qnan() -> F32 { apfloat::qnan<u32:8, u32:23>() }
pub fn zero(sign: u1) -> F32 { apfloat::zero<u32:8, u32:23>(sign) }
pub fn one(sign: u1) -> F32 { apfloat::one<u32:8, u32:23>(sign) }
pub fn inf(sign: u1) -> F32 { apfloat::inf<u32:8, u32:23>(sign) }


pub fn unbiased_exponent(f: F32) -> s8 {
  apfloat::unbiased_exponent<u32:8, u32:23>(f)
}
pub fn bias(unbiased_exponent_in: s8) -> u8 {
  apfloat::bias<u32:8, u32:23>(unbiased_exponent_in)
}
pub fn flatten(f: F32) -> u32 { apfloat::flatten<u32:8, u32:23>(f) }
pub fn unflatten(f: u32) -> F32 { apfloat::unflatten<u32:8, u32:23>(f) }
pub fn cast_from_fixed<NUM_SRC_BITS:u32>(s: sN[NUM_SRC_BITS]) -> F32 {
  apfloat::cast_from_fixed<u32:8, u32:23>(s)
}
pub fn cast_to_fixed<NUM_DST_BITS:u32>(to_cast: F32) -> sN[NUM_DST_BITS] {
  apfloat::cast_to_fixed<NUM_DST_BITS, u32:8, u32:23>(to_cast)
}
pub fn subnormals_to_zero(f: F32) -> F32 {
  apfloat::subnormals_to_zero<u32:8, u32:23>(f)
}

pub fn is_inf(f: F32) -> u1 { apfloat::is_inf<u32:8, u32:23>(f) }
pub fn is_nan(f: F32) -> u1 { apfloat::is_nan<u32:8, u32:23>(f) }
pub fn is_zero_or_subnormal(f: F32) -> u1 {
  apfloat::is_zero_or_subnormal<u32:8, u32:23>(f)
}

pub fn eq_2(x: F32, y: F32) -> u1 {
  apfloat::eq_2<u32:8, u32:23>(x, y)
}

pub fn gt_2(x: F32, y: F32) -> u1 {
  apfloat::gt_2<u32:8, u32:23>(x, y)
}

pub fn gte_2(x: F32, y: F32) -> u1 {
  apfloat::gte_2<u32:8, u32:23>(x, y)
}

pub fn lt_2(x: F32, y: F32) -> u1 {
  apfloat::lt_2<u32:8, u32:23>(x, y)
}

pub fn lte_2(x: F32, y: F32) -> u1 {
  apfloat::lte_2<u32:8, u32:23>(x, y)
}

pub fn normalize(sign:u1, exp: u8, fraction_with_hidden: u24) -> F32 {
  apfloat::normalize<u32:8, u32:23>(sign, exp, fraction_with_hidden)
}

pub fn to_int<RESULT_SZ: u32>(x: F32) -> sN[RESULT_SZ] {
  apfloat::to_int<u32:8, u32:23, RESULT_SZ>(x)
}

// Just a convenience for the most common case.
pub fn to_int32(x: F32) -> s32 {
  apfloat::to_int<u32:8, u32:23, u32:32>(x)
}

pub const F32_ONE_FLAT = u32:0x3f800000;

pub fn tag(f: F32) -> FloatTag {
  apfloat::tag<u32:8, u32:23>(f)
}

#[test]
fn normalize_test() {
  let expected = F32 {
      sign: u1:0, bexp: u8:0x12, fraction: u23:0x7e_dcba };
  let actual = normalize(u1:0, u8:0x12, u24:0xfe_dcba);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:0, bexp: u8:0x0, fraction: u23:0x0 };
  let actual = normalize(u1:0, u8:0x1, u24:0x0);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:0, bexp: u8:0x0, fraction: u23:0x0 };
  let actual = normalize(u1:0, u8:0xfe, u24:0x0);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:1, bexp: u8:77, fraction: u23:0x0 };
  let actual = normalize(u1:1, u8:100, u24:1);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:1, bexp: u8:2, fraction: u23:0b000_1111_0000_0101_0000_0000 };
  let actual = normalize(
      u1:1, u8:10, u24:0b0000_0000_1000_1111_0000_0101);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:1, bexp: u8:10, fraction: u23:0b000_0000_1000_1111_0000_0101};
  let actual = normalize(
      u1:1, u8:10, u24:0b1000_0000_1000_1111_0000_0101);
  let _ = assert_eq(expected, actual);

  // Denormals should be flushed to zero.
  let expected = zero(u1:1);
  let actual = normalize(
      u1:1, u8:5, u24:0b0000_0000_1000_1111_0000_0101);
  let _ = assert_eq(expected, actual);

  let expected = zero(u1:0);
  let actual = normalize(
      u1:0, u8:2, u24:0b0010_0000_1000_1111_0000_0101);
  let _ = assert_eq(expected, actual);
  ()
}

#[test]
fn tag_test() {
  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, fraction: u23:0 }), FloatTag::ZERO);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:0, fraction: u23:0 }), FloatTag::ZERO);
  let _ = assert_eq(tag(zero(u1:0)), FloatTag::ZERO);
  let _ = assert_eq(tag(zero(u1:1)), FloatTag::ZERO);

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, fraction: u23:1 }), FloatTag::SUBNORMAL);
  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, fraction: u23:0x7f_ffff }), FloatTag::SUBNORMAL);

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:12, fraction: u23:0 }), FloatTag::NORMAL);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:254, fraction: u23:0x7f_ffff }), FloatTag::NORMAL);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:1, fraction: u23:1 }), FloatTag::NORMAL);

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:255, fraction: u23:0 }), FloatTag::INFINITY);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:255, fraction: u23:0 }), FloatTag::INFINITY);
  let _ = assert_eq(tag(inf(u1:0)), FloatTag::INFINITY);
  let _ = assert_eq(tag(inf(u1:1)), FloatTag::INFINITY);

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:255, fraction: u23:1 }), FloatTag::NAN);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:255, fraction: u23:0x7f_ffff }), FloatTag::NAN);
  let _ = assert_eq(tag(qnan()), FloatTag::NAN);
  ()
}

pub fn fixed_fraction(input_float: F32) -> u23 {
  let input_fraction_magnitude: u25 = u2:0b01 ++ input_float.fraction;
  let unbiased_input_float_exponent: s8 = unbiased_exponent(input_float);

  let input_fixed_magnitude: u25 = match unbiased_input_float_exponent as s8 > s8:0 {
    true =>
      let fraction_left_shift = unbiased_input_float_exponent as u3;
      input_fraction_magnitude << (fraction_left_shift as u25),
    _ =>
      let fraction_right_shift = (-unbiased_input_float_exponent) as u5;
      input_fraction_magnitude >> (fraction_right_shift as u25)
  };

  let input_fraction_part_magnitude: u24 = input_fixed_magnitude as u23 as u24;
  let fixed_fraction: u24 =
    if input_float.sign && input_fraction_part_magnitude != u24:0 {
      (u24:1<<u24:23) - input_fraction_part_magnitude
    } else {
      input_fraction_part_magnitude
    }
 ;
  fixed_fraction as u23
}

// Converts the given signed integer, into a floating-point number, rounding
// the resulting fraction with the usual guard, round, sticky bits according to
// the usual "round to nearest, half to even" rounding mode.
pub fn from_int32(x: s32) -> F32 {
  let sign = (x >> 31) as u1;
  let fraction = if sign { -x as u31 } else { x as u31 };
  let lz = clz(fraction) as u8;

  // Shift the fraction to be max width, normalize it, and drop the
  // leading/implicit one.
  let fraction = (fraction as u34 << (lz + u8:3)) as u33;

  let exp = u8:30 - lz;
  let bexp = exp + u8:127;

  // Rounding! Add the sticky bits back in to the shifted fraction.
  let sticky = (fraction & u33:0x7f != u33:0) as u31;
  let fraction = (fraction >> 7) as u26 | sticky as u26;

  let normal_chunk = fraction[0:3];
  let half_way_chunk = fraction[2:4];
  let do_round_up =
      if (normal_chunk > u3:0x4) | (half_way_chunk == u2:0x3) { u1:1 }
      else { u1:0 };
  let fraction = if do_round_up { fraction as u27 + u27:0x8 }
                 else { fraction as u27 };

  // See if rounding caused overflow, and if so, update the exponent.
  let overflow = fraction & (u27:1 << 26) != u27:0;
  let bexp = if overflow { bexp + u8:1 } else { bexp };
  let fraction = (fraction >> 3) as u23;

  let result = F32 { sign: sign, bexp: bexp, fraction: fraction };

  let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23:0 };
  let result =
      if bexp <= u8:126 && fraction == u23:0 { zero } else { result };

  // -INT_MAX is a special case; making it positive can drop its value.
  let is_neg_int_max = x == s32:-2147483648;
  let neg_int_max = F32 { sign: u1:1, bexp: u8:158, fraction: u23:0 };
  let result =
      if is_neg_int_max { neg_int_max } else { result };
  result
}

#[test]
fn from_int32_test() {
  let expected = F32 { sign: u1:0, bexp: u8:0, fraction: u23:0 };
  let actual = from_int32(s32:0);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:0, fraction: u23:0 };
  let actual = from_int32(s32:0);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:127, fraction: u23:0 };
  let actual = from_int32(s32:1);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:1, bexp: u8:127, fraction: u23:0 };
  let actual = from_int32(s32:-1);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 };
  let actual = from_int32(s32:2);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:1, bexp: u8:128, fraction: u23:0 };
  let actual = from_int32(s32:-2);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:156, fraction: u23:0x7fffff };
  let actual = from_int32(s32:1073741760);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:156, fraction: u23:0x3fffff };
  let actual = from_int32(s32:805306304);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:157, fraction: u23:0x7fffff };
  let actual = from_int32(s32:2147483583);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:158, fraction: u23:0x0 };
  let actual = from_int32(s32:2147483647);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:1, bexp: u8:158, fraction: u23:0x0 };
  let actual = from_int32(s32:-2147483647);
  let _ = assert_eq(expected, actual);

  let expected = F32 { sign: u1:1, bexp: u8:158, fraction: u23:0x0 };
  let actual = from_int32(s32:-2147483648);
  let _ = assert_eq(expected, actual);
  ()
}
