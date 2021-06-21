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
pub fn unbiased_exponent(f: F32) -> u9 {
  apfloat::unbiased_exponent<u32:8, u32:23>(f)
}
pub fn bias(unbiased_exponent_in: u9) -> u8 {
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

pub fn normalize(sign:u1, exp: u8, sfd_with_hidden: u24) -> F32 {
  apfloat::normalize<u32:8, u32:23>(sign, exp, sfd_with_hidden)
}

pub const F32_ONE_FLAT = u32:0x3f800000;

pub fn tag(f: F32) -> FloatTag {
  apfloat::tag<u32:8, u32:23>(f)
}

#![test]
fn normalize_test() {
  let expected = F32 {
      sign: u1:0, bexp: u8:0x12, sfd: u23:0x7e_dcba };
  let actual = normalize(u1:0, u8:0x12, u24:0xfe_dcba);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:0, bexp: u8:0x0, sfd: u23:0x0 };
  let actual = normalize(u1:0, u8:0x1, u24:0x0);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:0, bexp: u8:0x0, sfd: u23:0x0 };
  let actual = normalize(u1:0, u8:0xfe, u24:0x0);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:1, bexp: u8:77, sfd: u23:0x0 };
  let actual = normalize(u1:1, u8:100, u24:1);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:1, bexp: u8:2, sfd: u23:0b000_1111_0000_0101_0000_0000 };
  let actual = normalize(
      u1:1, u8:10, u24:0b0000_0000_1000_1111_0000_0101);
  let _ = assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:1, bexp: u8:10, sfd: u23:0b000_0000_1000_1111_0000_0101};
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

#![test]
fn tag_test() {
  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, sfd: u23:0 }), FloatTag::ZERO);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:0, sfd: u23:0 }), FloatTag::ZERO);
  let _ = assert_eq(tag(zero(u1:0)), FloatTag::ZERO);
  let _ = assert_eq(tag(zero(u1:1)), FloatTag::ZERO);

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, sfd: u23:1 }), FloatTag::SUBNORMAL);
  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, sfd: u23:0x7f_ffff }), FloatTag::SUBNORMAL);

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:12, sfd: u23:0 }), FloatTag::NORMAL);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:254, sfd: u23:0x7f_ffff }), FloatTag::NORMAL);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:1, sfd: u23:1 }), FloatTag::NORMAL);

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:255, sfd: u23:0 }), FloatTag::INFINITY);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:255, sfd: u23:0 }), FloatTag::INFINITY);
  let _ = assert_eq(tag(inf(u1:0)), FloatTag::INFINITY);
  let _ = assert_eq(tag(inf(u1:1)), FloatTag::INFINITY);

  let _ = assert_eq(tag(F32 { sign: u1:0, bexp: u8:255, sfd: u23:1 }), FloatTag::NAN);
  let _ = assert_eq(tag(F32 { sign: u1:1, bexp: u8:255, sfd: u23:0x7f_ffff }), FloatTag::NAN);
  let _ = assert_eq(tag(qnan()), FloatTag::NAN);
  ()
}

pub fn fixed_fraction(input_float: F32) -> u23 {
  let input_significand_magnitude: u25 = u2:0b01 ++ input_float.sfd;
  let unbiased_input_float_exponent: u9 = unbiased_exponent(input_float);

  let input_fixed_magnitude: u25 = match unbiased_input_float_exponent as s9 > s9:0 {
    true =>
      let significand_left_shift = unbiased_input_float_exponent as u3;
      input_significand_magnitude << (significand_left_shift as u25),
    _ =>
      let significand_right_shift = (-unbiased_input_float_exponent) as u5;
      input_significand_magnitude >> (significand_right_shift as u25)
  };

  let input_fraction_part_magnitude: u24 = input_fixed_magnitude as u23 as u24;
  let fixed_fraction: u24 = (u24:1<<u24:23) - input_fraction_part_magnitude
    if input_float.sign && input_fraction_part_magnitude != u24:0
    else input_fraction_part_magnitude
 ;
  fixed_fraction as u23
}

