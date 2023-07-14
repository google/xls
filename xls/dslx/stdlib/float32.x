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
import std

// TODO(rspringer): Make u32:8 and u32:23 symbolic constants. Currently, such
// constants don't propagate correctly and fail to resolve when in parametric
// specifications.
pub type F32 = apfloat::APFloat<8, 23>;
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
  assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:0, bexp: u8:0x0, fraction: u23:0x0 };
  let actual = normalize(u1:0, u8:0x1, u24:0x0);
  assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:0, bexp: u8:0x0, fraction: u23:0x0 };
  let actual = normalize(u1:0, u8:0xfe, u24:0x0);
  assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:1, bexp: u8:77, fraction: u23:0x0 };
  let actual = normalize(u1:1, u8:100, u24:1);
  assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:1, bexp: u8:2, fraction: u23:0b000_1111_0000_0101_0000_0000 };
  let actual = normalize(
      u1:1, u8:10, u24:0b0000_0000_1000_1111_0000_0101);
  assert_eq(expected, actual);

  let expected = F32 {
      sign: u1:1, bexp: u8:10, fraction: u23:0b000_0000_1000_1111_0000_0101};
  let actual = normalize(
      u1:1, u8:10, u24:0b1000_0000_1000_1111_0000_0101);
  assert_eq(expected, actual);

  // Denormals should be flushed to zero.
  let expected = zero(u1:1);
  let actual = normalize(
      u1:1, u8:5, u24:0b0000_0000_1000_1111_0000_0101);
  assert_eq(expected, actual);

  let expected = zero(u1:0);
  let actual = normalize(
      u1:0, u8:2, u24:0b0010_0000_1000_1111_0000_0101);
  assert_eq(expected, actual);
  ()
}

#[test]
fn tag_test() {
  assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, fraction: u23:0 }), FloatTag::ZERO);
  assert_eq(tag(F32 { sign: u1:1, bexp: u8:0, fraction: u23:0 }), FloatTag::ZERO);
  assert_eq(tag(zero(u1:0)), FloatTag::ZERO);
  assert_eq(tag(zero(u1:1)), FloatTag::ZERO);

  assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, fraction: u23:1 }), FloatTag::SUBNORMAL);
  assert_eq(tag(F32 { sign: u1:0, bexp: u8:0, fraction: u23:0x7f_ffff }), FloatTag::SUBNORMAL);

  assert_eq(tag(F32 { sign: u1:0, bexp: u8:12, fraction: u23:0 }), FloatTag::NORMAL);
  assert_eq(tag(F32 { sign: u1:1, bexp: u8:254, fraction: u23:0x7f_ffff }), FloatTag::NORMAL);
  assert_eq(tag(F32 { sign: u1:1, bexp: u8:1, fraction: u23:1 }), FloatTag::NORMAL);

  assert_eq(tag(F32 { sign: u1:0, bexp: u8:255, fraction: u23:0 }), FloatTag::INFINITY);
  assert_eq(tag(F32 { sign: u1:1, bexp: u8:255, fraction: u23:0 }), FloatTag::INFINITY);
  assert_eq(tag(inf(u1:0)), FloatTag::INFINITY);
  assert_eq(tag(inf(u1:1)), FloatTag::INFINITY);

  assert_eq(tag(F32 { sign: u1:0, bexp: u8:255, fraction: u23:1 }), FloatTag::NAN);
  assert_eq(tag(F32 { sign: u1:1, bexp: u8:255, fraction: u23:0x7f_ffff }), FloatTag::NAN);
  assert_eq(tag(qnan()), FloatTag::NAN);
  ()
}

pub fn fixed_fraction(input_float: F32) -> u23 {
  let input_fraction_magnitude: u25 = u2:0b01 ++ input_float.fraction;
  let unbiased_input_float_exponent: s8 = unbiased_exponent(input_float);

  let input_fixed_magnitude: u25 = match unbiased_input_float_exponent as s8 > s8:0 {
    true => {
      let fraction_left_shift = unbiased_input_float_exponent as u3;
      input_fraction_magnitude << (fraction_left_shift as u25)
    },
    _ => {
      let fraction_right_shift = (-unbiased_input_float_exponent) as u5;
      input_fraction_magnitude >> (fraction_right_shift as u25)
    }
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
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:0, fraction: u23:0 };
  let actual = from_int32(s32:0);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:127, fraction: u23:0 };
  let actual = from_int32(s32:1);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:1, bexp: u8:127, fraction: u23:0 };
  let actual = from_int32(s32:-1);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 };
  let actual = from_int32(s32:2);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:1, bexp: u8:128, fraction: u23:0 };
  let actual = from_int32(s32:-2);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:156, fraction: u23:0x7fffff };
  let actual = from_int32(s32:1073741760);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:156, fraction: u23:0x3fffff };
  let actual = from_int32(s32:805306304);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:157, fraction: u23:0x7fffff };
  let actual = from_int32(s32:2147483583);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:0, bexp: u8:158, fraction: u23:0x0 };
  let actual = from_int32(s32:2147483647);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:1, bexp: u8:158, fraction: u23:0x0 };
  let actual = from_int32(s32:-2147483647);
  assert_eq(expected, actual);

  let expected = F32 { sign: u1:1, bexp: u8:158, fraction: u23:0x0 };
  let actual = from_int32(s32:-2147483648);
  assert_eq(expected, actual);
  ()
}

pub fn add(x: F32, y: F32) -> F32 {
  apfloat::add(x, y)
}

pub fn sub(x: F32, y: F32) -> F32 {
  apfloat::sub(x, y)
}

pub fn mul(x: F32, y: F32) -> F32 {
  apfloat::mul(x, y)
}

pub fn fma(a: F32, b: F32, c: F32) -> F32 {
  apfloat::fma(a, b, c)
}


// Floating point fast (approximate) inverse square root. This should be able to
// compute 1.0 / sqrt(x) using fewer hardware resources than using a sqrt and
// division module, although this hasn't been benchmarked yet. Latency is
// expected to be lower as well. The tradeoff is that this offers slighlty less
// precision (error is < 0.2% in worst case). The accuracy-resources tradeoff
// can be adjusted by changing the number of Newton's method iterations
// (default 1).
//
// Note:
//  - Input denormals are treated as/flushed to 0. (denormals-are-zero / DAZ).
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
//  - We emit a single, canonical representation for NaN (qnan) but accept
//    all NaN respresentations as input.
//
// Reference: https://en.wikipedia.org/wiki/Fast_inverse_square_root

// Computes an approximation of 1.0 / sqrt(x). NUM_REFINEMENTS can be increased
// to tradeoff more hardware resources for more accuracy.
pub fn fast_rsqrt_config_refinements<NUM_REFINEMENTS: u32 = {u32:1}>(x: F32) -> F32 {
  const zero_point_five = F32 {sign: u1:0,
                               bexp: u8:0x7e,
                               fraction:  u23:0};
  const one_point_five  = F32 {sign: u1:0,
                               bexp: u8:0x7f,
                               fraction:  u1:1 ++ u22:0};
  const magic_number = u32:0x5f3759df;

  // Flush subnormal input.
  let x = subnormals_to_zero(x);

  let approx = unflatten(
                  magic_number - (flatten(x) >> u32:1));
  let half_x = mul(x, zero_point_five);

  // Refine solution w/ Newton's method.
  let result = for (idx, approx): (u32, F32) in range (u32:0, NUM_REFINEMENTS) {
    let prod = mul(half_x, approx);
    let prod = mul(prod, approx);
    let nprod = F32{sign: !prod.sign, bexp: prod.bexp, fraction: prod.fraction};
    let diff = add(one_point_five, nprod);
    mul(approx, diff)
  } (approx);

  // I don't *think* it is possible to underflow / have a subnormal result
  // here. In order to have a subnormal result, x would have to be so large
  // that it overflows to infinity (handled below).

  // Special cases.
  // 1/sqrt(inf) -> 0, 1/sqrt(-inf) -> NaN (handled below along
  // with other negative numbers).
  let result = if is_inf(x) { zero(x.sign) } else { result };
  // 1/sqrt(x < 0) -> NaN
  let result = if x.sign == u1:1 { qnan() } else { result };
  // 1/sqrt(NaN) -> NaN.
  let result = if is_nan(x) { x } else { result };
  // 1/sqrt(0) -> inf, 1/sqrt(-0) -> -inf
  let result = if is_zero_or_subnormal(x) { inf(x.sign) }
               else { result };
  result
}

pub fn fast_rsqrt(x: F32) -> F32 {
  fast_rsqrt_config_refinements<u32:1>(x)
}

#[test]
fn fast_sqrt_test() {
  // Test Special cases.
  assert_eq(fast_rsqrt(zero(u1:0)),
    inf(u1:0));
  assert_eq(fast_rsqrt(zero(u1:1)),
    inf(u1:1));
  assert_eq(fast_rsqrt(inf(u1:0)),
    zero(u1:0));
  assert_eq(fast_rsqrt(inf(u1:1)),
    qnan());
  assert_eq(fast_rsqrt(qnan()),
    qnan());
  assert_eq(fast_rsqrt(one(u1:1)),
    qnan());
  let pos_denormal = F32{sign: u1:0, bexp: u8:0, fraction: u23:99};
  assert_eq(fast_rsqrt(pos_denormal),
    inf(u1:0));
  let neg_denormal = F32{sign: u1:1, bexp: u8:0, fraction: u23:99};
  assert_eq(fast_rsqrt(neg_denormal),
    inf(u1:1));
  ()
}


// ldexp (load exponent) computes fraction * 2^exp.
// Note:
//  - Input denormals are treated as/flushed to 0.
//      (denormals-are-zero / DAZ).  Similarly,
//      denormal results are flushed to 0.
//  - No exception flags are raised/reported.
//  - We emit a single, canonical representation for
//      NaN (qnan) but accept all NaN respresentations
//      as input

// Returns fraction * 2^exp
pub fn ldexp(fraction: F32, exp: s32) -> F32 {
  // TODO(jbaileyhandle):  Remove after testing.

  const max_exponent = bias(s8:0) as s33;
  const min_exponent = s33:1 - (bias(s8:0) as s33);

  // Flush subnormal input.
  let fraction = subnormals_to_zero(fraction);

  // Increase the exponent of fraction by 'exp'.
  // If this was not a DAZ module, we'd have to deal
  // with denormal 'fraction' here.
  let exp = signex(exp, s33:0)
              + signex(unbiased_exponent(fraction), s33:0);
  let result = F32 {sign: fraction.sign,
                    bexp: bias(exp as s8),
                    fraction: fraction.fraction };

  // Handle overflow.
  let result = if exp > max_exponent { inf(fraction.sign) }
               else { result };

  // Hanlde underflow, taking into account the case that underflow
  // rounds back up to a normal number.
  // If this was not a DAZ module, we'd have to deal
  // with denormal 'result' here.
  let underflow_result =
    if exp == (min_exponent - s33:1) && fraction.fraction == std::mask_bits<u32:23>() {
      F32{sign: fraction.sign, bexp: u8:1, fraction: u23:0}
    } else {
      zero(fraction.sign)
    };
  let result = if exp < min_exponent { underflow_result }
               else { result };
  // Flush subnormal output.
  let result = subnormals_to_zero(result);

  // Handle special cases.
  let result = if is_zero_or_subnormal(fraction) || is_inf(fraction) { fraction }
               else { result };
  let result = if is_nan(fraction) { qnan() }
               else { result };
  result
}

#[test]
fn ldexp_test() {
  // Test Special cases.
  assert_eq(ldexp(zero(u1:0), s32:1),
    zero(u1:0));
  assert_eq(ldexp(zero(u1:1), s32:1),
    zero(u1:1));
  assert_eq(ldexp(inf(u1:0), s32:-1),
    inf(u1:0));
  assert_eq(ldexp(inf(u1:1), s32:-1),
    inf(u1:1));
  assert_eq(ldexp(qnan(), s32:1),
    qnan());

  // Subnormal input.
  let pos_denormal = F32{sign: u1:0, bexp: u8:0, fraction: u23:99};
  assert_eq(ldexp(pos_denormal, s32:1),
    zero(u1:0));
  let neg_denormal = F32{sign: u1:1, bexp: u8:0, fraction: u23:99};
  assert_eq(ldexp(neg_denormal, s32:1),
    zero(u1:1));

  // Output subnormal, flush to zero.
  let almost_denormal = F32{sign: u1:0, bexp: u8:1, fraction: u23:99};
  assert_eq(ldexp(pos_denormal, s32:-1),
    zero(u1:0));

  // Subnormal result rounds up to normal number.
  let frac = F32{sign: u1:0, bexp: u8:10, fraction: u23:0x7fffff};
  let expected = F32{sign: u1:0, bexp: u8:1, fraction: u23:0};
  assert_eq(ldexp(frac, s32:-10), expected);
  let frac = F32{sign: u1:1, bexp: u8:10, fraction: u23:0x7fffff};
  let expected = F32{sign: u1:1, bexp: u8:1, fraction: u23:0};
  assert_eq(ldexp(frac, s32:-10), expected);

  // Large positive input exponents.
  let frac = F32{sign: u1:0, bexp: u8:128, fraction: u23:0x0};
  let expected = inf(u1:0);
  assert_eq(ldexp(frac, s32:0x7FFFFFFF - s32:1), expected);
  let frac = F32{sign: u1:0, bexp: u8:128, fraction: u23:0x0};
  let expected = inf(u1:0);
  assert_eq(ldexp(frac, s32:0x7FFFFFFF), expected);
  let frac = F32{sign: u1:1, bexp: u8:128, fraction: u23:0x0};
  let expected = inf(u1:1);
  assert_eq(ldexp(frac, s32:0x7FFFFFFF - s32:1), expected);
  let frac = F32{sign: u1:1, bexp: u8:128, fraction: u23:0x0};
  let expected = inf(u1:1);
  assert_eq(ldexp(frac, s32:0x7FFFFFFF), expected);

  // Large negative input exponents.
  let frac = F32{sign: u1:0, bexp: u8:126, fraction: u23:0x0};
  let expected = zero(u1:0);
  assert_eq(ldexp(frac, s32:0x80000000 + s32:0x1), expected);
  let frac = F32{sign: u1:0, bexp: u8:126, fraction: u23:0x0};
  let expected = zero(u1:0);
  assert_eq(ldexp(frac, s32:0x80000000), expected);
  let frac = F32{sign: u1:1, bexp: u8:126, fraction: u23:0x0};
  let expected = zero(u1:1);
  assert_eq(ldexp(frac, s32:0x80000000 + s32:0x1), expected);
  let frac = F32{sign: u1:1, bexp: u8:126, fraction: u23:0x0};
  let expected = zero(u1:1);
  assert_eq(ldexp(frac, s32:0x80000000), expected);

  // Other large exponents from reported bug #462.
  let frac = unflatten(u32:0xd3fefd2b);
  let expected = inf(u1:1);
  assert_eq(ldexp(frac, s32:0x7ffffffd), expected);
  let frac = unflatten(u32:0x36eba93e);
  let expected = zero(u1:0);
  assert_eq(ldexp(frac, s32:0x80000010), expected);
  let frac = unflatten(u32:0x8a87c096);
  let expected = zero(u1:1);
  assert_eq(ldexp(frac, s32:0x80000013), expected);
  let frac = unflatten(u32:0x71694e37);
  let expected = inf(u1:0);
  assert_eq(ldexp(frac, s32:0x7fffffbe), expected);

  ()
}

