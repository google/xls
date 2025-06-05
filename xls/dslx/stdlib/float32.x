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
import apfloat;
import std;

pub const F32_EXP_SZ = u32:8;  // Exponent bits
pub const F32_FRACTION_SZ = u32:23;  // Fraction bits
pub const F32_TOTAL_SZ = u32:1 + F32_EXP_SZ + F32_FRACTION_SZ;

pub type F32 = apfloat::APFloat<F32_EXP_SZ, F32_FRACTION_SZ>;
pub type FloatTag = apfloat::APFloatTag;

pub type TaggedF32 = (FloatTag, F32);

pub fn qnan() -> F32 { apfloat::qnan<F32_EXP_SZ, F32_FRACTION_SZ>() }

pub fn is_nan(f: F32) -> bool { apfloat::is_nan<F32_EXP_SZ, F32_FRACTION_SZ>(f) }

pub fn inf(sign: u1) -> F32 { apfloat::inf<F32_EXP_SZ, F32_FRACTION_SZ>(sign) }

pub fn is_inf(f: F32) -> bool { apfloat::is_inf<F32_EXP_SZ, F32_FRACTION_SZ>(f) }

pub fn is_pos_inf(f: F32) -> bool { apfloat::is_pos_inf<F32_EXP_SZ, F32_FRACTION_SZ>(f) }

pub fn is_neg_inf(f: F32) -> bool { apfloat::is_neg_inf<F32_EXP_SZ, F32_FRACTION_SZ>(f) }

pub fn zero(sign: u1) -> F32 { apfloat::zero<F32_EXP_SZ, F32_FRACTION_SZ>(sign) }

pub fn one(sign: u1) -> F32 { apfloat::one<F32_EXP_SZ, F32_FRACTION_SZ>(sign) }

pub fn max_normal(sign: u1) -> F32 { apfloat::max_normal<F32_EXP_SZ, F32_FRACTION_SZ>(sign) }

pub fn negate(x: F32) -> F32 { apfloat::negate(x) }

pub fn max_normal_exp() -> sN[F32_EXP_SZ] { apfloat::max_normal_exp<F32_EXP_SZ>() }

pub fn min_normal_exp() -> sN[F32_EXP_SZ] { apfloat::min_normal_exp<F32_EXP_SZ>() }

pub fn unbiased_exponent(f: F32) -> sN[F32_EXP_SZ] {
    apfloat::unbiased_exponent<F32_EXP_SZ, F32_FRACTION_SZ>(f)
}

pub fn bias(unbiased_exponent_in: sN[F32_EXP_SZ]) -> uN[F32_EXP_SZ] {
    apfloat::bias(unbiased_exponent_in)
}

pub fn flatten(f: F32) -> uN[F32_TOTAL_SZ] { apfloat::flatten<F32_EXP_SZ, F32_FRACTION_SZ>(f) }

pub fn unflatten(f: uN[F32_TOTAL_SZ]) -> F32 { apfloat::unflatten<F32_EXP_SZ, F32_FRACTION_SZ>(f) }

pub fn ldexp(f: F32, e: s32) -> F32 { apfloat::ldexp(f, e) }

pub fn cast_from_fixed_using_rne<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> F32 {
    apfloat::cast_from_fixed_using_rne<F32_EXP_SZ, F32_FRACTION_SZ>(s)
}

pub fn cast_from_fixed_using_rz<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> F32 {
    apfloat::cast_from_fixed_using_rz<F32_EXP_SZ, F32_FRACTION_SZ>(s)
}

pub fn cast_to_fixed<NUM_DST_BITS: u32>(to_cast: F32) -> sN[NUM_DST_BITS] {
    apfloat::cast_to_fixed<NUM_DST_BITS, F32_EXP_SZ, F32_FRACTION_SZ>(to_cast)
}

pub fn subnormals_to_zero(f: F32) -> F32 {
    apfloat::subnormals_to_zero<F32_EXP_SZ, F32_FRACTION_SZ>(f)
}

pub fn is_zero_or_subnormal(f: F32) -> bool {
    apfloat::is_zero_or_subnormal<F32_EXP_SZ, F32_FRACTION_SZ>(f)
}

pub fn eq_2(x: F32, y: F32) -> bool { apfloat::eq_2<F32_EXP_SZ, F32_FRACTION_SZ>(x, y) }

pub fn gt_2(x: F32, y: F32) -> bool { apfloat::gt_2<F32_EXP_SZ, F32_FRACTION_SZ>(x, y) }

pub fn gte_2(x: F32, y: F32) -> bool { apfloat::gte_2<F32_EXP_SZ, F32_FRACTION_SZ>(x, y) }

pub fn lt_2(x: F32, y: F32) -> bool { apfloat::lt_2<F32_EXP_SZ, F32_FRACTION_SZ>(x, y) }

pub fn lte_2(x: F32, y: F32) -> bool { apfloat::lte_2<F32_EXP_SZ, F32_FRACTION_SZ>(x, y) }

pub fn normalize
    (sign: u1, exp: uN[F32_EXP_SZ], fraction_with_hidden: uN[u32:1 + F32_FRACTION_SZ]) -> F32 {
    apfloat::normalize<F32_EXP_SZ, F32_FRACTION_SZ>(sign, exp, fraction_with_hidden)
}

pub fn to_int<RESULT_SZ: u32>(x: F32) -> sN[RESULT_SZ] {
    apfloat::to_int<F32_EXP_SZ, F32_FRACTION_SZ, RESULT_SZ>(x)
}

// Just a convenience for the most common case.
pub fn to_int32(x: F32) -> s32 { apfloat::to_int<F32_EXP_SZ, F32_FRACTION_SZ, u32:32>(x) }

pub fn to_uint<RESULT_SZ: u32>(x: F32) -> uN[RESULT_SZ] { apfloat::to_uint<RESULT_SZ>(x) }

pub fn to_uint32(x: F32) -> u32 { apfloat::to_uint<u32:32>(x) }

pub const F32_ONE_FLAT = u32:0x3f800000;

pub fn tag(f: F32) -> FloatTag { apfloat::tag<F32_EXP_SZ, F32_FRACTION_SZ>(f) }

#[test]
fn normalize_test() {
    type ExpBits = uN[F32_EXP_SZ];
    type FractionBits = uN[F32_FRACTION_SZ];
    type WideFractionBits = uN[u32:1 + F32_FRACTION_SZ];

    let expected = F32 { sign: u1:0, bexp: ExpBits:0x12, fraction: FractionBits:0x7e_dcba };
    let actual = normalize(u1:0, ExpBits:0x12, WideFractionBits:0xfe_dcba);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:0x0, fraction: FractionBits:0x0 };
    let actual = normalize(u1:0, ExpBits:0x1, WideFractionBits:0x0);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:0x0, fraction: FractionBits:0x0 };
    let actual = normalize(u1:0, ExpBits:0xfe, WideFractionBits:0x0);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:1, bexp: ExpBits:77, fraction: FractionBits:0x0 };
    let actual = normalize(u1:1, ExpBits:100, WideFractionBits:1);
    assert_eq(expected, actual);

    let expected =
        F32 { sign: u1:1, bexp: ExpBits:2, fraction: FractionBits:0b000_1111_0000_0101_0000_0000 };
    let actual = normalize(u1:1, ExpBits:10, WideFractionBits:0b0000_0000_1000_1111_0000_0101);
    assert_eq(expected, actual);

    let expected =
        F32 { sign: u1:1, bexp: ExpBits:10, fraction: FractionBits:0b000_0000_1000_1111_0000_0101 };
    let actual = normalize(u1:1, ExpBits:10, WideFractionBits:0b1000_0000_1000_1111_0000_0101);
    assert_eq(expected, actual);

    // Denormals should be flushed to zero.
    let expected = zero(u1:1);
    let actual = normalize(u1:1, ExpBits:5, WideFractionBits:0b0000_0000_1000_1111_0000_0101);
    assert_eq(expected, actual);

    let expected = zero(u1:0);
    let actual = normalize(u1:0, ExpBits:2, WideFractionBits:0b0010_0000_1000_1111_0000_0101);
    assert_eq(expected, actual);
}

#[test]
fn tag_test() {
    type ExpBits = uN[F32_EXP_SZ];
    type FractionBits = uN[F32_FRACTION_SZ];

    assert_eq(tag(F32 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:0 }), FloatTag::ZERO);
    assert_eq(tag(F32 { sign: u1:1, bexp: ExpBits:0, fraction: FractionBits:0 }), FloatTag::ZERO);
    assert_eq(tag(zero(u1:0)), FloatTag::ZERO);
    assert_eq(tag(zero(u1:1)), FloatTag::ZERO);

    assert_eq(
        tag(F32 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:1 }), FloatTag::SUBNORMAL);
    assert_eq(
        tag(F32 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:0x7f_ffff }),
        FloatTag::SUBNORMAL);

    assert_eq(tag(F32 { sign: u1:0, bexp: ExpBits:12, fraction: FractionBits:0 }), FloatTag::NORMAL);
    assert_eq(
        tag(F32 { sign: u1:1, bexp: ExpBits:254, fraction: FractionBits:0x7f_ffff }),
        FloatTag::NORMAL);
    assert_eq(tag(F32 { sign: u1:1, bexp: ExpBits:1, fraction: FractionBits:1 }), FloatTag::NORMAL);

    assert_eq(
        tag(F32 { sign: u1:0, bexp: ExpBits:255, fraction: FractionBits:0 }), FloatTag::INFINITY);
    assert_eq(
        tag(F32 { sign: u1:1, bexp: ExpBits:255, fraction: FractionBits:0 }), FloatTag::INFINITY);
    assert_eq(tag(inf(u1:0)), FloatTag::INFINITY);
    assert_eq(tag(inf(u1:1)), FloatTag::INFINITY);

    assert_eq(tag(F32 { sign: u1:0, bexp: ExpBits:255, fraction: FractionBits:1 }), FloatTag::NAN);
    assert_eq(
        tag(F32 { sign: u1:1, bexp: ExpBits:255, fraction: FractionBits:0x7f_ffff }), FloatTag::NAN);
    assert_eq(tag(qnan()), FloatTag::NAN);
}

// TODO(hzeller): 2023-10-03 Use types derived from F32_{EXP,FRACTION}_SZ
// TODO(hzeller): 2023-10-03 Write a unit test.
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
        },
    };

    let input_fraction_part_magnitude: u24 = input_fixed_magnitude as u23 as u24;
    let fixed_fraction: u24 = if input_float.sign && input_fraction_part_magnitude != u24:0 {
        (u24:1 << F32_FRACTION_SZ) - input_fraction_part_magnitude
    } else {
        input_fraction_part_magnitude
    };
    fixed_fraction as u23
}

fn from_int32_internal(sign: u1, fraction: u32, lz: u8) -> F32 {
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
        if (normal_chunk > u3:0x4) | (half_way_chunk == u2:0x3) { u1:1 } else { u1:0 };
    let fraction = if do_round_up { fraction as u27 + u27:0x8 } else { fraction as u27 };

    // See if rounding caused overflow, and if so, update the exponent.
    let overflow = fraction & (u27:1 << 26) != u27:0;
    let bexp = if overflow { bexp + u8:1 } else { bexp };
    let fraction = (fraction >> 3) as u23;

    let result = F32 { sign, bexp, fraction };

    let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23:0 };
    let result = if bexp <= u8:126 && fraction == u23:0 { zero } else { result };
    result
}

// Converts the given signed integer, into a floating-point number, rounding
// the resulting fraction with the usual guard, round, sticky bits according to
// the usual "round to nearest, half to even" rounding mode.
pub fn from_int32(x: s32) -> F32 {
    // -INT_MAX is a special case; making it positive can drop its value.
    let is_neg_int_max = x == s32:-2147483648;
    let neg_int_max = F32 { sign: u1:1, bexp: u8:158, fraction: u23:0 };
    if is_neg_int_max {
        neg_int_max
    } else {
        let sign = (x >> 31) as u1;
        let fraction = if sign { -x as u31 } else { x as u31 };
        let lz = clz(fraction) as u8;
        from_int32_internal(sign, fraction as u32, lz)
    }
}

// Converts the given unsigned integer, into a floating-point number, rounding
// the resulting fraction with the usual guard, round, sticky bits according to
// the usual "round to nearest, half to even" rounding mode.
pub fn from_uint32(x: u32) -> F32 {
    let sign = u1:0;
    let fraction = x as u32;
    let lz = (clz(x) as u8) - u8:1;
    from_int32_internal(sign, fraction, lz)
}

#[test]
fn from_int32_test() {
    type ExpBits = uN[F32_EXP_SZ];
    type FractionBits = uN[F32_FRACTION_SZ];

    let expected = F32 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:0 };
    let actual = from_int32(s32:0);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:0 };
    let actual = from_int32(s32:0);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:127, fraction: FractionBits:0 };
    let actual = from_int32(s32:1);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:1, bexp: ExpBits:127, fraction: FractionBits:0 };
    let actual = from_int32(s32:-1);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:128, fraction: FractionBits:0 };
    let actual = from_int32(s32:2);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:1, bexp: ExpBits:128, fraction: FractionBits:0 };
    let actual = from_int32(s32:-2);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:156, fraction: FractionBits:0x7fffff };
    let actual = from_int32(s32:1073741760);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:156, fraction: FractionBits:0x3fffff };
    let actual = from_int32(s32:805306304);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:157, fraction: FractionBits:0x7fffff };
    let actual = from_int32(s32:2147483583);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:158, fraction: FractionBits:0x0 };
    let actual = from_int32(s32:2147483647);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:1, bexp: ExpBits:158, fraction: FractionBits:0x0 };
    let actual = from_int32(s32:-2147483647);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:1, bexp: ExpBits:158, fraction: FractionBits:0x0 };
    let actual = from_int32(s32:-2147483648);
    assert_eq(expected, actual);
}

#[test]
fn from_uint32_test() {
    type ExpBits = uN[F32_EXP_SZ];
    type FractionBits = uN[F32_FRACTION_SZ];

    let expected = F32 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:0 };
    let actual = from_uint32(u32:0);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:127, fraction: FractionBits:0 };
    let actual = from_uint32(u32:1);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:128, fraction: FractionBits:0 };
    let actual = from_uint32(u32:2);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:156, fraction: FractionBits:0x7fffff };
    let actual = from_uint32(u32:1073741760);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:156, fraction: FractionBits:0x3fffff };
    let actual = from_uint32(u32:805306304);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:157, fraction: FractionBits:0x7fffff };
    let actual = from_uint32(u32:2147483583);
    assert_eq(expected, actual);

    let expected = F32 { sign: u1:0, bexp: ExpBits:158, fraction: FractionBits:0x0 };
    let actual = from_uint32(u32:2147483647);
    assert_eq(expected, actual);
}

pub fn add(x: F32, y: F32) -> F32 { apfloat::add(x, y) }

pub fn sub(x: F32, y: F32) -> F32 { apfloat::sub(x, y) }

pub fn mul(x: F32, y: F32) -> F32 { apfloat::mul(x, y) }

pub fn fma(a: F32, b: F32, c: F32) -> F32 { apfloat::fma(a, b, c) }

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
    const ZERO_POINT_FIVE = F32 { sign: u1:0, bexp: u8:0x7e, fraction: u23:0 };
    const ONE_POINT_FIVE = F32 { sign: u1:0, bexp: u8:0x7f, fraction: u1:1 ++ u22:0 };
    const MAGIC_NUMBER = u32:0x5f3759df;

    // Flush subnormal input.
    let x = subnormals_to_zero(x);

    let approx = unflatten(MAGIC_NUMBER - (flatten(x) >> u32:1));
    let half_x = mul(x, ZERO_POINT_FIVE);

    // Refine solution w/ Newton's method.
    let result = for (idx, approx): (u32, F32) in u32:0..NUM_REFINEMENTS {
        let prod = mul(half_x, approx);
        let prod = mul(prod, approx);
        let nprod = F32 { sign: !prod.sign, bexp: prod.bexp, fraction: prod.fraction };
        let diff = add(ONE_POINT_FIVE, nprod);
        mul(approx, diff)
    }(approx);

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
    let result = if is_zero_or_subnormal(x) { inf(x.sign) } else { result };
    result
}

pub fn fast_rsqrt(x: F32) -> F32 { fast_rsqrt_config_refinements<u32:1>(x) }

#[test]
fn fast_sqrt_test() {
    // Test Special cases.
    assert_eq(fast_rsqrt(zero(u1:0)), inf(u1:0));
    assert_eq(fast_rsqrt(zero(u1:1)), inf(u1:1));
    assert_eq(fast_rsqrt(inf(u1:0)), zero(u1:0));
    assert_eq(fast_rsqrt(inf(u1:1)), qnan());
    assert_eq(fast_rsqrt(qnan()), qnan());
    assert_eq(fast_rsqrt(one(u1:1)), qnan());
    type ExpBits = uN[F32_EXP_SZ];
    type FractionBits = uN[F32_FRACTION_SZ];

    let pos_denormal = F32 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:99 };
    assert_eq(fast_rsqrt(pos_denormal), inf(u1:0));
    let neg_denormal = F32 { sign: u1:1, bexp: ExpBits:0, fraction: FractionBits:99 };
    assert_eq(fast_rsqrt(neg_denormal), inf(u1:1));
}

pub fn has_fractional_part(f: F32) -> bool { apfloat::has_fractional_part(f) }

pub fn has_negative_exponent(f: F32) -> bool { apfloat::has_negative_exponent(f) }

pub fn ceil_with_denorms(f: F32) -> F32 { apfloat::ceil_with_denorms(f) }

pub fn ceil_daz(f: F32) -> F32 { apfloat::ceil_daz(f) }

pub fn floor_with_denorms(f: F32) -> F32 { apfloat::floor_with_denorms(f) }

pub fn floor_daz(f: F32) -> F32 { apfloat::floor_daz(f) }

pub fn trunc(f: F32) -> F32 { apfloat::trunc(f) }

pub fn round<ROUND_STYLE: apfloat::RoundStyle = {apfloat::RoundStyle::TIES_TO_EVEN}>
    (f: F32) -> F32 {
    apfloat::round<F32::EXP_SIZE, F32::FRACTION_SIZE, ROUND_STYLE>(f)
}

#[test]
fn round_test() {
    let minus_tiny_f32 = F32 {
        sign: u1:1,
        bexp: bias(min_normal_exp()),
        fraction: u23:0b110_1011_0000_0000_0000_0000,
    };
    assert_eq(zero(u1:1), round<apfloat::RoundStyle::TIES_TO_EVEN>(minus_tiny_f32));

    // 1.5
    let one_dot_five =
        F32 { sign: u1:0, bexp: bias(s8:0), fraction: u23:0b100_0000_0000_0000_0000_0000 };
    // 2.0
    let expected = F32 { sign: u1:0, bexp: bias(s8:1), fraction: u23:0 };
    assert_eq(expected, round<apfloat::RoundStyle::TIES_TO_EVEN>(one_dot_five));

    // -1.5
    let minus_one_dot_five =
        F32 { sign: u1:1, bexp: bias(s8:0), fraction: u23:0b100_0000_0000_0000_0000_0000 };
    // -2.0
    let expected = F32 { sign: u1:1, bexp: bias(s8:1), fraction: u23:0 };
    assert_eq(expected, round<apfloat::RoundStyle::TIES_TO_EVEN>(minus_one_dot_five));
}

#[quickcheck]
fn int_roundtrip(x: s25) -> bool {
    // every integer between 0 and 0x100_0000 can be exactly represented
    to_int32(from_int32(x as s32)) == x as s32
}

#[quickcheck]
fn uint_roundtrip(x: u24) -> bool {
    // every integer between 0 and 0x100_0000 can be exactly represented, so only cover up to u24,
    // i.e. values that fit in an s25.
    to_uint<u32:24>(from_int32(x as s32)) == x
}

#[quickcheck]
fn uint_roundtrip_as_u32(x: u24) -> bool { to_uint32(from_int32(x as s32)) == x as u32 }
