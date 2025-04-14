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

// 64-bit floating point routines.
import apfloat;

pub const F64_EXP_SZ = u32:11;  // Exponent bits
pub const F64_FRACTION_SZ = u32:52;  // Fraction bits
pub const F64_TOTAL_SZ = u32:1 + F64_EXP_SZ + F64_FRACTION_SZ;

pub type F64 = apfloat::APFloat<F64_EXP_SZ, F64_FRACTION_SZ>;
pub type FloatTag = apfloat::APFloatTag;

pub type TaggedF64 = (FloatTag, F64);

pub fn qnan() -> F64 { apfloat::qnan<F64_EXP_SZ, F64_FRACTION_SZ>() }

pub fn is_nan(f: F64) -> bool { apfloat::is_nan<F64_EXP_SZ, F64_FRACTION_SZ>(f) }

pub fn inf(sign: u1) -> F64 { apfloat::inf<F64_EXP_SZ, F64_FRACTION_SZ>(sign) }

pub fn is_inf(f: F64) -> bool { apfloat::is_inf<F64_EXP_SZ, F64_FRACTION_SZ>(f) }

pub fn is_pos_inf(f: F64) -> bool { apfloat::is_pos_inf<F64_EXP_SZ, F64_FRACTION_SZ>(f) }

pub fn is_neg_inf(f: F64) -> bool { apfloat::is_neg_inf<F64_EXP_SZ, F64_FRACTION_SZ>(f) }

pub fn zero(sign: u1) -> F64 { apfloat::zero<F64_EXP_SZ, F64_FRACTION_SZ>(sign) }

pub fn one(sign: u1) -> F64 { apfloat::one<F64_EXP_SZ, F64_FRACTION_SZ>(sign) }

pub fn max_normal(sign: u1) -> F64 { apfloat::max_normal<F64_EXP_SZ, F64_FRACTION_SZ>(sign) }

pub fn negate(x: F64) -> F64 { apfloat::negate(x) }

pub fn max_normal_exp() -> sN[F64_EXP_SZ] { apfloat::max_normal_exp<F64_EXP_SZ>() }

pub fn min_normal_exp() -> sN[F64_EXP_SZ] { apfloat::min_normal_exp<F64_EXP_SZ>() }

pub fn unbiased_exponent(f: F64) -> sN[F64_EXP_SZ] {
    apfloat::unbiased_exponent<F64_EXP_SZ, F64_FRACTION_SZ>(f)
}

pub fn bias(unbiased_exponent_in: sN[F64_EXP_SZ]) -> uN[F64_EXP_SZ] {
    apfloat::bias(unbiased_exponent_in)
}

pub fn flatten(f: F64) -> uN[F64_TOTAL_SZ] { apfloat::flatten<F64_EXP_SZ, F64_FRACTION_SZ>(f) }

pub fn unflatten(f: uN[F64_TOTAL_SZ]) -> F64 { apfloat::unflatten<F64_EXP_SZ, F64_FRACTION_SZ>(f) }

pub fn ldexp(f: F64, e: s32) -> F64 { apfloat::ldexp(f, e) }

pub fn cast_from_fixed_using_rne<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> F64 {
    apfloat::cast_from_fixed_using_rne<F64_EXP_SZ, F64_FRACTION_SZ>(s)
}

pub fn cast_from_fixed_using_rz<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> F64 {
    apfloat::cast_from_fixed_using_rz<F64_EXP_SZ, F64_FRACTION_SZ>(s)
}

pub fn cast_to_fixed<NUM_DST_BITS: u32>(to_cast: F64) -> sN[NUM_DST_BITS] {
    apfloat::cast_to_fixed<NUM_DST_BITS, F64_EXP_SZ, F64_FRACTION_SZ>(to_cast)
}

pub fn subnormals_to_zero(f: F64) -> F64 {
    apfloat::subnormals_to_zero<F64_EXP_SZ, F64_FRACTION_SZ>(f)
}

pub fn is_zero_or_subnormal(f: F64) -> bool {
    apfloat::is_zero_or_subnormal<F64_EXP_SZ, F64_FRACTION_SZ>(f)
}

pub fn eq_2(x: F64, y: F64) -> bool { apfloat::eq_2<F64_EXP_SZ, F64_FRACTION_SZ>(x, y) }

pub fn gt_2(x: F64, y: F64) -> bool { apfloat::gt_2<F64_EXP_SZ, F64_FRACTION_SZ>(x, y) }

pub fn gte_2(x: F64, y: F64) -> bool { apfloat::gte_2<F64_EXP_SZ, F64_FRACTION_SZ>(x, y) }

pub fn lt_2(x: F64, y: F64) -> bool { apfloat::lt_2<F64_EXP_SZ, F64_FRACTION_SZ>(x, y) }

pub fn lte_2(x: F64, y: F64) -> bool { apfloat::lte_2<F64_EXP_SZ, F64_FRACTION_SZ>(x, y) }

pub fn normalize
    (sign: u1, exp: uN[F64_EXP_SZ], fraction_with_hidden: uN[u32:1 + F64_FRACTION_SZ]) -> F64 {
    apfloat::normalize<F64_EXP_SZ, F64_FRACTION_SZ>(sign, exp, fraction_with_hidden)
}

pub fn tag(f: F64) -> FloatTag { apfloat::tag<F64_EXP_SZ, F64_FRACTION_SZ>(f) }

pub fn to_int<RESULT_SZ: u32>(x: F64) -> sN[RESULT_SZ] {
    apfloat::to_int<F64_EXP_SZ, F64_FRACTION_SZ, RESULT_SZ>(x)
}

pub fn to_int64(x: F64) -> s64 { apfloat::to_int<F64_EXP_SZ, F64_FRACTION_SZ, u32:64>(x) }

#[test]
fn normalize_test() {
    type ExpBits = uN[F64_EXP_SZ];
    type FractionBits = uN[F64_FRACTION_SZ];
    type WideFractionBits = uN[u32:1 + F64_FRACTION_SZ];

    let expected = F64 { sign: u1:0, bexp: ExpBits:0x2, fraction: FractionBits:0xf_fffe_dcba_0000 };
    let actual = normalize(u1:0, ExpBits:0x12, WideFractionBits:0x1f_fffe_dcba);
    assert_eq(expected, actual);

    let expected = F64 { sign: u1:0, bexp: ExpBits:0x0, fraction: FractionBits:0x0 };
    let actual = normalize(u1:0, ExpBits:0x1, WideFractionBits:0x0);
    assert_eq(expected, actual);

    let expected = F64 { sign: u1:0, bexp: ExpBits:0x0, fraction: FractionBits:0x0 };
    let actual = normalize(u1:0, ExpBits:0xfe, WideFractionBits:0x0);
    assert_eq(expected, actual);

    let expected = F64 { sign: u1:1, bexp: ExpBits:0x4d, fraction: FractionBits:0x0 };
    let actual = normalize(u1:1, ExpBits:0x81, WideFractionBits:1);
    assert_eq(expected, actual);
}

#[test]
fn tag_test() {
    type ExpBits = uN[F64_EXP_SZ];
    type FractionBits = uN[F64_FRACTION_SZ];

    assert_eq(tag(F64 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:0 }), FloatTag::ZERO);
    assert_eq(tag(F64 { sign: u1:1, bexp: ExpBits:0, fraction: FractionBits:0 }), FloatTag::ZERO);
    assert_eq(tag(zero(u1:0)), FloatTag::ZERO);
    assert_eq(tag(zero(u1:1)), FloatTag::ZERO);

    assert_eq(
        tag(F64 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:1 }), FloatTag::SUBNORMAL);
    assert_eq(
        tag(F64 { sign: u1:0, bexp: ExpBits:0, fraction: FractionBits:0x7f_ffff }),
        FloatTag::SUBNORMAL);

    assert_eq(tag(F64 { sign: u1:0, bexp: ExpBits:12, fraction: FractionBits:0 }), FloatTag::NORMAL);
    assert_eq(
        tag(F64 { sign: u1:1, bexp: ExpBits:0x7fe, fraction: FractionBits:0x7f_ffff }),
        FloatTag::NORMAL);
    assert_eq(tag(F64 { sign: u1:1, bexp: ExpBits:1, fraction: FractionBits:1 }), FloatTag::NORMAL);

    assert_eq(
        tag(F64 { sign: u1:0, bexp: ExpBits:0x7ff, fraction: FractionBits:0 }), FloatTag::INFINITY);
    assert_eq(
        tag(F64 { sign: u1:1, bexp: ExpBits:0x7ff, fraction: FractionBits:0 }), FloatTag::INFINITY);
    assert_eq(tag(inf(u1:0)), FloatTag::INFINITY);
    assert_eq(tag(inf(u1:1)), FloatTag::INFINITY);

    assert_eq(tag(F64 { sign: u1:0, bexp: ExpBits:0x7ff, fraction: FractionBits:1 }), FloatTag::NAN);
    assert_eq(
        tag(F64 { sign: u1:1, bexp: ExpBits:0x7ff, fraction: FractionBits:0x7f_ffff }),
        FloatTag::NAN);
    assert_eq(tag(qnan()), FloatTag::NAN);
}

pub fn add(x: F64, y: F64) -> F64 { apfloat::add(x, y) }

pub fn sub(x: F64, y: F64) -> F64 { apfloat::sub(x, y) }

pub fn mul(x: F64, y: F64) -> F64 { apfloat::mul(x, y) }

pub fn fma(a: F64, b: F64, c: F64) -> F64 { apfloat::fma(a, b, c) }

pub fn has_fractional_part(f: F64) -> bool { apfloat::has_fractional_part(f) }

pub fn has_negative_exponent(f: F64) -> bool { apfloat::has_negative_exponent(f) }

pub fn ceil(f: F64) -> F64 { apfloat::ceil(f) }

pub fn floor(f: F64) -> F64 { apfloat::floor(f) }

pub fn trunc(f: F64) -> F64 { apfloat::trunc(f) }

pub fn round<ROUND_STYLE: apfloat::RoundStyle = {apfloat::RoundStyle::TIES_TO_EVEN}>
    (f: F64) -> F64 {
    apfloat::round<F64::EXP_SIZE, F64::FRACTION_SIZE, ROUND_STYLE>(f)
}

#[test]
fn round_test() {
    let minus_tiny_f64 =
        F64 { sign: u1:1, bexp: bias(min_normal_exp()), fraction: u52:0xf_ffff_ff00_0000 };
    assert_eq(zero(u1:1), round<apfloat::RoundStyle::TIES_TO_EVEN>(minus_tiny_f64));

    const POINT_FIVE_FRAC = u52:0x8_0000_0000_0000;
    // 1.5
    let one_dot_five = F64 { sign: u1:0, bexp: bias(s11:0), fraction: POINT_FIVE_FRAC };
    // 2.0
    let expected = F64 { sign: u1:0, bexp: bias(s11:1), fraction: u52:0 };
    assert_eq(expected, round<apfloat::RoundStyle::TIES_TO_EVEN>(one_dot_five));

    // -1.5
    let minus_one_dot_five = F64 { sign: u1:1, bexp: bias(s11:0), fraction: POINT_FIVE_FRAC };
    // -2.0
    let expected = F64 { sign: u1:1, bexp: bias(s11:1), fraction: u52:0 };
    assert_eq(expected, round<apfloat::RoundStyle::TIES_TO_EVEN>(minus_one_dot_five));
}
