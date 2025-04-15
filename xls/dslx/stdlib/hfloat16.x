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

// hfloat16 routines:
// https://en.wikipedia.org/wiki/Half-precision_floating-point_format

import apfloat;
import std;

pub type HF16 = apfloat::APFloat<u32:5, u32:10>;
pub type FloatTag = apfloat::APFloatTag;
pub type TaggedHF16 = (FloatTag, HF16);

pub fn qnan() -> HF16 { apfloat::qnan<HF16::EXP_SIZE, HF16::FRACTION_SIZE>() }

pub fn is_nan(f: HF16) -> bool { apfloat::is_nan<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(f) }

pub fn inf(sign: u1) -> HF16 { apfloat::inf<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(sign) }

pub fn is_inf(f: HF16) -> bool { apfloat::is_inf<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(f) }

pub fn is_pos_inf(f: HF16) -> bool { apfloat::is_pos_inf<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(f) }

pub fn is_neg_inf(f: HF16) -> bool { apfloat::is_neg_inf<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(f) }

pub fn zero(sign: u1) -> HF16 { apfloat::zero<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(sign) }

pub fn one(sign: u1) -> HF16 { apfloat::one<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(sign) }

pub fn max_normal(sign: u1) -> HF16 {
    apfloat::max_normal<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(sign)
}

pub fn negate(x: HF16) -> HF16 { apfloat::negate(x) }

pub fn max_normal_exp() -> sN[HF16::EXP_SIZE] { apfloat::max_normal_exp<HF16::EXP_SIZE>() }

pub fn min_normal_exp() -> sN[HF16::EXP_SIZE] { apfloat::min_normal_exp<HF16::EXP_SIZE>() }

pub fn unbiased_exponent(f: HF16) -> sN[HF16::EXP_SIZE] {
    apfloat::unbiased_exponent<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(f)
}

pub fn bias(unbiased_exponent_in: sN[HF16::EXP_SIZE]) -> uN[HF16::EXP_SIZE] {
    apfloat::bias(unbiased_exponent_in)
}

pub fn flatten(f: HF16) -> uN[HF16::TOTAL_SIZE] {
    apfloat::flatten<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(f)
}

pub fn unflatten(f: uN[HF16::TOTAL_SIZE]) -> HF16 {
    apfloat::unflatten<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(f)
}

pub fn ldexp(f: HF16, e: s32) -> HF16 { apfloat::ldexp(f, e) }

pub fn cast_from_fixed_using_rne<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> HF16 {
    apfloat::cast_from_fixed_using_rne<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(s)
}

pub fn cast_from_fixed_using_rz<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> HF16 {
    apfloat::cast_from_fixed_using_rz<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(s)
}

pub fn cast_to_fixed<NUM_DST_BITS: u32>(to_cast: HF16) -> sN[NUM_DST_BITS] {
    apfloat::cast_to_fixed<NUM_DST_BITS, HF16::EXP_SIZE, HF16::FRACTION_SIZE>(to_cast)
}

pub fn subnormals_to_zero(f: HF16) -> HF16 {
    apfloat::subnormals_to_zero<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(f)
}

pub fn is_zero_or_subnormal(f: HF16) -> bool {
    apfloat::is_zero_or_subnormal<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(f)
}

pub fn eq_2(x: HF16, y: HF16) -> bool { apfloat::eq_2<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(x, y) }

pub fn gt_2(x: HF16, y: HF16) -> bool { apfloat::gt_2<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(x, y) }

pub fn gte_2(x: HF16, y: HF16) -> bool { apfloat::gte_2<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(x, y) }

pub fn lt_2(x: HF16, y: HF16) -> bool { apfloat::lt_2<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(x, y) }

pub fn lte_2(x: HF16, y: HF16) -> bool { apfloat::lte_2<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(x, y) }

pub fn normalize
    (sign: u1, exp: uN[HF16::EXP_SIZE], fraction_with_hidden: uN[HF16::FRACTION_SIZE + u32:1])
    -> HF16 {
    apfloat::normalize<HF16::EXP_SIZE, HF16::FRACTION_SIZE>(sign, exp, fraction_with_hidden)
}

pub fn to_int<RESULT_SZ: u32>(x: HF16) -> sN[RESULT_SZ] {
    apfloat::to_int<HF16::EXP_SIZE, HF16::FRACTION_SIZE, RESULT_SZ>(x)
}

pub fn to_int16(x: HF16) -> s16 { apfloat::to_int<HF16::EXP_SIZE, HF16::FRACTION_SIZE, u32:16>(x) }

pub fn to_uint<RESULT_SZ: u32>(x: HF16) -> uN[RESULT_SZ] { apfloat::to_uint<RESULT_SZ>(x) }

pub fn to_uint16(x: HF16) -> u16 { apfloat::to_uint<u32:16>(x) }

pub fn tag(f: HF16) -> FloatTag { apfloat::tag(f) }

pub fn add(x: HF16, y: HF16) -> HF16 { apfloat::add(x, y) }

pub fn sub(x: HF16, y: HF16) -> HF16 { apfloat::sub(x, y) }

pub fn mul(x: HF16, y: HF16) -> HF16 { apfloat::mul(x, y) }

pub fn fma(a: HF16, b: HF16, c: HF16) -> HF16 { apfloat::fma(a, b, c) }

pub fn has_fractional_part(f: HF16) -> bool { apfloat::has_fractional_part(f) }

pub fn has_negative_exponent(f: HF16) -> bool { apfloat::has_negative_exponent(f) }

pub fn ceil(f: HF16) -> HF16 { apfloat::ceil(f) }

pub fn floor(f: HF16) -> HF16 { apfloat::floor(f) }

pub fn trunc(f: HF16) -> HF16 { apfloat::trunc(f) }

pub fn round<ROUND_STYLE: apfloat::RoundStyle = {apfloat::RoundStyle::TIES_TO_EVEN}>
    (f: HF16) -> HF16 {
    apfloat::round<HF16::EXP_SIZE, HF16::FRACTION_SIZE, ROUND_STYLE>(f)
}

#[test]
fn round_test() {
    let minus_tiny_hf16 = HF16 {
        sign: u1:1,
        bexp: bias(min_normal_exp()),
        fraction: std::unsigned_max_value<HF16::FRACTION_SIZE>(),
    };
    assert_eq(zero(u1:1), round<apfloat::RoundStyle::TIES_TO_EVEN>(minus_tiny_hf16));

    // 1.5
    let one_dot_five = HF16 {
        sign: u1:0,
        bexp: bias(sN[HF16::EXP_SIZE]:0),
        fraction: uN[HF16::FRACTION_SIZE]:1 << (HF16::FRACTION_SIZE - u32:1),
    };
    // 2.0
    let expected =
        HF16 { sign: u1:0, bexp: bias(sN[HF16::EXP_SIZE]:1), fraction: uN[HF16::FRACTION_SIZE]:0 };
    assert_eq(expected, round<apfloat::RoundStyle::TIES_TO_EVEN>(one_dot_five));

    // -1.5
    let minus_one_dot_five = HF16 {
        sign: u1:1,
        bexp: bias(sN[HF16::EXP_SIZE]:0),
        fraction: uN[HF16::FRACTION_SIZE]:1 << (HF16::FRACTION_SIZE - u32:1),
    };
    // -2.0
    let expected =
        HF16 { sign: u1:1, bexp: bias(sN[HF16::EXP_SIZE]:1), fraction: uN[HF16::FRACTION_SIZE]:0 };
    assert_eq(expected, round<apfloat::RoundStyle::TIES_TO_EVEN>(minus_one_dot_five));
}

pub fn from_float32(f32: apfloat::APFloat<u32:8, u32:23>) -> HF16 {
    apfloat::downcast_rne<HF16::FRACTION_SIZE, HF16::EXP_SIZE>(f32)
}

#[quickcheck]
fn float32_to_hfloat16_no_precision_loss
    (sign: u1, bexp: uN[HF16::EXP_SIZE], fraction: uN[HF16::FRACTION_SIZE]) -> bool {
    let exp = unbiased_exponent(HF16 { sign, bexp, fraction });
    const F32_EXP_SZ = u32:8;
    const F32_FRAC_SZ = u32:23;
    let exp_f32 = exp as sN[F32_EXP_SZ];
    let bexp_f32 = apfloat::bias<F32_EXP_SZ>(exp_f32);
    let fraction_f32 = fraction ++ uN[F32_FRAC_SZ - HF16::FRACTION_SIZE]:0;
    type F32 = apfloat::APFloat<F32_EXP_SZ, F32_FRAC_SZ>;
    const HF16_EXP_MAX = std::unsigned_max_value<HF16::EXP_SIZE>();
    (bexp == uN[HF16::EXP_SIZE]:0) || (bexp == HF16_EXP_MAX) ||
    (from_float32(F32 { sign, bexp: bexp_f32, fraction: fraction_f32 }) ==
    HF16 { sign, bexp, fraction })
}

#[quickcheck]
fn float32_to_hfloat16_subnormals_flushed_to_zero(sign: u1, fraction: u23) -> bool {
    const F32_EXP_SZ = u32:8;
    const F32_FRAC_SZ = u32:23;
    type F32 = apfloat::APFloat<F32_EXP_SZ, F32_FRAC_SZ>;
    from_float32(F32 { sign, bexp: uN[F32_EXP_SZ]:0, fraction }) == zero(sign)
}
// TODO: google/xls#2036 - add increment_fraction.
// TODO: google/xls#2036 - add from_int8.
