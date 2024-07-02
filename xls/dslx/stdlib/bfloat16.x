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

// bfloat16 routines.
import apfloat;

pub const BF16_EXP_SZ = u32:8;  // Exponent bits
pub const BF16_FRACTION_SZ = u32:7;  // Fraction bits
pub const BF16_TOTAL_SZ = u32:1 + BF16_EXP_SZ + BF16_FRACTION_SZ;

pub type BF16 = apfloat::APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ>;
pub type FloatTag = apfloat::APFloatTag;
pub type TaggedBF16 = (FloatTag, BF16);

pub fn qnan() -> BF16 { apfloat::qnan<BF16_EXP_SZ, BF16_FRACTION_SZ>() }

pub fn is_nan(f: BF16) -> bool { apfloat::is_nan<BF16_EXP_SZ, BF16_FRACTION_SZ>(f) }

pub fn inf(sign: u1) -> BF16 { apfloat::inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(sign) }

pub fn is_inf(f: BF16) -> bool { apfloat::is_inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(f) }

pub fn is_pos_inf(f: BF16) -> bool { apfloat::is_pos_inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(f) }

pub fn is_neg_inf(f: BF16) -> bool { apfloat::is_neg_inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(f) }

pub fn zero(sign: u1) -> BF16 { apfloat::zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(sign) }

pub fn one(sign: u1) -> BF16 { apfloat::one<BF16_EXP_SZ, BF16_FRACTION_SZ>(sign) }

pub fn negate(x: BF16) -> BF16 { apfloat::negate(x) }

pub fn max_normal_exp() -> s8 { apfloat::max_normal_exp<BF16_EXP_SZ>() }

pub fn min_normal_exp() -> s8 { apfloat::min_normal_exp<BF16_EXP_SZ>() }

pub fn unbiased_exponent(f: BF16) -> s8 {
    apfloat::unbiased_exponent<BF16_EXP_SZ, BF16_FRACTION_SZ>(f)
}

pub fn bias(unbiased_exponent_in: s8) -> u8 {
    apfloat::bias<BF16_EXP_SZ, BF16_FRACTION_SZ>(unbiased_exponent_in)
}

pub fn flatten(f: BF16) -> u16 { apfloat::flatten<BF16_EXP_SZ, BF16_FRACTION_SZ>(f) }

pub fn unflatten(f: u16) -> BF16 { apfloat::unflatten<BF16_EXP_SZ, BF16_FRACTION_SZ>(f) }

pub fn ldexp(f: BF16, e: s32) -> BF16 { apfloat::ldexp(f, e) }

pub fn cast_from_fixed_using_rne<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> BF16 {
    apfloat::cast_from_fixed_using_rne<BF16_EXP_SZ, BF16_FRACTION_SZ>(s)
}

pub fn cast_from_fixed_using_rz<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> BF16 {
    apfloat::cast_from_fixed_using_rz<BF16_EXP_SZ, BF16_FRACTION_SZ>(s)
}

pub fn cast_to_fixed<NUM_DST_BITS: u32>(to_cast: BF16) -> sN[NUM_DST_BITS] {
    apfloat::cast_to_fixed<NUM_DST_BITS, BF16_EXP_SZ, BF16_FRACTION_SZ>(to_cast)
}

pub fn subnormals_to_zero(f: BF16) -> BF16 {
    apfloat::subnormals_to_zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(f)
}

pub fn is_zero_or_subnormal(f: BF16) -> bool {
    apfloat::is_zero_or_subnormal<BF16_EXP_SZ, BF16_FRACTION_SZ>(f)
}

pub fn eq_2(x: BF16, y: BF16) -> bool { apfloat::eq_2<BF16_EXP_SZ, BF16_FRACTION_SZ>(x, y) }

pub fn gt_2(x: BF16, y: BF16) -> bool { apfloat::gt_2<BF16_EXP_SZ, BF16_FRACTION_SZ>(x, y) }

pub fn gte_2(x: BF16, y: BF16) -> bool { apfloat::gte_2<BF16_EXP_SZ, BF16_FRACTION_SZ>(x, y) }

pub fn lt_2(x: BF16, y: BF16) -> bool { apfloat::lt_2<BF16_EXP_SZ, BF16_FRACTION_SZ>(x, y) }

pub fn lte_2(x: BF16, y: BF16) -> bool { apfloat::lte_2<BF16_EXP_SZ, BF16_FRACTION_SZ>(x, y) }

pub fn normalize(sign: u1, exp: u8, fraction_with_hidden: u8) -> BF16 {
    apfloat::normalize<BF16_EXP_SZ, BF16_FRACTION_SZ>(sign, exp, fraction_with_hidden)
}

pub fn to_int<RESULT_SZ: u32>(x: BF16) -> sN[RESULT_SZ] {
    apfloat::to_int<BF16_EXP_SZ, BF16_FRACTION_SZ, RESULT_SZ>(x)
}

pub fn to_int16(x: BF16) -> s16 { apfloat::to_int<BF16_EXP_SZ, BF16_FRACTION_SZ, BF16_TOTAL_SZ>(x) }

pub fn tag(f: BF16) -> FloatTag { apfloat::tag(f) }

// Increments the fraction of the input BF16 by one and returns the
// normalized result. Input must be a normal *non-zero* number.
pub fn increment_fraction(input: BF16) -> BF16 {
    // Add the hidden bit and one (the increment amount) to the fractional part.
    // If it overflows 8 bits the number must be normalized.
    let new_fraction = u9:0x80 + (input.fraction as u9) + u9:1;
    let new_fraction_msb = new_fraction[8+:u1];
    match (new_fraction_msb, input.bexp >= u8:0xfe) {
        // Overflow to infinity.
        (true, true) => inf(input.sign),
        // Significand overflowed, normalize.
        (true, false) => BF16 {
            sign: input.sign, bexp: input.bexp + u8:1, fraction: new_fraction[1+:u7]
        },
        // No normalization required.
        (_, _) => BF16 { sign: input.sign, bexp: input.bexp, fraction: new_fraction[:7] },
    }
}

#[test]
fn increment_fraction_bf16_test() {
    // No normalization required.
    assert_eq(
        increment_fraction(BF16 { sign: u1:0, bexp: u8:42, fraction: u7:0 }),
        BF16 { sign: u1:0, bexp: u8:42, fraction: u7:1 });
    assert_eq(
        increment_fraction(BF16 { sign: u1:1, bexp: u8:42, fraction: u7:0 }),
        BF16 { sign: u1:1, bexp: u8:42, fraction: u7:1 });
    assert_eq(
        increment_fraction(BF16 { sign: u1:0, bexp: u8:42, fraction: u7:12 }),
        BF16 { sign: u1:0, bexp: u8:42, fraction: u7:13 });
    assert_eq(
        increment_fraction(BF16 { sign: u1:0, bexp: u8:254, fraction: u7:0x3f }),
        BF16 { sign: u1:0, bexp: u8:254, fraction: u7:0x40 });

    // Normalization required.
    assert_eq(
        increment_fraction(BF16 { sign: u1:1, bexp: u8:1, fraction: u7:0x7f }),
        BF16 { sign: u1:1, bexp: u8:2, fraction: u7:0 });
    assert_eq(
        increment_fraction(BF16 { sign: u1:0, bexp: u8:123, fraction: u7:0x7f }),
        BF16 { sign: u1:0, bexp: u8:124, fraction: u7:0 });

    // Overflow to infinity.
    assert_eq(increment_fraction(BF16 { sign: u1:0, bexp: u8:254, fraction: u7:0x7f }), inf(u1:0));
    assert_eq(increment_fraction(BF16 { sign: u1:1, bexp: u8:254, fraction: u7:0x7f }), inf(u1:1));
}

pub fn add(x: BF16, y: BF16) -> BF16 { apfloat::add(x, y) }

pub fn sub(x: BF16, y: BF16) -> BF16 { apfloat::sub(x, y) }

pub fn mul(x: BF16, y: BF16) -> BF16 { apfloat::mul(x, y) }

pub fn fma(a: BF16, b: BF16, c: BF16) -> BF16 { apfloat::fma(a, b, c) }

pub fn has_fractional_part(f: BF16) -> bool { apfloat::has_fractional_part(f) }

pub fn has_negative_exponent(f: BF16) -> bool { apfloat::has_negative_exponent(f) }

pub fn ceil(f: BF16) -> BF16 { apfloat::ceil(f) }

pub fn floor(f: BF16) -> BF16 { apfloat::floor(f) }
