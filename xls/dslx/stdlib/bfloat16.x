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
import std;

pub type BF16 = apfloat::APFloat<u32:8, u32:7>;
pub type FloatTag = apfloat::APFloatTag;
pub type TaggedBF16 = (FloatTag, BF16);

pub fn qnan() -> BF16 { apfloat::qnan<BF16::EXP_SIZE, BF16::FRACTION_SIZE>() }

pub fn is_nan(f: BF16) -> bool { apfloat::is_nan<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(f) }

pub fn inf(sign: u1) -> BF16 { apfloat::inf<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(sign) }

pub fn is_inf(f: BF16) -> bool { apfloat::is_inf<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(f) }

pub fn is_pos_inf(f: BF16) -> bool { apfloat::is_pos_inf<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(f) }

pub fn is_neg_inf(f: BF16) -> bool { apfloat::is_neg_inf<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(f) }

pub fn zero(sign: u1) -> BF16 { apfloat::zero<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(sign) }

pub fn one(sign: u1) -> BF16 { apfloat::one<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(sign) }

pub fn negate(x: BF16) -> BF16 { apfloat::negate(x) }

pub fn max_normal_exp() -> s8 { apfloat::max_normal_exp<BF16::EXP_SIZE>() }

pub fn min_normal_exp() -> s8 { apfloat::min_normal_exp<BF16::EXP_SIZE>() }

pub fn unbiased_exponent(f: BF16) -> s8 {
    apfloat::unbiased_exponent<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(f)
}

pub fn bias(unbiased_exponent_in: s8) -> u8 { apfloat::bias(unbiased_exponent_in) }

pub fn flatten(f: BF16) -> u16 { apfloat::flatten<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(f) }

pub fn unflatten(f: u16) -> BF16 { apfloat::unflatten<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(f) }

pub fn ldexp(f: BF16, e: s32) -> BF16 { apfloat::ldexp(f, e) }

pub fn cast_from_fixed_using_rne<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> BF16 {
    apfloat::cast_from_fixed_using_rne<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(s)
}

pub fn cast_from_fixed_using_rz<NUM_SRC_BITS: u32>(s: sN[NUM_SRC_BITS]) -> BF16 {
    apfloat::cast_from_fixed_using_rz<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(s)
}

pub fn cast_to_fixed<NUM_DST_BITS: u32>(to_cast: BF16) -> sN[NUM_DST_BITS] {
    apfloat::cast_to_fixed<NUM_DST_BITS, BF16::EXP_SIZE, BF16::FRACTION_SIZE>(to_cast)
}

pub fn subnormals_to_zero(f: BF16) -> BF16 {
    apfloat::subnormals_to_zero<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(f)
}

pub fn is_zero_or_subnormal(f: BF16) -> bool {
    apfloat::is_zero_or_subnormal<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(f)
}

pub fn eq_2(x: BF16, y: BF16) -> bool { apfloat::eq_2<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(x, y) }

pub fn gt_2(x: BF16, y: BF16) -> bool { apfloat::gt_2<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(x, y) }

pub fn gte_2(x: BF16, y: BF16) -> bool { apfloat::gte_2<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(x, y) }

pub fn lt_2(x: BF16, y: BF16) -> bool { apfloat::lt_2<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(x, y) }

pub fn lte_2(x: BF16, y: BF16) -> bool { apfloat::lte_2<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(x, y) }

pub fn normalize(sign: u1, exp: u8, fraction_with_hidden: u8) -> BF16 {
    apfloat::normalize<BF16::EXP_SIZE, BF16::FRACTION_SIZE>(sign, exp, fraction_with_hidden)
}

pub fn to_int<RESULT_SZ: u32>(x: BF16) -> sN[RESULT_SZ] {
    apfloat::to_int<BF16::EXP_SIZE, BF16::FRACTION_SIZE, RESULT_SZ>(x)
}

pub fn to_int16(x: BF16) -> s16 { apfloat::to_int<BF16::EXP_SIZE, BF16::FRACTION_SIZE, u32:16>(x) }

pub fn to_uint<RESULT_SZ: u32>(x: BF16) -> uN[RESULT_SZ] { apfloat::to_uint<RESULT_SZ>(x) }

pub fn to_uint16(x: BF16) -> u16 { apfloat::to_uint<u32:16>(x) }

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
            sign: input.sign,
            bexp: input.bexp + u8:1,
            fraction: new_fraction[1+:u7],
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

pub fn trunc(f: BF16) -> BF16 { apfloat::trunc(f) }

pub fn round<ROUND_STYLE: apfloat::RoundStyle = {apfloat::RoundStyle::TIES_TO_EVEN}>
    (f: BF16) -> BF16 {
    apfloat::round<BF16::EXP_SIZE, BF16::FRACTION_SIZE, ROUND_STYLE>(f)
}

#[test]
fn round_test() {
    let minus_tiny_bf16 = BF16 { sign: u1:1, bexp: bias(min_normal_exp()), fraction: u7:0b1101011 };
    assert_eq(zero(u1:1), round<apfloat::RoundStyle::TIES_TO_EVEN>(minus_tiny_bf16));

    // 1.5
    let one_dot_five = BF16 { sign: u1:0, bexp: bias(s8:0), fraction: u7:0b1000000 };
    // 2.0
    let expected = BF16 { sign: u1:0, bexp: bias(s8:1), fraction: u7:0b0000000 };
    assert_eq(expected, round<apfloat::RoundStyle::TIES_TO_EVEN>(one_dot_five));

    // -1.5
    let minus_one_dot_five = BF16 { sign: u1:1, bexp: bias(s8:0), fraction: u7:0b1000000 };
    // -2.0
    let expected = BF16 { sign: u1:1, bexp: bias(s8:1), fraction: u7:0b0000000 };
    assert_eq(expected, round<apfloat::RoundStyle::TIES_TO_EVEN>(minus_one_dot_five));
}

pub fn from_float32(f32: apfloat::APFloat<u32:8, u32:23>) -> BF16 {
    apfloat::downcast_rne<BF16::FRACTION_SIZE, BF16::EXP_SIZE>(f32)
}

// Converts the given signed integer to bfloat16. For s8, all values can be
// captured exactly, so no need to round or handle overflow.
pub fn from_int8(x: s8) -> BF16 {
    const MAX_EXPONENT = u4:6;
    const BIAS = u8:127;

    let sign = std::msb(x as u8);
    let unsigned = if sign { -x as u7 } else { x as u7 };

    // Remove leading 1.
    let lz = clz(unsigned) as u4;
    let fraction = unsigned << (lz + u4:1);

    let exp = MAX_EXPONENT - lz;
    let bexp = exp as u8 + BIAS;

    let result = BF16 { sign, bexp, fraction };

    // Handle special cases: zero and max negative s8.
    let result = if unsigned == u7:0 { zero(sign) } else { result };
    let max_neg_s8 = BF16 { sign: u1:1, bexp: u8:134, fraction: u7:0 };
    let result = if x == s8:-128 { max_neg_s8 } else { result };
    result
}

#[test]
fn from_int8_test() {
    let expected = BF16 { sign: u1:0, bexp: u8:130, fraction: u7:64 };
    let actual = from_int8(s8:12);
    assert_eq(expected, actual);

    let expected = one(u1:1);
    let actual = from_int8(s8:-1);
    assert_eq(expected, actual);

    let val = s8:35;
    let actual = to_int16(from_int8(val));
    assert_eq(val as s16, actual);

    let val = s8:-35;
    let actual = to_int16(from_int8(val));
    assert_eq(val as s16, actual);

    let val = s8:127;
    let actual = to_int16(from_int8(val));
    assert_eq(val as s16, actual);

    let val = s8:-150;
    let actual = to_int16(from_int8(val));
    assert_eq(val as s16, actual);

    let val = s8:-42;
    let actual = to_int16(from_int8(val));
    assert_eq(val as s16, actual);

    let val = s8:0;
    let actual = to_int16(from_int8(val));
    assert_eq(val as s16, actual);

    let val = s8:-128;
    let actual = to_int16(from_int8(val));
    assert_eq(val as s16, actual);
}

// NB s5 to ensure no rounding during the add occurs.
#[quickcheck]
fn add_less_than_one_half_round_check(f_i: s5) -> bool {
    let f = f_i as s8;
    let flt = from_int8(f);
    let is_neg = f < s8:0;
    let one_half_less_a_bit = BF16 { sign: is_neg, bexp: bias(s8:-2), fraction: u7:0b011_0000 };
    round<apfloat::RoundStyle::TIES_TO_EVEN>(add(flt, one_half_less_a_bit)) == flt &&
    round<apfloat::RoundStyle::TIES_TO_AWAY>(add(flt, one_half_less_a_bit)) == flt
}

// NB s5 to ensure no rounding during the add occurs.
#[quickcheck]
fn add_more_than_one_half_round_check(f_i: s5) -> bool {
    let f = f_i as s8;
    let flt = from_int8(f);
    let is_neg = f < s8:0;
    let flt_plus_one = from_int8(f + if is_neg { s8:-1 } else { s8:1 });
    let one_half_plus_a_bit = BF16 { sign: is_neg, bexp: bias(s8:-1), fraction: u7:0b111_0000 };
    round<apfloat::RoundStyle::TIES_TO_EVEN>(add(flt, one_half_plus_a_bit)) == flt_plus_one &&
    round<apfloat::RoundStyle::TIES_TO_AWAY>(add(flt, one_half_plus_a_bit)) == flt_plus_one
}

// NB s5 to ensure no rounding during the add occurs.
#[quickcheck]
fn add_one_half_round_to_even_check(f_i: s5) -> bool {
    let f = f_i as s8;
    let flt = from_int8(f);
    let is_neg = f < s8:0;

    let flt_plus_one = from_int8(f + if is_neg { s8:-1 } else { s8:1 });

    let f_is_even = !(f as u8)[0+:u1];
    let one_half = BF16 { sign: is_neg, bexp: bias(s8:-1), fraction: u7:0 };
    if f_is_even {
        round<apfloat::RoundStyle::TIES_TO_EVEN>(add(flt, one_half)) == flt
    } else {
        round<apfloat::RoundStyle::TIES_TO_EVEN>(add(flt, one_half)) == flt_plus_one
    }
}

// NB s5 to ensure no rounding during the add occurs.
#[quickcheck]
fn add_one_half_round_to_away_check(f_i: s5) -> bool {
    let f = f_i as s8;
    let flt = from_int8(f);
    let is_neg = f < s8:0;
    let flt_plus_one = from_int8(f + if is_neg { s8:-1 } else { s8:1 });
    let one_half = BF16 { sign: is_neg, bexp: bias(s8:-1), fraction: u7:0 };
    round<apfloat::RoundStyle::TIES_TO_AWAY>(add(flt, one_half)) == flt_plus_one
}

#[quickcheck]
fn int_roundtrip(x: s8) -> bool { to_int16(from_int8(x)) == x as s16 }

#[quickcheck]
fn uint_roundtrip(x: u7) -> bool { to_uint<u32:7>(from_int8(x as s8)) == x }

#[quickcheck]
fn uint_roundtrip_as_u16(x: u7) -> bool { to_uint16(from_int8(x as s8)) == x as u16 }

#[quickcheck]
fn float32_to_bfloat16_no_precision_loss(sign: u1, exp: u8, frac_part: u7) -> bool {
    type F32 = apfloat::APFloat<u32:8, u32:23>;
    (exp == u8:0) || (exp == u8::MAX) ||
    (from_float32(F32 { sign, bexp: exp, fraction: frac_part ++ u16:0 }) ==
    BF16 { sign, bexp: exp, fraction: frac_part })
}

#[quickcheck]
fn float32_to_bfloat16_subnormals_flushed_to_zero(sign: u1, fraction: u23) -> bool {
    type F32 = apfloat::APFloat<u32:8, u32:23>;
    from_float32(F32 { sign, bexp: u8:0, fraction }) == zero(sign)
}

// Converts the given unsigned integer to bfloat16. For u8, all values can be
// captured exactly, so no need to round or handle overflow.
pub fn from_uint8(x: u8) -> BF16 {
    const MAX_EXPONENT = u4:7;
    const BIAS = std::signed_max_value<BF16::EXP_SIZE>() as u8;  // 127

    // Remove leading 1.
    let lz = clz(x) as u4;
    // Shifted by at last 1 so lowest bit is 0
    let fraction = (x << (lz + u4:1))[1+:u7];

    let exp = MAX_EXPONENT - lz;
    let bexp = exp as u8 + BIAS;

    let sign = u1:0;
    let result = BF16 { sign, bexp, fraction };

    // Handle special cases: zero and max negative s8.
    if x == u8:0 { zero(sign) } else { result }
}

#[test]
fn from_uint8_test() {
    let expected = BF16 { sign: u1:0, bexp: u8:130, fraction: u7:64 };
    let actual = from_uint8(u8:12);
    assert_eq(expected, actual);

    let expected = one(u1:0);
    let actual = from_uint8(u8:1);
    assert_eq(expected, actual);

    let val = u8:35;
    let actual = to_uint16(from_uint8(val));
    assert_eq(val as u16, actual);

    let val = u8:127;
    let actual = to_uint16(from_uint8(val));
    assert_eq(val as u16, actual);

    let val = u8:255;
    let actual = to_uint16(from_uint8(val));
    assert_eq(val as u16, actual);

    let val = u8:0;
    let actual = to_uint16(from_uint8(val));
    assert_eq(val as u16, actual);
}

#[quickcheck]
fn uint8_roundtrip(x: u8) -> bool { to_uint16(from_uint8(x)) == x as u16 }
