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

// Arbitrary-precision floating point routines.
import std;
import abs_diff;

pub struct APFloat<EXP_SZ: u32, FRACTION_SZ: u32> {
    sign: bits[1],  // Sign bit.
    bexp: bits[EXP_SZ],  // Biased exponent.
    fraction: bits[FRACTION_SZ],  // Fractional part (no hidden bit).
}

impl APFloat<EXP_SZ, FRACTION_SZ> {
    const EXP_SIZE = EXP_SZ;
    const FRACTION_SIZE = FRACTION_SZ;
    const TOTAL_SIZE = u32:1 + EXP_SZ + FRACTION_SZ;
}

pub enum APFloatTag : u3 {
    NAN = 0,
    INFINITY = 1,
    SUBNORMAL = 2,
    ZERO = 3,
    NORMAL = 4,
}

pub fn tag<EXP_SZ: u32, FRACTION_SZ: u32>(input_float: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloatTag {
    const EXPR_MASK = std::mask_bits<EXP_SZ>();
    match (input_float.bexp, input_float.fraction) {
        (uN[EXP_SZ]:0, uN[FRACTION_SZ]:0) => APFloatTag::ZERO,
        (uN[EXP_SZ]:0, _) => APFloatTag::SUBNORMAL,
        (EXPR_MASK, uN[FRACTION_SZ]:0) => APFloatTag::INFINITY,
        (EXPR_MASK, _) => APFloatTag::NAN,
        (_, _) => APFloatTag::NORMAL,
    }
}

// Returns a quiet NaN.
pub fn qnan<EXP_SZ: u32, FRACTION_SZ: u32>() -> APFloat<EXP_SZ, FRACTION_SZ> {
    APFloat<EXP_SZ, FRACTION_SZ> {
        sign: bits[1]:0,
        bexp: std::mask_bits<EXP_SZ>() as bits[EXP_SZ],
        fraction: bits[FRACTION_SZ]:1 << ((FRACTION_SZ - u32:1) as bits[FRACTION_SZ]),
    }
}

#[test]
fn qnan_test() {
    let expected = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0xff, fraction: u23:0x400000 };
    let actual = qnan<u32:8, u32:23>();
    assert_eq(actual, expected);

    let expected = APFloat<u32:4, u32:2> { sign: u1:0, bexp: u4:0xf, fraction: u2:0x2 };
    let actual = qnan<u32:4, u32:2>();
    assert_eq(actual, expected);
}

// Returns whether or not the given APFloat represents NaN.
pub fn is_nan<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction != bits[FRACTION_SZ]:0)
}

// Returns a positive or a negative infinity depending upon the given sign parameter.
pub fn inf<EXP_SZ: u32, FRACTION_SZ: u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    APFloat<EXP_SZ, FRACTION_SZ> {
        sign,
        bexp: std::mask_bits<EXP_SZ>(),
        fraction: bits[FRACTION_SZ]:0,
    }
}

#[test]
fn inf_test() {
    let expected = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0xff, fraction: u23:0x0 };
    let actual = inf<u32:8, u32:23>(u1:0);
    assert_eq(actual, expected);

    let expected = APFloat<u32:4, u32:2> { sign: u1:0, bexp: u4:0xf, fraction: u2:0x0 };
    let actual = inf<u32:4, u32:2>(u1:0);
    assert_eq(actual, expected);
}

// Returns whether or not the given APFloat represents an infinite quantity.
pub fn is_inf<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction == bits[FRACTION_SZ]:0)
}

// Returns whether or not the given APFloat represents a positive infinite quantity.
pub fn is_pos_inf<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    is_inf(x) && !x.sign
}

// Returns whether or not the given APFloat represents a negative infinite quantity.
pub fn is_neg_inf<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    is_inf(x) && x.sign
}

// Returns the absolute value of x unless it is a NaN, in which case it will
// return a quiet NaN.
pub fn abs<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    APFloat { sign: u1:0, bexp: x.bexp, fraction: x.fraction }
}

#[test]
fn abs_test() {
    let expected = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x0 };
    let actual =
        abs<u32:8, u32:23>(APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0x7f, fraction: u23:0x0 });
    assert_eq(actual, expected);

    let expected = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0xff, fraction: u23:0x3645A2 };
    let actual = abs<u32:8, u32:23>(
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0xff, fraction: u23:0x3645A2 });
    assert_eq(actual, expected);
}

// Returns a positive or negative zero depending upon the given sign parameter.
pub fn zero<EXP_SZ: u32, FRACTION_SZ: u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    APFloat<EXP_SZ, FRACTION_SZ> { sign, bexp: bits[EXP_SZ]:0, fraction: bits[FRACTION_SZ]:0 }
}

#[test]
fn zero_test() {
    let expected = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x0, fraction: u23:0x0 };
    let actual = zero<u32:8, u32:23>(u1:0);
    assert_eq(actual, expected);

    let expected = APFloat<u32:4, u32:2> { sign: u1:1, bexp: u4:0x0, fraction: u2:0x0 };
    let actual = zero<u32:4, u32:2>(u1:1);
    assert_eq(actual, expected);
}

// Returns one or minus one depending upon the given sign parameter.
pub fn one<EXP_SZ: u32, FRACTION_SZ: u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    const MASK_SZ: u32 = EXP_SZ - u32:1;
    APFloat<EXP_SZ, FRACTION_SZ> {
        sign,
        bexp: std::mask_bits<MASK_SZ>() as bits[EXP_SZ],
        fraction: bits[FRACTION_SZ]:0,
    }
}

#[test]
fn one_test() {
    let expected = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x0 };
    let actual = one<u32:8, u32:23>(u1:0);
    assert_eq(actual, expected);

    let expected = APFloat<u32:4, u32:2> { sign: u1:0, bexp: u4:0x7, fraction: u2:0x0 };
    let actual = one<u32:4, u32:2>(u1:0);
    assert_eq(actual, expected);
}

// Returns the largest possible normal value of the APFloat type, or its negative, depending upon
// the given sign parameter.
pub fn max_normal<EXP_SZ: u32, FRACTION_SZ: u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    APFloat<EXP_SZ, FRACTION_SZ> {
        sign,
        bexp: (std::mask_bits<EXP_SZ>() << 1),
        fraction: std::mask_bits<FRACTION_SZ>(),
    }
}

#[test]
fn max_normal_test() {
    let expected = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fffff };
    let actual = max_normal<u32:8, u32:23>(u1:0);
    assert_eq(actual, expected);

    let expected = APFloat<u32:4, u32:2> { sign: u1:1, bexp: u4:0xe, fraction: u2:0x3 };
    let actual = max_normal<u32:4, u32:2>(u1:1);
    assert_eq(actual, expected);
}

// Returns the negative of x unless it is a NaN, in which case it will
// change it from a quiet to signaling NaN or from signaling to a quiet NaN.
pub fn negate<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    type Float = APFloat<EXP_SZ, FRACTION_SZ>;
    Float { sign: !x.sign, bexp: x.bexp, fraction: x.fraction }
}

// Maximum value of the exponent for normal numbers with EXP_SZ bits in the exponent field.
pub fn max_normal_exp<EXP_SZ: u32>() -> sN[EXP_SZ] {
    ((uN[EXP_SZ]:1 << (EXP_SZ - u32:1)) - uN[EXP_SZ]:1) as sN[EXP_SZ]
}

#[test]
fn test_max_normal_exp() {
    assert_eq(max_normal_exp<u32:8>(), s8:127);
    assert_eq(max_normal_exp<u32:11>(), s11:1023);
}

// Minimum value of the exponent for normal numbers with EXP_SZ bits in the exponent field.
pub fn min_normal_exp<EXP_SZ: u32>() -> sN[EXP_SZ] {
    let minus_min_normal_exp = ((uN[EXP_SZ]:1 << (EXP_SZ - u32:1)) - uN[EXP_SZ]:2) as sN[EXP_SZ];
    -minus_min_normal_exp
}

#[test]
fn test_min_normal_exp() {
    assert_eq(min_normal_exp<u32:8>(), s8:-126);
    assert_eq(min_normal_exp<u32:11>(), s11:-1022);
}

// Returns 2^(EXP_SZ-1)-1, which is the exponent bias used for encoding
// the given APFloat type. For example, this would return 127 for float32.
pub fn exponent_bias<EXP_SZ: u32, FRACTION_SZ: u32>(f: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[EXP_SZ] {
    std::signed_max_value<EXP_SZ>() as sN[EXP_SZ]
}

#[test]
fn exponent_bias_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let expected = s8:127;
    let input = one<F32_EXP_SZ, F32_FRACTION_SZ>(false);
    let actual = exponent_bias(input);
    assert_eq(actual, expected);

    const BF19_EXP_SZ = u32:8;
    const BF19_FRACTION_SZ = u32:10;
    let expected = s8:127;
    let input = one<BF19_EXP_SZ, BF19_FRACTION_SZ>(false);
    let actual = exponent_bias(input);
    assert_eq(actual, expected);

    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    let expected = s8:127;
    let input = one<BF16_EXP_SZ, BF16_FRACTION_SZ>(false);
    let actual = exponent_bias(input);
    assert_eq(actual, expected);

    const FP16_EXP_SZ = u32:5;
    const FP16_FRACTION_SZ = u32:10;
    let expected = s5:15;
    let input = one<FP16_EXP_SZ, FP16_FRACTION_SZ>(false);
    let actual = exponent_bias(input);
    assert_eq(actual, expected);
}

// Returns the unbiased exponent.
// For normal numbers it is `bexp - 2^(EXP_SZ - 1) + 1`.
// For zero and subnormals, it is `1 - 2^(EXP_SZ-1)`.
// For infinity and `NaN`, it is `-2^(EXP_SZ - 1)`.
//
// For example, for single precision IEEE numbers, the unbiased exponent is
// `bexp - 127`, for zero and subnormal numbers it is `-127`, and for infinity
// and `NaN` it is `-128`.
pub fn unbiased_exponent<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[EXP_SZ] {
    let bias = exponent_bias(f);
    (f.bexp as sN[EXP_SZ]) - bias
}

#[test]
fn unbiased_exponent_zero_test() {
    let expected = s8:-127;
    let actual = unbiased_exponent<u32:8, u32:23>(zero<u32:8, u32:23>(false));
    assert_eq(actual, expected);

    let actual = unbiased_exponent<u32:8, u32:23>(zero<u32:8, u32:23>(true));
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_one_test() {
    let expected = s8:0;
    let actual = unbiased_exponent<u32:8, u32:23>(one<u32:8, u32:23>(false));
    assert_eq(actual, expected);

    let actual = unbiased_exponent<u32:8, u32:23>(one<u32:8, u32:23>(true));
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_two_test() {
    let expected = s8:1;
    let two = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:128, fraction: u23:0 };
    let actual = unbiased_exponent<u32:8, u32:23>(two);
    assert_eq(actual, expected);

    let minus_two = APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:128, fraction: u23:0 };
    let actual = unbiased_exponent<u32:8, u32:23>(minus_two);
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_half_test() {
    let half = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:126, fraction: u23:0 };
    let expected = s8:-1;
    let actual = unbiased_exponent<u32:8, u32:23>(half);
    assert_eq(actual, expected);

    let minus_half = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:126, fraction: u23:0 };
    let actual = unbiased_exponent<u32:8, u32:23>(minus_half);
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_subnormal_test() {
    let expected = s8:-127;
    let actual = unbiased_exponent<u32:8, u32:23>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0, fraction: u23:42 });
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_inf_nan_test() {
    let expected = s8:-128;

    // inf
    let actual = unbiased_exponent(inf<u32:8, u32:23>(false));
    assert_eq(actual, expected);

    // -inf
    let actual = unbiased_exponent(inf<u32:8, u32:23>(true));
    assert_eq(actual, expected);

    // NaN
    let actual = unbiased_exponent(qnan<u32:8, u32:23>());
    assert_eq(actual, expected);
}

// Returns the biased exponent which is equal to `unbiased_exponent + 2^(EXP_SZ - 1)`
//
// Notice: Since the function only takes as input the unbiased exponent, it cannot
// distinguish between zero and subnormal numbers, or NaN and infinity, respectively.
pub fn bias<EXP_SZ: u32>(unbiased_exponent: sN[EXP_SZ]) -> uN[EXP_SZ] {
    const BIAS = std::signed_max_value<EXP_SZ>();
    (unbiased_exponent + BIAS) as uN[EXP_SZ]
}

#[test]
fn bias_test() {
    // Normal.
    let expected = u8:127;
    let actual = bias<u32:8>(s8:0);
    assert_eq(expected, actual);

    // Inf or NaN.
    let expected = u8:255;
    let actual = bias<u32:8>(s8:-128);
    assert_eq(expected, actual);

    // Zero or subnormal.
    let expected = u8:0;
    let actual = bias<u32:8>(s8:-127);
    assert_eq(expected, actual);
}

// Returns a bit string of size `1 + EXP_SZ + FRACTION_SZ` where the first bit
// is the sign bit, the next `EXP_SZ` bit encode the biased exponent and the
// last `FRACTION_SZ` are the significand without the hidden bit.
pub fn flatten<EXP_SZ: u32, FRACTION_SZ: u32, TOTAL_SZ: u32 = {u32:1 + EXP_SZ + FRACTION_SZ}>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> bits[TOTAL_SZ] {
    x.sign ++ x.bexp ++ x.fraction
}

// Returns a `APFloat` struct whose flattened version would be the input string.
pub fn unflatten<EXP_SZ: u32, FRACTION_SZ: u32, TOTAL_SZ: u32 = {u32:1 + EXP_SZ + FRACTION_SZ}>
    (x: bits[TOTAL_SZ]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    const SIGN_OFFSET: u32 = EXP_SZ + FRACTION_SZ;
    APFloat<EXP_SZ, FRACTION_SZ> {
        sign: (x >> (SIGN_OFFSET as bits[TOTAL_SZ])) as bits[1],
        bexp: (x >> (FRACTION_SZ as bits[TOTAL_SZ])) as bits[EXP_SZ],
        fraction: x as bits[FRACTION_SZ],
    }
}

// TODO(google/xls#1566): more apfloat functions should take additional configuration
pub enum RoundStyle : u1 {
    // The input is rounded to the closest number that can be represented by an integer. In the case
    // of an exact tie the input is rounded to the even integer.
    TIES_TO_EVEN = 0,
    // The input is rounded to the closest number that can be represented by an integer. In the case
    // of an exact tie the input is rounded to the number with the larger absolute value.
    TIES_TO_AWAY = 1,
}

// Round to nearest, ties controlled by round_style.
// if truncated bits > halfway bit: round up.
// if truncated bits < halfway bit: round down.
// if truncated bits == halfway bit:
//   if TIES_TO_EVEN:
//     if halfway bit is odd: round_up
//     if halfway bit is even: round_down
//   if TIES_TO_AWAY: round_up
fn does_lsb_round_up<FRACTION_SZ: u32>
    (lsb_index: u32, fraction: uN[FRACTION_SZ], round_style: RoundStyle) -> bool {
    // retained
    //    |
    //    |  /- round-bit
    //    |  |
    //    |  |  /- sticky
    // /----\|/--\
    // ABCDEFGHIJK
    assert!(lsb_index >= u32:1, "apfloat_round_without_residue");
    assert!(lsb_index <= FRACTION_SZ as u32, "apfloat_round_without_lsb");
    // Extract the first bit which needs to be cut.
    let first_lost_bit_idx = lsb_index - u32:1;
    let round_bit = fraction[first_lost_bit_idx+:u1];
    match round_style {
        RoundStyle::TIES_TO_EVEN => {
            // Extract the last bit which is retained.
            let lsb = lsb_index < FRACTION_SZ && fraction[lsb_index+:u1];
            // Whether any bits before the round_bit are 1.
            let sticky = std::or_reduce_lsb(fraction, lsb_index - u32:1);

            //  L R S
            //  X 0 X   --> Round down (less than half)
            //  0 1 0   --> Round down (half, already even)
            //  1 1 0   --> Round up (half, to even)
            //  X 1 1   --> Round up (greater than half)
            (round_bit && sticky) || (round_bit && lsb)
        },
        RoundStyle::TIES_TO_AWAY => { round_bit },
    }
}

#[test]
fn does_lsb_round_up_test() {
    // 0b1000.0
    assert_eq(does_lsb_round_up(u32:1, u5:0b10000, RoundStyle::TIES_TO_EVEN), false);
    assert_eq(does_lsb_round_up(u32:1, u5:0b10000, RoundStyle::TIES_TO_AWAY), false);
    // 0b1000.1
    assert_eq(does_lsb_round_up(u32:1, u5:0b10001, RoundStyle::TIES_TO_EVEN), false);
    assert_eq(does_lsb_round_up(u32:1, u5:0b10001, RoundStyle::TIES_TO_AWAY), true);
    // 0b1001.1
    assert_eq(does_lsb_round_up(u32:1, u5:0b10011, RoundStyle::TIES_TO_EVEN), true);
    assert_eq(does_lsb_round_up(u32:1, u5:0b10011, RoundStyle::TIES_TO_AWAY), true);
    // 0b100.11
    assert_eq(does_lsb_round_up(u32:2, u5:0b10011, RoundStyle::TIES_TO_EVEN), true);
    assert_eq(does_lsb_round_up(u32:2, u5:0b10011, RoundStyle::TIES_TO_AWAY), true);
    // 0b10.011
    assert_eq(does_lsb_round_up(u32:3, u5:0b10011, RoundStyle::TIES_TO_EVEN), false);
    assert_eq(does_lsb_round_up(u32:3, u5:0b10011, RoundStyle::TIES_TO_AWAY), false);
    // 0b.00011
    assert_eq(does_lsb_round_up(u32:5, u5:0b00011, RoundStyle::TIES_TO_EVEN), false);
    assert_eq(does_lsb_round_up(u32:5, u5:0b00011, RoundStyle::TIES_TO_AWAY), false);
    // 0b.10000
    assert_eq(does_lsb_round_up(u32:5, u5:0b10000, RoundStyle::TIES_TO_EVEN), false);
    assert_eq(does_lsb_round_up(u32:5, u5:0b10000, RoundStyle::TIES_TO_AWAY), true);
    // 0b.10011
    assert_eq(does_lsb_round_up(u32:5, u5:0b10011, RoundStyle::TIES_TO_EVEN), true);
    assert_eq(does_lsb_round_up(u32:5, u5:0b10011, RoundStyle::TIES_TO_AWAY), true);
}

#[test]
fn round_up_az_test() {
    // >halfway bit.
    assert_eq(does_lsb_round_up(u32:3, u5:0b01101, RoundStyle::TIES_TO_AWAY), true);
    // <halfway bit.
    assert_eq(does_lsb_round_up(u32:3, u5:0b01001, RoundStyle::TIES_TO_AWAY), false);
    // ==halfway bit and lsb odd.
    assert_eq(does_lsb_round_up(u32:3, u5:0b01100, RoundStyle::TIES_TO_AWAY), true);
    // ==halfway bit and lsb even.
    assert_eq(does_lsb_round_up(u32:3, u5:0b00100, RoundStyle::TIES_TO_AWAY), true);
    // 0 fraction.
    assert_eq(does_lsb_round_up(u32:3, u5:0b000000, RoundStyle::TIES_TO_AWAY), false);
    // max lsb index, >halfway bit.
    assert_eq(does_lsb_round_up(u32:0b111, u8:0b11000001, RoundStyle::TIES_TO_AWAY), true);
    // max lsb index, <halfway bit.
    assert_eq(does_lsb_round_up(u32:0b111, u8:0b10000000, RoundStyle::TIES_TO_AWAY), false);
    // max lsb index, ==halfway bit and lsb odd.
    assert_eq(does_lsb_round_up(u32:0b111, u8:0b11000000, RoundStyle::TIES_TO_AWAY), true);
    // max lsb index, ==halfway bit and lsb even.
    assert_eq(does_lsb_round_up(u32:0b111, u8:0b01000000, RoundStyle::TIES_TO_AWAY), true);
}

#[test]
fn round_up_ne_test() {
    // >halfway bit.
    assert_eq(does_lsb_round_up(u32:3, u5:0b01101, RoundStyle::TIES_TO_EVEN), true);
    // <halfway bit.
    assert_eq(does_lsb_round_up(u32:3, u5:0b01001, RoundStyle::TIES_TO_EVEN), false);
    // ==halfway bit and lsb odd.
    assert_eq(does_lsb_round_up(u32:3, u5:0b01100, RoundStyle::TIES_TO_EVEN), true);
    // ==halfway bit and lsb even.
    assert_eq(does_lsb_round_up(u32:3, u5:0b00100, RoundStyle::TIES_TO_EVEN), false);
    // 0 fraction.
    assert_eq(does_lsb_round_up(u32:3, u5:0b000000, RoundStyle::TIES_TO_EVEN), false);
    // max lsb index, >halfway bit.
    assert_eq(does_lsb_round_up(u32:0b111, u8:0b11000001, RoundStyle::TIES_TO_EVEN), true);
    // max lsb index, <halfway bit.
    assert_eq(does_lsb_round_up(u32:0b111, u8:0b10000000, RoundStyle::TIES_TO_EVEN), false);
    // max lsb index, ==halfway bit and lsb odd.
    assert_eq(does_lsb_round_up(u32:0b111, u8:0b11000000, RoundStyle::TIES_TO_EVEN), true);
    // max lsb index, ==halfway bit and lsb even.
    assert_eq(does_lsb_round_up(u32:0b111, u8:0b01000000, RoundStyle::TIES_TO_EVEN), false);
}

// Casts the fixed point number to a floating point number using RNE
// (Round to Nearest Even) as the rounding mode.
// TODO(google/xls#1656): Consolidate apfloat "round with ties to nearest even" logic
pub fn cast_from_fixed_using_rne<EXP_SZ: u32, FRACTION_SZ: u32, NUM_SRC_BITS: u32>
    (to_cast: sN[NUM_SRC_BITS]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    const UEXP_SZ: u32 = EXP_SZ + u32:1;
    const EXTENDED_FRACTION_SZ: u32 = FRACTION_SZ + NUM_SRC_BITS;

    // Determine sign.
    let is_negative = to_cast < sN[NUM_SRC_BITS]:0;

    // Determine exponent.
    let abs_magnitude = std::abs(to_cast) as uN[NUM_SRC_BITS];
    let lz = clz(abs_magnitude);
    let num_trailing_nonzeros = (NUM_SRC_BITS as uN[NUM_SRC_BITS]) - lz;

    // TODO(sameeragarwal): The following computation of exp can overflow if
    // num_trailing_nonzeros is larger than what uN[UEXP_SZ] can hold.
    let exp = (num_trailing_nonzeros as uN[UEXP_SZ]) - uN[UEXP_SZ]:1;
    let max_exp_exclusive = uN[UEXP_SZ]:1 << ((EXP_SZ as uN[UEXP_SZ]) - uN[UEXP_SZ]:1);
    let is_inf = exp >= max_exp_exclusive;
    let bexp = bias(exp as sN[EXP_SZ]);

    // Determine fraction (pre-rounding).
    //
    // TODO(sameeragarwal): This concatenation of FRACTION_SZ zeros followed by a shift
    // and slice seems excessive, worth exploring if the compiler is able to optimize it
    // or a smaller manual implementation will be more efficient.
    let extended_fraction = abs_magnitude ++ uN[FRACTION_SZ]:0;
    let fraction = extended_fraction >>
                   ((num_trailing_nonzeros - uN[NUM_SRC_BITS]:1) as uN[EXTENDED_FRACTION_SZ]);
    let fraction = fraction[0:FRACTION_SZ as s32];

    // Round fraction (round to nearest, half to even).
    let lsb_idx = (num_trailing_nonzeros as uN[EXTENDED_FRACTION_SZ]) - uN[EXTENDED_FRACTION_SZ]:1;
    let round_up = if lsb_idx == uN[EXTENDED_FRACTION_SZ]:0 {
        // already exact
        false
    } else if lsb_idx > EXTENDED_FRACTION_SZ as uN[EXTENDED_FRACTION_SZ] {
        // Overflowed lsb idx
        true
    } else {
        does_lsb_round_up(lsb_idx as u32, extended_fraction, RoundStyle::TIES_TO_EVEN)
    };
    let fraction = if round_up { fraction + uN[FRACTION_SZ]:1 } else { fraction };

    // Check if rounding up causes an exponent increment.
    let overflow = round_up && (fraction == uN[FRACTION_SZ]:0);
    let bexp = if overflow { (bexp + uN[EXP_SZ]:1) } else { bexp };

    // Check if rounding up caused us to overflow to infinity.
    let is_inf = is_inf || bexp == std::mask_bits<EXP_SZ>();
    let is_zero = abs_magnitude == uN[NUM_SRC_BITS]:0;
    match (is_inf, is_zero) {
        (true, false) => inf<EXP_SZ, FRACTION_SZ>(is_negative),
        // Technically is_inf and is_zero should never be true at the same time, however, currently
        // the way is_inf is computed it can be true while is_zero is true. So we allow is_inf to be
        // anything when deciding if the output should be zero or not.
        //
        // Further, input when zero does not have a sign, so we will always return +0.0
        (_, true) => zero<EXP_SZ, FRACTION_SZ>(false),
        (false, false) => APFloat<EXP_SZ, FRACTION_SZ> { sign: is_negative, bexp, fraction },
    }
}

#[test]
fn cast_from_fixed_using_rne_test() {
    // Zero is a special case.
    let zero_float = zero<u32:4, u32:4>(u1:0);
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:0), zero_float);

    // +/-1
    let one_float = one<u32:4, u32:4>(u1:0);
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:1), one_float);
    let none_float = one<u32:4, u32:4>(u1:1);
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:-1), none_float);

    // +/-4
    let four_float = APFloat<u32:4, u32:4> { sign: u1:0, bexp: u4:9, fraction: u4:0 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:4), four_float);
    let nfour_float = APFloat<u32:4, u32:4> { sign: u1:1, bexp: u4:9, fraction: u4:0 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:-4), nfour_float);

    // Cast maximum representable exponent in target format.
    let max_representable = APFloat<u32:4, u32:4> { sign: u1:0, bexp: u4:14, fraction: u4:0 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:128), max_representable);

    // Cast minimum non-representable exponent in target format.
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:256), inf<u32:4, u32:4>(u1:0));

    // Test rounding - maximum truncated bits that will round down, even fraction.
    let truncate = APFloat<u32:4, u32:4> { sign: u1:0, bexp: u4:14, fraction: u4:0 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:131), truncate);

    // Test rounding - maximum truncated bits that will round down, odd fraction.
    let truncate = APFloat<u32:4, u32:4> { sign: u1:0, bexp: u4:14, fraction: u4:1 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:139), truncate);

    // Test rounding - halfway and already even, round down
    let truncate = APFloat<u32:4, u32:4> { sign: u1:0, bexp: u4:14, fraction: u4:0 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:132), truncate);

    // Test rounding - halfway and odd, round up
    let round_up = APFloat<u32:4, u32:4> { sign: u1:0, bexp: u4:14, fraction: u4:2 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:140), round_up);

    // Test rounding - over halfway and even, round up
    let round_up = APFloat<u32:4, u32:4> { sign: u1:0, bexp: u4:14, fraction: u4:1 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:133), round_up);

    // Test rounding - over halfway and odd, round up
    let round_up = APFloat<u32:4, u32:4> { sign: u1:0, bexp: u4:14, fraction: u4:2 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:141), round_up);

    // Test rounding - Rounding up increases exponent.
    let round_inc_exponent = APFloat<u32:4, u32:4> { sign: u1:0, bexp: u4:14, fraction: u4:0 };
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:126), round_inc_exponent);
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:127), round_inc_exponent);

    // Test rounding - Rounding up overflows to infinity.
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:252), inf<u32:4, u32:4>(u1:0));
    assert_eq(cast_from_fixed_using_rne<u32:4, u32:4>(sN[32]:254), inf<u32:4, u32:4>(u1:0));
}

// Casts the fixed point number to a floating point number using RZ
// (Round to Zero) as the rounding mode.
pub fn cast_from_fixed_using_rz<EXP_SZ: u32, FRACTION_SZ: u32, NUM_SRC_BITS: u32>
    (to_cast: sN[NUM_SRC_BITS]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    const UEXP_SZ: u32 = EXP_SZ + u32:1;
    const EXTENDED_FRACTION_SZ: u32 = FRACTION_SZ + NUM_SRC_BITS;

    // Determine sign.
    let is_negative = to_cast < sN[NUM_SRC_BITS]:0;

    // Determine exponent.
    let abs_magnitude = std::abs(to_cast) as uN[NUM_SRC_BITS];
    let lz = clz(abs_magnitude);
    let num_trailing_nonzeros = (NUM_SRC_BITS as uN[NUM_SRC_BITS]) - lz;

    // TODO(sameeragarwal): The following computation of exp can overflow if
    // num_trailing_nonzeros is larger than what uN[UEXP_SZ] can hold.
    let exp = (num_trailing_nonzeros as uN[UEXP_SZ]) - uN[UEXP_SZ]:1;
    let max_exp_exclusive = uN[UEXP_SZ]:1 << ((EXP_SZ as uN[UEXP_SZ]) - uN[UEXP_SZ]:1);
    let is_inf = exp >= max_exp_exclusive;
    let bexp = bias(exp as sN[EXP_SZ]);

    // Determine fraction (pre-rounding).
    //
    // TODO(sameeragarwal): This concatenation of FRACTION_SZ zeros followed by a shift
    // and slice seems excessive, worth exploring if the compiler is able to optimize it
    // or a smaller manual implementation will be more efficient.
    let extended_fraction = abs_magnitude ++ uN[FRACTION_SZ]:0;
    let fraction = extended_fraction >>
                   ((num_trailing_nonzeros - uN[NUM_SRC_BITS]:1) as uN[EXTENDED_FRACTION_SZ]);
    let fraction = fraction[0:FRACTION_SZ as s32];

    let is_zero = abs_magnitude == uN[NUM_SRC_BITS]:0;

    match (is_inf, is_zero) {
        (true, false) => inf<EXP_SZ, FRACTION_SZ>(is_negative),
        // Technically is_inf and is_zero should never be true at the same time, however, currently
        // the way is_inf is computed it can be true while is_zero is true. So we allow is_inf to be
        // anything when deciding if the output should be zero or not.
        //
        // Further, input when zero does not have a sign, so we will always return +0.0
        (_, true) => zero<EXP_SZ, FRACTION_SZ>(false),
        (false, false) => APFloat<EXP_SZ, FRACTION_SZ> { sign: is_negative, bexp, fraction },
    }
}

#[test]
fn cast_from_fixed_using_rz_test() {
    const EXP_SZ = u32:5;
    const FRAC_SZ = u32:5;
    // This gives us the maximum number of bits before things go to infinity
    assert_eq(max_normal_exp<EXP_SZ>(), sN[EXP_SZ]:15);

    type Float = APFloat<EXP_SZ, FRAC_SZ>;
    assert_eq(cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0), zero<EXP_SZ, FRAC_SZ>(false));
    assert_eq(cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:1), one<EXP_SZ, FRAC_SZ>(false));
    assert_eq(cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:1), one<EXP_SZ, FRAC_SZ>(true));

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:2),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:1), fraction: uN[FRAC_SZ]:0 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:2),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:1), fraction: uN[FRAC_SZ]:0 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:3),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:1), fraction: uN[FRAC_SZ]:0b10000 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:3),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:1), fraction: uN[FRAC_SZ]:0b10000 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b111000),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:5), fraction: uN[FRAC_SZ]:0b11000 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b111000),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:5), fraction: uN[FRAC_SZ]:0b11000 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b1110000),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11000 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b1110000),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11000 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b111111),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:5), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b111111),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:5), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b1111110),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b1111110),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b1111111),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b1111111),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b01111111111111111),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:15), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b01111111111111111),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:15), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b00000011111111111),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:10), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b00000011111111111),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:10), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b00000011111111000),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:10), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b00000011111111000),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:10), fraction: uN[FRAC_SZ]:0b11111 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[20]:0b01000000000000000),
        Float { sign: false, bexp: bias(sN[EXP_SZ]:15), fraction: uN[FRAC_SZ]:0 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[20]:0b01000000000000000),
        Float { sign: true, bexp: bias(sN[EXP_SZ]:15), fraction: uN[FRAC_SZ]:0 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[20]:0b010000000000000000),
        inf<EXP_SZ, FRAC_SZ>(false));

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[20]:0b010000000000000000),
        inf<EXP_SZ, FRAC_SZ>(true));

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[20]:0b010000111000000000),
        inf<EXP_SZ, FRAC_SZ>(false));

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[20]:0b010001110000000000),
        inf<EXP_SZ, FRAC_SZ>(true));

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[30]:0b010000111000000000),
        inf<EXP_SZ, FRAC_SZ>(false));

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[30]:0b010001110000000000),
        inf<EXP_SZ, FRAC_SZ>(true));
}

// Returns true if x == 0 or x is a subnormal number.
pub fn is_zero_or_subnormal<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    x.bexp == uN[EXP_SZ]:0
}

pub fn subnormals_to_zero<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    if is_zero_or_subnormal(x) { zero<EXP_SZ, FRACTION_SZ>(x.sign) } else { x }
}

// Upcast the given apfloat to an apfloat with fraction and exponent size at least as large as the
// input's. Subnormal inputs are supported; they are converted to normal numbers of the same value
// when the exponent size is larger, or preserved otherwise.
pub fn upcast_with_denorms
    <TO_EXP_SZ: u32, TO_FRACTION_SZ: u32, FROM_EXP_SZ: u32, FROM_FRACTION_SZ: u32>
    (f: APFloat<FROM_EXP_SZ, FROM_FRACTION_SZ>) -> APFloat<TO_EXP_SZ, TO_FRACTION_SZ> {
    const IS_UPCAST = TO_EXP_SZ >= FROM_EXP_SZ && TO_FRACTION_SZ >= FROM_FRACTION_SZ;
    const SAME_SIZE = TO_EXP_SZ == FROM_EXP_SZ && TO_FRACTION_SZ == FROM_FRACTION_SZ;
    const_assert!(IS_UPCAST);

    // Force this function to reduce to either a trivial or real upcast depending on how it's
    // parameterized. The `match` in the `else` block would otherwise obscure the fact that it's a
    // no-op.
    if SAME_SIZE {
        // The reconstruction is necessary because it's invalid to explicitly just return `f`
        // without a `constexpr if`-type construct.
        APFloat {
            sign: f.sign,
            bexp: f.bexp as uN[TO_EXP_SZ],
            fraction: f.fraction as uN[TO_FRACTION_SZ],
        }
    } else {
        // use `sN+1` to preserve source bexp sign.
        const FROM_EXP_SZ_PLUS_1 = FROM_EXP_SZ + u32:1;
        type FromExpOffsetT = sN[FROM_EXP_SZ_PLUS_1];
        type ToExpOffsetT = sN[TO_EXP_SZ];
        // substract `2^(FROM_EXP_SZ-1) - 1` to retrieve the true exponent.
        const FROM_EXP_SZ_MINUS_1 = FROM_EXP_SZ - u32:1;
        const FROM_EXP_OFFSET = (FromExpOffsetT:1 << FROM_EXP_SZ_MINUS_1) - FromExpOffsetT:1;
        // add 2^(TO_EXP_SZ-1) - 1 to contruct back offset encoded exponent.
        const TO_EXP_SZ_MINUS_1 = TO_EXP_SZ - u32:1;
        const TO_EXP_OFFSET = (ToExpOffsetT:1 << TO_EXP_SZ_MINUS_1) - ToExpOffsetT:1;

        match tag(f) {
            APFloatTag::NAN => qnan<TO_EXP_SZ, TO_FRACTION_SZ>(),
            APFloatTag::INFINITY => inf<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign),
            APFloatTag::ZERO => zero<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign),
            APFloatTag::SUBNORMAL => {
                const FROM_TO_FRACTION_SHIFT = TO_FRACTION_SZ - FROM_FRACTION_SZ;
                let (to_bexp, to_fraction_shift) = if FROM_EXP_SZ == TO_EXP_SZ {
                    // still subnormal, just shift fraction.
                    (f.bexp as uN[TO_EXP_SZ], FROM_TO_FRACTION_SHIFT)
                } else {
                    // upcast subnormal to normal.
                    let subnormal_exp = (-FROM_EXP_OFFSET + FromExpOffsetT:1) as ToExpOffsetT;
                    let subnormal_to_normal_shift = clz(f.fraction) + uN[FROM_FRACTION_SZ]:1;
                    let upcasted_exp = subnormal_exp - (subnormal_to_normal_shift as ToExpOffsetT);
                    let to_bexp = (upcasted_exp as ToExpOffsetT + TO_EXP_OFFSET) as uN[TO_EXP_SZ];
                    (to_bexp, FROM_TO_FRACTION_SHIFT + subnormal_to_normal_shift as u32)
                };
                // shift fraction to destination size.
                let to_fraction = (f.fraction as uN[TO_FRACTION_SZ]) << to_fraction_shift;
                APFloat { sign: f.sign, bexp: to_bexp, fraction: to_fraction }
            },
            APFloatTag::NORMAL => {
                let from_exp = f.bexp as FromExpOffsetT - FROM_EXP_OFFSET;
                let to_bexp = (from_exp as ToExpOffsetT + TO_EXP_OFFSET) as uN[TO_EXP_SZ];
                // shift fraction to destination size.
                let FROM_TO_FRACTION_SHIFT = TO_FRACTION_SZ - FROM_FRACTION_SZ;
                let to_fraction = (f.fraction as uN[TO_FRACTION_SZ]) << FROM_TO_FRACTION_SHIFT;
                APFloat { sign: f.sign, bexp: to_bexp, fraction: to_fraction }
            },
        }
    }
}

#[test]
fn upcast_with_denorms_test() {
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    const F64_EXP_SZ = u32:11;
    const F64_FRACTION_SZ = u32:52;

    let one_bf16 = one<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let one_f64 = one<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0);
    let one_dot_5_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> {
        fraction: u7:1 << (BF16_FRACTION_SZ - u32:1),
        ..one_bf16
    };
    let one_dot_5_f64 = APFloat<F64_EXP_SZ, F64_FRACTION_SZ> {
        fraction: u52:1 << (F64_FRACTION_SZ - u32:1),
        ..one_f64
    };
    let zero_f64 = zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0);
    let neg_zero_f64 = zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1);
    let zero_bf16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let neg_zero_bf16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);

    assert_eq(upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(one_bf16), one_f64);
    assert_eq(upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(zero_bf16), zero_f64);
    assert_eq(upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(neg_zero_bf16), neg_zero_f64);
    assert_eq(upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(one_dot_5_bf16), one_dot_5_f64);
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(qnan<BF16_EXP_SZ, BF16_FRACTION_SZ>()),
        qnan<F64_EXP_SZ, F64_FRACTION_SZ>());
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0)),
        inf<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0));
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1)),
        inf<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1));
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0)),
        zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0));
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1)),
        zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1));

    // same size
    assert_eq(upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(zero_bf16), zero_bf16);
    assert_eq(upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(neg_zero_bf16), neg_zero_bf16);
    assert_eq(upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(one_bf16), one_bf16);
    assert_eq(upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(one_dot_5_bf16), one_dot_5_bf16);
}

#[test]
fn upcast_with_denorms_denormal_test() {
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    const F64_EXP_SZ = u32:11;
    const F64_FRACTION_SZ = u32:52;

    let denormal_bf16 = APFloat {
        sign: u1:0,
        bexp: uN[BF16_EXP_SZ]:0,
        fraction: uN[BF16_FRACTION_SZ]:1 << (BF16_FRACTION_SZ - u32:1) | uN[BF16_FRACTION_SZ]:1,
    };
    let neg_denormal_bf16 = APFloat { sign: u1:1, ..denormal_bf16 };
    assert_eq(upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(denormal_bf16), denormal_bf16);
    assert_eq(
        upcast_with_denorms<BF16_EXP_SZ, BF16_FRACTION_SZ>(neg_denormal_bf16), neg_denormal_bf16);
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(denormal_bf16),
        APFloat {
            sign: u1:0,
            bexp: uN[F64_EXP_SZ]:0x380,
            fraction: uN[F64_FRACTION_SZ]:0x400000000000,
        });
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(neg_denormal_bf16),
        APFloat {
            sign: u1:1,
            bexp: uN[F64_EXP_SZ]:0x380,
            fraction: uN[F64_FRACTION_SZ]:0x400000000000,
        });

    let min_plus_denormal_bf16 =
        APFloat { sign: u1:0, bexp: uN[BF16_EXP_SZ]:0, fraction: uN[BF16_FRACTION_SZ]:1 };

    let max_plus_denormal_bf16 = APFloat {
        sign: u1:0,
        bexp: uN[BF16_EXP_SZ]:0,
        fraction: std::mask_bits<BF16_FRACTION_SZ>(),
    };
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(min_plus_denormal_bf16),
        APFloat { sign: u1:0, bexp: uN[F64_EXP_SZ]:0x37a, fraction: uN[F64_FRACTION_SZ]:0 });
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, F64_FRACTION_SZ>(max_plus_denormal_bf16),
        APFloat {
            sign: u1:0,
            bexp: uN[F64_EXP_SZ]:0x380,
            fraction: uN[F64_FRACTION_SZ]:0xfc00000000000,
        });

    // same exponent size upcast.
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    assert_eq(
        upcast_with_denorms<F32_EXP_SZ, F32_FRACTION_SZ>(min_plus_denormal_bf16),
        APFloat { sign: u1:0, bexp: uN[F32_EXP_SZ]:0, fraction: uN[F32_FRACTION_SZ]:0x10000 });
    assert_eq(
        upcast_with_denorms<F32_EXP_SZ, F32_FRACTION_SZ>(max_plus_denormal_bf16),
        APFloat { sign: u1:0, bexp: uN[F32_EXP_SZ]:0, fraction: uN[F32_FRACTION_SZ]:0x7f0000 });

    // same fraction size upcast.
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, BF16_FRACTION_SZ>(min_plus_denormal_bf16),
        APFloat { sign: u1:0, bexp: uN[F64_EXP_SZ]:0x37a, fraction: uN[BF16_FRACTION_SZ]:0 });
    assert_eq(
        upcast_with_denorms<F64_EXP_SZ, BF16_FRACTION_SZ>(max_plus_denormal_bf16),
        APFloat { sign: u1:0, bexp: uN[F64_EXP_SZ]:0x380, fraction: uN[BF16_FRACTION_SZ]:0x7e });
}

// Upcast the given apfloat to an apfloat with fraction and exponent size at least as large as the
// input's. Note: denormal inputs get flushed to zero.
pub fn upcast_daz<TO_EXP_SZ: u32, TO_FRACTION_SZ: u32, FROM_EXP_SZ: u32, FROM_FRACTION_SZ: u32>
    (f: APFloat<FROM_EXP_SZ, FROM_FRACTION_SZ>) -> APFloat<TO_EXP_SZ, TO_FRACTION_SZ> {
    upcast_with_denorms<TO_EXP_SZ, TO_FRACTION_SZ>(subnormals_to_zero(f))
}

#[test]
fn upcast_daz_test() {
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    const F64_EXP_SZ = u32:11;
    const F64_FRACTION_SZ = u32:52;

    let one_bf16 = one<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let one_f64 = one<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0);
    let one_dot_5_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> {
        fraction: u7:1 << (BF16_FRACTION_SZ - u32:1),
        ..one_bf16
    };
    let one_dot_5_f64 = APFloat<F64_EXP_SZ, F64_FRACTION_SZ> {
        fraction: u52:1 << (F64_FRACTION_SZ - u32:1),
        ..one_f64
    };
    let zero_f64 = zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0);
    let neg_zero_f64 = zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1);
    let zero_bf16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let neg_zero_bf16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);

    assert_eq(upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(one_bf16), one_f64);
    assert_eq(upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(zero_bf16), zero_f64);
    assert_eq(upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(neg_zero_bf16), neg_zero_f64);
    assert_eq(upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(one_dot_5_bf16), one_dot_5_f64);
    assert_eq(
        upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(qnan<BF16_EXP_SZ, BF16_FRACTION_SZ>()),
        qnan<F64_EXP_SZ, F64_FRACTION_SZ>());
    assert_eq(
        upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0)),
        inf<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0));
    assert_eq(
        upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1)),
        inf<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1));
    assert_eq(
        upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0)),
        zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0));
    assert_eq(
        upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1)),
        zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1));

    // same size
    assert_eq(upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(zero_bf16), zero_bf16);
    assert_eq(upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(neg_zero_bf16), neg_zero_bf16);
    assert_eq(upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(one_bf16), one_bf16);
    assert_eq(upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(one_dot_5_bf16), one_dot_5_bf16);
}

#[test]
fn upcast_daz_denormal_test() {
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;
    const F64_EXP_SZ = u32:11;
    const F64_FRACTION_SZ = u32:52;

    let zero_bf16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let neg_zero_bf16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);
    let zero_f64 = zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0);
    let neg_zero_f64 = zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1);

    let denormal_bf16 = APFloat {
        sign: u1:0,
        bexp: uN[BF16_EXP_SZ]:0,
        fraction: uN[BF16_FRACTION_SZ]:1 << (BF16_FRACTION_SZ - u32:1) | uN[BF16_FRACTION_SZ]:1,
    };
    let neg_denormal_bf16 = APFloat { sign: u1:1, ..denormal_bf16 };
    assert_eq(upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(denormal_bf16), zero_bf16);
    assert_eq(upcast_daz<BF16_EXP_SZ, BF16_FRACTION_SZ>(neg_denormal_bf16), neg_zero_bf16);
    assert_eq(upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(denormal_bf16), zero_f64);
    assert_eq(upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(neg_denormal_bf16), neg_zero_f64);

    let min_plus_denormal_bf16 =
        APFloat { sign: u1:0, bexp: uN[BF16_EXP_SZ]:0, fraction: uN[BF16_FRACTION_SZ]:1 };

    let max_plus_denormal_bf16 = APFloat {
        sign: u1:0,
        bexp: uN[BF16_EXP_SZ]:0,
        fraction: std::mask_bits<BF16_FRACTION_SZ>(),
    };
    assert_eq(upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(min_plus_denormal_bf16), zero_f64);
    assert_eq(upcast_daz<F64_EXP_SZ, F64_FRACTION_SZ>(max_plus_denormal_bf16), zero_f64);

    // same exponent size upcast.
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    assert_eq(upcast_daz<F32_EXP_SZ, F32_FRACTION_SZ>(min_plus_denormal_bf16), zero_f32);
    assert_eq(upcast_daz<F32_EXP_SZ, F32_FRACTION_SZ>(max_plus_denormal_bf16), zero_f32);

    // same fraction size upcast.
    assert_eq(
        upcast_daz<F64_EXP_SZ, BF16_FRACTION_SZ>(min_plus_denormal_bf16),
        zero<F64_EXP_SZ, BF16_FRACTION_SZ>(u1:0));
    assert_eq(
        upcast_daz<F64_EXP_SZ, BF16_FRACTION_SZ>(max_plus_denormal_bf16),
        zero<F64_EXP_SZ, BF16_FRACTION_SZ>(u1:0));
}

// Rounds a normal apfloat to lower precision in fractional bits, while the
// exponent size remains fixed. Ties round based on RoundStyle. The result is
// undefined for subnormals, NaN and infinity.
fn downcast_fractional<TO_FRACTION_SZ: u32, FROM_FRACTION_SZ: u32, EXP_SZ: u32>
    (f: APFloat<EXP_SZ, FROM_FRACTION_SZ>, round_style: RoundStyle)
    -> APFloat<EXP_SZ, TO_FRACTION_SZ> {
    const_assert!(FROM_FRACTION_SZ >= TO_FRACTION_SZ);

    let lsb_index = FROM_FRACTION_SZ - TO_FRACTION_SZ;
    let truncated_fraction = f.fraction[lsb_index as s32:FROM_FRACTION_SZ as s32];

    let round_up = does_lsb_round_up(lsb_index, f.fraction, round_style);

    let renormalize = round_up && and_reduce(truncated_fraction);

    // bexp: If the fraction rolled over when rounded up, then need to increment the exponent.
    // Rollover from the largest representable value will naturally go to infinity as desired.

    APFloat {
        sign: f.sign,
        bexp: if renormalize { f.bexp + uN[EXP_SZ]:1 } else { f.bexp },
        fraction: if round_up {
            truncated_fraction + uN[TO_FRACTION_SZ]:1
        } else {
            truncated_fraction
        },
    }
}

#[test]
fn downcast_fractional_raz_fp32_to_bf16_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;

    // normals
    let one_bf16 = one<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    assert_eq(downcast_fractional<BF16_FRACTION_SZ>(one_f32, RoundStyle::TIES_TO_AWAY), one_bf16);

    let minus_one_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:1, ..one_bf16 };
    let minus_one_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..one_f32 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(minus_one_f32, RoundStyle::TIES_TO_AWAY),
        minus_one_bf16);

    // fraction with no rounding necessary
    let one_dot_5_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: u23:0x400000, ..one_f32 };
    let one_dot_5_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { fraction: u7:0x40, ..one_bf16 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(one_dot_5_f32, RoundStyle::TIES_TO_AWAY),
        one_dot_5_bf16);

    // rounds down
    let pi_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x490fdb };
    let pi_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u7:0x49 };
    assert_eq(downcast_fractional<BF16_FRACTION_SZ>(pi_f32, RoundStyle::TIES_TO_AWAY), pi_bf16);

    // rounds up
    let one_third_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x7d, fraction: u23:0x2aaaab };
    let one_third_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x7d, fraction: u7:0x2b };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(one_third_f32, RoundStyle::TIES_TO_AWAY),
        one_third_bf16);

    // rounds up, tie to away
    let rne_up_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x408000 };
    let rne_up_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u7:0x41 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(rne_up_f32, RoundStyle::TIES_TO_AWAY), rne_up_bf16);

    // rounds up, tie to away
    let rne_up_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x418000 };
    let rne_up_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u7:0x42 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(rne_up_f32, RoundStyle::TIES_TO_AWAY), rne_up_bf16);

    // round up to inf
    let inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let just_above_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fe000 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(just_above_max_normal_bf16, RoundStyle::TIES_TO_AWAY),
        inf_bf16);

    let just_below_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x7fe000 };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(just_below_max_normal_bf16, RoundStyle::TIES_TO_AWAY),
        minus_inf_bf16);

    // round up to inf
    let max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fffff };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(max_normal_f32, RoundStyle::TIES_TO_AWAY), inf_bf16);

    let minus_max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x7fffff };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(minus_max_normal_f32, RoundStyle::TIES_TO_AWAY),
        minus_inf_bf16);
}

#[test]
fn downcast_fractional_rne_fp32_to_bf16_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;

    // normals
    let one_bf16 = one<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    assert_eq(downcast_fractional<BF16_FRACTION_SZ>(one_f32, RoundStyle::TIES_TO_EVEN), one_bf16);

    let minus_one_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:1, ..one_bf16 };
    let minus_one_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..one_f32 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(minus_one_f32, RoundStyle::TIES_TO_EVEN),
        minus_one_bf16);

    // fraction with no rounding necessary
    let one_dot_5_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: u23:0x400000, ..one_f32 };
    let one_dot_5_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { fraction: u7:0x40, ..one_bf16 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(one_dot_5_f32, RoundStyle::TIES_TO_EVEN),
        one_dot_5_bf16);

    // rounds down
    let pi_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x490fdb };
    let pi_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u7:0x49 };
    assert_eq(downcast_fractional<BF16_FRACTION_SZ>(pi_f32, RoundStyle::TIES_TO_EVEN), pi_bf16);

    // rounds up
    let one_third_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x7d, fraction: u23:0x2aaaab };
    let one_third_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x7d, fraction: u7:0x2b };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(one_third_f32, RoundStyle::TIES_TO_EVEN),
        one_third_bf16);

    // rounds down, tie to even
    let rne_down_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x408000 };
    let rne_down_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u7:0x40 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(rne_down_f32, RoundStyle::TIES_TO_EVEN), rne_down_bf16);

    // rounds up, tie to even
    let rne_up_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x418000 };
    let rne_up_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u7:0x42 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(rne_up_f32, RoundStyle::TIES_TO_EVEN), rne_up_bf16);

    // round up to inf
    let inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let just_above_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fe000 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(just_above_max_normal_bf16, RoundStyle::TIES_TO_EVEN),
        inf_bf16);

    let just_below_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x7fe000 };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(just_below_max_normal_bf16, RoundStyle::TIES_TO_EVEN),
        minus_inf_bf16);

    // round up to inf
    let max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fffff };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(max_normal_f32, RoundStyle::TIES_TO_EVEN), inf_bf16);

    let minus_max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x7fffff };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(minus_max_normal_f32, RoundStyle::TIES_TO_EVEN),
        minus_inf_bf16);
}

// Perform downcasting that converts a normal number into a subnormal. f must
// have an unbiased exponent less than the minumum normal exponent of the target
// float type (checked via assert!).
fn downcast_to_subnormal
    <TO_FRACTION_SZ: u32, TO_EXP_SZ: u32, FROM_FRACTION_SZ: u32, FROM_EXP_SZ: u32>
    (f: APFloat<FROM_EXP_SZ, FROM_FRACTION_SZ>, round_style: RoundStyle)
    -> APFloat<TO_EXP_SZ, TO_FRACTION_SZ> {
    const TO_BIAS = std::signed_max_value<TO_EXP_SZ>() as sN[FROM_EXP_SZ];
    let uexp = unbiased_exponent(f);
    // Check for over- and underflow of the exponent in the target type.
    assert!(
        uexp < (min_normal_exp<TO_EXP_SZ>() as sN[FROM_EXP_SZ]),
        "apfloat_downcast_to_subnormal_called_on_normal_number");

    // either a zero or a subnormal.
    // 32 bits is more than large enough for any reasonable
    // floating point numbers exponent. Narrowing should crush it down.
    let right_shift_cnt = -(TO_BIAS + uexp) as u32 + u32:1;
    if right_shift_cnt > (TO_FRACTION_SZ + u32:1) {
        // actually underflows
        zero<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign)
    } else {
        const SMALL_FRAC_OFF = FROM_FRACTION_SZ - TO_FRACTION_SZ;
        // Truncate the trailing bits of the fraction.
        let truncated_frac = f.fraction[SMALL_FRAC_OFF+:uN[TO_FRACTION_SZ]];
        // Add the implied leading 1
        let full_frac = u1:0b1 ++ truncated_frac;
        // Shift the bits over.
        let unrounded_subnormal_frac = full_frac[right_shift_cnt+:uN[TO_FRACTION_SZ]];

        let round_up = does_lsb_round_up(
            right_shift_cnt as u32 + SMALL_FRAC_OFF, u1:1 ++ f.fraction, round_style);

        let subnormal_frac = if round_up {
            unrounded_subnormal_frac + uN[TO_FRACTION_SZ]:1
        } else {
            unrounded_subnormal_frac
        };

        // Technically the subnormal frac is good enough but this is
        // easier to see through.
        let rounds_to_normal = right_shift_cnt == u32:1 && subnormal_frac == uN[TO_FRACTION_SZ]:0;

        if rounds_to_normal {
            APFloat { sign: f.sign, bexp: uN[TO_EXP_SZ]:1, fraction: uN[TO_FRACTION_SZ]:0 }
        } else {
            APFloat { sign: f.sign, bexp: uN[TO_EXP_SZ]:0, fraction: subnormal_frac }
        }
    }
}

// Round the apfloat to an apfloat with smaller fraction and/or smaller exponent
// size. Ties round to 'round_style'. Values with exponents above the target
// exponent range will overflow to infinity. Values with exponents below the
// target exponent range will result in subnormal result or zero.
//
// f must be normal.
fn downcast_normal<TO_FRACTION_SZ: u32, TO_EXP_SZ: u32, FROM_FRACTION_SZ: u32, FROM_EXP_SZ: u32>
    (f: APFloat<FROM_EXP_SZ, FROM_FRACTION_SZ>, round_style: RoundStyle)
    -> APFloat<TO_EXP_SZ, TO_FRACTION_SZ> {
    assert!(tag(f) == APFloatTag::NORMAL, "apfloat_downcast_normal_called_on_non_normal");
    // Downcast the fraction. This may increment the exponent.
    let f_cast = if FROM_FRACTION_SZ > TO_FRACTION_SZ {
        downcast_fractional<TO_FRACTION_SZ>(f, round_style)
    } else {
        APFloat { sign: f.sign, bexp: f.bexp, fraction: f.fraction as uN[TO_FRACTION_SZ] }
    };
    const INF_EXP = std::unsigned_max_value<FROM_EXP_SZ>();
    const CAN_GENERATE_SUBNORMALS = FROM_EXP_SZ > TO_EXP_SZ;
    // Check for overflow to infinity due to rounding the fractional part up.
    if FROM_FRACTION_SZ > TO_FRACTION_SZ && f_cast.bexp == INF_EXP {
        inf<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign)
    } else {
        // Check for over- and underflow of the exponent in the target type.
        const TO_BIAS = std::signed_max_value<TO_EXP_SZ>() as sN[FROM_EXP_SZ];
        let uexp = unbiased_exponent<FROM_EXP_SZ, TO_FRACTION_SZ>(f_cast);
        if CAN_GENERATE_SUBNORMALS && uexp > TO_BIAS {
            // NB In the no-subnormals case the fraction/bias is already inf.
            inf<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign)
        } else if CAN_GENERATE_SUBNORMALS && uexp <= -TO_BIAS {
            downcast_to_subnormal<TO_FRACTION_SZ, TO_EXP_SZ>(f, round_style)
        } else {
            APFloat {
                sign: f_cast.sign,
                bexp: bias(uexp as sN[TO_EXP_SZ]),
                fraction: f_cast.fraction,
            }
        }
    }
}

// Round the apfloat to an apfloat with smaller fraction and/or smaller exponent
// size. Ties round to 'round_style'. Values with exponents above the target
// exponent range will overflow to infinity. Values with exponents below the
// target exponent range will result in subnormal result or zero.
// Subnormal inputs are flushed to zero before conversion.
pub fn downcast<TO_FRACTION_SZ: u32, TO_EXP_SZ: u32, FROM_FRACTION_SZ: u32, FROM_EXP_SZ: u32>
    (f: APFloat<FROM_EXP_SZ, FROM_FRACTION_SZ>, round_style: RoundStyle)
    -> APFloat<TO_EXP_SZ, TO_FRACTION_SZ> {
    const_assert!(FROM_EXP_SZ >= TO_EXP_SZ);
    const_assert!(FROM_FRACTION_SZ >= TO_FRACTION_SZ);
    // Guard against using this method on identical types.
    const_assert!((FROM_FRACTION_SZ > TO_FRACTION_SZ) || (FROM_EXP_SZ > TO_EXP_SZ));

    match tag(f) {
        APFloatTag::NAN => qnan<TO_EXP_SZ, TO_FRACTION_SZ>(),
        APFloatTag::INFINITY => inf<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign),
        APFloatTag::ZERO => zero<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign),
        APFloatTag::SUBNORMAL => {
            // TODO(allight): We could try to convert a subnormal to a subnormal but its not clear
            // if this would ever actually be required with reasonable bit-widths.
            zero<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign)
        },
        APFloatTag::NORMAL => downcast_normal<TO_FRACTION_SZ, TO_EXP_SZ>(f, round_style),
    }
}

#[quickcheck]
fn does_downcast_stay_within_one_ulp_rounded(f: APFloat<u32:8, u32:13>) -> bool {
    type HF16 = APFloat<u32:5, u32:10>;
    let to_away = downcast<HF16::FRACTION_SIZE, HF16::EXP_SIZE>(f, RoundStyle::TIES_TO_AWAY);
    let to_even = downcast<HF16::FRACTION_SIZE, HF16::EXP_SIZE>(f, RoundStyle::TIES_TO_EVEN);

    let one_ulp_away = if is_nan(to_away) {
        !is_nan(to_even)
    } else if is_inf(to_away) {
        // to-away rounded up to infinity but even didn't
        unbiased_exponent(to_even) == max_normal_exp<HF16::EXP_SIZE>() &&
        to_even.fraction == all_ones!<u10>()
    } else if to_even.bexp == u5:0 && to_even.fraction == u10:0 {
        // to_even rounded down to zero but to_away rounded away from it.
        to_away.bexp == u5:0 && to_away.fraction == u10:1
    } else if !is_zero_or_subnormal(to_away) && is_zero_or_subnormal(to_even) {
        // to-away rounded up to normal but even didn't
        to_away.bexp == u5:1 && to_away.fraction == u10:0 && to_even.fraction == all_ones!<u10>()
    } else if to_even.bexp != to_away.bexp {
        // to-away rounded up but to_even rounded down
        to_even.bexp + u5:1 == to_away.bexp && to_even.fraction == all_ones!<u10>() &&
        to_away.fraction == u10:0
    } else {
        // subnormals different rounding
        to_even.fraction + u10:1 == to_away.fraction
    };
    to_away.sign == to_even.sign && (to_away == to_even || one_ulp_away)
}

#[test]
fn downcast_special() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const FP16_EXP_SZ = u32:5;
    const FP16_FRACTION_SZ = u32:10;

    // qnan -> qnan
    let qnan_f32 = qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    let qnan_fp16 = qnan<FP16_EXP_SZ, FP16_FRACTION_SZ>();
    assert_eq(
        downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(qnan_f32, RoundStyle::TIES_TO_EVEN), qnan_fp16);
    assert_eq(
        downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(qnan_f32, RoundStyle::TIES_TO_AWAY), qnan_fp16);

    // inf -> inf
    let inf_f32 = inf<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let inf_fp16 = inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:0);
    assert_eq(downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(inf_f32, RoundStyle::TIES_TO_EVEN), inf_fp16);
    assert_eq(downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(inf_f32, RoundStyle::TIES_TO_AWAY), inf_fp16);

    let minus_inf_f32 = inf<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    let minus_inf_fp16 = inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:1);
    assert_eq(
        downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_inf_f32, RoundStyle::TIES_TO_EVEN),
        minus_inf_fp16);
    assert_eq(
        downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_inf_f32, RoundStyle::TIES_TO_AWAY),
        minus_inf_fp16);

    // +/- 0.0 -> same-signed 0
    let zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let zero_fp16 = zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:0);
    assert_eq(
        downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(zero_f32, RoundStyle::TIES_TO_EVEN), zero_fp16);
    assert_eq(
        downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(zero_f32, RoundStyle::TIES_TO_AWAY), zero_fp16);

    let minus_zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    let minus_zero_fp16 = zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:1);
    assert_eq(
        downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_zero_f32, RoundStyle::TIES_TO_EVEN),
        minus_zero_fp16);
    assert_eq(
        downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_zero_f32, RoundStyle::TIES_TO_AWAY),
        minus_zero_fp16);

    // stays normalized
    type F32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ>;
    const SMALLER_EXP_SZ = F32_EXP_SZ - u32:4;
    type Smaller = APFloat<SMALLER_EXP_SZ, F32_FRACTION_SZ>;
    let f = F32 {
        sign: u1:0,
        bexp: bias<F32_EXP_SZ>(min_normal_exp<SMALLER_EXP_SZ>() as s8),
        fraction: u23:0x123,
    };
    let exp = Smaller { sign: u1:0, bexp: u4:1, fraction: u23:0x123 };
    assert_eq(downcast<F32_FRACTION_SZ, SMALLER_EXP_SZ>(f, RoundStyle::TIES_TO_EVEN), exp);
    assert_eq(downcast<F32_FRACTION_SZ, SMALLER_EXP_SZ>(f, RoundStyle::TIES_TO_AWAY), exp);
}

#[test]
fn downcast_generates_subnormal() {
    // unbiased exponent is -1, 0, inf, or subnormal/zero
    type E2F5 = APFloat<u32:2, u32:5>;
    type E4F6 = APFloat<u32:4, u32:6>;

    // binary float 0b0.001000
    let not_subnormal = E4F6 {
        sign: false,
        bexp: bias<E4F6::EXP_SIZE>(min_normal_exp<E2F5::EXP_SIZE>() as s4 - s4:3),
        fraction: u6:0b000000,
    };
    let expected = E2F5 { sign: false, bexp: u2:0, fraction: u5:0b00100 };
    assert_eq(downcast<u32:5, u32:2>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:5, u32:2>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);

    // Check rounding occurs.
    // slightly larger ranges
    type E6F8 = APFloat<u32:6, u32:8>;
    // Middle.
    let not_subnormal = E6F8 {
        sign: false,
        bexp: bias<E6F8::EXP_SIZE>(min_normal_exp<E4F6::EXP_SIZE>() as s6 - s6:3),
        // 0b1.0101_0000
        fraction: u8:0b0101_0000,
    };
    // 0b0.001010[10000]
    let expected_to_even = E4F6 { sign: false, bexp: u4:0, fraction: u6:0b00_1010 };
    let expected_to_away = E4F6 { sign: false, bexp: u4:0, fraction: u6:0b00_1011 };
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected_to_even);
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected_to_away);

    // always down.
    let not_subnormal = E6F8 {
        sign: false,
        bexp: bias<E6F8::EXP_SIZE>(min_normal_exp<E4F6::EXP_SIZE>() as s6 - s6:4),
        // 0b1.0101_0000
        fraction: u8:0b0101_0000,
    };
    // 0b0.000101[010000]
    let expected = E4F6 { sign: false, bexp: u4:0, fraction: u6:0b00_0101 };
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);

    // always up.
    let not_subnormal = E6F8 {
        sign: false,
        bexp: bias<E6F8::EXP_SIZE>(min_normal_exp<E4F6::EXP_SIZE>() as s6 - s6:5),
        // 0b1.0101_0000
        fraction: u8:0b0101_0000,
    };
    // 0b0.000010[1010000]
    let expected = E4F6 { sign: false, bexp: u4:0, fraction: u6:0b00_0011 };
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);

    // always up all the way off.
    let not_subnormal = E6F8 {
        sign: false,
        bexp: bias<E6F8::EXP_SIZE>(min_normal_exp<E4F6::EXP_SIZE>() as s6 - s6:7),
        // 0b1.0101_0000
        fraction: u8:0b0101_0000,
    };
    // 0b0.000000[101010000]
    let expected = E4F6 { sign: false, bexp: u4:0, fraction: u6:0b00_0001 };
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);

    type F32 = APFloat<u32:8, u32:23>;
    type HF16 = APFloat<u32:5, u32:10>;
    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:1),
        fraction: u23:0,
    };
    let expected = HF16 { sign: false, bexp: u5:0, fraction: u10:0b10_0000_0000 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:2),
        fraction: u23:0,
    };
    let expected = HF16 { sign: false, bexp: u5:0, fraction: u10:0b01_0000_0000 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:3),
        fraction: u23:0,
    };
    let expected = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1000_0000 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:4),
        fraction: u23:0,
    };
    let expected = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_0100_0000 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:10),
        fraction: u23:0,
    };
    let expected = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_0000_0001 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);

    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:11),
        fraction: u23:0,
    };
    let expected_away = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_0000_0001 };
    let expected_even = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_0000_0000 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected_away);

    // Rounds up
    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:3),
        fraction: u10:0b00_0000_0110 ++ u13:0,
    };
    let expected_away = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1000_0001 };
    let expected_even = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1000_0001 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected_away);

    // Rounds down
    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:3),
        fraction: u10:0b00_0000_0011 ++ u13:0,
    };
    let expected_away = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1000_0000 };
    let expected_even = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1000_0000 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected_away);

    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:3),
        fraction: u10:0b00_0000_0100 ++ u13:0,
    };
    let expected_away = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1000_0001 };
    let expected_even = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1000_0000 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected_away);

    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:3),
        fraction: u10:0b00_0000_1100 ++ u13:0,
    };
    let expected_away = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1000_0010 };
    let expected_even = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1000_0010 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected_away);

    let not_subnormal = F32 {
        sign: false,
        bexp: bias<u32:8>(min_normal_exp<HF16::EXP_SIZE>() as s8 - s8:3),
        fraction: u23:0b100_1001_0111_1011_1110_1001,
    };
    let expected_away = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1100_1001 };
    let expected_even = HF16 { sign: false, bexp: u5:0, fraction: u10:0b00_1100_1001 };
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected_even);
    assert_eq(downcast<u32:10, u32:5>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected_away);

    type F64 = APFloat<u32:11, u32:52>;
    let not_subnormal = F64 {
        sign: false,
        bexp: bias<F64::EXP_SIZE>(min_normal_exp<F32::EXP_SIZE>() as s11 - s11:3),
        fraction: u52:0b1000_1000_0110_0001_0010_1001_1010_0000_0001_0010_1100_1010_0001,
    };
    let expected = F32 { sign: false, bexp: u8:0, fraction: u23:0b001_1000_1000_0110_0001_0011 };
    assert_eq(downcast<u32:23, u32:8>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:23, u32:8>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);
}

#[test]
fn downcast_rounds_subnormal_to_normal() {
    type E4F6 = APFloat<u32:4, u32:6>;
    type E6F8 = APFloat<u32:6, u32:8>;

    // binary float 0b0.000000000000111111111 = 0b1.11111111 * 2**-7
    let not_subnormal = E6F8 { sign: false, bexp: bias<u32:6>(s6:-7), fraction: u8:0b1111_1111 };

    // ends up with a subnormal with 3 1s off the end of the mantisa and all
    // other bits 1. This rounds up to pushing it out of the subnormal regime.
    let expected = E4F6 { sign: false, bexp: bias<u32:4>(s4:-6), fraction: u6:0 };
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_EVEN), expected);
    assert_eq(downcast<u32:6, u32:4>(not_subnormal, RoundStyle::TIES_TO_AWAY), expected);
}

// Round the apfloat to an apfloat with smaller fraction and/or smaller exponent size.
// Ties round to even (LSB = 0) and denormal inputs are treated as zero and denormal outputs get
// flushed to zero. Values with exponents above the target exponent range will
// overflow to infinity. Values with exponents below the target exponent range
// will underflow to zero. Can round to infinity.
// Deprecated: Use 'downcast' instead.
pub fn downcast_rne<TO_FRACTION_SZ: u32, TO_EXP_SZ: u32, FROM_FRACTION_SZ: u32, FROM_EXP_SZ: u32>
    (f: APFloat<FROM_EXP_SZ, FROM_FRACTION_SZ>) -> APFloat<TO_EXP_SZ, TO_FRACTION_SZ> {
    subnormals_to_zero(downcast<TO_FRACTION_SZ, TO_EXP_SZ>(f, RoundStyle::TIES_TO_EVEN))
}

#[test]
fn downcast_raz_fp32_to_fp16_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const FP16_EXP_SZ = u32:5;
    const FP16_FRACTION_SZ = u32:10;

    // qnan -> qnan
    let qnan_f32 = qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    let qnan_fp16 = qnan<FP16_EXP_SZ, FP16_FRACTION_SZ>();
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(qnan_f32, RoundStyle::TIES_TO_AWAY)), qnan_fp16);

    // inf -> inf
    let inf_f32 = inf<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let inf_fp16 = inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:0);
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(inf_f32, RoundStyle::TIES_TO_AWAY)), inf_fp16);

    let minus_inf_f32 = inf<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    let minus_inf_fp16 = inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:1);
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_inf_f32, RoundStyle::TIES_TO_AWAY)),
        minus_inf_fp16);

    // +/- 0.0 -> same-signed 0
    let zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let zero_fp16 = zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:0);
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(zero_f32, RoundStyle::TIES_TO_AWAY)), zero_fp16);

    let minus_zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    let minus_zero_fp16 = zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:1);
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_zero_f32, RoundStyle::TIES_TO_AWAY)),
        minus_zero_fp16);

    // subnormals flushed to same-signed 0.0
    let subnormal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0, fraction: u23:0x7fffff };
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(subnormal_f32, RoundStyle::TIES_TO_AWAY)),
        zero_fp16);

    let minus_subnormal_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..subnormal_f32 };
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_subnormal_f32, RoundStyle::TIES_TO_AWAY)),
        minus_zero_fp16);

    // normals
    let one_fp16 = one<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:0);
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(one_f32, RoundStyle::TIES_TO_AWAY)), one_fp16);

    let minus_one_fp16 = APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:1, ..one_fp16 };
    let minus_one_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..one_f32 };
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_one_f32, RoundStyle::TIES_TO_AWAY)),
        minus_one_fp16);

    // fraction with no rounding necessary
    let one_dot_5_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: u23:0x400000, ..one_f32 };
    let one_dot_5_fp16 = APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { fraction: u10:0x200, ..one_fp16 };
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(one_dot_5_f32, RoundStyle::TIES_TO_AWAY)),
        one_dot_5_fp16);

    // rounds down
    let not_that_close_to_four_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x7fc000 };
    let almost_four_fp16 =
        APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:0, bexp: u5:0x10, fraction: u10:0x3fe };
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                not_that_close_to_four_f32, RoundStyle::TIES_TO_AWAY)), almost_four_fp16);

    // rounds up tie to away
    let getting_closer_to_four_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x7fe000 };
    let four_fp16 =
        APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:0, bexp: u5:0x10, fraction: u10:0x3ff };
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                getting_closer_to_four_f32, RoundStyle::TIES_TO_AWAY)), four_fp16);

    // rounds up tie to away.
    let even_closer_to_four_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x7ff000 };
    let four_fp16 =
        APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:0, bexp: u5:0x11, fraction: u10:0x0 };
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                even_closer_to_four_f32, RoundStyle::TIES_TO_AWAY)), four_fp16);

    // rounds up
    let soooo_close_to_four_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x7ff100 };
    let four_fp16 =
        APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:0, bexp: u5:0x11, fraction: u10:0x0 };
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(
                soooo_close_to_four_f32, RoundStyle::TIES_TO_AWAY)), four_fp16);

    // round up to inf
    let max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fffff };
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(max_normal_f32, RoundStyle::TIES_TO_AWAY)),
        inf_fp16);

    let minus_max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x7fffff };
    let minus_inf_fp16 = inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:1);
    assert_eq(
        subnormals_to_zero(
            downcast<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_max_normal_f32, RoundStyle::TIES_TO_AWAY)),
        minus_inf_fp16);
}

#[test]
fn downcast_rne_fp32_to_fp16_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const FP16_EXP_SZ = u32:5;
    const FP16_FRACTION_SZ = u32:10;

    // qnan -> qnan
    let qnan_f32 = qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    let qnan_fp16 = qnan<FP16_EXP_SZ, FP16_FRACTION_SZ>();
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(qnan_f32), qnan_fp16);

    // inf -> inf
    let inf_f32 = inf<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let inf_fp16 = inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:0);
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(inf_f32), inf_fp16);

    let minus_inf_f32 = inf<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    let minus_inf_fp16 = inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:1);
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_inf_f32), minus_inf_fp16);

    // +/- 0.0 -> same-signed 0
    let zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let zero_fp16 = zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:0);
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(zero_f32), zero_fp16);

    let minus_zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    let minus_zero_fp16 = zero<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:1);
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_zero_f32), minus_zero_fp16);

    // subnormals flushed to same-signed 0.0
    let subnormal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0, fraction: u23:0x7fffff };
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(subnormal_f32), zero_fp16);

    let minus_subnormal_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..subnormal_f32 };
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_subnormal_f32), minus_zero_fp16);

    // normals
    let one_fp16 = one<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:0);
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(one_f32), one_fp16);

    let minus_one_fp16 = APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:1, ..one_fp16 };
    let minus_one_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..one_f32 };
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_one_f32), minus_one_fp16);

    // fraction with no rounding necessary
    let one_dot_5_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: u23:0x400000, ..one_f32 };
    let one_dot_5_fp16 = APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { fraction: u10:0x200, ..one_fp16 };
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(one_dot_5_f32), one_dot_5_fp16);

    // rounds down
    let not_that_close_to_four_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x7fc000 };
    let almost_four_fp16 =
        APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:0, bexp: u5:0x10, fraction: u10:0x3fe };
    assert_eq(
        downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(not_that_close_to_four_f32), almost_four_fp16);

    // rounds down tie to even
    let getting_closer_to_four_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x7fe000 };
    let almost_four_fp16 =
        APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:0, bexp: u5:0x10, fraction: u10:0x3ff };
    assert_eq(
        downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(getting_closer_to_four_f32), almost_four_fp16);

    // rounds up tie to even.
    let even_closer_to_four_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x7ff000 };
    let four_fp16 =
        APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:0, bexp: u5:0x11, fraction: u10:0x0 };
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(even_closer_to_four_f32), four_fp16);

    // rounds up
    let soooo_close_to_four_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x7ff100 };
    let four_fp16 =
        APFloat<FP16_EXP_SZ, FP16_FRACTION_SZ> { sign: u1:0, bexp: u5:0x11, fraction: u10:0x0 };
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(soooo_close_to_four_f32), four_fp16);

    // round up to inf
    let max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fffff };
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(max_normal_f32), inf_fp16);

    let minus_max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x7fffff };
    let minus_inf_fp16 = inf<FP16_EXP_SZ, FP16_FRACTION_SZ>(u1:1);
    assert_eq(downcast_rne<FP16_FRACTION_SZ, FP16_EXP_SZ>(minus_max_normal_f32), minus_inf_fp16);
}

#[test]
fn downcast_rne_fp32_to_bf16_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;

    // qnan -> qnan
    let qnan_f32 = qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    let qnan_fp16 = qnan<BF16_EXP_SZ, BF16_FRACTION_SZ>();
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(qnan_f32), qnan_fp16);

    // inf -> inf
    let inf_f32 = inf<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let inf_fp16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(inf_f32), inf_fp16);

    let minus_inf_f32 = inf<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    let minus_inf_fp16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_inf_f32), minus_inf_fp16);

    // +/- 0.0 -> same-signed 0
    let zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let zero_fp16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(zero_f32), zero_fp16);

    let minus_zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    let minus_zero_fp16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_zero_f32), minus_zero_fp16);

    // subnormals flushed to same-signed 0.0
    let subnormal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0, fraction: u23:0x7fffff };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(subnormal_f32), zero_fp16);

    let minus_subnormal_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..subnormal_f32 };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_subnormal_f32), minus_zero_fp16);

    // normals
    let one_bf16 = one<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(one_f32), one_bf16);

    let minus_one_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:1, ..one_bf16 };
    let minus_one_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..one_f32 };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_one_f32), minus_one_bf16);

    // fraction with no rounding necessary
    let one_dot_5_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: u23:0x400000, ..one_f32 };
    let one_dot_5_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { fraction: u7:0x40, ..one_bf16 };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(one_dot_5_f32), one_dot_5_bf16);

    // rounds down
    let pi_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x490fdb };
    let pi_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u7:0x49 };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(pi_f32), pi_bf16);

    // rounds up
    let one_third_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x7d, fraction: u23:0x2aaaab };
    let one_third_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x7d, fraction: u7:0x2b };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(one_third_f32), one_third_bf16);

    // rounds down, tie to even
    let rne_down_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x408000 };
    let rne_down_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u7:0x40 };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(rne_down_f32), rne_down_bf16);

    // rounds up, tie to even
    let rne_up_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x418000 };
    let rne_up_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:0, bexp: u8:0x80, fraction: u7:0x42 };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(rne_up_f32), rne_up_bf16);

    // round up to inf
    let inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let just_above_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fe000 };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(just_above_max_normal_bf16), inf_bf16);

    let just_below_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x7fe000 };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);
    assert_eq(
        downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(just_below_max_normal_bf16), minus_inf_bf16);

    // round up to inf
    let max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fffff };
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(max_normal_f32), inf_bf16);

    let minus_max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x7fffff };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);
    assert_eq(downcast_rne<BF16_FRACTION_SZ, BF16_EXP_SZ>(minus_max_normal_f32), minus_inf_bf16);
}

// Returns a normalized APFloat with the given components.
// 'fraction_with_hidden' is the fraction (including the hidden bit). This
// function only normalizes in the direction of decreasing the exponent. Input
// must be a normal number or zero. Dernormals are flushed to zero in the
// result.
pub fn normalize<EXP_SZ: u32, FRACTION_SZ: u32, WIDE_FRACTION: u32 = {FRACTION_SZ + u32:1}>
    (sign: bits[1], exp: bits[EXP_SZ], fraction_with_hidden: bits[WIDE_FRACTION])
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    let leading_zeros = clz(fraction_with_hidden) as bits[FRACTION_SZ];
    let zero_value = zero<EXP_SZ, FRACTION_SZ>(sign);
    let zero_fraction = WIDE_FRACTION as bits[FRACTION_SZ];
    let normalized_fraction =
        (fraction_with_hidden << (leading_zeros as bits[WIDE_FRACTION])) as bits[FRACTION_SZ];

    let is_denormal = exp <= (leading_zeros as bits[EXP_SZ]);
    match (is_denormal, leading_zeros) {
        // Significand is zero.
        (_, zero_fraction) => zero_value,
        // Flush denormals to zero.
        (true, _) => zero_value,
        // Normalize.
        _ => APFloat {
            sign,
            bexp: exp - (leading_zeros as bits[EXP_SZ]),
            fraction: normalized_fraction,
        },
    }
}

// ldexp (load exponent) computes fraction * 2^exp.
// Note:
//  - Input denormals are treated as/flushed to 0.
//      (denormals-are-zero / DAZ).  Similarly,
//      denormal results are flushed to 0.
//  - No exception flags are raised/reported.
//  - We emit a single, canonical representation for
//      NaN (qnan) but accept all NaN representations
//      as input
//
// Returns fraction * 2^exp
pub fn ldexp<EXP_SZ: u32, FRACTION_SZ: u32>
    (fraction: APFloat<EXP_SZ, FRACTION_SZ>, exp: s32) -> APFloat<EXP_SZ, FRACTION_SZ> {
    type Float = APFloat<EXP_SZ, FRACTION_SZ>;

    const MAX_EXPONENT = max_normal_exp<EXP_SZ>() as s33;
    const MIN_EXPONENT = min_normal_exp<EXP_SZ>() as s33;

    // Flush subnormal input.
    let fraction = subnormals_to_zero(fraction);

    // Increase the exponent of fraction by 'exp'. If this was not a DAZ module,
    // we'd have to deal with denormal 'fraction' here.
    let exp = signex(exp, s33:0) + signex(unbiased_exponent<EXP_SZ, FRACTION_SZ>(fraction), s33:0);
    let result =
        Float { sign: fraction.sign, bexp: bias(exp as sN[EXP_SZ]), fraction: fraction.fraction };

    // Handle overflow.
    let result = if exp > MAX_EXPONENT { inf<EXP_SZ, FRACTION_SZ>(fraction.sign) } else { result };

    // Handle underflow, taking into account the case that underflow rounds back
    // up to a normal number. If this was not a DAZ module, we'd have to deal with
    // denormal 'result' here.
    let underflow_result =
        if exp == (MIN_EXPONENT - s33:1) && fraction.fraction == std::mask_bits<FRACTION_SZ>() {
            Float { sign: fraction.sign, bexp: uN[EXP_SZ]:1, fraction: uN[FRACTION_SZ]:0 }
        } else {
            zero<EXP_SZ, FRACTION_SZ>(fraction.sign)
        };

    let result = if exp < MIN_EXPONENT { underflow_result } else { result };
    // Flush subnormal output.
    let result = subnormals_to_zero(result);

    // Handle special cases.
    let result = if is_zero_or_subnormal(fraction) || is_inf(fraction) { fraction } else { result };
    let result = if is_nan(fraction) { qnan<EXP_SZ, FRACTION_SZ>() } else { result };
    result
}

#[test]
fn ldexp_test() {
    // Test Special cases.
    assert_eq(ldexp(zero<u32:8, u32:23>(u1:0), s32:1), zero<u32:8, u32:23>(u1:0));
    assert_eq(ldexp(zero<u32:8, u32:23>(u1:1), s32:1), zero<u32:8, u32:23>(u1:1));
    assert_eq(ldexp(inf<u32:8, u32:23>(u1:0), s32:-1), inf<u32:8, u32:23>(u1:0));
    assert_eq(ldexp(inf<u32:8, u32:23>(u1:1), s32:-1), inf<u32:8, u32:23>(u1:1));
    assert_eq(ldexp(qnan<u32:8, u32:23>(), s32:1), qnan<u32:8, u32:23>());

    // Subnormal input.
    let pos_denormal = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:0, bexp: u8:0, fraction: u23:99 };
    assert_eq(ldexp(pos_denormal, s32:1), zero<u32:8, u32:23>(u1:0));
    let neg_denormal = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:1, bexp: u8:0, fraction: u23:99 };
    assert_eq(ldexp(neg_denormal, s32:1), zero<u32:8, u32:23>(u1:1));

    // Output subnormal, flush to zero.
    assert_eq(ldexp(pos_denormal, s32:-1), zero<u32:8, u32:23>(u1:0));

    // Subnormal result rounds up to normal number.
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:0, bexp: u8:10, fraction: u23:0x7fffff };
    let expected = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:0, bexp: u8:1, fraction: u23:0 };
    assert_eq(ldexp(frac, s32:-10), expected);
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:1, bexp: u8:10, fraction: u23:0x7fffff };
    let expected = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:1, bexp: u8:1, fraction: u23:0 };
    assert_eq(ldexp(frac, s32:-10), expected);

    // Large positive input exponents.
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:0, bexp: u8:128, fraction: u23:0x0 };
    let expected = inf<u32:8, u32:23>(u1:0);
    assert_eq(ldexp(frac, s32:0x7FFFFFFF - s32:1), expected);
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:0, bexp: u8:128, fraction: u23:0x0 };
    let expected = inf<u32:8, u32:23>(u1:0);
    assert_eq(ldexp(frac, s32:0x7FFFFFFF), expected);
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:1, bexp: u8:128, fraction: u23:0x0 };
    let expected = inf<u32:8, u32:23>(u1:1);
    assert_eq(ldexp(frac, s32:0x7FFFFFFF - s32:1), expected);
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:1, bexp: u8:128, fraction: u23:0x0 };
    let expected = inf<u32:8, u32:23>(u1:1);
    assert_eq(ldexp(frac, s32:0x7FFFFFFF), expected);

    // Large negative input exponents.
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:0, bexp: u8:126, fraction: u23:0x0 };
    let expected = zero<u32:8, u32:23>(u1:0);
    assert_eq(ldexp(frac, s32:0x80000000 + s32:0x1), expected);
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:0, bexp: u8:126, fraction: u23:0x0 };
    let expected = zero<u32:8, u32:23>(u1:0);
    assert_eq(ldexp(frac, s32:0x80000000), expected);
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:1, bexp: u8:126, fraction: u23:0x0 };
    let expected = zero<u32:8, u32:23>(u1:1);
    assert_eq(ldexp(frac, s32:0x80000000 + s32:0x1), expected);
    let frac = APFloat<EXP_SZ, FRACTION_SZ> { sign: u1:1, bexp: u8:126, fraction: u23:0x0 };
    let expected = zero<u32:8, u32:23>(u1:1);
    assert_eq(ldexp(frac, s32:0x80000000), expected);

    // Other large exponents from reported bug #462.
    let frac = unflatten<u32:8, u32:23>(u32:0xd3fefd2b);
    let expected = inf<u32:8, u32:23>(u1:1);
    assert_eq(ldexp(frac, s32:0x7ffffffd), expected);
    let frac = unflatten<u32:8, u32:23>(u32:0x36eba93e);
    let expected = zero<u32:8, u32:23>(u1:0);
    assert_eq(ldexp(frac, s32:0x80000010), expected);
    let frac = unflatten<u32:8, u32:23>(u32:0x8a87c096);
    let expected = zero<u32:8, u32:23>(u1:1);
    assert_eq(ldexp(frac, s32:0x80000013), expected);
    let frac = unflatten<u32:8, u32:23>(u32:0x71694e37);
    let expected = inf<u32:8, u32:23>(u1:0);
    assert_eq(ldexp(frac, s32:0x7fffffbe), expected);
}

// Casts the floating point number to a fixed point number.
// Unrepresentable numbers are cast to the minimum representable
// number (largest magnitude negative number).
pub fn cast_to_fixed<NUM_DST_BITS: u32, EXP_SZ: u32, FRACTION_SZ: u32>
    (to_cast: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[NUM_DST_BITS] {
    const UEXP_SZ: u32 = EXP_SZ + u32:1;
    const EXTENDED_FIXED_SZ: u32 = NUM_DST_BITS + u32:1 + FRACTION_SZ + NUM_DST_BITS;

    const MIN_FIXED_VALUE = (uN[NUM_DST_BITS]:1 <<
                            ((NUM_DST_BITS as uN[NUM_DST_BITS]) - uN[NUM_DST_BITS]:1)) as
                            sN[NUM_DST_BITS];
    const MAX_EXPONENT = NUM_DST_BITS - u32:1;

    // Convert to fixed point and truncate fractional bits.
    let exp = unbiased_exponent(to_cast);
    let result = (uN[NUM_DST_BITS]:0 ++ u1:1 ++ to_cast.fraction ++ uN[NUM_DST_BITS]:0) as
                 uN[EXTENDED_FIXED_SZ];
    let result = result >>
                 ((FRACTION_SZ as uN[EXTENDED_FIXED_SZ]) + (NUM_DST_BITS as uN[EXTENDED_FIXED_SZ]) -
                 (exp as uN[EXTENDED_FIXED_SZ]));
    let result = result[0:NUM_DST_BITS as s32] as sN[NUM_DST_BITS];
    let result = if to_cast.sign { -result } else { result };

    // NaN and too-large inputs --> MIN_FIXED_VALUE
    let overflow = (exp as u32) >= MAX_EXPONENT;
    let result = if overflow || is_nan(to_cast) { MIN_FIXED_VALUE } else { result };
    // Underflow / to_cast < 1 --> 0
    let result = if to_cast.bexp < bias(sN[EXP_SZ]:0) { sN[NUM_DST_BITS]:0 } else { result };

    result
}

#[test]
fn cast_to_fixed_test() {
    // Cast +/-0.0
    assert_eq(cast_to_fixed<u32:32>(zero<u32:8, u32:23>(u1:0)), s32:0);
    assert_eq(cast_to_fixed<u32:32>(zero<u32:8, u32:23>(u1:1)), s32:0);

    // Cast +/-1.0
    assert_eq(cast_to_fixed<u32:32>(one<u32:8, u32:23>(u1:0)), s32:1);
    assert_eq(cast_to_fixed<u32:32>(one<u32:8, u32:23>(u1:1)), s32:-1);

    // Cast +/-1.5 --> +/- 1
    let one_point_five =
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7f, fraction: u1:1 ++ u22:0 };
    assert_eq(cast_to_fixed<u32:32>(one_point_five), s32:1);
    let n_one_point_five =
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0x7f, fraction: u1:1 ++ u22:0 };
    assert_eq(cast_to_fixed<u32:32>(n_one_point_five), s32:-1);

    // Cast +/-4.0
    let four = cast_from_fixed_using_rne<u32:8, u32:23>(s32:4);
    let neg_four = cast_from_fixed_using_rne<u32:8, u32:23>(s32:-4);
    assert_eq(cast_to_fixed<u32:32>(four), s32:4);
    assert_eq(cast_to_fixed<u32:32>(neg_four), s32:-4);

    // Cast 7
    let seven = cast_from_fixed_using_rne<u32:8, u32:23>(s32:7);
    assert_eq(cast_to_fixed<u32:32>(seven), s32:7);

    // Cast big number (more digits left of decimal than hidden bit + fraction).
    let big_num = (u1:0 ++ std::mask_bits<u32:23>() ++ u8:0) as s32;
    let fp_big_num = cast_from_fixed_using_rne<u32:8, u32:23>(big_num);
    assert_eq(cast_to_fixed<u32:32>(fp_big_num), big_num);

    // Cast large, non-overflowing numbers.
    let big_fit =
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:127 + u8:30, fraction: u23:0x7fffff };
    assert_eq(cast_to_fixed<u32:32>(big_fit), (u1:0 ++ u24:0xffffff ++ u7:0) as s32);
    let big_fit =
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:127 + u8:30, fraction: u23:0x7fffff };
    assert_eq(cast_to_fixed<u32:32>(big_fit), (s32:0 - (u1:0 ++ u24:0xffffff ++ u7:0) as s32));

    // Cast barely overflowing postive number.
    let big_overflow =
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:127 + u8:31, fraction: u23:0x0 };
    assert_eq(cast_to_fixed<u32:32>(big_overflow), (u1:1 ++ u31:0) as s32);

    // This produces the largest negative int, but doesn't actually
    // overflow
    let max_negative =
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:127 + u8:31, fraction: u23:0x0 };
    assert_eq(cast_to_fixed<u32:32>(max_negative), (u1:1 ++ u31:0) as s32);

    // Negative overflow.
    let negative_overflow =
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:127 + u8:31, fraction: u23:0x1 };
    assert_eq(cast_to_fixed<u32:32>(negative_overflow), (u1:1 ++ u31:0) as s32);

    // NaN input.
    assert_eq(cast_to_fixed<u32:32>(qnan<u32:8, u32:23>()), (u1:1 ++ u31:0) as s32);
}

// Returns u1:1 if x == y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn eq_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    if !(is_nan(x) || is_nan(y)) {
        ((flatten(x) == flatten(y)) || (is_zero_or_subnormal(x) && is_zero_or_subnormal(y)))
    } else {
        u1:0
    }
}

#[test]
fn test_fp_eq_2() {
    let neg_zero = zero<u32:8, u32:23>(u1:1);
    let zero = zero<u32:8, u32:23>(u1:0);
    let neg_one = one<u32:8, u32:23>(u1:1);
    let one = one<u32:8, u32:23>(u1:0);
    let two = APFloat<8, 23> { bexp: one.bexp + uN[8]:1, ..one };
    let neg_inf = inf<u32:8, u32:23>(u1:1);
    let inf = inf<u32:8, u32:23>(u1:0);
    let nan = qnan<u32:8, u32:23>();
    let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
    let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

    // Test unequal.
    assert_eq(eq_2(one, two), u1:0);
    assert_eq(eq_2(two, one), u1:0);

    // Test equal.
    assert_eq(eq_2(neg_zero, zero), u1:1);
    assert_eq(eq_2(one, one), u1:1);
    assert_eq(eq_2(two, two), u1:1);

    // Test equal (subnormals and zero).
    assert_eq(eq_2(zero, zero), u1:1);
    assert_eq(eq_2(zero, neg_zero), u1:1);
    assert_eq(eq_2(zero, denormal_1), u1:1);
    assert_eq(eq_2(denormal_2, denormal_1), u1:1);

    // Test negatives.
    assert_eq(eq_2(one, neg_one), u1:0);
    assert_eq(eq_2(neg_one, one), u1:0);
    assert_eq(eq_2(neg_one, neg_one), u1:1);

    // Special case - inf.
    assert_eq(eq_2(inf, one), u1:0);
    assert_eq(eq_2(neg_inf, inf), u1:0);
    assert_eq(eq_2(inf, inf), u1:1);
    assert_eq(eq_2(neg_inf, neg_inf), u1:1);

    // Special case - NaN (always returns false).
    assert_eq(eq_2(one, nan), u1:0);
    assert_eq(eq_2(neg_one, nan), u1:0);
    assert_eq(eq_2(inf, nan), u1:0);
    assert_eq(eq_2(nan, inf), u1:0);
    assert_eq(eq_2(nan, nan), u1:0);
}

// Returns u1:1 if x > y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn gt_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    // Flush denormals.
    let x = subnormals_to_zero(x);
    let y = subnormals_to_zero(y);

    let gt_exp = x.bexp > y.bexp;
    let eq_exp = x.bexp == y.bexp;
    let gt_fraction = x.fraction > y.fraction;
    let abs_gt = gt_exp || (eq_exp && gt_fraction);
    let result = match (x.sign, y.sign) {
        // Both positive.
        (u1:0, u1:0) => abs_gt,
        // x positive, y negative.
        (u1:0, u1:1) => u1:1,
        // x negative, y positive.
        (u1:1, u1:0) => u1:0,
        // Both negative.
        _ => !abs_gt && !eq_2(x, y),
    };

    if !(is_nan(x) || is_nan(y)) { result } else { u1:0 }
}

#[test]
fn test_fp_gt_2() {
    let zero = zero<u32:8, u32:23>(u1:0);
    let neg_one = one<u32:8, u32:23>(u1:1);
    let one = one<u32:8, u32:23>(u1:0);
    let two = APFloat<u32:8, u32:23> { bexp: one.bexp + u8:1, ..one };
    let neg_two = APFloat<u32:8, u32:23> { bexp: neg_one.bexp + u8:1, ..neg_one };
    let neg_inf = inf<u32:8, u32:23>(u1:1);
    let inf = inf<u32:8, u32:23>(u1:0);
    let nan = qnan<u32:8, u32:23>();
    let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
    let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

    // Test unequal.
    assert_eq(gt_2(one, two), u1:0);
    assert_eq(gt_2(two, one), u1:1);

    // Test equal.
    assert_eq(gt_2(one, one), u1:0);
    assert_eq(gt_2(two, two), u1:0);
    assert_eq(gt_2(denormal_1, denormal_2), u1:0);
    assert_eq(gt_2(denormal_2, denormal_1), u1:0);
    assert_eq(gt_2(denormal_1, zero), u1:0);

    // Test negatives.
    assert_eq(gt_2(zero, neg_one), u1:1);
    assert_eq(gt_2(neg_one, zero), u1:0);
    assert_eq(gt_2(one, neg_one), u1:1);
    assert_eq(gt_2(neg_one, one), u1:0);
    assert_eq(gt_2(neg_one, neg_one), u1:0);
    assert_eq(gt_2(neg_two, neg_two), u1:0);
    assert_eq(gt_2(neg_one, neg_two), u1:1);
    assert_eq(gt_2(neg_two, neg_one), u1:0);

    // Special case - inf.
    assert_eq(gt_2(inf, one), u1:1);
    assert_eq(gt_2(inf, neg_one), u1:1);
    assert_eq(gt_2(inf, two), u1:1);
    assert_eq(gt_2(neg_two, neg_inf), u1:1);
    assert_eq(gt_2(inf, inf), u1:0);
    assert_eq(gt_2(neg_inf, inf), u1:0);
    assert_eq(gt_2(inf, neg_inf), u1:1);
    assert_eq(gt_2(neg_inf, neg_inf), u1:0);

    // Special case - NaN (always returns false).
    assert_eq(gt_2(one, nan), u1:0);
    assert_eq(gt_2(nan, one), u1:0);
    assert_eq(gt_2(neg_one, nan), u1:0);
    assert_eq(gt_2(nan, neg_one), u1:0);
    assert_eq(gt_2(inf, nan), u1:0);
    assert_eq(gt_2(nan, inf), u1:0);
    assert_eq(gt_2(nan, nan), u1:0);
}

// Returns u1:1 if x >= y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn gte_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    gt_2(x, y) || eq_2(x, y)
}

#[test]
fn test_fp_gte_2() {
    let zero = zero<u32:8, u32:23>(u1:0);
    let neg_one = one<u32:8, u32:23>(u1:1);
    let one = one<u32:8, u32:23>(u1:0);
    let two = APFloat<u32:8, u32:23> { bexp: one.bexp + u8:1, ..one };
    let neg_two = APFloat<u32:8, u32:23> { bexp: neg_one.bexp + u8:1, ..neg_one };
    let neg_inf = inf<u32:8, u32:23>(u1:1);
    let inf = inf<u32:8, u32:23>(u1:0);
    let nan = qnan<u32:8, u32:23>();
    let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
    let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

    // Test unequal.
    assert_eq(gte_2(one, two), u1:0);
    assert_eq(gte_2(two, one), u1:1);

    // Test equal.
    assert_eq(gte_2(one, one), u1:1);
    assert_eq(gte_2(two, two), u1:1);
    assert_eq(gte_2(denormal_1, denormal_2), u1:1);
    assert_eq(gte_2(denormal_2, denormal_1), u1:1);
    assert_eq(gte_2(denormal_1, zero), u1:1);

    // Test negatives.
    assert_eq(gte_2(zero, neg_one), u1:1);
    assert_eq(gte_2(neg_one, zero), u1:0);
    assert_eq(gte_2(one, neg_one), u1:1);
    assert_eq(gte_2(neg_one, one), u1:0);
    assert_eq(gte_2(neg_one, neg_one), u1:1);
    assert_eq(gte_2(neg_two, neg_two), u1:1);
    assert_eq(gte_2(neg_one, neg_two), u1:1);
    assert_eq(gte_2(neg_two, neg_one), u1:0);

    // Special case - inf.
    assert_eq(gte_2(inf, one), u1:1);
    assert_eq(gte_2(inf, neg_one), u1:1);
    assert_eq(gte_2(inf, two), u1:1);
    assert_eq(gte_2(neg_two, neg_inf), u1:1);
    assert_eq(gte_2(inf, inf), u1:1);
    assert_eq(gte_2(neg_inf, inf), u1:0);
    assert_eq(gte_2(inf, neg_inf), u1:1);
    assert_eq(gte_2(neg_inf, neg_inf), u1:1);

    // Special case - NaN (always returns false).
    assert_eq(gte_2(one, nan), u1:0);
    assert_eq(gte_2(nan, one), u1:0);
    assert_eq(gte_2(neg_one, nan), u1:0);
    assert_eq(gte_2(nan, neg_one), u1:0);
    assert_eq(gte_2(inf, nan), u1:0);
    assert_eq(gte_2(nan, inf), u1:0);
    assert_eq(gte_2(nan, nan), u1:0);
}

// Returns u1:1 if x <= y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn lte_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    if !(is_nan(x) || is_nan(y)) { !gt_2(x, y) } else { u1:0 }
}

#[test]
fn test_fp_lte_2() {
    let zero = zero<u32:8, u32:23>(u1:0);
    let neg_one = one<u32:8, u32:23>(u1:1);
    let one = one<u32:8, u32:23>(u1:0);
    let two = APFloat<u32:8, u32:23> { bexp: one.bexp + u8:1, ..one };
    let neg_two = APFloat<u32:8, u32:23> { bexp: neg_one.bexp + u8:1, ..neg_one };
    let neg_inf = inf<u32:8, u32:23>(u1:1);
    let inf = inf<u32:8, u32:23>(u1:0);
    let nan = qnan<u32:8, u32:23>();
    let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
    let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

    // Test unequal.
    assert_eq(lte_2(one, two), u1:1);
    assert_eq(lte_2(two, one), u1:0);

    // Test equal.
    assert_eq(lte_2(one, one), u1:1);
    assert_eq(lte_2(two, two), u1:1);
    assert_eq(lte_2(denormal_1, denormal_2), u1:1);
    assert_eq(lte_2(denormal_2, denormal_1), u1:1);
    assert_eq(lte_2(denormal_1, zero), u1:1);

    // Test negatives.
    assert_eq(lte_2(zero, neg_one), u1:0);
    assert_eq(lte_2(neg_one, zero), u1:1);
    assert_eq(lte_2(one, neg_one), u1:0);
    assert_eq(lte_2(neg_one, one), u1:1);
    assert_eq(lte_2(neg_one, neg_one), u1:1);
    assert_eq(lte_2(neg_two, neg_two), u1:1);
    assert_eq(lte_2(neg_one, neg_two), u1:0);
    assert_eq(lte_2(neg_two, neg_one), u1:1);

    // Special case - inf.
    assert_eq(lte_2(inf, one), u1:0);
    assert_eq(lte_2(inf, neg_one), u1:0);
    assert_eq(lte_2(inf, two), u1:0);
    assert_eq(lte_2(neg_two, neg_inf), u1:0);
    assert_eq(lte_2(inf, inf), u1:1);
    assert_eq(lte_2(neg_inf, inf), u1:1);
    assert_eq(lte_2(inf, neg_inf), u1:0);
    assert_eq(lte_2(neg_inf, neg_inf), u1:1);

    // Special case - NaN (always returns false).
    assert_eq(lte_2(one, nan), u1:0);
    assert_eq(lte_2(nan, one), u1:0);
    assert_eq(lte_2(neg_one, nan), u1:0);
    assert_eq(lte_2(nan, neg_one), u1:0);
    assert_eq(lte_2(inf, nan), u1:0);
    assert_eq(lte_2(nan, inf), u1:0);
    assert_eq(lte_2(nan, nan), u1:0);
}

// Returns u1:1 if x < y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn lt_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    if !(is_nan(x) || is_nan(y)) { !gte_2(x, y) } else { u1:0 }
}

#[test]
fn test_fp_lt_2() {
    let zero = zero<u32:8, u32:23>(u1:0);
    let neg_one = one<u32:8, u32:23>(u1:1);
    let one = one<u32:8, u32:23>(u1:0);
    let two = APFloat<u32:8, u32:23> { bexp: one.bexp + u8:1, ..one };
    let neg_two = APFloat<u32:8, u32:23> { bexp: neg_one.bexp + u8:1, ..neg_one };
    let neg_inf = inf<u32:8, u32:23>(u1:1);
    let inf = inf<u32:8, u32:23>(u1:0);
    let nan = qnan<u32:8, u32:23>();
    let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
    let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

    // Test unequal.
    assert_eq(lt_2(one, two), u1:1);
    assert_eq(lt_2(two, one), u1:0);

    // Test equal.
    assert_eq(lt_2(one, one), u1:0);
    assert_eq(lt_2(two, two), u1:0);
    assert_eq(lt_2(denormal_1, denormal_2), u1:0);
    assert_eq(lt_2(denormal_2, denormal_1), u1:0);
    assert_eq(lt_2(denormal_1, zero), u1:0);

    // Test negatives.
    assert_eq(lt_2(zero, neg_one), u1:0);
    assert_eq(lt_2(neg_one, zero), u1:1);
    assert_eq(lt_2(one, neg_one), u1:0);
    assert_eq(lt_2(neg_one, one), u1:1);
    assert_eq(lt_2(neg_one, neg_one), u1:0);
    assert_eq(lt_2(neg_two, neg_two), u1:0);
    assert_eq(lt_2(neg_one, neg_two), u1:0);
    assert_eq(lt_2(neg_two, neg_one), u1:1);

    // Special case - inf.
    assert_eq(lt_2(inf, one), u1:0);
    assert_eq(lt_2(inf, neg_one), u1:0);
    assert_eq(lt_2(inf, two), u1:0);
    assert_eq(lt_2(neg_two, neg_inf), u1:0);
    assert_eq(lt_2(inf, inf), u1:0);
    assert_eq(lt_2(neg_inf, inf), u1:1);
    assert_eq(lt_2(inf, neg_inf), u1:0);
    assert_eq(lt_2(neg_inf, neg_inf), u1:0);

    // Special case - NaN (always returns false).
    assert_eq(lt_2(one, nan), u1:0);
    assert_eq(lt_2(nan, one), u1:0);
    assert_eq(lt_2(neg_one, nan), u1:0);
    assert_eq(lt_2(nan, neg_one), u1:0);
    assert_eq(lt_2(inf, nan), u1:0);
    assert_eq(lt_2(nan, inf), u1:0);
    assert_eq(lt_2(nan, nan), u1:0);
}

// Helper to convert to a signed or unsigned integer.
fn to_signed_or_unsigned_int<RESULT_SZ: u32, RESULT_SIGNED: bool, EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> xN[RESULT_SIGNED][RESULT_SZ] {
    const WIDE_FRACTION: u32 = FRACTION_SZ + u32:1;
    const MAX_FRACTION_SZ: u32 = std::max(RESULT_SZ, WIDE_FRACTION);

    const INT_MIN = if RESULT_SIGNED {
        (uN[MAX_FRACTION_SZ]:1 << (RESULT_SZ - u32:1))  // or rather, its negative.
    } else {
        uN[MAX_FRACTION_SZ]:0
    };
    const INT_MAX = if RESULT_SIGNED {
        (uN[MAX_FRACTION_SZ]:1 << (RESULT_SZ - u32:1)) - uN[MAX_FRACTION_SZ]:1
    } else {
        (uN[MAX_FRACTION_SZ]:1 << RESULT_SZ) - uN[MAX_FRACTION_SZ]:1
    };

    let exp = unbiased_exponent(x);

    // True significand (including implicit leading bit) as an integer. Will need to be shifted
    // so that MSB represents the value 1 based on the effective exponent (exp - FRACTION_SZ).
    let fraction = (x.fraction as uN[WIDE_FRACTION] | (uN[WIDE_FRACTION]:1 << FRACTION_SZ)) as
                   uN[MAX_FRACTION_SZ];

    let max_exp = if RESULT_SIGNED { RESULT_SZ as s32 - s32:1 } else { RESULT_SZ as s32 };
    let exp_oob = exp as s32 >= max_exp;
    let result = if exp_oob || is_inf(x) || (!RESULT_SIGNED && x.sign) {
        // Clamp if out of bounds, infinite.
        if x.sign { INT_MIN } else { INT_MAX }
    } else if is_nan(x) {
        uN[MAX_FRACTION_SZ]:0
    } else if exp < sN[EXP_SZ]:0 {
        uN[MAX_FRACTION_SZ]:0
    } else if exp == sN[EXP_SZ]:0 {
        uN[MAX_FRACTION_SZ]:1
    } else {
        // For most cases, we need to either shift the "ones" place from FRACTION_SZ + 1 bits down
        // closer to 0 (if the effective exponent is negative) else we need to move it away from 0
        // if the effective exponent is positive.
        let effective_exp = (exp as s32) - (FRACTION_SZ as s32);
        if effective_exp < s32:0 {
            (fraction >> (-effective_exp as u32))
        } else if effective_exp > s32:0 {
            (fraction << effective_exp as u32)
        } else {
            fraction
        }
    };

    // Needed for matching conditional consequent/alternative types.
    type ReturnT = xN[RESULT_SIGNED][RESULT_SZ];
    if RESULT_SIGNED {
        // Reduce to the target size, preserving signedness.
        let result = result as sN[RESULT_SZ];
        let result = if !x.sign { result } else { -result };
        result as ReturnT
    } else {
        // Already unsigned, just size correctly.
        result as ReturnT
    }
}

// Returns the signed integer part of the input float, truncating any
// fractional bits if necessary.
// Exceptional cases:
// NaN                  -> sN[RESULT_SZ]::ZERO
// +Inf                 -> sN[RESULT_SZ]::MAX
// -Inf                 -> sN[RESULT_SZ]::MIN
// +0, -0, subnormal    -> sN[RESULT_SZ]::ZERO
// > sN[RESULT_SZ]::MAX -> sN[RESULT_SZ]::MAX
// < sN[RESULT_SZ]::MIN -> sN[RESULT_SZ]::MIN
pub fn to_int<EXP_SZ: u32, FRACTION_SZ: u32, RESULT_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[RESULT_SZ] {
    to_signed_or_unsigned_int<RESULT_SZ, true>(x)
}

// TODO(rspringer): Create a broadly-applicable normalize test, that
// could be used for multiple type instantiations (without needing
// per-specialization data to be specified by a user).

#[test]
fn to_int_test() {
    // +0
    let expected = s32:0;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x0, fraction: u23:0x0 });
    assert_eq(expected, actual);

    // smallest positive subnormal
    let expected = s32:0;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x0, fraction: u23:0x1 });
    assert_eq(expected, actual);

    // largest subnormal
    let expected = s32:0;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x0, fraction: u23:0x7fffff });
    assert_eq(expected, actual);

    // smallest positive normal
    let expected = s32:0;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x1, fraction: u23:0x0 });
    assert_eq(expected, actual);

    // largest < 1
    let expected = s32:0;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7e, fraction: u23:0x7fffff });
    assert_eq(expected, actual);

    // 1
    let expected = s32:1;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x0 });
    assert_eq(expected, actual);

    let expected = s32:1;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7f, fraction: u23:0xa5a5 });
    assert_eq(expected, actual);

    // 2
    let expected = s32:2;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x0 });
    assert_eq(expected, actual);

    let expected = s32:2;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x80, fraction: u23:0xa5a5 });
    assert_eq(expected, actual);

    let expected = s32:0xa5a5;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x8e, fraction: u23:0x25a500 });
    assert_eq(expected, actual);

    let expected = s32:23;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x83, fraction: u23:0x380000 });
    assert_eq(expected, actual);

    // unbiased exp == EXP_SZ = 23
    let expected = s32:0xffffff;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x96, fraction: u23:0x7fffff });
    assert_eq(expected, actual);

    let expected = s16:0xa5a;
    let actual = to_int<u32:8, u32:23, u32:16>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x8a, fraction: u23:0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0xa5;
    let actual = to_int<u32:8, u32:23, u32:16>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x86, fraction: u23:0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0x14b;
    let actual = to_int<u32:8, u32:23, u32:16>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x87, fraction: u23:0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0x296;
    let actual = to_int<u32:8, u32:23, u32:16>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x88, fraction: u23:0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16::MAX;
    let actual = to_int<u32:8, u32:23, u32:16>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x8f, fraction: u23:0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s24:0x14b4b;
    let actual = to_int<u32:8, u32:23, u32:24>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x8f, fraction: u23:0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s32:0x14b4b;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x8f, fraction: u23:0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0xa;
    let actual = to_int<u32:8, u32:23, u32:16>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x82, fraction: u23:0x25a5a5 });
    assert_eq(expected, actual);

    let expected = s16:0x5;
    let actual = to_int<u32:8, u32:23, u32:16>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x81, fraction: u23:0x25a5a5 });
    assert_eq(expected, actual);

    // s32::MAX + 1.0
    let expected = s32::MAX;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x9e, fraction: u23:0x0 });
    assert_eq(expected, actual);

    // s32::MAX + 1.0, fits in s64 so no clamping
    let expected = (s32::MAX as s64) + s64:1;
    let actual = to_int<u32:8, u32:23, u32:64>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x9e, fraction: u23:0x0 });
    assert_eq(expected, actual);

    // large s64 sized integer as exact float32
    let expected = s64:0x1234_5000_0000_0000;
    let actual = to_int<u32:8, u32:23, u32:64>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0xbb, fraction: u23:0x11a280 });
    assert_eq(expected, actual);

    // largest normal
    let expected = s32::MAX;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fffff });
    assert_eq(expected, actual);

    // inf
    let expected = s32::MAX;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0xff, fraction: u23:0x0 });
    assert_eq(expected, actual);

    // -0
    let expected = s32:0;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0x0, fraction: u23:0x0 });
    assert_eq(expected, actual);

    // negative subnormal
    let expected = s32:0;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0x0, fraction: u23:0x7fffff });
    assert_eq(expected, actual);

    // -2
    let expected = s32:-2;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0x80, fraction: u23:0x0 });
    assert_eq(expected, actual);

    let expected = s32:-2;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0x80, fraction: u23:0x1 });
    assert_eq(expected, actual);

    // -inf
    const S32_MIN = s32:1 << 31;
    let expected = S32_MIN;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0xff, fraction: u23:0x0 });
    assert_eq(expected, actual);

    // nan
    let expected = s32:0;
    let actual = to_int<u32:8, u32:23, u32:32>(qnan<u32:8, u32:23>());
    assert_eq(expected, actual);
}

// Cast the input float to an unsigned integer. Any fractional bits are truncated
// and negative values are clamped to 0.
// Exceptional cases:
// NaN                   -> uN[RESULT_SZ]::ZERO
// +Inf                  -> uN[RESULT_SZ]::MAX
// -Inf                  -> uN[RESULT_SZ]::ZERO
// +0, -0, subnormal     -> uN[RESULT_SZ]::ZERO
// > uN[RESULT_SZ]::MAX  -> uN[RESULT_SZ]::MAX
// < uN[RESULT_SZ]::ZERO -> uN[RESULT_SZ]::ZERO
pub fn to_uint<RESULT_SZ: u32, EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> uN[RESULT_SZ] {
    to_signed_or_unsigned_int<RESULT_SZ, false>(x)
}

#[test]
fn to_uint_test() {
    // +0
    assert_eq(
        u32:0,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x0, fraction: u23:0x0 }));

    // smallest positive subnormal
    assert_eq(
        u32:0,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x0, fraction: u23:0x1 }));

    // largest subnormal
    assert_eq(
        u32:0,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x0, fraction: u23:0x7fffff }));

    // smallest positive normal
    assert_eq(
        u32:0,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x1, fraction: u23:0x0 }));

    // largest < 1
    assert_eq(
        u32:0,
        to_uint<u32:32>(
            APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7e, fraction: u23:0x7fffff }));

    // 1
    assert_eq(
        u32:1,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x0 }));

    // smallest > 1
    assert_eq(
        u32:1,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7f, fraction: u23:0x1 }));

    // pi
    assert_eq(
        u32:3,
        to_uint<u32:32>(
            APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x80, fraction: u23:0x490fdb }));

    // largest normal
    assert_eq(
        u32::MAX,
        to_uint<u32:32>(
            APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0xfe, fraction: u23:0x7fffff }));

    // inf
    assert_eq(u32::MAX, to_uint<u32:32>(inf<u32:8, u32:23>(u1:0)));

    // -0
    assert_eq(
        u32:0,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x0, fraction: u23:0x0 }));

    // -2
    assert_eq(
        u32:0,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0x80, fraction: u23:0x0 }));

    // -2
    assert_eq(
        u32:0,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0x80, fraction: u23:0x0 }));

    // largest < -2
    assert_eq(
        u32:0,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:1, bexp: u8:0x80, fraction: u23:0x1 }));

    // -inf
    assert_eq(u32:0, to_uint<u32:32>(inf<u32:8, u32:23>(u1:1)));

    // nan
    assert_eq(u32:0, to_uint<u32:32>(qnan<u32:8, u32:23>()));

    // s32::MAX + 1.0, should fit as uint32.
    assert_eq(
        u32:0x80000000,
        to_uint<u32:32>(APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x9e, fraction: u23:0x0 }));

    // basic spot check with different format (bfloat16)
    // smallest > 1
    assert_eq(
        u16:1,
        to_uint<u32:16>(APFloat<u32:8, u32:7> { sign: u1:0, bexp: u8:0x7f, fraction: u7:0x1 }));

    // largest normal
    assert_eq(
        u16::MAX,
        to_uint<u32:16>(APFloat<u32:8, u32:7> { sign: u1:0, bexp: u8:0x0fe, fraction: u7:0x7f }));

    // -2
    assert_eq(
        u4:0, to_uint<u32:4>(APFloat<u32:8, u32:7> { sign: u1:1, bexp: u8:0x80, fraction: u7:0 }));
}

fn compound_adder<WIDTH: u32>(a: uN[WIDTH], b: uN[WIDTH]) -> (uN[WIDTH], uN[WIDTH]) {
    (a + b, a + b + uN[WIDTH]:1)
}

// Calculate difference of two positive values and return values in sign-magnitude
// form. Returns sign-magnitude tuple (|a| - |b| <= 0, abs(|a| - |b|)).
// Note, this returns -0 if (a == b), which is used in our application, which is good
// for testing if strictly |a| > |b|.
fn sign_magnitude_difference<WIDTH: u32>(a: uN[WIDTH], b: uN[WIDTH]) -> (bool, uN[WIDTH]) {
    let abs_diff_result = abs_diff::abs_diff(a, b);
    (!abs_diff::is_x_larger(abs_diff_result), abs_diff::to_corrected(abs_diff_result))
}

#[test]
fn sign_magnitude_difference_test() {
    // The way we like our ones differencer is to yield a -0 for (a == b), i.e. result.0==(a >= b)
    assert_eq(sign_magnitude_difference(u8:0, u8:0), (true, u8:0));
    assert_eq(sign_magnitude_difference(u8:42, u8:42), (true, u8:0));
    assert_eq(sign_magnitude_difference(u8:255, u8:255), (true, u8:0));

    // Make sure this works for very small width; exhaustive for u1
    assert_eq(sign_magnitude_difference(u1:0, u1:0), (true, u1:0));
    assert_eq(sign_magnitude_difference(u1:1, u1:1), (true, u1:0));
    assert_eq(sign_magnitude_difference(u1:0, u1:1), (true, u1:1));
    assert_eq(sign_magnitude_difference(u1:1, u1:0), (false, u1:1));

    // Exhaustive for u2
    for (left, _): (u2, ()) in u2:0..u2:3 {
        for (right, _): (u2, ()) in u2:0..u2:3 {
            assert_eq(
                sign_magnitude_difference(left, right),
                ((right >= left), if right >= left { right - left } else { left - right }));
        }(());
    }(());

    // Exhaustive for u8
    for (left, _): (u8, ()) in u8:0..u8:255 {
        for (right, _): (u8, ()) in u8:0..u8:255 {
            assert_eq(
                sign_magnitude_difference(left, right),
                ((right >= left), if right >= left { right - left } else { left - right }));
        }(());
    }(());

    // Close to overflow is handled correctly
    assert_eq(sign_magnitude_difference(u8:255, u8:0), (false, u8:255));
    assert_eq(sign_magnitude_difference(u8:255, u8:5), (false, u8:250));
    assert_eq(sign_magnitude_difference(u8:0, u8:255), (true, u8:255));
    assert_eq(sign_magnitude_difference(u8:5, u8:255), (true, u8:250));
}

// Manually apply optimization https://github.com/google/xls/issues/1217
fn or_last_bit<WIDTH: u32>(value: bits[WIDTH], lsb: u1) -> bits[WIDTH] {
    bit_slice_update(value, u1:0, (value[0:1] | lsb))
}

// Floating point addition based on a generalization of IEEE 754 single-precision floating-point
// addition, with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact fraction values in the NaN case.

// The bit widths of different float components are given
// in comments throughout this implementation, listed
// relative to the widths of a standard float32.
pub fn add<EXP_SZ: u32, FRACTION_SZ: u32>
    (a: APFloat<EXP_SZ, FRACTION_SZ>, b: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    // WIDE_EXP: Widened exponent to capture a possible carry bit.
    const WIDE_EXP: u32 = EXP_SZ + u32:1;
    // CARRY_EXP: WIDE_EXP plus one sign bit.
    const CARRY_EXP: u32 = u32:1 + WIDE_EXP;

    // WIDE_FRACTION: Widened fraction to contain full precision + rounding
    // (sign, hidden as well as guard, round, sticky) bits.
    const SIGN_BIT = u32:1;
    const HIDDEN_BIT = u32:1;
    const GUARD_ROUND_STICKY_BITS = u32:3;
    const FRACTION = HIDDEN_BIT + FRACTION_SZ;
    const SIGNED_FRACTION = SIGN_BIT + FRACTION;
    const WIDE_FRACTION: u32 = SIGNED_FRACTION + GUARD_ROUND_STICKY_BITS;
    // CARRY_FRACTION: WIDE_FRACTION plus one bit to capture a possible carry bit.
    const CARRY_FRACTION: u32 = u32:1 + WIDE_FRACTION;

    // NORMALIZED_FRACTION: WIDE_FRACTION minus one bit for post normalization
    // (where the implicit leading 1 bit is dropped).
    const NORMALIZED_FRACTION: u32 = WIDE_FRACTION - u32:1;

    // Step 0: Swap operands for x to contain value with greater exponent
    let (a_is_smaller, shift) = sign_magnitude_difference(a.bexp, b.bexp);
    let (x, y) = if a_is_smaller { (b, a) } else { (a, b) };

    // Step 1: add hidden bit.
    let fraction_x = (u1:1 ++ x.fraction) as uN[FRACTION];
    let fraction_y = (u1:1 ++ y.fraction) as uN[FRACTION];

    // Flush denormals to 0.
    let fraction_x = if x.bexp == uN[EXP_SZ]:0 { uN[FRACTION]:0 } else { fraction_x };
    let fraction_y = if y.bexp == uN[EXP_SZ]:0 { uN[FRACTION]:0 } else { fraction_y };

    // Provide space for guard, round and sticky
    let wide_x = fraction_x as uN[WIDE_FRACTION] << GUARD_ROUND_STICKY_BITS;
    let wide_y = fraction_y as uN[WIDE_FRACTION] << GUARD_ROUND_STICKY_BITS;

    // Shift the smaller fraction to align with the largest exponent.
    // x is already the larger, so no alignment needed which is done for y.
    // Use the extra time on x to negate if we have an effective subtraction.
    let addend_x = wide_x as sN[WIDE_FRACTION];
    let addend_x = if x.sign != y.sign { -addend_x } else { addend_x };

    // The smaller (y) needs to be shifted.
    // Calculate the sticky bit - set to 1 if any set bits have to be shifted
    // shifted out of the fraction.
    let sticky = std::or_reduce_lsb(wide_y, shift);
    let addend_y = or_last_bit(wide_y >> shift, sticky) as sN[WIDE_FRACTION];

    // Step 2: Do some addition!
    // Add one bit to capture potential carry: s28 -> s29.
    let fraction = (addend_x as sN[CARRY_FRACTION]) + (addend_y as sN[CARRY_FRACTION]);
    let fraction_is_zero = fraction == sN[CARRY_FRACTION]:0;
    let result_sign = match (fraction_is_zero, fraction < sN[CARRY_FRACTION]:0) {
        (true, _) => x.sign && y.sign,  // So that -0.0 + -0.0 results in -0.0
        (false, true) => !y.sign,
        _ => y.sign,
    };

    // Get the absolute value of the result then chop off the sign bit: s29 -> u28.
    let abs_fraction =
        (if fraction < sN[CARRY_FRACTION]:0 { -fraction } else { fraction }) as uN[WIDE_FRACTION];

    // Step 3: Normalize the fraction (shift until the leading bit is a 1).
    // If the carry bit is set, shift right one bit (to capture the new bit of
    // precision) - but don't drop the sticky bit!
    let carry_bit = abs_fraction[-1:];
    let carry_fraction = (abs_fraction >> u32:1) as uN[NORMALIZED_FRACTION];
    let carry_fraction = or_last_bit(carry_fraction, abs_fraction[0:1]);

    // If we cancelled higher bits, then we'll need to shift left.
    // Leading zeroes will be 1 if there's no carry or cancellation.
    let leading_zeroes = std::clzt(abs_fraction);

    // Manually apply https://github.com/google/xls/issues/1274
    let cancel_fraction = abs_fraction as uN[WIDE_FRACTION + u32:1] << leading_zeroes;
    let cancel_fraction = (cancel_fraction >> u32:1) as uN[NORMALIZED_FRACTION];
    let shifted_fraction = if carry_bit { carry_fraction } else { cancel_fraction };

    // Step 4: Rounding.
    // Rounding down is a no-op, since we eventually have to shift off
    // the extra precision bits, so we only need to be concerned with
    // rounding up. We only support round to nearest, half to even
    // mode. This means we round up if:
    //  - The last three bits are greater than 1/2 way between
    //    values, i.e., the last three bits are > 0b100.
    //  - We're exactly 1/2 way between values (0b100) and bit 3 is 1
    //    (i.e., 0x...1100). In other words, if we're "halfway", we round
    //    in whichever direction makes the last bit in the fraction 0.
    let normal_chunk = shifted_fraction[0:3];
    let half_way_chunk = shifted_fraction[2:4];
    let do_round_up = (normal_chunk > u3:0x4) || (half_way_chunk == u2:0x3);

    // We again need an extra bit for carry.
    let rounded_fraction = if do_round_up {
        (shifted_fraction as uN[WIDE_FRACTION]) + uN[WIDE_FRACTION]:0x8
    } else {
        shifted_fraction as uN[WIDE_FRACTION]
    };
    let rounding_carry = rounded_fraction[-1:];

    // After rounding, we can chop off the extra precision bits.
    // As with normalization, if we carried, we need to shift right
    // an extra place.
    let fraction_shift =
        GUARD_ROUND_STICKY_BITS as u3 + (if rounded_fraction[-1:] { u3:1 } else { u3:0 });
    let result_fraction = (rounded_fraction >> fraction_shift) as uN[FRACTION_SZ];

    // Finally, adjust the exponent based on addition and rounding -
    // each bit of carry or cancellation moves it by one place.
    let wide_exponent = (x.bexp as sN[CARRY_EXP]) + (rounding_carry as sN[CARRY_EXP]) +
                        sN[CARRY_EXP]:1 - (leading_zeroes as sN[CARRY_EXP]);
    let wide_exponent = if fraction_is_zero { sN[CARRY_EXP]:0 } else { wide_exponent };

    // Chop off the sign bit.
    let wide_exponent = if wide_exponent < sN[CARRY_EXP]:0 {
        uN[WIDE_EXP]:0
    } else {
        wide_exponent as uN[WIDE_EXP]
    };

    // Extra bonus step 5: special case handling!

    // If the exponent underflowed, don't bother with denormals. Just flush to 0.
    let result_fraction =
        if wide_exponent < uN[WIDE_EXP]:1 { uN[FRACTION_SZ]:0 } else { result_fraction };

    // Handle exponent overflow infinities.
    const MAX_EXPONENT = std::mask_bits<EXP_SZ>();
    const SATURATED_EXPONENT = MAX_EXPONENT as uN[WIDE_EXP];
    let result_fraction =
        if wide_exponent < SATURATED_EXPONENT { result_fraction } else { uN[FRACTION_SZ]:0 };
    let result_exponent =
        if wide_exponent < SATURATED_EXPONENT { wide_exponent as uN[EXP_SZ] } else { MAX_EXPONENT };

    // Handle arg infinities.
    let is_operand_inf = is_inf<EXP_SZ, FRACTION_SZ>(x) || is_inf<EXP_SZ, FRACTION_SZ>(y);
    let result_exponent = if is_operand_inf { MAX_EXPONENT } else { result_exponent };
    let result_fraction = if is_operand_inf { uN[FRACTION_SZ]:0 } else { result_fraction };
    // Result infinity is negative iff all infinite operands are neg.
    let has_pos_inf = (is_inf<EXP_SZ, FRACTION_SZ>(x) && (x.sign == u1:0)) ||
                      (is_inf<EXP_SZ, FRACTION_SZ>(y) && (y.sign == u1:0));
    let result_sign = if is_operand_inf { !has_pos_inf } else { result_sign };

    // Handle NaN; NaN trumps infinities, so we handle it last.
    // -inf + inf = NaN, i.e., if we have both positive and negative inf.
    let has_neg_inf = (is_inf<EXP_SZ, FRACTION_SZ>(x) && (x.sign == u1:1)) ||
                      (is_inf<EXP_SZ, FRACTION_SZ>(y) && (y.sign == u1:1));
    let is_result_nan = is_nan<EXP_SZ, FRACTION_SZ>(x) || is_nan<EXP_SZ, FRACTION_SZ>(y) ||
                        (has_pos_inf && has_neg_inf);
    const FRACTION_HIGH_BIT = uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:1);
    let result_exponent = if is_result_nan { MAX_EXPONENT } else { result_exponent };
    let result_fraction = if is_result_nan { FRACTION_HIGH_BIT } else { result_fraction };
    let result_sign = if is_result_nan { u1:0 } else { result_sign };

    // Finally (finally!), construct the output float.
    APFloat<EXP_SZ, FRACTION_SZ> {
        sign: result_sign,
        bexp: result_exponent,
        fraction: result_fraction as uN[FRACTION_SZ],
    }
}

// IEEE floating-point subtraction (and comparisons that are implemented using subtraction),
// with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact fraction values in the NaN case).
pub fn sub<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    let y = APFloat<EXP_SZ, FRACTION_SZ> { sign: !y.sign, bexp: y.bexp, fraction: y.fraction };
    add(x, y)
}

// add is thoroughly tested elsewhere so a few simple tests is sufficient.
#[test]
fn test_sub() {
    let one = one<u32:8, u32:23>(u1:0);
    let two = add<u32:8, u32:23>(one, one);
    let neg_two = APFloat<u32:8, u32:23> { sign: u1:1, ..two };
    let three = add<u32:8, u32:23>(one, two);
    let four = add<u32:8, u32:23>(two, two);

    assert_eq(sub(four, one), three);
    assert_eq(sub(four, two), two);
    assert_eq(sub(four, three), one);
    assert_eq(sub(three, two), one);
    assert_eq(sub(two, four), neg_two);
}

// Returns the product of `x` and `y`, with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact fraction values in the NaN case).
pub fn mul<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    // WIDE_EXP: Widened exponent to capture a possible carry bit.
    const WIDE_EXP: u32 = EXP_SZ + u32:1;
    // SIGNED_EXP: WIDE_EXP plus one sign bit.
    const SIGNED_EXP: u32 = WIDE_EXP + u32:1;

    // ROUNDING_FRACTION: Result fraction with one extra bit to capture
    // potential carry if rounding up.
    const ROUNDING_FRACTION: u32 = FRACTION_SZ + u32:1;

    // WIDE_FRACTION: Widened fraction to contain full precision + rounding
    // (guard & sticky) bits.
    const WIDE_FRACTION: u32 = FRACTION_SZ + FRACTION_SZ + u32:2;
    // FRACTION_ROUNDING_BIT: Position of the first rounding bit in the "wide" FRACTION.
    const FRACTION_ROUNDING_BIT: u32 = FRACTION_SZ - u32:1;

    // STICKY_FRACTION: Location of the sticky bit in the wide FRACTION (same as
    // "ROUNDING_FRACTION", but it's easier to understand the code if it has its own name).
    const STICKY_FRACTION: u32 = FRACTION_SZ + u32:1;

    // 0. Check if either operand is 0 (flushing subnorms to 0).
    let has_0_arg = is_zero_or_subnormal(x) || is_zero_or_subnormal(y);

    // 1. Get and expand mantissas.
    let x_fraction = (x.fraction as uN[WIDE_FRACTION]) |
                     (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));
    let y_fraction = (y.fraction as uN[WIDE_FRACTION]) |
                     (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));

    // 2. Multiply integer mantissas, flushing subnormals to 0.
    let fraction = if has_0_arg { uN[WIDE_FRACTION]:0 } else { x_fraction * y_fraction };

    // 3. Add non-biased exponents.
    //  - Remove the bias from the exponents, add them, then restore the bias.
    //  - Simplifies from
    //      (A - 127) + (B - 127) + 127 = exp
    //    to
    //      A + B - 127 = exp
    let bias = std::mask_bits<EXP_SZ>() as sN[SIGNED_EXP] >> uN[SIGNED_EXP]:1;
    let exp = (x.bexp as sN[SIGNED_EXP]) + (y.bexp as sN[SIGNED_EXP]) - bias;

    // Here is where we'd handle subnormals if we cared to.
    // If the exponent remains < 0, even after reapplying the bias,
    // then we'd calculate the extra exponent needed to get back to 0.
    // We'd set the result exponent to 0 and shift the fraction to the right
    // to capture that "extra" exponent.
    // Since we just flush subnormals, we don't have to do any of that.
    // Instead, if we're multiplying by 0, the result is 0.
    let exp = if has_0_arg { sN[SIGNED_EXP]:0 } else { exp };

    // 4. Normalize. Adjust the fraction until our leading 1 is
    // bit 47 (the first past the 46 bits of actual fraction).
    // That'll be a shift of 1 or 0 places (since we're multiplying
    // two values with leading 1s in bit 24).
    let fraction_shift = fraction[-1:] as uN[WIDE_FRACTION];

    // If there is a leading 1, then we need to shift to the right one place -
    // that means we gained a new significant digit at the top.
    // Dont forget to maintain the sticky bit!
    let sticky = fraction[0:1] as uN[WIDE_FRACTION];
    let fraction = fraction >> fraction_shift;
    let fraction = fraction | sticky;

    // Update the exponent if we shifted.
    let exp = exp + (fraction_shift as sN[SIGNED_EXP]);

    // If the value is currently subnormal, then we need to shift right by one
    // space: a subnormal value doesn't have the leading 1, and thus has one
    // fewer significant digits than normal numbers - in a sense, the -1th bit
    // is the least significant (0) bit.
    // Rounding (below) expects the least significant digit to start at position
    // 0, so we shift subnormals to the left by one position to match normals.
    // Again, track the sticky bit. This could be combined with the shift
    // above, but it's easier to understand (and comment) if separated, and the
    // optimizer will clean it up anyway.
    let sticky = fraction[0:1] as uN[WIDE_FRACTION];
    let fraction = if exp <= sN[SIGNED_EXP]:0 { fraction >> uN[WIDE_FRACTION]:1 } else { fraction };
    let fraction = fraction | sticky;

    // 5. Round - we use nearest, half to even rounding.
    // - We round down if less than 1/2 way between values, i.e.
    //   if bit 23 is 0. Rounding down is equivalent to doing nothing.
    // - We round up if we're more than 1/2 way, i.e., if bit 23
    //   is set along with any bit lower than 23.
    // - If halfway (bit 23 set and no bit lower), then we round;
    //   whichever direction makes the result even. In other words,
    //   we round up if bit 25 is set.
    let is_half_way = fraction[FRACTION_ROUNDING_BIT as s32:FRACTION_SZ as s32] &
                      (fraction[0:FRACTION_ROUNDING_BIT as s32] == uN[FRACTION_ROUNDING_BIT]:0);
    let greater_than_half_way = fraction[FRACTION_ROUNDING_BIT as s32:FRACTION_SZ as s32] &
                                (fraction[0:FRACTION_ROUNDING_BIT as s32] !=
                                uN[FRACTION_ROUNDING_BIT]:0);
    let do_round_up = greater_than_half_way ||
                      (is_half_way & fraction[FRACTION_SZ as s32:STICKY_FRACTION as s32]);

    // We're done with the extra precision bits now, so shift the
    // fraction into its almost-final width, adding one extra
    // bit for potential rounding overflow.
    let fraction = (fraction >> (FRACTION_SZ as uN[WIDE_FRACTION])) as uN[FRACTION_SZ];
    let fraction = fraction as uN[ROUNDING_FRACTION];
    let fraction = if do_round_up { fraction + uN[ROUNDING_FRACTION]:1 } else { fraction };

    // Adjust the exponent if we overflowed during rounding.
    // After checking for subnormals, we don't need the sign bit anymore.
    let exp = if fraction[-1:] { exp + sN[SIGNED_EXP]:1 } else { exp };
    let is_subnormal = exp <= sN[SIGNED_EXP]:0;

    // We're done - except for special cases...
    let result_sign = x.sign != y.sign;
    let result_exp = exp as uN[WIDE_EXP];
    let result_fraction = fraction as uN[FRACTION_SZ];

    // 6. Special cases!
    // - Subnormals: flush to 0.
    let result_exp = if is_subnormal { uN[WIDE_EXP]:0 } else { result_exp };
    let result_fraction = if is_subnormal { uN[FRACTION_SZ]:0 } else { result_fraction };

    // - Overflow infinites - saturate exp, clear fraction.
    let high_exp = std::mask_bits<EXP_SZ>();
    let result_fraction =
        if result_exp < (high_exp as uN[WIDE_EXP]) { result_fraction } else { uN[FRACTION_SZ]:0 };
    let result_exp =
        if result_exp < (high_exp as uN[WIDE_EXP]) { result_exp as uN[EXP_SZ] } else { high_exp };

    // - Arg infinites. Any arg is infinite == result is infinite.
    let is_operand_inf = is_inf<EXP_SZ, FRACTION_SZ>(x) || is_inf<EXP_SZ, FRACTION_SZ>(y);
    let result_exp = if is_operand_inf { high_exp } else { result_exp };
    let result_fraction = if is_operand_inf { uN[FRACTION_SZ]:0 } else { result_fraction };

    // - NaNs. NaN trumps infinities, so we handle it last.
    //   inf * 0 = NaN, i.e.,
    let has_nan_arg = is_nan<EXP_SZ, FRACTION_SZ>(x) || is_nan<EXP_SZ, FRACTION_SZ>(y);
    let has_inf_arg = is_inf<EXP_SZ, FRACTION_SZ>(x) || is_inf<EXP_SZ, FRACTION_SZ>(y);
    let is_result_nan = has_nan_arg || (has_0_arg && has_inf_arg);
    let result_exp = if is_result_nan { high_exp } else { result_exp };
    let nan_fraction = uN[FRACTION_SZ]:1 << (FRACTION_SZ as uN[FRACTION_SZ] - uN[FRACTION_SZ]:1);
    let result_fraction = if is_result_nan { nan_fraction } else { result_fraction };
    let result_sign = if is_result_nan { u1:0 } else { result_sign };

    APFloat<EXP_SZ, FRACTION_SZ> { sign: result_sign, bexp: result_exp, fraction: result_fraction }
}

// Simple utility struct for holding the result of the multiplication step.
struct Product<EXP_CARRY: u32, WIDE_FRACTION: u32> {
    sign: u1,
    bexp: uN[EXP_CARRY],
    fraction: uN[WIDE_FRACTION],
}

// Returns true if the given Product is infinite.
fn is_product_inf<EXP_CARRY: u32, WIDE_FRACTION: u32>
    (p: Product<EXP_CARRY, WIDE_FRACTION>) -> bool {
    p.bexp == std::mask_bits<EXP_CARRY>() && p.fraction == uN[WIDE_FRACTION]:0
}

// Returns true if the given Product is NaN.
fn is_product_nan<EXP_CARRY: u32, WIDE_FRACTION: u32>
    (p: Product<EXP_CARRY, WIDE_FRACTION>) -> bool {
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
fn mul_no_round
    <EXP_SZ: u32, FRACTION_SZ: u32, WIDE_FRACTION: u32 = {(FRACTION_SZ + u32:1) * u32:2},
     EXP_CARRY: u32 = {EXP_SZ + u32:1}, EXP_SIGN_CARRY: u32 = {EXP_SZ + u32:2}>
    (a: APFloat<EXP_SZ, FRACTION_SZ>, b: APFloat<EXP_SZ, FRACTION_SZ>) -> Product {
    // These steps are taken from apfloat_mul_2.x; look there for full comments.
    // Widen the fraction to full size and prepend the formerly-implicit "1".
    let a_fraction = (a.fraction as uN[WIDE_FRACTION]) | (uN[WIDE_FRACTION]:1 << FRACTION_SZ);
    let b_fraction = (b.fraction as uN[WIDE_FRACTION]) | (uN[WIDE_FRACTION]:1 << FRACTION_SZ);

    let has_0_arg = a.bexp == uN[EXP_SZ]:0 || b.bexp == uN[EXP_SZ]:0;

    // Flush subnorms, and multiply.
    let fraction = if has_0_arg { uN[WIDE_FRACTION]:0 } else { a_fraction * b_fraction };

    // Normalize - shift left one place if the top bit is 0.
    let fraction_shift = fraction[-1:] as uN[WIDE_FRACTION];
    let fraction = if fraction_shift == uN[WIDE_FRACTION]:0 { fraction << 1 } else { fraction };

    // e.g., for floats, 0xff -> 0x7f, A.K.A. 127, the exponent bias.
    let bias = std::signed_max_value<EXP_SZ>() as sN[EXP_SIGN_CARRY];
    let bexp = (a.bexp as sN[EXP_SIGN_CARRY]) + (b.bexp as sN[EXP_SIGN_CARRY]) - bias +
               (fraction_shift as sN[EXP_SIGN_CARRY]);
    let bexp = if a.bexp == bits[EXP_SZ]:0 || b.bexp == bits[EXP_SZ]:0 {
        sN[EXP_SIGN_CARRY]:0
    } else {
        bexp
    };

    // Note that we usually flush subnormals. Here, we preserve what we can for
    // compatability with reference implementations.
    // We only do this for the internal product - we otherwise don't handle
    // subnormal values (we flush them to 0).
    let is_subnormal = bexp <= sN[EXP_SIGN_CARRY]:0;
    let result_exp = if is_subnormal { uN[EXP_CARRY]:0 } else { bexp as uN[EXP_CARRY] };
    let sub_exp = std::abs(bexp) as uN[EXP_CARRY];
    let result_fraction = if is_subnormal { fraction >> sub_exp } else { fraction };

    // - Overflow infinites - saturate exp, clear fraction.
    let high_exp = std::mask_bits<EXP_CARRY>();
    let result_fraction = if result_exp < high_exp { result_fraction } else { uN[WIDE_FRACTION]:0 };
    let result_exp = if result_exp < high_exp { result_exp as uN[EXP_CARRY] } else { high_exp };

    // - Arg infinites. Any arg is infinite == result is infinite.
    let is_operand_inf = is_inf(a) || is_inf(b);
    let result_exp = if is_operand_inf { high_exp } else { result_exp };
    let result_fraction =
        if is_operand_inf { uN[WIDE_FRACTION]:0 } else { result_fraction as uN[WIDE_FRACTION] };

    // - NaNs. NaN trumps infinities, so we handle it last.
    //   inf * 0 = NaN, i.e.,
    let has_nan_arg = is_nan(a) || is_nan(b);
    let has_inf_arg = is_inf(a) || is_inf(b);
    let is_result_nan = has_nan_arg || (has_0_arg && has_inf_arg);
    let result_exp = if is_result_nan { high_exp } else { result_exp };
    let nan_fraction = uN[WIDE_FRACTION]:1 << (uN[WIDE_FRACTION]:1 - uN[WIDE_FRACTION]:1);
    let result_fraction = if is_result_nan { nan_fraction } else { result_fraction };

    let result_sign = a.sign != b.sign;
    let result_sign = if is_result_nan { u1:0 } else { result_sign };

    Product { sign: result_sign, bexp: result_exp, fraction: result_fraction }
}

// Fused multiply-add for any given APFloat configuration.
//
// This implementation uses (2 * (FRACTION + 1)) bits of precision for the
// multiply fraction and (3 * (FRACTION + 1)) for the add.
// The results have been tested (not exhaustively, of course! It's a 96-bit
// input space for binary32!) to be bitwise identical to those produced by
// glibc/libm 2.31 (for IEEE binary32 formats).
//
// The fundamentals of the multiply and add are the same as those in the
// standalone ops - the differences arise in the extra precision bits and the
// handling thereof (e.g., 72 vs. 24 bits for the add, for binary32).
//
// Many of the steps herein are fully described in the standalone adder or
// multiplier modules, but abridged comments are present here where useful.
pub fn fma<EXP_SZ: u32, FRACTION_SZ: u32>
    (a: APFloat<EXP_SZ, FRACTION_SZ>, b: APFloat<EXP_SZ, FRACTION_SZ>,
     c: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    // EXP_CARRY: One greater than EXP_SZ, to hold a carry bit.
    const EXP_CARRY: u32 = EXP_SZ + u32:1;
    // EXP_SIGN_CARRY: One greater than EXP_CARRY, to hold a sign bit.
    const EXP_SIGN_CARRY: u32 = EXP_CARRY + u32:1;
    // WIDE_FRACTION: Fully-widened fraction to hold all rounding bits.
    const WIDE_FRACTION: u32 = (FRACTION_SZ + u32:1) * u32:3 + u32:1;
    // WIDE_FRACTION_CARRY: WIDE_FRACTION plus one carry bit.
    const WIDE_FRACTION_CARRY: u32 = WIDE_FRACTION + u32:1;
    // WIDE_FRACTION_SIGN_CARRY: WIDE_FRACTION_CARRY plus one sign bit.
    const WIDE_FRACTION_SIGN_CARRY: u32 = WIDE_FRACTION_CARRY + u32:1;

    // WIDE_FRACTION_LOW_BIT: Position of the LSB in the final fraction within a
    // WIDE_FRACTION element. All bits with lower index are for rounding.
    const WIDE_FRACTION_LOW_BIT: u32 = WIDE_FRACTION - FRACTION_SZ;

    // WIDE_FRACTION_TOP_ROUNDING: One less than WIDE_FRACTION_LOW_BIT, in other words the
    // most-significant rounding bit.
    const WIDE_FRACTION_TOP_ROUNDING: u32 = WIDE_FRACTION_LOW_BIT - u32:1;

    let ab = mul_no_round<EXP_SZ, FRACTION_SZ>(a, b);

    let (ab_exp_smaller, exp_difference) =
        sign_magnitude_difference(ab.bexp, c.bexp as uN[EXP_CARRY]);
    let (greater_exp, greater_sign) =
        if ab_exp_smaller { (c.bexp as uN[EXP_CARRY], c.sign) } else { (ab.bexp, ab.sign) };

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
    // The sticky bits are set to whether anything non-zero gets shifted out.
    let rshift_ab = if ab_exp_smaller { exp_difference } else { uN[EXP_CARRY]:0 };
    let rshift_c = if ab_exp_smaller { uN[EXP_CARRY]:0 } else { exp_difference };
    let shifted_ab = wide_ab >> rshift_ab;
    let shifted_c = wide_c >> rshift_c;
    let sticky_ab = std::or_reduce_lsb(wide_ab, rshift_ab) as uN[WIDE_FRACTION];
    let sticky_c = std::or_reduce_lsb(wide_c, rshift_c) as uN[WIDE_FRACTION];

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
    let abs_fraction = (if sum_fraction < sN[WIDE_FRACTION_SIGN_CARRY]:0 {
                           -sum_fraction
                       } else {
                           sum_fraction
                       }) as
                       uN[WIDE_FRACTION_CARRY];

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
    let shifted_fraction = match (carry_bit, cancel) {
        (true, false) => carry_fraction,
        (false, true) => cancel_fraction,
        (false, false) => abs_fraction as uN[WIDE_FRACTION],
        _ => fail!("carry_and_cancel", uN[WIDE_FRACTION]:0),
    };

    // Similar to the rounding in apfloat_add_2, except that the fraction
    // starts at the bit below instead of bit 3.
    // For binary32, normal_chunk will be bits 0-48 (inclusive), stopping
    // immediately below the first bit in the final fraction.
    let normal_chunk = shifted_fraction[0:(WIDE_FRACTION_LOW_BIT - u32:1) as s32];
    let half_way_chunk =
        shifted_fraction[(WIDE_FRACTION_LOW_BIT - u32:2) as s32:(WIDE_FRACTION_LOW_BIT as s32)];
    let half_of_extra = uN[WIDE_FRACTION_TOP_ROUNDING]:1 << (WIDE_FRACTION_LOW_BIT - u32:2);
    let do_round_up =
        if (normal_chunk > half_of_extra) | (half_way_chunk == u2:0x3) { u1:1 } else { u1:0 };
    let rounded_fraction = if do_round_up {
        shifted_fraction as uN[WIDE_FRACTION_CARRY] +
        (uN[WIDE_FRACTION_CARRY]:1 << (WIDE_FRACTION_LOW_BIT - u32:1))
    } else {
        shifted_fraction as uN[WIDE_FRACTION_CARRY]
    };

    let rounding_carry = rounded_fraction[-1:];
    let result_fraction = (rounded_fraction >>
                          ((WIDE_FRACTION_LOW_BIT - u32:1) as uN[WIDE_FRACTION_CARRY])) as
                          uN[FRACTION_SZ];

    let bexp = greater_exp as sN[EXP_SIGN_CARRY] + rounding_carry as sN[EXP_SIGN_CARRY] +
               sN[EXP_SIGN_CARRY]:1 - leading_zeroes as sN[EXP_SIGN_CARRY];
    let bexp = if fraction_is_zero { sN[EXP_SIGN_CARRY]:0 } else { bexp };
    let bexp = if bexp < sN[EXP_SIGN_CARRY]:0 { uN[EXP_CARRY]:0 } else { (bexp as uN[EXP_CARRY]) };

    // Standard special case handling follows.

    // If the exponent underflowed, don't bother with denormals. Just flush to 0.
    let result_fraction = if bexp == uN[EXP_CARRY]:0 { uN[FRACTION_SZ]:0 } else { result_fraction };

    // Handle exponent overflow infinities.
    let saturated_exp = std::mask_bits<EXP_SZ>() as uN[EXP_CARRY];
    let max_exp = std::mask_bits<EXP_SZ>();
    let result_fraction = if bexp < saturated_exp { result_fraction } else { uN[FRACTION_SZ]:0 };
    let result_exp = if bexp < saturated_exp { bexp as uN[EXP_SZ] } else { max_exp };

    // Handle arg infinities.
    let is_operand_inf = is_product_inf(ab) | is_inf(c);
    let result_exp = if is_operand_inf { max_exp } else { result_exp };
    let result_fraction = if is_operand_inf { uN[FRACTION_SZ]:0 } else { result_fraction };
    // Result infinity is negative iff all infinite operands are neg.
    let has_pos_inf = (is_product_inf(ab) & (ab.sign == u1:0)) | (is_inf(c) & (c.sign == u1:0));
    let result_sign = if is_operand_inf { !has_pos_inf } else { result_sign };

    // Handle NaN; NaN trumps infinities, so we handle it last.
    // -inf + inf = NaN, i.e., if we have both positive and negative inf.
    let has_neg_inf = (is_product_inf(ab) & (ab.sign == u1:1)) | (is_inf(c) & (c.sign == u1:1));
    let is_result_nan = is_product_nan(ab) | is_nan(c) | (has_pos_inf & has_neg_inf);
    let result_exp = if is_result_nan { max_exp } else { result_exp };
    let result_fraction =
        if is_result_nan { uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:4) } else { result_fraction };
    let result_sign = if is_result_nan { u1:0 } else { result_sign };
    let is_result_inf = has_pos_inf | has_neg_inf;

    APFloat<EXP_SZ, FRACTION_SZ> {
        sign: result_sign,
        bexp: result_exp as uN[EXP_SZ],
        fraction: result_fraction as uN[FRACTION_SZ],
    }
}

#[test]
fn smoke() {
    type F32 = APFloat<8, 23>;
    let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23:0 };
    let one_point_one = F32 { sign: u1:0, bexp: u8:127, fraction: u23:0xccccd };
    let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23:0x58cccd };
    let a = twenty_seven_point_one;
    let b = one_point_one;
    let c = zero;
    let expected = F32 { sign: u1:0, bexp: u8:0x83, fraction: u23:0x6e7ae2 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_f32() {
    type F32 = APFloat<8, 23>;
    let one_point_zero = F32 { sign: u1:0, bexp: u8:127, fraction: u23:0 };
    let a = one_point_zero;
    let b = one_point_zero;
    let c = one_point_zero;
    let expected = F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_f64() {
    type F64 = APFloat<11, 52>;
    let one_point_zero = F64 { sign: u1:0, bexp: u11:1023, fraction: u52:0 };
    let a = one_point_zero;
    let b = one_point_zero;
    let c = one_point_zero;
    let expected = F64 { sign: u1:0, bexp: u11:1024, fraction: u52:0 };
    let actual = fma<u32:11, u32:52>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_bf16() {
    type BF16 = APFloat<8, 7>;
    let one_point_zero = BF16 { sign: u1:0, bexp: u8:127, fraction: u7:0 };
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
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0, bexp: u8:0x97, fraction: u23:0x565d43 };
    let b = F32 { sign: u1:1, bexp: u8:0x77, fraction: u23:0x319a49 };
    let c = F32 { sign: u1:0, bexp: u8:0x87, fraction: u23:0x642891 };
    let expected = F32 { sign: u1:1, bexp: u8:0x90, fraction: u23:0x144598 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn manual_case_a_f64() {
    type F64 = APFloat<11, 52>;
    let a = F64 { sign: u1:0, bexp: u11:0x417, fraction: u52:0x565d43 };
    let b = F64 { sign: u1:1, bexp: u11:0x3f7, fraction: u52:0x319a49 };
    let c = F64 { sign: u1:0, bexp: u11:0x407, fraction: u52:0x642891 };
    let expected = F64 { sign: u1:1, bexp: u11:0x40e, fraction: u52:0xfe000010f26c7 };
    let actual = fma<u32:11, u32:52>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn manual_case_a_bf16() {
    type BF16 = APFloat<8, 7>;
    let a = BF16 { sign: u1:0, bexp: u8:0x97, fraction: u7:0x2b };
    let b = BF16 { sign: u1:1, bexp: u8:0x77, fraction: u7:0x18 };
    let c = BF16 { sign: u1:0, bexp: u8:0x87, fraction: u7:0x32 };
    let expected = BF16 { sign: u1:1, bexp: u8:0x8f, fraction: u7:0x4a };
    let actual = fma<u32:8, u32:7>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_zero() {
    type F32 = APFloat<8, 23>;
    let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23:0 };
    let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23:0x58cccd };
    let a = twenty_seven_point_one;
    let b = twenty_seven_point_one;
    let c = zero;
    let expected = F32 { sign: u1:0, bexp: u8:0x88, fraction: u23:0x379a3e };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one() {
    type F32 = APFloat<8, 23>;
    let one_point_zero = F32 { sign: u1:0, bexp: u8:127, fraction: u23:0 };
    let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23:0x58cccd };
    let a = twenty_seven_point_one;
    let b = twenty_seven_point_one;
    let c = one_point_zero;
    let expected = F32 { sign: u1:0, bexp: u8:0x88, fraction: u23:0x37da3e };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one_point_one() {
    type F32 = APFloat<8, 23>;
    let one_point_one = F32 { sign: u1:0, bexp: u8:127, fraction: u23:0xccccd };
    let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23:0x58cccd };
    let a = twenty_seven_point_one;
    let b = twenty_seven_point_one;
    let c = one_point_one;
    let expected = F32 { sign: u1:0, bexp: u8:0x88, fraction: u23:0x37e0a4 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_a() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x50, fraction: u23:0x1a8ddc };
    let b = F32 { sign: u1:0x1, bexp: u8:0xcb, fraction: u23:0xee7ac };
    let c = F32 { sign: u1:0x1, bexp: u8:0xb7, fraction: u23:0x609f18 };
    let expected = F32 { sign: u1:1, bexp: u8:0xb7, fraction: u23:0x609f18 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_b() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0x23, fraction: u23:0x4d3a41 };
    let b = F32 { sign: u1:0x0, bexp: u8:0x30, fraction: u23:0x35a901 };
    let c = F32 { sign: u1:0x0, bexp: u8:0x96, fraction: u23:0x627c62 };
    let expected = F32 { sign: u1:0, bexp: u8:0x96, fraction: u23:0x627c62 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_c() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x71, fraction: u23:0x2f0932 };
    let b = F32 { sign: u1:0x0, bexp: u8:0xe5, fraction: u23:0x416b76 };
    let c = F32 { sign: u1:0x0, bexp: u8:0xcb, fraction: u23:0x5fd32a };
    let expected = F32 { sign: u1:1, bexp: u8:0xd8, fraction: u23:0x4386a };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_d() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0xac, fraction: u23:0x1d0d22 };
    let b = F32 { sign: u1:0x0, bexp: u8:0xdb, fraction: u23:0x2fe688 };
    let c = F32 { sign: u1:0x0, bexp: u8:0xa9, fraction: u23:0x2be1d2 };
    let expected = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x0 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_e() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0x7b, fraction: u23:0x25e79f };
    let b = F32 { sign: u1:0x1, bexp: u8:0xff, fraction: u23:0x207370 };
    let c = F32 { sign: u1:0x1, bexp: u8:0x39, fraction: u23:0x6bb348 };
    let expected = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x80000 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_f() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0xe0, fraction: u23:0x3cdaa8 };
    let b = F32 { sign: u1:0x1, bexp: u8:0x96, fraction: u23:0x52549c };
    let c = F32 { sign: u1:0x0, bexp: u8:0x1c, fraction: u23:0x21e0fd };
    let expected = F32 { sign: u1:0, bexp: u8:0xf8, fraction: u23:0x1b29c9 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_g() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0xc4, fraction: u23:0x73b59a };
    let b = F32 { sign: u1:0x0, bexp: u8:0xa6, fraction: u23:0x1631c0 };
    let c = F32 { sign: u1:0x0, bexp: u8:0x29, fraction: u23:0x5b3d33 };
    let expected = F32 { sign: u1:1, bexp: u8:0xec, fraction: u23:0xefbc5 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_h() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x9b, fraction: u23:0x3f50d4 };
    let b = F32 { sign: u1:0x0, bexp: u8:0x7b, fraction: u23:0x4beeb5 };
    let c = F32 { sign: u1:0x1, bexp: u8:0x37, fraction: u23:0x6ad17c };
    let expected = F32 { sign: u1:1, bexp: u8:0x98, fraction: u23:0x18677d };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_i() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0x66, fraction: u23:0x36e592 };
    let b = F32 { sign: u1:0x0, bexp: u8:0xc8, fraction: u23:0x2b5bf1 };
    let c = F32 { sign: u1:0x0, bexp: u8:0x52, fraction: u23:0x12900b };
    let expected = F32 { sign: u1:0, bexp: u8:0xaf, fraction: u23:0x74da11 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_j() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x88, fraction: u23:0x0f0e03 };
    let b = F32 { sign: u1:0x1, bexp: u8:0xb9, fraction: u23:0x36006d };
    let c = F32 { sign: u1:0x1, bexp: u8:0xaa, fraction: u23:0x358b6b };
    let expected = F32 { sign: u1:0, bexp: u8:0xc2, fraction: u23:0x4b6865 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_k() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x29, fraction: u23:0x2fd76d };
    let b = F32 { sign: u1:0x1, bexp: u8:0xce, fraction: u23:0x63eded };
    let c = F32 { sign: u1:0x0, bexp: u8:0xfd, fraction: u23:0x21adee };
    let expected = F32 { sign: u1:0, bexp: u8:0xfd, fraction: u23:0x21adee };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_l() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0x6a, fraction: u23:0x09c1b9 };
    let b = F32 { sign: u1:0x1, bexp: u8:0x7c, fraction: u23:0x666a52 };
    let c = F32 { sign: u1:0x1, bexp: u8:0x80, fraction: u23:0x626bcf };
    let expected = F32 { sign: u1:1, bexp: u8:0x80, fraction: u23:0x626bcf };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_m() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x70, fraction: u23:0x41e2db };
    let b = F32 { sign: u1:0x1, bexp: u8:0xd1, fraction: u23:0x013c17 };
    let c = F32 { sign: u1:0x0, bexp: u8:0xb9, fraction: u23:0x30313f };
    let expected = F32 { sign: u1:0, bexp: u8:0xc2, fraction: u23:0x4419bf };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_n() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x33, fraction: u23:0x537374 };
    let b = F32 { sign: u1:0x0, bexp: u8:0x40, fraction: u23:0x78fa62 };
    let c = F32 { sign: u1:0x1, bexp: u8:0x09, fraction: u23:0x7cfb29 };
    let expected = F32 { sign: u1:1, bexp: u8:0x09, fraction: u23:0x7cfb36 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_o() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0x94, fraction: u23:0x1aeb90 };
    let b = F32 { sign: u1:0x1, bexp: u8:0x88, fraction: u23:0x1ab376 };
    let c = F32 { sign: u1:0x1, bexp: u8:0x9d, fraction: u23:0x15dd1e };
    let expected = F32 { sign: u1:1, bexp: u8:0x9e, fraction: u23:0x288cde };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_p() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0x88, fraction: u23:0x1ebb00 };
    let b = F32 { sign: u1:0x1, bexp: u8:0xf6, fraction: u23:0x0950b6 };
    let c = F32 { sign: u1:0x0, bexp: u8:0xfd, fraction: u23:0x6c314b };
    let expected = F32 { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x5e77d4 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_q() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0xda, fraction: u23:0x5b328f };
    let b = F32 { sign: u1:0x1, bexp: u8:0x74, fraction: u23:0x157da3 };
    let c = F32 { sign: u1:0x0, bexp: u8:0x1b, fraction: u23:0x6a3f25 };
    let expected = F32 { sign: u1:1, bexp: u8:0xd0, fraction: u23:0x000000 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_r() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x34, fraction: u23:0x4da000 };
    let b = F32 { sign: u1:0x0, bexp: u8:0xf4, fraction: u23:0x4bc400 };
    let c = F32 { sign: u1:0x1, bexp: u8:0x33, fraction: u23:0x54476d };
    let expected = F32 { sign: u1:1, bexp: u8:0xaa, fraction: u23:0x23ab4f };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_s() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x27, fraction: u23:0x732d83 };
    let b = F32 { sign: u1:0x1, bexp: u8:0xbb, fraction: u23:0x4b2dcd };
    let c = F32 { sign: u1:0x0, bexp: u8:0x3a, fraction: u23:0x65e4bd };
    let expected = F32 { sign: u1:0, bexp: u8:0x64, fraction: u23:0x410099 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_t() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0x17, fraction: u23:0x070770 };
    let b = F32 { sign: u1:0x1, bexp: u8:0x86, fraction: u23:0x623b39 };
    let c = F32 { sign: u1:0x0, bexp: u8:0x1e, fraction: u23:0x6ea761 };
    let expected = F32 { sign: u1:1, bexp: u8:0x0c, fraction: u23:0x693bc0 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_u() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0xb1, fraction: u23:0x0c8800 };
    let b = F32 { sign: u1:0x1, bexp: u8:0xc6, fraction: u23:0x2b3800 };
    let c = F32 { sign: u1:0x0, bexp: u8:0x22, fraction: u23:0x00c677 };
    let expected = F32 { sign: u1:1, bexp: u8:0xf8, fraction: u23:0x3bfb2b };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_v() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0x90, fraction: u23:0x04a800 };
    let b = F32 { sign: u1:0x1, bexp: u8:0x1f, fraction: u23:0x099cb0 };
    let c = F32 { sign: u1:0x0, bexp: u8:0x28, fraction: u23:0x4d6497 };
    let expected = F32 { sign: u1:1, bexp: u8:0x30, fraction: u23:0x0dd0cf };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_w() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0x90, fraction: u23:0x0fdde1 };
    let b = F32 { sign: u1:0x0, bexp: u8:0xa8, fraction: u23:0x663085 };
    let c = F32 { sign: u1:0x0, bexp: u8:0x9b, fraction: u23:0x450d69 };
    let expected = F32 { sign: u1:0, bexp: u8:0xba, fraction: u23:0x015c9c };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_x() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x4c, fraction: u23:0x5ca821 };
    let b = F32 { sign: u1:0x0, bexp: u8:0x87, fraction: u23:0x14808c };
    let c = F32 { sign: u1:0x0, bexp: u8:0x1c, fraction: u23:0x585ccf };
    let expected = F32 { sign: u1:1, bexp: u8:0x55, fraction: u23:0x000000 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_y() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x0, bexp: u8:0xc5, fraction: u23:0x3a123b };
    let b = F32 { sign: u1:0x1, bexp: u8:0x3b, fraction: u23:0x7ee4d9 };
    let c = F32 { sign: u1:0x0, bexp: u8:0x4f, fraction: u23:0x1d4ddc };
    let expected = F32 { sign: u1:1, bexp: u8:0x82, fraction: u23:0x39446d };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_z() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0xd4, fraction: u23:0x1b858b };
    let b = F32 { sign: u1:0x1, bexp: u8:0x9e, fraction: u23:0x59fa23 };
    let c = F32 { sign: u1:0x1, bexp: u8:0xb5, fraction: u23:0x3520e4 };
    let expected = F32 { sign: u1:0x0, bexp: u8:0xf4, fraction: u23:0x046c29 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn fail_case_aa() {
    type F32 = APFloat<8, 23>;
    let a = F32 { sign: u1:0x1, bexp: u8:0x9b, fraction: u23:0x3ac78d };
    let b = F32 { sign: u1:0x0, bexp: u8:0x3b, fraction: u23:0x542cbb };
    let c = F32 { sign: u1:0x1, bexp: u8:0x09, fraction: u23:0x0c609e };
    let expected = F32 { sign: u1:0x1, bexp: u8:0x58, fraction: u23:0x1acde3 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

// Returns whether or not the given APFloat has a fractional part.
pub fn has_fractional_part<EXP_SZ: u32, FRACTION_SZ: u32>(f: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    f.bexp < bias(FRACTION_SZ as sN[EXP_SZ])
}

#[test]
fn has_fractional_part_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    type F32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ>;
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let big_f32 =
        F32 { sign: u1:0, bexp: bias(sN[F32_EXP_SZ]:32), fraction: uN[F32_FRACTION_SZ]:0x123 };
    assert_eq(has_fractional_part(one_f32), true);
    assert_eq(has_fractional_part(big_f32), false);

    const BF16_EXP_SZ = u32:5;
    const BF16_FRACTION_SZ = u32:10;
    type BF16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ>;
    let one_bf16 = one<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let big_bf16 =
        BF16 { sign: u1:0, bexp: bias(sN[BF16_EXP_SZ]:10), fraction: uN[BF16_FRACTION_SZ]:0x12 };
    assert_eq(has_fractional_part(one_bf16), true);
    assert_eq(has_fractional_part(big_bf16), false);
}

// Returns whether or not the given APFloat has an negative exponent.
pub fn has_negative_exponent<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    f.bexp < bias(sN[EXP_SZ]:0)
}

#[test]
fn has_negative_exponent_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let zero_f32 = zero<EXP_SZ, FRACTION_SZ>(u1:0);
    let zero_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:1), ..zero_f32 };
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(has_negative_exponent(zero_dot_5_f32), true);
    assert_eq(has_negative_exponent(one_f32), false);
}

// Round up exponent and fraction integral part, ignoring the sign.
fn round_up_no_sign_positive_exp<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    // unbias expononent to signed (but in that case positive) integer.
    let exp = unbiased_exponent(f);
    // compute fractional mask according to unbiased exponent.
    let fractional_mask = std::mask_bits<FRACTION_SZ>() >> (exp as u32);

    // add mask to round up integral bits of the fraction.
    // example: (bfloat3 w/ exp=3)
    // S: sign
    // E: biased exponent
    // I: integral bits of the fraction
    // F: fractional bits of the fraction

    // APFloat<8, 7>:         SEEE EEEE EIII FFFF
    // fraction:                         III FFFF
    // fractional_mask:                  000 1111
    // fraction carry:                    (1) if any of `F` is set
    // bexp carry:                   (1) if III overflow.
    //
    // if any of the fractional bits (F) are set:
    // adding the mask to the fraction causes the fractional bits to overflow and carry up to
    // integral bits (I) of the fraction, incrementing it by 1 (rounding it up to the next integer).
    let fraction_up =
        f.fraction as uN[FRACTION_SZ + u32:1] + fractional_mask as uN[FRACTION_SZ + u32:1];

    // if the integral bits of the fraction overflowed: forward the carry to the biased exponent.
    let bexp_with_carry = f.bexp + fraction_up[FRACTION_SZ+:u1] as uN[EXP_SZ];
    // mask out fractional part to get a round number.
    let fraction_integral = fraction_up[0+:uN[FRACTION_SZ]] & !fractional_mask;
    APFloat { sign: f.sign, bexp: bexp_with_carry, fraction: fraction_integral }
}

// Round down exponent and fraction integral part, ignoring the sign.
fn round_down_no_sign_positive_exp<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    // unbias expononent to signed (but in that case positive) integer.
    let exp = unbiased_exponent(f);
    // compute fractional mask according to unbiased exponent.
    let fractional_mask = std::mask_bits<FRACTION_SZ>() >> (exp as u32);
    // mask out fractional part to get a round number.
    let fraction_integral = f.fraction & !fractional_mask;
    APFloat { sign: f.sign, bexp: f.bexp, fraction: fraction_integral }
}

// Finds the nearest integral `APFloat` greater than or equal to `f`.
pub fn ceil<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {

    match tag(f) {
        APFloatTag::NAN => qnan<EXP_SZ, FRACTION_SZ>(),
        APFloatTag::INFINITY => inf<EXP_SZ, FRACTION_SZ>(f.sign),
        APFloatTag::ZERO => zero<EXP_SZ, FRACTION_SZ>(f.sign),
        _ => {
            if !has_fractional_part(f) {
                f  // if no fractional part: already rounded.
            } else if has_negative_exponent(f) {
                if f.sign != u1:0 {
                    // if sign negative: round to -0.
                    zero<EXP_SZ, FRACTION_SZ>(u1:1)
                } else {
                    // if sign positive: round to 1.
                    one<EXP_SZ, FRACTION_SZ>(u1:0)
                }
            } else if f.sign == u1:0 {
                // if positive: round up.
                round_up_no_sign_positive_exp(f)
            } else {
                // if negative: round down.
                round_down_no_sign_positive_exp(f)
            }
        },
    }
}

#[test]
fn ceil_fractional_midpoint_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    let two_f32 = add(one_f32, one_f32);
    let minus_two_f32 = F32 { sign: u1:1, ..two_f32 };
    let three_f32 = add(one_f32, two_f32);
    let two_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:2), ..two_f32 };
    let minus_two_dot_5_f32 = F32 { sign: u1:1, ..two_dot_5_f32 };
    assert_eq(ceil(two_dot_5_f32), three_f32);
    assert_eq(ceil(minus_two_dot_5_f32), minus_two_f32);
}

#[test]
fn ceil_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    let two_f32 = add(one_f32, one_f32);
    let minus_two_f32 = F32 { sign: u1:1, ..two_f32 };
    let three_f32 = add(one_f32, two_f32);
    let two_dot_0000002_f32 = F32 { fraction: uN[FRACTION_SZ]:1, ..two_f32 };
    assert_eq(ceil(two_dot_0000002_f32), three_f32);
    let minus_two_dot_0000002_f32 = F32 { sign: u1:1, ..two_dot_0000002_f32 };
    assert_eq(ceil(minus_two_dot_0000002_f32), minus_two_f32);
}

#[test]
fn ceil_big_integral_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let big_f32 = F32 { sign: u1:0, bexp: bias(sN[EXP_SZ]:32), fraction: uN[FRACTION_SZ]:0x123 };
    assert_eq(ceil(big_f32), big_f32);
}

#[test]
fn ceil_already_round_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(ceil(one_f32), one_f32);
}

#[test]
fn ceil_special() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let inf_f32 = inf<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(ceil(inf_f32), inf_f32);
    let minus_inf_f32 = F32 { sign: u1:1, ..inf_f32 };
    assert_eq(ceil(minus_inf_f32), minus_inf_f32);
    let zero_f32 = zero<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(ceil(zero_f32), zero_f32);
    let minus_zero_f32 = F32 { sign: u1:1, ..zero_f32 };
    assert_eq(ceil(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = qnan<EXP_SZ, FRACTION_SZ>();
    assert_eq(ceil(qnan_f32), qnan_f32);
}

#[test]
fn ceil_zero_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let zero_f32 = zero<EXP_SZ, FRACTION_SZ>(u1:0);
    let zero_dot_something_f32 =
        F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:1), ..zero_f32 };
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(ceil(zero_dot_something_f32), one_f32);
    let minus_zero_f32 = zero<EXP_SZ, FRACTION_SZ>(u1:1);
    let minus_zero_dot_something_f32 = F32 { sign: u1:1, ..zero_dot_something_f32 };
    assert_eq(ceil(minus_zero_dot_something_f32), minus_zero_f32);
}

// Finds the nearest integral `APFloat` lower than or equal to `f`.
pub fn floor<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {

    match tag(f) {
        APFloatTag::NAN => qnan<EXP_SZ, FRACTION_SZ>(),
        APFloatTag::INFINITY => inf<EXP_SZ, FRACTION_SZ>(f.sign),
        APFloatTag::ZERO => zero<EXP_SZ, FRACTION_SZ>(f.sign),
        _ => {
            if !has_fractional_part(f) {
                f  // if no fractional part: already rounded.
            } else if has_negative_exponent(f) {
                if f.sign != u1:0 {
                    // if sign negative: round to -1.
                    one<EXP_SZ, FRACTION_SZ>(u1:1)
                } else {
                    // if sign positive: round to 0.
                    zero<EXP_SZ, FRACTION_SZ>(u1:0)
                }
            } else if f.sign == u1:1 {
                // if negative: round up.
                round_up_no_sign_positive_exp(f)
            } else {
                // if positive: round down.
                round_down_no_sign_positive_exp(f)
            }
        },
    }
}

#[test]
fn floor_fractional_midpoint_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    let two_f32 = add(one_f32, one_f32);
    let three_f32 = add(one_f32, two_f32);
    let minus_three_f32 = F32 { sign: u1:1, ..three_f32 };
    let two_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:2), ..two_f32 };
    let minus_two_dot_5_f32 = F32 { sign: u1:1, ..two_dot_5_f32 };
    assert_eq(floor(two_dot_5_f32), two_f32);
    assert_eq(floor(minus_two_dot_5_f32), minus_three_f32);
}

#[test]
fn floor_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    let two_f32 = add(one_f32, one_f32);
    let three_f32 = add(one_f32, two_f32);
    let minus_three_f32 = F32 { sign: u1:1, ..three_f32 };
    let two_dot_5000002_f32 =
        F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:2) | uN[FRACTION_SZ]:1, ..two_f32 };
    let minus_two_dot_0000002_f32 = F32 { sign: u1:1, ..two_dot_5000002_f32 };
    assert_eq(floor(two_dot_5000002_f32), two_f32);
    assert_eq(floor(minus_two_dot_0000002_f32), minus_three_f32);
}

#[test]
fn floor_big_integral_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let big_f32 = F32 { sign: u1:0, bexp: bias(sN[EXP_SZ]:32), fraction: uN[FRACTION_SZ]:0x123 };
    assert_eq(floor(big_f32), big_f32);
}

#[test]
fn floor_already_round_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(floor(one_f32), one_f32);
}

#[test]
fn floor_special() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let inf_f32 = inf<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(floor(inf_f32), inf_f32);
    let minus_inf_f32 = F32 { sign: u1:1, ..inf_f32 };
    assert_eq(floor(minus_inf_f32), minus_inf_f32);
    let zero_f32 = zero<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(floor(zero_f32), zero_f32);
    let minus_zero_f32 = F32 { sign: u1:1, ..zero_f32 };
    assert_eq(floor(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = qnan<EXP_SZ, FRACTION_SZ>();
    assert_eq(floor(qnan_f32), qnan_f32);
}

#[test]
fn floor_zero_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let zero_f32 = zero<EXP_SZ, FRACTION_SZ>(u1:0);
    let zero_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:1), ..zero_f32 };
    assert_eq(floor(zero_dot_5_f32), zero_f32);
    let minus_one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:1);
    let minus_zero_dot_5_f32 = F32 { sign: u1:1, ..zero_dot_5_f32 };
    assert_eq(floor(minus_zero_dot_5_f32), minus_one_f32);
}

// Returns an `APFloat` with all its `fraction` bits past the decimal point set to `0`.
pub fn trunc<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {

    match tag(f) {
        APFloatTag::NAN => qnan<EXP_SZ, FRACTION_SZ>(),
        APFloatTag::INFINITY => inf<EXP_SZ, FRACTION_SZ>(f.sign),
        APFloatTag::ZERO => zero<EXP_SZ, FRACTION_SZ>(f.sign),
        _ => {
            if !has_fractional_part(f) {
                f  // if no fractional part: already truncated.
            } else if has_negative_exponent(f) {
                zero<EXP_SZ, FRACTION_SZ>(f.sign)
            } else {
                round_down_no_sign_positive_exp(f)
            }
        },
    }
}

#[test]
fn trunc_fractional_midpoint_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    let two_f32 = add(one_f32, one_f32);
    let two_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:2), ..two_f32 };
    let minus_two_f32 = F32 { sign: u1:1, ..two_f32 };
    let minus_two_dot_5_f32 = F32 { sign: u1:1, ..two_dot_5_f32 };
    assert_eq(trunc(two_dot_5_f32), two_f32);
    assert_eq(trunc(minus_two_dot_5_f32), minus_two_f32);
}

#[test]
fn trunc_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    let two_f32 = add(one_f32, one_f32);
    let two_dot_5000002_f32 =
        F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:2) | uN[FRACTION_SZ]:1, ..two_f32 };
    let minus_two_f32 = F32 { sign: u1:1, ..two_f32 };
    let minus_two_dot_0000002_f32 = F32 { sign: u1:1, ..two_dot_5000002_f32 };
    assert_eq(trunc(two_dot_5000002_f32), two_f32);
    assert_eq(trunc(minus_two_dot_0000002_f32), minus_two_f32);
}

#[test]
fn trunc_big_integral_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let big_f32 = F32 { sign: u1:0, bexp: bias(sN[EXP_SZ]:32), fraction: uN[FRACTION_SZ]:0x123 };
    assert_eq(trunc(big_f32), big_f32);
}

#[test]
fn trunc_already_round_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    let one_f32 = one<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(trunc(one_f32), one_f32);
}

#[test]
fn trunc_special() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let inf_f32 = inf<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(trunc(inf_f32), inf_f32);
    let minus_inf_f32 = F32 { sign: u1:1, ..inf_f32 };
    assert_eq(trunc(minus_inf_f32), minus_inf_f32);
    let zero_f32 = zero<EXP_SZ, FRACTION_SZ>(u1:0);
    assert_eq(trunc(zero_f32), zero_f32);
    let minus_zero_f32 = F32 { sign: u1:1, ..zero_f32 };
    assert_eq(trunc(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = qnan<EXP_SZ, FRACTION_SZ>();
    assert_eq(trunc(qnan_f32), qnan_f32);
}

#[test]
fn trunc_zero_fractional_test() {
    const EXP_SZ = u32:8;
    const FRACTION_SZ = u32:23;
    type F32 = APFloat<EXP_SZ, FRACTION_SZ>;
    let zero_f32 = zero<EXP_SZ, FRACTION_SZ>(u1:0);
    let zero_dot_5_f32 = F32 { fraction: uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:1), ..zero_f32 };
    assert_eq(trunc(zero_dot_5_f32), zero_f32);
    let minus_zero_f32 = zero<EXP_SZ, FRACTION_SZ>(u1:1);
    let minus_zero_dot_5_f32 = F32 { sign: u1:1, ..zero_dot_5_f32 };
    assert_eq(trunc(minus_zero_dot_5_f32), minus_zero_f32);
}

fn round_normal<EXP_SZ: u32, FRACTION_SZ: u32, ROUND_STYLE: RoundStyle = {RoundStyle::TIES_TO_EVEN}>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {

    const ROUND_STYLE_IS_TIES_TO_EVEN: bool = ROUND_STYLE == RoundStyle::TIES_TO_EVEN;
    const ROUND_STYLE_IS_TIES_TO_AWAY: bool = ROUND_STYLE == RoundStyle::TIES_TO_AWAY;
    const EXP_MAX = std::mask_bits<EXP_SZ>();
    const FRACTION_MAX = std::mask_bits<FRACTION_SZ>();
    const FRACTION_SZ_PLUS_ONE = FRACTION_SZ + u32:1;
    const_assert!(ROUND_STYLE_IS_TIES_TO_EVEN || ROUND_STYLE_IS_TIES_TO_AWAY);
    if !has_fractional_part(f) {
        f
    } else {
        let exp = unbiased_exponent(f);
        if exp < sN[EXP_SZ]:-1 {
            // abs(f) < 0.5
            zero<EXP_SZ, FRACTION_SZ>(f.sign)
        } else if exp == sN[EXP_SZ]:-1 {
            if ROUND_STYLE_IS_TIES_TO_EVEN && f.fraction == uN[FRACTION_SZ]:0 {
                // abs(f) == 0.5
                zero<EXP_SZ, FRACTION_SZ>(f.sign)
            } else {
                // TIES_TO_AWAY will always be one.
                // 0.5 < abs(f) < 1.0
                one<EXP_SZ, FRACTION_SZ>(f.sign)
            }
        } else {
            let round_up = if exp == sN[EXP_SZ]:0 {
                let two_exp_0 = u1:1;
                let widened_fraction = two_exp_0 ++ f.fraction;
                does_lsb_round_up(FRACTION_SZ as u32, widened_fraction, ROUND_STYLE)
            } else {
                let lsb_index = (FRACTION_SZ as uN[EXP_SZ] - exp as uN[EXP_SZ]);
                does_lsb_round_up(lsb_index as u32, f.fraction, ROUND_STYLE)
            };
            if round_up {
                round_up_no_sign_positive_exp(f)
            } else {
                round_down_no_sign_positive_exp(f)
            }
        }
    }
}

// Returns an `APFloat` of the same precision as `f` rounded to the nearest integer
// Using the "tie-to-even" (RoundStyle::TIES_TO_EVEN) rounding style:
// - in case of a tie (`f` half-way between two integers): rounding to the nearest even integer.
// Using the "tie-to-away" (RoundStyle::TIES_TO_AWAY) rounding style:
// - in case of a tie (`f` half-way between two integers): rounding to the integer with the largest
//   absolute value.
pub fn round<EXP_SZ: u32, FRACTION_SZ: u32, ROUND_STYLE: RoundStyle = {RoundStyle::TIES_TO_EVEN}>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    match tag(f) {
        APFloatTag::NAN => qnan<EXP_SZ, FRACTION_SZ>(),
        APFloatTag::INFINITY => inf<EXP_SZ, FRACTION_SZ>(f.sign),
        APFloatTag::ZERO => zero<EXP_SZ, FRACTION_SZ>(f.sign),
        APFloatTag::SUBNORMAL => zero<EXP_SZ, FRACTION_SZ>(f.sign),
        _ => round_normal<EXP_SZ, FRACTION_SZ, ROUND_STYLE>(f),
    }
}

#[test]
fn round_fractional_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let two_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> {
        sign: u1:0,
        bexp: uN[F32_EXP_SZ]:0x80,
        fraction: uN[F32_FRACTION_SZ]:0,
    };
    let minus_two_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..two_f32 };
    let three_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> {
        sign: u1:0,
        bexp: uN[F32_EXP_SZ]:0x80,
        fraction: uN[F32_FRACTION_SZ]:0x400000,
    };
    let minus_three_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..three_f32 };
    // round(2.0000002) == 2.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x1, ..two_f32 }), two_f32);
    // round(-2.0000002) == -2.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x1, ..minus_two_f32 }),
        minus_two_f32);
    // round(2.9999997) == 3.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x3fffff, ..two_f32 }),
        three_f32);
    // round(-2.9999997) = -3.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x3fffff, ..minus_two_f32 }),
        minus_three_f32);
}

#[test]
fn round_fractional_subnormal_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let minus_zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);

    // round(1e-45) == 0.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x1, ..zero_f32 }), zero_f32);
    // round(-1e-45) == -0.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x1, ..minus_zero_f32 }),
        minus_zero_f32);
    // round(1.1754942e-38) == 0.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x7fffff, ..zero_f32 }),
        zero_f32);
    // round(-1.1754942e-38) == -0.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x7fffff, ..minus_zero_f32 }),
        minus_zero_f32);
}

#[test]
fn round_fractional_negative_exp_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let minus_zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);

    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let minus_one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    // round(1.1754945e-38) == 0.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:0,
                bexp: uN[F32_EXP_SZ]:0x1,
                fraction: uN[F32_FRACTION_SZ]:0x1,
            }), zero_f32);
    // round(-1.1754945e-38) == -0.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:1,
                bexp: uN[F32_EXP_SZ]:0x1,
                fraction: uN[F32_FRACTION_SZ]:0x1,
            }), minus_zero_f32);
    // round(0.49999997) == 0.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:0,
                bexp: uN[F32_EXP_SZ]:0x7d,
                fraction: uN[F32_FRACTION_SZ]:0x7fffff,
            }), zero_f32);
    // round(-0.49999997) == -0.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:1,
                bexp: uN[F32_EXP_SZ]:0x7d,
                fraction: uN[F32_FRACTION_SZ]:0x7fffff,
            }), minus_zero_f32);
    // round(0.5) == 0.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:0,
                bexp: uN[F32_EXP_SZ]:0x7e,
                fraction: uN[F32_FRACTION_SZ]:0,
            }), zero_f32);
    // round(-0.5) == -0.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:1,
                bexp: uN[F32_EXP_SZ]:0x7e,
                fraction: uN[F32_FRACTION_SZ]:0,
            }), minus_zero_f32);
    // round(0.50000006) == 1.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:0,
                bexp: uN[F32_EXP_SZ]:0x7e,
                fraction: uN[F32_FRACTION_SZ]:1,
            }), one_f32);
    // round(-0.50000006) == -1.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:1,
                bexp: uN[F32_EXP_SZ]:0x7e,
                fraction: uN[F32_FRACTION_SZ]:1,
            }), minus_one_f32);
    // round(0.9999999) == 1.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:0,
                bexp: uN[F32_EXP_SZ]:0x7e,
                fraction: uN[F32_FRACTION_SZ]:0x7ffffe,
            }), one_f32);
    // round(-0.9999999) == -1.0
    assert_eq(
        round(
            APFloat<u32:8, u32:23> {
                sign: u1:1,
                bexp: uN[F32_EXP_SZ]:0x7e,
                fraction: uN[F32_FRACTION_SZ]:0x7ffffe,
            }), minus_one_f32);
}

#[test]
fn round_fractional_zero_exp_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    let minus_one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:1);
    let two_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> {
        sign: u1:0,
        bexp: uN[F32_EXP_SZ]:0x80,
        fraction: uN[F32_FRACTION_SZ]:0,
    };
    let minus_two_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..two_f32 };
    // round(1.0000001) == 1.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x1, ..one_f32 }), one_f32);
    // round(-1.0000001) == -1.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x1, ..minus_one_f32 }),
        minus_one_f32);
    // round(1.9999999) == 2.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x7fffff, ..one_f32 }), two_f32);
    // round(-1.9999999) == -2.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x7fffff, ..minus_one_f32 }),
        minus_two_f32);
}

#[test]
fn round_fractional_midpoint_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let two_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> {
        sign: u1:0,
        bexp: uN[F32_EXP_SZ]:0x80,
        fraction: uN[F32_FRACTION_SZ]:0,
    };
    let minus_two_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..two_f32 };
    let three_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> {
        sign: u1:0,
        bexp: uN[F32_EXP_SZ]:0x80,
        fraction: uN[F32_FRACTION_SZ]:0x400000,
    };
    let minus_three_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..three_f32 };
    let four_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> {
        sign: u1:0,
        bexp: uN[F32_EXP_SZ]:0x81,
        fraction: uN[F32_FRACTION_SZ]:0,
    };
    let minus_four_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..four_f32 };
    // round(3.5) == 4.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x600000, ..three_f32 }),
        four_f32);
    // round(-3.5) = -4.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x600000, ..minus_three_f32 }),
        minus_four_f32);
    // round(2.5) == 2.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x200000, ..two_f32 }), two_f32);
    // round(-2.5) == -2.0
    assert_eq(
        round(APFloat<u32:8, u32:23> { fraction: uN[F32_FRACTION_SZ]:0x200000, ..minus_two_f32 }),
        minus_two_f32);
}

#[test]
fn round_already_round_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    assert_eq(round(one_f32), one_f32);
}

#[test]
fn round_special() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    let inf_f32 = inf<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    assert_eq(round(inf_f32), inf_f32);
    let minus_inf_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..inf_f32 };
    assert_eq(round(minus_inf_f32), minus_inf_f32);
    let zero_f32 = zero<F32_EXP_SZ, F32_FRACTION_SZ>(u1:0);
    assert_eq(round(zero_f32), zero_f32);
    let minus_zero_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: u1:1, ..zero_f32 };
    assert_eq(round(minus_zero_f32), minus_zero_f32);
    let qnan_f32 = qnan<F32_EXP_SZ, F32_FRACTION_SZ>();
    assert_eq(round(qnan_f32), qnan_f32);
}

#[quickcheck]
fn round_never_assert_fails(f: APFloat<u32:8, u32:23>) -> bool {
    // nb asserts are checked.
    round(f);
    true
}

#[quickcheck]
fn downcast_never_assert_fails(f: APFloat<u32:8, u32:23>, round_style: RoundStyle) -> bool {
    // nb asserts are checked.
    downcast<u32:10, u32:8>(f, round_style);
    downcast<u32:7, u32:8>(f, round_style);
    downcast<u32:3, u32:4>(f, round_style);
    downcast<u32:2, u32:5>(f, round_style);
    true
}

#[quickcheck]
fn cast_from_fixed_using_rne_never_assert_fails(v: s64) -> bool {
    // nb asserts are checked.
    cast_from_fixed_using_rne<u32:4, u32:4>(v);
    cast_from_fixed_using_rne<u32:8, u32:8>(v);
    cast_from_fixed_using_rne<u32:8, u32:23>(v);
    cast_from_fixed_using_rne<u32:20, u32:43>(v);
    cast_from_fixed_using_rne<u32:25, u32:60>(v);
    true
}
