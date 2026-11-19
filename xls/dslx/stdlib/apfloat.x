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
import lza;

pub struct APFloat<EXP_SZ: u32, FRACTION_SZ: u32> {
    sign: bits[1],  // Sign bit.
    bexp: bits[EXP_SZ],  // Biased exponent.
    fraction: bits[FRACTION_SZ],  // Fractional part (no hidden bit).
}

impl APFloat<EXP_SZ, FRACTION_SZ> {
    const EXP_SIZE = EXP_SZ;
    const FRACTION_SIZE = FRACTION_SZ;
    const TOTAL_SIZE = 1 + EXP_SZ + FRACTION_SZ;
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
        (0, 0) => APFloatTag::ZERO,
        (0, _) => APFloatTag::SUBNORMAL,
        (EXPR_MASK, 0) => APFloatTag::INFINITY,
        (EXPR_MASK, _) => APFloatTag::NAN,
        (_, _) => APFloatTag::NORMAL,
    }
}

// Returns a quiet NaN.
pub fn qnan<EXP_SZ: u32, FRACTION_SZ: u32>() -> APFloat<EXP_SZ, FRACTION_SZ> {
    APFloat<EXP_SZ, FRACTION_SZ> {
        sign: 0,
        bexp: std::mask_bits<EXP_SZ>() as bits[EXP_SZ],
        fraction: bits[FRACTION_SZ]:1 << ((FRACTION_SZ - 1) as bits[FRACTION_SZ]),
    }
}

// Returns whether or not the given APFloat represents NaN.
pub fn is_nan<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction != 0)
}

// Returns a positive or a negative infinity depending upon the given sign parameter.
pub fn inf<EXP_SZ: u32, FRACTION_SZ: u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    APFloat<EXP_SZ, FRACTION_SZ> { sign, bexp: std::mask_bits<EXP_SZ>(), fraction: 0 }
}

// Returns whether or not the given APFloat represents an infinite quantity.
pub fn is_inf<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction == 0)
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

// Returns a positive or negative zero depending upon the given sign parameter.
pub fn zero<EXP_SZ: u32, FRACTION_SZ: u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    APFloat<EXP_SZ, FRACTION_SZ> { sign, bexp: 0, fraction: 0 }
}

// Returns one or minus one depending upon the given sign parameter.
pub fn one<EXP_SZ: u32, FRACTION_SZ: u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    const MASK_SZ: u32 = EXP_SZ - 1;
    APFloat<EXP_SZ, FRACTION_SZ> {
        sign,
        bexp: std::mask_bits<MASK_SZ>() as bits[EXP_SZ],
        fraction: 0,
    }
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

// Returns the negative of x unless it is a NaN, in which case it will
// change it from a quiet to signaling NaN or from signaling to a quiet NaN.
pub fn negate<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    type Float = APFloat<EXP_SZ, FRACTION_SZ>;
    Float { sign: !x.sign, bexp: x.bexp, fraction: x.fraction }
}

// Maximum value of the exponent for normal numbers with EXP_SZ bits in the exponent field.
pub fn max_normal_exp<EXP_SZ: u32>() -> sN[EXP_SZ] {
    ((uN[EXP_SZ]:1 << (EXP_SZ - 1)) - 1) as sN[EXP_SZ]
}

// Minimum value of the exponent for normal numbers with EXP_SZ bits in the exponent field.
pub fn min_normal_exp<EXP_SZ: u32>() -> sN[EXP_SZ] {
    let minus_min_normal_exp = ((uN[EXP_SZ]:1 << (EXP_SZ - 1)) - 2) as sN[EXP_SZ];
    -minus_min_normal_exp
}

// Returns 2^(EXP_SZ-1)-1, which is the exponent bias used for encoding
// the given APFloat type. For example, this would return 127 for float32.
pub fn exponent_bias<EXP_SZ: u32, FRACTION_SZ: u32>(f: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[EXP_SZ] {
    std::signed_max_value<EXP_SZ>() as sN[EXP_SZ]
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

// Returns the biased exponent which is equal to `unbiased_exponent + 2^(EXP_SZ - 1)`
//
// Notice: Since the function only takes as input the unbiased exponent, it cannot
// distinguish between zero and subnormal numbers, or NaN and infinity, respectively.
pub fn bias<EXP_SZ: u32>(unbiased_exponent: sN[EXP_SZ]) -> uN[EXP_SZ] {
    const BIAS = std::signed_max_value<EXP_SZ>();
    (unbiased_exponent + BIAS) as uN[EXP_SZ]
}

// Returns a bit string of size `1 + EXP_SZ + FRACTION_SZ` where the first bit
// is the sign bit, the next `EXP_SZ` bit encode the biased exponent and the
// last `FRACTION_SZ` are the significand without the hidden bit.
pub fn flatten<EXP_SZ: u32, FRACTION_SZ: u32, TOTAL_SZ: u32 = {1 + EXP_SZ + FRACTION_SZ}>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> bits[TOTAL_SZ] {
    x.sign ++ x.bexp ++ x.fraction
}

// Returns a `APFloat` struct whose flattened version would be the input string.
pub fn unflatten<EXP_SZ: u32, FRACTION_SZ: u32, TOTAL_SZ: u32 = {1 + EXP_SZ + FRACTION_SZ}>
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

// Returns true when a value with the given LSB, guard (halfway) bit, and
// sticky bit should round up under Round to Nearest, ties to Even (RNE).
//
//  L G S
//  X 0 X   --> Round down (less than half)
//  0 1 0   --> Round down (half, already even)
//  1 1 0   --> Round up (half, to even)
//  X 1 1   --> Round up (greater than half)
fn round_to_nearest_even(lsb: bool, guard: bool, sticky: bool) -> bool { guard && (lsb || sticky) }

// Applies a round-up increment to an unrounded mantissa using modulo-wrap
// addition. When `unrounded_frac` is all-ones and `do_round_up` is true, the
// FRACTION_SZ-bit addition wraps to 0 (matching the fractional bits of 2.0 =
// 1.0 * 2^1) and `rounding_carry` is asserted to increment the exponent.
fn apply_rounding<FRACTION_SZ: u32>
    (unrounded_frac: uN[FRACTION_SZ], do_round_up: bool) -> (uN[FRACTION_SZ], bool) {
    let rounding_carry = and_reduce(unrounded_frac) && do_round_up;
    let rounded_frac = unrounded_frac + (do_round_up as uN[FRACTION_SZ]);
    (rounded_frac, rounding_carry)
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
    assert!(lsb_index >= 1, "apfloat_round_without_residue");
    assert!(lsb_index <= FRACTION_SZ as u32, "apfloat_round_without_lsb");
    // Extract the first bit which needs to be cut.
    let first_lost_bit_idx = lsb_index - 1;
    let round_bit = fraction[first_lost_bit_idx+:u1];
    match round_style {
        RoundStyle::TIES_TO_EVEN => {
            // Extract the last bit which is retained.
            let lsb = lsb_index < FRACTION_SZ && fraction[lsb_index+:u1];
            // Whether any bits before the round_bit are 1.
            let sticky = std::or_reduce_lsb(fraction, lsb_index - 1);

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
    // Wide enough to hold both max_exp_exclusive (2^(EXP_SZ - 1)) and num_trailing_nonzeros
    // (up to NUM_SRC_BITS) without truncating or wrapping when NUM_SRC_BITS > 2^(EXP_SZ + 1).
    const UEXP_SZ: u32 = std::max(EXP_SZ + 1, std::clog2(NUM_SRC_BITS + 1));
    const EXTENDED_FRACTION_SZ: u32 = FRACTION_SZ + NUM_SRC_BITS;

    // Determine sign.
    let is_negative = to_cast < 0;

    // Determine exponent.
    let abs_magnitude = std::abs(to_cast) as uN[NUM_SRC_BITS];
    let lz = clz(abs_magnitude);
    let num_trailing_nonzeros = (NUM_SRC_BITS as uN[NUM_SRC_BITS]) - lz;

    let exp = (num_trailing_nonzeros as uN[UEXP_SZ]) - 1;
    let max_exp_exclusive = uN[UEXP_SZ]:1 << ((EXP_SZ as uN[UEXP_SZ]) - 1);
    let is_inf = exp >= max_exp_exclusive;
    let bexp = bias(exp as sN[EXP_SZ]);

    // Determine fraction (pre-rounding).
    //
    // TODO(sameeragarwal): This concatenation of FRACTION_SZ zeros followed by a shift
    // and slice seems excessive, worth exploring if the compiler is able to optimize it
    // or a smaller manual implementation will be more efficient.
    let extended_fraction = abs_magnitude ++ uN[FRACTION_SZ]:0;
    let fraction = extended_fraction >> ((num_trailing_nonzeros - 1) as uN[EXTENDED_FRACTION_SZ]);
    let fraction = fraction[0:FRACTION_SZ as s32];

    // Round fraction (round to nearest, half to even).
    let lsb_idx = (num_trailing_nonzeros as uN[EXTENDED_FRACTION_SZ]) - 1;
    let round_up = if lsb_idx == 0 {
        // already exact
        false
    } else if lsb_idx > EXTENDED_FRACTION_SZ as uN[EXTENDED_FRACTION_SZ] {
        // Overflowed lsb idx
        true
    } else {
        does_lsb_round_up(lsb_idx as u32, extended_fraction, RoundStyle::TIES_TO_EVEN)
    };
    let fraction = if round_up { fraction + 1 } else { fraction };

    // Check if rounding up causes an exponent increment.
    let overflow = round_up && (fraction == 0);
    let bexp = if overflow { (bexp + 1) } else { bexp };

    // Check if rounding up caused us to overflow to infinity.
    let is_inf = is_inf || bexp == std::mask_bits<EXP_SZ>();
    let is_zero = abs_magnitude == 0;
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

// Casts the fixed point number to a floating point number using RZ
// (Round to Zero) as the rounding mode.
pub fn cast_from_fixed_using_rz<EXP_SZ: u32, FRACTION_SZ: u32, NUM_SRC_BITS: u32>
    (to_cast: sN[NUM_SRC_BITS]) -> APFloat<EXP_SZ, FRACTION_SZ> {
    // Wide enough to hold both max_exp_exclusive (2^(EXP_SZ - 1)) and num_trailing_nonzeros
    // (up to NUM_SRC_BITS) without truncating or wrapping when NUM_SRC_BITS > 2^(EXP_SZ + 1).
    const UEXP_SZ: u32 = std::max(EXP_SZ + 1, std::clog2(NUM_SRC_BITS + 1));
    const EXTENDED_FRACTION_SZ: u32 = FRACTION_SZ + NUM_SRC_BITS;

    // Determine sign.
    let is_negative = to_cast < 0;

    // Determine exponent.
    let abs_magnitude = std::abs(to_cast) as uN[NUM_SRC_BITS];
    let lz = clz(abs_magnitude);
    let num_trailing_nonzeros = (NUM_SRC_BITS as uN[NUM_SRC_BITS]) - lz;

    let exp = (num_trailing_nonzeros as uN[UEXP_SZ]) - 1;
    let max_exp_exclusive = uN[UEXP_SZ]:1 << ((EXP_SZ as uN[UEXP_SZ]) - 1);
    let is_inf = exp >= max_exp_exclusive;
    let bexp = bias(exp as sN[EXP_SZ]);

    // Determine fraction (pre-rounding).
    //
    // TODO(sameeragarwal): This concatenation of FRACTION_SZ zeros followed by a shift
    // and slice seems excessive, worth exploring if the compiler is able to optimize it
    // or a smaller manual implementation will be more efficient.
    let extended_fraction = abs_magnitude ++ uN[FRACTION_SZ]:0;
    let fraction = extended_fraction >> ((num_trailing_nonzeros - 1) as uN[EXTENDED_FRACTION_SZ]);
    let fraction = fraction[0:FRACTION_SZ as s32];

    let is_zero = abs_magnitude == 0;

    match (is_inf, is_zero) {
        // For round-to-zero 'the result shall be the format’s floating-point
        // number closest to and no greater in magnitude than the infinitely
        // precise result' so since this is an integer (and therefore
        // non-infinite) if the result would be out of range we return the
        // largest (in magnitude) representable finite number.
        (true, false) => max_normal<EXP_SZ, FRACTION_SZ>(is_negative),
        // Technically is_inf and is_zero should never be true at the same time, however, currently
        // the way is_inf is computed it can be true while is_zero is true. So we allow is_inf to be
        // anything when deciding if the output should be zero or not.
        //
        // Further, input when zero does not have a sign, so we will always return +0.0
        (_, true) => zero<EXP_SZ, FRACTION_SZ>(false),
        (false, false) => APFloat<EXP_SZ, FRACTION_SZ> { sign: is_negative, bexp, fraction },
    }
}

// Returns true if x == 0 or x is a subnormal number.
pub fn is_zero_or_subnormal<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    x.bexp == 0
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
        const FROM_EXP_SZ_PLUS_1 = FROM_EXP_SZ + 1;
        type FromExpOffsetT = sN[FROM_EXP_SZ_PLUS_1];
        type ToExpOffsetT = sN[TO_EXP_SZ];
        // substract `2^(FROM_EXP_SZ-1) - 1` to retrieve the true exponent.
        const FROM_EXP_SZ_MINUS_1 = FROM_EXP_SZ - 1;
        const FROM_EXP_OFFSET = (FromExpOffsetT:1 << FROM_EXP_SZ_MINUS_1) - 1;
        // add 2^(TO_EXP_SZ-1) - 1 to contruct back offset encoded exponent.
        const TO_EXP_SZ_MINUS_1 = TO_EXP_SZ - 1;
        const TO_EXP_OFFSET = (ToExpOffsetT:1 << TO_EXP_SZ_MINUS_1) - 1;

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
                    let subnormal_exp = (-FROM_EXP_OFFSET + 1) as ToExpOffsetT;
                    let subnormal_to_normal_shift = clz(f.fraction) + 1;
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

// Upcast the given apfloat to an apfloat with fraction and exponent size at least as large as the
// input's. Note: denormal inputs get flushed to zero.
pub fn upcast_daz<TO_EXP_SZ: u32, TO_FRACTION_SZ: u32, FROM_EXP_SZ: u32, FROM_FRACTION_SZ: u32>
    (f: APFloat<FROM_EXP_SZ, FROM_FRACTION_SZ>) -> APFloat<TO_EXP_SZ, TO_FRACTION_SZ> {
    upcast_with_denorms<TO_EXP_SZ, TO_FRACTION_SZ>(subnormals_to_zero(f))
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
        bexp: if renormalize { f.bexp + 1 } else { f.bexp },
        fraction: if round_up { truncated_fraction + 1 } else { truncated_fraction },
    }
}

#[test]
fn downcast_fractional_raz_fp32_to_bf16_test() {
    const F32_EXP_SZ = u32:8;
    const F32_FRACTION_SZ = u32:23;
    const BF16_EXP_SZ = u32:8;
    const BF16_FRACTION_SZ = u32:7;

    // normals
    let one_bf16 = one<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    assert_eq(downcast_fractional<BF16_FRACTION_SZ>(one_f32, RoundStyle::TIES_TO_AWAY), one_bf16);

    let minus_one_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 1, ..one_bf16 };
    let minus_one_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..one_f32 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(minus_one_f32, RoundStyle::TIES_TO_AWAY),
        minus_one_bf16);

    // fraction with no rounding necessary
    let one_dot_5_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: 0x400000, ..one_f32 };
    let one_dot_5_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { fraction: 0x40, ..one_bf16 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(one_dot_5_f32, RoundStyle::TIES_TO_AWAY),
        one_dot_5_bf16);

    // rounds down
    let pi_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x490fdb };
    let pi_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x49 };
    assert_eq(downcast_fractional<BF16_FRACTION_SZ>(pi_f32, RoundStyle::TIES_TO_AWAY), pi_bf16);

    // rounds up
    let one_third_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x7d, fraction: 0x2aaaab };
    let one_third_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x7d, fraction: 0x2b };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(one_third_f32, RoundStyle::TIES_TO_AWAY),
        one_third_bf16);

    // rounds up, tie to away
    let rne_up_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x408000 };
    let rne_up_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x41 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(rne_up_f32, RoundStyle::TIES_TO_AWAY), rne_up_bf16);

    // rounds up, tie to away
    let rne_up_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x418000 };
    let rne_up_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x42 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(rne_up_f32, RoundStyle::TIES_TO_AWAY), rne_up_bf16);

    // round up to inf
    let inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let just_above_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0xfe, fraction: 0x7fe000 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(just_above_max_normal_bf16, RoundStyle::TIES_TO_AWAY),
        inf_bf16);

    let just_below_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, bexp: 0xfe, fraction: 0x7fe000 };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(just_below_max_normal_bf16, RoundStyle::TIES_TO_AWAY),
        minus_inf_bf16);

    // round up to inf
    let max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0xfe, fraction: 0x7fffff };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(max_normal_f32, RoundStyle::TIES_TO_AWAY), inf_bf16);

    let minus_max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, bexp: 0xfe, fraction: 0x7fffff };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);
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
    let one_bf16 = one<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let one_f32 = one<F32_EXP_SZ, F32_FRACTION_SZ>(0);
    assert_eq(downcast_fractional<BF16_FRACTION_SZ>(one_f32, RoundStyle::TIES_TO_EVEN), one_bf16);

    let minus_one_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 1, ..one_bf16 };
    let minus_one_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, ..one_f32 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(minus_one_f32, RoundStyle::TIES_TO_EVEN),
        minus_one_bf16);

    // fraction with no rounding necessary
    let one_dot_5_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { fraction: 0x400000, ..one_f32 };
    let one_dot_5_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { fraction: 0x40, ..one_bf16 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(one_dot_5_f32, RoundStyle::TIES_TO_EVEN),
        one_dot_5_bf16);

    // rounds down
    let pi_f32 = APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x490fdb };
    let pi_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x49 };
    assert_eq(downcast_fractional<BF16_FRACTION_SZ>(pi_f32, RoundStyle::TIES_TO_EVEN), pi_bf16);

    // rounds up
    let one_third_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x7d, fraction: 0x2aaaab };
    let one_third_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x7d, fraction: 0x2b };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(one_third_f32, RoundStyle::TIES_TO_EVEN),
        one_third_bf16);

    // rounds down, tie to even
    let rne_down_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x408000 };
    let rne_down_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x40 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(rne_down_f32, RoundStyle::TIES_TO_EVEN), rne_down_bf16);

    // rounds up, tie to even
    let rne_up_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x418000 };
    let rne_up_bf16 =
        APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: 0, bexp: 0x80, fraction: 0x42 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(rne_up_f32, RoundStyle::TIES_TO_EVEN), rne_up_bf16);

    // round up to inf
    let inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(0);
    let just_above_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0xfe, fraction: 0x7fe000 };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(just_above_max_normal_bf16, RoundStyle::TIES_TO_EVEN),
        inf_bf16);

    let just_below_max_normal_bf16 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, bexp: 0xfe, fraction: 0x7fe000 };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(just_below_max_normal_bf16, RoundStyle::TIES_TO_EVEN),
        minus_inf_bf16);

    // round up to inf
    let max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 0, bexp: 0xfe, fraction: 0x7fffff };
    assert_eq(
        downcast_fractional<BF16_FRACTION_SZ>(max_normal_f32, RoundStyle::TIES_TO_EVEN), inf_bf16);

    let minus_max_normal_f32 =
        APFloat<F32_EXP_SZ, F32_FRACTION_SZ> { sign: 1, bexp: 0xfe, fraction: 0x7fffff };
    let minus_inf_bf16 = inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(1);
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
    let right_shift_cnt = -(TO_BIAS + uexp) as u32 + 1;
    if right_shift_cnt > (TO_FRACTION_SZ + 1) {
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

        let subnormal_frac =
            if round_up { unrounded_subnormal_frac + 1 } else { unrounded_subnormal_frac };

        // Technically the subnormal frac is good enough but this is
        // easier to see through.
        let rounds_to_normal = right_shift_cnt == 1 && subnormal_frac == 0;

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

// Returns a normalized APFloat with the given components.
// 'fraction_with_hidden' is the fraction (including the hidden bit). This
// function only normalizes in the direction of decreasing the exponent. Input
// must be a normal number or zero. Dernormals are flushed to zero in the
// result.
pub fn normalize<EXP_SZ: u32, FRACTION_SZ: u32, WIDE_FRACTION: u32 = {FRACTION_SZ + 1}>
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
        if exp == (MIN_EXPONENT - 1) && fraction.fraction == std::mask_bits<FRACTION_SZ>() {
            Float { sign: fraction.sign, bexp: 1, fraction: 0 }
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

// Casts the floating point number to a fixed point number.
// Unrepresentable numbers are cast to the minimum representable
// number (largest magnitude negative number).
pub fn cast_to_fixed<NUM_DST_BITS: u32, EXP_SZ: u32, FRACTION_SZ: u32>
    (to_cast: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[NUM_DST_BITS] {
    const EXTENDED_FIXED_SZ: u32 = NUM_DST_BITS + 1 + FRACTION_SZ + NUM_DST_BITS;

    const MIN_FIXED_VALUE =
        (uN[NUM_DST_BITS]:1 << ((NUM_DST_BITS as uN[NUM_DST_BITS]) - 1)) as sN[NUM_DST_BITS];
    const MAX_EXPONENT = NUM_DST_BITS - 1;

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
    let result = if to_cast.bexp < bias(sN[EXP_SZ]:0) { 0 } else { result };

    result
}

// Returns u1:1 if x == y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn eq_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    if !(is_nan(x) || is_nan(y)) {
        ((flatten(x) == flatten(y)) || (is_zero_or_subnormal(x) && is_zero_or_subnormal(y)))
    } else {
        0
    }
}

// Returns u1:1 if x > y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn gt_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    if eq_2(x, y) {
        false
    } else {
        // Flush denormals.
        let x = subnormals_to_zero(x);
        let y = subnormals_to_zero(y);

        let gt_exp = x.bexp > y.bexp;
        let eq_exp = x.bexp == y.bexp;
        let gt_fraction = x.fraction > y.fraction;
        let abs_gt = gt_exp || (eq_exp && gt_fraction);
        let result = match (x.sign, y.sign) {
            // Both positive.
            (0, 0) => abs_gt,
            // x positive, y negative.
            (0, 1) => true,
            // x negative, y positive.
            (1, 0) => false,
            // Both negative.
            _ => !abs_gt && !eq_2(x, y),
        };

        if !(is_nan(x) || is_nan(y)) { result } else { false }
    }
}

// Returns u1:1 if x >= y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn gte_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    gt_2(x, y) || eq_2(x, y)
}

// Returns u1:1 if x <= y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn lte_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    if !(is_nan(x) || is_nan(y)) { !gt_2(x, y) } else { 0 }
}

// Returns u1:1 if x < y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn lt_2<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    if !(is_nan(x) || is_nan(y)) { !gte_2(x, y) } else { false }
}

// Helper to convert to a signed or unsigned integer.
fn to_signed_or_unsigned_int<RESULT_SZ: u32, RESULT_SIGNED: bool, EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> xN[RESULT_SIGNED][RESULT_SZ] {
    const WIDE_FRACTION: u32 = FRACTION_SZ + 1;
    const MAX_FRACTION_SZ: u32 = std::max(RESULT_SZ, WIDE_FRACTION);

    const INT_MIN = if RESULT_SIGNED {
        (uN[MAX_FRACTION_SZ]:1 << (RESULT_SZ - 1))  // or rather, its negative.
    } else {
        0
    };
    const INT_MAX = if RESULT_SIGNED {
        (uN[MAX_FRACTION_SZ]:1 << (RESULT_SZ - 1)) - 1
    } else {
        (uN[MAX_FRACTION_SZ]:1 << RESULT_SZ) - 1
    };

    let exp = unbiased_exponent(x);

    // True significand (including implicit leading bit) as an integer. Will need to be shifted
    // so that MSB represents the value 1 based on the effective exponent (exp - FRACTION_SZ).
    let fraction = (x.fraction as uN[WIDE_FRACTION] | (uN[WIDE_FRACTION]:1 << FRACTION_SZ)) as
                   uN[MAX_FRACTION_SZ];

    let max_exp = if RESULT_SIGNED { RESULT_SZ as s32 - 1 } else { RESULT_SZ as s32 };
    let exp_oob = exp as s32 >= max_exp;
    let result = if exp_oob || is_inf(x) || (!RESULT_SIGNED && x.sign) {
        // Clamp if out of bounds, infinite.
        if x.sign { INT_MIN } else { INT_MAX }
    } else if is_nan(x) {
        uN[MAX_FRACTION_SZ]:0
    } else if exp < 0 {
        uN[MAX_FRACTION_SZ]:0
    } else if exp == 0 {
        uN[MAX_FRACTION_SZ]:1
    } else {
        // For most cases, we need to either shift the "ones" place from FRACTION_SZ + 1 bits down
        // closer to 0 (if the effective exponent is negative) else we need to move it away from 0
        // if the effective exponent is positive.
        let effective_exp = (exp as s32) - (FRACTION_SZ as s32);
        if effective_exp < 0 {
            (fraction >> (-effective_exp as u32))
        } else if effective_exp > 0 {
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

// Calculate difference of two values and return values in sign-magnitude form. Returns
// sign-magnitude tuple (a - b <= 0, |a - b|).
//
// NOTE: This returns -0 if (a == b). We take advantage of this elsewhere; it makes it trivial to
//       test if a is strictly greater than b.
fn sign_magnitude_difference<WIDTH: u32, SIGNED: bool>
    (a: xN[SIGNED][WIDTH], b: xN[SIGNED][WIDTH]) -> (bool, uN[WIDTH]) {
    // If signed, bias the inputs into an unsigned domain, so we can use unsigned abs_diff.
    let a_biased = const if SIGNED { !std::msb(a) ++ (a as uN[WIDTH])[:-1] } else { a };
    let b_biased = const if SIGNED { !std::msb(b) ++ (b as uN[WIDTH])[:-1] } else { b };
    let abs_diff_result = abs_diff::abs_diff(a_biased, b_biased);
    (!abs_diff::is_x_larger(abs_diff_result), abs_diff::to_corrected(abs_diff_result))
}

#[test]
fn unsigned_sign_magnitude_difference_test() {
    // The way we like our ones differencer is to yield a -0 for (a == b), i.e. result.0==(a >= b)
    assert_eq(sign_magnitude_difference(u8:0, u8:0), (true, u8:0));
    assert_eq(sign_magnitude_difference(u8:42, u8:42), (true, u8:0));
    assert_eq(sign_magnitude_difference(u8:255, u8:255), (true, u8:0));

    // Make sure this works for very small width; exhaustive for u1
    assert_eq(sign_magnitude_difference(u1:0, u1:0), (true, u1:0));
    assert_eq(sign_magnitude_difference(u1:1, u1:1), (true, u1:0));
    assert_eq(sign_magnitude_difference(u1:0, u1:1), (true, u1:1));
    assert_eq(sign_magnitude_difference(u1:1, u1:0), (false, u1:1));

    // Close to overflow is handled correctly
    assert_eq(sign_magnitude_difference(u8:255, u8:0), (false, u8:255));
    assert_eq(sign_magnitude_difference(u8:255, u8:5), (false, u8:250));
    assert_eq(sign_magnitude_difference(u8:250, u8:0), (false, u8:250));
    assert_eq(sign_magnitude_difference(u8:0, u8:255), (true, u8:255));
    assert_eq(sign_magnitude_difference(u8:5, u8:255), (true, u8:250));
    assert_eq(sign_magnitude_difference(u8:0, u8:250), (true, u8:250));
}

#[test]
fn signed_sign_magnitude_difference_test() {
    assert_eq(sign_magnitude_difference(s8:0, s8:0), (true, u8:0));
    assert_eq(sign_magnitude_difference(s8:-42, s8:-42), (true, u8:0));
    assert_eq(sign_magnitude_difference(s8:-6, s8:5), (true, u8:11));
    assert_eq(sign_magnitude_difference(s8:5, s8:-6), (false, u8:11));
    assert_eq(sign_magnitude_difference(s8:-128, s8:127), (true, u8:255));
    assert_eq(sign_magnitude_difference(s8:127, s8:-128), (false, u8:255));

    // Make sure this works for very small width; exhaustive for s1
    assert_eq(sign_magnitude_difference(s1:0, s1:0), (true, u1:0));
    assert_eq(sign_magnitude_difference(s1:-1, s1:-1), (true, u1:0));
    assert_eq(sign_magnitude_difference(s1:0, s1:-1), (false, u1:1));
    assert_eq(sign_magnitude_difference(s1:-1, s1:0), (true, u1:1));

    // Close to overflow is handled correctly
    assert_eq(sign_magnitude_difference(s8:127, s8:-128), (false, u8:255));
    assert_eq(sign_magnitude_difference(s8:122, s8:-128), (false, u8:250));
    assert_eq(sign_magnitude_difference(s8:127, s8:-123), (false, u8:250));
    assert_eq(sign_magnitude_difference(s8:-128, s8:127), (true, u8:255));
    assert_eq(sign_magnitude_difference(s8:-128, s8:122), (true, u8:250));
    assert_eq(sign_magnitude_difference(s8:-123, s8:127), (true, u8:250));
}

fn sign_magnitude_difference_equivalence<WIDTH: u32, SIGNED: bool>
    (left: xN[SIGNED][WIDTH], right: xN[SIGNED][WIDTH]) -> bool {
    let (a_is_smaller, diff) = sign_magnitude_difference(left, right);
    let expected_diff =
        if right >= left { (right - left) as uN[WIDTH] } else { (left - right) as uN[WIDTH] };
    (a_is_smaller == (right >= left)) && (diff == expected_diff)
}

#[quickcheck(exhaustive)]
fn sign_magnitude_difference_u2_equivalence(left: u2, right: u2) -> bool {
    sign_magnitude_difference_equivalence(left, right)
}

#[quickcheck(exhaustive)]
fn sign_magnitude_difference_u8_equiivalence(left: u8, right: u8) -> bool {
    sign_magnitude_difference_equivalence(left, right)
}

#[quickcheck(exhaustive)]
fn sign_magnitude_difference_s2_equivalence(left: s2, right: s2) -> bool {
    sign_magnitude_difference_equivalence(left, right)
}

#[quickcheck(exhaustive)]
fn sign_magnitude_difference_s8_equivalence(left: s8, right: s8) -> bool {
    sign_magnitude_difference_equivalence(left, right)
}

// Manually apply optimization https://github.com/google/xls/issues/1217
fn or_last_bit<WIDTH: u32>(value: bits[WIDTH], lsb: u1) -> bits[WIDTH] {
    bit_slice_update(value, u1:0, (value[0:1] | lsb))
}

// Adjusts the biased exponent after normalization and rounding, flushing
// underflows/zeros to 0 and saturating overflows to infinity.
fn adjust_exponent_and_saturate
    <EXP_SZ: u32, FRACTION_SZ: u32, EXP_IN_SZ: u32, LZ_SZ: u32, SIGNED_EXP: u32 = {EXP_SZ + 2}>
    (greater_exp: uN[EXP_IN_SZ], leading_zeroes: uN[LZ_SZ], rounding_carry: bool,
     fraction_is_zero: bool, rounded_fraction: uN[FRACTION_SZ]) -> (uN[EXP_SZ], uN[FRACTION_SZ]) {
    const MAX_EXPONENT = std::mask_bits<EXP_SZ>();
    let base_bexp = (greater_exp as sN[SIGNED_EXP]) + 1 - (leading_zeroes as sN[SIGNED_EXP]);
    let bexp = if rounding_carry { base_bexp + 1 } else { base_bexp };
    let is_underflow_or_zero = fraction_is_zero || bexp <= 0;
    let is_overflow = bexp >= (MAX_EXPONENT as sN[SIGNED_EXP]);
    let result_exp = if is_underflow_or_zero {
        0
    } else if is_overflow {
        MAX_EXPONENT
    } else {
        bexp as uN[EXP_SZ]
    };
    let result_fraction = if is_underflow_or_zero || is_overflow { 0 } else { rounded_fraction };
    (result_exp, result_fraction)
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
pub fn add<EXP_SZ: u32, FRACTION_SZ: u32, USE_LZA: bool = {false}>
    (a: APFloat<EXP_SZ, FRACTION_SZ>, b: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    // WIDE_FRACTION: Widened fraction to contain full precision + rounding
    // (sign, hidden as well as guard, round, sticky) bits.
    const SIGN_BIT = u32:1;
    const HIDDEN_BIT = u32:1;
    const GUARD_ROUND_STICKY_BITS = u32:3;
    const FRACTION = HIDDEN_BIT + FRACTION_SZ;
    const SIGNED_FRACTION = SIGN_BIT + FRACTION;
    const WIDE_FRACTION: u32 = SIGNED_FRACTION + GUARD_ROUND_STICKY_BITS;
    // CARRY_FRACTION: WIDE_FRACTION plus one bit to capture a possible carry bit.
    const CARRY_FRACTION: u32 = 1 + WIDE_FRACTION;

    // NORMALIZED_FRACTION: WIDE_FRACTION minus two bits (dropping round and sticky), used for
    // normalization.
    const NORMALIZED_FRACTION: u32 = WIDE_FRACTION - 2;

    // Step 0: Swap operands for x to contain value with greater exponent
    let (a_is_smaller, shift) = sign_magnitude_difference(a.bexp, b.bexp);
    let (x, y) = if a_is_smaller { (b, a) } else { (a, b) };

    // Step 1: add hidden bit.
    let fraction_x = (u1:1 ++ x.fraction) as uN[FRACTION];
    let fraction_y = (u1:1 ++ y.fraction) as uN[FRACTION];

    // Flush denormals to 0.
    let fraction_x = if x.bexp == 0 { 0 } else { fraction_x };
    let fraction_y = if y.bexp == 0 { 0 } else { fraction_y };

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
    let fraction_is_zero = fraction == 0;
    let result_sign = match (fraction_is_zero, fraction < 0) {
        (true, _) => x.sign && y.sign,  // So that -0.0 + -0.0 results in -0.0
        (false, true) => !y.sign,
        _ => y.sign,
    };

    // Get the absolute value of the result then chop off the sign bit: s29 -> u28.
    let abs_fraction = (if fraction < 0 { -fraction } else { fraction }) as uN[WIDE_FRACTION];

    // Step 3: Normalize the fraction (shift until the leading bit is a 1).
    // Drop round and sticky bits before normalization: they can never contain the
    // leading 1, and when >= 2 bits cancel (leading_zeroes >= 3), shift <= 1 so
    // round and sticky are identically zero.
    let raw_norm = abs_fraction[2:] as uN[NORMALIZED_FRACTION];
    let carry_bit = std::msb(raw_norm);

    const CLZ_WIDTH: u32 = std::clog2(WIDE_FRACTION + 1);
    let init_shift = if USE_LZA {
        // For subtraction, lza is off-by-one at most and will be corrected later by examining
        // shifted_norm. For addition, clz is 0 when there is a carry and 1 when there is no carry.
        let (approx_lz, _) =
            lza::lza<WIDE_FRACTION>(wide_x as uN[WIDE_FRACTION], addend_y as uN[WIDE_FRACTION]);
        if x.sign != y.sign { approx_lz } else { !carry_bit as uN[CLZ_WIDTH] }
    } else {
        // Leading zeroes will be 1 if there's no carry or cancellation.
        std::clzt(raw_norm) as uN[CLZ_WIDTH]
    };

    let shifted_norm = raw_norm << init_shift;
    let (shifted_norm, leading_zeroes) = if USE_LZA && !carry_bit {
        // Leading zeroes will be off-by-one at most if there is cancellation. As an optimization,
        // we correct this in a separate shift-left instead of using LZA detection.
        let lz_off_by_one = !std::msb(shifted_norm);
        let final_norm = if lz_off_by_one { shifted_norm << 1 } else { shifted_norm };
        let final_lz = if lz_off_by_one { init_shift + 1 } else { init_shift };
        (final_norm, final_lz)
    } else {
        (shifted_norm, init_shift)
    };
    let unrounded_frac = shifted_norm[2+:uN[FRACTION_SZ]];

    // Step 4: Rounding (decision decoupled from the normalization barrel shift).
    // We only support round-to-nearest, half-to-even (RTNE) mode. Because >= 2-bit cancellation
    // (leading_zeroes >= 3) only occurs when shift <= 1, where the round and sticky bits
    // (abs_fraction[0:2]) are identically zero and the guard shifts into the mantissa, rounding up
    // is only possible in three cases:
    //  - Carry-out (leading_zeroes == 0): LSB is bit 4, guard is bit 3, sticky is bits [0:3].
    //  - Normal (leading_zeroes == 1): LSB is bit 3, guard is bit 2, sticky is bits [0:2].
    //  - 1-bit cancel (leading_zeroes == 2): LSB is bit 2, guard is bit 1, sticky is bit 0.
    let do_round_up = if abs_fraction[FRACTION_SZ + 4+:u1] {
        round_to_nearest_even(
            abs_fraction[4+:u1], abs_fraction[3+:u1], or_reduce(abs_fraction[0:3]))
    } else if abs_fraction[FRACTION_SZ + 3+:u1] {
        round_to_nearest_even(
            abs_fraction[3+:u1], abs_fraction[2+:u1], or_reduce(abs_fraction[0:2]))
    } else if abs_fraction[FRACTION_SZ + 2+:u1] {
        round_to_nearest_even(abs_fraction[2+:u1], abs_fraction[1+:u1], abs_fraction[0+:u1])
    } else {
        false
    };
    let (result_fraction, rounding_carry) = apply_rounding(unrounded_frac, do_round_up);

    // Step 5: Adjust exponent and handle underflow/overflow/special cases.
    let (result_exponent, result_fraction) = adjust_exponent_and_saturate<EXP_SZ>(
        x.bexp, leading_zeroes, rounding_carry, fraction_is_zero, result_fraction);

    // Handle arg infinities.
    const MAX_EXPONENT = std::mask_bits<EXP_SZ>();
    let is_operand_inf = is_inf<EXP_SZ, FRACTION_SZ>(x) || is_inf<EXP_SZ, FRACTION_SZ>(y);
    let result_exponent = if is_operand_inf { MAX_EXPONENT } else { result_exponent };
    let result_fraction = if is_operand_inf { 0 } else { result_fraction };
    // Result infinity is negative iff all infinite operands are neg.
    let has_pos_inf = (is_inf<EXP_SZ, FRACTION_SZ>(x) && (x.sign == 0)) ||
                      (is_inf<EXP_SZ, FRACTION_SZ>(y) && (y.sign == 0));
    let result_sign = if is_operand_inf { !has_pos_inf } else { result_sign };

    // Handle NaN; NaN trumps infinities, so we handle it last.
    // -inf + inf = NaN, i.e., if we have both positive and negative inf.
    let has_neg_inf = (is_inf<EXP_SZ, FRACTION_SZ>(x) && (x.sign == 1)) ||
                      (is_inf<EXP_SZ, FRACTION_SZ>(y) && (y.sign == 1));
    let is_result_nan = is_nan<EXP_SZ, FRACTION_SZ>(x) || is_nan<EXP_SZ, FRACTION_SZ>(y) ||
                        (has_pos_inf && has_neg_inf);
    const FRACTION_HIGH_BIT = uN[FRACTION_SZ]:1 << (FRACTION_SZ - 1);
    let result_exponent = if is_result_nan { MAX_EXPONENT } else { result_exponent };
    let result_fraction = if is_result_nan { FRACTION_HIGH_BIT } else { result_fraction };
    let result_sign = if is_result_nan { 0 } else { result_sign };

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
pub fn sub<EXP_SZ: u32, FRACTION_SZ: u32, USE_LZA: bool = {false}>
    (x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    let y = APFloat<EXP_SZ, FRACTION_SZ> { sign: !y.sign, bexp: y.bexp, fraction: y.fraction };
    add<EXP_SZ, FRACTION_SZ, USE_LZA>(x, y)
}

// Rounds a normalized significand `norm_sig` (explicit leading 1 at MSB,
// followed by `FRACTION_SZ` fraction bits and `ROUND_BITS` rounding bits) with
// biased pre-rounding exponent `bexp_pre_round` using Round-to-Nearest-Even
// (RNE), flushing subnormals (post-rounding `bexp <= 0`) to zero and
// saturating overflows (`bexp >= max_exp`) to infinity.
//
// When `bexp_pre_round <= 0`, the significand is shifted right by 1 bit
// (preserving the sticky bit) before rounding so values in
// `[min_normal - 0.5 * ULP_subnormal, min_normal)` round up and carry into
// `min_normal` (`bexp == 1`) rather than flushing to zero.
fn round_and_saturate_ftz
    <EXP_SZ: u32, FRACTION_SZ: u32, NORM_WIDTH: u32, SIGNED_EXP: u32 = {EXP_SZ + 2}>
    (bexp_pre_round: sN[SIGNED_EXP], norm_sig: uN[NORM_WIDTH], is_zero: bool)
    -> (uN[EXP_SZ], uN[FRACTION_SZ]) {
    const_assert!(NORM_WIDTH >= FRACTION_SZ + 2);
    const ROUND_BITS: u32 = NORM_WIDTH - 1 - FRACTION_SZ;
    const MAX_EXPONENT: uN[EXP_SZ] = std::mask_bits<EXP_SZ>();

    let sig = if bexp_pre_round <= sN[SIGNED_EXP]:0 {
        or_last_bit(norm_sig >> 1, norm_sig[0:1])
    } else {
        norm_sig
    };

    let unrounded_frac = sig[ROUND_BITS+:uN[FRACTION_SZ]];
    let lsb = unrounded_frac[0:1];
    let guard = sig[(ROUND_BITS - u32:1)+:u1];
    let sticky = if ROUND_BITS > u32:1 { or_reduce(sig[0:(ROUND_BITS - 1) as s32]) } else { false };
    let do_round_up = round_to_nearest_even(lsb, guard, sticky);
    let (rounded_frac, rounding_carry) = apply_rounding(unrounded_frac, do_round_up);

    let bexp = if rounding_carry { bexp_pre_round + sN[SIGNED_EXP]:1 } else { bexp_pre_round };
    let is_underflow_or_zero = is_zero || bexp <= sN[SIGNED_EXP]:0;
    let is_overflow = bexp >= (MAX_EXPONENT as sN[SIGNED_EXP]);
    let result_exp = if is_underflow_or_zero {
        uN[EXP_SZ]:0
    } else if is_overflow {
        MAX_EXPONENT
    } else {
        bexp as uN[EXP_SZ]
    };
    let result_fraction =
        if is_underflow_or_zero || is_overflow { uN[FRACTION_SZ]:0 } else { rounded_frac };
    (result_exp, result_fraction)
}

#[test]
fn round_and_saturate_ftz_test() {
    // E4M3 (EXP_SZ=4, FRACTION_SZ=3), NORM_WIDTH=7 (1 implicit + 3 frac + 3 round bits).
    // Normal tie-to-even: even LSB rounds down, odd LSB rounds up.
    assert_eq(round_and_saturate_ftz<4, 3>(s6:7, u7:0b1_010_100, false), (u4:7, u3:0b010));
    assert_eq(round_and_saturate_ftz<4, 3>(s6:7, u7:0b1_011_100, false), (u4:7, u3:0b100));
    // Normal carry into next exponent binade.
    assert_eq(round_and_saturate_ftz<4, 3>(s6:7, u7:0b1_111_100, false), (u4:8, u3:0b000));
    // Subnormal boundary (bexp_pre_round == 0): midpoint 1.111 * 2^-7 carries into min_normal.
    assert_eq(round_and_saturate_ftz<4, 3>(s6:0, u7:0b1_111_000, false), (u4:1, u3:0b000));
    // Just below midpoint stays subnormal and flushes to zero.
    assert_eq(round_and_saturate_ftz<4, 3>(s6:0, u7:0b1_110_111, false), (u4:0, u3:0b000));
    // Overflow saturates to infinity; is_zero always flushes to zero.
    assert_eq(round_and_saturate_ftz<4, 3>(s6:14, u7:0b1_111_100, false), (u4:15, u3:0b000));
    assert_eq(round_and_saturate_ftz<4, 3>(s6:7, u7:0b1_111_100, true), (u4:0, u3:0b000));
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
    // SIGNED_EXP: Exponent width plus one carry bit and one sign bit.
    const SIGNED_EXP: u32 = EXP_SZ + 2;

    // WIDE_FRACTION: Widened fraction to contain full precision + rounding
    // (guard & sticky) bits.
    const WIDE_FRACTION: u32 = FRACTION_SZ + FRACTION_SZ + 2;

    // 0. Check if either operand is 0 (flushing subnorms to 0).
    let has_0_arg = is_zero_or_subnormal(x) || is_zero_or_subnormal(y);

    // 1. Get and expand mantissas.
    let x_fraction = (x.fraction as uN[WIDE_FRACTION]) |
                     (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));
    let y_fraction = (y.fraction as uN[WIDE_FRACTION]) |
                     (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));

    // 2. Multiply integer mantissas, flushing subnormals to 0.
    let fraction = if has_0_arg { 0 } else { x_fraction * y_fraction };

    // 3. Add non-biased exponents.
    //  - Remove the bias from the exponents, add them, then restore the bias.
    //  - Simplifies from
    //      (A - 127) + (B - 127) + 127 = exp
    //    to
    //      A + B - 127 = exp
    let bias = std::mask_bits<EXP_SZ>() as sN[SIGNED_EXP] >> 1;
    let exp = (x.bexp as sN[SIGNED_EXP]) + (y.bexp as sN[SIGNED_EXP]) - bias;

    // Here is where we'd handle subnormals if we cared to.
    // If the exponent remains < 0, even after reapplying the bias,
    // then we'd calculate the extra exponent needed to get back to 0.
    // We'd set the result exponent to 0 and shift the fraction to the right
    // to capture that "extra" exponent.
    // Since we just flush subnormals, we don't have to do any of that.
    // Instead, if we're multiplying by 0, the result is 0.
    let exp = if has_0_arg { 0 } else { exp };

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

    // Round (RNE), flush subnormals to zero, and saturate overflows to infinity.
    let result_sign = x.sign != y.sign;
    let high_exp = std::mask_bits<EXP_SZ>();
    let (result_exp, result_fraction) =
        round_and_saturate_ftz<EXP_SZ, FRACTION_SZ>(exp, fraction[:-1], has_0_arg);

    // Special cases (operand infinities and NaNs).

    // - Arg infinites. Any arg is infinite == result is infinite.
    let is_operand_inf = is_inf<EXP_SZ, FRACTION_SZ>(x) || is_inf<EXP_SZ, FRACTION_SZ>(y);
    let result_exp = if is_operand_inf { high_exp } else { result_exp };
    let result_fraction = if is_operand_inf { 0 } else { result_fraction };

    // - NaNs. NaN trumps infinities, so we handle it last.
    //   inf * 0 = NaN, i.e.,
    let has_nan_arg = is_nan<EXP_SZ, FRACTION_SZ>(x) || is_nan<EXP_SZ, FRACTION_SZ>(y);
    let has_inf_arg = is_inf<EXP_SZ, FRACTION_SZ>(x) || is_inf<EXP_SZ, FRACTION_SZ>(y);
    let is_result_nan = has_nan_arg || (has_0_arg && has_inf_arg);
    let result_exp = if is_result_nan { high_exp } else { result_exp };
    let nan_fraction = uN[FRACTION_SZ]:1 << (FRACTION_SZ as uN[FRACTION_SZ] - 1);
    let result_fraction = if is_result_nan { nan_fraction } else { result_fraction };
    let result_sign = if is_result_nan { 0 } else { result_sign };

    APFloat<EXP_SZ, FRACTION_SZ> { sign: result_sign, bexp: result_exp, fraction: result_fraction }
}

// Simple utility struct for holding the full result of a floating-point multiplication; bexp may be
// negative (representing a subnormal result), and fraction still contains the leading 1.
//
// If `is_nan` is true, then we make no guarantee about the value of any other members.
struct RawProduct<SIGNED_EXP: u32, WIDE_FRACTION: u32> {
    is_nan: u1,
    sign: u1,
    bexp: sN[SIGNED_EXP],
    fraction: uN[WIDE_FRACTION],
}

// Returns true if the given RawProduct is infinite.
fn is_raw_product_inf<SIGNED_EXP: u32, WIDE_FRACTION: u32>
    (p: RawProduct<SIGNED_EXP, WIDE_FRACTION>) -> bool {
    !p.is_nan && p.bexp == std::signed_max_value<SIGNED_EXP>()
}

// Returns the full result of a floating-point multiplication, without any rounding or truncation.
// The only exception is that input denormals are flushed to/treated as 0.
pub fn raw_mul
    <EXP_SZ: u32, FRACTION_SZ_X: u32, FRACTION_SZ_Y: u32, SIGNED_EXP: u32 = {EXP_SZ + 2},
     WIDE_FRACTION: u32 = {FRACTION_SZ_X + FRACTION_SZ_Y + 2}>
    (x: APFloat<EXP_SZ, FRACTION_SZ_X>, y: APFloat<EXP_SZ, FRACTION_SZ_Y>)
    -> RawProduct<SIGNED_EXP, WIDE_FRACTION> {
    // 0. Check if either operand is 0 (flushing subnorms to 0).
    let has_0_arg = is_zero_or_subnormal(x) || is_zero_or_subnormal(y);

    // 1. Get and expand significands.
    let x_significand = u1:1 ++ x.fraction;
    let y_significand = u1:1 ++ y.fraction;

    // 2. Multiply integer significands, flushing input subnormals to 0.
    let full_product = if has_0_arg { 0 } else { std::umul(x_significand, y_significand) };

    // 3. Add non-biased exponents.
    //  - Remove the bias from the exponents, add them, then restore the bias.
    //  - Simplifies from
    //      (A - 127) + (B - 127) + 127 = exp
    //    to
    //      A + B - 127 = exp
    let bias = std::mask_bits<EXP_SZ>() as sN[SIGNED_EXP] >> 1;
    let exp = (x.bexp as sN[SIGNED_EXP]) + (y.bexp as sN[SIGNED_EXP]) - bias;

    // Here is where we'd handle subnormals if we cared to.
    // If the exponent remains < 0, even after reapplying the bias,
    // then we'd calculate the extra exponent needed to get back to 0.
    // We'd set the result exponent to 0 and shift the fraction to the right
    // to capture that "extra" exponent.
    // Since we just flush subnormals, we don't have to do any of that.
    // Instead, if we're multiplying by 0, the result is 0.
    let exp = if has_0_arg { 0 } else { exp };

    // 4. Normalize.
    // The result is either in the lower binade [1.0, 2.0) or the upper binade [2.0, 4.0), and the
    // MSB tells us which one. This tells us where the leading 1 is (either the MSB or the next
    // bit), and how much to shift to get the real significand.
    let in_upper_binade = std::msb(full_product);
    let product_significand = if in_upper_binade { full_product } else { (full_product << 1) };

    // Update the exponent if we're in the upper binade (and thus didn't have to shift the
    // significand); we gained a new significant bit.
    let exp = exp + (in_upper_binade as sN[SIGNED_EXP]);

    // We're done - except for special cases...
    let result_sign = x.sign != y.sign;
    let result_exp = exp;
    let result_fraction = product_significand;

    // 5. Special cases!
    // We don't flush subnormals to zero here; users can handle that later.
    // The exponent can't be saturated at this point, so we can leave that to users as well.
    // We just need to handle infinite args and NaNs.

    // - Arg infinites. Any arg is infinite == result is infinite.
    let is_operand_inf = is_inf(x) || is_inf(y);
    let result_exp = if is_operand_inf { std::signed_max_value<SIGNED_EXP>() } else { result_exp };
    let result_fraction = if is_operand_inf { 0 } else { result_fraction };

    // - NaNs. NaN trumps infinities, so we handle it last.
    //   inf * 0 = NaN, and so on.
    let has_nan_arg = is_nan(x) || is_nan(y);
    let has_inf_arg = is_inf(x) || is_inf(y);
    let is_result_nan = has_nan_arg || (has_0_arg && has_inf_arg);

    RawProduct<SIGNED_EXP, WIDE_FRACTION> {
        is_nan: is_result_nan,
        sign: result_sign,
        bexp: result_exp,
        fraction: result_fraction,
    }
}

// Returns the full-precision product of `x` and `y`, without rounding. The only exception is that
// both input and output denormals are treated as/flushed to 0.
pub fn full_precision_mul
    <EXP_SZ: u32, FRACTION_SZ_X: u32, FRACTION_SZ_Y: u32,
     FRACTION_SZ: u32 = {FRACTION_SZ_X + FRACTION_SZ_Y + 1}>
    (x: APFloat<EXP_SZ, FRACTION_SZ_X>, y: APFloat<EXP_SZ, FRACTION_SZ_Y>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
    // Prevent users from overriding the (calculated) return type.
    const_assert!(FRACTION_SZ == FRACTION_SZ_X + FRACTION_SZ_Y + 1);

    // SIGNED_EXP: the size of the exponent, along with one sign bit and one carry bit
    const SIGNED_EXP = EXP_SZ + 2;

    // Start with the "raw" product - this is the full result of the multiplication, with
    // potentially-negative oversized exponent and an untruncated fraction.
    let raw_product = raw_mul(x, y);

    // There's only a little work left to do - standardize NaNs, overflow/saturate infinites, flush
    // subnormals to 0, and drop the leading 1 from the fraction for normals,

    let special_exp = std::unsigned_max_value<EXP_SZ>();
    if raw_product.is_nan {
        // - NaNs: standardize on quiet NaN
        qnan<EXP_SZ, FRACTION_SZ>()
    } else if raw_product.bexp >= (special_exp as sN[SIGNED_EXP]) {
        // - Overflow infinites: saturate exp and clear fraction
        inf<EXP_SZ, FRACTION_SZ>(raw_product.sign)
    } else if raw_product.bexp <= 0 {
        // - Subnormals: flush to 0
        zero<EXP_SZ, FRACTION_SZ>(raw_product.sign)
    } else {
        // - Normals: just drop the leading 1 from the fraction (taken care of by narrowing),
        //            while narrowing the exponent (which already fits).
        APFloat<EXP_SZ, FRACTION_SZ> {
            sign: raw_product.sign,
            bexp: raw_product.bexp as uN[EXP_SZ],
            fraction: raw_product.fraction as uN[FRACTION_SZ],
        }
    }
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
    const EXP_CARRY: u32 = EXP_SZ + 1;
    // EXP_SIGN_CARRY: One greater than EXP_CARRY, to hold a sign bit.
    const EXP_SIGN_CARRY: u32 = EXP_CARRY + 1;
    // WIDE_FRACTION: Fully-widened fraction to hold all rounding bits.
    const WIDE_FRACTION: u32 = (FRACTION_SZ + 1) * 3 + 1;
    // WIDE_FRACTION_CARRY: WIDE_FRACTION plus one carry bit.
    const WIDE_FRACTION_CARRY: u32 = WIDE_FRACTION + 1;
    // WIDE_FRACTION_SIGN_CARRY: WIDE_FRACTION_CARRY plus one sign bit.
    const WIDE_FRACTION_SIGN_CARRY: u32 = WIDE_FRACTION_CARRY + 1;

    let ab = raw_mul(a, b);

    let (ab_exp_smaller, exp_difference) =
        sign_magnitude_difference(ab.bexp, c.bexp as sN[EXP_SIGN_CARRY]);
    let exp_difference = exp_difference as uN[EXP_CARRY];
    let (greater_exp, greater_sign) =
        if ab_exp_smaller { (c.bexp as sN[EXP_SIGN_CARRY], c.sign) } else { (ab.bexp, ab.sign) };

    // Make the implicit leading `1` explicit and flush subnormal `c` to 0 (already
    // done for `a` and `b` inside raw_mul()).
    let wide_c = c.fraction as uN[WIDE_FRACTION] | (uN[WIDE_FRACTION]:1 << FRACTION_SZ);
    let wide_c = if c.bexp == 0 { 0 } else { wide_c };

    // Align AB and C so that the implicit '1' is in the MSB.
    // For binary32: so shift by 73-48 for AB, and 73-24 for C.
    let wide_ab = (ab.fraction as uN[WIDE_FRACTION]) << (WIDE_FRACTION - ((FRACTION_SZ + 1) * 2));
    let wide_c = wide_c << (WIDE_FRACTION - (FRACTION_SZ + 1));

    // Shift the operands into their correct positions.
    // The sticky bits are set to whether anything non-zero gets shifted out.
    let rshift_ab = if ab_exp_smaller { exp_difference } else { 0 };
    let rshift_c = if ab_exp_smaller { 0 } else { exp_difference };
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
    let fraction_is_zero = sum_fraction == 0;
    let result_sign = match (fraction_is_zero, sum_fraction < 0) {
        // If the result is zero then because we can only differ from the sign
        // of at most one of the addition operands, we are negative only if both
        // the (a*b) and c are negative zero.
        (true, _) => ab.sign && c.sign,
        (false, true) => !greater_sign,
        _ => greater_sign,
    };

    // Chop off the sign bit (after applying it, if necessary).
    let abs_fraction =
        (if sum_fraction < 0 { -sum_fraction } else { sum_fraction }) as uN[WIDE_FRACTION_CARRY];

    // Normalize.
    let carry_bit = abs_fraction[-1:];
    let carry_fraction = (abs_fraction >> 1) as uN[WIDE_FRACTION];
    let carry_fraction = carry_fraction | (abs_fraction[0:1] as uN[WIDE_FRACTION]);

    // If high bits were cancelled, shift the result back into the MSB (ignoring
    // the zeroed carry bit, which is handled above).
    let leading_zeroes = clz(abs_fraction);
    let cancel = leading_zeroes > 1;
    let cancel_fraction = (abs_fraction << (leading_zeroes - 1)) as uN[WIDE_FRACTION];
    let shifted_fraction = match (carry_bit, cancel) {
        (true, false) => carry_fraction,
        (false, true) => cancel_fraction,
        (false, false) => abs_fraction as uN[WIDE_FRACTION],
        _ => fail!("carry_and_cancel", uN[WIDE_FRACTION]:0),
    };

    let bexp_pre_round = greater_exp + 1 - leading_zeroes as sN[EXP_SIGN_CARRY];
    let max_exp = std::mask_bits<EXP_SZ>();
    let (result_exp, result_fraction) = round_and_saturate_ftz<EXP_SZ, FRACTION_SZ>(
        bexp_pre_round, shifted_fraction, fraction_is_zero);

    // Handle arg infinities.
    let is_ab_inf = is_raw_product_inf(ab);
    let is_operand_inf = is_ab_inf | is_inf(c);
    let result_exp = if is_operand_inf { max_exp } else { result_exp };
    let result_fraction = if is_operand_inf { 0 } else { result_fraction };
    // Result infinity is negative iff all infinite operands are neg.
    let has_pos_inf = (is_ab_inf & (ab.sign == 0)) | (is_inf(c) & (c.sign == 0));
    let result_sign = if is_operand_inf { !has_pos_inf } else { result_sign };

    // Handle NaN; NaN trumps infinities, so we handle it last.
    // -inf + inf = NaN, i.e., if we have both positive and negative inf.
    let has_neg_inf = (is_ab_inf & (ab.sign == 1)) | (is_inf(c) & (c.sign == 1));
    let is_result_nan = ab.is_nan | is_nan(c) | (has_pos_inf & has_neg_inf);
    if is_result_nan {
        qnan<EXP_SZ, FRACTION_SZ>()
    } else {
        APFloat<EXP_SZ, FRACTION_SZ> {
            sign: result_sign,
            bexp: result_exp as uN[EXP_SZ],
            fraction: result_fraction as uN[FRACTION_SZ],
        }
    }
}

// Returns whether or not the given APFloat has a fractional part.
pub fn has_fractional_part<EXP_SZ: u32, FRACTION_SZ: u32>(f: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    f.bexp < bias(FRACTION_SZ as sN[EXP_SZ])
}

// Returns whether or not the given APFloat has an negative exponent.
pub fn has_negative_exponent<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
    f.bexp < bias(sN[EXP_SZ]:0)
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
    let fraction_up = f.fraction as uN[FRACTION_SZ + 1] + fractional_mask as uN[FRACTION_SZ + 1];

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

// Finds the nearest integral `APFloat` greater than or equal to `f` (round toward positive).
// Subnormal inputs are supported and rounded as per IEEE-754 to be no less than the infinitely
// precise result.
pub fn ceil_with_denorms<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {

    match tag(f) {
        APFloatTag::NAN => qnan<EXP_SZ, FRACTION_SZ>(),
        APFloatTag::INFINITY => inf<EXP_SZ, FRACTION_SZ>(f.sign),
        APFloatTag::ZERO => zero<EXP_SZ, FRACTION_SZ>(f.sign),
        _ => {
            if !has_fractional_part(f) {
                f  // if no fractional part: already rounded.
            } else if has_negative_exponent(f) {
                if f.sign != 0 {
                    // if sign negative: round to -0.
                    zero<EXP_SZ, FRACTION_SZ>(1)
                } else {
                    // if sign positive: round to 1.
                    one<EXP_SZ, FRACTION_SZ>(0)
                }
            } else if f.sign == 0 {
                // if positive: round up.
                round_up_no_sign_positive_exp(f)
            } else {
                // if negative: round down.
                round_down_no_sign_positive_exp(f)
            }
        },
    }
}

// Finds the nearest integral `APFloat` greater than or equal to `f` (round toward positive).
// Note: denormal inputs get flushed to zero.
pub fn ceil_daz<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    ceil_with_denorms(subnormals_to_zero(f))
}

// Finds the nearest integral `APFloat` lower than or equal to `f` (round toward negative).
// Subnormal inputs are supported and rounded as per IEEE-754 to be no greater than the infinitely
// precise result.
pub fn floor_with_denorms<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {

    match tag(f) {
        APFloatTag::NAN => qnan<EXP_SZ, FRACTION_SZ>(),
        APFloatTag::INFINITY => inf<EXP_SZ, FRACTION_SZ>(f.sign),
        APFloatTag::ZERO => zero<EXP_SZ, FRACTION_SZ>(f.sign),
        _ => {
            if !has_fractional_part(f) {
                f  // if no fractional part: already rounded.
            } else if has_negative_exponent(f) {
                if f.sign != 0 {
                    // if sign negative: round to -1.
                    one<EXP_SZ, FRACTION_SZ>(1)
                } else {
                    // if sign positive: round to 0.
                    zero<EXP_SZ, FRACTION_SZ>(0)
                }
            } else if f.sign == 1 {
                // if negative: round up.
                round_up_no_sign_positive_exp(f)
            } else {
                // if positive: round down.
                round_down_no_sign_positive_exp(f)
            }
        },
    }
}

// Finds the nearest integral `APFloat` lower than or equal to `f` (round toward negative).
// Note: denormal inputs get flushed to zero.
pub fn floor_daz<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    floor_with_denorms(subnormals_to_zero(f))
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

fn round_normal<EXP_SZ: u32, FRACTION_SZ: u32, ROUND_STYLE: RoundStyle = {RoundStyle::TIES_TO_EVEN}>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {

    const ROUND_STYLE_IS_TIES_TO_EVEN: bool = ROUND_STYLE == RoundStyle::TIES_TO_EVEN;
    const ROUND_STYLE_IS_TIES_TO_AWAY: bool = ROUND_STYLE == RoundStyle::TIES_TO_AWAY;
    const_assert!(ROUND_STYLE_IS_TIES_TO_EVEN || ROUND_STYLE_IS_TIES_TO_AWAY);
    if !has_fractional_part(f) {
        f
    } else {
        let exp = unbiased_exponent(f);
        if exp < -1 {
            // abs(f) < 0.5
            zero<EXP_SZ, FRACTION_SZ>(f.sign)
        } else if exp == -1 {
            if ROUND_STYLE_IS_TIES_TO_EVEN && f.fraction == 0 {
                // abs(f) == 0.5
                zero<EXP_SZ, FRACTION_SZ>(f.sign)
            } else {
                // TIES_TO_AWAY will always be one.
                // 0.5 < abs(f) < 1.0
                one<EXP_SZ, FRACTION_SZ>(f.sign)
            }
        } else {
            let round_up = if exp == 0 {
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
