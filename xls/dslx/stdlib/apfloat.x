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

pub struct APFloat<EXP_SZ: u32, FRACTION_SZ: u32> {
    sign: bits[1],  // Sign bit.
    bexp: bits[EXP_SZ],  // Biased exponent.
    fraction: bits[FRACTION_SZ],  // Fractional part (no hidden bit).
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
        fraction: bits[FRACTION_SZ]:1 << ((FRACTION_SZ - u32:1) as bits[FRACTION_SZ])
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
        sign, bexp: std::mask_bits<EXP_SZ>(), fraction: bits[FRACTION_SZ]:0
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
        sign, bexp: std::mask_bits<MASK_SZ>() as bits[EXP_SZ], fraction: bits[FRACTION_SZ]:0
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

// Returns the unbiased exponent. For normal numbers it is
// `bexp - 2^EXP_SZ + 1`` and for subnormals it is, `2 - 2^EXP_SZ``. For
// infinity and `NaN``, there are no guarantees, as the unbiased exponent has
// no meaning in that case.
//
// For example, for single precision normal numbers the unbiased exponent is
// `bexp - 127`` and for subnormal numbers it is `-126`.
pub fn unbiased_exponent<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[EXP_SZ] {
    const UEXP_SZ: u32 = EXP_SZ + u32:1;
    const MASK_SZ: u32 = EXP_SZ - u32:1;
    let bias = std::mask_bits<MASK_SZ>() as sN[UEXP_SZ];
    let subnormal_exp = (sN[UEXP_SZ]:1 - bias) as sN[EXP_SZ];
    let bexp = f.bexp as sN[UEXP_SZ];
    let uexp = (bexp - bias) as sN[EXP_SZ];
    if f.bexp == bits[EXP_SZ]:0 { subnormal_exp } else { uexp }
}

#[test]
fn unbiased_exponent_zero_test() {
    let expected = s8:0;
    let actual = unbiased_exponent<u32:8, u32:23>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:127, fraction: u23:0 });
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_positive_test() {
    let expected = s8:1;
    let actual = unbiased_exponent<u32:8, u32:23>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:128, fraction: u23:0 });
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_negative_test() {
    let expected = s8:-1;
    let actual = unbiased_exponent<u32:8, u32:23>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:126, fraction: u23:0 });
    assert_eq(actual, expected);
}

#[test]
fn unbiased_exponent_subnormal_test() {
    let expected = s8:-126;
    let actual = unbiased_exponent<u32:8, u32:23>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0, fraction: u23:0 });
    assert_eq(actual, expected);
}

// Returns the biased exponent which is equal to `unbiased_exponent + 2^EXP_SZ - 1`
//
// Since the function only takes as input the unbiased exponent, it cannot
// distinguish between normal and subnormal numbers, as a result it assumes that
// the input is the exponent for a normal number.
pub fn bias<EXP_SZ: u32, FRACTION_SZ: u32>(unbiased_exponent: sN[EXP_SZ]) -> bits[EXP_SZ] {
    const UEXP_SZ: u32 = EXP_SZ + u32:1;
    const MASK_SZ: u32 = EXP_SZ - u32:1;
    let bias = std::mask_bits<MASK_SZ>() as sN[UEXP_SZ];
    let extended_unbiased_exp = unbiased_exponent as sN[UEXP_SZ];
    (extended_unbiased_exp + bias) as bits[EXP_SZ]
}

#[test]
fn bias_test() {
    let expected = u8:127;
    let actual = bias<u32:8, u32:23>(s8:0);
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
        fraction: x as bits[FRACTION_SZ]
    }
}

//  Round to nearest, ties to even (aka roundTiesToEven).
// if truncated bits > halfway bit: round up.
// if truncated bits < halfway bit: round down.
// if truncated bits == halfway bit and lsb bit is odd: round up.
// if truncated bits == halfway bit and lsb bit is even: round down.
fn rne<FRACTION_SZ: u32, LSB_INDEX_SZ: u32 = {std::clog2(FRACTION_SZ)}>
    (fraction: uN[FRACTION_SZ], lsb_idx: uN[LSB_INDEX_SZ]) -> bool {
    let lsb_bit_mask = uN[FRACTION_SZ]:1 << lsb_idx;
    let halfway_idx = lsb_idx as uN[FRACTION_SZ] - uN[FRACTION_SZ]:1;
    let halfway_bit_mask = uN[FRACTION_SZ]:1 << halfway_idx;
    let trunc_mask = (uN[FRACTION_SZ]:1 << lsb_idx) - uN[FRACTION_SZ]:1;
    let trunc_bits = trunc_mask & fraction;
    let trunc_bits_gt_half = trunc_bits > halfway_bit_mask;
    let trunc_bits_are_halfway = trunc_bits == halfway_bit_mask;
    let to_fraction_is_odd = (fraction & lsb_bit_mask) == lsb_bit_mask;
    let round_to_even = trunc_bits_are_halfway && to_fraction_is_odd;
    let round_up = trunc_bits_gt_half || round_to_even;
    round_up
}

#[test]
fn rne_test() {
    assert_eq(rne(u5:0b01101, u3:3), true);  // >halfway bit.
    assert_eq(rne(u5:0b01001, u3:3), false);  // <halfway bit.
    assert_eq(rne(u5:0b01100, u3:3), true);  // ==halfway bit and lsb odd.
    assert_eq(rne(u5:0b00100, u3:3), false);  // ==halfway bit and lsb even.
    assert_eq(rne(u5:0b000000, u3:3), false);  // 0 fraction.
    assert_eq(rne(u8:0b11000001, u3:0b111), true);  // max lsb index, >halfway bit.
    assert_eq(rne(u8:0b10000000, u3:0b111), false);  // max lsb index, <halfway bit.
    assert_eq(rne(u8:0b11000000, u3:0b111), true);  // max lsb index, ==halfway bit and lsb odd.
    assert_eq(rne(u8:0b01000000, u3:0b111), false);  // max lsb index, ==halfway bit and lsb even.
    assert_eq(rne(u5:0b11111, u3:0b111), true);  // overflow lsb index.
}

#[quickcheck]
fn rne_overflow_always_rounds_up(f: u5) -> bool { rne(f, u3:0b111) }

// Casts the fixed point number to a floating point number using RNE
// (Round to Nearest Even) as the rounding mode.
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
    let bexp = bias<EXP_SZ, FRACTION_SZ>(exp as sN[EXP_SZ]);

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
    let round_up = rne(extended_fraction, lsb_idx as uN[std::clog2(EXTENDED_FRACTION_SZ)]);
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
        _ => qnan<EXP_SZ, FRACTION_SZ>(),
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
    let bexp = bias<EXP_SZ, FRACTION_SZ>(exp as sN[EXP_SZ]);

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
        _ => qnan<EXP_SZ, FRACTION_SZ>(),
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
        Float { sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:1), fraction: uN[FRAC_SZ]:0 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:2),
        Float { sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:1), fraction: uN[FRAC_SZ]:0 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:3),
        Float {
            sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:1), fraction: uN[FRAC_SZ]:0b10000
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:3),
        Float {
            sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:1), fraction: uN[FRAC_SZ]:0b10000
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b111000),
        Float {
            sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:5), fraction: uN[FRAC_SZ]:0b11000
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b111000),
        Float {
            sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:5), fraction: uN[FRAC_SZ]:0b11000
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b1110000),
        Float {
            sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11000
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b1110000),
        Float {
            sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11000
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b111111),
        Float {
            sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:5), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b111111),
        Float {
            sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:5), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b1111110),
        Float {
            sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b1111110),
        Float {
            sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b1111111),
        Float {
            sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b1111111),
        Float {
            sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:6), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b01111111111111111),
        Float {
            sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:15), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b01111111111111111),
        Float {
            sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:15), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b00000011111111111),
        Float {
            sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:10), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b00000011111111111),
        Float {
            sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:10), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[17]:0b00000011111111000),
        Float {
            sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:10), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[17]:0b00000011111111000),
        Float {
            sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:10), fraction: uN[FRAC_SZ]:0b11111
        });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(sN[20]:0b01000000000000000),
        Float { sign: false, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:15), fraction: uN[FRAC_SZ]:0 });

    assert_eq(
        cast_from_fixed_using_rz<EXP_SZ, FRAC_SZ>(-sN[20]:0b01000000000000000),
        Float { sign: true, bexp: bias<EXP_SZ, FRAC_SZ>(sN[EXP_SZ]:15), fraction: uN[FRAC_SZ]:0 });

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

// Upcast the given apfloat to another (larger) apfloat representation.
// Note: denormal inputs get flushed to zero.
pub fn upcast<TO_EXP_SZ: u32, TO_FRACTION_SZ: u32, FROM_EXP_SZ: u32, FROM_FRACTION_SZ: u32>
    (f: APFloat<FROM_EXP_SZ, FROM_FRACTION_SZ>) -> APFloat<TO_EXP_SZ, TO_FRACTION_SZ> {
    const FROM_SZ: u32 = u32:1 + FROM_EXP_SZ + FROM_FRACTION_SZ;
    const TO_SZ: u32 = u32:1 + TO_EXP_SZ + TO_FRACTION_SZ;
    const IS_UPCAST = FROM_SZ < TO_SZ;
    const SAME_SIZE = TO_EXP_SZ == FROM_EXP_SZ && TO_FRACTION_SZ == FROM_FRACTION_SZ;
    const_assert!(IS_UPCAST || SAME_SIZE);

    if SAME_SIZE {
        // Force this function to reduce to either a trivial (i.e. just a `subnormals_to_zero`
        // call) or real upcast depending on how it's parameterized. The `match` in the `else` block
        // would otherwise obscure the fact that it's dead. The `flatten` and `unflatten` will
        // disappear, and are necessary because it's invalid to explicitly just return `f` without a
        // `constexpr if`-type construct.
        subnormals_to_zero(
            unflatten<TO_EXP_SZ, TO_FRACTION_SZ>(
                flatten(f) as uN[u32:1 + TO_EXP_SZ + TO_FRACTION_SZ]))
    } else {
        match tag(f) {
            APFloatTag::NAN => qnan<TO_EXP_SZ, TO_FRACTION_SZ>(),
            APFloatTag::INFINITY => inf<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign),
            APFloatTag::ZERO => zero<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign),
            APFloatTag::SUBNORMAL => zero<TO_EXP_SZ, TO_FRACTION_SZ>(f.sign),
            APFloatTag::NORMAL => {
                // use `sN+1` to preserve source bexp sign.
                const FROM_EXP_SZ_PLUS_1 = FROM_EXP_SZ + u32:1;
                type FromExpOffsetT = sN[FROM_EXP_SZ_PLUS_1];
                type ToExpOffsetT = sN[TO_EXP_SZ];
                // substract `2^(FROM_EXP_SZ-1) - 1` to retrieve the true exponent.
                const FROM_EXP_SZ_MINUS_1 = FROM_EXP_SZ - u32:1;
                const FROM_EXP_OFFSET =
                    (FromExpOffsetT:1 << FROM_EXP_SZ_MINUS_1) - FromExpOffsetT:1;
                let from_exp = f.bexp as FromExpOffsetT - FROM_EXP_OFFSET;
                // add 2^(TO_EXP_SZ-1) - 1 to contrust back offset encoded exponent.
                const TO_EXP_SZ_MINUS_1 = TO_EXP_SZ - u32:1;
                const TO_EXP_OFFSET = (ToExpOffsetT:1 << TO_EXP_SZ_MINUS_1) - ToExpOffsetT:1;
                let to_bexp = (from_exp as ToExpOffsetT + TO_EXP_OFFSET) as uN[TO_EXP_SZ];
                // shift fraction to destination size.
                let FROM_TO_FRACTION_SHIFT = TO_FRACTION_SZ - FROM_FRACTION_SZ;
                let to_fraction = (f.fraction as uN[TO_FRACTION_SZ]) << FROM_TO_FRACTION_SHIFT;
                APFloat { sign: f.sign, bexp: to_bexp, fraction: to_fraction }
            },
            _ => fail!("unsupported_kind", qnan<TO_EXP_SZ, TO_FRACTION_SZ>()),
        }
    }
}

#[test]
fn upcast_test() {
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
    let denormal_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { bexp: u8:0, ..one_dot_5_bf16 };
    let zero_f64 = zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0);
    let neg_denormal_bf16 = APFloat<BF16_EXP_SZ, BF16_FRACTION_SZ> { sign: u1:1, ..denormal_bf16 };
    let neg_zero_f64 = zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1);
    let zero_bf16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0);
    let neg_zero_bf16 = zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1);

    assert_eq(upcast<F64_EXP_SZ, F64_FRACTION_SZ>(one_bf16), one_f64);
    assert_eq(upcast<F64_EXP_SZ, F64_FRACTION_SZ>(one_dot_5_bf16), one_dot_5_f64);
    assert_eq(
        upcast<F64_EXP_SZ, F64_FRACTION_SZ>(qnan<BF16_EXP_SZ, BF16_FRACTION_SZ>()),
        qnan<F64_EXP_SZ, F64_FRACTION_SZ>());
    assert_eq(
        upcast<F64_EXP_SZ, F64_FRACTION_SZ>(inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0)),
        inf<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0));
    assert_eq(
        upcast<F64_EXP_SZ, F64_FRACTION_SZ>(inf<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1)),
        inf<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1));
    assert_eq(
        upcast<F64_EXP_SZ, F64_FRACTION_SZ>(zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:0)),
        zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:0));
    assert_eq(
        upcast<F64_EXP_SZ, F64_FRACTION_SZ>(zero<BF16_EXP_SZ, BF16_FRACTION_SZ>(u1:1)),
        zero<F64_EXP_SZ, F64_FRACTION_SZ>(u1:1));
    assert_eq(upcast<F64_EXP_SZ, F64_FRACTION_SZ>(denormal_bf16), zero_f64);
    assert_eq(upcast<F64_EXP_SZ, F64_FRACTION_SZ>(neg_denormal_bf16), neg_zero_f64);
    assert_eq(upcast<BF16_EXP_SZ, BF16_FRACTION_SZ>(denormal_bf16), zero_bf16);
    assert_eq(upcast<BF16_EXP_SZ, BF16_FRACTION_SZ>(neg_denormal_bf16), neg_zero_bf16);
    assert_eq(upcast<BF16_EXP_SZ, BF16_FRACTION_SZ>(zero_bf16), zero_bf16);
    assert_eq(upcast<BF16_EXP_SZ, BF16_FRACTION_SZ>(one_bf16), one_bf16);
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
            sign, bexp: exp - (leading_zeros as bits[EXP_SZ]), fraction: normalized_fraction
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
    let result = Float {
        sign: fraction.sign,
        bexp: bias<EXP_SZ, FRACTION_SZ>(exp as sN[EXP_SZ]),
        fraction: fraction.fraction
    };

    // Handle overflow.
    let result = if exp > MAX_EXPONENT { inf<EXP_SZ, FRACTION_SZ>(fraction.sign) } else { result };

    // Handle underflow, taking into account the case that underflow rounds back
    // up to a normal number. If this was not a DAZ module, we'd have to deal with
    // denormal 'result' here.
    let underflow_result = if exp == (MIN_EXPONENT - s33:1) &&
    fraction.fraction == std::mask_bits<FRACTION_SZ>() {
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
    let result = if to_cast.bexp < bias<EXP_SZ, FRACTION_SZ>(sN[EXP_SZ]:0) {
        sN[NUM_DST_BITS]:0
    } else {
        result
    };

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

// Returns an APFloat with all its bits past the decimal point set to 0.
pub fn round_towards_zero<EXP_SZ: u32, FRACTION_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
    const EXTENDED_FRACTION_SZ: u32 = FRACTION_SZ + u32:1;
    let exp = unbiased_exponent(x) as s32;
    let mask = !((u32:1 << ((FRACTION_SZ as u32) - (exp as u32))) - u32:1);
    let trunc_fraction = x.fraction & (mask as uN[FRACTION_SZ]);
    let result =
        APFloat<EXP_SZ, FRACTION_SZ> { sign: x.sign, bexp: x.bexp, fraction: trunc_fraction };

    let result = if exp >= (FRACTION_SZ as s32) { x } else { result };
    let result = if exp < s32:0 { zero<EXP_SZ, FRACTION_SZ>(x.sign) } else { result };
    let result = if is_nan<EXP_SZ, FRACTION_SZ>(x) { qnan<EXP_SZ, FRACTION_SZ>() } else { result };
    let result = if x.bexp == (bits[EXP_SZ]:255) { x } else { result };
    result
}

#[test]
fn round_towards_zero_test() {
    // Special cases.
    assert_eq(round_towards_zero(zero<u32:8, u32:23>(u1:0)), zero<u32:8, u32:23>(u1:0));
    assert_eq(round_towards_zero(zero<u32:8, u32:23>(u1:1)), zero<u32:8, u32:23>(u1:1));
    assert_eq(round_towards_zero(qnan<u32:8, u32:23>()), qnan<u32:8, u32:23>());
    assert_eq(round_towards_zero(inf<u32:8, u32:23>(u1:0)), inf<u32:8, u32:23>(u1:0));
    assert_eq(round_towards_zero(inf<u32:8, u32:23>(u1:1)), inf<u32:8, u32:23>(u1:1));

    // Truncate all.
    let fraction = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:50, fraction: u23:0x7fffff };
    assert_eq(round_towards_zero(fraction), zero<u32:8, u32:23>(u1:0));

    let fraction = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:126, fraction: u23:0x7fffff };
    assert_eq(round_towards_zero(fraction), zero<u32:8, u32:23>(u1:0));

    // Truncate all but hidden bit.
    let fraction = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:127, fraction: u23:0x7fffff };
    assert_eq(round_towards_zero(fraction), one<u32:8, u32:23>(u1:0));

    // Truncate some.
    let fraction = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:128, fraction: u23:0x7fffff };
    let trunc_fraction =
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:128, fraction: u23:0x400000 };
    assert_eq(round_towards_zero(fraction), trunc_fraction);

    let fraction = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:149, fraction: u23:0x7fffff };
    let trunc_fraction =
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:149, fraction: u23:0x7ffffe };
    assert_eq(round_towards_zero(fraction), trunc_fraction);

    // Truncate none.
    let fraction = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:200, fraction: u23:0x7fffff };
    assert_eq(round_towards_zero(fraction), fraction);

    let fraction = APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:200, fraction: u23:0x7fffff };
    assert_eq(round_towards_zero(fraction), fraction);
}

// Returns the signed integer part of the input float, truncating any
// fractional bits if necessary.
pub fn to_int<EXP_SZ: u32, FRACTION_SZ: u32, RESULT_SZ: u32>
    (x: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[RESULT_SZ] {
    const WIDE_FRACTION: u32 = FRACTION_SZ + u32:1;
    const MAX_FRACTION_SZ: u32 = std::umax(RESULT_SZ, WIDE_FRACTION);
    let exp = unbiased_exponent(x);

    let fraction = (x.fraction as uN[WIDE_FRACTION] | (uN[WIDE_FRACTION]:1 << FRACTION_SZ)) as
                   uN[MAX_FRACTION_SZ];

    // Switch between base special cases before doing fancier cases below.
    // Default case: exponent == FRACTION_SZ.
    let result = match (exp, x.fraction) {
        (sN[EXP_SZ]:0, _) => uN[MAX_FRACTION_SZ]:1,
        (sN[EXP_SZ]:1, uN[FRACTION_SZ]:0) => uN[MAX_FRACTION_SZ]:0,
        (sN[EXP_SZ]:1, uN[FRACTION_SZ]:1) => uN[MAX_FRACTION_SZ]:1,
        _ => fraction,
    };

    let result = if exp < sN[EXP_SZ]:0 { uN[MAX_FRACTION_SZ]:0 } else { result };

    // For most cases, we need to either shift the "ones" place from
    // FRACTION_SZ + 1 bits down closer to 0 (if exp < FRACTION_SZ) else we
    // need to move it away from 0 if the exponent is bigger.
    let result = if (exp as u32) < FRACTION_SZ {
        (fraction >> (FRACTION_SZ - (exp as u32)))
    } else {
        result
    };
    let result =
        if exp as u32 > FRACTION_SZ { (fraction << ((exp as u32) - FRACTION_SZ)) } else { result };

    // Clamp high if out of bounds, infinite, or NaN.
    let exp_oob = exp as s32 >= (RESULT_SZ as s32 - s32:1);
    let result = if exp_oob || is_inf(x) || is_nan(x) {
        (uN[MAX_FRACTION_SZ]:1 << (RESULT_SZ - u32:1))
    } else {
        result
    };

    // Reduce to the target size, preserving signedness.
    let result = result as sN[MAX_FRACTION_SZ];
    let result = if !x.sign { result } else { -result };
    result as sN[RESULT_SZ]
}

// TODO(rspringer): Create a broadly-applicable normalize test, that
// could be used for multiple type instantiations (without needing
// per-specialization data to be specified by a user).

#[test]
fn to_int_test() {
    let expected = s32:0;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x1, fraction: u23:0x0 });
    assert_eq(expected, actual);

    let expected = s32:1;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x7f, fraction: u23:0xa5a5 });
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

    let expected = s16:0x8000;
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

    let expected = s32:0x80000000;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0x9e, fraction: u23:0x0 });
    assert_eq(expected, actual);

    let expected = s32:0x80000000;
    let actual = to_int<u32:8, u32:23, u32:32>(
        APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0xff, fraction: u23:0x0 });
    assert_eq(expected, actual);
}

fn compound_adder<WIDTH: u32>(a: uN[WIDTH], b: uN[WIDTH]) -> (uN[WIDTH], uN[WIDTH]) {
    (a + b, a + b + uN[WIDTH]:1)
}

// Calculate difference of two positive values and return values in sign-magnitude
// form. Returns sign-magnitude tuple (|a| - |b| <= 0, abs(|a| - |b|)).
// Note, this returns -0 if (a == b), which is used in our application, which is good
// for testing if strictly |a| > |b|.
fn sign_magnitude_difference<WIDTH: u32>(a: uN[WIDTH], b: uN[WIDTH]) -> (bool, uN[WIDTH]) {
    // 1's complement internally, then use the following observation.
    //    abs(|A| - |B|) =   |A| + |~B| + 1 iff |A| - |B| >  0 (end around carry needed)
    //                     ~(|A| + |~B|)    iff |A| - |B| <= 0
    // We use the compound_adder() to efficiently prepare sum + 1 to select result

    type WidthWithCarry = uN[WIDTH + u32:1];
    let (sum, incremented_sum) = compound_adder(a as WidthWithCarry, !b as WidthWithCarry);
    let a_is_less_equal: bool = !sum[-1:];  // Sign bit overflow in the carry

    let abs_difference = if a_is_less_equal { !sum } else { incremented_sum };
    (a_is_less_equal, abs_difference as uN[WIDTH])
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
    let leading_zeroes = clz(abs_fraction);

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
        sign: result_sign, bexp: result_exponent, fraction: result_fraction as uN[FRACTION_SZ]
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

    // 1. Get and expand mantissas.
    let x_fraction = (x.fraction as uN[WIDE_FRACTION]) |
                     (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));
    let y_fraction = (y.fraction as uN[WIDE_FRACTION]) |
                     (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));

    // 1a. Flush subnorms to 0.
    let x_fraction = if is_zero_or_subnormal(x) { uN[WIDE_FRACTION]:0 } else { x_fraction };
    let y_fraction = if is_zero_or_subnormal(y) { uN[WIDE_FRACTION]:0 } else { y_fraction };

    // 2. Multiply integer mantissas.
    let fraction = x_fraction * y_fraction;

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
    let exp =
        if is_zero_or_subnormal(x) || is_zero_or_subnormal(y) { sN[SIGNED_EXP]:0 } else { exp };

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
    let has_0_arg = is_zero_or_subnormal(x) || is_zero_or_subnormal(y);
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

    // Flush subnorms.
    let a_fraction = if a.bexp == uN[EXP_SZ]:0 { uN[WIDE_FRACTION]:0 } else { a_fraction };
    let b_fraction = if b.bexp == uN[EXP_SZ]:0 { uN[WIDE_FRACTION]:0 } else { b_fraction };
    let fraction = a_fraction * b_fraction;

    // Normalize - shift left one place if the top bit is 0.
    let fraction_shift = fraction[-1:] as uN[WIDE_FRACTION];
    let fraction = if fraction_shift == uN[WIDE_FRACTION]:0 { fraction << 1 } else { fraction };

    // e.g., for floats, 0xff -> 0x7f, A.K.A. 127, the exponent bias.
    let bias = std::mask_bits<EXP_SZ>() as sN[EXP_SIGN_CARRY] >> 1;
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
    let has_0_arg = a.bexp == uN[EXP_SZ]:0 || b.bexp == uN[EXP_SZ]:0;
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

    let greater_exp =
        if ab.bexp > c.bexp as uN[EXP_CARRY] { ab.bexp } else { c.bexp as uN[EXP_CARRY] };
    let greater_sign = if ab.bexp > c.bexp as uN[EXP_CARRY] { ab.sign } else { c.sign };

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
    let rshift_ab = greater_exp - ab.bexp;
    let rshift_c = greater_exp - (c.bexp as uN[EXP_CARRY]);
    let shifted_ab = wide_ab >> rshift_ab;
    let shifted_c = wide_c >> rshift_c;

    // Calculate the sticky bits.
    let dropped_ab = wide_ab << ((WIDE_FRACTION as uN[EXP_CARRY] - rshift_ab) as uN[WIDE_FRACTION]);
    let dropped_c = wide_c << ((WIDE_FRACTION as uN[EXP_CARRY] - rshift_c) as uN[WIDE_FRACTION]);
    let dropped_c = if rshift_c >= (WIDE_FRACTION as uN[EXP_CARRY]) { wide_c } else { dropped_c };
    let sticky_ab = (dropped_ab != uN[WIDE_FRACTION]:0) as uN[WIDE_FRACTION];
    let sticky_c = (dropped_c != uN[WIDE_FRACTION]:0) as uN[WIDE_FRACTION];

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
        fraction: result_fraction as uN[FRACTION_SZ]
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
