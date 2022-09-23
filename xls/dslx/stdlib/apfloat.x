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
import std

pub struct APFloat<EXP_SZ:u32, FRACTION_SZ:u32> {
  sign: bits[1],  // Sign bit.
  bexp: bits[EXP_SZ],  // Biased exponent.
  fraction:  bits[FRACTION_SZ],  // Fractional part (no hidden bit).
}

pub enum APFloatTag : u3 {
  NAN       = 0,
  INFINITY  = 1,
  SUBNORMAL = 2,
  ZERO      = 3,
  NORMAL    = 4,
}

pub fn tag<EXP_SZ:u32, FRACTION_SZ:u32>(input_float: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloatTag {
  const EXPR_MASK = std::mask_bits<EXP_SZ>();
  match (input_float.bexp, input_float.fraction) {
    (uN[EXP_SZ]:0, uN[FRACTION_SZ]:0) => APFloatTag::ZERO,
    (uN[EXP_SZ]:0,            _) => APFloatTag::SUBNORMAL,
    (   EXPR_MASK, uN[FRACTION_SZ]:0) => APFloatTag::INFINITY,
    (   EXPR_MASK,            _) => APFloatTag::NAN,
    (           _,            _) => APFloatTag::NORMAL,
  }
}

pub fn qnan<EXP_SZ:u32, FRACTION_SZ:u32>() -> APFloat<EXP_SZ, FRACTION_SZ> {
  APFloat<EXP_SZ, FRACTION_SZ> {
    sign: bits[1]:0,
    bexp: std::mask_bits<EXP_SZ>() as bits[EXP_SZ],
    fraction: bits[FRACTION_SZ]:1 << ((FRACTION_SZ - u32:1) as bits[FRACTION_SZ])
  }
}

#[test]
fn qnan_test() {
  let expected = APFloat<u32:8, u32:23> {
    sign: u1:0, bexp: u8:0xff, fraction: u23:0x400000,
  };
  let actual = qnan<u32:8, u32:23>();
  let _ = assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:0, bexp: u4:0xf, fraction: u2:0x2,
  };
  let actual = qnan<u32:4, u32:2>();
  let _ = assert_eq(actual, expected);
  ()
}

pub fn zero<EXP_SZ:u32, FRACTION_SZ:u32>(sign: bits[1])
    -> APFloat<EXP_SZ, FRACTION_SZ> {
  APFloat<EXP_SZ, FRACTION_SZ>{
    sign: sign,
    bexp: bits[EXP_SZ]:0,
    fraction: bits[FRACTION_SZ]:0 }
}

#[test]
fn zero_test() {
  let expected = APFloat<u32:8, u32:23> {
    sign: u1:0, bexp: u8:0x0, fraction: u23:0x0,
  };
  let actual = zero<u32:8, u32:23>(u1:0);
  let _ = assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:1, bexp: u4:0x0, fraction: u2:0x0,
  };
  let actual = zero<u32:4, u32:2>(u1:1);
  let _ = assert_eq(actual, expected);
  ()
}

pub fn one<EXP_SZ:u32, FRACTION_SZ:u32, MASK_SZ:u32 = EXP_SZ - u32:1>(
    sign: bits[1])
    -> APFloat<EXP_SZ, FRACTION_SZ> {
  APFloat<EXP_SZ, FRACTION_SZ>{
    sign: sign,
    bexp: std::mask_bits<MASK_SZ>() as bits[EXP_SZ],
    fraction: bits[FRACTION_SZ]:0
  }
}

#[test]
fn one_test() {
  let expected = APFloat<u32:8, u32:23> {
    sign: u1:0, bexp: u8:0x7f, fraction: u23:0x0,
  };
  let actual = one<u32:8, u32:23>(u1:0);
  let _ = assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:0, bexp: u4:0x7, fraction: u2:0x0,
  };
  let actual = one<u32:4, u32:2>(u1:0);
  let _ = assert_eq(actual, expected);
  ()
}

pub fn inf<EXP_SZ:u32, FRACTION_SZ:u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
  APFloat<EXP_SZ, FRACTION_SZ>{
    sign: sign,
    bexp: std::mask_bits<EXP_SZ>(),
    fraction: bits[FRACTION_SZ]:0
  }
}

#[test]
fn inf_test() {
  let expected = APFloat<u32:8, u32:23> {
    sign: u1:0, bexp: u8:0xff, fraction: u23:0x0,
  };
  let actual = inf<u32:8, u32:23>(u1:0);
  let _ = assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:0, bexp: u4:0xf, fraction: u2:0x0,
  };
  let actual = inf<u32:4, u32:2>(u1:0);
  let _ = assert_eq(actual, expected);
  ()
}

// Returns the unbiased exponent.
// For normal numbers this is bexp - 127 and for subnormal numbers this is -126.
//
// More generally, for normal numbers it is bexp - 2^EXP_SZ + 1 and for
// subnormals it is, 2 - 2^EXP_SZ.
//
// For infinity and nan, there are no guarantees, as the unbiased exponent has
// no meaning in that case.
pub fn unbiased_exponent<EXP_SZ:u32,
                         FRACTION_SZ:u32,
                         UEXP_SZ:u32 = EXP_SZ + u32:1,
                         MASK_SZ:u32 = EXP_SZ - u32:1>
                         (f: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[EXP_SZ] {
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
  let _ = assert_eq(actual, expected);
  ()
}

#[test]
fn unbiased_exponent_positive_test() {
  let expected = s8:1;
  let actual = unbiased_exponent<u32:8, u32:23>(
      APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:128, fraction: u23:0 });
  let _ = assert_eq(actual, expected);
  ()
}

#[test]
fn unbiased_exponent_negative_test() {
  let expected = s8:-1;
  let actual = unbiased_exponent<u32:8, u32:23>(
      APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:126, fraction: u23:0 });
  let _ = assert_eq(actual, expected);
  ()
}

#[test]
fn unbiased_exponent_subnormal_test() {
  let expected = s8:-126;
  let actual = unbiased_exponent<u32:8, u32:23>(
      APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0, fraction: u23:0 });
  let _ = assert_eq(actual, expected);
  ()
}

// Returns the biased exponent.
//
// Since the function only takes as input the unbiased exponent, it cannot
// distinguish between normal and subnormal numbers, as a result it assumes that
// the input is the exponent for a normal number.
//
// As a result the answer is just unbiased_exponent + 2^EXP_SZ - 1
pub fn bias<EXP_SZ: u32,
            FRACTION_SZ: u32,
            UEXP_SZ: u32 = EXP_SZ + u32:1,
            MASK_SZ: u32 = EXP_SZ - u32:1>
            (unbiased_exponent: sN[EXP_SZ]) -> bits[EXP_SZ] {
  let bias = std::mask_bits<MASK_SZ>() as sN[UEXP_SZ];
  let extended_unbiased_exp = unbiased_exponent as sN[UEXP_SZ];
  (extended_unbiased_exp + bias) as bits[EXP_SZ]
}

#[test]
fn bias_test() {
  let expected = u8:127;
  let actual = bias<u32:8, u32:23>(s8:0);
  let _ = assert_eq(expected, actual);
  ()
}

pub fn flatten<EXP_SZ:u32, FRACTION_SZ:u32, TOTAL_SZ:u32 = u32:1+EXP_SZ+FRACTION_SZ>(
    x: APFloat<EXP_SZ, FRACTION_SZ>) -> bits[TOTAL_SZ] {
  x.sign ++ x.bexp ++ x.fraction
}

pub fn unflatten<EXP_SZ:u32, FRACTION_SZ:u32,
    TOTAL_SZ:u32 = u32:1+EXP_SZ+FRACTION_SZ,
    SIGN_OFFSET:u32 = EXP_SZ+FRACTION_SZ>(
    x: bits[TOTAL_SZ]) -> APFloat<EXP_SZ, FRACTION_SZ> {
  APFloat<EXP_SZ, FRACTION_SZ>{
      sign: (x >> (SIGN_OFFSET as bits[TOTAL_SZ])) as bits[1],
      bexp: (x >> (FRACTION_SZ as bits[TOTAL_SZ])) as bits[EXP_SZ],
      fraction: x as bits[FRACTION_SZ],
  }
}

// Cast the fixed point number to a floating point number.
pub fn cast_from_fixed<EXP_SZ:u32, FRACTION_SZ:u32, UEXP_SZ:u32 = EXP_SZ + u32:1,
  NUM_SRC_BITS:u32, EXTENDED_FRACTION_SZ:u32 = FRACTION_SZ + NUM_SRC_BITS>
  (to_cast: sN[NUM_SRC_BITS]) -> APFloat<EXP_SZ, FRACTION_SZ> {
  // Determine sign.
  let sign = (to_cast as uN[NUM_SRC_BITS])[(NUM_SRC_BITS-u32:1) as s32 : NUM_SRC_BITS as s32];

  // Determine exponent.
  let abs_magnitude = (if sign == u1:0 { to_cast } else { -to_cast }) as uN[NUM_SRC_BITS];
  let lz = clz(abs_magnitude);
  let num_trailing_nonzeros = (NUM_SRC_BITS as uN[NUM_SRC_BITS]) - lz;
  let exp = (num_trailing_nonzeros as uN[UEXP_SZ]) - uN[UEXP_SZ]:1;
  let max_exp_exclusive = uN[UEXP_SZ]:1 << ((EXP_SZ as uN[UEXP_SZ]) - uN[UEXP_SZ]:1);
  let is_inf = exp >= max_exp_exclusive;
  let bexp = bias<EXP_SZ, FRACTION_SZ>(exp as sN[EXP_SZ]);

  // Determine fraction (pre-rounding).
  let extended_fraction = abs_magnitude ++ uN[FRACTION_SZ]:0;
  let fraction = extended_fraction >>
    ((num_trailing_nonzeros - uN[NUM_SRC_BITS]:1) as uN[EXTENDED_FRACTION_SZ]);
  let fraction = fraction[0 : FRACTION_SZ as s32];

  // Round fraction (round to nearest, half to even).
  let lsb_idx = (num_trailing_nonzeros as uN[EXTENDED_FRACTION_SZ])
    - uN[EXTENDED_FRACTION_SZ]:1;
  let halfway_idx = lsb_idx - uN[EXTENDED_FRACTION_SZ]:1;
  let halfway_bit_mask = uN[EXTENDED_FRACTION_SZ]:1 << halfway_idx;
  let trunc_mask = (uN[EXTENDED_FRACTION_SZ]:1 << lsb_idx) - uN[EXTENDED_FRACTION_SZ]:1;
  let trunc_bits = trunc_mask & extended_fraction;
  let trunc_bits_gt_half = trunc_bits > halfway_bit_mask;
  let trunc_bits_are_halfway = trunc_bits == halfway_bit_mask;
  let fraction_is_odd = fraction[0:1] == u1:1;
  let round_to_even = trunc_bits_are_halfway && fraction_is_odd;
  let round_up = trunc_bits_gt_half || round_to_even;
  let fraction = if round_up { fraction + uN[FRACTION_SZ]:1 } else { fraction };

  // Check if rounding up causes an exponent increment.
  let overflow = round_up && (fraction == uN[FRACTION_SZ]:0);
  let bexp = if overflow { (bexp + uN[EXP_SZ]:1) } else { bexp };

  // Check if rounding up caused us to overflow to infinity.
  let is_inf = is_inf || bexp == std::mask_bits<EXP_SZ>();

  let result =
    APFloat<EXP_SZ, FRACTION_SZ>{
      sign: sign,
      bexp: bexp,
      fraction: fraction
  };

  let is_zero = abs_magnitude == uN[NUM_SRC_BITS]:0;
  let result = if is_inf { inf<EXP_SZ, FRACTION_SZ>(sign) } else { result };
  let result = if is_zero { zero<EXP_SZ, FRACTION_SZ>(sign) } else { result };
  result
}

#[test]
fn cast_from_fixed_test() {
  // Zero is a special case.
  let zero_float = zero<u32:4, u32:4>(u1:0);
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:0), zero_float);

  // +/-1
  let one_float = one<u32:4, u32:4>(u1:0);
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:1), one_float);
  let none_float = one<u32:4, u32:4>(u1:1);
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:-1), none_float);

  // +/-4
  let four_float =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:9,
      fraction: u4:0
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:4), four_float);
  let nfour_float =
    APFloat<u32:4, u32:4>{
      sign: u1:1,
      bexp: u4:9,
      fraction: u4:0
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:-4), nfour_float);

  // Cast maximum representable exponent in target format.
  let max_representable =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:0
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:128), max_representable);

  // Cast minimum non-representable exponent in target format.
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:256),
                    inf<u32:4, u32:4>(u1:0));

  // Test rounding - maximum truncated bits that will round down, even fraction.
  let truncate =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:0
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:131),
                    truncate);

  // Test rounding - maximum truncated bits that will round down, odd fraction.
  let truncate =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:1
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:139),
                    truncate);

  // Test rounding - halfway and already even, round down
  let truncate =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:0
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:132),
                    truncate);

  // Test rounding - halfway and odd, round up
  let round_up =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:2
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:140),
                    round_up);

  // Test rounding - over halfway and even, round up
  let round_up =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:1
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:133),
                    round_up);

  // Test rounding - over halfway and odd, round up
  let round_up =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:2
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:141),
                    round_up);

  // Test rounding - Rounding up increases exponent.
  let round_inc_exponent =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:0
    };
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:126),
                    round_inc_exponent);
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:127),
                    round_inc_exponent);

  // Test rounding - Rounding up overflows to infinity.
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:252),
                    inf<u32:4, u32:4>(u1:0));
  let _ = assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:254),
                    inf<u32:4, u32:4>(u1:0));
  ()
}

pub fn subnormals_to_zero<EXP_SZ:u32, FRACTION_SZ:u32>(
    x: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
  if x.bexp == bits[EXP_SZ]:0 { zero<EXP_SZ, FRACTION_SZ>(x.sign) } else { x }
}

// Returns a normalized APFloat with the given components. 'fraction_with_hidden' is the
// fraction (including the hidden bit). This function only normalizes in the
// direction of decreasing the exponent. Input must be a normal number or
// zero. Dernormals are flushed to zero in the result.
pub fn normalize<EXP_SZ:u32, FRACTION_SZ:u32, WIDE_FRACTION:u32 = FRACTION_SZ + u32:1>(
    sign: bits[1], exp: bits[EXP_SZ], fraction_with_hidden: bits[WIDE_FRACTION])
    -> APFloat<EXP_SZ, FRACTION_SZ> {
  let leading_zeros = clz(fraction_with_hidden) as bits[FRACTION_SZ];
  let zero_value = zero<EXP_SZ, FRACTION_SZ>(sign);
  let zero_fraction = WIDE_FRACTION as bits[FRACTION_SZ];
  let normalized_fraction = (fraction_with_hidden << (leading_zeros as bits[WIDE_FRACTION])) as bits[FRACTION_SZ];

  let is_denormal = exp <= (leading_zeros as bits[EXP_SZ]);
  match (is_denormal, leading_zeros) {
    // Significand is zero.
    (_, zero_fraction) => zero_value,
    // Flush denormals to zero.
    (true, _) => zero_value,
    // Normalize.
    _ => APFloat { sign: sign,
                   bexp: exp - (leading_zeros as bits[EXP_SZ]),
                   fraction: normalized_fraction },
  }
}

// Returns whether or not the given APFloat represents an infinite quantity.
pub fn is_inf<EXP_SZ:u32, FRACTION_SZ:u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction == bits[FRACTION_SZ]:0)
}

// Returns whether or not the given F32 represents NaN.
pub fn is_nan<EXP_SZ:u32, FRACTION_SZ:u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction != bits[FRACTION_SZ]:0)
}

// Returns true if x == 0 or x is a subnormal number.
pub fn is_zero_or_subnormal<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  x.bexp == uN[EXP_SZ]:0
}

// Cast the floating point number to a fixed point number.
// Unrepresentable numbers are cast to the minimum representable
// number (largest magnitude negative number).
pub fn cast_to_fixed<NUM_DST_BITS:u32, EXP_SZ:u32, FRACTION_SZ:u32,
  UEXP_SZ:u32 = EXP_SZ + u32:1,
  EXTENDED_FIXED_SZ:u32 = NUM_DST_BITS + u32:1 + FRACTION_SZ + NUM_DST_BITS>
  (to_cast: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[NUM_DST_BITS] {

  const MIN_FIXED_VALUE = (uN[NUM_DST_BITS]:1 << (
    (NUM_DST_BITS as uN[NUM_DST_BITS]) - uN[NUM_DST_BITS]:1))
    as sN[NUM_DST_BITS];
  const MAX_EXPONENT = NUM_DST_BITS - u32:1;

  // Convert to fixed point and truncate fractional bits.
  let exp = unbiased_exponent(to_cast);
  let result = (uN[NUM_DST_BITS]:0 ++ u1:1
                ++ to_cast.fraction ++ uN[NUM_DST_BITS]:0)
                as uN[EXTENDED_FIXED_SZ];
  let result = result >>
    ((FRACTION_SZ as uN[EXTENDED_FIXED_SZ])
    + (NUM_DST_BITS as uN[EXTENDED_FIXED_SZ])
    - (exp as uN[EXTENDED_FIXED_SZ]));
  let result = result[0:NUM_DST_BITS as s32] as sN[NUM_DST_BITS];
  let result = if to_cast.sign { -result } else { result };

  // NaN and too-large inputs --> MIN_FIXED_VALUE
  let overflow = (exp as u32) >= MAX_EXPONENT;
  let result = if overflow || is_nan(to_cast) { MIN_FIXED_VALUE }
               else { result };
  // Underflow / to_cast < 1 --> 0
  let result =
    if to_cast.bexp < bias<EXP_SZ, FRACTION_SZ>(sN[EXP_SZ]:0) { sN[NUM_DST_BITS]:0 }
    else { result };

  result
}

#[test]
fn cast_to_fixed_test() {
  // Cast +/-0.0
  let _ = assert_eq(
    cast_to_fixed<u32:32>(zero<u32:8, u32:23>(u1:0)), s32:0);
  let _ = assert_eq(
    cast_to_fixed<u32:32>(zero<u32:8, u32:23>(u1:1)), s32:0);

  // Cast +/-1.0
  let _ = assert_eq(
    cast_to_fixed<u32:32>(one<u32:8, u32:23>(u1:0)), s32:1);
  let _ = assert_eq(
    cast_to_fixed<u32:32>(one<u32:8, u32:23>(u1:1)), s32:-1);

  // Cast +/-1.5 --> +/- 1
  let one_point_five = APFloat<u32:8, u32:23>{sign: u1:0,
                                              bexp: u8:0x7f,
                                              fraction:  u1:1 ++ u22:0};
  let _ = assert_eq(
    cast_to_fixed<u32:32>(one_point_five), s32:1);
  let n_one_point_five = APFloat<u32:8, u32:23>{sign: u1:1,
                                                bexp: u8:0x7f,
                                                fraction:  u1:1 ++ u22:0};
  let _ = assert_eq(
    cast_to_fixed<u32:32>(n_one_point_five), s32:-1);

  // Cast +/-4.0
  let four = cast_from_fixed<u32:8, u32:23>(s32:4);
  let neg_four = cast_from_fixed<u32:8, u32:23>(s32:-4);
  let _ = assert_eq(
    cast_to_fixed<u32:32>(four), s32:4);
  let _ = assert_eq(
    cast_to_fixed<u32:32>(neg_four), s32:-4);

  // Cast 7
  let seven = cast_from_fixed<u32:8, u32:23>(s32:7);
  let _ = assert_eq(
    cast_to_fixed<u32:32>(seven), s32:7);

  // Cast big number (more digits left of decimal than hidden bit + fraction).
  let big_num = (u1:0 ++ std::mask_bits<u32:23>() ++ u8:0) as s32;
  let fp_big_num = cast_from_fixed<u32:8, u32:23>(big_num);
  let _ = assert_eq(
    cast_to_fixed<u32:32>(fp_big_num), big_num);

  // Cast large, non-overflowing numbers.
  let big_fit = APFloat<u32:8, u32:23>{sign: u1:0,
                                       bexp: u8:127 + u8:30,
                                       fraction: u23:0x7fffff};
  let _ = assert_eq(
    cast_to_fixed<u32:32>(big_fit),
    (u1:0 ++ u24:0xffffff ++ u7:0) as s32);
  let big_fit = APFloat<u32:8, u32:23>{sign: u1:1,
                                       bexp: u8:127 + u8:30,
                                       fraction: u23:0x7fffff};
  let _ = assert_eq(
    cast_to_fixed<u32:32>(big_fit),
    (s32:0 - (u1:0 ++ u24:0xffffff ++ u7:0) as s32));


  // Cast barely overflowing postive number.
  let big_overflow = APFloat<u32:8, u32:23>{sign: u1:0,
                                            bexp: u8:127 + u8:31,
                                            fraction: u23:0x0};
  let _ = assert_eq(
    cast_to_fixed<u32:32>(big_overflow),
    (u1:1 ++ u31:0) as s32);


  // This produces the largest negative int, but doesn't actually
  // overflow
  let max_negative = APFloat<u32:8, u32:23>{sign: u1:1,
                                            bexp: u8:127 + u8:31,
                                            fraction: u23:0x0};
  let _ = assert_eq(
    cast_to_fixed<u32:32>(max_negative),
    (u1:1 ++ u31:0) as s32);


  // Negative overflow.
  let negative_overflow = APFloat<u32:8, u32:23>{sign: u1:1,
                                            bexp: u8:127 + u8:31,
                                            fraction: u23:0x1};
  let _ = assert_eq(
    cast_to_fixed<u32:32>(negative_overflow),
    (u1:1 ++ u31:0) as s32);


  // NaN input.
  let _ = assert_eq(
    cast_to_fixed<u32:32>(qnan<u32:8, u32:23>()),
    (u1:1 ++ u31:0) as s32);

  ()
}

// Returns u1:1 if x == y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn eq_2<EXP_SZ: u32, FRACTION_SZ: u32>(
    x: APFloat<EXP_SZ, FRACTION_SZ>,
    y: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  if !(is_nan(x) || is_nan(y)) {
    ((flatten(x) == flatten(y))
          || (is_zero_or_subnormal(x) && is_zero_or_subnormal(y)))
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
  let two = APFloat<8, 23> {bexp: one.bexp + uN[8]:1, ..one};
  let neg_inf = inf<u32:8, u32:23>(u1:1);
  let inf = inf<u32:8, u32:23>(u1:0);
  let nan = qnan<u32:8, u32:23>();
  let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
  let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

  // Test unequal.
  let _ = assert_eq(eq_2(one, two), u1:0);
  let _ = assert_eq(eq_2(two, one), u1:0);

  // Test equal.
  let _ = assert_eq(eq_2(neg_zero, zero), u1:1);
  let _ = assert_eq(eq_2(one, one), u1:1);
  let _ = assert_eq(eq_2(two, two), u1:1);

  // Test equal (subnormals and zero).
  let _ = assert_eq(eq_2(zero, zero), u1:1);
  let _ = assert_eq(eq_2(zero, neg_zero), u1:1);
  let _ = assert_eq(eq_2(zero, denormal_1), u1:1);
  let _ = assert_eq(eq_2(denormal_2, denormal_1), u1:1);

  // Test negatives.
  let _ = assert_eq(eq_2(one, neg_one), u1:0);
  let _ = assert_eq(eq_2(neg_one, one), u1:0);
  let _ = assert_eq(eq_2(neg_one, neg_one), u1:1);

  // Special case - inf.
  let _ = assert_eq(eq_2(inf, one), u1:0);
  let _ = assert_eq(eq_2(neg_inf, inf), u1:0);
  let _ = assert_eq(eq_2(inf, inf), u1:1);
  let _ = assert_eq(eq_2(neg_inf, neg_inf), u1:1);

  // Special case - NaN (always returns false).
  let _ = assert_eq(eq_2(one, nan), u1:0);
  let _ = assert_eq(eq_2(neg_one, nan), u1:0);
  let _ = assert_eq(eq_2(inf, nan), u1:0);
  let _ = assert_eq(eq_2(nan, inf), u1:0);
  let _ = assert_eq(eq_2(nan, nan), u1:0);

  ()
}

// Returns u1:1 if x > y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn gt_2<EXP_SZ: u32, FRACTION_SZ: u32>(
    x: APFloat<EXP_SZ, FRACTION_SZ>,
    y: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  // Flush denormals.
  let x = subnormals_to_zero(x);
  let y = subnormals_to_zero(y);

  let gt_exp = x.bexp > y.bexp;
  let eq_exp = x.bexp == y.bexp;
  let gt_fraction = x.fraction > y.fraction;
  let abs_gt = gt_exp || (eq_exp && gt_fraction);
  let result = match(x.sign, y.sign) {
    // Both positive.
    (u1:0, u1:0) => abs_gt,
    // x positive, y negative.
    (u1:0, u1:1) => u1:1,
    // x negative, y positive.
    (u1:1, u1:0) => u1:0,
    // Both negative.
    _ => !abs_gt && !eq_2(x,y)
  };


  if !(is_nan(x) || is_nan(y)) { result }
  else { u1:0 }
}

#[test]
fn test_fp_gt_2() {
  let zero = zero<u32:8, u32:23>(u1:0);
  let neg_one = one<u32:8, u32:23>(u1:1);
  let one = one<u32:8, u32:23>(u1:0);
  let two = APFloat<u32:8, u32:23>{bexp: one.bexp + u8:1, ..one};
  let neg_two = APFloat<u32:8, u32:23>{bexp: neg_one.bexp + u8:1, ..neg_one};
  let neg_inf = inf<u32:8, u32:23>(u1:1);
  let inf = inf<u32:8, u32:23>(u1:0);
  let nan = qnan<u32:8, u32:23>();
  let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
  let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

  // Test unequal.
  let _ = assert_eq(gt_2(one, two), u1:0);
  let _ = assert_eq(gt_2(two, one), u1:1);

  // Test equal.
  let _ = assert_eq(gt_2(one, one), u1:0);
  let _ = assert_eq(gt_2(two, two), u1:0);
  let _ = assert_eq(gt_2(denormal_1, denormal_2), u1:0);
  let _ = assert_eq(gt_2(denormal_2, denormal_1), u1:0);
  let _ = assert_eq(gt_2(denormal_1, zero), u1:0);

  // Test negatives.
  let _ = assert_eq(gt_2(zero, neg_one), u1:1);
  let _ = assert_eq(gt_2(neg_one, zero), u1:0);
  let _ = assert_eq(gt_2(one, neg_one), u1:1);
  let _ = assert_eq(gt_2(neg_one, one), u1:0);
  let _ = assert_eq(gt_2(neg_one, neg_one), u1:0);
  let _ = assert_eq(gt_2(neg_two, neg_two), u1:0);
  let _ = assert_eq(gt_2(neg_one, neg_two), u1:1);
  let _ = assert_eq(gt_2(neg_two, neg_one), u1:0);

  // Special case - inf.
  let _ = assert_eq(gt_2(inf, one), u1:1);
  let _ = assert_eq(gt_2(inf, neg_one), u1:1);
  let _ = assert_eq(gt_2(inf, two), u1:1);
  let _ = assert_eq(gt_2(neg_two, neg_inf), u1:1);
  let _ = assert_eq(gt_2(inf, inf), u1:0);
  let _ = assert_eq(gt_2(neg_inf, inf), u1:0);
  let _ = assert_eq(gt_2(inf, neg_inf), u1:1);
  let _ = assert_eq(gt_2(neg_inf, neg_inf), u1:0);

  // Special case - NaN (always returns false).
  let _ = assert_eq(gt_2(one, nan), u1:0);
  let _ = assert_eq(gt_2(nan, one), u1:0);
  let _ = assert_eq(gt_2(neg_one, nan), u1:0);
  let _ = assert_eq(gt_2(nan, neg_one), u1:0);
  let _ = assert_eq(gt_2(inf, nan), u1:0);
  let _ = assert_eq(gt_2(nan, inf), u1:0);
  let _ = assert_eq(gt_2(nan, nan), u1:0);

  ()
}

// Returns u1:1 if x >= y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn gte_2<EXP_SZ: u32, FRACTION_SZ: u32>(
    x: APFloat<EXP_SZ, FRACTION_SZ>,
    y: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  gt_2(x, y) || eq_2(x,y)
}

#[test]
fn test_fp_gte_2() {
  let zero = zero<u32:8, u32:23>(u1:0);
  let neg_one = one<u32:8, u32:23>(u1:1);
  let one = one<u32:8, u32:23>(u1:0);
  let two = APFloat<u32:8, u32:23>{bexp: one.bexp + u8:1, ..one};
  let neg_two = APFloat<u32:8, u32:23>{bexp: neg_one.bexp + u8:1, ..neg_one};
  let neg_inf = inf<u32:8, u32:23>(u1:1);
  let inf = inf<u32:8, u32:23>(u1:0);
  let nan = qnan<u32:8, u32:23>();
  let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
  let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

  // Test unequal.
  let _ = assert_eq(gte_2(one, two), u1:0);
  let _ = assert_eq(gte_2(two, one), u1:1);

  // Test equal.
  let _ = assert_eq(gte_2(one, one), u1:1);
  let _ = assert_eq(gte_2(two, two), u1:1);
  let _ = assert_eq(gte_2(denormal_1, denormal_2), u1:1);
  let _ = assert_eq(gte_2(denormal_2, denormal_1), u1:1);
  let _ = assert_eq(gte_2(denormal_1, zero), u1:1);

  // Test negatives.
  let _ = assert_eq(gte_2(zero, neg_one), u1:1);
  let _ = assert_eq(gte_2(neg_one, zero), u1:0);
  let _ = assert_eq(gte_2(one, neg_one), u1:1);
  let _ = assert_eq(gte_2(neg_one, one), u1:0);
  let _ = assert_eq(gte_2(neg_one, neg_one), u1:1);
  let _ = assert_eq(gte_2(neg_two, neg_two), u1:1);
  let _ = assert_eq(gte_2(neg_one, neg_two), u1:1);
  let _ = assert_eq(gte_2(neg_two, neg_one), u1:0);

  // Special case - inf.
  let _ = assert_eq(gte_2(inf, one), u1:1);
  let _ = assert_eq(gte_2(inf, neg_one), u1:1);
  let _ = assert_eq(gte_2(inf, two), u1:1);
  let _ = assert_eq(gte_2(neg_two, neg_inf), u1:1);
  let _ = assert_eq(gte_2(inf, inf), u1:1);
  let _ = assert_eq(gte_2(neg_inf, inf), u1:0);
  let _ = assert_eq(gte_2(inf, neg_inf), u1:1);
  let _ = assert_eq(gte_2(neg_inf, neg_inf), u1:1);

  // Special case - NaN (always returns false).
  let _ = assert_eq(gte_2(one, nan), u1:0);
  let _ = assert_eq(gte_2(nan, one), u1:0);
  let _ = assert_eq(gte_2(neg_one, nan), u1:0);
  let _ = assert_eq(gte_2(nan, neg_one), u1:0);
  let _ = assert_eq(gte_2(inf, nan), u1:0);
  let _ = assert_eq(gte_2(nan, inf), u1:0);
  let _ = assert_eq(gte_2(nan, nan), u1:0);

  ()
}

// Returns u1:1 if x <= y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn lte_2<EXP_SZ: u32, FRACTION_SZ: u32>(
    x: APFloat<EXP_SZ, FRACTION_SZ>,
    y: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  if !(is_nan(x) || is_nan(y)) { !gt_2(x,y) }
  else { u1:0 }
}

#[test]
fn test_fp_lte_2() {
  let zero = zero<u32:8, u32:23>(u1:0);
  let neg_one = one<u32:8, u32:23>(u1:1);
  let one = one<u32:8, u32:23>(u1:0);
  let two = APFloat<u32:8, u32:23>{bexp: one.bexp + u8:1, ..one};
  let neg_two = APFloat<u32:8, u32:23>{bexp: neg_one.bexp + u8:1, ..neg_one};
  let neg_inf = inf<u32:8, u32:23>(u1:1);
  let inf = inf<u32:8, u32:23>(u1:0);
  let nan = qnan<u32:8, u32:23>();
  let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
  let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

  // Test unequal.
  let _ = assert_eq(lte_2(one, two), u1:1);
  let _ = assert_eq(lte_2(two, one), u1:0);

  // Test equal.
  let _ = assert_eq(lte_2(one, one), u1:1);
  let _ = assert_eq(lte_2(two, two), u1:1);
  let _ = assert_eq(lte_2(denormal_1, denormal_2), u1:1);
  let _ = assert_eq(lte_2(denormal_2, denormal_1), u1:1);
  let _ = assert_eq(lte_2(denormal_1, zero), u1:1);

  // Test negatives.
  let _ = assert_eq(lte_2(zero, neg_one), u1:0);
  let _ = assert_eq(lte_2(neg_one, zero), u1:1);
  let _ = assert_eq(lte_2(one, neg_one), u1:0);
  let _ = assert_eq(lte_2(neg_one, one), u1:1);
  let _ = assert_eq(lte_2(neg_one, neg_one), u1:1);
  let _ = assert_eq(lte_2(neg_two, neg_two), u1:1);
  let _ = assert_eq(lte_2(neg_one, neg_two), u1:0);
  let _ = assert_eq(lte_2(neg_two, neg_one), u1:1);

  // Special case - inf.
  let _ = assert_eq(lte_2(inf, one), u1:0);
  let _ = assert_eq(lte_2(inf, neg_one), u1:0);
  let _ = assert_eq(lte_2(inf, two), u1:0);
  let _ = assert_eq(lte_2(neg_two, neg_inf), u1:0);
  let _ = assert_eq(lte_2(inf, inf), u1:1);
  let _ = assert_eq(lte_2(neg_inf, inf), u1:1);
  let _ = assert_eq(lte_2(inf, neg_inf), u1:0);
  let _ = assert_eq(lte_2(neg_inf, neg_inf), u1:1);

  // Special case - NaN (always returns false).
  let _ = assert_eq(lte_2(one, nan), u1:0);
  let _ = assert_eq(lte_2(nan, one), u1:0);
  let _ = assert_eq(lte_2(neg_one, nan), u1:0);
  let _ = assert_eq(lte_2(nan, neg_one), u1:0);
  let _ = assert_eq(lte_2(inf, nan), u1:0);
  let _ = assert_eq(lte_2(nan, inf), u1:0);
  let _ = assert_eq(lte_2(nan, nan), u1:0);

  ()
}

// Returns u1:1 if x < y.
// Denormals are Zero (DAZ).
// Always returns false if x or y is NaN.
pub fn lt_2<EXP_SZ: u32, FRACTION_SZ: u32>(
    x: APFloat<EXP_SZ, FRACTION_SZ>,
    y: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  if !(is_nan(x) || is_nan(y)) { !gte_2(x,y) }
  else { u1:0 }
}

#[test]
fn test_fp_lt_2() {
  let zero = zero<u32:8, u32:23>(u1:0);
  let neg_one = one<u32:8, u32:23>(u1:1);
  let one = one<u32:8, u32:23>(u1:0);
  let two = APFloat<u32:8, u32:23>{bexp: one.bexp + u8:1, ..one};
  let neg_two = APFloat<u32:8, u32:23>{bexp: neg_one.bexp + u8:1, ..neg_one};
  let neg_inf = inf<u32:8, u32:23>(u1:1);
  let inf = inf<u32:8, u32:23>(u1:0);
  let nan = qnan<u32:8, u32:23>();
  let denormal_1 = unflatten<u32:8, u32:23>(u32:1);
  let denormal_2 = unflatten<u32:8, u32:23>(u32:2);

  // Test unequal.
  let _ = assert_eq(lt_2(one, two), u1:1);
  let _ = assert_eq(lt_2(two, one), u1:0);

  // Test equal.
  let _ = assert_eq(lt_2(one, one), u1:0);
  let _ = assert_eq(lt_2(two, two), u1:0);
  let _ = assert_eq(lt_2(denormal_1, denormal_2), u1:0);
  let _ = assert_eq(lt_2(denormal_2, denormal_1), u1:0);
  let _ = assert_eq(lt_2(denormal_1, zero), u1:0);

  // Test negatives.
  let _ = assert_eq(lt_2(zero, neg_one), u1:0);
  let _ = assert_eq(lt_2(neg_one, zero), u1:1);
  let _ = assert_eq(lt_2(one, neg_one), u1:0);
  let _ = assert_eq(lt_2(neg_one, one), u1:1);
  let _ = assert_eq(lt_2(neg_one, neg_one), u1:0);
  let _ = assert_eq(lt_2(neg_two, neg_two), u1:0);
  let _ = assert_eq(lt_2(neg_one, neg_two), u1:0);
  let _ = assert_eq(lt_2(neg_two, neg_one), u1:1);

  // Special case - inf.
  let _ = assert_eq(lt_2(inf, one), u1:0);
  let _ = assert_eq(lt_2(inf, neg_one), u1:0);
  let _ = assert_eq(lt_2(inf, two), u1:0);
  let _ = assert_eq(lt_2(neg_two, neg_inf), u1:0);
  let _ = assert_eq(lt_2(inf, inf), u1:0);
  let _ = assert_eq(lt_2(neg_inf, inf), u1:1);
  let _ = assert_eq(lt_2(inf, neg_inf), u1:0);
  let _ = assert_eq(lt_2(neg_inf, neg_inf), u1:0);

  // Special case - NaN (always returns false).
  let _ = assert_eq(lt_2(one, nan), u1:0);
  let _ = assert_eq(lt_2(nan, one), u1:0);
  let _ = assert_eq(lt_2(neg_one, nan), u1:0);
  let _ = assert_eq(lt_2(nan, neg_one), u1:0);
  let _ = assert_eq(lt_2(inf, nan), u1:0);
  let _ = assert_eq(lt_2(nan, inf), u1:0);
  let _ = assert_eq(lt_2(nan, nan), u1:0);

  ()
}

// Set all bits past the decimal point to 0.
pub fn round_towards_zero<EXP_SZ:u32, FRACTION_SZ:u32,
    EXTENDED_FRACTION_SZ:u32 = FRACTION_SZ + u32:1>(
    x: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ> {
  let exp = unbiased_exponent(x) as s32;
  let mask = !((u32:1 << ((FRACTION_SZ as u32) - (exp as u32)))
                - u32:1);
  let trunc_fraction = x.fraction & (mask as uN[FRACTION_SZ]);
  let result = APFloat<EXP_SZ, FRACTION_SZ> {
                sign: x.sign,
                bexp: x.bexp,
                fraction:  trunc_fraction};

  let result = if (exp >= (FRACTION_SZ as s32)) { x } else { result };
  let result = if (exp < s32:0) { zero<EXP_SZ, FRACTION_SZ>(x.sign) }
               else { result };
  let result = if is_nan<EXP_SZ, FRACTION_SZ>(x) { qnan<EXP_SZ, FRACTION_SZ>() }
               else { result };
  let result = if (x.bexp == (bits[EXP_SZ]:255)) { x } else { result };
  result
}

#[test]
fn round_towards_zero_test() {
  // Special cases.
  let _ = assert_eq(round_towards_zero(zero<u32:8, u32:23>(u1:0)),
    zero<u32:8, u32:23>(u1:0));
  let _ = assert_eq(round_towards_zero(zero<u32:8, u32:23>(u1:1)),
    zero<u32:8, u32:23>(u1:1));
  let _ = assert_eq(round_towards_zero(qnan<u32:8, u32:23>()),
    qnan<u32:8, u32:23>());
  let _ = assert_eq(round_towards_zero(inf<u32:8, u32:23>(u1:0)),
    inf<u32:8, u32:23>(u1:0));
  let _ = assert_eq(round_towards_zero(inf<u32:8, u32:23>(u1:1)),
    inf<u32:8, u32:23>(u1:1));

  // Truncate all.
  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:50,
                    fraction:  u23: 0x7fffff
                  };
  let _ = assert_eq(round_towards_zero(fraction),
    zero<u32:8, u32:23>(u1:0));

  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:126,
                    fraction:  u23: 0x7fffff
                  };
  let _ = assert_eq(round_towards_zero(fraction),
    zero<u32:8, u32:23>(u1:0));

  // Truncate all but hidden bit.
  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:127,
                    fraction:  u23: 0x7fffff
                  };
  let _ = assert_eq(round_towards_zero(fraction),
    one<u32:8, u32:23>(u1:0));

  // Truncate some.
  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:128,
                    fraction:  u23: 0x7fffff
                  };
  let trunc_fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:128,
                    fraction:  u23: 0x400000
                  };
  let _ = assert_eq(round_towards_zero(fraction),
    trunc_fraction);

  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:149,
                    fraction:  u23: 0x7fffff
                  };
  let trunc_fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:149,
                    fraction:  u23: 0x7ffffe
                  };
  let _ = assert_eq(round_towards_zero(fraction),
    trunc_fraction);

  // Truncate none.
  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:200,
                    fraction:  u23: 0x7fffff
                  };
  let _ = assert_eq(round_towards_zero(fraction),
    fraction);

  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:200,
                    fraction:  u23: 0x7fffff
                  };
  let _ = assert_eq(round_towards_zero(fraction),
    fraction);

  ()
}

// Converts the given floating-point number into an unsigned integer, truncating
// any fractional bits if necessary.
// TODO(rspringer): 2021-08-06: We seem to be unable to call std::umax in
// template specification. Why?
pub fn to_int<EXP_SZ: u32, FRACTION_SZ: u32, RESULT_SZ:u32,
            WIDE_FRACTION: u32 = FRACTION_SZ + u32:1,
            MAX_FRACTION_SZ: u32 = if RESULT_SZ > WIDE_FRACTION { RESULT_SZ } else { WIDE_FRACTION }>(
    x: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[RESULT_SZ] {
  let exp = unbiased_exponent(x);

  let fraction =
      (x.fraction as uN[WIDE_FRACTION] | (uN[WIDE_FRACTION]:1 << FRACTION_SZ))
      as uN[MAX_FRACTION_SZ];

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
  let result =
      if exp as u32 < FRACTION_SZ { (fraction >> (FRACTION_SZ - (exp as u32))) }
      else { result };
  let result =
      if exp as u32 > FRACTION_SZ { (fraction << ((exp as u32) - FRACTION_SZ)) }
      else { result };

  // Clamp high if out of bounds, infinite, or NaN.
  let exp_oob = exp as s32 >= (RESULT_SZ as s32 - s32:1);
  let result =
      if exp_oob || is_inf(x) || is_nan(x) { (uN[MAX_FRACTION_SZ]:1 << (RESULT_SZ - u32:1)) }
      else { result };

  // Reduce to the target size, preserving signedness.
  let result = result as sN[MAX_FRACTION_SZ];
  let result = if !x.sign { result } else { -result };
  result as sN[RESULT_SZ]
}

#[test]
fn to_int_test() {
  let expected = s32:0;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x1, fraction: u23:0x0});
  let _ = assert_eq(expected, actual);

  let expected = s32:1;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x7f, fraction: u23:0xa5a5});
  let _ = assert_eq(expected, actual);

  let expected = s32:2;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x80, fraction: u23:0xa5a5});
  let _ = assert_eq(expected, actual);

  let expected = s32:0xa5a5;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8e, fraction: u23:0x25a500});
  let _ = assert_eq(expected, actual);

  let expected = s32:23;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x83, fraction: u23:0x380000});
  let _ = assert_eq(expected, actual);

  let expected = s16:0xa5a;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8a, fraction: u23:0x25a5a5});
  let _ = assert_eq(expected, actual);

  let expected = s16:0xa5;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x86, fraction: u23:0x25a5a5});
  let _ = assert_eq(expected, actual);

  let expected = s16:0x14b;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x87, fraction: u23:0x25a5a5});
  let _ = assert_eq(expected, actual);

  let expected = s16:0x296;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x88, fraction: u23:0x25a5a5});
  let _ = assert_eq(expected, actual);

  let expected = s16:0x8000;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8f, fraction: u23:0x25a5a5});
  let _ = assert_eq(expected, actual);

  let expected = s24:0x14b4b;
  let actual = to_int<u32:8, u32:23, u32:24>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8f, fraction: u23:0x25a5a5});
  let _ = assert_eq(expected, actual);

  let expected = s32:0x14b4b;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8f, fraction: u23:0x25a5a5});
  let _ = assert_eq(expected, actual);

  let expected = s16:0xa;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x82, fraction: u23:0x25a5a5});
  let _ = assert_eq(expected, actual);

  let expected = s16:0x5;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x81, fraction: u23:0x25a5a5});
  let _ = assert_eq(expected, actual);

  let expected = s32:0x80000000;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x9e, fraction: u23:0x0});
  let _ = assert_eq(expected, actual);

  let expected = s32:0x80000000;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0xff, fraction: u23:0x0});
  let _ = assert_eq(expected, actual);
  ()
}

// TODO(rspringer): Create a broadly-applicable normalize test, that
// could be used for multiple type instantiations (without needing
// per-specialization data to be specified by a user).
