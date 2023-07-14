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

pub fn tag<EXP_SZ:u32, FRACTION_SZ:u32>(
           input_float: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloatTag {
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
  assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:0, bexp: u4:0xf, fraction: u2:0x2,
  };
  let actual = qnan<u32:4, u32:2>();
  assert_eq(actual, expected);
  ()
}

// Returns a positive or negative zero depending upon the given sign parameter.
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
  assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:1, bexp: u4:0x0, fraction: u2:0x0,
  };
  let actual = zero<u32:4, u32:2>(u1:1);
  assert_eq(actual, expected);
  ()
}

// Returns one or minus one depending upon the given sign parameter.
pub fn one<EXP_SZ:u32, FRACTION_SZ:u32>(
           sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
  const MASK_SZ:u32 = EXP_SZ - u32:1;
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
  assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:0, bexp: u4:0x7, fraction: u2:0x0,
  };
  let actual = one<u32:4, u32:2>(u1:0);
  assert_eq(actual, expected);
  ()
}

// Returns a positive or a negative infinity depending upon the given sign parameter.
pub fn inf<EXP_SZ:u32, FRACTION_SZ:u32>(
           sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ> {
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
  assert_eq(actual, expected);

  let expected = APFloat<u32:4, u32:2> {
    sign: u1:0, bexp: u4:0xf, fraction: u2:0x0,
  };
  let actual = inf<u32:4, u32:2>(u1:0);
  assert_eq(actual, expected);
  ()
}

// Returns the unbiased exponent. For normal numbers it is
// `bexp - 2^EXP_SZ + 1`` and for subnormals it is, `2 - 2^EXP_SZ``. For
// infinity and `NaN``, there are no guarantees, as the unbiased exponent has
// no meaning in that case.
//
// For example, for single precision normal numbers the unbiased exponent is
// `bexp - 127`` and for subnormal numbers it is `-126`.
pub fn unbiased_exponent<EXP_SZ:u32,FRACTION_SZ:u32>(
                         f: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[EXP_SZ] {
  const UEXP_SZ:u32 = EXP_SZ + u32:1;
  const MASK_SZ:u32 = EXP_SZ - u32:1;
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
  ()
}

#[test]
fn unbiased_exponent_positive_test() {
  let expected = s8:1;
  let actual = unbiased_exponent<u32:8, u32:23>(
      APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:128, fraction: u23:0 });
  assert_eq(actual, expected);
  ()
}

#[test]
fn unbiased_exponent_negative_test() {
  let expected = s8:-1;
  let actual = unbiased_exponent<u32:8, u32:23>(
      APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:126, fraction: u23:0 });
  assert_eq(actual, expected);
  ()
}

#[test]
fn unbiased_exponent_subnormal_test() {
  let expected = s8:-126;
  let actual = unbiased_exponent<u32:8, u32:23>(
      APFloat<u32:8, u32:23> { sign: u1:0, bexp: u8:0, fraction: u23:0 });
  assert_eq(actual, expected);
  ()
}

// Returns the biased exponent which is equal to `unbiased_exponent + 2^EXP_SZ - 1`
//
// Since the function only takes as input the unbiased exponent, it cannot
// distinguish between normal and subnormal numbers, as a result it assumes that
// the input is the exponent for a normal number.
pub fn bias<EXP_SZ: u32,FRACTION_SZ: u32>(
            unbiased_exponent: sN[EXP_SZ]) -> bits[EXP_SZ] {
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
  ()
}

// Returns a bit string of size `1 + EXP_SZ + FRACTION_SZ` where the first bit
// is the sign bit, the next `EXP_SZ` bit encode the biased exponent and the
// last `FRACTION_SZ` are the significand without the hidden bit.
pub fn flatten<EXP_SZ:u32, FRACTION_SZ:u32, TOTAL_SZ:u32 = {u32:1+EXP_SZ+FRACTION_SZ}>(
    x: APFloat<EXP_SZ, FRACTION_SZ>) -> bits[TOTAL_SZ] {
  x.sign ++ x.bexp ++ x.fraction
}

// Returns a `APFloat` struct whose flattened version would be the input string.
pub fn unflatten<EXP_SZ:u32, FRACTION_SZ:u32,
                 TOTAL_SZ:u32 = {u32:1+EXP_SZ+FRACTION_SZ}>(
                 x: bits[TOTAL_SZ]) -> APFloat<EXP_SZ, FRACTION_SZ> {
  const SIGN_OFFSET:u32 = EXP_SZ+FRACTION_SZ;
  APFloat<EXP_SZ, FRACTION_SZ>{
      sign: (x >> (SIGN_OFFSET as bits[TOTAL_SZ])) as bits[1],
      bexp: (x >> (FRACTION_SZ as bits[TOTAL_SZ])) as bits[EXP_SZ],
      fraction: x as bits[FRACTION_SZ],
  }
}

// Casts the fixed point number to a floating point number using RNE
// (Round to Nearest Even) as the rounding mode.
pub fn cast_from_fixed<EXP_SZ:u32, FRACTION_SZ:u32, NUM_SRC_BITS:u32>(
                       to_cast: sN[NUM_SRC_BITS])
    -> APFloat<EXP_SZ, FRACTION_SZ> {
  const UEXP_SZ:u32 = EXP_SZ + u32:1;
  const EXTENDED_FRACTION_SZ:u32 = FRACTION_SZ + NUM_SRC_BITS;
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
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:0), zero_float);

  // +/-1
  let one_float = one<u32:4, u32:4>(u1:0);
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:1), one_float);
  let none_float = one<u32:4, u32:4>(u1:1);
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:-1), none_float);

  // +/-4
  let four_float =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:9,
      fraction: u4:0
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:4), four_float);
  let nfour_float =
    APFloat<u32:4, u32:4>{
      sign: u1:1,
      bexp: u4:9,
      fraction: u4:0
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:-4), nfour_float);

  // Cast maximum representable exponent in target format.
  let max_representable =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:0
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:128), max_representable);

  // Cast minimum non-representable exponent in target format.
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:256),
                    inf<u32:4, u32:4>(u1:0));

  // Test rounding - maximum truncated bits that will round down, even fraction.
  let truncate =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:0
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:131),
                    truncate);

  // Test rounding - maximum truncated bits that will round down, odd fraction.
  let truncate =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:1
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:139),
                    truncate);

  // Test rounding - halfway and already even, round down
  let truncate =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:0
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:132),
                    truncate);

  // Test rounding - halfway and odd, round up
  let round_up =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:2
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:140),
                    round_up);

  // Test rounding - over halfway and even, round up
  let round_up =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:1
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:133),
                    round_up);

  // Test rounding - over halfway and odd, round up
  let round_up =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:2
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:141),
                    round_up);

  // Test rounding - Rounding up increases exponent.
  let round_inc_exponent =
    APFloat<u32:4, u32:4>{
      sign: u1:0,
      bexp: u4:14,
      fraction: u4:0
    };
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:126),
                    round_inc_exponent);
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:127),
                    round_inc_exponent);

  // Test rounding - Rounding up overflows to infinity.
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:252),
                    inf<u32:4, u32:4>(u1:0));
  assert_eq(cast_from_fixed<u32:4, u32:4>(sN[32]:254),
                    inf<u32:4, u32:4>(u1:0));
  ()
}

pub fn subnormals_to_zero<EXP_SZ:u32, FRACTION_SZ:u32>(
                          x: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
  if x.bexp == bits[EXP_SZ]:0 { zero<EXP_SZ, FRACTION_SZ>(x.sign) } else { x }
}

// Returns a normalized APFloat with the given components.
// 'fraction_with_hidden' is the fraction (including the hidden bit). This
// function only normalizes in the direction of decreasing the exponent. Input
// must be a normal number or zero. Dernormals are flushed to zero in the
// result.
pub fn normalize<EXP_SZ:u32, FRACTION_SZ:u32,
                 WIDE_FRACTION:u32 = {FRACTION_SZ + u32:1}>(
                 sign: bits[1], exp: bits[EXP_SZ],
                 fraction_with_hidden: bits[WIDE_FRACTION])
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
pub fn is_inf<EXP_SZ:u32, FRACTION_SZ:u32>(
              x: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction == bits[FRACTION_SZ]:0)
}

// Returns whether or not the given APFloat represents NaN.
pub fn is_nan<EXP_SZ:u32, FRACTION_SZ:u32>(
              x: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction != bits[FRACTION_SZ]:0)
}

// Returns true if x == 0 or x is a subnormal number.
pub fn is_zero_or_subnormal<EXP_SZ: u32, FRACTION_SZ: u32>(
                            x: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  x.bexp == uN[EXP_SZ]:0
}

// Casts the floating point number to a fixed point number.
// Unrepresentable numbers are cast to the minimum representable
// number (largest magnitude negative number).
pub fn cast_to_fixed<NUM_DST_BITS:u32, EXP_SZ:u32, FRACTION_SZ:u32>(
                     to_cast: APFloat<EXP_SZ, FRACTION_SZ>)
    -> sN[NUM_DST_BITS] {
  const UEXP_SZ:u32 = EXP_SZ + u32:1;
  const EXTENDED_FIXED_SZ:u32 = NUM_DST_BITS + u32:1 + FRACTION_SZ + NUM_DST_BITS;

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
  assert_eq(
    cast_to_fixed<u32:32>(zero<u32:8, u32:23>(u1:0)), s32:0);
  assert_eq(
    cast_to_fixed<u32:32>(zero<u32:8, u32:23>(u1:1)), s32:0);

  // Cast +/-1.0
  assert_eq(
    cast_to_fixed<u32:32>(one<u32:8, u32:23>(u1:0)), s32:1);
  assert_eq(
    cast_to_fixed<u32:32>(one<u32:8, u32:23>(u1:1)), s32:-1);

  // Cast +/-1.5 --> +/- 1
  let one_point_five = APFloat<u32:8, u32:23>{sign: u1:0,
                                              bexp: u8:0x7f,
                                              fraction:  u1:1 ++ u22:0};
  assert_eq(
    cast_to_fixed<u32:32>(one_point_five), s32:1);
  let n_one_point_five = APFloat<u32:8, u32:23>{sign: u1:1,
                                                bexp: u8:0x7f,
                                                fraction:  u1:1 ++ u22:0};
  assert_eq(
    cast_to_fixed<u32:32>(n_one_point_five), s32:-1);

  // Cast +/-4.0
  let four = cast_from_fixed<u32:8, u32:23>(s32:4);
  let neg_four = cast_from_fixed<u32:8, u32:23>(s32:-4);
  assert_eq(
    cast_to_fixed<u32:32>(four), s32:4);
  assert_eq(
    cast_to_fixed<u32:32>(neg_four), s32:-4);

  // Cast 7
  let seven = cast_from_fixed<u32:8, u32:23>(s32:7);
  assert_eq(
    cast_to_fixed<u32:32>(seven), s32:7);

  // Cast big number (more digits left of decimal than hidden bit + fraction).
  let big_num = (u1:0 ++ std::mask_bits<u32:23>() ++ u8:0) as s32;
  let fp_big_num = cast_from_fixed<u32:8, u32:23>(big_num);
  assert_eq(
    cast_to_fixed<u32:32>(fp_big_num), big_num);

  // Cast large, non-overflowing numbers.
  let big_fit = APFloat<u32:8, u32:23>{sign: u1:0,
                                       bexp: u8:127 + u8:30,
                                       fraction: u23:0x7fffff};
  assert_eq(
    cast_to_fixed<u32:32>(big_fit),
    (u1:0 ++ u24:0xffffff ++ u7:0) as s32);
  let big_fit = APFloat<u32:8, u32:23>{sign: u1:1,
                                       bexp: u8:127 + u8:30,
                                       fraction: u23:0x7fffff};
  assert_eq(
    cast_to_fixed<u32:32>(big_fit),
    (s32:0 - (u1:0 ++ u24:0xffffff ++ u7:0) as s32));


  // Cast barely overflowing postive number.
  let big_overflow = APFloat<u32:8, u32:23>{sign: u1:0,
                                            bexp: u8:127 + u8:31,
                                            fraction: u23:0x0};
  assert_eq(
    cast_to_fixed<u32:32>(big_overflow),
    (u1:1 ++ u31:0) as s32);


  // This produces the largest negative int, but doesn't actually
  // overflow
  let max_negative = APFloat<u32:8, u32:23>{sign: u1:1,
                                            bexp: u8:127 + u8:31,
                                            fraction: u23:0x0};
  assert_eq(
    cast_to_fixed<u32:32>(max_negative),
    (u1:1 ++ u31:0) as s32);


  // Negative overflow.
  let negative_overflow = APFloat<u32:8, u32:23>{sign: u1:1,
                                            bexp: u8:127 + u8:31,
                                            fraction: u23:0x1};
  assert_eq(
    cast_to_fixed<u32:32>(negative_overflow),
    (u1:1 ++ u31:0) as s32);


  // NaN input.
  assert_eq(
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

  ()
}

// Returns an APFloat with all its bits past the decimal point set to 0.
pub fn round_towards_zero<EXP_SZ:u32, FRACTION_SZ:u32>(
                          x: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
  const EXTENDED_FRACTION_SZ:u32 = FRACTION_SZ + u32:1;
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
  assert_eq(round_towards_zero(zero<u32:8, u32:23>(u1:0)),
    zero<u32:8, u32:23>(u1:0));
  assert_eq(round_towards_zero(zero<u32:8, u32:23>(u1:1)),
    zero<u32:8, u32:23>(u1:1));
  assert_eq(round_towards_zero(qnan<u32:8, u32:23>()),
    qnan<u32:8, u32:23>());
  assert_eq(round_towards_zero(inf<u32:8, u32:23>(u1:0)),
    inf<u32:8, u32:23>(u1:0));
  assert_eq(round_towards_zero(inf<u32:8, u32:23>(u1:1)),
    inf<u32:8, u32:23>(u1:1));

  // Truncate all.
  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:50,
                    fraction:  u23: 0x7fffff
                  };
  assert_eq(round_towards_zero(fraction),
    zero<u32:8, u32:23>(u1:0));

  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:126,
                    fraction:  u23: 0x7fffff
                  };
  assert_eq(round_towards_zero(fraction),
    zero<u32:8, u32:23>(u1:0));

  // Truncate all but hidden bit.
  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:127,
                    fraction:  u23: 0x7fffff
                  };
  assert_eq(round_towards_zero(fraction),
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
  assert_eq(round_towards_zero(fraction),
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
  assert_eq(round_towards_zero(fraction),
    trunc_fraction);

  // Truncate none.
  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:200,
                    fraction:  u23: 0x7fffff
                  };
  assert_eq(round_towards_zero(fraction),
    fraction);

  let fraction = APFloat<u32:8, u32:23> {
                    sign: u1:0,
                    bexp: u8:200,
                    fraction:  u23: 0x7fffff
                  };
  assert_eq(round_towards_zero(fraction),
    fraction);

  ()
}

// Returns the signed integer part of the input float, truncating any
// fractional bits if necessary.
pub fn to_int<EXP_SZ: u32, FRACTION_SZ: u32, RESULT_SZ:u32>(
              x: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[RESULT_SZ] {
  const WIDE_FRACTION: u32 = FRACTION_SZ + u32:1;
  const MAX_FRACTION_SZ: u32 = std::umax(RESULT_SZ, WIDE_FRACTION);
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

// TODO(rspringer): Create a broadly-applicable normalize test, that
// could be used for multiple type instantiations (without needing
// per-specialization data to be specified by a user).

#[test]
fn to_int_test() {
  let expected = s32:0;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x1, fraction: u23:0x0});
  assert_eq(expected, actual);

  let expected = s32:1;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x7f, fraction: u23:0xa5a5});
  assert_eq(expected, actual);

  let expected = s32:2;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x80, fraction: u23:0xa5a5});
  assert_eq(expected, actual);

  let expected = s32:0xa5a5;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8e, fraction: u23:0x25a500});
  assert_eq(expected, actual);

  let expected = s32:23;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x83, fraction: u23:0x380000});
  assert_eq(expected, actual);

  let expected = s16:0xa5a;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8a, fraction: u23:0x25a5a5});
  assert_eq(expected, actual);

  let expected = s16:0xa5;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x86, fraction: u23:0x25a5a5});
  assert_eq(expected, actual);

  let expected = s16:0x14b;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x87, fraction: u23:0x25a5a5});
  assert_eq(expected, actual);

  let expected = s16:0x296;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x88, fraction: u23:0x25a5a5});
  assert_eq(expected, actual);

  let expected = s16:0x8000;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8f, fraction: u23:0x25a5a5});
  assert_eq(expected, actual);

  let expected = s24:0x14b4b;
  let actual = to_int<u32:8, u32:23, u32:24>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8f, fraction: u23:0x25a5a5});
  assert_eq(expected, actual);

  let expected = s32:0x14b4b;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x8f, fraction: u23:0x25a5a5});
  assert_eq(expected, actual);

  let expected = s16:0xa;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x82, fraction: u23:0x25a5a5});
  assert_eq(expected, actual);

  let expected = s16:0x5;
  let actual = to_int<u32:8, u32:23, u32:16>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x81, fraction: u23:0x25a5a5});
  assert_eq(expected, actual);

  let expected = s32:0x80000000;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0x9e, fraction: u23:0x0});
  assert_eq(expected, actual);

  let expected = s32:0x80000000;
  let actual = to_int<u32:8, u32:23, u32:32>(
      APFloat<u32:8, u32:23>{sign: u1:0, bexp: u8:0xff, fraction: u23:0x0});
  assert_eq(expected, actual);
  ()
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
pub fn add<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>,
                                          y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
  // WIDE_EXP: Widened exponent to capture a possible carry bit.
  const WIDE_EXP: u32 =EXP_SZ + u32:1;
  // CARRY_EXP: WIDE_EXP plus one sign bit.
  const CARRY_EXP: u32 =WIDE_EXP + u32:1;
  // WIDE_FRACTION: Widened fraction to contain full precision + rounding
  // (guard & sticky) bits.
  const WIDE_FRACTION: u32 =FRACTION_SZ + u32:5;
  // CARRY_FRACTION: WIDE_FRACTION plus one bit to capture a possible carry bit.
  const CARRY_FRACTION: u32 =WIDE_FRACTION + u32:1;
  // NORMALIZED_FRACTION: WIDE_FRACTION minus one bit for post normalization
  // (where the implicit leading 1 bit is dropped).
  const NORMALIZED_FRACTION: u32 =WIDE_FRACTION - u32:1;

  // Step 1: align the fractions.
  //  - Bit widths: Base fraction: u23.
  //  - Add the implied/leading 1 bit: u23 -> u24
  //  - Add a sign bit: u24 -> u25
  let fraction_high_bit = uN[FRACTION_SZ]:1 << (FRACTION_SZ as uN[FRACTION_SZ] - uN[FRACTION_SZ]:1);
  let wide_fraction_high_bit = fraction_high_bit as uN[WIDE_FRACTION] << uN[WIDE_FRACTION]:1;
  let wide_x = ((x.fraction as uN[WIDE_FRACTION]) | wide_fraction_high_bit) << uN[WIDE_FRACTION]:3;
  let wide_y = ((y.fraction as uN[WIDE_FRACTION]) | wide_fraction_high_bit) << uN[WIDE_FRACTION]:3;

  // Flush denormals to 0.
  let wide_x = if x.bexp == uN[EXP_SZ]:0 { uN[WIDE_FRACTION]:0 } else { wide_x };
  let wide_y = if y.bexp == uN[EXP_SZ]:0 { uN[WIDE_FRACTION]:0 } else { wide_y };

  // Shift the fractions to align with the largest exponent.
  let greater_exp = if x.bexp > y.bexp { x } else { y };
  let shift_x = greater_exp.bexp - x.bexp;
  let shift_y = greater_exp.bexp - y.bexp;
  let shifted_x = (wide_x >> (shift_x as uN[WIDE_FRACTION])) as sN[WIDE_FRACTION];
  let shifted_y = (wide_y >> (shift_y as uN[WIDE_FRACTION])) as sN[WIDE_FRACTION];

  // Calculate the sticky bits - set to 1 if any set bits were
  // shifted out of the fractions.
  let dropped_x = wide_x << ((WIDE_FRACTION as uN[EXP_SZ] - shift_x) as uN[WIDE_FRACTION]);
  let dropped_y = wide_y << ((WIDE_FRACTION as uN[EXP_SZ] - shift_y) as uN[WIDE_FRACTION]);
  let sticky_x = dropped_x > uN[WIDE_FRACTION]:0;
  let sticky_y = dropped_y > uN[WIDE_FRACTION]:0;
  let addend_x = shifted_x | (sticky_x as sN[WIDE_FRACTION]);
  let addend_y = shifted_y | (sticky_y as sN[WIDE_FRACTION]);

  // Invert the mantissa if its source has a different sign than
  // the larger value.
  let addend_x = if x.sign != greater_exp.sign { -addend_x } else { addend_x };
  let addend_y = if y.sign != greater_exp.sign { -addend_y } else { addend_y };

  // Step 2: Do some addition!
  // Add one bit to capture potential carry: s28 -> s29.
  let fraction = (addend_x as sN[CARRY_FRACTION]) + (addend_y as sN[CARRY_FRACTION]);
  let fraction_is_zero = fraction == sN[CARRY_FRACTION]:0;
  let result_sign = match (fraction_is_zero, fraction < sN[CARRY_FRACTION]:0) {
    (true, _) => u1:0,
    (false, true) => !greater_exp.sign,
    _ => greater_exp.sign,
  };

  // Get the absolute value of the result then chop off the sign bit: s29 -> u28.
  let abs_fraction = (if fraction < sN[CARRY_FRACTION]:0 { -fraction } else { fraction }) as uN[WIDE_FRACTION];

  // Step 3: Normalize the fraction (shift until the leading bit is a 1).
  // If the carry bit is set, shift right one bit (to capture the new bit of
  // precision) - but don't drop the sticky bit!
  let carry_bit = abs_fraction[-1:];
  let carry_fraction = (abs_fraction >> uN[WIDE_FRACTION]:1) as uN[NORMALIZED_FRACTION];
  let carry_fraction = carry_fraction | (abs_fraction[0:1] as uN[NORMALIZED_FRACTION]);

  // If we cancelled higher bits, then we'll need to shift left.
  // Leading zeroes will be 1 if there's no carry or cancellation.
  let leading_zeroes = clz(abs_fraction);
  let cancel = leading_zeroes > uN[WIDE_FRACTION]:1;
  let cancel_fraction = (abs_fraction << (leading_zeroes - uN[WIDE_FRACTION]:1)) as uN[NORMALIZED_FRACTION];
  let shifted_fraction = match(carry_bit, cancel) {
    (true, false) => carry_fraction,
    (false, true) => cancel_fraction,
    (false, false) => abs_fraction as uN[NORMALIZED_FRACTION],
    _ => fail!("carry_and_cancel", uN[NORMALIZED_FRACTION]:0)
  };

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
  let do_round_up =
      if (normal_chunk > u3:0x4) | (half_way_chunk == u2:0x3) { u1:1 }
      else { u1:0 };

  // We again need an extra bit for carry.
  let rounded_fraction = if do_round_up { (shifted_fraction as uN[WIDE_FRACTION]) + uN[WIDE_FRACTION]:0x8 }
      else { shifted_fraction as uN[WIDE_FRACTION] };
  let rounding_carry = rounded_fraction[-1:];

  // After rounding, we can chop off the extra precision bits.
  // As with normalization, if we carried, we need to shift right
  // an extra place.
  let fraction_shift = uN[WIDE_FRACTION]:3 +
      (if rounded_fraction[-1:] { uN[WIDE_FRACTION]:1 } else { uN[WIDE_FRACTION]:0 });
  let result_fraction = (rounded_fraction >> fraction_shift) as uN[FRACTION_SZ];

  // Finally, adjust the exponent based on addition and rounding -
  // each bit of carry or cancellation moves it by one place.
  let wide_exponent =
      (greater_exp.bexp as sN[CARRY_EXP]) +
      (rounding_carry as sN[CARRY_EXP]) +
      sN[CARRY_EXP]:1 - (leading_zeroes as sN[CARRY_EXP]);
  let wide_exponent = if fraction_is_zero { sN[CARRY_EXP]:0 } else { wide_exponent };

  // Chop off the sign bit.
  let wide_exponent =
      if wide_exponent < sN[CARRY_EXP]:0 { uN[WIDE_EXP]:0 }
      else { wide_exponent as uN[WIDE_EXP] };

  // Extra bonus step 5: special case handling!

  // If the exponent underflowed, don't bother with denormals. Just flush to 0.
  let result_fraction = if wide_exponent < uN[WIDE_EXP]:1 { uN[FRACTION_SZ]:0 } else { result_fraction };

  // Handle exponent overflow infinities.
  let saturated_exp = std::mask_bits<EXP_SZ>() as uN[WIDE_EXP];
  let max_exp = std::mask_bits<EXP_SZ>();
  let result_fraction =
      if wide_exponent < saturated_exp { result_fraction }
      else { uN[FRACTION_SZ]:0 };
  let result_exponent =
      if wide_exponent < saturated_exp { wide_exponent as uN[EXP_SZ] }
      else { max_exp };

  // Handle arg infinities.
  let is_operand_inf = is_inf<EXP_SZ, FRACTION_SZ>(x) |
      is_inf<EXP_SZ, FRACTION_SZ>(y);
  let result_exponent = if is_operand_inf { max_exp } else { result_exponent };
  let result_fraction = if is_operand_inf { uN[FRACTION_SZ]:0 } else { result_fraction };
  // Result infinity is negative iff all infinite operands are neg.
  let has_pos_inf = (is_inf<EXP_SZ, FRACTION_SZ>(x) & (x.sign == u1:0)) |
                    (is_inf<EXP_SZ, FRACTION_SZ>(y) & (y.sign == u1:0));
  let result_sign = if is_operand_inf { !has_pos_inf } else { result_sign };

  // Handle NaN; NaN trumps infinities, so we handle it last.
  // -inf + inf = NaN, i.e., if we have both positive and negative inf.
  let has_neg_inf =
      (is_inf<EXP_SZ, FRACTION_SZ>(x) & (x.sign == u1:1)) |
      (is_inf<EXP_SZ, FRACTION_SZ>(y) & (y.sign == u1:1));
  let is_result_nan = is_nan<EXP_SZ, FRACTION_SZ>(x) |
      is_nan<EXP_SZ, FRACTION_SZ>(y) | (has_pos_inf & has_neg_inf);
  let result_exponent = if is_result_nan { max_exp } else { result_exponent };
  let result_fraction = if is_result_nan { fraction_high_bit } else { result_fraction };
  let result_sign = if is_result_nan { u1:0 } else { result_sign };

  // Finally (finally!), construct the output float.
  APFloat<EXP_SZ, FRACTION_SZ> { sign: result_sign, bexp: result_exponent,
                            fraction: result_fraction as uN[FRACTION_SZ] }
}


// IEEE floating-point subtraction (and comparisons that are implemented using subtraction),
// with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact fraction values in the NaN case).
pub fn sub<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>,
                                          y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
  let y = APFloat<EXP_SZ, FRACTION_SZ>{sign: !y.sign, bexp: y.bexp, fraction: y.fraction};
  add(x, y)
}

// add is thoroughly tested elsewhere so a few simple tests is sufficient.
#[test]
fn test_sub() {
  let one = one<u32:8, u32:23>(u1:0);
  let two = add<u32:8, u32:23>(one, one);
  let neg_two = APFloat<u32:8, u32:23>{sign: u1:1, ..two};
  let three = add<u32:8, u32:23>(one, two);
  let four = add<u32:8, u32:23>(two, two);

  assert_eq(sub(four, one), three);
  assert_eq(sub(four, two), two);
  assert_eq(sub(four, three), one);
  assert_eq(sub(three, two), one);
  assert_eq(sub(two, four), neg_two);
  ()
}

// Returns the product of `x` and `y`, with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact fraction values in the NaN case).
pub fn mul<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>,
                                          y: APFloat<EXP_SZ, FRACTION_SZ>)
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
  let x_fraction = (x.fraction as uN[WIDE_FRACTION]) | (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));
  let y_fraction = (y.fraction as uN[WIDE_FRACTION]) | (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));

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
  let exp = if is_zero_or_subnormal(x) || is_zero_or_subnormal(y) { sN[SIGNED_EXP]:0 } else { exp };

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
  let is_half_way =
      fraction[FRACTION_ROUNDING_BIT as s32 : FRACTION_SZ as s32] &
      (fraction[0:FRACTION_ROUNDING_BIT as s32] == uN[FRACTION_ROUNDING_BIT]:0);
  let greater_than_half_way =
      fraction[FRACTION_ROUNDING_BIT as s32 : FRACTION_SZ as s32] &
      (fraction[0:FRACTION_ROUNDING_BIT as s32] != uN[FRACTION_ROUNDING_BIT]:0);
  let do_round_up =
      greater_than_half_way || (is_half_way & fraction[FRACTION_SZ as s32 : STICKY_FRACTION as s32]);

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
  let result_fraction = if result_exp < (high_exp as uN[WIDE_EXP]) { result_fraction } else { uN[FRACTION_SZ]:0 };
  let result_exp = if result_exp < (high_exp as uN[WIDE_EXP]) { result_exp as uN[EXP_SZ] } else { high_exp };

  // - Arg infinites. Any arg is infinite == result is infinite.
  let is_operand_inf = is_inf<EXP_SZ, FRACTION_SZ>(x) ||
      is_inf<EXP_SZ, FRACTION_SZ>(y);
  let result_exp = if is_operand_inf { high_exp } else { result_exp };
  let result_fraction = if is_operand_inf { uN[FRACTION_SZ]:0 } else { result_fraction };

  // - NaNs. NaN trumps infinities, so we handle it last.
  //   inf * 0 = NaN, i.e.,
  let has_0_arg = is_zero_or_subnormal(x) || is_zero_or_subnormal(y);
  let has_nan_arg = is_nan<EXP_SZ, FRACTION_SZ>(x) ||
      is_nan<EXP_SZ, FRACTION_SZ>(y);
  let has_inf_arg = is_inf<EXP_SZ, FRACTION_SZ>(x) ||
      is_inf<EXP_SZ, FRACTION_SZ>(y);
  let is_result_nan = has_nan_arg || (has_0_arg && has_inf_arg);
  let result_exp = if is_result_nan { high_exp } else { result_exp };
  let nan_fraction = uN[FRACTION_SZ]:1 << (FRACTION_SZ as uN[FRACTION_SZ] - uN[FRACTION_SZ]:1);
  let result_fraction = if is_result_nan { nan_fraction } else { result_fraction };
  let result_sign = if is_result_nan { u1:0 } else { result_sign };

  APFloat<EXP_SZ, FRACTION_SZ>{
      sign: result_sign, bexp: result_exp, fraction: result_fraction }
}

// Simple utility struct for holding the result of the multiplication step.
struct Product<EXP_CARRY: u32, WIDE_FRACTION: u32> {
  sign: u1,
  bexp: uN[EXP_CARRY],
  fraction: uN[WIDE_FRACTION],
}

// Returns true if the given Product is infinite.
fn is_product_inf<EXP_CARRY: u32, WIDE_FRACTION: u32>(p: Product<EXP_CARRY, WIDE_FRACTION>) -> u1 {
  p.bexp == std::mask_bits<EXP_CARRY>() && p.fraction == uN[WIDE_FRACTION]:0
}

// Returns true if the given Product is NaN.
fn is_product_nan<EXP_CARRY: u32, WIDE_FRACTION: u32>(p: Product<EXP_CARRY, WIDE_FRACTION>) -> u1 {
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
fn mul_no_round<EXP_SZ: u32, FRACTION_SZ: u32,
                WIDE_FRACTION: u32 = {(FRACTION_SZ + u32:1) * u32:2},
                EXP_CARRY: u32 = {EXP_SZ + u32:1},
                EXP_SIGN_CARRY: u32 = {EXP_SZ + u32:2}>(
                a: APFloat<EXP_SZ, FRACTION_SZ>,
                b: APFloat<EXP_SZ, FRACTION_SZ>) -> Product {
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
  let bexp = (a.bexp as sN[EXP_SIGN_CARRY]) + (b.bexp as sN[EXP_SIGN_CARRY]) -
      bias + (fraction_shift as sN[EXP_SIGN_CARRY]);
  let bexp =
      if a.bexp == bits[EXP_SZ]:0 || b.bexp == bits[EXP_SZ]:0 {
        sN[EXP_SIGN_CARRY]:0
      } else {
        bexp
      };

  // Note that we usually flush subnormals. Here, we preserve what we can for
  // compatability with reference implementations.
  // We only do this for the internal product - we otherwise don't handle
  // subnormal values (we flush them to 0).
  let is_subnormal = bexp <= sN[EXP_SIGN_CARRY]:0;
  let result_exp =
      if is_subnormal { uN[EXP_CARRY]:0 } else { bexp as uN[EXP_CARRY] };
  let sub_exp = std::abs(bexp) as uN[EXP_CARRY];
  let result_fraction = if is_subnormal { fraction >> sub_exp } else { fraction };

  // - Overflow infinites - saturate exp, clear fraction.
  let high_exp = std::mask_bits<EXP_CARRY>();
  let result_fraction = if result_exp < high_exp { result_fraction } else { uN[WIDE_FRACTION]:0 };
  let result_exp = if result_exp < high_exp { result_exp as uN[EXP_CARRY] } else { high_exp };

  // - Arg infinites. Any arg is infinite == result is infinite.
  let is_operand_inf = is_inf(a) || is_inf(b);
  let result_exp = if is_operand_inf { high_exp } else { result_exp };
  let result_fraction = if is_operand_inf {
    uN[WIDE_FRACTION]:0
  } else {
    result_fraction as uN[WIDE_FRACTION]
  };

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
pub fn fma<EXP_SZ: u32, FRACTION_SZ: u32>(a: APFloat<EXP_SZ, FRACTION_SZ>,
                                          b: APFloat<EXP_SZ, FRACTION_SZ>,
                                          c: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
  // EXP_CARRY: One greater than EXP_SZ, to hold a carry bit.
  const EXP_CARRY: u32 =EXP_SZ + u32:1;
  // EXP_SIGN_CARRY: One greater than EXP_CARRY, to hold a sign bit.
  const EXP_SIGN_CARRY: u32 =EXP_CARRY + u32:1;
  // WIDE_FRACTION: Fully-widened fraction to hold all rounding bits.
  const WIDE_FRACTION: u32 =(FRACTION_SZ + u32:1) * u32:3 + u32:1;
  // WIDE_FRACTION_CARRY: WIDE_FRACTION plus one carry bit.
  const WIDE_FRACTION_CARRY: u32 =WIDE_FRACTION + u32:1;
  // WIDE_FRACTION_SIGN_CARRY: WIDE_FRACTION_CARRY plus one sign bit.
  const WIDE_FRACTION_SIGN_CARRY: u32 =WIDE_FRACTION_CARRY + u32:1;
  // WIDE_FRACTION_LOW_BIT: Position of the LSB in the final fraction within a
  // WIDE_FRACTION element. All bits with lower index are for rounding.
  const WIDE_FRACTION_LOW_BIT: u32 =WIDE_FRACTION - FRACTION_SZ;
  // WIDE_FRACTION_TOP_ROUNDING: One less than WIDE_FRACTION_LOW_BIT, in other words the
  // most-significant rounding bit.
  const WIDE_FRACTION_TOP_ROUNDING: u32 =WIDE_FRACTION_LOW_BIT - u32:1;

  let ab = mul_no_round<EXP_SZ, FRACTION_SZ>(a, b);

  let greater_exp = if ab.bexp > c.bexp as uN[EXP_CARRY] { ab.bexp }
                    else { c.bexp as uN[EXP_CARRY] };
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
  let dropped_ab =
      wide_ab << ((WIDE_FRACTION as uN[EXP_CARRY] - rshift_ab) as uN[WIDE_FRACTION]);
  let dropped_c =
      wide_c << ((WIDE_FRACTION as uN[EXP_CARRY] - rshift_c) as uN[WIDE_FRACTION]);
  let dropped_c =
      if rshift_c >= (WIDE_FRACTION as uN[EXP_CARRY]) { wide_c } else { dropped_c };
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
  let abs_fraction =
      (if sum_fraction < sN[WIDE_FRACTION_SIGN_CARRY]:0 { -sum_fraction }
       else { sum_fraction })
      as uN[WIDE_FRACTION_CARRY];

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
  let shifted_fraction = match(carry_bit, cancel) {
    (true, false) => carry_fraction,
    (false, true) => cancel_fraction,
    (false, false) => abs_fraction as uN[WIDE_FRACTION],
    _ => fail!("carry_and_cancel", uN[WIDE_FRACTION]:0)
  };

  // Similar to the rounding in apfloat_add_2, except that the fraction
  // starts at the bit below instead of bit 3.
  // For binary32, normal_chunk will be bits 0-48 (inclusive), stopping
  // immediately below the first bit in the final fraction.
  let normal_chunk = shifted_fraction[0:(WIDE_FRACTION_LOW_BIT - u32:1) as s32];
  let half_way_chunk =
      shifted_fraction[(WIDE_FRACTION_LOW_BIT - u32:2) as s32:(WIDE_FRACTION_LOW_BIT as s32)];
  let half_of_extra =
      uN[WIDE_FRACTION_TOP_ROUNDING]:1 << (WIDE_FRACTION_LOW_BIT - u32:2);
  let do_round_up =
      if (normal_chunk > half_of_extra) | (half_way_chunk == u2:0x3) { u1:1 }
      else { u1:0 };
  let rounded_fraction =
      if do_round_up {
        shifted_fraction as uN[WIDE_FRACTION_CARRY] +
            (uN[WIDE_FRACTION_CARRY]:1 << (WIDE_FRACTION_LOW_BIT - u32:1))
      } else {
        shifted_fraction as uN[WIDE_FRACTION_CARRY]
      };

  let rounding_carry = rounded_fraction[-1:];
  let result_fraction =
      (rounded_fraction >> ((WIDE_FRACTION_LOW_BIT - u32:1) as uN[WIDE_FRACTION_CARRY]))
      as uN[FRACTION_SZ];

  let bexp =
      greater_exp as sN[EXP_SIGN_CARRY] +
      rounding_carry as sN[EXP_SIGN_CARRY] + sN[EXP_SIGN_CARRY]:1 -
      leading_zeroes as sN[EXP_SIGN_CARRY];
  let bexp = if fraction_is_zero { sN[EXP_SIGN_CARRY]:0 } else { bexp };
  let bexp =
      if bexp < sN[EXP_SIGN_CARRY]:0 { uN[EXP_CARRY]:0 }
      else { (bexp as uN[EXP_CARRY]) };

  // Standard special case handling follows.

  // If the exponent underflowed, don't bother with denormals. Just flush to 0.
  let result_fraction = if bexp == uN[EXP_CARRY]:0 { uN[FRACTION_SZ]:0 }
                        else { result_fraction };

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
  let has_pos_inf = (is_product_inf(ab) & (ab.sign == u1:0)) |
                    (is_inf(c) & (c.sign == u1:0));
  let result_sign = if is_operand_inf { !has_pos_inf } else { result_sign };

  // Handle NaN; NaN trumps infinities, so we handle it last.
  // -inf + inf = NaN, i.e., if we have both positive and negative inf.
  let has_neg_inf =
      (is_product_inf(ab) & (ab.sign == u1:1)) |
      (is_inf(c) & (c.sign == u1:1));
  let is_result_nan = is_product_nan(ab) |
      is_nan(c) | (has_pos_inf & has_neg_inf);
  let result_exp = if is_result_nan { max_exp } else { result_exp };
  let result_fraction =
      if is_result_nan {
        uN[FRACTION_SZ]:1 << (FRACTION_SZ - u32:4)
      } else {
        result_fraction
      };
  let result_sign = if is_result_nan { u1:0 } else { result_sign };
  let is_result_inf = has_pos_inf | has_neg_inf;

  APFloat<EXP_SZ, FRACTION_SZ>{ sign: result_sign, bexp: result_exp as uN[EXP_SZ],
                           fraction: result_fraction as uN[FRACTION_SZ] }
}

#[test]
fn smoke() {
    type F32 = APFloat<8,23>;
    let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23: 0 };
    let one_point_one = F32 { sign: u1:0, bexp: u8:127, fraction: u23: 0xccccd };
    let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23: 0x58cccd };
    let a = twenty_seven_point_one;
    let b = one_point_one;
    let c = zero;
    let expected = F32 { sign: u1:0, bexp: u8:0x83, fraction: u23:0x6e7ae2 };
    let actual = fma<u32:8, u32:23>(a, b, c);
    assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_f32() {
  type F32 = APFloat<8,23>;
  let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23: 0 };
  let one_point_zero = F32 { sign: u1:0, bexp: u8:127, fraction: u23: 0 };
  let a = one_point_zero;
  let b = one_point_zero;
  let c = one_point_zero;
  let expected = F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn one_x_one_plus_one_f64() {
  type F64 = APFloat<11,52>;
  let zero = F64 { sign: u1:0, bexp: u11:0, fraction: u52: 0 };
  let one_point_zero = F64 { sign: u1:0, bexp: u11:1023, fraction: u52: 0 };
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
  let zero = BF16 { sign: u1:0, bexp: u8:0, fraction: u7: 0 };
  let one_point_zero = BF16 { sign: u1:0, bexp: u8:127, fraction: u7: 0 };
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
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0, bexp: u8:0x97, fraction: u23:0x565d43 };
  let b = F32 { sign: u1:1, bexp: u8:0x77, fraction: u23:0x319a49 };
  let c = F32 { sign: u1:0, bexp: u8:0x87, fraction: u23:0x642891 };
  let expected = F32 { sign: u1:1, bexp: u8:0x90, fraction: u23:0x144598 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn manual_case_a_f64() {
  type F64 = APFloat<11,52>;
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
  type F32 = APFloat<8,23>;
  let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23: 0 };
  let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23: 0x58cccd };
  let a = twenty_seven_point_one;
  let b = twenty_seven_point_one;
  let c = zero;
  let expected = F32 { sign: u1:0, bexp: u8:0x88, fraction: u23:0x379a3e };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one() {
  type F32 = APFloat<8,23>;
  let zero = F32 { sign: u1:0, bexp: u8:0, fraction: u23: 0 };
  let one_point_zero = F32 { sign: u1:0, bexp: u8:127, fraction: u23: 0 };
  let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23: 0x58cccd };
  let a = twenty_seven_point_one;
  let b = twenty_seven_point_one;
  let c = one_point_zero;
  let expected = F32 { sign: u1:0, bexp: u8:0x88, fraction: u23:0x37da3e };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn twenty_seven_point_one_x_twenty_seven_point_one_plus_one_point_one() {
  type F32 = APFloat<8,23>;
  let one_point_one = F32 { sign: u1:0, bexp: u8:127, fraction: u23: 0xccccd };
  let twenty_seven_point_one = F32 { sign: u1:0, bexp: u8:131, fraction: u23: 0x58cccd };
  let a = twenty_seven_point_one;
  let b = twenty_seven_point_one;
  let c = one_point_one;
  let expected = F32 { sign: u1:0, bexp: u8:0x88, fraction: u23:0x37e0a4 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_a() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x50, fraction: u23:0x1a8ddc };
  let b = F32 { sign: u1:0x1, bexp: u8:0xcb, fraction: u23:0xee7ac };
  let c = F32 { sign: u1:0x1, bexp: u8:0xb7, fraction: u23:0x609f18 };
  let expected = F32 { sign: u1:1, bexp: u8:0xb7, fraction: u23:0x609f18 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_b() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0x23, fraction: u23:0x4d3a41 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x30, fraction: u23:0x35a901 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x96, fraction: u23:0x627c62 };
  let expected = F32 { sign: u1:0, bexp: u8:0x96, fraction: u23:0x627c62 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_c() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x71, fraction: u23:0x2f0932 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xe5, fraction: u23:0x416b76 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xcb, fraction: u23:0x5fd32a };
  let expected = F32 { sign: u1:1, bexp: u8:0xd8, fraction: u23:0x4386a };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_d() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0xac, fraction: u23:0x1d0d22 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xdb, fraction: u23:0x2fe688 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xa9, fraction: u23:0x2be1d2 };
  let expected = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x0 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_e() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0x7b, fraction: u23:0x25e79f };
  let b = F32 { sign: u1:0x1, bexp: u8:0xff, fraction: u23:0x207370 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x39, fraction: u23:0x6bb348 };
  let expected = F32 { sign: u1:0, bexp: u8:0xff, fraction: u23:0x80000 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_f() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0xe0, fraction: u23:0x3cdaa8 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x96, fraction: u23:0x52549c };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1c, fraction: u23:0x21e0fd };
  let expected = F32 { sign: u1:0, bexp: u8:0xf8, fraction: u23:0x1b29c9 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_g() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0xc4, fraction: u23:0x73b59a };
  let b = F32 { sign: u1:0x0, bexp: u8:0xa6, fraction: u23:0x1631c0 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x29, fraction: u23:0x5b3d33 };
  let expected = F32 { sign: u1:1, bexp: u8:0xec, fraction: u23:0xefbc5 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_h() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x9b, fraction: u23:0x3f50d4 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x7b, fraction: u23:0x4beeb5 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x37, fraction: u23:0x6ad17c };
  let expected = F32 { sign: u1:1, bexp: u8:0x98, fraction: u23:0x18677d };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_i() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0x66, fraction: u23:0x36e592 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xc8, fraction: u23:0x2b5bf1 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x52, fraction: u23:0x12900b };
  let expected = F32 { sign: u1:0, bexp: u8:0xaf, fraction: u23:0x74da11 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_j() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x88, fraction: u23:0x0f0e03 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xb9, fraction: u23:0x36006d };
  let c = F32 { sign: u1:0x1, bexp: u8:0xaa, fraction: u23:0x358b6b };
  let expected = F32 { sign: u1:0, bexp: u8:0xc2, fraction: u23:0x4b6865 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_k() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x29, fraction: u23:0x2fd76d };
  let b = F32 { sign: u1:0x1, bexp: u8:0xce, fraction: u23:0x63eded };
  let c = F32 { sign: u1:0x0, bexp: u8:0xfd, fraction: u23:0x21adee };
  let expected = F32 { sign: u1:0, bexp: u8:0xfd, fraction: u23:0x21adee };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_l() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0x6a, fraction: u23:0x09c1b9 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x7c, fraction: u23:0x666a52 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x80, fraction: u23:0x626bcf };
  let expected = F32 { sign: u1:1, bexp: u8:0x80, fraction: u23:0x626bcf };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_m() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x70, fraction: u23:0x41e2db };
  let b = F32 { sign: u1:0x1, bexp: u8:0xd1, fraction: u23:0x013c17 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xb9, fraction: u23:0x30313f };
  let expected = F32 { sign: u1:0, bexp: u8:0xc2, fraction: u23:0x4419bf };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_n() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x33, fraction: u23:0x537374 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x40, fraction: u23:0x78fa62 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x09, fraction: u23:0x7cfb29 };
  let expected = F32 { sign: u1:1, bexp: u8:0x09, fraction: u23:0x7cfb36 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_o() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0x94, fraction: u23:0x1aeb90 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x88, fraction: u23:0x1ab376 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x9d, fraction: u23:0x15dd1e };
  let expected = F32 { sign: u1:1, bexp: u8:0x9e, fraction: u23:0x288cde };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_p() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0x88, fraction: u23:0x1ebb00 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xf6, fraction: u23:0x0950b6 };
  let c = F32 { sign: u1:0x0, bexp: u8:0xfd, fraction: u23:0x6c314b };
  let expected = F32 { sign: u1:1, bexp: u8:0xfe, fraction: u23:0x5e77d4 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_q() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0xda, fraction: u23:0x5b328f };
  let b = F32 { sign: u1:0x1, bexp: u8:0x74, fraction: u23:0x157da3 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1b, fraction: u23:0x6a3f25 };
  let expected = F32 { sign: u1:1, bexp: u8:0xd0, fraction: u23:0x000000 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_r() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x34, fraction: u23:0x4da000 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xf4, fraction: u23:0x4bc400 };
  let c = F32 { sign: u1:0x1, bexp: u8:0x33, fraction: u23:0x54476d };
  let expected = F32 { sign: u1:1, bexp: u8:0xaa, fraction: u23:0x23ab4f };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_s() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x27, fraction: u23:0x732d83 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xbb, fraction: u23:0x4b2dcd };
  let c = F32 { sign: u1:0x0, bexp: u8:0x3a, fraction: u23:0x65e4bd };
  let expected = F32 { sign: u1:0, bexp: u8:0x64, fraction: u23:0x410099 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_t() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0x17, fraction: u23:0x070770 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x86, fraction: u23:0x623b39 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1e, fraction: u23:0x6ea761 };
  let expected = F32 { sign: u1:1, bexp: u8:0x0c, fraction: u23:0x693bc0 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_u() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0xb1, fraction: u23:0x0c8800 };
  let b = F32 { sign: u1:0x1, bexp: u8:0xc6, fraction: u23:0x2b3800 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x22, fraction: u23:0x00c677 };
  let expected = F32 { sign: u1:1, bexp: u8:0xf8, fraction: u23:0x3bfb2b };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_v() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0x90, fraction: u23:0x04a800 };
  let b = F32 { sign: u1:0x1, bexp: u8:0x1f, fraction: u23:0x099cb0 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x28, fraction: u23:0x4d6497 };
  let expected = F32 { sign: u1:1, bexp: u8:0x30, fraction: u23:0x0dd0cf };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_w() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0x90, fraction: u23:0x0fdde1 };
  let b = F32 { sign: u1:0x0, bexp: u8:0xa8, fraction: u23:0x663085 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x9b, fraction: u23:0x450d69 };
  let expected = F32 { sign: u1:0, bexp: u8:0xba, fraction: u23:0x015c9c };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_x() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x4c, fraction: u23:0x5ca821 };
  let b = F32 { sign: u1:0x0, bexp: u8:0x87, fraction: u23:0x14808c };
  let c = F32 { sign: u1:0x0, bexp: u8:0x1c, fraction: u23:0x585ccf };
  let expected = F32 { sign: u1:1, bexp: u8:0x55, fraction: u23:0x000000 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_y() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x0, bexp: u8:0xc5, fraction: u23:0x3a123b };
  let b = F32 { sign: u1:0x1, bexp: u8:0x3b, fraction: u23:0x7ee4d9 };
  let c = F32 { sign: u1:0x0, bexp: u8:0x4f, fraction: u23:0x1d4ddc };
  let expected = F32 { sign: u1:1, bexp: u8:0x82, fraction: u23:0x39446d };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_z() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0xd4, fraction: u23:0x1b858b };
  let b = F32 { sign: u1:0x1, bexp: u8:0x9e, fraction: u23:0x59fa23 };
  let c = F32 { sign: u1:0x1, bexp: u8:0xb5, fraction: u23:0x3520e4 };
  let expected = F32 { sign: u1:0x0, bexp: u8:0xf4, fraction: u23:0x046c29 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}

#[test]
fn fail_case_aa() {
  type F32 = APFloat<8,23>;
  let a = F32 { sign: u1:0x1, bexp: u8:0x9b, fraction: u23:0x3ac78d };
  let b = F32 { sign: u1:0x0, bexp: u8:0x3b, fraction: u23:0x542cbb };
  let c = F32 { sign: u1:0x1, bexp: u8:0x09, fraction: u23:0x0c609e };
  let expected = F32 { sign: u1:0x1, bexp: u8:0x58, fraction: u23:0x1acde3 };
  let actual = fma<u32:8, u32:23>(a, b, c);
  assert_eq(expected, actual)
}
