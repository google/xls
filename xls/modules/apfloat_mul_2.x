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

// This file implements [most of] IEEE 754 single-precision
// floating point multiplication, with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact fraction values in the NaN case).

import apfloat
import std

type APFloat = apfloat::APFloat;

// Determines if the given value is 0, taking into account
// flushing subnormals.
fn is_zero<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  x.bexp == uN[EXP_SZ]:0
}

// Usage:
//  - EXP_SZ: The number of bits in the exponent.
//  - FRACTION_SZ: The number of bits in the fractional part of the FP number.
//  - x, y: The two floating-point numbers to multiply.
//
// Derived parametrics:
//  - WIDE_EXP: Widened exponent to capture a possible carry bit.
//  - SIGNED_EXP: WIDE_EXP plus one sign bit.
//  - ROUNDING_FRACTION: Result fraction with one extra bit to capture
//    potential carry if rounding up.
//  - WIDE_FRACTION: Widened fraction to contain full precision + rounding
//    (guard & sticky) bits.
//  - FRACTION_ROUNDING_BIT: Position of the first rounding bit in the "wide" FRACTION.
//  - STICKY_FRACTION: Location of the sticky bit in the wide FRACTION (same as
//    "ROUNDING_FRACTION", but it's easier to understand the code if it
//    has its own name).
pub fn apfloat_mul_2<
    EXP_SZ: u32,
    FRACTION_SZ: u32,
    WIDE_EXP: u32 = EXP_SZ + u32:1,
    SIGNED_EXP: u32 = WIDE_EXP + u32:1,
    ROUNDING_FRACTION: u32 = FRACTION_SZ + u32:1,
    WIDE_FRACTION: u32 = FRACTION_SZ + FRACTION_SZ + u32:2,
    FRACTION_ROUNDING_BIT: u32 = FRACTION_SZ - u32:1,
    STICKY_FRACTION: u32 = FRACTION_SZ + u32:1>(
    x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) ->
    APFloat<EXP_SZ, FRACTION_SZ> {
  // 1. Get and expand mantissas.
  let x_fraction = (x.fraction as uN[WIDE_FRACTION]) | (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));
  let y_fraction = (y.fraction as uN[WIDE_FRACTION]) | (uN[WIDE_FRACTION]:1 << (FRACTION_SZ as uN[WIDE_FRACTION]));

  // 1a. Flush subnorms to 0.
  let x_fraction = if is_zero(x) { uN[WIDE_FRACTION]:0 } else { x_fraction };
  let y_fraction = if is_zero(y) { uN[WIDE_FRACTION]:0 } else { y_fraction };

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
  let exp = if is_zero(x) || is_zero(y) { sN[SIGNED_EXP]:0 } else { exp };

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
  let is_operand_inf = apfloat::is_inf<EXP_SZ, FRACTION_SZ>(x) ||
      apfloat::is_inf<EXP_SZ, FRACTION_SZ>(y);
  let result_exp = if is_operand_inf { high_exp } else { result_exp };
  let result_fraction = if is_operand_inf { uN[FRACTION_SZ]:0 } else { result_fraction };

  // - NaNs. NaN trumps infinities, so we handle it last.
  //   inf * 0 = NaN, i.e.,
  let has_0_arg = is_zero(x) || is_zero(y);
  let has_nan_arg = apfloat::is_nan<EXP_SZ, FRACTION_SZ>(x) ||
      apfloat::is_nan<EXP_SZ, FRACTION_SZ>(y);
  let has_inf_arg = apfloat::is_inf<EXP_SZ, FRACTION_SZ>(x) ||
      apfloat::is_inf<EXP_SZ, FRACTION_SZ>(y);
  let is_result_nan = has_nan_arg || (has_0_arg && has_inf_arg);
  let result_exp = if is_result_nan { high_exp } else { result_exp };
  let nan_fraction = uN[FRACTION_SZ]:1 << (FRACTION_SZ as uN[FRACTION_SZ] - uN[FRACTION_SZ]:1);
  let result_fraction = if is_result_nan { nan_fraction } else { result_fraction };
  let result_sign = if is_result_nan { u1:0 } else { result_sign };

  APFloat<EXP_SZ, FRACTION_SZ>{
      sign: result_sign, bexp: result_exp, fraction: result_fraction }
}
