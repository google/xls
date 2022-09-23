// =================================================================
// Copyright 2011, 2012, 2013, 2014 The Regents of the University of California.
// Copyright 2021 The XLS Authors
// All rights reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//  1. Redistributions of source code must retain the above copyright notice,
//     this list of conditions, and the following disclaimer.
//  2. Redistributions in binary form must reproduce the above copyright notice,
//     this list of conditions, and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//  3. Neither the name of the University nor the names of its contributors may
//     be used to endorse or promote products derived from this software without
//     specific prior written permission.
// THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
// DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// =================================================================

// This file implements [most of] IEEE 754 single-precision floating point
// division, with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//      (denormals-are-zero / DAZ).
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other conforming
// implementations, modulo exact fraction values in the NaN case (we emit a
// single, canonical representation for NaN (qnan) but accept all NaN
// respresentations as input).
import float32
import std

type F32 = float32::F32;

// Determines if the given value is 0, taking into account
// flushing subnormals.
fn is_zero(x: F32) -> u1 {
  x.bexp == u8:0
}

pub fn fpdiv_2x32(x: F32, y: F32) -> F32 {
  // 1. Get and expand mantissas.
  let x_fraction = (x.fraction as u64) | u64:0x80_0000;
  let y_fraction = (y.fraction as u64) | u64:0x80_0000;

  // 1a. Flush subnorms to 0.
  let x_fraction = if x.bexp == u8:0 { u64:0 } else { x_fraction };
  let y_fraction = if y.bexp == u8:0 { u64:0 } else { y_fraction };

  // 2. Subtract non-biased exponents.
  //  - Remove the bias from the exponents, subtract them, then restore the bias.
  //  - Simplifies from
  //      (A - 127) - (B - 127) + 127 = exp
  //    to
  //      A + B + 127 = exp
  let exp = (x.bexp as s10) - (y.bexp as s10) + s10:0x7f;

  // 3. Shift numerator and adjust exponent.
  let exp = if x_fraction < y_fraction { exp - s10:1 } else { exp };
  let x_fraction = if x_fraction < y_fraction { x_fraction << u64:31 } else { x_fraction << u64:30 };

  // 4. Divide integer mantissas.
  let fraction = std::iterative_div(x_fraction, y_fraction) as u32;

  // 5. Account for remainder / error.
  let fraction_has_bit_in_six_lsbs = fraction[0:6] != u6:0;
  let remainder_detected = (y_fraction * fraction as u64) != x_fraction;
  let set_fraction_lsb = !fraction_has_bit_in_six_lsbs && remainder_detected;
  let fraction = if set_fraction_lsb { fraction | u32:1 } else { fraction };

  // 6. Check rounding conditions.
  // We use nearest, half to even rounding.
  // - We round down if less than 1/2 way between values, i.e.
  // - We round up if we're more than 1/2 way
  // - If halfway, then we round whichever direction makes the
  //   result even.
  let round_bits = fraction[0:7];
  let is_half_way = round_bits[-1:] & (round_bits[:-1] == u6:0);
  let greater_than_half_way = round_bits[-1:] & (round_bits[:-1] != u6:0);

  // We're done with the extra precision bits now, so shift the
  // fraction into its almost-final width, adding one extra
  // bit for potential rounding overflow.
  let fraction = (fraction >> u32:7) as u23;
  let fraction = fraction as u24;
  let do_round_up = greater_than_half_way || (is_half_way & fraction[0:1]);
  let fraction = if do_round_up { fraction + u24:1 } else { fraction };

  // Adjust the exponent if we overflowed during rounding.
  // After checking for subnormals, we don't need the sign bit anymore.
  let exp = if fraction[-1:] { exp + s10:1 } else { exp };
  let is_subnormal = exp <= s10:0;

  // We're done - except for special cases...
  let result_sign = x.sign ^ y.sign;
  let result_exp = exp as u9;
  let result_fraction = fraction as u23;

  // 6. Special cases!
  // - Subnormals: flush to 0.
  let result_exp = if is_subnormal { u9:0 } else { result_exp };
  let result_fraction = if is_subnormal { u23:0 } else { result_fraction };

  // - Overflow infinites. Exp to 255, clear fraction.
  let result_fraction = if result_exp < u9:0xff { result_fraction } else { u23:0} ;
  let result_exp = if result_exp < u9:0xff { result_exp as u8 } else { u8:0xff };

  // - If the denominator is 0 or the numerator is infinity,
  // the result is infinity.
  let divide_by_zero = is_zero(y);
  let divide_inf = float32::is_inf(x);
  let is_result_inf = divide_by_zero || divide_inf;
  let result_exp = if is_result_inf { u8:0xff } else { result_exp };
  let result_fraction = if is_result_inf { u23:0 } else { result_fraction };

  // - If the numerator is 0 or the denominator is infinity,
  // the result is 0.
  let divide_by_inf = float32::is_inf(y);
  let divide_zero = is_zero(x);
  let is_result_zero = divide_by_inf || divide_zero;
  let result_exp = if is_result_zero { u8:0 } else { result_exp };
  let result_fraction = if is_result_zero { u23:0 } else { result_fraction };

  // Preliminary result until we check for NaN output.
  let result = F32 { sign: result_sign, bexp: result_exp, fraction: result_fraction };

  // - NaNs. NaN cases have highest priority, so we handle them last.
  //  If the numerator or denominator is NaN, the result is NaN.
  //  If we divide inf / inf or 0 / 0 , the result ist NaN.
  let has_nan_arg = float32::is_nan(x) || float32::is_nan(y);
  let zero_divides_zero = is_zero(x) && is_zero(y);
  let inf_divides_inf = float32::is_inf(x) && float32::is_inf(y);
  let is_result_nan = has_nan_arg || zero_divides_zero || inf_divides_inf;
  let result = if is_result_nan { float32::qnan() } else { result };

  result
}


// Test special cases explicitly.
#[test]
fn special_cases_tests() {
  // Try all combinations of these special values.
  let nan  = float32::qnan();
  let inf  = float32::inf(u1:0);
  let zero = float32::zero(u1:0);
  let one  = float32::one(u1:0);

  let result = fpdiv_2x32(zero, zero);
  let _ = assert_eq(result, nan);

  let result = fpdiv_2x32(zero, inf);
  let _ = assert_eq(result, zero);

  let result = fpdiv_2x32(zero, nan);
  let _ = assert_eq(result, nan);

  let result = fpdiv_2x32(zero, one);
  let _ = assert_eq(result, zero);

  let result = fpdiv_2x32(inf, zero);
  let _ = assert_eq(result, inf);

  let result = fpdiv_2x32(inf, inf);
  let _ = assert_eq(result, nan);

  let result = fpdiv_2x32(inf, nan);
  let _ = assert_eq(result, nan);

  let result = fpdiv_2x32(inf, one);
  let _ = assert_eq(result, inf);

  let result = fpdiv_2x32(nan, zero);
  let _ = assert_eq(result, nan);

  let result = fpdiv_2x32(nan, inf);
  let _ = assert_eq(result, nan);

  let result = fpdiv_2x32(nan, nan);
  let _ = assert_eq(result, nan);

  let result = fpdiv_2x32(nan, one);
  let _ = assert_eq(result, nan);

  let result = fpdiv_2x32(one, zero);
  let _ = assert_eq(result, inf);

  let result = fpdiv_2x32(one, inf);
  let _ = assert_eq(result, zero);

  let result = fpdiv_2x32(one, nan);
  let _ = assert_eq(result, nan);

  let result = fpdiv_2x32(one, one);
  let _ = assert_eq(result, one);

  // Check dividing inf by non-one, non-nan,
  // non-zero, non-inf number.
  let two  = F32 { sign: u1:0, bexp: u8:128, fraction: u23:0 };
  let result = fpdiv_2x32(inf, two);
  let _ = assert_eq(result, inf);

  ()
}
