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
// implementations, modulo exact significand values in the NaN case (we emit a
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
  let x_sfd = (x.sfd as u64) | u64:0x80_0000;
  let y_sfd = (y.sfd as u64) | u64:0x80_0000;

  // 1a. Flush subnorms to 0.
  let x_sfd = u64:0 if x.bexp == u8:0 else x_sfd;
  let y_sfd = u64:0 if y.bexp == u8:0 else y_sfd;

  // 2. Subtract non-biased exponents.
  //  - Remove the bias from the exponents, subtract them, then restore the bias.
  //  - Simplifies from
  //      (A - 127) - (B - 127) + 127 = exp
  //    to
  //      A + B + 127 = exp
  let exp = (x.bexp as s10) - (y.bexp as s10) + s10:0x7f;

  // 3. Shift numerator and adjust exponent.
  let exp = exp - s10:1 if x_sfd < y_sfd else exp;
  let x_sfd = x_sfd << u64:31 if x_sfd < y_sfd else x_sfd << u64:30;

  // 4. Divide integer mantissas.
  let sfd = std::iterative_div(x_sfd, y_sfd) as u32;

  // 5. Account for remainder / error.
  let sfd_has_bit_in_six_lsbs = sfd[0:6] != u6:0;
  let remainder_detected = (y_sfd * sfd as u64) != x_sfd;
  let set_sfd_lsb = !sfd_has_bit_in_six_lsbs && remainder_detected;
  let sfd = sfd | u32:1 if set_sfd_lsb else sfd;

  // 6. Check rounding conditions.
  // We use nearest, half to even rounding.
  // - We round down if less than 1/2 way between values, i.e.
  // - We round up if we're more than 1/2 way
  // - If halfway, then we round whichever direction makes the
  //   result even.
  let round_bits = sfd[0:7];
  let is_half_way = round_bits[-1:] & (round_bits[:-1] == u6:0);
  let greater_than_half_way = round_bits[-1:] & (round_bits[:-1] != u6:0);

  // We're done with the extra precision bits now, so shift the
  // significand into its almost-final width, adding one extra
  // bit for potential rounding overflow.
  let sfd = (sfd >> u32:7) as u23;
  let sfd = sfd as u24;
  let do_round_up = greater_than_half_way || (is_half_way & sfd[0:1]);
  let sfd = sfd + u24:1 if do_round_up else sfd;

  // Adjust the exponent if we overflowed during rounding.
  // After checking for subnormals, we don't need the sign bit anymore.
  let exp = exp + s10:1 if sfd[-1:] else exp;
  let is_subnormal = exp <= s10:0;

  // We're done - except for special cases...
  let result_sign = x.sign ^ y.sign;
  let result_exp = exp as u9;
  let result_sfd = sfd as u23;

  // 6. Special cases!
  // - Subnormals: flush to 0.
  let result_exp = u9:0 if is_subnormal else result_exp;
  let result_sfd = u23:0 if is_subnormal else result_sfd;

  // - Overflow infinites. Exp to 255, clear sfd.
  let result_sfd = result_sfd if result_exp < u9:0xff else u23:0;
  let result_exp = result_exp as u8 if result_exp < u9:0xff else u8:0xff;

  // - If the denominator is 0 or the numerator is infinity,
  // the result is infinity.
  let divide_by_zero = is_zero(y);
  let divide_inf = float32::is_inf(x);
  let is_result_inf = divide_by_zero || divide_inf;
  let result_exp = u8:0xff if is_result_inf else result_exp;
  let result_sfd = u23:0 if is_result_inf else result_sfd;

  // - If the numerator is 0 or the denominator is infinity,
  // the result is 0.
  let divide_by_inf = float32::is_inf(y);
  let divide_zero = is_zero(x);
  let is_result_zero = divide_by_inf || divide_zero;
  let result_exp = u8:0 if is_result_zero else result_exp;
  let result_sfd = u23:0 if is_result_zero else result_sfd;

  // Preliminary result until we check for NaN output.
  let result = F32 { sign: result_sign, bexp: result_exp, sfd: result_sfd };

  // - NaNs. NaN cases have highest priority, so we handle them last.
  //  If the numerator or denominator is NaN, the result is NaN.
  //  If we divide inf / inf or 0 / 0 , the result ist NaN.
  let has_nan_arg = float32::is_nan(x) || float32::is_nan(y);
  let zero_divides_zero = is_zero(x) && is_zero(y);
  let inf_divides_inf = float32::is_inf(x) && float32::is_inf(y);
  let is_result_nan = has_nan_arg || zero_divides_zero || inf_divides_inf;
  let result = float32::qnan() if is_result_nan else result;

  result
}


// Test special cases explicitly.
#![test]
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
  let two  = F32 { sign: u1:0, bexp: u8:128, sfd: u23:0 };
  let result = fpdiv_2x32(inf, two);
  let _ = assert_eq(result, inf);

  ()
}
