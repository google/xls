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
// conforming implementations (modulo exact significand
// values in the NaN case.
import float32
import std

type F32 = float32::F32;

// Determines if the given value is 0, taking into account
// flushing subnormals.
fn is_zero(x: F32) -> u1 {
  x.bexp == u8:0
}

pub fn fpmul_2x32(x: F32, y: F32) -> F32 {
  // 1. Get and expand mantissas.
  let x_sfd = (x.sfd as u48) | u48:0x80_0000;
  let y_sfd = (y.sfd as u48) | u48:0x80_0000;

  // 1a. Flush subnorms to 0.
  let x_sfd = u48:0 if x.bexp == u8:0 else x_sfd;
  let y_sfd = u48:0 if y.bexp == u8:0 else y_sfd;

  // 2. Multiply integer mantissas.
  let sfd: u48 = x_sfd * y_sfd;

  // 3. Add non-biased exponents.
  //  - Remove the bias from the exponents, add them, then restore the bias.
  //  - Simplifies from
  //      (A - 127) + (B - 127) + 127 = exp
  //    to
  //      A + B - 127 = exp
  let exp = (x.bexp as s10) + (y.bexp as s10) - s10:0x7f;

  // Here is where we'd handle subnormals if we cared to.
  // If the exponent remains < 0, even after reapplying the bias,
  // then we'd calculate the extra exponent needed to get back to 0.
  // We'd set the result exponent to 0 and shift the sfd to the right
  // to capture that "extra" exponent.
  // Since we just flush subnormals, we don't have to do any of that.
  // Instead, if we're multiplying by 0, the result is 0.
  let exp = s10:0 if is_zero(x) || is_zero(y) else exp;

  // 4. Normalize. Adjust the significand until our leading 1 is;
  // bit 47 (the first past the 46 bits of actual significand).
  // That'll be a shift of 1 or 0 places (since we're multiplying
  // two values with leading 1s in bit 24).
  let sfd_shift = sfd[-1:] as u48;

  // If there is a leading 1, then we need to shift to the right one place -
  // that means we gained a new significant digit at the top.
  // Dont forget to maintain the sticky bit!
  let sticky = sfd[0:1] as u48;
  let sfd = sfd >> sfd_shift;
  let sfd = sfd | sticky;

  // Update the exponent if we shifted.
  let exp = exp + (sfd_shift as s10);
  // If the value is currently subnormal, then we need to shift right by one
  // space: a subnormal value doesn't have the leading 1, and thus has one
  // fewer significant digits than normal numbers - in a sense, the -1th bit
  // is the least significant (0) bit.
  // Rounding (below) expects the least significant digit to start at position
  // 0, so we shift subnormals to the left by one position to match normals.
  // Again, track the sticky bit. This could be combined with the shift
  // above, but it's easier to understand (and comment) if separated, and the
  // optimizer will clean it up anyway.
  let sticky = sfd[0:1] as u48;
  let sfd = sfd >> u48:1 if exp <= s10:0 else sfd;
  let sfd = sfd | sticky;

  // 5. Round - we use nearest, half to even rounding.
  // - We round down if less than 1/2 way between values, i.e.
  //   if bit 23 is 0. Rounding down is equivalent to doing nothing.
  // - We round up if we're more than 1/2 way, i.e., if bit 23
  //   is set along with any bit lower than 23.
  // - If halfway (bit 23 set and no bit lower), then we round;
  //   whichever direction makes the result even. In other words,
  //   we round up if bit 25 is set.
  let is_half_way = sfd[22:23] & (sfd[0:22] == u22:0);
  let greater_than_half_way = sfd[22:23] & (sfd[0:22] != u22:0);
  let do_round_up = greater_than_half_way || (is_half_way & sfd[23:24]);

  // We're done with the extra precision bits now, so shift the
  // significand into its almost-final width, adding one extra
  // bit for potential rounding overflow.
  let sfd = (sfd >> u48:23) as u23;
  let sfd = sfd as u24;
  let sfd = sfd + u24:1 if do_round_up else sfd;

  // Adjust the exponent if we overflowed during rounding.
  // After checking for subnormals, we don't need the sign bit anymore.
  let exp = exp + s10:1 if sfd[-1:] else exp;
  let is_subnormal = exp <= s10:0;

  // We're done - except for special cases...
  let result_sign = x.sign != y.sign;
  let result_exp = exp as u9;
  let result_sfd = sfd as u23;

  // 6. Special cases!
  // - Subnormals: flush to 0.
  let result_exp = u9:0 if is_subnormal else result_exp;
  let result_sfd = u23:0 if is_subnormal else result_sfd;

  // - Overflow infinites. Exp to 255, clear sfd.
  let result_sfd = result_sfd if result_exp < u9:0xff else u23:0;
  let result_exp = result_exp as u8 if result_exp < u9:0xff else u8:0xff;

  // - Arg infinites. Any arg is infinite == result is infinite.
  let is_operand_inf = float32::is_inf(x) || float32::is_inf(y);
  let result_exp = u8:0xff if is_operand_inf else result_exp;
  let result_sfd = u23:0 if is_operand_inf else result_sfd;

  // - NaNs. NaN trumps infinities, so we handle it last.
  //   inf * 0 = NaN, i.e.,
  let has_0_arg = is_zero(x) || is_zero(y);
  let has_nan_arg = float32::is_nan(x) || float32::is_nan(y);
  let has_inf_arg = float32::is_inf(x) || float32::is_inf(y);
  let is_result_nan = has_nan_arg || (has_0_arg && has_inf_arg);
  let result_exp = u8:0xff if is_result_nan else result_exp;
  let result_sfd = u23:0x40_0000 if is_result_nan else result_sfd;
  let result_sign = u1:0 if is_result_nan else result_sign;

  F32 { sign: result_sign, bexp: result_exp, sfd: result_sfd }
}
