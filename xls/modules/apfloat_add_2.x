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

// This file implements floating point addition based on a
// generalization of IEEE 754 single-precision floating-point
// addition, with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact significand
// values in the NaN case.

// The bit widths of different float components are given
// in comments throughout this implementation, listed
// relative to the widths of a standard float32.
import apfloat
import std

type APFloat = apfloat::APFloat;

// Usage:
//  - EXP_SZ: The number of bits in the exponent.
//  - SFD_SZ: The number of bits in the significand (see
//        https://en.wikipedia.org/wiki/Significand for "significand"
//        vs "mantissa" naming).
//  - x, y: The two floating-point numbers to add.
//
// Derived parametrics:
//  - WIDE_EXP: Widened exponent to capture a possible carry bit.
//  - CARRY_EXP: WIDE_EXP plus one sign bit.
//  - WIDE_SFD: Widened sfd to contain full precision + rounding
//    (guard & sticky) bits.
//  - CARRY_SFD: WIDE_SFD plus one bit to capture a possible carry bit.
//  - NORMALIZED_SFD: WIDE_SFD minus one bit for post normalization
//    (where the implicit leading 1 bit is dropped).
pub fn add<EXP_SZ: u32, SFD_SZ: u32,
    WIDE_EXP: u32 = EXP_SZ + u32:1,
    CARRY_EXP: u32 = WIDE_EXP + u32:1,
    WIDE_SFD: u32 = SFD_SZ + u32:5,
    CARRY_SFD: u32 = WIDE_SFD + u32:1,
    NORMALIZED_SFD: u32 = WIDE_SFD - u32:1>(
    x: APFloat<EXP_SZ, SFD_SZ>, y: APFloat<EXP_SZ, SFD_SZ>) ->
    APFloat<EXP_SZ, SFD_SZ> {
  // Step 1: align the significands.
  //  - Bit widths: Base significand: u23.
  //  - Add the implied/leading 1 bit: u23 -> u24
  //  - Add a sign bit: u24 -> u25
  let sfd_high_bit = uN[SFD_SZ]:1 << (SFD_SZ as uN[SFD_SZ] - uN[SFD_SZ]:1);
  let wide_sfd_high_bit = sfd_high_bit as uN[WIDE_SFD] << uN[WIDE_SFD]:1;
  let wide_x = ((x.sfd as uN[WIDE_SFD]) | wide_sfd_high_bit) << uN[WIDE_SFD]:3;
  let wide_y = ((y.sfd as uN[WIDE_SFD]) | wide_sfd_high_bit) << uN[WIDE_SFD]:3;

  // Flush denormals to 0.
  let wide_x = uN[WIDE_SFD]:0 if x.bexp == uN[EXP_SZ]:0 else wide_x;
  let wide_y = uN[WIDE_SFD]:0 if y.bexp == uN[EXP_SZ]:0 else wide_y;

  // Shift the significands to align with the largest exponent.
  let greater_exp = x if x.bexp > y.bexp else y;
  let shift_x = greater_exp.bexp - x.bexp;
  let shift_y = greater_exp.bexp - y.bexp;
  let shifted_x = (wide_x >> (shift_x as uN[WIDE_SFD])) as sN[WIDE_SFD];
  let shifted_y = (wide_y >> (shift_y as uN[WIDE_SFD])) as sN[WIDE_SFD];

  // Calculate the sticky bits - set to 1 if any set bits were
  // shifted out of the significands.
  let dropped_x = wide_x << ((WIDE_SFD as uN[EXP_SZ] - shift_x) as uN[WIDE_SFD]);
  let dropped_y = wide_y << ((WIDE_SFD as uN[EXP_SZ] - shift_y) as uN[WIDE_SFD]);
  let sticky_x = dropped_x > uN[WIDE_SFD]:0;
  let sticky_y = dropped_y > uN[WIDE_SFD]:0;
  let addend_x = shifted_x | (sticky_x as sN[WIDE_SFD]);
  let addend_y = shifted_y | (sticky_y as sN[WIDE_SFD]);

  // Invert the mantissa if its source has a different sign than
  // the larger value.
  let addend_x = -addend_x if x.sign != greater_exp.sign else addend_x;
  let addend_y = -addend_y if y.sign != greater_exp.sign else addend_y;

  // Step 2: Do some addition!
  // Add one bit to capture potential carry: s28 -> s29.
  let sfd = (addend_x as sN[CARRY_SFD]) + (addend_y as sN[CARRY_SFD]);
  let sfd_is_zero = sfd == sN[CARRY_SFD]:0;
  let result_sign = match (sfd_is_zero, sfd < sN[CARRY_SFD]:0) {
    (true, _) => u1:0,
    (false, true) => !greater_exp.sign,
    _ => greater_exp.sign,
  };

  // Get the absolute value of the result then chop off the sign bit: s29 -> u28.
  let abs_sfd = (-sfd if sfd < sN[CARRY_SFD]:0 else sfd) as uN[WIDE_SFD];

  // Step 3: Normalize the significand (shift until the leading bit is a 1).
  // If the carry bit is set, shift right one bit (to capture the new bit of
  // precision) - but don't drop the sticky bit!
  let carry_bit = abs_sfd[-1:];
  let carry_sfd = (abs_sfd >> uN[WIDE_SFD]:1) as uN[NORMALIZED_SFD];
  let carry_sfd = carry_sfd | (abs_sfd[0:1] as uN[NORMALIZED_SFD]);

  // If we cancelled higher bits, then we'll need to shift left.
  // Leading zeroes will be 1 if there's no carry or cancellation.
  let leading_zeroes = clz(abs_sfd);
  let cancel = leading_zeroes > uN[WIDE_SFD]:1;
  let cancel_sfd = (abs_sfd << (leading_zeroes - uN[WIDE_SFD]:1)) as uN[NORMALIZED_SFD];
  let shifted_sfd = match(carry_bit, cancel) {
    (true, false) => carry_sfd,
    (false, true) => cancel_sfd,
    (false, false) => abs_sfd as uN[NORMALIZED_SFD],
    _ => fail!(uN[NORMALIZED_SFD]:666)
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
  //    in whichever direction makes the last bit in the significand 0.
  let normal_chunk = shifted_sfd[0:3];
  let half_way_chunk = shifted_sfd[2:4];
  let do_round_up =
      u1:1 if (normal_chunk > u3:0x4) | (half_way_chunk == u2:0x3)
      else u1:0;

  // We again need an extra bit for carry.
  let rounded_sfd = (shifted_sfd as uN[WIDE_SFD]) + uN[WIDE_SFD]:0x8 if do_round_up
      else (shifted_sfd as uN[WIDE_SFD]);
  let rounding_carry = rounded_sfd[-1:];

  // After rounding, we can chop off the extra precision bits.
  // As with normalization, if we carried, we need to shift right
  // an extra place.
  let sfd_shift = uN[WIDE_SFD]:3 +
      (uN[WIDE_SFD]:1 if rounded_sfd[-1:] else uN[WIDE_SFD]:0);
  let result_sfd = (rounded_sfd >> sfd_shift) as uN[SFD_SZ];

  // Finally, adjust the exponent based on addition and rounding -
  // each bit of carry or cancellation moves it by one place.
  let wide_exponent =
      (greater_exp.bexp as sN[CARRY_EXP]) +
      (rounding_carry as sN[CARRY_EXP]) +
      sN[CARRY_EXP]:1 - (leading_zeroes as sN[CARRY_EXP]);
  let wide_exponent = sN[CARRY_EXP]:0 if sfd_is_zero else wide_exponent;

  // Chop off the sign bit.
  let wide_exponent =
      uN[WIDE_EXP]:0
      if wide_exponent < sN[CARRY_EXP]:0 else
      (wide_exponent as uN[WIDE_EXP]);

  // Extra bonus step 5: special case handling!

  // If the exponent underflowed, don't bother with denormals. Just flush to 0.
  let result_sfd = uN[SFD_SZ]:0 if wide_exponent < uN[WIDE_EXP]:1 else result_sfd;

  // Handle exponent overflow infinities.
  let saturated_exp = std::mask_bits<EXP_SZ>() as uN[WIDE_EXP];
  let max_exp = std::mask_bits<EXP_SZ>();
  let result_sfd =
      result_sfd
      if wide_exponent < saturated_exp
      else uN[SFD_SZ]:0;
  let result_exponent =
      wide_exponent as uN[EXP_SZ]
      if wide_exponent < saturated_exp
      else max_exp;

  // Handle arg infinities.
  let is_operand_inf = apfloat::is_inf<EXP_SZ, SFD_SZ>(x) |
      apfloat::is_inf<EXP_SZ, SFD_SZ>(y);
  let result_exponent = max_exp if is_operand_inf else result_exponent;
  let result_sfd = uN[SFD_SZ]:0 if is_operand_inf else result_sfd;
  // Result infinity is negative iff all infinite operands are neg.
  let has_pos_inf = (apfloat::is_inf<EXP_SZ, SFD_SZ>(x) & (x.sign == u1:0)) |
                    (apfloat::is_inf<EXP_SZ, SFD_SZ>(y) & (y.sign == u1:0));
  let result_sign = !has_pos_inf if is_operand_inf else result_sign;

  // Handle NaN; NaN trumps infinities, so we handle it last.
  // -inf + inf = NaN, i.e., if we have both positive and negative inf.
  let has_neg_inf =
      (apfloat::is_inf<EXP_SZ, SFD_SZ>(x) & (x.sign == u1:1)) |
      (apfloat::is_inf<EXP_SZ, SFD_SZ>(y) & (y.sign == u1:1));
  let is_result_nan = apfloat::is_nan<EXP_SZ, SFD_SZ>(x) |
      apfloat::is_nan<EXP_SZ, SFD_SZ>(y) | (has_pos_inf & has_neg_inf);
  let result_exponent = max_exp if is_result_nan else result_exponent;
  let result_sfd = sfd_high_bit if is_result_nan else result_sfd;
  let result_sign = u1:0 if is_result_nan else result_sign;

  // Finally (finally!), construct the output float.
  APFloat<EXP_SZ, SFD_SZ> { sign: result_sign, bexp: result_exponent,
                            sfd: result_sfd as uN[SFD_SZ] }
}
