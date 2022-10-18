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
// conforming implementations (modulo exact fraction values in the NaN case.

// The bit widths of different float components are given
// in comments throughout this implementation, listed
// relative to the widths of a standard float32.
import apfloat
import std

type APFloat = apfloat::APFloat;

// Usage:
//  - EXP_SZ: The number of bits in the exponent.
//  - FRACTION_SZ: The number of bits in the fractional part of the FP value.
//  - x, y: The two floating-point numbers to add.
//
// Derived parametrics:
//  - WIDE_EXP: Widened exponent to capture a possible carry bit.
//  - CARRY_EXP: WIDE_EXP plus one sign bit.
//  - WIDE_FRACTION: Widened fraction to contain full precision + rounding
//    (guard & sticky) bits.
//  - CARRY_FRACTION: WIDE_FRACTION plus one bit to capture a possible carry bit.
//  - NORMALIZED_FRACTION: WIDE_FRACTION minus one bit for post normalization
//    (where the implicit leading 1 bit is dropped).
pub fn add<EXP_SZ: u32, FRACTION_SZ: u32,
    WIDE_EXP: u32 = EXP_SZ + u32:1,
    CARRY_EXP: u32 = WIDE_EXP + u32:1,
    WIDE_FRACTION: u32 = FRACTION_SZ + u32:5,
    CARRY_FRACTION: u32 = WIDE_FRACTION + u32:1,
    NORMALIZED_FRACTION: u32 = WIDE_FRACTION - u32:1>(
    x: APFloat<EXP_SZ, FRACTION_SZ>, y: APFloat<EXP_SZ, FRACTION_SZ>) ->
    APFloat<EXP_SZ, FRACTION_SZ> {
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
  let is_operand_inf = apfloat::is_inf<EXP_SZ, FRACTION_SZ>(x) |
      apfloat::is_inf<EXP_SZ, FRACTION_SZ>(y);
  let result_exponent = if is_operand_inf { max_exp } else { result_exponent };
  let result_fraction = if is_operand_inf { uN[FRACTION_SZ]:0 } else { result_fraction };
  // Result infinity is negative iff all infinite operands are neg.
  let has_pos_inf = (apfloat::is_inf<EXP_SZ, FRACTION_SZ>(x) & (x.sign == u1:0)) |
                    (apfloat::is_inf<EXP_SZ, FRACTION_SZ>(y) & (y.sign == u1:0));
  let result_sign = if is_operand_inf { !has_pos_inf } else { result_sign };

  // Handle NaN; NaN trumps infinities, so we handle it last.
  // -inf + inf = NaN, i.e., if we have both positive and negative inf.
  let has_neg_inf =
      (apfloat::is_inf<EXP_SZ, FRACTION_SZ>(x) & (x.sign == u1:1)) |
      (apfloat::is_inf<EXP_SZ, FRACTION_SZ>(y) & (y.sign == u1:1));
  let is_result_nan = apfloat::is_nan<EXP_SZ, FRACTION_SZ>(x) |
      apfloat::is_nan<EXP_SZ, FRACTION_SZ>(y) | (has_pos_inf & has_neg_inf);
  let result_exponent = if is_result_nan { max_exp } else { result_exponent };
  let result_fraction = if is_result_nan { fraction_high_bit } else { result_fraction };
  let result_sign = if is_result_nan { u1:0 } else { result_sign };

  // Finally (finally!), construct the output float.
  APFloat<EXP_SZ, FRACTION_SZ> { sign: result_sign, bexp: result_exponent,
                            fraction: result_fraction as uN[FRACTION_SZ] }
}
