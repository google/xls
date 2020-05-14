// This file implements most of IEEE-754 single-precision
// floating-point addition, with the following exceptions:
//  - Both input and output denormals are treated as/flushed to 0.
//  - Only round-to-nearest mode is supported.
//  - No exception flags are raised/reported.
// In all other cases, results should be identical to other
// conforming implementations (modulo exact significand
// values in the NaN case.
import float32

type F32 = float32::F32;

fn fpadd_2x32(x: F32, y: F32) -> F32 {
  // Step 1: align the significands.
  //  - Bit widths: Base significant: u23.
  //  - Add the implied/leading 1 bit: u23 -> u24
  //  - Add a sign bit: u24 -> u25
  let wide_x = ((x.sfd as u28) | u28:0x800000) << u28:3 in
  let wide_y = ((y.sfd as u28) | u28:0x800000) << u28:3 in

  // Flush denormals to 0.
  let wide_x = u28:0 if x.bexp == u8:0 else wide_x in
  let wide_y = u28:0 if y.bexp == u8:0 else wide_y in

  // Shift the significands to align with the largest exponent.
  let greater_exp = x if x.bexp > y.bexp else y in
  let shift_x = greater_exp.bexp - x.bexp in
  let shift_y = greater_exp.bexp - y.bexp in
  let shifted_x = (wide_x >> (shift_x as u28)) as s28 in
  let shifted_y = (wide_y >> (shift_y as u28)) as s28 in

  // Calculate the sticky bits - set to 1 if any set bits were
  // shifted out of the significands.
  let dropped_x = wide_x << ((u8:28 - shift_x) as u28) in
  let dropped_y = wide_y << ((u8:28 - shift_y) as u28) in
  let sticky_x = dropped_x > u28:0 in
  let sticky_y = dropped_y > u28:0 in
  let addend_x = shifted_x | (sticky_x as s28) in
  let addend_y = shifted_y | (sticky_y as s28) in

  // Invert the mantissa if its source has a different sign than
  // the larger value.
  let addend_x = -addend_x if x.sign != greater_exp.sign else addend_x in
  let addend_y = -addend_y if y.sign != greater_exp.sign else addend_y in

  // Step 2: Do some addition!
  // Add one bit to capture potential carry: s28 -> s29.
  let sfd = (addend_x as s29) + (addend_y as s29) in
  let sfd_is_zero = sfd == s29:0 in
  let result_sign = match (sfd_is_zero, sfd < s29:0) {
    (true, _) => u1:0;
    (false, true) => ~greater_exp.sign;
    _ => greater_exp.sign;
  } in

  // Get the absolute value of the result then chop off the sign bit: s29 -> u28.
  let abs_sfd = (-sfd if sfd < s29:0 else sfd) as u28 in

  // Step 3: Normalize the significand (shift until the leading bit is a 1).
  // If the carry bit is set, shift right one bit (to capture the new bit of
  // precision) - but don't drop the sticky bit!
  let carry_bit = abs_sfd[-1:] in
  let carry_sfd = (abs_sfd >> u28:1) as u27 in
  let carry_sfd = carry_sfd | (abs_sfd[0:1] as u27) in

  // If we cancelled higher bits, then we'll need to shift left.
  // Leading zeroes will be 1 if there's no carry or cancellation.
  let leading_zeroes = clz(abs_sfd) in
  let cancel = leading_zeroes > u28:1 in
  let cancel_sfd = (abs_sfd << (leading_zeroes - u28:1)) as u27 in
  let shifted_sfd = match(carry_bit, cancel) {
    (true, false) => carry_sfd;
    (false, true) => cancel_sfd;
    (false, false) => abs_sfd as u27;
    _ => fail!(u27:666)
  } in

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
  let normal_chunk = shifted_sfd[0:3] in
  let half_way_chunk = shifted_sfd[2:4] in
  let do_round_up =
      u1:1 if (normal_chunk > u3:0x4) | (half_way_chunk == u2:0x3)
      else u1:0 in

  // We again need an extra bit for carry.
  let rounded_sfd = (shifted_sfd as u28) + u28:0x8 if do_round_up
      else (shifted_sfd as u28) in
  let rounding_carry = rounded_sfd[-1:] in

  // After rounding, we can chop off the extra precision bits.
  // As with normalization, if we carried, we need to shift right
  // an extra place.
  let sfd_shift = u28:3 + (u28:1 if rounded_sfd[-1:] else u28:0) in
  let result_sfd = (rounded_sfd >> sfd_shift) as u23 in

  // Finally, adjust the exponent based on addition and rounding -
  // each bit of carry or cancellation moves it by one place.
  let wide_exponent = (greater_exp.bexp as s10) + (rounding_carry as s10) +
      s10:1 - (leading_zeroes as s10) in
  let wide_exponent = s10:0 if sfd_is_zero else wide_exponent in

  // Chop off the sign bit.
  let wide_exponent = u9:0 if wide_exponent < s10:0 else (wide_exponent as u9) in

  // Extra bonus step 5: special case handling!

  // If the exponent underflowed, don't bother with denormals. Just flush to 0.
  let result_sfd = u23:0 if wide_exponent < u9:1 else result_sfd in

  // Handle exponent overflow infinities.
  let result_sfd = result_sfd if wide_exponent < u9:255 else u23:0 in
  let result_exponent = wide_exponent as u8 if wide_exponent < u9:255 else u8:255 in

  // Handle arg infinities.
  let is_operand_inf = float32::is_inf(x) | float32::is_inf(y) in
  let result_exponent = u8:255 if is_operand_inf else result_exponent in
  let result_sfd = u23:0 if is_operand_inf else result_sfd in
  // Result infinity is negative iff all infinite operands are neg.
  let has_pos_inf = (float32::is_inf(x) & (x.sign == u1:0)) |
                    (float32::is_inf(y) & (y.sign == u1:0)) in
  let result_sign = ~has_pos_inf if is_operand_inf else result_sign in

  // Handle NaN; NaN trumps infinities, so we handle it last.
  // -inf + inf = NaN, i.e., if we have both positive and negative inf.
  let has_neg_inf =
      (float32::is_inf(x) & (x.sign == u1:1)) | (float32::is_inf(y) & (y.sign == u1:1)) in
  let is_result_nan = float32::is_nan(x) | float32::is_nan(y) | (has_pos_inf & has_neg_inf) in
  let result_exponent = u8:255 if is_result_nan else result_exponent in
  let result_sfd = u23:0x400000 if is_result_nan else result_sfd in
  let result_sign = u1:0 if is_result_nan else result_sign in

  // Finally (finally!), construct the output float.
  F32 { sign: result_sign, bexp: result_exponent, sfd: result_sfd as u23 }
}
