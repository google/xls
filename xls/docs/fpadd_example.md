# DSLX Example: Floating-point addition

This document explains how floating-point addition is implemented in DSLX, along
with a discussion on the algorithm and how it maps to DSLX.

This document assumes familiarity with floating-point numbers in general
(layout, precision, error, etc.).

[TOC]

## Background

Floating-point addition, like any FP operation, is much more complicated than
integer addition, and has many more steps.

1.  Expand significands: Floating-point operations are computed with bits beyond
    that in their normal representations for increased precision. For IEEE 754
    numbers, there are three extra, called the guard, rounding and sticky bits.
    The first two behave normally, but the last, the "sticky" bit, is special.
    During shift operations (below), if a "1" value is ever shifted into the
    sticky bit, it "sticks" - the bit will remain "1" through any further shift
    operations. In this step, the significands are expanded by these three bits.
1.  Align significands: To ensure that significands are added with appropriate
    magnitudes, they must be aligned according to their exponents. To do so, the
    smaller significant needs to be shifted to the right (each right shift is
    equivalent to increasing the exponent by one).
    -   The extra precision bits are populated in this shift.
    -   As part of this step, the leading 1 bit... and a sign bit Note: The
        sticky bit is calculated and applied in this step.
1.  Sign-adjustment: if the significands differ in sign, then the significand
    with the smaller initial exponent needs to be (two's complement) negated.
1.  Add the significands and capture the carry bit. Note that, if the signs of
    the significands differs, then this could result in higher bits being
    cleared.
1.  Normalize the significands: Shift the result so that the leading '1' is
    present in the proper space. This means shifting right one place if the
    result set the carry bit, and to the left some number of places if high bits
    were cleared.
    -   The sticky bit must be preserved in any of these shifts!
1.  Rounding: Here, the extra precision bits are examined to determine if the
    result significand's last bit should be rounded up. IEEE 754 supports five
    rounding modes:
    -   Round towards 0: just chop off the extra precision bits.
    -   Round towards +infinity: round up if any extra precision bits are set.
    -   Round towards -infinity: round down if any extra precision bits are set.
    -   Round to nearest, ties away from zero: Rounds to the nearest value. In
        cases where the extra precision bits are halfway between values, i.e.,
        0b100, then the result is rounded up for positive numbers and down for
        negative ones.
    -   Round to nearest, ties to even: Rounds to the nearest value. In cases
        where the extra precision bits are halfway between values, then the
        result is rounded in whichever direction causes the LSB of the result
        significant to be 0.
        -   This is the most commonly-used rounding mode.
        -   This is [currently] the only supported mode by the DSLX
            implementation.
1.  Special case handling: The results are examined for special cases such as
    NaNs, infinities, or (optionally) subnormals.

## DSLX implementation

With an understanding of the algorithm above, the DSLX implementation is
relatively straightforward. "Interesting" chunks are described below.

### Result sign determination

The sign of the result will normally be the same as the sign of the operand with
the greater exponent, but there are two extra cases to consider. If the operands
have the same exponent, then the sign will be that of the greater significand,
and if the result is 0, then we favor positive 0 vs. negative 0.

```
  let sfd = (addend_x as s29) + (addend_y as s29) in
  let sfd_is_zero = sfd == s29:0 in
  let result_sign = match (sfd_is_zero, sfd < s29:0) {
    (true, _) => u1:0;
    (false, true) => ~greater_exp.sign;
    _ => greater_exp.sign;
  } in
```

### Rounding

As complicated as rounding is to describe, its implementation is relatively
straightforward.

```
  let normal_chunk = shifted_sfd[0:3] in
  let half_way_chunk = shifted_sfd[2:4] in
  let do_round_up =
      u1:1 if (normal_chunk > u3:0x4) | (half_way_chunk == u2:0x3)
      else u1:0 in

  // We again need an extra bit for carry.
  let rounded_sfd = (shifted_sfd as u28) + u28:0x8 if do_round_up
      else (shifted_sfd as u28) in
  let rounding_carry = rounded_sfd[-1:] in
```

The behavior of logic descriptions - even in a higher level language such as
DSLX - can be non-obvious to a new reader, so extensive comments, such as those
here, are invaluable.
