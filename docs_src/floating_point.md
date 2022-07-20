# Floating-point routines

XLS provides implementations of several floating-point operations and may add
more at any time. Here are listed notable details of our implementations or of
floating-point operations in general. Unless otherwise specified, not possible
or out-of-scope, all operations and types should be IEEE-754 compliant.

For example, floating-point exceptions have not been implemented; they're
outside our current scope. The numeric results of multiplication, on the other
hand, should _exactly_ match those of any other compliant implementation.

## APFloat

Floating-point operations are, in general, defined by the same sequence of steps
regardless of their underlying bit widths: fractional parts must be expanded
then aligned, then an operation (add, multiply, etc.) must be performed,
interpreting the fractions as integral types, followed by rounding, special case
handling, and reconstructing an output FP type.

This observation leads to the possibility of _generic_ floating-point routines:
a fully parameterized add, for example, which can be instantiated with and 8-bit
exponent and 23-bit fractional part for binary32 types, and an 11-bit exponent
and 52-bit fractional part for binary64 types. Even more interesting, a
hypothetical bfloat32 type could _immediately_ be supported by, say,
instantiating that adder with, say, 15 exponent bits and 16 fractional ones.

As much as possible, XLS implements FP operations in terms of its
[`APFloat`](https://github.com/google/xls/tree/main/xls/dslx/stdlib/apfloat.x)
(arbitrary-precision floating-point) type. `APFloat` is a parameterized
floating-point structure with a fixed one-bit sign and specifiable exponent and
fractional part size. For ease of use, common types, such as
[`float32`](https://github.com/google/xls/tree/main/xls/dslx/stdlib/float32.x), are
defined in terms of those `APFloat` types.

For example, the generic "is X infinite" operation is defined in `apfloat.x` as:

```dslx-snippet
// Returns whether or not the given APFloat represents an infinite quantity.
pub fn is_inf<EXP_SZ:u32, FRACTION_SZ:u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> u1 {
  (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction == bits[FRACTION_SZ]:0)
}
```

Whereas in `float32.x`, `F32` is defined as:

```dslx-snippet
pub type F32 = apfloat::APFloat<u32:8, u32:23>;
```

and `is_inf` is exposed as:

```dslx-snippet
pub fn is_inf(f: F32) -> u1 { apfloat::is_inf<u32:8, u32:23>(f) }
```

In this way, users can refer to `F32` types and can use them as and with
`float32::is_inf(f)`, giving them simplified access to a generic operation.

More complex functionality such as addition and multiplication are defined in
standalone modules, e.g,
[`apfloat_add_2.x`](https://github.com/google/xls/tree/main/xls/dslx/modules/apfloat_add_2.x)
defining `apfloat_add_2::add` for the `APFloat` type
[`fp32_add_2.x`](https://github.com/google/xls/tree/main/xls/dslx/modules/fp32_add_2.x)
instantiating the operation for the `float32` type.

## Supported operations

Here are listed the routines so far implemented in XLS. Unless otherwise
specified, operations are implemented in terms of APFloats such that they can
support any precisions (aside from corner cases, such as a zero-byte fractional
part).

## Operation details

### Add/sub

Floating-point addition, like any FP operation, is much more complicated than
integer addition, and has many more steps. Being the first operation described,
we'll take extra care to explain floating-point addition:

1.  **Expand fractions:** Floating-point operations are computed with bits
    beyond that in their normal representations for increased precision. For
    IEEE 754 numbers, there are three extra, called the guard, rounding and
    sticky bits. The first two behave normally, but the last, the "sticky" bit,
    is special. During shift operations (below), if a "1" value is ever shifted
    into the sticky bit, it "sticks" - the bit will remain "1" through any
    further shift operations. In this step, the fractions are expanded by these
    three bits.
1.  **Align fractions:** To ensure that fractions are added with appropriate
    magnitudes, they must be aligned according to their exponents. To do so, the
    smaller significant needs to be shifted to the right (each right shift is
    equivalent to increasing the exponent by one).
    -   The extra precision bits are populated in this shift.
    -   As part of this step, the leading 1 bit... and a sign bit Note: The
        sticky bit is calculated and applied in this step.
1.  **Sign-adjustment:** if the fractions differ in sign, then the fraction with
    the smaller initial exponent needs to be (two's complement) negated.
1.  **Add the fractions and capture the carry bit.** Note that, if the signs of
    the fractions differs, then this could result in higher bits being cleared.
1.  **Normalize the fractions:** Shift the result so that the leading '1' is
    present in the proper space. This means shifting right one place if the
    result set the carry bit, and to the left some number of places if high bits
    were cleared.
    -   The sticky bit must be preserved in any of these shifts!
1.  **Rounding:** Here, the extra precision bits are examined to determine if
    the result fraction's last bit should be rounded up. IEEE 754 supports five
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
1.  **Special case handling:** The results are examined for special cases such
    as NaNs, infinities, or (optionally) subnormals.

#### Result sign determination

The sign of the result will normally be the same as the sign of the operand with
the greater exponent, but there are two extra cases to consider. If the operands
have the same exponent, then the sign will be that of the greater fraction, and
if the result is 0, then we favor positive 0 vs. negative 0 (types are as for a
C `float` implementation):

```dslx-snippet
  let fraction = (addend_x as s29) + (addend_y as s29);
  let fraction_is_zero = fraction == s29:0;
  let result_sign = match (fraction_is_zero, fraction < s29:0) {
    (true, _) => u1:0,
    (false, true) => !greater_exp.sign,
    _ => greater_exp.sign,
  };
```

#### Rounding

As complicated as rounding is to describe, its implementation is relatively
straightforward (types are as for a C `float` implementation):

```dslx-snippet
  let normal_chunk = shifted_fraction[0:3];
  let half_way_chunk = shifted_fraction[2:4];
  let do_round_up =
      u1:1 if (normal_chunk > u3:0x4) | (half_way_chunk == u2:0x3)
      else u1:0;

  // We again need an extra bit for carry.
  let rounded_fraction = (shifted_fraction as u28) + u28:0x8 if do_round_up
      else (shifted_fraction as u28);
  let rounding_carry = rounded_fraction[-1:];
```

### Mul

TODO(rspringer): 2021-04-06: This.

### FMA

The `fma` operation (again, fused multiply-add) is a three-operand operation
that computes the product of the first two and the sum of that with the third.
The IEEE 754-2008 description of the operation states that the operation should
be performed "as if with unbounded range and precision", limited only by
rounding of the final result. In other words, this differs from a sequence of a
separate multiply followed by an add in that there is only a single rounding
step (instead of the two involved in separate operations).

In practice, this means A) that the precision of an FMA is higher than
individual ops, and thus that B) an FMA requires significantly more internal
precision bits than naively expected.

For binary32 inputs, to achieve the standard-specified precision, the initial
mul requires the usual 48 ((23 fraction + 1 "hidden") * 2) fraction bits.
When performing the subsequent add step, though, it is necessary to maintain
*72* fraction bits ((23 fraction + 1 "hidden") * 3). Fortunately, this sum
includes the guard, round, and sticky bits (so we don't need 75). The
mathematical derivation of the exact amount will not be given here (as I've not
done it), but the same calculated size would apply for other data types (i.e.,
54 * 2 = 108 and 54 * 3 = 162 for binary64).

Aside from determining the necessary precision bits, the FMA implementation is
rather straightforward, especially after reviewing the adder and multiplier.

## Testing

Several different methods are used to test these routines, depending on
applicability. These are:

-   Reference comparison: exhaustive testing
-   Reference comparison: space-sampling
-   Formal proving

When comparing to a reference, a natural question is the stability of the
reference, i.e., is the reference answer the same across all versions or
environments? Will the answer given by glibc/libm on AArch64 be the same as one
given by a hardware FMA unit on a GPU? Fortunately, all "correct"
implementations will give the same results for the same inputs.\* In addition,
POSIX has the same result-precision language. It's worth noting that -ffast-math
doesn't currently affect FMA emission/fusion/fission/etc.

\* - There are operations for which this is not true. Transcendental ops may
differ between implementations due to the
[_table maker's dilemma_](https://en.wikipedia.org/wiki/Rounding).

### Exhaustive testing

This is the happiest case - where the input space is so small that we can
iterate over every possible input, effectively treating the input as a binary
iteration counter. Sadly, this is uncommon (except, perhaps for ML math), as
binary32 is the precision floor for most problems, and a 64-bit input space is
well beyond our current abilities. Still - if your problem _can_ be exhaustively
tested (with respect to a trusted reference), it _should_ be exhaustively
tested!

None of our current ops are tested in this way, although the bf16 cases
could/should be.

### Space-sampling

When the problem input space is too large for exhaustive testing, then random
samples can be tested instead. This approach can't give complete verification of
an implementation, but, given enough samples, it can yield a high degree of
confidence.

The existing modules are tested in this way. This could be improved by
preventing re-testing of any given sample (at the cost of memory and, perhaps,
atomic/locking costs) and by identifying interesting "corner cases" of the input
space and focusing on those.

### Formal verification

This sort of testing utilizes our formal solver infrastructure to prove
correctness with the solver's internal FP implementation. This is fully
described in the
[solvers documentation](./solvers.md).
