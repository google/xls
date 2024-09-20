# Floating-point routines

XLS provides implementations of several floating-point operations and may add
more at any time. Here are listed notable details of our implementations or of
floating-point operations in general. Unless otherwise specified, not possible
or out-of-scope, all operations and types should be IEEE-754 compliant.

For example, floating-point exceptions have not been implemented; they're
outside our current scope. The numeric results of multiplication, on the other
hand, should *exactly* match those of any other compliant implementation.

## APFloat

Floating-point operations are, in general, defined by the same sequence of steps
regardless of their underlying bit widths: fractional parts must be expanded
then aligned, then an operation (add, multiply, etc.) must be performed,
interpreting the fractions as integral types, followed by rounding, special case
handling, and reconstructing an output FP type.

This observation leads to the possibility of *generic* floating-point routines:
a fully parameterized add, for example, which can be instantiated with and 8-bit
exponent and 23-bit fractional part for binary32 types, and an 11-bit exponent
and 52-bit fractional part for binary64 types. Even more interesting, a
hypothetical bfloat32 type could *immediately* be supported by, say,
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
pub fn is_inf<EXP_SZ:u32, FRACTION_SZ:u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool {
  (x.bexp == std::mask_bits<EXP_SZ>() && x.fraction == bits[FRACTION_SZ]:0)
}
```

Whereas in `float32.x`, `F32` is defined as:

```dslx-snippet
pub type F32 = apfloat::APFloat<u32:8, u32:23>;
```

and `is_inf` is exposed as:

```dslx-snippet
pub fn is_inf(f: F32) -> bool { apfloat::is_inf<u32:8, u32:23>(f) }
```

In this way, users can refer to `F32` types and can use them as and with
`float32::is_inf(f)`, giving them simplified access to a generic operation.

## Supported operations

Here are listed the routines so far implemented in XLS. Unless otherwise
specified, operations are implemented in terms of `APFloat`s such that they can
support any precisions (aside from corner cases, such as a zero-byte fractional
part).

### `apfloat::tag`

```dslx-snippet
pub fn tag<EXP_SZ:u32, FRACTION_SZ:u32>(input_float: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloatTag
```

Returns the type of float as one of `APFloatTag::ZERO`, `APFloatTag::SUBNORMAL`,
`APFloatTag::INFINITY`, `APFloatTag::NAN` and `APFloatTag::NORMAL`.

### `apfloat::qnan`

```dslx-snippet
pub fn qnan<EXP_SZ:u32, FRACTION_SZ:u32>() -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns a [`quiet NaN`](https://en.wikipedia.org/wiki/NaN#Quiet_NaN).

### `apfloat::is_nan`

```dslx-snippet
pub fn is_nan<EXP_SZ:u32, FRACTION_SZ:u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns whether or not the given `APFloat` represents `NaN`.

### `apfloat::inf`

```dslx-snippet
pub fn inf<EXP_SZ:u32, FRACTION_SZ:u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns a positive or a negative infinity depending upon the given sign
parameter.

### `apfloat::is_inf`

```dslx-snippet
pub fn is_inf<EXP_SZ:u32, FRACTION_SZ:u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns whether or not the given `APFloat` represents an infinite quantity.

### `apfloat::is_pos_inf`

```dslx-snippet
pub fn is_pos_inf<EXP_SZ:u32, FRACTION_SZ:u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns whether or not the given `APFloat` represents a positive infinite quantity.

### `apfloat::is_neg_inf`

```dslx-snippet
pub fn is_neg_inf<EXP_SZ:u32, FRACTION_SZ:u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns whether or not the given `APFloat` represents a negative infinite quantity.

### `apfloat::zero`

```dslx-snippet
pub fn zero<EXP_SZ:u32, FRACTION_SZ:u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns a positive or negative zero depending upon the given sign parameter.

### `apfloat::one`

```dslx-snippet
pub fn one<EXP_SZ:u32, FRACTION_SZ:u32>(sign: bits[1]) -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns one or minus one depending upon the given sign parameter.

### `apfloat::negate`

```dslx-snippet
pub fn negate<EXP_SZ:u32, FRACTION_SZ:u32>(x: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns the negative of `x` unless it is a `NaN`, in which case it will change it
from a quiet to signaling `NaN` or from signaling to a quiet `NaN`.

### `apfloat::max_normal_exp`

```dslx-snippet
pub fn max_normal_exp<EXP_SZ : u32>() -> sN[EXP_SZ]
```

Maximum value of the exponent for normal numbers with EXP_SZ bits in the
exponent field. For single precision floats this value is 127.

### `apfloat::min_normal_exp`

```dslx-snippet
pub fn min_normal_exp<EXP_SZ : u32>() -> sN[EXP_SZ]
```

Minimum value of the exponent for normal numbers with EXP_SZ bits in the
exponent field. For single precision floats this value is -126.

### `apfloat::unbiased_exponent`

```dslx-snippet
pub fn unbiased_exponent<EXP_SZ:u32, FRACTION_SZ:u32>(f: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[EXP_SZ]
```

Returns the unbiased exponent. For normal numbers it is `bexp - 2^EXP_SZ +
1``and for subnormals it is,`2 - 2^EXP_SZ``. For infinity and `NaN``, there are
no guarantees, as the unbiased exponent has no meaning in that case.

For example, for single precision normal numbers the unbiased exponent is
`bexp - 127``and for subnormal numbers it is`-126`.

### `apfloat::bias`

```dslx-snippet
pub fn bias<EXP_SZ: u32, FRACTION_SZ: u32>(unbiased_exponent: sN[EXP_SZ]) -> bits[EXP_SZ]
```

Returns the biased exponent which is equal to `unbiased_exponent + 2^EXP_SZ - 1`

Since the function only takes as input the unbiased exponent, it cannot
distinguish between normal and subnormal numbers, as a result it assumes that
the input is the exponent for a normal number.

### `apfloat::flatten`

```dslx-snippet
pub fn flatten<EXP_SZ:u32, FRACTION_SZ:u32,
               TOTAL_SZ:u32 = {u32:1+EXP_SZ+FRACTION_SZ}>(
               x: APFloat<EXP_SZ, FRACTION_SZ>)
    -> bits[TOTAL_SZ]
```

Returns a bit string of size `1 + EXP_SZ + FRACTION_SZ` where the first bit is
the sign bit, the next `EXP_SZ` bit encode the biased exponent and the last
`FRACTION_SZ` are the significand without the hidden bit.

### `apfloat::unflatten`

```dslx-snippet
pub fn unflatten<EXP_SZ:u32, FRACTION_SZ:u32,
                 TOTAL_SZ:u32 = {u32:1+EXP_SZ+FRACTION_SZ}>(x: bits[TOTAL_SZ])
    -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns a `APFloat` struct whose flattened version would be the input string
`x`.

### `apfloat:ldexp`

```dslx-snippet
pub fn ldexp<EXP_SZ:u32, FRACTION_SZ:u32>(
             fraction: APFloat<EXP_SZ, FRACTION_SZ>,
             exp: s32) -> APFloat<EXP_SZ, FRACTION_SZ>
```

`ldexp` (load exponent) computes `fraction * 2^exp`. Note that:

 - Input denormals are treated as/flushed to 0. (denormals-are-zero / DAZ).
   Similarly, denormal results are flushed to 0.
 - No exception flags are raised/reported.
 - We emit a single, canonical representation for NaN (qnan) but accept all
   `NaN` representations as input.

### `apfloat::cast_from_fixed_using_rne`

```dslx-snippet
pub fn cast_from_fixed_using_rne<EXP_SZ:u32, FRACTION_SZ:u32, NUM_SRC_BITS:u32>(
                                 to_cast: sN[NUM_SRC_BITS])
    -> APFloat<EXP_SZ, FRACTION_SZ> {
```

Casts the fixed point number to a floating point number using RNE (Round to
Nearest Even) as the [rounding mode](https://en.wikipedia.org/wiki/Rounding).

### `apfloat::cast_from_fixed_using_rz`

```dslx-snippet
pub fn cast_from_fixed_using_rz<EXP_SZ:u32, FRACTION_SZ:u32, NUM_SRC_BITS:u32>(
                                 to_cast: sN[NUM_SRC_BITS])
    -> APFloat<EXP_SZ, FRACTION_SZ> {
```

Casts the fixed point number to a floating point number using RZ (Round to Zero)
as the [rounding mode](https://en.wikipedia.org/wiki/Rounding).

### `apfloat::upcast`

```dslx-snippet
pub fn upcast<TO_EXP_SZ: u32, TO_FRACTION_SZ: u32, FROM_EXP_SZ: u32, FROM_FRACTION_SZ: u32>
    (f: APFloat<FROM_EXP_SZ, FROM_FRACTION_SZ>) -> APFloat<TO_EXP_SZ, TO_FRACTION_SZ> {
```

Upcast the given apfloat to another (larger) apfloat representation. Note:
denormal inputs get flushed to zero.

### `apfloat::downcast_fractional_rne`

```dslx-snippet
pub fn downcast_fractional_rne<TO_FRACTION_SZ: u32, FROM_FRACTION_SZ: u32, EXP_SZ: u32>
    (f: APFloat<EXP_SZ, FROM_FRACTION_SZ>) -> APFloat<EXP_SZ, TO_FRACTION_SZ> {
```

Round the apfloat to lower precision in fractional bits, while the exponent size
remains fixed. Ties round to even (LSB = 0) and denormal inputs get flushed to
zero.

### `apfloat::normalize`

```dslx-snippet
pub fn normalize<EXP_SZ:u32, FRACTION_SZ:u32,
                 WIDE_FRACTION:u32 = {FRACTION_SZ + u32:1}>(
                 sign: bits[1], exp: bits[EXP_SZ],
                 fraction_with_hidden: bits[WIDE_FRACTION])
    -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns a normalized APFloat with the given components. `fraction_with_hidden`
is the fraction (including the hidden bit). This function only normalizes in the
direction of decreasing the exponent. Input must be a normal number or zero.
Subnormals/Denormals are flushed to zero in the result.

### `apfloat::is_zero_or_subnormal`

```dslx-snippet
pub fn is_zero_or_subnormal<EXP_SZ: u32, FRACTION_SZ: u32>(
                            x: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns `true` if `x == 0` or `x` is a subnormal number.

### `apfloat::cast_to_fixed`

```dslx-snippet
pub fn cast_to_fixed<NUM_DST_BITS:u32, EXP_SZ:u32, FRACTION_SZ:u32>(
                     to_cast: APFloat<EXP_SZ, FRACTION_SZ>)
    -> sN[NUM_DST_BITS]
```

Casts the floating point number to a fixed point number. Unrepresentable numbers
are cast to the minimum representable number (largest magnitude negative
number).

### `apfloat::eq_2`

```dslx-snippet
pub fn eq_2<EXP_SZ: u32, FRACTION_SZ: u32>(
            x: APFloat<EXP_SZ, FRACTION_SZ>,
            y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns `true` if `x == y`. Denormals are Zero (DAZ). Always returns `false` if
`x` or `y` is `NaN`.

### `apfloat::gt_2`

```dslx-snippet
pub fn gt_2<EXP_SZ: u32, FRACTION_SZ: u32>(
            x: APFloat<EXP_SZ, FRACTION_SZ>,
            y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns `true` if `x > y`. Denormals are Zero (DAZ). Always returns `false` if
`x` or `y` is `NaN`.

### `apfloat::gte_2`

```dslx-snippet
pub fn gte_2<EXP_SZ: u32, FRACTION_SZ: u32>(
             x: APFloat<EXP_SZ, FRACTION_SZ>,
             y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns `true` if `x >= y`. Denormals are Zero (DAZ). Always returns `false` if
`x` or `y` is `NaN`.

### `apfloat::lte_2`

```dslx-snippet
pub fn lte_2<EXP_SZ: u32, FRACTION_SZ: u32>(
             x: APFloat<EXP_SZ, FRACTION_SZ>,
             y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns `true` if `x <= y`. Denormals are Zero (DAZ). Always returns `false` if
`x` or `y` is `NaN`.

### `apfloat::lt_2`

```dslx-snippet
pub fn lt_2<EXP_SZ: u32, FRACTION_SZ: u32>(
            x: APFloat<EXP_SZ, FRACTION_SZ>,
            y: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

Returns `true` if `x < y`. Denormals are Zero (DAZ). Always returns `false` if
`x` or `y` is `NaN`.

### `apfloat::to_int`

```dslx-snippet
pub fn to_int<EXP_SZ: u32, FRACTION_SZ: u32, RESULT_SZ:u32>(
              x: APFloat<EXP_SZ, FRACTION_SZ>) -> sN[RESULT_SZ]
```

Returns the signed integer part of the input float, truncating any fractional
bits if necessary.

Exceptional cases:

X operand                          | `sN[RESULT_SZ]` value
---------------------------------- | -----------------------
`NaN`                              | `sN[RESULT_SZ]::ZERO`
`+Inf`                             | `sN[RESULT_SZ]::MAX`
`-Inf`                             | `sN[RESULT_SZ]::MIN`
+0.0, -0.0 or any subnormal number | `sN[RESULT_SZ]::ZERO`
`> sN[RESULT_SZ]::MAX`             | `sN[RESULT_SZ]::MAX`
`< sN[RESULT_SZ]::MIN`             | `sN[RESULT_SZ]::MIN`

### `apfloat::to_uint`

```dslx-snippet
pub fn to_uint<RESULT_SZ:u32, EXP_SZ: u32, FRACTION_SZ: u32>(
               x: APFloat<EXP_SZ, FRACTION_SZ>) -> uN[RESULT_SZ]
```

Casts the input float to the nearest unsigned integer. Any fractional bits are
truncated and negative floats are clamped to 0.

Exceptional cases:

X operand                          | `uN[RESULT_SZ]` value
---------------------------------- | ---------------------
`NaN`                              | `uN[RESULT_SZ]::ZERO`
`+Inf`                             | `uN[RESULT_SZ]::MAX`
`-Inf`                             | `uN[RESULT_SZ]::ZERO`
+0.0, -0.0 or any subnormal number | `uN[RESULT_SZ]::ZERO`
`> uN[RESULT_SZ]::MAX`             | `uN[RESULT_SZ]::MAX`
`< uN[RESULT_SZ]::ZERO`            | `uN[RESULT_SZ]::ZERO`

### `apfloat::add/sub`

```dslx-snippet
pub fn add<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>,
                                          y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ>

pub fn sub<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>,
                                          y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns the sum/difference of `x` and `y` based on a generalization of IEEE 754
single-precision floating-point addition, with the following exceptions:

-   Both input and output denormals are treated as/flushed to 0.
-   Only round-to-nearest mode is supported.
-   No exception flags are raised/reported.

In all other cases, results should be identical to other conforming
implementations (modulo exact fraction values in the `NaN` case.

### `apfloat::has_fractional_part`

Returns whether or not the given APFloat has a fractional part.

```dslx-snippet
pub fn has_fractional_part<EXP_SZ: u32, FRACTION_SZ: u32>(f: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

### `has_negative_exponent`

Returns whether or not the given APFloat has an negative exponent.

```dslx-snippet
pub fn has_negative_exponent<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> bool
```

### `apfloat::ceil`

```dslx-snippet
pub fn ceil<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns the nearest integral `APFloat` of the same precision as `f` whose value
is greater than or equal to `f`.

### `apfloat::floor`

```dslx-snippet
pub fn floor<EXP_SZ: u32, FRACTION_SZ: u32>
    (f: APFloat<EXP_SZ, FRACTION_SZ>) -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns the nearest integral `APFloat` of the same precision as `f` whose value
is lesser than or equal to `f`.

### `apfloat::trunc`

```dslx-snippet
pub fn trunc<EXP_SZ:u32, FRACTION_SZ:u32>(
                          x: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns an `APFloat` of the same precision as `f` with all the fractional bits
set to `0`.

#### Implementation details

Floating-point addition, like any FP operation, is much more complicated than
integer addition, and has many more steps. Being the first operation described,
we'll take extra care to explain floating-point addition:

<!-- mdformat off(nested lists are rendered differently in mkdocs) -->

1. **Expand fractions:** Floating-point operations are computed with bits
   beyond that in their normal representations for increased precision. For
   IEEE 754 numbers, there are three extra, called the guard, rounding and
   sticky bits. The first two behave normally, but the last, the "sticky" bit,
   is special. During shift operations (below), if a "1" value is ever shifted
   into the sticky bit, it "sticks" - the bit will remain "1" through any
   further shift operations. In this step, the fractions are expanded by these
   three bits.
1. **Align fractions:** To ensure that fractions are added with appropriate
   magnitudes, they must be aligned according to their exponents. To do so, the
   smaller significant needs to be shifted to the right (each right shift is
   equivalent to increasing the exponent by one).

   * The extra precision bits are populated in this shift.
   * As part of this step, the leading 1 bit... and a sign bit Note: The
     sticky bit is calculated and applied in this step.

1. **Sign-adjustment:** if the fractions differ in sign, then the fraction with
   the smaller initial exponent needs to be (two's complement) negated.

1. **Add the fractions and capture the carry bit.** Note that, if the signs of
   the fractions differs, then this could result in higher bits being cleared.

1. **Normalize the fractions:** Shift the result so that the leading '1' is
   present in the proper space. This means shifting right one place if the
   result set the carry bit, and to the left some number of places if high bits
   were cleared.

   * The sticky bit must be preserved in any of these shifts!

1. **Rounding:** Here, the extra precision bits are examined to determine if
   the result fraction's last bit should be rounded up. IEEE 754 supports five
   rounding modes:

   * Round towards 0: just chop off the extra precision bits.
   * Round towards +infinity: round up if any extra precision bits are set.
   * Round towards -infinity: round down if any extra precision bits are set.
   * Round to nearest, ties away from zero: Rounds to the nearest value. In
     cases where the extra precision bits are halfway between values, i.e.,
     0b100, then the result is rounded up for positive numbers and down for
     negative ones.
   * Round to nearest, ties to even: Rounds to the nearest value. In cases
     where the extra precision bits are halfway between values, then the
     result is rounded in whichever direction causes the LSB of the result
     significant to be 0.

     * This is the most commonly-used rounding mode.
     * This is [currently] the only supported mode by the DSLX implementation.

1. **Special case handling:** The results are examined for special cases such
   as NaNs, infinities, or (optionally) subnormals.

<!-- mdformat on -->

##### Result sign determination

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

##### Rounding

As complicated as rounding is to describe, its implementation is relatively
straightforward (types are as for a C `float` implementation):

```dslx-snippet
  let normal_chunk = shifted_fraction[0:3];
  let half_way_chunk = shifted_fraction[2:4];
  let do_round_up =
      if (normal_chunk > u3:0x4) | (half_way_chunk == u2:0x3) { u1:1 }
      else { u1:0 };

  // We again need an extra bit for carry.
  let rounded_fraction = if do_round_up { (shifted_fraction as u28) + u28:0x8 }
                         else { shifted_fraction as u28 };
  let rounding_carry = rounded_fraction[-1:];
```

### `apfloat::mul`

```dslx-snippet
pub fn mul<EXP_SZ: u32, FRACTION_SZ: u32>(x: APFloat<EXP_SZ, FRACTION_SZ>,
                                          y: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ>
```

Returns the product of `x` and `y`, with the following exceptions:

-   Both input and output denormals are treated as/flushed to 0.
-   Only round-to-nearest mode is supported.
-   No exception flags are raised/reported.

In all other cases, results should be identical to other conforming
implementations (modulo exact fraction values in the NaN case).

### `apfloat::fma`

```dslx-snippet
pub fn fma<EXP_SZ: u32, FRACTION_SZ: u32>(a: APFloat<EXP_SZ, FRACTION_SZ>,
                                          b: APFloat<EXP_SZ, FRACTION_SZ>,
                                          c: APFloat<EXP_SZ, FRACTION_SZ>)
    -> APFloat<EXP_SZ, FRACTION_SZ> {
```

Returns the Fused Multiply and Add (FMA) result of computing `a*b + c`.

The FMA operation is a three-operand operation that computes the product of the
first two and the sum of that with the third. The IEEE 754-2008 description of
the operation states that the operation should be performed "as if with
unbounded range and precision", limited only by rounding of the final result. In
other words, this differs from a sequence of a separate multiply followed by an
add in that there is only a single rounding step (instead of the two involved in
separate operations).

In practice, this means A) that the precision of an FMA is higher than
individual ops, and thus that B) an FMA requires significantly more internal
precision bits than naively expected.

For binary32 inputs, to achieve the standard-specified precision, the initial
mul requires the usual 48 ((23 fraction + 1 "hidden") * 2) fraction bits. When
performing the subsequent add step, though, it is necessary to maintain *72*
fraction bits ((23 fraction + 1 "hidden") * 3). Fortunately, this sum includes
the guard, round, and sticky bits (so we don't need 75). The mathematical
derivation of the exact amount will not be given here (as I've not done it), but
the same calculated size would apply for other data types (i.e., 54 * 2 = 108
and 54 * 3 = 162 for binary64).

Aside from determining the necessary precision bits, the FMA implementation is
rather straightforward, especially after reviewing the adder and multiplier.

## float64

To help with the `float64` or
[double precision](https://en.wikipedia.org/wiki/Double-precision_floating-point_format)
floating point numbers, `float32.x` defines the following type aliases.

```dslx-snippet
pub type F64 = apfloat::APFloat<11, 52>;
pub type FloatTag = apfloat::APFloatTag;
pub type TaggedF64 = (FloatTag, F64);
```

Besides `float64` specializations of the functions in `apfloat.x`, the following
functions are defined just for `float64`.

### `float64::to_int64`

```dslx-snippet
pub fn to_int64(x: F64) -> s64;
```

Convert the `F64` struct into a 64 bit integer.

## float32

To help with the `float32` or
[single precision](https://en.wikipedia.org/wiki/Single-precision_floating-point_format)
floating point numbers, `float32.x` defines the following type aliases.

```dslx-snippet
pub type F32 = apfloat::APFloat<8, 23>;
pub type FloatTag = apfloat::APFloatTag;
pub type TaggedF32 = (FloatTag, F32);
pub const F32_ONE_FLAT = u32:0x3f800000;
```

Besides `float32` specializations of the functions in `apfloat.x`, the following
functions are defined just for `float32`.

### `float32::to_int32`, `float32::to_uint32`, `float32::from_int32`

```dslx-snippet
pub fn to_int32(x: F32) -> s32
pub fn to_uint32(x: F32) -> u32
pub fn from_int32(x: s32) -> F32
```

Convert the `F32` struct to a 32 bit signed/unsigned integer, or from a 32 bit
signed integer to an `F32`.

# `float32::fixed_fraction`
```dslx-snippet
pub fn fixed_fraction(input_float: F32) -> u23
```

TBD

# `float32::fast_rsqrt_config_refinements`/`float32::fast_rsqrt`
```dslx-snippet
pub fn fast_rsqrt_config_refinements<NUM_REFINEMENTS: u32 = {u32:1}>(x: F32) -> F32
pub fn fast_rsqrt(x: F32) -> F32
```

Floating point (fast (approximate) inverse square root)[
https://en.wikipedia.org/wiki/Fast_inverse_square_root]. This should be able to
compute `1.0 / sqrt(x)` using fewer hardware resources than using a `sqrt` and
division module, although this hasn't been benchmarked yet. Latency is expected
to be lower as well. The tradeoff is that this offers slighlty less precision
(error is < 0.2% in worst case). Note that:

 - Input denormals are treated as/flushed to 0. (denormals-are-zero / DAZ).
 - Only round-to-nearest mode is supported.
 - No exception flags are raised/reported.
 - We emit a single, canonical representation for NaN (qnan) but accept
   all NaN respresentations as input.

`fast_rsqrt_config_refinements` allows the user to specify the number of
Computes an approximation of 1.0 / sqrt(x). `NUM_REFINEMENTS` can be increased
to tradeoff more hardware resources for more accuracy.

`fast_rsqrt` does exactly one refinement.

## bfloat16

To help with the
[`bfloat16`](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
floating point numbers, `bfloat16.x` defines the following type aliases.

```dslx-snippet
pub type BF16 = apfloat::APFloat<8, 7>;
pub type FloatTag = apfloat::APFloatTag;
pub type TaggedBF16 = (FloatTag, BF16);
```
Besides `bfloat16` specializations of the functions in `apfloat.x`, the following
functions are defined just for `bfloat16`.

### `bfloat16:to_int16`, `bfloat16:to_uint16`

```dslx-snippet
pub fn to_int16(x: BF16) -> s16
pub fn to_uint16(x: BF16) -> u16
```

Convert the `BF16` struct to a 16 bit signed/unsigned integer.

### `bfloat16::from_int8`
```dslx-snippet
pub fn from_int8(x: s8) -> BF16
```

Converts the given signed integer to bfloat16. For s8, all values can be
captured exactly, so no need to round or handle overflow.

### `bfloat16::from_float32`
```dslx-snippet
pub fn from_float32(x: F32) -> BF16
```

Converts the given float32 to bfloat16. Ties round to even (LSB = 0) and
denormal inputs get flushed to zero.

### `bfloat16:increment_fraction`
```dslx-snippet
pub fn increment_fraction(input: BF16) -> BF16
```

Increments the fraction of the input BF16 by one and returns the normalized
result. Input must be a normal *non-zero* number.

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
implementations will give the same results for the same inputs. [^2] In
addition, POSIX has the same result-precision language. It's worth noting that
-ffast-math doesn't currently affect FMA emission/fusion/fission/etc.

[^2]: There are operations for which this is not true. Transcendental ops may

differ between implementations due to the
[*table maker's dilemma*](https://en.wikipedia.org/wiki/Rounding).

### Exhaustive testing

This is the happiest case - where the input space is so small that we can
iterate over every possible input, effectively treating the input as a binary
iteration counter. Sadly, this is uncommon (except, perhaps for ML math), as
binary32 is the precision floor for most problems, and a 64-bit input space is
well beyond our current abilities. Still - if your problem *can* be exhaustively
tested (with respect to a trusted reference), it *should* be exhaustively
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
