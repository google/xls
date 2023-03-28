# Standard Library

This page documents the DSLX standard library.

[TOC]

# `std.x`

## Bits Type Properties

### `std::?_min_value`

```dslx-snippet
pub fn unsigned_min_value<N: u32>() -> uN[N]
pub fn signed_min_value<N: u32>() -> sN[N]
```

Returns the minimum signed or unsigned value contained in N bits.

### `std::?_max_value`

```dslx-snippet
pub fn unsigned_max_value<N: u32>() -> uN[N];
pub fn signed_max_value<N: u32>() -> sN[N];
```

Returns the maximum signed or unsigned value contained in N bits.

### `std::sizeof_?`

```dslx-snippet
pub fn sizeof_unsigned<N: u32>(x : uN[N]) -> u32
pub fn sizeof_signed<N: u32>(x : sN[N]) -> u32
```

Returns the number of bits (sizeof) of unsigned or signed bit value.

## Bit Manipulation Functions

### `std::lsb`

```dslx-snippet
pub fn lsb<N: u32>(x: uN[N]) -> u1
```

Extracts the LSB (Least Significant Bit) from the value `x` and returns it.

### `std::convert_to_bits_msb0`

```dslx-snippet
pub fn convert_to_bits_msb0<N: u32>(x: bool[N]) -> uN[N]
```

Converts an array of `N` bools to a `bits[N]` value.

**Note well:** the boolean value at **index 0** of the array becomes the **most
significant bit** in the resulting bit value. Similarly, the last index of the
array becomes the **least significant bit** in the resulting bit value.

```dslx
import std

#[test]
fn convert_to_bits_test() {
  let _ = assert_eq(u3:0b001, std::convert_to_bits(bool[3]:[false, false, true]));
  let _ = assert_eq(u3:0b100, std::convert_to_bits(bool[3]:[true, false, false]));
  ()
}
```

There's always a source of confusion in these orderings:

* Mathematically we often indicate the least significant digit as "digit 0"
* *But*, in a number as we write the digits from left-to-right on a piece of
  paper, if you made an array from the written characters, the digit at "array
  index 0" would be the most significant bit.

So, it's somewhat ambiguous whether "index 0" in the array would become the
least significant bit or the most significant bit. This routine uses the "as it
looks on paper" conversion; e.g. `[true, false, false]` becomes `0b100`.

### `std::convert_to_bools_lsb0`

```dslx-snippet
pub fn fn convert_to_bools_lsb0<N:u32>(x: uN[N]) -> bool[N]
```

Convert a "word" of bits to a corresponding array of booleans.

**Note well:** The least significant bit of the word becomes index 0 in the
array.

### `std::mask_bits`

```dslx-snippet
pub fn mask_bits<X: u32>() -> bits[X]
```

Returns a value with X bits set (of type bits[X]).

### `std::concat3`

```dslx-snippet
pub fn concat3<X: u32, Y: u32, Z: u32, R: u32 = X + Y + Z>(x: bits[X], y: bits[Y], z: bits[Z]) -> bits[R]
```

Concatenates 3 values of arbitrary bitwidths to a single value.

### `std::rrot`

```dslx-snippet
pub fn rrot<N: u32>(x: bits[N], y: bits[N]) -> bits[N]
```

Rotate `x` right by `y` bits.

### `std::popcount`

```dslx-snippet
pub fn popcount<N: u32>(x: bits[N]) -> bits[N]
```

Counts the number of bits in `x` that are '1'.

## Mathematical Functions

### `std::bounded_minus_1`

```dslx-snippet
pub fn bounded_minus_1<N: u32>(x: uN[N]) -> uN[N]
```

Returns the value of `x - 1` with saturation at `0`.

### `std::abs`

```dslx-snippet
pub fn abs<BITS: u32>(x: sN[BITS]) -> sN[BITS]
```

Returns the absolute value of `x` as a signed number.

### `std::is_pow2`

```dslx-snippet
pub fn is_pow2<N: u32>(x: uN[N]) -> bool
```

Returns true when x is a non-zero power-of-two.

### `std::?mul`

```dslx-snippet
pub fn umul<N: u32, M: u32, R: u32 = N + M>(x: uN[N], y: uN[M]) -> uN[R]
pub fn smul<N: u32, M: u32, R: u32 = N + M>(x: sN[N], y: sN[M]) -> sN[R]
```

Returns product of `x` (`N` bits) and `y` (`M` bits) as an `N+M` bit value.

### `std::iterative_div`

```dslx-snippet
pub fn iterative_div<N: u32, DN: u32 = N * u32:2>(x: uN[N], y: uN[N]) -> uN[N]
```

Calculate `x / y` one bit at a time. This is an alternative to using the
division operator '/' which may not synthesize nicely.

### `std::div_pow2`

```dslx-snippet
pub fn div_pow2<N: u32>(x: bits[N], y: bits[N]) -> bits[N]
```

Returns `x / y` where `y` must be a non-zero power-of-two.

### `std::mod_pow2`

```dslx-snippet
pub fn mod_pow2<N: u32>(x: bits[N], y: bits[N]) -> bits[N]
```

Returns `x % y` where `y` must be a non-zero power-of-two.

### `std::ceil_div`

```dslx-snippet
pub fn ceil_div<N: u32>(x: uN[N], y: uN[N]) -> uN[N]
```

Returns the ceiling of (x divided by y).

### `std::round_up_to_nearest`

```
pub fn round_up_to_nearest(x: u32, y: u32) -> u32
```

Returns `x` rounded up to the nearest multiple of `y`.

### `std::?pow`

```dslx-snippet
pub fn upow<N: u32>(x: uN[N], n: uN[N]) -> uN[N]
pub fn spow<N: u32>(x: sN[N], n: uN[N]) -> sN[N]
```

Performs integer exponentiation as in Hacker's Delight, Section 11-3. Only
non-negative exponents are allowed, hence the uN parameter for spow.

### `std::clog2`

```dslx-snippet
pub fn clog2<N: u32>(x: bits[N]) -> bits[N]
```

Returns `ceiling(log2(x))`, with one exception: When `x = 0`, this function
differs from the true mathematical function: `clog2(0) = 0` where as
`ceil(log2(0)) = -infinity`

This function is frequently used to calculate the number of bits required to
represent `x` possibilities. With this interpretation, it is sensible to define
`clog2(0) = 0`.

Example: `clog2(7) = 3`.

### `std:flog2`

```dslx-snippet
pub fn flog2<N: u32>(x: bits[N]) -> bits[N]
```

Returns `floor(log2(x))`, with one exception:

When x=0, this function differs from the true mathematical function: `flog2(0) =
0` where as `floor(log2(0)) = -infinity`

This function is frequently used to calculate the number of bits required to
represent an unsigned integer `n` to define `flog2(0) = 0`, so that `flog(n)+1`
represents the number of bits needed to represent the `n`.

Example: `flog2(7) = 2`, `flog2(8) = 3`.

### `std::?max`

```dslx-snippet
pub fn smax<N: u32>(x: sN[N], y: sN[N]) -> sN[N]
pub fn umax<N: u32>(x: uN[N], y: uN[N]) -> uN[N]
```

Returns the maximum of two integers.

### `std::umin`

```dslx-snippet
pub fn umin<N: u32>(x: uN[N], y: uN[N]) -> uN[N]
```

Returns the minimum of two unsigned integers.

## Misc Functions

### `Signed comparison - std::{sge, sgt, sle, slt}`

```dslx-snippet
pub fn sge<N: u32>(x: uN[N], y: uN[N]) -> bool
pub fn sgt<N: u32>(x: uN[N], y: uN[N]) -> bool
pub fn sle<N: u32>(x: uN[N], y: uN[N]) -> bool
pub fn slt<N: u32>(x: uN[N], y: uN[N]) -> bool
```

**Explicit signed comparison** helpers for working with unsigned values, can be
a bit more convenient and a bit more explicit intent than doing casting of left
hand side and right hand side.

### `std::find_index`

```dslx-snippet
pub fn find_index<BITS: u32, ELEMS: u32>( array: uN[BITS][ELEMS], x: uN[BITS]) -> (bool, u32)
```

Returns (`found`, `index`) given an array and the element to find within the
array.

Note that when `found` is false, the `index` is `0` -- `0` is provided instead
of a value like `-1` to prevent out-of-bounds accesses from occurring if the
index is used in a match expression (which will eagerly evaluate all of its
arms), to prevent it from creating an error at simulation time if the value is
ultimately discarded from the unselected match arm.

# `acm_random.x`

Port of
[ACM random](https://github.com/google/or-tools/blob/66b8d230798f9b8a3c98c26a997daf04974400b8/ortools/base/random.cc#L35)
number generator to DSLX.

DO NOT use `acm_random.x` for any application where security -- unpredictability
of subsequent output and previous output -- is needed. ACMRandom is in *NO*
*WAY* a cryptographically secure pseudorandom number generator, and using it
where recipients of its output may wish to guess earlier/later output values
would be very bad.

## `acm_random::rng_deterministic_seed`

```dslx-snippet
pub fn rng_deterministic_seed() -> u32
```

Returns a fixed seed for use in the random number generator.

## `acm_random::rng_new`

```dslx-snippet
pub fn rng_new(seed: u32) -> State
```

Create the state for a new random number generator using the given seed.

## `acm_random::rng_next`

```dslx-snippet
pub fn rng_next(s: State) -> (State, u32)
```

Returns a pseudo-random number in the range `[1, 2^31-2]`.

Note that this is one number short on both ends of the full range of
non-negative 32-bit integers, which range from `0` to `2^31-1`.

## `acm_random::rng_next64`

```dslx-snippet
pub fn rng_next(s: State) -> (State, u64)
```

Returns a pseudo random number in the range `[1, (2^31-2)^2]`.

Note that this does not cover all non-negative values of int64, which range from
`0` to `2^63-1`. **The top two bits are ALWAYS ZERO**.
