# Tutorial: parametric types and functions

This tutorial demonstrates how types and functions can be parameterized to
enable them to work on data of different formats and layouts, e.g., for a
function `foo` to work on both u16 and u32 data types, and anywhere in between.

It's recommended that you're familiar with the concepts in the previous
tutorial,
"[float-to-int conversion"](../tutorials/float_to_int.md)
before following this tutorial.

## Simple parametrics

Consider the simple example of the `umax` function
[in the DSLX standard library](https://github.com/google/xls/tree/main/xls/dslx/stdlib/std.x):

```dslx
pub fn umax<N: u32>(x: uN[N], y: uN[N]) -> uN[N] {
  if x > y { x } else { y }
}
```

Most of this function looks like other DSLX functions you may have seen, except
for the new-style parameter, `N`. The declaration of `N` inside angle brackets
denotes that `N` is a _parametric value_ whose value is a build-time invariant
that will be specified by the caller. In other words, changing regular function
parameters pumps different values through the circuit, while changing parametric
values changes the circuit itself.

Here, `N` is used to define the widths of the input and output types. It's plain
to see, then, that specifying `N = 16` would calculate the maximum of two 16-bit
numbers, whereas `N = 273` would calculate the maximum of two 273-bit numbers.
That being said, the smaller the circuit, the faster, smaller, and lower-power
it will be, so `N` should be as small as possible (but no smaller!).

There are two ways invoke a parametric function: the first is to explicitly
specify all parametric values, and the second is to rely on the language to
infer them:

Explicit specification:

```dslx
import std;

fn foo(a: u32, b: u16) -> u64 {
  std::umax<u32:64>(a as u64, b as u64)
}
```

Here, the user has directly told the language what the values of all parametrics
are.

Parametric inference:

```dslx
import std;

fn foo(a: u32, b: u16) -> u64 {
  std::umax(a as u64, b as u64)
}
```

Here, though, the language is able to determine that `N` is 64, since that
matches the types of the arguments to `umax`, and since both arg types agree.
There may be times where inference isn't possible - for example, when there
exist parametrics that aren't referenced in the argument list:

```dslx
fn my_parametric_sum<N: u32>(a: u32, b: u32) -> uN[N] {
  let a_mod = a as uN[N];
  let b_mod = a as uN[N];
  a_mod + b_mod
}
```

To invoke this function, explicit specification is required.

## Derived parametrics

It's common, when using parametric types, to need types _similar, but not
identical to_ the parametric type. Consider calculating the unbiased
floating-point exponent (from the previous tutorial): while the _biased_
exponent was 8 bits wide, the calculated _unbiased_ exponent was 9 bits wide due
to the additional sign bit. In this situation, if `EXP_SZ` was 8, then it'd be
handy to also have a `SIGNED_EXP_SZ` symbol that was equal to 9. This can be
done as follows:

```dslx-snippet
fn unbias_exponent<EXP_SZ: u32, SIGNED_EXP_SZ: u32 = EXP_SZ + u32:1>(
    exp: uN[EXP_SZ]) -> sN[SIGNED_EXP_SZ] {
  exp as sN[SIGNED_EXP_SZ] - sN[SIGNED_EXP_SZ]:???
}
```

Oh no! Specifying parametrics in this way has revealed a problem! If we
parameterize types, then in some situations, we'll need to also parameterize
*values*!

Of course, we'd not be writing this tutorial if that wasn't possible. DSLX
supports "constexpr"-style evaluation, whereby constant expressions can be
evaluated at interpretation or compilation time. In this case, we just need an
expression that can calculate the correct bias adjustment: `(sN[SIGNED_EXP_SZ]:1
<< (EXP_SZ - u32:1)) - sN[SIGNED_EXP_SZ]:1`

This is a bit unwieldy in practice, so we can wrap it in a function:

```dslx
fn bias_scaler<N: u32, WIDE_N: u32 = {N + u32:1}>() -> sN[WIDE_N] {
  (sN[WIDE_N]:1 << (N - u32:1)) - sN[WIDE_N]:1
}

fn unbias_exponent<EXP_SZ: u32, SIGNED_EXP_SZ: u32 = {EXP_SZ + u32:1}>(
    exp: uN[EXP_SZ]) -> sN[SIGNED_EXP_SZ] {
  exp as sN[SIGNED_EXP_SZ] - bias_scaler<EXP_SZ>()
}
```

## Parameterized float-to-int

Finally, consider the 32-bit float-to-int program from the previous tutorial.
That program was restricted to converting from one specific type to another. If,
however, we wanted to convert from, say a `double` to an `int32_t`, we'd have to
write a new function, even though the basic logic would be the same.

Instead, armed with parametrics, we can write a single function to handle *all*
such conversions - even to floating-point formats we haven't considered!

The first step in such a parameterization is to have a working single-typed
example, which we take from the previous codelab:

```dslx
pub struct float32 {
  sign: u1,
  bexp: u8,
  fraction: u23,
}

fn unbias_exponent(exp: u8) -> s9 {
  exp as s9 - s9:127
}

pub fn float_to_int(x: float32) -> s32 {
  let exp = unbias_exponent(x.bexp);

  // Add the implicit leading one.
  // Note that we need to add one bit to the fraction to hold it.
  let fraction = u33:1 << 23 | (x.fraction as u33);

  // Shift the result to the right if the exponent is less than 23.
  let fraction =
      if (exp as u8) < u8:23 { fraction >> (u8:23 - (exp as u8)) }
      else { fraction };

  // Shift the result to the left if the exponent is greater than 23.
  let fraction =
      if (exp as u8) > u8:23 { fraction << ((exp as u8) - u8:23) }
      else { fraction };

  let result = fraction as s32;
  let result = if x.sign { -result } else { result };
  result
}
```

Next is to identify all types needing parameterization, here being the intended
size of the result and the layout of the floating-point type itself; all other
types flow from that base definition:

*   `exp`: `float32::bexp` size + 1 sign bit
*   `fraction`: `float32::fraction` size + 1 implicit leading bit

Thus, the struct declaration and function signature will be:

```dslx-snippet
pub struct float<EXP_SZ: u32, FRACTION_SZ: u32> {
  sign: u1,
  bexp: uN[EXP_SZ],
  fraction: uN[FRACTION_SZ],
}

pub fn float_to_int<EXP_SZ: u32, FRACTION_SZ: u32, RESULT_SZ: u32>(
    x: float<EXP_SZ, FRACTION_SZ>) -> sN[RESULT_SZ] {
  ...
}
```

From there, the rest of the function can be populated by replacing the types in
the original implementation with the parameterized ones in the signature:

```dslx
pub struct float<EXP_SZ: u32, FRACTION_SZ: u32> {
  sign: u1,
  bexp: uN[EXP_SZ],
  fraction: uN[FRACTION_SZ],
}

fn bias_scaler<N: u32, WIDE_N: u32 = {N + u32:1}>() -> sN[WIDE_N] {
  (sN[WIDE_N]:1 << (N - u32:1)) - sN[WIDE_N]:1
}

fn unbias_exponent<EXP_SZ: u32, SIGNED_EXP_SZ: u32 = {EXP_SZ + u32:1}>(
    exp: uN[EXP_SZ]) -> sN[SIGNED_EXP_SZ] {
  exp as sN[SIGNED_EXP_SZ] - bias_scaler<EXP_SZ>()
}

pub fn float_to_int<
    EXP_SZ: u32, FRACTION_SZ: u32, RESULT_SZ: u32,
    WIDE_EXP_SZ: u32 = {EXP_SZ + u32:1},
    WIDE_FRACTION_SZ: u32 = {FRACTION_SZ + u32:1}>(
    x: float<EXP_SZ, FRACTION_SZ>) -> sN[RESULT_SZ] {
  let exp = unbias_exponent(x.bexp);

  let fraction = uN[WIDE_FRACTION_SZ]:1 << FRACTION_SZ |
      (x.fraction as uN[WIDE_FRACTION_SZ]);

  let fraction =
      if (exp as u32) < FRACTION_SZ { fraction >> (FRACTION_SZ - (exp as u32)) }
      else { fraction };

  let fraction =
      if (exp as u32) > FRACTION_SZ { fraction << ((exp as u32) - FRACTION_SZ) }
      else { fraction };

  let result = fraction as sN[RESULT_SZ];
  let result = if x.sign { -result } else { result };
  result
}
```

Note that `unbias_exponent()` didn't need type specification, since the type
could be inferred from the argument! (Also note that this implementation doesn't
contain the fixes from the missing cases from the previous tutorial. Exercise to
the reader: apply those fixes here, too!)

This technique underlies all of XLS' floating-point libraries. Common operations
are defined in common files, such as
[apfloat.x](https://github.com/google/xls/tree/main/xls/dslx/stdlib/apfloat.x) (general
utilities). Specializations of the above are then available in, e.g.,
[float32.x](https://github.com/google/xls/tree/main/xls/stdlib/float32.x) to hide internal
implementation details from end users.

With this technique, you can write single implementations of functionality that
can be applicable across all sorts of hardware configurations for minimal
additional cost. Try it out! Create an `0xbeef`-bit wide floating-point adder!
