# Tutorial: basic logic

This tutorial demonstrates how to use XLS to create a simple combinational
module, in this case one that performs floating-point-to-integer conversion.

The first task is to define the full semantics of the module. We wish for the
module to accept a
[IEEE-754 floating-point number](https://en.wikipedia.org/wiki/IEEE_754) and to
output the integer representing the same. All fractional elements will be
discarded. Overflow, NaN, and infinite values will be clamped to the maximum or
minimum representable integer value with the sign of the input number.

[TOC]

## 1. Bootstrapping

When creating a module, one first needs to define the signature and skeleton of
the entry function. If you recall, a IEEE binary32 (the C `float` type) has 1
sign bit, 8 [biased] exponent bits, and 23 fractional bits. These values can be
packed into a tuple, and so, the signature of our function can be defined as:

```dslx
pub fn float_to_int(
    x: (u1, u8, u23))
    -> s32 {
  s32:0xbeef
}
```

DSLX syntax is intended to follow Rust syntax as much as possible, so this may
look familiar if you're a Rustacean. In any case, let's walk through this code,
line-by-line:

1.  This line declares a public function (`pub fn`), named `float_to_int`. Since
    this function is "public", it can be referenced from other modules (i.e.,
    files) if
    [imported](../dslx_reference.md#imports)
    therein.
2.  Function parameter declarations! This function only takes one parameter,
    `x`, whose type follows its name. In this case, it's a *`tuple`*: a grouping
    of potentially disparate elements into a single quantity. A tuple is
    specified by listing a set of types in parentheses, as here. Our tuple has a
    1-bit element for the sign, an 8-bit element for the biased exponent, and a
    23-bit element for the fractional part. In what is the complete opposite of
    a coincidence, these fields match those of an IEEE float32 number. If a
    function takes more than one argument, they'll be comma-separated.
    -   `u1`, `u8`, and `u23` are all shortcuts for the type `uN[1]`, `uN[8]`,
        and `uN[23]`. The `uN[X]` construct declares an X-bit wide unsigned
        type. There is also `sN[X]`, which declares an X-bit wide *signed* type.
    -   Other type shortcuts exist such as `bits[X]` (alias for `uN[X]`), `bool`
        (alias for `uN[1]`), and `u[1-64]` and `s[1-64]`, being aliases for
        `uN[1]` through `uN[64]` and `sN[1]` through `sN[64]`.
3.  Function return type. This function returns a signed 32-bit type, matching
    the intentions of float-to-int conversion (since floats are signed).
4.  Finally, the last line: the final statement in a function is its return
    value. Here, we're unconditionally returning a signed 32-bit number with the
    value `48879`. This is only temporary to make the function syntactically
    valid - we're still learning the basics! Gimme a second!

As an aside, it's a good idea to keep a bookmark to the
[DSLX language reference](../dslx_reference.md)
handy. It has the full details on language features and syntax and even we XLS
devs frequently reference it.

Anyway...the tuple representation of our input is a bit cumbersome, so let's
define our floating-point number as a struct instead:

```dslx
pub struct float32 {
  sign: u1,
  bexp: u8,
  fraction: u23,
}

pub fn float_to_int(x: float32) -> s32 {
  s32:0xbeef
}
```

Finally, let's write a quick test to make sure things work. Add the following
code to your file.

```dslx-snippet
#[test]
fn float_to_int_test() {
  // 0xbeef in float32.
  let test_input = float32 {
    sign: u1:0x0,
    bexp: u8:0x8e,
    fraction: u23:0x3eef00
  };
  assert_eq(s32:0xbeef, float_to_int(test_input))
}
```

Now run the test through the DSLX interpreter:

```
$ ./bazel-bin/xls/dslx/interpreter_main float_to_int.x
```

You should see something like the following:

```
[ RUN UNITTEST  ] float_to_int_test
[            OK ]
[===============] 1 test(s) ran; 0 failed; 0 skipped.
```

If so, then congrats! You've written - and tested - your first DSLX module! Next
up: let's make it do something more interesting.

## 2. Simple logic

After getting the trivial module up and running, the next step is to add real
logic to the implementation. Recall that a floating-point number's fractional
part has an implicit leading `1`, so a floating-point number is representable as
an integer if its exponent is < 30, that is to say, if its value is between
[-2^31, 2^31).

To get that exponent, we need to _unbias_ it. In its binary representation, a
valid floating-point number's exponent is a value from 0 to 254, being an
8-bit value (an exponent of 255 indicates either NaN or an infinity).
The range of a floating-point number's exponent is from -128 to 127, though, so
we need to subtract 127 from that value to get the actual exponent. Let's write
a function to do just that:

```dslx
fn unbias_exponent(exp: u8) -> s9 {
  exp as s9 - s9:127
}
```

Notice that we need to expand the exponent to add on the sign bit before the
subtraction!

> **Note**: The repeated `s9` type specifications are a bit redundant. They're
> needed because we've not yet fully built out DSLX' type inference
> capabilities, but this is an area targeted for improvement.

Now that we can get the proper exponent, we can code up the rest of the simple
in-bound cases. To do that, we need to prepend that leading `1` and shift the
fractional part into its proper location in the final integer. Here's what that
looks like when we add that to our original function:

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

If we run this function with our original test case, it still works! Of course,
one should run additional test cases to see what happens with other inputs,
particularly because this implementation will fail for some important values.

Try adding tests on your own to find these cases - and to fix them! If you're
stumped, hints (and answers) are hidden below:

<details>
<summary>Missing case 1</summary>
What if the input is 0.0? What should the result be?

To fix this, add a specific check for a zero exponent and fractional part.
</details>

<details>
<summary>Missing case 2</summary>
Are NaNs or infinite numbers handled correctly?

To fix, add a special check for NaN or infinities at function end. Consider
making `is_inf` and `is_nan` functions!
</details>

## 3. Conclusion

I hope that these examples help you get a better grasp on DSLX and writing
modules in them. While our float-to-int function correctly handles
`float`-to-`int32_t` conversions, what if we wanted to convert `double` to
`int64_t`? Or even `float` to `int64_t`? _Even worse_, will we have to write
separate floating-point operators for every floating-point type we (or our
users) wish to support?

Fortunately, the answer is no! The next tutorial covers type parameterization,
and demonstrates how we can write a single int-to-float routine that covers all
our possible conversions. See you there!
