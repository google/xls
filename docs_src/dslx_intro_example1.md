# DSLX Example: Compute CRC32 Checksum

In this document we explain in detail the implementation of routine to compute a
CRC32 checksum on a single 8-bit input. We don't discuss the algorithm here,
only the language features necessary to implement the algorithm.

Refer to the full implementation, in `examples/dslx_intro/crc32_one_byte.x`,
while following this document.

### Function Prototype

Let's explain how the function is being defined:

```
fn crc32_one_byte(byte: u8, polynomial: u32, crc: u32) -> u32 {
```

Functions are defined starting with the keyword `fn`, followed by the function's
name, `crc32_one_byte` in this case. Then comes the list of parameters, followed
by the declaration of the return type.

This function accepts 3 parameters:

1.  `byte`, which is of type `u8`. `u8` is a shortcut for `bits[8]`
2.  `polynomial`, which is of type `u32`, a 32-bit type.
3.  `crc`, which is also of type `u32`

The return type, which is declared after the `->`, is also a `u32`.

The first line of the function's body is quite curious:

```
let crc: u32 = crc ^ u32:byte;
```

The expression to the right side of the `=` is easy to understand, it computes
the `xor` operation between the incoming parameters `crc` and `byte`, which has
been cast to a `u32`.

The `let` expression re-binds `crc` to the expression on the right. This looks
like a classic variable assignment. However, since these expressions are all
scoped by the `let` expression, the newly assigned `crc` values are different
and distinguishable from their previous values. In other words, the original
value of `crc` is not visible to anybody else after the re-binding.

The next line specifies a `for` loop. Index variable and accumulator are `i` and
`crc`, both of type `u32`. The iterable range expression specifies that the loop
should execute 8 times.

```
  // 8 rounds of updates.
  for (i, crc): (u32, u32) in range(u32:8) {
```

At the end of the loop, the calculated value is being assigned to the
accumulator `crc` - the last expression in the loop body is assigned to the
accumulator:

```
    let mask: u32 = -(crc & u32:1);
    (crc >> u32:1) ^ (polynomial & mask)
```

Finally, the accumulator's initial value is being passed to the `for` expression
as a parameter. This can be confusing, especially when compared to other
languages, where the init value typically is provided at or near the top of a
loop.

```
}(crc)
```
