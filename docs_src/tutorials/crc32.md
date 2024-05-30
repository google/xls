# Tutorial: `for` expressions

In this document we explain in detail the implementation of routine to compute a
CRC32 checksum on a single 8-bit input. We don't discuss the algorithm here,
only the language features necessary to implement the algorithm.

Refer to the
[full implementation](https://github.com/google/xls/tree/main/xls/examples/dslx_intro/crc32_one_byte.x)
while following this document.

### Function Prototype

The signature and first line of this function should look familiar enough now,
but the `for` construct is new: DSLX provides a means of iterating **a fixed
number of times** within a function.

A DSLX `for` loop has the following structure:

1.  The loop signature: this consists of three elements:
    1.  An `(index, <accumulator vars>)` tuple. The index holds the current
        iteration number, and the accumulator vars are user-specified data
        carried into the current iteration.
    2.  The type specification for the index/accumulators tuple. Note that the
        index type can be controlled by the user (i.e., doesn't have to be u32,
        but it should be able to hold all possible loop index values).
    3.  An
        [iterable](../dslx_reference.md),
        either the `range()` or `enumerate()` expressions, either of which
        dictates the number of iterations of the loop to complete.
2.  The loop body: this has the same general form as a DSLX function.
    Particularly noteworthy is that the loop body ends by stating the "return"
    value. In a `for` loop, this "return" value is either used as the input to
    the next iteration of the loop (for non-terminal iterations) or as the
    result of the entire expression (for the terminal iteration).

For this specific for loop, the index variable is unused, so we assign it to
`_`. This indicates to the DSLX frontend that "the variable is unused but that's
ok"; a trailing name after the underscore is also allowed to provide additional
context. The accumulator consists of a single variable `crc`. Both index and
accumulator are of type `u32`. The iterable range expression specifies that the
loop should execute 8 times.

```dslx-snippet
  // 8 rounds of updates.
  for (_, crc): (u32, u32) in range(u32:8) {
```

At the end of the loop, the calculated value is being assigned to the
accumulator `crc` - the last expression in the loop body is assigned to the
accumulator:

```dslx-snippet
    let mask: u32 = -(crc & u32:1);
    (crc >> u32:1) ^ (polynomial & mask)
```

Finally, the accumulator's initial value is being passed to the `for` expression
as a parameter. This can be confusing, especially when compared to other
languages, where the init value typically is provided at or near the top of a
loop.

```dslx-snippet
}(crc)
```

Since the `for` loop is the last expression in the function, it's also the
function's return value, but in other contexts, it could be assigned to a
variable and used elsewhere. In general, the result of a `for` expression can be
used in the same manner as any other expression's result.
