# DSLX Example: Prefix Scan Computation

In this document we explain in detail the implementation of a 8 byte prefix scan
computation. In order to understand the implementation, it is useful to
understand the intended functionality first.

For a given input of 8 bytes, the scan iterates from left to right over the
input and produces an output of the same size. Each element in the output
contains the count of duplicate values seen so far in the input. The counter
resets to 0 if a new value is found.

For example, for this input:

```
  let input = bits[8,32]:[0, 0, 0, 0, 0, 0, 0, 0]
```

the code should produce this output:

```
  bits[8,3]:[0, 1, 2, 3, 4, 5, 6, 7])
```

At index 0 it has not yet found any value, so it assigns a counter value of `0`.

At index 1 it finds the second occurrence of the value '0' (which is the 1st
duplicate) and therefore adds a 1 to the counter from index 0.

At index 2 it finds the third occurrence of the value '0' (which is the 2nd
duplicate) and therefore adds a 1 to the counter from index 1. And so on.

Correspondingly, for this input:

```
  let input = bits[8,32]:[0, 0, 1, 1, 2, 2, 3, 3]
```

it should produce:

```
  assert_eq(result, bits[8,3]:[0, 1, 0, 1, 0, 1, 0, 1])
```

The full listing is in `examples/dslx_intro/prefix_scan_equality.x`.

### Function prefix_scan_eq

The implementation displays a few interesting language features.

The function prototype is straigt-forward. Input is an array of 8 values of type
`u32`. Output is an array of size 8 holding 3-bit values (the maximum resulting
count can only be 7, which fits in 3 bits).

```
fn prefix_scan_eq(x: u32[8]) -> bits[8,3] {
```

The first let expression produces a tuple of 3 values. It only cares about the
last value `result`, so it stubs out the other two elements via the 'ignore'
placeholder `_`.

```
  let (_, _, result) =
```

Why a 3-Tuple? Because he following loop has tuple of three values as the
accumulator. The return type of the loop is the type of the accumulator, so
above let needs to be of the same type.

#### Enumerated Loop

Using tuples as the accumulator is a convenient way to model multiple
loop-carried values:

```
    for ((i, elem), (prior, count, result)): ((u32, u32), (u32, u3, bits[8,3]))
          in enumerate(x) {
```

The iterable of this loop is `enumerate(x)`. On each iteration, this construct
delivers a tuple consisting of current index and current element. This is
represented as the tuple `(i, elem)` in the `for` construct.

The loop next specifies the accumulator, which is a 3-tuple consisting of the
values named `prior`, `count`, and `result`.

The types of the iterable and accumulator are specified next. The iterable is a
tuple consisting of two `u32` values. The accumulator is more interesting, it is
a tuple consiting of a `u32` value (`prior`), a `u3` value (`count`), and a
2-dimension array type `bits[8, 3]`, which is an array holding 8 elements of
bit-width 3. This is the type of `result` in the accumulator.

Looping back to the prior `let` statement, it ignores the `prior` and `count`
members of the tuple and will only return the `result` part.

#### A Match Expression

The next expression is an interesting `match` expression. The let expression
binds the tuple `(to_place, new_count): (u3, u3)` to the result of the following
match expression:

```
let (to_place, new_count): (u3, u3) = match (i == u32:0, prior == elem) {
```

`to_place` will hold the value that is to be written at a given index.
`new_count` will contain the updated counter value.

The `match` expression evaluates two conditions in parallel:

*   is `i` == 0?
*   is the `prior` element the same as the current `elem`

Two tests mean there are four possible cases, which are all handled in the
following four cases:

```
      // i == 0 (no matter whether prior == elem or not):
      //    we set position 0 to 0 and update the new_counter to 1
      (true, true) => (u3:0, u3:1);
      (true, false) => (u3:0, u3:1);

      // if i != 0 - if the current element is the same as pior,
      //    set to_place to the value of the current count
      //    update new_counter with the increased counter value
      (false, true) => (count, count + u3:1);

      // if i != 0 - if current element is different from prior,
      //     set to_place back to 0
      //     set new_counter back to 1
      (false, false) => (u3:0, u3:1);
    } in
```

To update the result, we set index `i` in the `result` array to the value
`to_place` via the built-in `update` function, which produces a new value
`new_result`):

```
    let new_result: bits[8,3] = update(result, i, to_place) in
```

Finally the updated accumulator value is constructed, it is the last expression
in the loop:

```
    (elem, new_count, new_result)
```

Following the loop body, as an argument to the loop, we initialize the
accumulator in the following way.

*   set element `prior` to -1, in order to not match any other value.
*   set element `count` to 0.
*   set element `result` to 8 0's of size `u3`

```
}((u32:-1, u3:0, bits[8,3]:[u3:0, u3:0, u3:0, u3:0, u3:0, u3:0, u3:0, u3:0])) in
```

And, finally, the function simply returns `result`:

```
  result
}
```

### Testing

To test the two cases we've described above, we add the following two test cases
right to this implementation file:

```
test prefix_scan_eq_all_zero {
  let input = bits[8,32]:[0, 0, 0, 0, 0, 0, 0, 0] in
  let result = prefix_scan_eq(input) in
  assert_eq(result, bits[8,3]:[0, 1, 2, 3, 4, 5, 6, 7])
}

test prefix_scan_eq_doubles {
  let input = bits[8,32]:[0, 0, 1, 1, 2, 2, 3, 3] in
  let result = prefix_scan_eq(input) in
  assert_eq(result, bits[8,3]:[0, 1, 0, 1, 0, 1, 0, 1])
}
```
