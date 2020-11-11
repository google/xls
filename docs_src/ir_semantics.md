# XLS: IR semantics

[TOC]

The XLS IR is a pure dataflow-oriented IR that has the static-single-assignment
property, but is specialized for generating circuitry. Notably, it includes high
level parallel patterns. The aim is to create effective circuit designs through
a "lifted" understanding of the high-level operations and their semantics,
instead of trying to reverse all relevant properties via dependence analysis,
which often cannot take advantage of high level knowledge that the designer
holds in their mind at design time.

This document describes the semantics of the XLS intermediate representation
(IR) including data types, operations, and textual representation.

## Data types

### Bits

A vector of bits with a fixed width.

**Type syntax:**

`bits[N]` where `N` is the number of bits.

**Value syntax:**

*   A literal decimal number. Example: `42`.
*   A binary number prefixed with `0b`. Example: `0b10101`
*   A hexadecimal number: `0x`. Example: `0xdeadbeef`

The representation may optionally include the bit width in which case the type
is prefixed before the literal: `bits[N]:$literal`. Example: `bits[8]:0xab`.

### Array

A one-dimensional array of elements of the same type with a fixed number of
elements. An array can contain bits, arrays, or tuples as elements. May be empty
(in which case the type of the array element cannot be automatically deduced).

**Type syntax:**

`$type[N]`: an array containing `N` elements of type `$type`. Examples:

*   Two-element array of 8-bit bits type: `bits[8][2]`
*   Three-element array of tuple type: `(bits[32], bits[2])[3]`

**Value syntax:**

`[$value_1, ... , $value_N]` where `$value_n` is the value of the `n`-th
element. Examples:

*   Array of bits elements with explicit bit count: `[bits[8]:10, bits[8]:30]`
*   Three-element array consisting of two-element arrays of bits elements: `[[1,
    2], [3, 4], [5, 6]]]`

### Tuple

An ordered set of fixed size containing elements with potentially different
types. tuples can contain bits, arrays, or tuples as elements. May be empty.

**Type syntax:**

`($type_{0}, ..., $type_{N-1})` where `N` is the number of elements and where
`$type_n` is the type of the `n`-th element.

**Value syntax:**

`($value_{0}, ..., $value_{N-1})` where `$value_n` is the value of the `n`-th
element. Examples:

*   Tuple containing two bits elements: `(0b100, 0b101)`
*   A nested tuple containing various element types: `((1, 2), 42, [5, 6])`

### Token

A type used to enforce ordering between channel operations. The token type has
no value and all tokens are identical. A token is purely symbolic / semantic and
has no correlate in hardware.

**Type syntax:**

`token`

## Operations

Operations share a common syntax and have both positional and keyword arguments
à la Python. Positional arguments are ordered and must appear first in the
argument list. Positional arguments are exclusively the identifiers of the
operands. Keyword arguments are unordered and must appear after the positional
arguments. Keyword arguments can include arbitrary value types.

```
result = operation(pos_arg_0, ..., pos_arg_N, keyword_0=value0, ..., keyword_M=valueM, ...)
```

**Common keyword arguments**

| Keyword | Type             | Required | Default | Description           |
| ------- | ---------------- | -------- | ------- | --------------------- |
| `pos`   | `SourceLocation` | no       |         | The source location   |
:         :                  :          :         : associated with this  :
:         :                  :          :         : operation. The syntax :
:         :                  :          :         : is a triplet of       :
:         :                  :          :         : comma-separated       :
:         :                  :          :         : integer values\:      :
:         :                  :          :         : `Fileno,Lineno,Colno` :

### Unary bitwise operations

Performs a bit-wise operation on a single bits-typed operand.

**Syntax**

```
result = identity(operand)
result = not(operand)
```

**Types**

Value     | Type
--------- | ---------
`operand` | `bits[N]`
`result`  | `bits[N]`

**Operations**

Operation  | Opcode          | Semantics
---------- | --------------- | -------------------
`identity` | `Op::kIdentity` | `result = operand`
`not`      | `Op::kNot`      | `result = ~operand`

### Variadic bitwise operations

Performs a bit-wise operation on one-or-more identically-typed bits operands. If
only a single argument is provided the operation is a no-op.

**Syntax**

```
result = and(operand_{0}, ..., operand_{N-1})
result = or(operand_{0}, ..., operand_{N-1})
result = xor(operand_{0}, ..., operand_{N-1})
```

**Types**

Value            | Type
---------------- | ---------
`operand_{i}`    | `bits[N]`
`result`         | `bits[N]`

**Operations**

Operation | Opcode     | Semantics
--------- | ---------- | ---------------------
`and`     | `Op::kAnd` | `result = lhs & rhs & ...`
`or`      | `Op::kOr`  | `result = lhs \| rhs \| ...`
`xor`     | `Op::kXor` | `result = lhs ^ rhs ^ ...`

### Arithmetic unary operations

Performs an arithmetic operation on a single bits-typed operand.

**Syntax**

```
result = neg(operand)
```

**Types**

Value     | Type
--------- | ---------
`operand` | `bits[N]`
`result`  | `bits[N]`

**Operations**

Operation | Opcode     | Semantics
--------- | ---------- | -------------------
`neg`     | `Op::kNeg` | `result = -operand`

### Arithmetic binary operations

Performs an arithmetic operation on a pair of bits operands. Unsigned operations
are prefixed with a 'u', and signed operations are prefixed with a 's'.

**Syntax**

```
result = add(lhs, rhs)
result = smul(lhs, rhs)
result = umul(lhs, rhs)
result = sdiv(lhs, rhs)
result = sub(lhs, rhs)
result = udiv(lhs, rhs)
```

**Types**

Currently signed and unsigned multiply support arbitrary width operands and
result. For all other arithmetic operations the operands and the result are the
same width. The expectation is that all arithmetic operations will eventually
support arbitrary widths.

**Operations**

Operation | Opcode      | Semantics
--------- | ----------- | --------------------------------------------
`add`     | `Op::kAnd`  | `result = lhs + rhs`
`sdiv`    | `Op::kSDiv` | `result = $signed(lhs) / $signed(rhs)` * **
`smod`    | `Op::kSMod` | `result = $signed(lhs) % $signed(rhs)` * ***
`smul`    | `Op::kSMul` | `result = $signed(lhs) * $signed(rhs)`
`sub`     | `Op::kSub`  | `result = lhs - rhs`
`udiv`    | `Op::kUDiv` | `result = lhs / rhs` * **
`umod`    | `Op::kUMod` | `result = lhs % rhs` * ***
`umul`    | `Op::kUMul` | `result = lhs * rhs`

\* Synthesizing division or modulus can lead to failing synthesis and/or
problems with timing closure. It is usually best not to rely on this Verilog
operator in practice, but instead explicitly instantiate a divider of choice.

\** Division rounds toward zero. For unsigned division this is the same as
truncation. If the divisor is zero, unsigned division produces a maximal
positive value. For signed division, if the divisor is zero the result is the
maximal positive value if the dividend is non-negative or the maximal negative
value if the dividend is negative.

\*** For signed modulus, the sign of the result of modulus matches the sign of
the left operand. If the right operand is zero the result is zero.

### Comparison operations

Performs a comparison on a pair of identically-typed bits operands. Unsigned
operations are prefixed with a 'u', and signed operations are prefixed with a
's'. Produces a result of bits[1] type.

**Syntax**

```
result = eq(lhs, rhs)
result = ne(lhs, rhs)
result = sge(lhs, rhs)
result = sgt(lhs, rhs)
result = sle(lhs, rhs)
result = slt(lhs, rhs)
result = uge(lhs, rhs)
result = ugt(lhs, rhs)
result = ule(lhs, rhs)
result = ult(lhs, rhs)
```

**Types**

Value    | Type
-------- | ---------
`lhs`    | `bits[N]`
`rhs`    | `bits[N]`
`result` | `bits[1]`

**Operations**

Operation | Opcode     | Semantics
--------- | ---------- | ---------------------
`eq`      | `Op::kEq`  | `result = lhs == rhs`
`ne`      | `Op::kNe`  | `result = lhs != rhs`
`sge`     | `Op::kSGe` | `result = lhs >= rhs`
`sgt`     | `Op::kSGt` | `result = lhs > rhs`
`sle`     | `Op::kSLe` | `result = lhs <= rhs`
`slt`     | `Op::kSLt` | `result = lhs < rhs`
`uge`     | `Op::kUGe` | `result = lhs >= rhs`
`ugt`     | `Op::kUGt` | `result = lhs > rhs`
`ule`     | `Op::kULe` | `result = lhs <= rhs`
`ult`     | `Op::kULt` | `result = lhs < rhs`

### Shift operations

Performs an shift operation on an input operand where the shift amount is
specified by a second operand.

**Syntax**

```
result = shll(operand, amount)
result = shra(operand, amount)
result = shrl(operand, amount)
```

**Types**

The shifted operand and the result of the shift are the same width. Widths of
the shift amount may be arbitrary.

**Operations**

Operation | Opcode      | Semantics
--------- | ----------- | --------------------------------------------------
`shll`    | `Op::kShll` | `result = lhs << rhs` *
`shra`    | `Op::kShra` | `result = lhs >>> rhs` (arithmetic shift right) **
`shrl`    | `Op::kShra` | `result = lhs >> rhs` *

\* Logically shifting greater than or equal to the number of bits in the `lhs`
produces a result of zero.

** Arithmetic right shifting greater than or equal to the number of bits in the
`lhs` produces a result equal to all of the bits set to the sign of the `lhs`.

### Extension operations

Extends a bit value to a new (larger) target bit-length.

**Syntax**

```
result = zero_ext(x, new_bit_count=42)
result = sign_ext(x, new_bit_count=42)
```

**Types**

Value           | Type
--------------- | ---------------------
`arg`           | `bits[N]`
`new_bit_count` | `int64`
`result`        | `bits[new_bit_count]`

Note: `new_bit_count` should be `>= N` or an error may be raised.

#### `zero_ext`

Zero-extends a value: turns its bit-length into the new target bit-length by
filling zeroes in the most significant bits.

#### `sign_ext`

Sign-extends a value: turns its bit-length into the new target bit-length by
filling in the most significant bits (MSbs) with the following policy:

*   ones in the MSbs if the MSb of the original value was set, or
*   zeros in the MSbs if the MSb of the original value was unset.

### Channel operations

These operations send or receive data over channels. Channels are monomorphic,
and each channel supports a fixed set of data types which are sent or received
in a single transaction.

#### **`receive`**

Receives data values from a specified channel. The number of data values `N` and
their type is determined by the channel.

```
result = receive(tkn, channel_id=<ch>)
```

**Types**

Value    | Type
-------- | -------------------------------
`tkn`    | `token`
`result` | `(token, T_{0}, ... , T_{N-1})`

**Keyword arguments**

| Keyword      | Type    | Required | Default | Description              |
| ------------ | ------- | -------- | ------- | ------------------------ |
| `channel_id` | `int64` | yes      |         | The ID of the channel to |
:              :         :          :         : receive data from        :

#### **`receive_if`**

Receives data values from a specified channel if and only if the predicate
operand is true. The number of data values `N` and their type is determined by
the channel.

```
result = receive_if(tkn, pred, channel_id=<ch>)
```

**Types**

Value    | Type
-------- | -------------------------------
`tkn`    | `token`
`pred`   | `bits[1]`
`result` | `(token, T_{0}, ... , T_{N-1})`

**Keyword arguments**

| Keyword      | Type    | Required | Default | Description              |
| ------------ | ------- | -------- | ------- | ------------------------ |
| `channel_id` | `int64` | yes      |         | The ID of the channel to |
:              :         :          :         : receive data from        :

If the predicate is false the data values in the result are zero-filled.

#### **`send`**

Sends data values to a specified channel. The number of data values `N` and
their type is determined by the channel.

```
result = send(tkn, data_{0}, ..., data_{N-1}, channel_id=<ch>)
```

**Types**

Value      | Type
---------- | -------
`tkn`      | `token`
`data_{i}` | `T_{i}`
`result`   | `token`

The types of operands `data_{i}` and the number `N` must match the types and
number of data elements supported by the channel.

**Keyword arguments**

| Keyword      | Type    | Required | Default | Description                   |
| ------------ | ------- | -------- | ------- | ----------------------------- |
| `channel_id` | `int64` | yes      |         | The ID of the channel to send |
:              :         :          :         : data to.                      :

#### **`send_if`**

Sends data values to a specified channel if and only if the predicate operand is
true. The number and type of data values is determined by the channel.

```
result = send_if(tkn, pred, data_{0}, ..., data_{N-1}, channel_id=<ch>)
```

**Types**

Value      | Type
---------- | ---------
`tkn`      | `token`
`pred`     | `bits[1]`
`data_{i}` | `T_{i}`
`result`   | `token`

The types of operands `data_{i}` and the number `N` must match the types and
number of data elements supported by the channel.

**Keyword arguments**

| Keyword      | Type    | Required | Default | Description                   |
| ------------ | ------- | -------- | ------- | ----------------------------- |
| `channel_id` | `int64` | yes      |         | The ID of the channel to send |
:              :         :          :         : data to.                      :

### Miscellaneous operations

#### **`after_all`**

Used to construct partial orderings among channel operations.

```
result = after_all(operand_{0}, ..., operand_{N-1})
```

**Types**

Value         | Type
------------- | -------
`operand_{i}` | `token`
`result`      | `token`

`after_all` can consume an arbitrary number of token operands including zero.

#### **`array`**

Constructs a array of its operands.

```
result = array(operand_{0}, ..., operand_{N-1})
```

**Types**

Value         | Type
------------- | ------
`operand_{i}` | `T`
`result`      | `T[N]`

Array can take an arbitrary number of operands including zero (which produces an
empty array).

#### **`array_index`**

Returns a single element from an array.

**Syntax**

```
result = array_index(data, index)
```

**Types**

Value    | Type
-------- | -----------------------------
`lhs`    | Array of elements of type `T`
`rhs`    | `bits[M]`
`result` | `T`

Returns the element at the index given by operand `index` from the array `data`.

TODO: Define out of bounds semantics for array_index.

#### **`array_update`**

Returns a modified copy of an array.

**Syntax**

```
result = array_update(array, index, value)
```

**Types**

Value    | Type
-------- | -----------------------------
`array`  | Array of elements of type `T`
`index`  | `bits[M]`*
`value`  | `T`
`result` | Array of elements of type `T`

\* M is arbitrary.

Returns a copy of the input array with the element at the index replaced with
the given value. If index is out of bounds, the returned array is identical to
the input array.

#### **`bit_slice`**

Slices a contiguous range of bits from a bits-typed operand.

**Syntax**

```
result = bit_slice(operand, start=<start>, width=<width>)
```

**Types**

Value     | Type
--------- | ---------------
`operand` | `bits[N]`
`result`  | `bits[<width>]`

**Keyword arguments**

| Keyword | Type    | Required | Default | Description       |
| ------- | ------- | -------- | ------- | ----------------- |
| `start` | `int64` | yes      |         | The starting bit  |
:         :         :          :         : of the slice.     :
:         :         :          :         : `start` is is     :
:         :         :          :         : zero-indexed      :
:         :         :          :         : where zero is the :
:         :         :          :         : least-significant :
:         :         :          :         : bit of the        :
:         :         :          :         : operand.          :
| `width` | `int64` | yes      |         | The width of the  |
:         :         :          :         : slice.            :

The bit-width of `operand` must be greater than or equal to `<start>` plus
`<width>`.

#### **`dynamic_bit_slice`**

Slices a contiguous range of bits from a bits-typed operand, with variable
starting index but fixed width. Out-of-bounds slicing is supported by
treating all out-of-bounds bits as having value 0.

**Syntax**

```
result = dynamic_bit_slice(operand, start, width=<width>)
```

**Types**

Value     | Type
--------- | ---------------
`operand` | `bits[N]`
`start`   | `bits[M]`
`result`  | `bits[<width>]`

`start` can be of arbitrary bit width. It will be interpreted
as an unsigned integer.

**Keyword arguments**

| Keyword | Type    | Required | Default | Description       |
| ------- | ------- | -------- | ------- | ----------------- |
| `width` | `int64` | yes      |         | The width of the  |
:         :         :          :         : slice.            :

#### **`concat`**

Concatenates and arbitrary number of bits-typed operands.

```
result = concat(operand{0}, ..., operand{n-1})
```

**Types**

Value         | Type
------------- | ------------------
`operand_{i}` | `bits[N_{i}]`
`result`      | `bits[Sum(N_{i})]`

This is equivalent to the verilog concat operator: `result = {arg0, ..., argN}`

#### **`counted_for`**

Invokes a fixed-trip count loop.

**Syntax**

```
result = counted_for(init, trip_count=<trip_count>, stride=<stride>, body=<body>, invariant_args=<inv_args>)
```

**Types**

Value    | Type
-------- | ----
`init`   | `T`
`result` | `T`

**Keyword arguments**

| Keyword          | Type     | Required | Default | Description               |
| ---------------- | -------- | -------- | ------- | ------------------------- |
| `trip_count`     | `int64`  | yes      |         | Trip count of the loop    |
:                  :          :          :         : (number of times that the :
:                  :          :          :         : loop body will be         :
:                  :          :          :         : executed)                 :
| `stride`         | `int64`  | no       | 1       | Stride of the induction   |
:                  :          :          :         : variable                  :
| `invariant_args` | array of | yes      |         | Names of the invariant    |
:                  : operands :          :         : operands as the loop body :
| `body`           | `string` | yes      |         | Name of the function to   |
:                  :          :          :         : use as the loop body      :

`counted_for` invokes the function `body` `trip_count` times, passing
loop-carried data that starts with value `init`.

*   The first argument passed to `body` is the induction variable -- presently,
    the induction variable always starts at zero and increments by `stride`
    after every trip.
*   The second argument passed to `body` is the loop-carry data. The return type
    of `body` must be the same as the type of the `init` loop carry data. The
    value returned from the last trip is the result of the `counted_for`
    expression.
*   All subsequent arguments passed to `body` are passed from `invariant_args`;
    e.g. if there are two members in `invariant_args` those values are passed as
    the third and fourth arguments.

Therefore `body` should have a signature that matches the following:

```
body(i, loop_carry_data[, invariant_arg0, invariant_arg1, ...])
```

Note that we currently inspect the `body` function to see what type of induction
variable (`i` above) it accepts in order to pass an `i` value of that type.

#### **`dynamic_counted_for`**

Invokes a dynamic-trip count loop.

**Syntax**

```
result = counted_for(init, trip_count, stride, body=<body>, invariant_args=<inv_args>)
```

**Types**

Value        | Type
------------ | ----
`init`       | `T`
`trip_count` | `bits[N], treated as unsigned`
`stride`     | `bits[M], treated as signed`,
`result`     | `T`

**Keyword arguments**

| Keyword          | Type     | Required | Default | Description               |
| ---------------- | -------- | -------- | ------- | ------------------------- |
| `invariant_args` | array of | yes      |         | Names of the invariant    |
:                  : operands :          :         : operands as the loop body :
| `body`           | `string` | yes      |         | Name of the function to   |
:                  :          :          :         : use as the loop body      :

`dynamic_counted_for` invokes the function `body` `trip_count` times, passing
loop-carried data that starts with value `init`. The induction variable is 
incremented by `stride` after each iteration.

*   The first argument passed to `body` is the induction variable -- presently,
    the induction variable always starts at zero and increments by `stride`
    after every trip.
*   The second argument passed to `body` is the loop-carry data. The return type
    of `body` must be the same as the type of the `init` loop carry data. The
    value returned from the last trip is the result of the `counted_for`
    expression.
*   All subsequent arguments passed to `body` are passed from `invariant_args`;
    e.g. if there are two members in `invariant_args` those values are passed as
    the third and fourth arguments.

Therefore `body` should have a signature that matches the following:

```
body(i, loop_carry_data, [invariant_arg0, invariant_arg1, ...])
```

Note that we currently inspect the `body` function to see what type of induction
variable (`i` above) it accepts in order to pass an `i` value of that type.
`trip_count` must have fewer bits than `i` and `stride` should have fewer than 
or equal number of bits to `i`.

#### **`decode`**

Implements a binary decoder.

```
result = decode(operand, width=<width>)
```

**Types**

Value     | Type
--------- | ---------
`operand` | `bits[N]`
`result`  | `bits[M]`

The result width `M` must be less than or equal to 2**`N` where `N` is the
operand width.

**Keyword arguments**

Keyword | Type    | Required | Default | Description
------- | ------- | -------- | ------- | -------------------
`width` | `int64` | yes      |         | Width of the result

`decode` converts the binary-encoded operand value into a one-hot result. For an
operand value of `n` interpreted as an unsigned number the `n`-th result bit and
only the `n`-th result bit is set. The width of the `decode` operation may be
less than the maximum value expressible by the input (2**`N` - 1). If the
encoded operand value is larger than the number of bits of the result the result
is zero.

#### **`encode`**

Implements a binary encoder.

```
result = encode(operand, width=<width>)
```

**Types**

Value     | Type
--------- | ---------
`operand` | `bits[N]`
`result`  | `bits[M]`

The result width `M` must be equal to $$\lceil \log_{2} N \rceil$$.

`encode` converts the one-hot operand value into a binary-encoded value of the
"hot" bit of the input. If the `n`-th bit and only the `n`-th bit of the operand
is set the result is equal the value `n` as an unsigned number.

If multiple bits of the input are set the result is equal to the logical or of
the results produced by the input bits individually. For example, if bit 3 and
bit 5 of an `encode` input are set the result is equal to 3 | 5 = 7.

If no bits of the input are set the result is zero.

#### **`invoke`**

Invokes a function.

**Syntax**

```
result = invoke(operand_{0}, ... , operand_{N-1}, to_apply=<to_apply>)
```

**Types**

Value    | Type
-------- | ----
`init`   | `T`
`result` | `T`

**Keyword arguments**

| Keyword    | Type     | Required | Default | Description                    |
| ---------- | -------- | -------- | ------- | ------------------------------ |
| `to_apply` | `string` | yes      |         | Name of the function to use as |
:            :          :          :         : the loop body                  :

TODO: finish

#### **`map`**

Applies a function to the elements of an array and returns the result as an
array.

**Syntax**

```
result = map(operand, to_apply=<to_apply>)
```

**Types**

Value     | Type
--------- | ----------
`operand` | `array[T]`
`result`  | `array[U]`

**Keyword arguments**

| Keyword    | Type     | Required | Default | Description                    |
| ---------- | -------- | -------- | ------- | ------------------------------ |
| `to_apply` | `string` | yes      |         | Name of the function to apply  |
:            :          :          :         : to each element of the operand :

TODO: finish

#### **`one_hot`**

Produces a bits value with exactly one bit set. The index of the set bit depends
upon the input value.

**Syntax**

```
result = one_hot(input, lsb_prio=true)
result = one_hot(input, lsb_prio=false)
```

**Types**

Value    | Type
-------- | -----------
`input`  | `bits[N]`
`result` | `bits[N+1]`

**Keyword arguments**

| Keyword    | Type     | Required | Default | Description                    |
| ---------- | -------- | -------- | ------- | ------------------------------ |
| `lsb_prio` | `bool`   | yes      |         | Whether the least significant  |
:            :          :          :         : bit (LSb) has priority.        :

For `lsb_prio=true`: result bit `i` for `0 <= i < N` is set in `result` iff bit
`i` is set in the input and all lower bits `j` for `j < i` are not set in the
input.

For `lsb_prio=false`: result bit `i` for `N-1 >= i >= 0` is set in `result` iff
bit `i` is set in the input and all higher (more significant) bits `j` for `j >
i` are not set in the input.

For **both** `lsb_prio=true` and `lsb_prio=false`, result bit `N` (the most
significant bit in the output) is only set if no bits in the input are set.

Examples:

* `one_hot(0b0011, lsb_prio=true)` => `0b00001` -- note that an extra MSb has
  been appended to the output to potentially represent the "all zeros" case.
* `one_hot(0b0111, lsb_prio=false)` => `0b00100`.
* `one_hot(0b00, lsb_prio=false)` => `0b100`.
* `one_hot(0b00, lsb_prio=true)` => `0b100` -- note the output for `one_hot` is
  the same for the all-zeros case regardless of whether `lsb_prio` is true or
  false.

This operation is useful for constructing match or switch operation semantics
where a condition is matched against an ordered set of cases and the first match
is chosen. It is also useful for one-hot canonicalizing, e.g. as a prelude to
counting leading/trailing zeros.

#### **`one_hot_sel`**

Selects between operands based on a one-hot selector.

**Syntax**

```
result = one_hot_sel(selector, cases=[case_{0}, ... , case_{N-1}])
```

**Types**

Value      | Type
---------- | ---------
`selector` | `bits[N]`
`case_{i}` | `T`
`result`   | `T`

The result is the logical OR of all cases `case_{i}` for which the corresponding
bit `i` is set in the selector. When selector is one-hot this performs a select
operation.

#### **`param`**

TODO: finish

#### **`reverse`**

Reverses the order of bits of its operand.

```
result = reverse(operand)
```

**Types**

Value     | Type
--------- | ---------
`operand` | `bits[N]`
`result`  | `bits[N]`

#### **`sel`**

Selects between operands based on a selector value.

**Syntax**

```
result = sel(selector, cases=[case_{0}, ... , case_{N-1}], default=<default>)
```

**Types**

Value      | Type
---------- | ---------
`selector` | `bits[M]`
`case_{i}` | `T`
`default`  | `T`
`result`   | `T`

#### **`tuple`**

Constructs a tuple of its operands.

```
result = tuple(operand_{0}, ..., operand_{N-1})
```

**Types**

Value         | Type
------------- | ------------------------
`operand_{i}` | `T_{i}`
`result`      | `(T_{0}, ... , T_{N-1})`

Tuple can take and arbitrary number of operands including zero (which produces
an empty tuple).

#### **`tuple_index`**

Returns a single element from a tuple-typed operand.

**Syntax**

```
result = tuple_index(operand, index=<index>)
```

**Types**

Value     | Type
--------- | ------------------------
`operand` | `(T_{0}, ... , T_{N-1})`
`result`  | `T_{<index>}`

**Keyword arguments**

Keyword | Type    | Required | Default | Description
------- | ------- | -------- | ------- | ---------------------------------
`index` | `int64` | yes      |         | Index of tuple element to produce
