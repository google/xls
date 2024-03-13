# XLS: IR semantics

[TOC]

The XLS IR is a pure dataflow-oriented IR that has the static-single-assignment
property, but is specialized for generating circuitry. The aim is to create
effective circuit designs through a "lifted" understanding of the high-level
operations and their semantics, instead of trying to reverse all relevant
properties via dependence analysis, which often cannot take advantage of high
level knowledge that the designer holds in their mind at design time.

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
elements. An array can contain bits, arrays, or tuples as elements. Empty
(zero-element) arrays are not supported.

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

## Functions, procs, and blocks

The XLS IR has three function-level abstractions each which hold a data-flow
graph of XLS IR operations: *functions*, *procs*, and *blocks*. Names of
function, procs and blocks must be unique among their respective abstractions
(functions, procs, and blocks). For example, a block cannot share a name with
another block but can share a name with a function.

### Function

A function is a stateless abstraction with a single-output which is computed
from zero or more input parameters. May invoke other functions.

### Proc

A Proc is a stateful abstraction with an arbitrarily-typed recurrent state.
Procs can communicate with other procs via channels which (abstractly) are
infinite-depth FIFOs with flow control. Channel communication is handled via
send and receive IR operations. Procs may invoke functions.

TODO(meheff): 2021/11/04 Expand to include more details.

### Block

A Block is an RTL-level abstraction used for code generation. It corresponds to
a single Verilog module. Procs and functions are converted to blocks as part of
the code generation process. Blocks may “invoke” other blocks via instantiation.
A block includes explicit representations of RTL constructs: ports, registers,
and instantiations. The constructs are scoped within the block.

#### Port

A port is a representation of an input or output to the block. These correspond
to ports on Verilog modules. Ports can be arbitrarily-typed. In the block, each
port is represented with a `input_port` or `output_port` operation.

#### Register

A register is a representation of a hardware register (flop). Registers can be
arbitrarily-typed. Each register must have a single `register_write` and a
single `register_read` operation for writing and reading the register
respectively.

Each register may optionally specify its reset behavior. The reset can be
specified to occur either synchronously or asynchronously and either on the
`reset` signal of the associated `register_write` being active-high or
active-low. If specified the reset value must match the type of the register. If
no reset behavior is specified then the `reset` argument of `register_write`
must be unset.

#### Instantiation

An instantiation is a block-scoped construct that represents a module
instantiation at the Verilog level. The instantiated object can be another
block, a FIFO (not yet supported), or a externally defined Verilog module (not
yet supported). The instantiation is integrated into the instantiating block
with `instantiation_input` and `instantiation_output` operations. There is a
one-to-one mapping between the instantiation input/output and the ports of the
instantiated objects.

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

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword | Type             | Required | Default | Description           |
| ------- | ---------------- | -------- | ------- | --------------------- |
| `pos`   | `SourceLocation` | no       |         | The source location associated with this operation. The syntax is a triplet of comma-separated integer values: `Fileno,Lineno,Colno` |

<!-- mdformat on -->

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

Value         | Type
------------- | ---------
`operand_{i}` | `bits[N]`
`result`      | `bits[N]`

**Operations**

Operation | Opcode     | Semantics
--------- | ---------- | ----------------------------
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
result = smod(lhs, rhs)
result = sub(lhs, rhs)
result = udiv(lhs, rhs)
result = umod(lhs, rhs)
result = smulp(lhs, rhs)
result = umulp(lhs, rhs)
```

**Types**

Currently signed and unsigned multiply, as wells as their partial product
variants, support arbitrary width operands and result. For all other arithmetic
operations the operands and the result are the same width. The expectation is
that all arithmetic operations will eventually support arbitrary widths.

**Operations**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Operation | Opcode       | Semantics                                    |
| --------- | ------------ | -------------------------------------------- |
| `add`     | `Op::kAdd`   | `result = lhs + rhs`                         |
| `sdiv`    | `Op::kSDiv`  | `result = $signed(lhs) / $signed(rhs)` * **  |
| `smod`    | `Op::kSMod`  | `result = $signed(lhs) % $signed(rhs)` * *** |
| `smul`    | `Op::kSMul`  | `result = $signed(lhs) * $signed(rhs)`       |
| `smulp`   | `Op::kSMulp` | `result[0] + result[1] = $signed(lhs) * $signed(rhs)` \*\*\*\* |
| `sub`     | `Op::kSub`   | `result = lhs - rhs`                         |
| `udiv`    | `Op::kUDiv`  | `result = lhs / rhs` * **                    |
| `umod`    | `Op::kUMod`  | `result = lhs % rhs` *                       |
| `umul`    | `Op::kUMul`  | `result = lhs * rhs`                         |
| `umulp`   | `Op::kUMulp` | `result[0] + result[1] = lhs * rhs` \*\*\*\* |

<!-- mdformat on -->

\* Synthesizing division or modulus can lead to failing synthesis and/or
problems with timing closure. It is usually best not to rely on this Verilog
operator in practice, but instead explicitly instantiate a divider of choice.

\** Division rounds toward zero. For unsigned division this is the same as
truncation. If the divisor is zero, unsigned division produces a maximal
positive value. For signed division, if the divisor is zero the result is the
maximal positive value if the dividend is non-negative or the maximal negative
value if the dividend is negative.

\*** The sign of the result of modulus matches the sign of the left operand. If
the right operand is zero the result is zero.

\*\*\*\* The partial product multiply variants return a two-element tuple with
both elements having the same type. The outputs are not fully constrained; the
operations are free to return any values that sum to the product `lhs * rhs`.

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
`shrl`    | `Op::kShrl` | `result = lhs >> rhs` *

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
`new_bit_count` | `int64_t`
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

Receives a data value from a specified channel. The type of the data value is
determined by the channel. An optional predicate value conditionally enables the
receive operation. An optional `blocking` attribute determines whether the
receive operation is blocking. A blocking receive waits (or blocks) until valid
data is present at the channel. Compared to a blocking receive, a non-blocking
receive has an additional entry in its return tuple of type `bits[1]` denoting
whether the data read is valid.

```
result = receive(tkn, predicate=<pred>, blocking=<bool>, channel_id=<ch>)
```

**Types**

Value    | Type
-------- | ---------------------------------------------------------------
`tkn`    | `token`
`pred`   | `bits[1]`
`result` | `(token, T)` if `blocking` == `true` else `(token, T, bits[1])`

**Keyword arguments**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword      | Type      | Required | Default | Description              |
| ------------ | --------- | -------- | ------- | ------------------------ |
| `predicate`  | `bits[1]` | no       |         | A value is received iff `predicate` is true |
| `blocking`   | `bool`    | no       | `true`  | Whether the receive is blocking             |
| `channel_id` | `int64_t` | yes      |         | The ID of the channel to receive data from  |

<!-- mdformat on -->

If the predicate is false the data values in the result are zero-filled.

#### **`send`**

Sends data to a specified channel. The type of the data values is determined by
the channel. An optional predicate value conditionally enables the send
operation.

```
result = send(tkn, data, predicate=<pred>, channel_id=<ch>)
```

**Types**

Value    | Type
-------- | ---------
`tkn`    | `token`
`data`   | `T`
`pred`   | `bits[1]`
`result` | `token`

The type of `data` must match the type supported by the channel.

**Keyword arguments**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword      | Type    | Required | Default | Description                   |
| ------------ | ------- | -------- | ------- | ----------------------------- |
| `predicate`  | `bits[1]` | no       |         | A value is sent iff `predicate` is true |
| `channel_id` | `int64_t` | yes      |         | The ID of the channel to send data to. |

<!-- mdformat on -->

### Array operations

#### **`array`**

Constructs an array of its operands.

```
result = array(operand_{0}, ..., operand_{N-1})
```

**Types**

Value         | Type
------------- | ------
`operand_{i}` | `T`
`result`      | `T[N]`

Array can take an arbitrary number of operands including zero (which produces an
empty array). The n-th operand becomes the n-th element of the array.

#### **`array_index`**

Returns a single element from an array.

**Syntax**

```
result = array_index(array, indices=[idx_{0}, ... , idx_{N-1}])
```

**Types**

Value     | Type
--------- | --------------------------------
`array`   | Array of at least `N` dimensions
`idx_{i}` | Arbitrary bits type
`result`  | `T`

Returns the element of `array` indexed by the indices `idx_{0} ... idx_{N-1}`.
The array must have at least as many dimensions as number of index elements `N`.
Each element `idx_{i}` indexes a dimension of `array`. The first element
`idx_{0}` indexes the outer most dimension, the second element `idx_{1}` indexes
the second outer most dimension, etc. The result type `T` is the type of `array`
with the `N` outer most dimensions removed.

Any out-of-bounds indices `idx_{i}` are clamped to the maximum in bounds index
for the respective dimension.

The table below shows examples of the result type `T` and the result expression
assuming input array operand `A`.

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Indices   | Array type      | result type `T` | Result expression            |
| --------- | --------------- | --------------- | ---------------------------- |
| `{1, 2}`  | `bits[3][4][5]` | `bits[3]`       | `A[1][2]`                    |
| `{10, 2}` | `bits[3][4][5]` | `bits[3]`       | `A[4][2]` (first index is out-of-bounds and clamped at the maximum index) |
| `{1}`     | `bits[3][4][5]` | `bits[3][4]`    | `A[1]`                       |
| `{}`      | `bits[3][4][5]` | `bits[3][4][5]` | `A`                          |
| `{}`      | `bits[32]`      | `bits[32]`      | `A`                          |

<!-- mdformat on -->

#### **`array_slice`**

Returns a slice of an array.

**Syntax**

```
result = array_slice(array, start, width=<width>)
```

**Types**

Value    | Type
-------- | -------------------------------------------------------------
`array`  | Array
`start`  | Arbitrary bits type
`result` | Array with same `element_type` as `array` and size of `width`

**Keyword arguments**

Keyword | Type      | Required | Default | Description
------- | --------- | -------- | ------- | ----------------------------------
`width` | `int64_t` | yes      |         | Width to make the resulting array.

Returns a copy of the segment of the input array consisting of the `<width>`
consecutive elements starting from `start`. If any element in that segment is
out-of-bounds of the original array the value at the corresponding index is the
final element in the array. This is consistent behavior with respect to the
index operation.

#### **`array_update`**

Returns a modified copy of an array.

**Syntax**

```
result = array_update(array, value, indices=[idx_{0}, ... , idx_{N-1}])
```

**Types**

Value     | Type
--------- | --------------------------------
`array`   | Array of at least `N` dimensions
`value`   | `T`
`idx_{i}` | Arbitrary bits type
`result`  | Same type as `array`

Returns a copy of the input array with the element at the given indices replaced
with the given value. If any index is out of bounds, the result is identical to
the input `array`. The indexing semantics is identical to `array_index` with the
exception of out-of-bounds behavior.

### Tuple operations

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

Keyword | Type      | Required | Default | Description
------- | --------- | -------- | ------- | ---------------------------------
`index` | `int64_t` | yes      |         | Index of tuple element to produce

### Bit-vector operations

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

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword | Type    | Required | Default | Description       |
| ------- | ------- | -------- | ------- | ----------------- |
| `start` | `int64_t` | yes      |         | The starting bit of the slice. `start` is is zero-indexed where zero is the least-significant bit of the operand. |
| `width` | `int64_t` | yes      |         | The width of the slice. |

<!-- mdformat on -->

The bit-width of `operand` must be greater than or equal to `<start>` plus
`<width>`.

#### **`bit_slice_update`**

Replaces a contiguous range of bits in a bits-typed operand at a variable start
index with a given value.

**Syntax**

```
result = bit_slice_update(operand, start, update_value)
```

**Types**

Value          | Type
-------------- | ---------
`operand`      | `bits[N]`
`start`        | `bits[I]`
`update_value` | `bits[M]`
`result`       | `bits[N]`

Evaluates to `operand` with the contiguous `M` bits starting at index `start`
replaced with `update_value`. Out-of-bound bits (which occur if `start + M > N`)
are ignored. Examples:

`operand`         | `start` | `update_value` | `result`
----------------- | ------- | -------------- | -----------------
`bits[16]:0xabcd` | `0`     | `bits[8]:0xff` | `bits[16]:0xabff`
`bits[16]:0xabcd` | `4`     | `bits[8]:0xff` | `bits[16]:0xaffd`
`bits[16]:0xabcd` | `12`    | `bits[8]:0xff` | `bits[16]:0xfbcd`
`bits[16]:0xabcd` | `16`    | `bits[8]:0xff` | `bits[16]:0xabcd`

#### **`dynamic_bit_slice`**

Slices a contiguous range of bits from a bits-typed operand, with variable
starting index but fixed width. Out-of-bounds slicing is supported by treating
all out-of-bounds bits as having value 0.

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

`start` can be of arbitrary bit width. It will be interpreted as an unsigned
integer.

**Keyword arguments**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword | Type    | Required | Default | Description       |
| ------- | ------- | -------- | ------- | ----------------- |
| `width` | `int64_t` | yes      |         | The width of the slice. |

<!-- mdformat on -->

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

Keyword | Type      | Required | Default | Description
------- | --------- | -------- | ------- | -------------------
`width` | `int64_t` | yes      |         | Width of the result

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

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword    | Type     | Required | Default | Description                    |
| ---------- | -------- | -------- | ------- | ------------------------------ |
| `lsb_prio` | `bool`   | yes      |         | Whether the least significant bit (LSb) has priority. |

<!-- mdformat on -->

For `lsb_prio=true`: result bit `i` for `0 <= i < N` is set in `result` iff bit
`i` is set in the input and all lower bits `j` for `j < i` are not set in the
input.

For `lsb_prio=false`: result bit `i` for `N-1 >= i >= 0` is set in `result` iff
bit `i` is set in the input and all higher (more significant) bits `j` for `j >
i` are not set in the input.

For **both** `lsb_prio=true` and `lsb_prio=false`, result bit `N` (the most
significant bit in the output) is only set if no bits in the input are set.

Examples:

*   `one_hot(0b0011, lsb_prio=true)` => `0b00001` -- note that an extra MSb has
    been appended to the output to potentially represent the "all zeros" case.
*   `one_hot(0b0111, lsb_prio=false)` => `0b00100`.
*   `one_hot(0b00, lsb_prio=false)` => `0b100`.
*   `one_hot(0b00, lsb_prio=true)` => `0b100` -- note the output for `one_hot`
    is the same for the all-zeros case regardless of whether `lsb_prio` is true
    or false.

This operation is useful for constructing match or switch operation semantics
where a condition is matched against an ordered set of cases and the first match
is chosen. It is also useful for one-hot canonicalizing, e.g. as a prelude to
counting leading/trailing zeros.

### Control-oriented operations

For context note that, in XLS, operations are evaluated eagerly in a very
general sense: all "branches" of computation may be evaluated in full before the
result is selected via an operation such as `one_hot_sel` or `sel`. This model
is amenable to pipeline-like hardware execution, where operations tend to be
fixed in some spatial area and operations execute a single function, while
interconnect is used for reconfiguration purposes.

Towards this eager-evaluation-capable model, operations used within a function
are generally not Turing-complete: operations such as `counted_for` require a
finite bound so that they could be implemented using a finite amount of pipeline
area. Operations such as `dynamic_counted_for` are an exception, where that
operation will only be possible to use in a time-multiplexed code generation
mode, such as the XLS sequential emitter, where arbitrary iteration to some
dynamic bound is likely to be possible.

#### **`param`**

A parameter to the current IR function, which can be used as an operand for
operations within the function.

**Syntax**

Parameters have a special syntactic form distinct from other nodes, where they
are listed directly in the function signature with their type.

```
fn f(x: bits[32]) -> bits[32] {
  ret identity.2 = identity(x, id=2)
}
```

**Types**

Value  | Type
------ | ------
`name` | `str`
`type` | `type`

#### **`sel`**

Selects between operands based on a selector value.

This behaves as if the `selector` indexes into the values given in `cases`,
providing `default` if it is indexing beyond the given `cases`.

**Syntax**

```
result = sel(selector, cases=[case_{0}, ... , case_{N-1}], default=<default>)
```

A default value must be provided **iff** the `selector` is not the correct width
for the `cases` array. That is, if the number of cases is less than
$2^{bitwidth(selector)}$ then a default value must be specified (because it must
be well defined what happens when the selector takes on values outside the case
range). If the selector is exactly the correct bitwidth a default value must not
be provided.

**Types**

Value      | Type
---------- | ---------
`selector` | `bits[M]`
`case_{i}` | `T`
`default`  | `T`
`result`   | `T`

#### **`one_hot_sel`**

Selects between operands based on a one-hot selector, `OR`-ing all selected
cases if more than one case is selected.

See `one_hot` for an example of the one-hot selector invariant. Note that when
the selector is not one-hot, this operation is still well defined.

Note that when `one_hot` operations are used to precondition the `selector`
operand to `one_hot_sel`, the XLS optimizer will try to determine when they are
unnecessary and subsequently eliminate them.

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

#### **`priority_sel`**

Selects between operands based on a selector, choosing the highest-priority case
if more than one case is selected. Each bit in the selector corresponds to a
case, with the least significant bit corresponding to the first case and having
the highest priority. If there are no bits in the selector set, no case is
selected and the default value of 0 is chosen.

See `one_hot` for an example of the one-hot selector invariant. Note that when
the selector is not one-hot, this operation is still well defined.

Note that when `one_hot` operations are used to precondition the `selector`
operand to `priority_sel`, the XLS optimizer will try to determine when they are
unnecessary and subsequently eliminate them.

**Syntax**

```
result = priority_sel(selector, cases=[case_{0}, ... , case_{N-1}])
```

**Types**

Value      | Type
---------- | ---------
`selector` | `bits[N]`
`case_{i}` | `T`
`result`   | `T`

The result is the first case `case_{i}` for which the corresponding bit `i` is
set in the selector. If the selector is known to be one-hot, then the
`priority_sel()` operation is equivalent to a `one_hot_sel()`.

#### **`invoke`**

Invokes a function. The return value for the invoked function is the result
value.

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

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword    | Type     | Required | Default | Description                    |
| ---------- | -------- | -------- | ------- | ------------------------------ |
| `to_apply` | `string` | yes      |         | Name of the function to use as the loop body |

<!-- mdformat on -->

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

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword    | Type     | Required | Default | Description                    |
| ---------- | -------- | -------- | ------- | ------------------------------ |
| `to_apply` | `string` | yes      |         | Name of the function to apply to each element of the operand |

<!-- mdformat on -->

#### **`dynamic_counted_for`**

Invokes a dynamic-trip count loop.

**Syntax**

```
result = counted_for(init, trip_count, stride, body=<body>, invariant_args=<inv_args>)
```

**Types**

Value        | Type
------------ | ------------------------------
`init`       | `T`
`trip_count` | `bits[N], treated as unsigned`
`stride`     | `bits[M], treated as signed`,
`result`     | `T`

**Keyword arguments**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword          | Type     | Required | Default | Description               |
| ---------------- | -------- | -------- | ------- | ------------------------- |
| `invariant_args` | array of | yes      |         | Names of the invariant operands as the loop body |
| `body`           | `string` | yes      |         | Name of the function to use as the loop body |

<!-- mdformat on -->

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

Code generation support for `dynamic_counted_for` is limited because the
pipeline generator cannot handle an unknown trip count.

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

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword          | Type     | Required | Default | Description               |
| ---------------- | -------- | -------- | ------- | ------------------------- |
| `trip_count`     | `int64_t`  | yes      |         | Trip count of the loop (number of times that the loop body will be executed) |
| `stride`         | `int64_t`  | no       | 1       | Stride of the induction variable |
| `invariant_args` | array of | yes      |         | Names of the invariant operands as the loop body |
| `body`           | `string` | yes      |         | Name of the function to use as the loop body |

<!-- mdformat on -->

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

### Sequencing operations

Some operations in XLS IR are sensitive to sequence order, similar to
[channel operations](#channel-operations), but are not themselves
channel-related. Tokens are used to determine the possible sequencing of these
effects, and `after_all` can be used to join together tokens as a sequencing
merge point for concurrent threads of execution described by different tokens.

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

### Other side-effecting operations

Aside from channels operations such as `send` and `receive` several other
operations have side-effects. Care must be taken when adding, removing, or
transforming these operations, e.g., in the optimizer.

#### **`assert`**

Raises an error at software run-time (DSLX/IR interpretation, JIT execution, RTL
simulation) if the given condition evaluates to false. The operation takes a
literal string attribute which is included in the error message. This is a
software-only operation and has no representation in the generated hardware.
Tokens are used to connect the operation to the graph and order with respect to
other side-effecting operations.

```
result = assert(tkn, condition, message=<string>)
result = assert(tkn, condition, message=<string>, label=<string>)
```

**Types**

Value       | Type
----------- | ---------
`tkn`       | `token`
`condition` | `bits[1]`
`result`    | `token`

**Keyword arguments**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword   | Type              | Required | Default | Description     |
| --------- | ----------------- | -------- | ------- | --------------- |
| `message` | `string`          | yes      |         | Message to include in raised error |
| `label`   | `optional string` | yes      |         | Label to associate with the assert statement in the generated (System)Verilog |

<!-- mdformat on -->

#### **`cover`**

Records the number of times the given condition evaluates to true. Just like
`assert`, this is a software-only construct and is not emitted in a final
hardware design. Tokens are used to sequence this operation in the graph.

```
result = cover(tkn, condition, label=<string>)
```

**Types**

Value       | Type
----------- | ---------
`tkn`       | `token`
`condition` | `bits[1]`
`result`    | `token`

**Keyword arguments**

Keyword | Type     | Required | Default | Description
------- | -------- | -------- | ------- | ---------------------------------
`label` | `string` | yes      |         | Name associated with the counter.

#### **`gate`**

Gates an arbitrarily-typed value based on a condition.

The result of the operation is the data operand if the condition is true,
otherwise the result is a zero value of the type of the data operand (i.e., the
value is gated off). A helpful mnemonic is to think of this as analogous to an
`AND` gate: if the condition is `true`, the value passes through, otherwise it's
zeroed.

This operation can reduce switching and may be used in power optimizations. This
is intended for use in operand gating for power reduction, and the compiler may
ultimately use it to perform register-level load-enable gating.

The operation is considered side-effecting to prevent removal of the operation
when the gated result (condition is false) is not observable. The 'side-effect'
of this operation is the effect it can have on power consumption.

Despite being 'side-effecting' this operation is special cased to still be
eligible for total removal by various passes. This will only be done in cases
where the gate is redundant, for example the condition is known to be false or
the data is known to be zero.

```
result = gate(condition, data)
```

**Types**

Value       | Type
----------- | ---------
`condition` | `bits[1]`
`data`      | `T`
`result`    | `T`

### RTL-level operations

These IR operations correspond to RTL-level constructs in the emitted Verilog.
These operations are added and used in the code generation process and may only
appear in blocks (not procs or functions).

#### **`input_port`**

Corresponds to an input port on a Verilog module.

**Syntax**

```
result = input_port()
```

**Types**

Value    | Type
-------- | ----
`result` | `T`

An input_port operation can be an arbitrary type.

#### **`output_port`**

Corresponds to an output port on a Verilog module. The value sent to the output
port is the data operand.

**Syntax**

```
result = output_port(data)
```

**Types**

Value    | Type
-------- | ----
`data`   | `T`
`result` | `T`

#### **`register_read`**

Reads a value from a register.

The register is defined on the block.

**Syntax**

```
result = register_read(register=<register_name>)
```

**Types**

Value    | Type
-------- | ----
`result` | `T`

The type `T` of the result of the operation is the type of the register.

**Keyword arguments**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword    | Type     | Required | Default | Description                  |
| ---------- | -------- | -------- | ------- | ---------------------------- |
| `register` | `string` | yes      |         | Name of the register to read |

<!-- mdformat on -->

#### **`register_write`**

Writes a value to a register.

The write to the register may be conditioned upon an optional load-enable and/or
reset signal. The register is defined on the block.

If `reset` is given the `register` associated with this read **must** have a
reset behavior set.

If the `reset` value matches the reset-active value of the register then the
`reset_value` of the register is written and the `data` is ignored.

If the `load_enable` argument is present the register will only be written if
the argument evaluates to `1`, remaining unchanged otherwise (i.e. if present it
is equivalent to `register_write.REG(sel(load_enable, {register_read.REG,
data}))`).

The `reset` and `load_enable` arguments affect the value written according to
the following table.

| `Register` reset      | `reset` value | `load_enable` value | new value     | {.sortable}
: behavior              :               :                     :               :
| --------------------- | ------------- | ------------------- | ------------- |
| `active_low == false` | `false` / `0` | not present         | `data`        |
| `active_low == false` | `true` / `1`  | not present         | `reset_value` |
| `active_low == true`  | `false` / `0` | not present         | `reset_value` |
| `active_low == true`  | `true` / `1`  | not present         | `data`        |
| `active_low == false` | `false` / `0` | `true` / `1`        | `data`        |
| `active_low == false` | `true` / `1`  | `true` / `1`        | `reset_value` |
| `active_low == true`  | `false` / `0` | `true` / `1`        | `reset_value` |
| `active_low == true`  | `true` / `1`  | `true` / `1`        | `data`        |
| `active_low == false` | `false` / `0` | `false` / `0`       | No change     |
| `active_low == false` | `true` / `1`  | `false` / `0`       | `reset_value` |
| `active_low == true`  | `false` / `0` | `false` / `0`       | `reset_value` |
| `active_low == true`  | `true` / `1`  | `false` / `0`       | No change     |
| not present           | not present   | `true` / `1`        | `data`        |
| not present           | not present   | `false` / `0`       | No change     |

**Syntax**

```
result = register_write(data, load_enable=<load_enable>, reset=<reset>, register=<register_name>)
```

**Types**

Value         | Type
------------- | --------------------
`data`        | `T`
`load_enable` | `bits[1]` (optional)
`reset`       | `bits[1]` (optional)
`result`      | `()` (empty tuple)

**Keyword arguments**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword    | Type     | Required | Default | Description                   |
| ---------- | -------- | -------- | ------- | ----------------------------- |
| `register` | `string` | yes      |         | Name of the register to write |

<!-- mdformat on -->

The type `T` of the data operand must be the same as the the type of the
register.

#### **`instantiation_input`**

Corresponds to a single input port of an instantiation.

An instantiation is a block-scoped construct that represents a module
instantiation at the Verilog level. Each `instantation_input` operation
corresponds to a particular port of the instantiated object, so generally a
single instantiation can have multiple associated `instantiation_input`
operations (one for each input port).

**Syntax**

```
result = instantiation_input(data, instantiation=<instantiation>, port_name=<port_name>)
```

**Types**

Value    | Type
-------- | ----
`data`   | `T`
`result` | `()`

**Keyword arguments**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword         | Type     | Required | Default | Description                 |
| --------------- | -------- | -------- | ------- | --------------------------- |
| `instantiation` | `string` | yes      |         | Name of the instantiation.  |
| `port_name`     | `string` | yes      |         | Name of the associated port of the instantiation. |

<!-- mdformat on -->

The type `T` of the data operand must be the same as the the type of the
associated input port of the instantiated object.

#### **`instantiation_output`**

Corresponds to a single output port of an instantiation.

An instantiation is a block-scoped construct that represents a module
instantiation at the Verilog level. Each `instantation_output` operation
corresponds to a output particular port of the instantiated object, so generally
a single instantiation can have multiple associated `instantiation_output`
operations (one for each output port).

**Syntax**

```
result = instantiation_output(instantiation=<instantiation>, port_name=<port_name>)
```

**Types**

Value    | Type
-------- | ----
`result` | `T`

**Keyword arguments**

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Keyword         | Type     | Required | Default | Description                 |
| --------------- | -------- | -------- | ------- | --------------------------- |
| `instantiation` | `string` | yes      |         | Name of the instantiation.  |
| `port_name`     | `string` | yes      |         | Name of the associated port of the instantiation. |

<!-- mdformat on -->

The type `T` of the result is type of the associated output port of the
instantiated object.
