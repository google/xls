# Data layout

For many uses, XLS types exist within their own conceptual space or domain, so
"portability" concerns don't exist. When interacting with the JIT, however, XLS
and host-native types must interact, so the data layouts of both must be
understood and possibly reconciled.

[TOC]

## XLS data layout

The concrete XLS [`Bits`](https://github.com/google/xls/tree/main/xls/ir/bits.h) type is
the ultimate container of actual data for any XLS IR type: tuples and arrays may
be contain any number of tuple, array, or bits types, but whatever the layout of
the type tree, all leaf nodes are Bits. When accessing the underlying storage of
a `Bits` via the `ToBytes()` member function, the results are returned in a
little-endian layout, i.e, with the least-significant data elements stored in
the lowest addressable location. For example, the 32-bit value 12,345,678
(0xBC614E), would be returned as:

```
High  <--  Low
0x 00 BC 61 4E
```

## Host data layout

Different architectures can use different native layouts. For example, x86 (and
descendants) use little-endian (i.e., `0x00 BC 61 4E`), and modern ARM can be
configurable as either. (There are actually other layouts, but they're best left
to the dustbin of history).

## JIT data layout

From the above, we can see that XLS' native layout differs from that of most
modern hosts. When compiling XLS code, the [LLVM] JIT understandably uses the
host's native layout. What this means is that any data fed into the JIT from XLS
will need to be byte-swapped before ingestion.

For Value or unpacked view input, this swapping is handled automatically, in
`LlvmIrRuntime::PackArgs()` (via `LlvmIrRuntime::BlitValueToBuffer()`) - and the
**un**swapping is also automatically performed in
`LlvmIrRuntime::UnpackBuffer()`. Thus, for these uses, no special action is
required of the user.

## Packed views

*However*, this is not the case for use of packed views. The motivating use case
for packed views is to allow users to map native types directly into JIT-usable
values - for example, to use an IEEE float32 (e.g., a C `float`) *directly*,
without needing to be exploded into a `bits[1]` for the sign, a `bits[8]` for
the exponent, and a `bits[23]` for the fractional part.

When creating a packed view from a C `float`, no special action is needed - that
`float` is in native host layout, which is the layout used by the JIT. If,
however, data is coming from XLS (perhaps a `float` converted into a Value,
manipulated in some way, then passed into the JIT), then the *user* must un-swap
the bits back to native layout. This is because the JIT has no way of knowing
the provenance of that data (if it's from a native type or XLS), so it's up to
the provider of that data to ensure proper layout.

Distilled into a simple rule of thumb: if packed view data is coming from XLS,
it needs to be byte swapped before being passed into the JIT.

### Tuple types

Another wrinkle is the usage of packed tuple views. When XLS emits a tuple type
in Verilog, the first element in the tuple declaration is placed in the most
significant bits, and so on, with the last-declared element placed in the least
significant bits. To match this layout, PackedTupleView elements must be also
declared from most significant to least significant element. This way, when
running on a host, the in-memory layout of input data matches that expected by
XLS tools. That means, using the usual float32 example, that the packed view
declaration is:

```
PackedTupleView<PackedBitsView<1>, PackedBitsView<8>, PackedBitsView<23>>
```

(The non-packed-View tuple declaration is much the same, but matters less, as it
doesn't directly correspond to in-memory data layout.)

Be aware of this layout when accessing elements in a PackedTupleView. In the
float32 above, accessing element 0 yields the sign bit (the most significant bit
in memory), and accessing element 2 yields the fractional part (the least
significant 23 bits in memory), as one would expect given the tuple type
declaration order.

While this may initially seem confusing, it suffices to remember that
PackedTupleView element declaration order is the "reverse" of the in-memory
order; refer to
[value_view_test.cc](https://github.com/google/xls/tree/main/xls/ir/value_view_test.cc)
and
[function_jit_test.cc](https://github.com/google/xls/tree/main/xls/jit/function_jit_test.cc)
for test examples, or the [generated] float32_add_jit_wrapper.h/cc and
[float32_add_test.cc](https://github.com/google/xls/tree/main/xls/dslx/stdlib/test/float32_add_test.cc)
for practical usage.
