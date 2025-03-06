# XLS: IR Lowering

[TOC]

As part of codegen, IR constructs are lowered to Verilog/SystemVerilog
constructs. In some cases, this lowering is simple and direct, but some IR
constructs don't map directly to Verilog constructs.

## Flattening

Some XLS types are flattened into simpler types when IR gets lowered to RTL.
This flattening is performed
[here](https://github.com/google/xls/tree/main/xls/codegen/flattening.cc) (also see
[the more-commented header file](https://github.com/google/xls/tree/main/xls/codegen/flattening.h)).
The following summarizes how types are flattened.

### Arrays

Arrays are flattened such that the last element occupies the most significant
bits. Elements are concatenated via the SystemVerilog concatenation operation.
This matches the flattening of SystemVerilog packed arrays (e.g. `logic [N:0]
foo`) to unpacked arrays declared like `logic bar[N:0]`. For example, a
4-element array of 4-bit UInts would be flattened as

```
[0x3, 0x4, 0x5, 0x6] => 0x6543
```

### Tuples

Tuples are flattened by concatenating each leaf element of the tuple. If an
element of a tuple is an array, this is equivalent to first flattening the array
and treating the flattened array as a leaf element. Concatenation is performed
via the SystemVerilog concatenation operation. The zero-th tuple element will
end up occupying the most significant bits in the flattened output. For example,
a 4-tuple of 4-bit UInts would be flattened as

```
(0x3, 0x4, 0x5, 0x6) => 0x3456
```

and a 2-tuple of length 2 arrays of 4-bit UInts would be flattened as

```
([0x3, 0x4], [0x5, 0x6]) => 0x4365
```

### Structs

DSLX structs are lowered to tuples in the IR, so there's no separate handling of
structs.

## Unrepresented Ops

Tokens are not represented in RTL.

Entities with zero width are unrepresented in RTL. Where a zero-width value is
used, a zero-valued literal can be substituted.

Asserts and covers are only represented when producing SystemVerilog, and are
unrepresented when producing Verilog.

## Assertions

For combinational blocks, assertions are emitted as SystemVerilog deferred
immediate assertions of the form:

```
[LABEL:] assert final (DISABLE_IFF || CONDITION) else $fatal(0, message)
```

For pipelined blocks, assertions are emitted as SystemVerilog concurrent
assertions of the form:

```
[LABEL:] assert property (
    @(CLOCKING_EVENT)
    [disable iff DISABLE_IFF] CONDITION)
  else $fatal(0, message);
```

If reset is present and asynchronous, the `DISABLE_IFF` expression is wrapped in
a call to `$sampled()`.

Note that the default codegen of assertions can be overridden via the
`--assert_format` codegen options, even if the output format is Verilog.
