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

Arrays are flattened such that the zero-th element occupies the most significant
bits. Elements are concatenated via the SystemVerilog concatenation operation.

### Tuples

Tuples are flattened by concatenating each leaf element of the tuple. If an
element of a tuple is an array, this is equivalent to first flattening the array
and treating the flattened array as a leaf element. Concatenation is performed
via the SystemVerilog concatenation operation. The zero-th tuple element will
end up occupy the most significant bits in the flattened output.

### Structs

DSXL structs are lowered to tuples in the IR, so there's no separate handling of
structs.

## Unrepresented Ops

Tokens are not represented in RTL.

Entities with zero width are unrepresented in RTL. Where a zero-width value is
used, a zero-valued literal can be substituted.

Asserts and covers are only represented when producing SystemVerilog, and are
unrepresented when producing Verilog.
