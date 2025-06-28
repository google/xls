# IR JIT Compiler

[TOC]

XLS provides a JIT compiler for evaluating functions written in the [XLS]
compiler intermediate representation (IR) at native machine speed.

## Usage

Given a DSLX file and build target, one can build and run it through the JIT by:

1.  Declaring a
    [`cc_xls_ir_jit_wrapper`](https://github.com/google/xls/tree/main/xls/build_rules/xls_build_defs.bzl)
    target matching the DSLX build target.
2.  Creating a JIT object and calling its `Run()` method. Using the 2-way
    floating-point adder as an example:

    ```c
     # include "xls/modules/fpadd_2x24_jit_wrapper.h"

     absl::StatusOr<Value> foo(Value a, Value b) {
       ...
       // Only create this once and re-use it; it's created here just as
       // an illustration.
       XLS_ASSIGN_OR_RETURN(Fpadd2x32 adder, Fpadd2x32::Create()); return
       adder.Run(a, b);
     }
    ```

The advantages of JIT compilation (or any compilation, for that matter) only
come into play when repeatedly using the compiled object, so programs should be
structured to create a JIT wrapper once and to reuse it many times, e.g., to
test a module across many - or even exhaustively, across all possible - inputs.

### Specialized matching

In many cases, the types used by DSLX designs map to native types, such as the
C/C++ `float`. In that case, a simplified wrapper call is available:

```
#include "xls/modules/fpadd_2x24_jit_wrapper.h"

absl::StatusOr<float> foo(float a, float b) {
  ...
  // Only create this once and re-use it; it's created here just as
  // an illustration.
  XLS_ASSIGN_OR_RETURN(Fpadd2x32 adder, Fpadd2x32::Create());
  return adder.Run(a, b);
}
```

When available, these simplified wrappers should be used for higher performance
(~30% in our measured cases). Currently, floats and integral types >= 64 bits in
this way. For non-native integral types, the generated wrapper will accept the
next larger native type e.g., `uint64_t` for `bits[47]`. Proper operation with
next-larger types depends on the input value being present in the
least-significant bits of the containing type.

These are higher performance because they avoid unnecessary marshaling of these
types into Views (e.g., a `float` outside the JIT -> View -> `float` inside the
JIT).

### Direct usage

The JIT is also available as a library with a straightforward interface:

```
absl::StatusOr<Value> RunOnJit(
    Function* function, absl::Span<const Value> args) {
  XLS_ASSIGN_OR_RETURN(auto jit, FunctionJit::Create(function));
  return jit->Run(args);
}
```

The IR JIT is the default backend for the
[eval_ir_main](./tools.md#ir-eval)
tool, which loads IR from disk and runs with args present on either the command
line or in a specified file.

## Design

Internally, the JIT converts XLS IR to LLVM IR and uses
[LLVM's ORC infrastructure](https://llvm.org/docs/ORCv2.html) to convert that
into native machine code. The details of compiling an LLVM IR program with ORC
are mostly generic and are available online - here are discussed details
specific to our usage in XLS.

XLS IR is converted to LLVM IR by recursively visiting every node in a function
using the DfsVisitor functions. Most nodes have relatively straightforward
implementations, e.g., in Concat, we create an empty value with the combined
width of the operands, and each is shifted and blitted into that value to
produce the result. Some operations, though, merit more discussion.

### Arg passing

When LLVM JIT-compiles a program, the resulting value is simply a function
pointer to the requested entry point (that can be called like any function
pointer). Calling such a function pointer with concrete-typed arguments, though,
is difficult: one must either make heavy [ab]use of C++ templates or "hide" the
argument types behind an opaque pointer. The latter approach is taken here.

When a compiled function is invoked (via `FunctionJit::Run()`), the typed input
args are "packed" into an opaque byte buffer which is passed into the new
function. Inside there, any references to an argument (via
`DfsVisitor::HandleParam()`) calculate the offset of that param in the opaque
buffer and load from there appropriately (this should happen at most once per
arg; LLVM and/or XLS should optimize away redundant loads).

A special case is for function invocations (inside the JITted function): for
these, the arguments already exist inside "LLVM-space", so there's no need for
unpacking args, so `LlvmFunction::getArg()` can be used as usual.

Results must be handled in a similar way - they could be of any type and will
need to be packed inside XLS types before returning, so there's a corresponding
argument unpacking phase at function exit.

For both packing and unpacking, LLVM's DataLayout must be used to determining
where input and output values will be placed, as LLVM will use those conventions
when, e.g., loading values from a struct.

### ArrayIndex

IRBuilder provides three means of extracting values from an aggregate type:

1.  `CreateGEP`: these use the getelementptr instruction, which requires a
    pointer-typed value (not the same thing as an array!). This requires holding
    a value in a specially-created allocation (via `CreateAlloca()` or in an
    input buffer).
1.  `CreateExtractElement`: returns the value at a given index in a
    *vector*-typed value.
1.  `CreateExtractValue`: returns the value at a given *constant index* in an
    aggregate value.

Unfortunately, #2 doesn't apply, as arrays aren't LLVM vectors, and #3 doesn't
apply, as an array index isn't necessarily a constant value. Uniformly managing
arrays as allocas doesn't scale well (consider the case of arrays of arrays of
tuples...), so for `ArrayIndex` nodes, we lazily create allocas for *only the
array of interest* and load the requested index from there.
