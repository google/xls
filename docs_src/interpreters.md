# Interpreters

XLS provides a several interpreters to assist in design validation across our
functional stack, from input DSLX down to the netlist level.

[TOC]

## DSLX

The DSLX interpreter (`//xls/dslx:interpreter_main`) operates on
DSLX `.x` files that contain both the design and unit tests to execute (present
as `#[test]` annotated functions).

The [adler32](https://github.com/google/xls/tree/main/xls/examples/adler32/adler32.x)
example demonstrates this: the design is encapsulated in the `main`,
`adler32_seq`, and `mod` functions, and the samples are present in the test
`adler32_one_char` (note that unit-style tests/interpretations of `adler32_seq`
and `mod` could also be present).

Interpreter targets are automatically generated for `dslx_test()` targets, so no
special declarations are necessary to wrap DSLX code.

To invoke these samples, execute the following:

```
bazel build -c opt //xls/examples/adler32:adler32_dslx_test
./bazel-bin/xls/examples/adler32/adler32_dslx_test
```

To execute directly via the interpreter, you can instead run:

```
$ bazel build -c opt //xls/dslx/interpreter_main
$ ./bazel-bin/xls/dslx/interpreter_main \
    ./xls/examples/adler32/adler32.x
```

These two methods are equivalent.

### Execution comparison

The DSL interpreter provides a flag, `--compare`, to implicitly compare its run
results to those of the IR-converted DSL functions. This helps "spot check"
consistency between IR and DSL execution (in addition to other methods used in
more generally in XLS, like the fuzzer).

The user may compare DSL execution to IR interpreter execution, IR JIT
execution, or not perform IR comparison at all.

```console
$ ./bazel-bin/xls/dslx/interpreter_main \
    ./xls/examples/adler32/adler32.x --compare=jit
$ ./bazel-bin/xls/dslx/interpreter_main \
    ./xls/examples/adler32/adler32.x --compare=interpreter
$ ./bazel-bin/xls/dslx/interpreter_main \
    ./xls/examples/adler32/adler32.x --compare=none
```

## IR

XLS provides two means of evaluating IR - interpretation and native host
compilation (the
[JIT](./ir_jit.md)). Both are
invoked in nearly the same way, via the
[`eval_ir_main`](https://github.com/google/xls/tree/main/xls/tools/eval_ir_main.cc) tool.

`eval_ir_main` supports a wide number of use cases, but the most common end-user
case will be to run a sample through a design. To evaluate a sample (1.0 + 2.5)
on the add function in
[floating-point adder](https://github.com/google/xls/tree/main/xls/dslx/stdlib/float32.x),
one would run the following:

```
bazel build -c opt //xls/tools:eval_ir_main
./bazel-bin/xls/tools/eval_ir_main    \
--input '(bits[1]: 0x0, bits[8]:0x7F, bits[23]:0x0); (bits[1]: 0x0, bits[8]:0x80, bits[23]:0x200000)'   \
./bazel-bin/xls/dslx/stdlib/float32_add.opt.ir
```

By default, this runs via the JIT. To use the interpreter, add the
`--use_llvm_jit=false` flag to the invocation.

`eval_ir_main` supports a broad set of options and modes of execution. Refer to
its [very thorough] `--help` documentation for full details.

## Netlists

Finally, compiled netlists can also be interpreted against input samples via the
aptly-named
[`netlist_interpreter_main`](https://github.com/google/xls/tree/main/xls/netlist/netlist_interpreter_main.cc)
tool. This tool currently only supports single sample evaluation (as illustrated
in the IR section above):

```
bazel build -c opt //xls/tools:netlist_interpreter_main
./bazel-bin/xls/netlist/netlist_interpreter_main \
  --netlist <path to netlist>
  --module  <module to evaluate>
  --cell_library[_proto] <path to the module's cell library [proto]>
  --inputs  <input sample, as above>
```

As XLS does not currently provide an sample/example netlist (TODO(rspringer)),
concrete values can't [yet] be provided here. The `--cell_library` flag merits
extra discussion, though.

During netlist compilation, a cell library is provided to indicate the
individual logic cells available for the design, and these cells are referenced
in the output netlist. The interpreter needs a description of these cells'
behaviors/functions, so the cell library must be provided here, as well. Many
cell libraries are very large (> 1GB), and can thus incur significant processing
overhead at startup, so we also accept pre-processed cell libraries, as
[`CellLibraryProto`](https://github.com/google/xls/tree/main/xls/netlist/netlist.proto)
messages, that contain much-abridged cell descriptions. The
[`function_extractor_main`](https://github.com/google/xls/tree/main/xls/netlist/function_extractor_main.cc)
tool can automatically perform this extraction for
[Liberty](https://www.synopsys.com/community/interoperability-programs/tap-in.html)-formatted
cell library descriptions.
