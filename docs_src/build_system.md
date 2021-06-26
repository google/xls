# Build system

XLS uses the [Bazel](http://bazel.build) build system for itself and all its
dependencies. Bazel is an easy to configure and use, and has powerful extension
facilities. (It's also
[well-documented](https://docs.bazel.build/versions/master/bazel-overview.html)!)
XLS provides a number of
[Starlark](https://docs.bazel.build/versions/master/skylark/language.html)
[rules](https://docs.bazel.build/versions/master/skylark/rules.html) and
[macros](https://docs.bazel.build/versions/master/skylark/macros.html) to define
a build flow.

[TOC]

## Rules

Below is a brief summary of the public rules. An extensive description of the
rules is available at
[xls/build_rules/xls_build_defs.bzl](https://github.com/google/xls/tree/main/xls/build_rules/xls_build_defs.bzl).
Examples using the rules are found at
[xls/build_rules/tests/BUILD](https://github.com/google/xls/tree/main/xls/build_rules/tests/BUILD).

### `xls_dslx_library`

A build rule that parses and type checks DSLX source files.

### `xls_dslx_module_library`

A build rule that parses and type checks a DSLX source file. The rule is the
gateway to other translation rules such as xls_dslx_ir and xls_dslx_verilog.

Currently, the DSLX executables (interpreter and converter) require a
'main source file' to perform evaluation and conversion. As a result, the
xls_dslx_module_library was introduced that includes an entry to indicate the
'main source file'.

This rule is an interim solution. When the DSLX executables (interpreter and
converter) are updated where the 'main source file' is not required, the
xls_dslx_module_library can be removed.

### `xls_dslx_test`

A test rule that executes the tests and quick checks of a DSLX source file.

### `xls_dslx_ir`

A build rule that converts a DSLX source file to an IR file.

### `xls_benchmark_ir`

An execution rule that executes the benchmark tool on an IR file.

### `xls_ir_equivalence_test`

A test rule that executes the equivalence tool on two IR files.

### `xls_eval_ir_test`

A test rule that executes the IR interpreter on an IR file.

### `xls_ir_opt_ir`

A build rule that optimizes an IR file.

### `xls_ir_verilog`

A build rule that generates a Verilog file from an IR file.

### `xls_ir_jit_wrapper`

A build rule that generates the sources for JIT invocation wrappers.

### `xls_dslx_verilog`

A build rule that generates a Verilog file from a DSLX source file. The rule
executes the `xls_dslx_ir`, `xls_ir_opt_ir` and `xls_ir_verilog` rules in that
order.

### `xls_dslx_opt_ir`

A build rule that generates an optimized IR file from a DSLX source file. The
rule executes the `xls_dslx_ir` and `xls_ir_opt_ir` rules in that order.

### `xls_dslx_opt_ir_test`

A test rule that executes the commands in the order presented in the list for
the following rules:

   1. xls_dslx_test

   2. xls_ir_equivalence_test

   3. xls_eval_ir_test

   4. xls_benchmark_ir

## Macros

Below is a brief summary of the public macros. An extensive description of the
macros is available at
[xls/build_rules/xls_build_defs.bzl](https://github.com/google/xls/tree/main/xls/build_rules/xls_build_defs.bzl).
Examples using the macros are found at
[xls/build_rules/tests/BUILD](https://github.com/google/xls/tree/main/xls/build_rules/tests/BUILD).

### `get_mangled_ir_symbol`

Returns the mangled IR symbol for the module/function combination.

### `cc_xls_ir_jit_wrapper`

The macro generates sources files (.cc and .h) using the xls_ir_jit_wrapper
rule. The source files are the input to a cc_library target with the same name
as this macro.

### `xls_ir_verilog_macro`

The macro instantiates the 'xls_ir_verilog' rule and
'enable_generated_file_wrapper' function. The generated files of the rule are
listed in the outs attribute of the rule. These generated files can be
referenced in other rules.

### `xls_dslx_verilog_macro`

The macro instantiates the 'xls_dslx_verilog' rule and
'enable_generated_file_wrapper' function. The generated files of the rule are
listed in the outs attribute of the rule. These generated files can be
referenced in other rules.

## Bazel queries

Understanding the build tree for a new project can be difficult, but fortunately
Bazel provides a
[powerful query mechanism](https://docs.bazel.build/versions/master/query.html).
`bazel query` enables a user to examine build targets, dependencies between
them, and much more. A few usage examples are provided here, but the full
documentation (linked above) is comprehensive.

### Finding transitive dependencies

To understand why, for example, the combinational verilog generator depends on
the ABSL container algorithm library, one could run:

```
$ bazel query 'somepath(//xls/codegen:combinational_generator, @com_google_absl//absl/algorithm:container)'
//xls/codegen:combinational_generator
//xls/codegen:vast
@com_google_absl//absl/algorithm:container
```

This result shows that one such path goes through the `:vast` target. Another
such path goes through the xls/ir:ir target, then the xls/ir:value target.
`somepath` provides _some_ path, not all paths (that's what `allpaths` is for).

### Finding dependees ("reverse dependencies")

Sometimes it's useful to identify the set of targets depending on some other
target - the `rdeps` query performs this:

```
$ bazel query 'rdeps(//xls/codegen:all, //xls/codegen:combinational_generator)'
//xls/codegen:flattening_test
//xls/ir:ir_test_base
//xls/codegen:combinational_generator_test
//xls/codegen:combinational_generator
```

This shows the transitive closure of all dependencies of the combinational
generator, with the starting set being all targets in `//xls/codegen:all`. This
set of dependees can quickly grow to be unmanageable, so keep the initial set
(the first argument) as small as possible, and consider specifying a third
argument for maximum search depth.
