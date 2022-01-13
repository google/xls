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

### `xls_benchmark_ir`

An execution rule that executes the benchmark tool on an IR file.

### `xls_ir_equivalence_test`

A test rule that executes the equivalence tool on two IR files.

### `xls_eval_ir_test`

A test rule that executes the IR interpreter on an IR file.

### `xls_dslx_opt_ir_test`

A test rule that executes the commands in the order presented in the list for
the following rules:

   1. xls_dslx_test

   1. xls_ir_equivalence_test

   1. xls_eval_ir_test

   1. xls_benchmark_ir

## Macros

Below is a brief summary of the public macros. An extensive description of the
macros is available at
[xls/build_rules/xls_build_defs.bzl](https://github.com/google/xls/tree/main/xls/build_rules/xls_build_defs.bzl).
Examples using the macros are found at
[xls/build_rules/tests/BUILD](https://github.com/google/xls/tree/main/xls/build_rules/tests/BUILD).

### `get_mangled_ir_symbol`

Returns the mangled IR symbol for the module/function combination.

### `cc_xls_ir_jit_wrapper`

The macro instantiates a macro that generates the source files (.cc and .h) for
JIT invocation wrappers and the 'enable_generated_file_wrapper' function. The
source files are the input to a cc_library target with the same name as this
macro.

### `xls_dslx_ir`<a id="xls_dslx_ir"></a>

The macro instantiates a rule that converts a DSLX source file to an IR file and
the 'enable_generated_file_wrapper' function. The generated files of the rule
are listed in the outs attribute of the rule. These generated files can be
referenced in other rules.

### `xls_ir_opt_ir`<a id="xls_ir_opt_ir"></a>

The macro instantiates a rule that optimizes an IR file and the
'enable_generated_file_wrapper' function. The generated files of the rule are
listed in the outs attribute of the rule. These generated files can be
referenced in other rules.

### `xls_ir_verilog`<a id="xls_ir_verilog"></a>

The macro instantiates a rule that generates a Verilog file from an IR file and
the 'enable_generated_file_wrapper' function. The generated files of the rule
are listed in the outs attribute of the rule. These generated files can be
referenced in other rules.

### `xls_dslx_opt_ir`

The macro instantiates a rule that:

1. converts a DSLX source file to an IR file ([xls_dslx_ir](#xls_dslx_ir)), and,
1. optimizes the IR file ([xls_ir_opt_ir](#xls_ir_opt_ir)).

The macro also instantiates the 'enable_generated_file_wrapper' function. The
generated files of the rule are listed in the outs attribute of the rule. These
generated files can be referenced in other rules.

### `xls_dslx_verilog`

The macro instantiates a rule that:

1. converts a DSLX source file to an IR file ([xls_dslx_ir](#xls_dslx_ir)),
1. optimizes the IR file ([xls_ir_opt_ir](#xls_ir_opt_ir)), and,
1. generates a Verilog file from an IR file ([xls_ir_verilog](#xls_ir_verilog)).

The macro also instantiates the 'enable_generated_file_wrapper' function. The
generated files of the rule are listed in the outs attribute of the rule. These
generated files can be referenced in other rules.

### `xls_verify_checksum`

Helper macro for checksumming files.

As projects cut releases or freeze, it's important to know that generated (e.g.
Verilog) code is never changing without having to actually check in the
generated artifact. This macro performs a checksum of generated files as an
integrity check. Users might use this macro to help enable confidence that there
is neither:

*   non-determinism in the toolchain, nor
*   an accidental dependence on a non-released toolchain (e.g. an accidental
    dependence on top-of-tree, where the toolchain is constantly changing)

Say there was a codegen rule producing `my_output.v`, a user might instantiate
something like:

```starlark
xls_verify_checksum(
    name = my_output_checksum",
    src = ":my_output.v",
    out = "my_output.frozen.v",
    sha256 = "d1bc8d3ba4afc7e109612cb73acbdddac052c93025aa1f82942edabb7deb82a1",
)
```

... and then take a dependency on `my_output.frozen.v` in the surrounding
project, knowing that it had been checksum-verified.

Taking a dependence on `my_output.v` directly may also be ok if the
`:my_output_checksum` target is also built (e.g. via the same wildcard build
request), but taking a dependence on the output `.frozen.v` file ensures that
the checking is an integral part of the downstream build-artifact-creation
process.

This macro also notably creates a test (in the package that instantiates it),
and so any wildcard testing of the directory will also check that the sha256
integrity holds.

Note that this mechanism will only checksum the Verilog source file itself --
in the future as XLS can pull in "foreign" Verilog constructs via tick-includes
or externally managed file lists, this mechanism will not guarantee the external
world also remains the same. As of the time of this writing, however, XLS
generated modules are self-contained, but foreign-function instantiation of e.g.
FIFOs and external modules _are_ expected upcoming features.

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
