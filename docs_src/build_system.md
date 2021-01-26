# Build system

XLS uses the [Bazel](http://bazel.build) build system for itself and all its
dependencies. Bazel is an easy to configure and use, and has powerful extension
facilities. (It's also
[well-documented](https://docs.bazel.build/versions/master/bazel-overview.html)!)
XLS provdes a number of
[Starlark](https://docs.bazel.build/versions/master/skylark/language.html)
[macros](https://docs.bazel.build/versions/master/skylark/macros.html),
described below, to simplify build target definition.

[TOC]

## Macros

Below are summaries of public macro functionality. Full documentation is
available in
[xls/build/build_defs.bzl](https://github.com/google/xls/tree/main/xls/build/build_defs.bzl).

### `dslx_codegen`

This macro generates Verilog from an DSLX target (currently given as a
`dslx_test` target). This target also accepts several key/value parameters (as
`configs` for generation:

*   `clock_period_ps`: The target clock period in picoseconds. Only used with
    the pipeline generator.
*   `clock_margin_percent`: The percent of the target clock period to reserve as
    "margin". Only used with the pipeline generator.
*   `entry`: The DSLX function to synthesize. If not given, a "best guess" will
    be made.
*   `flop_inputs`, `flop_outputs`: Whether to add flops at the inputs/outputs of
    the generated module. Only used with the pipeline generator.
*   `generator`: The code generator to be used. Can be either `combinational` or
    `pipeline`.
*   `module_name`: The desired name of the generated Verilog module.
*   `pipeline_stages`: The desired number of pipeline stages.
*   `reset`: The name of the reset signal to use, if any.
*   `reset_active_low`: Whether the reset signal is active low. Must also
    specify `reset` option. Only used with the pipeline generator.

### `dslx_jit_wrapper`

Generates a source/header file pair to wrap invocation of a DSLX function by the
JIT - this wrapper is a much more straightforward way to interact with the JIT
versus direct use. It accepts a DSLX target and entry function in the same
manner as above.

### `dslx_test`

Main driver for:

* Compiling DSLX files to IR
* Running DSLX test cases (defined alongside the designs)
* Proving DSLX/IR equivalence
* Generating a benchmark for the associated IR
* Proving logical equivalence for the optimized vs. unoptimized IR

In general, if one has a DSLX .x file, there should be an associated `dslx_test`
target - this is the entry point to "downstream" capabilities (IR
transformations, codegen, interpretation/JIT, etc.).

Note: This macro is targeted for refactoring, as it currently contains a broad
swath of not-always-directly-related functionality.

### `dslx_generated_rtl`

This macro is for internal developer use only, for generating RTL with
internally-released toolchains.

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
