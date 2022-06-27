# Build system

XLS uses the [Bazel](http://bazel.build) build system for itself and all its
dependencies. Bazel is an easy to configure and use, and has powerful extension
facilities. (It's also
[well-documented](https://bazel.build/start/bazel-intro)!)
XLS provides a number of
[Starlark](https://bazel.build/rules/language)
[rules](https://bazel.build/rules/rules) and
[macros](https://bazel.build/rules/macros) to define
a build flow.

[TOC]

## Rules

XLS build rules and macros are defined in
[xls/build_rules/xls_build_defs.bzl](https://github.com/google/xls/tree/main/xls/build_rules/xls_build_defs.bzl).

Examples using the rules and macros are found at
[xls/build_rules/tests/BUILD](https://github.com/google/xls/tree/main/xls/build_rules/tests/BUILD).

A detailed description of the bazel rules/macros can be found
[here](bazel_rules_macros.md).

## Bazel queries

Understanding the build tree for a new project can be difficult, but fortunately
Bazel provides a
[powerful query mechanism](https://bazel.build/reference/query).
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
