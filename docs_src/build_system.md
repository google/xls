# Build system

XLS uses the [Bazel](http://bazel.build) build system for itself and all its
dependencies. Bazel is an easy to configure and use, and has powerful extension
facilities. (It's also
[well-documented](https://bazel.build/start/bazel-intro)!) XLS provides a number
of [Starlark](https://bazel.build/rules/language)
[rules](https://bazel.build/rules/rules) and
[macros](https://bazel.build/rules/macros) to define a build flow.

[TOC]

## Whirlwind Intro To Bazel

Many developers are familiar with a make-style build flow. Bazel, by contrast,
provides more built-in structure for where generated files and binary artifacts
are placed, in order to keep the source tree unmodified and the build process
fully declarative / repeatable. In Bazel, one of the key principles is "the user
should not need to `bazel clean`".

A typical build command looks like:

```
$ bazel build -c opt //xls/tools:opt_main
```

The `-c opt` flag is requesting we produce an optimized build. Other options for
development are:

-   `-c fastbuild`: fewer optimizations, quicker turn around time on builds, and
-   `-c dbg`: debug binaries, minimal optimization level and debug information
    produced, e.g. for using binaries under `gdb`

Targets are referenced with `//` as the root of the current repository -- it is
generally optional. From there you specify the path to a directory with a
`BUILD` file, and then `:target_name` to reference a named target within that
`BUILD` file. In the case above, the build target referenced is a C++ binary --
its build definition is described by a `cc_binary` rule in the `xls/tools/BUILD`
file.

### Where the output files go

The above command notes the following in its output:

```
Target //xls/tools:opt_main up-to-date:
  bazel-bin/xls/tools/opt_main
```

We can see binary result files go to `bazel-bin` within our repository's root
directory. (Aside: `bazel-bin` is a convenient symlink to an out-of-tree
location where build artifacts are placed.)

**Generated** files that are intermediate entities in the build process are also
visible via a similar symlink, `bazel-out`. Within the following directory:

```
$ ls bazel-out/host/bin/xls/ir/
```

We can see files that were part of the build of the IR library, like `op.h` and
`op.cc`.

## XLS Project Build Rules

XLS provides a set of Bazel build rules and macros that allow users to
quickly/easily create XLS-based design artifacts -- analogous to the way C++,
Python, etc are done in Bazel. For example, `dslx_library` lets a user make a
library target written in XLS' Domain Specific Language frontend.

XLS build rules and macros are defined in
[xls/build_rules/xls_build_defs.bzl](https://github.com/google/xls/tree/main/xls/build_rules/xls_build_defs.bzl).

Examples using the rules and macros are found at
[xls/build_rules/tests/BUILD](https://github.com/google/xls/tree/main/xls/build_rules/tests/BUILD).

A detailed description of the bazel rules/macros can be found
[here](bazel_rules_macros.md).

## Bazel queries

Understanding the build tree for a new project can be difficult, but fortunately
Bazel provides a
[powerful query mechanism](https://bazel.build/reference/query). `bazel query`
enables a user to examine build targets, dependencies between them, and much
more. A few usage examples are provided here, but the full documentation (linked
above) is comprehensive.

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
`somepath` provides *some* path, not all paths (that's what `allpaths` is for).

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
set of dependencies can quickly grow to be unmanageable, so keep the initial set
(the first argument) as small as possible, and consider specifying a third
argument for maximum search depth.
