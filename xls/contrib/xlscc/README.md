# XLS[cc]

XLS[cc] is a a C++ HLS tool built on top of XLS. It generates XLS IR from a
subset of C++.

Original author:

Sean Purser-Haskell (seanhaskell@google.com, sean.purserhaskell@gmail.com):
https://github.com/spurserh

## Disclaimer

XLS[cc] is an alternate XLS frontend, primarily maintained by its original
contributor. It lives in the XLS repository to better leverage common
infrastructure.

The core XLS authors do not expect to maintain this tool outside of a
best-effort basis, but contributions are welcome!

### Summary

XLS[cc] is based on libclang, and so it supports most C++17 language features.

Some features that will never be supported are:

- Pointers
- Function pointers
- Virtual methods

Variable width integer support is provided by synth_only/xls_int.h

To see if specific features are supported, check translator_logic_test.cc for
unit tests.

### Sample Usage

(See more at https://google.github.io/xls/tutorials/xlscc_overview/)

To generate some verilog:

```console
echo "#pragma hls_top
int add3(int input) { return input+3; }" > test.cc

bazel build -c opt //xls/contrib/xlscc:xlscc //xls/tools:opt_main //xls/tools:codegen_main
bazel-bin/xls/contrib/xlscc/xlscc test.cc > test.ir
bazel-bin/xls/tools/opt_main test.ir > test.opt.ir
bazel-bin/xls/tools/codegen_main test.opt.ir --generator combinational
```

### Building XLS[cc] with Bazel

XLScc build rules and macros are defined in
[xls/contrib/xlscc/build_rules/xlscc_build_defs.bzl](https://github.com/google/xls/tree/main/xls/contrib/xlscc/build_rules/xlscc_build_defs.bzl).

Examples using the rules and macros are found at
[xls/contrib/xlscc/build_rules/tests/BUILD](https://github.com/google/xls/tree/main/xls/contrib/xlscc/build_rules/tests/BUILD).

A detailed description of the bazel rules/macros can be found
[here](bazel_rules_macros.md).
