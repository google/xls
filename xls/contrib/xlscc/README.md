# XLS[cc]

XLS[cc] is a a C++ HLS tool built on top of XLS. It generates XLS IR from a
subset of C++.

Original author:

Sean Purser-Haskell
seanhaskell@google.com  sean.purserhaskell@gmail.com
https://github.com/spurserh

### Summary ###

XLS[cc] is based on libclang, and so it supports most C++17 language features.

Some features that will never be supported are:
  - Pointers
  - Function pointers
  - Virtual methods

Variable width integer support is provided by synth_only/xls_int.h

To see if specific features are supported, check translator_test.cc for
unit tests.


### Sample Usage ###

To generate some verilog:

```console
echo "#pragma hls_top
int add2(int input) { return input+3; }" > test.cc

bazel build -c opt //xls/contrib/xlscc:xlscc //xls/tools:opt_main //xls/tools:codegen_main
bazel-bin/xls/contrib/xlscc/xlscc test.cc > test.ir
bazel-bin/xls/tools/opt_main test.ir > test.opt.ir
bazel-bin/xls/tools/codegen_main test.opt.ir --generator combinational
```

### Sample Usage to build a pipeline ###

```console
echo "#pragma hls_top
int test(int x) {
  int ret = 0;
  #pragma hls_unroll yes
  for(int i=0;i<32;++i) {
    ret += x * i;
  }
  return ret;
}" > test.cc

bazel build -c opt //xls/contrib/xlscc:xlscc //xls/tools:opt_main //xls/tools:codegen_main
bazel-bin/xls/contrib/xlscc/xlscc test.cc > test.ir
bazel-bin/xls/tools/opt_main test.ir > test.opt.ir
bazel-bin/xls/tools/codegen_main test.opt.ir --generator pipeline --pipeline_stages 4 --delay_model unit
```

### Sample Usage with xls_int.h ###

First, you will need to clone the ac_datatype git repository:

https://github.com/hlslibs/ac_types

Then, update the clang.args file below for your environment, and run the
subsequent commands.

```console

echo "-D__SYNTHESIS__
-I/path/to/your/xls
-I/path/to/your/ac_types/include
-I/usr/include/clang/10.0.1/include" > clang.args

echo "
#include \"xls/contrib/xlscc/synth_only/xls_int.h\"

#pragma hls_top
XlsInt<22, false> test(XlsInt<17, false> x) {
  return x * 3;
}" > test.cc

bazel build -c opt //xls/contrib/xlscc:xlscc //xls/tools:opt_main //xls/tools:codegen_main
bazel-bin/xls/contrib/xlscc/xlscc ~/tmp/test.cc --clang_args_file ~/tmp/clang.args > test.ir
bazel-bin/xls/tools/opt_main test.ir > test.opt.ir
bazel-bin/xls/tools/codegen_main test.opt.ir --generator combinational --entry test
```

