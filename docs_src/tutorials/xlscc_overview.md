# [xls/contrib] Tutorial: XLS[cc] Overview

[TOC]

This tutorial is aimed at walking you through getting a function written in C++
and then compiling into a working Verilog module.

This assumes that you've already been successful in building XLS. See
[Installing and building](../tutorials/hello_xls.md)
if not.

## Create your first C++ module.

XLS[cc] takes as input a single translation unit -- one `.cc` file. Other files
may be included in that one file, but only the top-level file should be
provided.

Create a file called `test.cc` with the following contents.

```c++
#pragma hls_top
int add3(int input) { return input + 3; }
```

Note that `#pragma hls_top` denotes the top-level function for the module. The
xls func or proc created will follow that function's interface.

## Translate into optimized XLS IR.

Now that the C++ function has been created, `xlscc` can be used to translate the
C++ into XLS IR. `opt_main` is used afterwards to optimize and transform the IR
into a form more easily synthesized into verilog.

```
$ ./bazel-bin/xls/contrib/xlscc/xlscc test.cc > test.ir
$ ./bazel-bin/xls/tools/opt_main test.ir > test.opt.ir
```

The resulting `test.opt.ir` file should look something like the following

```
package my_package

file_number 1 "./test.cc"

top fn add3(input: bits[32]) -> bits[32] {
  literal.2: bits[32] = literal(value=3, id=2, pos=[(1,2,23)])
  ret add.3: bits[32] = add(input, literal.2, id=3, pos=[(1,2,23)])
}
```

## Perform code-generation into a combinational Verilog block.

With the same IR, you can either generate a combinational block or a clocked
pipelined block with the `codegen_main` tool. In this section, we'll demonstrate
how to generate a combinational block.

```shell
$ ./bazel-bin/xls/tools/codegen_main test.opt.ir \
  --generator=combinational \
  --delay_model="unit" \
  --output_verilog_path=test.v \
  --module_name=xls_test \
  --top=add3
```

Below is a quick summary of each option:

1.  `--generator=combinational` states that `codegen_main` should generate a
    combinational module.
2.  `--delay_model="unit"` states to use the unit delay model. Additional delay
    models include asap7 and sky130.
3.  `--output_verilog_path=test.v` is where the output verilog should be written
    to.
4.  `--module_name=xls_test` states that the generated verilog module should
    have the name of `xls_test`.
5.  `--top=add3` states that the function that should be used for codegen is the
    function (`fn`) named `add3`.

The resulting `test.v` should have contents similar to the following

```
module xls_test(
  input wire [31:0] input,
  output wire [31:0] out
);
  wire [31:0] add_6;
  assign add_6 = input + 32'h0000_0003;
  assign out = add_6;
endmodule
```

## Create your second C++ module and generate an optimized IR file.

XLS[cc] supports two ways of handling looping C++ constructs -- it can unroll
the loop, or convert the loop into sequential logic. In this section, we'll
demonstrate loop unrolling.

Unrolled loops are annotated with `#pragma hls_unroll yes`. For example, create
a file called `test_unroll.cc` with the following contents.

```c++
#pragma hls_top
int test_unroll(int x) {
  int ret = 0;
  #pragma hls_unroll yes
  for(int i=0;i<32;++i) {
    ret += x * i;
  }
  return ret;
}
```

Then compile, and optimize the resulting IR

```
$ ./bazel-bin/xls/contrib/xlscc/xlscc test_unroll.cc > test_unroll.ir
$ ./bazel-bin/xls/tools/opt_main test_unroll.ir > test_unroll.opt.ir
```

## Perform code-generation into a pipelined Verilog block.

The previous section should have left you with an IR file called
`test_unroll.opt.ir` with a function with the signature `fn test_unroll(x:
bits[32]) -> bits[32]`. The function is likely too large to fit into a single
clock cycle so we'll create a pipelined module.

```shell
$ ./bazel-bin/xls/tools/codegen_main test_unroll.opt.ir \
  --generator=pipeline \
  --delay_model="asap7" \
  --output_verilog_path=test_unroll.v \
  --module_name=xls_test_unroll \
  --entry=test_unroll \
  --reset=rst \
  --reset_active_low=false \
  --reset_asynchronous=false \
  --pipeline_stages=5 \
  --flop_inputs=true \
  --flop_outputs=true
```

Below is a quick summary of each option:

1.  `--generator=pipeline` - `codegen_main` should generate a pipelined module.
2.  `--delay_model="asap7"` - use the asap7 delay model.
3.  `--output_verilog_path=test_unroll.v` - where the output verilog should be
    written to.
4.  `--module_name=xls_test_unroll` - the generated verilog module should have
    the name of `xls_unroll_test`.
5.  `--entry=test_unroll` - the function that should be used for codegen is the
    function (`fn`) named `test_unroll`.
6.  `--reset=rst` - there should be a reset signal named `rst`.
7.  `--reset_active_low=false` - a high reset signal means reset the module.
8.  `--reset_asynchronous=false` - rst is a synchronous reset signal.
9.  `--pipeline_stages=5` - create a 5 stage pipeline.
10. `--flop_inputs=true` and `--flop_outputs=true` - input and outputs for the
    block are registered.

## Additional XLS[cc] examples.

The above tutorials only touches upon the capabilities of XLS[cc]. XLS[cc] is
based on libclang and supports many C++17 features. Notable *unsupported*
features include pointers, function pointers, and virtual methods.

For developers, it is possible to check if a specific feature is supported by
checking
[translator_logic_test.cc](https://github.com/google/xls/tree/main/xls/contrib/xlscc/unit_tests/translator_logic_test.cc)
and other unit tests in the same directory.
