# [xls/contrib] Tutorial: XLS[cc] channels.

[TOC]

This tutorial is aimed at walking you through the implementation and synthesis
into Verilog of a sequential C++ block containing channels.

## Introduction to channels.

XLS implements channels via a FIFO-based (ready/valid/data) interface.

In C++, these channels are provided by a built-in template class called
`__xls_channel` supporting the two methods: `read()` and `write(val)`.

It can be aliased to the desired name like this:

```c++
template<typename T>
using OutputChannel = __xls_channel<T, __xls_channel_dir_Out>;

template<typename T>
using InputChannel = __xls_channel<T, __xls_channel_dir_In>;
```

An example of a usage is below, which reads an integer on the input channel,
multiplies it by 3, and writes it to the output channel.

```c++
class TestBlock {
public:
    InputChannel<int> in;
    OutputChannel<int> out;

    #pragma hls_top
    void Run() {
        auto x = in.read();
        out.write(3*x);
    }
};
```

## Translate into optimized XLS IR.

With the above setup complete, XLS IR can now be generated using a sequence of
`xlscc` and `opt_main`.

```shell
$ ./bazel-bin/xls/contrib/xlscc/xlscc test_channels.cc \
  --block_from_class TestBlock --block_pb meta.pb > test_channels.ir
$ ./bazel-bin/xls/tools/opt_main test_channels.ir > test_channels.opt.ir
```

Below is a quick summary of the options. 1. `--block_from_class TestBlock` -
tells XLS[cc] which class is the top block. 1. `--block_pb` - tells XLS[cc]
where to write the block's metadata description. This must be specified with
`--block_from_class`.

Note that unlike in the prior tutorial, XLS[cc] is used to generate
[XLS procs](../ir_semantics.md#proc)
rather than functions. This is to support the additional interface requirements
of channels.

## Note the metadata output

The file `meta.pb` now contains a description of the block which can be useful
for integration. In this example, the result is:

```textproto
channels {
  name: "in"
  is_input: true
  type: FIFO
  width_in_bits: 32
}
channels {
  name: "out"
  is_input: false
  type: FIFO
  width_in_bits: 32
}
name: "TestBlock"
```

## Perform code-generation into a pipelined Verilog block.

With the same IR, you can either generate a combinational block or a clocked
pipelined block with the `codegen_main` tool. In this section, we'll demonstrate
how to generate a pipelined block using the above C++ code.

```shell
$ ./bazel-bin/xls/tools/codegen_main test_channels.opt.ir \
  --generator=pipeline \
  --delay_model="sky130" \
  --output_verilog_path=test_channels.v \
  --module_name=xls_test \
  --top=TestBlock_proc \
  --reset=rst \
  --reset_active_low=false \
  --reset_asynchronous=false \
  --reset_data_path=true \
  --pipeline_stages=5 \
  --flop_inputs=true \
  --flop_outputs=true \
  --flop_inputs_kind=skid \
  --flop_outputs_kind=skid
```

Below is a quick summary of the options.

1.  `--delay_model="sky130"` - use the sky130 delay model.
2.  `--top=TestBlock_proc` - the proc that is the top-level is named
    `TestBlock_proc`. This should be the block class name given to XLS[cc] with
    a `_proc` suffix appended.
3.  `--flop_inputs_kind=skid` and `--flop_outputs_kind=skid` - control what type
    of I/O buffering is used. In this case, we configure a skid buffer at both
    the input and output.

## Additional XLS[cc] examples.

For developers, it is possible to check if a specific feature is supported by
checking
[translator_io_test.cc](https://github.com/google/xls/tree/main/xls/contrib/xlscc/unit_tests/translator_io_test.cc)
[translator_proc_test.cc](https://github.com/google/xls/tree/main/xls/contrib/xlscc/unit_tests/translator_proc_test.cc)
for unit tests.
