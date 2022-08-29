# Tutorial: XLS[cc] channels, and fixed-width integers.

[TOC]

This tutorial is aimed at walking you through the implementation and synthesis
into Verilog of a C++ function containing channels and fixed-width integers.

## Introduction to channels.

XLS implements channels via a FIFO-based (ready/valid/data) interface.

In C++, these channels are provided by a built-in template class called
`__xls_channel` supporting two methods: `read()` and `write(val)`.

To utilize channels include the special xls\_builtin header file: `#include
"/xls_builtin.h"`.

An example of a usage is below, which reads an integer on the input channel,
multiplies it by 3, and writes it to the output channel.

```c++
#include "/xls_builtin.h"

#pragma hls_top
void test_channels(__xls_channel<int>& in,
                   __xls_channel<int>& out) {
  out.write(3*in.read());
}
```

Note that an `ac_datatypes` compatibility layer is also provided so that the
same C++ code can be used for both simulation and XLS[cc] synthesis with just a
change in the include paths:
[ac_compat](https://github.com/google/xls/tree/main/xls/contrib/xlscc/synth_only/ac_compat).

## Introduction to fixed-width integers.

XLS also provides a template class for fixed-width integers. These are declared
using the template class `XlsInt<int Width, bool Signed = true>`.

To utilize fixed with integer types, include
[xls_int.h](https://github.com/google/xls/tree/main/xls/contrib/xlscc/synth_only/xls_int.h).

Create a `test_channels.cc` with the following contents for the rest of this
tutorial.

```c++
#include "/xls_builtin.h"
#include "xls_int.h"

#pragma hls_top
void test_channels(__xls_channel<XlsInt<17, false>>& in,
                __xls_channel<XlsInt<22, false>>& out) {
  out.write(3*in.read());
}
```

**NOTE** `"/xls_builtin.h"` has a `/` in order to reduce the chance of
collisions with other include files that may exist on the system.

## Configuring XLS[cc] for channel interfaces.

To support different of channel interfaces, a protofile is provided to XLS[cc]
to configure the direction and interface of the top level function.

The below textproto specifies that

1.  `in` is an input channel with FIFO (ready/valid) signaling

2.  `out` is an output channel with FIFO (ready/valid) signaling.

3.  The IR should be created with a top-level proc named `xls_test_proc`.

Create a file `test_channels.textproto` with the following contents.

```textproto
channels {
  name: "in"
  is_input: true
  type: FIFO
}
channels {
  name: "out"
  is_input: false
  type: FIFO
}
name: "xls_test"
```

Then convert it to a protobin. This protobin will be later provided to `xlscc`
with `--block_pb test_channels.pb`.

```
./bazel-bin/xls/tools/proto2bin test_channels.textproto --message xlscc.HLSBlock --output test_channels.pb
```

## Configuring XLS[cc] for fixed-width integers.

XLS[cc] fixed-width integers have a dependency on the `ac_datatypes` library.
Clone the repository (https://github.com/hlslibs) into a directory named
`ac_datatypes`.

```shell
git clone https://github.com/hlslibs/ac_types.git ac_datatypes
```

Then create the a `clang.args` file with the following contents to configure the
include paths and pre-define the `__SYNTHESIS__` name as a macro.

```
-D__SYNTHESIS__
-I/path/to/your/xls/contrib/xlscc/synth_only
-I/path/containing/ac_datatypes/..
```

## Translate into optimized XLS IR.

With the above setup complete, XLS IR can now be generated using a sequence of
`xlscc` and `opt_main`.

```shell
$ ./bazel-bin/xls/contrib/xlscc/xlscc test_channels.cc \
  --clang_args_file clang.args \
  --block_pb test_channels.pb > test_channels.ir
$ ./bazel-bin/xls/tools/opt_main test_channels.ir > test_channels.opt.ir
```

Note that unlike in the prior tutorial, XLS[cc] is used to generate
[XLS procs](../ir_semantics.md#proc)
rather than functions. This is to support the additional interface requirements
of channels.

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
  --top=xls_test_proc \
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

Below is a quick summary of the new options.

1.  `--delay_model="sky130"` - use the sky130 delay model.
2.  `--top=xls_test_proc` - the proc that is the top-level is named
    `xls_test_proc`. This should be the name specified in the textproto given to
    XLS[cc] with a `_proc` suffix appended.
3.  `--flop_inputs_kind=skid` and `--flop_outputs_kind=skid` - control what type
    of I/O buffering is used. In this case, we configure a skid buffer at both
    the input and output.
