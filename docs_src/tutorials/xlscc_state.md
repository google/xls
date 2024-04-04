# [xls/contrib] Tutorial: XLS[cc] state.

[TOC]

This tutorial is aimed at walking you through the implementation and synthesis
into Verilog of a C++ function containing state.

XLS[cc] may infer that in order to achieve a particular implementation of a C++
function, operations may occur over multiple cycles and require additional proc
state to be kept. Common constructs that may require this are static variables,
or loops that aren't unrolled. In this tutorial we give an example using static
variables.

## C++ Source

Create a file named `test_state.cc` with the following contents.

```c++
template<typename T>
using OutputChannel = __xls_channel<T, __xls_channel_dir_Out>;

class TestBlock {
public:
    OutputChannel<int> out;

    int count = 0;

    #pragma hls_top
    void Run() {
        out.write(count);
        count++;
    }
};
```

## Generate optimized XLS IR.

Use a combination of `xlscc` and `opt_main` to generate optimized XLS IR.

```shell
$ ./bazel-bin/xls/contrib/xlscc/xlscc test_state.cc \
  --block_from_class TestBlock --block_pb block.pb \
  > test_state.ir
$ ./bazel-bin/xls/tools/opt_main test_state.ir > test_state.opt.ir
```

## Perform code-generation into a pipelined Verilog block.

In this case, we will generate a single-stage pipeline without input and output
flops. This will result in a module with a 32-bit increment adder along with
32-bit of state.

```shell
$ ./bazel-bin/xls/tools/codegen_main test_state.opt.ir \
  --generator=pipeline \
  --delay_model="sky130" \
  --output_verilog_path=xls_counter.v \
  --module_name=xls_counter \
  --top=TestBlock_proc \
  --reset=rst \
  --reset_active_low=false \
  --reset_asynchronous=false \
  --reset_data_path=true \
  --pipeline_stages=1  \
  --flop_inputs=false \
  --flop_outputs=false
```

After running codegen, you should see a file named `xls_counter.v` with contents
similar to the following.

```
module xls_counter(
  input wire clk,
  input wire rst,
  input wire out_rdy,
  output wire [31:0] out,
  output wire out_vld
);
  reg [31:0] __st__1;
  wire literal_43;
  wire literal_40;
  wire [31:0] add_37;
  wire pipeline_enable;
  assign literal_43 = 1'h1;
  assign literal_40 = 1'h1;
  assign add_37 = __st__1 + 32'h0000_0001;
  assign pipeline_enable = literal_43 & literal_40 & out_rdy & (literal_43 & literal_40 & out_rdy);
  always_ff @ (posedge clk) begin
    if (rst) begin
      __st__1 <= 32'h0000_0000;
    end else begin
      __st__1 <= pipeline_enable ? add_37 : __st__1;
    end
  end
  assign out = __st__1;
  assign out_vld = literal_40 & literal_43 & 1'h1;
endmodule
```

## Additional XLS[cc] examples.

For developers, it is possible to check if a specific feature is supported by
checking
[translator_static_test.cc](https://github.com/google/xls/tree/main/xls/contrib/xlscc/unit_tests/translator_static_test.cc)
and
[translator_proc_test.cc](https://github.com/google/xls/tree/main/xls/contrib/xlscc/unit_tests/translator_proc_test.cc)
for unit tests.
