# [xls/contrib] Tutorial: XLS[cc] pipelined loops.

[TOC]

This tutorial is aimed at walking you through the implementation and synthesis
into Verilog of a C++ function containing a pipelined loop. Pipelined loops are
an automatic way of generating stateful logic that can often be more intuitive
and software-like than using explicit state (eg via `static`).

## C++ Source

Create a file named `test_loop.cc` with the following contents.

```c++
template<typename T>
using InputChannel = __xls_channel<T, __xls_channel_dir_In>;
template<typename T>
using OutputChannel = __xls_channel<T, __xls_channel_dir_Out>;

class TestBlock {
public:
    InputChannel<int> in;
    OutputChannel<int> out;

    #pragma hls_top
    void Run() {
    int sum = 0;
        #pragma hls_pipeline_init_interval 1
        for(int i=0;i<4;++i) {
            sum += in.read();
        }
        out.write(sum);
    }
};
```

## Generate optimized XLS IR.

Use a combination of `xlscc` and `opt_main` to generate optimized XLS IR.

```shell
$ ./bazel-bin/xls/contrib/xlscc/xlscc test_loop.cc \
  --block_from_class TestBlock --block_pb block.pb \
  > test_loop.ir
$ ./bazel-bin/xls/tools/opt_main test_loop.ir > test_loop.opt.ir
```

## Examine the optimized IR

`test_loop.opt.ir` should look like this, containing only one proc:

```
package my_package

file_number 1 "/usr/local/google/home/seanhaskell/tmp/tutorial_loop.cc"

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="""""")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="""""")

top proc TestBlock_proc(tkn: token, __for_1_proc_state__4: bits[1], __for_1_proc_state__5: bits[32], __for_1_proc_state__6: bits[32], __for_1_proc_activation__1: bits[1], __for_1_ctx_out_receive_holds_activation__1: bits[1], TestBlock_proc_activation__1: bits[1], after_all_186_holds_activation_0__1: bits[1], after_all_186_holds_activation_1__1: bits[1], after_all_84_holds_activation_0__1: bits[1], after_all_84_holds_activation_1__1: bits[1], after_all_84_holds_activation_2__1: bits[1], __for_1_ctx_in_receive_holds_activation__1: bits[1], init={1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0}) {
...
  next (after_all.354, __for_1_proc___first_tick_next_state, __for_1_proc_sum_next_state, __for_1_proc_i_next_state, __for_1_ctx_out_receive_activation_out, __for_1_ctx_out_receive_holds_activation_next__1, after_all_84_is_activated, after_all_186_holds_activation_0_next__1, after_all_186_holds_activation_1_next__1, after_all_84_holds_activation_0_next__1, after_all_84_holds_activation_1_next__1, after_all_84_holds_activation_2_next__1, __for_1_ctx_in_receive_holds_activation_next__1)
}

```

## Perform code-generation into a pipelined Verilog block.

In this case, we will generate a single-stage pipeline without input and output
flops. This will result in a module with a 32-bit increment adder along with
32-bit of state.

```shell
$ ./bazel-bin/xls/tools/codegen_main test_loop.opt.ir \
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

## Additional XLS[cc] examples.

For developers, it is possible to check if a specific feature is supported by
checking
[translator_proc_test.cc](https://github.com/google/xls/tree/main/xls/contrib/xlscc/unit_tests/translator_proc_test.cc)
for unit tests.
