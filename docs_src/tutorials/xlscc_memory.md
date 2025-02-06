# [xls/contrib] Tutorial: XLS[cc] memories.

[TOC]

This tutorial is aimed at walking you through usage of memories. Memories in
this context refer to externally implemented SRAMs. The example given will be
usage of a dual port "1 read 1 write" memory.

The channels tutorial should be followed as a prerequisite.

## C++ Source

Create a source file `test_memory.cc`

```c++
template<typename T>
using InputChannel = __xls_channel<T, __xls_channel_dir_In>;
template<typename T>
using OutputChannel = __xls_channel<T, __xls_channel_dir_Out>;
template<typename T, int Size>
using Memory = __xls_memory<T, Size>;

class TestBlock {
public:
    InputChannel<int> in;
    OutputChannel<int> out;
    Memory<short, 32> store;

    int addr = 0;

    #pragma hls_top
    void Run() {
        const int next_addr = (addr + 1) & 0b11111;
        store[addr] = in.read();
        out.write(store[next_addr]);
        addr = next_addr;
    }
};
```

## Translate to IR

```shell
$ ./bazel-bin/xls/contrib/xlscc/xlscc test_memory.cc \
  --block_from_class TestBlock --block_pb block.pb \
  > test_memory.ir
```

## Examine the metadata

The memory should appear in the resulting block `block.pb`:

```textproto

channels {
  name: "in"
  is_input: true
  type: CHANNEL_TYPE_FIFO
  width_in_bits: 32
}
channels {
  name: "out"
  is_input: false
  type: CHANNEL_TYPE_FIFO
  width_in_bits: 32
}
channels {
  name: "store"
  type: CHANNEL_TYPE_MEMORY
  width_in_bits: 16
  depth: 32
}
name: "TestBlock"
```

This block description information can be used to generate the RAM rewrites and
IO constraints specified in later steps.

## Optimize the IR with RAM rewrites

Create `rewrites.textproto` with the following contents:

```
rewrites {
  from_config {
    kind: RAM_ABSTRACT
    depth: 32
  }
  to_config {
    kind: RAM_1R1W
    depth: 32
  }
  from_channels_logical_to_physical: {
    key: "abstract_read_req"
    value: "store_read_request"
  }
  from_channels_logical_to_physical: {
    key: "abstract_read_resp"
    value: "store_read_response"
  }
  from_channels_logical_to_physical: {
    key: "abstract_write_req"
    value: "store_write_request"
  }
  from_channels_logical_to_physical: {
    key: "write_completion"
    value: "store_write_response"
  }
  to_name_prefix: "store_"
}
```

Then run `opt` with awareness of the RAM:

```shell
$ ./bazel-bin/xls/tools/opt_main test_memory.ir --ram_rewrites_pb rewrites.textproto > test_memory.opt.ir
```

## Generate Verilog with IO constraints

For this memory, we are assuming a fixed 1-cycle latency for reads and writes.
As such, we need to add these constraints to the `codegen` command, in addition
to making it aware of the RAM rewrites, so that the correct ports are generated.

```shell
$ ./bazel-bin/xls/tools/codegen_main test_memory.opt.ir \
  --generator=pipeline \
  --delay_model="sky130" \
  --output_verilog_path=memory_test.v \
  --module_name=memory_test \
  --top=TestBlock_proc \
  --reset=rst \
  --reset_active_low=false \
  --reset_asynchronous=false \
  --reset_data_path=true \
  --pipeline_stages=2  \
  --flop_inputs=false \
  --flop_outputs=false \
  --ram_configurations=ram:1R1W:store__read_req:store__read_resp:store__write_req:store__write_completion
```

Below is a quick summary of the options.

1.  `--ram_configurations=ram:1R1W:store__read_req...` This option informs
    codegen of the necessary information about the memory to generate the top
    level ports in the correct style. "ram" is the name prefix the ports will
    use.

## Additional XLS[cc] examples.

For developers, it is possible to check if a specific feature is supported by
checking
[translator_memory_test.cc](https://github.com/google/xls/tree/main/xls/contrib/xlscc/unit_tests/translator_memory_test.cc)
for unit tests.
