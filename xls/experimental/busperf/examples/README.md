# Busperf example

This example demonstrates how [busperf](https://antmicro.github.io/busperf/)
can be used to find backpressure bottlenecks in a pipeline. It compares two
versions of the same `Passthrough` -> `FIFO` -> `SlowConsumer` pipeline.
`BottleneckNoStall` provides a baseline where all stages can run at full
speed, while `BottleneckStall` slows down the consumer to create backpressure.
The testbench drives the pipeline at full throughput, so the difference in
the reports comes from the design itself. The stalled version shows
backpressure building up at the slow consumer and propagating through the
FIFO to the upstream channels.

## Generate the config

```sh
bazel build \
    //xls/experimental/busperf/examples:bottleneck_no_stall_bus_yaml \
    //xls/experimental/busperf/examples:bottleneck_stall_bus_yaml
```

The `scope` values in `BUILD` must match the testbench module and DUT instance
(`dut`).

## Capture a VCD

Build the Verilog and run the testbench with
[Verilator](https://verilator.org/) (tested with Verilator 5.052); for
installation follow the
[installation guide](https://verilator.org/guide/latest/install.html). Run
these from the repository root:

```sh
bazel build \
    //xls/experimental/busperf/examples:bottleneck_stall_verilog

verilator --binary --trace --trace-underscore --timing -Wno-fatal \
    --top-module tb_bottleneck_stall \
    -DDUT_MODULE=bottleneck_stall \
    -DTB_MODULE=tb_bottleneck_stall \
    '-DVCD_NAME="bottleneck_stall.vcd"' \
    bazel-bin/xls/experimental/busperf/examples/\
bottleneck_stall_verilog.sv \
    xls/experimental/busperf/examples/tb_bottleneck.v

./obj_dir/Vtb_bottleneck_stall
```

This produces `bottleneck_stall.vcd`. Replace `bottleneck_stall` with
`bottleneck_no_stall` to generate the baseline trace.

## Run busperf

See busperf's [installation guide](https://antmicro.github.io/busperf/install.html)
(tested with commit `a62c4c7c6cb3`).

Analyze the trace:

```sh
busperf analyze bottleneck_stall.vcd \
    bazel-bin/xls/experimental/busperf/examples/\
bottleneck_stall_bus_yaml.busperf.yaml \
    --text -o report.stats.txt

cat report.stats.txt
```

The no-stall pipeline has no backpressure:

```text
Busy 2000 | Backpressure 0
```

The stalled pipeline shows sustained backpressure on the FIFO-facing
channels:

```text
_data_r:                   Busy 503 | Backpressure 1497
SlowConsumer<4> _data_r:   Busy 500 | Backpressure 1500
```

For an HTML report, use `--html -o report.html` instead.
