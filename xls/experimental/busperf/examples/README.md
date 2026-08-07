# Busperf examples

The goal is to spot bottlenecks - stages that stall a design's throughput -
directly from a busperf report, without hand-transcribing any signal names.
`xls_busperf_setup` (see `//xls/experimental/busperf`) turns an XLS proc
design straight into that report.

This is demonstrated in `bottleneck.x`. It defines a small pipeline where one
stage may occasionally be too slow to keep up, stalling the dataflow feeding
into it. The same pipeline is built twice: once with that stage running at
full speed, and once with it stalling. Comparing the two reports side by
side shows what a bottleneck actually looks like in practice, next to a
healthy baseline.

## Generate a report

```sh
bazel build //xls/experimental/busperf/examples:bottleneck_stall_stats
cat bazel-bin/xls/experimental/busperf/examples/bottleneck_stall_stats.stats.txt
```

## Check that busperf sees the stall

Build both variants and compare:

```sh
bazel build //xls/experimental/busperf/examples:bottleneck_no_stall_stats \
    //xls/experimental/busperf/examples:bottleneck_stall_stats
```

Both testbenches drive their channels at full throttle (always valid, always
ready), so any backpressure comes from the design itself, not from the
testbench. In `bottleneck_no_stall`'s report, the `_data_r` row reads `Busy
2000 | Backpressure 0` - every cycle transfers. In `bottleneck_stall`'s
report, the same row reads `Busy 504 | Backpressure 1496` - the FIFO fills
and the upstream stage spends most of its time stalled behind it.

## Open a report in a browser

```sh
bazel run //xls/experimental/busperf/examples:bottleneck_stall_open_report
```
