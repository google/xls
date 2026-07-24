# XLS - busperf integration

This directory lets you analyze bottlenecks in an XLS proc design's
processing pipeline using [busperf](https://github.com/antmicro/busperf).
A bottleneck is a stage that cannot keep up with the stages feeding it,
causing the ready/valid handshake upstream to stall. busperf makes this
visible as measurable backpressure, once it has a bus description telling
it which signals in your design form each ready/valid channel.

## Layout

- `build_rules/` contains the Bazel rules and macros.
- `busperf_yaml_generator.{h,cc}` and `xls_sig_to_busperf_main.cc` implement
  the tool that turns an XLS codegen signature into a busperf YAML bus
  description.
- `examples/` contains example designs demonstrating the full pipeline, from
  a DSLX design to a busperf report.
- `tests/` contains tests for the rules and the generator.

## Bazel rules

- `xls_busperf_yaml` generates busperf's YAML bus description for a DSLX
  proc.
- `busperf_analyze` runs busperf itself against a VCD and a busperf YAML
  bus description to produce a text or HTML report.

## How to use it

Call `xls_busperf_yaml` in a `BUILD` file, supplying the DSLX proc you want
to analyze, to generate its busperf YAML bus description as a build
artifact. From there, simulate your design to produce a VCD however you
normally would, and call `busperf_analyze` on that VCD and the generated
YAML to get a text or HTML report. See `examples/README.md` for a
complete, working reference of this pipeline end to end, including the
exact commands to build a report and the numbers to look for that confirm
busperf has actually caught a bottleneck.
