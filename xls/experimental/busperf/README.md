# XLS - busperf integration

This directory generates a [busperf](https://github.com/antmicro/busperf) YAML
bus description from an XLS proc design's codegen signature. busperf uses this
description, together with a simulation VCD, to measure ready/valid handshake
backpressure and pinpoint pipeline bottlenecks.

## Layout

- `build_defs.bzl` contains the Bazel rule.
- `busperf_yaml_generator.{h,cc}` and `xls_sig_to_busperf_main.cc` implement
  the tool that turns an XLS codegen signature into a busperf YAML bus
  description.

## Bazel rules

- `busperf_yaml` generates busperf's YAML bus description from an existing
  codegen signature, e.g. the `<name>.sig.textproto` output of an
  `xls_dslx_verilog` target.

## How to use it

Call `busperf_yaml` in a `BUILD` file, pointing `signature` at the
`.sig.textproto` output of the `xls_dslx_verilog` target for the design you
want to analyze, to generate its busperf YAML bus description as a build
artifact:

```python
xls_dslx_verilog(
    name = "foo_verilog",
    dslx_top = "Foo",
    library = ":foo_dslx",
    ...
)

busperf_yaml(
    name = "foo_bus_yaml",
    signature = ":foo_verilog.sig.textproto",
    scope = "tb_foo.dut",
)
```
