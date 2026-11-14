# XLS Tools Quick Start

This document is a quick start guide through the use of the individual XLS
tools, from DSL input to RTL generation.

**Note:** This guide assumes you have
[set up your system so it can build the XLS tools via Bazel](README.md#building-from-source).
There is currently no binary tools distribution so building from source is
required.

Create a file `/tmp/simple_add.x` with the following contents:

```
fn add(x: u32, y: u32) -> u32 {
  x + y + u32:0  // Something to optimize.
}

#[test]
fn test_add() {
  assert_eq(add(u32:2, u32:3), u32:5)
}
```

This contains a function, and a unit test of that function.

## Interpreting the DSL file

Now, run it through the DSL interpreter -- the DSL interpreter is useful for
interactive development and debugging.

```
$ bazel run -c opt //xls/dslx:interpreter_main -- /tmp/simple_add.x
[ RUN      ] add
[       OK ] add
```

The DSL interpreter is the execution engine running the test shown.

In lieu of using bazel run for the subsequent commands, this document will
assume `bazel build -c opt //xls/...` has been completed so the
binaries in `./bazel-bin` can be used directly:

```
$ ./bazel-bin/xls/dslx/interpreter_main /tmp/simple_add.x
[ RUN      ] add
[       OK ] add
```

## DSL to IR conversion

To convert the DSL file to IR, run the following command:

```
$ ./bazel-bin/xls/dslx/ir_convert/ir_converter_main --top=add /tmp/simple_add.x > /tmp/simple_add.ir
```

## IR optimization

To optimize the IR, use the `opt_main` tool:

```
$ ./bazel-bin/xls/tools/opt_main /tmp/simple_add.ir > /tmp/simple_add.opt.ir
```

Check the output of `diff -U8 /tmp/simple_add*.ir` to see that the optimizer
eliminated the useless add-with-zero.

## Verilog RTL generation

To generate RTL from the optimized IR, use the `codegen_main` tool:

```
$ ./bazel-bin/xls/tools/codegen_main --pipeline_stages=1 --delay_model=unit /tmp/simple_add.opt.ir > /tmp/simple_add.v
```

## Measuring and reducing area

To inspect XLS's block-level bill of materials (registers, adders, comparators,
and multiplexers), pass `--block_metrics_path=/tmp/simple_add.metrics.textproto`
to `codegen_main` and format the summary with `print_bom`:

```
$ ./bazel-bin/xls/tools/codegen_main --pipeline_stages=1 --delay_model=unit --block_metrics_path=/tmp/simple_add.metrics.textproto /tmp/simple_add.opt.ir > /tmp/simple_add.v
$ ./bazel-bin/xls/tools/print_bom --root_path=/tmp --file_pattern=simple_add\\.metrics\\.textproto
```

To measure synthesized generic gate-level cell counts (`$_*`) from the generated
RTL, run Yosys with `synth -flatten` and `stat` (or add `dfflibmap -liberty
<target.lib>; abc -liberty <target.lib>; stat -liberty <target.lib>` to measure
technology-mapped cell count and physical area on a target standard-cell
library):

```
$ yosys -p "read_verilog -sv /tmp/simple_add.v; synth -flatten -top __simple_add__add; stat"
```

When optimizing a DSLX design for post-synthesis cell count or physical area,
see
[Measuring and Reducing Area](codegen_options.md#measuring-and-reducing-area)
for guidance on `proc` state encodings (binary, one-hot, shift-register, and
Johnson counters) and how synchronous reset and load-enable multiplexer lowering
interacts with target libraries that do or do not provide synchronous-reset /
clock-enable flip-flops (`$_SDFF*` / `$_SDFFE_*`).

## IR visualizer

To get a graphical view of the IR files, use the IR visualization tool:

```
$ ./bazel-bin/xls/visualization/ir_viz/app --delay_model=unit --preload_ir_path=/tmp/simple_add.ir
```

This starts a server on localhost port 5000 by default, so you can access it
from your machine as `http://localhost:5000` in a web browser.
