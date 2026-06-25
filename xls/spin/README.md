# XLS Spin Integration

This toolset integrates the [SPIN](https://spinroot.com) model checker into
the XLS build system to formally verify DSLX hardware designs. It is an
experimental feature and is subject to change.

## Motivation

The XLS DSLX interpreter simulates hardware designs by running all operations
in a fixed, sequential order. This works correctly for a class of designs
called Kahn Process Networks (KPNs), where the result depends only on the
input data and not on the timing or order of execution. However, many XLS
designs break this property by using constructs such as bounded FIFOs
that can fill up and block, or non-blocking reads that behave differently
depending on whether data happens to be present. For these designs, the order
in which operations execute matters, and different orderings can produce
different results or expose bugs. Because the interpreter always follows one
fixed order, it can miss entire classes of failures: deadlocks where the
system stalls permanently, livelocks where processes keep running but make no
useful progress, and race conditions where two processes interfere with each
other in a timing-dependent way. A design can pass all interpreter tests and
still fail in real hardware, because the interpreter never explored the
problematic execution order.

## How it works

The toolset translates XLS designs to Promela, the input language of SPIN,
and runs SPIN on the resulting model. Unlike the interpreter, SPIN does not
follow a single execution order. It supports two operating modes: a guided
simulation that follows one specific execution path, and an exhaustive search
that explores all possible interleavings of concurrent processes. SPIN detects
deadlocks by default, and can optionally detect livelocks and channel overflow
with additional options. A trace-comparison layer allows the channel events
produced by a SPIN simulation to be compared against those from the DSLX
interpreter, surfacing any divergence as a test failure.

## Directory layout

| File / dir | Role |
|---|---|
| `promela_generator.h/.cc` | Converts XLS IR to a SPIN model. |
| `spin_runner.h/.cc` | Runs the full DSLX-to-SPIN verification pipeline programmatically. |
| `trace_compare.h/.cc` | Checks that SPIN and the DSLX interpreter produce the same channel events. |
| `promela_converter_main.cc` | CLI tool for generating a SPIN model from an IR file. |
| `dslx_trace_filter_main.cc` | CLI tool for preparing a DSLX interpreter trace for comparison. |
| `promela_trace_compare_main.cc` | CLI tool for comparing a SPIN trace against a DSLX trace. |
| `defs.bzl` | Public Bazel rules for integrating SPIN verification into a build. |
| `priv-defs.bzl` | Private Bazel rules used internally for testing and development. |
| `testdata/` | Checked-in fixtures used by the test suite. |
| `examples/` | Working DSLX proc examples with full verification target suites. |

## Getting started

SPIN verification is enabled via `dslx_test_args` on `xls_dslx_test`:

```python
xls_dslx_test(
    name = "my_proc_test",
    srcs = ["my_proc.x"],
    dslx_test_args = {
        "guided_model_check": "True",      # guided simulation, compares traces with DSLX
        "exhaustive_model_check": "True",  # exhaustive search over all interleavings
    },
)
```

The `examples/` directory contains working designs with both modes enabled
and is the best place to see how to structure a proc for verification.

For cases where the test rule is not sufficient, `defs.bzl` provides dedicated
Bazel rules that give more control over the pipeline:

| Rule | Purpose |
|---|---|
| `xls_dslx_spin` | DSLX to SPIN model build action |
| `spin_guided_test` | Guided SPIN simulation as `bazel test` |
| `spin_exhaustive_test` | Exhaustive SPIN verification as `bazel test` |

These rules expose additional options such as livelock detection, throughput
modeling, and channel overflow assertions. See `defs.bzl` for the full list.
Additional private rules intended for development are available in `priv-defs.bzl`.

## Known limitations

- **Interleaving at channel boundaries only.** SPIN explores all interleavings
  between concurrent procs, but within a single proc the operations execute in
  the same fixed order as in DSLX. The scheduler may reorder those operations
  in real hardware, so interleavings between operations within a proc are not
  covered.

- **32-bit integer limit.** SPIN represents all scalar types as 32-bit
  signed integers. XLS `bits[N]` with N > 32 is rejected at generation time.
  Sub-32-bit types are emulated by masking after every arithmetic operation.

- **One test proc per run.** The verification pipeline targets a single
  `#[test_proc]` at a time. When a DSLX file contains multiple test procs, one
  is selected via `--spin_top` or by picking the first match. Each proc must be
  verified in a separate `bazel test` target.
