# XLS / SPIN examples

The DSLX interpreter executes one fixed schedule per run, so it can't catch
bugs that only appear under a different interleaving of a concurrent design.
These examples translate DSLX procs to
[Promela](https://spinroot.com/spin/Man/Intro.html) and check them with
[SPIN](https://spinroot.com/spin/whatispin.html)
instead, which can reason about all schedules.

Each example has two test targets:

| Target | What it does |
|---|---|
| `*_dslx_guided_test` | DSLX interpreter + one guided SPIN simulation; trace comparison |
| `*_dslx_exhaustive_test` | DSLX interpreter + exhaustive SPIN state-space search |

## counter

`counter.x` sends incrementing u32 values on a channel. Well-behaved example
that produces the same output in DSLX and SPIN.

```
bazel test //xls/spin/examples:counter_dslx_guided_test
bazel test //xls/spin/examples:counter_dslx_exhaustive_test
```

Expected result: both pass.

## order_dependence

An `Arbiter` forwards requests to two parametric `Worker` procs and collects
their responses via a `Receiver` using non-blocking receives. The two workers
run concurrently, making the response non-deterministic. The DSLX interpreter
misses this because it runs one fixed schedule; SPIN finds it under both guided
and exhaustive search.

Both targets are tagged `manual` because they intentionally fail:

```
bazel test --build_tests_only --test_tag_filters=-manual //xls/spin/examples:all  # skips both
bazel test //xls/spin/examples:order_dependence_guided_test     --tags=            # fails
bazel test //xls/spin/examples:order_dependence_exhaustive_test --tags=            # fails
```

`order_dependence_guided_test` runs a single SPIN simulation path. SPIN picks an
interleaving where the racy `Worker` fires in a different order than the DSLX
interpreter, so the `assert_eq` fires in the Promela model but not in DSLX.

`order_dependence_exhaustive_test` exhaustively explores all interleavings and
also finds the assertion violation.

## Test artifacts

All test targets write to Bazel's undeclared test outputs:

```
bazel-testlogs/xls/spin/examples/<target>/test.outputs/outputs.zip
```

- Guided tests: `guided/model.pml`, `guided/spin_trace.json`,
  `guided/dslx_trace.textproto`, `guided/spin_output.log`.
- Exhaustive tests: the counterexample `<model>.pml.trail` on failure. To
  replay it: put it alongside `<model>.pml` and run `spin -t -p -g <model>.pml`.
