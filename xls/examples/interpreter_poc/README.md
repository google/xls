# DSLX Interpreter Randomization Enhancements

Three enhancements to the DSLX bytecode interpreter that expose bugs which pass
silently under the default model but would surface in real hardware: bounded FIFO
simulation, proc scheduling randomization, and mid-tick yield.

Each enhancement has a pair of Bazel targets: one that passes (bug hidden) and
one that fails (bug detected). The failing target is tagged `manual`.

To see channel traces add `--test_env=GLOG_alsologtostderr=1 --test_output=all`.


## 1. Bounded FIFO simulation (`--simulate_bounded_fifos`)

Default channels never fill, so backpressure is never exercised. With this flag,
channels declared with a depth stall the sender when full, forcing sends and
receives to interleave as they would on real hardware.

### `bounded_fifo.x`

`Passthrough` forwards values through two depth-1 channels. Without simulation
all three sends complete before `Passthrough` ever runs. With simulation the
channel fills after the first send and `Passthrough` must drain it before the
next send can proceed. Both tests pass - the difference is when operations occur.

```
# Bug hidden - all three sends complete before Passthrough runs:
bazel test //xls/examples/interpreter_poc:bounded_fifo_without_bounded_fifos_dslx_test

# Backpressure active - sends and drains interleave:
bazel test //xls/examples/interpreter_poc:bounded_fifo_with_bounded_fifos_dslx_test
```

Both targets have `trace_channels` enabled. To see the channel activity:

```
bazel test //xls/examples/interpreter_poc:bounded_fifo_with_bounded_fifos_dslx_test \
  --test_env=GLOG_alsologtostderr=1 --test_output=all
```

### `bounded_fifo_impl.x`

Same scenario using the impl-style proc syntax (`impl` block, `fn new`/`fn next`,
channels accessed via `self`).

```
bazel test //xls/examples/interpreter_poc:bounded_fifo_impl_without_bounded_fifos_dslx_test
bazel test //xls/examples/interpreter_poc:bounded_fifo_impl_with_bounded_fifos_dslx_test
```


## 2. Proc scheduling randomization (`--randomize_proc_execution --seed=N`)

Procs run in a fixed insertion order every tick. A design that only works under
one particular ordering silently passes. With this flag the order is shuffled
each tick, exercising schedules that fixed ordering never reaches.

### `scheduling_bug.x`

`PriorityMux` non-blocking-checks channel A and falls back to B if A is empty.
The test asserts every output came from A. Under fixed ordering WriterA always
runs before `PriorityMux` so A always has data. Under randomized scheduling
`PriorityMux` may run first, find A empty, read from B, and fire the assertion.

```
# Bug hidden (passes):
bazel test //xls/examples/interpreter_poc:scheduling_bug_without_randomization_dslx_test

# Bug detected (assertion fires):
bazel test //xls/examples/interpreter_poc:scheduling_bug_with_randomization_dslx_test
```

### `valid_data_race.x`

Two independent procs drive a valid flag and data on separate channels. The test
asserts both are available at the same time. Under fixed ordering both procs
complete before the test checks. Under randomized scheduling one may lag, leaving
one channel empty and firing the assertion.

```
# Bug hidden (passes):
bazel test //xls/examples/interpreter_poc:valid_data_race_without_randomization_dslx_test

# Bug detected (assertion fires):
bazel test //xls/examples/interpreter_poc:valid_data_race_with_randomization_dslx_test
```

## 3. Mid-tick yield (`--mid_tick_yield`)

Scheduling randomization shuffles proc order at tick boundaries. Mid-tick yield
goes further: after every channel op a proc is suspended and another may run
before it resumes. This exposes assumptions that multiple ops within one `next()`
call are uninterruptible.

### `send_atomicity.x`

`Sender` sends a ready flag then data in the same `next()`. The test asserts: if
ready is visible, data must be too. Without mid-tick yield `Sender` completes
both sends before any other proc runs. With it `Sender` is suspended after the
first send and the test observes ready=true but data=false.

```
# Bug hidden (passes):
bazel test //xls/examples/interpreter_poc:send_atomicity_without_mid_tick_yield_dslx_test

# Bug detected (assertion fires):
bazel test //xls/examples/interpreter_poc:send_atomicity_with_mid_tick_yield_dslx_test
```
