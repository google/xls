# DSLX Tutorial: How to use procs

A proc has:

*   A `proc` declaration listing its channels and state members.
*   An `impl` block typically containing:
    *   A constructor function (conventionally `new`) that initializes the
        proc's channels and state, and spawns any dependent/child procs needed
        for execution.
    *   A recurrent (i.e., infinitely looping) *`next`* function that contains
        the actual logic to be executed by the proc.

A critical component of procs is communication: every proc needs a means to
share data with other procs, or else it'd just be spinning dead code. These
means are *channels*: entities into which data can be sent and from which data
can be received. Each channel has a send and a receive endpoint: data inserted
into a channel by a `send` op can be pulled out by a `recv` op.

## Example proc

Many concepts here are more easily explained via example, so here's a possible
DSLX implementation of
[FMAC (fused multiply-accumulate)](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation),
which computes `C = A * B + C`:

```dslx
#![feature(explicit_state_access)]

import float32;

type F32 = float32::F32;

proc Fmac {
    input_a_consumer: chan<F32> in,
    input_b_consumer: chan<F32> in,
    output_producer: chan<F32> out,
    state: F32,
}

impl Fmac {
    fn new
        (input_a_consumer: chan<F32> in, input_b_consumer: chan<F32> in,
         output_producer: chan<F32> out) -> Self {
        Self { input_a_consumer, input_b_consumer, output_producer, state: zero!<F32>() }
    }

    fn next(self) {
        let state = read(self.state);
        let tok = join();
        let (tok_a, input_a) = recv(tok, self.input_a_consumer);
        let (tok_b, input_b) = recv(tok, self.input_b_consumer);
        let result = float32::fma(input_a, input_b, state);
        let tok = join(tok_a, tok_b);
        let tok = send(tok, self.output_producer, result);
        write(self.state, result);
    }
}
```

(In practice, a FMAC unit would want a reset signal, but we've leaving that out
for simplicity.)

There's a lot to unpack here, so we'll walk through the example.

The first part of interest is the declaration of the proc member values in `proc
Fmac`: the three channels and the `state` member (`F32`). In DSLX, proc channel
members are constant values initialized when the proc is constructed, while
state members hold values that evolve across activations (read via `read(...)`
and updated via `write(...)`). Member values are accessed inside the `next`
function via `self.<member>`.

Next up is the `impl Fmac` block and its `new` constructor function. When a proc
is constructed and spawned (following this example, we'll show how procs are
spawned), values are passed to `new`. Inside `new`, any constant values can be
computed, any necessary child procs can be spawned, and finally the proc's
channel and initial state members are initialized in the returned `Self { ... }`
struct literal.

After that, we encounter `next(self)`. This function serves as the real "body"
of the Proc. The `next` function maintains and evolves the proc's recurrent
state and is responsible for communicating with the outside world, as well. In
our example, we first read the current state with `read(self.state)`, followed
by receiving the input values for the computation. The token elements are used
to sequence events: since the receives can happen in parallel, they can share
the same token, but since sending the output must happen after that, their
result tokens are "joined" (think joining two threads of execution in software),
and the result is used to sequence the send. Between the communication routines
is the actual computation.

At the end of `next`, we update the state with `write(self.state, result)`. This
updated value becomes the state read by the next iteration.

## Spawning procs

In any real design, procs will form a network[^proc_network], where one proc
will spawn any number of child procs, which themselves might spawn other procs.
(The "root" proc will be instantiated by some outside component in the outside
RTL environment.) Procs are spawned at configuration time (typically inside a
parent proc's constructor function) by constructing the child proc and calling
`.spawn()` on it, as part of statically configuring the hardware network to
construct.

[^proc_network]: The proc network itself will form a tree, but channels may make
    point-to-point connections between any two procs.

As an example, spawning a couple of our procs above would look as follows:

```dslx-snippet
proc Spawner {
    fmac_1_a_producer: chan<F32> out,
    fmac_1_b_producer: chan<F32> out,
    fmac_1_output_consumer: chan<F32> in,
    fmac_2_a_producer: chan<F32> out,
    fmac_2_b_producer: chan<F32> out,
    fmac_2_output_consumer: chan<F32> in,
}

impl Spawner {
    fn new() -> Self {
        let (fmac_1_a_p, fmac_1_a_c) = chan<F32>("fmac_1_a");
        let (fmac_1_b_p, fmac_1_b_c) = chan<F32>("fmac_1_b");
        let (fmac_1_output_p, fmac_1_output_c) = chan<F32>("fmac_1_output");
        Fmac::new(fmac_1_a_c, fmac_1_b_c, fmac_1_output_p).spawn();

        let (fmac_2_a_p, fmac_2_a_c) = chan<F32>("fmac_2_a");
        let (fmac_2_b_p, fmac_2_b_c) = chan<F32>("fmac_2_b");
        let (fmac_2_output_p, fmac_2_output_c) = chan<F32>("fmac_2_output");
        Fmac::new(fmac_2_a_c, fmac_2_b_c, fmac_2_output_p).spawn();

        Self {
            fmac_1_a_producer: fmac_1_a_p,
            fmac_1_b_producer: fmac_1_b_p,
            fmac_1_output_consumer: fmac_1_output_c,
            fmac_2_a_producer: fmac_2_a_p,
            fmac_2_b_producer: fmac_2_b_p,
            fmac_2_output_consumer: fmac_2_output_c,
        }
    }

    // ...
}
```

For each child proc, we first declare the necessary channels (each channel
declaration produces a producer and consumer channel, respectively), then we
construct the child proc and call `.spawn()` on it. The arguments to `new`
configure the child proc. A `.spawn()` call produces no value, hence no `let` on
the left-hand side.

## Advanced features

### Channel arrays and loop-based spawning

Many hardware layouts have regular arrays of components, such as systolic
arrays, vector units, etc. Individually specifying these quickly grows
cumbersome, so users can instead declare arrays of channels and spawn procs
inside `unroll_for!` loops. This looks as follows:

```dslx-snippet
proc Spawner4x4 {
    input_producers: chan<F32>[4][4] out,
    output_consumers: chan<F32>[4][4] in,
}

impl Spawner4x4 {
    fn new() -> Self {
        let (input_producers, input_consumers) = chan<F32>[4][4]("node_input");
        let (output_producers, output_consumers) = chan<F32>[4][4]("node_output");

        unroll_for! (i, _): (u32, ()) in u32:0..u32:4 {
            unroll_for! (j, _): (u32, ()) in u32:0..u32:4 {
                Node::new(input_consumers[i][j], output_producers[i][j]).spawn();
            }(());
        }(());

        Self { input_producers, output_consumers }
    }

    // ...
}
```

### Parametric procs

Just as with other DSLX constructs, Procs can be parameterized. Parametrics must
be specified at the proc level, and not at the component function level (i.e.,
not on `new` or `next`). Building off of the previous example, this looks as
follows:

```dslx-snippet
proc Parametric<N: u32, M: u32> {
    input_producers: chan<F32>[N][M] out,
    output_consumers: chan<F32>[N][M] in,
}

impl Parametric<N, M> {
    fn new() -> Self {
        let (input_producers, input_consumers) = chan<F32>[N][M]("node_input");
        let (output_producers, output_consumers) = chan<F32>[N][M]("node_output");

        unroll_for! (i, _): (u32, ()) in u32:0..N {
            unroll_for! (j, _): (u32, ()) in u32:0..M {
                Node::new(input_consumers[i][j], output_producers[i][j]).spawn();
            }(());
        }(());

        Self { input_producers, output_consumers }
    }

    // ...
}
```

These two features are very powerful together: users can specify a broad variety
of designs simply by adjusting a few parametric values.

## Proc testing

The DSLX interpreter supports testing procs via the `#[test]` attribute on a
`proc`. A test proc is very similar to a normal proc with the following changes:

*   A test proc is preceded by the `#[test]` attribute. This attribute notifies
    the interpreter that the following proc is a test proc.
*   A test proc's `new` constructor function in its `impl` block must accept a
    single argument: a boolean output channel for terminating interpretation.
    When the test is complete, the proc should send the test's status (`true` on
    success, `false` on failure) on that channel (commonly called the
    "terminator" channel).
*   The `#[test]` attribute on a proc also supports an optional
    `expected_fail_label` parameter (e.g.
    `#[test(expected_fail_label="my_label")]`). This allows writing tests that
    are expected to fail due to a specific `assert!` or `fail!` condition. The
    value must match the label on the failing assertion. If a failure with the
    matching label occurs, the test is treated as successful, and if no such
    failure happens, the test is treated as failed.

A skeletal example:

```dslx-snippet
#[test]
proc Tester {
    terminator: chan<bool> out,
    // ...
}

impl Tester {
    fn new(terminator: chan<bool> out) -> Self {
        ProcUnderTest::new(...).spawn();
        Self { terminator }
    }

    fn next(self) {
        let tok = join();

        // send and recv message to the proc under test.
        // ...

        // terminate the test interpretation.
        send(tok, self.terminator, true);
    }
}
```

The [FP32 fmac module](https://github.com/google/xls/tree/main/xls/examples/fp32_fmac.x)
has a more complete proc test that may be used for reference.

# Scheduling constraints

If you want to interface with something in the outside world that is latency
sensitive (for example, an SRAM -- though we have separate infrastructure for
making SRAMs work that builds on top of this feature), you can create external
channels representing the interface you want to use, e.g.:

```dslx-snippet
proc Main {
    req: chan<u32> out,
    resp: chan<u32> in,
    state: u32,
}

impl Main {
    fn new(req: chan<u32> out, resp: chan<u32> in) -> Self {
        Self { req, resp, state: u32:0 }
    }

    fn next(self) {
        let state = read(self.state);
        let request = state * state;
        let tok = send(join(), self.req, request);
        let (tok, response) = recv(tok, self.resp);
        write(self.state, state + u32:1);
    }
}
```

where the fact that `req` and `resp` are channel members of `Main`, and `Main`
is the top proc during IR conversion, is what makes them "external".

Then when you codegen this, you can pass in
`--io_constraints=foo__req:send:foo__resp:recv:2:2` where `foo__req` is the
mangled name of the channel, which you can see by examining the generated IR
prior to codegen. That constraint means "a send on any channel named `req` must
occur exactly two cycles before a receive on any channel named `resp`"; the `2`
is specified twice because it is possible to give a range of allowed cycle
differences.

For more details on `--io_constraints`, check out
[the docs](https://google.github.io/xls/codegen_options/#pipelining-and-scheduling-options).
For a complete example, see `//xls/examples:constraint_sv` and
associated build targets; the target you'd build to get the mangled channel
names is `:constraint_ir`.
