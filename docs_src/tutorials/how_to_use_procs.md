# DSLX Tutorial: How to use procs

A proc contains:

*   A *`init`* function that initializes the proc state.
*   A *`config`* function that spawns any other dependent/child procs needed for
    execution.
*   A recurrent (i.e., infinitely looping) *`next`* function that contains the
    actual logic to be executed by the proc.

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
import float32;

type F32 = float32::F32;

proc Fmac {
    input_a_consumer: chan<F32> in;
    input_b_consumer: chan<F32> in;
    output_producer: chan<F32> out;

    config(input_a_consumer: chan<F32> in, input_b_consumer: chan<F32> in,
           output_producer: chan<F32> out) {
        (input_a_consumer, input_b_consumer, output_producer)
    }

    init { zero!<F32>() }

    next(state: F32) {
        let tok = join();
        let (tok_a, input_a) = recv(tok, input_a_consumer);
        let (tok_b, input_b) = recv(tok, input_b_consumer);
        let result = float32::fma(input_a, input_b, state);
        let tok = join(tok_a, tok_b);
        let tok = send(tok, output_producer, result);
        result
    }
}
```

(In practice, a FMAC unit would want a reset signal, but we've leaving that out
for simplicity.)

There's a lot to unpack here, so we'll walk through the example.

The first part of interest is the declaration of the proc member values: the
three channels. These are similar to class data members in software. In DSLX,
proc members are constant values, and are set by the output of the `config`
function. Member values can be referred to inside the `next` function in the
same way as locally-declared data.

Next up is the `init` function, defining the initial value for the proc state.

Following is the `config` function itself. When a proc is "spawned", its given a
sets of values that get passed to the `config` function (following this example,
we'll show how procs are spawned). Inside a `config` function, any constant
values can be computed, any necessary procs can be spawned, and finally member
values are set by the return value. Member values are assigned in declaration
order: the first element of the return tuple corresponds to the first-declared
member, and so on.

After that, we encounter `next`. This function serves as the real "body" of the
Proc. The `next` function maintains and evolves the proc's recurrent state and
is responsible for communicating with the outside world, as well. In our
example, the first two lines are that communication, receiving the input values
for the computation. The token elements are used to sequence events: since the
receives can happen in parallel, they can share the same token, but since
sending the output must happen after that, their result tokens are "joined"
(think joining two threads of execution in software), and the result is used to
sequence the send. Between the communication routines is the actual computation.

At the end of the proc, we terminate with the `result` value. This final value
becomes the input state for the next iteration. This is how recurrent state is
managed by procs: a state value is provided to the `next` function, and the
result of that function is used as the next iteration's state input. Procs have
exactly one state value; to keep multiple elements in state, the state can be a
struct.

## Spawning procs

In any real design, procs will form a network[^proc_network], where one proc
will spawn any number of child procs, which themselves might spawn other procs.
(The "root" proc will be instantiated by some outside component in the outside
RTL environment.) Procs may only be spawned in `config` functions, as they're
part of statically configuring the hardware network to construct.

[^proc_network]: The proc network itself will form a tree, but channels may make
    point-to-point connections between any two procs.

As an example, spawning a couple of our procs above would look as follows:

```dslx-snippet
proc Spawner {
  fmac_1_a_producer: chan<F32> out;
  fmac_1_b_producer: chan<F32> out;
  fmac_1_output_consumer: chan<F32> in;
  fmac_2_a_producer: chan<F32> out;
  fmac_2_b_producer: chan<F32> out;
  fmac_2_output_consumer: chan<F32> in;

  // ...

  config() {
    let (fmac_1_a_p, fmac_1_a_c) = chan<F32>("fmac_1_a");
    let (fmac_1_b_p, fmac_1_b_c) = chan<F32>("fmac_1_b");
    let (fmac_1_output_p, fmac_1_output_c) = chan<F32>("fmac_1_output");
    spawn Fmac(fmac_1_a_c, fmac_1_b_c, fmac_1_output_p);

    let (fmac_2_a_p, fmac_2_a_c) = chan<F32>("fmac_2_a");
    let (fmac_2_b_p, fmac_2_b_c) = chan<F32>("fmac_2_b");
    let (fmac_2_output_p, fmac_2_output_c) = chan<F32>("fmac_2_output");
    spawn Fmac(fmac_2_a_c, fmac_2_b_c, fmac_2_output_p);

    (fmac_1_a_p, fmac_1_b_p, fmac_1_output_c,
     fmac_2_a_p, fmac_2_b_p, fmac_2_output_c)
  }

  // ...
}
```

For each child proc, we first declare the necessary channels (each channel
declaration produces a producer and consumer channel, respectively), then we
actually spawn it. The first set of arguments is passed to the child's config
function. A spawn produces no value, hence no `let` on the left-hand side.

## Advanced features

### Channel arrays and loop-based spawning

Many hardware layouts have regular arrays of components, such as systolic
arrays, vector units, etc. Individually specifying these quickly grows
cumbersome, so users can instead declare arrays of channels and spawn procs
inside `unroll_for!` loops. This looks as follows:

```dslx-snippet
proc Spawner4x4 {
  input_producers: chan<F32>[4][4] out;
  output_consumers: chan<F32>[4][4] in;

  // ...

  config() {
    let (input_producers, input_consumers) = chan<F32>[4][4]("node_input");
    let (output_producers, output_consumers) = chan<F32>[4][4]("node_output");

    unroll_for! (i, _) : (u32, ()) in u32:0..u32:4 {
      unroll_for! (j, _) : (u32, ()) in u32:0..u32:4 {
        spawn Node(input_consumers[i][j],
                   output_producers[i][j]);
      }(());
    }(());

    (input_producers, output_consumers)
  }

  // ...
}
```

### Parametric procs

Just as with other DSLX constructs, Procs can be parameterized. Parametrics must
be specified at the proc level, and not at the component function level (i.e.,
not on `config` or `next`). Building off of the previous example, this looks as
follows:

```dslx-snippet
proc Parametric<N: u32, M: u32> {
  input_producers: chan<F32>[N][M] out;
  output_consumers: chan<F32>[N][M] in;

  // ...

  config() {
    let (input_producers, input_consumers) = chan<F32>[N][M]("node_input");
    let (output_producers, output_consumers) = chan<F32>[N][M]("node_output");

    for (i, _) : (u32, ()) in u32:0..N {
      for (j, _) : (u32, ()) in u32:0..M {
        spawn Node(input_consumers[i][j],
                   output_producers[i][j]);
      }(());
    }(());

    (input_producers, output_consumers)
  }

  // ...
}
```

These two features are very powerful together: users can specify a broad variety
of designs simply by adjusting a few parametric values.

## Proc testing

The DSLX interpreter supports testing procs via the *test_proc* construct. A
test proc is very similar to a normal proc with the following changes:

*   A test proc is preceded by the `#[test_proc]` attribute. This attribute, as
    one might expect, notifies the interpreter that the following proc is a test
    proc.
*   A test proc's `config` function must accept a single argument: a boolean
    output channel for terminating interpretation. When the test is complete,
    the proc should send the test's status (`true` on success, `false` on
    failure) on that channel (commonly called the "terminator" channel).
*   The `#[test_proc]` attribute also supports an optional `expected_fail`
    parameter. This allows writing tests that are expected to fail due to
    a specific `assert!` or `fail!` condition. The value must match the label
    on the failing assertion. If a failure with the matching label occurs,
    the test is treated as successful, and if no such failure happens,
    the test is treated as failed.

A skeletal example:

```dslx-snippet
#[test_proc]
proc Tester {
  terminator: chan<bool> out;

  // ...

  config(terminator: chan<bool> out) {
    spawn proc_under_test(...);
    (terminator,)
  }

  next(state: ()) {
    let tok = join();

    // send and recv message to the proc under test.
    // ...

    // terminate the test interpretation.
    send(tok, terminator, true);
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
proc main {
  req: chan<u32> out;
  resp: chan<u32> in;

  init { u32: 0 }

  config(req: chan<u32> out, resp: chan<u32> in) {
    (req, resp)
  }

  next(state: u32) {
    let request = state * state;
    let tok = send(join(), req, request);
    let (tok, response) = recv(tok, resp);
    state + u32:1
  }
}
```

where the fact that `req` and `resp` are parameters of `config`, and `main` is
the top proc during IR conversion, is what makes them "external".

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
