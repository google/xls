# DSLX Tutorial: What is a Proc?

Up to this point, our tutorials have described stateless, non-communicating,
combinational modules. To add state or communication with other actors, we need
to venture into the exciting land of procs!

**Procs**, short for "communicating sequential processes", are the means by
which DSLX models sequential and stateful modules. DSLX's semantics are based on
[Kahn process networks](https://en.wikipedia.org/wiki/Kahn_process_networks), or
KPNs. A pure KPN is made up of independent computing units communicating with
each other in timing-insensitive ways; whenever possible, we try to make it easy
to express your desired behavior in this style.[^impure-kpn] In XLS, our
independent computing units are called processes, or *procs*. Each proc has a
fixed set of I/O interfaces (aka *channels*, usually FIFO queues), a fixed
amount of memory (aka *state*), and the ability to carry out a bounded amount of
computation on their state & inputs whenever they activate.

[^impure-kpn]: For when it's necessary, we also expose a limited number of ways
    to express timing-sensitive computations, which we'll discuss
    later.

We can think of each proc as *activating* as often as it can; e.g., in hardware,
up to once per clock cycle. Each activation proceeds as the information &
resources for it to run become available. For example, suppose the proc is
designed to:

*   read a number from each of channels A and B,
*   add the numbers together, and
*   write the result to channel C.

The first activation will wait until it can read the first numbers from each of
channels A and B, then write their sum as the first value on channel C. The
second activation will read the second number from each of channels A and B,
then write their sum as the second value on channel C... and so on.

!!! NOTE
    XLS guarantees that, for each I/O operation X, the first activation's
    action (X<sub>0</sub>) will happen before the second activation's action
    (X<sub>1</sub>), and so on. Therefore, the first activation will read the first
    numbers on channels A and B, and write the first number on channel C. However,
    if a proc includes multiple I/O operations - even on the same channel - and
    these operations need to happen in a specific order (whether in the same
    activation or between activations), there is no ordering guarantee by default.
    Instead, you can express that using **tokens**, as we'll discuss later.

## Channels

In general, **channels** provide possible I/O operations each activation can
carry out. A proc can receive from an input channel or send on an output channel
any number of times per activation (though for now, only once per
channel[^strictness]). There are multiple types of channels, which can make
things rather more complicated; for now, we'll discuss the standard
**streaming** channel type.

[^strictness]: At the moment, DSLX only supports channels where we can prove
    that at most one I/O operation can fire per channel per
    activation. Work is in progress to allow the user to opt into
    multiple I/O operations per channel per activation, and we expect
    this support to land soon. It will be opt-in, however, since this
    introduces the extra overhead of an arbiter to sequence the
    operations, and also adds potential backpressure since each
    channel can still accept at most one datum per clock cycle!

By default, receives are considered to be **blocking** operations; if no data is
available on the channel, the activation will **stall**[^stalling] until the
data becomes available. (This is part of how the defaults ensure that the
results are timing-insensitive, making it easier for you to use XLS to produce
more optimized hardware.) Once the data is available, the receiving proc will
take the value from the channel, removing it if the channel is a FIFO queue, and
proceed.

[^stalling]: When generating pipelined hardware, no later activation can proceed
    past the stalled stage. This may cause activations at earlier
    stages to stall as well, since the resources for them to run are
    not yet available. Activations at later stages can proceed as
    normal. Once the data becomes available, the pipeline will resume
    all stalled operations. See the pipelining documentation for more
    details.

Sends, by contrast, are considered to be **non-blocking** operations; by
default, XLS models channels as if they used infinite-depth queues, so sends can
always complete.

!!! NOTE
    In reality, when compiled to RTL, hardware FIFOs have finite depth. By
    default, XLS allows for **backpressure** on finite-depth channels. If the
    channel is not able to receive new data when the send should trigger, the
    activation will stall until the channel is ready. This always produces correct
    behavior in the absence of deadlocks, but can introduce deadlocks in RTL that
    did not exist at higher levels.

Using these, we can implement our first example, which reads from two channels
and writes the sum to a third:

```dslx
pub proc adder {
  A: chan<u32> in;
  B: chan<u32> in;
  C: chan<u32> out;

  // The initial value of the proc's state (empty in this case).
  init { () }

  // The interface used by anything that spawns this proc, which will need to
  // configure its inputs & outputs.
  config (A: chan<u32> in, B: chan<u32> in, C: chan<u32> out) {
    (A, B, C)
  }

  // The description of how this proc actually acts when running.
  next(st: ()) {
    let (tok_A, data_A) = recv(join(), A);
    let (tok_B, data_B) = recv(join(), B);
    let sum = data_A + data_B;
    let tok = join(tok_A, tok_B);
    send(tok, C, sum);
  }
}
```

You might be surprised at the extra values being passed to and returned from our
`recv` and `send` operations. These are *tokens*, which we will discuss in more
detail below; they are used to establish ordering between operations where data
dependencies aren't sufficient.[^required-tokens]

[^required-tokens]: For now, `send` and `recv` need to take and return tokens
    even when data dependencies are sufficient to express the
    ordering requirements. We do have work in progress to create
    a syntax where tokens are only used where necessary; watch
    this space!

More generally, sends and receives can also be conditional; we can decide
whether or not to carry out each operation based on a computed predicate. For
example, we can write a proc that:

*   reads a value *x* from channel A, and
*   if that input is zero:
    *   reads a value *y* from channel B, and
    *   writes *y* to channel C.
*   otherwise:
    *   writes *x* to channel C.

Therefore, each activation reads one value from channel A, reads either one or
zero values from channel B, and writes one value to channel C. We can implement
this as follows:

```dslx
pub proc fallback {
  A: chan<u32> in;
  B: chan<u32> in;
  C: chan<u32> out;

  init { () }

  config (A: chan<u32> in, B: chan<u32> in, C: chan<u32> out) {
    (A, B, C)
  }

  next(st: ()) {
    let (tok, x) = recv(join(), A);
    let (tok, y) = recv_if(tok, B, x == u32:0, u32:0);
    let val = if x != u32:0 {
      x
    } else {
      y
    };
    send(tok, C, val);
  }
}
```

We could also write this a bit more naturally by putting the channel-`B` receive
inside the correct branch of our `if` expression. As you might expect, the
receive will only trigger if that branch is taken.

```dslx
pub proc fallback {
  A: chan<u32> in;
  B: chan<u32> in;
  C: chan<u32> out;

  init { () }

  config (A: chan<u32> in, B: chan<u32> in, C: chan<u32> out) {
    (A, B, C)
  }

  next(st: ()) {
    let (tok, x) = recv(join(), A);
    let (tok, val) = if x != u32:0 {
      (tok, x)
    } else {
      recv(tok, B)
    };
    send(tok, C, val);
  }
}
```

!!! NOTE
    This also works with `match` expressions; side-effecting operations inside
    a match arm will only trigger if the arm is chosen.

### State

Each proc can also have its own **state elements**, each of which is a piece of
data of any allowed type. Each state element has an initial value, which is the
value seen by the first activation; beyond that, activation *N* sets the state
elements to values that can be read by activation *N*+1. This can include
setting them back to whatever value activation *N*-1 assigned to them,
effectively leaving them unchanged. It's worth noting that activation *N*+1 is
allowed to start before the state from activation *N* has fully resolved; it can
stall if it needs to read from the state, waiting until it can confirm that the
previous activation has set the state element that it needs.

For example, we can design a proc to implement a saturating accumulator, which
reads a value, adds the result to an accumulator (storing the maximum value if
the result would overflow), and returns the updated accumulator value:

```dslx
pub proc saturating_accumulator {
  ch_in: chan<u32> in;
  result: chan<u32> out;

  init { u32:0 }

  config (ch_in: chan<u32> in, result: chan<u32> out) {
    (ch_in, result)
  }

  next(accumulated: u32) {
    let (tok, data) = recv(join(), ch_in);
    let sum = (data as u33) + (accumulated as u33);
    let new_val = if sum > all_ones!<u32>() as u33 {
      all_ones!<u32>()
    } else {
      sum as u32
    };
    send(tok, result, new_val);

    // The last expression is the value the next activation will receive as its
    // state.
    new_val
  }
}
```

#### State and Throughput

If we try to generate pipelined hardware for this proc (as discussed in the
pipelining documentation) and cannot fit the saturating addition in a single
stage, then XLS will notice that it can take more than one cycle for each
activation to determine the next state after reading the current state. This
means this example will not be able to achieve **full throughput**; i.e., it
might be possible for activations to stall *internally*, waiting on the state
from the previous activation, even though all input channels are full and no
output channel is providing backpressure. By default, XLS assumes you're
expecting full throughput, and will emit an error explaining this failure in
terms of the **worst-case throughput** for the proc (the number of cycles that
can elapse between two activations with no externally-caused
stalls)[^inverse-throughput]; the error message will also include what
worst-case throughput *is* possible with your design as written, and how to let
XLS know if this is acceptable for your use case.

[^inverse-throughput]: This is technically an **inverse throughput**. The
    throughput of a proc is properly defined as the number of
    activations that occur per cycle... but rather than
    saying that your proc has WCT 1/2, it's more natural to
    think in inverse throughput - the number of cycles per
    activation - and write WCT 2.

```shell
Error: INVALID_ARGUMENT: Impossible to schedule proc <NAME> as specified; cannot achieve full throughput. Try `--worst_case_throughput=5`
```

On the other hand, we could also design a proc that compares the newest input
value to the previous value, clamps the difference to the range [-5, 5], and
sends the clamped difference:

```dslx
pub proc clamped_diff {
  ch_in: chan<s32> in;
  result: chan<s32> out;

  init { s32:0 }

  config (ch_in: chan<s32> in, result: chan<s32> out) {
    (ch_in, result)
  }

  next(prev: s32) {
    let (tok, val) = recv(join(), ch_in);
    let diff = (val as s33) - (prev as s33);
    let clamped_diff = if diff > s33:5 {
      s32:5
    } else if diff < s33:-5 {
      s32:-5
    } else {
      diff as s32
    };
    send(tok, result, clamped_diff);
    val
  }
}
```

Even if we can't fit both the subtraction and the clamping into a single
pipeline stage, XLS can still pipeline this example to achieve full throughput;
the new state value can be determined immediately on reading the value, so the
next activation can read that state before the first activation is done
computing its result.

### Tokens

For computation operations, XLS can tell which operations depend on which
others, and **schedule** them in hardware respecting these dependencies.
Sometimes, though, a dependency might be external to the proc. For example,
suppose our proc needs to send a message on channel A, then wait for a response
on channel B. If we wait for the response before we send the message, our
hardware will end up deadlocked.

To make sure XLS knows that the message needs to be sent first, we use a
**token**; this lets us express the dependency between these operations even
though the `send` does not produce any actual data that the `recv` can use.
Every operation where ordering effects can be important (generally because
they're visible at the interface), including I/O operations, returns a token and
can accept a token. For instance, to express the dependency we wrote above, we
can write:

```dslx-snippet
// ...
let request_tok = send(join(), A, msg);
// ...
let (response_tok, response) = recv(request_tok, B);
// ...
```

Since the `recv` depends on the token produced by the `send`, we know that the
receive operation should not be allowed to go off until the send operation has
completed.

Of course, it's possible for an operation to need to happen after *more* than
one predecessor. For a simple example, maybe our request needs to be sent in
multiple parts. For this case, we have the `join(tok...)` function, which takes
any number of tokens and returns a single token that depends on all of them. (In
fact, we've been using that already - we wrote `join()` in our previous examples
whenever we needed a token that depended on nothing!) In context, this might
look like:

```dslx-snippet
// ...
let request_part_1_tok = send(join(), A, request_part_1);
// ...
let request_part_2_tok = send(join(), A, request_part_2);
// ...
let request_part_3_tok = send(join(), A, request_part_3);
// ...
let full_request_tok = join(request_part_1_tok, request_part_2_tok, request_part_3_tok);
let (response_tok, response) = recv(full_request_tok, B);
// ...
```

In this example, our `send()`s can happen in any order, but we know that the
`recv()` will not block until after all the `send()`s have finished.

#### Cross-Activation Tokens

There are also contexts where we need to specify ordering constraints
**between** activations. As usual, when we want to communicate some context to
the next activation, we can use a state element; in this case, we can pass a
token as a state element.

For example, suppose we need to write a serialization interface that takes in a
complex struct and produces a sequence of four 32-bit values. If the input
channel receives inputs `A` and `B`, we need to make sure to produce `[A1, A2,
A3, A4, B1, B2, B3, B4]` on the output channel. We can pass a token within each
activation to make sure that the sends are properly sequenced, but we also need
to pass it to the *next* activation to make sure it doesn't start serializing
`B` until `A` is finished. (i.e., we want to prevent orders like `[A1, A2, B1,
...]`.) Using tokens in our state, we can write:

!!! NOTE
    This example will not work as written until DSLX supports multiple I/O
    operations per channel per activation.

```dslx-snippet
pub proc serialize {
  ch_in: chan<complex_struct> in;
  result: chan<u32> out;

  init { join() }

  config (ch_in: chan<complex_struct> in, result: chan<u32> out) {
    (ch_in, result)
  }

  next (tok: token) {
    let (input_tok, val) = recv(join(), ch_in);
    // ... (calculate val1)
    let tok = send(tok, result, val1);
    // ... (calculate val2)
    let tok = send(tok, result, val2);
    // ... (calculate val3)
    let tok = send(tok, result, val3);
    // ... (calculate val4)
    send(tok, result, val4)
  }
}
```

### Timing-Sensitive Operations

!!! WARNING
    Using timing-sensitive operations means that your circuit's behavior
    can depend on the exact details of scheduling. It is possible to write correct
    XLS code under these constraints, but the logic **must** work no matter how
    operations are pipelined, and **should** (within reason) be designed to be
    correct even if the number of stages in the pipeline varies. Testing/DV is
    substantially more difficult for timing-sensitive procs. As such, similar to how
    *unsafe* operations work in Rust, we recommend that you avoid using
    timing-sensitive operations except where strictly necessary. It's good practice
    to keep them confined to small well-understood procs that implement certain
    necessary behaviors.

Using streaming channels as documented above, there are some circuits that are
simply impossible to implement; for example, you cannot implement an
**arbiter**, a process that listens on multiple channels and forwards the
highest-priority message sending at any given time, while using only blocking
reads. All unimplementable circuits are **timing-sensitive**; their results
depend on *when* the inputs arrive on each channel, not just on the *order of
arrival*. This makes it much harder to write a provably correct design; however,
there are still times these circuits are needed!

For these times, XLS does include some timing-sensitive operations.

In particular, XLS has a **non-blocking receive** operation,
`recv_non_blocking(tok, ch, default)`. This attempts to read from the channel
`ch`. If the channel's queue is empty at the time of the read, it returns \
`(tok, default, false)`. Otherwise, it acts like a normal receive,
<span style="text-decoration:underline;">removing</span> the leading element
`data` from the `ch` queue and returning `(tok, data, true)`.

We can use this operation to implement a simple arbiter, combining two channels
(`data0_in` and `data1_in`) into one `result` by letting the higher-priority
data (from `data0_in`) through whenever it is ready with no delay, and sending
the lower-priority data (from `data1_in`) only when there's no higher-priority
message to send:

```dslx
struct Message {
  value1: u32,
  value2: u64,
  value3: bool
}

pub proc priority_arbiter {
  data0_in: chan<Message> in;
  data1_in: chan<Message> in;
  result: chan<(u1, Message)> out;

  init { () }

  config (data0_in: chan<Message> in, data1_in: chan<Message> in, result: chan<(u1, Message)> out) {
    (data0_in, data1_in, result)
  }

  next (st: ()) {
    let (tok, data0_msg, data0_valid) = recv_non_blocking(join(), data0_in, zero!<Message>());
    let (tok, data1_msg, data1_valid) = recv_if_non_blocking(tok, data1_in, !data0_valid, zero!<Message>());
    let (source, to_send) = if (data0_valid) {
      (u1:0, data0_msg)
    } else {
      (u1:1, data1_msg)
    };
    send_if(tok, result, data0_valid || data1_valid, (source, to_send));
  }
}
```
