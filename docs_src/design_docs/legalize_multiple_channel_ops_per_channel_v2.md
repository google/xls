# Legalize Multiple Channel Ops Per Channel (v2)

[TOC]

Written on 2025-04-08.

## Objective

Make channel legalization simpler, while improving QoR.

## Background

XLS codegen does not support multiple I/O operations per channel. However, many
circuits naturally require this.

For example, hardware designers routinely want to have (e.g.) two `send_if`s on
the same channel where at most one triggers per activation. We handle these
scenarios today by applying our "proven mutually exclusive" channel strictness;
we call out to Z3 to prove that the predicates cannot be active at the same
time, then merge the `send` operations into a single node that sends the
appropriate data for whichever original operation was actually activated.
However, we know of some cases where this merge is impossible, as it would
create token cycles. For example, consider the token chain `send(channel=A) ->
send(channel=B) -> send(channnel=A)`; if we merge the outer sends, we would need
to create a token cycle between the merged channel-A send and the channel-B
send.

For cases where we can't prove mutual exclusion, whether due to limits of Z3 or
circuit design, we invoke our current channel legalization pass. This generates
a complex arbitrating adapter, which (roughly) reifies the token graph of the
original proc & regulates the order in which the actions are allowed to resolve,
and connects one channel for each operation to this adapter - which serves as an
arbitrating mux/demux for the external channel. The design document for this can
be found
[here](legalize_multiple_channel_ops_per_channel_v1.md#minimum-viable-product).
However, this is known to introduce QoR issues; we'd previously leaned heavily
on proc inlining in hopes that this would allow other optimizations to apply.

## Proposal

### Overview

Rather than incorporating arbitration logic into a complex multiplexing adapter,
we put it all into the original proc. At that point, we only need a
multiplexer - a simple (stateless) block-IR construct that fans out outputs &
ties inputs together with an OR-tree.

To make this multiplexer safe, we add logic to the proc to guarantee that all
operations on channel C wait for all operations in earlier activations on the
same channel to resolve (by threading tokens through the proc's state), and
separately add scheduling constraints to guarantee that no stage contains two
non-exclusive operations for the same channel.

### Arbitration Design

Suppose we have N > 1 operations on the same channel C:

`op1(tok1, C, ...)`, ..., `opN(tokN, C, ...)`.

During legalization, we take each token produced by these operations and store
them in the proc's state as

`implicit_token__op1`, ..., `implicit_token__opN`.

We then replace `opI`'s token parameter (`tokI`) with

`after_all(tokI, implicit_token__op1, ..., implicit_token__opN)`.

This guarantees that no operation on this channel can proceed until all
operations from earlier activations have resolved.

Note that if we had two operations on channel C that both trigger in the same
activation, but were scheduled into different stages, there would be no conflict
between those operations in this activation - and we've already ensured that
there can be no conflict between operations in different activations. Instead,
there would be a token-state backedge from the later stage to the earlier stage.
As with all state backedges, this would limit throughput, but prevent conflict.
(As a side effect, this means that with `--worst_case_throughput=N`, you can
have up to N unconditional operations on the same channel.)

Therefore, all we need to do is ensure that if two operations on channel C can
trigger in the same activation, they aren't scheduled in the same stage. How we
handle this depends on the strictness for channel C.

#### Proven Mutually Exclusive

We invoke Z3 to prove that all operations are in fact exclusive, producing an
error if not. No additional scheduling constraints are necessary. Note that we
no longer insist on merging operations; we can choose to merge I/O operations as
an optimization, but this is not required. (Further note: this eliminates a
known issue in our current proven-mutually-exclusive handling, where merging
operations can produce a cycle.)

#### Runtime Mutually Exclusive

We add an assert that at most one of the operations' predicates is true. No
additional scheduling constraints are necessary. (We can optionally merge I/O
operations if we believe it will enable new optimizations.)

#### Total Order

We check that all operations on channel C are in fact ordered, and produce an
error if not. We add a scheduling constraint (either directly or in the form of
a `min_delay` node) that each operation must be scheduled at least one stage
after all of its predecessors.

#### Proven Ordered

We add scheduling constraints between two operations on channel C if and only if
there is a token path between those operations. For any operations not connected
by a token path, we invoke Z3 to prove that they are mutually exclusive.

#### Runtime Ordered

We add scheduling constraints between two operations on channel C if and only if
there is a token path between those operations. For any operations not connected
by a token path, we add an assert that they are not both triggered in the same
activation.

#### Arbitrary Static Order

We add scheduling constraints that the operations on channel C must be scheduled
at least one stage apart in topological-sort order.

### Multiplexer Design

We want to ensure that the multiplexer is extremely minimal for PPA purposes,
while adding no additional latency or FIFO depth. Backpressure should still work
through a multiplexer.

Ideally, our multiplexer should reduce to nothing more than fanouts of incoming
signals (e.g., valid for a recv-side multiplexer) and OR-trees of outgoing
signals (e.g., ready for a recv-side multiplexer).

I don't believe implementing this type of multiplexer in proc IR is possible
without a way to make a receive conditional on whether the corresponding send
will succeed immediately; without this, I believe the best we can implement
would be equivalent to adding a depth-1 FIFO with bypass. As such, this will
need to be built in block IR during codegen.

We can implement this by:

1.  adding support for multiple mutually-exclusive operations per channel to
    channel lowering (currently found in block conversion), building the fanouts
    & OR-trees as part of lowering, or
2.  rewriting each operation to use a different channel (recording these as a
    group) before block conversion, generating the corresponding multiplexer as
    a separate block, and ensuring that stitching knows to connect the
    multiplexer's channels to the proc after lowering.

(1) would be a cleaner option, especially since I've been given to understand
that codegen plans to support multiple users of each channel-end soon; (2)
requires creating more metadata to be carried through codegen, but lets us avoid
complicating channel lowering.

### Optimization Options

To reduce state overhead, we could fold all of the implicit tokens for a single
channel together, taking the `after_all` of the operations' tokens before
putting the result in state. However, this may reduce opportunities for dynamic
state feedback. To recover these, we would need to split the `next_value` node
for the `after_all` based on the predicates for the operations that provide its
inputs, which seems potentially complicated.

To increase opportunities for dynamic state writes: if an operation is
conditional, copy its predicate into the `next_value` node we create for its
implicit token, so the implicit "no change" `next_value` can apply as soon as we
know the operation will not be triggered.

## Worked Example

Suppose we start with a DSLX proc with two dependent RAM accesses, containing
the following:

```
next(..., initial=(...)) {
  ...
  let tok = send(join(), ram_req, addr);
  let (tok, x) = recv(tok, ram_resp);
  let (tok, x) = if d {
   let t = send(tok, ram_req, addr + g(x));
   recv(t, ram_resp)
  } else {
   (tok, x)
  };
  ...
}
```

This compiles to something like the following XLS IR:

```
proc next(..., initial=(...)) {
  ...
  tok1: token = literal(value=token);
  send.4: token = send(tok1, addr, channel=ram_req);
  recv.5: (token, u64) = recv(send.4, channel=ram_resp);
  tok2: token = tuple_index(recv.5, 0);
  x0: u64 = tuple_index(recv.5, 1);
  g: u64 = ...; // depends on x0
  new_addr: u64 = add(addr, g);
  send.6: token = send(tok2, new_addr, predicate=d, channel=ram_req);
  recv.7: (token, u64) = recv(send.6, predicate=d, channel=ram_resp);
  x1: u64 = tuple_index(recv.7, 1);
  x: u64 = priority_sel(f, cases=[x1], default=x0);
  ...
}
```

Applying our cross-activation guarding transform, we would get something close
to: \
(some nodes named for convenience)

```
proc next(..., implicit_token__send_4, implicit_token__recv_5,
               implicit_token__send_6, implicit_token__recv_7,
          initial=(..., token, token, token, token)) {
  ...
  tok1: token = literal(value=token);
  send_4_tok: token = after_all(tok1, implicit_token__send_4,
                                      implicit_token__send_6);
  send.4: token = send(send_4_tok, addr, channel=ram_req);
  next_send_4_tok: () = next_value(state_read=implicit_token__send_4,
                                   value=send.4);
  recv_5_tok: token = after_all(send.4, implicit_token__recv_5,
                                        implicit_token__recv_7);
  recv.5: (token, u64) = recv(recv_5_tok, channel=ram_resp);
  tok2: token = tuple_index(recv.5, 0);
  next_recv_5_tok: () = next_value(state_read=implicit_token__recv_5,
                                   value=tok2)
  x0: u64 = tuple_index(recv.5, 1);
  g: u64 = ... // depends on x0
  new_addr: u64 = add(addr, g);
  send_6_tok: token = after_all(tok2, implicit_token__send_4,
                                      implicit_token__send_6);
  send.6: token = send(send_6_tok, new_addr, predicate=d, channel=ram_req);
  next_send_6_tok: () = next_value(state_read=implicit_token_send_6,
                                   value=send.6, predicate=d);
  recv_7_tok: token = after_all(send.6, implicit_token__recv_5,
                                        implicit_token__recv_7);
  recv.7: (token, u64) = recv(recv_7_tok, predicate=d, channel=ram_resp);
  tok3: token = tuple_index(recv.7, 0);
  next_recv_7_tok: () = next_value(state_read=implicit_token__recv_7,
                                   value=tok3, predicate=d);
  x1: u64 = tuple_index(recv.7, 1);
  x: u64 = priority_sel(f, cases=[x1], default=x0);
  ...
}
```

We then apply our scheduling constraints. Let's assume the channel strictness is
set to Total Order. Since `send.4` and `send.6` are on the same channel
(`ram_req`), we'd check and note that there's a token path `send.4 -> recv.5 ->
tok2 -> send_6_tok -> send.6`; therefore, we must schedule `send.4` strictly
before `send.6`. Similarly, `recv.5` must be scheduled strictly before `recv.7`.

One possible schedule could have:

Stage 3:

*   `d`
*   `send.4`
*   `next_send_4_tok`

Stage 4:

*   `recv.5`
*   `next_recv_5_tok`

Stage 5:

*   `g`
*   `send.6`
*   `next_send_6_tok`

Stage 6:

*   `recv.7`
*   `next_recv_7_tok`

In this case, we find that there's a (predicated) state backedge from stage 5 to
stage 3 (from `next_send_6_tok` to `send.4`) as well as one from stage 6 to
stage 4 (from `next_recv_7_tok` to `recv.5`). This means we must have worst-case
throughput at least 3; for example, if activation 1's `f` is true, activation 2
will not be able to start its stage 3 until activation 1 has already finished
its stage 5, forcing the circuit to take at least 3 cycles per result.

On the other hand, if activation 1's `d` is false, the implicit "no change"
`next_value` for `implicit_token__send_6` will activate - and it will almost
certainly be scheduled in stage 3, the same stage as `send.4`, since `d` is
resolved in that stage. Similar logic applies to `recv.7`; the corresponding "no
change" `next_value` will end up scheduled into stage 4, the same stage as
`recv.4`. In other words, we generate a circuit that achieves full throughput as
long as `f` is false, and automatically introduces a 2-cycle bubble as necessary
to accommodate instances where `f` is true.

In all cases, however, we end up with at most one of the operations on each
channel active at any given time.
