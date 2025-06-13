# Tutorial: Dataflow & Time in XLS

## Introduction/Background

XLS IR and DSLX represent every aspect of a computation in terms of a *dataflow
graph*. Each operation (even reading from a parameter) is represented by a node;
if operation Y depends on the result of operation X, this is represented by an
edge from X to Y.[^1] If two operations have externally-visible effects (e.g.,
`send` and `receive` operations) and the order of the effects might matter even
if there is no data dependency, these operations will both take and produce a
*token*, which can be used to add dependency edges.[^2]

These graphs are *acyclic*; there is no path from any node back to itself. Any
form of recurrent computation (restricted to procs [or blocks, in the backend])
is therefore represented by an operation rather than a data dependency: reading
from or writing to a state parameter (in procs), or reading from or writing to a
register (in blocks). [^3]

Since these dataflow graphs are DAGs (directed acyclic graphs), they correspond
to a *weak partial order* on the set of operations; we know that if there is a
path of edges from X to Y, then X cannot execute after Y. Since this is a
**partial** order, there may not be a constraint in either direction; the
sequencing of X and Y may be unspecified. Also, since this is a **weak** order,
our computational model allows X to execute at the same "time" as Y.

> **NOTE:** "Time" is in scare quotes in the previous sentence for good reason!
>
> In reality, dependent operations can't execute at the same actual point in
> time, as it takes some amount of time for each operation to complete & for the
> data to move. However, the computational model at the heart of XLS
> intentionally leaves this detail out, separating the computation from the
> problem of making sure all combinational paths resolve within a clock cycle.

There are multiple ways to think about this partial order, and each corresponds
to its own useful perspective.

## Perspectives

### The Direct View (partially-ordered time)

We can, with some effort, think about this weak partial order directly! For
each pair of nodes, we know that whether or not there is a path from A to B, or
from B to A. Since our graph is acyclic, we know these can't be true
simultaneously, so we have three possibilities: A happens no later than B, B
happens no later than A, or **we have no information** and the ordering between
them is fully unspecified. This gives us an enormous amount of freedom for
purposes of dataflow analysis, optimizations, etc.

The dataflow graph fully determines the results of every operation that is
timing-insensitive (sometimes called "KPN-compatible") that completes
successfully (i.e., no deadlocks occur). However, this does not in general
determine the results of timing-sensitive operations, such as
`recv_non_blocking` (as well as other proposed non-KPN operations like `peek`);
more information is needed, both external and internal. External information may
include I/O constraints as well as the exact timing of various inputs &
responses. Internal information mostly means the actual execution order, which
depends both on the schedule and the final implementation.

Actually executing this dataflow graph in either serialized or pipelined form
requires us to take this order and *complete* it, deciding for each pair of
nodes whether A comes before or after B, or if A is instead simultaneous with B.
This amounts to turning it into one variation or other of a total order! This
process is also known as *scheduling*.

Other forms of dataflow graph execution exist, but the simpler ones start to
look more like fully-asynchronous circuitry, with nodes executing as their
inputs become ready & storing outputs until their dependents are ready to
receive them. This creates a large amount of overhead in a HW implementation, so
it's often not considered practical for ASIC design.[^4]

### Topological Sorting (Linearization)

We can view our weak partial order as if it were a **strong total order**,
adding nodes to an ordered list in some sequence where no node ever comes before
one of its predecessors. This is more commonly known as a topological sort.

This is how we represent the dataflow graph in text (in both DSLX and XLS IR),
since we need to pick some order in which to write things down. It's also what
all of our pre-scheduling interpreters & JITs do (for both functions and procs);
it's extremely efficient to do, always represents a legal evaluation/scheduling
of the computation, and guarantees that we don't need to deal with emulating
simultaneous execution.

Its downsides include that it arbitrarily chooses one order out of the set of
allowed orders, so anything timing-sensitive not reflected in the dataflow graph
may be missed, and it never deals with the possibility of simultaneous
resolution. It is also usually inefficient as a computational order when
implemented on anything but a serial processor.

### ASAP Dispatch

We can view our weak partial order as a **weak total order** by grouping all
nodes whose predecessors have already been addressed into a set together, then
iterating. We consider all nodes grouped together to be *simultaneous* with
respect to each other, but to come after all nodes in the earlier groups. The
result is more commonly known as an ASAP schedule.

This is as efficient as a topological sort, always represents a legal
evaluation/scheduling of the computation, and involves no arbitrary choices. It
also includes many cases of simultaneous resolution!

This can sometimes be inefficient from a storage/register perspective, and of
course it can be difficult to implement accurately on a serial CPU. We should
also keep in mind that despite initial appearances, the ASAP schedule is **not**
strictly speaking a worst-case schedule from the perspective of timing-sensitive
operations; it can reveal some timing-dependent problems, but this can also hide
the possibility that operation Y **could** be scheduled after operation X, even
though there is no dependency between them. For example, suppose we have the
sequence of operations:

```
let x: bits[32] = a + b;
let y: bits[32] = b + c;
let z: bits[32] = y + a;
let t: token = join();
let s1: token = send(t, ch1, x);
let s2: token = send(t, ch2, z);
```

In an ASAP context, this will always be scheduled as:

```
Step 0: {x, y, t}
Step 1: {z, s1}
Step 2: {s2}
```

However, it's completely legal for `s1` & `s2` to execute simultaneously, or
even for `s2` to finish resolving before `s1` begins! As such, since XLS
doesn't use a pure ASAP scheduler, we can't rely on this for safety - and in
fact, even if our interpreters/JITs used an ASAP schedule, there would still be
circumstances where the results differed between the (pre-scheduling) proc
interpreter & (post-scheduling) block interpreter.

## But what about...

### ... loopback channels?

From a scheduling perspective, these act like normal channels; the fact that
they connect one part of a proc to another part of the same proc is irrelevant.

### ... state parameters/variables?

These don't **natively** factor into the scheduling, and behave (from a dataflow
perspective) like any other input/output interface. During actual scheduling, we
normally try to keep the recurrence as short as possible, since longer
recurrences produce lower throughput.

<!-- Footnotes themselves at the bottom. -->

[^1]: This doesn't fully specify things for operations where operand order
    matters, since we don't know which incoming edge provides each operand...
    we fix this by augmenting Y with an order on the set of incoming edges.
[^2]: Note that operations that have no externally-visible effects should not
    take or produce tokens. When operations occur entirely internal to a
    function or proc, the only order that needs to be enforced is the
    data-dependency order.
[^3]: For example, if you want to write a loop where the number of iterations is
    data-dependent, you will likely need to implement a proc that stores the
    data being communicated between iterations in its state elements.
[^4]: For a good example of a project that starts with fully-dynamic execution
    and then tries to remove the overhead where possible, check out
    [Dynamatic](https://dynamatic.epfl.ch/).
