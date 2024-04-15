# XLS: Elaboration

[TOC]

XLS IR has a notion of instantiation and elaboration similar to RTL.[^1]
A hierarchy is a directed acyclic graph of elements connected via instantiation.
An elaboration flattens the hierarchy into a tree by walking all paths in the
hierarchy starting at a `top` proc/block where a path is a chain of
instantiations. The elaboration creates an "instance" object for each path
through the hierarchy from the top proc/block to each IR construct (channel or
instantiation).

This outlines what elaboration looks like for each kind of `FunctionBase`.

## Function

Functions are "instantiated" via `invoke` operations. `opt` inlines all function
invocations, so there is no separate elaboration of functions.

## Proc

New-style procs can instantiate child procs. They have a notion of "channel
references" that are a parent proc can bind to a child proc via instantiation,
which is how procs communicate with the procs they instantiate. These channel
references are bound to channel instances during elaboration.

Example proc hierarchy:

```
  proc leaf_proc<ch0: ... in, ch1: .... out>(...) { }

  proc other_proc<x: ... in, y: .... out>(...) {
    chan z(...)
    proc_instantiation other_inst0(x, z, proc=leaf_proc)
    proc_instantiation other_inst1(z, y, proc=leaf_proc)
  }

  proc my_top<a: ... in, b: ... out>(...) {
    chan c(...)
    chan d(...)
    proc_instantiation my_inst0(a, b, proc=other_proc)
    proc_instantiation my_inst1(c, c, proc=other_proc)
    proc_instantiation my_inst2(d, d, proc=leaf_proc)
  }
```

Elaborating this hierarchy from `my_top` yields the following elaboration tree.
Each line is a instance of either a proc or a channel.

```
 <a, b>my_top
   chan c
   chan d
   other_proc<x=a, y=b> [my_inst0]
     chan z
     leaf_proc<ch0=x, ch1=z> [other_inst0]
     leaf_proc<ch0=z, ch1=y> [other_inst1]
   other_proc<x=c, y=c> [my_inst1]
     chan z
     leaf_proc<ch0=x, ch1=z> [other_inst0]
     leaf_proc<ch0=z, ch1=y> [other_inst1]
   leaf_proc<ch0=d, ch1=d> [my_inst2]
```

There are five instances of `leaf_proc` as there are five paths from `top_proc`
to `leaf_proc` in the proc hierarchy.

## Block

Blocks can instantiate:

*   Other blocks
*   FIFOs
*   External modules

Blocks interact with their instantiations via `InstantiationInput` and
`InstantiationOutput` operations, which bind a value to a named port on the
instantiation.

A block elaboration has an instance for each instantiation and builds maps from
parent→child and child→parent ports or instantiation input/output ops.

### Topological Sorting

Elaborated blocks can be topo sorted via `ElaboratedTopoSort()`. Each element in
the sort is a pair of `Node*` and `BlockInstance*`. This means that blocks
instantiated multiple times will have their nodes appear in the sort multiple
times (once for each instantiation).

`InstantiationInput/InputPort` and `OutputPort/InstantiationOutput` pairs are
treated as edges in the DAG. This is in contrast to the "unelaborated"
`TopoSort()` function which only produces a topo sort for an isolated `Block`.
An example use-case of `ElaboratedTopoSort()` is evaluating IR with dependencies
that span the hierarchy.

Note that our implementations of `ElaboratedTopoSort()` and `TopoSort()` produce
the same order for blocks that have no instantiatiations.

### Visitors

In the same way that `ElaboratedTopoSort()` extends `TopoSort()` to elaborated
blocks, `ElaboratedBlockDfsVisitor` extends `DfsVisitor` to elaborated blocks.
The main difference is the signature for op handlers. For `DfsVisitor`, they
take the form `HandleNodeT(NodeT* node)`, but for `ElaboratedBlockDfsVisitor`
they take the form `HandleNodeT(NodeT* node, BlockInstance* instance)`.

Like `ElaboratedTopoSort()`, `InstantiationInput/InputPort` and
`OutputPort/InstantiationOutput` pairs are treated as edges in the DAG.

[^1]: Note that elaboration is internal to the compiler, users generally don't
    need to be aware of or orchestrate an elaboration.
