# XLS: IR Overview

[TOC]

Before providing a detailed specification of the IR, in this section we briefly
outline the ideas and philosophy behind the IR design and explain how to build,
modify, and navigate the IR.

The XLS IR is a dataflow-oriented IR that has the static-single-assignment (SSA)
property, but is specialized for generating circuitry. It started out as a
purely functional IR but over time more and more side-effecting operations had
to be introduced. Specifically:

*   XLS has a single IR representation which is used from the front-end down to
    the RTL-level. A single representation throughout the compiler enables
    maximal reuse of analysis and transformation components. Often compilers
    have different specialized IRs (or "dialects") for different levels of
    abstraction which can add complexity and inhibit reusability. However, in
    XLS this tradeoff between specialization and reusability is unnecessary
    because we start with a dataflow representation in the front end and can
    smoothly lower the IR down to the RTL-level which is itself dataflow.

*   XLS IR is *not* control-flow graph (CFG) based, as many other compiler
    infrastructures. The insight is that the CFG abstraction was developed to
    model serial execution on a CPU. In hardware, however, everything happens at
    all times and in parallel. A *sea-of-nodes* (SoN) representation much more
    closely resembles this reality, which is why we have chosen it.

    It further turns out that many optimization passes are rather trivial to
    implement in the SoN representation, in particular as it requires no
    explicit SSA updates. The SSA property is automatically maintained by the IR
    being functional.

TODO: High-level structure, package -> func,proc,block -> sea of nodes

TODO: How to navigate

TODO: Talk about basic types
