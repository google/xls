# Ideas and Projects

This document lists a few sample ideas, projects, and research ideas, to help
get started on contributing to XLS.

## Programming Languages

### XLS New Frontends

One of XLS' primary core focus areas was defining a compiler intermediate
representation that is powerful enough to express all required concepts, but
minimal enough to make it easy to target from other, new, or modified
programming languages. We are focusing on a functional domain-specific language,
but others are possible. There is work to target the IR from C++. There are many
other research systems out there with their own respective DSLs and other input
mechanisms. It would be interesting to connect those to allow comparisons. An
embedded DSL in Python could be developed, which is straightforward as the IR's
builder interfaces are already exported to Python.

## Core XLS

### ML for Delay Estimation

We currently estimate the delay of ops and op combos via benchmarks and the
theory of logical efforts. In principle we are trying to guess what the
commercial toolchains will do. This is a problem that seems to be just made for
ML, especially for new technology nodes or FPGA devices.

### Delay Estimation for a variety of Devices

We are focusing on only a small number of FPGAs and very specific ASIC Flows.
For the lager community out there we should add many more models, improve
automation of deriving a delay model, and/or try ML based approaches.

### Delay Estimation for Implicit Broadcasts

HLS often creates implicit broadcasts (wire load / fan out / wiring congestion)
in unrolled loops, deep pipelines, large memory blocks, etc. that lead to
frequency bottlenecks. Modeling these broadcasts in the delay model can help
mitigate or even completely solve such problems.

### Z3

We use Z3 for our logical equivalence checking, eg., to check that the optimized
and unoptimized IR have the same semantics. With a solver like this in place,
there are many more opportunities to apply it or make it more practical, for
example.

-   (Automatic) Partitioning of the input IR to reduce per-phase problem space
-   Add / compare with other formal verification tools.
-   There are other alternatives, e.g. [Boolector](https://boolector.github.io/)
-   Use cases we haven't thought of yet.

### Design Verification Flows

Provide mechanisms for "constrained random" approaches to hardware
verification -- either libraries to emulate capabilities of constraint-based
vector generation similar to UVM, or more automated approaches provided by
QuickCheck/Hypothesis.

## Implement new blocks and functions in XLS

### Extend XLS standard library

XLS has a small standard library which includes some basic utility functions and
some limited floating point support. Extending this library and developing more
complex functionality would improve the usability of XLS. Ideas:

-   FP Libraries Implement libraries for important FP operations, modes, widths,
    with all the relevant parameters. BFloat libraries
-   Fixed point libraries, similar to above
-   Exploit FPGA hard macros and BRAMs

XLS implementations of common hardware libraries - arbiters - counters -
encoders - fifos

### Three-stage RISC-V

Piccolo is a 3-stage RISC-V core. Implementation appears of reasonable size.

-   IF: ISA / Instruction Fetch / Decode Unit
-   DM: Data Memory stage
-   WB: Write Back stage
-   ALU
-   add/sub unit, mul unit, shifter unit
-   PLIC / Platform Level Interrupts architecture
-   Cache Hierarchy with various sizes, associativity, and replacement policies
-   MMU and L1-Cache
-   CSRs
-   Register File

### Support for Systolic Arrays as a "Parallel Pattern"

It is clear, especially in the domain of machine learning and signal processing,
that HLS will benefit from full support of 2D elements, such as systolic arrays
or specialized convolution engines. The idea here is to abstract these properly.
For example, one could add constructs to the IR, or we can extend the analysis
capabilities and add passes to construct them as needed. We also have to make
sure that the mechanisms are fully supported by all optimization and code
generation passes. This is a fascinating and wide-open research area and
intersects with the current work on concurrent blocks.

### Interoperability with other languages, FFI

Users may want to reuse building blocks written in other languages that are
pre-optimized. Support proper FFI. Instantiate external blocks by specifying
their properties in a way the scheduler can understand. Import type definitions
from system verilog descriptions e.g. as encoded in protobufs

## XLS Tools

### Visualization

XLS includes some minimal visualization and exploration tools. There is large
potential for tool builders to improve and add information, insights,
suggestions, etc.

### Source Correlation

It is always important to maintain a high level of productivity and utility for
a toolset like XLS provides. Source correlation, annotations, and many other
techniques are always improvable the enhance the debugging, visualization, and
also design verification experiences.

### Source Analysis

A linter/style checker for the DSL would be very helpful for users to write
high-quality HLS code.

## XLS Integration with other tools

### Add Verilator support to XLS

Verilator is a strong open-source (System)Verilog simulator. Currently XLS only
supports Icarus Verilog, a Verilog-only simulator. Verilator support in XLS will
provide much higher simulation performance and SystemVerilog support to open
source users. Unlike other simulators which can run testbench code, Verilator
only operates on synthesizable Verilog and emits C++ code for compilation and
execution. This unique flow needs to be integrating into XLS's simulation
framework.
