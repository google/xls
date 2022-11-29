<div align='center'>
<img src='https://google.github.io/xls/images/xls_logo_623_250.png' alt='XLS Logo'>
</div>

# **XLS**: Accelerated HW Synthesis

[**Docs**](https://google.github.io/xls/) | [**Quick Start**](https://google.github.io/xls/tools_quick_start/) | [**Tutorials**](https://google.github.io/xls/tutorials/) ![Ubuntu 20.04](https://github.com/google/xls/workflows/Continuous%20Integration/badge.svg) ![Ubuntu 22.04](https://github.com/google/xls/workflows/Nightly%20Ubuntu%2022.04/badge.svg)

## What is XLS?

XLS implements a High Level Synthesis (HLS) toolchain which produces
synthesizable designs (Verilog and SystemVerilog) from flexible, high-level
descriptions of functionality. It is fully Open Source: Apache 2 licensed and
developed via GitHub.

XLS (Accelerated HW Synthesis) aims to be the Software Development Kit (SDK) for
the End of Moore's Law (EoML) era. In this "age of specialization", software and
hardware engineers must do more co-design across their domain boundaries --
collaborate on shared artifacts, understand each other's cost models, and share
tooling/methodology. XLS attempts to leverage automation, software engineers,
and machine cycles to accelerate this overall process.

In effect, XLS enables the rapid development of *hardware IP* that also runs as
efficient *host software* via "software style" methodology. A single XLS source
program runs at native speeds for use in host software or a simulator, but that
source can also be used to generate hardware block output -- the XLS tools'
correctness ensures (and provides tools to help formally verify) that they are
functionally identical.

XLS is used inside of Google for generating feed-forward pipelines from
"building block" routines / libraries that can be easily retargeted, reused, and
composed in a latency-insensitive manner.

XLS also supports
[*concurrent processes*](https://google.github.io/xls/tutorials/intro_to_procs/),
in Communicating Sequential Processes (CSP) style, that allow pipelines to
communicate with each other and induct over time. This feature is still under
active development but today supports base use cases.

XLS is still experimental, undergoing rapid development, and not an officially
supported Google product. Expect bugs and sharp edges. Please help by trying it
out, running through [some tutorials](https://google.github.io/xls/tutorials/),
[reporting bugs](https://github.com/google/xls/issues), and letting us know
what you think!

## Building From Source

Currently, XLS must be built from source using the Bazel build system.

*Note:* Binary distributions of the XLS library are not currently available, but
we hope to enable them via continuous integration,
[see this issue](https://github.com/google/xls/issues/108).

The following instructions are for the Ubuntu 20.04 (Focal Fossa) and Ubuntu
22.04 (Jammy Jellyfish) Linux distributions.

We start by assuming
[Bazel has been installed](https://bazel.build/install/ubuntu).
On an average 8-core VM, a full initial build (including the C++ frontend) may take up to 6 hours.
A build without the C++ frontend may take about 2 hours. Please see the two corresponding
command lines below:

```console
~$ git clone https://github.com/google/xls.git
~$ cd xls

~/xls$ # Follow the bazel install instructions:
~/xls$ # https://bazel.build/install/ubuntu
~/xls$ # Afterwards we observe:
~/xls$ bazel --version
bazel 5.2.0

~/xls$ # Note we're going to tell Ubuntu that `/usr/bin/env python` is actually python3
~/xls$ # here, since that is not the case by default on Ubuntu 20.04.
~/xls$ # This is important. Without this step, you may experience cryptic error messages:
~/xls$ sudo apt install python3-distutils python3-dev libtinfo5 python-is-python3

~/xls$ # Now build/test in optimized build mode.
~/xls$ # If you don't plan on using the C++ frontend, which is not needed to get started,
~/xls$ # use this command line:
~/xls$ bazel test -c opt -- //xls/... -//xls/contrib/xlscc/...

~/xls$ # To build everything, including the C++ frontend:
~/xls$ bazel test -c opt -- //xls/...
```

Reference build/test environment setups are also provided via `Dockerfile`s:

```console
~$ git clone https://github.com/google/xls.git
~$ cd xls

~/xls$ # Several Dockerfiles are available to choose from:
~/xls$ docker build . -f Dockerfile-ubuntu-20.04 # Performs optimized build and test.
~/xls$ docker build . -f Dockerfile-ubuntu-20.10
~/xls$ docker build . -f Dockerfile-ubuntu-22.04
```

## Stack Diagram and Project Layout

Navigating a new code base can be daunting; the following description provides a
high-level view of the important directories and their intended organization /
purpose, and correspond to the components in this XLS stack diagram:

<div align='center'>
<img src='https://google.github.io/xls/images/xls_stack_diagram.png' alt='XLS Stack Diagram'>
</div>

* [`dependency_support`](https://github.com/google/xls/tree/main/dependency_support):
  Configuration files that load, build, and expose Bazel targets for *external*
  dependencies of XLS.
* [`docs`](https://github.com/google/xls/tree/main/docs): Generated documentation
  served via GitHub pages:
  [https://google.github.io/xls/](https://google.github.io/xls/)
* [`docs_src`](https://github.com/google/xls/tree/main/docs_src): Markdown file
  sources, rendered to `docs` via
  [mkdocs](https://google.github.io/xls/contributing/#rendering-documentation).
* [`xls`](https://github.com/google/xls/tree/main/xls): Project-named
  subdirectory within the repository, in common Bazel-project style.

  * [`build`](https://github.com/google/xls/tree/main/xls/BUILD): Build macros
    that create XLS artifacts; e.g. convert DSL to IR, create test targets for
    DSL code, etc.
  * [`codegen`](https://github.com/google/xls/tree/main/xls/codegen): Verilog
    AST (VAST) support to generate Verilog/SystemVerilog operations and FSMs.
    VAST is built up by components we call *generators* (e.g.
    PipelineGenerator, SequentialGenerator for FSMs) in the translation from XLS
    IR.
  * [`common`](https://github.com/google/xls/tree/main/xls/common): "base"
    functionality that layers on top of standard library usage. Generally we use
    [Abseil](https://abseil.io) versions of base constructs wherever possible.
  * [`contrib/xlscc`](https://github.com/google/xls/tree/main/xls/contrib/xlscc):
    Experimental C++ syntax support that targets XLS IR (alternative path to
    DSLX) developed by a sister team at Google, sharing the same open source /
    testing flow as the rest of the XLS project. May be of particular interest
    for teams with existing C++ HLS code bases.
  * [`data_structures`](https://github.com/google/xls/tree/main/xls/data_structures):
    Generic data structures used in XLS that augment standard libraries; e.g.
    BDDs, union find, min cut, etc.
  * [`delay_model`](https://github.com/google/xls/tree/main/xls/delay_model):
    Functionality to characterize, describe, and interpolate data delay for
    XLS IR operations on a target backend process. Already-characterized
    descriptions are placed in `xls/delay_model/models` and can be referred to via
    command line flags.
  * [`dslx`](https://github.com/google/xls/tree/main/xls/dslx): A DSL (called
    "DSLX") that mimics Rust, while being an immutable expression-language
    dataflow DSL with hardware-oriented features; e.g.  arbitrary bitwidths,
    entirely fixed size objects, fully analyzeable call graph. XLS team has found
    dataflow DSLs are a good fit to describe hardware as compared to languages
    designed assume von Neumann style computation.
  * [`fuzzer`](https://github.com/google/xls/tree/main/xls/fuzzer): A
    whole-stack multiprocess fuzzer that generates programs at the DSL level and
    cross-compares different execution engines (DSL interpreter, IR interpreter,
    IR JIT, code-generated-Verilog simulator). Designed so that it can easily be
    run on different nodes in a cluster simultaneously and accumulate shared
    findings.
  * [`examples`](https://github.com/google/xls/tree/main/xls/examples): Example
    computations that are tested and executable through the XLS stack.
  * [`experimental`](https://github.com/google/xls/tree/main/xls/experimental):
    Artifacts captured from experimental explorations.
  * [`interpreter`](https://github.com/google/xls/tree/main/xls/interpreter):
    Interpreter for XLS IR - useful for debugging and exploration. For cases
    needing throughput, consider using the JIT (below).
  * [`ir`](https://github.com/google/xls/tree/main/xls/ir):
    XLS IR definition, text parser/formatter, and facilities for abstract
    evaluation.
  * [`jit`](https://github.com/google/xls/tree/main/xls/jit):
    LLVM-based JIT for XLS IR. Enables native-speed execution of DSLX and XLS IR
    programs.
  * [`modules`](https://github.com/google/xls/tree/main/xls/modules):
    Hardware building block DSLX "libraries" (outside the DSLX standard library)
    that may be easily reused or instantiated in a broader design.
  * [`netlist`](https://github.com/google/xls/tree/main/xls/netlist): Libraries
    that parse/analyze/interpret netlist-level descriptions, as are
    generally given in simple structural Verilog with an associated cell library.
  * [`passes`](https://github.com/google/xls/tree/main/xls/passes): Passes that
    run on the XLS IR as part of optimization, before scheduling / code
    generation.
  * [`scheduling`](https://github.com/google/xls/tree/main/xls/scheduling):
    Scheduling algorithms, determine when operations execute (e.g. which
    pipeline stage) in a clocked design.
  * [`simulation`](https://github.com/google/xls/tree/main/xls/simulation):
    Code that wraps Verilog simulators and generates Verilog testbenches for XLS
    computations. [iverilog](https://github.com/steveicarus/iverilog) is
    currently used to simulate as it supports non-synthesizable testbench
    constructs.
  * [`solvers`](https://github.com/google/xls/tree/main/xls/solvers):
    Converters from XLS IR into SMT solver input, such that formal proofs can be
    run on XLS computations; e.g. Logical Equalence Checks between XLS IR and a
    netlist description. [Z3](https://github.com/Z3Prover/z3) is used as the
    solver engine.
  * [`synthesis`](https://github.com/google/xls/tree/main/xls/synthesis):
    Interface that wraps backend synthesis flows, such that tools can be
    retargeted e.g. between ASIC and FPGA flows.
  * [`tests`](https://github.com/google/xls/tree/main/xls/tests):
    Integration tests that span various top-level components of the XLS project.
  * [`tools`](https://github.com/google/xls/tree/main/xls/tools):
    [Many tools](https://google.github.io/xls/tools/) that work with the XLS
    system and its libraries in a decomposed way via command line interfaces.
  * [`uncore_rtl`](https://github.com/google/xls/tree/main/xls/uncore_rtl):
    Helper RTL that interfaces XLS-generated blocks with device top-level for e.g.
    FPGA experiments.
  * [`visualization`](https://github.com/google/xls/tree/main/xls/visualization):
    Visualization tools to inspect the XLS compiler/system interactively. See
    [IR visualization](https://google.github.io/xls/ir_visualization/).

## Community

Discussions about XLS - development, debugging, usage, and anything else -
should go to the [xls-dev mailing list](https://groups.google.com/g/xls-dev).

## Contributors

The following are
[contributors](https://github.com/google/xls/graphs/contributors) to the XLS
project, see our
[contributing documentation](https://google.github.io/xls/contributing/) and
[good first issues](https://github.com/google/xls/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)!

*   [Albert Magyar](https://github.com/albert-magyar)
*   [Amin Kalantar](https://github.com/aminiok1)
*   [Balint Christian](https://github.com/cbalint13)
*   [Blaok](https://github.com/Blaok)
*   [Brandon Jiang](https://github.com/brajiang)
*   [Brian Searls](https://github.com/briansrls)
*   [Chen-hao Chang](https://github.com/cchao)
*   [Chris Drake](https://github.com/cjdrake)
*   [Chris Leary](https://github.com/cdleary)
*   [Dan Killebrew](https://github.com/dkillebrew-g)
*   [Derek Lockhart](https://github.com/dmlockhart)
*   [Eric Astor](https://github.com/ericastor)
*   [Ethan Mahintorabi](https://github.com/QuantamHD)
*   [Felix Zhu](https://github.com/felixzhuologist)
*   [Georges Rotival](https://github.com/grotival)
*   [Hans Montero](https://github.com/hmontero1205)
*   [Iliyan Malchev](https://github.com/malchev)
*   [Johan Euphrosine](https://github.com/proppy)
*   [Jonathan Bailey](https://github.com/jbaileyhandle)
*   [Josh Varga](https://github.com/JoshVarga)
*   [Julian Viera](https://github.com/julianviera99)
*   [Kevin Harlley](https://github.com/kevineharlley)
*   [Leonardo Romor](https://github.com/lromor)
*   [Manav Kohli](https://github.com/manav-kohli)
*   [Mark Heffernan](https://github.com/meheff)
*   [Paul Rigge](https://github.com/grebe)
*   [Per Grön](https://github.com/per-gron)
*   [Ravi Nanavati](https://github.com/nanavati)
*   [Rebecca Chen (Pytype)](https://github.com/rchen152)
*   [Remy Goldschmidt](https://github.com/taktoa)
*   [Robert Hundt](https://github.com/rhundt)
*   [Rob Springer](https://github.com/RobSpringer)
*   [Sameer Agarwal](https://github.com/sandwichmaker)
*   [Sean Purser-Haskell](https://github.com/spurserh)
*   [Ted Hong](https://github.com/hongted)
*   [Ted Xie](https://github.com/ted-xie)
*   [Vincent Mirian](https://github.com/vincent-mirian-google)
