<div align='center'>
<img src='https://google.github.io/xls/images/xls_logo_623_250.png' alt='XLS Logo'>
</div>

# **XLS**: Accelerated HW Synthesis

[**Docs**](https://google.github.io/xls/) | [**Quick Start**](https://google.github.io/xls/tools_quick_start/)

## What is XLS?

The XLS (Accelerated HW Synthesis) project aims to enable the rapid development
of *hardware IP* that also runs as efficient *host software* via "software
style" methodology.

XLS implements a High Level Synthesis (HLS) toolchain which produces
synthesizable designs from flexible, high-level descriptions of functionality.
It is fully Open Source: Apache 2 licensed and developed via GitHub.

XLS is used inside of Google for generating feed-forward pipelines from
"building block" routines / libraries that can be easily retargeted, reused, and
composed in a latency-insensitive manner.

*Not yet available*, but active work in progress is the implementation of XLS
*concurrent processes*, in Communicating Sequential Processes (CSP) style, that
allow pipelines to communicate with each other and induct over time.

XLS is still experimental, undergoing rapid development, and not an officially
supported Google product. Expect bugs and sharp edges. Please help by trying it
out, [reporting bugs](https://github.com/google/xls/issues), and letting us know
what you think!

## Building From Source

Currently, XLS must be built from source using the Bazel build system.

*Note:* Binary distributions of the XLS library are not currently available, but
we hope to enable them via continuous integration, [see this issue](https://github.com/google/xls/issues/108).

The following instructions are for the Ubuntu 20.04 (Focal) Linux distribution.
Note that we start by assuming [Bazel has been
installed](https://docs.bazel.build/versions/master/install-ubuntu.html).

```console
# Follow the bazel install instructions:
# https://docs.bazel.build/versions/master/install-ubuntu.html
#
# Afterwards we observe:

$ bazel --version
bazel 3.2.0

$ sudo apt install python3-dev python3-distutils python3-dev libtinfo5

# py_binary currently assume they can refer to /usr/bin/env python
# even though Ubuntu 20.04 has no `python`, only `python3`.
# See https://github.com/bazelbuild/bazel/issues/8685

$ mkdir -p $HOME/opt/bin/
$ ln -s $(which python3) $HOME/opt/bin/python
$ echo 'export PATH=$HOME/opt/bin:$PATH' >> ~/.bashrc
$ source ~/.bashrc

$ bazel test -c opt ...
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

  * [`build`](https://github.com/google/xls/tree/main/xls/build): Build macros
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
  * [`dslx/fuzzer`](https://github.com/google/xls/tree/main/xls/dslx/fuzzer): A
    whole-stack multiprocess fuzzer that generates programs at the DSL level and
    cross-compares different execution engines (DSL interpreter, IR interpreter,
    IR JIT, code-generated-Verilog simulator). Designed so that it can easily be
    run on different nodes in a cluster simultaneously and accumulate shared
    findings.
  * [`examples`](https://github.com/google/xls/tree/main/xls/examples): Example
    computations that are tested and executable through the XLS stack.
  * [`experimental`](https://github.com/google/xls/tree/main/xls/experimental):
    Artifacts captured from experimental explorations.
  * [`ir`](https://github.com/google/xls/tree/main/xls/ir):
    XLS IR definition, text parser/formatter, and facilities for abstract
    evaluation and execution engines ([IR interpreter](interpreters.md),
    [JIT](ir_jit.md)).
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
    computations.
  * [`solvers`](https://github.com/google/xls/tree/main/xls/solvers):
    Converters from XLS IR into SMT solver input, such that formal proofs can be
    run on XLS computations; e.g. Logical Equalence Checks between XLS IR and a
    netlist description.
  * [`synthesis`](https://github.com/google/xls/tree/main/xls/synthesis):
    Interface that wraps backend synthesis flows, such that tools can be
    retargeted e.g. between ASIC and FPGA flows.
  * [`tests`](https://github.com/google/xls/tree/main/xls/tests):
    Integration tests that span various top-level components of the XLS project.
  * [`tools`](https://github.com/google/xls/tree/main/xls/tools):
    [Many tools](tools.md) that work with the XLS system and its libraries in a
    decomposed way via command line interfaces.
  * [`uncore_rtl`](https://github.com/google/xls/tree/main/xls/uncore_rtl):
    Helper RTL that interfaces XLS-generated blocks with device top-level for e.g.
    FPGA experiments.
  * [`visualization`](https://github.com/google/xls/tree/main/xls/visualzation):
    Visualization tools to inspect the XLS compiler/system interactively. See
    [IR visualization](ir_visualization.md).


## Community

Discussions about XLS - development, debugging, usage, and anything else -
should go to the [xls-dev mailing list](https://groups.google.com/g/xls-dev).

## Contributors

The following are
[contributors](https://github.com/google/xls/graphs/contributors) to the XLS
project:

* [Brandon Jiang](https://github.com/brajiang)
* [Chris Leary](https://github.com/cdleary)
* [Derek Lockhart](https://github.com/dmlockhart)
* [Hans Montero](https://github.com/hmontero1205)
* [Jonathan Bailey](https://github.com/jbaileyhandle)
* [Julian Viera](https://github.com/julianviera99)
* [Kevin Harlley](https://github.com/kevineharlley)
* [Mark Heffernan](https://github.com/meheffernan)
* [Per Grön](https://github.com/per-gron)
* [Rebecca Chen (Pytype)](https://github.com/rchen152)
* [Robert Hundt](https://github.com/rhundt)
* [Rob Springer](https://github.com/RobSpringer)
* [Sean Purser-Haskell](https://github.com/spurserh)
