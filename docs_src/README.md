<div align='center'>
<img src='https://google.github.io/xls/images/xls_logo.svg' alt='XLS Logo'>
</div>

# **XLS**: Accelerated HW Synthesis

<!-- GitHub banner -->

## What is XLS? What isn't XLS?

XLS implements a High Level Synthesis toolchain that produces synthesizable
designs (Verilog and SystemVerilog) from flexible, high-level descriptions of
functionality. It is Apache 2 licensed.

XLS (Accelerated HW Synthesis) aims to be the Software Development Kit (SDK) for
the End of Moore's Law (EoML) era. In this "age of specialization", software and
hardware engineers must do more co-design across their domain boundaries --
collaborate on shared artifacts, understand each other's cost models, and share
tooling/methodology. XLS attempts to leverage automation, software engineers,
and machine cycles to accelerate this overall process.

XLS enables the rapid development of *hardware IP* that also runs as efficient
*host software* via "software style" methodology. An XLS design runs at native
speeds for use in host software or a simulator, but that design can also
generate hardware block output -- the XLS tools' correctness ensures (and
provides tools to help formally verify) that they are functionally identical.

XLS supports both (optionally pipelined) functions with pure-wire I/O interfaces
and
[*concurrent processes*](https://google.github.io/xls/tutorials/what_is_a_proc/)
(or `proc`s). Procs are stateful, allowing induction over time, and include more
general communication interfaces.

## State of the Project

XLS is experimental, undergoing rapid development, and not an officially
supported Google product. Expect bugs and sharp edges. Please help by trying it
out, running through [some tutorials](https://google.github.io/xls/tutorials/),
[reporting bugs](https://github.com/google/xls/issues).

We are early stage and this has some practical effects:

-   We welcome your issues and PRs.
    -   Please try to lead with an issue. Engage us in conversation if you wish
        to upstream changes. Sending a PR without back and forth with us in an
        issue may be a longer road to success. If you believe your PR is ready
        and has not received a response within two business days, please ping
        the issue with what you think are next steps.
-   At the current point in its evolution, we regularly improve DSLX without
    considering backward compatibility.
    -   If you are building a corpus of hardware with XLS, please be thoughtful
        about your process for bringing in new versions of the compiler.

## Colab Notebooks

For a more setup-free and environment-independent way of trying out XLS, see our
colab notebooks:

-   [bit.ly/learn-xls](https://bit.ly/learn-xls): a "learn XLS in Y minutes"
    style walkthrough in DSLX, our Rust-inspired domain specific language (DSL).

-   [bit.ly/xls-playground](https://bit.ly/xls-playground): an XLS evaluation
    environment that can run the following interactively:

    -   XLS tests
    -   XLS→IR conversion
    -   IR→Verilog codegen
    -   Verilog synthesis via Yosys (using open PDKs ASAP7 and SKY130)
    -   Place-and-Route (P&R) via OpenROAD
    -   Power/Performance/Area (PPA) metric collection

## Install Latest Release

The following downloads the latest github repo release binaries for an x64 Linux
machine:

```bash
# Determine the url of the latest release tarball.
LATEST_XLS_RELEASE_TARBALL_URL=$(curl -s -L \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/google/xls/releases | \
  grep -m 1 -o 'https://.*/releases/download/.*\.tar\.gz')

# Download the tarball and unpack it, observe the version numbers for each of the included tools.
curl -O -L ${LATEST_XLS_RELEASE_TARBALL_URL}
tar -xzvvf xls-*.tar.gz
cd xls-*/
./interpreter_main --version
./ir_converter_main --version
./opt_main --version
./codegen_main --version
./proto_to_dslx_main --version
```

## Building From Source

Aside from the binary releases (available for x64 Linux as described above), and
the available colab notebooks, XLS must be built from source using the Bazel
build system.

The following instructions are for the Ubuntu 22.04 (Jammy Jellyfish) Linux
distribution.

On an average 8-core VM:

-   A full initial build **without the C++ front-end (e.g. "DSLX only") may take
    about 2 hours**,
-   **Including the C++ front-end may take up to 6 hours.**

Please see the two corresponding command lines below -- we start by assuming
[Bazel has been installed](https://bazel.build/install/ubuntu):

```console
~$ git clone https://github.com/google/xls.git
~$ cd xls

~/xls$ # Follow the bazel install instructions to install bazel 7
~/xls$ # https://bazel.build/install/ubuntu

~/xls$ # Note we're going to tell Ubuntu that `/usr/bin/env python` is actually python3
~/xls$ # here, since that has not been the case by default on past Ubuntus.
~/xls$ # This is important. Without this step, you may experience cryptic error messages:
~/xls$ sudo apt install python3-dev libtinfo6 python-is-python3

~/xls$ # Now build/test in optimized build mode.
~/xls$ # If you don't plan on using the C++ front-end, which is not strictly
~/xls$ # needed (i.e. DSLX front-end only), use this command line:
~/xls$ bazel test -c opt -- //xls/... -//xls/contrib/xlscc/...

~/xls$ # To build everything, including the C++ front-end:
~/xls$ bazel test -c opt -- //xls/...
```

Reference build/test environment setups are also provided via `Dockerfile`s, if
you have difficulty setting up the (limited set of) dependencies shown above in
your environment:

```console
~$ git clone https://github.com/google/xls.git
~$ cd xls
~/xls$ docker build . -f Dockerfile-ubuntu-22.04  # Performs optimized build-and-test.
```

### Adding Additional Build Caching

Many programmers are used to using programs like `ccache` to improve caching for
a build, but Bazel actually ships with very-high quality caching layers. In
particular, incremental builds are more safe.

However, there are circumstances where Bazel might decide to recompile files
where the results could have been cached locally - or where it might be safe to
reuse certain intermediate results, even after a `bazel clean`. To improve this,
you can tell Bazel to use a shared "disk cache", storing files persistently
elsewhere on disk; just create a directory somewhere (e.g.,
`~/.bazel_disk_cache/`), and then run:

```bash
echo "build --disk_cache=$(realpath ~/.bazel_disk_cache)" >> ~/.bazelrc
echo "test --disk_cache=$(realpath ~/.bazel_disk_cache)" >> ~/.bazelrc
```

!!! WARNING
    Bazel does not automate garbage collection of this directory, so it
    will grow over time without bounds. You will need to clean it up periodically,
    either manually or with an automated script.

Alternatively, you can add a [remote cache](https://bazel.build/remote/caching)
that takes care of garbage collection for you. This can be hosted on a personal
server or even on the local machine. We've personally had good results with
localhost instances of [bazel-remote](https://github.com/buchgr/bazel-remote/).

### Getting Clangd completions

A `compile_flags.txt` file compatible with clangd and similar tools can be
created by running `xls/dev_tools/make-compilation-db.sh`. Follow directions for
your editor to install clangd code completion.

## Stack Diagram and Project Layout

Navigating a new code base can be daunting; the following description provides a
high-level view of the important directories and their intended organization /
purpose, and correspond to the components in this XLS stack diagram:

<div align='center'>
<img src='https://google.github.io/xls/images/xls_stack_diagram.png' alt='XLS Stack Diagram'>
</div>

-   [`dependency_support`](https://github.com/google/xls/tree/main/dependency_support):
    Configuration files that load, build, and expose Bazel targets for
    *external* dependencies of XLS.
-   [`docs_src`](https://github.com/google/xls/tree/main/docs_src): Markdown
    file sources, rendered to `docs` via
    [mkdocs](https://google.github.io/xls/contributing/#rendering-documentation).
-   [`xls`](https://github.com/google/xls/tree/main/xls): Project-named
    subdirectory within the repository, in common Bazel-project style.
    -   [`build`](https://github.com/google/xls/tree/main/xls/BUILD): Build
        macros that create XLS artifacts; e.g. convert DSL to IR, create test
        targets for DSL code, etc.
    -   [`codegen`](https://github.com/google/xls/tree/main/xls/codegen):
        Verilog AST (VAST) support to generate Verilog/SystemVerilog operations
        and FSMs. VAST is built up by components we call *generators* (e.g.
        PipelineGenerator, SequentialGenerator for FSMs) in the translation from
        XLS IR.
    -   [`common`](https://github.com/google/xls/tree/main/xls/common): "base"
        functionality that layers on top of standard library usage. Generally we
        use [Abseil](https://abseil.io) versions of base constructs wherever
        possible.
    -   [`contrib/xlscc`](https://github.com/google/xls/tree/main/xls/contrib/xlscc):
        Experimental C++ syntax support that targets XLS IR (alternative path to
        DSLX) developed by a sister team at Google, sharing the same open source
        / testing flow as the rest of the XLS project. May be of particular
        interest for teams with existing C++ HLS code bases.
    -   [`data_structures`](https://github.com/google/xls/tree/main/xls/data_structures):
        Generic data structures used in XLS that augment standard libraries;
        e.g. BDDs, union find, min cut, etc.
    -   [`delay_model`](https://github.com/google/xls/tree/main/xls/estimators/delay_model):
        Functionality to characterize, describe, and interpolate data delay for
        XLS IR operations on a target backend process. Already-characterized
        descriptions are placed in `xls/estimators/delay_model/models` and can
        be referred to via command line flags.
    -   [`dslx`](https://github.com/google/xls/tree/main/xls/dslx): A DSL
        (called "DSLX") that mimics Rust, while being an immutable
        expression-language dataflow DSL with hardware-oriented features; e.g.
        arbitrary bitwidths, entirely fixed size objects, fully analyzeable call
        graph. XLS team has found dataflow DSLs are a good fit to describe
        hardware as compared to languages designed assume von Neumann style
        computation.
    -   [`fuzzer`](https://github.com/google/xls/tree/main/xls/fuzzer): A
        whole-stack multiprocess fuzzer that generates programs at the DSL level
        and cross-compares different execution engines (DSL interpreter, IR
        interpreter, IR JIT, code-generated-Verilog simulator). Designed so that
        it can easily be run on different nodes in a cluster simultaneously and
        accumulate shared findings.
    -   [`examples`](https://github.com/google/xls/tree/main/xls/examples):
        Example computations that are tested and executable through the XLS
        stack.
    -   [`experimental`](https://github.com/google/xls/tree/main/xls/experimental):
        Artifacts captured from experimental explorations.
    -   [`interpreter`](https://github.com/google/xls/tree/main/xls/interpreter):
        Interpreter for XLS IR - useful for debugging and exploration. For cases
        needing throughput, consider using the JIT (below).
    -   [`ir`](https://github.com/google/xls/tree/main/xls/ir): XLS IR
        definition, text parser/formatter, and facilities for abstract
        evaluation.
    -   [`jit`](https://github.com/google/xls/tree/main/xls/jit): LLVM-based JIT
        for XLS IR. Enables native-speed execution of DSLX and XLS IR programs.
    -   [`modules`](https://github.com/google/xls/tree/main/xls/modules):
        Hardware building block DSLX "libraries" (outside the DSLX standard
        library) that may be easily reused or instantiated in a broader design.
    -   [`netlist`](https://github.com/google/xls/tree/main/xls/netlist):
        Libraries that parse/analyze/interpret netlist-level descriptions, as
        are generally given in simple structural Verilog with an associated cell
        library.
    -   [`passes`](https://github.com/google/xls/tree/main/xls/passes): Passes
        that run on the XLS IR as part of optimization, before scheduling / code
        generation.
    -   [`scheduling`](https://github.com/google/xls/tree/main/xls/scheduling):
        Scheduling algorithms, determine when operations execute (e.g. which
        pipeline stage) in a clocked design.
    -   [`simulation`](https://github.com/google/xls/tree/main/xls/simulation):
        Code that wraps Verilog simulators and generates Verilog testbenches for
        XLS computations. [iverilog](https://github.com/steveicarus/iverilog) is
        currently used to simulate as it supports non-synthesizable testbench
        constructs.
    -   [`solvers`](https://github.com/google/xls/tree/main/xls/solvers):
        Converters from XLS IR into SMT solver input, such that formal proofs
        can be run on XLS computations; e.g. Logical Equalence Checks between
        XLS IR and a netlist description. [Z3](https://github.com/Z3Prover/z3)
        is used as the solver engine.
    -   [`synthesis`](https://github.com/google/xls/tree/main/xls/synthesis):
        Interface that wraps backend synthesis flows, such that tools can be
        retargeted e.g. between ASIC and FPGA flows.
    -   [`tests`](https://github.com/google/xls/tree/main/xls/tests):
        Integration tests that span various top-level components of the XLS
        project.
    -   [`tools`](https://github.com/google/xls/tree/main/xls/tools):
        [Many tools](https://google.github.io/xls/tools/) that work with the XLS
        system and its libraries in a decomposed way via command line
        interfaces.
    -   [`visualization`](https://github.com/google/xls/tree/main/xls/visualization):
        Visualization tools to inspect the XLS compiler/system interactively.
        See [IR visualization](https://google.github.io/xls/ir_visualization/).

## Community

Discussions about XLS - development, debugging, usage, etc:

-   Ideally happen in the
    [XLS repo GitHub discussions](https://github.com/google/xls/discussions)
-   But, if you feel email is a better venue for the discussion, there is also
    an [xls-dev mailing list](https://groups.google.com/g/xls-dev) -- please
    prefer GitHub discussions if possible as they are searchable and can be
    easily cross-referenced and converted to the issue tracker

## Contributors

The following are
[contributors](https://github.com/google/xls/graphs/contributors) to the XLS
project, see our
[contributing documentation](https://google.github.io/xls/contributing/) and
[good first issues](https://github.com/google/xls/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
if you're interested in contributing, or reach out via
[GitHub discussions](https://github.com/google/xls/discussions)!

-   [Aidan Kirk](https://github.com/aidankirk12)
-   [Albert Magyar](https://github.com/albert-magyar)
-   [Alex Light](https://github.com/allight)
-   [Amin Kalantar](https://github.com/aminiok1)
-   [Balint Christian](https://github.com/cbalint13)
-   [Blaok](https://github.com/Blaok)
-   [Brandon Jiang](https://github.com/brajiang)
-   [Brian Searls](https://github.com/briansrls)
-   [Chen-hao Chang](https://github.com/cchao)
-   [Chris Drake](https://github.com/cjdrake)
-   [Chris Leary](https://github.com/cdleary)
-   [Conor McCullough](https://github.com/crmymh)
-   [Dan Killebrew](https://github.com/dkillebrew-g)
-   [Derek Lockhart](https://github.com/dmlockhart)
-   [Eric Astor](https://github.com/ericastor)
-   [Ethan Mahintorabi](https://github.com/QuantamHD)
-   [Felix Zhu](https://github.com/felixzhuologist)
-   [Georges Rotival](https://github.com/grotival)
-   [Hanchen Ye](https://github.com/hanchenye)
-   [Hans Montero](https://github.com/hmontero1205)
-   [Henner Zeller](https://github.com/hzeller)
-   [Iliyan Malchev](https://github.com/malchev)
-   [Johan Euphrosine](https://github.com/proppy)
-   [Jonathan Bailey](https://github.com/jbaileyhandle)
-   [Josh Varga](https://github.com/JoshVarga)
-   [Julian Viera](https://github.com/julianviera99)
-   [Kevin Harlley](https://github.com/kevineharlley)
-   [Leonardo Romor](https://github.com/lromor)
-   [Manav Kohli](https://github.com/manav-kohli)
-   [Mark Heffernan](https://github.com/meheff)
-   [Paul Rigge](https://github.com/grebe)
-   [Per Grön](https://github.com/per-gron)
-   [Philipp Schilk](https://github.com/schilkp)
-   [Ravi Nanavati](https://github.com/nanavati)
-   [Rebecca Chen (Pytype)](https://github.com/rchen152)
-   [Remy Goldschmidt](https://github.com/taktoa)
-   [Robert Hundt](https://github.com/rhundt)
-   [Rob Springer](https://github.com/RobSpringer)
-   [Sameer Agarwal](https://github.com/sandwichmaker)
-   [Sean Purser-Haskell](https://github.com/spurserh)
-   [Ted Hong](https://github.com/hongted)
-   [Ted Xie](https://github.com/ted-xie)
-   [Tim Callahan](https://github.com/tcal-x)
-   [Vincent Mirian](https://github.com/vincent-mirian-google)
