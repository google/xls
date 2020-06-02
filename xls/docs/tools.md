# XLS Tools

An index of XLS developer tools.

## [`bdd_stats`](https://github.com/google/xls/tree/master/xls/tools/bdd_stats.cc)

Constructs a binary decision diagram (BDD) using a given XLS function and prints
various statistics about the BDD. BDD construction can be very slow in
pathological cases and this utility is useful for identifying the underlying
causes. Accepts arbitrary IR as input or a benchmark specified by name.



## [`benchmark_main`](https://github.com/google/xls/tree/master/xls/tools/benchmark_main.cc)

Prints numerous metrics and other information about an XLS IR file including:
total delay, critical path, codegen information, optimization time, etc. This
tool may be run against arbitrary IR not just the fixed set of XLS benchmarks.
The output of this tool is scraped by `run_benchmarks` to construct a table
comparing metrics against a mint CL across the benchmark suite.

## [`codegen_main`](https://github.com/google/xls/tree/master/xls/tools/codegen_main.cc)

Lowers an XLS IR file into Verilog. Options include emitting a feedforward
pipeline or a purely combinational block. Emits both a Verilog file and a module
signature which includes metadata about the block. The tool does not run any XLS
passes so unoptimized IR may fail if the IR contains constructs not expected by
the backend.



## [`eval_ir_main`](https://github.com/google/xls/tree/master/xls/tools/eval_ir_main.cc)

Evaluates an XLS IR file with user-specified or random inputs. Includes
features for evaluating the IR before and after optimizations which makes this
tool very useful for identifying optimization bugs.



## [`ir_minimizer_main`](https://github.com/google/xls/tree/master/xls/tools/ir_minimizer_main.cc)

Tool for reducing IR to a minimal test case based on an external test.



\##
[`check_ir_equivalence`](https://github.com/google/xls/tree/master/xls/tools/check_ir_equivalence_main.cc)

Verifies that two IR files (for example, optimized and unoptimized IR from the
same source) are logically equivalent.



## [`opt_main`](https://github.com/google/xls/tree/master/xls/tools/opt_main.cc)

Runs XLS IR through the optimization pipeline.

## [`simulate_module_main`](https://github.com/google/xls/tree/master/xls/tools/simulate_module_main.cc)

Runs an Verilog block emitted by XLS through a Verilog simulator. Requires both
the Verilog text and the module signature which includes metadata about the
block.



## [`solver`](https://github.com/google/xls/tree/master/xls/tools/solver.cc)

Uses a SMT solver (i.e. Z3) to prove properties of an XLS IR program from the
command line. Currently the set of "predicates" that the solver supports from
the command line are limited, but in theory it is capable of solving for
arbitrary IR-function-specified predicates.

This can be used to uncover opportunities for optimization that were missed, or
to prove equivalence of transformed representations with their original version.



## [`cell_library_extract_formula`](https://github.com/google/xls/tree/master/xls/tools/cell_library_extract_formula.cc)

Parses a cell library ".lib" file and extracts boolean formulas from it that
determine the functionality of cells. This is useful for LEC of the XLS IR
against the post-sythesis netlist.


