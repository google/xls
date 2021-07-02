# XLS Tools

An index of XLS developer tools.

## [`bdd_stats`](https://github.com/google/xls/tree/main/xls/tools/bdd_stats.cc)

Constructs a binary decision diagram (BDD) using a given XLS function and prints
various statistics about the BDD. BDD construction can be very slow in
pathological cases and this utility is useful for identifying the underlying
causes. Accepts arbitrary IR as input or a benchmark specified by name.

## [`benchmark_main`](https://github.com/google/xls/tree/main/xls/tools/benchmark_main.cc)

Prints numerous metrics and other information about an XLS IR file including:
total delay, critical path, codegen information, optimization time, etc. This
tool may be run against arbitrary IR not just the fixed set of XLS benchmarks.
The output of this tool is scraped by `run_benchmarks` to construct a table
comparing metrics against a mint CL across the benchmark suite.

## [`booleanify_main`](https://github.com/google/xls/tree/main/xls/tools/booleanify_main.cc)

Rewrites an XLS IR function in terms of its ops' fundamental AND/OR/NOT
constituents, i.e., makes all operations boolean, thus it's "booleanifying" the
function.

## [`codegen_main`](https://github.com/google/xls/tree/main/xls/tools/codegen_main.cc)

Lowers an XLS IR file into Verilog. Options include emitting a feedforward
pipeline or a purely combinational block. Emits both a Verilog file and a module
signature which includes metadata about the block. The tool does not run any XLS
passes so unoptimized IR may fail if the IR contains constructs not expected by
the backend.

## [`delay_info_main`](https://github.com/google/xls/tree/main/xls/tools/delay_info_main.cc)

Dumps delay information about an XLS function including per-node delay
information and critical-path.

## [`eval_ir_main`](https://github.com/google/xls/tree/main/xls/tools/eval_ir_main.cc)

Evaluates an XLS IR file with user-specified or random inputs. Includes
features for evaluating the IR before and after optimizations which makes this
tool very useful for identifying optimization bugs.

This tool accepts two [mutually exclusive] optional args,
`--input_validator_expr` and `--input_validator_path`, which allow the user to
specify an expression to "filter" potential input values to discard invalid
ones. For both, the filter must be a function, named `validator`, and must take
params of the same layout as the function under test. This function should
return true if the inputs are valid for the function and false otherwise.
`--input_validator_expr` lists the function as an inline command-line argument,
whereas `--input_validator_path` holds the path to a .x file containing the
validation function.

## [`ir_minimizer_main`](https://github.com/google/xls/tree/main/xls/tools/ir_minimizer_main.cc)

Tool for reducing IR to a minimal test case based on an external test.

## [`ir_stats_main`](https://github.com/google/xls/tree/main/xls/tools/ir_stats_main.cc)

Prints summary information/stats on an IR [Package] file. An example:

```
$ bazel-bin/xls/tools/ir_stats_main bazel-genfiles/xls/modules/fpadd_2x32.ir
Package "fpadd_2x32"
  Function: "__float32__is_inf"
    Signature: ((bits[1], bits[8], bits[23])) -> bits[1]
    Nodes: 8

  Function: "__float32__is_nan"
    Signature: ((bits[1], bits[8], bits[23])) -> bits[1]
    Nodes: 8

  Function: "__fpadd_2x32__fpadd_2x32"
    Signature: ((bits[1], bits[8], bits[23]), (bits[1], bits[8], bits[23])) -> (bits[1], bits[8], bits[23])
    Nodes: 252
```

## [`check_ir_equivalence`](https://github.com/google/xls/tree/main/xls/tools/check_ir_equivalence_main.cc)

Verifies that two IR files (for example, optimized and unoptimized IR from the
same source) are logically equivalent.

## [`opt_main`](https://github.com/google/xls/tree/main/xls/tools/opt_main.cc)

Runs XLS IR through the optimization pipeline.

## [`proto_to_dslx_main`](https://github.com/google/xls/tree/main/xls/tools/proto_to_dslx_main.cc)

Takes in a proto schema and a textproto instance thereof and outputs a DSLX
module containing a DSLX type and constant matching both inputs, respectively.

Not all protocol buffer types map to DSLX types, so there are some restrictions
or other behaviors requiring explanation:

1.  Only scalar and repeated fields are supported (i.e., no maps or oneofs,
    etc.).
1.  Only recursively-integral messages are supported, that is to say, a message
    may contain submessages, as long as all non-Message fields are integral.
1.  Since DSLX doesn't support variable arrays and Protocol Buffers don't
    support fixed-length repeated fields. To unify this, all instances of
    repeated-field-containing Messages must have the same size of their repeated
    members (declared as arrays in DSLX). This size will be calculated as the
    maximum size of any instance of that repeated field across all instances in
    the input textproto. For example, if a message `Foo` has a repeated field
    `bar`, and this message is present multiple times in the input textproto,
    say as:

    ```
      foo: {
        bar: 1
      }
      foo: {
        bar: 1
        bar: 2
      }
      foo: {
        bar: 1
        bar: 2
        bar: 3
      }
    ```

    the DSLX version of `Foo` will declare `bar` has a 3-element array. An
    accessory field, `bar_count`, will also be created, which will contain the
    number of valid entries in an actual instance of `Foo::bar`.

    The "Fields" example in
    `./xls/tools/testdata/proto_to_dslx_main.*` demonstrates this
    behavior.

## [`repl`](https://github.com/google/xls/tree/main/xls/tools/repl.cc)

Allows you to interactively run various parts of the compiler, including
parsing/typechecking (`:reload`), lowering/optimization (`:ir`), Verilog
codegen (`:verilog [identifier]`), and LLVM codegen (`:llvm`, not yet
implemented). You can also inspect the IR types of identifiers with `:type`,
and even imported identifiers can be accessed with `:type foo::bar`.

![animated GIF](./repl.gif)

## [`simulate_module_main`](https://github.com/google/xls/tree/main/xls/tools/simulate_module_main.cc)

Runs an Verilog block emitted by XLS through a Verilog simulator. Requires both
the Verilog text and the module signature which includes metadata about the
block.

## [`smtlib_emitter_main`](https://github.com/google/xls/tree/main/xls/tools/smtlib_emitter_main.cc)

Simple driver for Z3IrTranslator - converts a given IR function into its Z3
representation and outputs that translation as SMTLIB2.

## [`solver`](https://github.com/google/xls/tree/main/xls/tools/solver.cc)

Uses a SMT solver (i.e. Z3) to prove properties of an XLS IR program from the
command line. Currently the set of "predicates" that the solver supports from
the command line are limited, but in theory it is capable of solving for
arbitrary IR-function-specified predicates.

This can be used to uncover opportunities for optimization that were missed, or
to prove equivalence of transformed representations with their original version.

## [`cell_library_extract_formula`](https://github.com/google/xls/tree/main/xls/tools/cell_library_extract_formula.cc)

Parses a cell library ".lib" file and extracts boolean formulas from it that
determine the functionality of cells. This is useful for LEC of the XLS IR
against the post-sythesis netlist.

## [`dslx/highlight_main`](https://github.com/google/xls/tree/main/xls/dslx/highlight_main.cc)

Performs terminal-based color code highlighting of a DSL file.
