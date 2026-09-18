# XLS: Symbolic Execution

Symbolic execution explores an XLS IR function with *unknown* inputs,
enumerating every distinct path through the function and solving for a concrete
input that reaches each path. Where an interpreter answers "what does this
function do for input X?", symbolic execution answers "what are all the things
this function can do, and what input gets me to each of them?".

The primary entry point is the
[`symex_main`](https://github.com/google/xls/tree/main/xls/dev_tools/symex_main.cc) tool;
the engine is also usable as a C++ library.

[TOC]

## Usage

`symex_main` consumes XLS IR, so a DSLX design is converted first using
[`three_way_compare.x`](https://github.com/google/xls/tree/main/xls/solvers/symex/testdata/three_way_compare.x):

```dslx
pub fn three_way_compare(a: u32, b: u32) -> u32 {
    if a < b { u32:10 } else if a == b { u32:20 } else { u32:30 }
}
```

```
$ ir_converter_main --top=three_way_compare three_way_compare.x > three_way.ir
$ symex_main three_way.ir
Explored 3 feasible path(s) for function '__three_way_compare__three_way_compare':

Path #0:
  Inputs:
    a = bits[32]:4294967295
    b = bits[32]:0
  Result = bits[32]:30

Path #1:
  Inputs:
    a = bits[32]:33686666
    b = bits[32]:33686666
  Result = bits[32]:20

Path #2:
  Inputs:
    a = bits[32]:2048
    b = bits[32]:4294934527
  Don't cares: sel.8
  Result = bits[32]:10
```

There are three paths, one per arm of the comparison, each with an input that
reaches it. No test inputs were supplied — the solver derived them.

To emit the generated inputs as a machine-readable test vector instead:

```
$ symex_main three_way.ir --output_path= \
    --output_testvector_textproto=vectors.textproto
$ cat vectors.textproto
# proto-file: xls/tests/testvector.proto
# proto-message: SampleInputsProto

function_args {
  args: "bits[32]:4294967295; bits[32]:0"
  args: "bits[32]:33686666; bits[32]:33686666"
  args: "bits[32]:2048; bits[32]:4294934527"
}
```

## When to use symbolic execution

XLS has several ways to exercise a function. They answer different questions:

Tool                           | You supply         | You get
------------------------------ | ------------------ | -------
[`eval_ir_main`](tools.md#ir-eval) | one concrete input | that one result
[IR Fuzzer](ir_fuzzer.md)      | nothing            | many random inputs, unguided coverage
[Z3 translation](solvers.md)   | a property         | one proof about the whole function
**`symex_main`**               | **nothing**        | **every feasible path, plus a witness input for each**

Reach for symbolic execution when you want path-complete test vectors, want to
know how many behaviors a function actually has, or want to confirm that some
branch is reachable at all.

## Command-line reference

```
symex_main <path/to/design.ir> [flags]
```

Passing `-` as the input path reads from stdin.

Flag                            | Default | Meaning
------------------------------- | ------- | -------
`--top`                         | pkg top | Entry function to explore.
`--concrete_inputs`             | `""`    | `param=value` pairs, comma separated.
`--output_path`                 | `-`     | `-` is stdout; empty string suppresses text output.
`--output_testvector_textproto` | `""`    | Write generated inputs as `SampleInputsProto`.
`--max_paths`                   | `1000`  | Stop after this many paths; `0` is unlimited.
`--prune_unobservable`          | `true`  | Skip multiplexers that cannot affect the result.

`--concrete_inputs` accepts both untyped and typed literals, so
`--concrete_inputs=op=0,a=10` and `--concrete_inputs=op=bits[2]:0` are both
valid. Typed values are type-checked against the parameter.

## Reading the output

Each path block reports:

*   **`Inputs`** — a concrete witness, solved for so that execution follows this
    path. These are real, runnable arguments.
*   **`Don't cares`** — multiplexers whose selector cannot change the result on
    this path. See [observability don't cares](#observability-dont-cares).
*   **`Result`** — the function's return value. This is *not* taken from the
    solver model; `symex_main` re-runs the witness through the IR interpreter,
    so it independently confirms the path.

## Concepts

### Paths and branch points

A path is one set of choices at the function's multiplexers. Only `sel` and
`priority_sel` nodes create branches, which in DSLX means `if` and `match`.
Path count is therefore driven by selects, not by lines of code: straight-line
arithmetic of any size is still a single path.

In hardware synthesis, conditional selection can be expressed either through
control flow or through bitwise dataflow. Because symbolic execution explores
branches at the IR multiplexer level (`sel` and `priority_sel`), only explicit
conditionals create multiple paths. Logically equivalent selection implemented
with bitwise masking evaluates both sides eagerly without multiplexers, so the
engine explores it as a single path:

```dslx
// Two paths: `if` introduces an IR multiplexer (`sel`).
if x == u16:42 { u16:0xdead } else { u16:0xbeef }

// One path: produces the same result, but bitwise masking emits no multiplexer.
let eq_mask = ((x == u16:42) as s1) as u16;
let ne_mask = !eq_mask;
(eq_mask & u16:0xdead) | (ne_mask & u16:0xbeef)
```

### Concolic inputs

Pinning a parameter to a constant prunes everything that constant rules out.
[`execute_alu.x`](https://github.com/google/xls/tree/main/xls/solvers/symex/testdata/execute_alu.x)
is an ALU with an opcode and two operands; unconstrained it has four paths, but
fixing the opcode to `ADD` leaves only the two ADD behaviors:

```
$ symex_main alu.ir --concrete_inputs=op=0
Explored 2 feasible path(s) for function '__execute_alu__execute_alu':

Path #0:
  Inputs:
    op = bits[2]:0
    a = bits[8]:0
    b = bits[8]:0
  Don't cares: sel.26
  Result = (bits[2]:0, bits[8]:0)

Path #1:
  Inputs:
    op = bits[2]:0
    a = bits[8]:173
    b = bits[8]:240
  Don't cares: sel.26
  Result = (bits[2]:1, bits[8]:0)
```

The second witness overflows 8 bits, so the ALU reports `OVERFLOW`. Note that
neither input was hand-written. The `Don't cares` line is explained under
[observability don't cares](#observability-dont-cares).

### Observability don't cares {#observability-dont-cares}

Hardware computes eagerly: an ALU evaluates its adder even on a cycle when the
AND result is selected. A naive explorer would enumerate every adder path once
per opcode, multiplying path counts by work that is thrown away.

When `--prune_unobservable` is enabled (the default), the engine notices that a
multiplexer's output is discarded on the current path, leaves its selector
unconstrained, and reports it as a don't care. That one path then stands for
every combination of that multiplexer's arms. For the ALU:

```
$ symex_main alu.ir                            # pruning on (default)
Explored 4 feasible path(s) ...
$ symex_main alu.ir --prune_unobservable=false # pruning off
Explored 6 feasible path(s) ...
```

Symbolic execution returns four paths instead of six, and the four still cover
the whole input domain. Pass `--prune_unobservable=false` if you specifically
need every arm enumerated.

Path count is not the only difference between the two modes. Pruning has to
decide consumers before producers to know when a multiplexer stops mattering,
which means a consumer's arm is fixed while the producer variables it refers to
are still unconstrained, and the solver carries that extra state. Exhaustive
exploration skips nothing, so it has no reason to pay that cost and decides
producers first instead.

### Path explosion {#path-explosion}

Path count is worst-case exponential in the number of selects.

!!! WARNING
    `--max_paths` defaults to `1000` and truncation is silent. If the tool
    reports exactly `max_paths` paths, the result is **not** exhaustive and no
    conclusion about full coverage is valid.

## C++ API

```c++
#include "xls/solvers/symex/symex_engine.h"

SymExOptions options;
options.concrete_inputs.BindParam("op", Value(UBits(0, 2)));
options.max_paths = 100;

XLS_ASSIGN_OR_RETURN(SymExEngine engine, SymExEngine::Create(ctx, options));
XLS_ASSIGN_OR_RETURN(std::vector<SymbolicPath> paths, engine.ExplorePaths(fn));

for (const SymbolicPath& path : paths) {
  XLS_ASSIGN_OR_RETURN(Value result,
                       DropInterpreterEvents(
                           InterpretFunction(fn, path.input_values())));
}
```

The caller owns the `Z3_context`. Each
[`SymbolicPath`](https://github.com/google/xls/tree/main/xls/solvers/symex/symbolic_path.h)
carries its `path_condition` and symbolic `return_value`, the
`branch_decisions` taken, any `unobservable_muxes`, and the witness as
`generated_test` / `input_values()` / `GetParamValue(name)`.

## Coverage guarantees

The reported paths are a verified decomposition of the function: the path
conditions are mutually exclusive, they cover 100% of the input domain, and
folding them back into a nested if-then-else yields a model logically equivalent
to translating the function as a whole. So a run that did not hit `--max_paths`
has enumerated every behavior the function has, and its generated inputs
exercise all of them.

These properties are checked by the engine's own test suite; see
[`test_util.h`](https://github.com/google/xls/tree/main/xls/solvers/symex/test_util.h) if
you are extending the engine.

## Limitations

*   `Function` only — procs and blocks are not supported.
*   Only `sel` and `priority_sel` are treated as branch points.
*   Solver-bound: wide selectors and long select chains get expensive.
*   `--max_paths` truncates silently; see [path explosion](#path-explosion).

## How it works

1.  **Encode** the function into Z3 up front, representing each multiplexer as
    an unconstrained SSA variable.
2.  **Traverse** multiplexers depth-first, push/pop-ing a solver frame per
    branch and backtracking as soon as a partial path is UNSAT.
3.  **Prune** multiplexers that can no longer reach the result. This is what
    dictates the traversal order: a multiplexer is a don't care only once every
    consumer of it has committed to an arm, so pruning must decide consumers
    before producers.
4.  **Extract** a witness from the solver model once every multiplexer on a
    feasible path is resolved.

See
[`symex_engine.h`](https://github.com/google/xls/tree/main/xls/solvers/symex/symex_engine.h)
and
[`mux_observability.h`](https://github.com/google/xls/tree/main/xls/solvers/symex/mux_observability.h)
for details.

## Related

*   [Formal solvers](solvers.md) — Z3 translation and equivalence checking.
*   [Interpreters](interpreters.md) — evaluating IR on concrete inputs.
*   [IR Fuzzer](ir_fuzzer.md) — randomized IR generation.
*   [Tools](tools.md) — index of XLS developer tools.
