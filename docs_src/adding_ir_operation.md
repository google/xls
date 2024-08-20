# Adding a new IR operation

XLS has about 60 different [opcodes](https://github.com/google/xls/tree/main/xls/ir/op.h)
and periodically new ones are added to extend functionality or improve the
expressiveness of the IR. XLS has many different components and adding a new
opcode involves changes to numerous places in the code. These changes, some of
which are optional, are described below:

1.  Add operation to [op.h](https://github.com/google/xls/tree/main/xls/ir/op.h).

    Opcodes are defined in the file `op.h` and IR node classes are defined in
    the file `nodes.h`.Every opcode has an associated node subclass derived from
    the `xls::Node` base class. Some opcodes such as `Op::kArray` have their own
    class (`Array`) because of the unique structure of the operation. Other
    opcodes such as the logical operations (`Op::kAnd`, `Op::kOr`, etc) share a
    common base class (`BinOp`).

    The first step to adding a new operations is to add an opcode, and
    potentially a new Node class. After adding the opcode numerous files will
    fail to build because switch statements over the set of opcodes will no
    longer be exhaustive. Add the necessary cases to each switch statement. The
    exact code in each case will, of course, be operation-specific. Initially
    the implementation might return an `absl::UnimplementedError` status until
    later changes add proper support for the new operation.

    As part of this change the new operations needs to be added to the DFS
    visitor class `DfsVisitor` by adding a handler method. This class is used
    throughout XLS to traverse the IR. This will also adding an implementation
    of this new method to many of the subclasses derived from `DfsVisitor`.

    [(Code example)](https://github.com/google/xls/commit/5fd739abe3e28f4198e07d45987522b12ebdf051)

1.  [IR Verifier](https://github.com/google/xls/tree/main/xls/ir/verifier.h)

    The IR verifier checks numerous invariants about the IR including
    operation-specific properties such as the number and type of operands. Add
    an additional handler method for the new operand and add appropriate
    operation-specific checks.

    [(Code example)](https://github.com/google/xls/commit/5fd739abe3e28f4198e07d45987522b12ebdf051)

1.  [IR Semantics document](https://google.github.io/xls/ir_semantics/)

    Describe the semantics and syntax of the new operation in the IR semantics
    document.

    [(Code example)](https://github.com/google/xls/commit/5fd739abe3e28f4198e07d45987522b12ebdf051)

1.  [Function builder](https://github.com/google/xls/tree/main/xls/ir/function_builder.h)

    The function builder is the primary API for constructing IR. If appropriate,
    add a method to the `BuilderBase` class which adds an IR node of the new
    type to a function.

    [(Code example)](https://github.com/google/xls/commit/eb12ef77d51e2d65f4295e80ffa944043f021b2f)

1.  [IR Parser](https://github.com/google/xls/tree/main/xls/ir/ir_parser.h)

    Add support for parsing of the new operation. The parser tests typically
    send a snippet of IR with the operation through the parser and text
    serialization and verifies that the output matches the original. Supporting
    the new operation may require modifying the `xls::Node::ToString` method to
    emit any special fields required by the operation.

    [(Code example)](https://github.com/google/xls/commit/eb12ef77d51e2d65f4295e80ffa944043f021b2f)

1.  [IR Interpreter](https://github.com/google/xls/tree/main/xls/interpreter/ir_interpreter.h)

    The IR interpreter has C++ implementations of all of the operations.
    Implement the new operation and add tests.

    [(Code example)](https://github.com/google/xls/commit/eb12ef77d51e2d65f4295e80ffa944043f021b2f)

1.  [IR Matcher](https://github.com/google/xls/tree/main/xls/ir/ir_matcher.h)

    The IR matcher is used in tests to enable easy matching of IR expressions.
    For example, the following tests that the return value of a function is the
    parameter `x` plus the parameter `y`:

    ```
    EXPECT_THAT(f->return_value(), m::Add(m::Param("x", m::Param("y")));
    ```

    If the new operation has no named attributed, IR matcher support is
    typically a single line using the macro `NODE_MATCHER`. Otherwise, a custom
    matcher should be added to enable matching the attribute as well.

    [(Code example)](https://github.com/google/xls/commit/eb12ef77d51e2d65f4295e80ffa944043f021b2f)

1.  [LLVM JIT](https://github.com/google/xls/tree/main/xls/jit/ir_builder_visitor.h)

    The LLVM JIT enables fast simulation of the XLS IR. The JIT constructs LLVM
    IR for each XLS operation which is then optimized by LLVM and runs natively
    on the host. Implement the new operation in the `FunctionBuilderVisitor`
    class.

    [(Code example)](https://github.com/google/xls/commit/eb12ef77d51e2d65f4295e80ffa944043f021b2f)

1.  [Code generation](https://github.com/google/xls/tree/main/xls/codegen/node_expressions.h)

    In XLS "code generation" refers to the generation of (System)Verilog from
    XLS IR. If the operation can be emitted as a single Verilog expression, then
    likely support for the new operation can be added to `node_expressions.h`,
    otherwise if the implementation requires multiple statements then support is
    added to `module_builder.h`.

    [(Code example)](https://github.com/google/xls/commit/ef08b552ac3738eb98484cc46a7396c89f7cbb7d)

1.  [Abstract evaluator](https://github.com/google/xls/tree/main/xls/ir/abstract_evaluator.h)

    The abstract evaluator enables evaluation of the XLS IR using different
    evaluation systems than Boolean algebra. Users define the semantics of
    simple logical operations such as and, or, and not. Then, the abstract
    evaluator interprets an IR function using these rules. One example use case
    is ternary logic which uses three logic values (true, false, and unknown)
    rather than two (true and false) Ternary evaluation is used by the optimizer
    to discover statically known bits in the IR graph. The abstract evaluator
    can also be used for translation of the IR to other representations. For
    example, IR is translated to the Z3 solver representation for formal
    verification using the abstract evaluator.

    If appropriate, the operation should be implemented in
    `AbstractNodeEvaluator` by providing an implementation which decomposes the
    operation into fundamental logical operations.

    [(Code example)](https://github.com/google/xls/commit/bda129fed73de323574d7955292753187af7bb20)

1.  [Z3 solver](https://github.com/google/xls/tree/main/xls/solvers/z3_ir_translator.h)

    The Z3 solver is used for theorem proving and logical equivalence checking
    between the IR in different stages of compilation and the netlist. To enable
    this functionality for the new operation, add a lowering of the operation to
    Z3's internal representation.

    [(Code example)](https://github.com/google/xls/commit/bda129fed73de323574d7955292753187af7bb20)

1.  [Delay model](https://github.com/google/xls/tree/main/xls/delay_model)

    In order to generate efficient circuits which meet timing requirement, XLS
    models the delay (in picoseconds) of each operation for different process
    technology nodes. This model is constructed by characterizing the process
    node using an EDA tool to synthesize the circuit and estimate delay.
    Typically, a new operation will need to be characterized by running numerous
    permutations of the operation (e.g., with different bit widths) through a
    synthesis flow, extracting delay, and building a delay model.

    [(Code example)](https://github.com/google/xls/commit/19bb886a1471b074159fcbf95fccce17fba40031)

1.  [DSLX frontend](https://github.com/google/xls/tree/main/xls/dslx)

    Most ops are used by the DSLX frontend in the lowering of DSLX to IR. The
    operation may be exposed directly as a builtin (or other operation) or used
    in the lowering of other AST nodes. In any case, some changes to the DSLX
    frontend will likely be necessary.

    [(Code example)](https://github.com/google/xls/commit/feeac2c4c0bcc73b529cb0c4a976abae47f96730)

1.  [Fuzzer](https://github.com/google/xls/tree/main/xls/fuzzer)

    The fuzzer generates random DSLX functions and random inputs to check and
    compare different parts of XLS, for example checking that un-optimized and
    optimized IR give the same outputs when interpreted. If there is an
    operation in DSLX that maps nicely onto the newly added operation, the
    fuzzer can be modified to generate functions with DSLX that exercise the new
    operation. This is done by adding a handler to `AstGenerator`. See
    [here](./fuzzer.md) for
    more details on how the fuzzer works and how to run it.

    [(Code example)](https://github.com/google/xls/commit/c09339f3b0c147031eadd626d1a856a860031e05)

1.  Operation-specific optimizations

    Typically, a new operation provides optimization opportunities unique to the
    node. The details, of course, will be vary for different operations.
    However, typically these are at least several easy optimizations which can
    be implemented.
