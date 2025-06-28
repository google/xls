# Bytecode interpreter

DSLX provides a [bytecode interpreter](https://en.wikipedia.org/wiki/Bytecode)
for expression evaluation. This style of interpreter can be started, stopped,
and resumed more easily than an AST-walking native interpreter, as its full
state can be captured as `{PC, stack}` instead of some traversal state in native
execution, which makes it very suitable for modeling independent processes, such
as `Proc`s.

!!! NOTE
    The bytecode interpreter system is under active construction and does not
    yet support the full set of DSLX functionality.

[TOC]

## Structure

[The interpreter](https://github.com/google/xls/tree/main/xls/dslx/bytecode/bytecode_interpreter.h)
is implemented as a
[stack virtual machine](https://en.wikipedia.org/wiki/Stack_machine): it
consists of a program counter (PC), a stack of frames, and "slot"-based locals
within a given stack frame (*conceptually* part of the stack frame, but tracked
separately in our implementation). Both the stack and local storage hold
[`InterpValues`](https://github.com/google/xls/tree/main/xls/dslx/interp_value.h), which
can hold all DSLX data types: bits, tuples, and arrays (and others), thus there
is no fundamental need for lower-level (i.e., byte) type representation. For the
purposes of [de]serialization, this may change in the future. Local data is
addressed by integer-typed "slots", being backed by a simple `std::vector`: in
other words, slot indices are dense. All slots must be pre-allocated to contain
all references to locals in the current function stack frame.

On each "tick", the interpreter reads the current instruction, as given by the
PC (conceptually, the only register in the virtual machine), executes the
described operation (usually consuming values from the stack), and places the
result on the stack.

## ISA

Each instruction consists of an opcode plus, optionally, some piece of data,
either `int64`- or `InterpValue`-typed, depending on the specific opcode.

The below opcodes are supported by the interpreter:

-   `ADD`: Adds the two values at the top of the stack.
-   `CALL`: Invokes the function given as the optional data argument, consuming
    a number of arguments from the stack as described by the function signature.
    The N'th parameter will be present as the N'th value down the stack (such
    that the last parameter will be the value initially on top of the stack.
-   `CREATE_TUPLE`: Groups together N items on the stack (given by the optional
    data argument into a single `InterpValue`.
-   `EXPAND_TUPLE`: Expands the N-tuple at stack top by one level, placing
    leading elements at stack top. In other words, expanding the tuple `(a, (b,
    c))` will result in a stack of `(b, c), a`, where `a` is on top of the
    stack.
-   `EQ`: Compares the two values on top of the stack for equality. Emits a
    single-bit value.
-   `LOAD`: Loads the value from locals slot `n`, where `n` is given by the
    optional data argument.
-   `LITERAL`: Places a literal value (given in the optional data argument) on
    top of the stack.
-   `STORE`: Stores the value at stack top into slot `n` in locals storage.

## Bytecode generation

The
[bytecode emitter](https://github.com/google/xls/tree/main/xls/dslx/bytecode/bytecode_emitter.h)
is responsible for converting a set of DSLX ASTs (one per function)) into a set
of linear bytecode representations. It does this via a postorder traversal of
the AST, converting XLS ops into bytecode instructions along the way, e.g.,
converting a DSLX `Binop` for adding two `NameRef`s into two `LOAD` instructions
(one for each `NameRef`) and one `ADD` instruction.

To do this, the emitter needs access to the full set of resolved type and import
information: in other words, it requires a fully-populated
[`ImportData`](https://github.com/google/xls/tree/main/xls/dslx/import.h) and the
top-level
[`TypeInfo`](https://github.com/google/xls/tree/main/xls/dslx/type_system/type_info.h) for
the module containing the function to convert. This places bytecode emission in
sequence after typechecking and deduction.

## Implementation details

### `map` builtin

The `map()` function built-in to DSLX accepts an array-typed value `x` and a
mapping function `f` with the signature `T -> U`; that is, it accepts a single
value of type `T` and returns a single value with the type `U`. In operation,
`map()` applies the mapping function `f` to every element in `x` and returns a
new array containing the results (with element `i` in the output corresponding
to element `i` in the input).

Conceptually, `map()` destructures to a `for` loop over the elements in `x`, and
that's essentially what the interpreter does with these opcodes. To avoid
modifying the currently executing bytecode, the interpreter instead creates a
new BytecodeFunction consisting of just that destructured `for` loop over the
inputs, followed by a CreateArray op to collect the output(s). Finally, the
interpreter begins execution of the new function by creating a new `Frame` on
top of the execution stack.
