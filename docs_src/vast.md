# Verilog Abstract Syntax Tree (VAST)

XLS outputs Verilog (or SystemVerilog) for synthesis and simulation. As a lowest
common denominator, Verilog output enables XLS generated designs to integrate
into existing design flows. To make generation of Verilog easier, XLS includes
an abstract representation of Verilog called VAST (Verilog Abstract Syntax
Tree). VAST is a C++ library which represents Verilog in a recursive tree data
structure which is simple to construct and manipulate programmatically. Verilog
source code is emitted directly from the VAST data structure.

VAST is intentionally not a complete representation of the Verilog language.
VAST is used to emit Verilog for the purposes of code generation within XLS.
Given this limited use case, VAST is much smaller and simpler than a complete
representation of the entire Verilog language as might be required for a parser,
for example.

## VAST Overview

Each supported Verilog construct is represented with a C++ class. These classes
form a type hierarchy with the class `VastNode` at the root. Objects are
gathered in tree-shaped structures to represent Verilog constructs. Ownership of
all VAST objects is maintained by a `VerilogFile` object which represents a
single file of Verilog source code. References between objects are stored as
plain pointers.

For example, consider the following Verilog expression:

```
  foo + 8
```

In VAST, this is represented with an object of the `BinaryInfix` class which is
derived from the `Expression` class representing arbitrary Verilog expressions.
A `BinaryInfix` object has three relevant data members:

`std::string op_;`
:   The string representation of the operation to perform (e.g., `+`).

`Expression* lhs_;`
:   The left-hand-side of the expression. In this example, this points to a
    `LogicRef` object (derived from `Expression` class) referring to a Verilog
    `reg` or `wire` variable.

`Expression* rhs_;`
:   The left-hand-side of the expression. In this example, this points to a
    `Literal` object (derived from `Expression` class) containing the number 8
    with unspecified bit width.

The `BinaryInfix` object representing `foo + 8` might be used within other
expressions or statements by referring to the object by pointer. For example,
the representation of the statement `assign bar = foo + 8` would contain an
`Expression*` pointer referring to the `foo + 8` object for the right-hand-side
of the assignment.

### Operator Precedence

To avoid ambiguity, operators in Verilog follow precedence rules. For example,
multiplication is higher precedence than addition so the expression `2 + 4 * 10`
evaluates to `42` (i.e., `2 + (4 * 10)`) not `60` (i.e., `(2 + 4) * 10`). In
VAST, expressions are built as a trees which is evaluated from the leaves to the
root. To ensure that the operations are evaluated in the correct order when
emitted as Verilog text, VAST automatically adds parentheses where appropriate.
For example, the VAST expression consisting of the product (`BinaryInfix` with
operation `*`) of `10` and the sum of `2` and `4` (`BinaryInfix` with operation
`+`) will be emitted as `10 * (2 + 4)`.

### Containers

VAST has a number of classes which hold a sequence of (pointers to) other VAST
objects. At the top-level, this includes the `VerilogFile` class which can hold
a sequence of objects such as include statements and modules. Verilog modules
themselves are represented with the `Module` class containing a sequence of
statements, declarations, comments, and other constructs. Other containers
include always blocks and functions.

### Emitting Verilog text

VAST classes include an `Emit` method which returns the represented Verilog
construct as a string. Typically, `Emit` is called on the top-level
`VerilogFile` object to create the text of the entire Verilog source file.
Underneath the hood, this method calls the `Emit` method on all contained VAST
objects and assembles the returned strings into the Verilog source code.

### SystemVerilog support

XLS can emit either Verilog or SystemVerilog so VAST supports both languages.
SystemVerilog constructs are included alongside Verilog constructs in VAST.
Examples of SystemVerilog features supported by VAST include:

*   `always_ff` procedure for modeling sequential logic (VAST `AlwaysFlop`
    class).

*   Array assignment pattern (VAST `ArrayAssignmentPattern` class). Example:
    `'{foo, bar, baz}`

*   Array declaration using sizes. Example: `reg [7:0] foo[42];`

Within VAST, there is no distinction between the two languages and it is up to
the user of VAST to only use the supported features for the target language
(Verilog or SystemVerilog).
