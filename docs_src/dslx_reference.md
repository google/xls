# DSLX Reference

### Overview

DSLX is a domain specific, dataflow-oriented functional language used to build
hardware that can also run effectively as host software. Within the XLS project,
DSLX is also referred to as "the DSL". The DSL targets the XLS compiler (via
conversion to XLS IR) to enable flows for FPGAs and ASICs.

DSLX mimics Rust, while being an immutable expression-based dataflow DSL with
hardware-oriented features; e.g. arbitrary bitwidths, entirely fixed size
objects, fully analyzeable call graph, etc. To avoid arbitrary new
syntax/semantics choices, the DSL mimics Rust where it is reasonably possible;
for example, integer conversions all follow the same semantics as Rust.

Note: There are *some* unnecessary differences today from Rust syntax due to
early experimentation, but they are quickly being removed to converge on Rust
syntax.

Note that other frontends to XLS core functionality will become available in the
future; e.g. [xlscc](https://github.com/google/xls/tree/main/xls/contrib/xlscc), for users
familiar with the C++-and-pragma style of HLS computation. XLS team develops the
DSL as part of the XLS project because we believe it can offer significant
advantages over the C++-with-pragmas approach.

Dataflow DSLs are a good fit for describing hardware, compared to languages
whose design assumes
[von Neumann style computation](https://en.wikipedia.org/wiki/Von_Neumann_architecture)
(global mutable state, sequential mutation by a sequential thread of control).
Using a Domain Specific Language (DSL) provides a more hardware-oriented
representation of a given computation that matches XLS compiler (IR) constructs
closely. The DSL also allows an exploration of HLS without being encumbered by
C++ language or compiler limitations such as non-portable pragmas, magic macros,
or semantically important syntactic conventions. The language is still
experimental and likely to change, but it is already useful for experimentation
and exploration.

This document provides a reference for DSLX, mostly by example. After perusing
it and learning about the language features, we recommend exploring the
following, detailed examples to learn how the language features are put to
action:

1.  [CRC32](./dslx_intro_example1.md)

2.  [Floating-point addition](./fpadd_example.md)

3.  [Prefix Sum Computation](./dslx_intro_example3.md)

In this document we use the function to compute a CRC32 checksum to describe
language features. The full code is in
[`examples/dslx_intro/crc32_one_byte.x`](https://github.com/google/xls/tree/main/xls/examples/dslx_intro/crc32_one_byte.x).

### Comments

Just as in languages like Rust/C++, comments start with `//` and last through
the end of the line.

### Identifiers

All identifiers, eg., for function names, parameters, and values, follow the
typical naming rules of other languages. The identifiers can start with a
character or an underscore, and can then contain more characters, underscores,
or numbers. Valid examples are:

```
a                 // valid
CamelCase         // valid
like_under_scores // valid
__also_ok         // valid
_Ok123_321        // valid
_                 // valid

2ab               // not valid
&ade              // not valid
```

However, we suggest the following **DSLX style rules**, which mirror the
[Rust naming conventions](https://doc.rust-lang.org/1.0.0/style/style/naming/README.html).

*   Functions are `written_like_this`

*   User-defined data types are `NamesLikeThis`

*   Constant bindings are `NAMES_LIKE_THIS`

*   `_` is the "black hole" identifier -- a name that you can bind to but should
    never read from, akin to Rust's wildcard pattern match or Python's "unused
    identifier" convention. It should never be referred to in an expression
    except as a "sink".

NOTE Since mutable locals are not supported, there is also
[support for "tick identifiers"](https://github.com/google/xls/issues/212),
where a ' character may appear anywhere after the first character of an
identifier to indicate "prime"; e.g. `let state' = update(state);`. By
convention ticks usually come at the end of an identifier. Since this is not
part of Rust's syntax, it is considered experimental at this time.

## Functions

Function definitions begin with the keyword `fn`, followed by the function name,
a parameter list to the function in parenthesis, followed by an `->` and the
return type of the function. After this, curly braces denote the begin and end
of the function body.

The list of parameters can be empty.

A single input file can contain many functions.

Simple examples:

```dslx
fn ret3() -> u32 {
   u32:3   // This function always returns 3.
}

fn add1(x: u32) -> u32 {
   x + u32:1  // Returns x + 1, but you knew that!
}
```

Functions return the result of their last computed expression as their return
value. There are no explicit return statements. By implication, functions return
exactly one expression; they can't return multiple expressions (but this may
change in the future as we migrate towards some Rust semantics).

Tuples should be returned if a function needs to return multiple values.

### Parameters

Parameters are written as pairs `name` followed by a colon `:` followed by the
`type` of that parameter. Each parameter needs to declare its own type.

Examples:

```
// a simple parameter x of type u32
   x: u32

// t is a tuple with 2 elements.
//   the 1st element is of type u32
//   the 2nd element is a tuple with 3 elements
//       the 1st element is of type u8
//       the 2nd element is another tuple with 1 element of type u16
//       the 3rd element is of type u8
   t: (u32, (u8, (u16,), u8))
```

### Parametric Functions

DSLX functions can be parameterized in terms of the types of its arguments and
in terms of types derived from other parametric values. For instance:

```dslx
fn double(n: u32) -> u32 {
  n * u32:2
}

fn self_append<A: u32, B: u32 = double(A)>(x: bits[A]) -> bits[B] {
  x++x
}

fn main() -> bits[10] {
  self_append(u5:1)
}
```

In `self_append(bits[5]:1)`, we see that `A = 5` based off of formal argument
instantiation. Using that value, we can evaluate `B = double(A=5)`. This derived
expression is analogous to C++'s constexpr – a simple expression that can be
evaluated at that point in compilation.

See
[advanced understanding](#advanced-understanding-parametricity-constraints-and-unification)
for more information on parametricity.

#### Explicit parametric instantiation

In some cases, parametric values cannot be inferred from function arguments,
such as in the
[`explicit_parametric_simple.x`](https://github.com/google/xls/tree/main/xls/dslx/tests/explicit_parametric_simple.x)
test:

```
fn add_one<E:u32, F:u32, G:u32 = E+F>(lhs: bits[E]) -> bits[G] { ... }
```

For this call to instantiable, both `E` and `F` must be specified. Since `F`
can't be inferred from an argument, we must rely on _explicit parametrics_:

```dslx-snippet
  add_one<u32:1, {u32:2 + u32:3}>(u1:1);
```

This invocation will bind `1` to `E`, `5` to `F`, and `6` to `G`. Note the curly
braces around the expression-defined parametric: simple literals and constant
references do not need braces (but they _can_ have them), but any other
expression requires them.

##### Expression ambiguity

Without curly braces, explicit parametric expressions could be ambiguous;
consider the following, slightly changed from the previous example:

```dslx-snippet
  add_one<u32:1, u32:2>(u32:3)>(u1:1);
```

Is the statement above computing `add_one<1, (2 > 3)>(1)`, or is it computing
`(zero<1, 2>(3)) > 1)`? Without additional (and subtle and perhaps surprising)
contextual precedence rules, this would be ambiguous and could lead to a parse
error or, even worse, unexpected behavior.

Fortunately, we can look to Rust for inspiration. Rust's const generics RPF
introduced the `{ }` syntax for disambiguating just this case in generic
specifications. With this, any expressions present in a parametric specification
must be contained within curly braces, as in the original example.

At present, if the braces are omitted, some unpredictable error will occur. Work
to improve this is tracked in
[XLS GitHub issue #321](https://github.com/google/xls/issues/321).

### Function Calls

Function calls are expressions and look and feel just like one would expect from
other languages. For example:

```dslx
fn callee(x: bits[32], y: bits[32]) -> bits[32] {
  x + y
}
fn caller() -> u32 {
  callee(u32:2, u32:3)
}
```

If more than one value should be returned by a function, a tuple type should be
returned.

## Types

### Bit Type

The most fundamental type in DSLX is a variable length bit type denoted as
`bits[n]`, where `n` is a constant. For example:

```
bits[0]   // possible, but, don't do that

bits[1]   // a single bit
uN[1]     // explicitly noting single bit is unsigned
u1        // convenient shorthand for bits[1]

bits[8]   // an 8-bit datatype, yes, a byte
u8        // convenient shorthand for bits[8]
bits[32]  // a 32-bit datatype
u32       // convenient shorthand for bits[32]
bits[256] // a 256-bit datatype
```

DSLX introduces aliases for commonly used types, such as `u8` for an 8-wide bit
type, or `u32` for a 32-bit wide bit type. These are defined up to `u64`.

All `u*`, `uN[*]`, and `bits[*]` types are interpreted as unsigned integers.
Signed integers are specified via `s*` and `sN[*]`. Similarly to unsigned
numbers, the `s*` shorthands are defined up to `s64`. For example:

```
sN[0]
s0

sN[1]
s1

sN[64]
s64

sN[256]
```

Signed numbers differ in their behavior from unsigned numbers primarily via
operations like comparisons, (variable width) multiplications, and divisions.

### Enum Types

DSLX supports enumerations as a way of defining a group of related, scoped,
named constants that do not pollute the module namespace. For example:

```dslx
enum Opcode : u3 {
  FIRE_THE_MISSILES = 0,
  BE_TIRED = 1,
  TAKE_A_NAP = 2,
}

fn get_my_favorite_opcode() -> Opcode {
  Opcode::FIRE_THE_MISSILES
}
```

Note the use of the double-colon to reference the enum value. This code
specifies that the enum behaves like a `u3`: its storage and extension (via
casting) behavior are defined to be those of a `u3`. Attempts to define an enum
value outside of the representable `u3` range will produce a compile time error.

```dslx-bad
enum Opcode : u3 {
  FOO = 8  // Causes compile time error!
}
```

Enums can be compared for equality/inequality, but they do not permit arithmetic
operations, they must be cast to numerical types in order to perform arithmetic:

```dslx
enum Opcode: u3 {
  NOP = 0,
  ADD = 1,
  SUB = 2,
  MUL = 3,
}

fn same_opcode(x: Opcode, y: Opcode) -> bool {
  x == y  // ok
}

fn next_in_sequence(x: Opcode, y: Opcode) -> bool {
  // x+1 == y // does not work, arithmetic!
  x as u3 + u3:1 == (y as u3)  // ok, casted first
}
```

As mentioned above, casting of enum-values works with the same casting/extension
rules that apply to the underlying enum type definition. For example, this cast
will sign extend because the source type for the enum is signed. (See
[numerical conversions](#numerical-conversions) for the full description of
extension/truncation behavior.)

```dslx
enum MySignedEnum : s3 {
  LOW = -1,
  ZERO = 0,
  HIGH = 1,
}

fn extend_to_32b(x: MySignedEnum) -> u32 {
  x as u32  // Sign-extends because the source type is signed.
}

#![test]
fn test_extend_to_32b() {
  assert_eq(extend_to_32b(MySignedEnum::LOW), u32:-1)
}
```

Casting *to* an enum is also permitted. However, in most cases errors from
invalid casting can only be found at runtime, e.g., in the DSL interpreter or
flagging a fatal error from hardware. Because of that, it is recommended to
avoid such casts as much as possible.

### Tuple Type

A tuple is a fixed-size ordered set, containing elements of heterogeneous types.
Tuples elements can be any type, e.g. bits, arrays, structs, tuples. Tuples may
be empty (an empty tuple is also known as the unit type), or contain one or more
types.

Examples of tuple values:

```dslx-snippet
// The unit type, carries no information.
let unit = ();

// A tuple containing two bits-typed elements.
let pair = (u3:0b100, u4:0b1101);
```

Example of a tuple type:

```dslx
// The type of a tuple with 2 elements.
//   the 1st element is of type u32
//   the 2nd element is a tuple with 3 elements
//       the 1st element is of type u8
//       the 2nd element is another tuple with 1 element of type u16
//       the 3rd element is of type u8
type MyTuple = (u32, (u8, (u16,), u8));
```

To access individual tuple elements use simple indices, starting at 0. For
example, to access the second element of a tuple (index 1):

```dslx
#![test]
fn test_tuple_access() {
  let t = (u32:2, u8:3);
  assert_eq(u8:3, t[1])
}
```

Tuples can be "destructured", similarly to how pattern matching works in
`match` expressions, which provides a convenient syntax to name elements of a
tuple for subsequent use. See `a` and `b` in the following:

```dslx-snippet
#![test]
fn test_tuple_destructure() {
  let t = (u32:2, u8:3);
  let (a, b) = t;
  let _ = assert_eq(u32:2, a);
  assert_eq(u8:3, b)
}
```

Just as values can be discarded in a `let` by using the "black hole identifier"
`_`, don't-care values can also be discarded when destructuring a tuple:

```dslx-snippet
#![test]
fn test_black_hole() {
  let t = (u32:2, u8:3, true);
  let (_, _, v) = t;
  assert_eq(v, true)
}
```

### Struct Types

Structures are similar to tuples, but provide two additional capabilities: we
name the slots (i.e. struct fields have names while tuple elements only have
positions), and we introduce a new type.

The following syntax is used to define a struct:

```dslx
struct Point {
  x: u32,
  y: u32
}
```

Once a struct is defined it can be constructed by naming the fields in any
order:

```dslx
struct Point {
  x: u32,
  y: u32,
}

#![test]
fn test_struct_equality() {
  let p0 = Point { x: u32:42, y: u32:64 };
  let p1 = Point { y: u32:64, x: u32:42 };
  assert_eq(p0, p1)
}
```

There is a simple syntax when creating a struct whose field names match the
names of in-scope values:

```dslx
struct Point { x: u32, y: u32, }

#![test]
fn test_struct_equality() {
  let x = u32:42;
  let y = u32:64;
  let p0 = Point { x, y };
  let p1 = Point { y, x };
  assert_eq(p0, p1)
}
```

Struct fields can also be accessed with "dot" syntax:

```dslx
struct Point {
  x: u32,
  y: u32,
}

fn f(p: Point) -> u32 {
  p.x + p.y
}

fn main() -> u32 {
  f(Point { x: u32:42, y: u32:64 })
}

#![test]
fn test_main() {
  assert_eq(u32:106, main())
}
```

Note that structs cannot be mutated "in place", the user must construct new
values by extracting the fields of the original struct mixed together with new
field values, as in the following:

```dslx
struct Point3 {
  x: u32,
  y: u32,
  z: u32,
}

fn update_y(p: Point3, new_y: u32) -> Point3 {
  Point3 { x: p.x, y: new_y, z: p.z }
}

fn main() -> Point3 {
  let p = Point3 { x: u32:42, y: u32:64, z: u32:256 };
  update_y(p, u32:128)
}

#![test]
fn test_main() {
  let want = Point3 { x: u32:42, y: u32:128, z: u32:256 };
  assert_eq(want, main())
}
```

#### Struct Update Syntax

The DSL has syntax for conveniently producing a new value with a subset of
fields updated to reduce verbosity. The "struct update" syntax is:

```dslx
struct Point3 {
  x: u32,
  y: u32,
  z: u32,
}

fn update_y(p: Point3) -> Point3 {
  Point3 { y: u32:42, ..p }
}

fn update_x_and_y(p: Point3) -> Point3 {
  Point3 { x: u32:42, y: u32:42, ..p }
}
```

#### Parametric Structs

DSLX also supports parametric structs. For more information on how
type-parametricity works, see the [parametric functions](#parametric-functions)
section.

```dslx
fn double(n: u32) -> u32 { n * u32:2 }

struct Point<N: u32, M: u32 = double(N)> {
  x: bits[N],
  y: bits[M],
}

fn make_point<A: u32, B: u32>(x: bits[A], y: bits[B]) -> Point<A, B> {
  Point{ x, y }
}

#![test]
fn test_struct_construction() {
  let p = make_point(u16:42, u32:42);
  assert_eq(u16:42, p.x)
}
```

#### Understanding Nominal Typing

As mentioned above, a struct definition introduces a new type. Structs are
nominally typed, as opposed to structurally typed (note that tuples are
structurally typed). This means that structs with different names have different
types, regardless of whether those structs have the same structure (i.e. even
when all the fields of two structures are identical, those structures are a
different type when they have a different name).

```dslx
struct Point {
  x: u32,
  y: u32,
}

struct Coordinate {
  x: u32,
  y: u32,
}

fn f(p: Point) -> u32 {
  p.x + p.y
}

#![test]
fn test_ok() {
  assert_eq(f(Point { x: u32:42, y: u32:64 }), u32:106)
}
```

```dslx-bad
#![test]
fn test_type_checker_error() {
  assert_eq(f(Coordinate { x: u32:42, y: u32:64 }), u32:106)
}
```

### Array Type

Arrays can be constructed via bracket notation. All values that make up the
array must have the same type. Arrays can be indexed with indexing notation
(`a[i]`) to retrieve a single element.

```dslx
fn main(a: u32[2], i: u1) -> u32 {
  a[i]
}

#![test]
fn test_main() {
  let x = u32:42;
  let y = u32:64;
  // Make an array with "bracket notation".
  let my_array: u32[2] = [x, y];
  let _ = assert_eq(main(my_array, u1:0), x);
  let _ = assert_eq(main(my_array, u1:1), y);
  ()
}
```

Because arrays with repeated trailing elements are common, the DSL supports
ellipsis (`...`) at the end of an array to fill the remainder of the array with
the last noted element. Because the compiler must know how many elements to
fill, in order to use the ellipsis the type must be annotated explicitly as
shown.

```dslx
fn make_array(x: u32) -> u32[3] {
  u32[3]:[u32:42, x, ...]
}

#![test]
fn test_make_array() {
  let _ = assert_eq(u32[3]:[u32:42, u32:42, u32:42], make_array(u32:42));
  let _ = assert_eq(u32[3]:[u32:42, u32:64, u32:64], make_array(u32:64));
  ()
}
```

TODO(meheff): Explain arrays and the intricacies of our bits type interpretation
and how it affects arrays of bits etc.

#### Character String Constants

Character strings are a special case of array types, being implicitly-sized
arrays of u8 elements. String constants can be used just as traditional arrays:

```dslx
fn add_one<N: u32>(input: u8[N]) -> u8[N] {
  for (i, result) : (u32, u8[N]) in range(u32:0, N) {
    update(result, i, result[i] + u8:1)
  }(input)
}

#![test]
fn test_main() {
  assert_eq("bcdef", add_one("abcde"))
}
```

DSLX string constants support the
[full Rust set of escape sequences](https://doc.rust-lang.org/reference/tokens.html) -
for multi-byte sequences, i.e., Unicode escapes, the resulting byte sequence
will be in printed order. In other words, the sequence `\u{89ab}` will result in
an array with the (binary) values `1000 1001 1010 1011` in sequence.

### Type Aliases

DLSX supports the definition of type aliases.

Type aliases can be used to provide a more human-readable name for an existing
type. The new name is on the left, the existing name on the right:

```dslx
type Weight = u6;
```

We can create an alias for an imported type:

```dslx
// Note: this imports an external file in the codebase.
import xls.dslx.tests.mod_imported

type MyEnum = mod_imported::MyEnum;

fn main(x: u8) -> MyEnum {
  x as MyEnum
}

#![test]
fn test_main() {
  let _ = assert_eq(main(u8:42), MyEnum::FOO);
  let _ = assert_eq(main(u8:64), MyEnum::BAR);
  ()
}
```

Type aliases can also provide a descriptive name for a tuple type (which is
otherwise anonymous). For example, to define a tuple type that represents a
float number with a sign bit, an 8-bit mantissa, and a 23-bit mantissa, one
would write:

```dslx
type F32 = (u1, u8, u23);
```

After this definition, the `F32` may be used as a type annotation
interchangeably with `(u1, u8, u23)`.

Note, however, that structs are generally preferred for this purpose, as they
are more readable and users do not need to rely on tuple elements having a
stable order in the future (i.e., they are resilient to refactoring).

### Type Casting

Bit types can be cast from one bit-width to another with the `as` keyword. Types
can be widened (increasing bit-width), narrowed (decreasing bit-width) and/or
changed between signed and unsigned. Some examples are found below. See
[Numerical Conversions](#numerical-conversions) for a description of the
semantics.

```dslx
#![test]
fn test_narrow_cast() {
  let twelve = u4:0b1100;
  assert_eq(twelve as u2, u2:0)
}

#![test]
fn test_widen_cast() {
  let three = u2:0b11;
  assert_eq(three as u4, u4:3)
}

#![test]
fn test_narrow_signed_cast() {
  let negative_seven = s4:0b1001;
  assert_eq(negative_seven as u2, u2:1)
}

#![test]
fn test_widen_signed_cast() {
  let negative_one = s2:0b11;
  assert_eq(negative_one as s4, s4:-1)
}

#![test]
fn test_widen_to_unsigned() {
  let negative_one = s2:0b11;
  assert_eq(negative_one as u3, u3:0b111)
}

#![test]
fn test_widen_to_signed() {
  let three = u2:0b11;
  assert_eq(three as u3, u3:0b011)
}
```

### Type Checking and Inference

DSLX performs type checking and produces an error if types in an expression
don't match up.

`let` expressions also perform type inference, which is quite convenient. For
example, instead of writing:

```dslx-snippet
let ch: u32 = (e & f) ^ ((!e) & g);
let (h, g, f): (u32, u32, u32) = (g, f, e);
```

one can write the following, as long as the types can be properly inferred:

```dslx-snippet
let ch = (e & f) ^ ((!e) & g);
let (h, g, f) = (g, f, e);
```

Note that type annotations can still be added and be used for program
understanding, as they they will be checked by DSLX.

### Type Inference Details

DSLX uses deductive type inference to check the types present in the program.
Deductive type inference is a set of (typically straight-forward) deduction
rules: Hindley-Milner style deductive type inference determines the result type
of a function with a rule that only observes the input types to that function.
(Note that operators like '+' are just slightly special functions in that they
have pre-defined special-syntax-rule names.)

#### Operator Example

For example, consider the binary (meaning takes two operands) / infix (meaning
it syntactically is placed in the center of its operands) '+' operator. The
simple deductive type inference rule for '+' is:

`(T, T) -> T`

Meaning that the left hand side operand to the '+' operator is of some type
(call it T), the right hand side operand to the '+' operator must be of that
same type, T, and the result of that operator is then (deduced) to be of the
same type as its operands, T.

Let's instantiate this rule in a function:

```dslx
fn add_wrapper(x: bits[2], y: bits[2]) -> bits[2] {
  x + y
}
```

This function wraps the '+' operator. It presents two arguments to the '+'
operator and then checks that the annotated return type on `add_wrapper` matches
the deduced type for the body of that function; that is, we ask the following
question of the '+' operator (since the type of the operands must be known at
the point the add is performed):

`(bits[2], bits[2]) -> ?`

To resolve the '?' the following procedure is being used:

*   Pattern match the rule given above `(T, T) -> T` to determine the type T:
    the left hand side operand is `bits[2]`, called T.
*   Check that the right hand side is also that same T, which it is: another
    `bits[2]`.
*   Deduce that the result type is that same type T: `bits[2]`.
*   That becomes the return type of the body of the function. Check that it is
    the same type as the annotated return type for the function, and it is!

The function is annotated to return `bits[2]`, and the deduced type of the body
is also `bits[2]`. Qed.

#### Type errors

A **type error** would occur in the following:

```dslx-bad
fn add_wrapper(x: bits[2], y: bits[3]) -> bits[2] {
  x + y
}
```

Applying the type deduction rule for '+' finds an inconsistency. The left hand
side operand has type `bits[2]`, called T, but the right hand side is `bits[3]`,
which is not the same as T. Because the deductive type inference rule does not
say what to do when the operand types are different, it results in a type error
which is flagged at this point in the program.

#### Let Bindings, Names, and the Environment

All expressions in the language's expression grammar have a deductive type
inference rule. The types must be known for inputs to an operator/function
(otherwise there'd be a use-before-definition error) and every expression has a
way to determine its type from its operand expressions.

A more interesting deduction rule comes into view with "let" expressions, which
are of the form:

`let $name: $annotated_type = $expr in $subexpr`

An example of this is:

`let x: u32 = u32:2 in x`

That's an expression which evaluates to the value '2' of type `u32`.

In a let expression like this, we say `$name` gets "bound" to a value of type
`$annotated_type`. The let typecheck must both check that `$expr` is of type
`$annotated_type`, as well as determine the type of `$subexpr`, which is the
type of the overall "let expression".

This leads to the deduction rule that "let just returns the type of `$subexpr`".
But, in this example, the subexpr needs some information from the outer `let`
expression, because if asked "what's the type of some symbol `y`" one
immediately asks "well what comes before that in the program text?"

Let bindings lead to the introduction of the notion of an *environment* that is
passed to type inference rules. The deduction rule says, "put the bindings that
`$name` is of type `$annotated_type` in the environment, then deduce the type of
`$subexpr`. Then we can simply say that the type of some identifier
`$identifier` is the type that we find looking up `$identifier` up in that
environment.

In the DSLX prototype code this environment is called the `Bindings`, and it
maps identifiers to the AST node that defines the name (`{Text: AstNode}`),
which can be combined with a mapping from AST node to its deduced type
(`{AstNode: ConcreteType}`) to resolve the type of an identifier.

## Statements

### Imports

DSLX modules can import other modules via the `import` keyword. Circular imports
are not permitted (the dependencies among DSLX modules must form a DAG, as in
languages like Go).

The import statement takes the following form (note the lack of semicolon):

```
import path.to.my.imported_module
```

With that statement, the module will be accessible as (the trailing identifier
after the last dot) `imported_module`; e.g. the program can refer to
`imported_module::IMPORTED_MODULE_PUBLIC_CONSTANT`.

NOTE Imports are relative to the Bazel "depot root" -- for external use of the
tools a `DSLX_PATH` will be exposed, akin to a `PYTHONPATH`, for users to
indicate paths where were should attempt module discovery.

NOTE Importing **does not** introduce any names into the current file other than
the one referred to by the import statement. That is, if `imported_module` had a
constant defined in it `FOO`, this is referred to via `imported_module::FOO`,
`FOO` does not "magically" get put in the current scope. This is analogous to
how wildcard imports are discouraged in other languages (e.g. `from import *` in
Python) on account of leading to "namespace pollution" and needing to specify
what happens when names conflict.

If you want to change the name of the imported module (for reference inside of
the importing file) you can use the `as` keyword:

```dslx-snippet
import path.to.my.imported_module as im
```

Just using the above construct,
`imported_module::IMPORTED_MODULE_PUBLIC_CONSTANT` is *not* valid, only
`im::IMPORTED_MODULE_PUBLIC_CONSTANT`. However, both statements can be used on
different lines:

```dslx-snippet
import path.to.my.imported_module
import path.to.my.imported_module as im
```

In this case, either `im::IMPORTED_MODULE_PUBLIC_CONSTANT` or
`imported_module::IMPORTED_MODULE_PUBLIC_CONSTANT` can be used to refer to the
same thing.

Here is an example using the same function via two different aliases for the
same module:

```dslx
// Note: this imports an external file in the codebase under two different
// names.
import xls.dslx.tests.mod_imported
import xls.dslx.tests.mod_imported as mi

fn main(x: u3) -> u1 {
  mod_imported::my_lsb(x) || mi::my_lsb(x)
}

#![test]
fn test_main() {
  assert_eq(u1:0b1, main(u3:0b001))
}
```

### Public module members

Module members are private by default and not accessible from any importing
module. To make a member public/visible to importing modules, the `pub` keyword
must be added as a prefix; e.g.

```dslx
const FOO = u32:42;      // Not accessible to importing modules.
pub const BAR = u32:64;  // Accessible to importing modules.
```

This applies to other things defined at module scope as well: functions, enums,
type aliases, etc.

```dslx
import xls.dslx.tests.mod_imported
import xls.dslx.tests.mod_imported as mi

fn main(x: u3) -> u1 {
  mod_imported::my_lsb(x) || mi::my_lsb(x)
}

#![test]
fn test_main() {
  assert_eq(u1:0b1, main(u3:0b001))
}
```

### Const

The `const` keyword is used to define module-level constant values. Named
constants should be usable anywhere a literal value can be used:

```dslx
const FOO = u8:42;

fn match_const(x: u8) -> u8 {
  match x {
    FOO => u8:0,
    _ => u8:42,
  }
}

#![test]
fn test_match_const_not_binding() {
  let _ = assert_eq(u8:42, match_const(u8:0));
  let _ = assert_eq(u8:42, match_const(u8:1));
  let _ = assert_eq(u8:0, match_const(u8:42));
  ()
}

fn h(t: (u8, (u16, u32))) -> u32 {
  match t {
    (FOO, (x, y)) => (x as u32) + y,
    (_, (y, u32:42)) => y as u32,
    _ => u32:7,
  }
}

#![test]
fn test_match_nested() {
  let _ = assert_eq(u32:3, h((u8:42, (u16:1, u32:2))));
  let _ = assert_eq(u32:1, h((u8:0, (u16:1, u32:42))));
  let _ = assert_eq(u32:7, h((u8:0, (u16:1, u32:0))));
  ()
}
```

## Expressions

### Unary Expressions

DSLX supports three types of unary expressions:

*   bit-wise not (the `!` operator)
*   negate (the `-` operator, computes the two's complement negation)

### Binary Expressions

DSLX supports a familiar set of binary expressions. There are two categories of
binary expressions. A category where both operands to the expression must be of
the same bit type (i.e., not arrays or tuples), and a category where the
operands can be of arbitrary bit types (i.e. shift expressions).

*   shift-right (`>>`)
*   shift-left (`<<`)
*   bit-wise or (`|`)
*   bit-wise and (`&`)
*   add (`+`)
*   subtract (`-`)
*   xor (`^`)
*   multiply (`*`)
*   logical or (`||`)
*   logical and (`&&`)

#### Shift Expressions

Shift expressions include: shift-right (logical) and shift-left. These are
binary operations that don't require the same type on the left and right hand
side. The right hand side must be unsigned, but it does not need to be the same
type or width as the left hand side, i.e. the type signature for these
operations is: `(xN[M], uN[N]) -> xN[M]`. If the right hand side is a literal
value it does not need to be type annotated. For example:

```dslx
fn shr_two(x: s32) -> s32 {
  x >> 2
}
```

Note that, as in Rust, the semantics of the shift-right (`>>`) operation depends
on the signedness of the left hand side. For a signed-type left hand side, the
shift-right (`>>`) operation performs a shift-right arithmetic and, for a
unsigned-type left hand side, the shift-right (`>>`) operation performs a
shift-right (logical).

### Comparison Expressions

For comparison expressions the types of both operands must match. However these
operations return a result of type `bits[1]`, aka `bool`.

*   equal (`==`)
*   not-equal (`!=`)
*   greater-equal (`>=`)
*   greater (`>`)
*   less-equal (`<=`)
*   less (`<`)

### Concat Expression

Bitwise concatenation is performed with the `++` operator. The value on the left
hand side becomes the most significant bits, the value on the right hand side
becomes the least significant bits. These may be chained together as shown
below:

```dslx
#![test]
fn test_bits_concat() {
  let _ = assert_eq(u8:0b11000000, u2:0b11 ++ u6:0b000000);
  let _ = assert_eq(u8:0b00000111, u2:0b00 ++ u6:0b000111);
  let _ = assert_eq(u6:0b100111, u1:1 ++ u2:0b00 ++ u3:0b111);
  let _ = assert_eq(u6:0b001000, u1:0 ++ u2:0b01 ++ u3:0b000);
  let _ = assert_eq(u32:0xdeadbeef, u16:0xdead ++ u16:0xbeef);
  ()
}
```

### Match Expression

Match expressions permit "pattern matching" on data, like a souped-up switch
statement. It can both test for values (like a conditional guard) and bind
values to identifiers for subsequent use. For example:

```dslx
fn f(t: (u8, u32)) -> u32 {
  match t {
    (u8:42, y) => y,
    (_, y) => y+u32:77
  }
}
```

If the first member of the tuple is the value is `42`, we pass the second tuple
member back as-is from the function. Otherwise, we add `77` to the value and
return that. The `_` symbolizes "I don't care about this value".

Just like literal constants, pattern matching can also match via named
constants; For example, consider this variation on the above:

```dslx
const MY_FAVORITE_NUMBER = u8:42;
fn f(t: (u8, u32)) -> u32 {
  match t {
    (MY_FAVORITE_NUMBER, y) => y,
    (_, y) => y+u32:77
  }
}
```

This also works with nested tuples; for example:

```dslx
const MY_FAVORITE_NUMBER = u8:42;
fn f(t: (u8, (u16, u32))) -> u32 {
  match t {
    (MY_FAVORITE_NUMBER, (y, z)) => y as u32 + z,
    (_, (y, u32:42)) => y as u32,
    _ => u32:7
  }
}
```

Here we use a "catch all" wildcard pattern in the last match arm to ensure the
match expression always matches the input somehow.

#### Redundant Patterns

`match` will flag an error if a _syntactically identical_ pattern is typed
twice; e.g.

```dslx-bad
const FOO = u32:42;
fn f(x: u32) -> u2 {
  match x {
    FOO => u2:0,
    FOO => u2:1,  // Identical pattern!
    _ => u2:2,
  }
}
```

Only the first pattern will ever match, so it is fully redundant (and therefore
likely a user error they'd like to be informed of). Note that _equivalent_ but
not _syntactically identical_ patterns will not be flagged in this way.

```dslx
const FOO = u32:42;
const BAR = u32:42;  // Compares `==` to `FOO`.
fn f(x: u32) -> u2 {
  match x {
    FOO => u2:0,
    BAR => u2:1,  // _Equivalent_ pattern, but not syntactically identical.
    _ => u2:2,
  }
}
```

### `let` Expression

let expressions work the same way as let expressions in other functional
languages (such as the ML family languages). let expressions provide a nested,
lexically-scoped, list of binding definitions. The scope of the binding is the
expression on the right hand side of the declaration. For example:

```dslx-snippet
let a: u32 = u32:1 + u32:2;
let b: u32 = a + u32:3;
b
```

would bind (and return as a value) the value `6` which corresponds to `b` when
evaluated. In effect there is little difference to other languages like C/C++ or
Python, where the same result would be achieved with code similar to this:

```python
a = 1 + 2
b = a + 3
return b
```

However, `let` expressions are lexically scoped. In above example, the value `3`
is bound to `a` only during the combined let expression sequence. There is no
other type of scoping in DSLX.

### Ternary If Expression

Note: ternary expression syntax is expected to change to mimic Rust's, see
[#318](https://github.com/google/xls/issues/318).

DSLX offers a ternary `if` expression, which is very similar to the Python
ternary `if`. Blueprint:

```
consequent if condition else alternate
```

This corresponds to the C/C++ ternary `?:` operator, but with the order of the
operands changed:

```
condition ? consequent : alternate
```

For example, in the FP adder module (modules/fpadd_2x32.x), there is code like
the following:

```
[...]
let result_sfd = result_sfd if wide_exponent < u9:255 else u23:0;
let result_exponent = wide_exponent as u8 if wide_exponent < u9:255 else u8:255;
```

### Iterable Expression

Iterable expressions are used in counted for loops. DSLX currently supports two
types of iterable expressions, `range` and `enumerate`.

The range expression `range(m, n)` produces values from m to n-1 (similar to how
typical loops are constructed in C/C++). This example will run from 0 to 4
(exclusive):

```
for (i, accum): (u32, u32) in range(u32:0, u32:4) {
```

`enumerate` iterates over the elements of an array type and produces pairs of
`(index, value)`, similar to enumeration constructs in languages like Python or
Go.

In the example below, the loop will iterate 8 times, following the array
dimension of `x`. Each iteration produces a tuple with the current index (`i`
ranging from 0 to 7) and the value at the index (`e = x[i]`).

```
fn prefix_scan_eq(x: u32[8]) -> bits[8,3] {
  let (_, _, result) =
    for ((i, e), (prior, count, result)): ((u32, u32), (u32, u3, bits[8,3]))
        in enumerate(x) {...
```

### for Expression

DSLX currently supports synthesis of "counted" for loops (loops that have a
clear upper bound on their number of iterations). These loops are capable of
being generated as unrolled pipeline stages: when generating a pipeline, the XLS
compiler will unroll and specialize the iterations.

NOTE In the future support for loops with an unbounded number of iterations may
be permitted, but will only be possible to synthesize as a time-multiplexed
implementation, since pipelines cannot be unrolled indefinitely.

#### Blueprint

```
for (index, accumulator): (type-of-index, type-of-accumulator) in iterable {
   body-expression
} (initial-accumulator-value)
```

The type annotation in the above "blueprint" is optional, but often helpful to
include for increased clarity.

Because DSLX is a pure dataflow description, a for loop is an expression that
produces a value. As a result, you grab the output of a for loop just like any
other expression:

```dslx-snippet
let final_accum = for (i, accum) in range(u32:0, u32:8) {
  let new_accum = f(accum);
  new_accum
}(init_accum);
```

Conceptually the for loop "evolves" the accumulator as it iterates, and
ultimately pops it out as the result of its evaluation.

#### Examples

Add up all values from 0 to 4 (exclusive). Note that we pass the accumulator's
initial value in as a parameter to this expression.

```dslx-snippet
for (i, accum): (u32, u32) in range(u32:0, u32:4) {
  accum + i
}(u32:0)
```

To add up values from 7 to 11 (exclusive), one would write:

```dslx-snippet
let base = u32:7;
for (i, accum): (u32, u32) in range(u32:0, u32:4) {
  accum + base + i
}(u32:0)
```

"Loop invariant" values (values that do not change as the loop runs) can be used
in the loop body, for example, note the use of `outer_thing` below:

```dslx-snippet
let outer_thing: u32 = u32:42;
for (i, accum): (u32, u32) in range(u32:0, u32:4) {
    accum + i + outer_thing
}(u32:0)
```

Both the index and accumulator can be of any valid type, in particular, the
accumulator can be a tuple type, which is useful for evolving a bunch of values.
For example, this for loop "evolves" two arrays:

```dslx-snippet
for (i, (xs, ys)): (u32, (u16[3], u8[3])) in range(u32:0, u32:4) {
  ...
}((init_xs, init_ys))
```

Note in the above example arrays are dataflow values just like anything else. To
conditionally update an array every other iteration:

```dslx-snippet
let result: u4[8] = for (i, array) in range(u32:0, u32:8) {
  // Update every other cell with the square of the index.
  update(array, i, i*i) if i % 2 == 0 else array
}(u4[8]:[0, ...]);
```

### Numerical Conversions {#numerical-conversions}

DSLX adopts the
[Rust rules](https://doc.rust-lang.org/1.30.0/book/first-edition/casting-between-types.html)
for semantics of numeric casts:

*   Casting from larger bit-widths to smaller bit-widths will truncate (to the
    LSbs).
*   Casting from a smaller bit-width to a larger bit-width will zero-extend if
    the source is unsigned, sign-extend if the source is signed.
*   Casting from a bit-width to its own bit-width, between signed/unsigned, is a
    no-op.

```dslx
#![test]
fn test_numerical_conversions() {
  let s8_m2 = s8:-2;
  let u8_m2 = u8:-2;
  // Sign extension (source type is signed).
  let _ = assert_eq(s32:-2, s8_m2 as s32);
  let _ = assert_eq(u32:-2, s8_m2 as u32);
  let _ = assert_eq(s16:-2, s8_m2 as s16);
  let _ = assert_eq(u16:-2, s8_m2 as u16);
  // Zero extension (source type is unsigned).
  let _ = assert_eq(u32:0xfe, u8_m2 as u32);
  let _ = assert_eq(s32:0xfe, u8_m2 as s32);
  // Nop (bitwidth is unchanged).
  let _ = assert_eq(s8:-2, s8_m2 as s8);
  let _ = assert_eq(s8:-2, u8_m2 as s8);
  let _ = assert_eq(u8:-2, u8_m2 as u8);
  let _ = assert_eq(s8:-2, u8_m2 as s8);
  ()
}
```

### Array Conversions

Casting to an array takes bits from the MSb to the LSb; that is, the group of
bits including the MSb ends up as element 0, the next group ends up as element
1, and so on.

Casting from an array to bits performs the inverse operation: element 0 becomes
the MSbs of the resulting value.

All casts between arrays and bits must have the same total bit count.

```dslx
fn cast_to_array(x: u6) -> u2[3] {
  x as u2[3]
}

fn cast_from_array(a: u2[3]) -> u6 {
  a as u6
}

fn concat_arrays(a: u2[3], b: u2[3]) -> u2[6] {
  a ++ b
}

#![test]
fn test_cast_to_array() {
  let a_value: u6 = u6:0b011011;
  let a: u2[3] = cast_to_array(a_value);
  let a_array = u2[3]:[1, 2, 3];
  let _ = assert_eq(a, a_array);
  // Note: converting back from array to bits gives the original value.
  let _ = assert_eq(a_value, cast_from_array(a));

  let b_value: u6 = u6:0b111001;
  let b_array: u2[3] = u2[3]:[3, 2, 1];
  let b: u2[3] = cast_to_array(b_value);
  let _ = assert_eq(b, b_array);
  let _ = assert_eq(b_value, cast_from_array(b));

  // Concatenation of bits is analogous to concatenation of their converted
  // arrays. That is:
  //
  //  convert(concat(a, b)) == concat(convert(a), convert(b))
  let concat_value: u12 = a_value ++ b_value;
  let concat_array: u2[6] = concat_value as u2[6];
  let _ = assert_eq(concat_array, concat_arrays(a_array, b_array));

  // Show a few classic "endianness" example using 8-bit array values.
  let x = u32:0xdeadbeef;
  let _ = assert_eq(x as u8[4], u8[4]:[0xde, 0xad, 0xbe, 0xef]);
  let y = u16:0xbeef;
  let _ = assert_eq(y as u8[2], u8[2]:[0xbe, 0xef]);

  ()
}
```

### Bit Slice Expressions

DSLX supports Python-style bit slicing over bits types. Note that bits are
numbered 0..N starting "from the right (as you would write it on paper)" --
least significant bit, AKA LSb -- for example, for the value `u7:0b100_0111`:

```
    Bit    6 5 4 3 2 1 0
  Value    1 0 0 0 1 1 1
```

A slice expression `[N:M]` means to get from bit `N` (inclusive) to bit `M`
exclusive. The start and limit in the slice expression must be signed integral
values.

Aside: This can be confusing, because the `N` stands to the left of `M` in the
expression, but bit `N` would be to the 'right' of `M` in the classical bit
numbering. Additionally, this is not the case in the classical array
visualization, where element 0 is usually drawn on the left.

For example, the expression `[0:2]` would yield:

```
    Bit    6 5 4 3 2 1 0
  Value    1 0 0 0 1 1 1
                     ^ ^  included
                   ^      excluded

  Result:  0b11
```

Note that, as of now, the indices for this `[N:M]` form must be literal numbers
(so the compiler can determine the width of the result). To perform a slice with
a non-literal-number start position, see the `+:` form described below.

The slicing operation also support the python style slices with offsets from
start or end. To visualize, one can think of `x[ : -1]` as the equivalent of
`x[from the start : bitwidth - 1]`. Correspondingly, `x[-1 : ]` can be
visualized as `[ bitwidth - 1 : to the end]`.

For example, to get all bits, *except* the MSb (from the beginning, until the
top element minus 1):

```dslx-snippet
x[:-1]
```

Or to get the two most significant bits:

```dslx-snippet
x[-2:]
```

This results in the nice property that a the original complete value can be
sliced into complementary slices such as `:-2` (all but the two most significant
bits) and `-2:` (the two most significant bits):

```dslx
#![test]
fn slice_into_two_pieces() {
  let x = u5:0b11000;
  let (lo, hi): (u3, u2) = (x[:-2], x[-2:]);
  let _ = assert_eq(hi, u2:0b11);
  let _ = assert_eq(lo, u3:0b000);
  ()
}
```

#### Width Slice

There is also a "width slice" form `x[start +: bits[N]]` - starting from a
specified bit, slice out the next `N` bits. This is equivalent to:
`bits[N]:(x >> start)`. The type can be specified as either signed or unsigned;
e.g. `[start +: s8]` will produce an 8-bit signed value starting at `start`,
whereas `[start +: u4]` will produce a 4-bit unsigned number starting at
`start`.

[Here are many more examples](https://github.com/google/xls/tree/main/xls/dslx/tests/bit_slice_syntax.x):

#### Bit Slice Examples

```dslx
// Identity function helper.
fn id<N: u32>(x: bits[N]) -> bits[N] { x }

#![test]
fn test_bit_slice_syntax() {
  let x = u6:0b100111;
  // Slice out two bits.
  let _ = assert_eq(u2:0b11, x[0:2]);
  let _ = assert_eq(u2:0b11, x[1:3]);
  let _ = assert_eq(u2:0b01, x[2:4]);
  let _ = assert_eq(u2:0b00, x[3:5]);

  // Slice out three bits.
  let _ = assert_eq(u3:0b111, x[0:3]);
  let _ = assert_eq(u3:0b011, x[1:4]);
  let _ = assert_eq(u3:0b001, x[2:5]);
  let _ = assert_eq(u3:0b100, x[3:6]);

  // Slice out from the end.
  let _ = assert_eq(u1:0b1, x[-1:]);
  let _ = assert_eq(u1:0b1, x[-1:6]);
  let _ = assert_eq(u2:0b10, x[-2:]);
  let _ = assert_eq(u2:0b10, x[-2:6]);
  let _ = assert_eq(u3:0b100, x[-3:]);
  let _ = assert_eq(u3:0b100, x[-3:6]);
  let _ = assert_eq(u4:0b1001, x[-4:]);
  let _ = assert_eq(u4:0b1001, x[-4:6]);

  // Slice both relative to the end (MSb).
  let _ = assert_eq(u2:0b01, x[-4:-2]);
  let _ = assert_eq(u2:0b11, x[-6:-4]);

  // Slice out from the beginning (LSb).
  let _ = assert_eq(u5:0b00111, x[:-1]);
  let _ = assert_eq(u4:0b0111, x[:-2]);
  let _ = assert_eq(u3:0b111, x[:-3]);
  let _ = assert_eq(u2:0b11, x[:-4]);
  let _ = assert_eq(u1:0b1, x[:-5]);

  // Slicing past the end just means we hit the end (as in Python).
  let _ = assert_eq(u1:0b1, x[5:7]);
  let _ = assert_eq(u1:0b1, x[-7:1]);
  let _ = assert_eq(bits[0]:0, x[-7:-6]);
  let _ = assert_eq(bits[0]:0, x[-6:-6]);
  let _ = assert_eq(bits[0]:0, x[6:6]);
  let _ = assert_eq(bits[0]:0, x[6:7]);
  let _ = assert_eq(u1:1, x[-6:-5]);

  // Slice of a slice.
  let _ = assert_eq(u2:0b11, x[:4][1:3]);

  // Slice of an invocation.
  let _ = assert_eq(u2:0b01, id(x)[2:4]);

  // Explicit-width slices.
  let _ = assert_eq(u2:0b01, x[2+:u2]);
  let _ = assert_eq(s3:0b100, x[3+:s3]);
  let _ = assert_eq(u3:0b001, x[5+:u3]);
  ()
}
```

### Advanced Understanding: Parametricity, Constraints, and Unification

An infamous wrinkle is introduced for parametric functions: consider the
following function:

```dslx-snippet
// (Note: DSLX does not currently support the `T: type` construct shown here,
// it is for example purposes only.)
fn add_wrapper<T: type, U: type>(x: T, y: U) -> T {
  x + y
}
```

Based on the inference rule, we know that '+' can only type check when the
operand types are the same. This means we can conclude that type `T` is the same
as type `U`. Once we determine this, we need to make sure anywhere `U` is used
it is consistent with the fact it is the same as `T`. In a sense the + operator
is "adding a constraint" that `T` is equivalent to `U`, and trying to check that
fact is valid is under the purview of type inference. The fact that the
constraint is added that `T` and `U` are the same type is referred to as
"unification", as what was previously two entities with potentially different
constraints now has a single set of constraints that comes from the union of its
operand types.

DSLX's typechecker will go through the body of parametric functions per
invocation. As such, the typechecker will always have the invocation's
parametric values for use in asserting type consistency against "constraints"
such as derived parametric expressions, body vs. annotated return type equality,
and expression inference rules.

### Operator Precedence

DSLX's operator precedence matches Rust's. Listed below are DSLX's operators in
descending precedence order. Binary operators at the same level share the same
associativity and will be grouped accordingly.

Operator                    | Associativity
--------------------------- | -------------
Unary `-` `!`               | n/a
`as`                        | Left to right
`*` `/` `%`                 | Left to right
`+` `-`                     | Left to right
`<<` `>>` `>>>`             | Left to right
`&`                         | Left to right
`^`                         | Left to right
`\|`                        | Left to right
`==` `!=` `<` `>` `<=` `>=` | Left to right
`&&`                        | Left to right
`\|\|`                      | Left to right

## Builtins

This section describes the built-in functions provided for use in the DSL that
do not need to be explicitly imported.

A note on "Parallel Primitives": the DSL is expected to grow additional support
for use of high-level parallel primitives over time, adding operators for
order-insensitive reductions, scans, groupings, and similar. By making these
operations known to the compiler in their high level form, we potentially enable
optimizations and analyses on their higher level ("lifted") form. As of now,
`map` is the sole parallel-primitive-oriented built-in.

### add_with_carry

Operation that produces the result of the add, as well as the carry bit as an
output. The binary add operators works similar to software programming
languages, preserving the length of the input operands, so this builtin can
assist when easy access to the carry out value is desired. Has the following
signature:

```
fn add_with_carry<N>(x: uN[N], y: uN[N]) -> (u1, uN[N])
```

### map

`map`, similarly to other languages, executes a transformation function on all
the elements of an original array to produce the resulting "mapped' array.
[For example](https://github.com/google/xls/tree/main/xls/dslx/tests/map_of_stdlib_parametric.x):
taking the absolute value of each element in an input array:

```dslx
import std

fn main(x: s3[3]) -> s3[3] {
  let y: s3[3] = map(x, std::abs);
  y
}

#![test]
fn main_test() {
  let got: s3[3] = main(s3[3]:[-1, 1, 0]);
  assert_eq(s3[3]:[1, 1, 0], got)
}
```

Note that map is special, in that we can pass it a callee *as if* it were a
value. As a function that "takes" a function as an argument, `map` is a special
builtin -- in language implementor parlance it is a *higher order function*.

Implementation note: Functions are not first class values in the DSL, so the
name of the function must be referred to directly.

Note: Novel higher order functions (e.g. if a user wanted to write their own
`map`) cannot currently be written in user-level DSL code.


### `clz`, `ctz`

DSLX provides the common "count leading zeroes" and "count trailing zeroes"
functions:

```dslx-snippet
  let x0 = u32:0x0FFFFFF8;
  let x1 = clz(x0);
  let x2 = ctz(x0);
  let _ = assert_eq(u32:4, x1);
  assert_eq(u32:3, x2)
```

### concat

Named variant of the binary `++` bits-concatenation operator. Has the following
signature:

```
fn concat<N, M, NPM=N+M>(x: uN[N], y: uN[M]) -> uN[NPM]
```

In the above signature, `x` becomes the most significant bits in the results,
whereas `y` becomes the less significant bits in the result.

### one_hot

Converts a value to one-hot form. Has the following signature:

```
fn one_hot<N, NP1=N+1>(x: uN[N], lsb_is_prio: bool) -> uN[NP1]
```

When `lsb_is_prio` is true, the least significant bit that is set becomes the
one-hot bit in the result. When it is false, the most significant bit that is
set becomes the one-hot bit in the result.

When all bits in the input are unset, the additional bit present in the output
value (MSb) becomes set.

Example usage:
[`dslx/tests/one_hot.x`](https://github.com/google/xls/tree/main/xls/dslx/tests/one_hot.x).

See also the
[IR semantics for the `one_hot` op](./ir_semantics.md#one_hot).

### signex

Casting has well-defined extension rules, but in some cases it is necessary to
be explicit about sign-extensions, if just for code readability. For this, there
is the `signex` builtin.

To invoke the `signex` builtin, provide it with the operand to sign extend
(lhs), as well as the target type to extend to: these operands may be either
signed or unsigned. Note that the *value* of the right hand side is ignored,
only its type is used to determine the result type of the sign extension.

```dslx
#![test]
fn test_signex() {
  let x = u8:-1;
  let s: s32 = signex(x, s32:0);
  let u: u32 = signex(x, u32:0);
  assert_eq(s as u32, u)
}
```

Note that both `s` and `u` contain the same bits in the above example.

### slice

Array-slice builtin operation. Note that the "want" argument is *not* used as a
value, but is just used to reflect the desired slice type. (Prior to constexprs
being passed to builtin functions, this was the canonical way to reflect a
constexpr in the type system.) Has the following signature:

```
fn slice<T: type, N, M, S>(xs: T[N], start: uN[M], want: T[S]) -> T[S]
```

### rev

`rev` is used to reverse the bits in an unsigned bits value. The LSb in the
input becomes the MSb in the result, the 2nd LSb becomes the 2nd MSb in the
result, and so on.

```dslx
// (Dummy) wrapper around reverse.
fn wrapper<N: u32>(x: bits[N]) -> bits[N] {
  rev(x)
}

// Target for IR conversion that works on u3s.
fn main(x: u3) -> u3 {
  wrapper(x)
}

// Reverse examples.
#![test]
fn test_reverse() {
  let _ = assert_eq(u3:0b100, main(u3:0b001));
  let _ = assert_eq(u3:0b001, main(u3:0b100));
  let _ = assert_eq(bits[0]:0, rev(bits[0]:0));
  let _ = assert_eq(u1:1, rev(u1:1));
  let _ = assert_eq(u2:0b10, rev(u2:0b01));
  let _ = assert_eq(u2:0b00, rev(u2:0b00));
  ()
}
```

### `bit_slice_update`

`bit_slice_update(subject, start, value)` returns a copy of the bits-typed value
`subject` where the contiguous bits starting at index `start` (where 0 is the
least-significant bit) are replaced with `value`. The bit-width of the returned
value is the same as the bit-width of `subject`. Any updated bit indices which
are out of bounds (if `start + bit-width(value) >= bit-width(subject)`) are
ignored. Example usage:
[`dslx/tests/bit_slice_update.x`](https://github.com/google/xls/tree/main/xls/dslx/tests/bit_slice_update.x).

### Bitwise reduction builtins: and_reduce, or_reduce, xor_reduce

These are unary reduction operations applied to a bits-typed value:

*   `and_reduce`: evaluates to bool:1 if all bits of the input are set, and 0
    otherwise.
*   `or_reduce`: evaluates to bool:1 if any bit of the input is set, and 0
    otherwise.
*   `xor_reduce`: evaluates to bool:1 if there is an odd number of bits set in
    the input, and 0 otherwise.

These functions return the identity element of the respective operation for
trivial (0 bit wide) inputs:

```dslx
#![test]
fn test_trivial_reduce() {
  let _ = assert_eq(and_reduce(bits[0]:0), true);
  let _ = assert_eq(or_reduce(bits[0]:0), false);
  let _ = assert_eq(xor_reduce(bits[0]:0), false);
  ()
}
```

### update

`update(array, index, new_value)` returns a copy of `array` where `array[index]`
has been replaced with `new_value`, and all other elements are unchanged. Note
that this is *not* an in-place update of the array, it is an "evolution" of
`array`. It is the compiler's responsibility to optimize by using mutation
instead of copying, when it's safe to do. The compiler makes a best effort to do
this, but can't guarantee the optimization is always made.

### assert_eq, assert_lt

In a unit test pseudo function all valid DSLX code is allowed. To evaluate test
results DSLX provides the `assert_eq` primitive (we'll add more of those in the
future). Here is an example of a `divceil` implementation with its corresponding
tests:

```dslx
fn divceil(x: u32, y: u32) -> u32 {
  (x-u32:1) / y + u32:1
}

#![test]
fn test_divceil() {
  let _ = assert_eq(u32:3, divceil(u32:5, u32:2));
  let _ = assert_eq(u32:2, divceil(u32:4, u32:2));
  let _ = assert_eq(u32:2, divceil(u32:3, u32:2));
  let _ = assert_eq(u32:1, divceil(u32:2, u32:2));
  _
}
```

Note that in this example, the final `let _ = ... in _` construct could be
omitted.

`assert_eq` cannot be synthesized into equivalent Verilog. Because of that it is
recommended to use it within `test` constructs (interpretation) only.

### trace!

DSLX supports printf-style debugging via the `trace!` builtin, which allows
dumping of current values to stdout. For example:

```dslx
fn decode_s_instruction(ins: u32) -> (u12, u5, u5, u3, u7) {
  let imm_11_5 = (ins >> u32:25);
  let rs2 = (ins >> u32:20) & u32:0x1F;
  let rs1 = (ins >> u32:15) & u32:0x1F;
  let funct3 = (ins >> u32:12) & u32:0x07;
  let imm_4_0 = (ins >> u32:7) & u32:0x1F;
  let opcode = ins & u32:0x7F;
  let _ = trace!(imm_11_5);
  let _ = trace!(imm_4_0);
  (((imm_11_5 as u7) ++ (imm_4_0 as u5)) as u12, rs2 as u5, rs1 as u5, funct3 as u3, opcode as u7)
}

```

would produce the following output, with each trace being annotated with its
corresponding source position:

```
[...]
[ RUN      ] decode_s_test_lsb
trace of imm_11_5 @ 69:17-69:27: bits[32]:0x1
trace of imm_4_0 @ 70:17-70:26: bits[32]:0x1
[...]
```

`trace` also returns the value passed to it, so it can be used inline, as in:

```
match trace!(my_thing) {
   [...]
}
```

To see the values of _all_ expressions during interpretation, invoke the
interpreter or test with the `--trace_all` flag:

```
$ ./interpreter_main clz.x -logtostderr -trace_all
[ RUN      ] clz
trace of (u3:0) @ clz.x:2:24: bits[3]:0x0
trace of (u3:0b111) @ clz.x:2:34-2:39: bits[3]:0x7
trace of clz((u3:0b111)) @ clz.x:2:30-2:40: bits[3]:0x0
trace of assert_eq((u3:0), clz((u3:0b111))) @ clz.x:2:20-2:41: ()
trace of (u3:1) @ clz.x:3:24: bits[3]:0x1
trace of (u3:0b011) @ clz.x:3:34-3:39: bits[3]:0x3
trace of clz((u3:0b011)) @ clz.x:3:30-3:40: bits[3]:0x1
trace of assert_eq((u3:1), clz((u3:0b011))) @ clz.x:3:20-3:41: ()
trace of (u3:2) @ clz.x:4:24: bits[3]:0x2
trace of (u3:0b001) @ clz.x:4:34-4:39: bits[3]:0x1
trace of clz((u3:0b001)) @ clz.x:4:30-4:40: bits[3]:0x2
trace of assert_eq((u3:2), clz((u3:0b001))) @ clz.x:4:20-4:41: ()
trace of (u3:3) @ clz.x:5:24: bits[3]:0x3
trace of (u3:0b000) @ clz.x:5:34-5:39: bits[3]:0x0
trace of clz((u3:0b000)) @ clz.x:5:30-5:40: bits[3]:0x3
trace of assert_eq((u3:3), clz((u3:0b000))) @ clz.x:5:20-5:41: ()
trace of () @ clz.x:6:3-6:5: ()
[       OK ] clz

```

Implementation note: tracing has no equivalent node in the IR (nor would such a
node make sense), so any `trace!` builtin invocations are silently dropped
during conversion to XLS IR.

### fail!

NOTE: this section describes work-in-progress functionality, currently `fail!`
will only trigger in DSL interpretation (it is discarded in IR conversion).
Support for converting `fail!` to XLS `assert` IR is tracked in
[#232](https://github.com/google/xls/issues/232) -- support for indicating the
assertion was triggered in the JIT is tracked in
[#308](https://github.com/google/xls/issues/308)

The `fail!` builtin indicates dataflow that should not be occurring in practice.
Its general signature is:

```
fail!(fallback_value)
```

The `fail!` builtin can be thought of as a "fatal assertion macro". It is used
to **annotate dataflow that should not occur in practice** and, if triggered,
should raise a fatal error in simulation (e.g. via a JIT-execution failure
status or a Verilog assertion when running in RTL simulation).

Note, however, that XLS will permit users to avoid inserting
fatal-error-signaling hardware that correspond to this `fail!` -- assuming it
will not be triggered in practice minimizes its cost in synthesized form. In
this situation, **when it is "erased", it acts as the identity function**,
propagating the `fallback_value`. This allows XLS to keep well defined semantics
even when fatal assertion hardware is not present.

Example: if only these two enum values shown should be possible (say, as a
documented [precondition](https://en.wikipedia.org/wiki/Precondition) for
`main`):

```dslx
enum EnumType: u2 {
  FIRST = 0,
  SECOND = 1,
}

fn main(x: EnumType) -> u32 {
  match x {
    EnumType::FIRST => u32:0,
    EnumType::SECOND => u32:1,
    _ => fail!(u32:0),
  }
}
```

The `fail!(u32:0)` above indicates that a) that match arm *should* not be
reached (and if it is in the JIT or RTL simulation it will cause an error status
or assertion failure respectively), but b) provides a fallback value to use (of
the appropriate type) in case it were to happen in synthesized gates which did
not insert fatal-error-indicating hardware.

### cover!

NOTE: This section describes work-in-progress functionality. Currently, `cover!`
has no effect. Progress is being tracked in
[#436](https://github.com/google/xls/issues/436).

The `cover!` builtin tracks how often some condition is satisfied. Its signature
is:

```
cover!(<name>, <condition>);
```

Where `name` is a function-unique literal string identifying the coverpoint and
`condition` is a boolean element. When `condition` is true, a counter with the
given name is incremented that can be inspected upon program termination.
Coverpoints can be used to give an indication of code "coverage", i.e. to see
what paths of a design are exercised in practice. The name of the coverpoint
must begin with either a letter or underscore, and its remainder must consist of
letters, digits, underscores, or dollar signs.

## Testing and Debugging

DSLX allows specifying tests right in the implementation file via the `test` and
`quickcheck` directives.

Having key test code in the implementation file serves two purposes. It helps to
ensure the code behaves as expected. Additionally it serves as 'executable'
documentation, similar in spirit to Python doc strings.

### Unit Tests

Unit tests are specified by the `test` directive, as seen below:

```dslx
#![test]
fn test_reverse() {
  let _ = assert_eq(u1:1, rev(u1:1));
  let _ = assert_eq(u2:0b10, rev(u2:0b01));
  let _ = assert_eq(u2:0b00, rev(u2:0b00));
  ()
}
```

The DSLX interpreter will execute all functions that are proceeded by a `test`
directive. These functions should be non-parametric, take no arguments, and
should return a unit-type.

Unless otherwise specified in the implementation's build configs, functions
called by unit tests are also converted to XLS IR and run through the
toolchain's LLVM JIT. The resulting values from the DSLX interpreter and the
LLVM JIT are compared against each other to assert equality. This is to ensure
DSLX implementations are IR-convertable and that IR translation is correct.

### QuickCheck

QuickCheck is a [testing framework concept][hughes-paper] founded on
property-based testing. Instead of specifying expected and test values,
QuickCheck asks for properties of the implementation that should hold true
against any input of the specified type(s). In DSLX, we use the `quickcheck`
directive to designate functions to be run via the toolchain's QuickCheck
framework. Here is an example that complements the unit testing of DSLX's `rev`
implementation from above:

```dslx
// Reversing a value twice gets you the original value.

#![quickcheck]
fn prop_double_reverse(x: u32) -> bool {
  x == rev(rev(x))
}
```

The DSLX interpreter will also execute all functions that are proceeded by a
`quickcheck` directive. These functions should be non-parametric and return a
`bool`. The framework will provide randomized input based on the types of the
arguments to the function (e.g. above, the framework will provided randomized
`u32`'s as `x`).

By default, the framework will run the function against 1000 sets of randomized
inputs. This default may be changed by specifying the `test_count` key in the
`quickcheck` directive before a particular test:

```
#![quickcheck(test_count=50000)]
```

The framework also allows programmers to specify a seed to use in generating the
random inputs, as opposed to letting the framework pick one. The seed chosen for
production can be found in the execution log.

For determinism, the DSLX interpreter should be run with the `seed` flag:
`./interpreter_main --seed=1234 <DSLX source file>`

[hughes-paper]: https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quick.pdf
