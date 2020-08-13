<!--* freshness: { owner: 'rhundt' reviewed: '2019-10-29' } *-->

# Introduction to DSLX

[TOC]

DSLX is a domain specific, functional language to build hardware. It targets the
XLS toolchain to enable flows for FPGAs and ASICs (note that other frontends
will become available in the future).

By using a DSL we get the best possible representation of a given problem,
allowing us to explore the full potential of HLS, without being encumbered by
C++ language or compiler limitations details. The language is still experimental
and likely to change, but it is already useful for experimentation and
exploration.

This document provides am introduction to DSLX, mostly by example. After
perusing it and learning about the language features, we recommend exploring the
following, detailed examples to learn how the language features are put to
action:

1.  [CRC32](./g3doc/dslx_intro_example1.md)

2.  [Floating-point addition](./g3doc/fpadd_example.md)

3.  [Prefix Sum Computation](./g3doc/dslx_intro_example3.md)

In this document we use the function to compute a CRC32 checksum to describe
language feature. The full code is in `examples/dslx_intro/crc32_one_byte.x`.

### Comments

Just as in languages like C/C++, comments start with `//` and last through the
end of the line.

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

However, we suggest the following **DSLX style rules**:

*   Functions are `written_like_this`

*   User-defined data types are `NamesLikeThis`

*   Constant bindings are `NAMES_LIKE_THIS`

*   `_` is the black-hole identifier, as in Python. It should never be used in a
    binding-reference.

## Functions

Function definitions begin with the keyword `fn`, followed by the function name,
a parameter list to the function in parenthesis, followed by an `->` and the
return type of the function. After this, curly braces denote the begin and end
of the function body.

The list of parameters can be empty.

A single input file can contain many functions.

Simple examples:

```
fn ret3() -> u32 {
   3   // this function always returns 3.
}

fn add1(x: u32) -> u32 {
   x + u32:1   // returns x + 1, but you knew that.
}
```

Functions return the result of their last computed expression as their return
value. There are no explicit return statements. Functions also don't have
multiple return statements.

Tuples should be returned if a function should return multiple values.

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

```
fn double(n: bits[32]) -> bits[32] {
  n * bits[32]:2
}

fn [A: u32, B: u32 = double(A)] self_append(x: bits[A]) -> bits[B] {
  x++x
}

fn main() -> bits[10] { self_append(bits[5]:1) }
```

In `self_append(bits[5]:1)`, we see that `A = 5` based off of formal argument
instantiation. Using that value, we can evaluate `B = double(A=5)`. This derived
expression is analogous to C++'s constexpr – a simple expression that can be
evaluated at that point in compilation.

See
[advanced understanding](#advanced-understanding-parametricity-constraints-and-unification)
for more information on parametricity.

### Function Calls

Function calls are expressions and look and feel just like one would expect from
other languages. For example:

```
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

DSLX introduces shortcuts for commonly used types, such as `u8` for an 8-wide
bit type, or `u32` for a 32-bit wide bit type. These are defined up to `u64`.

All ``u*`` and ``bits[*]`` types are interpreted as unsigned numbers. Signed
numbers are specified via ``sN[*]``, which is analogous to ``bits[*]`` and
``uN[*]`` which are the corresponding unsigned version of the bit width ``*``.
For example:

```
sN[0]
s0

sN[1]
s1

sN[64]
s64

sN[256]
```

Similarly to unsigned numbers, the `s*` shorthands are defined up to `s64`.

Signed numbers differ in their behavior from unsigned numbers primarily via
operations like comparisons, (variable width) multiplications, and divisions.

### Enum Types

DSLX supports enumerations as a way of defining a group of related, scoped,
named constants that do not pollute the module namespace. For example:

```
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

```
enum Opcode : u3 {
  FOO = 8  // Causes compile time error!
}
```

Enums can be compared for equality/inequality, but they do not permit arithmetic
operations, they must be cast to numerical types in order to perform arithmetic:

```
fn same_opcode(x: Opcode, y: Opcode) -> bool {
  x == y  // ok
}

fn next_in_sequence(x: Opcode, y: Opcode) -> bool {
  // x+1 == y // does not work, arithmetic!
  u3:x + u3:1 == u3:y  // ok, casted first
}
```

As mentioned above casting of enum-values works with the same casting/extension
rules that apply to the underlying enum type definition. For example, this cast
will sign extend because the source type for the enum is signed. (See
[numerical conversions](./g3doc/dslx_intro.md#numerical-conversions)
for the full description of extension/truncation behavior.)

```
enum MySignedEnum : s3 {
  LOW = -1
  ZERO = 0
  HIGH = 1
}

fn extend_to_32b(x: MySignedEnum) -> u32 {
  u32:x  // Sign-extends because the source type is signed.
}

test extend_to_32b {
  assert_eq(extend_to_32b(MySignedEnum::LOW), u32:-1)
}
```

Casting *to* an enum is also permitted. However, in most cases errors from
invalid casting can only be found at runtime, e.g., in the DSL interpreter or
flagging a fatal error from hardware. Because of that, it is recommended to
avoid such casts as much as possible.

### Tuple Type

A tuple is an ordered set of fixed size containing elements of potentially
different types. Tuples can contain bits, arrays, or other tuples.

Examples:

```
(0b100, 0b101) a tuple containing two bits elements

// A tuple with 2 elements.
//   the 1st element is of type u32
//   the 2nd element is a tuple with 3 elements
//       the 1st element is of type u8
//       the 2nd element is another tuple with 1 element of type u16
//       the 3rd element is of type u8
(u32, (u8, (u16,), u8)
```

To access individual tuple elements use simple indices, starting
at 0 (Member access by field names is work in progress). For example, to
access the 2nd element of a tuple (index 1):

```
let t = (u32:2, u8:3);
assert_eq(u8:3, t[u32:1])
```

Another way to "destructure" a tuple into names is to use tuple assignment:

```
let t = (u32:2, u8:3);
let (a, b) = t;
let _ = assert_eq(u32:2, a);
assert_eq(u8:3, b)
```

Just as values can be discarded in a `let` by using the "black hole identifier"
`_`, don't care values can also be discarded when destructuring a tuple:

```
let t = (u32:2, u8:3, true);
let (_, _, v) = t;
assert_eq(v, true)
```

### Struct Type

Structs are "sugar" on top of tuples that give names to the various slots and
have convenient ways of constructing / accessing the members.

The following syntax is used to define a struct:

```
struct Point {
  x: u32,
  y: u32
}
```

Once a struct is defined it can be constructed by naming the fields in any
order:

```
struct Point {
  x: u32,
  y: u32,
}

test struct_equality {
  let p0 = Point { x: u32:42, y: u32:64 };
  let p1 = Point { y: u32:64, x: u32:42 };
  assert_eq(p0, p1)
}
```

Struct fields can also be accessed with "dot" syntax:

```
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

test main {
  assert_eq(u32:106, main())
}
```

Note that structs cannot be mutated "in place", the user must construct new
values by extracting the fields of the original struct mixed together with new
field values, as in the following:

```
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

test main {
  let want = Point3 { x: u32:42, y: u32:128, z: u32:256 };
  assert_eq(want, main())
}
```

The DSL has syntax for conveniently producing a new value with a subset of
fields updated to "feel" like the convenience of mutation. The "struct update"
syntax is:

```
fn update_y(p: Point3) -> Point3 {
  Point3 { y: u32:42, ..p }
}

fn update_x_and_y(p: Point3) -> Point3 {
  Point3 { x: u32:42, y: u32:42, ..p }
}
```

Note that structs are not compatible with other structs that happen to have the
same definition (this is called "nominal typing"). For example:

```
  def test_nominal_typing(self):
    # Nominal typing not structural, e.g. OtherPoint cannot be passed where we
    # want a Point, even though their members are the same.
    self._typecheck(
        """
        struct Point {
          x: s8,
          y: u32,
        }
        struct OtherPoint {
          x: s8,
          y: u32
        }
        fn f(x: Point) -> Point { x }
        fn g() -> Point {
          let shp = OtherPoint { x: s8:255, y: u32:1024 };
          f(shp)
        }
        """,
        error='parameter type name: \'Point\'; argument type name: \'OtherPoint\''
    )

  def test_bad_enum_ref(self):
```

### Array Type

Arrays can be constructed via bracket notation. All values that make up the
array must have the same type. Arrays can be indexed with indexing notation
(`a[i]`) to retrieve a single element.

```
fn main(a: u32[2], i: u1) -> u32 {
  a[i]
}

test main {
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

```
fn make_array(x: u32) -> u32[3] {
  u32[3]:[u32:42, x, ...]
}

test make_array {
  let _ = assert_eq(u32[3]:[u32:42, u32:42, u32:42], make_array(u32:42));
  let _ = assert_eq(u32[3]:[u32:42, u32:64, u32:64], make_array(u32:64));
  ()
}
```

TODO(meheff): Explain arrays and the intricacies of our bits type interpretation
and how it affects arrays of bits etc.

### Type Aliases

DLSX supports the definition of type aliases.

For example, to define a tuple type to represent a float number with a sign bit,
an 8-bit mantissa, and a 23-bit mantissa, one would write:

```
type F32 = {
  u1,
  u8,
  u23,
}
```

After this definition, the `F32` may be used as a type annotation
interchangeably with `(u1, u8, u23)`.

Note, however, that structs are generally preferred as described above, as they
are more readable and users do not need to rely on tuple indices remaining the
same in the future.

For direct type aliasing, users can use the following coding style (similar to
the C++ `typedef` keyword):

```
type TypeWeight = u6;
```

### Type Casting

Bit types can be cast from one bit-width to another. To cast a given value to a
different bit-width, the required code looks similar to how other types are
being specified in DSLX: Simply prefix the value with the desired type, followed
by a `:`. In this example:

```
fn add_with_carry(x: bits[24], y: bits[24]) -> (u1, bits[24]) {
  let result = (u1:0 ++ x) + (u1:0 ++ y);
  (u1:(result >> bits[25]:24), bits[24]:result)
}
```

The type of `result` is inferred to be of type `bits[25]` (concatenation of a
24-bit value with an additional 1-bit value in the front).

The expression `bits[25]:24` specifies the type of the value `24` to be of
`bits[25]`. The larger expression `(u1:(result >> bits[25]:24)` casts the result
of the shift expression to a `u1`. Only a single bit is returned, the least
significant bit (the right-most bit).

The expression `bits[24]:result` casts the 25-bit value 'result' down to a
24-bit value, basically chopping off the leading bit.

### Type Checking and Inference

DSLX performs type checking and produces an error if types in an expression
don't match up.

`let` expressions also perform type inference, which is quite convenient.
For example, instead of writing:

```
let ch: u32 = (e & f) ^ ((!e) & g);
let (h, g, f): (u32, u32, u32) = (g, f, e);
```

one can write the following, as long as the types can be properly inferred:

```
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

```
fn add_wrapper(x: bits[2], y: bits[2]) -> bits[2] {
  x + y
}
```

This function wraps the '+' operator. It presents two arguments
to the '+' operator and then checks that the annotated return type on
`add_wrapper` matches the deduced type for the body of that function; that is,
we ask the following question of the '+' operator (since the type of the
operands must be known at the point the add is performed):

`(bits[2], bits[2]) -> ?`

To resolve the '?' the following procedure is being used:

*   Pattern match the rule given above `(T, T) -> T` to determine the type T:
    the left hand side operand is `bits[2]`, called T.
*   Check that the right hand side is also that same T, which it is:
    another `bits[2]`.
*   Deduce that the result type is that same type T: `bits[2]`.
*   That becomes the return type of the body of the function.
    Check that it is the same type as the annotated return type for the
    function, and it is!

The function is annotated to return `bits[2]`, and the deduced type of the
body is also `bits[2]`. Qed.

#### Type errors

A **type error** would occur in the following:

```
fn add_wrapper(x: bits[2], y: bits[3]) -> bits[2] {
  x + y
}
```

Applying the type deduction rule for '+' finds an
inconsistency. The left hand side operand has type `bits[2]`, called T,
but the right hand side is `bits[3]`, which is not the same as T.
Because the deductive type inference rule does not say what to do when the
operand types are different, it results in a type error which is flagged
at this point in the program.

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

This leads to the deduction rule that "let just returns the type of
`$subexpr`". But, in this example, the subexpr needs some
information from the outer `let` expression, because if asked "what's the
type of some symbol `y`" one immediately asks "well what comes before that in
the program text?"

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


## Expressions

### Unary Expressions

DSLX supports three types of unary expressions:

*   bit-wise not (the `!` operator)
*   negate (the `-` operator, computes the two's complement negation)

### Binary Expressions

DSLX support a familiar set of binary expressions. For those, both operands to
the expression must have the same type. This is true even for the shift
operators.

*   shift-right (`>>`)
*   shift-right arithmetic (`>>>`)
*   shift-left (`<<`)
*   bit-wise or (`|`)
*   bit-wise and (`&`)
*   add (`+`)
*   subtract (`-`)
*   xor (`^`)
*   multiply (`*`)
*   logical or (`||`)
*   logical and (`&&`)

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

TODO(meheff): Explain the intricacies of our bits type interpretation and how it
affects concat.

### Match Expression

Match expressions permit "pattern matching" on data, like a souped-up switch
statement. It can both test for values (like a conditional guard) and bind
values to identifiers for subsequent use. For example:

```
fn f(t: (u8, u32)) -> u32 {
  match t {
    (u8:42, y) => y;
    (_, y) => y+u8:77
  }
}
```

If the first member of the tuple is the value is `42`, we pass the second tuple
member back as-is from the function. Otherwise, we add `77` to the value and
return that. The `_` symbolizes "I don't care about this value".

Just like literal constants, pattern matching can also match via named
constants; For example, consider this variation on the above:

```
const MY_FAVORITE_NUMBER = u8:42;
fn f(t: (u8, u32)) -> u32 {
  match t {
    (MY_FAVORITE_NUMBER, y) => y;
    (_, y) => y+u8:77
  }
}
```

This also works with nested tuples; for example:

```
const MY_FAVORITE_NUMBER = u8:42;
fn f(t: (u8, (u16, u32))) -> u32 {
  match t {
    (MY_FAVORITE_NUMBER, (y, z)) => u32:y+z;
    (_, (y, u32:42)) => u32:y;
    _ => u32:7
  }
}
```

Here we use a "catch all" wildcard pattern in the last match arm to ensure the
match expression always matches the input somehow.


### let Expression

let expressions work the same way as let expressions in other functional
languages, such as the ML languages or Haskell. let expressions provide a
nested, lexically-scoped, list of declarations. The scope of the declaration is
the expression and the right hand side of the declaration. For example,

```
let a: u32 = u32:1 + u32:2;
let b: u32 = a + u32:3;
b
```

would bind (and return) the value `6` to `b`. In effect there is little
difference to other languages like C/C++ or python, where the same result would
be achieved with code similar to this:

```
a = 1+2
b = a+3
return b
```

However, let expressions are lexically scoped. In above example, the value 3 is
bound to `a` only during the combined let expression sequence. There is no other
type of scoping in DSLX.

### Ternary If Expression

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

Iterable expressions are used in counted for loops. DSLX currently support two
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

DSLX currently supports counted loops.

Blueprint:

```
for (index, accumulator): (type-of-index, type-of-accumulator) in
        iterable {
           body-expression
        } (initial-accumulator-value)
```

Examples:

Add up all values from 0 to 4 (exclusive). Note that we pass the accumulator's
initial value in as a parameter to this expression.

```
for (i, accum): (u32, u32)
in range(u32:0, u32:4) { accum + i }(u32:0)
```

To add up values from 7 to 11 (exclusive), one would write:

```
let base: u32 = u32:7;
for (i, accum): (u32, u32) in range(u32:0, u32:4) { accum + base + i }(u32:0)
```

Invariants can be used in the loop body, for example:

```
let outer_thing: u32 = u32:42;
for (i, accum): (u32, u32) in range(u32:0, u32:4) {
    accum + i + outer_thing
}(u32:0)
```

Both the index and accumulator can be of any valid type, in particular, they can
be tuple types. For example:

```
for ((i, e), (prior, count, result)): ((u32, u32), (u32, u3, bits[8,3]))
```


### Numerical Conversions

DSLX adopts the
[Rust rules](https://doc.rust-lang.org/1.30.0/book/first-edition/casting-between-types.html)
for semantics of numeric casts:

*   Casting from larger bit-widths to smaller bit-widths will truncate (to the
    LSbs).
*   Casting from a smaller bit-width to a larger bit-width will zero-extend if
    the source is unsigned, sign-extend if the source is signed.
*   Casting from a bit-width to its own bit-width, between signed/unsigned, is a
    no-op.

```
test numerical_conversions {
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

```
fn cast_to_array(x: u6) -> u2[3] {
  x as u2[3]
}

fn cast_from_array(a: u2[3]) -> u6 {
  a as u6
}

fn concat_arrays(a: u2[3], b: u2[3]) -> u2[6] {
  a ++ b
}

test cast_to_array {
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

### Advanced Understanding: Parametricity, Constraints, and Unification

An infamous wrinkle is introduced for parametric functions: consider the
following function:

```
fn [T: type, U: type] add_wrapper(x: T, y: U) -> T {
  x + y
}
```

Based on the inference rule, we know that '+' can only type check when the
operand types are the same. This means we can conclude that type `T` is the same
as type `U`. Once we determine this, we need to make sure anywhere `U` is used
it is consistent with the fact it is the same as `T`. In a sense the +
operator is "adding a constraint" that `T` is equivalent to `U`, and trying to
check that fact is valid is under the purview of type inference. The fact that
the constraint is added that `T` and `U` are the same type is referred to as
"unification", as what was previously two entities with potentially different
constraints now has a single set of constraints that comes from the union of its
operand types.

DSLX's typechecker will go through the body of parametric functions per
invocation. As such, the typechecker will always have the invocation's
parametric values for use in asserting type consistency against "constraints"
such as derived parametric expressions, body vs. annotated return type equality,
and expression inference rules.

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

```
import path.to.my.imported_module as im
```

Just using the above construct,
`imported_module::IMPORTED_MODULE_PUBLIC_CONSTANT` is *not* valid, only
`im::IMPORTED_MODULE_PUBLIC_CONSTANT`. However, both statements can be used on
different lines:

```
import path.to.my.imported_module
import path.to.my.imported_module as im
```

In this case, either `im::IMPORTED_MODULE_PUBLIC_CONSTANT` or
`imported_module::IMPORTED_MODULE_PUBLIC_CONSTANT` can be used to refer to the
same thing.

Here is an example using the same function via two different aliases for the
same module:

```
import xls.dslx.interpreter.tests.mod_imported
import xls.dslx.interpreter.tests.mod_imported as mi

fn main(x: u3) -> u1 {
  mod_imported::my_lsb(x) || mi::my_lsb(x)
}

test main {
  assert_eq(u1:0b1, main(u3:0b001))
}
```

### Public module members

Module members are private by default and not accessible from any importing
module. To make a member public/visible to importing modules, the `pub` keyword
must be added as a prefix; e.g.

```
const FOO = u32:42;      // Not accessible to importing modules.
pub const BAR = u32:64;  // Accessible to importing modules.
```

This applies to other things defined at module scope as well: functions, enums,
typedefs, etc.

```
import xls.dslx.interpreter.tests.mod_imported
import xls.dslx.interpreter.tests.mod_imported as mi

fn main(x: u3) -> u1 {
  mod_imported::my_lsb(x) || mi::my_lsb(x)
}

test main {
  assert_eq(u1:0b1, main(u3:0b001))
}
```

### Typedefs

To import a type defined in an imported module or make a convenient shorthand
for an existing type, the `typedef` construct can be used at module scope; e.g.
for the case of an enum:

```
import xls.dslx.interpreter.tests.mod_imported

type MyEnum = mod_imported::MyEnum;

fn main(x: u8) -> MyEnum {
  x as MyEnum
}

test main {
  let _ = assert_eq(main(u8:42), MyEnum::FOO);
  let _ = assert_eq(main(u8:64), MyEnum::BAR);
  ()
}
```

### Const

The `const` keyword is used to define module-level constant values. Named
constants should be usable anywhere a literal value can be used:

```
const FOO = u8:42;

fn match_const(x: u8) -> u8 {
  match x {
    FOO => u8:0;
    _ => u8:42;
  }
}

test match_const_not_binding {
  let _ = assert_eq(u8:42, match_const(u8:0));
  let _ = assert_eq(u8:42, match_const(u8:1));
  let _ = assert_eq(u8:0, match_const(u8:42));
  ()
}

fn h(t: (u8, (u16, u32))) -> u32 {
  match t {
    (FOO, (x, y)) => (x as u32) + y;
    (_, (y, u32:42)) => y as u32;
    _ => u32:7;
  }
}

test match_nested {
  let _ = assert_eq(u32:3, h((u8:42, (u16:1, u32:2))));
  let _ = assert_eq(u32:1, h((u8:0, (u16:1, u32:42))));
  let _ = assert_eq(u32:7, h((u8:0, (u16:1, u32:0))));
  ()
}
```

## Parallel Primitives

### map
### reduce
### group-by
TODO


## Builtins

### Bit Slicing

DSLX supports Python-style bit slicing over bits types. Note that bits are
numbered 0..N starting "from the right" (least significant bit, AKA LSb), for
example:

```
    Bit    6 5 4 3 2 1 0
  Value    1 0 0 0 1 1 1
```

A slice expression `[n:m]` means to get from bit `n` (inclusive)
to bit 'm' exclusive. This can be confusing, because the `n` stands to
the left of `m` in the expression, but bit `n` would be to the 'right'
of `m` in the classical bit numbering (note: Not in the classical
array visualization, where element 0 is usually drawn to the left).

For example, the expression `[0:2]` would yield:

```
    Bit    6 5 4 3 2 1 0
  Value    1 0 0 0 1 1 1
                     ^ ^  included
                   ^      excluded

  Result:  0b11
```

Note that, as of now, the indices for this `[n:m]` form must be literal numbers
(so the compiler can determine the width of the result). To perform a slice with
a non-literal-number start position, see the `+:` form described below.

The slicing operation also support the python style slices with offsets from
start or end. To visualize, one can think of `x[ : -1]` as the equivalent of
`x[from the start :  bitwidth - 1]`. Correspondingly, `x[-1 : ]` can be
visualized as `[ bitwidth - 1 : to the end]`.

For example, to get all bits, except the MSb (from the beginning,
until the top element minus 1):

```
x[:-1]
```

Or to get the left-most 2 bits (from bitwidth - 2, all the way to
the end):

```
x[-2:]
```

There is also a "counted" form `x[start +: bits[N]]` - starting from a specified
bit, slice out the next `N` bits. This is equivalent to: `bits[N]:(x >> start)`.
The type can be specified as either signed or unsigned; e.g. `[start +: s8]`
will produce an 8-bit signed value starting at `start`, whereas `[start +: u4]`
will produce a 4-bit unsigned number starting at `start`.

Here are many more examples:

```
// Identity function helper.
fn [N: u32] id(x: bits[N]) -> bits[N] { x }

test bit_slice_syntax {
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

### clz, ctz

DSLX provides the common "count leading zeroes" and "count trailing zeroes"
functions:

```
  let x0 = u32:0x0FFFFFF8;
  let x1 = clz(x0);
  let x2 = ctz(x0);
  let _ = assert_eq(u32:4, x1);
  assert_eq(u32:3, x2)
```

### signex

Casting has well-defined extension rules, but in some cases it is necessary to
be explicit about sign-extensions, if just for code readability. For this, there
is the `signex` builtin.

To invoke the `signex` builtin, provide it with the operand to sign extend
(lhs), as well as the target type to extend to: these operands may be either
signed or unsigned. Note that the *value* of the right hand side is ignored,
only its type is used to determine the result type of the sign extension.

```
  let x = u8:-1;
  let s: s32 = signex(x, s32:0);
  let u: u32 = signex(x, u32:0);
  assert_eq(u32:s, u)
```

Note that both `s` and `u` contain the same bits in the above example.

### rev

`rev` is used to reverse the bits in an unsigned bits value. The LSb in the
input becomes the MSb in the result, the 2nd LSb becomes the 2nd MSb in the
result, and so on.

```
// (Dummy) wrapper around reverse.
fn [N: u32] wrapper(x: bits[N]) -> bits[N] {
  rev(x)
}

// Target for IR conversion that works on u3s.
fn main(x: u3) -> u3 {
  wrapper(x)
}

// Reverse examples.
test reverse {
  let _ = assert_eq(u3:0b100, main(u3:0b001));
  let _ = assert_eq(u3:0b001, main(u3:0b100));
  let _ = assert_eq(bits[0]:0, rev(bits[0]:0));
  let _ = assert_eq(u1:1, rev(u1:1));
  let _ = assert_eq(u2:0b10, rev(u2:0b01));
  let _ = assert_eq(u2:0b00, rev(u2:0b00));
  ()
}
```

### Bitwise reductions

These are unary reduction operations applied to a bits-typed value:

*   `and_reduce`: evaluates to bits[N]:1 if all bits are set
*   `or_reduce`: evaluates to bits[N]:1 if any bit is set in the input, and 0
    otherwise.
*   `xor_reduce`: evaluates to bits[N]:1 if there is an odd number of bits set
    in the input, and 0 otherwise.

### update

`update(array, index, new_value)` updates `array` by replacing the value
previously at `index` with `new_value` and returns the updated array. Note that
this is not an in-place update of the array, it is an "evolution" of the array
value and it is up to the compiler to find places in which an in-place
replacement is viable.


## Testing and Debugging

DSLX allows specifying tests right in the implementation file via the `test`
construct, which looks similar to function definitions, but doesn't allow
parameters.

Having key test code in the implementation file serves two purposes. It helps
to ensure the code behaves as expected. Additionally it serves as
'executable' documentation, similar in spirit to Python doc strings.

### assert_eq

In a test pseudo function all valid DSLX code is allowed. To evaluate test
results DLSX provides the `assert_eq` primitive (we'll add more of those in the
future). Here is an example of a `divceil` implementation with its corresponding
tests:

```
fn divceil(x: u32, y: u32) -> u32 {
  (x-u32:1) / y + u32:1
}

test divceil {
  let _ = assert_eq(u32:3, divceil(u32:5, u32:2));
  let _ = assert_eq(u32:2, divceil(u32:4, u32:2));
  let _ = assert_eq(u32:2, divceil(u32:3, u32:2));
  let _ = assert_eq(u32:1, divceil(u32:2, u32:2));
  _
}

```

Note that in this example, the final `let _ = ... in _` construct could be
omitted.

`assert_eq` cannot be synthesized into equivalent Verilog. Because of that
it is recommended to use it within `test` constructs (interpretation) only.


### trace
DSLX supports printf-style debugging via the `trace` expression, which allows
dumping of current values to stdout. For example:

```
fn decode_s_instruction(ins: u32) -> (u12, u5, u5, u3, u7) {
   let imm_11_5 = (ins >> u32:25);
   let rs2 = (ins >> u32:20) & u32:0x1F;
   let rs1 = (ins >> u32:15) & u32:0x1F;
   let funct3 = (ins >> u32:12) & u32:0x07;
   let imm_4_0 = (ins >> u32:7) & u32:0x1F;
   let opcode = ins & u32:0x7F;
   let _ = trace(imm_11_5);
   let _ = trace(imm_4_0);
   (u12:(u7:imm_11_5 ++ u5:imm_4_0), u5:rs2, u5:rs1, u3:funct3, u7:opcode)
}

```

would produce the following output, with each trace being annotated with
its corresponding source position:

```
[...]
[ RUN      ] decode_s_test_lsb
trace of imm_11_5 @ 69:17-69:27: bits[32]:0x1
trace of imm_4_0 @ 70:17-70:26: bits[32]:0x1
[...]
```

`trace` also returns the value passed to it, so it can be used inline, as in:

```
match trace(my_thing) {
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

Tracing has no equivalent node in the IR (nor would such a node make sense), so
any `trace` nodes are silently dropped during conversion.

### fail!()

TODO(leary): Document the fail() expression.

# Appendix

## Operator Precedence

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
