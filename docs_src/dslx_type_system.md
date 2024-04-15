# DSLX: Type System

The DSL (frontend) performs a few primary tasks:

1.  Parsing text files to an AST representation.
2.  Typechecking the AST representation.
3.  Conversion of the AST to bytecode that can be interpreted.
4.  Conversion of the AST to XLS IR (from which it can be interpreted or
    optimized or scheduled or code generated, etc.)

Note that step #2 is an essential component for steps #3 and #4 -- the type
information computed in the type checking process is used by the bytecode
emission/evaluation and IR conversion processes.

(You could imagine a bytecode interpreter that did not use any pre-computed type
information and tried to derive it all dynamically, but that is not how the
system is set up -- using the type information from typechecking time avoids
redundant work and replication of similar code in a way that could get out of
sync.)

Aside: bytecode emission/interpretation may also be necessary for constexpr
(compile-time constant) evaluation, and so #2 and #3 will be interleaved to some
degree.

## Parametric Instantiation

The most interesting thing that happens in the type system, and one of the main
things that the DSL provides as a useful abstraction on top of XLS IR, is
*parametric instantiation*. This is where users write a parameterized function
(or proc) and instantiate it with particular concrete parmeters; e.g.

```dslx
// A parametric (identity) function.
fn p<N: u32>(x: bits[N]) -> bits[N] { x }

fn main() -> u8 {
    p(u8:42)  // Instantiates p with N=8
}

#[test]
fn test_main() {
    assert_eq(main(), u8:42)
}
```

This allows us to write more generic code as library-style functions, which
dovetails nicely with the facilities that XLS core has to schedule and optimize
across cycles.

With parametric instantiation as a feature, several questions around the nature
of the parameterized definition are raised; e.g.

*   When `p` is defined with a parametric `N`, should we check that the
    definition has no type errors "for all N"? (Note: we do not.)
*   If `p` is not instantiated anywhere, do we check that `p` has no type errors
    for "there exists some N"? (Note: we do not.)

These relate to the "laziness" of parametric instantiation. As a historical
example for comparison, C++ originally had template definitions as token
substitutions, not even ensuring that the definition could parse, more akin to
syntactic macros.

## Typechecked on Instantiation

In the DSL, as noted above, the definitions of parametric instances are parsed,
but not typechecked until instantiation, and errors are raised if the concrete
set of parameterized values cause a type error during instantiation. That is:

```dslx
fn p<N: u32>() -> bits[N] {
    N()  // type error on instantiation: cannot invoke a number
}
```

If there is no instantiation of `p`, this definition will parse, but the type
error will go unreported, because it is never instantiated, and thus never type
checked. If another function were to instantiate it by calling `p`, however, a
type error would occur due to that instantiation.

Similarly, we can consider a definition that does not work "for all N", but
works "for one N", and that's the only `N` we instantiate it with.

```dslx
fn p<N: u32>() -> bits[N] {
    const_assert!(N == u32:42);
    u42:64
}
fn main() -> u42 {
    p<u32:42>()  // this is fine
}
```

## Parametric Evaluation Ordering

There are three components to parametric invocations. (Note: "binding" refers to
the assignment of a value to each named parameter.)

1.  Binding explicit values (given in angle brackets, i.e. `<>`) given in the
    caller
2.  Binding actual arguments (passed by the caller) against the parametric
    bindings
3.  Filling in any "remaining holes" in the parametric bindings using *default
    expressions* in the parametric bindings

The three components are performed in that order.

### 1: Binding Explicit Values

In this example:

```dslx
fn p<A: u32, B: u32>() -> (bits[A], bits[B]) {
    (bits[A]:42, bits[B]:64)
}

fn main() -> (u8, u16) {
    p<u32:8, u32:16>()
}
```

The caller `main` explicitly binds the parametrics `A` and `B` by supplying
arguments in the angle brackets.

### 2: Binding Actual Arguments

In this example:

```dslx
fn p<A: u32>(x: bits[A]) -> bits[A] {
    x + bits[A]:1
}
fn main() -> u13 {
    p(u13:42)
}

#[test]
fn test_main() {
  assert_eq(main(), u13:43)
}
```

`main` is implicitly saying what `A` must be by passing a `u13` -- we know that
the parameter to `p` is declared to be a `bits[A]`, so we know that `A` must be
`13` since a `u13` was passed as the "actual" argument (i.e. argument value from
the caller).

Note that if you contradict an explicit binding and a binding from actual
arguments, you will get a type error; e.g. this will cause a type error:

```dslx-snippet
fn main() -> u13 {
    p<u32:14>(u13:42)  // explicit says 14 bits, actual arg is 13 bits
}
```

### 3: Default Expressions

In this example:

```dslx
fn p<A: u32, B: u32 = {A+A}>(x: bits[A]) -> bits[B] {
    x as bits[B]
}

fn main() -> u32 {
    p(u16:42)
}

#[test]
fn test_main() {
    assert_eq(main(), u32:42);
}
```

`main` is implicitly saying what `A` must be by passing a `u16`; however, `B` is
not specified; neither by an explicit parametric value (i.e. in `<>` in the
caller), nor implicitly by an actual arg that was passed. As a result, we go
evaluate the *default expression* for the parametric, and populate `B` with the
result of the expression `A+A`. Since `A` is `16`, `B` is `32`.

### Aside: Earlier Bindings in Later Types

Note: this is not generally necessary to know to use parametrics effectively,
but is useful in thinking through the design and power of parametric
instantiation.

One consequence of the ordering defined is that earlier parametric bindings can
be used to define the types of later parametric bindings; e.g.

```dslx
fn p<A: u32, B: bits[A] = {bits[A]:0}>() -> bits[A] {
    B
}

fn main() -> u8 { p<u32:8>() }

#[test]
fn test_main() {
    assert_eq(main(), u8:0)
}
```

Note that `main` uses an explicit parametric to define `A` as `8`. Then the
*type* of `B` is defined based on the *value* of `A`; i.e. the type of B is
defined to be `u8` as a result, and the value of `B` is defined to be a `u8:0`.
This is interesting because we used an earlier parametric binding to define a
later parametric binding's type.
