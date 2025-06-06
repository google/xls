# Type System V2

[TOC]

## Overview

Type System V2 (also called Type Inference V2, or TIv2) is a complete rewrite of
the DSLX type system.

From the TIV2 design doc, its goals are:

1.  Reduce the need for explicit types that seem to the DSLX programmer like
    they should be obvious.

1.  Continue to raise fatal errors for situations that are likely errors, such
    as enum mismatch, signed/unsigned mismatch, and implicit narrowing on
    assignment.

1.  Minimize pitfalls for code that successfully compiles. This means e.g. a
    statement like: `let d:u64 = (a_u16 + b_u16) + c_u32;` should not lose a
    carry bit at any intermediate step. SystemVerilog would agree with this.

1.  Integrate “normal” inference and parametric inference better than today, and
    make the latter less error-prone.

1.  Use a design that foresees support for type parametrics, and closures with
    untyped parameters.

## Definitions

concretization - the conversion of a `TypeAnnotation` into a `Type`. The `Type`
objects that TIv2 produces always have absolute dimensions (in the case of
arrays).

inference variable - an integer, bool, or type variable that type inference must
find a value for. All parametrics in DSLX source code are inference variables.
Type inference also internally defines variables of its own, e.g., to model the
equivalence of the types of binary operands.

`ParametricContext` - identifies an instantiation of a parametric entity
(`proc`, `fn`, or `struct`). Types and constants within the entity can vary
depending on the `ParametricContext`.

resolution - takes "indirect" `TypeAnnotations` and "expands" them. For example,
`MemberTypeAnnotation(S, "foo")` becomes the actual type of member "foo" of
`struct S`.

`TypeAnnotation` - the data structure that the frontend uses to represent types.
This is a hierarchy that mirrors the `Type` hierarchy, but can preserve any type
expression from DSLX source code, and can also represent the type of one entity
indirectly as a derivation from the type of another. `Type`, by contrast, always
uses absolute and reduced forms.

`TypeVariableTypeAnnotation` - an internal type annotation that the system uses
to reference a type variable for feeding its value forward.

`Type` - the data structure that the rest of the DSLX/XLS stack uses to
represent types. The ultimate output of both type systems. This is a hierarchy,
e.g., `TupleType`, `ArrayType`, etc.

`TypeInfo` - stores a mapping of AST nodes to `Type` objects (and their constant
values, where appropriate.)

unification - reconciles multiple type annotations associated with a type
variable, or produces a type mismatch error if this isn't feasible.

## Major components

!!! NOTE
    the list of "inputs" is simplified below. There are almost always other
    parameters such as `Module`, `FileTable`, `ParametricContext`, etc., which are
    also required to determine out the output. Only the major parameters are listed.

!!! NOTE
    Some algorithms are simplified for space purposes. Refer to the source
    code for full details.

### [`TypecheckModule`](https://github.com/google/xls/tree/main/xls/dslx/parse_and_typecheck.cc)

Top-level type-checking method. It decides to run v1 or v2.

**Input:** AST `Module`

**Output:** `TypecheckedModule`

**How:**

For v2: calls `PopulateTable` on the module and its dependencies to populate an
`InferenceTable`, then uses an `InferenceTableConverter` to convert it into its
`TypeInfo` hierarchy used by later compiler phases. This winds up performing all
required inference of implicit parametrics, resolution, unification and
concretization.

### [`PopulateTable`](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/populate_table.h) ([impl](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/populate_table.cc))

Populates the [`InferenceTable`](#inference-table) with type annotations and/or
type variables.

**Input:** AST `Module`

**Output:** `InferenceTable`

**How:**

Does a top-down traversal of the AST via `AstNodeVisitorWithDefault`. Each node
handler method collects explicit type annotations from the node, and assigns
them to the node in the `InferenceTable`, and defines and/or propagates type
variables.

For example, the `HandleConstantDef` method makes a type variable for the left
hand side node, then assigns it to that node via the `InferenceTable`. Then it
assigns the same type variable to the right hand side, also via the
`InferenceTable`. This connection of type variables is what subsequent steps
will use to make type inferences.

### [`InferenceTable`](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/inference_table.h) ([impl](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/inference_table.cc)) {#inference-table}

Stores information about type annotations, type variables, and other definitions
for the nodes in the AST. Other secondary data structures for tracking
parametric values and expressions are also in this file.

### [`InferenceTableConverter`](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/inference_table_converter.h) ([impl](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/inference_table_converter_impl.cc))

Builds the `Type` object(s) for the objects in the `InferenceTable`. The main
entry point is [`ConvertSubtree`](#convertsubtree) which is called on the
top-level `Module` by `TypecheckModule`.

**Input:** `InferenceTable`

**Output:** `TypeInfo` for the entire conversion

#### ConvertSubtree

**Input:** AST node

**Output:** populates `Type` for the given node in the appropriate `TypeInfo`

**How:**

1.  Figures out the "correct order" of its children nodes via
    `ConversionOrderVisitor`. The order generally means "constextpr dependency"
    order. Similarly, resolution for functions and implicit parametrics for
    invocations must come before the nodes they affect.
1.  For each node,
    1.  if it's an `Invocation`, return
        [`ConvertInvocation`](#convertinvocation)
    1.  else return [`GenerateTypeInfo`](#generatetypeinfo)

#### ConvertInvocation

**Input:** `Invocation` AST node

**Output:** populates `Type` for the given node in the appropriate `TypeInfo`

The outline of the algorithm is:

1.  Figures out the actual `Function` object by looking it up by its identifier
    in the `Module`.
1.  If not a parametric `Function`, determines the argument and return types and
    updates the inference table with those types. Returns
    [`GenerateTypeInfo`](#generatetypeinfo) with the information gleaned about
    the argument and return types. Otherwise, continues:
1.  Determine the effective concrete values of all parametrics for the
    invocation via [`SolveForParametrics`](#solveforparametrics)
1.  Determine the actual function type of the callee, given the parametrics.
1.  Determine the types of any actual arguments that were not determined already
    in order to infer parametric values.
1.  Produce `TypeInfo` for the contents of the callee function, or reuse an
    existing `TypeInfo` for another call that has the same effective parametric
    values.
1.  Determine the type of the invocation node (i.e., the function return type
    unified with the call site context).
1.  Returns [`GenerateTypeInfo`](#generatetypeinfo) with the information gleaned
    above

#### GenerateTypeInfo

**Input:** AST node, `ParametricContext`

**Output:** populates `Type` for the given node in the appropriate `TypeInfo`

**How:**

1.  First it resolves and unifies the type annotations for the given node.
1.  Processes specific subclasses of `TypeAnnotation`
1.  [`Concretize`](#concretize)s the given type annotation
1.  Validates the concrete type to perform last-minute checks on required types
1.  Tells the `TypeInfo` about the `type` object generated by `Concretize` in
    step 3
1.  Records const expressions.

#### Concretize

Convert the `TypeAnnotation` to a `Type`, replacing any remaining algebraic
expressions that weren't taken care of by unification.

Example: `uN[N + M]` becomes `uN[32]`.

### [`SolveForParametrics`](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/solve_for_parametrics.h) ([impl](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/solve_for_parametrics.cc)) {#solveforparametrics}

Given a set of parametric bindings, figure out how to satisfy parametric values.

For example: \
N: [u32:3] \
blah: [uN[N]]

returns: \
N: [u32:3] \
blah: [uN[u32:3]]

**Input:** set of `ParametricBinding`s

**Output:** map of `ParametricBinding` to `InterpValueOrTypeAnnotation`

### [`TypeAnnotationResolver`](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/type_annotation_resolver.h) ([impl](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/type_annotation_resolver.cc)) {#typeannotationresolver}

Wraps unification with resolution of indirect type annotations.

**Input:** a possibly "indirect" `TypeAnnotation`

**Output:** direct `TypeAnnotation` TBD (array or tuple or scalar?)

**What:**

1.  Converts indirect `TypeAnnotation`s into direct ones. E.g.,
    `MemberTypeAnnotation(S, "foo")` becomes the actual type of member `foo` in
    `struct S`.
1.  Calls [`UnifyTypeAnnotations`](#unifytypeannotations)

### [`UnifyTypeAnnotations`](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/unify_type_annotations.h) ([impl](https://github.com/google/xls/tree/main/xls/dslx/type_system_v2/unify_type_annotations.cc)) {#unifytypeannotations}

Reconciles the information in a vector of type annotations, producing either one
canonical annotation or a type mismatch error.

**Input:** a vector of `TypeAnnotation`s

**Output:** a single `TypeAnnotation`

## Example

TODO: mckeever - Add example DSLX source with inference table and trace dumps
