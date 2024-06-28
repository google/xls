# XLS Style Guide

The Google style guides recommend enforcing local consistency where stylistic
choices are not predefined. This file notes some of the choices we make locally
in the XLS project, with the relevant Google style guides
([C++](https://google.github.io/styleguide/cppguide.html),
[Python](https://google.github.io/styleguide/pyguide.html)) as their bases.

## C++

*   Align the pointer or reference modifier token with the type; e.g. `Foo&
    foo = ...` instead of `Foo &foo = ...`, and `Foo* foo = ...` instead of `Foo
    *foo= ...`.

*   Use `/*parameter_name=*/value` style comments if you choose to annotate
    arguments in a function invocation. `clang-tidy` recognizes this form, and
    provides a Tricorder notification if `parameter_name` is mismatched against
    the parameter name of the callee.

*   Prefer `int64_t` over `int` to avoid any possibility of overflow.

*   Always use `Status` or `StatusOr` for any error that a user could encounter.

*   Other than user-facing errors, use `Status` only in exceptional situations.
    For example, `Status` is good to signal that a required file does not exist
    but not for signaling that constant folding did not constant fold an
    expression.

    See [how heavyweight is StatusOr](#how-heavyweight-is-statusor) for more
    details on thinking about the costs involved.

*   Internal errors for conditions that should never be false can use `CHECK`,
    but may also use `Status` or `StatusOr`.

*   Prefer to brace single-statement blocks. Because the `XLS_ASSIGN_OR_RETURN`
    macro expands into multiple statements, this can cause problems when using
    unbraced single-statement blocks. Instead of XLS developers needing to think
    about individual cases of single statement blocks, we brace all single
    statement blocks.

*   Prefer using `XLS_ASSIGN_OR_RETURN` / `XLS_RETURN_IF_ERROR` when
    appropriate, but when binding a `StatusOr` wrapped value prefer to name it
    `thing_or` so that it can be referenced without the wrapper as `thing`; e.g.

    ```
    absl::StatusOr<Thing> thing_or = f();
    if (!thing_or.ok()) {
      // ... handling of the status via thing_or.status() and returning ...
    }
    const Thing& thing = thing_or.value();
    ```

*   Prefer `CHECK` to `DCHECK`, except that `DCHECK` can be used to verify
    conditions that it would be too expensive to verify in production, but that
    are fast enough to include outside of production.

*   Use `QCHECK` and `LOG(QFATAL)` during program startup when verifying startup
    parameters (i.e., flags); prefer `CHECK` and `LOG(FATAL)` in all other
    circumstances, as the `Q` variants suppress `atexit` handling (including
    `--cpu_profile`).

*   Follow the C++ style guide for capitalization guidelines; however, in the
    somewhat ambiguous case of I/O (short for Input/Output, which we use often),
    the slash counts as internal spacing and therefore the capitalization we use
    is `IO`, as in `WrapIO` or `StreamingIOReader`.

*   Prefer to use the `XLS_FRIEND_TEST` macro vs friending manually-mangled test
    names.

    At times it can be useful to test unit test a private/protected member of a
    class, and the `XLS_FRIEND_TEST` macro makes this possible. Note that the
    test case must live outside an unnamed namespace in the test file for the
    "friending" to work properly.

*   For simple const accessors, for the sake of consistency in the code base,
    and a weak preference towards the benefits of information hiding, prefer to
    return view types over the apparent type of the member; e.g.

    ```
    class MyClass {
     public:
      // This return type is preferrable to `const std::vector<uint64_t>&`.
      absl::Span<const uint64_t> values() const { return values_; }

     private:
      std::vector<uint64_t> values_;
    };
    ```

*   Follow the
    [style guide's](https://google.github.io/styleguide/cppguide.html#Run-Time_Type_Information__RTTI_)
    decision to avoid RTTI. In practice, this means `down_cast<>` should be used
    instead of `dynamic_cast<>`. However, the style guide says to avoid
    hand-implementing RTTI-like workarounds. The DSLX and XLScc frontends are
    places where avoiding RTTI would require implementing workarounds that end
    up looking a lot like RTTI, so `dynamic_cast<>` is common and accepted for
    those parts of the codebase. Elsewhere, especially with IR `Node` types,
    `down_cast<>` should be used instead.

*   Prefer `std::string_view` to `absl::string_view`. `absl::string_view` mainly
    differs from `std::string_view` in construction from nullptr, which our
    usage/callers do not depend upon. This decision lets us switch over to the
    more consistent end-state sooner. Although the style guide recommends we
    prefer `absl::string_view` for now, the rationale for why does not really
    apply to us and their target end state is clear.

*   XLS code is often written in a functional (i.e. separating functions from
    the [ideally immutable] structs they operate on) and layered style, which
    leads to `_utils.h` style translation units that layer on and compose
    functionality. Prefer the suffix `_utils.h` for these, vs `_helpers.h` or
    other alternatives.

*   Static member functions should be used sparingly, generally only for
    factories that call a private constructor. We prefer to document
    implementations with a `/* static */` comment as a reminder to readers (and
    writers that there is no `this` available). Comments are not an ideal way to
    mark this kind of information, but there should be a small number of these
    functions and as factories it is unlikely the static qualifier will be
    dropped in the future to put the comments out of sync.

*   We prefer `absl::visit` over `std::visit` as it is reportedly higher
    performance.

*   We use C++ standard-library filesystem functions and data structures, in the
    absence of an accepted open-source alternative other than Boost.

### Functions

*   Short or easily-explained argument lists (as defined by the developer) can
    be explained inline with the rest of the function comment. For more complex
    argument lists, the following pattern should be used:

    ```
    // <Function description>
    // Args:
    //   arg1: <arg1 description>
    //   arg2: <arg2 description>
    //   ...
    ```

### IR nodes

#### Class Hierarchy and OOP Design

A frequently asked question about XLS's design is how the IR class hierarchy
gels with Google style guide recommendations. This section is intended to
provide a rationale for "tagging" leaf node types and for using
`node->Is<NodeType>()` and `switch (node->op())` to form categories of node
types instead of a class inheritance taxonomy.

The base type `Node` encapsulates an element that takes input operands and
produces an output, along with some metadata like type, name, and references to
source locations. Each IR node (e.g. `add`, `send`, `concat`, etc.) extends
directly from `Node`.

Each `Node` defines `op()` and `Is<NodeType>()` methods which are more
performant alternatives to C++'s RTTI. For example,

```
if (node->Is<Send>()) {
  return node->As<Send>()->channel_id();  // As<Send>() is down_cast<Send*>()
}
```

or

```
switch (node->op()) {
  case Op::kSend:
    return down_cast<Send*>(node)->channel_id();
  // case Op:: ...
}
```

are common patterns on IR nodes.

In many contexts, this code would be a cry for better abstractions- some
reasonable ideas include:

1.  A `virtual std::optional<int64_t> Node::channel_id()` (or
    `absl::StatusOr<int64_t>`) implementation.
2.  A subclass or mixin trait like `ChannelNode` that extends `Node` for `Send`
    to derive.
3.  A visitor that implements similar functionality outside of the class
    hierarchy.

These ideas are generally not a good fit for IR nodes. The first idea's main
problem is that there are a lot of node types and the base class will become
huge if it needs to contain every property of each type of node. Furthermore,
the base class will be difficult to reason about without more structure- e.g.
does a node with a `channel_id` sometimes, always, or never also have a
`predicate`?

The second idea seems to address the problems of the first, but it is not clear
how to design a useful type hierarchy for IR nodes. The subset of nodes we care
about is very context dependent. The examples above invite a `ChannelNode`
abstraction, but other places in the code might care about unary vs. n-ary ops,
or ops that produce bare values vs. tuples, or some other way to group nodes. We
can find ourselves facing the first idea's complexity explosion if we make a
mixin trait for each pass, and there aren't good ways to inject mixin traits for
each compilation unit.

The third idea of using a visitor (or, similarly, a typeclass) is used in the
XLS codebase at times, but mostly where there's some well-defined behavior for
most kinds of nodes. If you want to pluck out an ad-hoc subset of nodes, you
need to make a new kind of visitor for that subset and it ends up being similar
to the code above. In the limit, you might need all arbitrary combinations which
will lead to too many visitor types to maintain centrally.

Using `node->Is<NodeType>()` or `switch (node->op())` are concise and readable
ways for the common task of operating on a new category of nodes. The typical
OOP tools we'd often use instead don't map well to the needs of an IR, so we
discourage adding to the base type or type hierarchy of IR nodes. We encourage
gathering categories that are reused in
[node_util.h](https://github.com/google/xls/tree/main/xls/ir/node_util.h).

It's also worth noting that `node->op()`, `node->Is<NodeType>()`,
`node->As<NodeType>()`, and `down_cast<NodeType*>(node)` are more performant
than C++ RTTI and `dynamic_cast<>`. C++ RTTI is not designed to be cheap and if
we used `dynamic_cast<>` instead of our own tags + `down_cast<>`, we expect that
would perform significantly worse. Performance is not the primary rationale for
the design decision discussed above, but the knock-on performance effects
further support the decision.

#### Passing Node Types

*   Unlike most data, IR elements should be passed as non-const pointers, even
    when expected to be const (which would usually indicate passing them as
    const references). Experience has shown that IR elements often develop
    non-const usages over time. Consider the case of IR analysis passes - those
    passes themselves rarely need to mutate their input data, but they build up
    data structures whose users often need to mutate their contents. In
    addition, treating elements as pointers makes equality comparisons more
    straightforward (avoid taking an address of a reference) and helps avoid
    accidental copies (assigning a reference to local, etc.). Non-const pointer
    usage propagates outwards such that the few cases where a const reference
    could *actually* be appropriate become odd outliers, so our guidance is that
    IR elements should uniformly be passed as non-const pointers.

*   A corollary to the above is that `nullptr` is generally not a valid input to
    functions taking IR elements. When an IR element is optional, we recommend
    explicitly using `std::optional<T*>`. We deviate from the style guide here
    because for IR elements `T*` sometimes means `const T&`, `T&`, or just `T`
    in addition to `T*`, but which is not apparent from the signature. However,
    using `nullptr` for IR element types is OK when the usage is fully
    encapsulated.

## Protocol buffers

*   Prefer to use
    [proto3](https://developers.google.com/protocol-buffers/docs/proto3#simple)
    specifications in all new protocol buffer files.

## FAQ

### How heavyweight is `StatusOr`?

What follows is the **general guidance on how absl::StatusOr is used** -- it is
used extensively throughout the XLS code base as an error-style indicator object
wrapper, so it is important to understand the mental model used for its cost.

Consider cost wise that: a) creating an **ok** `StatusOr` is cheap, b) creating
a **non-ok** `StatusOr` is expensive (that is, imagine the non-ok `Status`
within a `StatusOr` is the expensive part).

The implication being: if there's an API where "not found" is a reasonable
outcome, prefer `std::optional<>` as a return value to indicate that / go with
the grain of cost.

Something like a filesystem API would be a classic example -- where you
shouldn't be rooting around looking for files that aren't there -- so a
not-found `absl::StatusOr` result would be fine to use.

A good potential mental model is to imagine the program may run with logging of
a traceback for every non-ok status that is created. (This is related to a
debugging capability in Google internally called
`--util_status_save_stack_trace` that captures backtraces when error `Status`es
are created.) Ideally, with such a logging flag turned on, the screen wouldn't
fill up with "non error tracebacks", only tracebacks from events where something
really went wrong.
