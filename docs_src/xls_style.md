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
    could _actually_ be appropriate become odd outliers, so our guidance is that
    IR elements should uniformly be passed as non-const pointers.

## FAQ

### How heavyweight is `StatusOr`?

What follows is the **general guidance on how absl::StatusOr is used** -- it is
used extensively throughout the XLS code base as an error-style indicator object
wrapper, so it is important to understand the mental model used for its cost.

Consider cost wise that: a) creating an **ok** `StatusOr` is cheap, b) creating
a **non-ok** `StatusOr` is expensive (that is, imagine the non-ok `Status`
within a `StatusOr` is the expensive part).

The implication being: if there's an API where "not found" is a reasonable
outcome, prefer `absl::optional<>` as a return value to indicate that / go with
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
