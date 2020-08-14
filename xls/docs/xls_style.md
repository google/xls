<!--* freshness: { owner: 'leary' reviewed: '2019-09-01' } *-->

# XLS Style Guide

The Google style guides recommend enforcing local consistency where stylistic
choices are not pre-defined. This file notes some of the choices we make locally
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

*   Prefer `int64` over `int` to avoid any possibility of overflow.

*   Always use `Status` or `StatusOr` for any error that a user could encounter.

*   Other than user-facing errors, use `Status` only in exceptional situations.
    For example, `Status` is good to signal that a required file does not exist
    but not for signaling that constant folding did not constant fold an
    expression.

*   Internal errors for conditions that should never be false can use `CHECK`,
    but may also use `Status` or `StatusOr`.

*   Prefer `CHECK` to `DCHECK`, except that `DCHECK` can be used to verify
    conditions that it would be too expensive to verify in production, but that
    are fast enough to include outside of production.

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
