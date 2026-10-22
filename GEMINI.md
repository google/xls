## XLS (Accelerated Hardware Synthesis)

This file provides a context for the XLS project.

**Core Information:**

*   **What is XLS:** XLS is a 'Mid-Level' Synthesis toolchain for hardware
    development. It is similar and has some of the same goals as High-level
    Synthesis (HLS) tools however it operates at a generally lower level. It
    aims to improve development velocity, quality, reliability, and security.
*   **Open Source:** The core XLS toolchain and base libraries are open-sourced
    under an Apache2 license and hosted on GitHub at
    [https://github.com/google/xls](https://github.com/google/xls). The open
    source documentation is at
    [https://google.github.io/xls/](https://google.github.io/xls/).
*   **Bugs:** Public bugs are tracked on
    [GitHub Issues](https://github.com/google/xls/issues).

**Style Guides:**

*   **Google C++ Style Guide:** Always follow the Google C++ Style Guide
    (https://google.github.io/styleguide/cppguide.html) for all C++ code. This
    is the authoritative style guide.
*   **Other Google Style Guides:** Follow other relevant Google style guides for
    other languages (e.g., https://google.github.io/styleguide/pyguide.html for
    Python, https://google.github.io/styleguide/shellguide.html for shell
    scripts).
*   **XLS Style Guide:** In cases where the Google style guides are ambiguous or
    do not cover a specific situation, consult the XLS Style Guide
    (https://google.github.io/xls/style_guide, `docs_src/xls_style.md`) for
    additional guidance. However, the Google style guides always take
    precedence.
*   **Optional Non-Owning References (`absl::optional_ref`):** Use
    `absl::optional_ref<T>` instead of `std::optional<T*>` (or nullable `T*`)
    when passing or returning an optional non-owning reference to a type that is
    normally passed by reference.
    *   *Exception:* IR, AST, and VAST node/element types (e.g., `Node*`,
        `FunctionBase*`, `dslx::AstNode*`, `verilog::VastNode*`) are uniformly
        passed as pointers per the XLS Style Guide, so optional values of those
        types should continue to use `std::optional<T*>`.
    *   *Conditional initialization with `XLS_ASSIGN_OR_RETURN`:* Because
        `absl::optional_ref<T>` is not assignable after construction, when
        conditionally populating one with `XLS_ASSIGN_OR_RETURN`, assign into a
        temporary `T*` initialized to `nullptr` inside the `if` block and
        construct the `absl::optional_ref<T>` from that pointer afterward (do
        not use an immediately-invoked lambda).
    *   *Naming:* Avoid `maybe_` prefixes or `_ptr` suffixes on
        `absl::optional_ref` and `std::optional` variables; name variables after
        their underlying value or role.

**Development Workflow:**

*   **Build Tool:** Use `bazelisk` for building and testing the open source
    version.
*   To close GitHub issues, use keywords like `Fixes google/xls#NNN` in the
    commit message.

**Testing:**

*   **GitHub Actions:** CI for the open-source repo.

**Documentation:**

*   Docs are in `docs_src` and rendered with `mkdocs` at
    [https://google.github.io/xls/](https://google.github.io/xls/).
*   Documentation on DSLX (the standard XLS frontend language) can be found at
    `docs_src/dslx_reference.md` and `docs_src/dslx_std.md`, along with some
    useful tutorials under `docs_src/tutorials/`.
