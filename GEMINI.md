# XLS (Accelerated Hardware Synthesis)

This file provides context and agent instructions for the XLS project.

## Overview & Repository Layout

*   **What is XLS:** XLS is a "Mid-Level" Synthesis toolchain for hardware
    development. It shares goals with High-Level Synthesis (HLS) tools but
    operates at a lower level to improve development velocity, quality,
    reliability, and security.
*   **Open-Source Repository:** The core XLS toolchain and base libraries are
    open-sourced under the Apache 2.0 license at
    [https://github.com/google/xls](https://github.com/google/xls) with
    documentation at
    [https://google.github.io/xls/](https://google.github.io/xls/) (see
    `README.md` for directory layout and `CONTRIBUTING.md` for contribution
    guidelines).
*   **Bug Tracking:** Public bugs are tracked on
    [GitHub Issues](https://github.com/google/xls/issues).

## Building, Testing & Development Workflow

*   **Build Tool:** Use `bazelisk` (or `bazel`) for building and testing the
    open-source repository (configured in `.bazelrc`, with optional local
    overrides in `user.bazelrc`; see `docs_src/build_system.md` and
    `docs_src/bazel_rules_macros.md`).
*   **CI & Testing:** GitHub Actions runs continuous integration for the
    open-source repository (`.github/workflows/`).
*   **Closing GitHub Issues:** Use keywords like `Fixes google/xls#NNN` in the
    commit message.

## Style Guides & Code Conventions

*   **Google Style Guides (Authoritative):** Always follow the Google C++ Style
    Guide (https://google.github.io/styleguide/cppguide.html) for C++ and the
    corresponding Google style guides for other languages (e.g.,
    https://google.github.io/styleguide/pyguide.html for Python,
    https://google.github.io/styleguide/shellguide.html for shell scripts).
    Google style guides always take precedence.
*   **XLS Style Guide:** Where Google style guides are ambiguous or silent,
    follow the XLS Style Guide (https://google.github.io/xls/xls_style/,
    `docs_src/xls_style.md`).
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

## Documentation

*   **Locations:** Documentation sources live in `docs_src/` and are rendered
    with `mkdocs` (`mkdocs.yml`) at
    [https://google.github.io/xls/](https://google.github.io/xls/).
*   **DSLX Reference & Tutorials:** Documentation for DSLX (the XLS frontend
    language) is at `docs_src/dslx_reference.md` and `docs_src/dslx_std.md`,
    with tutorials under `docs_src/tutorials/` (in particular,
    `docs_src/tutorials/what_is_a_proc.md` for DSLX procs).
