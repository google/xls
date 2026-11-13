## XLS (Accelerated Hardware Synthesis)

This file provides context and engineering guidelines for the XLS project.

**Core Information:**

*   **What is XLS:** XLS is a 'Mid-Level' Synthesis toolchain for hardware
    development. It shares goals with High-Level Synthesis (HLS) tools but
    operates at a lower level to improve development velocity, quality,
    reliability, and security.
*   **Open Source:** The core XLS toolchain and base libraries are open-sourced
    under an Apache 2.0 license at
    [https://github.com/google/xls](https://github.com/google/xls), with
    documentation at
    [https://google.github.io/xls/](https://google.github.io/xls/) (source in
    `docs_src/`, including `docs_src/dslx_reference.md`, `docs_src/dslx_std.md`,
    and `docs_src/tutorials/`).
*   **Bugs & Issues:** Public bugs are tracked on
    [GitHub Issues](https://github.com/google/xls/issues). Use keywords like
    `Fixes google/xls#NNN` in commit messages to close issues.

**Style Guides & Code Quality:**

*   **Style Guides:** Always follow the Google style guides
    ([C++](https://google.github.io/styleguide/cppguide.html),
    [Python](https://google.github.io/styleguide/pyguide.html),
    [Shell](https://google.github.io/styleguide/shellguide.html)), which take
    precedence, and the XLS Style Guide
    ([https://google.github.io/xls/style_guide](https://google.github.io/xls/style_guide),
    `docs_src/xls_style.md`).
*   **Optional Non-Owning References (`absl::optional_ref`):** Use
    `absl::optional_ref<T>` instead of `std::optional<T*>` (or nullable `T*`)
    for optional non-owning references, except for IR, AST, and VAST
    node/element types (`Node*`, `FunctionBase*`, `dslx::AstNode*`,
    `verilog::VastNode*`), which are uniformly passed as pointers and use
    `std::optional<T*>` (see `docs_src/xls_style.md` for `XLS_ASSIGN_OR_RETURN`
    usage and naming).
*   **Conceptual Simplicity & Reuse:** Prefer the simplest code that solves the
    general problem at the root cause rather than patching a single caller or
    pass. Reuse existing XLS/Abseil utilities (`DecoratingDelayEstimator`,
    `AstNodeRecursiveVisitor`, `absl::FunctionRef`/`AnyInvocable` over
    `std::function`, `absl::SimpleAtoi`, `ABSL_FLAG`) and factor multi-statement
    lambdas or local classes into module-scope helpers.
*   **Error Handling:** Never silently saturate on integer overflow in
    scheduling/delay math or ignore statuses with `.IgnoreError()` when failure
    indicates a broken invariant; return `absl::InvalidArgumentError` or use
    `XLS_RET_CHECK` / `XLS_ASSIGN_OR_RETURN`.
*   **Codegen 1.5 Default:** Target Codegen 1.5 (`xls/codegen_v_1_5/`) for all
    new codegen bug fixes and features. Do not modify Codegen 1.0
    (`xls/codegen/`) unless trivial or explicitly requested.
*   **Idiomatic DSLX (Don't Transliterate Verilog):** Follow the DSLX section in
    `docs_src/xls_style.md` (`bool` over `u1`, `x[i+:bool]`, `std::msb(x)`,
    `enum`s, `pub struct` + `impl` over >3-element tuples, pure datapath
    functions separated from `proc` FSM state, `PascalCase` `proc` names, and
    `dslx_fmt --simplify_type_annotations -i`).
*   **Verification & Hermetic Tests:**
    *   **Build & CI:** Use `bazelisk` for building and testing; GitHub Actions
        runs CI.
    *   **Fail on base:** Verify that every new regression test fails without
        the fix and passes with it. Keep each change to one logical concern and
        include regression tests/crashers in the same change as the fix.
    *   **Small IR & fast tests:** Build test IR with
        `FunctionBuilder`/`ProcBuilder`/`BlockBuilder` (not
        `Parser::ParsePackage`). Keep `ScopedVerifyEquivalence` bit-widths small
        (e.g. `u4`–`u5`) so Z3 finishes in seconds—never paper over slow tests
        or flakes by bumping timeouts or adding retries.
    *   **Testbench X-prop:** `ModuleTestbench` intentionally drives `'x` when
        `valid` is deasserted to verify X-propagation; do not weaken this in
        tests.
*   **Comments:** Add concise comments explaining *why* non-obvious invariants
    exist, and keep transient synthesis/benchmark tables in commit messages
    rather than source or `BUILD` comments (using 50 seeds for seeded synthesis
    comparisons).
