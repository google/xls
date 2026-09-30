# Module-Scoped `configured_values` in DSLX

Document contact: [allight](https://github.com/allight)

Date: 2026-09-22

Relevant issues: [#4995](https://github.com/google/xls/issues/4995)

[TOC]

## Overview

DSLX provides the built-in function `configured_value_or<T>("key", default)` to
allow build-time configuration overrides of constants in hardware designs and
tests. Currently, `configured_values` can only be specified on terminal build
targets (`xls_dslx_ir`, `xls_dslx_opt_ir`, and `xls_dslx_test`) and are applied
exclusively to the top-level entry `Module` during parsing and typechecking. Any
`configured_value_or` calls located inside imported `xls_dslx_library` modules
silently ignore the configured overrides and fall back to their compile
defaults.

This document describes **Proposal 2 (Target/Module-Scoped
`configured_values`)** from [#4995](https://github.com/google/xls/issues/4995),
which introduces target-scoped `configured_values` on `xls_dslx_library`
targets, propagates module-scoped overrides through `DslxInfo` providers and CLI
flags (`key@mod_a+mod_b:value`), and adds a default-enabled compiler warning
(`unused_configured_value`) to catch misspelled or unreferenced configuration
keys across single- and multi-module scopes.

## Background & Problem Statement

### How `configured_value_or` Works Today

The DSLX built-in function has the signature:

```dslx-snippet
fn configured_value_or<T: type, N: u32>(label: u8[N], default_value: T) -> T;
```

During constant collection and type deduction (in
`NoteBuiltinInvocationConstExpr` within `xls/dslx/type_system/deduce_utils.cc`),
the compiler inspects `invocation->owner()->configured_values()`, which is an
`absl::flat_hash_map<std::string, std::string>` stored on the enclosing AST
`Module` (`xls/dslx/frontend/module.h`):

1.  If `key` is present in `invocation->owner()->configured_values()`, the
    compiler parses the associated string override into an `InterpValue` of the
    explicit parametric type `T` via `GetConfiguredValueAsInterpValue` and
    records it in `TypeInfo::NoteConstExpr`.

2.  If `key` is absent from `invocation->owner()->configured_values()`, the
    compiler evaluates `default_value` and binds the result to `invocation`.

During bytecode emission (`xls/dslx/bytecode/bytecode_emitter.cc`) and IR
conversion (`xls/dslx/ir_convert/function_converter.cc`), the resolved constant
expression is emitted directly from `TypeInfo`.

### Why Imported Library Modules Silently Fall Back to Defaults

Today, `configured_values` is only wired to the top-level entry module:

*   **Build Rules**: Only `xls_dslx_ir`, `xls_dslx_opt_ir`, and `xls_dslx_test`
    accept a `configured_values` dictionary attribute. `xls_dslx_library` does
    not accept `configured_values`, and `DslxInfo`
    (`xls/build_rules/xls_providers.bzl`) does not carry any configuration
    metadata to downstream dependents.

*   **Top-Level Entry Point**: `ParseAndTypecheck`
    (`xls/dslx/parse_and_typecheck.cc`) and `ConvertFilesToPackage`
    (`xls/dslx/ir_convert/ir_converter.cc`) call
    `module->SetConfiguredValues(options.configured_values)` solely on the entry
    `Module` parsed from the primary input file.

*   **Imported Modules (`DoImport`)**: When the entry module imports a
    dependency (`import foo.bar;`), `DoImport` and `DslxPathToModuleInfo`
    (`xls/dslx/import_routines.cc`) instantiate a fresh `Module` object and
    invoke `ftypecheck` without populating `module->SetConfiguredValues(...)`.
    Because `ImportData` (`xls/dslx/import_data.h`) does not store or dispatch
    configured values to imported modules,
    `invocation->owner()->configured_values()` is always empty for every
    imported module.

Consequently, when a hardware designer extracts shared constants or
parameterized IP blocks into an `xls_dslx_library`, all `configured_value_or`
calls inside that library silently evaluate to `default_value` during both
`xls_dslx_test` execution and `xls_dslx_ir` synthesis.

## Alternatives Considered

### Proposal 1: Global Propagation via `ImportData` (Rejected)

An initial prototype ([PR #4985](https://github.com/google/xls/pull/4985))
stored a single flat list of `configured_values` on `ImportData` and copied the
entire map into every `Module` instantiated inside `DslxPathToModuleInfo`.

While simple to implement, Proposal 1 was rejected for the following reasons:

*   **Implicit Global Blast Radius**: Any key specified on a top-level
    `xls_dslx_ir` or `xls_dslx_test` target globally mutates every transitive
    dependency in the import DAG that happens to query the same string key.

*   **Key Collisions Across Transitive Dependencies**: Two unrelated libraries
    in a large design (for example, `pcie_phy` and `noc_router`) cannot both use
    a common key name like `"fifo_depth"` or `"use_fast_mode"` without a single
    top-level override unintentionally overwriting both modules simultaneously.

*   **Action-at-a-Distance & Fragile Caching**: Library authors have no
    visibility in `BUILD` files into which libraries are meant to be
    parameterized by `configured_values`, and `xls_dslx_library` targets cannot
    validate or encapsulate their own configuration parameters during standalone
    `parse_and_typecheck_dslx_main` actions.

### Proposal 3: Source-Level Parameterized Module Imports (Deferred)

Proposal 3 envisions first-class DSLX language syntax for parameterizing modules
at the `import` statement site (similar to SystemVerilog parameterized
interfaces or ML functors):

```dslx-snippet
import foo.bar with { FIFO_DEPTH: u32:16 };
```

While source-level module functors are expressive and enable multiple
instantiations of the same module with distinct parameters within a single
translation unit, they require substantial changes to the DSLX grammar, AST,
`ImportTokens` cache keys in `ImportData`, and IR name mangling. Furthermore,
they do not directly solve the build-system requirement of injecting
synthesis-time overrides from Bazel `BUILD` targets without editing `.x` source
files. Proposal 2 solves the build-system configuration requirement cleanly
today while remaining orthogonal to future language-level module functors.

## Detailed Design: Proposal 2 (Target/Module-Scoped `configured_values`)

Proposal 2 scopes `configured_values` to the `xls_dslx_library` (or terminal
target) where they are declared, propagates them transitively via `DslxInfo`,
and binds each key exclusively to the DSLX module(s) belonging to that target's
scope.

### 1. Build Rules & Starlark Propagation

#### `xls_dslx_library` Attribute and Typechecking

We add `"configured_values": attr.string_dict(...)` to `_xls_dslx_library_attrs`
in `xls/build_rules/xls_dslx_rules.bzl`:

```python
_xls_dslx_library_attrs = {
    "configured_values": attr.string_dict(
        doc = "Dictionary of overrides to use for overridable constants " +
              "in DSLX processing for the modules in this library. " +
              "Format is \"key\":\"value\" pairs.",
    ),
    ...
}
```

When `_xls_dslx_library_impl` invokes `parse_and_typecheck_dslx_main`
(`_xls_dslx_parse_and_typecheck_tool`) on each file in `ctx.files.srcs`, it
passes the merged `--configured_values` flag containing both the library's own
scoped entries and those inherited from `ctx.attr.deps`. This guarantees that
library-level typechecking validates the configured override strings (including
type compatibility and enum resolution) at the library build step.

#### `DslxInfo` Provider Extension

We extend `DslxInfo` in `xls/build_rules/xls_providers.bzl` with a new
`configured_values` depset field:

```python
DslxInfo = provider(
    doc = "...",
    fields = {
        "configured_values": "Depset: A depset of scoped configured_value " +
                             "strings ('key@mod_a+mod_b:value') for this " +
                             "target and its transitive xls_dslx_library " +
                             "dependencies.",
        "dslx_placeholder_files": "...",
        "dslx_source_files": "...",
        "target_dslx_source_files": "...",
    },
)
```

For an `xls_dslx_library` target with `srcs = [f_1, ..., f_n]` and
`configured_values = {"k": "v"}`, Starlark computes the normalized DSLX module
identifier `mod_i` for each file `f_i` in `srcs` and formats each entry as:

```
k@mod_1+mod_2+...+mod_n:v
```

These formatted entries are placed in the `direct` list of a `depset` whose
`transitive` children are `[dep[DslxInfo].configured_values for dep in
ctx.attr.deps]`.

#### Propagation to Consumer Rules

All downstream DSLX rules that consume `xls_dslx_library` targets via `deps` or
`library` (`xls_dslx_library`, `xls_dslx_test`, `xls_dslx_ir`, and
`xls_dslx_opt_ir`) collect the transitive `DslxInfo.configured_values` depset
alongside their own direct `ctx.attr.configured_values` attribute and pass the
combined list via `--configured_values=...` to `parse_and_typecheck_dslx_main`,
`interpreter_main`, and `ir_converter_main`.

### 2. CLI Flag Encoding

To maintain full backwards compatibility with existing CLI invocations while
supporting module-scoped and multi-file library rules, `--configured_values`
accepts comma-separated entries in two formats:

<!-- mdformat off(reason: preserving standard GFM table for mkdocs and g3mark) -->
| Format | Scope | Semantics |
| :--- | :--- | :--- |
| `key:value` | Unscoped (Entry Module) | Applies only to the top-level entry module being compiled or tested. Preserves 100% backwards compatibility with existing CLI callers and direct `configured_values` attributes. |
| `key@mod_1+...+mod_n:value` | Module-Scoped | Applies to any module in the `+`-delimited scope set `{mod_1, ..., mod_n}`. Used by `xls_dslx_library` to scope a configuration override to the `.x` files in its `srcs`. |
<!-- mdformat on -->

Grouping a multi-file `xls_dslx_library`'s modules into a single
`key@mod_1+...+mod_n:value` entry (rather than emitting separate
`key@mod_i:value` entries per file) is required for accurate
`unused_configured_value` warning semantics: when a library has `srcs =
["mod_a.x", "mod_b.x"]`, often only one module (`mod_a.x`) calls
`configured_value_or` while `mod_b.x` is a helper or wrapper module in the same
target. Encoding the library's `srcs` as a single `+`-delimited group allows the
compiler to verify that *at least one* module in the library consumes `key`,
avoiding false-positive `unused_configured_value` warnings on `mod_b.x`.

Note that the separator between the scope specification and the value is the
**first** `:` after the `key@scope` prefix; the value itself may contain colons
(such as DSLX typed literals `u32:42`, `s32:-100`, or enum colon-refs
`MyEnum::C`). Specifically:

*   We split on the first `:` to separate `<lhs>` (`key` or
    `key@mod_1+...+mod_n`) from `<rhs>` (`value`, e.g. `MyEnum::C`).

*   If `<lhs>` contains `@`, we split `<lhs>` on `@` into `key` and the
    `+`-separated list of target module identifiers `mod_1, ..., mod_n`.

*   If `<lhs>` does not contain `@`, `key` is bound to the unscoped entry-module
    configuration map.

### 3. Module Identifier Normalization

A critical subtlety in Bazel and monorepo builds is ensuring that the module
identifier emitted by Starlark in `key@<module_id>:value` matches the module
identifier known to C++ during both `ParseAndTypecheck` (when a file is the
top-level entry module) and `DoImport` (when a file is imported via `import
...;`).

#### How C++ Names Modules

1.  **When Imported (`DoImport` in `import_routines.cc`)**:

    *   `ImportTokens` holds the dotted components of the `import` statement
        (for example, `xls.examples.foo` or `xls.examples.foo`), and
        `module->name()` is set to `subject.ToString()`.

    *   `dslx_path.source_path` holds the resolved relative path to the `.x`
        file (for example, `xls/examples/foo.x`).

2.  **When Compiled as the Entry Module (`ParseAndTypecheck`)**:

    *   `PathToName(entry_module_path)` derives `module_name` from the stem of
        the filename (for example, `"foo"`), while `entry_module_path` holds the
        relative path passed on the CLI (for example, `xls/examples/foo.x`).

#### Canonical Module Identifier Matching

To ensure consistent matching across Starlark, generated build outputs, external
workspaces, and standalone CLI usage without any build-system-specific or
repository-specific hardcoded path filters:

1.  **Starlark Normalization (`get_dslx_module_scope_for_file`)**: In Starlark,
    for each `File` `f` in `ctx.files.srcs`, we derive its search-path-relative
    dotted module identifier by taking `f.path`, stripping the file's search
    path roots (`f.root.path` and `f.owner.workspace_root`), removing the `.x`
    suffix, stripping leading `./` or `/`, and replacing `/` with `.` (yielding
    e.g. `xls.examples.foo`). All corresponding search roots
    (`ctx.genfiles_dir.path`, `ctx.bin_dir.path`, `f.root.path`, and
    `f.owner.workspace_root`) are passed via `--dslx_path`.

2.  **C++ Search-Path-Based Matching (`ImportData::ModuleMatchesScope`)**: In
    C++, `ImportData::ModuleMatchesScope(std::string_view scope_id,
    std::string_view module_name, const std::filesystem::path& module_path)`
    normalizes `module_path` strictly against the configured search paths:

    *   Strips any matching search path prefix (`additional_search_paths()`,
        `stdlib_path()`, or the VFS current working directory) from
        `module_path`.

    *   Converts the search-path-relative path (with `.x` removed and `/`
        replaced by `.`) to a dotted module identifier.

    *   Matches if the normalized `scope_id` equals the normalized
        `module_name`, equals the dotted `module_path`, or equals the
        search-path-relative dotted `module_path`.

### 4. Precedence, Diamond Deduplication, and Conflict Detection

In realistic build graphs, a library may be reached along multiple dependency
paths (diamond dependencies), or a terminal target (`xls_dslx_test`,
`xls_dslx_ir`) with `library = ":my_lib"` may wish to override a default
configured value set on `:my_lib`.

We enforce the following deterministic resolution rules:

1.  **Diamond Deduplication (Identical Values)**: Because
    `DslxInfo.configured_values` uses a Bazel `depset`, identical strings from
    diamond dependencies are automatically deduplicated in Starlark. In C++, if
    the same `(scope, key)` pair is provided multiple times with the **exact
    same value**, it is silently deduplicated.

2.  **Conflict Detection Across Scoped Entries (Different Values)**: If two
    scoped entries assign **different values** to the same `key` for an
    overlapping module `M` (for example, `key@M:val1` and `key@M:val2`), parser
    initialization immediately fails with an `absl::InvalidArgumentError`
    reporting the conflicting key, module, and values.

3.  **Consumer Target Precedence Over `library = ...`**: When a consumer target
    (`xls_dslx_test`, `xls_dslx_ir`, or `xls_dslx_opt_ir`) specifies `library =
    ":my_lib"` **and** also specifies its own `configured_values = {"key":
    "override_val"}`:

    *   If `"key"` in the consumer's `configured_values` is unscoped, it applies
        to the entry module(s) being tested or converted and **takes precedence
        over** (overrides) any `@`-scoped value for `"key"` inherited from
        `library[DslxInfo].configured_values` on that entry module.

    *   Furthermore, overriding a library's `@`-scoped key via an unscoped entry
        on the entry module marks both the unscoped entry and the library's
        scoped group as **used** so that no false-positive unused warning is
        emitted.

    *   If the consumer explicitly specifies a scoped key `"key@other_mod":
        "override_val"` in `ctx.attr.configured_values`, Starlark filters out or
        supersedes any inherited `DslxInfo.configured_values` entry for the same
        `(key, scope)` before constructing the CLI arguments.

## Detailed Design: Unused `configured_value` Warning (`unused_configured_value`)

### 1. Motivation

Because `configured_values` keys are arbitrary strings matched against string
literals in `configured_value_or<T>("key", default)`, a typo in a `BUILD` file
or CLI flag (for example, `configured_values = {"fifi_depth": "u32:16"}` instead
of `"fifo_depth"`) previously went completely unnoticed: the module silently
fell back to `default` and synthesized hardware with the wrong parameter.
Similarly, refactoring DSLX code to remove a `configured_value_or` call left
dead `configured_values` entries in `BUILD` files.

To prevent silent misconfigurations, we introduce a new compiler warning,
`unused_configured_value`, enabled by default in `kDefaultWarningsSet`.

### 2. `WarningKind::kUnusedConfiguredValue` and Bitset Widening

In `xls/dslx/warning_kind.h`:

*   `WarningKindInt` is currently `uint16_t` with `kWarningKindCount = 15` (bits
    `0..14`). Adding a 16th warning (`1 << 15`) would overflow signed/16-bit
    shift expressions such as `(WarningKindInt{1} << kWarningKindCount) - 1`
    when `kWarningKindCount == 16`.

*   We widen `WarningKindInt` from `uint16_t` to `uint32_t`:

    ```cpp
    using WarningKindInt = uint32_t;

    enum class WarningKind : WarningKindInt {
      ...
      kWidthSliceOutOfRange = 1 << 14,
      kUnusedConfiguredValue = 1 << 15,
    };
    constexpr WarningKindInt kWarningKindCount = 16;
    ```

*   We map `WarningKind::kUnusedConfiguredValue` to the flag name
    `"unused_configured_value"` in `xls/dslx/warning_kind.cc`.

*   Because `kDefaultWarningsSet` is constructed from `kAllWarningsSet` minus
    explicit opt-out warnings (`kShouldUseAssert` and
    `kAlreadyExhaustiveMatch`), `WarningKind::kUnusedConfiguredValue` is
    **enabled by default** (and treated as an error when
    `--warnings_as_errors=true`, which is the default in XLS tools and tests).
    Users can explicitly suppress it if needed via
    `--disable_warnings=unused_configured_value`.

### 3. Usage Tracking and Multi-File Scope Verification

Consider an `xls_dslx_library` with multiple source files:

```python
xls_dslx_library(
    name = "multi_file_lib",
    srcs = [
        "mod_a.x",
        "mod_b.x",
    ],
    configured_values = {
        "shared_param": "u32:64",
    },
)
```

Here, `"shared_param"` is encoded as `shared_param@pkg.mod_a+pkg.mod_b:u32:64`.
Frequently, only `mod_a.x` calls `configured_value_or<u32>("shared_param",
...)`, while `mod_b.x` is a helper or test module in the same library that
imports `mod_a`.

If we naively warned whenever *any* single module in `{mod_a, mod_b}` failed to
call `configured_value_or("shared_param", ...)`, then compiling or testing
`mod_b.x` would trigger a false-positive `unused_configured_value` warning if
`mod_b.x` did not touch `mod_a`, or if each module were checked in isolation!
Conversely, if *neither* `mod_a.x` nor `mod_b.x` calls
`configured_value_or("shared_param", ...)`, we **must** emit a warning.

We solve this across both C++ and Starlark as follows:

#### Data Structures in `ImportData` and `Module`

1.  **`ConfiguredValueGroup` in `ImportData`**: `ImportData` stores the parsed
    configuration rules as a list of `ConfiguredValueGroup` records:

    ```cpp
    struct ConfiguredValueGroup {
      std::string key;
      std::string value;
      // Empty vector indicates an unscoped entry (applies to the entry module).
      // Non-empty vector indicates a '+'-delimited module scope set.
      std::vector<std::string> scope_modules;
      bool used = false;
    };
    ```

2.  **Populating Each `Module`**: When the entry module is initialized in
    `ParseAndTypecheck` (or `ConvertFilesToPackage`) and when any imported
    module is initialized in `DslxPathToModuleInfo` (`import_routines.cc`),
    `ImportData` resolves all applicable `(key, value)` pairs for that module
    (applying entry-module precedence over scoped entries), records the
    `(module_name, key)` mapping for usage tracking, and populates
    `module->SetConfiguredValuesMap(...)`.

3.  **Recording Key Usage During Pre-Typecheck Analysis and Type Deduction**:

    *   **Pre-Typecheck Analysis (Uninstantiated Parametrics)**: A library
        module may define `pub fn p<N: u32>() -> u32 {
        configured_value_or<u32>("key", 0) }` or a parametric proc/impl without
        instantiating `p` inside the library itself. Because
        `NoteBuiltinInvocationConstExpr` only executes when a function body is
        concretely instantiated, relying solely on type deduction would emit a
        false-positive `unused_configured_value` warning when building the
        `xls_dslx_library` target. Therefore, during
        `SemanticsAnalysis::RunPreTypeCheckPass` (`semantics_analysis.cc`), the
        compiler walks `Invocation` AST nodes in the module whose callee is
        `configured_value_or` and extracts string literal keys to mark them as
        used on `ImportData`.

    *   **Type Deduction (`NoteBuiltinInvocationConstExpr` in
        `deduce_utils.cc`)**: Whenever `configured_value_or` looks up `key` in
        `invocation->owner()`, `import_data->NoteConfiguredValueUsed(...)` marks
        any matching `ConfiguredValueGroup` (both unscoped entry overrides and
        any `@`-scoped group whose `scope_modules` contains the module) as
        `used = true`.

4.  **Propagating `configured_values` Across `interpreter_main.cc` Modes**: In
    `xls/dslx/interpreter_main.cc`, `--configured_values` must be propagated not
    only to `parse_and_typecheck_options.configured_values`, but also to
    `options.convert_options.configured_values` (used when `--compare=jit` or
    `--compare=interpreter` converts modules to IR for comparison) and
    `ir_convert_options.configured_values` (used when `--lower_to_ir` validates
    IR lowering).

5.  **Verifying Scope Satisfaction After Typechecking**: At the completion of
    top-level `ParseAndTypecheck`:

    *   **Unscoped Entries (`scope_modules.empty()`)**: Must have `used == true`
        (i.e., used by the entry module). If `used == false`, a
        `WarningKind::kUnusedConfiguredValue` warning is added to the entry
        module's `WarningCollector` at a synthetic file-start `Span`
        (`Pos(fileno, 0, 0)`):

        ```
        Configured value `foo` (with value `u32:42`) was provided but not used in module `my_entry`.
        ```

    *   **Scoped Entries (`!scope_modules.empty()`)**: A scoped group
        `key@mod_1+...+mod_n:value` is considered active in a compilation unit
        if **at least one** module in `{mod_1, ..., mod_n}` was loaded during
        the compilation session (either as the entry module or via
        `ImportData`).

        *   For single-module scopes (`n == 1`), if `mod_1` was loaded (as the
            entry module or via import) and `used == false`, a
            `WarningKind::kUnusedConfiguredValue` warning is emitted.

        *   For multi-module scopes (`n > 1`), if **all** modules in `{mod_1,
            ..., mod_n}` were loaded in the session (or when
            `parse_and_typecheck_dslx_main` is invoked across all `srcs` of the
            `xls_dslx_library`) and `used == false` across the entire scope set,
            a `WarningKind::kUnusedConfiguredValue` warning is emitted:

            ```
            Configured value `shared_param` (with value `u32:64`) scoped to `pkg.mod_a+pkg.mod_b` was not used by any module in its scope.
            ```

        *   In `_xls_dslx_library_impl`, `parse_and_typecheck_dslx_main` is
            passed all `srcs` of the library in a single invocation so that
            every file in `srcs` is parsed and typechecked into the shared
            `ImportData` session before checking unused scoped
            `configured_values`. This guarantees that a key declared on a
            multi-file `xls_dslx_library` warns if and only if **none** of the
            files in that library's `srcs` use the key.

## Summary of Files Modified

<!-- mdformat off(reason: preserving standard GFM table for mkdocs and g3mark) -->
| File Path | Summary of Changes |
| :--- | :--- |
| `xls/dslx/warning_kind.h` | Widen `WarningKindInt` to `uint32_t`, add `WarningKind::kUnusedConfiguredValue = 1 << 15`, increment `kWarningKindCount` to 16. |
| `xls/dslx/warning_kind.cc` | Register `"unused_configured_value"` string mapping in `WarningKindToString` and `WarningKindFromString`. |
| `xls/dslx/frontend/module.h` | Replace `SetConfiguredValues` with `SetConfiguredValuesMap` accepting the resolved per-module map from `ImportData`. |
| `xls/dslx/frontend/semantics_analysis.cc` | Record `configured_value_or` key usage during `SemanticsAnalysis::RunPreTypeCheckPass` (including inside uninstantiated parametrics). |
| `xls/dslx/import_data.h` / `.cc` | Store `ConfiguredValueGroup` records on `ImportData`, resolve per-module configured values with conflict detection and entry-module override precedence, track usage across scopes, and check unused values. |
| `xls/dslx/import_routines.cc` | Populate `module->SetConfiguredValuesMap(...)` in imported modules inside `DslxPathToModuleInfo` using `import_data`. |
| `xls/dslx/parse_and_typecheck.cc` | Register `options.configured_values` on `import_data`, populate entry module configured values, preserve configured values across `TestFunctionTransformer` re-parses, and emit `WarningKind::kUnusedConfiguredValue` diagnostics. |
| `xls/dslx/parse_and_typecheck_dslx_main.cc` | Add `--configured_values` flag and support typechecking multiple input files in a shared `ImportData` session for multi-file `xls_dslx_library` verification. |
| `xls/dslx/interpreter_main.cc` | Propagate `configured_values` to `options.convert_options` (for `--compare=jit`) and `ir_convert_options` (for `--lower_to_ir`). |
| `xls/dslx/type_system/deduce_utils.cc` | Call `NoteConfiguredValueUsed` on `ImportData` when `configured_value_or` references a key. |
| `xls/build_rules/xls_providers.bzl` | Add `configured_values` depset field to `DslxInfo`. |
| `xls/build_rules/xls_common_rules.bzl` | Add `collect_dslx_configured_values` helper to collect transitive `@`-scoped entries and direct overrides. |
| `xls/build_rules/xls_dslx_rules.bzl` | Add `configured_values` attribute to `xls_dslx_library`, compute module-scoped `key@mod_a+mod_b:value` entries via `_format_scoped_configured_values`, propagate `DslxInfo.configured_values`, and pass `--configured_values` to `parse_and_typecheck_dslx_main`. |
| `xls/build_rules/xls_dslx_test.bzl` | Collect transitive `DslxInfo.configured_values` from `library` / `deps` / `dep`, merge with direct `ctx.attr.configured_values`, and pass to `interpreter_main`. |
| `xls/build_rules/xls_ir_rules.bzl` | Collect transitive `DslxInfo.configured_values` in `_convert_to_ir`, merge with direct `ctx.attr.configured_values`, and pass to `ir_converter_main`. |
<!-- mdformat on -->

## Testing & Rollout Plan

1.  **Unit Tests (C++ Type System, Bytecode, & IR Converter)**:

    *   Verify unscoped `key:value` applies only to the entry module and does
        **not** leak into imported modules (`typecheck_module_test.cc`,
        `ir_converter_test.cc`).

    *   Verify scoped `key@import.path:value` and `key@mod_a+mod_b:value` apply
        to matching imported modules and do not affect unrelated modules.

    *   Verify conflict detection raises an `INVALID_ARGUMENT` error when two
        scoped rules specify different values for the same key and module.

    *   Verify unscoped entry-module override takes precedence over an
        `@`-scoped rule targeting the entry module.

    *   Verify `WarningKind::kUnusedConfiguredValue` triggers for unused
        unscoped and unused scoped keys, does not trigger when at least one
        module in a `mod_a+mod_b` scope consumes the key, and can be suppressed
        via `--disable_warnings=unused_configured_value`.

2.  **Build Rule Integration Tests (`xls/examples/` & `build_rules/tests/`)**:

    *   Add `xls_dslx_library` targets with `configured_values` (both
        single-file and multi-file `srcs`).

    *   Add `xls_dslx_test`, `xls_dslx_ir`, and `xls_dslx_opt_ir` targets that
        depend on configured `xls_dslx_library` targets (including diamond
        dependencies and consumer overrides via `library = ...`).
