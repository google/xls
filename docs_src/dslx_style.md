# DSLX Style Guide

[TOC]

## Introduction

This guide defines the standard style for DSLX (`.x`) code in the XLS repository
and provides recommended conventions for external projects. Derived from
Google's Rust Style Guide, it optimizes for long-term readability, uniform
automated formatting, and unambiguous hardware semantics.

DSLX captures behavioral hardware intent rather than structural netlists. Write
code so a reviewer can verify bit-level behavior, protocol invariants, and
synthesized datapath structures without mentally simulating wires.

Three core principles guide these rules:

*   **Readability over writeability**: Optimize for reviewers reading the code
    months later rather than minimizing keystrokes while authoring.
*   **Consistent formatting and automation**: Rely on `dslx_fmt` and compiler
    warnings rather than ad-hoc formatting or local conventions so diffs stay
    clean and issues fail early.
*   **Hardware intent over RTL transliteration**: Model algebraic operations,
    state transitions, and transactions directly instead of mirroring legacy
    SystemVerilog constructs line by line.

Unless qualified with *prefer*, *should*, or *avoid*, rules in this guide are
required for new DSLX code. See also the [DSLX Reference](dslx_reference.md),
the [DSLX Standard Library](dslx_std.md), and the
[XLS Style Guide](xls_style.md).

## License and File Headers

Every `.x` file in the XLS Github repository must begin with the Apache 2.0
license header attributed to `The XLS Authors` (replacing the year with the
current year for new files):

```dslx-snippet
// Copyright 2026 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
```

External projects should use their own repository's license header.

## Formatting and Tooling

### Formatter (`dslx_fmt`)

*   **Format all `.x` files with `dslx_fmt`** (for example,
    `dslx_fmt -i <file>.x`) and enforce formatting in CI with
    `xls_dslx_fmt_test` so diffs remain uniform and free of whitespace noise.
*   **Run `dslx_fmt --simplify_type_annotations -i`** periodically or after
    refactors to strip redundant type annotations on literals when inferred from
    context (for example, `const Y: u32 = u32:42;` simplifies to
    `const Y: u32 = 42;`).
*   **Accept `dslx_fmt` layout defaults** (4-space indentation and a 100-column
    line limit) rather than fighting wrapping choices.
*   **Reserve formatter opt-outs for aligned tables**: Use
    `#[dslx_format_disable]` to exempt an entire function, or `// dslx-fmt::off`
    and `// dslx-fmt::on` (each on its own standalone line with no trailing
    text) to exempt `const` lookup tables, `struct`s, or inline regions where
    columnar alignment materially helps verification. Include a brief comment
    explaining why alignment is needed.

### Numeric Literals

*   **Use lowercase hexadecimal digits** (`0xdead_beef`, not `0xDEAD_BEEF`).
    Matching the lowercase `0x` prefix avoids jarring mixed case (`0xDEAD`) and
    prevents visual collisions with `SCREAMING_SNAKE_CASE` constants.
*   **Group long binary and hexadecimal literals with underscores** (`_`) at
    natural byte, nibble, or word boundaries (`0b1010_0110`, `0xffff_0001`) so
    digit counts are easy to verify.
*   **Omit redundant type prefixes on literals** when the expected type is clear
    from context (`x + 1`). Include the type prefix whenever bit width or
    signedness would otherwise be ambiguous (`u32:0`, `s8:-1`, `uN[W]:0`).

## Naming Conventions

Consistent capitalization indicates whether an identifier names a type, a
compile-time constant, an enum variant, or a runtime value without relying on
IDE hover hints.

<!-- mdformat off(multiline table cells not supported in mkdocs) -->

| Construct | Convention | Notes and Examples |
| :--- | :--- | :--- |
| Source files and modules | `snake_case` | Valid identifier (`abs_diff.x`, never `abs-diff.x`; `illegal_package_name`). |
| Functions, variables, and parameters | `snake_case` | `decode_header`, `next_state`. Reserve trailing `_test` for `#[test]` functions (`misleading_function_name`). |
| Struct fields and proc members | `snake_case` | `payload_len`, `in_ch` (`member_naming`). |
| Constants | `SCREAMING_SNAKE_CASE` | `MAX_WINDOW_BITS` (`constant_naming`). |
| Enum variants | `SCREAMING_SNAKE_CASE` | `OpCode::ADD`, `Status::OK` (matches constants; distinct from types). |
| Structs, enums, and type aliases | `PascalCase` | `DecodeResult`, `WordIdx`. Avoid C-style `_t` suffixes. |
| Procs | `PascalCase` | `Accumulator`, `AxiMaster` (`impl`-style procs define types). |
| Numeric and boolean parametrics | `SCREAMING_SNAKE_CASE` | `<WIDTH: u32, SIGNED: bool>`. |
| Type parametrics | `PascalCase` | `<T: type, StateT: type>` (requires `#![feature(generics)]`). |

<!-- mdformat on -->

### Prime Identifiers, Unused Bindings, and Keywords

*   Use descriptive `snake_case` suffixes (`x_next`, `valid_d1`) rather than
    prime/tick identifiers (`x'`, `x''`) in interfaces and persistent state.
    Ticks are experimental non-Rust syntax, require mangling in generated
    RTL/FFI symbols, and scale poorly across multi-stage pipelines (`_d1`,
    `_d2`); reserve them for short local mathematical derivations where equation
    symmetry helps readability.
*   Use `_` or prefix intentionally ignored parameters and destructured tuple
    elements with `_` (`_unused`) to document intent. Do not write
    `let _ = expr;` as a standalone statement; evaluate `expr;` directly to
    avoid `useless_let_binding`.
*   Avoid Verilog and SystemVerilog keywords (`module`, `reg`, `wire`, `logic`,
    `input`, `output`) for functions, parameters, struct fields, and exported
    identifiers so generated RTL stays readable without mangling (checked by the
    opt-in `verilog_keyword_name` warning).

### Interoperability Exceptions

Generated type bindings or strict pin-parity wrappers for external hardware
interfaces may retain upstream `_t` type suffixes or `k`-prefixed constant names
when renaming would break 1:1 parity audits. Handwritten DSLX modules must
follow the standard conventions above.

## Modules, Imports, and Visibility

### Imports

Place `import` statements at the top of the file (below the license header,
file-level attributes, and `//!` module doc comment):

1.  Standard library imports (`import apfloat;`, `import std;`), sorted
    alphabetically.
2.  Project or package imports, sorted alphabetically and separated from
    standard library imports by a blank line.

DSLX namespaces every imported symbol (`mod::symbol`), which eliminates
ambiguous unqualified calls. For example:

```dslx-snippet
import bfloat16;
import std;

import my_project.common;
import my_project.very_long_decoder_name as decoder;
```

*   Use `as` aliases only to shorten unwieldy module names or resolve name
    collisions.
*   Avoid `#![feature(use_syntax)]` unless your project has explicitly adopted
    it; standard `import` keeps symbol origins clear at every call site.

### File Layout

Order top-level items cleanly within DSLX's definition-before-use constraint
(each constant, type, or function must be declared before items that reference
it):

1.  License header
2.  File-level attributes (`#![...]`, if needed) and `//!` module doc comment
3.  `import` declarations
4.  Module-level `const` definitions
5.  Public `struct`, `enum`, and type-alias definitions (with their `impl`
    blocks)
6.  Private helper types and functions, ordered before the public `fn`s and
    `proc`s that call them
7.  Public `fn`s and `proc`s (with their `impl` blocks)
8.  In-file unit, QuickCheck, fuzz, and proc tests

### Visibility (`pub`)

Keep items private by default. Mark only the intended module interface (`pub`
types, top-level entry points, and shared constants) public so callers and
maintainers know which APIs external packages depend on.

## Documentation and Comments

Use Rust-style doc comments (`//!` and `///`), which `dslx_fmt` preserves so
documentation tools and IDE hover help distinguish API contracts from
implementation notes:

*   **`//!` (module doc comments)**: Explain the module's overall purpose,
    protocol assumptions, and invariants at the top of non-trivial files.
*   **`///` (item doc comments)**: Document every `pub` type, function, proc,
    method, and non-obvious constant in Markdown. Explain *what* the item
    models, field units or encodings, and caller preconditions rather than
    restating the signature.
*   **`//` (implementation comments)**: Explain *why* a bit trick, pipeline
    split, or area- or timing-motivated (PPA) restructuring is necessary.
*   **`TODO` comments**: Use `// TODO <issue-url> - <description>` or state a
    clear exit condition on every `// TODO` comment so stale workarounds can be
    traced and removed.
*   **No divider banners**: Avoid decorative comment banners (`////////`); use
    blank lines and well-factored helper functions instead.

## Types, Type Inference, and `impl` Blocks

### Boolean and Integer Types

*   **Use `bool` for single-bit truth values and flags**, not `u1` or `bits[1]`.
    Although `bool` and `u1` are type-equivalent in DSLX, `bool` clarifies
    intent for readers, eliminates `== u1:1` boilerplate in conditions, and
    pairs naturally with `x[i+:bool]`.
*   **Use `u1`..`u64` and `s1`..`s64`** for concrete numeric bit widths up to 64
    (reserving `u1` for 1-bit numeric values), and **`uN[W]` / `sN[W]`** for
    parametric widths or widths above 64 bits.
*   **Treat `bits[W]` as legacy.** Use `uN[W]` (or `u32`, `u8`, etc.) in new
    code and cleanups so unsigned semantics stay explicit in the type.
*   **Use explicit signed types (`s32`, `sN[W]`)** for signed values rather than
    repeatedly casting unsigned values before arithmetic shifts (`>>`) and
    signed comparisons.

### Structs, Enums, Type Aliases, and Tuples

*   **Prefer strongly typed `enum`s over raw bit constants** for opcodes, FSM
    states, and tags, using the narrowest matching underlying type
    (`enum Op : u2`). Unlike raw bit constants, `enum`s provide type safety,
    exhaustive `match` checking, and fuzzer generation restricted to defined
    variants.
*   **Prefer named `struct`s over tuples with more than three elements** or
    tuples with repeated types, because positional `.3` indexing invites silent
    field swaps. Small 2- or 3-element tuples (such as `(sum, carry)` or
    `(token, data)`) are idiomatic for immediate local returns.
*   **Use `#[derive(ToBits)]`** or explicit helper methods instead of manually
    concatenating (`++`) and slicing packed struct fields so serialization stays
    in sync when fields change.
*   **Use `impl` blocks for constructors and methods** (`zero`, `new`,
    `is_valid`) rather than standalone `foo_reset()` or `foo_is_valid()` free
    functions; `impl` co-locates behavior with the type, enables `Type::zero()`
    and `val.is_valid()` calls, and removes repetitive prefixes.
*   **Type aliases vs. newtype structs**: Use a type alias
    (`pub type Word = u32;`) when interchangeability is intended; wrap in a
    single-field `struct` (`pub struct CycleCount { value: u32 }`) when distinct
    units or invariants must not silently mix.

```dslx
//! Arithmetic decode unit illustrating idiomatic types and `impl` methods.

import std;

/// Operation performed by `eval_alu`.
pub enum OpCode : u2 {
    ADD = 0,
    SUB = 1,
    XOR = 2,
    PASS = 3,
}

/// Packed operands and control flags for `eval_alu`.
pub struct AluInput<WIDTH: u32> { op: OpCode, lhs: uN[WIDTH], rhs: uN[WIDTH], invert_rhs: bool }

impl AluInput<WIDTH> {
    /// Returns a zero-initialized `PASS` input.
    pub fn zero() -> Self {
        Self { op: OpCode::PASS, lhs: uN[WIDTH]:0, rhs: uN[WIDTH]:0, invert_rhs: false }
    }
}

/// Evaluates `input` and returns `(result, msb_is_set)`.
pub fn eval_alu<WIDTH: u32>(input: AluInput<WIDTH>) -> (uN[WIDTH], bool) {
    let rhs = if input.invert_rhs { !input.rhs } else { input.rhs };
    let res = match input.op {
        OpCode::ADD => input.lhs + rhs,
        OpCode::SUB => input.lhs - rhs,
        OpCode::XOR => input.lhs ^ rhs,
        OpCode::PASS => input.lhs,
    };
    (res, std::msb(res))
}

#[test]
fn test_eval_alu() {
    let base = AluInput<u32:8>::zero();
    assert_eq(eval_alu(base), (u8:0, false));
    let sub = AluInput { op: OpCode::SUB, lhs: u8:0, rhs: u8:1, invert_rhs: false };
    assert_eq(eval_alu(sub), (u8:0xff, true));
}
```

Avoid raw bitvector constants, `bits[W]`, `u1` booleans, manual MSB slices, and
wide anonymous tuples:

```dslx-bad
// Anti-pattern: legacy bits[W], u1 flags, magic constants, and wide tuple return.
fn bad_alu<W: u32>(op: bits[2], lhs: bits[W], rhs: bits[W], inv: u1) -> (bits[W], u1, u1, u1) {
    let r = if inv == u1:1 { !rhs } else { rhs };
    let out = if op == bits[2]:0 {
        lhs + r
    } else if op == bits[2]:1 {
        lhs - r
    } else {
        lhs
    };
    (out, out[(W - 1)+:u1] == u1:1, out == bits[W]:0, inv)
}
```

### Type Inference in `let` Bindings

Omit `let` type annotations when the type is clear from the expression on the
right-hand side (`let count = u32:0;`, `let decoded = parse_header(raw);`). Add
explicit annotations when bit widths, signedness, or parametric widening would
otherwise force readers to compute types mentally (`let sum: uN[W + 1] = ...;`).

## Expressions, Control Flow, and Assertions

### Bit Slicing and Built-In Helpers

*   **Prefer width-indexed slicing (`x[lo+:uN[W]]` and `x[i+:bool]`) over range
    slicing (`x[lo:hi]`)** when extracting fixed-width fields. Width slicing
    makes the extracted width explicit, prevents `hi - lo` off-by-one errors,
    and supports non-literal start indices.
*   **Use built-in and `std` helpers** instead of open-coding bit idioms:
    `std::msb(x)` and `std::lsb(x)` over manual bit slices (`x[(W - 1)+:bool]`,
    `x[0+:bool]`), `array_rev(x as u8[N])` over manual byte-swap concatenation,
    and `std::clog2` / `std::is_pow2` over ad-hoc math loops.

### Control Flow and Exhaustive `match`

*   **Prefer `match` over chained `if`/`else if`** when dispatching on `enum`s
    or bit patterns, and avoid catch-all `_` arms on `enum`s unless
    intentionally handling a subset. Omitting `_` ensures adding an `enum`
    variant triggers a compile error at every unhandled `match` rather than
    silently hitting a default arm. (The opt-in
    `--enable_warnings=already_exhaustive_match` warning flags redundant `_`
    arms after prior arms already cover every value.)
*   **Use `for` for uniform bounded loops**; use `unroll_for!` only when loop
    iterations differ in type or require compile-time parametric specialization
    that `for` cannot express.
*   **Return expression values directly from `if` and `match`** rather than
    shadowing a `let` binding across sequential statements; direct expression
    results keep dataflow dependencies clear.

### Assertions (`assert!` vs. `fail!`)

Prefer `assert!(cond, "descriptive_label")` over `if !cond { fail!(...); }`
(checked by the opt-in `--enable_warnings=should_use_assert` warning); reserve
`fail!` for unreachable expression branches that must return a typed fallback
value. Assertion labels must be valid Verilog identifiers—use concise, unique
names explaining the violated invariant (and `assert_fmt!` inside `unroll_for!`)
so simulation logs pinpoint failures cleanly. *(Note: `expected_fail_label` is
currently supported only on legacy `#[test_proc(expected_fail_label="...")]`
tests, not on `impl`-style `#[test]` procs; see `google/xls#4125`.)*

## Procs, Channels, and Hardware Intent

### Always Use `impl`-Style Procs

Write all new procs using modern **`impl`-style `proc` syntax** (`new` and
`next(self)`) rather than legacy `config`/`init`/`next` procs. `impl`-style
procs consolidate channel and state declarations cleanly, support helper
methods, and match modern `#[test]` proc syntax (stateful procs using `read` and
`write` require `#![feature(explicit_state_access)]`).

When migrating legacy procs with `dslx_fmt --convert_legacy_procs -i`, remove
`#![feature(generics)]` (and `#![feature(explicit_state_access)]` for stateless
procs) afterward if the proc does not need them.

### Channels and State Separation

*   **Model handshakes with native `chan<T>` channels** (`send`, `recv`,
    `send_if`, `recv_if`, `recv_non_blocking`) rather than embedding `.valid` or
    `.ready` fields inside payload structs. XLS codegen synthesizes flow-control
    signals directly from channel operations.
*   **Group coupled signals into a payload `struct` on one channel** instead of
    splitting synchronized fields across independent channels; grouping avoids
    redundant handshake logic and cross-channel deadlock hazards.
*   **Separate pure combinational logic from proc I/O and state updates.** Keep
    `next(self)` focused on `recv`, `send`, `read`, and `write`, while
    delegating computation and next-state transitions to pure `fn`s (or `impl`
    methods) that can be unit-tested and formally proven in isolation.

```dslx
#![feature(explicit_state_access)]

fn step_saturating(val: u8, inc: u8) -> u8 {
    let wide = (val as u9) + (inc as u9);
    if wide > u9:0xff { u8:0xff } else { wide as u8 }
}

/// Accumulates incoming unsigned bytes with saturation at `0xff`.
pub proc SaturatingAccum {
    in_ch: chan<u8> in,
    out_ch: chan<u8> out,
    acc: u8,
}

impl SaturatingAccum {
    /// Creates a `SaturatingAccum` initialized to `0`.
    pub fn new(in_ch: chan<u8> in, out_ch: chan<u8> out) -> Self {
        Self { in_ch, out_ch, acc: u8:0 }
    }

    fn next(self) {
        let acc = read(self.acc);
        let (tok, inc) = recv(join(), self.in_ch);
        let next_acc = step_saturating(acc, inc);
        send(tok, self.out_ch, next_acc);
        write(self.acc, next_acc);
    }
}

#[test]
fn test_step_saturating() { assert_eq(step_saturating(u8:250, u8:10), u8:255); }

#[test]
proc SaturatingAccumTest {
    terminator: chan<bool> out,
    in_s: chan<u8> out,
    out_r: chan<u8> in,
}

impl SaturatingAccumTest {
    fn new(terminator: chan<bool> out) -> Self {
        let (in_s, in_r) = chan<u8>("in");
        let (out_s, out_r) = chan<u8>("out");
        SaturatingAccum::new(in_r, out_s).spawn();
        Self { terminator, in_s, out_r }
    }

    fn next(self) {
        let tok = send(join(), self.in_s, u8:200);
        let (tok, v0) = recv(tok, self.out_r);
        assert_eq(v0, u8:200);
        let tok = send(tok, self.in_s, u8:100);
        let (tok, v1) = recv(tok, self.out_r);
        assert_eq(v1, u8:255);
        send(tok, self.terminator, true);
    }
}
```

## Attributes and Unstable Features

*   **Minimize `#![feature(...)]`**: Enable only features required by the module
    (such as `#![feature(explicit_state_access)]` for stateful `impl` procs, or
    `#![feature(channel_attributes)]` for `#[channel_strictness]` and
    `#[channel_flow_control]`). Unstable flags opt into evolving syntax or
    semantics that may change across compiler releases; remove unused flags
    immediately.
*   **Do not blanket-suppress naming warnings**: Avoid module-wide
    `#![allow(nonstandard_constant_naming)]` or
    `#![allow(nonstandard_member_naming)]` outside justified interoperability
    wrappers so real naming defects in handwritten code remain visible.
*   **Common item attributes**: Use `#[test]`, `#[quickcheck]`, `#[fuzz_test]`,
    and `#[fuzz_domain("...")]` for verification; `#[derive(ToBits)]` for bit
    serialization; `#[sv_type("...")]`, `#[extern_verilog("...")]`,
    `#[channel_strictness("...")]`, and `#[channel_flow_control("...")]` for RTL
    codegen contracts; and `#[dslx_format_disable]` only on functions where
    table alignment requires opting out of formatting.

## Testing and Fuzzing

**Place all tests in the same `.x` file as the code they verify** (`#[test]`
functions, `#[test]` procs, `#[quickcheck]` properties, and `#[fuzz_test]`
targets). Keeping tests alongside the implementation keeps invariants visible
during review and prevents test suites from drifting out of sync during
refactors. Synthesizing from a `dslx_top` entry point converts only the top
function or proc and its callees (and whole-module IR conversion omits `#[test]`
functions and procs), so unit tests add zero hardware area.

Select the verification mechanism suited to each invariant:

*   **`#[test]` functions**: Deterministic unit tests for edge cases, boundary
    overflows, and known golden vectors on pure helpers.
*   **`#[test]` procs**: End-to-end channel protocol verification using a
    `terminator: chan<bool> out` channel.
*   **`#[quickcheck]` (run via `xls_dslx_test`, or formally proved via
    `xls_dslx_prove_quickcheck_test`)**: Randomized or SMT-proved property
    checking for pure boolean predicates (`fn prop(...) -> bool`). Prefer formal
    proofs for algebraic identities and invariant preservation over small or
    moderate bit widths.
*   **`#[fuzz_test]` and `#[fuzz_domain("...")]` (with `dslx_fuzz_test`)**:
    Coverage-guided fuzzing over explicit input domains when formal proof does
    not scale. Use `#[fuzz_domain("DomainName")]` on structs paired with a
    domain builder function rather than unwieldy inline domain expressions, and
    **return `bool`** from `#[fuzz_test]` functions instead of calling `assert!`
    so the fuzzing framework can report counterexamples directly:

```dslx
#[fuzz_domain("RangePairDomain")]
struct RangePair { lo: u16, hi: u16 }

fn range_pair_domain() -> RangePairDomain { RangePairDomain { lo: u16:0..100, hi: u16:100..500 } }

#[fuzz_test(domains=`range_pair_domain()`)]
fn prop_ordered(p: RangePair) -> bool {
    p.lo < p.hi
}
```

See the [Bazel Rules and Macros](bazel_rules_macros.md) reference for
configuring `xls_dslx_test`, `xls_dslx_fmt_test`,
`xls_dslx_prove_quickcheck_test`, and `dslx_fuzz_test` targets.
