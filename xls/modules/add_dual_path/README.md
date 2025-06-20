# Dual-Path Floating-Point Adder

This directory contains a floating-point adder implemented in DSLX using a
classic **dual-path** algorithm. This particular algorithm just splits on
**exponent distance** as a predicate. It works on `APFloat` so the
implementation targets arbitrary exponent/fraction widths (parameterized via
generic parameters) and is validated for some of the most common formats:

-   **BF16** – exponent 8, fraction 7;
-   **FP32** – exponent 8, fraction 23.

## Why "dual-path"?

When adding two binary floating-point numbers the exponent distance between the
operands dictates the amount of mantissa alignment that is required -- and
consequently how complex normalization becomes.

-   **Near path** - covers the *potentially catastrophic cancellation*
    corner-case where the exponents differ by at most one. After the
    addition/subtraction a *leading-zero count* (CLZ) may be needed to
    renormalize the result.
-   **Far path** - taken when the exponents differ by two or more. In this
    regime the hidden-bit of the larger operand cannot be cancelled and at most
    one post-add shift is needed; as a result no CLZ is required.

Splitting the datapath specializes each path for its specific condition of
interest, which can lead to better critical path delay and creates power gating
opportunity.

The top-level function `dual_path::add_dual_path` chooses which path to take
using a comparison produced by `abs_diff`.

## File overview

| File               | Purpose                                                |
| ------------------ | ------------------------------------------------------ |
| `common.x`         | Small helpers shared by all sub-modules (e.g.          |
:                    : *flush-to-zero*, leading-bit split, masks).            :
| `sign_magnitude.x` | Sign-magnitude add/sub helper used by both paths.      |
| `near_path.x`      | Implements the *near* path, including optional CLZ via |
:                    : `std\:\:clzt`.                                         :
| `far_path.x`       | Implements the *far* path, including guard/sticky      |
:                    : generation and round-to-nearest-even logic.            :
| `dual_path.x`      | Top-level that selects near vs. far, handles NaN/Inf   |
:                    : cases, and provides BF16/FP32 convenience wrappers.    :
| `quickcheck_*.x`   | Property-based tests comparing the module result       |
:                    : against reference arithmetic in `apfloat`/`bfloat16`.  :
| `BUILD`            | Bazel targets for libraries, unit tests, QuickCheck    |
:                    : proofs, and formatting tests.                          :

## Implementation notes

-   **Rounding** – both paths implement IEEE-754 "round-to-nearest ties-to-even"
    using guard/sticky bits.
-   **Flush-to-zero** – subnormals are handled via the helper `common::ftz`,
    simplifying later datapath logic.
-   **CLZ vs CLZT** – the *near* path can optionally use the tree-shaped
    `std::clzt`. Toggle via `common::USE_CLZT`.
