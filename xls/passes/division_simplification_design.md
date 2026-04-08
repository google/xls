# Division Simplification by Selector and QueryEngine

Users often use division where the divisor takes a small set of values (e.g.,
powers of two or a selection of constants). This document outlines the design
for simplifying such divisions to avoid expensive hardware dividers.

## Design

The optimization is implemented in `value_set_simplification_pass.cc`. It uses
the `PartialInfoQueryEngine` (retrieved via
`context.SharedQueryEngine<PartialInfoQueryEngine>(f)`) to evaluate the set of
possible values for the divisor $Y$.

`PartialInfoQueryEngine` is preferred over standard ternary analysis because it
provides **interval/range information**. Range information is critical for
division; for example, if we know $Y$ is in $[2, 4]$, we know it takes at most 3
values, which ternary might miss if multiple bits are toggling.

### Rule 1: All Possible Values are Powers of Two (or Zero)

XLS semantics define division by 0 as: If the divisor is zero, unsigned division
produces a maximal positive value. For signed division, if the divisor is zero
the result is the maximal positive value if the dividend is non-negative or the
maximal negative value if the dividend is negative.

As such, zero is like a power of two in that dividing by it is close to free.

Since we always prioritize area:

-   **Constant Shifts Tree**: Used if $K \le \log_2 M + 1$ (where $M$ is max
    shift amount).
-   **Variable Shift Fallback**: Used if $K > \log_2 M + 1$.
    -   **Zero Guard**: Rewrite to `Y == 0 ? (appropriate value) : x >>
        Encode(Y)`.

!!! NOTE
    **Property Checks for Powers of Two**: To verify Rule 1, we scan the
    `IntervalSet` to count powers of two and identify if they cover all values.

### Rule 2: Some Possible Values are NOT Powers of Two

Let $L$ be the count of non-power-of-two constants.

**How Sinking Works**: We rewrite the single division `div(x, Y)` into a
`priority_sel` over the possible constant values of $Y$. The branches of the
select become `div(x, C_1)`, `div(x, C_2)`, etc. This replaces a
variable-divisor division with multiple constant-divisor divisions. We rely on
the existing `arith_simplification_pass.cc` (which runs in the same pipeline) to
recognize `div(x, Constant)` and replace it with a multiply-and-shift using the
reciprocal. We do **not** implement the reciprocal multiplication logic here!

-   **If an Area Model is available**: Query the area of a single multiplication
    of size $N$ and the muxes. If it is cheaper than a divider, sink it.
-   **If no Area Model is present**: Fallback to a safe universal limit of $L
    \le 2$ (replaces a divider with at most two multipliers, which is neutral
    area but a huge latency win).

**Caveats and Edge Cases**:

-   **Skip Single Literals**: If $Y$ is a single known literal constant, abort
    early. Let `arith_simplification_pass.cc` handle it natively.
-   **Division By Zero**: If the set of constants contains $0$, do not emit
    `div(x, 0)`. Instead, emit the XLS standard division-by-zero value directly
    for that branch (e.g., all bits set to 1).

### The Hybrid Case (Powers of Two + General Constants)

When we have a mix of powers of two ($K_{\text{p2}}$) and general constants
($L$), we have two choices for the powers of two:

-   **Option A (Separate)**: Keep powers of two as individual constants shifts.
    Total cases: $K_{\text{p2}} + L$.
-   **Option B (Grouped)**: Group all powers of two into a single variable shift
    case. Total cases: $L + 1$.

**Decision Rule (Consistent with Rule 1)**:

-   **If an Area Model is available**: Directly compare options A and B using
    the area estimator.
-   **If no Area Model is present**: Fallback to threshold $K_{\text{p2}} >
    \log_2 M + C$ (where $C = 1$ for `UDiv`, $C = 2$ for `SDiv`).

**Final Select Cardinality ($C_{\text{eff}}$)**:

-   For Option A: $C_{\text{eff}} = K_{\text{p2}} + L$
-   For Option B: $C_{\text{eff}} = L + 1$

**Profitability Sinking Rules**:

-   **If an Area Model is available**: Query the area of the chosen approach vs
    the divider.
-   **If no Area Model is present**: Limit to $L \le 2$ non-powers-of-two
    constants for Option A, or $L \le 1$ for Option B.

## Implementation Phases

To ensure a smooth and incremental rollout, we will split the implementation
into three phases:

### Phase 1: Rule 1 (Powers of Two Check)

Implement Rule 1 using `PartialInfoQueryEngine`. Use `AtMostBitOneTrue` to
detect powers of two and `Op::kEncode` to calculate shift amounts.

### Phase 2: Rule 2 (Implicit Select Sinking)

Implement Rule 2 using `PartialInfoQueryEngine`.

-   Extract sets of constants from small intervals in `IntervalSet`.
-   Synthesize a select tree.
-   Fallback to $L \le 2$ if no Area Model is available.

### Phase 3: Hybrid Cases

Merge the powers-of-two support and general constant support into the final
decision rule (Option A vs Option B).

--------------------------------------------------------------------------------

## Alternatives Considered

### 1. Simple Pattern Match in `arith_simplification_pass.cc`

Brittle check for `div(x, sel(c, [constants]))`.

-   **Reason for Rejection**: Misses hidden selectors and non-immediate
    constants.

### 2. Generic Lifting in `select_lifting_pass.cc`

Enable `UDiv`/`SDiv` for select lifting.

-   **Reason for Rejection**: Select Lifting pulls operations *out* of selects
    (e.g., `sel(c, [x/1, x/2]) -> x / sel(c)`). This is the opposite of what we
    want! We want **Select Sinking** (pushing the division into the select).
