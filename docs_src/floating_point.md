# Floating-point routines

XLS provides implementations of several floating-point operations and may add
more at any time. Here are listed notable details of our implementations or of
floating-point operations in general.

## Supported operations

Here are listed the routines so far implemented in XLS, and the precisions of
such. Precisions will be listed either by their IEEE-754-defined names, or
"APFloat", for arbitrary-precision floating-point. This indicates that the
operation is supported for any combination of exponent and fractional part width
(within reason, e.g., no 0-bit exponents). Full details on APFloat support are
below.

-   Add/sub: APFloat
-   Mul: APFloat
-   FMA (fused multiply-add): float32

## "APFloat" - Arbitrary-precision floating-point

TODO(rspringer): 2021-04-06: This.

## Operation details

### Add/sub

TODO(rspringer): 2021-04-06: This.

### Mul

TODO(rspringer): 2021-04-06: This.

### FMA

The `fma` operation (again, fused multiply-add) is a three-operand operation
that computes the product of the first two and the sum of that with the third.
The IEEE 754-2008 decription of the operation states that the operation should
be performed "as if with unbounded range and precision", limited only by
rounding of the final result. In other words, this differs from a sequence of a
separate multiply followed by an add in that there is only a single rounding
step (instead of the two involved in separate operations).

In practice, this means A) that the precision of an FMA is higher than
individual ops, and thus that B) an FMA requires significantly more internal
precision bits than naively expected.

For binary32 inputs, to acheive the standard-specified precision, the initial
mul requires the usual 48 ((23 fraction + 1 "hidden") * 2) fraction bits.
When performing the subsequent add step, though, it is necessary to maintain
*72* fraction bits ((23 fraction + 1 "hidden") * 3). Fortunately, this sum
includes the guard, round, and sticky bits (so we don't need 75). The
mathematical derivation of the exact amount will not be given here (as I've not
done it), but the same calculated size would apply for other data types (i.e.,
54 * 2 = 108 and 54 * 3 = 162 for binary64).

Aside from determining the necessary precision bits, the FMA implementation is
rather straightforward, especially after reviewing the adder and multiplier.

## Testing

Several different methods are used to test these routines, depending on
applicability. These are:

-   Reference comparison: exhaustive testing
-   Reference comparison: space-sampling
-   Formal proving

When comparing to a reference, a natural question is the stability of the
reference, i.e., is the reference answer the same across all versions or
environments? Will the answer given by glibc/libm on AArch64 be the same as one
given by a hardware FMA unit on a GPU? Fortunately, all "correct"
implementations will give the same results for the same inputs.\* In addition,
POSIX has the same result-precision language. It's worth noting that -ffast-math
doesn't currently affect FMA emission/fusion/fission/etc.

\* - There are operations for which this is not true. Transcendental ops may
differ between implementations due to the
[_table maker's dilemma_](https://en.wikipedia.org/wiki/Rounding).

### Exhaustive testing

This is the happiest case - where the input space is so small that we can
iterate over every possible input, effectively treating the input as a binary
iteration counter. Sadly, this is uncommon (except, perhaps for ML math), as
binary32 is the precision floor for most problems, and a 64-bit input space is
well beyond our current abilities. Still - if your problem _can_ be exhaustively
tested (with respect to a trusted reference), it _should_ be exhaustively
tested!

None of our current ops are tested in this way, although the bf16 cases
could/should be.

### Space-sampling

When the problem input space is too large for exhaustive testing, then random
samples can be tested instead. This approach can't give complete verification of
an implementation, but, given enough samples, it can yield a high degree of
confidence.

The existing modules are tested in this way. This could be improved by
preventing re-testing of any given sample (at the cost of memory and, perhaps,
atomic/locking costs) and by identifying interesting "corner cases" of the input
space and focusing on those.

### Formal verification

This sort of testing utilizes our formal solver infrastructure to prove
correctness with the solver's internal FP implementation. This is fully
described in the
[solvers documentation](./solvers.md).
