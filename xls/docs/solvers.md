# XLS Solvers

Programs that are represented as optimized XLS IR are converted into circuits
based on boolean logic, and so it is also possible to feed those as logical
operations to a theorem prover.

We have implemented that conversion with the Z3 theorem prover using its "bit
vector" type support. As a result, you can conceptually ask Z3 to prove any
predicate that can be expressed as XLS, over all possible parameter inputs.

See the [tools documentation](https://github.com/google/xls/tree/main/xls/g3doc/tools.md)
for usage information on related command line tools.

## Applications

This facility is expected to be useful to augment random testing. While
profiling the values in an XLS IR function that is given random stimulus, we may
observe bits that result from nodes that *appear* to be constant (but are not
created via a "literal" or a "concat" of a literal).

**Example:** Say the value resulting from `and.1234` in the graph appears to be
constant zero with all the stimulus provided via a fuzzer thus far -- the solver
provides a facility whereby we can ask "is there a counterexample to `and.1234`
always being zero?" and the solver will either say "no, it is always zero", or
it will yield a counterexample, or will not terminate within the allocated
deadline.

Assuming we can prove useful properties in a reasonable amount of time, we can
use this proof capability to help find interesting example inputs that provide
unique stimulus.

### Correctness WRT reference: [32-bit Floating-Point Adder](https://github.com/google/xls/tree/main/xls/modules/fpadd_2x32.x)

The full input space for a 32-bit adder is a whopping 64 bits - far more than is
possible to exhaustively test for correctness. Proving correctness via Z3,
however, is relatively straightforward: at a high level, one just compares the
output from the DSLX (translated into Z3) and comapres that to the same
operation performed solely in Z3.

In detail, the steps are:

1.  Translate the DSLX implementation into Z3 via
    `Z3Translator::CreateAndTranslate()`.
1.  Create a Z3 implementation of the same addition. This is nearly trivial, as
    Z3 helpfully has built-in support for floating-point values and theories.
1.  Take the result nodes from each "branch" above and create a new node
    subtracting the two. This is the absolute error. Note: Usually, one is
    interested in relative error when working with FP values, but here, our
    target is absolute equivalence, so absolute error sufficies (and is
    simpler).
1.  Create a Z3 node comparing that error to the maximum bound (here 0.0f).
1.  Feed that error node into a Z3 solver, asking it to prove that the error
    could be greater than that bound.

If the solver can not satisfy that criterion, then that means the error is
_never_ greater than that bound, i.e., that the implementations are equivalent
(with our 0.0f bound).

### Transform validity

It's usually not possible (or is merely extremely difficult) to write tests to
prove that an optimization/transform is safe across all input IR. By comparing
the optimized vs. unoptimized IR in a similar manner as the correctness proof
above, we can symbolically prove safety.

The only difference between this and the correctness proof is that both the
optimized and unoptimized IR need to be fed into the same Z3Translator (the
second via `Z3Translator::AddFunction()`) and the result nodes each are used in
the error comparison.

## Current Limitations

Hypothetically, any XLS function that computes a predicate (bool) can be fed to
Z3 for satisfiability testing. Currently a more limited set of predicates are
exposed that can be easily expressed on the command line; however, it should be
possible to provide:

a) an XLS IR file b) a set of nodes in the entry function c) a DSLX function
that computes a predicate on those nodes

Which would allow the user to compute arbitrary properties of nodes in the
function with the concise DSL syntax.

### Subroutines

Z3 doesn't intrinsically have support for subroutines, or as they're called in
Z3, "macros", instead requiring that
[all function calls be inlined](https://stackoverflow.com/questions/7740556/equivalent-of-define-fun-in-z3-api).

There is an extension that adds support for _recursive_ function decls and defs,
but in our experience, it doesn't behave the way we'd expect.

Consider the following example:

```
package p

fn mapper(value: bits[32]) -> bits[32] {
  ret value
}

fn main() -> bits[1] {
  literal_0: bits[32] = literal(value=0)
  literal_1: bits[1] = literal(value=1)
  elem_0: bits[32] = invoke(literal_0, to_apply=mapper)
  eq_12: bits[1] = eq(literal_0, elem_0)
  ret and_13: bits[1] = and(eq_12, literal_1)
}
```

Here, it's trivial for a human reader to see that the results are the same; the
output should be equal to 1. Z3, however, reports that this is not necessarily
the case, suggesting that `literal_0` and `elem_0` would not be equal in the
case where the input to `mapper` was 1...which is clearly never the case here.

To address this, we require that all subroutines (including those used in maps
and counted fors) be inlined before consumption by Z3.
