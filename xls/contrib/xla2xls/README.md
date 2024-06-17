# XLA2XLS

The XLA2XLS flow takes XLA HLO as an input, and generates:
- HW inference in XLS IR
- SW inference in C, optimizer / SIMD friendly
- Ground truth scalar operation graph for formal verification

Original author:

Sean Purser-Haskell (seanhaskell@google.com, sean.purserhaskell@gmail.com):
https://github.com/spurserh

## Disclaimer

XLA2XLS is an alternate XLS frontend, primarily maintained by its original
contributor. It lives in the XLS repository to better leverage common
infrastructure.

The core XLS authors do not expect to maintain this tool outside of a
best-effort basis, but contributions are welcome!

### Summary

The XLA2XLS flow's primary function is to generate hardware implementations of
inference for ML models. It takes XLA HLO as its input, and traces the scalar
operations done in performing inference by the XLA HLOEvalator to a database.

This database serves as ground truth for verification of generated code. This
can be extended to formal verification, as the graph of scalar operations
can be directly fed to a solver to check for equivalence with the generated
operations (when fully unrolled).

It can also be used directly to generate the code via "rolling up" the scalar
operation graph, in a bottom-up approach that avoids the need to re-implement
the semantics of each HLO operation.
