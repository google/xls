# Adder/Multiplier Proofs with SMT-LIB and SMT Solvers

SMT solvers like CVC4, Yices, and Z3 can produce and consume a universal language
called SMT-LIB, which allows us to compare logical equivalence checking for
these solvers on the same file. This directory contains a generator of n-bit
multiplication proofs in SMT-LIB.  

## Evaluating an smt2 file proof with CVC4, STP, Yices, or Z3

Once these SMT solvers are installed, they can be used directly on the command line. 
CVC4, STP, Yices, and Z3 infer input language by the file type, so to test a
solver we can enter the following in the xls/experimental/smtlib directory:

```
$ <solver_command> <smt2 file>
```

and the output we should see is:

```
unsat
```

meaning that the assertion made in the smt2 file is unsatisfiable. Files
produced by n\_bit\_mul\_generator.py contain the assertion that the "by-hand"
multiplication described in them is not equal to their builtin bitvector
multiplication. The result of ``unsat`` indicates that this assertion is
impossible to achieve, meaning the two implementations of multiplications are
logically equivalent. 
