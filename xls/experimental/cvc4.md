# Adder/Multiplier Proofs with CVC4

CVC4 is another SMT solver that we can use for logical equivalence checking. 
This directory contains a two-bit adder in SMT-LIB, a universal SMT language
that solvers like Z3 and CVC4 can consume and produce. 

## CVC4 Installation and Command Line Use

The binary for CVC4 can be found [here](https://cvc4.github.io/downloads.html).
Once installed, it can be used directly on the command line. CVC4 infers the
input language by the file type, so to run the two-bit adder through CVC4, we
can enter the following in the xls/experimental directory:

```
$ cvc4 two_bit_adder.smt2
```

and the output we should see is:

```
unsat
```

meaning that a difference in output between the two-bit adder and CVC4's builtin
adder is unsatisfiable. 

TODO (Julian): Add 4, 8, 16, 32 bit adder and multiplier files, and script to
compare speeds of proofs of all of these with Z3. 
