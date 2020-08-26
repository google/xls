# Bitvector Operation Proofs with SMT-LIB and SMT Solvers

Currently, XLS uses Z3, an SMT solver, for logical equivalence checking of IR
and netlist. While Z3 is widely regarded as an efficient SMT solver, we have
found that it can be too slow on bitvector operations, especially
multiplication. However, there are many other SMT solvers out there that may
be useful in XLS. This folder aims to identify speed differences between
Z3 and a few other solvers (CVC4, STP, and Yices), with the ability to add
on other solvers in future research.

SMT solvers like CVC4, STP, Yices, and Z3 can produce and consume a universal language
called SMT-LIB, which allows us to compare logical equivalence checking for

these solvers on the same file. This directory contains generators for n-bit
addition, multiplication, and shift proofs in SMT-LIB.

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

meaning that the assertion made in the smt2 file is unsatisfiable. The smt2 files
produced by the generator files in this folder contain the assertion that the "by-hand"
operation described in them is not equal to their builtin bitvector operation. The
result of ``unsat`` indicates that this assertion is impossible to achieve, meaning
the two implementations of the operation are logically equivalent.

# Plots

This folder also contains files (solvers\_op\_comparison\_bits\_list.py and
solvers\_op\_comparison\_nests\_list.py) to test these SMT solvers on bitvector
operation proofs, and store their performance in csv files. Once a csv file has
been created, we can then use plot\_solver\_speed\_data.py to plot the data in
the csv file.

# Results

Some plots for addition, mulitplication, and shift can be found in the
sub-folder solvers\_plots. Among CVC4, STP, Yices, and Z3, we have found no
significant difference in performance. However, there are some smaller cases
(such as 2-bit addition or 16-bit shift) where there is a difference in speed,
and it seems to indicate that some solvers are better at simplifying proofs in
certain cases. But unfortunately, we have yet to find a solver that can
automatically simplify 32-bit multiplication to be as fast as we desire (and it
is still unclear if this is even possible).

# Next Steps

There are three main areas we can explore to continue this research.

One is to test out more SMT solvers beyond the four studied in this folder. The files
described here should easily incorporate the addition of new SMT solvers that
can be used on the command line.

Another option to look into is using BDDs to solve our logical equivalence
checking speed issues. BDDs (Binary Decision Diagrams) are another form of
formal verification, like SMT solving. All equivalent logical expressions have
the exact same BDD, so once the BDD for the problem have been created,
verification doesn't take very long. However, BDDs are exponential in size, so
it is also possible that a solver using BDDs may take just as long. Q3B is an
readily available SMT solver that uses BDDs, and this may be a good place to
start.

The last and most difficult option is to investigate the internals of these SMT
solvers. Each SMT solver has its own, unclear way of simplifying the input
problem, and if we could figure out how they do this, we might be able to
simplify bitvector operations enough to reach the performance we need.
