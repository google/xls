# Plotting SMT Solver Speeds of Bitvector Operation SMT-LIB Proofs

Each of the plots in this folder have names of the form

\<operation\>\_nests\<list or value\>\_bits\<list or value\>

The \<operation\> is simply the operation that has been tested: *add* for
addition, *mul* for multiplication, and *shl* for shift left. Next is *nest*,
followed by either *list* and a number, or just a number. If *nest* is followed
by *list* and a number, this indicates that the number of nested operations is
the changing variable, not the bits.

For example, if the file's name includes *nestslist10*, then we know the
number of nested operations is the changing variable, and the maximum number
of nested operations is 10.

If *nest* is followed by just a number, then we know that the number of nested
operations is static. For example, if the file's name includes *nests10*, then
we know there are 10 nested operations.

The same idea applies to *bits*. If it's followed by *list* and a number, the
bit count is the changing variable, with the number indicating the max bit
count. If *bits* is followed by just a number, then the bit count doesn't
change, and that's the number of bits.

As a final example, let's look at one of the files in this folder,
mul\_nestslist10\_bits4.png. This plot shows the solver speeds on the
multiplication operation with input bitvectors of 4 bits, with the number
of nested multiplication operations varying up to 10 nested operations.
