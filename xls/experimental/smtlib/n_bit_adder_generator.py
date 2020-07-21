"""
This file receives a number n from the user and creates an smt2 file. The smt2 file
produced contains an n-bit adder, and checks the adder's equivalence with SMT-LIB's 
builtin addition. Instructions for using CVC4 can be found in cvc4.md. Once an smt2
file
is created, we can run:

$ cvc4 <filename>

The created smt2 file asserts that the adder and the builtin addition DO NOT produce
the same result, so the output we expect to see is:

$ unsat

meaning the adder and the builtin addition never produce different results. They are
logically equivalent. 
"""

def description_comments(n, f):
  f.write("; The following SMT-LIB verifies that a {bits}-bit adder is equivalent\n; to CVC4's built in bit-vector addition.\n\n""".format(bits = n))

def logic_and_variables(n, f):
  f.write("(set-logic ALL)\n\n")
  f.write("; Declare bit-vectors and proxies for indices\n")
  f.write("(declare-fun x () (_ BitVec {bits}))\n".format(bits = n))
  f.write("(declare-fun y () (_ BitVec {bits}))\n".format(bits = n))
  for i in range(n):
    for var in ["x", "y"]:
      f.write("(define-fun {v}{b} () (_ BitVec 1) ((_ extract {b} {b}) {v}))\n".format(v = var, b = i))
  f.write("\n")

def half_adder(n, f):
  f.write("; Half adder for bit {b}\n".format(b = n))
  f.write("(define-fun s{b} () (_ BitVec 1) (bvxor x{b} y{b}))\n".format(b = n))
  f.write("(define-fun c{b} () (_ BitVec 1) (bvand x{b} y{b}))\n\n".format(b = n))

def full_adder(n, f):
  f.write("; Full adder for bit {b}\n".format(b = n))
  f.write("(define-fun s{b} () (_ BitVec 1) (bvxor c{b_1} (bvxor x{b} y{b})))\n".format(b = n, b_1 = n - 1))
  f.write("(define-fun c{b} () (_ BitVec 1) (bvor (bvand (bvxor x{b} y{b}) c{b_1}) (bvand x{b} y{b})))\n\n".format(b = n, b_1 = n - 1))

def get_result_bits(n):
  string = ""
  for i in range(n-1, -1, -1):
    string += "s{} ".format(i)
  return string[:-1]

def make_sum(n, f):
  f.write("; Concatenate s bits to create sum\n")
  f.write("(define-fun sum () (_ BitVec {bits}) (concat {s}))\n\n".format(bits = n, s = get_result_bits(n)))

def assert_and_check_sat(n, f):
  f.write("; Compare {bits}-bit adder result and internal addition and solve\n".format(bits = n))
  f.write("(assert (not (= sum (bvadd x y))))\n")
  f.write("(check-sat)")

def n_bit_adder(n):
  f = open("adder_2x{}.smt2".format(n), "w")
  description_comments(n, f)
  logic_and_variables(n, f)
  half_adder(0, f)
  for i in range(1, n):
    full_adder(i, f)
  make_sum(n, f)
  assert_and_check_sat(n, f)
  f.close()

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

if __name__ == '__main__':
  n = input("Number of bits for each adder input: ")
  n_bit_adder(int(n))
