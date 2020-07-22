"""
This file receives a number n from the user in argv and creates an smt2 file. When 
we run:

$ python3 n_bit_adder_generator.py <n>

the smt2 file produced contains an n-bit adder that checks the adder's equivalence 
with SMT-LIB's builtin addition. Instructions for using CVC4 can be found in cvc4.md. 
Once an smt2 file is created, we can run:

$ cvc4 <filename>

The created smt2 file asserts that the adder and the builtin addition DO NOT produce
the same result, so the output we expect to see is:

$ unsat

meaning the adder and the builtin addition never produce different results. They are
logically equivalent. 
"""

import sys

def description_comments(n, f):
  print(
f"""; The following SMT-LIB verifies that a {n}-bit adder is equivalent
; to CVC4's built in bit-vector addition.
""", file=f)

def logic_and_variables(n, f):
  print(
f"""(set-logic ALL)

; Declare bit-vectors and proxies for indices
(declare-fun x () (_ BitVec {n}))
(declare-fun y () (_ BitVec {n}))""", file=f)
  for i in range(n):
    for var in ["x", "y"]:
      print(f"(define-fun {var}{i} () (_ BitVec 1) ((_ extract {i} {i}) {var}))", file=f)
  print("", file=f)

def half_adder(n, f):
  print(
f"""; Half adder for bit {n}
(define-fun s{n} () (_ BitVec 1) (bvxor x{n} y{n}))
(define-fun c{n} () (_ BitVec 1) (bvand x{n} y{n}))
""", file=f)

def full_adder(n, f):
  print(
f"""; Full adder for bit {n}
(define-fun s{n} () (_ BitVec 1) (bvxor c{n - 1} (bvxor x{n} y{n})))
(define-fun c{n} () (_ BitVec 1) (bvor (bvand (bvxor x{n} y{n}) c{n - 1}) (bvand x{n} y{n})))
""", file=f)

def get_result_bits(n):
  bits = []
  for i in range(n-1, -1, -1):
    bits.append(f"s{i}")
  return " ".join(bits)

def make_sum(n, f):
  print(
f"""; Concatenate s bits to create sum
(define-fun sum () (_ BitVec {n}) (concat {get_result_bits(n)}))
""", file=f)

def assert_and_check_sat(n, f):
  print(
f"""; Compare {n}-bit adder result and internal addition and solve
(assert (not (= sum (bvadd x y))))
(check-sat)""", file=f)

def n_bit_adder(n):
  with open(f"adder_2x{n}.smt2", "w") as f:
    description_comments(n, f)
    logic_and_variables(n, f)
    half_adder(0, f)
    for i in range(1, n):
      full_adder(i, f)
    make_sum(n, f)
    assert_and_check_sat(n, f)

if __name__ == '__main__':
  n_bit_adder(int(sys.argv[1]))
    
