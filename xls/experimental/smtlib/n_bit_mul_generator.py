"""
This file receives numbers from the --N flag, and for each number n, creates an smt2 
file containing an n-bit multiplier equivalence proof. For example, to create and smt2
file for 2-bit multiplication and an smt2 file for 4-bit multiplication, we can run

$ bazel-bin/xls/experimental/smtlib/n_bit_add_generator --N=2,4

Once these files are created, we can use an SMT solver to check the proof's 
satisfiability. Instructions for using a solver on an smt2 file can be found in solvers.md. If we wanted to use Z3, for example, on the 2-bit multiplication file, we can run:

$ z3 mul_2x2.smt2

The created smt2 file asserts that the multiplier and the builtin addition DO NOT 
produce the same result, so the output we expect to see is:

$ unsat

meaning the multiplier and the builtin multiplication never produce different results. 
They are logically equivalent. 
"""

import sys
from xls.common.gfile import open as gopen

from absl import app
from absl import flags

def list_contains_only_integers(L):
  """
  Input: list L
  Output: True if all of the elements of the list are strings that contain
          digits between 0-9, False otherwise. 
  Used to check that the list of numbers given by user contains only integer strings.
  """
  for elm in L:
    if not elm.isdigit():
      return False
  return True

FLAGS = flags.FLAGS
flags.DEFINE_list("N", None, "List of n values for each n-bit multiplication proof.")
flags.register_validator("N",
                         list_contains_only_integers,
                         message="--N must contain only integers.")
flags.mark_flag_as_required("N")

def description_comments(n, f):
  """
  Write comments to the top of the smt2 file describing the proof it contains:
  the operation, how many bits in the arguments, and how many operations. 
  """
  print(
f"""; The following SMT-LIB verifies that a {n}-bit multiplier is equivalent
; to CVC4's built in bit-vector multiplication.
""", file=f)

def logic_and_variables(n, f):
  """
  Write the set-logic for the proof (QF_BV is the bitvector logic), declare the input
  bitvector variables, and define variables for their indices. Note that x_i or y_i 
  corresponds to index i of that input bitvector. 
  """
  print(
f"""(set-logic QF_BV)

; Declare bit-vectors and proxies for indices""", file=f)
  for var in ["x", "y"]:
    print(f"(declare-fun {var} () (_ BitVec {n}))", file=f)
    for i in range(n):
      print(f"(define-fun {var}{i} () (_ BitVec 1) ((_ extract {i} {i}) {var}))", file=f)
  print("", file=f)

def get_concat_level_bits(i, n):
  """
  Input: integers i, n
  Output: string containing smt2 concat operation combining the bits of the 
          multiplication of the current variable by the i-th index of the 
          previous variable. 
  Example:
  >> get_concat_level_bits(1, 4)
  >> (concat m1_3 (concat m1_2 (concat m1_1 #b0)))
  """
  concats = [] 
  if i > 0:
    concats.append(f"(concat m{i}_{i} #b{'0' * i})")
  else:
    concats.append(f"m0_0")
  if i < (n - 1):
    for j in range(i + 1, n):
      rhs = concats[j - i - 1]
      concat = ["(concat", f"m{i}_{j}", rhs + ")"]
      concats.append(" ".join(concat))
  return concats[-1]

def mul_level(i, n, f):
  """
  Mutiply the current variable by the i-th index of the previous variable. This 
  is done by evaluating each of the output bits with a boolean expression and 
  then concatenating them together. 
  """
  print(f"; Multiply x by y{i}, shifting x bits accordingly", file=f)
  for j in range(i, n):
    print(f"(define-fun m{i}_{j} () (_ BitVec 1) (bvand x{j - i} y{i}))", file=f)
  print(f"(define-fun m{i} () (_ BitVec {n}) {get_concat_level_bits(i, n)})\n", file=f)

def get_result_bits(n):
  """
  Input: integer n
  Output: a string representing the n bits of the final multiplication result
  Example:
  >> get_result_bits(5)
  >> m4 m3 m2 m1 m0
  """
  bits = []
  for i in range(n-1, -1, -1):
    bits.append(f"m{i}")
  return " ".join(bits)

def make_mul(n, f):
  """
  Create the output of the multiplication. Get the bitvectors for the 
  multiplication at each index using get_result_bits, and add them all together.
  """
  print(
f"""; Add all m bit-vectors to create mul
(define-fun mul () (_ BitVec {n}) (bvadd {get_result_bits(n)}))
""", file=f)

def assert_and_check_sat(f):
  """
  Write the assertion that the output of the 'by-hand' multiplication (called 
  mul), does not equal the output of the builtin bvmul operation, and tell the
  solver to check the satisfiability. 
  """
  print(
"""; Assert and solve
(assert (not (= mul (bvmul x y))))
(check-sat)""", file=f)

def n_bit_mul_existing_file(n, f):
  """
  Given a file, write a multiplication proof into it with n-bit arguments.
  """
  description_comments(n, f)
  logic_and_variables(n, f)
  for i in range(n):
    mul_level(i, n, f)
  make_mul(n, f)
  assert_and_check_sat(f)

def n_bit_mul_new_file(n):
  """
  Create a new file, and in it, write a multiplication proof with
  n-bit arguments. 
  """
  with gopen(f"mul_2x{n}.smt2", "w+") as f:
    n_bit_mul_existing_file(n, f)

def main(argv):
  """
  Run the file. Take the list of integers from FLAGS.N, and for each integer
  n in N, make an smt2 file containing an n-bit-multiplication proof.
  """
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  else:
    for n_string in FLAGS.N:
      n_bit_mul_new_file(int(n_string))

if __name__ == '__main__':
  app.run(main)
