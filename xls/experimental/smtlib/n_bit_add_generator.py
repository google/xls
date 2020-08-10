"""
This file receives numbers from the --N flag, and for each number n, creates an smt2 
file containing an n-bit adder equivalence proof. For example, to create an smt2 
file for 2-bit addition and an smt2 file for 4-bit addition, we can run
(after building):

$ bazel-bin/xls/experimental/smtlib/n_bit_add_generator --N=2,4

Once these files are created, we can use an SMT solver to check the proof's 
satisfiability. Instructions for using a solver on an smt2 file can be found in solvers.md. If we wanted to use Z3, for example, on the 2-bit addition file, we can run:

$ z3 add_2x2.smt2

The created smt2 file asserts that the adder and the builtin addition DO NOT produce
the same result, so the output we expect to see is:

$ unsat

meaning the adder and the builtin addition never produce different results. They are
logically equivalent. 
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
flags.DEFINE_list("N", None, "List of n values for each n-bit addition proof.")
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
f"""; The following SMT-LIB verifies that a {n}-bit adder is equivalent
; to CVC4's built in bit-vector addition.
""", file=f)

def logic_and_variables(n, f):
  """
  Write the set-logic for the proof (QF_BV is the bitvector logic), declare the input
  bitvector variables, and define the variables for their indices. Note that x_i or y_i
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

def half_adder(n, f):
  """
  Define the sum and carry bits for a half adder.
  """
  print(
f"""; Half adder for bit {n}
(define-fun s{n} () (_ BitVec 1) (bvxor x{n} y{n}))
(define-fun c{n} () (_ BitVec 1) (bvand x{n} y{n}))
""", file=f)

def full_adder(n, f):
  """
  Define the sum and carry bits for a full adder.
  """
  print(
f"""; Full adder for bit {n}
(define-fun s{n} () (_ BitVec 1) (bvxor c{n - 1} (bvxor x{n} y{n})))
(define-fun c{n} () (_ BitVec 1) (bvor (bvand (bvxor x{n} y{n}) c{n - 1}) (bvand x{n} y{n})))
""", file=f)

def get_concat_result_bits(n):
  """
  Input: integer n
  Output: string containing smt2 concat operation combining the bits of the 
          addition result.
  Example:
  >> get_concat_level_bits(4)
  >> (concat s3 (concat s2 (concat s1 s0)))
  """
  concats = []
  for i in range(n):
    if i == 0:
      concats.append(f"s{i}")
    else:
      rhs = concats[i - 1]
      concat = ["(concat", f"s{i}", rhs + ")"]
      concats.append(" ".join(concat))
  return concats[-1]

def make_sum(n, f):
  """
  Create the output of the addition. Get the bitvectors for the 
  addition at each index using get_result_bits, and concatenate them
  all together. 
  """
  print(
f"""; Concatenate s bits to create sum
(define-fun sum () (_ BitVec {n}) {get_concat_result_bits(n)})
""", file=f)

def assert_and_check_sat(n, f):
  """
  Write the assertion that the output of the 'by-hand' addition (called sum)
  does not equal the output of the builtin bvadd operation, and tell the solver
  to check the satisfiability. 
  """
  print(
f"""; Compare {n}-bit adder result and internal addition and solve
(assert (not (= sum (bvadd x y))))
(check-sat)""", file=f)

def n_bit_add_existing_file(n, f):
  """
  Given a file, write an addition proof into it with n-bit arguments. 
  """
  description_comments(n, f)
  logic_and_variables(n, f)
  half_adder(0, f)
  for i in range(1, n):
    full_adder(i, f)
  make_sum(n, f)
  assert_and_check_sat(n, f)

def n_bit_add_new_file(n):
  """
  Create a new file, and in it, write an addition proof with n-bit arguments. 
  """
  with gopen(f"add_2x{n}.smt2", "w+") as f:
    n_bit_add_existing_file(n, f)

def main(argv):
  """
  Run the file. Take the list of integers from FLAGS.N, and for each integer
  n in N, make an smt2 file containing an n-bit-addition proof.
  """
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  else:
    for n_string in FLAGS.N:
      n_bit_add_new_file(int(n_string))

if __name__ == '__main__':
  app.run(main) 
