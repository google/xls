#
# Copyright 2020 The XLS Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=g-doc-args,g-bad-import-order,line-too-long

"""This file receives numbers from the --N flag, and for each number n, creates an smt2 file containing an n-bit multiplier equivalence proof.

For example, to create and smt2 file for 2-bit multiplication and an smt2 file
for 4-bit multiplication, we can run
$ bazel-bin/xls/experimental/smtlib/n_bit_add_generator --N=2,4

Once these files are created, we can use an SMT solver to check the proof's
satisfiability. Instructions for using a solver on an smt2 file can be found in
solvers.md. If we wanted to use Z3, for example, on the 2-bit multiplication
file, we can run:

$ z3 mul_2x2.smt2

The created smt2 file asserts that the multiplier and the builtin addition DO
NOT
produce the same result, so the output we expect to see is:

$ unsat

meaning the multiplier and the builtin multiplication never produce different
results.
They are logically equivalent.
"""

from absl import app
from absl import flags
from xls.common import gfile
from xls.experimental.smtlib import flags_checks

FLAGS = flags.FLAGS
flags.DEFINE_list(
    "N", None, "List of n values for each n-bit multiplication proof."
)
flags.register_validator(
    "N",
    flags_checks.list_contains_only_integers,
    message="--N must contain only integers.",
)
flags.mark_flag_as_required("N")


def description_comments(n, f):
  """Write comments to the top of the file describing what it does.

  Write comments to the top of the smt2 file describing the proof it contains:
  the operation, how many bits in the arguments, and how many operations.

  Args:

  n: An integer, the number of bits in each input bitvector.
  f: The file to write into.
  """
  print(
      f"""; The following SMT-LIB verifies that a {n}-bit multiplier is equivalent
; to SMTLIB's built in bit-vector multiplication.
""",
      file=f,
  )


def logic_and_variables(n, f):
  """Set the logic for the smt2 file, and declare/define variables.

  Write the set-logic for the proof (QF_BV is the bitvector logic), declare the
  input bitvector variables, and define variables for their indices. Note that
  x_i or y_i corresponds to index i of that input bitvector. Args: n: An
  integer, the number of bits in each bitvector.

  f: The file to write into.
  """
  print(
      """(set-logic QF_BV)
; Declare bit-vectors and proxies for indices""",
      file=f,
  )
  for var in ["x", "y"]:
    print(f"(declare-fun {var} () (_ BitVec {n}))", file=f)
    for i in range(n):
      print(
          f"(define-fun {var}{i} () (_ BitVec 1) ((_ extract {i} {i}) {var}))",
          file=f,
      )
  print("", file=f)


def get_concat_level_bits(i, n):
  """Returns the bits of multiplying the current variable by the i-th index of the previous one.

  Take in integers i and n, returns a string containing the smt2 concat
  operation combining the bits of the multiplication of the current variable
  by the i-th index of the previous variable.
  Args:

  i: An integer, the index of the previous variable
  n: An integer, the number of bits in each bitvector
  """
  concats = []
  if i > 0:
    concats.append(f"(concat m{i}_{i} #b{'0' * i})")
  else:
    concats.append("m0_0")
  if i < (n - 1):
    for j in range(i + 1, n):
      rhs = concats[j - i - 1]
      concat = ["(concat", f"m{i}_{j}", rhs + ")"]
      concats.append(" ".join(concat))
  return concats[-1]


def mul_level(i, n, f):
  """Writes the result of multiplying the current variable by index i of the previous variable.

  Mutiply the current variable by the i-th index of the previous variable. This
  is done by evaluating each of the output bits with a boolean expression and
  then concatenating them together.

  Args: i: An integer, the index of the previous variable n: An integer, the
  number of bits in each bitvector f: The file to be written into.
  """
  print(f"; Multiply x by y{i}, shifting x bits accordingly", file=f)
  for j in range(i, n):
    print(
        f"(define-fun m{i}_{j} () (_ BitVec 1) (bvand x{j - i} y{i}))", file=f
    )
  print(
      f"(define-fun m{i} () (_ BitVec {n}) {get_concat_level_bits(i, n)})\n",
      file=f,
  )


def get_result_bits(n):
  """Returns a string of the bitvectors to be added to produce the final output.

  Take in an integer n, and return a string representing the bitvectors to be
  added together to produce the final multiplication result.
  Args:

  n: An integer, the number of bits in each bitvector
  """
  bits = []
  for i in range(n - 1, -1, -1):
    bits.append(f"m{i}")
  return " ".join(bits)


def make_mul(n, f):
  """Write the final multiplication output.

  Create the output of the multiplication.

  Get the bitvectors for the multiplication at each index using
  get_result_bits, and add them all together.
  Args:

  n: An integer, the number of bits in the output bitvector.
  f: The file to write into.
  """
  print(
      f"""; Add all m bit-vectors to create mul
(define-fun mul () (_ BitVec {n}) (bvadd {get_result_bits(n)}))
""",
      file=f,
  )


def assert_and_check_sat(f):
  """Writes an (unsatisfiable) assertion and tells the solver to check it.

  Write the assertion that the output of the 'by-hand' multiplication (called
  mul), does not equal the output of the builtin bvmul operation, and tell the
  solver to check the satisfiability.

  Args:
    f: The file to write into.
  """
  print(
      """; Assert and solve
(assert (not (= mul (bvmul x y))))
(check-sat)""",
      file=f,
  )


def n_bit_mul_existing_file(n, f):
  """Given a file, write a multiplication proof into it with n-bit arguments.

  Args:

  n: An integer, the number of bits for the input and output bitvectors.
  f: The file to write the proof into.
  """
  description_comments(n, f)
  logic_and_variables(n, f)
  for i in range(n):
    mul_level(i, n, f)
  make_mul(n, f)
  assert_and_check_sat(f)


def n_bit_mul_new_file(n):
  """Create a new file, and write a multiplication proof with n-bit arguments.

  Args:

  n: An integer, the number of bits for the input and output bitvectors.
  """
  with gfile.open(f"mul_2x{n}.smt2", "w+") as f:
    n_bit_mul_existing_file(n, f)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  else:
    for n_string in FLAGS.N:
      n_bit_mul_new_file(int(n_string))


if __name__ == "__main__":
  app.run(main)
