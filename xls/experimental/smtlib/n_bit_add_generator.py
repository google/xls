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

"""Creates an SMTLIB2 file with an n-bit adder equivalence proof.

This file receives numbers from the --N flag, and for each number n, creates an
smt2 file containing an n-bit adder equivalence proof.

For example, to create an smt2 file for 2-bit addition and an smt2 file for
4-bit addition, we can run (after building):
$ bazel-bin/xls/experimental/smtlib/n_bit_add_generator --N=2,4

Once these files are created, we can use an SMT solver to check the proof's

satisfiability. Instructions for using a solver on an smt2 file can be found in
solvers.md. If we wanted to use Z3, for example, on the 2-bit addition file, we
can run:

$ z3 add_2x2.smt2

The created smt2 file asserts that the adder and the builtin addition DO NOT
produce
the same result, so the output we expect to see is:

$ unsat

meaning the adder and the builtin addition never produce different results. They
are
logically equivalent.
"""

from absl import app
from absl import flags
from xls.common import gfile
from xls.experimental.smtlib import flags_checks

FLAGS = flags.FLAGS
flags.DEFINE_list("N", None, "List of n values for each n-bit addition proof.")
flags.register_validator(
    "N",
    flags_checks.list_contains_only_integers,
    message="--N must contain only integers.",
)
flags.mark_flag_as_required("N")


def description_comments(n, f):
  """Write comments to the top of the file describing what it does.

  Write comments to the top of the smt2 file describing the proof it contains:
  the operations, how many bits in the arguments, and how many operations.

  Args:
    n: An integer, the number of bits in each input bitvector.
    f: The file to write into.
  """
  print(
      f"""; The following SMT-LIB verifies that a {n}-bit adder is equivalent
; to CVC4's built in bit-vector addition.
""",
      file=f,
  )


def logic_and_variables(n, f):
  """Set the logic for the smt2 file, and declare/define variables.

  Write the set-logic for the proof (QF_BV is the bitvector logic), declare the
  input bitvector variables, and define variables for their indices.
  Note that x_i or y_i corresponds to index i of that input bitvector.

  Args:
    n: An integer, the number of bits in each bitvector.
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


def half_adder(n, f):
  """Define the sum and carry bits for a half adder.

  Args:
    n: An integer, the number of bits for each bitvector.
    f: The file to write into.
  """
  print(
      f"""; Half adder for bit {n}
(define-fun s{n} () (_ BitVec 1) (bvxor x{n} y{n}))
(define-fun c{n} () (_ BitVec 1) (bvand x{n} y{n}))
""",
      file=f,
  )


def full_adder(n, f):
  """Defines the sum and carry bits for a full adder.

  Args:
    n: An integer, the number of bits in the output bitvector.
    f: The file to write into.
  """
  print(
      f"""; Full adder for bit {n}
(define-fun s{n} () (_ BitVec 1) (bvxor c{n - 1} (bvxor x{n} y{n})))
(define-fun c{n} () (_ BitVec 1) (bvor (bvand (bvxor x{n} y{n}) c{n - 1}) (bvand x{n} y{n})))
""",
      file=f,
  )


def get_concat_result_bits(n):
  """Creates a string of smt2 concat ops to combine the bits of the output sum.

  Args:
    n: An integer, the number of bits in the output bitvector.

  Returns:
    The combined string of concats.
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
  """Write the final addition output by concatenating all of the output bits.

  Args:
    n: An integer, the number of bits in the output bitvector.
    f: The file to write into.
  """
  print(
      f"""; Concatenate s bits to create sum
(define-fun sum () (_ BitVec {n}) {get_concat_result_bits(n)})
""",
      file=f,
  )


def assert_and_check_sat(n, f):
  """Write an (unsatisfiable) assertion and tell the solver to check it.

  Write the assertion that the output of the 'by-hand' addition (called
  sum), does not equal the output of the builtin bvadd operation, and tell
  the solver to check the satisfiability.

  Args:
    n: An integer, the number of bits in the output bitvector.
    f: The file to write into.
  """
  print(
      f"""; Compare {n}-bit adder result and internal addition and solve
(assert (not (= sum (bvadd x y))))
(check-sat)""",
      file=f,
  )


def n_bit_add_existing_file(n, f):
  """Given a file, write a multiplication proof into it with n-bit arguments.

  Args:
    n: An integer, the number of bits for the input and output bitvectors.
    f: The file to write the proof into.
  """
  description_comments(n, f)
  logic_and_variables(n, f)
  half_adder(0, f)
  for i in range(1, n):
    full_adder(i, f)
  make_sum(n, f)
  assert_and_check_sat(n, f)


def n_bit_add_new_file(n):
  """Create a new file, and write a multiplication proof with n-bit arguments.

  Args:
    n: An integer, the number of bits for the input and output bitvectors.
  """
  with gfile.open(f"add_2x{n}.smt2", "w+") as f:
    n_bit_add_existing_file(n, f)


def main(argv):
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  else:
    for n_string in FLAGS.N:
      n_bit_add_new_file(int(n_string))


if __name__ == "__main__":
  app.run(main)
