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
r"""Creates an SMTLIB2 file with an n-bit shift equivalence proof.

This file receives an integer from --n and an integer from --chains, and creates
an SMTLIB2 file containing an (n)-bit shifter equivalence proof with "chains"
nested operations. For example, to create an smt2 file for 4-bit multiplication
and 5 nested shift operations, we can run (after building):
$ bazel-bin/xls/experimental/smtlib/n_bit_nested_shift_generator \
    --n=4 --chains=5

Once an smt2 file is created, we can run:
$ <solver command> <filename>

The created SMTLIB2 file asserts that the shifter and the builtin shift DO NOT
produce the same result, so the output we expect to see is:
$ unsat

meaning the shifter and the builtin shift never produce different
results. They are logically equivalent.
"""

from xls.common import gfile


def description_comments(n, shifts, f):
  """Writes comments to the top of the output file describing what it does.

  Write comments to the top of the smt2 file describing the proof it contains:
  the operation, how many bits in the arguments, and how many operations.

  Args:
    n: An integer, the number of bits in each input bitvector.
    shifts: An integer, the number of nested shift operations.
    f: The file to write into.
  """
  print(
      f"""; The following SMT-LIB verifies that a chain of {shifts} {n}-bit shifts
; is equivalent to SMT-LIB's built in bit-vector shift.
""",
      file=f,
  )


def logic_and_variables(n, shifts, f):
  """Sets the logic for the smt2 file, and declare/define variables.

  Write the set-logic for the proof (QF_BV is the bitvector logic), declare the
  input bitvector variables, and define variables for their indices. Note that
  x_i_j corresponds to index j of the i-th input bitvector.

  Args:
    n: An integer, the number of bits in each input bitvector.
    shifts: An integer, the number of nested shift operations.
    f: The file to write into.
  """
  print(
      """(set-logic QF_BV)

; Declare bit-vectors and proxies for indices""",
      file=f,
  )
  for i in range(shifts + 1):
    print(f"(declare-fun x_{i} () (_ BitVec {n}))", file=f)
    for j in range(n):
      print(
          f"(define-fun x_{i}_{j} () (_ BitVec 1) ((_ extract {j} {j}) x_{i}))",
          file=f,
      )
  print("", file=f)


def get_shift_bit(char, var):
  """Returns the var argument if char is '1', otherwise (bvnot var).

  Args:
    char: A string, either '0' or '1'
    var: A string, representing a variable we've defined in the smt2 file.

  Returns:
    The value or its inversion.
  """
  if char == "0":
    return f"(bvnot {var})"
  elif char == "1":
    return var
  else:
    raise ValueError("Bit string contained value that wasn't 0 or 1")


def get_shift_bits(n, bit_string, shift):
  """Given a bit-string, return an expression to map to the bit-string.

  Each index of the given bit-string corresponds to the index of an n-bit
  bit vector. This function returns those bits, using the variable for the bit
  if the corresponding index in the bit-string is '1', and using the negation
  of the variable if the index is '0'.

  Args:
    n: An integer, the number of bits in the bitvector
    bit_string: A string of '0's and '1's
    shift: An integer, the nested shift operation that we're currently at.

  Returns:
    The shift op to extract the desired bit.
  """
  bits = []
  var = f"x_{shift}" if shift == 0 else f"shl_{shift - 1}"
  for i in range(n - 1, -1, -1):
    bits.append(get_shift_bit(bit_string[n - i - 1], f"{var}_{i}"))
  return " ".join(bits)


def yield_shift_conjunctions(i, n, shift):
  """Yields all of the expressions from get_shift_bits, from 0 to i.

  This function iterates from 0 to i (inclusive), and represents each number
  in the iteration as an n-bit bit-string. For each of these bit-strings, it
  yields the bit-vector anding of all of the bits in the bit-vector at the shift
  we are at, negating bits whose corresponding index in the bit-string have a
  '0'.

  Args:
    i: An integer, the index we iterate up to.
    n: An integer, the number of bits in the bit-string
    shift: An integer, the nested shift we're at.
  """
  if n == 1:
    yield get_shift_bits(n, format(i, f"0{n}b"), shift)
  else:
    for step in range(i + 1):
      bit_string = format(step, f"0{n}b")
      yield "(bvand " + get_shift_bits(n, bit_string, shift) + ")"


def shift_index(i, n, shift, f):
  """Writes the definition of bit i of the output of shift (shift).

  Bit i of the output of shift (shift) is defined by what bits in the shifting
  bitvector are '1' and '0'. For every case, bit i of the output will be equal
  to a certain bit in the input bitvector that is being shifted.

  Args:
    i: An integer, the index we iterate up to.
    n: An integer, the number of bits in the bit-string
    shift: An integer, the nested shift we're at.
    f: The file to write into.
  """
  clauses = []
  j = i
  for conjunction in yield_shift_conjunctions(i, n, shift):
    clauses.append(f"(bvand x_{shift + 1}_{j} {conjunction})")
    j -= 1
  if i == 0:
    assignment = " ".join(clauses)
  else:
    assignment = f"(bvor {' '.join(clauses)})"
  print(f"(define-fun shl_{shift}_{i} () (_ BitVec 1) {assignment})", file=f)


def concat_shift_indices(n, shift):
  """Returns the concat expression of the bits at shift (shift).

  Args:
    n: An integer, the number of bits in the bit-string
    shift: An integer, the nested shift we're at.
  """
  concats = [f"shl_{shift}_0"]
  for i in range(1, n):
    rhs = concats[i - 1]
    concat = ["(concat", f"shl_{shift}_{i}", rhs + ")"]
    concats.append(" ".join(concat))
  return concats[-1]


def shift_level(n, shift, f):
  """Write the output of shift (shift), concatenating all of its bits together.

  Args:
    n: An integer, the number of bits in the bit-string
    shift: An integer, the nested shift we're at.
    f: The file to write into.
  """
  for i in range(n):
    shift_index(i, n, shift, f)
  print(
      f"\n(define-fun shl_{shift} () (_ BitVec {n})"
      f" {concat_shift_indices(n, shift)})\n",
      file=f,
  )


def get_nested_expression(shifts):
  """Returns a string representing the addition of all the input bitvectors.

  Args:
    shifts: An integer, the number of nested shift operations.
  """
  nested_expressions = []
  for i in range(shifts):
    rhs = "x_0" if i == 0 else nested_expressions[i - 1]
    expression = ["(bvshl", f"x_{i + 1}", rhs + ")"]
    nested_expressions.append(" ".join(expression))
  return nested_expressions[-1]


def assert_and_check_sat(n, shifts, f):
  """Writes an (unsatisfiable) assertion and tell the solver to check it.

  Write the assertion that the output of the 'by-hand' shift, shl_(adders - 1),
  does not equal the output of the builtin bvshl operation, and tell the solver
  to check the satisfiability.

  Args:
    n: An integer, the number of bits in each bitvector.
    shifts: An integer, the number of nested shift operations.
    f: The file to write into.
  """
  print(
      f"""; Compare {n}-bit shift result and internal shift and solve
(assert (not (= shl_{shifts - 1} {get_nested_expression(shifts)})))
(check-sat)""",
      file=f,
  )


def n_bit_nested_shift_existing_file(n, shifts, f):
  """Given a file, write an n-bit shift proof with a chain of (shifts) shifts.

  Args:
    n: An integer, the number of bits in each bitvector.
    shifts: An integer, the number of nested shift operations.
    f: The file to write into.
  """
  description_comments(n, shifts, f)
  logic_and_variables(n, shifts, f)
  for shift in range(shifts):
    shift_level(n, shift, f)
  assert_and_check_sat(n, shifts, f)


def n_bit_nested_shift_new_file(n, shifts):
  """Makes a new file and writes an n-bit shift proof.

  Args:
    n: An integer, the number of bits in each bitvector.
    shifts: An integer, the number of nested shift operations.
  """
  with gfile.open(f"shift{shifts}_2x{n}.smt2", "w") as f:
    n_bit_nested_shift_existing_file(n, shifts, f)
