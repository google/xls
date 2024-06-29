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
"""Creates an SMTLIB2 file containing a chained series of muls.

This file receives an integer from --n and an integer from --chains, and creates
an SMTLIB2 file containing an (n)-bit multiplier equivalence proof with (chains)
nested mul operations. For example, to create an smt2 file for 4-bit
multiplication and 5 nested mul operations, we can run (after building)
$ bazel-bin/xls/experimental/smtlib/n_bit_nested_mul_generator --n=4 --chains=5

Once an smt2 file is created, we can run:
$ <solver command> <filename>

The created smt2 file asserts that the multiplier and the builtin multiplication
DO NOT produce the same result, so the output we expect to see is:
$ unsat

meaning the multiplier and the builtin multiplication never produce different
results. They are logically equivalent.
"""

from xls.common import gfile


def description_comments(n, muls, f):
  """Write comments to the top of the file describing what it does.

  Write comments to the top of the smt2 file describing the proof it contains:
  the operation, how many bits in the arguments, and how many operations.

  Args:
    n: An integer, the number of bits in each input bitvector.
    muls: An integer, the number of nested addition operations.
    f: The file to write into.
  """
  print(
      f"""; The following SMT-LIB verifies that a chain of {muls} {n}-bit multiplier
; is equivalent to SMT-LIB's built in bit-vector multiplication.
""",
      file=f,
  )


def logic_and_variables(n, muls, f):
  """Set the logic for the smt2 file, and declare/define variables.

  Write the set-logic for the proof (QF_BV is the bitvector logic), declare the
  input bitvector variables, and define variables for their indices. Note that
  x_i_j corresponds to index j of the i-th input bitvector.

  Args:
    n: An integer, the number of bits in each input bitvector.
    muls: An integer, the number of nested addition operations.
    f: The file to write into.
  """
  print(
      """(set-logic QF_BV)

; Declare bit-vectors and proxies for indices""",
      file=f,
  )
  for i in range(muls + 1):
    print(f"(declare-fun x_{i} () (_ BitVec {n}))", file=f)
    for j in range(n):
      print(
          f"(define-fun x_{i}_{j} () (_ BitVec 1) ((_ extract {j} {j}) x_{i}))",
          file=f,
      )
  print("", file=f)


def get_concat_level_bits(i, n, mul):
  """Create a string combining the bits of the current mul.

  Combine the bits of the multiplication of the current variable (at mul) by the
  i-th index of the previous variable.

  Args:
    i: An integer, the index of the previous variable.
    n: An integer, the number of bits in the bitvectors.
    mul: An integer, the index of the nested mul we're at.

  Returns:
    The resulting concat string.
  """
  concats = []
  if i > 0:
    concats.append(f"(concat m_{mul}_{i}_{i} #b{'0' * i})")
  else:
    concats.append(f"m_{mul}_0_0")
  if i < (n - 1):
    for j in range(i + 1, n):
      rhs = concats[j - i - 1]
      concat = ["(concat", f"m_{mul}_{i}_{j}", rhs + ")"]
      concats.append(" ".join(concat))
  return concats[-1]


def mul_level(i, n, mul, f):
  """Define index i of the (mul)-th multiplication.

  Mutiply the current variable by the i-th index of the previous variable. This
  is done by evaluating each of the output bits with a boolean expression and
  then concatenating them together.

  Args:
    i: An integer, the index of the previous variable.
    n: An integer, the number of bits in the bitvectors.
    mul: An integer, the index of the nested mul we're at.
    f: The file to write into.
  """
  prev_var = f"m_{mul - 1}" if mul > 0 else f"x_{mul}"
  print(
      f"; Multiply x_{mul + 1} by {prev_var}_{i}, shifting x_{mul + 1} bits"
      " accordingly",
      file=f,
  )
  for j in range(i, n):
    print(
        f"""(define-fun m_{mul}_{i}_{j} () (_ BitVec 1) (bvand x_{mul + 1}_{j - i} ((_ extract {i} {i}) {prev_var})))""",
        file=f,
    )
  print(
      f"""\n; Concatenate m_{mul}_{i} bits to create mul at level {mul}_{i}
(define-fun m_{mul}_{i} () (_ BitVec {n}) {get_concat_level_bits(i, n, mul)})""",
      file=f,
  )


def get_all_levels(n, mul):
  """Returns all the indices of the bitvector at (mul) to be added together.

  Args:
    n: An integer, the number of bits in the bitvector
    mul: An integer, the index of the nested mul we're at.
  """
  bits = []
  for i in range(n - 1, -1, -1):
    bits.append(f"m_{mul}_{i}")
  return " ".join(bits)


def make_mul(n, mul, f):
  """Create the output of the (mul)-th multiplication.

  Get the bitvectors for the multiplication at each index using get_all_levels,
  and add them all together to produce the result at the (mul)-th level.

  Args:
    n: An integer, the number of bits in the bitvector
    mul: An integer, the index of the nested mul we're at.
    f: The file to write into.
  """
  if n == 1:
    bv_string = f"{get_all_levels(n, mul)}"
  else:
    bv_string = f"(bvadd {get_all_levels(n, mul)})"
  print(
      f"""; Add all m bit-vectors to create mul
(define-fun m_{mul} () (_ BitVec {n}) {bv_string})
""",
      file=f,
  )


def get_nested_expression(muls):
  """Returns a string representing the multiplication of all the inputs.

  Args:
    muls: An integer, represents the number of nested mul operations.
  """
  nested_expressions = []
  for i in range(muls):
    rhs = "x_0" if i == 0 else nested_expressions[i - 1]
    expression = ["(bvmul", f"x_{i + 1}", rhs + ")"]
    nested_expressions.append(" ".join(expression))
  return nested_expressions[-1]


def assert_and_check_sat(muls, f):
  """Writes an (unsatisfiable) assertion and tell the solver to check it.

  Write the assertion that the output of the 'by-hand' multiplication,
  m_(muls - 1), does not equal the output of the builtin bvmul operation, and
  tell the solver to check the satisfiability.

  Args:
    muls: An integer, the number of nested multiplication operations.
    f: The file to write into.
  """
  print(
      f"""; Assert and solve
(assert (not (= m_{muls - 1} {get_nested_expression(muls)})))
(check-sat)""",
      file=f,
  )


def n_bit_nested_mul_existing_file(n, muls, f):
  """Writes out an n-bit multiplication proof with a chain of (muls) muls.

  Args:
    n: An integer, the number of bits in each bitvector.
    muls: An integer, the number of nested multiplication operations.
    f: The file to write into.
  """
  description_comments(n, muls, f)
  logic_and_variables(n, muls, f)
  for mul in range(muls):
    for i in range(n):
      mul_level(i, n, mul, f)
    make_mul(n, mul, f)
  assert_and_check_sat(muls, f)


def n_bit_nested_mul_new_file(n, muls):
  """Makes a new file and write an n-bit multiplication [chain] proof.

  Args:
    n: An integer, the number of bits in each bitvector.
    muls: An integer, the number of nested multiplication operations.
  """
  with gfile.open(f"mul{muls}_2x{n}.smt2", "w+") as f:
    n_bit_nested_mul_existing_file(n, muls, f)
