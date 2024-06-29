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
"""Creates an SMTLIB2 file containing an n-bit adder equivalence proof.

This file receives an integer from --n and an integer from --chains, and creates
an SMTLIB2 file containing an (n)-bit adder equivalence proof with (chains)
nested add operations. For example, to create an smt2 file for 4-bit addition
and 5 nested add operations, we can run (after building)

$ bazel-bin/xls/experimental/smtlib/n_bit_nested_add_generator --n=4 --chains=5

Once an smt2 file is created, we can run:

$ <solver command> <filename>

The created smt2 file asserts that the adder and the builtin addition DO NOT
produce the same result, so the output we expect to see is:

$ unsat

meaning the adder and the builtin addition never produce different
results. They are logically equivalent.
"""

from xls.common import gfile


def description_comments(n, adders, f):
  """Write comments to the top of the file describing what it does.

  Write comments to the top of the smt2 file describing the proof it contains:
  the operation, how many bits in the arguments, and how many operations.

  Args:
    n: An integer, the number of bits in each input bitvector.
    adders: An integer, the number of nested addition operations.
    f: The file to write into.
  """
  print(
      f"""; The following SMT-LIB verifies that a chain of {adders} {n}-bit
; adder is equivalent to SMT-LIB's built in bit-vector addition.
""",
      file=f,
  )


def logic_and_variables(n, adders, f):
  """Set the logic for the smt2 file, and declare/define variables.

  Write the set-logic for the proof (QF_BV is the bitvector logic), declare the
  input bitvector variables, and define variables for their indices. Note that
  x_i_j corresponds to index j of the i-th input bitvector.

  Args:
    n: An integer, the number of bits in each input bitvector.
    adders: An integer, the number of nested addition operations.
    f: The file to write into.
  """
  print(
      """(set-logic QF_BV)

; Declare bit-vectors and proxies for indices""",
      file=f,
  )
  for i in range(adders + 1):
    print(f"(declare-fun x_{i} () (_ BitVec {n}))", file=f)
    for j in range(n):
      print(
          f"(define-fun x_{i}_{j} () (_ BitVec 1) ((_ extract {j} {j}) x_{i}))",
          file=f,
      )
  print("", file=f)


def half_adder(bit, adder, f):
  """Write a half adder for adder (adder) in the chain at bit (bit).

  Args:
    bit: An integer, represents the current index.
    adder: An integer, represents the index of the add that we're at in the
      chain.
    f: The file to write into.
  """
  prev_var = f"s_{adder - 1}" if adder > 0 else f"x_{adder}"
  print(
      f"""; Half adder for bit {bit} at adder {adder}
(define-fun s_{adder}_{bit} () (_ BitVec 1) (bvxor x_{adder + 1}_{bit} {prev_var}_{bit}))
(define-fun c_{adder}_{bit} () (_ BitVec 1) (bvand x_{adder + 1}_{bit} {prev_var}_{bit}))
""",
      file=f,
  )


def full_adder(bit, adder, f):
  """Write a full adder for adder "adder" in the chain at bit "bit".

  Args:
    bit: An integer, represents the current index.
    adder: An integer, represents the index of the add that we're at in the
      chain.
    f: The file to write into.
  """
  prev_var = f"s_{adder - 1}" if adder > 0 else f"x_{adder}"
  print(
      f"""; Full adder for bit {bit} at adder {adder}
(define-fun s_{adder}_{bit} () (_ BitVec 1) (bvxor c_{adder}_{bit - 1} (bvxor x_{adder + 1}_{bit} {prev_var}_{bit})))
(define-fun c_{adder}_{bit} () (_ BitVec 1) (bvor (bvand (bvxor x_{adder + 1}_{bit} {prev_var}_{bit}) c_{adder}_{bit - 1}) (bvand x_{adder + 1}_{bit} {prev_var}_{bit})))
""",
      file=f,
  )


def get_concat_level_bits(n, adder):
  """Create a string of concat ops for the bits of the sum at the given adder.

  Args:
    n: An integer, the number of bits to concatenate
    adder: An integer, represents the index of the add that we're at in the
      chain.

  Returns:
    The chain of concats.
  """
  concats = [f"s_{adder}_0"]
  for i in range(1, n):
    rhs = concats[i - 1]
    concat = ["(concat", f"s_{adder}_{i}", rhs + ")"]
    concats.append(" ".join(concat))
  return concats[-1]


def level_sum(n, adder, f):
  """Writes the result sum at adder "adder".

  Writes the sum, concatenating the output bits using get_concat_level_bits.
  Args:
    n: An integer, the number of bits for the bitvector representing the sum.
    adder: An integer, represents the index of the add that we're at in the
      chain.
    f: The file to write into.
  """
  print(
      f"""; Concatenate s_{adder} bits to create sum at level {adder}
(define-fun s_{adder} () (_ BitVec {n}) {get_concat_level_bits(n, adder)})
""",
      file=f,
  )


def get_nested_expression(adders):
  """Return a string representing the addition of all the input bitvectors.

  Args:
    adders: An integer, the number of nested addition operations.

  Returns:
    The full addition string.
  """
  nested_expressions = []
  for i in range(adders):
    rhs = "x_0" if i == 0 else nested_expressions[i - 1]
    expression = ["(bvadd", f"x_{i + 1}", rhs + ")"]
    nested_expressions.append(" ".join(expression))
  return nested_expressions[-1]


def assert_and_check_sat(n, adders, f):
  """Writes an (unsatisfiable) assertion and tell the solver to check it.

  Write the assertion that the output of the 'by-hand' addition, s_(adders - 1),
  does not equal the output of the builtin bvadd operation, and tell the solver
  to check the satisfiability.

  Args:
    n: An integer, the number of bits in each bitvector.
    adders: An integer, the number of nested addition operations.
    f: The file to write into.
  """
  print(
      f"""; Compare {n}-bit adder result and internal addition and solve
(assert (not (= s_{adders - 1} {get_nested_expression(adders)})))
(check-sat)""",
      file=f,
  )


def n_bit_nested_add_existing_file(n, adders, f):
  """Given a file, write an n-bit addition proof with a chain of (adders) adds.

  Args:
    n: An integer, the number of bits in each bitvector.
    adders: An integer, the number of nested addition operations.
    f: The file to write into.
  """
  description_comments(n, adders, f)
  logic_and_variables(n, adders, f)
  for adder in range(adders):
    half_adder(0, adder, f)
    for i in range(1, n):
      full_adder(i, adder, f)
    level_sum(n, adder, f)
  assert_and_check_sat(n, adders, f)


def n_bit_nested_add_new_file(n, adders):
  """Make a new file and write an n-bit addition proof.

  Args:
    n: An integer, the number of bits in each bitvector.
    adders: An integer, the number of nested addition operations.
  """
  with gfile.open(f"add{adders}_2x{n}.smt2", "w") as f:
    n_bit_nested_add_existing_file(n, adders, f)
