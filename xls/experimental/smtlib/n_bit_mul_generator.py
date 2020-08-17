# Lint as: python3
#
# Copyright 2020 Google LLC
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

# pylint: disable=bad-continuation,missing-function-docstring
"""N-bit multiplier equivalence-proof generator.

This file receives a list of numbers from argv[1:], and creates an smt2
file containing an n-bit multiplier equivalence proof for each number n.
Instructions for using CVC4 can be found in cvc4.md. Once an smt2 file is
created, we can run:

$ cvc4 <filename>

The created smt2 file asserts that the multiplier and the builtin addition DO
NOT produce the same result, so the output we expect to see is:

$ unsat

meaning the multiplier and the builtin multiplication never produce different
results. They are logically equivalent.
"""

import sys


def description_comments(n, f):
  print(
      f"""; The following SMT-LIB verifies that a {n}-bit multiplier is equivalent
; to CVC4's built in bit-vector multiplication.
""",
      file=f)


def logic_and_variables(n, f):
  print(
      f"""(set-logic ALL)

; Declare bit-vectors and proxies for indices
(declare-fun x () (_ BitVec {n}))
(declare-fun y () (_ BitVec {n}))""",
      file=f)
  for i in range(n):
    for var in ["x", "y"]:
      print(
          f"(define-fun {var}{i} () (_ BitVec 1) ((_ extract {i} {i}) {var}))",
          file=f)
  print("", file=f)


def get_mul_level_bits(i, n):
  bits = []
  for j in range(n - 1, i - 1, -1):
    bits.append(f"m{i}_{j}")
  if i > 0:
    bits.append("#b" + ("0" * i))
  return " ".join(bits)


def mul_level(i, n, f):
  print(f"; Multiply x by y{i}, shifting x bits accordingly", file=f)
  for j in range(i, n):
    print(
        f"(define-fun m{i}_{j} () (_ BitVec 1) (bvand x{j - i} y{i}))", file=f)
  print(
      f"(define-fun m{i} () (_ BitVec {n}) "
      f"(concat {get_mul_level_bits(i, n)}))\n",
      file=f)


def get_result_bits(n):
  bits = []
  for i in range(n - 1, -1, -1):
    bits.append(f"m{i}")
  return " ".join(bits)


def make_mul(n, f):
  print(
      f"""; Add all m bit-vectors to create mul
(define-fun mul () (_ BitVec {n}) (bvadd {get_result_bits(n)}))
""",
      file=f)


def assert_and_check_sat(f):
  print(
      """; Assert and solve
(assert (not (= mul (bvmul x y))))
(check-sat)""",
      file=f)


def n_bit_mul_existing_file(n, f):
  description_comments(n, f)
  logic_and_variables(n, f)
  for i in range(n):
    mul_level(i, n, f)
  make_mul(n, f)
  assert_and_check_sat(f)


def n_bit_mul_new_file(n):
  with open(f"mul_2x{n}.smt2", "w+") as f:
    n_bit_mul_existing_file(n, f)


def main():
  for n in sys.argv[1:]:
    n_bit_mul_new_file(int(n))


if __name__ == "__main__":
  main()
