#
# Copyright 2025 The XLS Authors
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

"""Simple model of the bit-count query engine calculation for add/sub to prove its valid."""

from collections.abc import Sequence
import contextlib
import dataclasses
import sys

from absl import app
from absl import flags

import z3


_BIT_WIDTH = flags.DEFINE_integer(
    "bit_width",
    default=8,
    help="Bit width to prove leading bit counts for",
)

_PRINT_SMTLIB = flags.DEFINE_bool(
    "print_smtlib", default=False, help="Print SMT lib model/proof"
)


@contextlib.contextmanager
def _scope(s):
  """Creates a new z3 scope for the solver."""
  s.push()
  yield s
  s.pop()


@dataclasses.dataclass(frozen=True)
class Context:
  """Context for the z3 solver, including functions and the solver itself."""

  s: z3.Solver
  max_fn: z3.Function
  min_fn: z3.Function
  leading_ones: z3.Function

  def eval(self, v):
    """Evaluates a z3 value in the current model."""
    return self.s.model().evaluate(v)


def _make_max(s: z3.Solver):
  """Creates a max function in the given solver."""
  max_fn = z3.Function(
      "max",
      z3.IntSort(),
      z3.IntSort(),
      z3.IntSort(),
  )
  a, b = z3.Consts("a b", z3.IntSort())
  s.add(z3.ForAll([a, b], max_fn(a, b) == z3.If(a > b, a, b)))
  return max_fn


def _make_min(s: z3.Solver):
  """Creates a min function in the given solver."""
  min_fn = z3.Function(
      "min",
      z3.IntSort(),
      z3.IntSort(),
      z3.IntSort(),
  )
  a, b = z3.Consts("a b", z3.IntSort())
  s.add(z3.ForAll([a, b], min_fn(a, b) == z3.If(a < b, a, b)))
  return min_fn


def _make_count_leading_zeros(ctx: Context):
  cnt_zero_bits = z3.Function(
      "count_zero_bits", z3.BitVecSort(_BIT_WIDTH.value), z3.IntSort()
  )
  cnt_arg = z3.Const("zero_cnt_arg", z3.BitVecSort(_BIT_WIDTH.value))
  ctx.s.add(
      z3.ForAll([cnt_arg], cnt_zero_bits(cnt_arg) == ctx.leading_ones(~cnt_arg))
  )
  return cnt_zero_bits


def _make_count_sign_bits(ctx: Context, leading_zeros=None):
  """Creates a count_sign_bits function in the given context."""
  cnt_sign_bits = z3.Function(
      "count_sign_bits", z3.BitVecSort(_BIT_WIDTH.value), z3.IntSort()
  )
  cnt_arg = z3.Const("sign_cnt_arg", z3.BitVecSort(_BIT_WIDTH.value))
  leading_zeros = (
      leading_zeros
      if leading_zeros is not None
      else _make_count_leading_zeros(ctx)
  )
  ctx.s.add(
      z3.ForAll(
          [cnt_arg],
          cnt_sign_bits(cnt_arg)
          == ctx.max_fn(ctx.leading_ones(cnt_arg), leading_zeros(cnt_arg)),
      )
  )
  return cnt_sign_bits


def _check_one_bit_type(
    ctx: Context, sum_val, lhs, rhs, get_bits, is_sign: bool
) -> bool:
  """Check that sum bit-cnt is always bounded by one less than its source counts.

  Args:
    ctx: The z3 context.
    sum_val: The sum value.
    lhs: The left hand side value.
    rhs: The right hand side value.
    get_bits: A function to get the bit count.
    is_sign: True if the bit count is for a sign bit.

  Returns:
    True if the check passes, False otherwise.
  """
  add_sub_estimation = ctx.max_fn(
      ctx.min_fn(get_bits(lhs), get_bits(rhs)) - 1, 1 if is_sign else 0
  )
  ctx.s.add(z3.Not(get_bits(sum_val) >= add_sub_estimation))
  if _PRINT_SMTLIB.value:
    print(ctx.s.sexpr())
  if ctx.s.check() != z3.unsat:
    print(
        f"    - Failed with lhs={ctx.eval(lhs)} (leading"
        f" {ctx.eval(get_bits(lhs))}), rhs={ctx.eval(rhs)} (leading"
        f" {ctx.eval(get_bits(rhs))}), sum={ctx.eval(sum_val)} (leading"
        f" {ctx.eval(get_bits(sum_val))}),"
        f" estimated_leading_bits={ctx.eval(add_sub_estimation)}, model:"
        f" {ctx.s.model()}"
    )
    return False
  else:
    print("    - proven")
    return True


def _check_add_sub_counts(
    ctx: Context, sum_val, lhs_val, rhs_val, is_sub: bool
) -> bool:
  """Checks that sum bit-cnt is always bounded by one less than its source counts.

  Args:
    ctx: The z3 context.
    sum_val: The sum value.
    lhs_val: The left hand side value.
    rhs_val: The right hand side value.
    is_sub: True if the operation is subtraction.

  Returns:
    True if the check passes, False otherwise.
  """
  res = True

  if not is_sub:
    print("  - Proving zero-bits-count constraints")
    with _scope(ctx.s):
      res = (
          _check_one_bit_type(
              ctx,
              sum_val,
              lhs_val,
              rhs_val,
              _make_count_leading_zeros(ctx),
              is_sign=False,
          )
          and res
      )
    print("  - Proving one-bits-count constraints")
    with _scope(ctx.s):
      res = (
          _check_one_bit_type(
              ctx, sum_val, lhs_val, rhs_val, ctx.leading_ones, is_sign=False
          )
          and res
      )
  print("  - Proving sign-bits-count constraints")
  with _scope(ctx.s):
    res = (
        _check_one_bit_type(
            ctx,
            sum_val,
            lhs_val,
            rhs_val,
            _make_count_sign_bits(ctx),
            is_sign=True,
        )
        and res
    )
  return res


def _check_neg_counts(ctx: Context, res, inp) -> bool:
  """Checks that the negative of a value has a correct bit count.

  Args:
    ctx: The z3 context.
    res: The result of the negation.
    inp: The input value.

  Returns:
    True if the check passes, False otherwise.
  """
  print("  - Checking sign bit count")
  with _scope(ctx.s):
    cnt_zeros = _make_count_leading_zeros(ctx)
    cnt_sign_bits = _make_count_sign_bits(ctx, leading_zeros=cnt_zeros)
    no_leading_zeros = cnt_zeros(inp) == 0
    estimation = ctx.max_fn(
        cnt_sign_bits(inp) - z3.If(no_leading_zeros, 1, 0), 1
    )
    ctx.s.add(z3.Not(cnt_sign_bits(res) >= estimation))
    if _PRINT_SMTLIB.value:
      print(ctx.s.sexpr())
    if ctx.s.check() != z3.unsat:
      print(
          f"    - Failed with lhs={ctx.eval(inp)} (leading"
          f" {ctx.eval(cnt_sign_bits(inp))}), res={ctx.eval(res)} (leading"
          f" {ctx.eval(cnt_sign_bits(res))}),"
          f" estimated_leading_bits={ctx.eval(estimation)}, model:"
          f" {ctx.s.model()}"
      )
      return False
    else:
      print("    - proven")
      return True


def _define_leading_ones(s: z3.Solver):
  """Define a leading_ones function in the given solver.

  Defines an unterpreted leading_ones function and adds constraints that 'n'-1s
  followed by a 0 bit in MSB order is 'n'.

  Args:
    s: The z3 solver to define the leading_ones function in.

  Returns:
    The leading_ones z3 function.
  """
  leading_ones = z3.Function(
      "leading_ones", z3.BitVecSort(_BIT_WIDTH.value), z3.IntSort()
  )
  # Constrain outputs.
  other_bits = z3.Const("other_bits_0", z3.BitVecSort(_BIT_WIDTH.value - 1))
  value_with_top_bit_unset = z3.Concat(z3.BitVecVal(0, 1), other_bits)
  s.add(
      z3.ForAll(
          [other_bits],
          0 == leading_ones(value_with_top_bit_unset),
      )
  )
  for i in range(1, _BIT_WIDTH.value - 1):
    # i set bits in a row.
    top_bits = (1 << i) - 1
    other_bits = z3.Const(
        f"other_bits_{i}", z3.BitVecSort(_BIT_WIDTH.value - (i + 1))
    )
    value_with_top_bits_set = z3.Concat(
        z3.BitVecVal(top_bits, i),
        z3.BitVecVal(0, 1),
        other_bits,
    )
    assert value_with_top_bits_set.sort() == z3.BitVecSort(
        _BIT_WIDTH.value
    ), f"Expected bitvec, got {value_with_top_bits_set.sort()}"
    s.add(z3.ForAll([other_bits], i == leading_ones(value_with_top_bits_set)))
  all_but_one_bit_set = z3.Concat(
      z3.BitVecVal((1 << _BIT_WIDTH.value - 1) - 1, _BIT_WIDTH.value - 1),
      z3.BitVecVal(0, 1),
  )
  s.add(_BIT_WIDTH.value - 1 == leading_ones(all_but_one_bit_set))
  all_bits_set = z3.BitVecVal((1 << _BIT_WIDTH.value) - 1, _BIT_WIDTH.value)
  s.add(_BIT_WIDTH.value == leading_ones(all_bits_set))
  return leading_ones


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")
  s = z3.Solver()
  max_fn = _make_max(s)
  min_fn = _make_min(s)
  leading_ones = _define_leading_ones(s)
  ctx = Context(s=s, max_fn=max_fn, min_fn=min_fn, leading_ones=leading_ones)
  lhs = z3.BitVec("lhs", _BIT_WIDTH.value)
  rhs = z3.BitVec("rhs", _BIT_WIDTH.value)
  res = True
  with _scope(ctx.s):
    print("Proving common constraints")
    if ctx.s.check() == z3.unsat:
      print(f"    - FAIL: common asssertions unsat: {ctx.s.sexpr()}")
      sys.exit(2)
    else:
      print("    - Proven")
  print("Proving Neg constraints:")
  res = _check_neg_counts(ctx, -lhs, lhs) and res
  print("Proving Add constraints:")
  res = _check_add_sub_counts(ctx, lhs + rhs, lhs, rhs, is_sub=False) and res
  print("Proving Sub constraints:")
  res = _check_add_sub_counts(ctx, lhs - rhs, lhs, rhs, is_sub=True) and res

  if not res:
    sys.exit(1)


if __name__ == "__main__":
  app.run(main)
