#
# Copyright 2024 The XLS Authors
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

"""Basic functionality test for check_ir_equivalence_main."""

import subprocess
from typing import Optional, Tuple, Union

from absl.testing import absltest
from absl.testing import parameterized
from xls.common import runfiles


ADD_IR = """package add

top fn add(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
"""
# Version of ADD_IR that has no top function.
ADD_NO_TOP_IR = """package add_no_top

fn add(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
"""
NOT_ADD_IR = """package not_add

top fn not_add(x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  ret not.2: bits[32] = not(add.1)
}
"""
NOT_NOT_ADD_IR = """package not_not_add

top fn not_not_add(x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  not.2: bits[32] = not(add.1)
  ret not.3: bits[32] = not(not.2)
}
"""

PROC_IR = """package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only,
        flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only,
        flow_control=ready_valid)

top proc proc_tst(my_state: (), init={()}) {
  my_token: token = literal(value=token)
  rcv: (token, bits[32]) = receive(my_token, channel=in)
  data: bits[32] = tuple_index(rcv, index=1)
  rcv_token: token = tuple_index(rcv, index=0)
  send: token = send(rcv_token, data, channel=out)
  next (my_state)
}
"""

NEG_PROC_IR = """package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only,
        flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only,
        flow_control=ready_valid)

top proc neg_proc(my_state: (), init={()}) {
  my_token: token = literal(value=token)
  rcv: (token, bits[32]) = receive(my_token, channel=in)
  data: bits[32] = tuple_index(rcv, index=1)
  negate: bits[32] = neg(data)
  rcv_token: token = tuple_index(rcv, index=0)
  send: token = send(rcv_token, negate, channel=out)
  next (my_state)
}
"""

NEG_NEG_PROC_IR = """package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only,
        flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only,
        flow_control=ready_valid)

top proc neg_neg_proc(my_state: (), init={()}) {
  my_token: token = literal(value=token)
  rcv: (token, bits[32]) = receive(my_token, channel=in)
  data: bits[32] = tuple_index(rcv, index=1)
  negate: bits[32] = neg(data)
  negate2: bits[32] = neg(negate)
  rcv_token: token = tuple_index(rcv, index=0)
  send: token = send(rcv_token, negate2, channel=out)
  next (my_state)
}
"""

PROC_IR_WITH_ASSERT = """package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only,
        flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only,
        flow_control=ready_valid)

top proc proc_tst(my_state: (), init={()}) {
  my_token: token = literal(value=token)
  rcv: (token, bits[32]) = receive(my_token, channel=in)
  data: bits[32] = tuple_index(rcv, index=1)
  rcv_token: token = tuple_index(rcv, index=0)
  send: token = send(rcv_token, data, channel=out)
  zero: bits[32] = literal(value=0)
  nz: bits[1] = ne(data, zero)
  assert: token = assert(send, nz, message="foobar")
  next (my_state)
}
"""

SUBGRAPH_FN_A = """package subgraph_a

top fn f(x: bits[8], y: bits[8]) -> bits[8] {
  s: bits[8] = add(x, y)
  ret p: bits[8] = umul(s, s)
}
"""

SUBGRAPH_FN_B = """package subgraph_b

top fn f(x: bits[8], y: bits[8]) -> bits[8] {
  z: bits[8] = xor(x, x)
  s1: bits[8] = add(y, x)
  s2: bits[8] = add(s1, z)
  ret p: bits[8] = umul(s1, s2)
}
"""

SIMPLIFIED_NEQ_FN_A = """package neq_a

top fn f(x: bits[8], y: bits[8]) -> bits[8] {
  z: bits[8] = xor(x, x)
  ret r: bits[8] = add(z, y)
}
"""

SIMPLIFIED_NEQ_FN_B = """package neq_b

top fn f(x: bits[8], y: bits[8]) -> bits[8] {
  one: bits[8] = literal(value=1)
  ret r: bits[8] = add(y, one)
}
"""

PRIORITY_SEL_FN_A = """package prio_a

top fn f(op: bits[2], x: bits[8], y: bits[8]) -> bits[8] {
  z0: bits[2] = literal(value=0)
  z1: bits[2] = literal(value=1)
  c0: bits[1] = eq(op, z0)
  c1: bits[1] = eq(op, z1)
  sel_bits: bits[2] = concat(c1, c0)
  case0: bits[8] = add(x, y)
  case1: bits[8] = sub(x, y)
  dflt: bits[8] = literal(value=0)
  ret r: bits[8] = priority_sel(sel_bits, cases=[case0, case1], default=dflt)
}
"""

PRIORITY_SEL_FN_B = """package prio_b

top fn f(op: bits[2], x: bits[8], y: bits[8]) -> bits[8] {
  z0: bits[2] = literal(value=0)
  z1: bits[2] = literal(value=1)
  c0: bits[1] = eq(op, z0)
  c1: bits[1] = eq(op, z1)
  sel_bits: bits[2] = concat(c1, c0)
  case0: bits[8] = add(y, x)
  case1: bits[8] = sub(x, y)
  dflt: bits[8] = literal(value=0)
  ret r: bits[8] = priority_sel(sel_bits, cases=[case0, case1], default=dflt)
}
"""

STATEFUL_PROC_A = """package state_a

chan in_a(bits[8], id=0, kind=streaming, ops=receive_only,
          flow_control=ready_valid)
chan in_b(bits[8], id=1, kind=streaming, ops=receive_only,
          flow_control=ready_valid)
chan out(bits[8], id=2, kind=streaming, ops=send_only,
         flow_control=ready_valid)

top proc accum(st: bits[8], init={0}) {
  tkn: token = literal(value=token)
  rcv_a: (token, bits[8]) = receive(tkn, channel=in_a)
  rcv_b: (token, bits[8]) = receive(tkn, channel=in_b)
  a: bits[8] = tuple_index(rcv_a, index=1)
  b: bits[8] = tuple_index(rcv_b, index=1)
  prod: bits[8] = umul(a, b)
  next_st: bits[8] = add(st, prod)
  t0: token = tuple_index(rcv_a, index=0)
  t1: token = tuple_index(rcv_b, index=0)
  after: token = after_all(t0, t1)
  send_tkn: token = send(after, next_st, channel=out)
  next (next_st)
}
"""

STATEFUL_PROC_B = """package state_b

chan in_a(bits[8], id=0, kind=streaming, ops=receive_only,
          flow_control=ready_valid)
chan in_b(bits[8], id=1, kind=streaming, ops=receive_only,
          flow_control=ready_valid)
chan out(bits[8], id=2, kind=streaming, ops=send_only,
         flow_control=ready_valid)

top proc accum(st: bits[8], init={0}) {
  tkn: token = literal(value=token)
  rcv_a: (token, bits[8]) = receive(tkn, channel=in_a)
  rcv_b: (token, bits[8]) = receive(tkn, channel=in_b)
  a: bits[8] = tuple_index(rcv_a, index=1)
  b: bits[8] = tuple_index(rcv_b, index=1)
  z: bits[8] = xor(a, a)
  prod: bits[8] = umul(b, a)
  prod_z: bits[8] = add(prod, z)
  next_st: bits[8] = add(st, prod_z)
  t0: token = tuple_index(rcv_a, index=0)
  t1: token = tuple_index(rcv_b, index=0)
  after: token = after_all(t0, t1)
  send_tkn: token = send(after, next_st, channel=out)
  next (next_st)
}
"""

STATEFUL_PROC_NEQ = """package state_neq

chan in_a(bits[8], id=0, kind=streaming, ops=receive_only,
          flow_control=ready_valid)
chan in_b(bits[8], id=1, kind=streaming, ops=receive_only,
          flow_control=ready_valid)
chan out(bits[8], id=2, kind=streaming, ops=send_only,
         flow_control=ready_valid)

top proc accum(st: bits[8], init={0}) {
  tkn: token = literal(value=token)
  rcv_a: (token, bits[8]) = receive(tkn, channel=in_a)
  rcv_b: (token, bits[8]) = receive(tkn, channel=in_b)
  a: bits[8] = tuple_index(rcv_a, index=1)
  b: bits[8] = tuple_index(rcv_b, index=1)
  prod: bits[8] = umul(a, b)
  sum_st: bits[8] = add(st, prod)
  one: bits[8] = literal(value=1)
  next_st: bits[8] = add(sum_st, one)
  t0: token = tuple_index(rcv_a, index=0)
  t1: token = tuple_index(rcv_b, index=0)
  after: token = after_all(t0, t1)
  send_tkn: token = send(after, next_st, channel=out)
  next (next_st)
}
"""

_CHECK_EQUIV = runfiles.get_path("xls/dev_tools/check_ir_equivalence_main")


class CheckIrEquivalenceMainTest(parameterized.TestCase):

  def _run_check_equiv(
      self,
      ir_a: str,
      ir_b: str,
      act_count: Union[None, int] = 0,
      solver: Optional[str] = None,
      optimize_merged_ir: Optional[bool] = None,
      print_timing: Optional[bool] = None,
  ) -> Tuple[int, str, str]:
    """Returns `(returncode, output_message, stderr_message)`."""
    cmd = [
        _CHECK_EQUIV,
        self.create_tempfile(content=ir_a).full_path,
        self.create_tempfile(content=ir_b).full_path,
        f"--activation_count={act_count}",
    ]
    if solver is not None:
      cmd.append(f"--solver={solver}")
    if optimize_merged_ir is not None:
      cmd.append(
          f"--optimize_merged_ir={'true' if optimize_merged_ir else 'false'}"
      )
    if print_timing is not None:
      cmd.append(f"--print_timing={'true' if print_timing else 'false'}")
    res = subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout = res.stdout.decode("utf8")
    stderr = res.stderr.decode("utf8")
    return res.returncode, stdout, stderr

  def _check_equiv(
      self,
      ir_a: str,
      ir_b: str,
      act_count: Union[None, int] = 0,
      solver: Optional[str] = None,
      optimize_merged_ir: Optional[bool] = None,
  ) -> Tuple[bool, str, str]:
    """Returns `(success, output_message, stderr_message)`."""
    rc, stdout, stderr = self._run_check_equiv(
        ir_a,
        ir_b,
        act_count=act_count,
        solver=solver,
        optimize_merged_ir=optimize_merged_ir,
    )
    return rc == 0, stdout, stderr

  @parameterized.parameters(None, "z3", "bitwuzla")
  def test_detects_different_functions(self, solver: Optional[str]):
    res, stdout, _ = self._check_equiv(ADD_IR, NOT_ADD_IR, solver=solver)
    self.assertFalse(res)
    self.assertIn("Verified NOT equivalent", stdout)

  @parameterized.parameters(
      (None, False),
      (None, True),
      ("z3", True),
      ("bitwuzla", True),
  )
  def test_ignores_assert(
      self, solver: Optional[str], optimize_merged_ir: bool
  ):
    res, stdout, _ = self._check_equiv(
        PROC_IR_WITH_ASSERT,
        PROC_IR,
        act_count=3,
        solver=solver,
        optimize_merged_ir=optimize_merged_ir,
    )
    self.assertTrue(res)
    self.assertIn("Verified equivalent", stdout)

  @parameterized.parameters(None, "z3", "bitwuzla")
  def test_detects_equiv_functions(self, solver: Optional[str]):
    res, stdout, _ = self._check_equiv(ADD_IR, NOT_NOT_ADD_IR, solver=solver)
    self.assertTrue(res)
    self.assertIn("Verified equivalent", stdout)

  @parameterized.parameters(None, "z3", "bitwuzla")
  def test_detects_different_procs(self, solver: Optional[str]):
    res, stdout, _ = self._check_equiv(
        PROC_IR, NEG_PROC_IR, act_count=14, solver=solver
    )
    self.assertFalse(res)
    self.assertIn("Verified NOT equivalent", stdout)

  @parameterized.parameters(None, "z3", "bitwuzla")
  def test_detects_equiv_procs(self, solver: Optional[str]):
    res, stdout, _ = self._check_equiv(
        PROC_IR, NEG_NEG_PROC_IR, act_count=14, solver=solver
    )
    self.assertTrue(res)
    self.assertIn("Verified equivalent", stdout)

  @parameterized.parameters("z3", "bitwuzla")
  def test_optimize_merged_ir_equiv_functions(self, solver: str):
    rc, stdout, _ = self._run_check_equiv(
        SUBGRAPH_FN_A,
        SUBGRAPH_FN_B,
        solver=solver,
        optimize_merged_ir=True,
    )
    self.assertEqual(rc, 0)
    self.assertEqual(stdout, "Verified equivalent\n")

  @parameterized.parameters("z3", "bitwuzla")
  def test_optimize_merged_ir_neq_functions(self, solver: str):
    rc, stdout, _ = self._run_check_equiv(
        SIMPLIFIED_NEQ_FN_A,
        SIMPLIFIED_NEQ_FN_B,
        solver=solver,
        optimize_merged_ir=True,
    )
    self.assertEqual(rc, 255)
    self.assertIn("Verified NOT equivalent; results differ for input: ", stdout)
    self.assertIn("0x", stdout)

  @parameterized.parameters("z3", "bitwuzla")
  def test_optimize_merged_ir_equiv_procs_unrolled(self, solver: str):
    rc, stdout, _ = self._run_check_equiv(
        STATEFUL_PROC_A,
        STATEFUL_PROC_B,
        act_count=4,
        solver=solver,
        optimize_merged_ir=True,
    )
    self.assertEqual(rc, 0)
    self.assertEqual(stdout, "Verified equivalent\n")

  @parameterized.parameters("z3", "bitwuzla")
  def test_optimize_merged_ir_neq_procs_unrolled(self, solver: str):
    rc, stdout, _ = self._run_check_equiv(
        STATEFUL_PROC_A,
        STATEFUL_PROC_NEQ,
        act_count=3,
        solver=solver,
        optimize_merged_ir=True,
    )
    self.assertEqual(rc, 255)
    self.assertIn("Verified NOT equivalent; results differ for input: ", stdout)
    self.assertIn("-> bits[", stdout)

  @parameterized.parameters("z3", "bitwuzla")
  def test_optimize_merged_ir_priority_sel_invariant_asserts(self, solver: str):
    rc, stdout, _ = self._run_check_equiv(
        PRIORITY_SEL_FN_A,
        PRIORITY_SEL_FN_B,
        solver=solver,
        optimize_merged_ir=True,
    )
    self.assertEqual(rc, 0)
    self.assertEqual(stdout, "Verified equivalent\n")

  @parameterized.parameters(False, True)
  def test_print_timing_writes_only_to_stderr(self, optimize_merged_ir: bool):
    rc, stdout, stderr = self._run_check_equiv(
        SUBGRAPH_FN_A,
        SUBGRAPH_FN_B,
        solver="z3",
        optimize_merged_ir=optimize_merged_ir,
        print_timing=True,
    )
    self.assertEqual(rc, 0)
    self.assertEqual(stdout, "Verified equivalent\n")
    for key in (
        "inlining_ms=",
        "merge_ms=",
        "opt_pipeline_ms=",
        "merge_and_opt_ms=",
        "solver_ms=",
        "total_ms=",
    ):
      self.assertIn(key, stderr)

  def test_lhs_has_no_top(self):
    res, _, stderr = self._check_equiv(ADD_IR, ADD_NO_TOP_IR)
    self.assertFalse(res)
    self.assertIn("Package has no top entity: add_no_top", stderr)


if __name__ == "__main__":
  absltest.main()
