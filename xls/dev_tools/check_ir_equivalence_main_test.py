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
from typing import Tuple, Union

from absl.testing import absltest
from xls.common import runfiles


ADD_IR = """package add

top fn add(x: bits[32], y: bits[32]) -> bits[32] {
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
        flow_control=ready_valid, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only,
        flow_control=ready_valid, metadata="")

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
        flow_control=ready_valid, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only,
        flow_control=ready_valid, metadata="")

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
        flow_control=ready_valid, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only,
        flow_control=ready_valid, metadata="")

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

_CHECK_EQUIV = runfiles.get_path("xls/dev_tools/check_ir_equivalence_main")


class CheckIrEquivalenceMainTest(absltest.TestCase):

  def _check_equiv(
      self, ir_a: str, ir_b: str, act_count: Union[None, int] = 0
  ) -> Tuple[bool, str]:
    res = subprocess.run(
        [
            _CHECK_EQUIV,
            self.create_tempfile(content=ir_a).full_path,
            self.create_tempfile(content=ir_b).full_path,
            f"--activation_count={act_count}",
        ],
        check=False,
        stdout=subprocess.PIPE,
    )
    return res.returncode == 0, res.stdout.decode("utf8")

  def test_detects_different_functions(self):
    res, msg = self._check_equiv(ADD_IR, NOT_ADD_IR)
    self.assertFalse(res)
    self.assertIn("Verified NOT equivalent", msg)

  def test_detects_equiv_functions(self):
    res, msg = self._check_equiv(ADD_IR, NOT_NOT_ADD_IR)
    self.assertTrue(res)
    self.assertIn("Verified equivalent", msg)

  def test_detects_different_procs(self):
    res, msg = self._check_equiv(PROC_IR, NEG_PROC_IR, act_count=14)
    self.assertFalse(res)
    self.assertIn("Verified NOT equivalent", msg)

  def test_detects_equiv_procs(self):
    res, msg = self._check_equiv(PROC_IR, NEG_NEG_PROC_IR, act_count=14)
    self.assertTrue(res)
    self.assertIn("Verified equivalent", msg)


if __name__ == "__main__":
  absltest.main()
