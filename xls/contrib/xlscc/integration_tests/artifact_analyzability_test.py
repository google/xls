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

"""Tests that validate the outputs of XLScc are able to be analyzed in expected ways."""

import re

from absl.testing import absltest
from xls.common import runfiles
from xls.ir import xls_ir_interface_pb2 as ir_interface

_INPUT_UNSIGNED_LOOP_IR_FILE = "xls/contrib/xlscc/integration_tests/simple_unsigned_pipelined_loop_fsm.ir.interface.binpb"
_OUTPUT_UNSIGNED_LOOP_IR_FILE = "xls/contrib/xlscc/integration_tests/simple_unsigned_pipelined_loop_fsm.opt.ir.interface.binpb"


def _get_interface_proto(path: str) -> ir_interface.PackageInterfaceProto:
  return ir_interface.PackageInterfaceProto.FromString(
      runfiles.get_contents_as_bytes(path)
  )


class ArtifactAnalyzabilityTest(absltest.TestCase):

  def _for_loop_state_element(self, base_name: str) -> re.Pattern[str]:
    """Returns a regex that matches a for loop state element.

    Args:
      base_name: The base name of the for loop state element.

    Returns:
      A regex that matches a for loop state element.
    """
    return re.compile(f"^__for_[0-9]+_{base_name}(__.*)?")

  def test_unsigned_pipelined_loop_is_narrowed(self):
    input_interface = _get_interface_proto(_INPUT_UNSIGNED_LOOP_IR_FILE)
    output_interface = _get_interface_proto(_OUTPUT_UNSIGNED_LOOP_IR_FILE)
    self.assertLen(input_interface.procs, 1)
    self.assertLen(output_interface.procs, 1)
    # Check 'i' loop variable narrowed.
    target_element = self._for_loop_state_element("i")
    initial_for_loop_elements = [
        nv
        for nv in input_interface.procs[0].state
        if target_element.match(nv.name)
    ]
    final_for_loop_elements = [
        nv
        for nv in output_interface.procs[0].state
        if target_element.match(nv.name)
    ]
    self.assertLen(initial_for_loop_elements, 1)
    self.assertLen(final_for_loop_elements, 1)
    self.assertEqual(initial_for_loop_elements[0].type.bit_count, 32)
    self.assertEqual(final_for_loop_elements[0].type.bit_count, 3)

  def test_unsigned_pipelined_loop_non_loop_variable_not_narrowed(self):
    input_interface = _get_interface_proto(_INPUT_UNSIGNED_LOOP_IR_FILE)
    output_interface = _get_interface_proto(_OUTPUT_UNSIGNED_LOOP_IR_FILE)
    # Check 'a' variable stays wide.
    a_element = self._for_loop_state_element("a")
    initial_a_for_loop_elements = [
        nv
        for nv in input_interface.procs[0].state
        if a_element.match(nv.name)
    ]
    final_a_for_loop_elements = [
        nv
        for nv in output_interface.procs[0].state
        if a_element.match(nv.name)
    ]
    self.assertLen(initial_a_for_loop_elements, 1)
    self.assertLen(final_a_for_loop_elements, 1)
    self.assertEqual(initial_a_for_loop_elements[0].type.bit_count, 32)
    self.assertEqual(final_a_for_loop_elements[0].type.bit_count, 32)


if __name__ == "__main__":
  absltest.main()
