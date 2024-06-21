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
from xls.ir import xls_ir_interface_pb2

_INTEGRATION_TESTS_DIR = "xls/contrib/xlscc/integration_tests/"
_INPUT_UNSIGNED_LOOP_IR_FILE = f"{_INTEGRATION_TESTS_DIR}/simple_unsigned_pipelined_loop_fsm.ir.interface.binpb"
_OUTPUT_UNSIGNED_LOOP_IR_FILE = f"{_INTEGRATION_TESTS_DIR}/simple_unsigned_pipelined_loop_fsm.opt.ir.interface.binpb"
_INPUT_UNSIGNED_NESTED_LOOP_IR_FILE = f"{_INTEGRATION_TESTS_DIR}/simple_unsigned_nested_pipelined_loop_fsm.ir.interface.binpb"
_OUTPUT_UNSIGNED_NESTED_LOOP_IR_FILE = f"{_INTEGRATION_TESTS_DIR}/simple_unsigned_nested_pipelined_loop_fsm.opt.ir.interface.binpb"
_INPUT_SIGNED_LOOP_IR_FILE = (
    f"{_INTEGRATION_TESTS_DIR}/simple_pipelined_loop_fsm.ir.interface.binpb"
)
_OUTPUT_SIGNED_LOOP_IR_FILE = (
    f"{_INTEGRATION_TESTS_DIR}/simple_pipelined_loop_fsm.opt.ir.interface.binpb"
)
_INPUT_BACKWARDS_LOOP_IR_FILE = f"{_INTEGRATION_TESTS_DIR}/simple_backwards_pipelined_loop_fsm.ir.interface.binpb"
_OUTPUT_BACKWARDS_LOOP_IR_FILE = f"{_INTEGRATION_TESTS_DIR}/simple_backwards_pipelined_loop_fsm.opt.ir.interface.binpb"
_INPUT_SIGNED_NESTED_LOOP_IR_FILE = f"{_INTEGRATION_TESTS_DIR}/simple_nested_pipelined_loop_fsm.ir.interface.binpb"
_OUTPUT_SIGNED_NESTED_LOOP_IR_FILE = f"{_INTEGRATION_TESTS_DIR}/simple_nested_pipelined_loop_fsm.opt.ir.interface.binpb"


def _get_interface_proto(
    path: str,
) -> xls_ir_interface_pb2.PackageInterfaceProto:
  return xls_ir_interface_pb2.PackageInterfaceProto.FromString(
      runfiles.get_contents_as_bytes(path)
  )


def _for_loop_state_element_matcher(base_name: str) -> re.Pattern[str]:
  """Returns a regex that matches a for loop state element.

  Args:
    base_name: The base name of the for loop state element.

  Returns:
    A regex that matches a for loop state element.
  """
  return re.compile(f"^__for_[0-9]+_{base_name}(__.*)?")


class ArtifactAnalyzabilityTest(absltest.TestCase):

  def _assert_loop_variable_bit_width(
      self,
      ir_interface: xls_ir_interface_pb2.PackageInterfaceProto,
      for_variable: str,
      bit_count: int,
  ):
    """Asserts that a loop variable has the expected bit width.

    Args:
      ir_interface: The IR interface proto.
      for_variable: The name of the loop variable.
      bit_count: The expected bit count of the loop variable.
    """
    target_element = _for_loop_state_element_matcher(for_variable)
    self.assertLen(ir_interface.procs, 1)
    loop_elements = [
        nv
        for nv in ir_interface.procs[0].state
        if target_element.match(nv.name)
    ]
    self.assertLen(
        loop_elements,
        1,
        f"Multiple loop elements matched name pattern {target_element}",
    )
    self.assertEqual(
        loop_elements[0].type.bit_count,
        bit_count,
        f"Incorrect bit count for {for_variable}",
    )

  def test_unsigned_pipelined_loop_is_narrowed(self):
    input_interface = _get_interface_proto(_INPUT_UNSIGNED_LOOP_IR_FILE)
    output_interface = _get_interface_proto(_OUTPUT_UNSIGNED_LOOP_IR_FILE)
    self._assert_loop_variable_bit_width(
        input_interface, for_variable="i", bit_count=32
    )
    self._assert_loop_variable_bit_width(
        output_interface, for_variable="i", bit_count=3
    )

  def test_unsigned_pipelined_loop_non_loop_variable_not_narrowed(self):
    input_interface = _get_interface_proto(_INPUT_UNSIGNED_LOOP_IR_FILE)
    output_interface = _get_interface_proto(_OUTPUT_UNSIGNED_LOOP_IR_FILE)
    # Check 'a' variable stays wide.
    self._assert_loop_variable_bit_width(
        input_interface, for_variable="a", bit_count=32
    )
    self._assert_loop_variable_bit_width(
        output_interface, for_variable="a", bit_count=32
    )

  def test_unsigned_nested_pipelined_loop_is_narrowed(self):
    input_interface = _get_interface_proto(_INPUT_UNSIGNED_NESTED_LOOP_IR_FILE)
    output_interface = _get_interface_proto(
        _OUTPUT_UNSIGNED_NESTED_LOOP_IR_FILE
    )
    with self.subTest(name="outer_loop"):
      self._assert_loop_variable_bit_width(
          input_interface, for_variable="i", bit_count=32
      )
      self._assert_loop_variable_bit_width(
          output_interface, for_variable="i", bit_count=3
      )
    with self.subTest(name="inner_loop"):
      self._assert_loop_variable_bit_width(
          input_interface, for_variable="j", bit_count=32
      )
      self._assert_loop_variable_bit_width(
          output_interface, for_variable="j", bit_count=4
      )

  def test_signed_pipelined_loop_is_narrowed(self):
    input_interface = _get_interface_proto(_INPUT_SIGNED_LOOP_IR_FILE)
    output_interface = _get_interface_proto(_OUTPUT_SIGNED_LOOP_IR_FILE)
    self._assert_loop_variable_bit_width(
        input_interface, for_variable="i", bit_count=32
    )
    self._assert_loop_variable_bit_width(
        output_interface, for_variable="i", bit_count=3
    )

  def test_signed_nested_pipelined_loop_is_narrowed(self):
    input_interface = _get_interface_proto(_INPUT_SIGNED_NESTED_LOOP_IR_FILE)
    output_interface = _get_interface_proto(_OUTPUT_SIGNED_NESTED_LOOP_IR_FILE)
    with self.subTest(name="outer_loop"):
      self._assert_loop_variable_bit_width(
          input_interface, for_variable="i", bit_count=32
      )
      self._assert_loop_variable_bit_width(
          output_interface, for_variable="i", bit_count=3
      )
    with self.subTest(name="inner_loop"):
      self._assert_loop_variable_bit_width(
          input_interface, for_variable="j", bit_count=32
      )
      self._assert_loop_variable_bit_width(
          output_interface, for_variable="j", bit_count=4
      )


if __name__ == "__main__":
  absltest.main()
