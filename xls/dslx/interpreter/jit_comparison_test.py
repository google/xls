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

"""Tests for xls.dslx.interpreter.jit_comparison."""

from xls.dslx.interpreter import value as dslx_value
from xls.ir.python import value as ir_value
from xls.ir.python import bits as ir_bits
from xls.dslx.interpreter import jit_comparison
from absl.testing import absltest


class JitComparisonTest(absltest.TestCase):

  def test_bits_equivalence_ok(self):
    dslx_bits = dslx_value.Value.make_ubits(bit_count=4, value=4)
    ir_bits = jit_comparison.convert_interpreter_value_to_ir(dslx_bits)

    try:
      jit_comparison.compare_values(dslx_bits, ir_bits)
    except (AssertionError, jit_comparison.UnsupportedConversionError) as e:
      self.fail("Unexpected error raised: %s", e)

  def test_bits_equivalence_fail(self):
    dslx_bits = dslx_value.Value.make_ubits(bit_count=4, value=4)
    ir_bits_wrong_bit_count = ir_value.Value(ir_bits.UBits(value=4,
                                                           bit_count=5))

    try:
      jit_comparison.compare_values(dslx_bits, ir_bits_wrong_bit_count)
      self.fail("Expected equality assertion failure on bit count!")
    except AssertionError as _:
      pass

    ir_bits_wrong_value = ir_value.Value(ir_bits.UBits(value=3, bit_count=4))
    try:
      jit_comparison.compare_values(dslx_bits, ir_bits_wrong_value)
      self.fail("Expected equality assertion failure on value!")
    except AssertionError as _:
      pass

  def test_array_equivalence_ok(self):
    elements = []
    for i in range(5):
      elements.append(dslx_value.Value.make_ubits(bit_count=4, value=i))
    dslx_array = dslx_value.Value.make_array(elements)
    ir_array = jit_comparison.convert_interpreter_value_to_ir(dslx_array)

    try:
      jit_comparison.compare_values(dslx_array, ir_array)
    except (AssertionError, jit_comparison.UnsupportedConversionError) as e:
      self.fail("Unexpected error raised: %s", e)

  def test_array_equivalence_fail(self):
    dslx_elements = []
    ir_elements = []
    for i in range(5):
      dslx_elements.append(dslx_value.Value.make_ubits(bit_count=4, value=i))
      ir_elements.append(ir_value.Value(ir_bits.UBits(value=i, bit_count=4)))

    dslx_array = dslx_value.Value.make_array(dslx_elements)

    ir_elements_extra = (ir_elements
                         + [ir_value.Value(ir_bits.UBits(value=5,
                                                         bit_count=4))])
    ir_array_different_size = ir_value.Value.make_array(ir_elements_extra)

    try:
      jit_comparison.compare_values(dslx_array, ir_array_different_size)
      self.fail("Expected equality assertion failure on length!")
    except AssertionError as e:
      pass

    ir_elements_wrong = (ir_elements[:2]
                         + [ir_value.Value(ir_bits.UBits(value=5, bit_count=4))]
                         + ir_elements[3:])

    ir_array_different_element = ir_value.Value.make_array(ir_elements_wrong)

    try:
      jit_comparison.compare_values(dslx_array, ir_array_different_element)
      self.fail("Expected equality assertion failure on value!")
    except AssertionError as e:
      pass

  def test_tuple_equivalence_ok(self):
    members = []
    for i in range(5):
      members.append(dslx_value.Value.make_ubits(bit_count=4, value=i))
    dslx_tuple = dslx_value.Value.make_tuple(tuple(members))
    ir_tuple = jit_comparison.convert_interpreter_value_to_ir(dslx_tuple)

    try:
      jit_comparison.compare_values(dslx_tuple, ir_tuple)
    except (AssertionError, jit_comparison.UnsupportedConversionError) as e:
      self.fail("Unexpected error raised: %s", e)

  def test_tuple_equivalence_fail(self):
    dslx_members = []
    ir_members = []
    for i in range(5):
      dslx_members.append(dslx_value.Value.make_ubits(bit_count=4, value=i))
      ir_members.append(ir_value.Value(ir_bits.UBits(value=i, bit_count=4)))

    dslx_tuple = dslx_value.Value.make_tuple(tuple(dslx_members))

    ir_members_extra = (ir_members
                        + [ir_value.Value(ir_bits.UBits(value=5,
                                                        bit_count=4))])
    ir_tuple_different_size = ir_value.Value.make_tuple(ir_members_extra)

    try:
      jit_comparison.compare_values(dslx_tuple, ir_tuple_different_size)
      self.fail("Expected equality assertion failure on length!")
    except AssertionError as e:
      pass

    ir_members_wrong = (ir_members[:2]
                        + [ir_value.Value(ir_bits.UBits(value=5, bit_count=4))]
                        + ir_members[3:])

    ir_tuple_different_member = ir_value.Value.make_tuple(ir_members_wrong)

    try:
      jit_comparison.compare_values(dslx_tuple, ir_tuple_different_member)
      self.fail("Expected equality assertion failure on value!")
    except AssertionError as e:
      pass

  def test_large_bit_equivalence(self):
    dslx_ubits = dslx_value.Value.make_ubits(bit_count=512, value=(2**512 - 1))
    ir_ubits = jit_comparison.convert_interpreter_value_to_ir(dslx_ubits)

    try:
      jit_comparison.compare_values(dslx_ubits, ir_ubits)
    except (AssertionError, jit_comparison.UnsupportedConversionError) as e:
      self.fail("Unexpected error raised: %s", e)


    dslx_sbits = dslx_value.Value.make_sbits(bit_count=512,
                                             value=((2**512) // 2 - 1))
    ir_sbits = jit_comparison.convert_interpreter_value_to_ir(dslx_sbits)

    try:
      jit_comparison.compare_values(dslx_sbits, ir_sbits)
    except (AssertionError, jit_comparison.UnsupportedConversionError) as e:
      self.fail("Unexpected error raised: %s", e)

if __name__ == '__main__':
  absltest.main()
