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

    jit_comparison.compare_values(dslx_bits, ir_bits)

  def test_bits_equivalence_fail(self):
    dslx_bits = dslx_value.Value.make_ubits(bit_count=4, value=4)
    ir_bits_wrong_bit_count = ir_value.Value(ir_bits.UBits(value=4,
                                                           bit_count=5))

    with self.assertRaises(AssertionError):
      jit_comparison.compare_values(dslx_bits, ir_bits_wrong_bit_count)

    ir_bits_wrong_value = ir_value.Value(ir_bits.UBits(value=3, bit_count=4))

    with self.assertRaises(AssertionError):
      jit_comparison.compare_values(dslx_bits, ir_bits_wrong_value)

  def test_array_equivalence_ok(self):
    elements = []
    for i in range(5):
      elements.append(dslx_value.Value.make_ubits(bit_count=4, value=i))
    dslx_array = dslx_value.Value.make_array(elements)
    ir_array = jit_comparison.convert_interpreter_value_to_ir(dslx_array)

    jit_comparison.compare_values(dslx_array, ir_array)

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

    with self.assertRaises(AssertionError):
      jit_comparison.compare_values(dslx_array, ir_array_different_size)

    ir_elements_wrong = (ir_elements[:2]
                         + [ir_value.Value(ir_bits.UBits(value=5, bit_count=4))]
                         + ir_elements[3:])

    ir_array_different_element = ir_value.Value.make_array(ir_elements_wrong)

    with self.assertRaises(AssertionError):
      jit_comparison.compare_values(dslx_array, ir_array_different_element)

  def test_tuple_equivalence_ok(self):
    members = []
    for i in range(5):
      members.append(dslx_value.Value.make_ubits(bit_count=4, value=i))
    dslx_tuple = dslx_value.Value.make_tuple(tuple(members))
    ir_tuple = jit_comparison.convert_interpreter_value_to_ir(dslx_tuple)

    jit_comparison.compare_values(dslx_tuple, ir_tuple)

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

    with self.assertRaises(AssertionError):
      jit_comparison.compare_values(dslx_tuple, ir_tuple_different_size)

    ir_members_wrong = (ir_members[:2]
                        + [ir_value.Value(ir_bits.UBits(value=5, bit_count=4))]
                        + ir_members[3:])

    ir_tuple_different_member = ir_value.Value.make_tuple(ir_members_wrong)

    with self.assertRaises(AssertionError):
      jit_comparison.compare_values(dslx_tuple, ir_tuple_different_member)

  def test_bits_to_int(self):
    """Tests IR bit-value retrieval done at one 64-bit word at a time."""
    bit_count_0 = jit_comparison.int_to_bits(value=0, bit_count=0)
    self.assertEqual(0, jit_comparison.bits_to_int(bit_count_0, signed=False))

    bit_count_1 = jit_comparison.int_to_bits(value=0, bit_count=1)
    self.assertEqual(0, jit_comparison.bits_to_int(bit_count_1, signed=False))

    bit_count_63 = jit_comparison.int_to_bits(value=(2**63 - 1), bit_count=63)
    self.assertEqual((2**63 - 1), jit_comparison.bits_to_int(bit_count_63,
                                                             signed=False))
    bit_count_63_signed = jit_comparison.int_to_bits(value=-1, bit_count=63)
    self.assertEqual(-1, jit_comparison.bits_to_int(bit_count_63_signed,
                                                    signed=True))

    bit_count_64 = jit_comparison.int_to_bits(value=(2**64 - 1), bit_count=64)
    self.assertEqual((2**64 - 1), jit_comparison.bits_to_int(bit_count_64,
                                                             signed=False))
    bit_count_64_signed = jit_comparison.int_to_bits(value=-1, bit_count=64)
    self.assertEqual(-1, jit_comparison.bits_to_int(bit_count_64_signed,
                                                    signed=True))

    bit_count_127 = jit_comparison.int_to_bits(value=(2**127 - 1),
                                               bit_count=127)
    self.assertEqual((2**127 - 1), jit_comparison.bits_to_int(bit_count_127,
                                                          signed=False))
    bit_count_127_signed = jit_comparison.int_to_bits(value=-1, bit_count=127)
    self.assertEqual(-1, jit_comparison.bits_to_int(bit_count_127_signed,
                                                    signed=True))

    bit_count_128 = jit_comparison.int_to_bits(value=(2**128 - 1),
                                               bit_count=128)
    self.assertEqual((2**128 - 1), jit_comparison.bits_to_int(bit_count_128,
                                                              signed=False))
    bit_count_128_signed = jit_comparison.int_to_bits(value=-1, bit_count=128)
    self.assertEqual(-1, jit_comparison.bits_to_int(bit_count_128_signed,
                                                    signed=True))

if __name__ == '__main__':
  absltest.main()
