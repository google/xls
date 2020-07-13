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

"""Tests for xls.ir.python.type."""

from xls.common import test_base
from xls.ir.python import bits
from xls.ir.python import function_builder
from xls.ir.python import package
from xls.ir.python import type as ir_type
from xls.ir.python import value as ir_value


class TypeTest(test_base.TestCase):

  def test_type(self):
    pkg = package.Package('pkg')
    t = pkg.get_bits_type(4)

    self.assertIn('4', str(t))

  def test_bits_type(self):
    pkg = package.Package('pkg')
    t = pkg.get_bits_type(4)

    self.assertEqual(4, t.get_bit_count())

  def test_array_type(self):
    pkg = package.Package('pkg')
    bit_type = pkg.get_bits_type(4)
    t = pkg.get_array_type(3, bit_type)

    self.assertEqual(3, t.get_size())
    self.assertEqual(4, t.get_element_type().get_bit_count())

  def test_type_polymorphism(self):
    # Verify that pybind11 knows to down-cast Type objects.

    pkg = package.Package('pkg')
    bits_type = pkg.get_bits_type(4)
    array_type = pkg.get_array_type(3, bits_type)
    tuple_type = pkg.get_tuple_type([bits_type])

    self.assertIsInstance(
        pkg.get_array_type(1, bits_type).get_element_type(), ir_type.BitsType)
    self.assertIsInstance(
        pkg.get_array_type(1, array_type).get_element_type(), ir_type.ArrayType)
    self.assertIsInstance(
        pkg.get_array_type(1, tuple_type).get_element_type(), ir_type.TupleType)

  def test_function_type(self):
    pkg = package.Package('pname')
    builder = function_builder.FunctionBuilder('f_name', pkg)
    builder.add_param('x', pkg.get_bits_type(32))
    builder.add_literal_value(ir_value.Value(bits.UBits(7, 8)))
    fn = builder.build()
    fn_type = fn.get_type()

    self.assertIsInstance(fn_type, ir_type.FunctionType)
    self.assertIn('bits[32]', str(fn_type))
    self.assertEqual(8, fn_type.return_type().get_bit_count())
    self.assertEqual(1, fn_type.get_parameter_count())
    self.assertEqual(32, fn_type.get_parameter_type(0).get_bit_count())


if __name__ == '__main__':
  test_base.main()
