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

"""Tests for xls.ir.python.package."""

from xls.ir.python import bits
from xls.ir.python import fileno
from xls.ir.python import function
from xls.ir.python import function_builder
from xls.ir.python import package
from xls.ir.python import type as ir_type
from xls.ir.python import value as ir_value
from absl.testing import absltest


class PackageTest(absltest.TestCase):

  def test_package_methods(self):
    pkg = package.Package('pkg')
    fb = function_builder.FunctionBuilder('f', pkg)
    fb.add_literal_value(ir_value.Value(bits.UBits(7, 8)))
    fb.build()

    self.assertIn('pkg', pkg.dump_ir())
    self.assertIsInstance(pkg.get_bits_type(4), ir_type.BitsType)
    self.assertIsInstance(
        pkg.get_array_type(4, pkg.get_bits_type(4)), ir_type.ArrayType)
    self.assertIsInstance(
        pkg.get_tuple_type([pkg.get_bits_type(4)]), ir_type.TupleType)
    self.assertIsInstance(pkg.get_or_create_fileno('file'), fileno.Fileno)
    self.assertIsInstance(pkg.get_function('f'), function.Function)
    self.assertEqual(['f'], pkg.get_function_names())


if __name__ == '__main__':
  absltest.main()
