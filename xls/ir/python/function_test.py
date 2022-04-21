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
#

"""Tests for xls.ir.python.function."""

from xls.ir.python import bits
from xls.ir.python import function_builder
from xls.ir.python import package
from xls.ir.python import value
from absl.testing import absltest


def build_function(name='function_name'):
  pkg = package.Package('pname')
  builder = function_builder.FunctionBuilder(name, pkg)
  builder.add_param('x', pkg.get_bits_type(32))
  builder.add_literal_value(value.Value(bits.UBits(7, 8)))
  return builder.build()


class FunctionTest(absltest.TestCase):

  def test_methods(self):
    fn = build_function('function_name')

    self.assertIn('function_name', fn.dump_ir())
    self.assertEqual(fn.get_param_count(), 1)
    self.assertEqual(fn.get_param_bit_count(0), 32)
    self.assertEqual('function_name', fn.name)


if __name__ == '__main__':
  absltest.main()
