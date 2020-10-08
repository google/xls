# Lint as: python3
#
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

"""Tests for xls.interpreter.python.ir_interpreter."""

from xls.interpreter.python import ir_interpreter
from xls.ir.python import ir_parser
from absl.testing import absltest


class IrInterpreterTest(absltest.TestCase):

  def test_add_one(self):
    ir = """
    package test_package

    fn add_one(x: bits[32]) -> bits[32] {
      literal.2: bits[32] = literal(value=1)
      ret add.3: bits[32] = add(x, literal.2)
    }
    """
    p = ir_parser.Parser.parse_package(ir)
    f = p.get_function('add_one')
    args = dict(x=ir_parser.Parser.parse_typed_value('bits[32]:0xf00'))
    result = ir_interpreter.run_function_kwargs(f, args)
    self.assertEqual(result,
                     ir_parser.Parser.parse_typed_value('bits[32]:0xf01'))

    result = ir_interpreter.run_function(
        f,
        [ir_parser.Parser.parse_typed_value('bits[32]:0xf05')])
    self.assertEqual(result,
                     ir_parser.Parser.parse_typed_value('bits[32]:0xf06'))


if __name__ == '__main__':
  absltest.main()
