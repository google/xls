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

"""Tests for xls.ir.python.ir_parser."""

from xls.ir.python import ir_parser
from xls.ir.python import package as package_mod
from absl.testing import absltest


class IrParserTest(absltest.TestCase):

  def test_parse_value(self):
    p = package_mod.Package('test_package')
    u32 = p.get_bits_type(32)
    u8 = p.get_bits_type(8)
    t = p.get_tuple_type([u32, u8])
    s = '({}, {})'.format(0xdeadbeef, 0xcd)
    v = ir_parser.Parser.parse_value(s, t)
    self.assertEqual(s, str(v))

    v = ir_parser.Parser.parse_value('(0xdeadbeef, 0xcd)', t)
    self.assertEqual('(bits[32]:{}, bits[8]:{})'.format(0xdeadbeef, 0xcd),
                     v.to_str())

  def test_parse_typed_value(self):
    s = 'bits[32]:0x42'
    v = ir_parser.Parser.parse_typed_value(s)
    self.assertEqual('66', str(v))

  def test_parse_package(self):
    p = ir_parser.Parser.parse_package('package test_package')
    self.assertIsInstance(p, package_mod.Package)

if __name__ == '__main__':
  absltest.main()
