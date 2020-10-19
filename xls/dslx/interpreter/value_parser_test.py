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
# Lint as: python3

"""Tests for xls.dslx.interpreter.value_parser."""

from absl.testing import absltest
from xls.dslx.interpreter.value_parser import value_from_string
from xls.dslx.interpreter.value_parser import ValueParseError
from xls.dslx.python.interp_value import Value


class ValueParserTest(absltest.TestCase):

  def test_bits_value_hex(self):
    self.assertEqual(
        value_from_string('bits[3]:0x7'),
        Value.make_ubits(bit_count=3, value=7))
    self.assertEqual(
        value_from_string('bits[1]:0x1'),
        Value.make_ubits(bit_count=1, value=1))
    self.assertEqual(
        value_from_string('bits[1]:0x0'),
        Value.make_ubits(bit_count=1, value=0))
    self.assertEqual(
        value_from_string('bits[8]:0xff'),
        Value.make_ubits(bit_count=8, value=0xff))
    self.assertEqual(
        value_from_string('u8:0xff'), Value.make_ubits(bit_count=8, value=0xff))

  def test_tuple_values(self):
    self.assertEqual(
        value_from_string('(bits[8]:0xff, bits[2]:0x1)'),
        Value.make_tuple((
            Value.make_ubits(bit_count=8, value=0xff),
            Value.make_ubits(bit_count=2, value=1),
        )))

    self.assertEqual(value_from_string('()'), Value.make_tuple(()))

    # Tuple of single element.
    want = Value.make_tuple((Value.make_ubits(bit_count=2, value=1),))
    got = value_from_string('(bits[2]:0x1,)')
    self.assertEqual(want, got)

    with self.assertRaises(ValueParseError) as cm:
      value_from_string('(,)')

    self.assertIn('Unexpected token in value', str(cm.exception))

  def test_array_values(self):
    self.assertEqual(
        value_from_string('[(u8:0xff, u2:0x1), (u8:0, u2:3)]'),
        Value.make_array((
            Value.make_tuple((
                Value.make_ubits(bit_count=8, value=0xff),
                Value.make_ubits(bit_count=2, value=1),
            )),
            Value.make_tuple((
                Value.make_ubits(bit_count=8, value=0x0),
                Value.make_ubits(bit_count=2, value=3),
            )),
        )))


if __name__ == '__main__':
  absltest.main()
