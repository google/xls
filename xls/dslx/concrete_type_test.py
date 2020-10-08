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

"""Tests for xls.dslx.concrete_type."""

from absl.testing import absltest
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python.cpp_concrete_type import ArrayType
from xls.dslx.python.cpp_concrete_type import BitsType
from xls.dslx.python.cpp_concrete_type import ConcreteType
from xls.dslx.python.cpp_concrete_type import TupleType
from xls.dslx.python.cpp_parametric_expression import ParametricSymbol
from xls.dslx.python.cpp_pos import Pos
from xls.dslx.python.cpp_pos import Span


class ConcreteTypeTest(absltest.TestCase):

  def test_nil_tuple(self):
    nil = TupleType(members=())
    self.assertTrue(nil.is_nil())

    t = TupleType(members=(nil,))
    self.assertFalse(t.is_nil())

  def test_equality(self):
    fake_pos = Pos('<fake>', 0, 0)
    fake_span = Span(fake_pos, fake_pos)
    p = BitsType(signed=False, size=ParametricSymbol('N', fake_span))
    c = BitsType(signed=False, size=32)
    self.assertTrue(c.__eq__(c))
    self.assertFalse(c.__ne__(c))
    self.assertTrue(p.__eq__(p))
    self.assertFalse(p.__ne__(p))
    self.assertTrue(p.__ne__(c))
    self.assertFalse(p.__eq__(c))

  def test_array_vs_multidim_bits_equality(self):
    a = ArrayType(BitsType(signed=False, size=5), 7)
    self.assertEqual(str(a), 'uN[5][7]')
    self.assertEqual(7 * 5, a.get_total_bit_count())
    self.assertEqual(7, a.size)
    self.assertEqual(5, a.get_element_type().size)  # pytype: disable=attribute-error
    self.assertEqual((7, 5), a.get_all_dims())

    self.assertEqual((), TupleType(()).get_all_dims())

  def test_array_of_tuple_all_dims(self):
    a = ArrayType(TupleType(()), 7)
    self.assertEqual((7,), a.get_all_dims())

  def test_stringify(self):
    u32 = BitsType(signed=False, size=32)
    tabular = [
        # type size total_bit_count str
        (ArrayType(u32, 7), 7, 32 * 7, 'uN[32][7]'),
        (u32, 32, 32, 'uN[32]'),
    ]
    for t, size, total_bit_count, s in tabular:
      self.assertEqual(t.size, size)
      self.assertEqual(t.get_total_bit_count(), total_bit_count)
      self.assertEqual(str(t), s)

  def test_arrayness(self):
    tabular = [
        # (type, is_array, element_count)
        (TupleType(members=()), False, None),
        (BitsType(signed=False, size=5), False, None),
        (ArrayType(BitsType(False, 5), 7), True, 7),
        (ArrayType(TupleType(members=()), 7), True, 7),
    ]

    for t, is_array, element_count in tabular:
      self.assertEqual(isinstance(t, ArrayType), is_array, msg=str(t))
      if is_array:
        self.assertEqual(t.size, element_count, msg=str(t))

  def test_named_tuple_vs_tuple_compatibility(self):
    u32 = ConcreteType.U32
    u8 = ConcreteType.U8
    m = ast.Module('test')
    fake_pos = Pos('fake.x', 0, 0)
    fake_span = Span(fake_pos, fake_pos)
    name_def = ast.NameDef(m, fake_span, 'fake')
    s = ast.Struct(m, fake_span, name_def, (), (), False)
    named = TupleType((('x', u32), ('y', u8)), struct=s)
    unnamed = TupleType((u32, u8))
    self.assertTrue(named.compatible_with(unnamed))
    self.assertNotEqual(named, unnamed)
    self.assertEqual(named.tuple_names, ('x', 'y'))

  def test_array_bit_count(self):
    e = BitsType(signed=False, size=4)
    a = ArrayType(e, 3)
    self.assertEqual(a.get_total_bit_count(), 12)


if __name__ == '__main__':
  absltest.main()
