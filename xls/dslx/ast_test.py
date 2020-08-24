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

"""Tests for xls.dslx.ast."""

from xls.common import test_base
from xls.dslx import ast
from xls.dslx.scanner import Keyword
from xls.dslx.scanner import Token
from xls.dslx.scanner import TokenKind
from xls.dslx.span import Pos
from xls.dslx.span import Span


class _Collector(ast.AstVisitor):
  """Collects visited nodes for test condition checking."""

  def __init__(self):
    self.collected = []

  def visit_ArrayTypeAnnotation(self, node: ast.ArrayTypeAnnotation) -> None:
    self.collected.append(node)

  def visit_Number(self, node: ast.Number) -> None:
    self.collected.append(node)

  def visit_Index(self, node: ast.Index) -> None:
    self.collected.append(node)

  def visit_NameRef(self, node: ast.NameRef) -> None:
    self.collected.append(node)

  def visit_Unop(self, node: ast.Unop) -> None:
    self.collected.append(node)


class AstTest(test_base.TestCase):

  @property
  def fake_pos(self) -> Pos:
    return Pos('<fake>', 0, 0)

  @property
  def fake_span(self) -> Span:
    return Span(self.fake_pos, self.fake_pos)

  @property
  def five(self) -> ast.Number:
    return ast.Number(self.fake_span, '5')

  def test_stringify_type(self):
    fake_span = self.fake_span
    number_5 = ast.Number(fake_span, '5')
    number_2 = ast.Number(fake_span, '2')
    number_3 = ast.Number(fake_span, '3')
    bits_token = Token(TokenKind.KEYWORD, value=Keyword.BITS, span=fake_span)

    type_ = ast.make_builtin_type_annotation(fake_span, bits_token, (number_5,))
    self.assertEqual('bits[5]', str(type_))

    type_ = ast.make_builtin_type_annotation(
        fake_span, Token(TokenKind.KEYWORD, value=Keyword.U32, span=fake_span),
        (number_5,))
    self.assertEqual('u32[5]', str(type_))

    # "no-volume" bits array.
    # TODO(leary): 2020-08-24 delete bits in favor of uN
    type_ = ast.make_builtin_type_annotation(fake_span, bits_token, ())
    self.assertEqual('bits', str(type_))

    # TypeRef with dims.
    my_type_tok = Token(TokenKind.IDENTIFIER, value='MyType', span=fake_span)
    name_def = ast.NameDef(fake_span, 'MyType')
    type_def = ast.TypeDef(False, name_def, type_)
    type_ref = ast.TypeRef(fake_span, my_type_tok.value, type_def=type_def)
    type_ = ast.make_type_ref_type_annotation(fake_span, type_ref,
                                              (number_2, number_3))
    self.assertEqual('MyType[2][3]', str(type_))

  def test_stringify_single_member_tuple(self):
    fake_pos = Pos('<fake>', 0, 0)
    fake_span = Span(fake_pos, fake_pos)

    t = ast.XlsTuple(fake_span, (self.five,))
    self.assertEqual('(5,)', str(t))

  def test_visit_type(self):
    fake_span = self.fake_span
    five = self.five
    # Make a uN[5] type node.
    t = ast.make_builtin_type_annotation(
        fake_span,
        Token(TokenKind.KEYWORD, value=Keyword.BITS, span=fake_span),
        dims=(five,))
    assert isinstance(t, ast.ArrayTypeAnnotation), t

    c = _Collector()
    t.accept(c)
    self.assertEqual(c.collected, [five, t])

  def test_visit_index(self):
    fake_span = self.fake_span
    # Make a t[i] inde xnode.
    t = ast.NameRef(fake_span, 't', ast.NameDef(fake_span, 't'))
    i = ast.NameRef(fake_span, 'i', ast.NameDef(fake_span, 'i'))
    index = ast.Index(fake_span, t, i)

    c = _Collector()
    index.accept(c)
    self.assertEqual(c.collected, [t, i, index])

  def test_visit_unop(self):
    fake_span = self.fake_span
    i_def = ast.NameDef(fake_span, 'i')
    i_ref = ast.NameRef(fake_span, 'i', i_def)
    negated = ast.Unop(Token(TokenKind.MINUS, fake_span), i_ref)

    c = _Collector()
    negated.accept(c)
    self.assertEqual(c.collected, [i_ref, negated])

  def test_visit_match_multi_pattern(self):
    fake_pos = self.fake_pos
    fake_span = Span(fake_pos, fake_pos)
    e = ast.Number(fake_span, u'0xf00')
    p0 = ast.NameDefTree(fake_span, e)
    p1 = ast.NameDefTree(fake_span, e)
    arm = ast.MatchArm(patterns=(p0, p1), expr=e)
    c = _Collector()
    arm.accept(c)
    self.assertEqual(c.collected, [e])

  def test_unicode_hex_number(self):
    fake_pos = self.fake_pos
    fake_span = Span(fake_pos, fake_pos)
    n = ast.Number(fake_span, u'0xf00')
    self.assertEqual(0xf00, n.get_value_as_int())

  def test_hex_number_with_underscores(self):
    fake_pos = self.fake_pos
    fake_span = Span(fake_pos, fake_pos)
    n = ast.Number(fake_span, '0xf_abcde_1234')
    self.assertEqual(0xfabcde1234, n.get_value_as_int())

  def test_binary_number_with_underscores(self):
    fake_pos = self.fake_pos
    fake_span = Span(fake_pos, fake_pos)
    n = ast.Number(fake_span, u'0b1_0_0_1')
    self.assertEqual(9, n.get_value_as_int())

  def test_ndt_preorder(self):
    fake_pos = self.fake_pos
    fake_span = Span(fake_pos, fake_pos)
    t = ast.NameDef(fake_span, 't')
    u = ast.NameDef(fake_span, 'u')
    wrapped_t = ast.NameDefTree(fake_span, t)
    wrapped_u = ast.NameDefTree(fake_span, u)

    interior = ast.NameDefTree(fake_span, (wrapped_t, wrapped_u))
    outer = ast.NameDefTree(fake_span, (interior,))

    walk_data = []

    def walk(item: ast.NameDefTree, level: int, i: int):
      walk_data.append((item, level, i))

    outer.do_preorder(walk)

    self.assertLen(walk_data, 3)
    self.assertEqual(walk_data[0], (interior, 1, 0))
    self.assertEqual(walk_data[1], (wrapped_t, 2, 0))
    self.assertEqual(walk_data[2], (wrapped_u, 2, 1))

  def test_format_binop(self):
    fake_pos = self.fake_pos
    fake_span = Span(fake_pos, fake_pos)
    le = ast.Binop(
        Token(TokenKind.OANGLE_EQUALS, span=fake_span), self.five, self.five)
    self.assertEqual('(5) <= (5)', str(le))

  def test_type_annotation_properties(self):
    fake_span = self.fake_span
    number_5 = ast.Number(fake_span, '5')
    number_2 = ast.Number(fake_span, '2')
    number_3 = ast.Number(fake_span, '3')
    bits_token = Token(TokenKind.KEYWORD, value=Keyword.BITS, span=fake_span)
    un_token = Token(TokenKind.KEYWORD, value=Keyword.UN, span=fake_span)
    u32_token = Token(TokenKind.KEYWORD, value=Keyword.U32, span=fake_span)

    type_ = ast.make_builtin_type_annotation(fake_span, bits_token, (number_5,))
    self.assertEqual('bits[5]', str(type_))

    type_ = ast.make_builtin_type_annotation(fake_span, bits_token,
                                             (number_5, number_2))
    self.assertIsInstance(type_, ast.ArrayTypeAnnotation)
    self.assertEqual('bits[5][2]', str(type_))

    type_ = ast.make_builtin_type_annotation(fake_span, u32_token, ())
    self.assertEqual('u32', str(type_))

    type_ = ast.make_builtin_type_annotation(fake_span, u32_token, (number_3,))
    self.assertIsInstance(type_, ast.ArrayTypeAnnotation)
    self.assertEqual('u32[3]', str(type_))

    type_ = ast.make_builtin_type_annotation(fake_span, un_token, (number_2,))
    self.assertEqual('uN[2]', str(type_))

    type_ = ast.make_builtin_type_annotation(fake_span, un_token,
                                             (number_2, number_3))
    self.assertIsInstance(type_, ast.ArrayTypeAnnotation)
    self.assertEqual('uN[2][3]', str(type_))

    # TODO(leary): 2020-08-24 delete bits in favor of uN
    # "no-volume" bits array.
    type_ = ast.make_builtin_type_annotation(fake_span, bits_token, ())
    self.assertEqual('bits', str(type_))

    # TypeRef with dims.
    name_def = ast.NameDef(fake_span, 'MyType')
    type_def = ast.TypeDef(False, name_def, type_)
    type_ref = ast.TypeRef(fake_span, 'MyType', type_def=type_def)
    type_ = ast.make_type_ref_type_annotation(fake_span, type_ref,
                                              (number_2, number_3))
    self.assertIsInstance(type_, ast.ArrayTypeAnnotation)
    self.assertEqual('MyType[2][3]', str(type_))

    type_ = ast.TupleTypeAnnotation(
        fake_span,
        (ast.make_builtin_type_annotation(fake_span, bits_token, (number_5,)),
         ast.make_builtin_type_annotation(fake_span, bits_token, (number_2,))))
    self.assertIsInstance(type_, ast.TupleTypeAnnotation)
    self.assertEqual('(bits[5], bits[2])', str(type_))


if __name__ == '__main__':
  test_base.main()
