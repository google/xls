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
"""Tests for xls.dslx.python.cpp_ast."""

from xls.dslx.python import cpp_ast
from absl.testing import absltest


class CppAstTest(absltest.TestCase):

  fake_pos = cpp_ast.Pos('fake.x', 0, 0)
  fake_span = cpp_ast.Span(fake_pos, fake_pos)

  def test_simple_span(self):
    start = cpp_ast.Pos('fake.x', 1, 2)
    limit = cpp_ast.Pos('fake.x', 3, 4)
    span = cpp_ast.Span(start, limit)
    self.assertEqual(str(span), 'fake.x:2:3-4:5')

  def test_module_with_constant(self):
    m = cpp_ast.Module('test')
    name_def = cpp_ast.NameDef(m, self.fake_span, 'MOL')
    number = cpp_ast.Number(m, self.fake_span, '42')
    constant_def = cpp_ast.Constant(m, name_def, number)
    m.add_top(constant_def)
    self.assertEqual(str(m), 'const MOL = 42;')

  def test_binop(self):
    m = cpp_ast.Module('test')
    ft = cpp_ast.Number(m, self.fake_span, '42')
    sf = cpp_ast.Number(m, self.fake_span, '64')
    add = cpp_ast.Binop(m, self.fake_span, cpp_ast.BinopKind.ADD, ft, sf)
    self.assertEqual(str(add), '(42) + (64)')

  def test_identity_function(self):
    m = cpp_ast.Module('test')
    name_def_x = cpp_ast.NameDef(m, self.fake_span, 'x')
    name_ref_x = cpp_ast.NameRef(m, self.fake_span, 'x', name_def_x)
    type_u32 = cpp_ast.BuiltinTypeAnnotation(m, self.fake_span,
                                             cpp_ast.BuiltinType.U32)
    param_x = cpp_ast.Param(m, name_def_x, type_u32)
    name_def_f = cpp_ast.NameDef(m, self.fake_span, 'f')
    params = (param_x,)
    f = cpp_ast.Function(
        m,
        self.fake_span,
        name_def_f, (),
        params,
        type_u32,
        name_ref_x,
        public=False)
    self.assertEqual(str(f), 'fn f(x: u32) -> u32 {\n  x\n}')


if __name__ == '__main__':
  absltest.main()
