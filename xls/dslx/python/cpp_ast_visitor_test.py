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
"""Tests for xls.dslx.python.cpp_ast_visitor."""

from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_ast_visitor
from absl.testing import absltest


class Collector(cpp_ast_visitor.AstVisitor):

  def __init__(self):
    self.visited = []

  def visit_Number(self, n: ast.Number) -> None:
    self.visited.append(n)

  def visit_Array(self, n: ast.Array) -> None:
    self.visited.append(n)


class CppAstVisitorTest(absltest.TestCase):

  fake_pos = ast.Pos('fake.x', 0, 0)
  fake_span = ast.Span(fake_pos, fake_pos)

  def test_simple_number(self):
    m = ast.Module('test')
    n = ast.Number(m, self.fake_span, '42')
    self.assertEmpty(n.children)
    collector = Collector()
    cpp_ast_visitor.visit(n, collector)
    self.assertEqual(collector.visited, [n])

  def test_array_of_numbers(self):
    m = ast.Module('test')
    n0 = ast.Number(m, self.fake_span, '42')
    n1 = ast.Number(m, self.fake_span, '64')
    a = ast.Array(m, self.fake_span, [n0, n1], False)
    collector = Collector()
    cpp_ast_visitor.visit(a, collector)
    self.assertEqual(collector.visited, [n0, n1, a])


if __name__ == '__main__':
  absltest.main()
