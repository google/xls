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

"""Tests for xls.frontend.dslx.extract_conversion_order."""

from typing import Text

from absl import logging

from xls.common import test_base
from xls.dslx import ast
from xls.dslx import extract_conversion_order
from xls.dslx import fakefs_util
from xls.dslx import parse_and_typecheck


class ExtractConversionOrderTest(test_base.XlsTestCase):

  def _get_module(self, program: Text) -> ast.Module:
    filename = '/fake/test_program.x'
    with fakefs_util.scoped_fakefs(filename, program):
      m, _ = parse_and_typecheck.parse_text(
          program,
          'test_program',
          print_on_error=True,
          f_import=None,
          filename=filename)
      return m

  def test_get_callees(self):
    program = """
    fn f() -> u32 { u32:42 }
    fn main() -> u32 { f() }
    """
    m = self._get_module(program)
    self.assertEqual(((m.get_function('f'), m, ()),),
                     extract_conversion_order.get_callees(
                         m.get_function('main'), m, imports={}, bindings=()))

  def test_simple_linear_callgraph(self):
    program = """
    fn g() -> u32 { u32:42 }
    fn f() -> u32 { g() }
    fn main() -> u32 { f() }
    """
    m = self._get_module(program)
    order = extract_conversion_order.get_order(m, imports={})
    logging.info('order: %s', order)
    self.assertLen(order, 3)
    self.assertEqual(order[0].f.identifier, 'g')
    self.assertEqual(order[1].f.identifier, 'f')
    self.assertEqual(order[2].f.identifier, 'main')

  def test_parametric(self):
    program = """
    fn [N: u32] f(x: bits[N]) -> u32 { N }
    fn main() -> u32 { f(u2:0) }
    """
    m = self._get_module(program)
    order = extract_conversion_order.get_order(m, imports={})
    logging.info('order: %s', order)
    self.assertLen(order, 2)
    self.assertEqual(order[0].f.identifier, 'f')
    self.assertEqual(order[0].bindings, (('N', 2),))
    self.assertEqual(order[0].callees, ())
    self.assertEqual(order[1].f.identifier, 'main')
    self.assertEqual(order[1].bindings, ())
    f = m.get_function('f')
    self.assertEqual(order[1].callees, ((f, m, (('N', 2),)),))

  def test_transitive_parametric(self):
    program = """
    fn [M: u32] g(x: bits[M]) -> u32 { M }
    fn [N: u32] f(x: bits[N]) -> u32 { g(x) }
    fn main() -> u32 { f(u2:0) }
    """
    m = self._get_module(program)
    order = extract_conversion_order.get_order(m, imports={})
    logging.info('order: %s', order)
    self.assertLen(order, 3)
    self.assertEqual(order[0].f.identifier, 'g')
    self.assertEqual(order[0].bindings, (('M', 2),))
    self.assertEqual(order[1].f.identifier, 'f')
    self.assertEqual(order[1].bindings, (('N', 2),))
    self.assertEqual(order[2].f.identifier, 'main')
    self.assertEqual(order[2].bindings, ())

  def test_builtin_is_elided(self):
    program = """
    fn main() -> u32 { fail!(u32:0) }
    """
    m = self._get_module(program)
    order = extract_conversion_order.get_order(m, imports={})
    logging.info('order: %s', order)
    self.assertLen(order, 1)
    self.assertEqual(order[0].f.identifier, 'main')
    self.assertEqual(order[0].bindings, ())


if __name__ == '__main__':
  test_base.main()
