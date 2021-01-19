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

"""Tests for xls.dslx.extract_conversion_order."""

from typing import Text, Tuple

from absl.testing import absltest
from xls.common import test_base
from xls.dslx import fakefs_test_util
from xls.dslx import parse_and_typecheck
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_ir_converter
from xls.dslx.python import cpp_type_info as type_info_mod
from xls.dslx.python.cpp_type_info import SymbolicBindings
from xls.dslx.python.import_routines import ImportCache


class ExtractConversionOrderTest(absltest.TestCase):

  def _get_module(self,
                  program: Text) -> Tuple[ast.Module, type_info_mod.TypeInfo]:
    filename = '/fake/test_program.x'
    with fakefs_test_util.scoped_fakefs(filename, program):
      m, type_info = parse_and_typecheck.parse_text(
          program,
          'test_program',
          print_on_error=True,
          import_cache=ImportCache(),
          additional_search_paths=(),
          filename=filename)
      return m, type_info

  def test_simple_linear_callgraph(self):
    program = """
    fn g() -> u32 { u32:42 }
    fn f() -> u32 { g() }
    fn main() -> u32 { f() }
    """
    m, type_info = self._get_module(program)
    order = cpp_ir_converter.get_conversion_order(m, type_info)
    self.assertLen(order, 3)
    self.assertEqual(order[0].f.identifier, 'g')
    self.assertEqual(order[1].f.identifier, 'f')
    self.assertEqual(order[2].f.identifier, 'main')

  def test_parametric(self):
    program = """
    fn f<N: u32>(x: bits[N]) -> u32 { N }
    fn main() -> u32 { f(u2:0) }
    """
    m, type_info = self._get_module(program)
    order = cpp_ir_converter.get_conversion_order(m, type_info)
    self.assertLen(order, 2)
    self.assertEqual(order[0].f.identifier, 'f')
    self.assertEqual(order[0].bindings, SymbolicBindings([('N', 2)]))
    self.assertEqual(order[1].f.identifier, 'main')
    self.assertEqual(order[1].bindings, SymbolicBindings())

  def test_transitive_parametric(self):
    program = """
    fn g<M: u32>(x: bits[M]) -> u32 { M }
    fn f<N: u32>(x: bits[N]) -> u32 { g(x) }
    fn main() -> u32 { f(u2:0) }
    """
    m, type_info = self._get_module(program)
    order = cpp_ir_converter.get_conversion_order(m, type_info)
    self.assertLen(order, 3)
    self.assertEqual(order[0].f.identifier, 'g')
    self.assertEqual(order[0].bindings, SymbolicBindings([('M', 2)]))
    self.assertEqual(order[1].f.identifier, 'f')
    self.assertEqual(order[1].bindings, SymbolicBindings([('N', 2)]))
    self.assertEqual(order[2].f.identifier, 'main')
    self.assertEqual(order[2].bindings, SymbolicBindings())

  def test_builtin_is_elided(self):
    program = """
    fn main() -> u32 { fail!(u32:0) }
    """
    m, type_info = self._get_module(program)
    order = cpp_ir_converter.get_conversion_order(m, type_info)
    self.assertLen(order, 1)
    self.assertEqual(order[0].f.identifier, 'main')
    self.assertEqual(order[0].bindings, SymbolicBindings())


if __name__ == '__main__':
  test_base.main()
