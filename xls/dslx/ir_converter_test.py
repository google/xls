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

"""Tests for xls.dslx.ir_converter."""

import subprocess
import sys
import textwrap
from typing import Text

from xls.common import runfiles
from xls.common import test_base
from xls.dslx import fakefs_test_util
from xls.dslx import ir_converter
from xls.dslx import parser_helpers
from xls.dslx import span
from xls.dslx.python import cpp_ast as ast
from xls.dslx.python import cpp_parser
from xls.dslx.python import cpp_type_info
from xls.dslx.python import cpp_typecheck
from xls.dslx.python.import_routines import ImportCache


# IR parser binary. Reads from stdin and tries to parse the text as a package.
_PARSE_IR = 'xls/tools/parse_ir'


class IrConverterTest(test_base.TestCase):

  def parse_dsl_text(self, program):
    program = textwrap.dedent(program)
    filename = '/fake/test_module.x'
    with fakefs_test_util.scoped_fakefs(filename, program):
      m = parser_helpers.parse_text(
          program, name='test_module', print_on_error=True, filename=filename)
      self.assertEqual(m.name, 'test_module')
      return m

  def parse_and_convert(self, program: str):
    program = textwrap.dedent(program)
    filename = '/fake/test_module.x'
    with fakefs_test_util.scoped_fakefs(filename, program):
      try:
        m = parser_helpers.parse_text(
            program, name='test_module', print_on_error=True, filename=filename)
        self.assertEqual(m.name, 'test_module')
        node_to_type = cpp_typecheck.check_module(
            m, ImportCache(), additional_search_paths=())
        return ir_converter.convert_module(
            m, node_to_type, emit_positions=False)
      except (span.PositionalError, cpp_parser.CppParseError) as e:
        parser_helpers.pprint_positional_error(e)
        raise

  def assert_ir_equals_and_parses(self, text: Text, expected: Text) -> None:
    """Verify the given text parses as IR and matches expected."""
    expected = textwrap.dedent(expected)
    self.assertMultiLineEqual(text, expected)
    parse_ir = runfiles.get_path(_PARSE_IR)
    process = subprocess.Popen([parse_ir],
                               stdin=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
      _, pstderr = process.communicate(text.encode('utf-8'))
    except subprocess.CalledProcessError:
      print(text, file=sys.stderr)
      raise
    self.assertEqual(process.returncode, 0, pstderr.decode('utf-8'))

  def _typecheck(self, m: ast.Module) -> cpp_type_info.TypeInfo:
    import_cache = ImportCache()
    return cpp_typecheck.check_module(
        m, import_cache, additional_search_paths=())

  def test_two_plus_two(self):
    m = self.parse_dsl_text("""\
    fn two_plus_two() -> u32 {
      u32:2 + u32:2
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(m, 'two_plus_two',
                                                  node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__two_plus_two() -> bits[32] {
          literal.1: bits[32] = literal(value=2, id=1, pos=0,1,6)
          literal.2: bits[32] = literal(value=2, id=2, pos=0,1,14)
          ret add.3: bits[32] = add(literal.1, literal.2, id=3, pos=0,1,8)
        }
        """)

  def test_negative_x(self):
    m = self.parse_dsl_text("""\
    fn negate(x: u32) -> u32 {
      -x
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(m, 'negate', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__negate(x: bits[32]) -> bits[32] {
          ret neg.2: bits[32] = neg(x, id=2, pos=0,1,2)
        }
        """)

  def test_let_binding(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let x: u32 = u32:2;
      x+x
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[32] {
          literal.1: bits[32] = literal(value=2, id=1, pos=0,1,19)
          ret add.2: bits[32] = add(literal.1, literal.1, id=2, pos=0,2,3)
        }
        """)

  def test_let_tuple_binding(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let t = (u32:2, u32:3);
      let (x, y) = t;
      x+y
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[32] {
          literal.1: bits[32] = literal(value=2, id=1, pos=0,1,15)
          literal.2: bits[32] = literal(value=3, id=2, pos=0,1,22)
          t: (bits[32], bits[32]) = tuple(literal.1, literal.2, id=3, pos=0,1,10)
          x: bits[32] = tuple_index(t, index=0, id=4, pos=0,2,7)
          y: bits[32] = tuple_index(t, index=1, id=5, pos=0,2,10)
          ret add.6: bits[32] = add(x, y, id=6, pos=0,3,3)
        }
        """)

  def test_let_tuple_binding_nested(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let t = (u32:2, (u32:3, (u32:4,), u32:5));
      let (x, (y, (z,), a)) = t;
      x+y+z+a
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[32] {
          literal.3: bits[32] = literal(value=4, id=3)
          literal.2: bits[32] = literal(value=3, id=2)
          tuple.4: (bits[32]) = tuple(literal.3, id=4)
          literal.5: bits[32] = literal(value=5, id=5)
          literal.1: bits[32] = literal(value=2, id=1)
          tuple.6: (bits[32], (bits[32]), bits[32]) = tuple(literal.2, tuple.4, literal.5, id=6)
          t: (bits[32], (bits[32], (bits[32]), bits[32])) = tuple(literal.1, tuple.6, id=7)
          tuple_index.9: (bits[32], (bits[32]), bits[32]) = tuple_index(t, index=1, id=9)
          x: bits[32] = tuple_index(t, index=0, id=8)
          y: bits[32] = tuple_index(tuple_index.9, index=0, id=10)
          tuple_index.11: (bits[32]) = tuple_index(tuple_index.9, index=1, id=11)
          add.14: bits[32] = add(x, y, id=14)
          z: bits[32] = tuple_index(tuple_index.11, index=0, id=12)
          add.15: bits[32] = add(add.14, z, id=15)
          a: bits[32] = tuple_index(tuple_index.9, index=2, id=13)
          ret add.16: bits[32] = add(add.15, a, id=16)
        }
        """)

  def test_struct(self):
    m = self.parse_dsl_text("""\
    struct S {
      zub: u8,
      qux: u8,
    }

    fn f(a: S, b: S) -> u8 {
      let foo = a.zub + b.qux;
      (S { zub: u8:42, qux: u8:0 }).zub + (S { zub: u8:22, qux: u8:11 }).zub

    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(a: (bits[8], bits[8]), b: (bits[8], bits[8])) -> bits[8] {
          literal.6: bits[8] = literal(value=42, id=6)
          literal.7: bits[8] = literal(value=0, id=7)
          literal.10: bits[8] = literal(value=22, id=10)
          literal.11: bits[8] = literal(value=11, id=11)
          tuple.8: (bits[8], bits[8]) = tuple(literal.6, literal.7, id=8)
          tuple.12: (bits[8], bits[8]) = tuple(literal.10, literal.11, id=12)
          a_zub: bits[8] = tuple_index(a, index=0, id=3)
          b_qux: bits[8] = tuple_index(b, index=1, id=4)
          zub: bits[8] = tuple_index(tuple.8, index=0, id=9)
          zub__1: bits[8] = tuple_index(tuple.12, index=0, id=13)
          foo: bits[8] = add(a_zub, b_qux, id=5)
          ret add.14: bits[8] = add(zub, zub__1, id=14)
        }
        """)

  def test_concat(self):
    m = self.parse_dsl_text("""\
    fn f(x: bits[31]) -> u32 {
      bits[1]:1 ++ x
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[31]) -> bits[32] {
          literal.2: bits[1] = literal(value=1, id=2, pos=0,1,10)
          ret concat.3: bits[32] = concat(literal.2, x, id=3, pos=0,1,12)
        }
        """)

  def test_tuple_of_parameters(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8, y: u8) -> (u8, u8) {
      (x, y)
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8], y: bits[8]) -> (bits[8], bits[8]) {
          ret tuple.3: (bits[8], bits[8]) = tuple(x, y, id=3, pos=0,1,2)
        }
        """)

  def test_tuple_of_literals(self):
    m = self.parse_dsl_text("""\
    fn f() -> (u8, u8) {
      (u8:0xaa, u8:0x55)
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> (bits[8], bits[8]) {
          literal.1: bits[8] = literal(value=170, id=1, pos=0,1,6)
          literal.2: bits[8] = literal(value=85, id=2, pos=0,1,15)
          ret tuple.3: (bits[8], bits[8]) = tuple(literal.1, literal.2, id=3, pos=0,1,2)
        }
        """)

  def test_counted_for(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      for (i, accum): (u32, u32) in range(u32:0, u32:4) {
        accum + i
      }(u32:0)
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f_counted_for_0_body(i: bits[32], accum: bits[32]) -> bits[32] {
          ret add.5: bits[32] = add(accum, i, id=5)
        }

        fn __test_module__f() -> bits[32] {
          literal.1: bits[32] = literal(value=0, id=1)
          literal.2: bits[32] = literal(value=4, id=2)
          ret counted_for.6: bits[32] = counted_for(literal.1, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body, id=6)
        }
        """)

  def test_counted_for_destructuring(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let t = for (i, (x, y)): (u32, (u32, u8)) in range(u32:0, u32:4) {
        (x + i, y)
      }((u32:0, u8:0));
      t[0]
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f_counted_for_0_body(i: bits[32], __loop_carry: (bits[32], bits[8])) -> (bits[32], bits[8]) {
          literal.7: bits[1] = literal(value=1, id=7)
          literal.9: bits[1] = literal(value=1, id=9)
          tuple_index.8: bits[32] = tuple_index(__loop_carry, index=0, id=8)
          and.10: bits[1] = and(literal.7, literal.9, id=10)
          literal.12: bits[1] = literal(value=1, id=12)
          add.14: bits[32] = add(tuple_index.8, i, id=14)
          tuple_index.11: bits[8] = tuple_index(__loop_carry, index=1, id=11)
          and.13: bits[1] = and(and.10, literal.12, id=13)
          ret tuple.15: (bits[32], bits[8]) = tuple(add.14, tuple_index.11, id=15)
        }

        fn __test_module__f() -> bits[32] {
          literal.1: bits[32] = literal(value=0, id=1)
          literal.2: bits[8] = literal(value=0, id=2)
          tuple.3: (bits[32], bits[8]) = tuple(literal.1, literal.2, id=3)
          t: (bits[32], bits[8]) = counted_for(tuple.3, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body, id=16)
          literal.4: bits[32] = literal(value=4, id=4)
          literal.17: bits[32] = literal(value=0, id=17)
          ret tuple_index.18: bits[32] = tuple_index(t, index=0, id=18)
        }
        """)

  def test_counted_for_parametric_const(self):
    m = self.parse_dsl_text("""\
    fn f<N: u32>(x: bits[N]) -> u32 {
      for (i, accum): (u32, u32) in range(u32:0, N) {
        accum + i
      }(u32:0)
    }
    fn main() -> u32 {
      f(bits[2]:0)
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f__2_counted_for_0_body(i: bits[32], accum: bits[32]) -> bits[32] {
          ret add.6: bits[32] = add(accum, i, id=6)
        }

        fn __test_module__f__2(x: bits[2]) -> bits[32] {
          literal.3: bits[32] = literal(value=0, id=3)
          literal.2: bits[32] = literal(value=2, id=2)
          ret counted_for.7: bits[32] = counted_for(literal.3, trip_count=2, stride=1, body=____test_module__f__2_counted_for_0_body, id=7)
        }

        fn __test_module__main() -> bits[32] {
          literal.8: bits[2] = literal(value=0, id=8)
          ret invoke.9: bits[32] = invoke(literal.8, to_apply=__test_module__f__2, id=9)
        }
        """)

  def test_counted_for_invoking_function_from_body(self):
    m = self.parse_dsl_text("""\
    fn my_id(x: u32) -> u32 { x }
    fn f() -> u32 {
      for (i, accum): (u32, u32) in range(u32:0, u32:4) {
        my_id(accum + i)
      }(u32:0)
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__my_id(x: bits[32]) -> bits[32] {
          ret identity.2: bits[32] = identity(x, id=2)
        }

        fn ____test_module__f_counted_for_0_body(i: bits[32], accum: bits[32]) -> bits[32] {
          add.7: bits[32] = add(accum, i, id=7)
          ret invoke.8: bits[32] = invoke(add.7, to_apply=__test_module__my_id, id=8)
        }

        fn __test_module__f() -> bits[32] {
          literal.3: bits[32] = literal(value=0, id=3)
          literal.4: bits[32] = literal(value=4, id=4)
          ret counted_for.9: bits[32] = counted_for(literal.3, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body, id=9)
        }
        """)

  def test_counted_for_with_loop_invariants(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let outer_thing: u32 = u32:42;
      let other_outer_thing: u32 = u32:24;
      for (i, accum): (u32, u32) in range(u32:0, u32:4) {
        accum + i + outer_thing + other_outer_thing
      }(u32:0)
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f_counted_for_0_body(i: bits[32], accum: bits[32], other_outer_thing: bits[32], outer_thing: bits[32]) -> bits[32] {
          add.9: bits[32] = add(accum, i, id=9)
          add.10: bits[32] = add(add.9, outer_thing, id=10)
          ret add.11: bits[32] = add(add.10, other_outer_thing, id=11)
        }

        fn __test_module__f() -> bits[32] {
          literal.3: bits[32] = literal(value=0, id=3)
          literal.2: bits[32] = literal(value=24, id=2)
          literal.1: bits[32] = literal(value=42, id=1)
          literal.4: bits[32] = literal(value=4, id=4)
          ret counted_for.12: bits[32] = counted_for(literal.3, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body, invariant_args=[literal.2, literal.1], id=12)
        }
        """)

  def test_counted_for_with_tuple_accumulator(self):
    m = self.parse_dsl_text("""\
    fn f() -> (u32, u32) {
      for (i, (a, b)): (u32, (u32, u32)) in range(u32:0, u32:4) {
        (a+b, b+u32:1)
      }((u32:0, u32:1))
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f_counted_for_0_body(i: bits[32], __loop_carry: (bits[32], bits[32])) -> (bits[32], bits[32]) {
          literal.7: bits[1] = literal(value=1, id=7)
          literal.9: bits[1] = literal(value=1, id=9)
          tuple_index.8: bits[32] = tuple_index(__loop_carry, index=0, id=8)
          tuple_index.11: bits[32] = tuple_index(__loop_carry, index=1, id=11)
          literal.15: bits[32] = literal(value=1, id=15)
          and.10: bits[1] = and(literal.7, literal.9, id=10)
          literal.12: bits[1] = literal(value=1, id=12)
          add.14: bits[32] = add(tuple_index.8, tuple_index.11, id=14)
          add.16: bits[32] = add(tuple_index.11, literal.15, id=16)
          and.13: bits[1] = and(and.10, literal.12, id=13)
          ret tuple.17: (bits[32], bits[32]) = tuple(add.14, add.16, id=17)
        }

        fn __test_module__f() -> (bits[32], bits[32]) {
          literal.1: bits[32] = literal(value=0, id=1)
          literal.2: bits[32] = literal(value=1, id=2)
          tuple.3: (bits[32], bits[32]) = tuple(literal.1, literal.2, id=3)
          literal.4: bits[32] = literal(value=4, id=4)
          ret counted_for.18: (bits[32], bits[32]) = counted_for(tuple.3, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body, id=18)
        }
        """)

  def test_index(self):
    m = self.parse_dsl_text("""\
    fn f(x: uN[32][4]) -> u32 {
      x[u32:0]
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[32][4]) -> bits[32] {
          literal.2: bits[32] = literal(value=0, id=2, pos=0,1,8)
          ret array_index.3: bits[32] = array_index(x, indices=[literal.2], id=3, pos=0,1,3)
        }
        """)

  def test_invoke_nullary(self):
    m = self.parse_dsl_text("""\
    fn callee() -> u32 {
      u32:42
    }
    fn caller() -> u32 {
      callee()
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__callee() -> bits[32] {
          ret literal.1: bits[32] = literal(value=42, id=1, pos=0,1,6)
        }

        fn __test_module__caller() -> bits[32] {
          ret invoke.2: bits[32] = invoke(to_apply=__test_module__callee, id=2, pos=0,4,8)
        }
        """)

  def test_invoke_multiple_args(self):
    m = self.parse_dsl_text("""\
    fn callee(x: bits[32], y: bits[32]) -> bits[32] {
      x + y
    }
    fn caller() -> u32 {
      callee(u32:2, u32:3)
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__callee(x: bits[32], y: bits[32]) -> bits[32] {
          ret add.3: bits[32] = add(x, y, id=3, pos=0,1,4)
        }

        fn __test_module__caller() -> bits[32] {
          literal.4: bits[32] = literal(value=2, id=4, pos=0,4,13)
          literal.5: bits[32] = literal(value=3, id=5, pos=0,4,20)
          ret invoke.6: bits[32] = invoke(literal.4, literal.5, to_apply=__test_module__callee, id=6, pos=0,4,8)
        }
        """)

  def test_cast_of_add(self):
    m = self.parse_dsl_text("""\
    fn main(x: u8, y: u8) -> u32 {
      (x + y) as u32
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8], y: bits[8]) -> bits[32] {
          add.3: bits[8] = add(x, y, id=3)
          ret zero_ext.4: bits[32] = zero_ext(add.3, new_bit_count=32, id=4)
        }
        """)

  def test_identity(self):
    m = self.parse_dsl_text("""\
    fn main(x: u8) -> u8 {
      x
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8]) -> bits[8] {
          ret identity.2: bits[8] = identity(x, id=2, pos=0,1,2)
        }
        """)

  def test_ternary(self):
    m = self.parse_dsl_text("""\
    fn main(x: bool) -> u8 {
      u8:42 if x else u8:24
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[1]) -> bits[8] {
          literal.3: bits[8] = literal(value=24, id=3, pos=0,1,21)
          literal.2: bits[8] = literal(value=42, id=2, pos=0,1,5)
          ret sel.4: bits[8] = sel(x, cases=[literal.3, literal.2], id=4, pos=0,1,8)
        }
        """)

  def test_package_level_constant_array_access(self):
    m = self.parse_dsl_text("""\
    const FOO = u8[2]:[1, 2];
    fn f() -> u8 { FOO[u32:0] }
    fn g() -> u8 { FOO[u32:1] }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[8] {
          FOO: bits[8][2] = literal(value=[1, 2], id=3)
          literal.4: bits[32] = literal(value=0, id=4)
          literal.1: bits[8] = literal(value=1, id=1)
          literal.2: bits[8] = literal(value=2, id=2)
          ret array_index.5: bits[8] = array_index(FOO, indices=[literal.4], id=5)
        }

        fn __test_module__g() -> bits[8] {
          FOO: bits[8][2] = literal(value=[1, 2], id=8)
          literal.9: bits[32] = literal(value=1, id=9)
          literal.6: bits[8] = literal(value=1, id=6)
          literal.7: bits[8] = literal(value=2, id=7)
          ret array_index.10: bits[8] = array_index(FOO, indices=[literal.9], id=10)
        }
        """)

  def test_package_level_constant_array(self):
    m = self.parse_dsl_text("""\
    const FOO = u8[2]:[1, 2];
    fn f() -> u8[2] { FOO }
    fn g() -> u8[2] { FOO }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[8][2] {
          FOO: bits[8][2] = literal(value=[1, 2], id=3, pos=0,0,18)
          literal.1: bits[8] = literal(value=1, id=1, pos=0,0,19)
          literal.2: bits[8] = literal(value=2, id=2, pos=0,0,22)
          ret identity.4: bits[8][2] = identity(FOO, id=4, pos=0,1,18)
        }

        fn __test_module__g() -> bits[8][2] {
          FOO: bits[8][2] = literal(value=[1, 2], id=7, pos=0,0,18)
          literal.5: bits[8] = literal(value=1, id=5, pos=0,0,19)
          literal.6: bits[8] = literal(value=2, id=6, pos=0,0,22)
          ret identity.8: bits[8][2] = identity(FOO, id=8, pos=0,2,18)
        }
        """)

  def test_match(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8) -> u2 {
      match x {
        u8:42 => u2:0,
        u8:64 => u2:1,
        _ => u2:2
      }
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[2] {
          literal.7: bits[8] = literal(value=64, id=7)
          literal.4: bits[8] = literal(value=42, id=4)
          eq.8: bits[1] = eq(literal.7, x, id=8)
          eq.5: bits[1] = eq(literal.4, x, id=5)
          concat.10: bits[2] = concat(eq.8, eq.5, id=10)
          one_hot.11: bits[3] = one_hot(concat.10, lsb_prio=true, id=11)
          literal.6: bits[2] = literal(value=0, id=6)
          literal.9: bits[2] = literal(value=1, id=9)
          literal.3: bits[2] = literal(value=2, id=3)
          literal.2: bits[1] = literal(value=1, id=2)
          ret one_hot_sel.12: bits[2] = one_hot_sel(one_hot.11, cases=[literal.6, literal.9, literal.3], id=12)
        }
        """)

  def test_match_dense(self):
    m = self.parse_dsl_text("""\
    fn f(x: u2) -> u8 {
      match x {
        u2:0 => u8:42,
        u2:1 => u8:64,
        u2:2 => u8:128,
        _ => u8:255
      }
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[2]) -> bits[8] {
          literal.10: bits[2] = literal(value=2, id=10)
          literal.7: bits[2] = literal(value=1, id=7)
          literal.4: bits[2] = literal(value=0, id=4)
          eq.11: bits[1] = eq(literal.10, x, id=11)
          eq.8: bits[1] = eq(literal.7, x, id=8)
          eq.5: bits[1] = eq(literal.4, x, id=5)
          concat.13: bits[3] = concat(eq.11, eq.8, eq.5, id=13)
          one_hot.14: bits[4] = one_hot(concat.13, lsb_prio=true, id=14)
          literal.6: bits[8] = literal(value=42, id=6)
          literal.9: bits[8] = literal(value=64, id=9)
          literal.12: bits[8] = literal(value=128, id=12)
          literal.3: bits[8] = literal(value=255, id=3)
          literal.2: bits[1] = literal(value=1, id=2)
          ret one_hot_sel.15: bits[8] = one_hot_sel(one_hot.14, cases=[literal.6, literal.9, literal.12, literal.3], id=15)
        }
        """)

  def test_match_dense_consts(self):
    m = self.parse_dsl_text("""\
    type MyU2 = u2;
    const ZERO = MyU2:0;
    const ONE = MyU2:1;
    const TWO = MyU2:2;
    fn f(x: u2) -> u8 {
      match x {
        ZERO => u8:42,
        ONE => u8:64,
        TWO => u8:128,
        _ => u8:255
      }
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[2]) -> bits[8] {
          literal.3: bits[2] = literal(value=2, id=3)
          literal.2: bits[2] = literal(value=1, id=2)
          literal.4: bits[2] = literal(value=0, id=4)
          eq.11: bits[1] = eq(literal.3, x, id=11)
          eq.9: bits[1] = eq(literal.2, x, id=9)
          eq.7: bits[1] = eq(literal.4, x, id=7)
          concat.13: bits[3] = concat(eq.11, eq.9, eq.7, id=13)
          one_hot.14: bits[4] = one_hot(concat.13, lsb_prio=true, id=14)
          literal.8: bits[8] = literal(value=42, id=8)
          literal.10: bits[8] = literal(value=64, id=10)
          literal.12: bits[8] = literal(value=128, id=12)
          literal.6: bits[8] = literal(value=255, id=6)
          literal.5: bits[1] = literal(value=1, id=5)
          ret one_hot_sel.15: bits[8] = one_hot_sel(one_hot.14, cases=[literal.8, literal.10, literal.12, literal.6], id=15)
        }
        """)

  def test_match_with_let(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8) -> u2 {
      match x {
        u8:42 => let x = u2:0; x,
        u8:64 => let x = u2:1; x,
        _ => let x = u2:2; x
      }
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[2] {
          literal.7: bits[8] = literal(value=64, id=7)
          literal.4: bits[8] = literal(value=42, id=4)
          eq.8: bits[1] = eq(literal.7, x, id=8)
          eq.5: bits[1] = eq(literal.4, x, id=5)
          concat.10: bits[2] = concat(eq.8, eq.5, id=10)
          one_hot.11: bits[3] = one_hot(concat.10, lsb_prio=true, id=11)
          literal.6: bits[2] = literal(value=0, id=6)
          literal.9: bits[2] = literal(value=1, id=9)
          literal.3: bits[2] = literal(value=2, id=3)
          literal.2: bits[1] = literal(value=1, id=2)
          ret one_hot_sel.12: bits[2] = one_hot_sel(one_hot.11, cases=[literal.6, literal.9, literal.3], id=12)
        }
        """)

  def test_match_identity(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8) -> u2 {
      match x {
        u8:42 => u2:3,
        _ => x as u2
      }
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[2] {
          literal.4: bits[8] = literal(value=42, id=4)
          eq.5: bits[1] = eq(literal.4, x, id=5)
          concat.7: bits[1] = concat(eq.5, id=7)
          one_hot.8: bits[2] = one_hot(concat.7, lsb_prio=true, id=8)
          literal.6: bits[2] = literal(value=3, id=6)
          bit_slice.3: bits[2] = bit_slice(x, start=0, width=2, id=3)
          literal.2: bits[1] = literal(value=1, id=2)
          ret one_hot_sel.9: bits[2] = one_hot_sel(one_hot.8, cases=[literal.6, bit_slice.3], id=9)
        }
        """)

  def test_match_package_level_constant(self):
    m = self.parse_dsl_text("""\
    const FOO = u8:0xff;
    fn f(x: u8) -> u2 {
      match x {
        FOO => u2:0,
        _ => x as u2
      }
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[2] {
          literal.2: bits[8] = literal(value=255, id=2)
          eq.5: bits[1] = eq(literal.2, x, id=5)
          concat.7: bits[1] = concat(eq.5, id=7)
          one_hot.8: bits[2] = one_hot(concat.7, lsb_prio=true, id=8)
          literal.6: bits[2] = literal(value=0, id=6)
          bit_slice.4: bits[2] = bit_slice(x, start=0, width=2, id=4)
          literal.3: bits[1] = literal(value=1, id=3)
          ret one_hot_sel.9: bits[2] = one_hot_sel(one_hot.8, cases=[literal.6, bit_slice.4], id=9)
        }
        """)

  def test_bool_literals(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8) -> bool {
      true if x == u8:42 else false
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[1] {
          literal.2: bits[8] = literal(value=42, id=2)
          eq.3: bits[1] = eq(x, literal.2, id=3)
          literal.5: bits[1] = literal(value=0, id=5)
          literal.4: bits[1] = literal(value=1, id=4)
          ret sel.6: bits[1] = sel(eq.3, cases=[literal.5, literal.4], id=6)
        }
        """)

  def test_parametric_invocation(self):
    m = self.parse_dsl_text("""\
    fn parametric_id<N: u32>(x: bits[N]) -> bits[N] {
      x+(N as bits[N])
    }

    fn main(x: u8) -> u8 {
      parametric_id(x)
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__parametric_id__8(x: bits[8]) -> bits[8] {
          literal.2: bits[32] = literal(value=8, id=2)
          bit_slice.3: bits[8] = bit_slice(literal.2, start=0, width=8, id=3)
          ret add.4: bits[8] = add(x, bit_slice.3, id=4)
        }

        fn __test_module__main(x: bits[8]) -> bits[8] {
          ret invoke.6: bits[8] = invoke(x, to_apply=__test_module__parametric_id__8, id=6)
        }
        """)

  def test_transitive_parametric_invocation(self):
    m = self.parse_dsl_text("""\
    fn parametric_id<N: u32>(x: bits[N]) -> bits[N] {
      x+(N as bits[N])
    }
    fn parametric_id_wrapper<M: u32>(x: bits[M]) -> bits[M] {
      parametric_id(x)
    }
    fn main(x: u8) -> u8 {
      parametric_id_wrapper(x)
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__parametric_id__8(x: bits[8]) -> bits[8] {
          literal.2: bits[32] = literal(value=8, id=2)
          bit_slice.3: bits[8] = bit_slice(literal.2, start=0, width=8, id=3)
          ret add.4: bits[8] = add(x, bit_slice.3, id=4)
        }

        fn __test_module__parametric_id_wrapper__8(x: bits[8]) -> bits[8] {
          literal.6: bits[32] = literal(value=8, id=6)
          ret invoke.7: bits[8] = invoke(x, to_apply=__test_module__parametric_id__8, id=7)
        }

        fn __test_module__main(x: bits[8]) -> bits[8] {
          ret invoke.9: bits[8] = invoke(x, to_apply=__test_module__parametric_id_wrapper__8, id=9)
        }
        """)

  def test_invocation_multi_symbol(self):
    m = self.parse_dsl_text("""\
    fn parametric<M: u32, N: u32, R: u32 = M + N>(x: bits[M], y: bits[N]) -> bits[R] {
      x ++ y
    }
    fn main() -> u8 {
      parametric(bits[3]:0, bits[5]:1)
    }""")
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__parametric__3_5_8(x: bits[3], y: bits[5]) -> bits[8] {
          literal.3: bits[32] = literal(value=3, id=3, pos=0,0,14)
          literal.4: bits[32] = literal(value=5, id=4, pos=0,0,22)
          literal.5: bits[32] = literal(value=8, id=5, pos=0,0,30)
          ret concat.6: bits[8] = concat(x, y, id=6, pos=0,1,4)
        }

        fn __test_module__main() -> bits[8] {
          literal.7: bits[3] = literal(value=0, id=7, pos=0,4,21)
          literal.8: bits[5] = literal(value=1, id=8, pos=0,4,32)
          ret invoke.9: bits[8] = invoke(literal.7, literal.8, to_apply=__test_module__parametric__3_5_8, id=9, pos=0,4,12)
        }
        """)

  def test_identity_final_arg(self):
    m = self.parse_dsl_text("""
    fn main(x0: u19, x3: u29) -> u29 {
        let x15: u29 = u29:0;
        let x17: u19 = (x0) + (x15 as u19);
        x3
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x0: bits[19], x3: bits[29]) -> bits[29] {
          literal.3: bits[29] = literal(value=0, id=3)
          bit_slice.4: bits[19] = bit_slice(literal.3, start=0, width=19, id=4)
          x17: bits[19] = add(x0, bit_slice.4, id=5)
          ret identity.6: bits[29] = identity(x3, id=6)
        }
        """)

  def test_bit_slice_cast(self):
    m = self.parse_dsl_text("""
    fn main(x: u2) -> u1 {
      x as u1
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[2]) -> bits[1] {
          ret bit_slice.2: bits[1] = bit_slice(x, start=0, width=1, id=2)
        }
        """)

  def test_tuple_index(self):
    m = self.parse_dsl_text("""
    fn main() -> u8 {
      let t = (u32:3, u8:4);
      t[1]
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main() -> bits[8] {
          literal.1: bits[32] = literal(value=3, id=1)
          literal.2: bits[8] = literal(value=4, id=2)
          t: (bits[32], bits[8]) = tuple(literal.1, literal.2, id=3)
          literal.4: bits[32] = literal(value=1, id=4)
          ret tuple_index.5: bits[8] = tuple_index(t, index=1, id=5)
        }
        """)

  def test_match_under_let(self):
    m = self.parse_dsl_text("""
    fn main(x: u8) -> u8 {
      let t = match x {
        u8:42 => u8:0xff,
        _ => x
      };
      t
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8]) -> bits[8] {
          literal.3: bits[8] = literal(value=42, id=3)
          eq.4: bits[1] = eq(literal.3, x, id=4)
          concat.6: bits[1] = concat(eq.4, id=6)
          one_hot.7: bits[2] = one_hot(concat.6, lsb_prio=true, id=7)
          literal.5: bits[8] = literal(value=255, id=5)
          literal.2: bits[1] = literal(value=1, id=2)
          ret t: bits[8] = one_hot_sel(one_hot.7, cases=[literal.5, x], id=8)
        }
        """)

  def test_module_level_constant_dims(self):
    m = self.parse_dsl_text("""
    const BATCH_SIZE = u32:17;

    fn main(x: u32[BATCH_SIZE]) -> u32 {
      x[u32:16]
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[32][17]) -> bits[32] {
          literal.2: bits[32] = literal(value=16, id=2)
          ret array_index.3: bits[32] = array_index(x, indices=[literal.2], id=3)
        }
        """)

  def test_parametric_ir_conversion(self):
    m = self.parse_dsl_text("""
    fn parametric<N: u32>(x: bits[N]) -> u32 {
      N
    }

    fn main() -> u32 {
      parametric(bits[2]:0) + parametric(bits[3]:0)
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__parametric__2(x: bits[2]) -> bits[32] {
          literal.2: bits[32] = literal(value=2, id=2)
          ret identity.3: bits[32] = identity(literal.2, id=3)
        }

        fn __test_module__parametric__3(x: bits[3]) -> bits[32] {
          literal.5: bits[32] = literal(value=3, id=5)
          ret identity.6: bits[32] = identity(literal.5, id=6)
        }

        fn __test_module__main() -> bits[32] {
          literal.7: bits[2] = literal(value=0, id=7)
          literal.9: bits[3] = literal(value=0, id=9)
          invoke.8: bits[32] = invoke(literal.7, to_apply=__test_module__parametric__2, id=8)
          invoke.10: bits[32] = invoke(literal.9, to_apply=__test_module__parametric__3, id=10)
          ret add.11: bits[32] = add(invoke.8, invoke.10, id=11)
        }
        """)

  def test_fail_is_elided(self):
    m = self.parse_dsl_text("""
    fn main() -> u32 {
      fail!(u32:42)
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main() -> bits[32] {
          literal.1: bits[32] = literal(value=42, id=1)
          ret identity.2: bits[32] = identity(literal.1, id=2)
        }
        """)

  def test_nested_tuple_signature(self):
    m = self.parse_dsl_text("""
    type Foo = u3;

    type MyTup = (u6, u1);

    type TupOfThings = (u1, MyTup, Foo);

    type MoreStructured = (
      TupOfThings[3],
      u3,
      u1,
    );

    type Data = (u64, u1);

    fn main(r: u9, l: u10, input: MoreStructured) -> (u9, u10, Data) {
      (u9:0, u10:0, (u64:0, u1:0))
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(r: bits[9], l: bits[10], input: ((bits[1], (bits[6], bits[1]), bits[3])[3], bits[3], bits[1])) -> (bits[9], bits[10], (bits[64], bits[1])) {
          literal.6: bits[64] = literal(value=0, id=6)
          literal.7: bits[1] = literal(value=0, id=7)
          literal.4: bits[9] = literal(value=0, id=4)
          literal.5: bits[10] = literal(value=0, id=5)
          tuple.8: (bits[64], bits[1]) = tuple(literal.6, literal.7, id=8)
          ret tuple.9: (bits[9], bits[10], (bits[64], bits[1])) = tuple(literal.4, literal.5, tuple.8, id=9)
        }
        """)

  def test_array_update(self):
    m = self.parse_dsl_text("""
    fn main(input: u8[2]) -> u8[2] {
      update(input, u32:1, u8:0x42)
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(input: bits[8][2]) -> bits[8][2] {
          literal.3: bits[8] = literal(value=66, id=3)
          literal.2: bits[32] = literal(value=1, id=2)
          ret array_update.4: bits[8][2] = array_update(input, literal.3, indices=[literal.2], id=4)
        }
        """)

  def test_array_update_in_loop(self):
    m = self.parse_dsl_text("""
    fn main() -> u8[2] {
      for (i, accum): (u32, u8[2]) in range(u32:0, u32:2) {
        update(accum, i, i as u8)
      }(u8[2]:[0, 0])
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__main_counted_for_0_body(i: bits[32], accum: bits[8][2]) -> bits[8][2] {
          bit_slice.7: bits[8] = bit_slice(i, start=0, width=8, id=7)
          ret array_update.8: bits[8][2] = array_update(accum, bit_slice.7, indices=[i], id=8)
        }

        fn __test_module__main() -> bits[8][2] {
          literal.3: bits[8][2] = literal(value=[0, 0], id=3)
          literal.1: bits[8] = literal(value=0, id=1)
          literal.2: bits[8] = literal(value=0, id=2)
          literal.4: bits[32] = literal(value=2, id=4)
          ret counted_for.9: bits[8][2] = counted_for(literal.3, trip_count=2, stride=1, body=____test_module__main_counted_for_0_body, id=9)
        }
        """)

  def test_array_ellipsis(self):
    converted = self.parse_and_convert("""
    fn main() -> u8[2] {
      u8[2]:[0, ...]
    }
    """)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main() -> bits[8][2] {
          literal.1: bits[8] = literal(value=0, id=1)
          ret literal.2: bits[8][2] = literal(value=[0, 0], id=2)
        }
        """)

  def test_non_const_array_ellipsis(self):
    m = self.parse_dsl_text("""
    fn main(x: bits[8]) -> u8[4] {
      u8[4]:[u8:0, x, ...]
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8]) -> bits[8][4] {
          literal.2: bits[8] = literal(value=0, id=2)
          ret array.3: bits[8][4] = array(literal.2, x, x, x, id=3)
        }
        """)

  def test_counted_for_parametric_ref_in_body(self):
    m = self.parse_dsl_text("""\
    fn f<N:u32>(init: bits[N]) -> bits[N] {
      for (i, accum): (u32, bits[N]) in range(u32:0, u32:4) {
        accum as bits[N]
      }(init)
    }

    fn main() -> u32 {
      f(u32:0)
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f__32_counted_for_0_body(i: bits[32], accum: bits[32]) -> bits[32] {
          ret zero_ext.6: bits[32] = zero_ext(accum, new_bit_count=32, id=6)
        }

        fn __test_module__f__32(init: bits[32]) -> bits[32] {
          literal.2: bits[32] = literal(value=32, id=2)
          literal.3: bits[32] = literal(value=4, id=3)
          ret counted_for.7: bits[32] = counted_for(init, trip_count=4, stride=1, body=____test_module__f__32_counted_for_0_body, id=7)
        }

        fn __test_module__main() -> bits[32] {
          literal.8: bits[32] = literal(value=0, id=8)
          ret invoke.9: bits[32] = invoke(literal.8, to_apply=__test_module__f__32, id=9)
        }
        """)

  def test_signed_comparisons(self):
    m = self.parse_dsl_text("""
    fn main(x: u32, y: u32) -> bool {
      sgt(x, y) && slt(x, y) && sge(x, y) && sle(x, y)
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[32], y: bits[32]) -> bits[1] {
          sgt.3: bits[1] = sgt(x, y, id=3)
          slt.4: bits[1] = slt(x, y, id=4)
          and.5: bits[1] = and(sgt.3, slt.4, id=5)
          sge.6: bits[1] = sge(x, y, id=6)
          and.7: bits[1] = and(and.5, sge.6, id=7)
          sle.8: bits[1] = sle(x, y, id=8)
          ret and.9: bits[1] = and(and.7, sle.8, id=9)
        }
        """)

  def test_signed_comparisons_via_signed_numbers(self):
    m = self.parse_dsl_text("""
    fn main(x: s32, y: s32) -> bool {
      x > y && x < y && x >= y && x <= y
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[32], y: bits[32]) -> bits[1] {
          sgt.3: bits[1] = sgt(x, y, id=3)
          slt.4: bits[1] = slt(x, y, id=4)
          and.5: bits[1] = and(sgt.3, slt.4, id=5)
          sge.6: bits[1] = sge(x, y, id=6)
          and.7: bits[1] = and(and.5, sge.6, id=7)
          sle.8: bits[1] = sle(x, y, id=8)
          ret and.9: bits[1] = and(and.7, sle.8, id=9)
        }
        """)

  def test_signex(self):
    m = self.parse_dsl_text("""
    fn main(x: u8) -> u32 {
      signex(x, u32:0)
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8]) -> bits[32] {
          literal.2: bits[32] = literal(value=0, id=2)
          ret sign_ext.3: bits[32] = sign_ext(x, new_bit_count=32, id=3)
        }
        """)

  def test_signex_accepts_signed_output_type(self):
    m = self.parse_dsl_text("""
    fn main(x: u8) -> s32 {
      signex(x, s32:0)
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8]) -> bits[32] {
          literal.2: bits[32] = literal(value=0, id=2)
          ret sign_ext.3: bits[32] = sign_ext(x, new_bit_count=32, id=3)
        }
        """)

  def test_extend_conversions(self):
    m = self.parse_dsl_text("""
    fn main(x: u8, y: s8) -> (u32, u32, s32, s32) {
      (x as u32, y as u32, x as s32, y as s32)
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8], y: bits[8]) -> (bits[32], bits[32], bits[32], bits[32]) {
          zero_ext.3: bits[32] = zero_ext(x, new_bit_count=32, id=3)
          sign_ext.4: bits[32] = sign_ext(y, new_bit_count=32, id=4)
          zero_ext.5: bits[32] = zero_ext(x, new_bit_count=32, id=5)
          sign_ext.6: bits[32] = sign_ext(y, new_bit_count=32, id=6)
          ret tuple.7: (bits[32], bits[32], bits[32], bits[32]) = tuple(zero_ext.3, sign_ext.4, zero_ext.5, sign_ext.6, id=7)
        }
        """)

  def test_one_hot_sel_splat_variadic(self):
    m = self.parse_dsl_text("""
    fn main(s: u2) -> u32 {
      one_hot_sel(s, u32[2]:[2, 3])
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(s: bits[2]) -> bits[32] {
          literal.2: bits[32] = literal(value=2, id=2)
          literal.3: bits[32] = literal(value=3, id=3)
          literal.4: bits[32][2] = literal(value=[2, 3], id=4)
          ret one_hot_sel.5: bits[32] = one_hot_sel(s, cases=[literal.2, literal.3], id=5)
        }
        """)

  def test_const_sized_array_in_typedef(self):
    m = self.parse_dsl_text("""
    const THING_COUNT = u32:2;
    type Foo = (
      u32[THING_COUNT]
    );
    fn get_thing(x: Foo, i: u32) -> u32 {
      let things: u32[THING_COUNT] = x[0];
      things[i]
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__get_thing(x: (bits[32][2]), i: bits[32]) -> bits[32] {
          things: bits[32][2] = tuple_index(x, index=0, id=4)
          literal.3: bits[32] = literal(value=0, id=3)
          ret array_index.5: bits[32] = array_index(things, indices=[i], id=5)
        }
        """)

  def test_enum_use(self):
    m = self.parse_dsl_text("""
    enum Foo : u32 {
      THING = 0,
      OTHER = 1,
    }
    fn f(x: Foo) -> Foo {
      Foo::OTHER if x == Foo::THING else Foo::THING
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[32]) -> bits[32] {
          literal.2: bits[32] = literal(value=0, id=2)
          eq.3: bits[1] = eq(x, literal.2, id=3)
          literal.5: bits[32] = literal(value=0, id=5)
          literal.4: bits[32] = literal(value=1, id=4)
          ret sel.6: bits[32] = sel(eq.3, cases=[literal.5, literal.4], id=6)
        }
        """)

  def test_single_element_bits_array_param(self):
    m = self.parse_dsl_text("""
    fn f(x: u32[1]) -> u32[1] {
      x
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[32][1]) -> bits[32][1] {
          ret identity.2: bits[32][1] = identity(x, id=2)
        }
        """)

  def test_single_element_enum_array_param(self):
    m = self.parse_dsl_text("""\
    enum Foo : u2 {}
    fn f(x: Foo[1]) -> Foo[1] {
      x
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[2][1]) -> bits[2][1] {
          ret identity.2: bits[2][1] = identity(x, id=2)
        }
        """)

  def test_bit_slice_syntax(self):
    m = self.parse_dsl_text("""\
    fn f(x: u4) -> u2 {
      x[:2]+x[-2:]+x[1:3]+x[-3:-1]+x[0:-2]
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        textwrap.dedent("""\
        package test_module

        fn __test_module__f(x: bits[4]) -> bits[2] {
          bit_slice.2: bits[2] = bit_slice(x, start=0, width=2, id=2)
          bit_slice.3: bits[2] = bit_slice(x, start=2, width=2, id=3)
          add.4: bits[2] = add(bit_slice.2, bit_slice.3, id=4)
          bit_slice.5: bits[2] = bit_slice(x, start=1, width=2, id=5)
          add.6: bits[2] = add(add.4, bit_slice.5, id=6)
          bit_slice.7: bits[2] = bit_slice(x, start=1, width=2, id=7)
          add.8: bits[2] = add(add.6, bit_slice.7, id=8)
          bit_slice.9: bits[2] = bit_slice(x, start=0, width=2, id=9)
          ret add.10: bits[2] = add(add.8, bit_slice.9, id=10)
        }
        """), converted)

  def test_width_slice(self):
    m = self.parse_dsl_text("""\
    fn f(x: u32, y: u32) -> u8 {
      x[2+:u8]+x[y+:u8]
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        textwrap.dedent("""\
        package test_module

        fn __test_module__f(x: bits[32], y: bits[32]) -> bits[8] {
          literal.3: bits[32] = literal(value=2, id=3)
          dynamic_bit_slice.4: bits[8] = dynamic_bit_slice(x, literal.3, width=8, id=4)
          dynamic_bit_slice.5: bits[8] = dynamic_bit_slice(x, y, width=8, id=5)
          ret add.6: bits[8] = add(dynamic_bit_slice.4, dynamic_bit_slice.5, id=6)
        }
        """), converted)

  def test_basic_struct(self):
    m = self.parse_dsl_text("""\
    struct Point {
      x: u32,
      y: u32,
    }

    fn f(xy: u32) -> Point {
      Point { x: xy, y: xy }
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(xy: bits[32]) -> (bits[32], bits[32]) {
          ret tuple.2: (bits[32], bits[32]) = tuple(xy, xy, id=2)
        }
        """)

  def test_splat_struct_instance(self):
    m = self.parse_dsl_text("""\
    struct Point {
      x: u32,
      y: u32,
    }

    fn f(p: Point, new_y: u32) -> Point {
      Point { y: new_y, ..p }
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(p: (bits[32], bits[32]), new_y: bits[32]) -> (bits[32], bits[32]) {
          tuple_index.3: bits[32] = tuple_index(p, index=0, id=3)
          ret tuple.4: (bits[32], bits[32]) = tuple(tuple_index.3, new_y, id=4)
        }
        """)

  def test_array_concat_0(self):
    m = self.parse_dsl_text("""\
    fn f(in1: u32[2]) -> u32 {
      let x : u32[4] = in1 ++ in1;
      x[u32:0]
    }
    """)
    node_to_type = self._typecheck(m)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(in1: bits[32][2]) -> bits[32] {
          x: bits[32][4] = array_concat(in1, in1, id=2)
          literal.3: bits[32] = literal(value=0, id=3)
          ret array_index.4: bits[32] = array_index(x, indices=[literal.3], id=4)
        }
        """)


if __name__ == '__main__':
  test_base.main()
