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

"""Tests for xls.dslx.ir_converter."""

import subprocess
import sys
import textwrap
from typing import Text

from absl.testing import absltest
from xls.common import runfiles
from xls.dslx import fakefs_util
from xls.dslx import ir_converter
from xls.dslx import parser_helpers
from xls.dslx import typecheck


# IR parser binary. Reads from stdin and tries to parse the text as a package.
_PARSE_IR = 'xls/tools/parse_ir'


class IrConverterTest(absltest.TestCase):

  def parse_dsl_text(self, program):
    program = textwrap.dedent(program)
    filename = '/fake/test_module.x'
    with fakefs_util.scoped_fakefs(filename, program):
      m = parser_helpers.parse_text(
          program, name='test_module', print_on_error=True, filename=filename)
      self.assertEqual(m.name, 'test_module')
      return m

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

  def test_two_plus_two(self):
    m = self.parse_dsl_text("""\
    fn two_plus_two() -> u32 {
      u32:2 + u32:2
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(m, 'two_plus_two',
                                                  node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__two_plus_two() -> bits[32] {
          literal.1: bits[32] = literal(value=2, pos=0,1,6)
          literal.2: bits[32] = literal(value=2, pos=0,1,14)
          ret add.3: bits[32] = add(literal.1, literal.2, pos=0,1,8)
        }
        """)

  def test_negative_x(self):
    m = self.parse_dsl_text("""\
    fn negate(x: u32) -> u32 {
      -x
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(m, 'negate', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__negate(x: bits[32]) -> bits[32] {
          ret neg.2: bits[32] = neg(x, pos=0,1,2)
        }
        """)

  def test_let_binding(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let x: u32 = u32:2 in
      x+x
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[32] {
          literal.1: bits[32] = literal(value=2, pos=0,1,19)
          ret add.2: bits[32] = add(literal.1, literal.1, pos=0,2,3)
        }
        """)

  def test_let_tuple_binding(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let t = (u32:2, u32:3) in
      let (x, y) = t in
      x+y
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[32] {
          literal.1: bits[32] = literal(value=2, pos=0,1,15)
          literal.2: bits[32] = literal(value=3, pos=0,1,22)
          tuple.3: (bits[32], bits[32]) = tuple(literal.1, literal.2, pos=0,1,10)
          tuple_index.4: bits[32] = tuple_index(tuple.3, index=0, pos=0,2,7)
          tuple_index.5: bits[32] = tuple_index(tuple.3, index=1, pos=0,2,10)
          ret add.6: bits[32] = add(tuple_index.4, tuple_index.5, pos=0,3,3)
        }
        """)

  def test_let_tuple_binding_nested(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let t = (u32:2, (u32:3, (u32:4,), u32:5)) in
      let (x, (y, (z,), a)) = t in
      x+y+z+a
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[32] {
          literal.3: bits[32] = literal(value=4)
          literal.2: bits[32] = literal(value=3)
          tuple.4: (bits[32]) = tuple(literal.3)
          literal.5: bits[32] = literal(value=5)
          literal.1: bits[32] = literal(value=2)
          tuple.6: (bits[32], (bits[32]), bits[32]) = tuple(literal.2, tuple.4, literal.5)
          tuple.7: (bits[32], (bits[32], (bits[32]), bits[32])) = tuple(literal.1, tuple.6)
          tuple_index.9: (bits[32], (bits[32]), bits[32]) = tuple_index(tuple.7, index=1)
          tuple_index.8: bits[32] = tuple_index(tuple.7, index=0)
          tuple_index.10: bits[32] = tuple_index(tuple_index.9, index=0)
          tuple_index.11: (bits[32]) = tuple_index(tuple_index.9, index=1)
          add.14: bits[32] = add(tuple_index.8, tuple_index.10)
          tuple_index.12: bits[32] = tuple_index(tuple_index.11, index=0)
          add.15: bits[32] = add(add.14, tuple_index.12)
          tuple_index.13: bits[32] = tuple_index(tuple_index.9, index=2)
          ret add.16: bits[32] = add(add.15, tuple_index.13)
        }
        """)

  def test_concat(self):
    m = self.parse_dsl_text("""\
    fn f(x: bits[31]) -> u32 {
      bits[1]:1 ++ x
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[31]) -> bits[32] {
          literal.2: bits[1] = literal(value=1, pos=0,1,10)
          ret concat.3: bits[32] = concat(literal.2, x, pos=0,1,12)
        }
        """)

  def test_tuple_of_parameters(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8, y: u8) -> (u8, u8) {
      (x, y)
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8], y: bits[8]) -> (bits[8], bits[8]) {
          ret tuple.3: (bits[8], bits[8]) = tuple(x, y, pos=0,1,2)
        }
        """)

  def test_tuple_of_literals(self):
    m = self.parse_dsl_text("""\
    fn f() -> (u8, u8) {
      (u8:0xaa, u8:0x55)
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> (bits[8], bits[8]) {
          literal.1: bits[8] = literal(value=170, pos=0,1,6)
          literal.2: bits[8] = literal(value=85, pos=0,1,15)
          ret tuple.3: (bits[8], bits[8]) = tuple(literal.1, literal.2, pos=0,1,2)
        }
        """)

  def test_counted_for(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      for (i, accum): (u32, u32) in range(u32:0, u32:4) {
        accum + i
      }(u32:0)
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f_counted_for_0_body(i: bits[32], accum: bits[32]) -> bits[32] {
          ret add.5: bits[32] = add(accum, i)
        }

        fn __test_module__f() -> bits[32] {
          literal.1: bits[32] = literal(value=0)
          literal.2: bits[32] = literal(value=4)
          ret counted_for.6: bits[32] = counted_for(literal.1, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body)
        }
        """)

  def test_counted_for_destructuring(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let t = for (i, (x, y)): (u32, (u32, u8)) in range(u32:0, u32:4) {
        (x + i, y)
      }((u32:0, u8:0)) in
      t[u32:0]
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f_counted_for_0_body(i: bits[32], __loop_carry: (bits[32], bits[8])) -> (bits[32], bits[8]) {
          literal.7: bits[1] = literal(value=1)
          literal.9: bits[1] = literal(value=1)
          tuple_index.8: bits[32] = tuple_index(__loop_carry, index=0)
          and.10: bits[1] = and(literal.7, literal.9)
          literal.12: bits[1] = literal(value=1)
          add.14: bits[32] = add(tuple_index.8, i)
          tuple_index.11: bits[8] = tuple_index(__loop_carry, index=1)
          and.13: bits[1] = and(and.10, literal.12)
          ret tuple.15: (bits[32], bits[8]) = tuple(add.14, tuple_index.11)
        }

        fn __test_module__f() -> bits[32] {
          literal.1: bits[32] = literal(value=0)
          literal.2: bits[8] = literal(value=0)
          tuple.3: (bits[32], bits[8]) = tuple(literal.1, literal.2)
          counted_for.16: (bits[32], bits[8]) = counted_for(tuple.3, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body)
          literal.4: bits[32] = literal(value=4)
          literal.17: bits[32] = literal(value=0)
          ret tuple_index.18: bits[32] = tuple_index(counted_for.16, index=0)
        }
        """)

  def test_counted_for_parametric_const(self):
    m = self.parse_dsl_text("""\
    fn [N: u32] f(x: bits[N]) -> u32 {
      for (i, accum): (u32, u32) in range(u32:0, N) {
        accum + i
      }(u32:0)
    }
    fn main() -> u32 {
      f(bits[2]:0)
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f__2_counted_for_0_body(i: bits[32], accum: bits[32]) -> bits[32] {
          ret add.6: bits[32] = add(accum, i)
        }

        fn __test_module__f__2(x: bits[2]) -> bits[32] {
          literal.3: bits[32] = literal(value=0)
          literal.2: bits[32] = literal(value=2)
          ret counted_for.7: bits[32] = counted_for(literal.3, trip_count=2, stride=1, body=____test_module__f__2_counted_for_0_body)
        }

        fn __test_module__main() -> bits[32] {
          literal.8: bits[2] = literal(value=0)
          ret invoke.9: bits[32] = invoke(literal.8, to_apply=__test_module__f__2)
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
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__my_id(x: bits[32]) -> bits[32] {
          ret identity.2: bits[32] = identity(x)
        }

        fn ____test_module__f_counted_for_0_body(i: bits[32], accum: bits[32]) -> bits[32] {
          add.7: bits[32] = add(accum, i)
          ret invoke.8: bits[32] = invoke(add.7, to_apply=__test_module__my_id)
        }

        fn __test_module__f() -> bits[32] {
          literal.3: bits[32] = literal(value=0)
          literal.4: bits[32] = literal(value=4)
          ret counted_for.9: bits[32] = counted_for(literal.3, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body)
        }
        """)

  def test_counted_for_with_loop_invariants(self):
    m = self.parse_dsl_text("""\
    fn f() -> u32 {
      let outer_thing: u32 = u32:42 in
      let other_outer_thing: u32 = u32:24 in
      for (i, accum): (u32, u32) in range(u32:0, u32:4) {
        accum + i + outer_thing + other_outer_thing
      }(u32:0)
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f_counted_for_0_body(i: bits[32], accum: bits[32], other_outer_thing: bits[32], outer_thing: bits[32]) -> bits[32] {
          add.9: bits[32] = add(accum, i)
          add.10: bits[32] = add(add.9, outer_thing)
          ret add.11: bits[32] = add(add.10, other_outer_thing)
        }

        fn __test_module__f() -> bits[32] {
          literal.3: bits[32] = literal(value=0)
          literal.2: bits[32] = literal(value=24)
          literal.1: bits[32] = literal(value=42)
          literal.4: bits[32] = literal(value=4)
          ret counted_for.12: bits[32] = counted_for(literal.3, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body, invariant_args=[literal.2, literal.1])
        }
        """)

  def test_counted_for_with_tuple_accumulator(self):
    m = self.parse_dsl_text("""\
    fn f() -> (u32, u32) {
      for (i, (a, b)): (u32, (u32, u32)) in range(u32:0, u32:4) {
        (a+b, b+u32:1)
      }((u32:0, u32:1))
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(
        m, 'f', node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f_counted_for_0_body(i: bits[32], __loop_carry: (bits[32], bits[32])) -> (bits[32], bits[32]) {
          literal.7: bits[1] = literal(value=1)
          literal.9: bits[1] = literal(value=1)
          tuple_index.8: bits[32] = tuple_index(__loop_carry, index=0)
          tuple_index.11: bits[32] = tuple_index(__loop_carry, index=1)
          literal.15: bits[32] = literal(value=1)
          and.10: bits[1] = and(literal.7, literal.9)
          literal.12: bits[1] = literal(value=1)
          add.14: bits[32] = add(tuple_index.8, tuple_index.11)
          add.16: bits[32] = add(tuple_index.11, literal.15)
          and.13: bits[1] = and(and.10, literal.12)
          ret tuple.17: (bits[32], bits[32]) = tuple(add.14, add.16)
        }

        fn __test_module__f() -> (bits[32], bits[32]) {
          literal.1: bits[32] = literal(value=0)
          literal.2: bits[32] = literal(value=1)
          tuple.3: (bits[32], bits[32]) = tuple(literal.1, literal.2)
          literal.4: bits[32] = literal(value=4)
          ret counted_for.18: (bits[32], bits[32]) = counted_for(tuple.3, trip_count=4, stride=1, body=____test_module__f_counted_for_0_body)
        }
        """)

  def test_index(self):
    m = self.parse_dsl_text("""\
    fn f(x: uN[32][4]) -> u32 {
      x[u32:0]
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_one_function(m, 'f', node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[32][4]) -> bits[32] {
          literal.2: bits[32] = literal(value=0, pos=0,1,8)
          ret array_index.3: bits[32] = array_index(x, literal.2, pos=0,1,3)
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
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__callee() -> bits[32] {
          ret literal.1: bits[32] = literal(value=42, pos=0,1,6)
        }

        fn __test_module__caller() -> bits[32] {
          ret invoke.2: bits[32] = invoke(to_apply=__test_module__callee, pos=0,4,8)
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
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__callee(x: bits[32], y: bits[32]) -> bits[32] {
          ret add.3: bits[32] = add(x, y, pos=0,1,4)
        }

        fn __test_module__caller() -> bits[32] {
          literal.4: bits[32] = literal(value=2, pos=0,4,13)
          literal.5: bits[32] = literal(value=3, pos=0,4,20)
          ret invoke.6: bits[32] = invoke(literal.4, literal.5, to_apply=__test_module__callee, pos=0,4,8)
        }
        """)

  def test_cast_of_add(self):
    m = self.parse_dsl_text("""\
    fn main(x: u8, y: u8) -> u32 {
      (x + y) as u32
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8], y: bits[8]) -> bits[32] {
          add.3: bits[8] = add(x, y)
          ret zero_ext.4: bits[32] = zero_ext(add.3, new_bit_count=32)
        }
        """)

  def test_identity(self):
    m = self.parse_dsl_text("""\
    fn main(x: u8) -> u8 {
      x
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8]) -> bits[8] {
          ret identity.2: bits[8] = identity(x, pos=0,1,2)
        }
        """)

  def test_ternary(self):
    m = self.parse_dsl_text("""\
    fn main(x: bool) -> u8 {
      u8:42 if x else u8:24
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[1]) -> bits[8] {
          literal.3: bits[8] = literal(value=24, pos=0,1,21)
          literal.2: bits[8] = literal(value=42, pos=0,1,5)
          ret sel.4: bits[8] = sel(x, cases=[literal.3, literal.2], pos=0,1,8)
        }
        """)

  def test_package_level_constant_array_access(self):
    m = self.parse_dsl_text("""\
    const FOO = u8[2]:[1, 2];
    fn f() -> u8 { FOO[u32:0] }
    fn g() -> u8 { FOO[u32:1] }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[8] {
          literal.3: bits[8][2] = literal(value=[1, 2])
          literal.4: bits[32] = literal(value=0)
          literal.1: bits[8] = literal(value=1)
          literal.2: bits[8] = literal(value=2)
          ret array_index.5: bits[8] = array_index(literal.3, literal.4)
        }

        fn __test_module__g() -> bits[8] {
          literal.8: bits[8][2] = literal(value=[1, 2])
          literal.9: bits[32] = literal(value=1)
          literal.6: bits[8] = literal(value=1)
          literal.7: bits[8] = literal(value=2)
          ret array_index.10: bits[8] = array_index(literal.8, literal.9)
        }
        """)

  def test_package_level_constant_array(self):
    m = self.parse_dsl_text("""\
    const FOO = u8[2]:[1, 2];
    fn f() -> u8[2] { FOO }
    fn g() -> u8[2] { FOO }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f() -> bits[8][2] {
          literal.3: bits[8][2] = literal(value=[1, 2], pos=0,0,18)
          literal.1: bits[8] = literal(value=1, pos=0,0,19)
          literal.2: bits[8] = literal(value=2, pos=0,0,22)
          ret identity.4: bits[8][2] = identity(literal.3, pos=0,1,18)
        }

        fn __test_module__g() -> bits[8][2] {
          literal.7: bits[8][2] = literal(value=[1, 2], pos=0,0,18)
          literal.5: bits[8] = literal(value=1, pos=0,0,19)
          literal.6: bits[8] = literal(value=2, pos=0,0,22)
          ret identity.8: bits[8][2] = identity(literal.7, pos=0,2,18)
        }
        """)

  def test_match(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8) -> u2 {
      match x {
        u8:42 => u2:0;
        u8:64 => u2:1;
        _ => u2:2
      }
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[2] {
          literal.7: bits[8] = literal(value=64)
          literal.4: bits[8] = literal(value=42)
          eq.8: bits[1] = eq(literal.7, x)
          eq.5: bits[1] = eq(literal.4, x)
          concat.10: bits[2] = concat(eq.8, eq.5)
          one_hot.11: bits[3] = one_hot(concat.10, lsb_prio=true)
          literal.6: bits[2] = literal(value=0)
          literal.9: bits[2] = literal(value=1)
          literal.3: bits[2] = literal(value=2)
          literal.2: bits[1] = literal(value=1)
          ret one_hot_sel.12: bits[2] = one_hot_sel(one_hot.11, cases=[literal.6, literal.9, literal.3])
        }
        """)

  def test_match_dense(self):
    m = self.parse_dsl_text("""\
    fn f(x: u2) -> u8 {
      match x {
        u2:0 => u8:42;
        u2:1 => u8:64;
        u2:2 => u8:128;
        _ => u8:255
      }
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[2]) -> bits[8] {
          literal.10: bits[2] = literal(value=2)
          literal.7: bits[2] = literal(value=1)
          literal.4: bits[2] = literal(value=0)
          eq.11: bits[1] = eq(literal.10, x)
          eq.8: bits[1] = eq(literal.7, x)
          eq.5: bits[1] = eq(literal.4, x)
          concat.13: bits[3] = concat(eq.11, eq.8, eq.5)
          one_hot.14: bits[4] = one_hot(concat.13, lsb_prio=true)
          literal.6: bits[8] = literal(value=42)
          literal.9: bits[8] = literal(value=64)
          literal.12: bits[8] = literal(value=128)
          literal.3: bits[8] = literal(value=255)
          literal.2: bits[1] = literal(value=1)
          ret one_hot_sel.15: bits[8] = one_hot_sel(one_hot.14, cases=[literal.6, literal.9, literal.12, literal.3])
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
        ZERO => u8:42;
        ONE => u8:64;
        TWO => u8:128;
        _ => u8:255
      }
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[2]) -> bits[8] {
          literal.3: bits[2] = literal(value=2)
          literal.2: bits[2] = literal(value=1)
          literal.4: bits[2] = literal(value=0)
          eq.11: bits[1] = eq(literal.3, x)
          eq.9: bits[1] = eq(literal.2, x)
          eq.7: bits[1] = eq(literal.4, x)
          concat.13: bits[3] = concat(eq.11, eq.9, eq.7)
          one_hot.14: bits[4] = one_hot(concat.13, lsb_prio=true)
          literal.8: bits[8] = literal(value=42)
          literal.10: bits[8] = literal(value=64)
          literal.12: bits[8] = literal(value=128)
          literal.6: bits[8] = literal(value=255)
          literal.5: bits[1] = literal(value=1)
          ret one_hot_sel.15: bits[8] = one_hot_sel(one_hot.14, cases=[literal.8, literal.10, literal.12, literal.6])
        }
        """)

  def test_match_with_let(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8) -> u2 {
      match x {
        u8:42 => let x = u2:0 in x;
        u8:64 => let x = u2:1 in x;
        _ => let x = u2:2 in x
      }
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[2] {
          literal.7: bits[8] = literal(value=64)
          literal.4: bits[8] = literal(value=42)
          eq.8: bits[1] = eq(literal.7, x)
          eq.5: bits[1] = eq(literal.4, x)
          concat.10: bits[2] = concat(eq.8, eq.5)
          one_hot.11: bits[3] = one_hot(concat.10, lsb_prio=true)
          literal.6: bits[2] = literal(value=0)
          literal.9: bits[2] = literal(value=1)
          literal.3: bits[2] = literal(value=2)
          literal.2: bits[1] = literal(value=1)
          ret one_hot_sel.12: bits[2] = one_hot_sel(one_hot.11, cases=[literal.6, literal.9, literal.3])
        }
        """)

  def test_match_identity(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8) -> u2 {
      match x {
        u8:42 => u2:3;
        _ => x as u2
      }
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[2] {
          literal.4: bits[8] = literal(value=42)
          eq.5: bits[1] = eq(literal.4, x)
          concat.7: bits[1] = concat(eq.5)
          one_hot.8: bits[2] = one_hot(concat.7, lsb_prio=true)
          literal.6: bits[2] = literal(value=3)
          bit_slice.3: bits[2] = bit_slice(x, start=0, width=2)
          literal.2: bits[1] = literal(value=1)
          ret one_hot_sel.9: bits[2] = one_hot_sel(one_hot.8, cases=[literal.6, bit_slice.3])
        }
        """)

  def test_match_package_level_constant(self):
    m = self.parse_dsl_text("""\
    const FOO = u8:0xff;
    fn f(x: u8) -> u2 {
      match x {
        FOO => u2:0;
        _ => x as u2
      }
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[2] {
          literal.2: bits[8] = literal(value=255)
          eq.5: bits[1] = eq(literal.2, x)
          concat.7: bits[1] = concat(eq.5)
          one_hot.8: bits[2] = one_hot(concat.7, lsb_prio=true)
          literal.6: bits[2] = literal(value=0)
          bit_slice.4: bits[2] = bit_slice(x, start=0, width=2)
          literal.3: bits[1] = literal(value=1)
          ret one_hot_sel.9: bits[2] = one_hot_sel(one_hot.8, cases=[literal.6, bit_slice.4])
        }
        """)

  def test_bool_literals(self):
    m = self.parse_dsl_text("""\
    fn f(x: u8) -> bool {
      true if x == u8:42 else false
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[8]) -> bits[1] {
          literal.2: bits[8] = literal(value=42)
          eq.3: bits[1] = eq(x, literal.2)
          literal.5: bits[1] = literal(value=0)
          literal.4: bits[1] = literal(value=1)
          ret sel.6: bits[1] = sel(eq.3, cases=[literal.5, literal.4])
        }
        """)

  def test_parametric_invocation(self):
    m = self.parse_dsl_text("""\
    fn [N: u32] parametric_id(x: bits[N]) -> bits[N] {
      x+(N as bits[N])
    }

    fn main(x: u8) -> u8 {
      parametric_id(x)
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__parametric_id__8(x: bits[8]) -> bits[8] {
          literal.2: bits[32] = literal(value=8)
          bit_slice.3: bits[8] = bit_slice(literal.2, start=0, width=8)
          ret add.4: bits[8] = add(x, bit_slice.3)
        }

        fn __test_module__main(x: bits[8]) -> bits[8] {
          ret invoke.6: bits[8] = invoke(x, to_apply=__test_module__parametric_id__8)
        }
        """)

  def test_transitive_parametric_invocation(self):
    m = self.parse_dsl_text("""\
    fn [N: u32] parametric_id(x: bits[N]) -> bits[N] {
      x+(N as bits[N])
    }
    fn [M: u32] parametric_id_wrapper(x: bits[M]) -> bits[M] {
      parametric_id(x)
    }
    fn main(x: u8) -> u8 {
      parametric_id_wrapper(x)
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__parametric_id__8(x: bits[8]) -> bits[8] {
          literal.2: bits[32] = literal(value=8)
          bit_slice.3: bits[8] = bit_slice(literal.2, start=0, width=8)
          ret add.4: bits[8] = add(x, bit_slice.3)
        }

        fn __test_module__parametric_id_wrapper__8(x: bits[8]) -> bits[8] {
          literal.6: bits[32] = literal(value=8)
          ret invoke.7: bits[8] = invoke(x, to_apply=__test_module__parametric_id__8)
        }

        fn __test_module__main(x: bits[8]) -> bits[8] {
          ret invoke.9: bits[8] = invoke(x, to_apply=__test_module__parametric_id_wrapper__8)
        }
        """)

  def test_invocation_multi_symbol(self):
    m = self.parse_dsl_text("""\
    fn [M: u32, N: u32] parametric(x: bits[M], y: bits[N]) -> bits[M+N] {
      x ++ y
    }
    fn main() -> u8 {
      parametric(bits[3]:0, bits[5]:1)
    }""")
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(m, node_to_type)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__parametric__3_5(x: bits[3], y: bits[5]) -> bits[8] {
          literal.3: bits[32] = literal(value=3, pos=0,0,4)
          literal.4: bits[32] = literal(value=5, pos=0,0,12)
          ret concat.5: bits[8] = concat(x, y, pos=0,1,4)
        }

        fn __test_module__main() -> bits[8] {
          literal.6: bits[3] = literal(value=0, pos=0,4,21)
          literal.7: bits[5] = literal(value=1, pos=0,4,32)
          ret invoke.8: bits[8] = invoke(literal.6, literal.7, to_apply=__test_module__parametric__3_5, pos=0,4,12)
        }
        """)

  def test_identity_final_arg(self):
    m = self.parse_dsl_text("""
    fn main(x0: u19, x3: u29) -> u29 {
        let x15: u29 = u29:0 in
        let x17: u19 = (x0) + (x15 as u19) in
        x3
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x0: bits[19], x3: bits[29]) -> bits[29] {
          literal.3: bits[29] = literal(value=0)
          bit_slice.4: bits[19] = bit_slice(literal.3, start=0, width=19)
          add.5: bits[19] = add(x0, bit_slice.4)
          ret identity.6: bits[29] = identity(x3)
        }
        """)

  def test_bit_slice_cast(self):
    m = self.parse_dsl_text("""
    fn main(x: u2) -> u1 {
      x as u1
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[2]) -> bits[1] {
          ret bit_slice.2: bits[1] = bit_slice(x, start=0, width=1)
        }
        """)

  def test_tuple_index(self):
    m = self.parse_dsl_text("""
    fn main() -> u8 {
      let t = (u32:3, u8:4) in
      t[u32:1]
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main() -> bits[8] {
          literal.1: bits[32] = literal(value=3)
          literal.2: bits[8] = literal(value=4)
          tuple.3: (bits[32], bits[8]) = tuple(literal.1, literal.2)
          literal.4: bits[32] = literal(value=1)
          ret tuple_index.5: bits[8] = tuple_index(tuple.3, index=1)
        }
        """)

  def test_match_under_let(self):
    m = self.parse_dsl_text("""
    fn main(x: u8) -> u8 {
      let t = match x {
        u8:42 => u8:0xff;
        _ => x
      } in
      t
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8]) -> bits[8] {
          literal.3: bits[8] = literal(value=42)
          eq.4: bits[1] = eq(literal.3, x)
          concat.6: bits[1] = concat(eq.4)
          one_hot.7: bits[2] = one_hot(concat.6, lsb_prio=true)
          literal.5: bits[8] = literal(value=255)
          literal.2: bits[1] = literal(value=1)
          ret one_hot_sel.8: bits[8] = one_hot_sel(one_hot.7, cases=[literal.5, x])
        }
        """)

  def test_module_level_constant_dims(self):
    m = self.parse_dsl_text("""
    const BATCH_SIZE = u32:17;

    fn main(x: u32[BATCH_SIZE]) -> u32 {
      x[u32:16]
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[32][17]) -> bits[32] {
          literal.2: bits[32] = literal(value=16)
          ret array_index.3: bits[32] = array_index(x, literal.2)
        }
        """)

  def test_parametric_ir_conversion(self):
    m = self.parse_dsl_text("""
    fn [N: u32] parametric(x: bits[N]) -> u32 {
      N
    }

    fn main() -> u32 {
      parametric(bits[2]:0) + parametric(bits[3]:0)
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__parametric__2(x: bits[2]) -> bits[32] {
          literal.2: bits[32] = literal(value=2)
          ret identity.3: bits[32] = identity(literal.2)
        }

        fn __test_module__parametric__3(x: bits[3]) -> bits[32] {
          literal.5: bits[32] = literal(value=3)
          ret identity.6: bits[32] = identity(literal.5)
        }

        fn __test_module__main() -> bits[32] {
          literal.7: bits[2] = literal(value=0)
          literal.9: bits[3] = literal(value=0)
          invoke.8: bits[32] = invoke(literal.7, to_apply=__test_module__parametric__2)
          invoke.10: bits[32] = invoke(literal.9, to_apply=__test_module__parametric__3)
          ret add.11: bits[32] = add(invoke.8, invoke.10)
        }
        """)

  def test_fail_is_elided(self):
    m = self.parse_dsl_text("""
    fn main() -> u32 {
      fail!(u32:42)
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main() -> bits[32] {
          literal.1: bits[32] = literal(value=42)
          ret identity.2: bits[32] = identity(literal.1)
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
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(r: bits[9], l: bits[10], input: ((bits[1], (bits[6], bits[1]), bits[3])[3], bits[3], bits[1])) -> (bits[9], bits[10], (bits[64], bits[1])) {
          literal.6: bits[64] = literal(value=0)
          literal.7: bits[1] = literal(value=0)
          literal.4: bits[9] = literal(value=0)
          literal.5: bits[10] = literal(value=0)
          tuple.8: (bits[64], bits[1]) = tuple(literal.6, literal.7)
          ret tuple.9: (bits[9], bits[10], (bits[64], bits[1])) = tuple(literal.4, literal.5, tuple.8)
        }
        """)

  def test_array_update(self):
    m = self.parse_dsl_text("""
    fn main(input: u8[2]) -> u8[2] {
      update(input, u32:1, u8:0x42)
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(input: bits[8][2]) -> bits[8][2] {
          literal.4: bits[32] = literal(value=0)
          literal.2: bits[32] = literal(value=1)
          literal.8: bits[32] = literal(value=1)
          eq.6: bits[1] = eq(literal.4, literal.2)
          array_index.5: bits[8] = array_index(input, literal.4)
          literal.3: bits[8] = literal(value=66)
          eq.10: bits[1] = eq(literal.8, literal.2)
          array_index.9: bits[8] = array_index(input, literal.8)
          sel.7: bits[8] = sel(eq.6, cases=[array_index.5, literal.3])
          sel.11: bits[8] = sel(eq.10, cases=[array_index.9, literal.3])
          ret array.12: bits[8][2] = array(sel.7, sel.11)
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
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__main_counted_for_0_body(i: bits[32], accum: bits[8][2]) -> bits[8][2] {
          literal.8: bits[32] = literal(value=0)
          literal.12: bits[32] = literal(value=1)
          eq.10: bits[1] = eq(literal.8, i)
          array_index.9: bits[8] = array_index(accum, literal.8)
          bit_slice.7: bits[8] = bit_slice(i, start=0, width=8)
          eq.14: bits[1] = eq(literal.12, i)
          array_index.13: bits[8] = array_index(accum, literal.12)
          sel.11: bits[8] = sel(eq.10, cases=[array_index.9, bit_slice.7])
          sel.15: bits[8] = sel(eq.14, cases=[array_index.13, bit_slice.7])
          ret array.16: bits[8][2] = array(sel.11, sel.15)
        }

        fn __test_module__main() -> bits[8][2] {
          literal.3: bits[8][2] = literal(value=[0, 0])
          literal.1: bits[8] = literal(value=0)
          literal.2: bits[8] = literal(value=0)
          literal.4: bits[32] = literal(value=2)
          ret counted_for.17: bits[8][2] = counted_for(literal.3, trip_count=2, stride=1, body=____test_module__main_counted_for_0_body)
        }
        """)

  def test_array_ellipsis(self):
    m = self.parse_dsl_text("""
    fn main() -> u8[2] {
      u8[2]:[0, ...]
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main() -> bits[8][2] {
          literal.1: bits[8] = literal(value=0)
          ret literal.2: bits[8][2] = literal(value=[0, 0])
        }
        """)

  def test_counted_for_parametric_ref_in_body(self):
    m = self.parse_dsl_text("""\
    fn [N: u32] f(init: bits[N]) -> bits[N] {
      for (i, accum): (u32, bits[N]) in range(u32:0, u32:4) {
        accum as bits[N]
      }(init)
    }

    fn main() -> u32 {
      f(u32:0)
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn ____test_module__f__32_counted_for_0_body(i: bits[32], accum: bits[32]) -> bits[32] {
          ret zero_ext.6: bits[32] = zero_ext(accum, new_bit_count=32)
        }

        fn __test_module__f__32(init: bits[32]) -> bits[32] {
          literal.2: bits[32] = literal(value=32)
          literal.3: bits[32] = literal(value=4)
          ret counted_for.7: bits[32] = counted_for(init, trip_count=4, stride=1, body=____test_module__f__32_counted_for_0_body)
        }

        fn __test_module__main() -> bits[32] {
          literal.8: bits[32] = literal(value=0)
          ret invoke.9: bits[32] = invoke(literal.8, to_apply=__test_module__f__32)
        }
        """)

  def test_signed_comparisons(self):
    m = self.parse_dsl_text("""
    fn main(x: u32, y: u32) -> bool {
      sgt(x, y) and slt(x, y) and sge(x, y) and sle(x, y)
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[32], y: bits[32]) -> bits[1] {
          sgt.3: bits[1] = sgt(x, y)
          slt.4: bits[1] = slt(x, y)
          and.5: bits[1] = and(sgt.3, slt.4)
          sge.6: bits[1] = sge(x, y)
          and.7: bits[1] = and(and.5, sge.6)
          sle.8: bits[1] = sle(x, y)
          ret and.9: bits[1] = and(and.7, sle.8)
        }
        """)

  def test_signed_comparisons_via_signed_numbers(self):
    m = self.parse_dsl_text("""
    fn main(x: s32, y: s32) -> bool {
      x > y and x < y and x >= y and x <= y
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[32], y: bits[32]) -> bits[1] {
          sgt.3: bits[1] = sgt(x, y)
          slt.4: bits[1] = slt(x, y)
          and.5: bits[1] = and(sgt.3, slt.4)
          sge.6: bits[1] = sge(x, y)
          and.7: bits[1] = and(and.5, sge.6)
          sle.8: bits[1] = sle(x, y)
          ret and.9: bits[1] = and(and.7, sle.8)
        }
        """)

  def test_signex(self):
    m = self.parse_dsl_text("""
    fn main(x: u8) -> u32 {
      signex(x, u32:0)
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8]) -> bits[32] {
          literal.2: bits[32] = literal(value=0)
          ret sign_ext.3: bits[32] = sign_ext(x, new_bit_count=32)
        }
        """)

  def test_signex_accepts_signed_output_type(self):
    m = self.parse_dsl_text("""
    fn main(x: u8) -> s32 {
      signex(x, s32:0)
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8]) -> bits[32] {
          literal.2: bits[32] = literal(value=0)
          ret sign_ext.3: bits[32] = sign_ext(x, new_bit_count=32)
        }
        """)

  def test_extend_conversions(self):
    m = self.parse_dsl_text("""
    fn main(x: u8, y: s8) -> (u32, u32, s32, s32) {
      (x as u32, y as u32, x as s32, y as s32)
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(x: bits[8], y: bits[8]) -> (bits[32], bits[32], bits[32], bits[32]) {
          zero_ext.3: bits[32] = zero_ext(x, new_bit_count=32)
          sign_ext.4: bits[32] = sign_ext(y, new_bit_count=32)
          zero_ext.5: bits[32] = zero_ext(x, new_bit_count=32)
          sign_ext.6: bits[32] = sign_ext(y, new_bit_count=32)
          ret tuple.7: (bits[32], bits[32], bits[32], bits[32]) = tuple(zero_ext.3, sign_ext.4, zero_ext.5, sign_ext.6)
        }
        """)

  def test_one_hot_sel_splat_variadic(self):
    m = self.parse_dsl_text("""
    fn main(s: u2) -> u32 {
      one_hot_sel(s, u32[2]:[2, 3])
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__main(s: bits[2]) -> bits[32] {
          literal.4: bits[32][2] = literal(value=[2, 3])
          literal.5: bits[32] = literal(value=0)
          literal.7: bits[32] = literal(value=1)
          array_index.6: bits[32] = array_index(literal.4, literal.5)
          array_index.8: bits[32] = array_index(literal.4, literal.7)
          literal.2: bits[32] = literal(value=2)
          literal.3: bits[32] = literal(value=3)
          ret one_hot_sel.9: bits[32] = one_hot_sel(s, cases=[array_index.6, array_index.8])
        }
        """)

  def test_const_sized_array_in_typedef(self):
    m = self.parse_dsl_text("""
    const THING_COUNT = u32:2;
    type Foo = (
      u32[THING_COUNT]
    );
    fn get_thing(x: Foo, i: u32) -> u32 {
      let things: u32[THING_COUNT] = x[u32:0] in
      things[i]
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__get_thing(x: (bits[32][2]), i: bits[32]) -> bits[32] {
          tuple_index.4: bits[32][2] = tuple_index(x, index=0)
          literal.3: bits[32] = literal(value=0)
          ret array_index.5: bits[32] = array_index(tuple_index.4, i)
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
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[32]) -> bits[32] {
          literal.2: bits[32] = literal(value=0)
          eq.3: bits[1] = eq(x, literal.2)
          literal.5: bits[32] = literal(value=0)
          literal.4: bits[32] = literal(value=1)
          ret sel.6: bits[32] = sel(eq.3, cases=[literal.5, literal.4])
        }
        """)

  def test_single_element_bits_array_param(self):
    m = self.parse_dsl_text("""
    fn f(x: u32[1]) -> u32[1] {
      x
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[32][1]) -> bits[32][1] {
          ret identity.2: bits[32][1] = identity(x)
        }
        """)

  def test_single_element_enum_array_param(self):
    m = self.parse_dsl_text("""\
    enum Foo : u2 {}
    fn f(x: Foo[1]) -> Foo[1] {
      x
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(x: bits[2][1]) -> bits[2][1] {
          ret identity.2: bits[2][1] = identity(x)
        }
        """)

  def test_bit_slice_syntax(self):
    m = self.parse_dsl_text("""\
    fn f(x: u4) -> u2 {
      x[:2]+x[-2:]+x[1:3]+x[-3:-1]+x[0:-2]
    }
    """)
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        textwrap.dedent("""\
        package test_module

        fn __test_module__f(x: bits[4]) -> bits[2] {
          bit_slice.2: bits[2] = bit_slice(x, start=0, width=2)
          bit_slice.3: bits[2] = bit_slice(x, start=2, width=2)
          add.4: bits[2] = add(bit_slice.2, bit_slice.3)
          bit_slice.5: bits[2] = bit_slice(x, start=1, width=2)
          add.6: bits[2] = add(add.4, bit_slice.5)
          bit_slice.7: bits[2] = bit_slice(x, start=1, width=2)
          add.8: bits[2] = add(add.6, bit_slice.7)
          bit_slice.9: bits[2] = bit_slice(x, start=0, width=2)
          ret add.10: bits[2] = add(add.8, bit_slice.9)
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
    node_to_type = typecheck.check_module(m, f_import=None)
    converted = ir_converter.convert_module(
        m, node_to_type, emit_positions=False)
    self.assert_ir_equals_and_parses(
        converted, """\
        package test_module

        fn __test_module__f(xy: bits[32]) -> (bits[32], bits[32]) {
          ret tuple.2: (bits[32], bits[32]) = tuple(xy, xy)
        }
        """)


if __name__ == '__main__':
  absltest.main()
