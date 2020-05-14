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
#
# Lint as: python3

"""Tests for xls.dslx.interpreter."""

import io
import sys
import textwrap
from typing import Text

import unittest.mock as mock
from pyfakefs import fake_filesystem_unittest as ffu

from xls.dslx.concrete_type import ArrayType
from xls.dslx.interpreter import interpreter
from xls.dslx.interpreter import parse_and_interpret
from xls.dslx.xls_type_error import XlsTypeError
from absl.testing import absltest as unittest


class InterpreterTest(unittest.TestCase):

  def _parse_and_test(self, program: Text, trace_all: bool = False) -> None:
    with ffu.Patcher() as patcher:
      filename = '/fake/test_module.x'
      patcher.fs.CreateFile(filename, contents=program)
      self.assertFalse(
          parse_and_interpret.parse_and_test(
              program, 'test_module', trace_all=trace_all, filename=filename))

  def test_two_plus_two_module_test(self):
    program = textwrap.dedent("""\
    test two_plus_two_is_four {
      let x: u32 = u32:2 in
      let y: u32 = x + x in
      let expected: u32 = u32:4 in
      assert_eq(y, expected)
    }
    """)
    self._parse_and_test(program)

  def test_two_plus_two_fail_module_test(self):
    program = textwrap.dedent("""\
    test two_plus_two_is_four {
      let x: u32 = u32:2 in
      let y: u32 = x + x in
      let expected: u32 = u32:5 in
      assert_eq(y, expected)
    }
    """)
    with self.assertRaises(interpreter.FailureError):
      self._parse_and_test(program)

  def test_pad_bits_via_concat(self):
    program = textwrap.dedent("""\
    test pad_to_four_bits {
      let x: bits[2] = bits[2]:0b10 in
      let y: bits[4] = bits[2]:0 ++ x in
      let expected: bits[4] = bits[4]:0b0010 in
      assert_eq(y, expected)
    }
    """)
    self._parse_and_test(program)

  def test_invocation(self):
    program = textwrap.dedent("""\
    fn id(x: u32) -> u32 { x }
    test identity_invocation {
      assert_eq(u32:42, id(u32:42))
    }
    """)
    self._parse_and_test(program)

  def test_cast(self):
    program = textwrap.dedent("""\
    test cast_to_u32 {
      let x: bits[3] = bits[3]:0b010 in
      assert_eq(u32:4, (x + x) as u32)
    }
    """)
    self._parse_and_test(program)

  def test_parametric_invocation(self):
    program = textwrap.dedent("""\
    fn [N: u32] id(x: bits[N]) -> bits[N] { x }
    test different_parametric_invocations {
      assert_eq(bits[5]:0b01111, id(bits[2]:0b01) ++ id(bits[3]:0b111))
    }
    """)
    self._parse_and_test(program)

  def test_parametric_binding(self):
    program = textwrap.dedent("""\
    fn [N: u32] add_num_bits(x: bits[N]) -> bits[N] { x+(N as bits[N]) }
    test different_parametric_invocations {
      assert_eq(bits[2]:3, add_num_bits(bits[2]:1))
    }
    """)
    self._parse_and_test(program)

  def test_simple_subtract(self):
    program = textwrap.dedent("""\
    test simple_subtract {
      let x: u32 = u32:5 in
      let y: u32 = u32:4 in
      assert_eq(u32:1, x-y)
    }
    """)
    self._parse_and_test(program)

  def test_subtract_a_negative_u32(self):
    program = textwrap.dedent("""\
    test simple_subtract {
      let x: u32 = u32:5 in
      let y: u32 = u32:-1 in
      assert_eq(u32:6, x-y)
    }
    """)
    self._parse_and_test(program)

  def test_add_a_negative_u32(self):
    program = textwrap.dedent("""\
    test simple_add {
      let x: u32 = u32:0 in
      let y: u32 = u32:-2 in
      assert_eq(u32:-2, x+y)
    }
    """)
    self._parse_and_test(program)

  def test_tree_binding(self):
    program = textwrap.dedent("""\
    test tree_binding {
      let (w, (x,), (y, (z,))): (u32, (u32,), (u32, (u32,))) =
        (u32:1, (u32:2,), (u32:3, (u32:4,))) in
      assert_eq(u32:10, w+x+y+z)
    }
    """)
    self._parse_and_test(program)

  def test_add_wraparound(self):
    program = textwrap.dedent("""\
    test simple_add {
      let x: u32 = (u32:1<<u32:31)+((u32:1<<u32:31)-u32:1) in
      let y: u32 = u32:1 in
      assert_eq(u32:0, x+y)
    }
    """)
    self._parse_and_test(program)

  def test_add_with_carry_u1(self):
    program = textwrap.dedent("""\
    test simple_add {
      let x: u1 = u1:1 in
      assert_eq((u1:1, u1:0), add_with_carry(x, x))
    }
    """)
    self._parse_and_test(program)

  def test_array_index(self):
    program = textwrap.dedent("""\
    test indexing {
      let x: u32[3] = u32[3]:[1, 2, 3] in
      let y: u32 = u32:1 in
      assert_eq(u32:2, x[y])
    }
    """)
    self._parse_and_test(program)

  def test_tuple_index(self):
    program = textwrap.dedent("""\
    test indexing {
      let t = (u1:0, u8:1, u32:2) in
      assert_eq(u32:2, t[u32:2])
    }
    """)
    self._parse_and_test(program)

  def test_concrete_type_from_array_of_u32s(self):
    elements = (
        interpreter.Value.make_u32(0xf00),
        interpreter.Value.make_u32(0xba5),
    )
    v = interpreter.Value.make_array(elements)
    concrete_type = interpreter.concrete_type_from_value(v)
    self.assertIsInstance(concrete_type, ArrayType)
    self.assertEqual((2, 32), concrete_type.get_all_dims())

  def test_for_over_array(self):
    program = textwrap.dedent("""\
    test for_over_array {
      let a: u32[3] = u32[3]:[1, 2, 3] in
      let result: u32 = for (value, accum): (u32, u32) in a {
        accum + value
      }(u32:0) in
      assert_eq(u32:6, result)
    }
    """)
    self._parse_and_test(program)

  def test_for_over_range(self):
    program = textwrap.dedent("""\
    test for_over_array {
      let result: u32 = for (value, accum): (u32, u32) in range(u32:1, u32:4) {
        accum + value
      }(u32:0) in
      assert_eq(u32:6, result)
    }
    """)
    self._parse_and_test(program)

  def test_for_over_range_u8(self):
    program = textwrap.dedent("""\
    test for_over_array {
      let result: u8 = for (value, accum): (u8, u8) in range(u8:1, u8:4) {
        accum + value
      }(u8:0) in
      assert_eq(u8:6, result)
    }
    """)
    self._parse_and_test(program)

  def test_conflicting_parametric_bindings(self):
    program = textwrap.dedent("""\
    fn [N: u32] parametric(x: bits[N], y: bits[N]) -> bits[1] {
      x == bits[N]:1 and y == bits[N]:2
    }
    test parametric_conflict {
      let a: bits[2] = bits[2]:0b10 in
      let b: bits[3] = bits[3]:0b110 in
      parametric(a, b)
    }
    """)
    with self.assertRaises(XlsTypeError) as cm:
      self._parse_and_test(program)
    self.assertIn(
        'Parametric value N was bound to different values at different places in invocation; saw: 2; then: 3',
        str(cm.exception))

  def test_inequality(self):
    program = textwrap.dedent("""
    test not_equals {
      let _: () = assert_eq(false, u32:0 != u32:0) in
      let _: () = assert_eq(true, u32:1 != u32:0) in
      let _: () = assert_eq(true, u32:1 != u32:-1) in
      let _: () = assert_eq(true, u32:1 != u32:2) in
      ()
    }
    """)
    self._parse_and_test(program)

  def test_array_equality(self):
    program = textwrap.dedent("""
    test array_equality {
      let a: u8[4] = u8[4]:[1,2,3,4] in
      assert_eq(a, a)
    }
    """)
    self._parse_and_test(program)

  def test_derived_parametric(self):
    program = textwrap.dedent("""\
    fn [X: u32, Y: u32 = X+X, Z: u32 = Y+u32:1] parametric(
          x: bits[X]) -> (u32, u32, u32) {
      (X, Y, Z)
    }
    test parametric {
      assert_eq((u32:2, u32:4, u32:5), parametric(bits[2]:0))
    }
    """)
    self._parse_and_test(program)

  def test_bool_not(self):
    program = """
    fn bool_not(x: bool) -> bool {
      not x
    }
    test bool_not {
      let _: () = assert_eq(true, bool_not(false)) in
      let _: () = assert_eq(false, bool_not(true)) in
      ()
    }
    """
    self._parse_and_test(program)

  def test_fail_incomplete_match(self):
    program = textwrap.dedent("""\
    test incomplete_match_failure {
      let x: u32 = u32:42 in
      let _: u32 = match x {
        u32:64 => u32:77
      } in
      ()
    }
    """)
    with self.assertRaises(interpreter.FailureError):
      self._parse_and_test(program)

  def test_while(self):
    program = """
    test while_lt {
      let i: u32 = while carry < u32:9 {
        carry + u32:2
      }(u32:0) in
      assert_eq(u32:10, i)
    }
    """
    self._parse_and_test(program)

  def test_boolean_literal_needs_no_type_annotation(self):
    program = """
    fn returns_bool() -> bool {
      false
    }
    test returns_bool {
      assert_eq(false, returns_bool())
    }
    """
    self._parse_and_test(program)

  def test_trace(self):
    program = """
    fn fut() -> u32 {
      while carry < u32:2 {
        trace(carry + u32:1)
      }(u32:0)
    }
    test while_lt {
      assert_eq(u32:2, fut())
    }
    """
    mock_stderr = io.StringIO()
    with mock.patch('sys.stderr', mock_stderr):
      self._parse_and_test(program)

    self.assertIn('4:14-4:29: bits[32]:0x1', mock_stderr.getvalue())
    self.assertIn('4:14-4:29: bits[32]:0x2', mock_stderr.getvalue())

  def test_slice_builtin(self):
    program = """
    fn non_test_slice(x: u8[4], start: u32) -> u8[3] {
      slice(x, start, u8[3]:[0, 0, 0])
    }
    test slice {
      let a: u8[4] = u8[4]:[1, 2, 3, 4] in
      let _: () = assert_eq(u8[2]:[1, 2], slice(a, u32:0, u8[2]:[0, 0])) in
      let _: () = assert_eq(u8[2]:[3, 4], slice(a, u32:2, u8[2]:[0, 0])) in
      let _: () = assert_eq(u8[3]:[2, 3, 4], slice(a, u32:1, u8[3]:[0, 0, 0])) in
      let _: () = assert_eq(u8[3]:[2, 3, 4], non_test_slice(a, u32:1)) in
      ()
    }
    """
    self._parse_and_test(program)

  def test_array_literal_ellipsis(self):
    program = """
    fn non_test_slice(x: u8[4], start: u32) -> u8[3] {
      slice(x, start, u8[3]:[0, ...])
    }
    test slice {
      let a: u8[4] = u8[4]:[4, ...] in
      let _: () = assert_eq(u8[3]:[4, 4, 4], non_test_slice(a, u32:1)) in
      let _: () = assert_eq(u8[3]:[4, 4, 4], u8[3]:[4, ...]) in
      let _: () = assert_eq(u8:4, (u8[3]:[4, ...])[u32:2]) in
      ()
    }
    """
    self._parse_and_test(program)

  def test_destructure(self):
    program = """
    test destructure {
      let t = (u32:2, u8:3) in
      let (a, b) = t in
      let _ = assert_eq(u32:2, a) in
      assert_eq(u8:3, b)
    }
    """
    self._parse_and_test(program)

  def test_destructure_black_hole_identifier(self):
    program = """
    test destructure {
      let t = (u32:2, u8:3, true) in
      let (_, _, v) = t in
      assert_eq(v, true)
    }
    """
    self._parse_and_test(program)

  def test_struct_with_const_sized_array(self):
    program = """
    const THING_COUNT = u32:2;
    type Foo = (
      u32[THING_COUNT]
    );
    fn get_thing(x: Foo, i: u32) -> u32 {
      let things: u32[THING_COUNT] = x[u32:0] in
      things[i]
    }
    test foo {
      let foo: Foo = (u32[THING_COUNT]:[42, 64],) in
      let _ = assert_eq(u32:42, get_thing(foo, u32:0)) in
      let _ = assert_eq(u32:64, get_thing(foo, u32:1)) in
      ()
    }
    """
    self._parse_and_test(program)

  def test_cast_array_to_wrong_bit_count(self):
    program = textwrap.dedent("""\
    test cast_array_to_wrong_bit_count {
      let x = u2[2]:[2, 3] in
      assert_eq(u3:0, x as u3)
    }
    """)
    with self.assertRaises(XlsTypeError) as cm:
      self._parse_and_test(program)
    self.assertIn(
        'uN[2][2] vs uN[3]: Cannot cast from expression type uN[2][2] to uN[3].',
        str(cm.exception))

  def test_cast_enum_oob_causes_fail(self):
    program = textwrap.dedent("""\
    enum Foo : u32 {
      FOO = 1
    }
    fn f(x: u32) -> Foo {
      x as Foo
    }
    test cast_enum_oob_causes_fail {
      let foo: Foo = f(u32:2) in
      assert_eq(true, foo != Foo::FOO)
    }
    """)
    with self.assertRaises(interpreter.FailureError):
      self._parse_and_test(program)

  def test_const_array_of_enum_refs(self):
    program = """
    enum MyEnum: u2 {
      FOO = 2,
      BAR = 3,
    }
    const A = MyEnum[2]:[MyEnum::FOO, MyEnum::BAR];
    test t {
      let _ = assert_eq(MyEnum::FOO, A[u32:0]) in
      let _ = assert_eq(MyEnum::BAR, A[u32:1]) in
      ()
    }
    """
    self._parse_and_test(program)

  def test_typedef_array(self):
    program = """
    type MyType = u2;
    type StructLike = (
      MyType[2],
    );
    fn f(s: StructLike) -> StructLike {
      let updated: StructLike = (
        [s[u32:0][u32:0]+MyType:1, s[u32:0][u32:1]+MyType:1],
      ) in
      updated
    }
    test t {
      let s: StructLike = (MyType[2]:[MyType:0, MyType:1],) in
      let s_2 = f(s) in
      assert_eq(s_2, (u2[2]:[u2:1, u2:2],))
    }
    """
    self._parse_and_test(program)

  def test_trace_all(self):
    # To test tracing output, we'll run a program, capture stderr, and make sure
    # that the desired traces (and _only_ the desired traces) are present.
    program = """
    test t {
      let x0 = u8:32 in
      let _ = trace(x0) in
      let x1 = clz(x0) in
      let x2 = x0 + x1 in
      assert_eq(x2, u8:34)
    }
    """

    # Capture stderr.
    old_stderr = sys.stderr
    sys.stderr = captured_stderr = io.StringIO()
    self._parse_and_test(program, trace_all=True)
    sys.stderr = old_stderr

    # Verify x0, x1, and x2 are traced.
    self.assertIn('trace of x0', captured_stderr.getvalue())
    self.assertIn('trace of x1', captured_stderr.getvalue())
    self.assertIn('trace of x2', captured_stderr.getvalue())

    # Now verify no "trace" or "let" lines or function references.
    self.assertNotIn('Tag.FUNCTION', captured_stderr.getvalue())
    self.assertNotIn('trace of trace', captured_stderr.getvalue())


if __name__ == '__main__':
  unittest.main()
