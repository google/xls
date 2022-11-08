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
"""Tests for xls.dslx.interpreter."""

import subprocess as subp
import textwrap
from typing import Text

from xls.common import runfiles
from xls.common import test_base

_INTERP_PATH = runfiles.get_path('xls/dslx/interpreter_main')


class InterpreterTest(test_base.TestCase):

  def _parse_and_test(self,
                      program: Text,
                      compare: str = 'jit',
                      want_error: bool = False) -> str:
    temp_file = self.create_tempfile(content=program)
    cmd = [_INTERP_PATH, temp_file.full_path]
    cmd.append('--compare=%s' % compare)
    p = subp.run(cmd, check=False, stderr=subp.PIPE, encoding='utf-8')
    if want_error:
      self.assertNotEqual(p.returncode, 0)
    else:
      self.assertEqual(p.returncode, 0, msg=p.stderr)
    return p.stderr

  def test_two_plus_two_module_test(self):
    program = textwrap.dedent("""\
    #[test]
    fn two_plus_two_is_four_test() {
      let x: u32 = u32:2;
      let y: u32 = x + x;
      let expected: u32 = u32:4;
      assert_eq(y, expected)
    }
    """)
    self._parse_and_test(program)

  def test_two_plus_two_fail_module_test(self):
    program = textwrap.dedent("""\
    #[test]
    fn two_plus_two_is_four_test() {
      let x: u32 = u32:2;
      let y: u32 = x + x;
      let expected: u32 = u32:5;
      assert_eq(y, expected)
    }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(':6:12-6:25', stderr)

  def test_pad_bits_via_concat(self):
    program = textwrap.dedent("""\
    #[test]
    fn pad_to_four_bits_test() {
      let x: bits[2] = bits[2]:0b10;
      let y: bits[4] = bits[2]:0 ++ x;
      let expected: bits[4] = bits[4]:0b0010;
      assert_eq(y, expected)
    }
    """)
    self._parse_and_test(program)

  def test_invocation(self):
    program = textwrap.dedent("""\
    fn id(x: u32) -> u32 { x }
    #[test]
    fn identity_invocation_test() {
      assert_eq(u32:42, id(u32:42))
    }
    """)
    self._parse_and_test(program)

  def test_cast(self):
    program = textwrap.dedent("""\
    #[test]
    fn cast_to_u32_test() {
      let x: bits[3] = bits[3]:0b010;
      assert_eq(u32:4, (x + x) as u32)
    }
    """)
    self._parse_and_test(program)

  def test_parametric_invocation(self):
    program = textwrap.dedent("""\
    fn id<N: u32>(x: bits[N]) -> bits[N] { x }
    #[test]
    fn different_parametric_invocations_test() {
      assert_eq(bits[5]:0b01111, id(bits[2]:0b01) ++ id(bits[3]:0b111))
    }
    """)
    self._parse_and_test(program)

  def test_parametric_binding(self):
    program = textwrap.dedent("""\
    fn add_num_bits<N: u32>(x: bits[N]) -> bits[N] { x+(N as bits[N]) }
    #[test]
    fn different_parametric_invocations_test() {
      assert_eq(bits[2]:3, add_num_bits(bits[2]:1))
    }
    """)
    self._parse_and_test(program)

  def test_simple_subtract(self):
    program = textwrap.dedent("""\
    #[test]
    fn simple_subtract_test() {
      let x: u32 = u32:5;
      let y: u32 = u32:4;
      assert_eq(u32:1, x-y)
    }
    """)
    self._parse_and_test(program)

  def test_tree_binding(self):
    program = textwrap.dedent("""\
    #[test]
    fn tree_binding_test() {
      let (w, (x,), (y, (z,))): (u32, (u32,), (u32, (u32,))) =
        (u32:1, (u32:2,), (u32:3, (u32:4,)));
      assert_eq(u32:10, w+x+y+z)
    }
    """)
    self._parse_and_test(program)

  def test_add_wraparound(self):
    program = textwrap.dedent("""\
    #[test]
    fn simple_add_test() {
      let x: u32 = (u32:1<<u32:31)+((u32:1<<u32:31)-u32:1);
      let y: u32 = u32:1;
      assert_eq(u32:0, x+y)
    }
    """)
    self._parse_and_test(program)

  def test_add_with_carry_u1(self):
    program = textwrap.dedent("""\
    #[test]
    fn simple_add_test() {
      let x: u1 = u1:1;
      assert_eq((u1:1, u1:0), add_with_carry(x, x))
    }
    """)
    self._parse_and_test(program)

  def test_array_index(self):
    program = textwrap.dedent("""\
    #[test]
    fn indexing_test() {
      let x: u32[3] = u32[3]:[1, 2, 3];
      let y: u32 = u32:1;
      assert_eq(u32:2, x[y])
    }
    """)
    self._parse_and_test(program)

  def test_tuple_index(self):
    program = textwrap.dedent("""\
    #[test]
    fn indexing_test() {
      let t = (u1:0, u8:1, u32:2);
      assert_eq(u32:2, t.2)
    }
    """)
    self._parse_and_test(program)

  def test_for_over_array(self):
    program = textwrap.dedent("""\
    #[test]
    fn for_over_array_test() {
      let a: u32[3] = u32[3]:[1, 2, 3];
      let result: u32 = for (value, accum): (u32, u32) in a {
        accum + value
      }(u32:0);
      assert_eq(u32:6, result)
    }
    """)
    self._parse_and_test(program)

  def test_for_over_range(self):
    program = textwrap.dedent("""\
    #[test]
    fn for_over_array_test() {
      let result: u32 = for (value, accum): (u32, u32) in range(u32:1, u32:4) {
        accum + value
      }(u32:0);
      assert_eq(u32:6, result)
    }
    """)
    self._parse_and_test(program)

  def test_for_over_range_u8(self):
    program = textwrap.dedent("""\
    #[test]
    fn for_over_array_test() {
      let result: u8 = for (value, accum): (u8, u8) in range(u8:1, u8:4) {
        accum + value
      }(u8:0);
      assert_eq(u8:6, result)
    }
    """)
    self._parse_and_test(program)

  def test_conflicting_parametric_bindings(self):
    program = textwrap.dedent("""\
    fn parametric<N: u32>(x: bits[N], y: bits[N]) -> bits[1] {
      x == bits[N]:1 && y == bits[N]:2
    }
    #[test]
    fn parametric_conflict_test() {
      let a: bits[2] = bits[2]:0b10;
      let b: bits[3] = bits[3]:0b110;
      parametric(a, b)
    }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(
        'Parametric value N was bound to different values at different places in invocation; saw: 2; then: 3',
        stderr)

  def test_inequality(self):
    program = textwrap.dedent("""
    #[test]
    fn not_equals_test() {
      let _: () = assert_eq(false, u32:0 != u32:0);
      let _: () = assert_eq(true, u32:1 != u32:0);
      let _: () = assert_eq(true, s32:1 != s32:-1);
      let _: () = assert_eq(true, u32:1 != u32:2);
      ()
    }
    """)
    self._parse_and_test(program)

  def test_array_equality(self):
    program = textwrap.dedent("""
    #[test]
    fn array_equality_test() {
      let a: u8[4] = u8[4]:[1,2,3,4];
      assert_eq(a, a)
    }
    """)
    self._parse_and_test(program)

  def test_derived_parametric(self):
    program = textwrap.dedent("""\
    fn parametric<X: u32, Y: u32 = X+X, Z: u32 = Y+u32:1>(
          x: bits[X]) -> (u32, u32, u32) {
      (X, Y, Z)
    }
    #[test]
    fn parametric_test() {
      assert_eq((u32:2, u32:4, u32:5), parametric(bits[2]:0))
    }
    """)
    self._parse_and_test(program)

  def test_bool_not(self):
    program = """
    fn bool_not(x: bool) -> bool {
      !x
    }
    #[test]
    fn bool_not_test() {
      let _: () = assert_eq(true, bool_not(false));
      let _: () = assert_eq(false, bool_not(true));
      ()
    }
    """
    self._parse_and_test(program)

  def test_fail_incomplete_match(self):
    program = textwrap.dedent("""\
    #[test]
    fn incomplete_match_failure_test() {
      let x: u32 = u32:42;
      let _: u32 = match x {
        u32:64 => u32:77
      };
      ()
    }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(
        'The program being interpreted failed! The value was not matched',
        stderr)
    self.assertIn(':4:16-6:4', stderr)

  def test_boolean_literal_needs_no_type_annotation(self):
    program = """
    fn returns_bool() -> bool {
      false
    }
    #[test]
    fn returns_bool_test() {
      assert_eq(false, returns_bool())
    }
    """
    self._parse_and_test(program)

  def test_failing_test_gives_error_retcode(self):
    program = """
    #[test]
    fn failing_test() {
      assert_eq(false, true)
    }
    """
    program_file = self.create_tempfile(content=program)
    cmd = [_INTERP_PATH, program_file.full_path]
    p = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=False)
    self.assertNotEqual(p.returncode, 0)

  def test_passing_then_failing_test_gives_error_retcode(self):
    program = """
    #[test]
    fn passing_test() {
      assert_eq(true, true)
    }
    #[test]
    fn failing_test() {
      assert_eq(false, true)
    }
    """
    program_file = self.create_tempfile(content=program)
    cmd = [_INTERP_PATH, program_file.full_path]
    p = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=False)
    self.assertNotEqual(p.returncode, 0)

  def test_failing_then_passing_test_gives_error_retcode(self):
    program = """
    #[test]
    fn failing_test() {
      assert_eq(false, true)
    }
    #[test]
    fn passing_test() {
      assert_eq(true, true)
    }
    """
    program_file = self.create_tempfile(content=program)
    cmd = [_INTERP_PATH, program_file.full_path]
    p = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=False)
    self.assertNotEqual(p.returncode, 0)

  def test_passing_test_returncode(self):
    program = """
    #[test]
    fn passing_test() {
      assert_eq(true, true)
    }
    """
    program_file = self.create_tempfile(content=program)
    cmd = [_INTERP_PATH, program_file.full_path]
    p = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=False)
    self.assertEqual(p.returncode, 0)

  def test_trace(self):
    program = """
    fn fut() -> u32 {
      let x = u32:0;
      let _ = trace!(x + u32:1);
      let x = u32:1;
      let _ = trace!(x + u32:1);
      x + u32:1
    }

    #[test]
    fn trace_test() {
      assert_eq(u32:2, fut())
    }
    """
    program_file = self.create_tempfile(content=program)
    # Trace is logged with XLS_LOG(INFO) so log to stderr to capture output.
    cmd = [
        _INTERP_PATH, '--compare=none', '--alsologtostderr',
        program_file.full_path
    ]
    result = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=True)
    self.assertIn('4:21-4:32: 1', result.stderr)
    self.assertIn('6:21-6:32: 2', result.stderr)

  def test_trace_s32(self):
    program = """
    #[test]
    fn trace_test() {
      let x = u32:4;
      let _ = trace!(x);
      let x = s32:-1;
      let _ = trace!(x);
      ()
    }
    """
    program_file = self.create_tempfile(content=program)
    # Trace is logged with XLS_LOG(INFO) so log to stderr to capture output.
    cmd = [
        _INTERP_PATH, '--compare=none', '--alsologtostderr',
        program_file.full_path
    ]
    result = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=True)
    self.assertIn('5:21-5:24: 4', result.stderr)
    self.assertIn('7:21-7:24: -1', result.stderr)

  def test_trace_fmt_hello(self):
    program = """
    fn main() {
      let x = u8:0xF0;
      let _ = trace_fmt!("Hello world!");
      let _ = trace_fmt!("x is {}, {:#x} in hex and {:#b} in binary", x, x, x);
      ()
    }

    #[test]
    fn hello_test() {
      main()
    }
    """
    program_file = self.create_tempfile(content=program)
    cmd = [_INTERP_PATH, '--alsologtostderr', program_file.full_path]
    result = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=True)
    print(result.stderr)
    self.assertIn('Hello world!', result.stderr)
    self.assertIn('x is 240, 0xf0 in hex and 0b1111_0000 in binary',
                  result.stderr)

  def test_bitslice_syntax(self):
    program = """
    #[test]
    fn slice_test() {
      let x = u4:0b1001;
      let _ = assert_eq(x[0:2], u2:0b01);
      let _ = assert_eq(x[2:4], u2:0b10);
      ()
    }
    """
    self._parse_and_test(program)

  def test_slice_builtin(self):
    program = """
    fn non_test_slice(x: u8[4], start: u32) -> u8[3] {
      slice(x, start, u8[3]:[0, 0, 0])
    }
    #[test]
    fn slice_test() {
      let a: u8[4] = u8[4]:[1, 2, 3, 4];
      let _: () = assert_eq(u8[2]:[1, 2], slice(a, u32:0, u8[2]:[0, 0]));
      let _: () = assert_eq(u8[2]:[3, 4], slice(a, u32:2, u8[2]:[0, 0]));
      let _: () = assert_eq(u8[3]:[2, 3, 4], slice(a, u32:1, u8[3]:[0, 0, 0]));
      let _: () = assert_eq(u8[3]:[2, 3, 4], non_test_slice(a, u32:1));
      ()
    }
    """
    # TODO(https://github.com/google/xls/issues/312): requires IR conversion of
    # slice.
    self._parse_and_test(program, compare='none')

  def test_array_literal_ellipsis(self):
    program = """
    fn non_test_slice(x: u8[4], start: u32) -> u8[3] {
      slice(x, start, u8[3]:[0, ...])
    }
    #[test]
    fn slice_test() {
      let a: u8[4] = u8[4]:[4, ...];
      let _: () = assert_eq(u8[3]:[4, 4, 4], non_test_slice(a, u32:1));
      let _: () = assert_eq(u8[3]:[4, 4, 4], u8[3]:[4, ...]);
      let _: () = assert_eq(u8:4, (u8[3]:[4, ...])[u32:2]);
      ()
    }
    """
    # TODO(https://github.com/google/xls/issues/312): requires IR conversion of
    # slice.
    self._parse_and_test(program, compare='none')

  def test_destructure(self):
    program = """
    #[test]
    fn destructure_test() {
      let t = (u32:2, u8:3);
      let (a, b) = t;
      let _ = assert_eq(u32:2, a);
      assert_eq(u8:3, b)
    }
    """
    self._parse_and_test(program)

  def test_destructure_black_hole_identifier(self):
    program = """
    #[test]
    fn destructure_test() {
      let t = (u32:2, u8:3, true);
      let (_, _, v) = t;
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
      let things: u32[THING_COUNT] = x.0;
      things[i]
    }
    #[test]
    fn foo_test() {
      let foo: Foo = (u32[THING_COUNT]:[42, 64],);
      let _ = assert_eq(u32:42, get_thing(foo, u32:0));
      let _ = assert_eq(u32:64, get_thing(foo, u32:1));
      ()
    }
    """
    self._parse_and_test(program)

  def test_cast_array_to_wrong_bit_count(self):
    program = textwrap.dedent("""\
    #[test]
    fn cast_array_to_wrong_bit_count_test() {
      let x = u2[2]:[2, 3];
      assert_eq(u3:0, x as u3)
    }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(
        'uN[2][2] vs uN[3]: Cannot cast from expression type uN[2][2] to uN[3].',
        stderr)

  def test_cast_enum_oob_causes_fail(self):
    program = textwrap.dedent("""\
    enum Foo : u32 {
      FOO = 1
    }
    fn f(x: u32) -> Foo {
      x as Foo
    }
    #[test]
    fn cast_enum_oob_causes_fail_test() {
      let foo: Foo = f(u32:2);
      assert_eq(true, foo != Foo::FOO)
    }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn('Value is not valid for enum', stderr)

  def test_const_array_of_enum_refs(self):
    program = """
    enum MyEnum: u2 {
      FOO = 2,
      BAR = 3,
    }
    const A = MyEnum[2]:[MyEnum::FOO, MyEnum::BAR];
    #[test]
    fn t_test() {
      let _ = assert_eq(MyEnum::FOO, A[u32:0]);
      let _ = assert_eq(MyEnum::BAR, A[u32:1]);
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
        [s.0[u32:0]+MyType:1, s.0[u32:1]+MyType:1],
      );
      updated
    }
    #[test]
    fn t_test() {
      let s: StructLike = (MyType[2]:[MyType:0, MyType:1],);
      let s_2 = f(s);
      assert_eq(s_2, (u2[2]:[u2:1, u2:2],))
    }
    """
    self._parse_and_test(program)

  def test_assert_eq_failure_arrays(self):
    program = """
    #[test]
    fn t_test() {
      assert_eq(s32[2]:[1, 2], s32[2]:[3, 4])
    }
    """
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn('lhs: sN[32][2]:[s32:1, s32:2]\n', stderr)
    self.assertIn('first differing index: 0', stderr)

  def test_first_failing_test(self):
    program = textwrap.dedent("""\
    #[test] fn first_test() { assert_eq(false, true) }
    #[test] fn second_test() { assert_eq(true, true) }
    #[test] fn third_test() { assert_eq(true, true) }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(':1:36-1:49', stderr)
    self.assertIn('were not equal', stderr)

  def test_second_failing_test(self):
    program = textwrap.dedent("""\
    #[test] fn first_test() { assert_eq(true, true) }
    #[test] fn second_test() { assert_eq(false, true) }
    #[test] fn third_test() { assert_eq(true, true) }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(':2:37-2:50', stderr)
    self.assertIn('were not equal', stderr)

  def test_third_failing_test(self):
    program = textwrap.dedent("""\
    #[test] fn first_test() { assert_eq(true, true) }
    #[test] fn second_test() { assert_eq(true, true) }
    #[test] fn third_test() { assert_eq(false, true) }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(':3:36-3:49', stderr)
    self.assertIn('were not equal', stderr)

  def test_wide_shifts(self):
    program = textwrap.dedent("""\
                              #[test]
                              fn simple_add_test() {
      let x: uN[96] = uN[96]:0xaaaa_bbbb_cccc_dddd_eeee_ffff;
      let big: uN[96] = uN[96]:0x9999_9999_9999_9999_9999_9999;
      let four: uN[96] = uN[96]:0x4;
      // Test a value which fits in an int64_t as a signed number,
      // but not in a uint64_t an unsigned number.
      let does_not_fit_in_uint64: uN[65] = uN[65]:0x1_ffff_ffff_ffff_ffff;
      let _ = assert_eq(x >> big, uN[96]:0);
      let _ = assert_eq(x >> four, uN[96]:0x0aaa_abbb_bccc_cddd_deee_efff);
      let _ = assert_eq(x << big, uN[96]:0);
      let _ = assert_eq(x << does_not_fit_in_uint64, uN[96]:0);
      assert_eq(x << four, uN[96]:0xaaab_bbbc_cccd_ddde_eeef_fff0)
    }
    """)
    self._parse_and_test(program)

  def test_wide_ashr(self):
    program = textwrap.dedent("""\
    #[test]
    fn simple_add_test() {
      let x: sN[80] = sN[80]:0x8000_0000_0000_0000_0000 >> uN[80]:0x0aaa_bbbb_cccc_dddd_eeee;
      assert_eq(sN[80]:0xffff_ffff_ffff_ffff_ffff, x)
    }
    """)
    self._parse_and_test(program)


if __name__ == '__main__':
  test_base.main()
