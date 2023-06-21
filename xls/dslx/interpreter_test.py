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

"""Tests for the DSLX interpreter.

Note that generally we want DSLX test to be in `xls/dslx/tests`, but this can
test properties of the interpreter binary itself that cannot be verified in a
normal DSLX test environment, e.g. that it flags the first failing test in a
file, etc.
"""

import subprocess as subp
import textwrap
from typing import Sequence

from xls.common import runfiles
from xls.common import test_base

_INTERP_PATH = runfiles.get_path('xls/dslx/interpreter_main')


class InterpreterTest(test_base.TestCase):

  def _parse_and_test(
      self,
      program: str,
      compare: str = 'jit',
      want_error: bool = False,
      alsologtostderr: bool = False,
      extra_flags: Sequence[str] = (),
  ) -> str:
    temp_file = self.create_tempfile(content=program)
    cmd = [_INTERP_PATH, temp_file.full_path]
    cmd.append('--compare=%s' % compare)
    if alsologtostderr:
      cmd.append('--alsologtostderr')
    cmd.extend(extra_flags)
    p = subp.run(cmd, check=False, stderr=subp.PIPE, encoding='utf-8')
    if want_error:
      self.assertNotEqual(p.returncode, 0)
    else:
      self.assertEqual(p.returncode, 0, msg=p.stderr)
    return p.stderr

  def test_two_plus_two_fail_module_test(self):
    """Tests that we flag assertion failure locations with no highlighting."""
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
    self.assertIn(textwrap.dedent("""\
    0004:   let y: u32 = x + x;
    0005:   let expected: u32 = u32:5;
    0006:   assert_eq(y, expected)
    ~~~~~~~~~~~~~~~~~^-----------^ FailureError: The program being interpreted failed!
      lhs: u32:4
      rhs: u32:5
      were not equal
    0007: }
    """), stderr)

  def test_conflicting_parametric_bindings(self):
    """Tests a conflict in a deduced parametric value.

    Note that this is really a typecheck test.
    """
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

  def test_fail_incomplete_match(self):
    """Tests that interpreter runtime-fails on incomplete match pattern set."""
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

  def test_failing_test_gives_error_retcode(self):
    """Tests that a failing DSLX test results in an error return code."""
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
    """Tests that a passing test then failing test gives an error retcode."""
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
    """Tests that the interpreter gives a 0 retcode for a passing test file."""
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
    """Tests that we see trace output in stderr."""
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
    """Tests that we can trace s32 values."""
    program = """
    #[test]
    fn trace_test() {
      let x = u32:4;
      let _ = trace!(x);
      let x = s32:-1;
      let _ = trace!(x);
    }
    """
    stderr = self._parse_and_test(
        program, want_error=False, alsologtostderr=True
    )
    self.assertIn('5:21-5:24: 4', stderr)
    self.assertIn('7:21-7:24: -1', stderr)

  def test_trace_fmt_hello(self):
    """Tests that basic trace formatting works."""
    program = """
    fn main() {
      let x = u8:0xF0;
      trace_fmt!("Hello world!");
      trace_fmt!("x is {}, {:#x} in hex and {:#b} in binary", x, x, x);
    }

    #[test]
    fn hello_test() {
      main()
    }
    """
    stderr = self._parse_and_test(
        program, want_error=False, alsologtostderr=True
    )
    self.assertIn('Hello world!', stderr)
    self.assertIn('x is 240, 0xf0 in hex and 0b1111_0000 in binary', stderr)

  def test_trace_fmt_struct_field_fmt_pref(self):
    """Tests trace-formatting of a struct with a format preference."""
    program = """
    struct MyStruct {
      x: u32,
    }
    fn main() {
      let s = MyStruct{x: u32:42};
      let _ = trace_fmt!("s as hex: {:#x}", s);
      let _ = trace_fmt!("s as bin: {:#b}", s);
      ()
    }

    #[test]
    fn hello_test() {
      main()
    }
    """
    # Note: we have to pass `--log_prefix=false` here to match multi-line
    # logging output easily.
    program_file = self.create_tempfile(content=program)
    cmd = [
        _INTERP_PATH,
        '--alsologtostderr',
        '--log_prefix=false',
        program_file.full_path,
    ]
    result = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=False)
    print(result.stderr)
    self.assertIn('s as hex: MyStruct {\n  x: 0x2a\n}', result.stderr)
    self.assertIn('s as bin: MyStruct {\n  x: 0b10_1010\n}', result.stderr)

  def test_trace_fmt_array_of_struct(self):
    """Tests that we can apply trace formatting to an array of structs."""
    program = """
    struct MyStruct {
      x: u32,
    }
    fn main() {
      let s = MyStruct{x: u32:42};
      let a = MyStruct[1]:[s];
      let _ = trace_fmt!("a as hex: {:#x}", a);
      let _ = trace_fmt!("a as bin: {:#b}", a);
      ()
    }

    #[test]
    fn hello_test() {
      main()
    }
    """
    # Note: we have to pass `--log_prefix=false` here to match multi-line
    # logging output easily.
    program_file = self.create_tempfile(content=program)
    cmd = [
        _INTERP_PATH,
        '--alsologtostderr',
        '--log_prefix=false',
        program_file.full_path,
    ]
    result = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=False)
    print(result.stderr)
    self.assertIn('a as hex: [MyStruct {\n  x: 0x2a\n}]', result.stderr)
    self.assertIn('a as bin: [MyStruct {\n  x: 0b10_1010\n}]', result.stderr)

  def test_trace_fmt_array_of_enum(self):
    """Tests we can trace-format an array of enum values."""
    program = """
    enum MyEnum : u2 {
      ONE = 1,
      TWO = 2,
    }
    fn main() {
      let a = MyEnum[2]:[MyEnum::ONE, MyEnum::TWO];
      let _ = trace_fmt!("a: {:#x}", a);
      ()
    }

    #[test]
    fn hello_test() {
      main()
    }
    """
    stderr = self._parse_and_test(
        program, want_error=False, alsologtostderr=True
    )
    self.assertIn('a: [MyEnum::ONE, MyEnum::TWO]', stderr)

  def test_trace_fmt_tuple_of_enum(self):
    """Tests that we can trace format a tuple that includes enum values."""
    program = """
    enum MyEnum : u2 {
      ONE = 1,
      TWO = 2,
    }
    fn main() {
      let t = (MyEnum::ONE, MyEnum::TWO, u32:42);
      let _ = trace_fmt!("t: {:#x}", t);
      ()
    }

    #[test]
    fn hello_test() {
      main()
    }
    """
    stderr = self._parse_and_test(
        program, want_error=False, alsologtostderr=True
    )
    self.assertIn('t: (MyEnum::ONE, MyEnum::TWO, 0x2a)', stderr)

  def test_cast_array_to_wrong_bit_count(self):
    """Tests that casing an array to the wrong bit count causes a failure."""
    program = textwrap.dedent("""\
    #[test]
    fn cast_array_to_wrong_bit_count_test() {
      let x = u2[2]:[2, 3];
      assert_eq(u3:0, x as u3)
    }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(textwrap.dedent("""\
    0002: fn cast_array_to_wrong_bit_count_test() {
    0003:   let x = u2[2]:[2, 3];
    0004:   assert_eq(u3:0, x as u3)
    ~~~~~~~~~~~~~~~~~~~~~~~~^-----^ XlsTypeError: uN[2][2] vs uN[3]: Cannot cast from expression type uN[2][2] to uN[3].
    0005: }
    """), stderr)

  def test_cast_enum_oob_causes_fail(self):
    """Tests casting an out-of-bound value to enum causes a runtime failure."""
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

  def test_assert_eq_failure_arrays(self):
    """Tests array equality failure prints an appropriate error message."""
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
    """Tests that we identify the first failing test in a file of tests."""
    program = textwrap.dedent("""\
    #[test] fn first_test() { assert_eq(false, true) }
    #[test] fn second_test() { assert_eq(true, true) }
    #[test] fn third_test() { assert_eq(true, true) }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(':1:36-1:49', stderr)
    self.assertIn('were not equal', stderr)

  def test_second_failing_test(self):
    """Tests that we identify the second failing test in a file of tests."""
    program = textwrap.dedent("""\
    #[test] fn first_test() { assert_eq(true, true) }
    #[test] fn second_test() { assert_eq(false, true) }
    #[test] fn third_test() { assert_eq(true, true) }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(':2:37-2:50', stderr)
    self.assertIn('were not equal', stderr)

  def test_third_failing_test(self):
    """Tests that we identify the third failing test in a file of tests."""
    program = textwrap.dedent("""\
    #[test] fn first_test() { assert_eq(true, true) }
    #[test] fn second_test() { assert_eq(true, true) }
    #[test] fn third_test() { assert_eq(false, true) }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(':3:36-3:49', stderr)
    self.assertIn('were not equal', stderr)

  def test_large_generated_function_body(self):
    """Tests we handle a large function body without stack overflow errors."""
    # This fails at 6KiStatements in opt due to deep recursion.
    # Picked this number so it passes with ASAN.
    nesting = 1024 * 1
    rest = '\n'.join('  let x%d = x%d;' % (i, i - 1) for i in range(1, nesting))
    program = textwrap.dedent("""\
    fn f() -> u32 {
      let x0 = u32:42;
      %s
      // Make a fail label since those should be unique at function scope.
      let _ = if x0 != u32:42 {
        fail!("impossible", ())
      } else {
        ()
      };
      x%d
    }
    #[test]
    fn test_f() {
      assert_eq(u32:42, f());
    }
    """) % (rest, nesting - 1)
    self._parse_and_test(program)

  def test_default_format_preference(self):
    """Tests the error message from an assert_eq is formatted correctly."""
    program = textwrap.dedent("""\
    #[test]
    fn failing_test() {
      assert_eq(u32:20, u32:30)
    }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn('lhs: u32:20', stderr)
    self.assertIn('rhs: u32:30', stderr)

  def test_hex_format_preference(self):
    """Tests the error message from an assert_eq is formatted correctly."""
    program = textwrap.dedent("""\
    #[test]
    fn failing_test() {
      assert_eq(u32:20, u32:30)
    }
    """)
    stderr = self._parse_and_test(
        program, want_error=True, extra_flags=('--format_preference=hex',)
    )
    self.assertIn('lhs: u32:0x14', stderr)
    self.assertIn('rhs: u32:0x1e', stderr)


if __name__ == '__main__':
  test_base.main()
