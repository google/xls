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

import dataclasses
import os
import subprocess as subp
import sys
import tempfile
import textwrap
from typing import Dict, List, Optional, Sequence, Tuple, Union
import xml.etree.ElementTree as ET

from xls.common import runfiles
from xls.common import test_base

_INTERP_PATH = runfiles.get_path('xls/dslx/interpreter_main')


@dataclasses.dataclass
class WantError:
  kind: str  # e.g. ArgCountMismatch


class InterpreterTest(test_base.TestCase):

  def _parse_and_test(
      self,
      program: str,
      *,
      compare: str = 'jit',
      want_error: bool = False,
      alsologtostderr: bool = False,
      warnings_as_errors: bool = True,
      disable_warnings: Sequence[str] = (),
      extra_flags: Sequence[str] = (),
      test_filter: Optional[str] = None,
      extra_env: Optional[Dict[str, str]] = None,
  ) -> str:
    temp_file = self.create_tempfile(content=program)
    cmd = [_INTERP_PATH, temp_file.full_path]
    cmd.append('--compare=%s' % compare)
    if alsologtostderr:
      cmd.append('--alsologtostderr')
    if not warnings_as_errors:
      cmd.append('--warnings_as_errors=false')
    if disable_warnings:
      cmd.append('--disable_warnings=%s' % ','.join(disable_warnings))
    cmd.extend(extra_flags)
    env = {} if test_filter is None else dict(TESTBRIDGE_TEST_ONLY=test_filter)
    env.update(extra_env if extra_env else {})
    p = subp.run(cmd, check=False, stderr=subp.PIPE, encoding='utf-8', env=env)
    if want_error:
      self.assertNotEqual(p.returncode, 0)
    else:
      self.assertEqual(p.returncode, 0, msg=p.stderr)
    return p.stderr

  def test_cause_scan_error(self):
    """Tests that we flag scan error locations."""
    program = textwrap.dedent("""\
    fn main(x: u32) -> u32 {
        ~x  // note: in DSLX, as in Rust, it's bang (!) not tilde (~)
    }
    """)
    stderr = self._parse_and_test(program, want_error=True)
    self.assertIn(':2:5-2:5', stderr)
    self.assertIn(
        textwrap.dedent("""\
    0002:     ~x  // note: in DSLX, as in Rust, it's bang (!) not tilde (~)
    ~~~~~~~~~~^ ScanError: Unrecognized character: '~' (0x7e)
    0003: }
    """),
        stderr,
    )

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
    self.assertIn(
        textwrap.dedent("""\
    0004:   let y: u32 = x + x;
    0005:   let expected: u32 = u32:5;
    0006:   assert_eq(y, expected)
    ~~~~~~~~~~~~~~~~~^-----------^ FailureError: The program being interpreted failed!
      lhs: u32:4
      rhs: u32:5
      were not equal
    0007: }
    """),
        stderr,
    )

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
        'Parametric value N was bound to different values at different places'
        ' in invocation; saw: 2; then: 3',
        stderr,
    )

  def test_fail_incomplete_match(self):
    """Tests that interpreter runtime-fails on incomplete match pattern set."""
    program = textwrap.dedent("""\
    #[test]
    fn incomplete_match_failure_test() {
      let x: u32 = u32:42;
      match x {
        u32:64 => u32:77
      };
    }
    """)
    # warnings-as-errors off because we have a useless match expression just to
    # trigger the type error.
    stderr = self._parse_and_test(
        program, warnings_as_errors=False, want_error=True
    )
    self.assertIn(
        'Match pattern is not exhaustive',
        stderr,
    )

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
      trace!(x + u32:1);
      let x = u32:1;
      trace!(x + u32:1);
      x + u32:1
    }

    #[test]
    fn trace_test() {
      assert_eq(u32:2, fut())
    }
    """
    program_file = self.create_tempfile(content=program)
    # Trace is logged with LOG(INFO) so log to stderr to capture output.
    cmd = [
        _INTERP_PATH,
        '--compare=none',
        '--alsologtostderr',
        program_file.full_path,
    ]
    result = subp.run(cmd, stderr=subp.PIPE, encoding='utf-8', check=True)
    self.assertIn(': 1', result.stderr)
    self.assertIn(': 2', result.stderr)

  def test_trace_s32(self):
    """Tests that we can trace s32 values."""
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
    stderr = self._parse_and_test(
        program,
        want_error=False,
        alsologtostderr=True,
        warnings_as_errors=False,
    )
    self.assertIn(':5] trace of x: 4', stderr)
    self.assertIn(':7] trace of x: -1', stderr)

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
      trace_fmt!("s as hex: {:#x}", s);
      trace_fmt!("s as bin: {:#b}", s);
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
    self.assertRegex(result.stderr, r's as hex:\s+MyStruct\s+{\s+x: 0x2a')
    self.assertRegex(result.stderr, r's as bin:\s+MyStruct\s+{\s+x: 0b10_1010')

  def test_trace_fmt_array_of_struct(self):
    """Tests that we can apply trace formatting to an array of structs."""
    program = """
    struct MyStruct {
      x: u32,
    }
    fn main() {
      let s = MyStruct{x: u32:42};
      let a = MyStruct[1]:[s];
      trace_fmt!("a as hex: {:#x}", a);
      trace_fmt!("a as bin: {:#b}", a);
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
    self.assertRegex(result.stderr, r'a as hex: \[\s+MyStruct\s+{\s+x: 0x2a')
    self.assertRegex(
        result.stderr, r'a as bin: \[\s+MyStruct\s+{\s+x: 0b10_1010'
    )

  def test_trace_fmt_array_of_enum(self):
    """Tests we can trace-format an array of enum values."""
    program = """
    enum MyEnum : u2 {
      ONE = 1,
      TWO = 2,
    }
    fn main() {
      let a = MyEnum[2]:[MyEnum::ONE, MyEnum::TWO];
      trace_fmt!("a: {:#x}", a);
    }

    #[test]
    fn hello_test() {
      main()
    }
    """
    stderr = self._parse_and_test(
        program, want_error=False, alsologtostderr=True
    )
    self.assertRegex(stderr, r'MyEnum::ONE\s+// u2:1')
    self.assertRegex(stderr, r'MyEnum::TWO\s+// u2:2')

  def test_trace_fmt_array_of_ints(self):
    """Tests we can trace-format an array of u8 values."""
    program = """
    fn main() {
      let a = u8[2]:[u8:1, u8:2];
      trace_fmt!("a: {:#x}", a);
    }

    #[test]
    fn hello_test() {
      main()
    }
    """
    stderr = self._parse_and_test(
        program, want_error=False, alsologtostderr=True
    )
    self.assertIn(r'a: [', stderr)
    self.assertIn(r'0x1', stderr)
    self.assertIn(r'0x2', stderr)

  def test_trace_fmt_tuple_of_enum(self):
    """Tests that we can trace format a tuple that includes enum values."""
    program = """
    enum MyEnum : u2 {
      ONE = 1,
      TWO = 2,
    }
    fn main() {
      let t = (MyEnum::ONE, MyEnum::TWO, u32:42);
      trace_fmt!("t: {:#x}", t);
    }

    #[test]
    fn hello_test() {
      main()
    }
    """
    stderr = self._parse_and_test(
        program, want_error=False, alsologtostderr=True
    )
    self.assertIn('t: (', stderr)
    self.assertIn('MyEnum::ONE', stderr)
    self.assertIn('MyEnum::TWO', stderr)
    self.assertIn('0x2a', stderr)

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
    self.assertIn(
        textwrap.dedent("""\
    0002: fn cast_array_to_wrong_bit_count_test() {
    0003:   let x = u2[2]:[2, 3];
    0004:   assert_eq(u3:0, x as u3)
    ~~~~~~~~~~~~~~~~~~~~~~~~^-----^ XlsTypeError: Cannot cast from expression type uN[2][2] to uN[3].
    Type mismatch:
       uN[2][2]
    vs uN[3]
    0005: }
    """),
        stderr,
    )

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
    self.assertRegex(stderr, r'<\s+s32:1')
    self.assertRegex(stderr, r'>\s+s32:3')
    self.assertRegex(stderr, r'<\s+s32:2')
    self.assertRegex(stderr, r'>\s+s32:4')
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
      assert!(x0 == u32:42, "impossible");
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

  def test_smulp_value_or_type_error(self):
    cases: List[Tuple[Union[WantError, str], List[str]]] = [
        (WantError('ArgCountMismatch'), []),  # nullary
        (WantError('ArgCountMismatch'), ['s32:42']),  # unary
        ('u32:2688', ['s32:42', 's32:64']),  # signed
        (WantError('TypeInferenceError'), ['u32:42', 'u32:64']),  # unsigned
        (
            WantError('ArgCountMismatch'),
            ['s32:42', 's32:64', 's32:96'],
        ),  # ternary
    ]
    for want, args in cases:
      print('args:', args, file=sys.stderr)
      print('want:', want, file=sys.stderr)

      args_str = ', '.join(args)
      want_str = 's32:0' if isinstance(want, WantError) else want
      assert isinstance(want_str, str), want_str

      program = textwrap.dedent("""\
      #[test]
      fn smulp_test() {
        let (p0, p1) = smulp(%s);
        let sum = (p0 + p1);
        assert_eq(%s, sum)
      }
      """ % (args_str, want_str))
      if isinstance(want, WantError):
        stderr = self._parse_and_test(program, want_error=True)
        self.assertIn(want.kind, stderr)
      else:
        stderr = self._parse_and_test(program, want_error=False)

  def test_disable_warning(self):
    """Tests that we can disable one kind of warning."""
    program = """
    fn f() { let _ = (); }
    """
    self._parse_and_test(
        program,
        want_error=False,
        alsologtostderr=True,
        warnings_as_errors=True,
        disable_warnings=['useless_let_binding'],
    )

  def test_disable_warning_does_not_affect_another_kind(self):
    """Disable one kind of warning without affecting another."""
    program = """
    fn f(a: u32, b: u32, c: u32) -> (u32, u32, u32) { (a, b, c,) }
    """
    stderr = self._parse_and_test(
        program,
        want_error=True,
        alsologtostderr=True,
        warnings_as_errors=True,
        disable_warnings=['useless_let_binding'],
    )
    self.assertIn(
        'Tuple expression (with >1 element) is on a single line, but has a'
        ' trailing comma.',
        stderr,
    )

  def test_env_based_test_filter(self):
    """Tests environment-variable based filtering."""
    with tempfile.TemporaryDirectory() as tmpdir:
      output_xml = os.path.join(tmpdir, 'test.xml')

      program = """
      #[test] fn test_one() {}
      #[test] fn test_two() {}
      """
      stderr = self._parse_and_test(
          program,
          alsologtostderr=True,
          test_filter='.*_two',
          extra_env=dict(XML_OUTPUT_FILE=output_xml),
      )
      self.assertIn(
          '2 test(s) ran; 0 failed; 1 skipped',
          stderr,
      )

      with open(output_xml) as f:
        xml_got = f.read()

      root = ET.fromstring(xml_got)
      self.assertLen(root.findall(".//testcase[@name='test_one']"), 1)
      self.assertLen(root.findall(".//testcase[@name='test_two']"), 1)
      self.assertLen(root.findall('.//failure'), 0)

  def test_env_based_test_filter_one_failing(self):
    """Tests environment-variable based filtering."""
    program = """
    #[test] fn test_failing() { assert_eq(false, true) }
    #[test] fn test_passing() {}
    """
    # First we only run the failing test.
    with tempfile.TemporaryDirectory() as tmpdir:
      output_xml = os.path.join(tmpdir, 'test.xml')
      stderr = self._parse_and_test(
          program,
          alsologtostderr=True,
          want_error=True,
          test_filter='.*_failing',
          extra_env=dict(XML_OUTPUT_FILE=output_xml),
      )
      self.assertIn(
          '2 test(s) ran; 1 failed; 1 skipped',
          stderr,
      )

      with open(output_xml) as f:
        xml_got = f.read()

      root = ET.fromstring(xml_got)
      self.assertLen(
          root.findall(
              ".//testcase[@name='test_failing'][@result='completed']"
          ),
          1,
      )
      self.assertLen(
          root.findall(".//testcase[@name='test_failing']/failure"), 1
      )
      self.assertLen(
          root.findall(".//testcase[@name='test_passing'][@result='filtered']"),
          1,
      )
      self.assertLen(root.findall('.//failure'), 1)

    # Now filter to the passing one.
    with tempfile.TemporaryDirectory() as tmpdir:
      output_xml = os.path.join(tmpdir, 'test.xml')
      stderr = self._parse_and_test(
          program,
          alsologtostderr=True,
          test_filter='.*_passing',
          extra_env=dict(XML_OUTPUT_FILE=output_xml),
      )
      self.assertIn(
          '2 test(s) ran; 0 failed; 1 skipped',
          stderr,
      )

      with open(output_xml) as f:
        xml_got = f.read()

      root = ET.fromstring(xml_got)
      self.assertLen(
          root.findall(
              ".//testcase[@name='test_passing'][@result='completed']"
          ),
          1,
      )
      self.assertLen(
          root.findall(".//testcase[@name='test_failing'][@result='filtered']"),
          1,
      )
      self.assertLen(
          root.findall(".//testcase[@name='test_failing']/failure"), 0
      )
      self.assertLen(root.findall('.//failure'), 0)

  def test_flag_based_test_filter(self):
    """Tests environment-variable based filtering."""
    program = """
    #[test] fn test_one() {}
    #[test] fn test_two() {}
    """
    with tempfile.TemporaryDirectory() as tmpdir:
      output_xml = os.path.join(tmpdir, 'test.xml')
      stderr = self._parse_and_test(
          program,
          alsologtostderr=True,
          extra_flags=['--test_filter', '.*_two'],
          extra_env=dict(XML_OUTPUT_FILE=output_xml),
      )
      self.assertIn(
          '2 test(s) ran; 0 failed; 1 skipped',
          stderr,
      )

      with open(output_xml) as f:
        xml_got = f.read()

      print('xml_got:', xml_got)
      root = ET.fromstring(xml_got)
      print('root:', root)
      self.assertLen(
          root.findall(".//testcase[@name='test_two'][@result='completed']"), 1
      )
      self.assertLen(
          root.findall(".//testcase[@name='test_one'][@result='filtered']"), 1
      )
      self.assertLen(root.findall('.//failure'), 0)

  def test_both_test_filters_errors(self):
    """Tests environment-variable based filtering."""
    program = """
    #[test] fn test_one() {}
    #[test] fn test_two() {}
    """
    stderr = self._parse_and_test(
        program,
        alsologtostderr=True,
        want_error=True,
        test_filter='.*_two',
        extra_flags=['--test_filter=".*_two"'],
    )
    self.assertIn(
        'only one is allowed',
        stderr,
    )

  def test_alternative_stdlib_path(self):
    with tempfile.TemporaryDirectory(suffix='stdlib') as stdlib_dir:
      # Make a std.x file in our fake stdlib.
      with open(os.path.join(stdlib_dir, 'std.x'), 'w') as fake_std:
        print('pub fn my_stdlib_func(x: u32) -> u32 { x }', file=fake_std)

      # Invoke the function in our fake std.x which should be appropriately
      # resolved via the dslx_stdlib_path flag.
      program = """
      import std;

      #[test]
      fn test_alternative_stdlib() { assert_eq(std::my_stdlib_func(u32:42), u32:42); }
      """
      self._parse_and_test(
          program,
          alsologtostderr=True,
          extra_flags=[f'--dslx_stdlib_path={stdlib_dir}'],
      )


if __name__ == '__main__':
  test_base.main()
