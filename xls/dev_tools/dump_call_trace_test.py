#
# Copyright 2026 The XLS Authors
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
"""Tests for xls.dev_tools.dump_call_trace."""

import os
import re
import subprocess
import tempfile
import textwrap
from typing import List

from absl.testing import absltest

from xls.common import runfiles


DSLX_PROGRAM = r'''
fn baz(z: u32) -> u32 { z + u32:1 }
fn bar(y: u32) -> u32 { baz(y + u32:1) }
fn foo(x: u32) -> u32 { bar(x + u32:1) }

#[test]
fn call_trace_test() {
  let _ = foo(u32:10);
  let _ = baz(u32:2);
  ()
}
'''

DSLX_FOR_LOOP_PROGRAM = r'''
fn foo(x: u32) -> u32 { x + u32:1 }

#[test]
fn for_loop_inside_test() {
  let _sum = for (value, accum): (u32, u32) in u32:1..u32:3 {
    let _ = foo(value);
    let _ = foo(value + u32:10);
    accum
  }(u32:0);
  ()
}
'''

def _write_textproto(path: str, text: str) -> None:
  with open(path, 'w', encoding='utf-8') as f:
    f.write(text)


def _line_col(program: str, needle: str) -> tuple[int, int]:
  for idx, line in enumerate(program.splitlines(), start=1):
    col = line.find(needle)
    if col != -1:
      return idx, col
  raise ValueError(f'needle not found: {needle}')


def _file_line_col(path: str, needle: str) -> tuple[int, int]:
  with open(path, encoding='utf-8') as f:
    return _line_col(f.read(), needle)


def _run_dump(input_path: str, extra_flags: List[str] | None = None) -> str:
  tool = runfiles.get_path('xls/dev_tools/dump_call_trace')
  cmd = [tool, f'--input={input_path}', '--log_prefix=false']
  if extra_flags:
    cmd.extend(extra_flags)
  proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE, encoding='utf-8')
  # Tool prints to stdout.
  return proc.stdout


def _normalize_locations(text: str) -> str:
  return re.sub(r':\d+:\d+:', ':<loc>:', text)


class DumpCallTraceTest(absltest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmpdir = tempfile.TemporaryDirectory()

  def tearDown(self):
    self._tmpdir.cleanup()
    super().tearDown()

  def _proto_path(self, name: str) -> str:
    return os.path.join(self._tmpdir.name, name)

  def _run_and_write_results_proto(
      self, dslx_program: str, filename: str,
      interpreter: str = 'dslx-interpreter') -> str:
    prog_path = self._proto_path('prog.x')
    _write_textproto(prog_path, dslx_program)
    out_path = self._proto_path(filename)
    interp = runfiles.get_path('xls/dslx/interpreter_main')
    cmd = [
        interp,
        prog_path,
        '--compare=none',
        f'--evaluator={interpreter}',
        '--trace_calls',
        '--warnings_as_errors=false',
        f'--output_results_proto={out_path}',
        '--log_prefix=false',
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE, encoding='utf-8')
    return out_path

  def test_basic(self):
    path = self._run_and_write_results_proto(DSLX_PROGRAM, 'results.textproto')
    out = _run_dump(path)
    dslx = self._proto_path('prog.x')
    call_line, call_col = _file_line_col(dslx, 'fn call_trace_test()')
    foo_line, foo_col = _file_line_col(dslx, 'foo(u32:10)')
    bar_line, bar_col = _file_line_col(dslx, 'baz(y + u32:1)')
    baz_line, baz_col = _file_line_col(dslx, 'z + u32:1')
    baz_call_line, baz_call_col = _file_line_col(dslx, 'baz(u32:2)')
    expected = textwrap.dedent(f"""
    {dslx}:{call_line}:{call_col}: call_trace_test(
    )
      {dslx}:{foo_line}:{foo_col}: foo(
        bits[32]:10
      )
        {dslx}:{bar_line}:{bar_col}: bar(
          bits[32]:11
        )
          {dslx}:{baz_line}:{baz_col}: baz(
            bits[32]:12
          )
          baz(...) => bits[32]:13
        bar(...) => bits[32]:13
      foo(...) => bits[32]:13
      {dslx}:{baz_call_line}:{baz_call_col}: baz(
        bits[32]:2
      )
      baz(...) => bits[32]:3
    call_trace_test(...) => ()
    """).lstrip()
    self.assertEqual(_normalize_locations(out),
                     _normalize_locations(expected))

  def test_basic_ir_interpreter(self):
    path = self._run_and_write_results_proto(
      DSLX_PROGRAM, 'results.textproto', 'ir-interpreter')
    out = _run_dump(path,
                    extra_flags=['--function=__itok__prog__call_trace_test'])
    dslx = self._proto_path('prog.x')
    foo_line, foo_col = _file_line_col(dslx, 'foo(u32:10)')
    bar_line, bar_col = _file_line_col(dslx, 'baz(y + u32:1)')
    baz_line, baz_col = _file_line_col(dslx, 'z + u32:1')
    baz_call_line, baz_call_col = _file_line_col(dslx, 'baz(u32:2)')
    expected = textwrap.dedent(f"""
    <unknown>: __itok__prog__call_trace_test(
      token
      bits[1]:1
    )
      {dslx}:{foo_line}:{foo_col}: __prog__foo(
        bits[32]:10
      )
        {dslx}:{bar_line}:{bar_col}: __prog__bar(
          bits[32]:11
        )
          {dslx}:{baz_line}:{baz_col}: __prog__baz(
            bits[32]:12
          )
          __prog__baz(...) => bits[32]:13
        __prog__bar(...) => bits[32]:13
      __prog__foo(...) => bits[32]:13
      {dslx}:{baz_call_line}:{baz_call_col}: __prog__baz(
        bits[32]:2
      )
      __prog__baz(...) => bits[32]:3
    __itok__prog__call_trace_test(...) => (token, ())
    """).lstrip()
    self.assertEqual(_normalize_locations(out),
                     _normalize_locations(expected))

  def test_function_filter(self):
    path = self._run_and_write_results_proto(DSLX_PROGRAM,
                                             'results_fn.textproto')
    out = _run_dump(path, ['--function=bar'])
    dslx = self._proto_path('prog.x')
    expected = textwrap.dedent(f"""
    {dslx}:3:27: bar(
      bits[32]:11
    )
      {dslx}:2:27: baz(
        bits[32]:12
      )
      baz(...) => bits[32]:13
    bar(...) => bits[32]:13
    """).lstrip()
    self.assertEqual(_normalize_locations(out),
                     _normalize_locations(expected))

  def test_for_loop_in_dslx_test(self):
    out_path = self._run_and_write_results_proto(
      DSLX_FOR_LOOP_PROGRAM, 'for_loop_results.textproto')
    out = _run_dump(out_path)
    dslx = self._proto_path('prog.x')
    test_line, test_col = _file_line_col(dslx, 'fn for_loop_inside_test()')
    foo_line, foo_col = _file_line_col(dslx, 'foo(value);')
    foo2_line, foo2_col = _file_line_col(dslx, 'foo(value + u32:10);')
    expected = textwrap.dedent(f"""
    {dslx}:{test_line}:{test_col}: for_loop_inside_test(
    )
      {dslx}:{foo_line}:{foo_col}: foo(
        bits[32]:1
      )
      foo(...) => bits[32]:2
      {dslx}:{foo2_line}:{foo2_col}: foo(
        bits[32]:11
      )
      foo(...) => bits[32]:12
      {dslx}:{foo_line}:{foo_col}: foo(
        bits[32]:2
      )
      foo(...) => bits[32]:3
      {dslx}:{foo2_line}:{foo2_col}: foo(
        bits[32]:12
      )
      foo(...) => bits[32]:13
    for_loop_inside_test(...) => ()
    """).lstrip()
    self.assertEqual(_normalize_locations(out),
                     _normalize_locations(expected))

  def test_for_loop_ir_interpreter(self):
    path = self._run_and_write_results_proto(
      DSLX_FOR_LOOP_PROGRAM,
      'for_loop_ir_results.textproto', 'ir-interpreter')
    out = _run_dump(
      path, extra_flags=['--function=__itok__prog__for_loop_inside_test'])
    dslx = self._proto_path('prog.x')
    expected = textwrap.dedent(f"""
    <unknown>: __itok__prog__for_loop_inside_test(
      token
      bits[1]:1
    )
      {dslx}:5:13: ____itok__prog__for_loop_inside_test_counted_for_0_body(
        bits[32]:0
        (
          token
          bits[1]:1
          bits[32]:0
        )
      )
        {dslx}:6:15: __prog__foo(
          bits[32]:1
        )
        __prog__foo(...) => bits[32]:2
        {dslx}:7:15: __prog__foo(
          bits[32]:11
        )
        __prog__foo(...) => bits[32]:12
      ____itok__prog__for_loop_inside_test_counted_for_0_body(...) => (token, bits[1]:1, bits[32]:0)
      {dslx}:5:13: ____itok__prog__for_loop_inside_test_counted_for_0_body(
        bits[32]:1
        (
          token
          bits[1]:1
          bits[32]:0
        )
      )
        {dslx}:6:15: __prog__foo(
          bits[32]:2
        )
        __prog__foo(...) => bits[32]:3
        {dslx}:7:15: __prog__foo(
          bits[32]:12
        )
        __prog__foo(...) => bits[32]:13
      ____itok__prog__for_loop_inside_test_counted_for_0_body(...) => (token, bits[1]:1, bits[32]:0)
    __itok__prog__for_loop_inside_test(...) => (token, ())
    """).lstrip()
    self.assertEqual(_normalize_locations(out),
                     _normalize_locations(expected))

if __name__ == '__main__':
  absltest.main()
