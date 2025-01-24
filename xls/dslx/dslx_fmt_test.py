# Copyright 2023 The XLS Authors
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

"""Tests for dslx_fmt tool (binary)."""

import subprocess as subp
import textwrap

from absl.testing import absltest
from xls.common import runfiles


_DSLX_FMT_PATH = runfiles.get_path('xls/dslx/dslx_fmt')


class DslxFmtTest(absltest.TestCase):

  def _run(self, contents: str, in_place: bool = False) -> str:
    f = self.create_tempfile(content=contents)
    if in_place:
      output = subp.check_call([_DSLX_FMT_PATH, '-i', f.full_path])
      return f.read_text()

    # When not in-place we take the result from stdout.
    output = subp.check_output([_DSLX_FMT_PATH, f.full_path], encoding='utf-8')
    return output

  def test_small_no_spaces_example(self):
    contents = 'fn f()->u32{u32:42}'
    want = textwrap.dedent("""\
    fn f() -> u32 { u32:42 }
    """)
    self.assertEqual(self._run(contents), want)

  def test_disabled(self):
    contents = '// dslx-fmt::off\nfn f()->u32{u32:42}'
    self.assertEqual(self._run(contents), contents)

  def test_disabled_in_place(self):
    contents = '// dslx-fmt::off\nfn f()->u32{u32:42}'
    self.assertEqual(self._run(contents, in_place=True), contents)

  def test_small_no_spaces_example_in_place(self):
    contents = 'fn f()->u32{u32:42}'
    want = textwrap.dedent("""\
    fn f() -> u32 { u32:42 }
    """)
    self.assertEqual(self._run(contents, in_place=True), want)

  def test_multi_file_in_place_fmt(self):
    contents0 = 'fn f()->u32{u32:42}'
    contents1 = 'fn g()->u32{u32:42}'
    want0 = 'fn f() -> u32 { u32:42 }\n'
    want1 = 'fn g() -> u32 { u32:42 }\n'
    f0 = self.create_tempfile(content=contents0)
    f1 = self.create_tempfile(content=contents1)
    subp.check_output([_DSLX_FMT_PATH, '-i', f0.full_path, f1.full_path])
    self.assertEqual(f0.read_text(), want0)
    self.assertEqual(f1.read_text(), want1)

  def test_no_args(self):
    with self.assertRaises(subp.CalledProcessError) as e:
      subp.check_output([_DSLX_FMT_PATH], encoding='utf-8', stderr=subp.PIPE)
    self.assertIn('No command-line arguments to format', str(e.exception.stderr))

  def test_error_for_multifile_with_stdin(self):
    with self.assertRaises(subp.CalledProcessError) as e:
      subp.check_output([_DSLX_FMT_PATH, '-i', '/tmp/dne.x', '-'], encoding='utf-8',
                        stderr=subp.PIPE)
    self.assertIn('Cannot have stdin along with file arguments', str(e.exception.stderr))

  def test_error_for_not_in_place_multifile(self):
    with self.assertRaises(subp.CalledProcessError) as e:
      subp.check_output([_DSLX_FMT_PATH, '/tmp/dne.x', '/tmp/dne2.x'], encoding='utf-8',
                        stderr=subp.PIPE)
    self.assertIn('Cannot have multiple input files when in-place formatting is disabled',
                  str(e.exception.stderr))

  def test_error_for_opportunistic_postcondition_multifile(self):
    with self.assertRaises(subp.CalledProcessError) as e:
      subp.check_output(
        [_DSLX_FMT_PATH, '--opportunistic_postcondition', '/tmp/dne.x', '/tmp/dne2.x'],
        encoding='utf-8', stderr=subp.PIPE)
    self.assertIn('Cannot have multiple input files when opportunistic-postcondition is enabled',
                  str(e.exception.stderr))

  def test_error_for_error_on_changes_multifile(self):
    with self.assertRaises(subp.CalledProcessError) as e:
      subp.check_output([_DSLX_FMT_PATH, '--error_on_changes', '/tmp/dne.x', '/tmp/dne2.x'],
                        encoding='utf-8', stderr=subp.PIPE)
    self.assertIn('Cannot have multiple input files when error-on-changes is enabled',
                  str(e.exception.stderr))

  def test_error_for_stdin_in_place(self):
    with self.assertRaises(subp.CalledProcessError) as e:
      subp.check_output([_DSLX_FMT_PATH, '-i', '-'], encoding='utf-8', stderr=subp.PIPE)
    self.assertIn('Cannot format stdin with in-place formatting', str(e.exception.stderr))

  def test_stdin_streaming_mode(self):
    p = subp.run([_DSLX_FMT_PATH, '-'], input='fn f()->u32{u32:42}', encoding='utf-8',
                 stdout=subp.PIPE, stderr=subp.PIPE)
    print('stderr:\n', p.stderr)
    p.check_returncode()
    self.assertEqual(p.stdout, 'fn f() -> u32 { u32:42 }\n')


if __name__ == '__main__':
  absltest.main()
