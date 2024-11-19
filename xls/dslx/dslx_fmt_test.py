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


if __name__ == '__main__':
  absltest.main()
