# Copyright 2021 The XLS Authors
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
"""Tests for highlight_main."""

import re
import subprocess as subp

from absl.testing import absltest
from xls.common import runfiles

_HIGHLIGHT_MAIN_PATH = runfiles.get_path('xls/dslx/highlight_main')
_SHA256_PATH = runfiles.get_path('xls/examples/sha256.x')
_ANSI_ESCAPE_RE = re.compile(r'\x1b\[[0-9;]*m')


class HighlightMainTest(absltest.TestCase):

  def test_highlight_then_remove_ansi_codes(self):
    with open(_SHA256_PATH) as f:
      original_text = f.read()
    output = subp.check_output(
        [_HIGHLIGHT_MAIN_PATH, _SHA256_PATH], encoding='utf-8'
    )
    self.assertNotEmpty(_ANSI_ESCAPE_RE.findall(output))
    escapes_removed = re.sub(_ANSI_ESCAPE_RE, '', output)
    self.assertMultiLineEqual(escapes_removed, original_text)

  def test_highlighting_constructs(self):
    # Includes a keyword, a number, a comment, a builtin function, a type, and
    # an identifier.
    s = """\
fn f(x: u32) -> u32 {
  trace!(u32:42)  // Yay!
}"""
    f = self.create_tempfile(content=s)
    output = subp.check_output(
        [_HIGHLIGHT_MAIN_PATH, f.full_path], encoding='utf-8'
    )
    red = '\x1b[1;31m'
    green = '\x1b[1;32m'
    yellow = '\x1b[1;33m'
    blue = '\x1b[1;34m'
    cyan = '\x1b[1;36m'
    reset = '\x1b[1;0m'
    want = f"""\
{yellow}fn{reset} f(x: {green}u32{reset}) -> {green}u32{reset} {{
  {cyan}trace!{reset}({green}u32{reset}:{red}42{reset})  {blue}// Yay!
{reset}}}"""
    self.assertEqual(repr(output), repr(want))


if __name__ == '__main__':
  absltest.main()
