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
"""Tests for strip_comments_main."""

import subprocess as subp

from absl.testing import absltest
from xls.common import runfiles

_STRIP_COMMENTS_MAIN_PATH = runfiles.get_path('xls/dslx/strip_comments_main')
_SHA256_PATH = runfiles.get_path('xls/examples/sha256.x')


class StripCommentsMainTest(absltest.TestCase):

  def test_sha256_example(self):
    with open(_SHA256_PATH) as f:
      original_text = f.read()
    output = subp.check_output(
        [_STRIP_COMMENTS_MAIN_PATH, _SHA256_PATH], encoding='utf-8'
    )
    # Check all comments have been stripped out.
    self.assertNotIn('//', output)
    self.assertNotEqual(original_text, output)

  def test_unterminated_string(self):
    original = '"'
    f = self.create_tempfile(content=original)

    p = subp.run(
        [_STRIP_COMMENTS_MAIN_PATH, f.full_path],
        encoding='utf-8',
        stderr=subp.PIPE,
        check=False,
    )
    self.assertNotEqual(p.returncode, 0)
    self.assertIn(
        'Reached end of file without finding a closing double quote.', p.stderr
    )

    output = subp.check_output(
        [_STRIP_COMMENTS_MAIN_PATH, f.full_path, '--original_on_error'],
        encoding='utf-8',
    )
    # Check even though it was a scanner error we got the original.
    self.assertEqual(output, original)


if __name__ == '__main__':
  absltest.main()
