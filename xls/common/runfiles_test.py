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

from absl.testing import absltest
from xls.common import runfiles


class RunfilesTest(absltest.TestCase):
  """Test for runfiles."""

  def testGetPath(self):

    def runfile_contents(path):
      with open(runfiles.get_path(path), 'r') as f:
        return f.read()

    self.assertEqual(runfile_contents('xls/common/testdata/foo.txt'), 'FOO\n')
    self.assertEqual(runfile_contents('xls/common/testdata/bar.txt'), 'BAR\n')

  def testGetContentsAsText(self):
    self.assertEqual(
        runfiles.get_contents_as_text('xls/common/testdata/foo.txt'), 'FOO\n'
    )
    self.assertEqual(
        runfiles.get_contents_as_text('xls/common/testdata/bar.txt'), 'BAR\n'
    )

  def testNonexistantFile(self):
    with self.assertRaises(FileNotFoundError):
      runfiles.get_path('not/a/file.txt')
    with self.assertRaises(FileNotFoundError):
      runfiles.get_contents_as_text('not/a/file.txt')


if __name__ == '__main__':
  absltest.main()
