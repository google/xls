# Lint as: python3
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
"""Tests for xls.dslx.interpreter.parse_and_interpret."""

import io

import unittest.mock as mock
from pyfakefs import fake_filesystem_unittest as ffu

from xls.dslx.interpreter import parse_and_interpret
from absl.testing import absltest


class ParseAndInterpretTest(absltest.TestCase):

  def test_assertion_failure_prints_positionally(self):
    program = """
    #![test]
    fn foo_test() {
      assert_eq(false, true)
    }
    """
    mock_stderr = io.StringIO()
    filename = 'test_filename.x'
    with ffu.Patcher() as patcher:
      patcher.fs.CreateFile(filename, contents=program)
      with mock.patch('sys.stderr', mock_stderr):
        parse_and_interpret.parse_and_test(
            program, 'test_program', filename=filename, raise_on_error=False)
    self.assertIn('* 0004:       assert_eq(false, true)',
                  mock_stderr.getvalue())
    self.assertIn(
        '        ~~~~~~~~~~~~~~~^-----------^ The program being interpreted failed!',
        mock_stderr.getvalue())


if __name__ == '__main__':
  absltest.main()
