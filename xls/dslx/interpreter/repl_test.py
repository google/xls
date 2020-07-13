# Lint as: python3
#
# Copyright 2020 Google LLC
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
"""Tests for xls.dslx.interpreter.repl binary."""

import subprocess as subp

from xls.common import runfiles
from xls.common import test_base

REPL_PATH = runfiles.get_path('xls/dslx/interpreter/repl')


class ReplTest(test_base.TestCase):

  def test_simple_command(self):
    # For some unclear reason pylint doesn't like the input keyword here but it
    # works fine.
    # pylint: disable=unexpected-keyword-arg
    output = subp.check_output([REPL_PATH], input=b'u32:2+u32:2')\
        .decode('utf-8')
    self.assertEqual(output, 'bits[32]:0x4\n')

  def test_stdlib_expr(self):
    # pylint: disable=unexpected-keyword-arg
    output = subp.check_output([REPL_PATH], input=b'std::smul(sN[2]:0b11, sN[3]:0b001)')\
        .decode('utf-8')
    self.assertEqual(output, 'bits[5]:0x1f\n')


if __name__ == '__main__':
  test_base.main()
