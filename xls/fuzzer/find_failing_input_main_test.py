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

"""Tests for find_failing_input_main binary."""

import subprocess

from xls.common import runfiles
from xls.common import test_base

FIND_FAILING_INPUT_MAIN = runfiles.get_path(
    'xls/fuzzer/find_failing_input_main'
)

ADD_IR = """package foo

top fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
"""


class EvalMainTest(test_base.TestCase):

  def test_input_file_no_failure(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(
        content='\n'.join(
            ('bits[32]:0x42; bits[32]:0x123', 'bits[32]:0x10; bits[32]:0xf0f')
        )
    )
    comp = subprocess.run(
        [
            FIND_FAILING_INPUT_MAIN,
            '--input_file=' + input_file.full_path,
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertNotEqual(comp, 0)
    self.assertIn(
        'No input found which results in a mismatch',
        comp.stderr.decode('utf-8'),
    )

  def test_empty_input_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(content='')
    comp = subprocess.run(
        [
            FIND_FAILING_INPUT_MAIN,
            '--input_file=' + input_file.full_path,
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
        check=False,
    )
    self.assertNotEqual(comp, 0)
    self.assertIn(
        'No input found which results in a mismatch',
        comp.stderr.decode('utf-8'),
    )

  def test_input_file_with_failure(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(
        content='\n'.join((
            'bits[32]:0x0; bits[32]:0x0',
            'bits[32]:0x42; bits[32]:0x123',
            'bits[32]:0x0; bits[32]:0x0',
        ))
    )
    result = subprocess.check_output(
        [
            FIND_FAILING_INPUT_MAIN,
            '--input_file=' + input_file.full_path,
            '--alsologtostderr',
            '--test_only_inject_jit_result=bits[32]:0x0',
            ir_file.full_path,
        ],
        stderr=subprocess.PIPE,
    )
    self.assertEqual(result.decode('utf-8'), 'bits[32]:0x42; bits[32]:0x123')


if __name__ == '__main__':
  test_base.main()
