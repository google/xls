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

import subprocess

from absl.testing import absltest
from xls.common import runfiles

PROC_CONSTANCY_CHECKER_MAIN_PATH = runfiles.get_path(
    'xls/dev_tools/proc_constancy_checker_main'
)

TEST_IR_PATH = runfiles.get_path('xls/dev_tools/testdata/constancy_test.ir')


class ProcConstancyCheckerMainTest(absltest.TestCase):

  def test_constancy_detection_node_mode(self):
    cmd = [
        PROC_CONSTANCY_CHECKER_MAIN_PATH,
        f'--ir_path={TEST_IR_PATH}',
        '--unroll_count=3',
        '--mode=node',
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        check=False,
    )
    self.assertEqual(p.returncode, 0, msg=p.stderr)
    self.assertIn('CONSTANT NODE DETECTED', p.stdout)
    self.assertIn("'const_zero'", p.stdout)
    self.assertNotIn("CONSTANT NODE DETECTED: 'trailing_zero'", p.stdout)

  def test_constancy_detection_bit_mode(self):
    cmd = [
        PROC_CONSTANCY_CHECKER_MAIN_PATH,
        f'--ir_path={TEST_IR_PATH}',
        '--unroll_count=3',
        '--mode=bit',
        '--node_filter=trailing_zero',
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        check=False,
    )
    self.assertEqual(p.returncode, 0, msg=p.stderr)
    self.assertIn('CONSTANT BIT DETECTED', p.stdout)
    self.assertIn("'trailing_zero' bit [0]", p.stdout)
    self.assertIn("'trailing_zero' bit [1]", p.stdout)

  def test_fail_on_constants_flag(self):
    cmd = [
        PROC_CONSTANCY_CHECKER_MAIN_PATH,
        f'--ir_path={TEST_IR_PATH}',
        '--unroll_count=3',
        '--mode=node',
        '--fail_on_constants',
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        check=False,
    )
    self.assertNotEqual(p.returncode, 0)
    self.assertIn('CONSTANT NODE DETECTED', p.stdout)
    self.assertIn("'const_zero'", p.stdout)

  def test_fail_on_constants_flag_bit_mode(self):
    cmd = [
        PROC_CONSTANCY_CHECKER_MAIN_PATH,
        f'--ir_path={TEST_IR_PATH}',
        '--unroll_count=3',
        '--mode=bit',
        '--node_filter=trailing_zero',
        '--fail_on_constants',
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        check=False,
    )
    self.assertNotEqual(p.returncode, 0)
    self.assertIn('CONSTANT BIT DETECTED', p.stdout)
    self.assertIn("'trailing_zero' bit [0]", p.stdout)
    self.assertIn("'trailing_zero' bit [1]", p.stdout)

  def test_node_filter(self):
    cmd = [
        PROC_CONSTANCY_CHECKER_MAIN_PATH,
        f'--ir_path={TEST_IR_PATH}',
        '--unroll_count=3',
        '--mode=node',
        '--node_filter=add1',
    ]
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding='utf-8',
        check=False,
    )
    self.assertEqual(p.returncode, 0, msg=p.stderr)
    self.assertIn('Constant Checks:     0', p.stdout)
    self.assertNotIn("'const_zero'", p.stdout)


if __name__ == '__main__':
  absltest.main()
