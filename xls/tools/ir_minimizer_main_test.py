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

import os
import stat
import subprocess

from xls.common import runfiles
from absl.testing import absltest

IR_MINIMIZER_MAIN_PATH = runfiles.get_path('xls/tools/ir_minimizer_main')

ADD_IR = """package foo

fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  not.1: bits[32] = not(x, id=1)
  add.2: bits[32] = add(not.1, y, id=2)
  ret not.3: bits[32] = not(add.2, id=3)
}
"""


class IrMinimizerMainTest(absltest.TestCase):

  def _write_sh_script(self, path, commands):
    with open(path, 'w') as f:
      f.write('#!/bin/sh -e\n')
      f.write('\n'.join(commands))
    st = os.stat(path)
    os.chmod(path, st.st_mode | stat.S_IXUSR)

  def test_minimize_add_no_remove_params(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, ['/bin/grep add $1'])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params=false', ir_file.full_path
    ])
    self.assertEqual(
        minimized_ir.decode('utf-8'), """package foo

fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  literal.5: bits[32] = literal(value=0, id=5)
  ret add.2: bits[32] = add(literal.5, literal.5, id=2)
}
""")

  def test_minimize_add_remove_params(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, ['/bin/grep add $1'])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params', ir_file.full_path
    ])
    self.assertEqual(
        minimized_ir.decode('utf-8'), """package foo

fn foo() -> bits[32] {
  literal.5: bits[32] = literal(value=0, id=5)
  ret add.2: bits[32] = add(literal.5, literal.5, id=2)
}
""")

  def test_no_reduction_possible(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    # Shell script is run with -e so if any of the greps fail then the script
    # fails.
    self._write_sh_script(test_sh_file.full_path, [
        '/bin/grep not.1.*x $1', '/bin/grep add.2.*not.1.*y $1',
        '/bin/grep not.3.*add.2 $1'
    ])
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
        '--can_remove_params', ir_file.full_path
    ])
    self.assertEqual(minimized_ir.decode('utf-8'), ADD_IR)

  def test_verify_return_code(self):
    # If the test script never successfully runs, then ir_minimizer_main should
    # return nonzero.
    ir_file = self.create_tempfile(content=ADD_IR)
    test_sh_file = self.create_tempfile()
    self._write_sh_script(test_sh_file.full_path, ['exit 1'])
    with self.assertRaises(subprocess.CalledProcessError):
      subprocess.check_call([
          IR_MINIMIZER_MAIN_PATH, '--test_executable=' + test_sh_file.full_path,
          '--can_remove_params', ir_file.full_path
      ])

  def test_minimize_jit_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    minimized_ir = subprocess.check_output([
        IR_MINIMIZER_MAIN_PATH, '--test_llvm_jit',
        '--simplify_with_optimization_pipeline',
        '--input=bits[32]:0x42; bits[32]:0x123',
        '--test_only_inject_jit_result=bits[32]:0x22', ir_file.full_path
    ],
                                           stderr=subprocess.PIPE)
    # The minimizer should reduce the test case to just a literal.
    self.assertIn('ret literal', minimized_ir.decode('utf-8'))

  def test_minimize_jit_mismatch_but_no_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run([
        IR_MINIMIZER_MAIN_PATH, '--test_llvm_jit',
        '--simplify_with_optimization_pipeline',
        '--input=bits[32]:0x42; bits[32]:0x123', ir_file.full_path
    ],
                          stderr=subprocess.PIPE,
                          check=False)
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('main function provided does not fail',
                  comp.stderr.decode('utf-8'))


if __name__ == '__main__':
  absltest.main()
