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

"""Tests for xls.tools.simulate_module_main."""

import subprocess

from xls.common import runfiles
from xls.common import test_base

CODEGEN_MAIN_PATH = runfiles.get_path('xls/tools/codegen_main')
SIMULATE_MODULE_MAIN_PATH = runfiles.get_path('xls/tools/simulate_module_main')

ADD_IR = """package add

top fn add(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
"""


class SimulateModuleMainTest(test_base.TestCase):

  def test_single_arg_inline(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    verilog_file = self.create_tempfile()
    signature_file = self.create_tempfile()
    subprocess.check_call([
        CODEGEN_MAIN_PATH,
        '--generator=combinational',
        '--output_verilog_path=' + verilog_file.full_path,
        '--output_signature_path=' + signature_file.full_path,
        '--alsologtostderr',
        ir_file.full_path,
    ])
    result = subprocess.check_output([
        SIMULATE_MODULE_MAIN_PATH, '--verilog_simulator=iverilog',
        '--file_type=verilog', '--alsologtostderr', '--v=1',
        '--signature_file=' + signature_file.full_path,
        '--args=bits[32]:7; bits[32]:123', verilog_file.full_path
    ])
    self.assertEqual('bits[32]:0x82', result.decode('utf-8').strip())

  def test_multi_arg_from_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    verilog_file = self.create_tempfile()
    signature_file = self.create_tempfile()
    subprocess.check_call([
        CODEGEN_MAIN_PATH,
        '--generator=combinational',
        '--output_verilog_path=' + verilog_file.full_path,
        '--output_signature_path=' + signature_file.full_path,
        '--alsologtostderr',
        ir_file.full_path,
    ])
    args_file = self.create_tempfile(content="""
      bits[32]:0xf00; bits[32]:1

      bits[32]:2; bits[32]:40

    """)
    result = subprocess.check_output([
        SIMULATE_MODULE_MAIN_PATH, '--verilog_simulator=iverilog',
        '--file_type=verilog', '--alsologtostderr', '--v=1',
        '--signature_file=' + signature_file.full_path,
        '--args_file=' + args_file.full_path, verilog_file.full_path
    ])
    self.assertMultiLineEqual('bits[32]:0xf01\nbits[32]:0x2a\n',
                              result.decode('utf-8'))


if __name__ == '__main__':
  test_base.main()
