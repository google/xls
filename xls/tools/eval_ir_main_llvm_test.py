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

import subprocess

from absl.testing import parameterized
from xls.common import runfiles
from xls.common import test_base

EVAL_IR_MAIN_PATH = runfiles.get_path('xls/tools/eval_ir_main')

ADD_IR = """package foo

top fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
"""


class EvalMainLLVMTest(parameterized.TestCase):
  """Tests which need to be able to hit llvm tools & run their interpreters/etc.

  This means that we can't run with msan since that doesn't mix well with the
  way that lli functions.
  """

  def test_one_input_jit_interpreter(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH,
        '--input=bits[32]:0x42; bits[32]:0x123',
        '--use_llvm_jit=true',
        '--use_llvm_jit_interpreter=true',
        ir_file.full_path,
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x165')

if __name__ == '__main__':
  test_base.main()
