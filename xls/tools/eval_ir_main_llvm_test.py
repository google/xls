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

import ctypes
import struct
import subprocess

from absl.testing import absltest
from xls.common import runfiles
from xls.common import test_base

EVAL_IR_MAIN_PATH = runfiles.get_path('xls/tools/eval_ir_main')

ADD_IR = """package foo

top fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
"""


class EvalMainLLVMTest(absltest.TestCase):
  """Tests which need to be able to hit llvm tools & run their interpreters/etc.

  This means that we can't run with msan since that doesn't mix well with the
  way that lli functions.
  """

  def test_generate_main_includes_used_inputs(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    main_wrapper = self.create_tempfile()
    jitted_code_file = self.create_tempfile()
    inputs = [
        (1, 3),  # 4
        (3, 5),  # 8
        (ctypes.c_uint32(-1).value, 33),  # 32
        (ctypes.c_uint32(-33).value, 33),  # 0
        (33, ctypes.c_uint32(-45).value),  # -12
    ]
    expected_results = tuple((l + r) & 0xFFFFFFFF for l, r in inputs)
    input_flags = []
    for left, right in inputs:
      input_flags.append(f'bits[32]:{hex(left)}; bits[32]:{hex(right)}')
    input_file = self.create_tempfile(content='\n'.join(input_flags))
    argv = [
        EVAL_IR_MAIN_PATH,
        '--llvm_jit_main_wrapper_write_is_linked',
        f'--llvm_jit_main_wrapper_output={main_wrapper.full_path}',
        f'--llvm_jit_ir_output={jitted_code_file.full_path}',
        f'--input_file={input_file.full_path}',
        ir_file.full_path,
    ]
    comp = subprocess.run(
        argv, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Check the results are what we expect.
    self.assertEqual(comp.returncode, 0, comp.stderr)
    results = comp.stdout.decode().splitlines()
    for input_values, expected, result in zip(
        inputs, expected_results, results
    ):
      result_num = int(result.split(':')[1], 16)
      l, r = input_values
      self.assertEqual(
          result_num,
          expected,
          f'unexpected result for {l} + {r} = {result}. Expected:'
          f' {hex(expected)}',
      )

    # link llvm bitcode.
    linked_bitcode = self.create_tempfile()
    subprocess.run(
        [
            runfiles.get_path('llvm/llvm-link', repository='llvm-project'),
            '-S',
            '-o',
            linked_bitcode.full_path,
            main_wrapper.full_path,
            jitted_code_file.full_path,
        ],
        check=True,
    )
    # Run lli
    lli_run = subprocess.run(
        [
            runfiles.get_path('llvm/lli', repository='llvm-project'),
            linked_bitcode.full_path,
        ],
        stdout=subprocess.PIPE,
        check=True,
        encoding=None,
    )
    self.assertEqual(
        struct.unpack('I' * (len(lli_run.stdout) // 4), lli_run.stdout),
        expected_results,
        f'raw bytes are: {lli_run.stdout}',
    )


if __name__ == '__main__':
  test_base.main()
