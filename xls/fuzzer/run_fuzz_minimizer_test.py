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

"""Tests for xls.fuzzer.run_fuzz."""

import os
import subprocess
from xls.common import runfiles
from xls.common import test_base
from xls.fuzzer import run_fuzz
from xls.fuzzer import sample_runner
from xls.fuzzer.python import cpp_sample as sample

# IR parser binary. Reads in a IR file and parses it. Raises an error on
# failure.
PARSE_IR = runfiles.get_path('xls/tools/parse_ir')


class RunFuzzMinimizzerTest(test_base.TestCase):

  def test_minimize_ir_minimization_possible(self):
    # Add an invalid codegen flag to inject an error into the running of the
    # sample. The error is unconditional so IR minimization should be able to
    # reduce the sample to a minimal function (just returns a parameter).
    s = sample.Sample(
        'fn main(x: u8) -> u8 { -x }',
        sample.SampleOptions(
            input_is_dslx=True,
            ir_converter_args=['--top=main'],
            codegen=True,
            codegen_args=('--invalid_flag!!!',)),
        sample.parse_args_batch('bits[8]:7\nbits[8]:100'))
    success = test_base.TempFileCleanup.SUCCESS  # type: test_base.TempFileCleanup
    run_dir = self.create_tempdir(cleanup=success).full_path
    with self.assertRaises(sample_runner.SampleError):
      run_fuzz.run_sample(s, run_dir=run_dir)
    minimized_ir_path = run_fuzz.minimize_ir(s, run_dir)
    self.assertIsNotNone(minimized_ir_path)
    self.assertIn('ir_minimizer_test.sh', os.listdir(run_dir))
    # Validate the minimized IR.
    with open(minimized_ir_path, 'r') as f:
      contents = f.read()
      self.assertIn('package ', contents)
      self.assertIn('fn ', contents)
      # It should be reduced to simply a literal.
      self.assertIn('ret literal', contents)
    # And verify the minimized IR parses.
    subprocess.check_call([PARSE_IR, minimized_ir_path])

  def test_minimize_ir_no_minimization_possible(self):
    # Verify that IR minimization at least generates a minimization test script
    # and doesn't blow up if the IR is not minimizable. In this case, "not
    # minimizable" means that no error is ever generated when running the
    # sample.
    s = sample.Sample(
        'fn main(x: u8) -> u8 { -x }',
        sample.SampleOptions(
            input_is_dslx=True, ir_converter_args=['--top=main']),
        sample.parse_args_batch('bits[8]:7\nbits[8]:100'))
    success = test_base.TempFileCleanup.SUCCESS  # type: test_base.TempFileCleanup
    run_dir = self.create_tempdir(cleanup=success).full_path
    run_fuzz.run_sample(s, run_dir=run_dir)
    self.assertIsNone(run_fuzz.minimize_ir(s, run_dir))
    dir_contents = os.listdir(run_dir)
    self.assertIn('ir_minimizer_test.sh', dir_contents)

  def test_minimize_jit_interpreter_mismatch(self):
    s = sample.Sample(
        'fn main(x: u8) -> u8 { !x }',
        sample.SampleOptions(
            input_is_dslx=True, ir_converter_args=['--top=main']),
        sample.parse_args_batch('bits[8]:0xff\nbits[8]:0x42'))
    success = test_base.TempFileCleanup.SUCCESS  # type: test_base.TempFileCleanup
    run_dir = self.create_tempdir(cleanup=success).full_path
    run_fuzz.run_sample(s, run_dir=run_dir)
    minimized_ir_path = run_fuzz.minimize_ir(
        s, run_dir, inject_jit_result='bits[32]:0x0')
    self.assertIsNotNone(minimized_ir_path)
    with open(minimized_ir_path, 'r') as f:
      contents = f.read()
      self.assertIn('package ', contents)
      self.assertIn('fn ', contents)
      # It should be reduced to simply a literal.
      self.assertIn('ret literal', contents)
    # And verify the minimized IR parses.
    subprocess.check_call([PARSE_IR, minimized_ir_path])


if __name__ == '__main__':
  test_base.main()
