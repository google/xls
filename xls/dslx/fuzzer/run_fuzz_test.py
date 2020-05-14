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
#
# Lint as: python3

"""Tests for xls.dslx.fuzzer.run_fuzz."""

import os
import random
import subprocess

from absl import flags

from xls.dslx.fuzzer import ast_generator
from xls.dslx.fuzzer import run_fuzz
from xls.dslx.fuzzer import sample
from xls.dslx.fuzzer import sample_runner
from xls.common import runfiles
from absl.testing import absltest
from absl.testing import parameterized

flags.DEFINE_boolean('codegen', True,
                     'Whether to generate Verilog for generated samples.')
flags.DEFINE_boolean('update_golden', False,
                     'Whether to update golden reference files.')
FLAGS = flags.FLAGS

X_SUB_Y_IR = """package x_sub_y

fn main(x: bits[8], y: bits[8]) -> bits[8] {
  ret sub.1: bits[8] = sub(x, y)
}
"""

# IR parser binary. Reads in a IR file and parses it. Raises an error on
# failure.
PARSE_IR = runfiles.get_path('xls/tools/parse_ir')


class RunFuzzTest(parameterized.TestCase):

  KWARGS = {
      'calls_per_sample': 4,
      'save_temps': False,
      'sample_count': 4,
      'return_samples': True,
      'codegen': True
  }
  GOLDEN_REFERENCE_FMT = 'xls/dslx/fuzzer/testdata/run_fuzz_test{codegen}.seed_{seed}_sample_{sample}.x'
  SEED_TO_CHECK_LIMIT = 2
  SAMPLE_TO_CHECK_LIMIT = 1

  def setUp(self):
    super(RunFuzzTest, self).setUp()
    self.KWARGS['codegen'] = FLAGS.codegen

  def _get_ast_options(self) -> ast_generator.AstGeneratorOptions:
    return ast_generator.AstGeneratorOptions(blacklist_divide=True)

  def test_repeatable_within_process(self):
    samples0 = run_fuzz.run_fuzz(
        random.Random(7), self._get_ast_options(), **self.KWARGS)
    samples1 = run_fuzz.run_fuzz(
        random.Random(7), self._get_ast_options(), **self.KWARGS)
    self.assertEqual(samples0, samples1)

  @parameterized.named_parameters(*tuple(
      dict(testcase_name='seed_{}'.format(x), seed=x) for x in range(47)))
  def test_first_n_seeds(self, seed):
    if FLAGS.update_golden and seed >= self.SEED_TO_CHECK_LIMIT:
      # Skip running unnecessary tests if updating golden because the test is
      # slow and runs unsharded.
      return
    samples = run_fuzz.run_fuzz(
        random.Random(seed), self._get_ast_options(), **self.KWARGS)
    for i in range(self.KWARGS['sample_count']):
      if seed < self.SEED_TO_CHECK_LIMIT and i < self.SAMPLE_TO_CHECK_LIMIT:
        path = self.GOLDEN_REFERENCE_FMT.format(
            seed=seed, sample=i, codegen='' if FLAGS.codegen else '_no_codegen')
        if FLAGS.update_golden:
          with open(path, 'w') as f:
            f.write(samples[i].input_text)
        else:
          expected = runfiles.get_contents_as_text(path)
          self.assertMultiLineEqual(expected, samples[i].input_text)

  def test_minimize_ir_minimization_possible(self):
    # Add an invalid codegen flag to inject an error into the running of the
    # sample. The error is unconditional so IR minimization should be able to
    # reduce the sample to a minimal function (just returns a parameter).
    s = sample.Sample(
        'fn main(x: u8) -> u8 { -x }',
        sample.SampleOptions(codegen=True, codegen_args=('--invalid_flag!!!',)),
        sample.parse_args_batch('bits[8]:7\nbits[8]:100'))
    run_dir = self.create_tempdir(
        cleanup=absltest.TempFileCleanup.SUCCESS).full_path
    with self.assertRaises(sample_runner.SampleError):
      run_fuzz.run_sample(s, run_dir=run_dir)
    minimized_ir_path = run_fuzz.minimize_ir(s, run_dir)
    self.assertIsNotNone(minimized_ir_path)
    self.assertIn('ir_minimizer_test.sh', os.listdir(run_dir))
    # Sanity check the minimized IR.
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
    s = sample.Sample('fn main(x: u8) -> u8 { -x }', sample.SampleOptions(),
                      sample.parse_args_batch('bits[8]:7\nbits[8]:100'))
    run_dir = self.create_tempdir(
        cleanup=absltest.TempFileCleanup.SUCCESS).full_path
    run_fuzz.run_sample(s, run_dir=run_dir)
    self.assertIsNone(run_fuzz.minimize_ir(s, run_dir))
    dir_contents = os.listdir(run_dir)
    self.assertIn('ir_minimizer_test.sh', dir_contents)

  def test_minimize_jit_interpreter_mismatch(self):
    s = sample.Sample('fn main(x: u8) -> u8 { ~x }', sample.SampleOptions(),
                      sample.parse_args_batch('bits[8]:0xff\nbits[8]:0x42'))
    run_dir = self.create_tempdir(
        cleanup=absltest.TempFileCleanup.SUCCESS).full_path
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
  absltest.main()
