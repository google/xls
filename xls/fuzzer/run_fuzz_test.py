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
from typing import Optional

from absl import flags

from absl.testing import parameterized
from xls.common import test_base
from xls.fuzzer import run_fuzz
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from xls.fuzzer.python import cpp_sample as sample

_CALLS_PER_SAMPLE = 8
_SAMPLE_COUNT = 200
_PROC_TICKS = 100

_WIDE = flags.DEFINE_boolean(
    'wide', default=False, help='Run with wide bits types.')
_GENERATE_PROC = flags.DEFINE_boolean(
    'generate_proc', default=False, help='Generate a proc sample.')


def _get_crasher_dir() -> Optional[str]:
  """Returns the directory in which to write crashers.

  Crashers are written to the undeclared outputs directory, if it is
  available. Otherwise a temporary directory is created.
  """
  if 'TEST_UNDECLARED_OUTPUTS_DIR' in os.environ:
    crasher_dir = os.path.join(os.environ['TEST_UNDECLARED_OUTPUTS_DIR'],
                               'crashers')
    if not os.path.exists(crasher_dir):
      os.mkdir(crasher_dir)
    return crasher_dir
  return None


class RunFuzzTest(parameterized.TestCase):

  def setUp(self):
    super(RunFuzzTest, self).setUp()
    self._crasher_dir = _get_crasher_dir()

  def _create_tempdir(self) -> str:
    # Don't cleanup temporary directory if test fails.
    return self.create_tempdir(
        cleanup=test_base.TempFileCleanup.SUCCESS).full_path

  def _get_ast_options(self) -> ast_generator.AstGeneratorOptions:
    return ast_generator.AstGeneratorOptions(
        generate_proc=_GENERATE_PROC.value,
        max_width_bits_types=128 if _WIDE.value else 64,
    )

  def _get_sample_options(self) -> sample.SampleOptions:
    return sample.SampleOptions(
        input_is_dslx=True,
        ir_converter_args=['--top=main'],
        calls_per_sample=0 if _GENERATE_PROC.value else _CALLS_PER_SAMPLE,
        convert_to_ir=True,
        optimize_ir=True,
        proc_ticks=_PROC_TICKS if _GENERATE_PROC.value else 0,
        codegen=False,
        simulate=False)

  def _run_fuzz(self, seed: int) -> sample.Sample:
    return run_fuzz.generate_sample_and_run(
        ast_generator.ValueGenerator(seed),
        self._get_ast_options(),
        self._get_sample_options(),
        run_dir=self._create_tempdir(),
        crasher_dir=self._crasher_dir)

  def test_repeatable_within_process(self):
    sample0 = self._run_fuzz(7)
    sample1 = self._run_fuzz(7)
    self.assertEqual(sample0, sample1)

  def test_different_seeds_produces_different_samples(self):
    sample0 = self._run_fuzz(10)
    sample1 = self._run_fuzz(11)
    self.assertNotEqual(sample0, sample1)

  def test_sequential_samples_are_different(self):
    rng = ast_generator.ValueGenerator(42)
    sample0 = run_fuzz.generate_sample_and_run(
        rng,
        self._get_ast_options(),
        self._get_sample_options(),
        run_dir=self._create_tempdir(),
        crasher_dir=self._crasher_dir)
    sample1 = run_fuzz.generate_sample_and_run(
        rng,
        self._get_ast_options(),
        self._get_sample_options(),
        run_dir=self._create_tempdir(),
        crasher_dir=self._crasher_dir)
    self.assertNotEqual(sample0, sample1)

  @parameterized.named_parameters(*tuple(
      dict(testcase_name='seed_{}'.format(x), seed=x) for x in range(40)))
  def test_first_n_seeds(self, seed):
    for i in range(_SAMPLE_COUNT):
      self._run_fuzz(_SAMPLE_COUNT * seed + i)

if __name__ == '__main__':
  test_base.main()
