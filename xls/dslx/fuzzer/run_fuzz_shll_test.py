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

"""Tests for shll-specific fuzzing."""

import random

from absl import flags

from absl.testing import absltest
from absl.testing import parameterized
from xls.common import runfiles
from xls.dslx import ast
from xls.dslx.fuzzer import ast_generator
from xls.dslx.fuzzer import run_fuzz

flags.DEFINE_boolean('update_golden', False,
                     'Whether to update golden reference files.')
FLAGS = flags.FLAGS


class RunFuzzShllTest(parameterized.TestCase):

  KWARGS = {
      'calls_per_sample': 4,
      'save_temps': False,
      'sample_count': 4,
      'return_samples': True,
      'codegen': False,
  }
  GOLDEN_REFERENCE_FMT = 'xls/dslx/fuzzer/testdata/run_fuzz_shll_test.seed_{seed}_sample_{sample}.x'
  SEED_TO_CHECK_LIMIT = 2
  SAMPLE_TO_CHECK_LIMIT = 1

  @parameterized.named_parameters(*tuple(
      dict(testcase_name='seed_{}'.format(x), seed=x) for x in range(50)))
  def test_first_n_seeds(self, seed):
    if FLAGS.update_golden and seed >= self.SEED_TO_CHECK_LIMIT:
      # Skip running unnecessary tests if updating golden because the test is
      # slow and runs unsharded.
      return
    rng = random.Random(seed)
    samples = run_fuzz.run_fuzz(
        rng,
        ast_generator.AstGeneratorOptions(
            blacklist_divide=True, binop_whitelist=[ast.Binop.SHLL]),
        **self.KWARGS)
    for i in range(self.KWARGS['sample_count']):
      if seed < self.SEED_TO_CHECK_LIMIT and i < self.SAMPLE_TO_CHECK_LIMIT:
        path = self.GOLDEN_REFERENCE_FMT.format(seed=seed, sample=i)
        if FLAGS.update_golden:
          with open(path, 'w') as f:
            f.write(samples[i].input_text)
        else:
          # rstrip to avoid miscompares from trailing newline at EOF.
          expected = runfiles.get_contents_as_text(path).rstrip()
          self.assertMultiLineEqual(expected, samples[i].input_text)


if __name__ == '__main__':
  absltest.main()
