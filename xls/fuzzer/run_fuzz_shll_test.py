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

"""Tests for shll-specific fuzzing."""

from absl.testing import absltest
from absl.testing import parameterized
from xls.dslx.python import ast
from xls.fuzzer import run_fuzz
from xls.fuzzer.python import cpp_ast_generator as ast_generator


class RunFuzzShllTest(parameterized.TestCase):

  KWARGS = {
      'calls_per_sample': 4,
      'save_temps': False,
      'sample_count': 4,
      'return_samples': True,
      'codegen': False,
  }
  SEED_TO_CHECK_LIMIT = 2

  @parameterized.named_parameters(*tuple(
      dict(testcase_name='seed_{}'.format(x), seed=x) for x in range(50)))
  def test_first_n_seeds(self, seed):
    rng = ast_generator.RngState(seed)
    run_fuzz.run_fuzz(
        rng,
        ast_generator.AstGeneratorOptions(
            disallow_divide=True, binop_allowlist=[ast.BinopKind.SHLL]),
        **self.KWARGS)


if __name__ == '__main__':
  absltest.main()
