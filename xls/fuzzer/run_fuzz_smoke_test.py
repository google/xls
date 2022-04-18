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

"""Smoke test of a few samples in a single shard for the fuzzer."""

from xls.fuzzer import run_fuzz
from xls.fuzzer.python import cpp_ast_generator as ast_generator
from absl.testing import absltest


class RunFuzzSmokeTest(absltest.TestCase):

  KWARGS = {
      'calls_per_sample': 4,
      'save_temps': False,
      'sample_count': 32,
  }

  def _get_options(self, codegen: bool) -> ast_generator.AstGeneratorOptions:
    # TODO(https://github.com/google/xls/issues/469) Can't emit gate operation
    # with the current fuzzer codegen strategy.
    return ast_generator.AstGeneratorOptions(
        disallow_divide=True, emit_gate=not codegen)

  def test_codegen(self):
    kwargs = dict(self.KWARGS)
    kwargs['codegen'] = True
    run_fuzz.run_fuzz(
        ast_generator.RngState(0), self._get_options(kwargs['codegen']),
        **kwargs)

  def test_no_codegen(self):
    kwargs = dict(self.KWARGS)
    kwargs['codegen'] = False
    run_fuzz.run_fuzz(
        ast_generator.RngState(0), self._get_options(kwargs['codegen']),
        **kwargs)


if __name__ == '__main__':
  absltest.main()
