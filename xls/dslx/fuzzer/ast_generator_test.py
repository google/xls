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

"""Tests for xls.dslx.fuzzer.ast_generator."""

import random

from xls.dslx import fakefs_test_util
from xls.dslx import parser_helpers
from xls.dslx import typecheck
from xls.dslx.fuzzer import ast_generator
from xls.dslx.span import PositionalError
from absl.testing import absltest


class AstGeneratorTest(absltest.TestCase):

  def test_generates_valid_functions(self):
    g = ast_generator.AstGenerator(
        random.Random(0), ast_generator.AstGeneratorOptions())
    for i in range(32):
      print('Generating sample', i)
      _, m = g.generate_function_in_module('main', 'test')
      text = str(m)
      filename = '/fake/test_sample.x'
      with fakefs_test_util.scoped_fakefs(filename, text):
        try:
          module = parser_helpers.parse_text(
              text, name='test_sample', print_on_error=True, filename=filename)
          typecheck.check_module(module, f_import=None)
        except PositionalError as e:
          parser_helpers.pprint_positional_error(e)
          raise

  def _test_repeatable(self, seed: int) -> None:

    def generate_one(g):
      _, m = g.generate_function_in_module('main', 'test')
      text = str(m)
      return text

    count_to_generate = 128
    g = ast_generator.AstGenerator(
        random.Random(seed), ast_generator.AstGeneratorOptions())
    fs = [generate_one(g) for _ in range(count_to_generate)]
    g.reset(seed=seed)
    gs = [generate_one(g) for _ in range(count_to_generate)]
    self.assertEqual(fs, gs)

    # Create a fresh generator as well.
    g = ast_generator.AstGenerator(
        random.Random(seed), ast_generator.AstGeneratorOptions())
    hs = [generate_one(g) for _ in range(count_to_generate)]
    self.assertEqual(gs, hs)

  def test_repeatable_seed0(self):
    self.maxDiff = None  # pylint: disable=invalid-name
    self._test_repeatable(seed=0)

  def test_repeatable_seed1(self):
    self.maxDiff = None  # pylint: disable=invalid-name
    self._test_repeatable(seed=1)


if __name__ == '__main__':
  absltest.main()
