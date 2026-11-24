#
# Copyright 2026 The XLS Authors
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

# pylint: disable=g-redundant-test-module-docstring

"""Tests for eval_dslx_main."""

import subprocess

from absl.testing import parameterized
from xls.common import runfiles
from xls.common import test_base

EVAL_DSLX_MAIN_PATH = runfiles.get_path('xls/tools/eval_dslx_main')


class EvalDslxMainTest(parameterized.TestCase):

  def _eval(self, dslx, inputs):
    source = self.create_tempfile('test.x', content=dslx)
    return subprocess.run(
        [
            EVAL_DSLX_MAIN_PATH,
            source.full_path,
            '--entry=main',
            f'--input={inputs}',
        ],
        capture_output=True,
        text=True,
        check=False,
    )

  @parameterized.parameters(('s8', 's8:-64'), ('u8', 'u8:64'))
  def test_right_shift_uses_parameter_signedness(self, param_type, expected):
    result = self._eval(
        f'fn main(x: {param_type}) -> {param_type} {{ x >> u32:1 }}',
        'bits[8]:0x80',
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(result.stdout.strip(), f'Result: {expected}')

  def test_nested_and_multiple_arguments(self):
    result = self._eval(
        'fn main(x: (s8[2], u8), y: s8) -> ((s8[2], u8), s8) { (x, y) }',
        '([bits[8]:0x80, bits[8]:0xff], bits[8]:0x80); bits[8]:0xfe',
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    self.assertEqual(
        result.stdout.strip(), 'Result: (([s8:-128, s8:-1], u8:128), s8:-2)'
    )

  @parameterized.named_parameters(
      ('wrong_element_width', '[bits[7]:1, bits[7]:2]'),
      ('wrong_array_length', '[bits[8]:1]'),
      ('bits_instead_of_array', 'bits[8]:1'),
      ('tuple_instead_of_array', '(bits[8]:1, bits[8]:2)'),
      ('too_many_arguments', '[bits[8]:1, bits[8]:2]; bits[8]:3'),
  )
  def test_rejects_mismatched_arguments(self, inputs):
    result = self._eval('fn main(x: s8[2]) -> s8[2] { x }', inputs)
    self.assertEqual(result.returncode, 1, result.stderr)
    self.assertIn('INVALID_ARGUMENT', result.stderr)

  def test_parametric_entry_requires_wrapper(self):
    result = self._eval(
        'fn main<N: u32>(x: sN[N]) -> sN[N] { x }', 'bits[8]:0x80'
    )
    self.assertEqual(result.returncode, 1, result.stderr)
    self.assertIn('Use a non-parametric wrapper', result.stderr)


if __name__ == '__main__':
  test_base.main()
