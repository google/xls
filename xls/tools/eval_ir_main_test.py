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

import subprocess

from xls.common import runfiles
from xls.common import test_base
from absl.testing import absltest

EVAL_IR_MAIN_PATH = runfiles.get_path('xls/tools/eval_ir_main')

ADD_IR = """package foo

fn foo(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
"""

TUPLE_IR = """package foo

fn foo(x: (bits[8], bits[32])) -> ((bits[8], bits[32])) {
  ret tuple.1: ((bits[8], bits[32])) = tuple(x)
}
"""


class EvalMainTest(absltest.TestCase):

  def test_one_input_jit(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH, '--input=bits[32]:0x42; bits[32]:0x123',
        '--use_llvm_jit=true', ir_file.full_path
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x165')

  def test_one_input_nojit(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH, '--input=bits[32]:0x42; bits[32]:0x123',
        '--use_llvm_jit=true', ir_file.full_path
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x165')

  def test_input_missing_arg(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run(
        [EVAL_IR_MAIN_PATH, '--input=bits[32]:0x42', ir_file.full_path],
        stderr=subprocess.PIPE,
        check=False)
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('Arg list has the wrong size: 1 vs expected 2',
                  comp.stderr.decode('utf-8'))

  def test_one_input_with_expected(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH, '--input=bits[32]:0x42; bits[32]:0x123',
        '--expected=bits[32]:0x165', ir_file.full_path
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x165')

  def test_one_input_with_failed_expected(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run([
        EVAL_IR_MAIN_PATH, '--input=bits[32]:0x42; bits[32]:0x123',
        '--expected=bits[32]:0x123', ir_file.full_path
    ],
                          stderr=subprocess.PIPE,
                          check=False)
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('Miscompare for input "bits[32]:0x42; bits[32]:0x123"',
                  comp.stderr.decode('utf-8'))

  def test_input_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(
        content='\n'.join(('bits[32]:0x42; bits[32]:0x123',
                           'bits[32]:0x10; bits[32]:0xf0f')))
    results = subprocess.check_output([
        EVAL_IR_MAIN_PATH, '--input_file=' + input_file.full_path,
        ir_file.full_path
    ])
    self.assertSequenceEqual(('bits[32]:0x165', 'bits[32]:0xf1f'),
                             results.decode('utf-8').strip().split('\n'))

  def test_input_file_extra_whitespace(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    # Empty lines and extra whitespace in the arg file should be ignored.
    input_file = self.create_tempfile(
        content='\n'.join(('bits[32]:0x42; bits[32]:0x123', '',
                           'bits[32]:0x10; bits[32]:0xf0f', '')))
    results = subprocess.check_output([
        EVAL_IR_MAIN_PATH, '--input_file=' + input_file.full_path,
        ir_file.full_path
    ])
    self.assertSequenceEqual(('bits[32]:0x165', 'bits[32]:0xf1f'),
                             results.decode('utf-8').strip().split('\n'))

  def test_input_file_with_expected_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(
        content='\n'.join(('bits[32]:0x42; bits[32]:0x123',
                           'bits[32]:0x10; bits[32]:0xf0f')))
    expected_file = self.create_tempfile(content='\n'.join(('bits[32]:0x165',
                                                            'bits[32]:0xf1f')))
    results = subprocess.check_output([
        EVAL_IR_MAIN_PATH, '--input_file=' + input_file.full_path,
        '--expected_file=' + expected_file.full_path, ir_file.full_path
    ])
    self.assertSequenceEqual(('bits[32]:0x165', 'bits[32]:0xf1f'),
                             results.decode('utf-8').strip().split('\n'))

  def test_input_file_with_failed_expected_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(
        content='\n'.join(('bits[32]:0x42; bits[32]:0x123',
                           'bits[32]:0x10; bits[32]:0x00')))
    expected_file = self.create_tempfile(content='\n'.join(('bits[32]:0x165',
                                                            'bits[32]:0xf1f')))
    comp = subprocess.run([
        EVAL_IR_MAIN_PATH, '--input_file=' + input_file.full_path,
        '--expected_file=' + expected_file.full_path, ir_file.full_path
    ],
                          stderr=subprocess.PIPE,
                          check=False)
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('Miscompare for input "bits[32]:0x10; bits[32]:0x0"',
                  comp.stderr.decode('utf-8'))

  def test_empty_input_file(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    input_file = self.create_tempfile(content='')
    results = subprocess.check_output([
        EVAL_IR_MAIN_PATH, '--input_file=' + input_file.full_path,
        ir_file.full_path
    ])
    self.assertEqual(results.decode('utf-8'), '')

  def test_tuple_in_out(self):
    ir_file = self.create_tempfile(content=TUPLE_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH, '--input=(bits[8]:0x42, bits[32]:0x123)',
        ir_file.full_path
    ])
    self.assertEqual(
        result.decode('utf-8').strip(), '((bits[8]:0x42, bits[32]:0x123))')

  def test_random_inputs(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output(
        [EVAL_IR_MAIN_PATH, '--random_inputs=42', ir_file.full_path])
    # There should be 42 results.
    self.assertLen(result.decode('utf-8').strip().split('\n'), 42)
    # And with overwhelming probability they should all be different.
    self.assertLen(set(result.decode('utf-8').strip().split('\n')), 42)

  def test_jit_result_injection(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    result = subprocess.check_output([
        EVAL_IR_MAIN_PATH, '--input=bits[32]:0x42; bits[32]:0x123',
        '--test_only_inject_jit_result=bits[32]:0x22', '--use_llvm_jit=true',
        ir_file.full_path
    ])
    self.assertEqual(result.decode('utf-8').strip(), 'bits[32]:0x22')

  def test_test_llvm_jit_no_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run([
        EVAL_IR_MAIN_PATH, '--input=bits[32]:0x42; bits[32]:0x123',
        '--test_llvm_jit', ir_file.full_path
    ],
                          check=False)
    self.assertEqual(comp.returncode, 0)

  def test_test_llvm_jit_mismatch(self):
    ir_file = self.create_tempfile(content=ADD_IR)
    comp = subprocess.run([
        EVAL_IR_MAIN_PATH, '--input=bits[32]:0x42; bits[32]:0x123',
        '--test_llvm_jit', '--test_only_inject_jit_result=bits[32]:0x22',
        ir_file.full_path
    ],
                          stderr=subprocess.PIPE,
                          check=False)
    self.assertNotEqual(comp.returncode, 0)
    self.assertIn('Miscompare for input "bits[32]:0x42; bits[32]:0x123"',
                  comp.stderr.decode('utf-8'))


if __name__ == '__main__':
  test_base.main()
