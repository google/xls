# Lint as: python3
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
"""Tests for op module generation."""

import sys

from absl.testing import absltest
from xls.common import runfiles
from xls.common.python import init_xls
from xls.delay_model import op_module_generator as opgen


def setUpModule():
  # This is required so that module initializers are called including those
  # which register delay models.
  init_xls.init_xls(sys.argv)


class OpModuleGeneratorTest(absltest.TestCase):

  def test_8_bit_add(self):
    self.assertEqual(
        """package add_characterization

fn main(op0: bits[8], op1: bits[8]) -> bits[8] {
  ret add.1: bits[8] = add(op0, op1)
}""",
        opgen.generate_ir_package(
            'add', output_type='bits[8]', operand_types=('bits[8]', 'bits[8]')))

  def test_mixed_width_mul(self):
    self.assertEqual(
        """package umul_characterization

fn main(op0: bits[27], op1: bits[5]) -> bits[42] {
  ret umul.1: bits[42] = umul(op0, op1)
}""",
        opgen.generate_ir_package(
            'umul', output_type='bits[42]', operand_types=('bits[27]', 'bits[5]')))

  def test_array_update(self):
    self.assertEqual(
        """package array_update_characterization

fn main(op0: bits[17][42], op1: bits[3], op2: bits[17]) -> bits[17][42] {
  ret array_update.1: bits[17][42] = array_update(op0, op1, op2)
}""",
        opgen.generate_ir_package(
            'array_update',
            output_type='bits[17][42]',
            operand_types=('bits[17][42]', 'bits[3]', 'bits[17]')))

  def test_sign_extend(self):
    self.assertEqual(
        """package sign_ext_characterization

fn main(op0: bits[16]) -> bits[32] {
  ret sign_ext.1: bits[32] = sign_ext(op0, new_bit_count=32)
}""",
        opgen.generate_ir_package(
            'sign_ext',
            output_type='bits[32]',
            operand_types=('bits[16]',),
            attributes=(('new_bit_count', '32'),)))

  def test_add_with_literal_operand(self):
    self.assertEqual(
        """package add_characterization

fn main(op0: bits[32]) -> bits[32] {
  literal.1: bits[32] = literal(value=0xd82c07cd)
  ret add.2: bits[32] = add(op0, literal.1)
}""",
        opgen.generate_ir_package(
            'add',
            output_type='bits[32]',
            operand_types=('bits[32]', 'bits[32]'),
            literal_operand=1))

  def test_array_update_with_literal_operand(self):
    self.assertEqual(
        """package array_update_characterization

fn main(op1: bits[3], op2: bits[17]) -> bits[17][4] {
  literal.1: bits[17][4] = literal(value=[0x1b058, 0xc53e, 0x18412, 0x1c7ce])
  ret array_update.2: bits[17][4] = array_update(literal.1, op1, op2)
}""",
        opgen.generate_ir_package(
            'array_update',
            output_type='bits[17][4]',
            operand_types=('bits[17][4]', 'bits[3]', 'bits[17]'),
            literal_operand=0))

  def test_invalid_ir(self):
    with self.assertRaisesRegex(Exception, 'does not match expected type'):
      opgen.generate_ir_package(
          'and',
          output_type='bits[17]',
          operand_types=('bits[123]', 'bits[17]'))

  def test_8_bit_add_verilog(self):
    verilog_text = opgen.generate_verilog_module(
        'add_module',
        'add',
        output_type='bits[8]',
        operand_types=('bits[8]', 'bits[8]')).verilog_text
    self.assertIn('module add_module', verilog_text)
    self.assertIn('p0_op0 + p0_op1', verilog_text)

  def test_parallel_add_verilog(self):
    modules = (
        opgen.generate_verilog_module(
            'add8_module',
            'add',
            output_type='bits[8]',
            operand_types=('bits[8]', 'bits[8]')),
        opgen.generate_verilog_module(
            'add16_module',
            'add',
            output_type='bits[16]',
            operand_types=('bits[16]', 'bits[16]')),
        opgen.generate_verilog_module(
            'add24_module',
            'add',
            output_type='bits[24]',
            operand_types=('bits[24]', 'bits[24]')),
    )
    parallel_module = opgen.generate_parallel_module(modules, 'foo')
    self.assertEqual(
        parallel_module,
        runfiles.get_contents_as_text(
            'xls/delay_model/testdata/parallel_op_module.vtxt'))


if __name__ == '__main__':
  absltest.main()
