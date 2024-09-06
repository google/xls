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
"""Tests for op module generation."""

from absl.testing import absltest
from xls.common import runfiles
from xls.estimators import op_module_generator as opgen


class OpModuleGeneratorTest(absltest.TestCase):

  def test_8_bit_add(self):
    self.assertEqual(
        """package add_characterization

top fn main(op0: bits[8], op1: bits[8]) -> bits[8] {
  ret result: bits[8] = add(op0, op1)
}""",
        opgen.generate_ir_package(
            'add', output_type='bits[8]', operand_types=('bits[8]', 'bits[8]')
        ),
    )

  def test_mixed_width_mul(self):
    self.assertEqual(
        """package umul_characterization

top fn main(op0: bits[27], op1: bits[5]) -> bits[42] {
  ret result: bits[42] = umul(op0, op1)
}""",
        opgen.generate_ir_package(
            'umul',
            output_type='bits[42]',
            operand_types=('bits[27]', 'bits[5]'),
        ),
    )

  def test_array_update(self):
    self.assertEqual(
        """package array_update_characterization

top fn main(op0: bits[17][42], op1: bits[17], op2: bits[3]) -> bits[17][42] {
  ret result: bits[17][42] = array_update(op0, op1, indices=[op2])
}""",
        opgen.generate_ir_package(
            'array_update',
            output_type='bits[17][42]',
            operand_types=('bits[17][42]', 'bits[17]', 'bits[3]'),
        ),
    )

  def test_array_index(self):
    self.assertEqual(
        """package array_index_characterization

top fn main(op0: bits[17][42][10], op1: bits[32], op2: bits[3]) -> bits[17] {
  ret result: bits[17] = array_index(op0, indices=[op1, op2])
}""",
        opgen.generate_ir_package(
            'array_index',
            output_type='bits[17]',
            operand_types=('bits[17][42][10]', 'bits[32]', 'bits[3]'),
        ),
    )

  def test_select(self):
    self.assertEqual(
        """package sel_characterization

top fn main(op0: bits[1], op1: bits[32], op2: bits[32]) -> bits[32] {
  ret result: bits[32] = sel(op0, cases=[op1, op2])
}""",
        opgen.generate_ir_package(
            'sel',
            output_type='bits[32]',
            operand_types=('bits[1]', 'bits[32]', 'bits[32]'),
        ),
    )

  def test_select_with_default(self):
    self.assertEqual(
        """package sel_characterization

top fn main(op0: bits[2], op1: bits[32], op2: bits[32], op3: bits[32]) -> bits[32] {
  ret result: bits[32] = sel(op0, cases=[op1, op2], default=op3)
}""",
        opgen.generate_ir_package(
            'sel',
            output_type='bits[32]',
            operand_types=('bits[2]', 'bits[32]', 'bits[32]', 'bits[32]'),
        ),
    )

  def test_select_invalid_selector_type(self):
    with self.assertRaises(ValueError):
      opgen.generate_ir_package(
          'sel',
          output_type='bits[32]',
          operand_types=('bits[1][2]', 'bits[32]', 'bits[32]'),
      )

  def test_one_hot_select(self):
    self.assertEqual(
        """package one_hot_sel_characterization

top fn main(op0: bits[2], op1: bits[32], op2: bits[32]) -> bits[32] {
  ret result: bits[32] = one_hot_sel(op0, cases=[op1, op2])
}""",
        opgen.generate_ir_package(
            'one_hot_sel',
            output_type='bits[32]',
            operand_types=('bits[2]', 'bits[32]', 'bits[32]'),
        ),
    )

  def test_priority_select(self):
    self.assertEqual(
        """package priority_sel_characterization

top fn main(op0: bits[4], op1: bits[64], op2: bits[64], op3: bits[64], op4: bits[64], op5: bits[64]) -> bits[64] {
  ret result: bits[64] = priority_sel(op0, cases=[op1, op2, op3, op4], default=op5)
}""",
        opgen.generate_ir_package(
            'priority_sel',
            output_type='bits[64]',
            operand_types=(
                'bits[4]',
                'bits[64]',
                'bits[64]',
                'bits[64]',
                'bits[64]',
                'bits[64]',
            ),
        ),
    )

  def test_sign_extend(self):
    self.assertEqual(
        """package sign_ext_characterization

top fn main(op0: bits[16]) -> bits[32] {
  ret result: bits[32] = sign_ext(op0, new_bit_count=32)
}""",
        opgen.generate_ir_package(
            'sign_ext',
            output_type='bits[32]',
            operand_types=('bits[16]',),
            attributes=(('new_bit_count', '32'),),
        ),
    )

  def test_add_with_literal_operand(self):
    self.assertEqual(
        """package add_characterization

top fn main(op0: bits[32]) -> bits[32] {
  op1: bits[32] = literal(value=0xd82c07cd)
  ret result: bits[32] = add(op0, op1)
}""",
        opgen.generate_ir_package(
            'add',
            output_type='bits[32]',
            operand_types=('bits[32]', 'bits[32]'),
            literal_operand=1,
        ),
    )

  def test_array_update_with_literal_operand(self):
    self.assertEqual(
        """package array_update_characterization

top fn main(op1: bits[17], op2: bits[3]) -> bits[17][4] {
  op0: bits[17][4] = literal(value=[0x1b058, 0xc53e, 0x18412, 0x1c7ce])
  ret result: bits[17][4] = array_update(op0, op1, indices=[op2])
}""",
        opgen.generate_ir_package(
            'array_update',
            output_type='bits[17][4]',
            operand_types=('bits[17][4]', 'bits[17]', 'bits[3]'),
            literal_operand=0,
        ),
    )

  def test_invalid_ir(self):
    with self.assertRaisesRegex(Exception, 'does not match expected type'):
      opgen.generate_ir_package(
          'and', output_type='bits[17]', operand_types=('bits[123]', 'bits[17]')
      )

  def test_8_bit_add_verilog(self):
    ir_text = opgen.generate_ir_package(
        'add', output_type='bits[8]', operand_types=('bits[8]', 'bits[8]')
    )
    verilog_text = opgen.generate_verilog_module(
        'add_module', ir_text
    ).verilog_text
    self.assertIn('module add_module', verilog_text)
    self.assertIn('p0_op0 + p0_op1', verilog_text)

  def test_parallel_add_verilog(self):
    add8_ir = opgen.generate_ir_package(
        'add', output_type='bits[8]', operand_types=('bits[8]', 'bits[8]')
    )
    add16_ir = opgen.generate_ir_package(
        'add', output_type='bits[16]', operand_types=('bits[16]', 'bits[16]')
    )
    add24_ir = opgen.generate_ir_package(
        'add', output_type='bits[24]', operand_types=('bits[24]', 'bits[24]')
    )

    modules = (
        opgen.generate_verilog_module('add8_module', add8_ir),
        opgen.generate_verilog_module('add16_module', add16_ir),
        opgen.generate_verilog_module('add24_module', add24_ir),
    )
    parallel_module = opgen.generate_parallel_module(modules, 'foo')
    self.assertEqual(
        parallel_module,
        runfiles.get_contents_as_text(
            'xls/estimators/testdata/parallel_op_module.vtxt'
        ),
    )


if __name__ == '__main__':
  absltest.main()
