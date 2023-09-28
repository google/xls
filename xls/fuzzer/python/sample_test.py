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

from absl.testing import absltest
from xls.dslx.python.interp_value import interp_value_from_ir_string
from xls.dslx.python.interp_value import Value
from xls.fuzzer.python import cpp_sample as sample


class SampleTest(absltest.TestCase):

  def test_parse_args(self):
    self.assertEqual(sample.parse_args(''), tuple())
    self.assertEqual(
        sample.parse_args('bits[8]:42'),
        (interp_value_from_ir_string('bits[8]:42'),))
    self.assertEqual(
        sample.parse_args('bits[8]:42; bits[16]:1234'),
        (interp_value_from_ir_string('bits[8]:42'),
         interp_value_from_ir_string('bits[16]:1234')))
    self.assertEqual(
        sample.parse_args(
            'bits[8]:42; (bits[8]:0x42, (bits[16]:0x33, bits[8]:0x44))'),
        (interp_value_from_ir_string('bits[8]:42'),
         Value.make_tuple(
             (interp_value_from_ir_string('bits[8]:0x42'),
              Value.make_tuple(
                  (interp_value_from_ir_string('bits[16]:0x33'),
                   interp_value_from_ir_string('bits[8]:0x44')))))))
    self.assertEqual(
        sample.parse_args('[bits[8]:0x42, bits[8]:0x43, bits[8]:0x44]'),
        (Value.make_array(
            tuple(
                interp_value_from_ir_string(f'bits[8]:{v}')
                for v in (0x42, 0x43, 0x44))),))

  def test_function_sample_to_crasher(self):
    s = sample.Sample(
        'fn main(x: u8, y: u8) -> u8 {\nx + y\n}',
        sample.SampleOptions(
            input_is_dslx=True,
            ir_converter_args=['--top=main'],
            calls_per_sample=42,
            codegen=True,
            codegen_args=('--generator=pipeline', '--pipeline_stages=2'),
            simulate=True,
            simulator='goat simulator',
            use_system_verilog=True),
        sample.parse_args_batch(
            'bits[8]:42; bits[8]:11\nbits[8]:44; bits[8]:99'))
    crasher = s.to_crasher('oh no\nI crashed\n')
    self.assertTrue(
        crasher.startswith('// Copyright'),
        msg=f'Crasher does not start with copyright:\n{crasher}')
    self.assertIn(R'oh no\nI crashed', crasher)
    self.assertIn(R'bits[8]:0x2a; bits[8]:0xb', crasher)
    self.assertIn(R'bits[8]:0x2c; bits[8]:0x63', crasher)

  def test_to_crasher_with_error_message(self):
    s = sample.Sample(
        'fn main(x: u8, y: u8) -> u8 {\nx + y\n}',
        sample.SampleOptions(
            input_is_dslx=True,
            ir_converter_args=['--top=main'],
            calls_per_sample=42,
            codegen=True,
            codegen_args=('--generator=pipeline', '--pipeline_stages=2'),
            simulate=True,
            simulator='goat simulator',
            use_system_verilog=True),
        sample.parse_args_batch(
            'bits[8]:42; bits[8]:11\nbits[8]:44; bits[8]:99'))
    crasher = s.to_crasher('oh no\nI crashed\n')
    self.assertTrue(
        crasher.startswith('// Copyright'),
        msg=f'Crasher does not start with copyright:\n{crasher}')
    # Split D.N.S. string to avoid triggering presubmit checks.
    self.assertIn(
        'DO NOT ' + 'SUBMIT Insert link to GitHub issue here.', crasher
    )
    self.assertIn(R'oh no\nI crashed', crasher)
    self.assertIn(R'bits[8]:0x2a; bits[8]:0xb', crasher)
    self.assertIn(R'bits[8]:0x2c; bits[8]:0x63', crasher)


if __name__ == '__main__':
  absltest.main()
