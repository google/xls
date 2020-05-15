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
#
# Lint as: python3

import json
import textwrap

from xls.dslx.fuzzer import sample
from absl.testing import absltest
from xls.dslx.interpreter.value import Value


class SampleTest(absltest.TestCase):

  def test_parse_args(self):
    self.assertEqual(sample.parse_args(''), tuple())
    self.assertEqual(
        sample.parse_args('bits[8]:42'), (Value.make_ubits(8, 42),))
    self.assertEqual(
        sample.parse_args('bits[8]:42; bits[16]:1234'),
        (Value.make_ubits(8, 42), Value.make_ubits(16, 1234)))
    self.assertEqual(
        sample.parse_args(
            'bits[8]:42; (bits[8]:0x42, (bits[16]:0x33, bits[8]:0x44))'),
        (Value.make_ubits(8, 42),
         Value.make_tuple(
             (Value.make_ubits(8, 0x42),
              Value.make_tuple(
                  (Value.make_ubits(16, 0x33), Value.make_ubits(8, 0x44)))))))
    self.assertEqual(
        sample.parse_args('[bits[8]:0x42, bits[8]:0x43, bits[8]:0x44]'),
        (Value.make_array(
            tuple(Value.make_ubits(8, v) for v in (0x42, 0x43, 0x44))),))

  def test_options_to_json(self):
    sample_json = json.loads(
        sample.SampleOptions(
            codegen=True,
            codegen_args=('--generator=pipeline', '--pipeline_stages=2'),
            simulate=True,
            simulator='iverilog').to_json())
    self.assertEqual(sample_json['input_is_dslx'], True)
    self.assertEqual(sample_json['codegen'], True)
    self.assertSequenceEqual(sample_json['codegen_args'],
                             ('--generator=pipeline', '--pipeline_stages=2'))
    self.assertEqual(sample_json['simulator'], 'iverilog')

  def test_options_from_json(self):
    json_text = ('{"input_is_dslx": true, "convert_to_ir": true, "optimize_ir":'
                 ' true, "use_jit": true, "codegen": true, "codegen_args": '
                 '["--generator=pipeline", "--pipeline_stages=2"], "simulate": '
                 'false, "simulator": null, "use_system_verilog": true}')
    expected_object = sample.SampleOptions(
        input_is_dslx=True,
        convert_to_ir=True,
        optimize_ir=True,
        use_jit=True,
        codegen=True,
        codegen_args=('--generator=pipeline', '--pipeline_stages=2'),
        simulate=False,
        simulator=None,
        use_system_verilog=True)
    self.assertEqual(sample.SampleOptions.from_json(json_text), expected_object)
    self.assertEqual(
        sample.SampleOptions.from_json(json_text).to_json(), json_text)

  def test_to_crasher(self):
    s = sample.Sample(
        'fn main(x: u8, y: u8) -> u8 {\nx + y\n}',
        sample.SampleOptions(
            input_is_dslx=True,
            codegen=True,
            codegen_args=('--generator=pipeline', '--pipeline_stages=2'),
            simulate=True,
            simulator='goat simulator',
            use_system_verilog=True),
        sample.parse_args_batch(
            'bits[8]:42; bits[8]:11\nbits[8]:44; bits[8]:99'))
    self.assertEqual(
        s.to_crasher(),
        textwrap.dedent("""\
        // options: {"input_is_dslx": true, "convert_to_ir": true, "optimize_ir": true, "use_jit": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=2"], "simulate": true, "simulator": "goat simulator", "use_system_verilog": true}
        // args: bits[8]:0x2a; bits[8]:0xb
        // args: bits[8]:0x2c; bits[8]:0x63
        fn main(x: u8, y: u8) -> u8 {
        x + y
        }
        """))

  def test_from_ir_crasher_with_codegen(self):
    crasher = textwrap.dedent("""\
    // options: {"input_is_dslx": false, "convert_to_ir": false, "optimize_ir": true, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=2"], "simulate": true, "simulator": "goat simulator", "use_system_verilog": false}
    // args: bits[8]:0x2a; bits[8]:0xb
    // args: bits[8]:0x2c; bits[8]:0x63
    package foo
    fn bar(x: bits[16], y: bits[16) -> bits[16] {
      ret add.1: bits[16] = add(x, y)
    }
    """)
    self.assertEqual(
        sample.Sample.from_crasher(crasher),
        sample.Sample(
            textwrap.dedent("""\
          package foo
          fn bar(x: bits[16], y: bits[16) -> bits[16] {
            ret add.1: bits[16] = add(x, y)
          }"""),
            sample.SampleOptions(
                input_is_dslx=False,
                convert_to_ir=False,
                optimize_ir=True,
                codegen=True,
                codegen_args=('--generator=pipeline', '--pipeline_stages=2'),
                simulate=True,
                simulator='goat simulator',
                use_system_verilog=False),
            sample.parse_args_batch(
                'bits[8]:0x2a; bits[8]:0xb\nbits[8]:0x2c; bits[8]:0x63')))

  def test_from_crasher_without_codegen(self):
    crasher = textwrap.dedent("""\
      // options: {"input_is_dslx": false, "convert_to_ir": false, "optimize_ir": true, "codegen": true, "codegen_args": null, "simulate": false, "simulator": null, "use_system_verilog": true}
      // args: bits[8]:0x2a; bits[8]:0xb
      // args: bits[8]:0x2c; bits[8]:0x63
      package foo
      fn bar(x: bits[16], y: bits[16) -> bits[16] {
        ret add.1: bits[16] = add(x, y)
      }""")
    self.assertEqual(
        sample.Sample.from_crasher(crasher),
        sample.Sample(
            textwrap.dedent("""\
            package foo
            fn bar(x: bits[16], y: bits[16) -> bits[16] {
              ret add.1: bits[16] = add(x, y)
            }"""),
            sample.SampleOptions(
                input_is_dslx=False,
                convert_to_ir=False,
                codegen=True,
                codegen_args=None,
                simulate=False,
                simulator=None,
                use_system_verilog=True),
            sample.parse_args_batch(
                'bits[8]:0x2a; bits[8]:0xb\nbits[8]:0x2c; bits[8]:0x63')))


if __name__ == '__main__':
  absltest.main()
