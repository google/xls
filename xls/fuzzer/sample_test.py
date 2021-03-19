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

import json
import textwrap

from xls.fuzzer.python import cpp_sample as sample
from absl.testing import absltest
from xls.dslx.python.interp_value import Value


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
    json_str = sample.SampleOptions(
        codegen=True,
        codegen_args=('--generator=pipeline', '--pipeline_stages=2'),
        simulate=True,
        simulator='iverilog').to_json()
    sample_json = json.loads(json_str)
    self.assertEqual(sample_json['input_is_dslx'], True)
    self.assertEqual(sample_json['codegen'], True)
    self.assertSequenceEqual(sample_json['codegen_args'],
                             ('--generator=pipeline', '--pipeline_stages=2'))
    self.assertEqual(sample_json['simulator'], 'iverilog')

  def test_options_from_json(self):
    json_text = (
        '{"codegen": true, "codegen_args": ["--generator=pipeline", '
        '"--pipeline_stages=2"], "convert_to_ir": true, "input_is_dslx": true,'
        ' "optimize_ir": true, "simulate": false, "simulator": null, '
        '"use_jit": true, "use_system_verilog": true}')
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

  def test_options_from_json_empty(self):
    json_text = '{}'
    want = sample.SampleOptions()
    got = sample.SampleOptions.from_json(json_text)
    self.assertEqual(got, want)
    want_text = (
        '{"codegen": false, "codegen_args": null, "convert_to_ir": true, '
        '"input_is_dslx": true, "optimize_ir": true, "simulate": false, '
        '"simulator": null, "use_jit": true, "use_system_verilog": true}')
    self.assertEqual(got.to_json(), want_text)

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
    crasher = s.to_crasher()
    self.assertTrue(
        crasher.startswith('// Copyright'),
        msg=f'Crasher does not start with copyright:\n{crasher}')
    self.assertIn(
        textwrap.dedent("""\
        // options: {"codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=2"], "convert_to_ir": true, "input_is_dslx": true, "optimize_ir": true, "simulate": true, "simulator": "goat simulator", "use_jit": true, "use_system_verilog": true}
        // args: bits[8]:0x2a; bits[8]:0xb
        // args: bits[8]:0x2c; bits[8]:0x63
        fn main(x: u8, y: u8) -> u8 {
        x + y
        }
        """), crasher)

  def test_to_crasher_with_error_message(self):
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
    crasher = s.to_crasher('oh no\nI crashed\n')
    self.assertTrue(
        crasher.startswith('// Copyright'),
        msg=f'Crasher does not start with copyright:\n{crasher}')
    # Split D.N.S. string to avoid triggering presubmit checks.
    self.assertIn('Issue: DO NOT ' + 'SUBMIT Insert link to GitHub issue here.',
                  crasher)
    self.assertIn(
        textwrap.dedent("""\
        // Exception:
        // oh no
        // I crashed"""), crasher)
    self.assertIn(
        textwrap.dedent("""\
        // options: {"codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=2"], "convert_to_ir": true, "input_is_dslx": true, "optimize_ir": true, "simulate": true, "simulator": "goat simulator", "use_jit": true, "use_system_verilog": true}
        // args: bits[8]:0x2a; bits[8]:0xb
        // args: bits[8]:0x2c; bits[8]:0x63
        fn main(x: u8, y: u8) -> u8 {
        x + y
        }
        """), crasher)

  def test_from_ir_crasher_with_codegen(self):
    crasher = textwrap.dedent("""\
    // options: {"codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=2"], "convert_to_ir": false, "input_is_dslx": false, "optimize_ir": true, "simulate": true, "simulator": "goat simulator", "use_system_verilog": false}
    // args: bits[8]:0x2a; bits[8]:0xb
    // args: bits[8]:0x2c; bits[8]:0x63
    package foo
    fn bar(x: bits[16], y: bits[16) -> bits[16] {
      ret add.1: bits[16] = add(x, y)
    }
    """)
    got = sample.Sample.from_crasher(crasher)
    want = sample.Sample(
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
            'bits[8]:0x2a; bits[8]:0xb\nbits[8]:0x2c; bits[8]:0x63'))
    self.assertMultiLineEqual(want.to_crasher(), got.to_crasher())
    self.assertEqual(got, want)

  def test_from_crasher_without_codegen(self):
    crasher = textwrap.dedent("""\
      // options: {"input_is_dslx": false, "convert_to_ir": false, "optimize_ir": true, "codegen": true, "codegen_args": null, "simulate": false, "simulator": null, "use_system_verilog": true}
      // args: bits[8]:0x2a; bits[8]:0xb
      // args: bits[8]:0x2c; bits[8]:0x63
      package foo
      fn bar(x: bits[16], y: bits[16) -> bits[16] {
        ret add.1: bits[16] = add(x, y)
      }""")
    got = sample.Sample.from_crasher(crasher)
    want = sample.Sample(
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
            'bits[8]:0x2a; bits[8]:0xb\nbits[8]:0x2c; bits[8]:0x63'))
    self.assertEqual(got, want)


if __name__ == '__main__':
  absltest.main()
