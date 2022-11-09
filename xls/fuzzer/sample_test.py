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
        '{"calls_per_sample": 42, "codegen": true, '
        '"codegen_args": ["--generator=pipeline", '
        '"--pipeline_stages=2"], "convert_to_ir": true, "input_is_dslx": true, '
        '"ir_converter_args": ["--top=main"], "optimize_ir": true, '
        '"proc_ticks": null, "simulate": false, "simulator": null, '
        '"timeout_seconds": 42, "top_type": 0, "use_jit": true, '
        '"use_system_verilog": true}')
    expected_object = sample.SampleOptions(
        input_is_dslx=True,
        ir_converter_args=['--top=main'],
        calls_per_sample=42,
        convert_to_ir=True,
        optimize_ir=True,
        use_jit=True,
        codegen=True,
        codegen_args=('--generator=pipeline', '--pipeline_stages=2'),
        simulate=False,
        simulator=None,
        use_system_verilog=True,
        timeout_seconds=42)
    self.assertEqual(sample.SampleOptions.from_json(json_text), expected_object)
    self.assertEqual(
        sample.SampleOptions.from_json(json_text).to_json(), json_text)

  def test_options_from_json_empty(self):
    json_text = '{}'
    want = sample.SampleOptions()
    got = sample.SampleOptions.from_json(json_text)
    self.assertEqual(got, want)
    want_text = (
        '{"calls_per_sample": 1, "codegen": false, "codegen_args": null, '
        '"convert_to_ir": true, '
        '"input_is_dslx": true, "ir_converter_args": null, '
        '"optimize_ir": true, "proc_ticks": null, '
        '"simulate": false, "simulator": null, "timeout_seconds": null, '
        '"top_type": 0, "use_jit": true, "use_system_verilog": true}')
    self.assertEqual(got.to_json(), want_text)

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
    self.assertIn(
        textwrap.dedent("""\
        // Exception:
        // oh no
        // I crashed"""), crasher)
    self.assertIn(
        textwrap.dedent("""\
        // options: {"calls_per_sample": 42, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=2"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "proc_ticks": null, "simulate": true, "simulator": "goat simulator", "timeout_seconds": null, "top_type": 0, "use_jit": true, "use_system_verilog": true}
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
    self.assertIn('Issue: DO NOT ' + 'SUBMIT Insert link to GitHub issue here.',
                  crasher)
    self.assertIn(
        textwrap.dedent("""\
        // Exception:
        // oh no
        // I crashed"""), crasher)
    self.assertIn(
        textwrap.dedent("""\
        // options: {"calls_per_sample": 42, "codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=2"], "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": true, "proc_ticks": null, "simulate": true, "simulator": "goat simulator", "timeout_seconds": null, "top_type": 0, "use_jit": true, "use_system_verilog": true}
        // args: bits[8]:0x2a; bits[8]:0xb
        // args: bits[8]:0x2c; bits[8]:0x63
        fn main(x: u8, y: u8) -> u8 {
        x + y
        }
        """), crasher)

  def test_serialize_deserialize_with_codegen(self):
    crasher = textwrap.dedent("""\
    // options: {"codegen": true, "codegen_args": ["--generator=pipeline", "--pipeline_stages=2"], "convert_to_ir": false, "input_is_dslx": false, "optimize_ir": true, "simulate": true, "simulator": "goat simulator", "use_system_verilog": false}
    // args: bits[8]:0x2a; bits[8]:0xb
    // args: bits[8]:0x2c; bits[8]:0x63
    package foo
    fn bar(x: bits[16], y: bits[16) -> bits[16] {
      ret add.1: bits[16] = add(x, y)
    }
    """)
    got = sample.Sample.deserialize(crasher)
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
    self.assertMultiLineEqual(want.serialize(), got.serialize())
    self.assertEqual(got, want)

  def test_deserialize_without_codegen(self):
    crasher = textwrap.dedent("""\
      // options: {"input_is_dslx": false, "convert_to_ir": false, "optimize_ir": true, "codegen": true, "codegen_args": null, "simulate": false, "simulator": null, "use_system_verilog": true}
      // args: bits[8]:0x2a; bits[8]:0xb
      // args: bits[8]:0x2c; bits[8]:0x63
      package foo
      fn bar(x: bits[16], y: bits[16) -> bits[16] {
        ret add.1: bits[16] = add(x, y)
      }""")
    got = sample.Sample.deserialize(crasher)
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

  def test_serialize_deserialize_proc(self):
    crasher = textwrap.dedent("""\
    // options: {"calls_per_sample" : 0, "convert_to_ir": true, "input_is_dslx": true, "ir_converter_args": ["--top=main"], "optimize_ir": false, "proc_ticks": 42, "top_type": 1}
    // args: bits[32]:0x10; bits[1]:1
    // ir_channel_names: sample__input, sample__enable
    const DEFAULT_INIT_STATE = u32:42;
    proc main {
      input: chan<u32> in;
      enable: chan<bool> in;
      result: chan<u32> out;

      config(input: chan<u32> in,
            enable: chan<bool> in,
            result: chan<u32> out
            ) {
        (input, enable, result)
      }

      next(tok: token, state: u32) {
        let (tok_input, input_val) = recv(tok, input);
        let (tok_enable, enable_val) = recv(tok, enable);
        let tok_recv = join(tok_input, tok_enable);

        let result_val = if enable_val {
          input_val + u32:1
        } else {
          input_val
        }
        let tok_send = send(tok_recv, result, result_val);
        (result_val)
      }
    }
    """)
    got = sample.Sample.deserialize(crasher)
    want = sample.Sample(
        textwrap.dedent("""\
            const DEFAULT_INIT_STATE = u32:42;
            proc main {
              input: chan<u32> in;
              enable: chan<bool> in;
              result: chan<u32> out;

              config(input: chan<u32> in,
                    enable: chan<bool> in,
                    result: chan<u32> out
                    ) {
                (input, enable, result)
              }

              next(tok: token, state: u32) {
                let (tok_input, input_val) = recv(tok, input);
                let (tok_enable, enable_val) = recv(tok, enable);
                let tok_recv = join(tok_input, tok_enable);

                let result_val = if enable_val {
                  input_val + u32:1
                } else {
                  input_val
                }
                let tok_send = send(tok_recv, result, result_val);
                (result_val)
              }
            }"""),
        sample.SampleOptions(
            input_is_dslx=True,
            convert_to_ir=True,
            ir_converter_args=['--top=main'],
            optimize_ir=False,
            top_type=sample.TopType.proc,
            calls_per_sample=0,
            proc_ticks=42),
        sample.parse_args_batch('bits[32]:0x10; bits[1]:1'),
        ['sample__input', 'sample__enable'])
    self.assertMultiLineEqual(want.serialize(), got.serialize())
    self.assertEqual(got, want)

if __name__ == '__main__':
  absltest.main()
