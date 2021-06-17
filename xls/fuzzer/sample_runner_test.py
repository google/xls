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

import os
from typing import Tuple, Text

from xls.common import check_simulator
from xls.common import test_base
from xls.dslx.python.interp_value import interp_value_from_ir_string
from xls.fuzzer import sample_runner
from xls.fuzzer.python import cpp_sample as sample


def _read_file(dirname: Text, filename: Text) -> Text:
  """Returns the contents of the file in the given directory as a string."""
  with open(os.path.join(dirname, filename), 'r') as f:
    return f.read()


def _split_nonempty_lines(dirname: Text, filename: Text) -> Tuple[Text]:
  """Returns a tuple of the stripped non-empty lines in the file."""
  return tuple(
      l.strip() for l in _read_file(dirname, filename).splitlines() if l)


class SampleRunnerTest(test_base.TestCase):

  def _make_sample_dir(self):
    # Keep the directory around (no cleanup) if the test fails for easier
    # debugging.
    success = test_base.TempFileCleanup.SUCCESS  # type: test_base.TempFileCleanup
    return self.create_tempdir(cleanup=success).full_path

  def test_interpret_dslx_single_value(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    runner.run(
        sample.Sample(dslx_text, sample.SampleOptions(convert_to_ir=False), [[
            interp_value_from_ir_string('bits[8]:42'),
            interp_value_from_ir_string('bits[8]:100')
        ]]))
    self.assertEqual(
        _read_file(sample_dir, 'sample.x.results').strip(), 'bits[8]:0x8e')

  def test_interpret_dslx_multiple_values(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    runner.run(
        sample.Sample(dslx_text, sample.SampleOptions(convert_to_ir=False),
                      [[
                          interp_value_from_ir_string('bits[8]:42'),
                          interp_value_from_ir_string('bits[8]:100')
                      ],
                       [
                           interp_value_from_ir_string('bits[8]:222'),
                           interp_value_from_ir_string('bits[8]:240')
                       ]]))
    self.assertSequenceEqual(
        _split_nonempty_lines(sample_dir, 'sample.x.results'),
        ['bits[8]:0x8e', 'bits[8]:0xce'])

  def test_interpret_invalid_dslx(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'syntaxerror!!! fn main(x: u8, y: u8) -> u8 { x + y }'
    with self.assertRaises(sample_runner.SampleError):
      runner.run(
          sample.Sample(dslx_text, sample.SampleOptions(convert_to_ir=False), [[
              interp_value_from_ir_string('bits[8]:42'),
              interp_value_from_ir_string('bits[8]:100')
          ]]))
    # Verify the exception text is written out to file.
    self.assertIn('Expected start of top-level construct',
                  _read_file(sample_dir, 'exception.txt'))

  def test_dslx_to_ir(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    runner.run(
        sample.Sample(dslx_text, sample.SampleOptions(optimize_ir=False)))
    self.assertIn('package sample', _read_file(sample_dir, 'sample.ir'))

  def test_evaluate_ir(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    runner.run(
        sample.Sample(dslx_text, sample.SampleOptions(optimize_ir=False),
                      [[
                          interp_value_from_ir_string('bits[8]:42'),
                          interp_value_from_ir_string('bits[8]:100')
                      ],
                       [
                           interp_value_from_ir_string('bits[8]:222'),
                           interp_value_from_ir_string('bits[8]:240')
                       ]]))
    self.assertSequenceEqual(
        _split_nonempty_lines(sample_dir, 'sample.x.results'),
        ['bits[8]:0x8e', 'bits[8]:0xce'])
    self.assertSequenceEqual(
        _split_nonempty_lines(sample_dir, 'sample.ir.results'),
        ['bits[8]:0x8e', 'bits[8]:0xce'])

  def test_evaluate_ir_wide(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: bits[100], y: bits[100]) -> bits[100] { x + y }'
    runner.run(
        sample.Sample(
            dslx_text, sample.SampleOptions(optimize_ir=False),
            [[
                interp_value_from_ir_string('bits[100]:{0:#x}'.format(10**30)),
                interp_value_from_ir_string('bits[100]:{0:#x}'.format(10**30)),
            ],
             [
                 interp_value_from_ir_string('bits[100]:{0:#x}'.format(2**80)),
                 interp_value_from_ir_string('bits[100]:{0:#x}'.format(2**81)),
             ]]))
    self.assertSequenceEqual(
        _split_nonempty_lines(sample_dir, 'sample.x.results'), [
            'bits[100]:0x9_3e59_39a0_8ce9_dbd4_8000_0000',
            'bits[100]:0x3_0000_0000_0000_0000_0000'
        ])
    self.assertSequenceEqual(
        _split_nonempty_lines(sample_dir, 'sample.ir.results'), [
            'bits[100]:0x9_3e59_39a0_8ce9_dbd4_8000_0000',
            'bits[100]:0x3_0000_0000_0000_0000_0000'
        ])

  def test_interpret_mixed_signedness(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: s8) -> s8 { (x as s8) + y }'
    runner.run(
        sample.Sample(dslx_text, sample.SampleOptions(optimize_ir=False), [[
            interp_value_from_ir_string('bits[8]:42'),
            interp_value_from_ir_string('bits[8]:100').to_signed()
        ]]))
    self.assertEqual(
        _read_file(sample_dir, 'sample.x.results').strip(), 'bits[8]:0x8e')

  def test_interpret_mixed_signedness_unsigned_inputs(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: s8) -> s8 { (x as s8) + y }'
    runner.run(
        sample.Sample(dslx_text, sample.SampleOptions(optimize_ir=False),
                      sample.parse_args_batch('bits[8]:0xb0; bits[8]:0x0a')))
    self.assertEqual(
        _read_file(sample_dir, 'sample.x.results').strip(), 'bits[8]:0xba')

  def test_evaluate_ir_miscompare_single_result(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    runner._evaluate_ir = lambda *args: (interp_value_from_ir_string('bits[8]:1'
                                                                    ),)
    with self.assertRaises(sample_runner.SampleError) as e:
      runner.run(
          sample.Sample(dslx_text, sample.SampleOptions(optimize_ir=False), [[
              interp_value_from_ir_string('bits[8]:42'),
              interp_value_from_ir_string('bits[8]:100')
          ]]))
    self.assertIn(
        'Result miscompare for sample 0:\n'
        'args: bits[8]:0x2a; bits[8]:0x64\n'
        'evaluated unopt IR (JIT), evaluated unopt IR (interpreter) =\n'
        '   bits[8]:0x1\n'
        'interpreted DSLX =\n'
        '   bits[8]:0x8e', str(e.exception))
    self.assertIn('Result miscompare for sample 0',
                  _read_file(sample_dir, 'exception.txt'))

  def test_evaluate_ir_miscompare_multiple_results(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    results = (interp_value_from_ir_string('bits[8]:100'),
               interp_value_from_ir_string('bits[8]:1'))
    runner._evaluate_ir = lambda *args: results
    with self.assertRaises(sample_runner.SampleError) as e:
      runner.run(
          sample.Sample(dslx_text, sample.SampleOptions(optimize_ir=False),
                        [[
                            interp_value_from_ir_string('bits[8]:40'),
                            interp_value_from_ir_string('bits[8]:60')
                        ],
                         [
                             interp_value_from_ir_string('bits[8]:2'),
                             interp_value_from_ir_string('bits[8]:1')
                         ]]))
    self.assertIn('Result miscompare for sample 1', str(e.exception))
    self.assertIn('Result miscompare for sample 1',
                  _read_file(sample_dir, 'exception.txt'))

  def test_evaluate_ir_miscompare_number_of_results(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    # Give back two results instead of one.
    def fake_evaluate_ir(*_):
      return (interp_value_from_ir_string('bits[8]:100'),
              interp_value_from_ir_string('bits[8]:100'))

    runner._evaluate_ir = fake_evaluate_ir
    args_batch = [(interp_value_from_ir_string('bits[8]:42'),
                   interp_value_from_ir_string('bits[8]:64'))]
    with self.assertRaises(sample_runner.SampleError) as e:
      runner.run(
          sample.Sample(dslx_text, sample.SampleOptions(optimize_ir=False),
                        args_batch))
    self.assertIn(
        'Results for evaluated unopt IR (JIT) has 2 values, interpreted DSLX has 1',
        str(e.exception))
    self.assertIn(
        'Results for evaluated unopt IR (JIT) has 2 values, interpreted DSLX has 1',
        _read_file(sample_dir, 'exception.txt'))

  def test_interpret_opt_ir(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    runner.run(
        sample.Sample(dslx_text, sample.SampleOptions(), [[
            interp_value_from_ir_string('bits[8]:42'),
            interp_value_from_ir_string('bits[8]:100')
        ]]))
    self.assertIn('package sample', _read_file(sample_dir, 'sample.opt.ir'))
    self.assertSequenceEqual(
        _split_nonempty_lines(sample_dir, 'sample.opt.ir.results'),
        ['bits[8]:0x8e'])

  def test_interpret_opt_ir_miscompare(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    results = [
        (interp_value_from_ir_string('bits[8]:100'),),  # correct result
        (interp_value_from_ir_string('bits[8]:100'),),  # correct result
        (interp_value_from_ir_string('bits[8]:0'),),  # incorrect result
        (interp_value_from_ir_string('bits[8]:100'),),  # correct result
    ]

    def result_gen(*_):
      return results.pop(0)

    runner._evaluate_ir = result_gen
    with self.assertRaises(sample_runner.SampleError) as e:
      runner.run(
          sample.Sample(dslx_text, sample.SampleOptions(), [[
              interp_value_from_ir_string('bits[8]:40'),
              interp_value_from_ir_string('bits[8]:60')
          ]]))
    self.assertIn('Result miscompare for sample 0', str(e.exception))
    self.assertIn('evaluated opt IR (JIT)', str(e.exception))

  def test_codegen_combinational(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    runner.run(
        sample.Sample(
            dslx_text,
            sample.SampleOptions(
                codegen=True,
                codegen_args=['--generator=combinational'],
                simulate=True), [[
                    interp_value_from_ir_string('bits[8]:42'),
                    interp_value_from_ir_string('bits[8]:100')
                ]]))

    self.assertIn('endmodule', _read_file(sample_dir, 'sample.v'))
    # A combinational block should not have a blocking assignment.
    self.assertNotIn('<=', _read_file(sample_dir, 'sample.v'))
    self.assertSequenceEqual(
        _split_nonempty_lines(sample_dir, 'sample.v.results'), ['bits[8]:0x8e'])

  def test_codegen_combinational_wrong_results(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    runner._simulate = lambda *args: (interp_value_from_ir_string('bits[8]:1'),)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    with self.assertRaises(sample_runner.SampleError) as e:
      runner.run(
          sample.Sample(
              dslx_text,
              sample.SampleOptions(
                  codegen=True,
                  codegen_args=['--generator=combinational'],
                  simulate=True,
                  simulator='iverilog'), [[
                      interp_value_from_ir_string('bits[8]:42'),
                      interp_value_from_ir_string('bits[8]:100')
                  ]]))
    self.assertIn('Result miscompare for sample 0', str(e.exception))

  @test_base.skipIf(not check_simulator.runs_system_verilog(),
                    'uses SystemVerilog')
  def test_codegen_pipeline(self):
    sample_dir = self._make_sample_dir()
    print('sample_dir = ' + sample_dir)
    runner = sample_runner.SampleRunner(sample_dir)
    dslx_text = 'fn main(x: u8, y: u8) -> u8 { x + y }'
    runner.run(
        sample.Sample(
            dslx_text,
            sample.SampleOptions(
                codegen=True,
                codegen_args=('--generator=pipeline', '--pipeline_stages=2'),
                simulate=True), [[
                    interp_value_from_ir_string('bits[8]:42'),
                    interp_value_from_ir_string('bits[8]:100')
                ]]))
    # A pipelined block should have a blocking assignment.
    self.assertIn('<=', _read_file(sample_dir, 'sample.v'))
    self.assertSequenceEqual(
        _split_nonempty_lines(sample_dir, 'sample.v.results'), ['bits[8]:0x8e'])

  def test_ir_input(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    ir_text = """package foo

    fn foo(x: bits[8], y: bits[8]) -> bits[8] {
      ret add.1: bits[8] = add(x, y)
    }
    """
    runner.run(
        sample.Sample(ir_text, sample.SampleOptions(input_is_dslx=False), [[
            interp_value_from_ir_string('bits[8]:42'),
            interp_value_from_ir_string('bits[8]:100')
        ]]))
    self.assertIn('package foo', _read_file(sample_dir, 'sample.ir'))
    self.assertIn('package foo', _read_file(sample_dir, 'sample.opt.ir'))
    self.assertSequenceEqual(
        _split_nonempty_lines(sample_dir, 'sample.opt.ir.results'),
        ['bits[8]:0x8e'])

  def test_bad_ir_input(self):
    sample_dir = self._make_sample_dir()
    runner = sample_runner.SampleRunner(sample_dir)
    ir_text = """bogus ir string"""
    with self.assertRaises(sample_runner.SampleError):
      runner.run(
          sample.Sample(ir_text, sample.SampleOptions(input_is_dslx=False)))
    self.assertIn('Expected \'package\' keyword',
                  _read_file(sample_dir, 'opt_main.stderr'))
    self.assertRegex(
        _read_file(sample_dir, 'exception.txt'),
        '.*opt_main.*returned non-zero exit status')


if __name__ == '__main__':
  test_base.main()
