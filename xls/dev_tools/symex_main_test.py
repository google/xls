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
"""Tests for symex_main CLI tool."""

import subprocess

from google.protobuf import text_format

from xls.common import gfile
from xls.common import runfiles
from xls.common import test_base
from xls.tests import testvector_pb2

SYMEX_MAIN_PATH = runfiles.get_path('xls/dev_tools/symex_main')

_TEST_IR = """package test_pkg

top fn execute_alu(op: bits[2], a: bits[8], b: bits[8]) -> (bits[1], bits[8]) {
  literal_0: bits[8] = literal(value=0, id=1)
  literal_false: bits[1] = literal(value=0, id=2)
  literal_true: bits[1] = literal(value=1, id=3)

  add_res: bits[8] = add(a, b, id=4)
  and_res: bits[8] = and(a, b, id=5)

  literal_255: bits[9] = literal(value=255, id=6)
  a_ext: bits[9] = zero_ext(a, new_bit_count=9, id=7)
  b_ext: bits[9] = zero_ext(b, new_bit_count=9, id=8)
  sum_9: bits[9] = add(a_ext, b_ext, id=9)
  overflow: bits[1] = ugt(sum_9, literal_255, id=10)

  add_out: bits[8] = sel(overflow, cases=[add_res, literal_0], id=11)
  add_status: bits[1] = sel(overflow, cases=[literal_false, literal_true], id=12)

  out_val: bits[8] = sel(op, cases=[add_out, and_res], default=literal_0, id=13)
  out_status: bits[1] = sel(op, cases=[add_status, literal_false], default=literal_true, id=14)

  ret result: (bits[1], bits[8]) = tuple(out_status, out_val, id=15)
}
"""


class SymexMainTest(test_base.TestCase):

  def setUp(self):
    super().setUp()
    self.ir_file = self.create_tempfile(content=_TEST_IR)

  def test_basic_text_output(self):
    output = subprocess.check_output(
        [
            SYMEX_MAIN_PATH,
            self.ir_file.full_path,
            '--top=execute_alu',
        ],
        stderr=subprocess.STDOUT,
    ).decode('utf-8')
    self.assertIn('Explored 6 feasible path(s)', output)
    self.assertIn('Path #0:', output)
    self.assertIn('Path #1:', output)
    self.assertIn('Path #2:', output)
    self.assertIn('Path #3:', output)
    self.assertIn('Path #4:', output)
    self.assertIn('Path #5:', output)

  def test_default_top_function(self):
    # Tests that omitting --top automatically infers the package top function.
    output = subprocess.check_output(
        [
            SYMEX_MAIN_PATH,
            self.ir_file.full_path,
        ],
        stderr=subprocess.STDOUT,
    ).decode('utf-8')
    self.assertIn('Explored 6 feasible path(s)', output)

  def test_concrete_inputs(self):
    output = subprocess.check_output(
        [
            SYMEX_MAIN_PATH,
            self.ir_file.full_path,
            '--top=execute_alu',
            '--concrete_inputs=op=0',
        ],
        stderr=subprocess.STDOUT,
    ).decode('utf-8')
    self.assertIn('Explored 2 feasible path(s)', output)
    self.assertIn('op = bits[2]:0', output)

  def test_output_testvector_textproto(self):
    out_file = self.create_tempfile()
    output = subprocess.check_output(
        [
            SYMEX_MAIN_PATH,
            self.ir_file.full_path,
            '--top=execute_alu',
            f'--output_testvector_textproto={out_file.full_path}',
        ],
        stderr=subprocess.STDOUT,
    ).decode('utf-8')
    self.assertIn('Explored 6 feasible path(s)', output)
    with gfile.open(out_file.full_path, 'r') as f:
      proto = text_format.Parse(f.read(), testvector_pb2.SampleInputsProto())

    self.assertTrue(proto.HasField('function_args'))
    self.assertLen(proto.function_args.args, 6)

  def test_max_paths(self):
    output = subprocess.check_output(
        [
            SYMEX_MAIN_PATH,
            self.ir_file.full_path,
            '--top=execute_alu',
            '--max_paths=2',
        ],
        stderr=subprocess.STDOUT,
    ).decode('utf-8')
    self.assertIn('Explored 2 feasible path(s)', output)

  def test_multiple_concrete_inputs(self):
    output = subprocess.check_output(
        [
            SYMEX_MAIN_PATH,
            self.ir_file.full_path,
            '--top=execute_alu',
            '--concrete_inputs=op=0,a=10',
        ],
        stderr=subprocess.STDOUT,
    ).decode('utf-8')
    self.assertIn('Explored 2 feasible path(s)', output)
    self.assertIn('op = bits[2]:0', output)
    self.assertIn('a = bits[8]:10', output)

  def test_invalid_param_error(self):
    proc = subprocess.run(
        [
            SYMEX_MAIN_PATH,
            self.ir_file.full_path,
            '--top=execute_alu',
            '--concrete_inputs=nonexistent=0',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("does not have a parameter named 'nonexistent'", proc.stderr)

  def test_malformed_concrete_input_error(self):
    proc = subprocess.run(
        [
            SYMEX_MAIN_PATH,
            self.ir_file.full_path,
            '--top=execute_alu',
            '--concrete_inputs=op_without_equal',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn('Invalid concrete input format', proc.stderr)

  def test_missing_input_file_error(self):
    proc = subprocess.run(
        [
            SYMEX_MAIN_PATH,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn('Missing required input IR file', proc.stderr)

  def test_negative_max_paths_error(self):
    proc = subprocess.run(
        [
            SYMEX_MAIN_PATH,
            self.ir_file.full_path,
            '--top=execute_alu',
            '--max_paths=-1',
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn('--max_paths must be non-negative', proc.stderr)


if __name__ == '__main__':
  test_base.main()
