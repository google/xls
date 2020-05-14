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

"""Tests for xls.tools.codegen_main."""

import subprocess

from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from xls.codegen import module_signature_pb2
from xls.common import runfiles
from xls.common import test_base

CODEGEN_MAIN_PATH = runfiles.get_path('xls/tools/codegen_main')
SHA256_IR_PATH = runfiles.get_path('xls/examples/sha256.opt.ir')

NOT_ADD_IR = """package not_add

fn not_add(x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  ret not.2: bits[32] = not(add.1)
}
"""


class CodeGenMainTest(parameterized.TestCase):

  def test_combinational(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)

    signature_path = test_base.create_named_output_text_file(
        'combinational_sig.pbtxt')
    verilog_path = test_base.create_named_output_text_file('combinational.v')

    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--entry=not_add', '--output_signature_path=' + signature_path,
        '--output_verilog_path=' + verilog_path, ir_file.full_path
    ])

    with open(verilog_path, 'r') as f:
      self.assertIn('module not_add(', f.read())

    with open(signature_path, 'r') as f:
      sig_proto = text_format.Parse(f.read(),
                                    module_signature_pb2.ModuleSignatureProto())
      self.assertEqual(sig_proto.module_name, 'not_add')
      self.assertTrue(sig_proto.HasField('combinational'))

  def test_combinational_verilog_to_stdout(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--entry=not_add', ir_file.full_path
    ]).decode('utf-8')
    self.assertIn('module not_add(', verilog)

  @parameterized.parameters(range(1, 6))
  def test_fixed_pipeline_length(self, pipeline_stages):
    signature_path = test_base.create_named_output_text_file(
        f'sha256.{pipeline_stages}_stage.sig.pbtxt')
    verilog_path = test_base.create_named_output_text_file(
        f'sha256.{pipeline_stages}_stage.v')
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--generator=pipeline',
        '--pipeline_stages=' + str(pipeline_stages), '--alsologtostderr',
        '--output_signature_path=' + signature_path,
        '--output_verilog_path=' + verilog_path, SHA256_IR_PATH
    ])

    with open(verilog_path, 'r') as f:
      verilog = f.read()
      self.assertIn(f'// ===== Pipe stage {pipeline_stages}', verilog)
      self.assertNotIn(f'// ===== Pipe stage {pipeline_stages + 1}', verilog)

    with open(signature_path, 'r') as f:
      sig_proto = text_format.Parse(f.read(),
                                    module_signature_pb2.ModuleSignatureProto())
      self.assertTrue(sig_proto.HasField('pipeline'))
      self.assertEqual(sig_proto.pipeline.latency, pipeline_stages + 1)

  @parameterized.parameters([500, 1000, 1500])
  def test_fixed_clock_period(self, clock_period_ps):
    verilog_path = test_base.create_named_output_text_file(
        f'sha256.clock_{clock_period_ps}_ps.v')
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--generator=pipeline',
        '--clock_period_ps=' + str(clock_period_ps), '--alsologtostderr',
        '--output_verilog_path=' + verilog_path, SHA256_IR_PATH
    ])

  def test_clock_period_and_pipeline_stages(self):
    pipeline_stages = 20
    clock_period_ps = 5000
    verilog_path = test_base.create_named_output_text_file(
        f'sha256.clock_{clock_period_ps}_ps_pipeline_stages_{pipeline_stages}.v'
    )
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--generator=pipeline',
        '--pipeline_stages=' + str(pipeline_stages),
        '--clock_period_ps=' + str(clock_period_ps), '--alsologtostderr',
        '--output_verilog_path=' + verilog_path, SHA256_IR_PATH
    ])

  def test_custom_module_name(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=3',
        '--clock_period_ps=1500', '--alsologtostderr', '--entry=not_add',
        '--module_name=foo_qux_baz', ir_file.full_path
    ]).decode('utf-8')
    self.assertIn('module foo_qux_baz(', verilog)


if __name__ == '__main__':
  absltest.main()
