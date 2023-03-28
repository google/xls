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

"""Tests for xls.tools.codegen_main."""

import inspect
import os
import subprocess

from absl import flags
from google.protobuf import text_format
from absl.testing import absltest
from absl.testing import parameterized
from xls.codegen import module_signature_pb2
from xls.common import runfiles
from xls.common import test_base

_UPDATE_GOLDEN = flags.DEFINE_bool(
    'test_update_golden_files', False,
    'whether to update golden reference files')
_XLS_SOURCE_DIR = flags.DEFINE_string(
    'xls_source_dir', '',
    'base path to use to update golden files, note this will cause writes '
    'within the given directory')
CODEGEN_MAIN_PATH = runfiles.get_path('xls/tools/codegen_main')
SHA256_IR_PATH = runfiles.get_path('xls/examples/sha256.opt.ir')

NOT_ADD_IR = """package not_add

fn not_add(x: bits[32], y: bits[32]) -> bits[32] {
  add.1: bits[32] = add(x, y)
  ret not.2: bits[32] = not(add.1)
}
"""

NEG_PROC_IR = """package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only,
        flow_control=ready_valid, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only,
        flow_control=ready_valid, metadata="")

proc neg_proc(my_token: token, my_state: (), init={()}) {
  rcv: (token, bits[32]) = receive(my_token, channel_id=0)
  data: bits[32] = tuple_index(rcv, index=1)
  negate: bits[32] = neg(data)
  rcv_token: token = tuple_index(rcv, index=0)
  send: token = send(rcv_token, negate, channel_id=1)
  next (send, my_state)
}
"""

ASSERT_IR = """package assert_example

fn invert_with_assert(x: bits[1]) -> bits[1] {
  after_all.1: token = after_all()
  assert.2: token = assert(after_all.1, x, message="assert message", id=2)
  ret not.3: bits[1] = not(x)
}
"""

GATE_IR = """package gate_example

fn gate_example(x: bits[32], y: bits[1]) -> bits[32] {
  ret gate.1: bits[32] = gate(y, x)
}
"""


class CodeGenMainTest(parameterized.TestCase):

  def _compare_to_golden(self, got: str) -> None:
    """Compares the obtained verilog to a golden reference file.

    Writes to file if the --update_golden flag was given.

    Args:
      got: Verilog obtained from XLS to compare against the golden reference.
    """
    caller: str = inspect.stack()[1].function
    path: str = f'xls/tools/testdata/codegen_main_test__{caller}.vtxt'
    if _UPDATE_GOLDEN.value:
      dirpath, xls = os.path.split(_XLS_SOURCE_DIR.value.rstrip('/'))
      assert xls == 'xls', (
          _XLS_SOURCE_DIR.value,
          'should end with `/xls/` got',
          xls,
      )
      path_to_write: str = os.path.join(dirpath, path)
      with open(path_to_write, 'w') as f:
        f.write(got)
    else:
      want: str = runfiles.get_contents_as_text(path)
      self.assertMultiLineEqual(got, want)

  def test_combinational(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)

    signature_path = test_base.create_named_output_text_file(
        'combinational_sig.textproto')
    verilog_path = test_base.create_named_output_text_file('combinational.v')

    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=not_add', '--output_signature_path=' + signature_path,
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
        '--top=not_add', ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  @parameterized.parameters(range(1, 6))
  def test_fixed_pipeline_length(self, pipeline_stages):
    signature_path = test_base.create_named_output_text_file(
        f'sha256.{pipeline_stages}_stage.sig.textproto')
    verilog_path = test_base.create_named_output_text_file(
        f'sha256.{pipeline_stages}_stage.v')
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--delay_model=unit',
        '--pipeline_stages=' + str(pipeline_stages), '--reset_data_path=false',
        '--alsologtostderr', '--output_signature_path=' + signature_path,
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
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--reset_data_path=false',
        '--delay_model=unit', '--clock_period_ps=' + str(clock_period_ps),
        '--alsologtostderr', '--output_verilog_path=' + verilog_path,
        SHA256_IR_PATH
    ])

  def test_clock_period_and_pipeline_stages(self):
    pipeline_stages = 5
    clock_period_ps = 5000
    verilog_path = test_base.create_named_output_text_file(
        f'sha256.clock_{clock_period_ps}_ps_pipeline_stages_{pipeline_stages}.v'
    )
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--delay_model=unit',
        '--pipeline_stages=' + str(pipeline_stages),
        '--clock_period_ps=' + str(clock_period_ps), '--reset_data_path=false',
        '--alsologtostderr', '--output_verilog_path=' + verilog_path,
        SHA256_IR_PATH
    ])

  def test_custom_module_name(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--delay_model=unit',
        '--pipeline_stages=3', '--reset_data_path=false',
        '--clock_period_ps=1500', '--alsologtostderr', '--top=not_add',
        '--module_name=foo_qux_baz', ir_file.full_path
    ]).decode('utf-8')
    self.assertIn('module foo_qux_baz(', verilog)

  def test_pipeline_system_verilog(self):
    verilog_path = test_base.create_named_output_text_file('sha256.sv')
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--use_system_verilog', '--generator=pipeline',
        '--delay_model=unit', '--pipeline_stages=10', '--reset_data_path=false',
        '--alsologtostderr', '--output_verilog_path=' + verilog_path,
        SHA256_IR_PATH
    ])

    with open(verilog_path, 'r') as f:
      verilog = f.read()
      self.assertIn('always_ff', verilog)
      self.assertNotIn('always @ (*)', verilog)

  def test_pipeline_no_system_verilog(self):
    verilog_path = test_base.create_named_output_text_file('sha256.v')
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--nouse_system_verilog', '--generator=pipeline',
        '--delay_model=unit', '--pipeline_stages=10', '--reset_data_path=false',
        '--alsologtostderr', '--output_verilog_path=' + verilog_path,
        SHA256_IR_PATH
    ])

    with open(verilog_path, 'r') as f:
      verilog = f.read()
      self.assertNotIn('always_ff', verilog)
      self.assertIn('always @ (posedge clk)', verilog)

  def test_separate_lines(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=not_add', '--separate_lines', ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  def test_proc_verilog_port_default_suffix(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=neg_proc', ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  def test_proc_verilog_port_nondefault_suffix(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=neg_proc', '--streaming_channel_data_suffix=_d',
        '--streaming_channel_ready_suffix=_r',
        '--streaming_channel_valid_suffix=_v', ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  def test_assert_format(self):
    ir_file = self.create_tempfile(content=ASSERT_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=invert_with_assert',
        "--assert_format='`MY_ASSERT({condition}, \"{message}\")'",
        ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  def test_gate_format(self):
    ir_file = self.create_tempfile(content=GATE_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=gate_example',
        "--gate_format='assign {output} = `MY_GATE({input}, {condition})'",
        ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  def test_flop_inputs(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--reset_data_path=false', '--delay_model=unit', '--alsologtostderr',
        '--top=not_add', '--flop_inputs', ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  def test_flop_outputs_false(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--reset_data_path=false', '--delay_model=unit', '--alsologtostderr',
        '--top=not_add', '--flop_outputs=false', ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  def test_add_idle_output_proc(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--reset=rst', '--reset_data_path=false', '--delay_model=unit',
        '--alsologtostderr', '--top=neg_proc', '--add_idle_output',
        ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  def test_flop_output_kind_skid(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--reset=rst', '--reset_data_path=false', '--delay_model=unit',
        '--alsologtostderr', '--top=neg_proc', '--flop_outputs_kind=skid',
        ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)

  def test_flop_output_kind_zerolatency(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--reset=rst', '--reset_data_path=false', '--delay_model=unit',
        '--alsologtostderr', '--top=neg_proc',
        '--flop_outputs_kind=zerolatency', ir_file.full_path
    ]).decode('utf-8')
    self._compare_to_golden(verilog)


if __name__ == '__main__':
  absltest.main()
