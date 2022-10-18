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
    self.assertIn('module not_add(', verilog)

  @parameterized.parameters(range(1, 6))
  def test_fixed_pipeline_length(self, pipeline_stages):
    signature_path = test_base.create_named_output_text_file(
        f'sha256.{pipeline_stages}_stage.sig.textproto')
    verilog_path = test_base.create_named_output_text_file(
        f'sha256.{pipeline_stages}_stage.v')
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--delay_model=unit',
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
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--delay_model=unit',
        '--clock_period_ps=' + str(clock_period_ps), '--alsologtostderr',
        '--output_verilog_path=' + verilog_path, SHA256_IR_PATH
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
        '--clock_period_ps=' + str(clock_period_ps), '--alsologtostderr',
        '--output_verilog_path=' + verilog_path, SHA256_IR_PATH
    ])

  def test_custom_module_name(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--delay_model=unit',
        '--pipeline_stages=3', '--clock_period_ps=1500', '--alsologtostderr',
        '--top=not_add', '--module_name=foo_qux_baz', ir_file.full_path
    ]).decode('utf-8')
    self.assertIn('module foo_qux_baz(', verilog)

  def test_pipeline_system_verilog(self):
    verilog_path = test_base.create_named_output_text_file('sha256.sv')
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--use_system_verilog', '--generator=pipeline',
        '--delay_model=unit', '--pipeline_stages=10', '--alsologtostderr',
        '--output_verilog_path=' + verilog_path, SHA256_IR_PATH
    ])

    with open(verilog_path, 'r') as f:
      verilog = f.read()
      self.assertIn('always_ff', verilog)
      self.assertNotIn('always @ (*)', verilog)

  def test_pipeline_no_system_verilog(self):
    verilog_path = test_base.create_named_output_text_file('sha256.v')
    subprocess.check_call([
        CODEGEN_MAIN_PATH, '--nouse_system_verilog', '--generator=pipeline',
        '--delay_model=unit', '--pipeline_stages=10', '--alsologtostderr',
        '--output_verilog_path=' + verilog_path, SHA256_IR_PATH
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
    self.assertEqual("""module not_add(
  input wire [31:0] x,
  input wire [31:0] y,
  output wire [31:0] out
);
  wire [31:0] add_7;
  wire [31:0] not_8;
  assign add_7 = x + y;
  assign not_8 = ~add_7;
  assign out = not_8;
endmodule
""", verilog)

  def test_proc_verilog_port_default_suffix(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=neg_proc', ir_file.full_path
    ]).decode('utf-8')

    self.assertIn('module neg_proc(', verilog)
    self.assertIn('wire [31:0] in', verilog)
    self.assertIn('wire in_vld', verilog)
    self.assertIn('wire in_rdy', verilog)

    self.assertIn('wire [31:0] out', verilog)
    self.assertIn('wire out_vld', verilog)
    self.assertIn('wire out_rdy', verilog)

  def test_proc_verilog_port_nondefault_suffix(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=neg_proc', '--streaming_channel_data_suffix=_d',
        '--streaming_channel_ready_suffix=_r',
        '--streaming_channel_valid_suffix=_v', ir_file.full_path
    ]).decode('utf-8')

    self.assertIn('module neg_proc(', verilog)
    self.assertIn('wire [31:0] in_d', verilog)
    self.assertIn('wire in_v', verilog)
    self.assertIn('wire in_r', verilog)

    self.assertIn('wire [31:0] out_d', verilog)
    self.assertIn('wire out_v', verilog)
    self.assertIn('wire out_r', verilog)

  def test_assert_format(self):
    ir_file = self.create_tempfile(content=ASSERT_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=invert_with_assert',
        "--assert_format='`MY_ASSERT({condition}, \"{message}\")'",
        ir_file.full_path
    ]).decode('utf-8')

    self.assertIn('module invert_with_assert', verilog)
    self.assertIn('MY_ASSERT', verilog)

  def test_gate_format(self):
    ir_file = self.create_tempfile(content=GATE_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=combinational', '--alsologtostderr',
        '--top=gate_example',
        "--gate_format='assign {output} = `MY_GATE({input}, {condition})'",
        ir_file.full_path
    ]).decode('utf-8')

    self.assertIn('module gate_example', verilog)
    self.assertIn('MY_GATE', verilog)

  def test_flop_inputs(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--delay_model=unit', '--alsologtostderr', '--top=not_add',
        '--flop_inputs', ir_file.full_path
    ]).decode('utf-8')
    self.assertEqual(
        """module not_add(
  input wire clk,
  input wire [31:0] x,
  input wire [31:0] y,
  output wire [31:0] out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [31:0] p0_x;
  reg [31:0] p0_y;
  always_ff @ (posedge clk) begin
    p0_x <= x;
    p0_y <= y;
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_add_11_comb;
  wire [31:0] p1_not_12_comb;
  assign p1_add_11_comb = p0_x + p0_y;
  assign p1_not_12_comb = ~p1_add_11_comb;

  // Registers for pipe stage 1:
  reg [31:0] p1_not_12;
  always_ff @ (posedge clk) begin
    p1_not_12 <= p1_not_12_comb;
  end
  assign out = p1_not_12;
endmodule
""", verilog)

  def test_flop_outputs_false(self):
    ir_file = self.create_tempfile(content=NOT_ADD_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--delay_model=unit', '--alsologtostderr', '--top=not_add',
        '--flop_outputs=false', ir_file.full_path
    ]).decode('utf-8')
    self.assertEqual(
        """module not_add(
  input wire clk,
  input wire [31:0] x,
  input wire [31:0] y,
  output wire [31:0] out
);
  // ===== Pipe stage 0:

  // Registers for pipe stage 0:
  reg [31:0] p0_x;
  reg [31:0] p0_y;
  always_ff @ (posedge clk) begin
    p0_x <= x;
    p0_y <= y;
  end

  // ===== Pipe stage 1:
  wire [31:0] p1_add_11_comb;
  assign p1_add_11_comb = p0_x + p0_y;
  assign out = ~p1_add_11_comb;
endmodule
""", verilog)

  def test_add_idle_output_proc(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--reset=rst', '--delay_model=unit', '--alsologtostderr',
        '--top=neg_proc', '--add_idle_output', ir_file.full_path
    ]).decode('utf-8')
    self.assertEqual(
        """\
module neg_proc(
  input wire clk,
  input wire rst,
  input wire [31:0] in,
  input wire in_vld,
  input wire out_rdy,
  output wire [31:0] out,
  output wire out_vld,
  output wire in_rdy,
  output wire idle
);
  reg [31:0] __in_reg;
  reg __in_valid_reg;
  reg [31:0] __out_reg;
  reg __out_valid_reg;
  wire and_20;
  wire literal_23;
  wire out_valid_inv;
  wire __out_vld_buf;
  wire out_valid_load_en;
  wire out_load_en;
  wire pipeline_enable;
  wire in_valid_inv;
  wire in_valid_load_en;
  wire in_load_en;
  wire [31:0] negate;
  assign and_20 = __in_valid_reg;
  assign literal_23 = 1'h1;
  assign out_valid_inv = ~__out_valid_reg;
  assign __out_vld_buf = and_20 & literal_23 & 1'h1;
  assign out_valid_load_en = out_rdy | out_valid_inv;
  assign out_load_en = __out_vld_buf & out_valid_load_en;
  assign pipeline_enable = literal_23 & and_20 & out_load_en & (literal_23 & and_20 & out_load_en);
  assign in_valid_inv = ~__in_valid_reg;
  assign in_valid_load_en = pipeline_enable | in_valid_inv;
  assign in_load_en = in_vld & in_valid_load_en;
  assign negate = -__in_reg;
  always_ff @ (posedge clk) begin
    if (rst) begin
      __in_reg <= 32'h0000_0000;
      __in_valid_reg <= 1'h0;
      __out_reg <= 32'h0000_0000;
      __out_valid_reg <= 1'h0;
    end else begin
      __in_reg <= in_load_en ? in : __in_reg;
      __in_valid_reg <= in_valid_load_en ? in_vld : __in_valid_reg;
      __out_reg <= out_load_en ? negate : __out_reg;
      __out_valid_reg <= out_valid_load_en ? __out_vld_buf : __out_valid_reg;
    end
  end
  assign out = __out_reg;
  assign out_vld = __out_valid_reg;
  assign in_rdy = in_load_en;
  assign idle = ~(__in_valid_reg | __out_valid_reg | in_vld | __out_valid_reg);
endmodule
""", verilog)

  def test_flop_output_kind_skid(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--reset=rst', '--delay_model=unit', '--alsologtostderr',
        '--top=neg_proc', '--flop_outputs_kind=skid', ir_file.full_path
    ]).decode('utf-8')
    self.assertEqual(
        """\
module neg_proc(
  input wire clk,
  input wire rst,
  input wire [31:0] in,
  input wire in_vld,
  input wire out_rdy,
  output wire [31:0] out,
  output wire out_vld,
  output wire in_rdy
);
  reg [31:0] __in_reg;
  reg __in_valid_reg;
  reg [31:0] __out_reg;
  reg [31:0] __out_skid_reg;
  reg __out_valid_reg;
  reg __out_valid_skid_reg;
  wire out_from_skid_rdy;
  wire literal_23;
  wire and_20;
  wire __out_vld_buf;
  wire pipeline_enable;
  wire in_valid_inv;
  wire out_data_valid_load_en;
  wire out_to_is_not_rdy;
  wire in_valid_load_en;
  wire out_data_is_sent_to;
  wire out_skid_data_load_en;
  wire out_skid_valid_set_zero;
  wire [31:0] out_select;
  wire out_valid_or;
  wire in_load_en;
  wire [31:0] negate;
  wire out_data_valid_load_en__1;
  wire out_skid_valid_load_en;
  assign out_from_skid_rdy = ~__out_valid_skid_reg;
  assign literal_23 = 1'h1;
  assign and_20 = __in_valid_reg;
  assign __out_vld_buf = and_20 & literal_23 & 1'h1;
  assign pipeline_enable = literal_23 & and_20 & out_from_skid_rdy & (literal_23 & and_20 & out_from_skid_rdy);
  assign in_valid_inv = ~__in_valid_reg;
  assign out_data_valid_load_en = __out_vld_buf & out_from_skid_rdy;
  assign out_to_is_not_rdy = ~out_rdy;
  assign in_valid_load_en = pipeline_enable | in_valid_inv;
  assign out_data_is_sent_to = __out_valid_reg & out_rdy & out_from_skid_rdy;
  assign out_skid_data_load_en = __out_valid_reg & out_data_valid_load_en & out_to_is_not_rdy;
  assign out_skid_valid_set_zero = __out_valid_skid_reg & out_rdy;
  assign out_select = __out_valid_skid_reg ? __out_skid_reg : __out_reg;
  assign out_valid_or = __out_valid_reg | __out_valid_skid_reg;
  assign in_load_en = in_vld & in_valid_load_en;
  assign negate = -__in_reg;
  assign out_data_valid_load_en__1 = out_data_is_sent_to | out_data_valid_load_en;
  assign out_skid_valid_load_en = out_skid_data_load_en | out_skid_valid_set_zero;
  always_ff @ (posedge clk) begin
    if (rst) begin
      __in_reg <= 32'h0000_0000;
      __in_valid_reg <= 1'h0;
      __out_reg <= 32'h0000_0000;
      __out_skid_reg <= 32'h0000_0000;
      __out_valid_reg <= 1'h0;
      __out_valid_skid_reg <= 1'h0;
    end else begin
      __in_reg <= in_load_en ? in : __in_reg;
      __in_valid_reg <= in_valid_load_en ? in_vld : __in_valid_reg;
      __out_reg <= out_data_valid_load_en ? negate : __out_reg;
      __out_skid_reg <= out_skid_data_load_en ? __out_reg : __out_skid_reg;
      __out_valid_reg <= out_data_valid_load_en__1 ? __out_vld_buf : __out_valid_reg;
      __out_valid_skid_reg <= out_skid_valid_load_en ? out_from_skid_rdy : __out_valid_skid_reg;
    end
  end
  assign out = out_select;
  assign out_vld = out_valid_or;
  assign in_rdy = in_load_en;
endmodule
""", verilog)

  def test_flop_output_kind_zerolatency(self):
    ir_file = self.create_tempfile(content=NEG_PROC_IR)
    verilog = subprocess.check_output([
        CODEGEN_MAIN_PATH, '--generator=pipeline', '--pipeline_stages=1',
        '--reset=rst', '--delay_model=unit', '--alsologtostderr',
        '--top=neg_proc', '--flop_outputs_kind=zerolatency', ir_file.full_path
    ]).decode('utf-8')
    self.assertEqual(
        """\
module neg_proc(
  input wire clk,
  input wire rst,
  input wire [31:0] in,
  input wire in_vld,
  input wire out_rdy,
  output wire [31:0] out,
  output wire out_vld,
  output wire in_rdy
);
  reg [31:0] __in_reg;
  reg __in_valid_reg;
  reg [31:0] __out_skid_reg;
  reg __out_valid_skid_reg;
  wire out_from_skid_rdy;
  wire literal_23;
  wire and_20;
  wire pipeline_enable;
  wire in_valid_inv;
  wire __out_vld_buf;
  wire out_to_is_not_rdy;
  wire [31:0] negate;
  wire in_valid_load_en;
  wire out_skid_data_load_en;
  wire out_skid_valid_set_zero;
  wire [31:0] out_select;
  wire out_valid_or;
  wire in_load_en;
  wire out_skid_valid_load_en;
  assign out_from_skid_rdy = ~__out_valid_skid_reg;
  assign literal_23 = 1'h1;
  assign and_20 = __in_valid_reg;
  assign pipeline_enable = literal_23 & and_20 & out_from_skid_rdy & (literal_23 & and_20 & out_from_skid_rdy);
  assign in_valid_inv = ~__in_valid_reg;
  assign __out_vld_buf = and_20 & literal_23 & 1'h1;
  assign out_to_is_not_rdy = ~out_rdy;
  assign negate = -__in_reg;
  assign in_valid_load_en = pipeline_enable | in_valid_inv;
  assign out_skid_data_load_en = __out_vld_buf & out_from_skid_rdy & out_to_is_not_rdy;
  assign out_skid_valid_set_zero = __out_valid_skid_reg & out_rdy;
  assign out_select = __out_valid_skid_reg ? __out_skid_reg : negate;
  assign out_valid_or = __out_vld_buf | __out_valid_skid_reg;
  assign in_load_en = in_vld & in_valid_load_en;
  assign out_skid_valid_load_en = out_skid_data_load_en | out_skid_valid_set_zero;
  always_ff @ (posedge clk) begin
    if (rst) begin
      __in_reg <= 32'h0000_0000;
      __in_valid_reg <= 1'h0;
      __out_skid_reg <= 32'h0000_0000;
      __out_valid_skid_reg <= 1'h0;
    end else begin
      __in_reg <= in_load_en ? in : __in_reg;
      __in_valid_reg <= in_valid_load_en ? in_vld : __in_valid_reg;
      __out_skid_reg <= out_skid_data_load_en ? negate : __out_skid_reg;
      __out_valid_skid_reg <= out_skid_valid_load_en ? out_from_skid_rdy : __out_valid_skid_reg;
    end
  end
  assign out = out_select;
  assign out_vld = out_valid_or;
  assign in_rdy = in_load_en;
endmodule
""", verilog)


if __name__ == '__main__':
  absltest.main()
