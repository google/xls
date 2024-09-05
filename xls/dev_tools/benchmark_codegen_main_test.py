#
# Copyright 2022 The XLS Authors
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

from absl.testing import absltest
from xls.common import runfiles
from xls.common import test_base

BENCHMARK_CODEGEN_MAIN_PATH = runfiles.get_path(
    'xls/dev_tools/benchmark_codegen_main'
)

OPT_IR = """package add

top fn my_function(a: bits[32], b: bits[32]) -> bits[32] {
  sum: bits[32] = add(a, b)
  not_sum: bits[32] = not(sum)
  ret not_not_sum: bits[32] = not(not_sum)
}
"""

BLOCK_IR = """package add

top block my_block(clk: clock, a: bits[32], b: bits[32], out: bits[32]) {
  reg a_reg(bits[32])
  reg b_reg(bits[32])
  reg sum_reg(bits[32])

  a: bits[32] = input_port(name=a)
  a_d: () = register_write(a, register=a_reg)
  a_q: bits[32] = register_read(register=a_reg)

  b: bits[32] = input_port(name=b)
  b_d: () = register_write(b, register=b_reg)
  b_q: bits[32] = register_read(register=b_reg)

  sum: bits[32] = add(a_q, b_q)
  sum_d: () = register_write(sum, register=sum_reg)
  sum_q: bits[32] = register_read(register=sum_reg)

  not_sum_q: bits[32] = not(sum_q)
  not_not_sum_q: bits[32] = not(not_sum_q)

  out: () = output_port(not_not_sum_q, name=out)
}
"""

SIMPLE_VERILOG = """module main(
  input wire [31:0] x,
  output wire [31:0] out
);
  assign out = x;
endmodule
"""

COMBINATIONAL_BLOCK_IR = """package add

top block my_function(a: bits[32], b: bits[32], out: bits[32]) {
  a: bits[32] = input_port(name=a, id=6)
  b: bits[32] = input_port(name=b, id=7)
  sum: bits[32] = add(a, b, id=8)
  not_sum: bits[32] = not(sum, id=9)
  not_not_sum: bits[32] = not(not_sum, id=10)
  out: () = output_port(not_not_sum, name=out, id=11)
}
"""

COMBINATIONAL_SIMPLE_VERILOG = """module my_function(
  input wire [31:0] a,
  input wire [31:0] b,
  output wire [31:0] out
);
  wire [31:0] sum;
  wire [31:0] not_sum;
  wire [31:0] not_not_sum;
  assign sum = a + b;
  assign not_sum = ~sum;
  assign not_not_sum = ~not_sum;
  assign out = not_not_sum;
endmodule
"""


class CodeGenMainTest(test_base.TestCase):

  def test_simple_block(self):
    opt_ir_file = self.create_tempfile(content=OPT_IR)
    block_ir_file = self.create_tempfile(content=BLOCK_IR)
    verilog_file = self.create_tempfile(content=SIMPLE_VERILOG)
    output = subprocess.check_output([
        BENCHMARK_CODEGEN_MAIN_PATH,
        '--delay_model=unit',
        '--clock_period_ps=10',
        opt_ir_file.full_path,
        block_ir_file.full_path,
        verilog_file.full_path,
    ]).decode('utf-8')

    self.assertIn('Flop count: 96', output)
    self.assertIn('Has feedthrough path: false', output)
    self.assertIn('Max reg-to-reg delay: 1ps', output)
    self.assertIn('Max input-to-reg delay: 0ps', output)
    self.assertIn('Max reg-to-output delay: 2ps', output)
    self.assertIn('Lines of Verilog: 7', output)
    self.assertIn('Codegen time:', output)
    self.assertIn('Scheduling time:', output)

  def test_simple_block_no_delay_model(self):
    opt_ir_file = self.create_tempfile(content=OPT_IR)
    block_ir_file = self.create_tempfile(content=BLOCK_IR)
    verilog_file = self.create_tempfile(content=SIMPLE_VERILOG)
    output = subprocess.check_output([
        BENCHMARK_CODEGEN_MAIN_PATH,
        '--measure_codegen_timing=false',
        opt_ir_file.full_path,
        block_ir_file.full_path,
        verilog_file.full_path,
    ]).decode('utf-8')

    self.assertIn('Flop count: 96', output)
    self.assertIn('Has feedthrough path: false', output)
    self.assertNotIn('Max reg-to-reg delays', output)
    self.assertNotIn('Max input-to-reg delay', output)
    self.assertNotIn('Max reg-to-output delay', output)
    self.assertIn('Lines of Verilog: 7', output)

  def test_simple_block_combinational(self):
    opt_ir_file = self.create_tempfile(content=OPT_IR)
    block_ir_file = self.create_tempfile(content=COMBINATIONAL_BLOCK_IR)
    verilog_file = self.create_tempfile(content=COMBINATIONAL_SIMPLE_VERILOG)
    output = subprocess.check_output([
        BENCHMARK_CODEGEN_MAIN_PATH,
        opt_ir_file.full_path,
        block_ir_file.full_path,
        verilog_file.full_path,
        '--generator=combinational',
    ]).decode('utf-8')
    self.assertIn('Codegen time:', output)
    self.assertNotIn('Scheduling time:', output)
    self.assertIn('Flop count: 0', output)
    self.assertIn('Has feedthrough path: true', output)
    self.assertIn('Lines of Verilog: 14', output)
    self.assertNotIn('Max reg-to-reg delay: 1ps', output)
    self.assertNotIn('Max input-to-reg delay: 0ps', output)
    self.assertNotIn('Max reg-to-output delay: 2ps', output)


if __name__ == '__main__':
  absltest.main()
