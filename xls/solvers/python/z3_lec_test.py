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
"""Tests for xls.solvers.python.z3_lec."""

import sys

from xls.common import runfiles
from xls.common.python import init_xls
from xls.solvers.python import z3_lec
from absl.testing import absltest


def setUpModule():
  init_xls.init_xls(sys.argv)


class Z3LecTest(absltest.TestCase):

  def test_simple_lec(self):
    ir_text = """package p
    top fn main(a: bits[2], b: bits[2]) -> bits[2] {
      ret add.3: bits[2] = add(a, b)
    }
    """

    netlist_text = """module main(clk, a_1_, a_0_, b_1_, b_0_, out_1_, out_0_);
      input clk, a_1_, a_0_, b_1_, b_0_;
      output out_1_, out_0_;
      wire p0_a_1_, p0_a_0_, p0_b_1_, p0_b_0_, p0_add_3_comb_0_, p0_add_3_comb_1_, carry, high;

      DFF p0_a_reg_1_ ( .D(a_1_), .CLK(clk), .Q(p0_a_1_) );
      DFF p0_a_reg_0_ ( .D(a_0_), .CLK(clk), .Q(p0_a_0_) );
      DFF p0_b_reg_1_ ( .D(b_1_), .CLK(clk), .Q(p0_b_1_) );
      DFF p0_b_reg_0_ ( .D(b_0_), .CLK(clk), .Q(p0_b_0_) );

      XOR out_0_cell ( .A(p0_a_0_), .B(p0_b_0_), .Z(p0_add_3_comb_0_) );

      AND carry_cell ( .A(p0_a_0_), .B(p0_b_0_), .Z(carry) );
      XOR high_cell ( .A(p0_a_1_), .B(p0_b_1_), .Z(high) );
      XOR out_1_cell ( .A(high), .B(carry), .Z(p0_add_3_comb_1_) );

      DFF p0_add_3_reg_1_ ( .D(p0_add_3_comb_1_), .CLK(clk), .Q(out_1_) );
      DFF p0_add_3_reg_0_ ( .D(p0_add_3_comb_0_), .CLK(clk), .Q(out_0_) );
    endmodule
    """

    proto_text = runfiles.get_contents_as_text(
        'xls/netlist/fake_cell_library.textproto')
    result = z3_lec.run(ir_text, netlist_text, 'main', proto_text)
    self.assertTrue(result)


if __name__ == '__main__':
  absltest.main()
