// Copyright 2026 The XLS Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

`timescale 1ns/1ps

module tb_hier;
  reg clk = 0;
  reg rst = 1'b1;

  wire [31:0] in0 = 32'hDEADBEEF;
  wire in0_vld = !rst;
  wire in0_rdy;
  wire [31:0] out0;
  wire out0_vld;
  wire out0_rdy = !rst;

  wire [31:0] in1 = 32'hDEADBEEF;
  wire in1_vld = !rst;
  wire in1_rdy;
  wire [31:0] out1;
  wire out1_vld;
  wire out1_rdy = !rst;

  hier dut (
      .clk(clk),
      .rst(rst),
      ._in0(in0),
      ._in0_vld(in0_vld),
      ._in0_rdy(in0_rdy),
      ._out0(out0),
      ._out0_vld(out0_vld),
      ._out0_rdy(out0_rdy),
      ._in1(in1),
      ._in1_vld(in1_vld),
      ._in1_rdy(in1_rdy),
      ._out1(out1),
      ._out1_vld(out1_vld),
      ._out1_rdy(out1_rdy)
  );

  always #5 clk = ~clk;

  initial begin
    $dumpfile("hier.vcd");
    $dumpvars(0, tb_hier);

    repeat (4) @(posedge clk);
    rst = 1'b0;

    repeat (2000) @(posedge clk);
    $finish;
  end
endmodule
