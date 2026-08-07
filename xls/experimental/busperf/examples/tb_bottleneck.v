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

// Shared testbench for bottleneck_no_stall and bottleneck_stall, driving
// data_r/data_s at full throttle. DUT_MODULE, TB_MODULE, and VCD_NAME come
// from xls_busperf_setup as `iverilog -D...` flags.

`timescale 1ns/1ps

module `TB_MODULE;
  reg clk = 0;
  reg rst = 1'b1;

  wire [31:0] data_r = 32'hDEADBEEF;
  wire data_r_vld = !rst;
  wire data_r_rdy;

  wire [31:0] data_s;
  wire data_s_vld;
  wire data_s_rdy = !rst;

  `DUT_MODULE dut (
      .clk(clk),
      .rst(rst),
      ._data_r(data_r),
      ._data_r_vld(data_r_vld),
      ._data_r_rdy(data_r_rdy),
      ._data_s(data_s),
      ._data_s_vld(data_s_vld),
      ._data_s_rdy(data_s_rdy)
  );

  always #5 clk = ~clk;

  initial begin
    $dumpfile(`VCD_NAME);
    $dumpvars(0, `TB_MODULE);

    repeat (4) @(posedge clk);
    rst = 1'b0;

    repeat (2000) @(posedge clk);
    $finish;
  end
endmodule
