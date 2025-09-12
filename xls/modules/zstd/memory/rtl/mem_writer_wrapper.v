// Copyright 2024 The XLS Authors
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

// This module wraps the Memory Writer verilog sources generated from DSLX to
// form a DUT for verilog tests with consistent IO.

`default_nettype none

module mem_writer_wrapper (
    input wire clk,
    input wire rst,

    input wire [31:0] req_data,
    input wire req_vld,
    output wire req_rdy,

    input wire [48:0] data_in_data,
    input wire data_in_vld,
    output wire data_in_rdy,

    output wire resp_data,
    output wire resp_vld,
    input  wire resp_rdy,

    output wire [3:0] axi_aw_awid,
    output wire [15:0] axi_aw_awaddr,
    output wire [2:0] axi_aw_awsize,
    output wire [7:0] axi_aw_awlen,
    output wire [1:0] axi_aw_awburst,
    output wire axi_aw_awvalid,
    input wire axi_aw_awready,

    output wire [31:0] axi_w_wdata,
    output wire [3:0] axi_w_wstrb,
    output wire [0:0] axi_w_wlast,
    output wire axi_w_wvalid,
    input wire axi_w_wready,

    input wire [2:0] axi_b_bresp,
    input wire [3:0] axi_b_bid,
    input wire axi_b_bvalid,
    output wire axi_b_bready
);

  wire [15:0] req_f_addr;
  wire [15:0] req_f_length;

  wire [31:0] data_in_f_data;
  wire [15:0] data_in_f_length;
  wire [0:0] data_in_f_last;

  wire [36:0] axi_w_data;
  wire axi_w_vld;
  wire axi_w_rdy;

  wire [32:0] axi_aw_data;
  wire axi_aw_vld;
  wire axi_aw_rdy;

  wire [6:0] axi_b_data;
  wire axi_b_rdy;
  wire axi_b_vld;

  assign {req_f_addr, req_f_length} = req_data;

  assign {data_in_f_data, data_in_f_length, data_in_f_last} = data_in_data;

  assign {axi_aw_awid, axi_aw_awaddr, axi_aw_awsize, axi_aw_awlen, axi_aw_awburst} = axi_aw_data;
  assign axi_aw_awvalid = axi_aw_vld;
  assign axi_aw_rdy = axi_aw_awready;

  assign {axi_w_wdata, axi_w_wstrb, axi_w_wlast} = axi_w_data;
  assign axi_w_wvalid = axi_w_vld;
  assign axi_w_rdy = axi_w_wready;

  assign axi_b_data = {axi_b_bresp, axi_b_bid};
  assign axi_b_vld = axi_b_bvalid;
  assign axi_b_bready = axi_b_rdy;

  mem_writer mem_writer (
      .clk(clk),
      .rst(rst),

      // MemWriter Write Request
      .mem_writer__req_in_r_data(req_data),
      .mem_writer__req_in_r_vld (req_vld),
      .mem_writer__req_in_r_rdy (req_rdy),

      // Data to write
      .mem_writer__data_in_r_data(data_in_data),
      .mem_writer__data_in_r_vld (data_in_vld),
      .mem_writer__data_in_r_rdy (data_in_rdy),

      // Response channel
      .mem_writer__resp_s_data(resp_data),
      .mem_writer__resp_s_rdy (resp_rdy),
      .mem_writer__resp_s_vld (resp_vld),

      // Memory AXI
      .mem_writer__axi_w_s_data(axi_w_data),
      .mem_writer__axi_w_s_vld (axi_w_vld),
      .mem_writer__axi_w_s_rdy (axi_w_rdy),

      .mem_writer__axi_aw_s_data(axi_aw_data),
      .mem_writer__axi_aw_s_vld (axi_aw_vld),
      .mem_writer__axi_aw_s_rdy (axi_aw_rdy),

      .mem_writer__axi_b_r_data(axi_b_data),
      .mem_writer__axi_b_r_vld (axi_b_vld),
      .mem_writer__axi_b_r_rdy (axi_b_rdy)
  );

endmodule : mem_writer_wrapper
