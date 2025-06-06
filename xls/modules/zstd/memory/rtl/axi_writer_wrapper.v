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

// This module wraps the AXI Writer verilog sources generated from DSLX to
// form a DUT for verilog tests with consistent IO.

`default_nettype none

module axi_writer_wrapper (
    input wire clk,
    input wire rst,

    output wire write_resp_data,
    output wire write_resp_vld,
    input  wire write_resp_rdy,

    input wire [31:0] write_req_data,
    input wire write_req_vld,
    output wire write_req_rdy,

    input wire [31:0] axi_st_read_tdata,
    input wire [3:0] axi_st_read_tstr,
    input wire [3:0] axi_st_read_tkeep,
    input wire [0:0] axi_st_read_tlast,
    input wire [3:0] axi_st_read_tid,
    input wire [3:0] axi_st_read_tdest,
    input wire axi_st_read_tvalid,
    output wire axi_st_read_tready,

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

  wire [32:0] axi_writer__ch_axi_aw_data;
  wire [36:0] axi_writer__ch_axi_w_data;
  wire [ 6:0] axi_writer__ch_axi_b_data;

  wire [15:0] write_req_data_address;
  wire [15:0] write_req_data_length;

  wire [48:0] axi_st_read_data;

  assign {write_req_data_address, write_req_data_length} = write_req_data;

  assign { axi_aw_awid,
           axi_aw_awaddr,
           axi_aw_awsize,
           axi_aw_awlen,
           axi_aw_awburst } = axi_writer__ch_axi_aw_data;

  assign {axi_w_wdata, axi_w_wstrb, axi_w_wlast} = axi_writer__ch_axi_w_data;

  assign axi_writer__ch_axi_b_data = {axi_b_bresp, axi_b_bid};

  assign axi_st_read_data = {
    axi_st_read_tdata,
    axi_st_read_tstr,
    axi_st_read_tkeep,
    axi_st_read_tlast,
    axi_st_read_tid,
    axi_st_read_tdest
  };

  axi_writer axi_writer (
      .clk(clk),
      .rst(rst),

      .axi_writer__ch_write_req_data(write_req_data),
      .axi_writer__ch_write_req_rdy (write_req_rdy),
      .axi_writer__ch_write_req_vld (write_req_vld),

      .axi_writer__ch_write_resp_rdy (write_resp_rdy),
      .axi_writer__ch_write_resp_vld (write_resp_vld),
      .axi_writer__ch_write_resp_data(write_resp_data),

      .axi_writer__ch_axi_aw_data(axi_writer__ch_axi_aw_data),
      .axi_writer__ch_axi_aw_rdy (axi_aw_awready),
      .axi_writer__ch_axi_aw_vld (axi_aw_awvalid),

      .axi_writer__ch_axi_w_data(axi_writer__ch_axi_w_data),
      .axi_writer__ch_axi_w_rdy (axi_w_wready),
      .axi_writer__ch_axi_w_vld (axi_w_wvalid),

      .axi_writer__ch_axi_b_data(axi_writer__ch_axi_b_data),
      .axi_writer__ch_axi_b_rdy (axi_b_bready),
      .axi_writer__ch_axi_b_vld (axi_b_bvalid),

      .axi_writer__ch_axi_st_read_data(axi_st_read_data),
      .axi_writer__ch_axi_st_read_rdy (axi_st_read_tready),
      .axi_writer__ch_axi_st_read_vld (axi_st_read_tvalid)
  );


endmodule : axi_writer_wrapper
