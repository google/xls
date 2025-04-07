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

// This module wraps the Memory Reader verilog sources generated from DSLX to
// form a DUT for verilog tests with consistent IO.

`default_nettype none

module mem_reader_wrapper #(
    parameter DSLX_DATA_W = 64,
    parameter DSLX_ADDR_W = 16,
    parameter AXI_DATA_W  = 128,
    parameter AXI_ADDR_W  = 16,
    parameter AXI_DEST_W  = 8,
    parameter AXI_ID_W    = 8,

    parameter CTRL_W   = (DSLX_ADDR_W),
    parameter REQ_W    = (2 * DSLX_ADDR_W),
    parameter RESP_W   = (1 + DSLX_DATA_W + DSLX_ADDR_W + 1),
    parameter AXI_AR_W = (AXI_ID_W + AXI_ADDR_W + 28),
    parameter AXI_R_W  = (AXI_ID_W + AXI_DATA_W + 4)
) (
    input wire clk,
    input wire rst,

    output wire             req_rdy,
    input  wire             req_vld,
    input  wire [REQ_W-1:0] req_data,

    output wire              resp_vld,
    input  wire              resp_rdy,
    output wire [RESP_W-1:0] resp_data,

    output wire                  axi_ar_arvalid,
    input  wire                  axi_ar_arready,
    output wire [  AXI_ID_W-1:0] axi_ar_arid,
    output wire [AXI_ADDR_W-1:0] axi_ar_araddr,
    output wire [           3:0] axi_ar_arregion,
    output wire [           7:0] axi_ar_arlen,
    output wire [           2:0] axi_ar_arsize,
    output wire [           1:0] axi_ar_arburst,
    output wire [           3:0] axi_ar_arcache,
    output wire [           2:0] axi_ar_arprot,
    output wire [           3:0] axi_ar_arqos,

    input  wire                  axi_r_rvalid,
    output wire                  axi_r_rready,
    input  wire [  AXI_ID_W-1:0] axi_r_rid,
    input  wire [AXI_DATA_W-1:0] axi_r_rdata,
    input  wire [           2:0] axi_r_rresp,
    input  wire                  axi_r_rlast
);

  wire [AXI_AR_W-1:0] axi_ar_data;
  wire                axi_ar_rdy;
  wire                axi_ar_vld;

  assign axi_ar_rdy = axi_ar_arready;

  assign axi_ar_arvalid = axi_ar_vld;
  assign {
    axi_ar_arid,
    axi_ar_araddr,
    axi_ar_arregion,
    axi_ar_arlen,
    axi_ar_arsize,
    axi_ar_arburst,
    axi_ar_arcache,
    axi_ar_arprot,
    axi_ar_arqos
} = axi_ar_data;

  wire [AXI_R_W-1:0] axi_r_data;
  wire               axi_r_vld;
  wire               axi_r_rdy;

  assign axi_r_data = {axi_r_rid, axi_r_rdata, axi_r_rresp, axi_r_rlast};
  assign axi_r_vld = axi_r_rvalid;

  assign axi_r_rready = axi_r_rdy;

  mem_reader_adv mem_reader_adv (
      .clk(clk),
      .rst(rst),

      .mem_reader__req_r_data(req_data),
      .mem_reader__req_r_rdy (req_rdy),
      .mem_reader__req_r_vld (req_vld),

      .mem_reader__resp_s_data(resp_data),
      .mem_reader__resp_s_rdy (resp_rdy),
      .mem_reader__resp_s_vld (resp_vld),

      .mem_reader__axi_ar_s_data(axi_ar_data),
      .mem_reader__axi_ar_s_rdy (axi_ar_rdy),
      .mem_reader__axi_ar_s_vld (axi_ar_vld),

      .mem_reader__axi_r_r_data(axi_r_data),
      .mem_reader__axi_r_r_vld (axi_r_vld),
      .mem_reader__axi_r_r_rdy (axi_r_rdy)
  );

endmodule
