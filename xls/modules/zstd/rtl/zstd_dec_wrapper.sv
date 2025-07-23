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

// This module wraps the ZSTD Decoder verilog sources generated from DSLX to
// form a DUT with consistent IO.
// The wrapper also contains an AXI crossbar module for merging the traffic
// from multiple AXI Managers comming from the ZSTD Decoder into a single AXI
// master that could be e.g. routed to the memory.

`default_nettype none

module zstd_dec_wrapper #(
    parameter int AXI_DATA_W  = 64,
    parameter int AXI_ADDR_W  = 32,
    parameter int S_AXI_ID_W  = 4,
    parameter int M_AXI_ID_W  = 8,
    parameter int AXI_STRB_W  = 8,
    parameter int AWUSER_WIDTH = 1,
    parameter int WUSER_WIDTH = 1,
    parameter int BUSER_WIDTH = 1,
    parameter int ARUSER_WIDTH = 1,
    parameter int RUSER_WIDTH = 1
) (
    input wire clk,
    input wire rst,

    // AXI Manager interface for the memory connection
    output wire [M_AXI_ID_W-1:0]    memory_axi_aw_awid,
    output wire [AXI_ADDR_W-1:0]    memory_axi_aw_awaddr,
    output wire [7:0]               memory_axi_aw_awlen,
    output wire [2:0]               memory_axi_aw_awsize,
    output wire [1:0]               memory_axi_aw_awburst,
    output wire                     memory_axi_aw_awlock,
    output wire [3:0]               memory_axi_aw_awcache,
    output wire [2:0]               memory_axi_aw_awprot,
    output wire [3:0]               memory_axi_aw_awqos,
    output wire [3:0]               memory_axi_aw_awregion,
    output wire [AWUSER_WIDTH-1:0]  memory_axi_aw_awuser,
    output wire                     memory_axi_aw_awvalid,
    input  wire                     memory_axi_aw_awready,
    output wire [AXI_DATA_W-1:0]    memory_axi_w_wdata,
    output wire [AXI_STRB_W-1:0]    memory_axi_w_wstrb,
    output wire                     memory_axi_w_wlast,
    output wire [WUSER_WIDTH-1:0]   memory_axi_w_wuser,
    output wire                     memory_axi_w_wvalid,
    input  wire                     memory_axi_w_wready,
    input  wire [M_AXI_ID_W-1:0]    memory_axi_b_bid,
    input  wire [2:0]               memory_axi_b_bresp,
    input  wire [BUSER_WIDTH-1:0]   memory_axi_b_buser,
    input  wire                     memory_axi_b_bvalid,
    output wire                     memory_axi_b_bready,
    output wire [M_AXI_ID_W-1:0]    memory_axi_ar_arid,
    output wire [AXI_ADDR_W-1:0]    memory_axi_ar_araddr,
    output wire [7:0]               memory_axi_ar_arlen,
    output wire [2:0]               memory_axi_ar_arsize,
    output wire [1:0]               memory_axi_ar_arburst,
    output wire                     memory_axi_ar_arlock,
    output wire [3:0]               memory_axi_ar_arcache,
    output wire [2:0]               memory_axi_ar_arprot,
    output wire [3:0]               memory_axi_ar_arqos,
    output wire [3:0]               memory_axi_ar_arregion,
    output wire [ARUSER_WIDTH-1:0]  memory_axi_ar_aruser,
    output wire                     memory_axi_ar_arvalid,
    input  wire                     memory_axi_ar_arready,
    input  wire [M_AXI_ID_W-1:0]    memory_axi_r_rid,
    input  wire [AXI_DATA_W-1:0]    memory_axi_r_rdata,
    input  wire [2:0]               memory_axi_r_rresp,
    input  wire                     memory_axi_r_rlast,
    input  wire [RUSER_WIDTH-1:0]   memory_axi_r_ruser,
    input  wire                     memory_axi_r_rvalid,
    output wire                     memory_axi_r_rready,

    // AXI Subordinate interface for the CSR access
    input wire [S_AXI_ID_W-1:0]     csr_axi_aw_awid,
    input wire [AXI_ADDR_W-1:0]     csr_axi_aw_awaddr,
    input wire [7:0]                csr_axi_aw_awlen,
    input wire [2:0]                csr_axi_aw_awsize,
    input wire [1:0]                csr_axi_aw_awburst,
    input wire                      csr_axi_aw_awlock,
    input wire [3:0]                csr_axi_aw_awcache,
    input wire [2:0]                csr_axi_aw_awprot,
    input wire [3:0]                csr_axi_aw_awqos,
    input wire [3:0]                csr_axi_aw_awregion,
    input wire [AWUSER_WIDTH-1:0]   csr_axi_aw_awuser,
    input wire                      csr_axi_aw_awvalid,
    output wire                     csr_axi_aw_awready,
    input wire [AXI_DATA_W-1:0]     csr_axi_w_wdata,
    input wire [AXI_STRB_W-1:0]     csr_axi_w_wstrb,
    input wire                      csr_axi_w_wlast,
    input wire [WUSER_WIDTH-1:0]    csr_axi_w_wuser,
    input wire                      csr_axi_w_wvalid,
    output wire                     csr_axi_w_wready,
    output wire [S_AXI_ID_W-1:0]    csr_axi_b_bid,
    output wire [2:0]               csr_axi_b_bresp,
    output wire [BUSER_WIDTH-1:0]   csr_axi_b_buser,
    output wire                     csr_axi_b_bvalid,
    input wire                      csr_axi_b_bready,
    input wire [S_AXI_ID_W-1:0]     csr_axi_ar_arid,
    input wire [AXI_ADDR_W-1:0]     csr_axi_ar_araddr,
    input wire [7:0]                csr_axi_ar_arlen,
    input wire [2:0]                csr_axi_ar_arsize,
    input wire [1:0]                csr_axi_ar_arburst,
    input wire                      csr_axi_ar_arlock,
    input wire [3:0]                csr_axi_ar_arcache,
    input wire [2:0]                csr_axi_ar_arprot,
    input wire [3:0]                csr_axi_ar_arqos,
    input wire [3:0]                csr_axi_ar_arregion,
    input wire [ARUSER_WIDTH-1:0]   csr_axi_ar_aruser,
    input wire                      csr_axi_ar_arvalid,
    output wire                     csr_axi_ar_arready,
    output wire [S_AXI_ID_W-1:0]    csr_axi_r_rid,
    output wire [AXI_DATA_W-1:0]    csr_axi_r_rdata,
    output wire [2:0]               csr_axi_r_rresp,
    output wire                     csr_axi_r_rlast,
    output wire [RUSER_WIDTH-1:0]   csr_axi_r_ruser,
    output wire                     csr_axi_r_rvalid,
    input wire                      csr_axi_r_rready,

    output wire                     notify_data,
    output wire                     notify_vld,
    input  wire                     notify_rdy
);

  /*
   * MemReader AXI interfaces
   */
  // RawBlockDecoder
  wire                  raw_block_decoder_axi_ar_arvalid;
  wire                  raw_block_decoder_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] raw_block_decoder_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] raw_block_decoder_axi_ar_araddr;
  wire [           3:0] raw_block_decoder_axi_ar_arregion;
  wire [           7:0] raw_block_decoder_axi_ar_arlen;
  wire [           2:0] raw_block_decoder_axi_ar_arsize;
  wire [           1:0] raw_block_decoder_axi_ar_arburst;
  wire [           3:0] raw_block_decoder_axi_ar_arcache;
  wire [           2:0] raw_block_decoder_axi_ar_arprot;
  wire [           3:0] raw_block_decoder_axi_ar_arqos;

  wire                  raw_block_decoder_axi_r_rvalid;
  wire                  raw_block_decoder_axi_r_rready;
  wire [S_AXI_ID_W-1:0] raw_block_decoder_axi_r_rid;
  wire [AXI_DATA_W-1:0] raw_block_decoder_axi_r_rdata;
  wire [           2:0] raw_block_decoder_axi_r_rresp;
  wire                  raw_block_decoder_axi_r_rlast;


  // BlockHeaderDecoder
  wire                  block_header_decoder_axi_ar_arvalid;
  wire                  block_header_decoder_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] block_header_decoder_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] block_header_decoder_axi_ar_araddr;
  wire [           3:0] block_header_decoder_axi_ar_arregion;
  wire [           7:0] block_header_decoder_axi_ar_arlen;
  wire [           2:0] block_header_decoder_axi_ar_arsize;
  wire [           1:0] block_header_decoder_axi_ar_arburst;
  wire [           3:0] block_header_decoder_axi_ar_arcache;
  wire [           2:0] block_header_decoder_axi_ar_arprot;
  wire [           3:0] block_header_decoder_axi_ar_arqos;

  wire                  block_header_decoder_axi_r_rvalid;
  wire                  block_header_decoder_axi_r_rready;
  wire [S_AXI_ID_W-1:0] block_header_decoder_axi_r_rid;
  wire [AXI_DATA_W-1:0] block_header_decoder_axi_r_rdata;
  wire [           2:0] block_header_decoder_axi_r_rresp;
  wire                  block_header_decoder_axi_r_rlast;


  // FrameHeaderDecoder
  wire                  frame_header_decoder_axi_ar_arvalid;
  wire                  frame_header_decoder_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] frame_header_decoder_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] frame_header_decoder_axi_ar_araddr;
  wire [           3:0] frame_header_decoder_axi_ar_arregion;
  wire [           7:0] frame_header_decoder_axi_ar_arlen;
  wire [           2:0] frame_header_decoder_axi_ar_arsize;
  wire [           1:0] frame_header_decoder_axi_ar_arburst;
  wire [           3:0] frame_header_decoder_axi_ar_arcache;
  wire [           2:0] frame_header_decoder_axi_ar_arprot;
  wire [           3:0] frame_header_decoder_axi_ar_arqos;

  wire                  frame_header_decoder_axi_r_rvalid;
  wire                  frame_header_decoder_axi_r_rready;
  wire [S_AXI_ID_W-1:0] frame_header_decoder_axi_r_rid;
  wire [AXI_DATA_W-1:0] frame_header_decoder_axi_r_rdata;
  wire [           2:0] frame_header_decoder_axi_r_rresp;
  wire                  frame_header_decoder_axi_r_rlast;


  /*
   * MemWriter AXI interfaces
   */

  // Output Writer
  wire [S_AXI_ID_W-1:0] output_axi_aw_awid;
  wire [AXI_ADDR_W-1:0] output_axi_aw_awaddr;
  wire [           2:0] output_axi_aw_awsize;
  wire [           7:0] output_axi_aw_awlen;
  wire [           1:0] output_axi_aw_awburst;
  wire                  output_axi_aw_awvalid;
  wire                  output_axi_aw_awready;

  wire [AXI_DATA_W-1:0] output_axi_w_wdata;
  wire [AXI_STRB_W-1:0] output_axi_w_wstrb;
  wire                  output_axi_w_wlast;
  wire                  output_axi_w_wvalid;
  wire                  output_axi_w_wready;

  wire [S_AXI_ID_W-1:0] output_axi_b_bid;
  wire [           2:0] output_axi_b_bresp;
  wire                  output_axi_b_bvalid;
  wire                  output_axi_b_bready;

  /*
   * XLS Channels representing AXI interfaces
   */

  localparam int XlsAxiAwW = AXI_ADDR_W + S_AXI_ID_W + 3 + 2 + 8;
  localparam int XlsAxiWW = AXI_DATA_W + AXI_STRB_W + 1;
  localparam int XlsAxiBW = 3 + S_AXI_ID_W;
  localparam int XlsAxiArW = S_AXI_ID_W + AXI_ADDR_W + 4 + 8 + 3 + 2 + 4 + 3 + 4;
  localparam int XlsAxiRW = S_AXI_ID_W + AXI_DATA_W + 3 + 1;
  // CSR
  wire [XlsAxiAwW-1:0] zstd_dec__csr_axi_aw;
  wire                 zstd_dec__csr_axi_aw_rdy;
  wire                 zstd_dec__csr_axi_aw_vld;
  wire [XlsAxiWW-1:0]  zstd_dec__csr_axi_w;
  wire                 zstd_dec__csr_axi_w_rdy;
  wire                 zstd_dec__csr_axi_w_vld;
  wire [ XlsAxiBW-1:0] zstd_dec__csr_axi_b;
  wire                 zstd_dec__csr_axi_b_rdy;
  wire                 zstd_dec__csr_axi_b_vld;
  wire [XlsAxiArW-1:0] zstd_dec__csr_axi_ar;
  wire                 zstd_dec__csr_axi_ar_rdy;
  wire                 zstd_dec__csr_axi_ar_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__csr_axi_r;
  wire                 zstd_dec__csr_axi_r_rdy;
  wire                 zstd_dec__csr_axi_r_vld;

  // Frame Header Decoder
  wire [XlsAxiArW-1:0] zstd_dec__fh_axi_ar;
  wire                 zstd_dec__fh_axi_ar_rdy;
  wire                 zstd_dec__fh_axi_ar_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__fh_axi_r;
  wire                 zstd_dec__fh_axi_r_rdy;
  wire                 zstd_dec__fh_axi_r_vld;

  // Block Header Decoder
  wire [XlsAxiArW-1:0] zstd_dec__bh_axi_ar;
  wire                 zstd_dec__bh_axi_ar_rdy;
  wire                 zstd_dec__bh_axi_ar_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__bh_axi_r;
  wire                 zstd_dec__bh_axi_r_rdy;
  wire                 zstd_dec__bh_axi_r_vld;

  // Raw Block Decoder
  wire [XlsAxiArW-1:0] zstd_dec__raw_axi_ar;
  wire                 zstd_dec__raw_axi_ar_rdy;
  wire                 zstd_dec__raw_axi_ar_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__raw_axi_r;
  wire                 zstd_dec__raw_axi_r_rdy;
  wire                 zstd_dec__raw_axi_r_vld;

  // Output Memory Interface
  wire [XlsAxiAwW-1:0] zstd_dec__output_axi_aw;
  wire                 zstd_dec__output_axi_aw_rdy;
  wire                 zstd_dec__output_axi_aw_vld;
  wire [XlsAxiWW-1:0]  zstd_dec__output_axi_w;
  wire                 zstd_dec__output_axi_w_rdy;
  wire                 zstd_dec__output_axi_w_vld;
  wire [XlsAxiBW-1:0]  zstd_dec__output_axi_b;
  wire                 zstd_dec__output_axi_b_rdy;
  wire                 zstd_dec__output_axi_b_vld;

  /*
   * Mapping XLS Channels to AXI channels fields
   */

  // CSR
  assign zstd_dec__csr_axi_aw = {
      csr_axi_aw_awid,
      csr_axi_aw_awaddr,
      csr_axi_aw_awsize,
      csr_axi_aw_awlen,
      csr_axi_aw_awburst
      };
  assign  zstd_dec__csr_axi_aw_vld = csr_axi_aw_awvalid;
  assign csr_axi_aw_awready = zstd_dec__csr_axi_aw_rdy;
  assign zstd_dec__csr_axi_w = {
      csr_axi_w_wdata,
      csr_axi_w_wstrb,
      csr_axi_w_wlast
      };
  assign zstd_dec__csr_axi_w_vld = csr_axi_w_wvalid;
  assign csr_axi_w_wready = zstd_dec__csr_axi_w_rdy;
  assign {
      csr_axi_b_bresp,
      csr_axi_b_bid
      } = zstd_dec__csr_axi_b;
  assign csr_axi_b_bvalid = zstd_dec__csr_axi_b_vld;
  assign zstd_dec__csr_axi_b_rdy = csr_axi_b_bready;
  assign zstd_dec__csr_axi_ar = {
      csr_axi_ar_arid,
      csr_axi_ar_araddr,
      csr_axi_ar_arregion,
      csr_axi_ar_arlen,
      csr_axi_ar_arsize,
      csr_axi_ar_arburst,
      csr_axi_ar_arcache,
      csr_axi_ar_arprot,
      csr_axi_ar_arqos
      };
  assign zstd_dec__csr_axi_ar_vld = csr_axi_ar_arvalid;
  assign csr_axi_ar_arready = zstd_dec__csr_axi_ar_rdy;
  assign {
      csr_axi_r_rid,
      csr_axi_r_rdata,
      csr_axi_r_rresp,
      csr_axi_r_rlast
      } = zstd_dec__csr_axi_r;
  assign csr_axi_r_rvalid = zstd_dec__csr_axi_r_vld;
  assign zstd_dec__csr_axi_r_rdy = csr_axi_r_rready;

  // Frame Header Decoder
  assign {
      frame_header_decoder_axi_ar_arid,
      frame_header_decoder_axi_ar_araddr,
      frame_header_decoder_axi_ar_arregion,
      frame_header_decoder_axi_ar_arlen,
      frame_header_decoder_axi_ar_arsize,
      frame_header_decoder_axi_ar_arburst,
      frame_header_decoder_axi_ar_arcache,
      frame_header_decoder_axi_ar_arprot,
      frame_header_decoder_axi_ar_arqos
      } = zstd_dec__fh_axi_ar;
  assign frame_header_decoder_axi_ar_arvalid = zstd_dec__fh_axi_ar_vld;
  assign zstd_dec__fh_axi_ar_rdy = frame_header_decoder_axi_ar_arready;
  assign zstd_dec__fh_axi_r = {
      frame_header_decoder_axi_r_rid,
      frame_header_decoder_axi_r_rdata,
      frame_header_decoder_axi_r_rresp,
      frame_header_decoder_axi_r_rlast};
  assign zstd_dec__fh_axi_r_vld = frame_header_decoder_axi_r_rvalid;
  assign frame_header_decoder_axi_r_rready = zstd_dec__fh_axi_r_rdy;

  // Block Header Decoder
  assign {
      block_header_decoder_axi_ar_arid,
      block_header_decoder_axi_ar_araddr,
      block_header_decoder_axi_ar_arregion,
      block_header_decoder_axi_ar_arlen,
      block_header_decoder_axi_ar_arsize,
      block_header_decoder_axi_ar_arburst,
      block_header_decoder_axi_ar_arcache,
      block_header_decoder_axi_ar_arprot,
      block_header_decoder_axi_ar_arqos
      } = zstd_dec__bh_axi_ar;
  assign block_header_decoder_axi_ar_arvalid = zstd_dec__bh_axi_ar_vld;
  assign zstd_dec__bh_axi_ar_rdy = block_header_decoder_axi_ar_arready;
  assign zstd_dec__bh_axi_r = {
      block_header_decoder_axi_r_rid,
      block_header_decoder_axi_r_rdata,
      block_header_decoder_axi_r_rresp,
      block_header_decoder_axi_r_rlast};
  assign zstd_dec__bh_axi_r_vld = block_header_decoder_axi_r_rvalid;
  assign block_header_decoder_axi_r_rready = zstd_dec__bh_axi_r_rdy;

  // Raw Block Decoder
  assign {
      raw_block_decoder_axi_ar_arid,
      raw_block_decoder_axi_ar_araddr,
      raw_block_decoder_axi_ar_arregion,
      raw_block_decoder_axi_ar_arlen,
      raw_block_decoder_axi_ar_arsize,
      raw_block_decoder_axi_ar_arburst,
      raw_block_decoder_axi_ar_arcache,
      raw_block_decoder_axi_ar_arprot,
      raw_block_decoder_axi_ar_arqos
      } = zstd_dec__raw_axi_ar;
  assign raw_block_decoder_axi_ar_arvalid = zstd_dec__raw_axi_ar_vld;
  assign zstd_dec__raw_axi_ar_rdy = raw_block_decoder_axi_ar_arready;
  assign zstd_dec__raw_axi_r = {
      raw_block_decoder_axi_r_rid,
      raw_block_decoder_axi_r_rdata,
      raw_block_decoder_axi_r_rresp,
      raw_block_decoder_axi_r_rlast};
  assign zstd_dec__raw_axi_r_vld = raw_block_decoder_axi_r_rvalid;
  assign raw_block_decoder_axi_r_rready = zstd_dec__raw_axi_r_rdy;

  // Output Writer
  assign {
      output_axi_aw_awid,
      output_axi_aw_awaddr,
      output_axi_aw_awsize,
      output_axi_aw_awlen,
      output_axi_aw_awburst
      } = zstd_dec__output_axi_aw;
  assign output_axi_aw_awvalid = zstd_dec__output_axi_aw_vld;
  assign zstd_dec__output_axi_aw_rdy = output_axi_aw_awready;
  assign {
      output_axi_w_wdata,
      output_axi_w_wstrb,
      output_axi_w_wlast
      } = zstd_dec__output_axi_w;
  assign output_axi_w_wvalid = zstd_dec__output_axi_w_vld;
  assign zstd_dec__output_axi_w_rdy = output_axi_w_wready;
  assign zstd_dec__output_axi_b = {
      output_axi_b_bresp,
      output_axi_b_bid
      };
  assign zstd_dec__output_axi_b_vld = output_axi_b_bvalid;
  assign output_axi_b_bready = zstd_dec__output_axi_b_rdy;

  assign csr_axi_b_buser = 1'b0;
  assign csr_axi_r_ruser = 1'b0;
  assign notify_data = notify_vld;

  // Axi Subordinate Interface for axi_ram_s__0
  wire                  axi_ram_s__0_axi_ar_arvalid;
  wire                  axi_ram_s__0_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__0_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__0_axi_ar_araddr;
  wire [           3:0] axi_ram_s__0_axi_ar_arregion;
  wire [           7:0] axi_ram_s__0_axi_ar_arlen;
  wire [           2:0] axi_ram_s__0_axi_ar_arsize;
  wire [           1:0] axi_ram_s__0_axi_ar_arburst;
  wire [           3:0] axi_ram_s__0_axi_ar_arcache;
  wire [           2:0] axi_ram_s__0_axi_ar_arprot;
  wire [           3:0] axi_ram_s__0_axi_ar_arqos;

  wire                  axi_ram_s__0_axi_r_rvalid;
  wire                  axi_ram_s__0_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__0_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__0_axi_r_rdata;
  wire [           2:0] axi_ram_s__0_axi_r_rresp;
  wire                  axi_ram_s__0_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__0;
  wire                 zstd_dec__axi_ram_ar_s__0_rdy;
  wire                 zstd_dec__axi_ram_ar_s__0_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__0;
  wire                 zstd_dec__axi_ram_r_r__0_rdy;
  wire                 zstd_dec__axi_ram_r_r__0_vld;

  assign {
      axi_ram_s__0_axi_ar_arid,
      axi_ram_s__0_axi_ar_araddr,
      axi_ram_s__0_axi_ar_arregion,
      axi_ram_s__0_axi_ar_arlen,
      axi_ram_s__0_axi_ar_arsize,
      axi_ram_s__0_axi_ar_arburst,
      axi_ram_s__0_axi_ar_arcache,
      axi_ram_s__0_axi_ar_arprot,
      axi_ram_s__0_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__0;
  assign axi_ram_s__0_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__0_vld;
  assign zstd_dec__axi_ram_ar_s__0_rdy = axi_ram_s__0_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__0 = {
      axi_ram_s__0_axi_r_rid,
      axi_ram_s__0_axi_r_rdata,
      axi_ram_s__0_axi_r_rresp,
      axi_ram_s__0_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__0_vld = axi_ram_s__0_axi_r_rvalid;
  assign axi_ram_s__0_axi_r_rready = zstd_dec__axi_ram_r_r__0_rdy;
  assign axi_ram_s__0_axi_r_rresp[2] = '0;

  // Axi Subordinate Interface for axi_ram_s__1
  wire                  axi_ram_s__1_axi_ar_arvalid;
  wire                  axi_ram_s__1_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__1_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__1_axi_ar_araddr;
  wire [           3:0] axi_ram_s__1_axi_ar_arregion;
  wire [           7:0] axi_ram_s__1_axi_ar_arlen;
  wire [           2:0] axi_ram_s__1_axi_ar_arsize;
  wire [           1:0] axi_ram_s__1_axi_ar_arburst;
  wire [           3:0] axi_ram_s__1_axi_ar_arcache;
  wire [           2:0] axi_ram_s__1_axi_ar_arprot;
  wire [           3:0] axi_ram_s__1_axi_ar_arqos;

  wire                  axi_ram_s__1_axi_r_rvalid;
  wire                  axi_ram_s__1_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__1_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__1_axi_r_rdata;
  wire [           2:0] axi_ram_s__1_axi_r_rresp;
  wire                  axi_ram_s__1_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__1;
  wire                 zstd_dec__axi_ram_ar_s__1_rdy;
  wire                 zstd_dec__axi_ram_ar_s__1_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__1;
  wire                 zstd_dec__axi_ram_r_r__1_rdy;
  wire                 zstd_dec__axi_ram_r_r__1_vld;

  assign {
      axi_ram_s__1_axi_ar_arid,
      axi_ram_s__1_axi_ar_araddr,
      axi_ram_s__1_axi_ar_arregion,
      axi_ram_s__1_axi_ar_arlen,
      axi_ram_s__1_axi_ar_arsize,
      axi_ram_s__1_axi_ar_arburst,
      axi_ram_s__1_axi_ar_arcache,
      axi_ram_s__1_axi_ar_arprot,
      axi_ram_s__1_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__1;
  assign axi_ram_s__1_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__1_vld;
  assign zstd_dec__axi_ram_ar_s__1_rdy = axi_ram_s__1_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__1 = {
      axi_ram_s__1_axi_r_rid,
      axi_ram_s__1_axi_r_rdata,
      axi_ram_s__1_axi_r_rresp,
      axi_ram_s__1_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__1_vld = axi_ram_s__1_axi_r_rvalid;
  assign axi_ram_s__1_axi_r_rready = zstd_dec__axi_ram_r_r__1_rdy;
  assign axi_ram_s__1_axi_r_rresp[2] = '0;


  // Axi Subordinate Interface for axi_ram_s__2
  wire                  axi_ram_s__2_axi_ar_arvalid;
  wire                  axi_ram_s__2_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__2_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__2_axi_ar_araddr;
  wire [           3:0] axi_ram_s__2_axi_ar_arregion;
  wire [           7:0] axi_ram_s__2_axi_ar_arlen;
  wire [           2:0] axi_ram_s__2_axi_ar_arsize;
  wire [           1:0] axi_ram_s__2_axi_ar_arburst;
  wire [           3:0] axi_ram_s__2_axi_ar_arcache;
  wire [           2:0] axi_ram_s__2_axi_ar_arprot;
  wire [           3:0] axi_ram_s__2_axi_ar_arqos;

  wire                  axi_ram_s__2_axi_r_rvalid;
  wire                  axi_ram_s__2_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__2_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__2_axi_r_rdata;
  wire [           2:0] axi_ram_s__2_axi_r_rresp;
  wire                  axi_ram_s__2_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__2;
  wire                 zstd_dec__axi_ram_ar_s__2_rdy;
  wire                 zstd_dec__axi_ram_ar_s__2_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__2;
  wire                 zstd_dec__axi_ram_r_r__2_rdy;
  wire                 zstd_dec__axi_ram_r_r__2_vld;

  assign {
      axi_ram_s__2_axi_ar_arid,
      axi_ram_s__2_axi_ar_araddr,
      axi_ram_s__2_axi_ar_arregion,
      axi_ram_s__2_axi_ar_arlen,
      axi_ram_s__2_axi_ar_arsize,
      axi_ram_s__2_axi_ar_arburst,
      axi_ram_s__2_axi_ar_arcache,
      axi_ram_s__2_axi_ar_arprot,
      axi_ram_s__2_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__2;
  assign axi_ram_s__2_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__2_vld;
  assign zstd_dec__axi_ram_ar_s__2_rdy = axi_ram_s__2_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__2 = {
      axi_ram_s__2_axi_r_rid,
      axi_ram_s__2_axi_r_rdata,
      axi_ram_s__2_axi_r_rresp,
      axi_ram_s__2_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__2_vld = axi_ram_s__2_axi_r_rvalid;
  assign axi_ram_s__2_axi_r_rready = zstd_dec__axi_ram_r_r__2_rdy;
  assign axi_ram_s__2_axi_r_rresp[2] = '0;


  // Axi Subordinate Interface for axi_ram_s__3
  wire                  axi_ram_s__3_axi_ar_arvalid;
  wire                  axi_ram_s__3_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__3_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__3_axi_ar_araddr;
  wire [           3:0] axi_ram_s__3_axi_ar_arregion;
  wire [           7:0] axi_ram_s__3_axi_ar_arlen;
  wire [           2:0] axi_ram_s__3_axi_ar_arsize;
  wire [           1:0] axi_ram_s__3_axi_ar_arburst;
  wire [           3:0] axi_ram_s__3_axi_ar_arcache;
  wire [           2:0] axi_ram_s__3_axi_ar_arprot;
  wire [           3:0] axi_ram_s__3_axi_ar_arqos;

  wire                  axi_ram_s__3_axi_r_rvalid;
  wire                  axi_ram_s__3_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__3_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__3_axi_r_rdata;
  wire [           2:0] axi_ram_s__3_axi_r_rresp;
  wire                  axi_ram_s__3_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__3;
  wire                 zstd_dec__axi_ram_ar_s__3_rdy;
  wire                 zstd_dec__axi_ram_ar_s__3_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__3;
  wire                 zstd_dec__axi_ram_r_r__3_rdy;
  wire                 zstd_dec__axi_ram_r_r__3_vld;

  assign {
      axi_ram_s__3_axi_ar_arid,
      axi_ram_s__3_axi_ar_araddr,
      axi_ram_s__3_axi_ar_arregion,
      axi_ram_s__3_axi_ar_arlen,
      axi_ram_s__3_axi_ar_arsize,
      axi_ram_s__3_axi_ar_arburst,
      axi_ram_s__3_axi_ar_arcache,
      axi_ram_s__3_axi_ar_arprot,
      axi_ram_s__3_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__3;
  assign axi_ram_s__3_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__3_vld;
  assign zstd_dec__axi_ram_ar_s__3_rdy = axi_ram_s__3_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__3 = {
      axi_ram_s__3_axi_r_rid,
      axi_ram_s__3_axi_r_rdata,
      axi_ram_s__3_axi_r_rresp,
      axi_ram_s__3_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__3_vld = axi_ram_s__3_axi_r_rvalid;
  assign axi_ram_s__3_axi_r_rready = zstd_dec__axi_ram_r_r__3_rdy;
  assign axi_ram_s__3_axi_r_rresp[2] = '0;


  // Axi Subordinate Interface for axi_ram_s__4
  wire                  axi_ram_s__4_axi_ar_arvalid;
  wire                  axi_ram_s__4_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__4_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__4_axi_ar_araddr;
  wire [           3:0] axi_ram_s__4_axi_ar_arregion;
  wire [           7:0] axi_ram_s__4_axi_ar_arlen;
  wire [           2:0] axi_ram_s__4_axi_ar_arsize;
  wire [           1:0] axi_ram_s__4_axi_ar_arburst;
  wire [           3:0] axi_ram_s__4_axi_ar_arcache;
  wire [           2:0] axi_ram_s__4_axi_ar_arprot;
  wire [           3:0] axi_ram_s__4_axi_ar_arqos;

  wire                  axi_ram_s__4_axi_r_rvalid;
  wire                  axi_ram_s__4_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__4_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__4_axi_r_rdata;
  wire [           2:0] axi_ram_s__4_axi_r_rresp;
  wire                  axi_ram_s__4_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__4;
  wire                 zstd_dec__axi_ram_ar_s__4_rdy;
  wire                 zstd_dec__axi_ram_ar_s__4_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__4;
  wire                 zstd_dec__axi_ram_r_r__4_rdy;
  wire                 zstd_dec__axi_ram_r_r__4_vld;

  assign {
      axi_ram_s__4_axi_ar_arid,
      axi_ram_s__4_axi_ar_araddr,
      axi_ram_s__4_axi_ar_arregion,
      axi_ram_s__4_axi_ar_arlen,
      axi_ram_s__4_axi_ar_arsize,
      axi_ram_s__4_axi_ar_arburst,
      axi_ram_s__4_axi_ar_arcache,
      axi_ram_s__4_axi_ar_arprot,
      axi_ram_s__4_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__4;
  assign axi_ram_s__4_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__4_vld;
  assign zstd_dec__axi_ram_ar_s__4_rdy = axi_ram_s__4_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__4 = {
      axi_ram_s__4_axi_r_rid,
      axi_ram_s__4_axi_r_rdata,
      axi_ram_s__4_axi_r_rresp,
      axi_ram_s__4_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__4_vld = axi_ram_s__4_axi_r_rvalid;
  assign axi_ram_s__4_axi_r_rready = zstd_dec__axi_ram_r_r__4_rdy;
  assign axi_ram_s__4_axi_r_rresp[2] = '0;

  // Axi Subordinate Interface for axi_ram_s__5
  wire                  axi_ram_s__5_axi_ar_arvalid;
  wire                  axi_ram_s__5_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__5_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__5_axi_ar_araddr;
  wire [           3:0] axi_ram_s__5_axi_ar_arregion;
  wire [           7:0] axi_ram_s__5_axi_ar_arlen;
  wire [           2:0] axi_ram_s__5_axi_ar_arsize;
  wire [           1:0] axi_ram_s__5_axi_ar_arburst;
  wire [           3:0] axi_ram_s__5_axi_ar_arcache;
  wire [           2:0] axi_ram_s__5_axi_ar_arprot;
  wire [           3:0] axi_ram_s__5_axi_ar_arqos;

  wire                  axi_ram_s__5_axi_r_rvalid;
  wire                  axi_ram_s__5_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__5_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__5_axi_r_rdata;
  wire [           2:0] axi_ram_s__5_axi_r_rresp;
  wire                  axi_ram_s__5_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__5;
  wire                 zstd_dec__axi_ram_ar_s__5_rdy;
  wire                 zstd_dec__axi_ram_ar_s__5_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__5;
  wire                 zstd_dec__axi_ram_r_r__5_rdy;
  wire                 zstd_dec__axi_ram_r_r__5_vld;

  assign {
      axi_ram_s__5_axi_ar_arid,
      axi_ram_s__5_axi_ar_araddr,
      axi_ram_s__5_axi_ar_arregion,
      axi_ram_s__5_axi_ar_arlen,
      axi_ram_s__5_axi_ar_arsize,
      axi_ram_s__5_axi_ar_arburst,
      axi_ram_s__5_axi_ar_arcache,
      axi_ram_s__5_axi_ar_arprot,
      axi_ram_s__5_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__5;
  assign axi_ram_s__5_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__5_vld;
  assign zstd_dec__axi_ram_ar_s__5_rdy = axi_ram_s__5_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__5 = {
      axi_ram_s__5_axi_r_rid,
      axi_ram_s__5_axi_r_rdata,
      axi_ram_s__5_axi_r_rresp,
      axi_ram_s__5_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__5_vld = axi_ram_s__5_axi_r_rvalid;
  assign axi_ram_s__5_axi_r_rready = zstd_dec__axi_ram_r_r__5_rdy;
  assign axi_ram_s__5_axi_r_rresp[2] = '0;

  // Axi Subordinate Interface for axi_ram_s__6
  wire                  axi_ram_s__6_axi_ar_arvalid;
  wire                  axi_ram_s__6_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__6_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__6_axi_ar_araddr;
  wire [           3:0] axi_ram_s__6_axi_ar_arregion;
  wire [           7:0] axi_ram_s__6_axi_ar_arlen;
  wire [           2:0] axi_ram_s__6_axi_ar_arsize;
  wire [           1:0] axi_ram_s__6_axi_ar_arburst;
  wire [           3:0] axi_ram_s__6_axi_ar_arcache;
  wire [           2:0] axi_ram_s__6_axi_ar_arprot;
  wire [           3:0] axi_ram_s__6_axi_ar_arqos;

  wire                  axi_ram_s__6_axi_r_rvalid;
  wire                  axi_ram_s__6_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__6_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__6_axi_r_rdata;
  wire [           2:0] axi_ram_s__6_axi_r_rresp;
  wire                  axi_ram_s__6_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__6;
  wire                 zstd_dec__axi_ram_ar_s__6_rdy;
  wire                 zstd_dec__axi_ram_ar_s__6_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__6;
  wire                 zstd_dec__axi_ram_r_r__6_rdy;
  wire                 zstd_dec__axi_ram_r_r__6_vld;

  assign {
      axi_ram_s__6_axi_ar_arid,
      axi_ram_s__6_axi_ar_araddr,
      axi_ram_s__6_axi_ar_arregion,
      axi_ram_s__6_axi_ar_arlen,
      axi_ram_s__6_axi_ar_arsize,
      axi_ram_s__6_axi_ar_arburst,
      axi_ram_s__6_axi_ar_arcache,
      axi_ram_s__6_axi_ar_arprot,
      axi_ram_s__6_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__6;
  assign axi_ram_s__6_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__6_vld;
  assign zstd_dec__axi_ram_ar_s__6_rdy = axi_ram_s__6_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__6 = {
      axi_ram_s__6_axi_r_rid,
      axi_ram_s__6_axi_r_rdata,
      axi_ram_s__6_axi_r_rresp,
      axi_ram_s__6_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__6_vld = axi_ram_s__6_axi_r_rvalid;
  assign axi_ram_s__6_axi_r_rready = zstd_dec__axi_ram_r_r__6_rdy;
  assign axi_ram_s__6_axi_r_rresp[2] = '0;

  // Axi Subordinate Interface for axi_ram_s__7
  wire                  axi_ram_s__7_axi_ar_arvalid;
  wire                  axi_ram_s__7_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__7_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__7_axi_ar_araddr;
  wire [           3:0] axi_ram_s__7_axi_ar_arregion;
  wire [           7:0] axi_ram_s__7_axi_ar_arlen;
  wire [           2:0] axi_ram_s__7_axi_ar_arsize;
  wire [           1:0] axi_ram_s__7_axi_ar_arburst;
  wire [           3:0] axi_ram_s__7_axi_ar_arcache;
  wire [           2:0] axi_ram_s__7_axi_ar_arprot;
  wire [           3:0] axi_ram_s__7_axi_ar_arqos;

  wire                  axi_ram_s__7_axi_r_rvalid;
  wire                  axi_ram_s__7_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__7_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__7_axi_r_rdata;
  wire [           2:0] axi_ram_s__7_axi_r_rresp;
  wire                  axi_ram_s__7_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__7;
  wire                 zstd_dec__axi_ram_ar_s__7_rdy;
  wire                 zstd_dec__axi_ram_ar_s__7_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__7;
  wire                 zstd_dec__axi_ram_r_r__7_rdy;
  wire                 zstd_dec__axi_ram_r_r__7_vld;

  assign {
      axi_ram_s__7_axi_ar_arid,
      axi_ram_s__7_axi_ar_araddr,
      axi_ram_s__7_axi_ar_arregion,
      axi_ram_s__7_axi_ar_arlen,
      axi_ram_s__7_axi_ar_arsize,
      axi_ram_s__7_axi_ar_arburst,
      axi_ram_s__7_axi_ar_arcache,
      axi_ram_s__7_axi_ar_arprot,
      axi_ram_s__7_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__7;
  assign axi_ram_s__7_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__7_vld;
  assign zstd_dec__axi_ram_ar_s__7_rdy = axi_ram_s__7_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__7 = {
      axi_ram_s__7_axi_r_rid,
      axi_ram_s__7_axi_r_rdata,
      axi_ram_s__7_axi_r_rresp,
      axi_ram_s__7_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__7_vld = axi_ram_s__7_axi_r_rvalid;
  assign axi_ram_s__7_axi_r_rready = zstd_dec__axi_ram_r_r__7_rdy;
  assign axi_ram_s__7_axi_r_rresp[2] = '0;

  // Axi Subordinate Interface for axi_ram_s__8
  wire                  axi_ram_s__8_axi_ar_arvalid;
  wire                  axi_ram_s__8_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__8_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__8_axi_ar_araddr;
  wire [           3:0] axi_ram_s__8_axi_ar_arregion;
  wire [           7:0] axi_ram_s__8_axi_ar_arlen;
  wire [           2:0] axi_ram_s__8_axi_ar_arsize;
  wire [           1:0] axi_ram_s__8_axi_ar_arburst;
  wire [           3:0] axi_ram_s__8_axi_ar_arcache;
  wire [           2:0] axi_ram_s__8_axi_ar_arprot;
  wire [           3:0] axi_ram_s__8_axi_ar_arqos;

  wire                  axi_ram_s__8_axi_r_rvalid;
  wire                  axi_ram_s__8_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__8_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__8_axi_r_rdata;
  wire [           2:0] axi_ram_s__8_axi_r_rresp;
  wire                  axi_ram_s__8_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__8;
  wire                 zstd_dec__axi_ram_ar_s__8_rdy;
  wire                 zstd_dec__axi_ram_ar_s__8_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__8;
  wire                 zstd_dec__axi_ram_r_r__8_rdy;
  wire                 zstd_dec__axi_ram_r_r__8_vld;

  assign {
      axi_ram_s__8_axi_ar_arid,
      axi_ram_s__8_axi_ar_araddr,
      axi_ram_s__8_axi_ar_arregion,
      axi_ram_s__8_axi_ar_arlen,
      axi_ram_s__8_axi_ar_arsize,
      axi_ram_s__8_axi_ar_arburst,
      axi_ram_s__8_axi_ar_arcache,
      axi_ram_s__8_axi_ar_arprot,
      axi_ram_s__8_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__8;
  assign axi_ram_s__8_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__8_vld;
  assign zstd_dec__axi_ram_ar_s__8_rdy = axi_ram_s__8_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__8 = {
      axi_ram_s__8_axi_r_rid,
      axi_ram_s__8_axi_r_rdata,
      axi_ram_s__8_axi_r_rresp,
      axi_ram_s__8_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__8_vld = axi_ram_s__8_axi_r_rvalid;
  assign axi_ram_s__8_axi_r_rready = zstd_dec__axi_ram_r_r__8_rdy;
  assign axi_ram_s__8_axi_r_rresp[2] = '0;

  // Axi Subordinate Interface for axi_ram_s__9
  wire                  axi_ram_s__9_axi_ar_arvalid;
  wire                  axi_ram_s__9_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__9_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__9_axi_ar_araddr;
  wire [           3:0] axi_ram_s__9_axi_ar_arregion;
  wire [           7:0] axi_ram_s__9_axi_ar_arlen;
  wire [           2:0] axi_ram_s__9_axi_ar_arsize;
  wire [           1:0] axi_ram_s__9_axi_ar_arburst;
  wire [           3:0] axi_ram_s__9_axi_ar_arcache;
  wire [           2:0] axi_ram_s__9_axi_ar_arprot;
  wire [           3:0] axi_ram_s__9_axi_ar_arqos;

  wire                  axi_ram_s__9_axi_r_rvalid;
  wire                  axi_ram_s__9_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__9_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__9_axi_r_rdata;
  wire [           2:0] axi_ram_s__9_axi_r_rresp;
  wire                  axi_ram_s__9_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__9;
  wire                 zstd_dec__axi_ram_ar_s__9_rdy;
  wire                 zstd_dec__axi_ram_ar_s__9_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__9;
  wire                 zstd_dec__axi_ram_r_r__9_rdy;
  wire                 zstd_dec__axi_ram_r_r__9_vld;

  assign {
      axi_ram_s__9_axi_ar_arid,
      axi_ram_s__9_axi_ar_araddr,
      axi_ram_s__9_axi_ar_arregion,
      axi_ram_s__9_axi_ar_arlen,
      axi_ram_s__9_axi_ar_arsize,
      axi_ram_s__9_axi_ar_arburst,
      axi_ram_s__9_axi_ar_arcache,
      axi_ram_s__9_axi_ar_arprot,
      axi_ram_s__9_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__9;
  assign axi_ram_s__9_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__9_vld;
  assign zstd_dec__axi_ram_ar_s__9_rdy = axi_ram_s__9_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__9 = {
      axi_ram_s__9_axi_r_rid,
      axi_ram_s__9_axi_r_rdata,
      axi_ram_s__9_axi_r_rresp,
      axi_ram_s__9_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__9_vld = axi_ram_s__9_axi_r_rvalid;
  assign axi_ram_s__9_axi_r_rready = zstd_dec__axi_ram_r_r__9_rdy;
  assign axi_ram_s__9_axi_r_rresp[2] = '0;


  // Axi Subordinate Interface for axi_ram_s__10
  wire                  axi_ram_s__10_axi_ar_arvalid;
  wire                  axi_ram_s__10_axi_ar_arready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__10_axi_ar_arid;
  wire [AXI_ADDR_W-1:0] axi_ram_s__10_axi_ar_araddr;
  wire [           3:0] axi_ram_s__10_axi_ar_arregion;
  wire [           7:0] axi_ram_s__10_axi_ar_arlen;
  wire [           2:0] axi_ram_s__10_axi_ar_arsize;
  wire [           1:0] axi_ram_s__10_axi_ar_arburst;
  wire [           3:0] axi_ram_s__10_axi_ar_arcache;
  wire [           2:0] axi_ram_s__10_axi_ar_arprot;
  wire [           3:0] axi_ram_s__10_axi_ar_arqos;

  wire                  axi_ram_s__10_axi_r_rvalid;
  wire                  axi_ram_s__10_axi_r_rready;
  wire [S_AXI_ID_W-1:0] axi_ram_s__10_axi_r_rid;
  wire [AXI_DATA_W-1:0] axi_ram_s__10_axi_r_rdata;
  wire [           2:0] axi_ram_s__10_axi_r_rresp;
  wire                  axi_ram_s__10_axi_r_rlast;

  wire [XlsAxiArW-1:0] zstd_dec__axi_ram_ar_s__10;
  wire                 zstd_dec__axi_ram_ar_s__10_rdy;
  wire                 zstd_dec__axi_ram_ar_s__10_vld;
  wire [ XlsAxiRW-1:0] zstd_dec__axi_ram_r_r__10;
  wire                 zstd_dec__axi_ram_r_r__10_rdy;
  wire                 zstd_dec__axi_ram_r_r__10_vld;

  assign {
      axi_ram_s__10_axi_ar_arid,
      axi_ram_s__10_axi_ar_araddr,
      axi_ram_s__10_axi_ar_arregion,
      axi_ram_s__10_axi_ar_arlen,
      axi_ram_s__10_axi_ar_arsize,
      axi_ram_s__10_axi_ar_arburst,
      axi_ram_s__10_axi_ar_arcache,
      axi_ram_s__10_axi_ar_arprot,
      axi_ram_s__10_axi_ar_arqos
      } = zstd_dec__axi_ram_ar_s__10;
  assign axi_ram_s__10_axi_ar_arvalid = zstd_dec__axi_ram_ar_s__10_vld;
  assign zstd_dec__axi_ram_ar_s__10_rdy = axi_ram_s__10_axi_ar_arready;

  assign zstd_dec__axi_ram_r_r__10 = {
      axi_ram_s__10_axi_r_rid,
      axi_ram_s__10_axi_r_rdata,
      axi_ram_s__10_axi_r_rresp,
      axi_ram_s__10_axi_r_rlast};
  assign zstd_dec__axi_ram_r_r__10_vld = axi_ram_s__10_axi_r_rvalid;
  assign axi_ram_s__10_axi_r_rready = zstd_dec__axi_ram_r_r__10_rdy;
  assign axi_ram_s__10_axi_r_rresp[2] = '0;


  // RAM instance for history_buffer_ram0
  logic [7:0] history_buffer_ram0_wr_data;
  logic [12:0] history_buffer_ram0_wr_addr;
  logic history_buffer_ram0_wr_en;
  logic history_buffer_ram0_wr_mask;
  logic [7:0] history_buffer_ram0_rd_data;
  logic [12:0] history_buffer_ram0_rd_addr;
  logic history_buffer_ram0_rd_en;
  logic history_buffer_ram0_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) history_buffer_ram0_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(history_buffer_ram0_wr_data),
      .wr_addr(history_buffer_ram0_wr_addr),
      .wr_en(history_buffer_ram0_wr_en),
      .wr_mask(history_buffer_ram0_wr_mask),
      .rd_data(history_buffer_ram0_rd_data),
      .rd_addr(history_buffer_ram0_rd_addr),
      .rd_en(history_buffer_ram0_rd_en),
      .rd_mask(history_buffer_ram0_rd_mask)
  );

  // RAM instance for history_buffer_ram1
  logic [7:0] history_buffer_ram1_wr_data;
  logic [12:0] history_buffer_ram1_wr_addr;
  logic history_buffer_ram1_wr_en;
  logic history_buffer_ram1_wr_mask;
  logic [7:0] history_buffer_ram1_rd_data;
  logic [12:0] history_buffer_ram1_rd_addr;
  logic history_buffer_ram1_rd_en;
  logic history_buffer_ram1_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) history_buffer_ram1_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(history_buffer_ram1_wr_data),
      .wr_addr(history_buffer_ram1_wr_addr),
      .wr_en(history_buffer_ram1_wr_en),
      .wr_mask(history_buffer_ram1_wr_mask),
      .rd_data(history_buffer_ram1_rd_data),
      .rd_addr(history_buffer_ram1_rd_addr),
      .rd_en(history_buffer_ram1_rd_en),
      .rd_mask(history_buffer_ram1_rd_mask)
  );

  // RAM instance for history_buffer_ram2
  logic [7:0] history_buffer_ram2_wr_data;
  logic [12:0] history_buffer_ram2_wr_addr;
  logic history_buffer_ram2_wr_en;
  logic history_buffer_ram2_wr_mask;
  logic [7:0] history_buffer_ram2_rd_data;
  logic [12:0] history_buffer_ram2_rd_addr;
  logic history_buffer_ram2_rd_en;
  logic history_buffer_ram2_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) history_buffer_ram2_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(history_buffer_ram2_wr_data),
      .wr_addr(history_buffer_ram2_wr_addr),
      .wr_en(history_buffer_ram2_wr_en),
      .wr_mask(history_buffer_ram2_wr_mask),
      .rd_data(history_buffer_ram2_rd_data),
      .rd_addr(history_buffer_ram2_rd_addr),
      .rd_en(history_buffer_ram2_rd_en),
      .rd_mask(history_buffer_ram2_rd_mask)
  );

  // RAM instance for history_buffer_ram3
  logic [7:0] history_buffer_ram3_wr_data;
  logic [12:0] history_buffer_ram3_wr_addr;
  logic history_buffer_ram3_wr_en;
  logic history_buffer_ram3_wr_mask;
  logic [7:0] history_buffer_ram3_rd_data;
  logic [12:0] history_buffer_ram3_rd_addr;
  logic history_buffer_ram3_rd_en;
  logic history_buffer_ram3_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) history_buffer_ram3_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(history_buffer_ram3_wr_data),
      .wr_addr(history_buffer_ram3_wr_addr),
      .wr_en(history_buffer_ram3_wr_en),
      .wr_mask(history_buffer_ram3_wr_mask),
      .rd_data(history_buffer_ram3_rd_data),
      .rd_addr(history_buffer_ram3_rd_addr),
      .rd_en(history_buffer_ram3_rd_en),
      .rd_mask(history_buffer_ram3_rd_mask)
  );

  // RAM instance for history_buffer_ram4
  logic [7:0] history_buffer_ram4_wr_data;
  logic [12:0] history_buffer_ram4_wr_addr;
  logic history_buffer_ram4_wr_en;
  logic history_buffer_ram4_wr_mask;
  logic [7:0] history_buffer_ram4_rd_data;
  logic [12:0] history_buffer_ram4_rd_addr;
  logic history_buffer_ram4_rd_en;
  logic history_buffer_ram4_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) history_buffer_ram4_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(history_buffer_ram4_wr_data),
      .wr_addr(history_buffer_ram4_wr_addr),
      .wr_en(history_buffer_ram4_wr_en),
      .wr_mask(history_buffer_ram4_wr_mask),
      .rd_data(history_buffer_ram4_rd_data),
      .rd_addr(history_buffer_ram4_rd_addr),
      .rd_en(history_buffer_ram4_rd_en),
      .rd_mask(history_buffer_ram4_rd_mask)
  );

  // RAM instance for history_buffer_ram5
  logic [7:0] history_buffer_ram5_wr_data;
  logic [12:0] history_buffer_ram5_wr_addr;
  logic history_buffer_ram5_wr_en;
  logic history_buffer_ram5_wr_mask;
  logic [7:0] history_buffer_ram5_rd_data;
  logic [12:0] history_buffer_ram5_rd_addr;
  logic history_buffer_ram5_rd_en;
  logic history_buffer_ram5_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) history_buffer_ram5_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(history_buffer_ram5_wr_data),
      .wr_addr(history_buffer_ram5_wr_addr),
      .wr_en(history_buffer_ram5_wr_en),
      .wr_mask(history_buffer_ram5_wr_mask),
      .rd_data(history_buffer_ram5_rd_data),
      .rd_addr(history_buffer_ram5_rd_addr),
      .rd_en(history_buffer_ram5_rd_en),
      .rd_mask(history_buffer_ram5_rd_mask)
  );

  // RAM instance for history_buffer_ram6
  logic [7:0] history_buffer_ram6_wr_data;
  logic [12:0] history_buffer_ram6_wr_addr;
  logic history_buffer_ram6_wr_en;
  logic history_buffer_ram6_wr_mask;
  logic [7:0] history_buffer_ram6_rd_data;
  logic [12:0] history_buffer_ram6_rd_addr;
  logic history_buffer_ram6_rd_en;
  logic history_buffer_ram6_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) history_buffer_ram6_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(history_buffer_ram6_wr_data),
      .wr_addr(history_buffer_ram6_wr_addr),
      .wr_en(history_buffer_ram6_wr_en),
      .wr_mask(history_buffer_ram6_wr_mask),
      .rd_data(history_buffer_ram6_rd_data),
      .rd_addr(history_buffer_ram6_rd_addr),
      .rd_en(history_buffer_ram6_rd_en),
      .rd_mask(history_buffer_ram6_rd_mask)
  );

  // RAM instance for history_buffer_ram7
  logic [7:0] history_buffer_ram7_wr_data;
  logic [12:0] history_buffer_ram7_wr_addr;
  logic history_buffer_ram7_wr_en;
  logic history_buffer_ram7_wr_mask;
  logic [7:0] history_buffer_ram7_rd_data;
  logic [12:0] history_buffer_ram7_rd_addr;
  logic history_buffer_ram7_rd_en;
  logic history_buffer_ram7_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) history_buffer_ram7_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(history_buffer_ram7_wr_data),
      .wr_addr(history_buffer_ram7_wr_addr),
      .wr_en(history_buffer_ram7_wr_en),
      .wr_mask(history_buffer_ram7_wr_mask),
      .rd_data(history_buffer_ram7_rd_data),
      .rd_addr(history_buffer_ram7_rd_addr),
      .rd_en(history_buffer_ram7_rd_en),
      .rd_mask(history_buffer_ram7_rd_mask)
  );

  // RAM instance for dpd_ram
  logic [15:0] dpd_ram_wr_data;
  logic [7:0] dpd_ram_wr_addr;
  logic dpd_ram_wr_en;
  logic dpd_ram_wr_mask;
  logic [15:0] dpd_ram_rd_data;
  logic [7:0] dpd_ram_rd_addr;
  logic dpd_ram_rd_en;
  logic dpd_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(16),
      .SIZE(256),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(8)
  ) dpd_ram_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(dpd_ram_wr_data),
      .wr_addr(dpd_ram_wr_addr),
      .wr_en(dpd_ram_wr_en),
      .wr_mask(dpd_ram_wr_mask),
      .rd_data(dpd_ram_rd_data),
      .rd_addr(dpd_ram_rd_addr),
      .rd_en(dpd_ram_rd_en),
      .rd_mask(dpd_ram_rd_mask)
  );

  // RAM instance for fse_tmp_ram
  logic [15:0] fse_tmp_ram_wr_data;
  logic [7:0] fse_tmp_ram_wr_addr;
  logic fse_tmp_ram_wr_en;
  logic fse_tmp_ram_wr_mask;
  logic [15:0] fse_tmp_ram_rd_data;
  logic [7:0] fse_tmp_ram_rd_addr;
  logic fse_tmp_ram_rd_en;
  logic fse_tmp_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(16),
      .SIZE(256),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(8)
  ) fse_tmp_ram_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(fse_tmp_ram_wr_data),
      .wr_addr(fse_tmp_ram_wr_addr),
      .wr_en(fse_tmp_ram_wr_en),
      .wr_mask(fse_tmp_ram_wr_mask),
      .rd_data(fse_tmp_ram_rd_data),
      .rd_addr(fse_tmp_ram_rd_addr),
      .rd_en(fse_tmp_ram_rd_en),
      .rd_mask(fse_tmp_ram_rd_mask)
  );

  // RAM instance for fse_tmp2_ram
  logic [7:0] fse_tmp2_ram_wr_data;
  logic [8:0] fse_tmp2_ram_wr_addr;
  logic fse_tmp2_ram_wr_en;
  logic fse_tmp2_ram_wr_mask;
  logic [7:0] fse_tmp2_ram_rd_data;
  logic [8:0] fse_tmp2_ram_rd_addr;
  logic fse_tmp2_ram_rd_en;
  logic fse_tmp2_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(512),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(9)
  ) fse_tmp2_ram_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(fse_tmp2_ram_wr_data),
      .wr_addr(fse_tmp2_ram_wr_addr),
      .wr_en(fse_tmp2_ram_wr_en),
      .wr_mask(fse_tmp2_ram_wr_mask),
      .rd_data(fse_tmp2_ram_rd_data),
      .rd_addr(fse_tmp2_ram_rd_addr),
      .rd_en(fse_tmp2_ram_rd_en),
      .rd_mask(fse_tmp2_ram_rd_mask)
  );

  // RAM instance for ll_def_fse_ram
  logic [31:0] ll_def_fse_ram_wr_data;
  logic [14:0] ll_def_fse_ram_wr_addr;
  logic ll_def_fse_ram_wr_en;
  logic ll_def_fse_ram_wr_mask;
  logic [31:0] ll_def_fse_ram_rd_data;
  logic [14:0] ll_def_fse_ram_rd_addr;
  logic ll_def_fse_ram_rd_en;
  logic ll_def_fse_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(32),
      .SIZE(256),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(15),
      .INIT_FILE("../xls/modules/zstd/zstd_dec_ll_fse_default.mem")
  ) ll_def_fse_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(ll_def_fse_ram_wr_data),
      .wr_addr(ll_def_fse_ram_wr_addr),
      .wr_en(ll_def_fse_ram_wr_en),
      .wr_mask(ll_def_fse_ram_wr_mask),
      .rd_data(ll_def_fse_ram_rd_data),
      .rd_addr(ll_def_fse_ram_rd_addr),
      .rd_en(ll_def_fse_ram_rd_en),
      .rd_mask(ll_def_fse_ram_rd_mask)
  );

  // RAM instance for ll_fse_ram
  logic [31:0] ll_fse_ram_wr_data;
  logic [14:0] ll_fse_ram_wr_addr;
  logic ll_fse_ram_wr_en;
  logic ll_fse_ram_wr_mask;
  logic [31:0] ll_fse_ram_rd_data;
  logic [14:0] ll_fse_ram_rd_addr;
  logic ll_fse_ram_rd_en;
  logic ll_fse_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(32),
      .SIZE(32768),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(15)
  ) ll_fse_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(ll_fse_ram_wr_data),
      .wr_addr(ll_fse_ram_wr_addr),
      .wr_en(ll_fse_ram_wr_en),
      .wr_mask(ll_fse_ram_wr_mask),
      .rd_data(ll_fse_ram_rd_data),
      .rd_addr(ll_fse_ram_rd_addr),
      .rd_en(ll_fse_ram_rd_en),
      .rd_mask(ll_fse_ram_rd_mask)
  );

  // RAM instance for ml_def_fse_ram
  logic [31:0] ml_def_fse_ram_wr_data;
  logic [14:0] ml_def_fse_ram_wr_addr;
  logic ml_def_fse_ram_wr_en;
  logic ml_def_fse_ram_wr_mask;
  logic [31:0] ml_def_fse_ram_rd_data;
  logic [14:0] ml_def_fse_ram_rd_addr;
  logic ml_def_fse_ram_rd_en;
  logic ml_def_fse_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(32),
      .SIZE(256),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(15),
      .INIT_FILE("../xls/modules/zstd/zstd_dec_ml_fse_default.mem")
  ) ml_def_fse_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(ml_def_fse_ram_wr_data),
      .wr_addr(ml_def_fse_ram_wr_addr),
      .wr_en(ml_def_fse_ram_wr_en),
      .wr_mask(ml_def_fse_ram_wr_mask),
      .rd_data(ml_def_fse_ram_rd_data),
      .rd_addr(ml_def_fse_ram_rd_addr),
      .rd_en(ml_def_fse_ram_rd_en),
      .rd_mask(ml_def_fse_ram_rd_mask)
  );

  // RAM instance for ml_fse_ram
  logic [31:0] ml_fse_ram_wr_data;
  logic [14:0] ml_fse_ram_wr_addr;
  logic ml_fse_ram_wr_en;
  logic ml_fse_ram_wr_mask;
  logic [31:0] ml_fse_ram_rd_data;
  logic [14:0] ml_fse_ram_rd_addr;
  logic ml_fse_ram_rd_en;
  logic ml_fse_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(32),
      .SIZE(32768),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(15)
  ) ml_fse_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(ml_fse_ram_wr_data),
      .wr_addr(ml_fse_ram_wr_addr),
      .wr_en(ml_fse_ram_wr_en),
      .wr_mask(ml_fse_ram_wr_mask),
      .rd_data(ml_fse_ram_rd_data),
      .rd_addr(ml_fse_ram_rd_addr),
      .rd_en(ml_fse_ram_rd_en),
      .rd_mask(ml_fse_ram_rd_mask)
  );

  // RAM instance for of_def_fse_ram
  logic [31:0] of_def_fse_ram_wr_data;
  logic [14:0] of_def_fse_ram_wr_addr;
  logic of_def_fse_ram_wr_en;
  logic of_def_fse_ram_wr_mask;
  logic [31:0] of_def_fse_ram_rd_data;
  logic [14:0] of_def_fse_ram_rd_addr;
  logic of_def_fse_ram_rd_en;
  logic of_def_fse_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(32),
      .SIZE(256),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(15),
      .INIT_FILE("../xls/modules/zstd/zstd_dec_of_fse_default.mem")
  ) of_def_fse_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(of_def_fse_ram_wr_data),
      .wr_addr(of_def_fse_ram_wr_addr),
      .wr_en(of_def_fse_ram_wr_en),
      .wr_mask(of_def_fse_ram_wr_mask),
      .rd_data(of_def_fse_ram_rd_data),
      .rd_addr(of_def_fse_ram_rd_addr),
      .rd_en(of_def_fse_ram_rd_en),
      .rd_mask(of_def_fse_ram_rd_mask)
  );

  // RAM instance for of_fse_ram
  logic [31:0] of_fse_ram_wr_data;
  logic [14:0] of_fse_ram_wr_addr;
  logic of_fse_ram_wr_en;
  logic of_fse_ram_wr_mask;
  logic [31:0] of_fse_ram_rd_data;
  logic [14:0] of_fse_ram_rd_addr;
  logic of_fse_ram_rd_en;
  logic of_fse_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(32),
      .SIZE(32768),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(15)
  ) of_fse_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(of_fse_ram_wr_data),
      .wr_addr(of_fse_ram_wr_addr),
      .wr_en(of_fse_ram_wr_en),
      .wr_mask(of_fse_ram_wr_mask),
      .rd_data(of_fse_ram_rd_data),
      .rd_addr(of_fse_ram_rd_addr),
      .rd_en(of_fse_ram_rd_en),
      .rd_mask(of_fse_ram_rd_mask)
  );

  // RAM instance for huffman_literals_prescan_ram
  logic [91:0] huffman_literals_prescan_ram_wr_data;
  logic [5:0] huffman_literals_prescan_ram_wr_addr;
  logic huffman_literals_prescan_ram_wr_en;
  logic huffman_literals_prescan_ram_wr_mask;
  logic [91:0] huffman_literals_prescan_ram_rd_data;
  logic [5:0] huffman_literals_prescan_ram_rd_addr;
  logic huffman_literals_prescan_ram_rd_en;
  logic huffman_literals_prescan_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(92),
      .SIZE(64),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(6)
  ) huffman_literals_prescan_ram_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(huffman_literals_prescan_ram_wr_data),
      .wr_addr(huffman_literals_prescan_ram_wr_addr),
      .wr_en(huffman_literals_prescan_ram_wr_en),
      .wr_mask(huffman_literals_prescan_ram_wr_mask),
      .rd_data(huffman_literals_prescan_ram_rd_data),
      .rd_addr(huffman_literals_prescan_ram_rd_addr),
      .rd_en(huffman_literals_prescan_ram_rd_en),
      .rd_mask(huffman_literals_prescan_ram_rd_mask)
  );

  // RAM instance for huffman_literals_weights_mem_ram
  logic [31:0] huffman_literals_weights_mem_ram_wr_data;
  logic [5:0] huffman_literals_weights_mem_ram_wr_addr;
  logic huffman_literals_weights_mem_ram_wr_en;
  logic [7:0] huffman_literals_weights_mem_ram_wr_mask;
  logic [31:0] huffman_literals_weights_mem_ram_rd_data;
  logic [5:0] huffman_literals_weights_mem_ram_rd_addr;
  logic huffman_literals_weights_mem_ram_rd_en;
  logic [7:0] huffman_literals_weights_mem_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(32),
      .SIZE(64),
      .NUM_PARTITIONS(8),
      .ADDR_WIDTH(6)
  ) huffman_literals_weights_mem_ram_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(huffman_literals_weights_mem_ram_wr_data),
      .wr_addr(huffman_literals_weights_mem_ram_wr_addr),
      .wr_en(huffman_literals_weights_mem_ram_wr_en),
      .wr_mask(huffman_literals_weights_mem_ram_wr_mask),
      .rd_data(huffman_literals_weights_mem_ram_rd_data),
      .rd_addr(huffman_literals_weights_mem_ram_rd_addr),
      .rd_en(huffman_literals_weights_mem_ram_rd_en),
      .rd_mask(huffman_literals_weights_mem_ram_rd_mask)
  );

  // RAM instance for literals_buffer_ram0
  logic [8:0] literals_buffer_ram0_wr_data;
  logic [12:0] literals_buffer_ram0_wr_addr;
  logic literals_buffer_ram0_wr_en;
  logic literals_buffer_ram0_wr_mask;
  logic [8:0] literals_buffer_ram0_rd_data;
  logic [12:0] literals_buffer_ram0_rd_addr;
  logic literals_buffer_ram0_rd_en;
  logic literals_buffer_ram0_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(9),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) literals_buffer_ram0_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(literals_buffer_ram0_wr_data),
      .wr_addr(literals_buffer_ram0_wr_addr),
      .wr_en(literals_buffer_ram0_wr_en),
      .wr_mask(literals_buffer_ram0_wr_mask),
      .rd_data(literals_buffer_ram0_rd_data),
      .rd_addr(literals_buffer_ram0_rd_addr),
      .rd_en(literals_buffer_ram0_rd_en),
      .rd_mask(literals_buffer_ram0_rd_mask)
  );

  // RAM instance for literals_buffer_ram1
  logic [8:0] literals_buffer_ram1_wr_data;
  logic [12:0] literals_buffer_ram1_wr_addr;
  logic literals_buffer_ram1_wr_en;
  logic literals_buffer_ram1_wr_mask;
  logic [8:0] literals_buffer_ram1_rd_data;
  logic [12:0] literals_buffer_ram1_rd_addr;
  logic literals_buffer_ram1_rd_en;
  logic literals_buffer_ram1_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(9),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) literals_buffer_ram1_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(literals_buffer_ram1_wr_data),
      .wr_addr(literals_buffer_ram1_wr_addr),
      .wr_en(literals_buffer_ram1_wr_en),
      .wr_mask(literals_buffer_ram1_wr_mask),
      .rd_data(literals_buffer_ram1_rd_data),
      .rd_addr(literals_buffer_ram1_rd_addr),
      .rd_en(literals_buffer_ram1_rd_en),
      .rd_mask(literals_buffer_ram1_rd_mask)
  );

  // RAM instance for literals_buffer_ram2
  logic [8:0] literals_buffer_ram2_wr_data;
  logic [12:0] literals_buffer_ram2_wr_addr;
  logic literals_buffer_ram2_wr_en;
  logic literals_buffer_ram2_wr_mask;
  logic [8:0] literals_buffer_ram2_rd_data;
  logic [12:0] literals_buffer_ram2_rd_addr;
  logic literals_buffer_ram2_rd_en;
  logic literals_buffer_ram2_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(9),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) literals_buffer_ram2_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(literals_buffer_ram2_wr_data),
      .wr_addr(literals_buffer_ram2_wr_addr),
      .wr_en(literals_buffer_ram2_wr_en),
      .wr_mask(literals_buffer_ram2_wr_mask),
      .rd_data(literals_buffer_ram2_rd_data),
      .rd_addr(literals_buffer_ram2_rd_addr),
      .rd_en(literals_buffer_ram2_rd_en),
      .rd_mask(literals_buffer_ram2_rd_mask)
  );

  // RAM instance for literals_buffer_ram3
  logic [8:0] literals_buffer_ram3_wr_data;
  logic [12:0] literals_buffer_ram3_wr_addr;
  logic literals_buffer_ram3_wr_en;
  logic literals_buffer_ram3_wr_mask;
  logic [8:0] literals_buffer_ram3_rd_data;
  logic [12:0] literals_buffer_ram3_rd_addr;
  logic literals_buffer_ram3_rd_en;
  logic literals_buffer_ram3_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(9),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) literals_buffer_ram3_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(literals_buffer_ram3_wr_data),
      .wr_addr(literals_buffer_ram3_wr_addr),
      .wr_en(literals_buffer_ram3_wr_en),
      .wr_mask(literals_buffer_ram3_wr_mask),
      .rd_data(literals_buffer_ram3_rd_data),
      .rd_addr(literals_buffer_ram3_rd_addr),
      .rd_en(literals_buffer_ram3_rd_en),
      .rd_mask(literals_buffer_ram3_rd_mask)
  );

  // RAM instance for literals_buffer_ram4
  logic [8:0] literals_buffer_ram4_wr_data;
  logic [12:0] literals_buffer_ram4_wr_addr;
  logic literals_buffer_ram4_wr_en;
  logic literals_buffer_ram4_wr_mask;
  logic [8:0] literals_buffer_ram4_rd_data;
  logic [12:0] literals_buffer_ram4_rd_addr;
  logic literals_buffer_ram4_rd_en;
  logic literals_buffer_ram4_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(9),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) literals_buffer_ram4_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(literals_buffer_ram4_wr_data),
      .wr_addr(literals_buffer_ram4_wr_addr),
      .wr_en(literals_buffer_ram4_wr_en),
      .wr_mask(literals_buffer_ram4_wr_mask),
      .rd_data(literals_buffer_ram4_rd_data),
      .rd_addr(literals_buffer_ram4_rd_addr),
      .rd_en(literals_buffer_ram4_rd_en),
      .rd_mask(literals_buffer_ram4_rd_mask)
  );

  // RAM instance for literals_buffer_ram5
  logic [8:0] literals_buffer_ram5_wr_data;
  logic [12:0] literals_buffer_ram5_wr_addr;
  logic literals_buffer_ram5_wr_en;
  logic literals_buffer_ram5_wr_mask;
  logic [8:0] literals_buffer_ram5_rd_data;
  logic [12:0] literals_buffer_ram5_rd_addr;
  logic literals_buffer_ram5_rd_en;
  logic literals_buffer_ram5_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(9),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) literals_buffer_ram5_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(literals_buffer_ram5_wr_data),
      .wr_addr(literals_buffer_ram5_wr_addr),
      .wr_en(literals_buffer_ram5_wr_en),
      .wr_mask(literals_buffer_ram5_wr_mask),
      .rd_data(literals_buffer_ram5_rd_data),
      .rd_addr(literals_buffer_ram5_rd_addr),
      .rd_en(literals_buffer_ram5_rd_en),
      .rd_mask(literals_buffer_ram5_rd_mask)
  );

  // RAM instance for literals_buffer_ram6
  logic [8:0] literals_buffer_ram6_wr_data;
  logic [12:0] literals_buffer_ram6_wr_addr;
  logic literals_buffer_ram6_wr_en;
  logic literals_buffer_ram6_wr_mask;
  logic [8:0] literals_buffer_ram6_rd_data;
  logic [12:0] literals_buffer_ram6_rd_addr;
  logic literals_buffer_ram6_rd_en;
  logic literals_buffer_ram6_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(9),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) literals_buffer_ram6_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(literals_buffer_ram6_wr_data),
      .wr_addr(literals_buffer_ram6_wr_addr),
      .wr_en(literals_buffer_ram6_wr_en),
      .wr_mask(literals_buffer_ram6_wr_mask),
      .rd_data(literals_buffer_ram6_rd_data),
      .rd_addr(literals_buffer_ram6_rd_addr),
      .rd_en(literals_buffer_ram6_rd_en),
      .rd_mask(literals_buffer_ram6_rd_mask)
  );

  // RAM instance for literals_buffer_ram7
  logic [8:0] literals_buffer_ram7_wr_data;
  logic [12:0] literals_buffer_ram7_wr_addr;
  logic literals_buffer_ram7_wr_en;
  logic literals_buffer_ram7_wr_mask;
  logic [8:0] literals_buffer_ram7_rd_data;
  logic [12:0] literals_buffer_ram7_rd_addr;
  logic literals_buffer_ram7_rd_en;
  logic literals_buffer_ram7_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(9),
      .SIZE(8192),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(13)
  ) literals_buffer_ram7_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(literals_buffer_ram7_wr_data),
      .wr_addr(literals_buffer_ram7_wr_addr),
      .wr_en(literals_buffer_ram7_wr_en),
      .wr_mask(literals_buffer_ram7_wr_mask),
      .rd_data(literals_buffer_ram7_rd_data),
      .rd_addr(literals_buffer_ram7_rd_addr),
      .rd_en(literals_buffer_ram7_rd_en),
      .rd_mask(literals_buffer_ram7_rd_mask)
  );

  // RAM instance for huffman_literals_weights_dpd_ram
  logic [15:0] huffman_literals_weights_dpd_ram_wr_data;
  logic [7:0] huffman_literals_weights_dpd_ram_wr_addr;
  logic huffman_literals_weights_dpd_ram_wr_en;
  logic huffman_literals_weights_dpd_ram_wr_mask;
  logic [15:0] huffman_literals_weights_dpd_ram_rd_data;
  logic [7:0] huffman_literals_weights_dpd_ram_rd_addr;
  logic huffman_literals_weights_dpd_ram_rd_en;
  logic huffman_literals_weights_dpd_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(16),
      .SIZE(256),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(8)
  ) huffman_literals_weights_dpd_ram_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(huffman_literals_weights_dpd_ram_wr_data),
      .wr_addr(huffman_literals_weights_dpd_ram_wr_addr),
      .wr_en(huffman_literals_weights_dpd_ram_wr_en),
      .wr_mask(huffman_literals_weights_dpd_ram_wr_mask),
      .rd_data(huffman_literals_weights_dpd_ram_rd_data),
      .rd_addr(huffman_literals_weights_dpd_ram_rd_addr),
      .rd_en(huffman_literals_weights_dpd_ram_rd_en),
      .rd_mask(huffman_literals_weights_dpd_ram_rd_mask)
  );

  // RAM instance for huffman_literals_weights_tmp_ram
  logic [15:0] huffman_literals_weights_tmp_ram_wr_data;
  logic [7:0] huffman_literals_weights_tmp_ram_wr_addr;
  logic huffman_literals_weights_tmp_ram_wr_en;
  logic huffman_literals_weights_tmp_ram_wr_mask;
  logic [15:0] huffman_literals_weights_tmp_ram_rd_data;
  logic [7:0] huffman_literals_weights_tmp_ram_rd_addr;
  logic huffman_literals_weights_tmp_ram_rd_en;
  logic huffman_literals_weights_tmp_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(16),
      .SIZE(256),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(8)
  ) huffman_literals_weights_tmp_ram_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(huffman_literals_weights_tmp_ram_wr_data),
      .wr_addr(huffman_literals_weights_tmp_ram_wr_addr),
      .wr_en(huffman_literals_weights_tmp_ram_wr_en),
      .wr_mask(huffman_literals_weights_tmp_ram_wr_mask),
      .rd_data(huffman_literals_weights_tmp_ram_rd_data),
      .rd_addr(huffman_literals_weights_tmp_ram_rd_addr),
      .rd_en(huffman_literals_weights_tmp_ram_rd_en),
      .rd_mask(huffman_literals_weights_tmp_ram_rd_mask)
  );

  // RAM instance for huffman_literals_weights_tmp2_ram
  logic [7:0] huffman_literals_weights_tmp2_ram_wr_data;
  logic [8:0] huffman_literals_weights_tmp2_ram_wr_addr;
  logic huffman_literals_weights_tmp2_ram_wr_en;
  logic huffman_literals_weights_tmp2_ram_wr_mask;
  logic [7:0] huffman_literals_weights_tmp2_ram_rd_data;
  logic [8:0] huffman_literals_weights_tmp2_ram_rd_addr;
  logic huffman_literals_weights_tmp2_ram_rd_en;
  logic huffman_literals_weights_tmp2_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(8),
      .SIZE(512),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(9)
  ) huffman_literals_weights_tmp2_ram_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(huffman_literals_weights_tmp2_ram_wr_data),
      .wr_addr(huffman_literals_weights_tmp2_ram_wr_addr),
      .wr_en(huffman_literals_weights_tmp2_ram_wr_en),
      .wr_mask(huffman_literals_weights_tmp2_ram_wr_mask),
      .rd_data(huffman_literals_weights_tmp2_ram_rd_data),
      .rd_addr(huffman_literals_weights_tmp2_ram_rd_addr),
      .rd_en(huffman_literals_weights_tmp2_ram_rd_en),
      .rd_mask(huffman_literals_weights_tmp2_ram_rd_mask)
  );

  // RAM instance for huffman_literals_weights_fse_ram
  logic [31:0] huffman_literals_weights_fse_ram_wr_data;
  logic [7:0] huffman_literals_weights_fse_ram_wr_addr;
  logic huffman_literals_weights_fse_ram_wr_en;
  logic huffman_literals_weights_fse_ram_wr_mask;
  logic [31:0] huffman_literals_weights_fse_ram_rd_data;
  logic [7:0] huffman_literals_weights_fse_ram_rd_addr;
  logic huffman_literals_weights_fse_ram_rd_en;
  logic huffman_literals_weights_fse_ram_rd_mask;

  ram_1r1w #(
      .DATA_WIDTH(32),
      .SIZE(256),
      .NUM_PARTITIONS(1),
      .ADDR_WIDTH(8)
  ) huffman_literals_weights_fse_ram_ram (
      .clk(clk),
      .rst(rst),
      .wr_data(huffman_literals_weights_fse_ram_wr_data),
      .wr_addr(huffman_literals_weights_fse_ram_wr_addr),
      .wr_en(huffman_literals_weights_fse_ram_wr_en),
      .wr_mask(huffman_literals_weights_fse_ram_wr_mask),
      .rd_data(huffman_literals_weights_fse_ram_rd_data),
      .rd_addr(huffman_literals_weights_fse_ram_rd_addr),
      .rd_en(huffman_literals_weights_fse_ram_rd_en),
      .rd_mask(huffman_literals_weights_fse_ram_rd_mask)
  );

  /*
   * ZSTD Decoder instance
   */


  ZstdDecoder ZstdDecoder (
        .clk(clk),
        .rst(rst),

        .zstd_dec__axi_ram_ar_s__0(zstd_dec__axi_ram_ar_s__0),
        .zstd_dec__axi_ram_ar_s__0_vld(zstd_dec__axi_ram_ar_s__0_vld),
        .zstd_dec__axi_ram_ar_s__0_rdy(zstd_dec__axi_ram_ar_s__0_rdy),
        .zstd_dec__axi_ram_r_r__0(zstd_dec__axi_ram_r_r__0),
        .zstd_dec__axi_ram_r_r__0_vld(zstd_dec__axi_ram_r_r__0_vld),
        .zstd_dec__axi_ram_r_r__0_rdy(zstd_dec__axi_ram_r_r__0_rdy),

        .zstd_dec__axi_ram_ar_s__1(zstd_dec__axi_ram_ar_s__1),
        .zstd_dec__axi_ram_ar_s__1_vld(zstd_dec__axi_ram_ar_s__1_vld),
        .zstd_dec__axi_ram_ar_s__1_rdy(zstd_dec__axi_ram_ar_s__1_rdy),
        .zstd_dec__axi_ram_r_r__1(zstd_dec__axi_ram_r_r__1),
        .zstd_dec__axi_ram_r_r__1_vld(zstd_dec__axi_ram_r_r__1_vld),
        .zstd_dec__axi_ram_r_r__1_rdy(zstd_dec__axi_ram_r_r__1_rdy),

        .zstd_dec__axi_ram_ar_s__2(zstd_dec__axi_ram_ar_s__2),
        .zstd_dec__axi_ram_ar_s__2_vld(zstd_dec__axi_ram_ar_s__2_vld),
        .zstd_dec__axi_ram_ar_s__2_rdy(zstd_dec__axi_ram_ar_s__2_rdy),
        .zstd_dec__axi_ram_r_r__2(zstd_dec__axi_ram_r_r__2),
        .zstd_dec__axi_ram_r_r__2_vld(zstd_dec__axi_ram_r_r__2_vld),
        .zstd_dec__axi_ram_r_r__2_rdy(zstd_dec__axi_ram_r_r__2_rdy),

        .zstd_dec__axi_ram_ar_s__3(zstd_dec__axi_ram_ar_s__3),
        .zstd_dec__axi_ram_ar_s__3_vld(zstd_dec__axi_ram_ar_s__3_vld),
        .zstd_dec__axi_ram_ar_s__3_rdy(zstd_dec__axi_ram_ar_s__3_rdy),
        .zstd_dec__axi_ram_r_r__3(zstd_dec__axi_ram_r_r__3),
        .zstd_dec__axi_ram_r_r__3_vld(zstd_dec__axi_ram_r_r__3_vld),
        .zstd_dec__axi_ram_r_r__3_rdy(zstd_dec__axi_ram_r_r__3_rdy),

        .zstd_dec__axi_ram_ar_s__4(zstd_dec__axi_ram_ar_s__4),
        .zstd_dec__axi_ram_ar_s__4_vld(zstd_dec__axi_ram_ar_s__4_vld),
        .zstd_dec__axi_ram_ar_s__4_rdy(zstd_dec__axi_ram_ar_s__4_rdy),
        .zstd_dec__axi_ram_r_r__4(zstd_dec__axi_ram_r_r__4),
        .zstd_dec__axi_ram_r_r__4_vld(zstd_dec__axi_ram_r_r__4_vld),
        .zstd_dec__axi_ram_r_r__4_rdy(zstd_dec__axi_ram_r_r__4_rdy),

        .zstd_dec__axi_ram_ar_s__5(zstd_dec__axi_ram_ar_s__5),
        .zstd_dec__axi_ram_ar_s__5_vld(zstd_dec__axi_ram_ar_s__5_vld),
        .zstd_dec__axi_ram_ar_s__5_rdy(zstd_dec__axi_ram_ar_s__5_rdy),
        .zstd_dec__axi_ram_r_r__5(zstd_dec__axi_ram_r_r__5),
        .zstd_dec__axi_ram_r_r__5_vld(zstd_dec__axi_ram_r_r__5_vld),
        .zstd_dec__axi_ram_r_r__5_rdy(zstd_dec__axi_ram_r_r__5_rdy),

        .zstd_dec__axi_ram_ar_s__6(zstd_dec__axi_ram_ar_s__6),
        .zstd_dec__axi_ram_ar_s__6_vld(zstd_dec__axi_ram_ar_s__6_vld),
        .zstd_dec__axi_ram_ar_s__6_rdy(zstd_dec__axi_ram_ar_s__6_rdy),
        .zstd_dec__axi_ram_r_r__6(zstd_dec__axi_ram_r_r__6),
        .zstd_dec__axi_ram_r_r__6_vld(zstd_dec__axi_ram_r_r__6_vld),
        .zstd_dec__axi_ram_r_r__6_rdy(zstd_dec__axi_ram_r_r__6_rdy),

        .zstd_dec__axi_ram_ar_s__7(zstd_dec__axi_ram_ar_s__7),
        .zstd_dec__axi_ram_ar_s__7_vld(zstd_dec__axi_ram_ar_s__7_vld),
        .zstd_dec__axi_ram_ar_s__7_rdy(zstd_dec__axi_ram_ar_s__7_rdy),
        .zstd_dec__axi_ram_r_r__7(zstd_dec__axi_ram_r_r__7),
        .zstd_dec__axi_ram_r_r__7_vld(zstd_dec__axi_ram_r_r__7_vld),
        .zstd_dec__axi_ram_r_r__7_rdy(zstd_dec__axi_ram_r_r__7_rdy),

        .zstd_dec__axi_ram_ar_s__8(zstd_dec__axi_ram_ar_s__8),
        .zstd_dec__axi_ram_ar_s__8_vld(zstd_dec__axi_ram_ar_s__8_vld),
        .zstd_dec__axi_ram_ar_s__8_rdy(zstd_dec__axi_ram_ar_s__8_rdy),
        .zstd_dec__axi_ram_r_r__8(zstd_dec__axi_ram_r_r__8),
        .zstd_dec__axi_ram_r_r__8_vld(zstd_dec__axi_ram_r_r__8_vld),
        .zstd_dec__axi_ram_r_r__8_rdy(zstd_dec__axi_ram_r_r__8_rdy),

        .zstd_dec__axi_ram_ar_s__9(zstd_dec__axi_ram_ar_s__9),
        .zstd_dec__axi_ram_ar_s__9_vld(zstd_dec__axi_ram_ar_s__9_vld),
        .zstd_dec__axi_ram_ar_s__9_rdy(zstd_dec__axi_ram_ar_s__9_rdy),
        .zstd_dec__axi_ram_r_r__9(zstd_dec__axi_ram_r_r__9),
        .zstd_dec__axi_ram_r_r__9_vld(zstd_dec__axi_ram_r_r__9_vld),
        .zstd_dec__axi_ram_r_r__9_rdy(zstd_dec__axi_ram_r_r__9_rdy),

        .zstd_dec__axi_ram_ar_s__10(zstd_dec__axi_ram_ar_s__10),
        .zstd_dec__axi_ram_ar_s__10_vld(zstd_dec__axi_ram_ar_s__10_vld),
        .zstd_dec__axi_ram_ar_s__10_rdy(zstd_dec__axi_ram_ar_s__10_rdy),
        .zstd_dec__axi_ram_r_r__10(zstd_dec__axi_ram_r_r__10),
        .zstd_dec__axi_ram_r_r__10_vld(zstd_dec__axi_ram_r_r__10_vld),
        .zstd_dec__axi_ram_r_r__10_rdy(zstd_dec__axi_ram_r_r__10_rdy),

        // Huffman literals memories
        .huffman_literals_weights_dpd_ram_rd_data(huffman_literals_weights_dpd_ram_rd_data),
        .huffman_literals_weights_dpd_ram_rd_addr(huffman_literals_weights_dpd_ram_rd_addr),
        .huffman_literals_weights_dpd_ram_rd_mask(huffman_literals_weights_dpd_ram_rd_mask),
        .huffman_literals_weights_dpd_ram_rd_en(huffman_literals_weights_dpd_ram_rd_en),
        .huffman_literals_weights_dpd_ram_wr_addr(huffman_literals_weights_dpd_ram_wr_addr),
        .huffman_literals_weights_dpd_ram_wr_data(huffman_literals_weights_dpd_ram_wr_data),
        .huffman_literals_weights_dpd_ram_wr_mask(huffman_literals_weights_dpd_ram_wr_mask),
        .huffman_literals_weights_dpd_ram_wr_en(huffman_literals_weights_dpd_ram_wr_en),

        .huffman_literals_weights_tmp_ram_rd_data(huffman_literals_weights_tmp_ram_rd_data),
        .huffman_literals_weights_tmp_ram_rd_addr(huffman_literals_weights_tmp_ram_rd_addr),
        .huffman_literals_weights_tmp_ram_rd_mask(huffman_literals_weights_tmp_ram_rd_mask),
        .huffman_literals_weights_tmp_ram_rd_en(huffman_literals_weights_tmp_ram_rd_en),
        .huffman_literals_weights_tmp_ram_wr_addr(huffman_literals_weights_tmp_ram_wr_addr),
        .huffman_literals_weights_tmp_ram_wr_data(huffman_literals_weights_tmp_ram_wr_data),
        .huffman_literals_weights_tmp_ram_wr_mask(huffman_literals_weights_tmp_ram_wr_mask),
        .huffman_literals_weights_tmp_ram_wr_en(huffman_literals_weights_tmp_ram_wr_en),

        .huffman_literals_weights_tmp2_ram_rd_data(huffman_literals_weights_tmp2_ram_rd_data),
        .huffman_literals_weights_tmp2_ram_rd_addr(huffman_literals_weights_tmp2_ram_rd_addr),
        .huffman_literals_weights_tmp2_ram_rd_mask(huffman_literals_weights_tmp2_ram_rd_mask),
        .huffman_literals_weights_tmp2_ram_rd_en(huffman_literals_weights_tmp2_ram_rd_en),
        .huffman_literals_weights_tmp2_ram_wr_addr(huffman_literals_weights_tmp2_ram_wr_addr),
        .huffman_literals_weights_tmp2_ram_wr_data(huffman_literals_weights_tmp2_ram_wr_data),
        .huffman_literals_weights_tmp2_ram_wr_mask(huffman_literals_weights_tmp2_ram_wr_mask),
        .huffman_literals_weights_tmp2_ram_wr_en(huffman_literals_weights_tmp2_ram_wr_en),

        .huffman_literals_weights_fse_ram_rd_data(huffman_literals_weights_fse_ram_rd_data),
        .huffman_literals_weights_fse_ram_rd_addr(huffman_literals_weights_fse_ram_rd_addr),
        .huffman_literals_weights_fse_ram_rd_mask(huffman_literals_weights_fse_ram_rd_mask),
        .huffman_literals_weights_fse_ram_rd_en(huffman_literals_weights_fse_ram_rd_en),
        .huffman_literals_weights_fse_ram_wr_addr(huffman_literals_weights_fse_ram_wr_addr),
        .huffman_literals_weights_fse_ram_wr_data(huffman_literals_weights_fse_ram_wr_data),
        .huffman_literals_weights_fse_ram_wr_mask(huffman_literals_weights_fse_ram_wr_mask),
        .huffman_literals_weights_fse_ram_wr_en(huffman_literals_weights_fse_ram_wr_en),

        // History buffers memories
        .history_buffer_ram0_rd_data(history_buffer_ram0_rd_data),
        .history_buffer_ram0_rd_addr(history_buffer_ram0_rd_addr),
        .history_buffer_ram0_rd_mask(history_buffer_ram0_rd_mask),
        .history_buffer_ram0_rd_en(history_buffer_ram0_rd_en),
        .history_buffer_ram0_wr_addr(history_buffer_ram0_wr_addr),
        .history_buffer_ram0_wr_data(history_buffer_ram0_wr_data),
        .history_buffer_ram0_wr_mask(history_buffer_ram0_wr_mask),
        .history_buffer_ram0_wr_en(history_buffer_ram0_wr_en),

        .history_buffer_ram1_rd_data(history_buffer_ram1_rd_data),
        .history_buffer_ram1_rd_addr(history_buffer_ram1_rd_addr),
        .history_buffer_ram1_rd_mask(history_buffer_ram1_rd_mask),
        .history_buffer_ram1_rd_en(history_buffer_ram1_rd_en),
        .history_buffer_ram1_wr_addr(history_buffer_ram1_wr_addr),
        .history_buffer_ram1_wr_data(history_buffer_ram1_wr_data),
        .history_buffer_ram1_wr_mask(history_buffer_ram1_wr_mask),
        .history_buffer_ram1_wr_en(history_buffer_ram1_wr_en),

        .history_buffer_ram2_rd_data(history_buffer_ram2_rd_data),
        .history_buffer_ram2_rd_addr(history_buffer_ram2_rd_addr),
        .history_buffer_ram2_rd_mask(history_buffer_ram2_rd_mask),
        .history_buffer_ram2_rd_en(history_buffer_ram2_rd_en),
        .history_buffer_ram2_wr_addr(history_buffer_ram2_wr_addr),
        .history_buffer_ram2_wr_data(history_buffer_ram2_wr_data),
        .history_buffer_ram2_wr_mask(history_buffer_ram2_wr_mask),
        .history_buffer_ram2_wr_en(history_buffer_ram2_wr_en),

        .history_buffer_ram3_rd_data(history_buffer_ram3_rd_data),
        .history_buffer_ram3_rd_addr(history_buffer_ram3_rd_addr),
        .history_buffer_ram3_rd_mask(history_buffer_ram3_rd_mask),
        .history_buffer_ram3_rd_en(history_buffer_ram3_rd_en),
        .history_buffer_ram3_wr_addr(history_buffer_ram3_wr_addr),
        .history_buffer_ram3_wr_data(history_buffer_ram3_wr_data),
        .history_buffer_ram3_wr_mask(history_buffer_ram3_wr_mask),
        .history_buffer_ram3_wr_en(history_buffer_ram3_wr_en),

        .history_buffer_ram4_rd_data(history_buffer_ram4_rd_data),
        .history_buffer_ram4_rd_addr(history_buffer_ram4_rd_addr),
        .history_buffer_ram4_rd_mask(history_buffer_ram4_rd_mask),
        .history_buffer_ram4_rd_en(history_buffer_ram4_rd_en),
        .history_buffer_ram4_wr_addr(history_buffer_ram4_wr_addr),
        .history_buffer_ram4_wr_data(history_buffer_ram4_wr_data),
        .history_buffer_ram4_wr_mask(history_buffer_ram4_wr_mask),
        .history_buffer_ram4_wr_en(history_buffer_ram4_wr_en),

        .history_buffer_ram5_rd_data(history_buffer_ram5_rd_data),
        .history_buffer_ram5_rd_addr(history_buffer_ram5_rd_addr),
        .history_buffer_ram5_rd_mask(history_buffer_ram5_rd_mask),
        .history_buffer_ram5_rd_en(history_buffer_ram5_rd_en),
        .history_buffer_ram5_wr_addr(history_buffer_ram5_wr_addr),
        .history_buffer_ram5_wr_data(history_buffer_ram5_wr_data),
        .history_buffer_ram5_wr_mask(history_buffer_ram5_wr_mask),
        .history_buffer_ram5_wr_en(history_buffer_ram5_wr_en),

        .history_buffer_ram6_rd_data(history_buffer_ram6_rd_data),
        .history_buffer_ram6_rd_addr(history_buffer_ram6_rd_addr),
        .history_buffer_ram6_rd_mask(history_buffer_ram6_rd_mask),
        .history_buffer_ram6_rd_en(history_buffer_ram6_rd_en),
        .history_buffer_ram6_wr_addr(history_buffer_ram6_wr_addr),
        .history_buffer_ram6_wr_data(history_buffer_ram6_wr_data),
        .history_buffer_ram6_wr_mask(history_buffer_ram6_wr_mask),
        .history_buffer_ram6_wr_en(history_buffer_ram6_wr_en),

        .history_buffer_ram7_rd_data(history_buffer_ram7_rd_data),
        .history_buffer_ram7_rd_addr(history_buffer_ram7_rd_addr),
        .history_buffer_ram7_rd_mask(history_buffer_ram7_rd_mask),
        .history_buffer_ram7_rd_en(history_buffer_ram7_rd_en),
        .history_buffer_ram7_wr_addr(history_buffer_ram7_wr_addr),
        .history_buffer_ram7_wr_data(history_buffer_ram7_wr_data),
        .history_buffer_ram7_wr_mask(history_buffer_ram7_wr_mask),
        .history_buffer_ram7_wr_en(history_buffer_ram7_wr_en),

        .dpd_ram_rd_data(dpd_ram_rd_data),
        .dpd_ram_rd_addr(dpd_ram_rd_addr),
        .dpd_ram_rd_mask(dpd_ram_rd_mask),
        .dpd_ram_rd_en(dpd_ram_rd_en),
        .dpd_ram_wr_addr(dpd_ram_wr_addr),
        .dpd_ram_wr_data(dpd_ram_wr_data),
        .dpd_ram_wr_mask(dpd_ram_wr_mask),
        .dpd_ram_wr_en(dpd_ram_wr_en),

        .fse_tmp_ram_rd_data(fse_tmp_ram_rd_data),
        .fse_tmp_ram_rd_addr(fse_tmp_ram_rd_addr),
        .fse_tmp_ram_rd_mask(fse_tmp_ram_rd_mask),
        .fse_tmp_ram_rd_en(fse_tmp_ram_rd_en),
        .fse_tmp_ram_wr_addr(fse_tmp_ram_wr_addr),
        .fse_tmp_ram_wr_data(fse_tmp_ram_wr_data),
        .fse_tmp_ram_wr_mask(fse_tmp_ram_wr_mask),
        .fse_tmp_ram_wr_en(fse_tmp_ram_wr_en),

        .fse_tmp2_ram_rd_data(fse_tmp2_ram_rd_data),
        .fse_tmp2_ram_rd_addr(fse_tmp2_ram_rd_addr),
        .fse_tmp2_ram_rd_mask(fse_tmp2_ram_rd_mask),
        .fse_tmp2_ram_rd_en(fse_tmp2_ram_rd_en),
        .fse_tmp2_ram_wr_addr(fse_tmp2_ram_wr_addr),
        .fse_tmp2_ram_wr_data(fse_tmp2_ram_wr_data),
        .fse_tmp2_ram_wr_mask(fse_tmp2_ram_wr_mask),
        .fse_tmp2_ram_wr_en(fse_tmp2_ram_wr_en),

        .ll_def_fse_ram_rd_data(ll_def_fse_ram_rd_data),
        .ll_def_fse_ram_rd_addr(ll_def_fse_ram_rd_addr),
        .ll_def_fse_ram_rd_mask(ll_def_fse_ram_rd_mask),
        .ll_def_fse_ram_rd_en(ll_def_fse_ram_rd_en),
        .ll_def_fse_ram_wr_addr(ll_def_fse_ram_wr_addr),
        .ll_def_fse_ram_wr_data(ll_def_fse_ram_wr_data),
        .ll_def_fse_ram_wr_mask(ll_def_fse_ram_wr_mask),
        .ll_def_fse_ram_wr_en(ll_def_fse_ram_wr_en),

        .ll_fse_ram_rd_data(ll_fse_ram_rd_data),
        .ll_fse_ram_rd_addr(ll_fse_ram_rd_addr),
        .ll_fse_ram_rd_mask(ll_fse_ram_rd_mask),
        .ll_fse_ram_rd_en(ll_fse_ram_rd_en),
        .ll_fse_ram_wr_addr(ll_fse_ram_wr_addr),
        .ll_fse_ram_wr_data(ll_fse_ram_wr_data),
        .ll_fse_ram_wr_mask(ll_fse_ram_wr_mask),
        .ll_fse_ram_wr_en(ll_fse_ram_wr_en),

        .ml_def_fse_ram_rd_data(ml_def_fse_ram_rd_data),
        .ml_def_fse_ram_rd_addr(ml_def_fse_ram_rd_addr),
        .ml_def_fse_ram_rd_mask(ml_def_fse_ram_rd_mask),
        .ml_def_fse_ram_rd_en(ml_def_fse_ram_rd_en),
        .ml_def_fse_ram_wr_addr(ml_def_fse_ram_wr_addr),
        .ml_def_fse_ram_wr_data(ml_def_fse_ram_wr_data),
        .ml_def_fse_ram_wr_mask(ml_def_fse_ram_wr_mask),
        .ml_def_fse_ram_wr_en(ml_def_fse_ram_wr_en),

        .ml_fse_ram_rd_data(ml_fse_ram_rd_data),
        .ml_fse_ram_rd_addr(ml_fse_ram_rd_addr),
        .ml_fse_ram_rd_mask(ml_fse_ram_rd_mask),
        .ml_fse_ram_rd_en(ml_fse_ram_rd_en),
        .ml_fse_ram_wr_addr(ml_fse_ram_wr_addr),
        .ml_fse_ram_wr_data(ml_fse_ram_wr_data),
        .ml_fse_ram_wr_mask(ml_fse_ram_wr_mask),
        .ml_fse_ram_wr_en(ml_fse_ram_wr_en),

        .of_def_fse_ram_rd_data(of_def_fse_ram_rd_data),
        .of_def_fse_ram_rd_addr(of_def_fse_ram_rd_addr),
        .of_def_fse_ram_rd_mask(of_def_fse_ram_rd_mask),
        .of_def_fse_ram_rd_en(of_def_fse_ram_rd_en),
        .of_def_fse_ram_wr_addr(of_def_fse_ram_wr_addr),
        .of_def_fse_ram_wr_data(of_def_fse_ram_wr_data),
        .of_def_fse_ram_wr_mask(of_def_fse_ram_wr_mask),
        .of_def_fse_ram_wr_en(of_def_fse_ram_wr_en),

        .of_fse_ram_rd_data(of_fse_ram_rd_data),
        .of_fse_ram_rd_addr(of_fse_ram_rd_addr),
        .of_fse_ram_rd_mask(of_fse_ram_rd_mask),
        .of_fse_ram_rd_en(of_fse_ram_rd_en),
        .of_fse_ram_wr_addr(of_fse_ram_wr_addr),
        .of_fse_ram_wr_data(of_fse_ram_wr_data),
        .of_fse_ram_wr_mask(of_fse_ram_wr_mask),
        .of_fse_ram_wr_en(of_fse_ram_wr_en),

        .huffman_literals_prescan_ram_rd_data(huffman_literals_prescan_ram_rd_data),
        .huffman_literals_prescan_ram_rd_addr(huffman_literals_prescan_ram_rd_addr),
        .huffman_literals_prescan_ram_rd_mask(huffman_literals_prescan_ram_rd_mask),
        .huffman_literals_prescan_ram_rd_en(huffman_literals_prescan_ram_rd_en),
        .huffman_literals_prescan_ram_wr_addr(huffman_literals_prescan_ram_wr_addr),
        .huffman_literals_prescan_ram_wr_data(huffman_literals_prescan_ram_wr_data),
        .huffman_literals_prescan_ram_wr_mask(huffman_literals_prescan_ram_wr_mask),
        .huffman_literals_prescan_ram_wr_en(huffman_literals_prescan_ram_wr_en),

        .huffman_literals_weights_mem_ram_rd_data(huffman_literals_weights_mem_ram_rd_data),
        .huffman_literals_weights_mem_ram_rd_addr(huffman_literals_weights_mem_ram_rd_addr),
        .huffman_literals_weights_mem_ram_rd_mask(huffman_literals_weights_mem_ram_rd_mask),
        .huffman_literals_weights_mem_ram_rd_en(huffman_literals_weights_mem_ram_rd_en),
        .huffman_literals_weights_mem_ram_wr_addr(huffman_literals_weights_mem_ram_wr_addr),
        .huffman_literals_weights_mem_ram_wr_data(huffman_literals_weights_mem_ram_wr_data),
        .huffman_literals_weights_mem_ram_wr_mask(huffman_literals_weights_mem_ram_wr_mask),
        .huffman_literals_weights_mem_ram_wr_en(huffman_literals_weights_mem_ram_wr_en),

        .literals_buffer_ram0_rd_data(literals_buffer_ram0_rd_data),
        .literals_buffer_ram0_rd_addr(literals_buffer_ram0_rd_addr),
        .literals_buffer_ram0_rd_mask(literals_buffer_ram0_rd_mask),
        .literals_buffer_ram0_rd_en(literals_buffer_ram0_rd_en),
        .literals_buffer_ram0_wr_addr(literals_buffer_ram0_wr_addr),
        .literals_buffer_ram0_wr_data(literals_buffer_ram0_wr_data),
        .literals_buffer_ram0_wr_mask(literals_buffer_ram0_wr_mask),
        .literals_buffer_ram0_wr_en(literals_buffer_ram0_wr_en),

        .literals_buffer_ram1_rd_data(literals_buffer_ram1_rd_data),
        .literals_buffer_ram1_rd_addr(literals_buffer_ram1_rd_addr),
        .literals_buffer_ram1_rd_mask(literals_buffer_ram1_rd_mask),
        .literals_buffer_ram1_rd_en(literals_buffer_ram1_rd_en),
        .literals_buffer_ram1_wr_addr(literals_buffer_ram1_wr_addr),
        .literals_buffer_ram1_wr_data(literals_buffer_ram1_wr_data),
        .literals_buffer_ram1_wr_mask(literals_buffer_ram1_wr_mask),
        .literals_buffer_ram1_wr_en(literals_buffer_ram1_wr_en),

        .literals_buffer_ram2_rd_data(literals_buffer_ram2_rd_data),
        .literals_buffer_ram2_rd_addr(literals_buffer_ram2_rd_addr),
        .literals_buffer_ram2_rd_mask(literals_buffer_ram2_rd_mask),
        .literals_buffer_ram2_rd_en(literals_buffer_ram2_rd_en),
        .literals_buffer_ram2_wr_addr(literals_buffer_ram2_wr_addr),
        .literals_buffer_ram2_wr_data(literals_buffer_ram2_wr_data),
        .literals_buffer_ram2_wr_mask(literals_buffer_ram2_wr_mask),
        .literals_buffer_ram2_wr_en(literals_buffer_ram2_wr_en),

        .literals_buffer_ram3_rd_data(literals_buffer_ram3_rd_data),
        .literals_buffer_ram3_rd_addr(literals_buffer_ram3_rd_addr),
        .literals_buffer_ram3_rd_mask(literals_buffer_ram3_rd_mask),
        .literals_buffer_ram3_rd_en(literals_buffer_ram3_rd_en),
        .literals_buffer_ram3_wr_addr(literals_buffer_ram3_wr_addr),
        .literals_buffer_ram3_wr_data(literals_buffer_ram3_wr_data),
        .literals_buffer_ram3_wr_mask(literals_buffer_ram3_wr_mask),
        .literals_buffer_ram3_wr_en(literals_buffer_ram3_wr_en),

        .literals_buffer_ram4_rd_data(literals_buffer_ram4_rd_data),
        .literals_buffer_ram4_rd_addr(literals_buffer_ram4_rd_addr),
        .literals_buffer_ram4_rd_mask(literals_buffer_ram4_rd_mask),
        .literals_buffer_ram4_rd_en(literals_buffer_ram4_rd_en),
        .literals_buffer_ram4_wr_addr(literals_buffer_ram4_wr_addr),
        .literals_buffer_ram4_wr_data(literals_buffer_ram4_wr_data),
        .literals_buffer_ram4_wr_mask(literals_buffer_ram4_wr_mask),
        .literals_buffer_ram4_wr_en(literals_buffer_ram4_wr_en),

        .literals_buffer_ram5_rd_data(literals_buffer_ram5_rd_data),
        .literals_buffer_ram5_rd_addr(literals_buffer_ram5_rd_addr),
        .literals_buffer_ram5_rd_mask(literals_buffer_ram5_rd_mask),
        .literals_buffer_ram5_rd_en(literals_buffer_ram5_rd_en),
        .literals_buffer_ram5_wr_addr(literals_buffer_ram5_wr_addr),
        .literals_buffer_ram5_wr_data(literals_buffer_ram5_wr_data),
        .literals_buffer_ram5_wr_mask(literals_buffer_ram5_wr_mask),
        .literals_buffer_ram5_wr_en(literals_buffer_ram5_wr_en),

        .literals_buffer_ram6_rd_data(literals_buffer_ram6_rd_data),
        .literals_buffer_ram6_rd_addr(literals_buffer_ram6_rd_addr),
        .literals_buffer_ram6_rd_mask(literals_buffer_ram6_rd_mask),
        .literals_buffer_ram6_rd_en(literals_buffer_ram6_rd_en),
        .literals_buffer_ram6_wr_addr(literals_buffer_ram6_wr_addr),
        .literals_buffer_ram6_wr_data(literals_buffer_ram6_wr_data),
        .literals_buffer_ram6_wr_mask(literals_buffer_ram6_wr_mask),
        .literals_buffer_ram6_wr_en(literals_buffer_ram6_wr_en),

        .literals_buffer_ram7_rd_data(literals_buffer_ram7_rd_data),
        .literals_buffer_ram7_rd_addr(literals_buffer_ram7_rd_addr),
        .literals_buffer_ram7_rd_mask(literals_buffer_ram7_rd_mask),
        .literals_buffer_ram7_rd_en(literals_buffer_ram7_rd_en),
        .literals_buffer_ram7_wr_addr(literals_buffer_ram7_wr_addr),
        .literals_buffer_ram7_wr_data(literals_buffer_ram7_wr_data),
        .literals_buffer_ram7_wr_mask(literals_buffer_ram7_wr_mask),
        .literals_buffer_ram7_wr_en(literals_buffer_ram7_wr_en),

        // CSR Interface
        .zstd_dec__csr_axi_aw_r(zstd_dec__csr_axi_aw),
        .zstd_dec__csr_axi_aw_r_vld(zstd_dec__csr_axi_aw_vld),
        .zstd_dec__csr_axi_aw_r_rdy(zstd_dec__csr_axi_aw_rdy),
        .zstd_dec__csr_axi_w_r(zstd_dec__csr_axi_w),
        .zstd_dec__csr_axi_w_r_vld(zstd_dec__csr_axi_w_vld),
        .zstd_dec__csr_axi_w_r_rdy(zstd_dec__csr_axi_w_rdy),
        .zstd_dec__csr_axi_b_s(zstd_dec__csr_axi_b),
        .zstd_dec__csr_axi_b_s_vld(zstd_dec__csr_axi_b_vld),
        .zstd_dec__csr_axi_b_s_rdy(zstd_dec__csr_axi_b_rdy),
        .zstd_dec__csr_axi_ar_r(zstd_dec__csr_axi_ar),
        .zstd_dec__csr_axi_ar_r_vld(zstd_dec__csr_axi_ar_vld),
        .zstd_dec__csr_axi_ar_r_rdy(zstd_dec__csr_axi_ar_rdy),
        .zstd_dec__csr_axi_r_s(zstd_dec__csr_axi_r),
        .zstd_dec__csr_axi_r_s_vld(zstd_dec__csr_axi_r_vld),
        .zstd_dec__csr_axi_r_s_rdy(zstd_dec__csr_axi_r_rdy),

        // FrameHeaderDecoder
        .zstd_dec__fh_axi_ar_s(zstd_dec__fh_axi_ar),
        .zstd_dec__fh_axi_ar_s_vld(zstd_dec__fh_axi_ar_vld),
        .zstd_dec__fh_axi_ar_s_rdy(zstd_dec__fh_axi_ar_rdy),
        .zstd_dec__fh_axi_r_r(zstd_dec__fh_axi_r),
        .zstd_dec__fh_axi_r_r_vld(zstd_dec__fh_axi_r_vld),
        .zstd_dec__fh_axi_r_r_rdy(zstd_dec__fh_axi_r_rdy),

        // BlockHeaderDecoder
        .zstd_dec__bh_axi_ar_s(zstd_dec__bh_axi_ar),
        .zstd_dec__bh_axi_ar_s_vld(zstd_dec__bh_axi_ar_vld),
        .zstd_dec__bh_axi_ar_s_rdy(zstd_dec__bh_axi_ar_rdy),
        .zstd_dec__bh_axi_r_r(zstd_dec__bh_axi_r),
        .zstd_dec__bh_axi_r_r_vld(zstd_dec__bh_axi_r_vld),
        .zstd_dec__bh_axi_r_r_rdy(zstd_dec__bh_axi_r_rdy),

        // RawBlockDecoder
        .zstd_dec__raw_axi_ar_s(zstd_dec__raw_axi_ar),
        .zstd_dec__raw_axi_ar_s_vld(zstd_dec__raw_axi_ar_vld),
        .zstd_dec__raw_axi_ar_s_rdy(zstd_dec__raw_axi_ar_rdy),
        .zstd_dec__raw_axi_r_r(zstd_dec__raw_axi_r),
        .zstd_dec__raw_axi_r_r_vld(zstd_dec__raw_axi_r_vld),
        .zstd_dec__raw_axi_r_r_rdy(zstd_dec__raw_axi_r_rdy),

        // Output Writer
        .zstd_dec__output_axi_aw_s(zstd_dec__output_axi_aw),
        .zstd_dec__output_axi_aw_s_vld(zstd_dec__output_axi_aw_vld),
        .zstd_dec__output_axi_aw_s_rdy(zstd_dec__output_axi_aw_rdy),
        .zstd_dec__output_axi_w_s(zstd_dec__output_axi_w),
        .zstd_dec__output_axi_w_s_vld(zstd_dec__output_axi_w_vld),
        .zstd_dec__output_axi_w_s_rdy(zstd_dec__output_axi_w_rdy),
        .zstd_dec__output_axi_b_r(zstd_dec__output_axi_b),
        .zstd_dec__output_axi_b_r_vld(zstd_dec__output_axi_b_vld),
        .zstd_dec__output_axi_b_r_rdy(zstd_dec__output_axi_b_rdy),

        // Other ports
        .zstd_dec__notify_s_vld(notify_vld),
        .zstd_dec__notify_s_rdy(notify_rdy)
  );

  assign frame_header_decoder_axi_r_rresp[2] = '0;
  assign block_header_decoder_axi_r_rresp[2] = '0;
  assign raw_block_decoder_axi_r_rresp[2] = '0;
  assign output_axi_b_bresp[2] = '0;
  assign memory_axi_b_bresp[2] = '0;
  assign memory_axi_r_rresp[2] = '0;
  /*
   * AXI Interconnect
   */

// parameter M_ID_WIDTH = S_ID_WIDTH+$clog2(S_COUNT),

  axi_crossbar_wrapper #(
      .DATA_WIDTH(AXI_DATA_W),
      .ADDR_WIDTH(AXI_ADDR_W),
      .M00_ADDR_WIDTH(AXI_ADDR_W),
      .M00_BASE_ADDR(32'd0),
      .STRB_WIDTH(AXI_STRB_W),
      .S_ID_WIDTH(S_AXI_ID_W),
      .M_ID_WIDTH(M_AXI_ID_W)
  ) axi_memory_interconnect (
      .clk(clk),
      .rst(rst),

      /*
       * AXI Subordinate interfaces
       */
      // FrameHeaderDecoder
      .s00_axi_awid('0),
      .s00_axi_awaddr('0),
      .s00_axi_awlen('0),
      .s00_axi_awsize('0),
      .s00_axi_awburst('0),
      .s00_axi_awlock('0),
      .s00_axi_awcache('0),
      .s00_axi_awprot('0),
      .s00_axi_awqos('0),
      .s00_axi_awuser('0),
      .s00_axi_awvalid('0),
      .s00_axi_awready(),
      .s00_axi_wdata('0),
      .s00_axi_wstrb('0),
      .s00_axi_wlast('0),
      .s00_axi_wuser('0),
      .s00_axi_wvalid('0),
      .s00_axi_wready(),
      .s00_axi_bid(),
      .s00_axi_bresp(),
      .s00_axi_buser(),
      .s00_axi_bvalid(),
      .s00_axi_bready('0),
      .s00_axi_arid(frame_header_decoder_axi_ar_arid),
      .s00_axi_araddr(frame_header_decoder_axi_ar_araddr),
      .s00_axi_arlen(frame_header_decoder_axi_ar_arlen),
      .s00_axi_arsize(frame_header_decoder_axi_ar_arsize),
      .s00_axi_arburst(frame_header_decoder_axi_ar_arburst),
      .s00_axi_arlock('0),
      .s00_axi_arcache(frame_header_decoder_axi_ar_arcache),
      .s00_axi_arprot(frame_header_decoder_axi_ar_arprot),
      .s00_axi_arqos(frame_header_decoder_axi_ar_arqos),
      .s00_axi_aruser('0),
      .s00_axi_arvalid(frame_header_decoder_axi_ar_arvalid),
      .s00_axi_arready(frame_header_decoder_axi_ar_arready),
      .s00_axi_rid(frame_header_decoder_axi_r_rid),
      .s00_axi_rdata(frame_header_decoder_axi_r_rdata),
      .s00_axi_rresp(frame_header_decoder_axi_r_rresp[1:0]),
      .s00_axi_rlast(frame_header_decoder_axi_r_rlast),
      .s00_axi_ruser(),
      .s00_axi_rvalid(frame_header_decoder_axi_r_rvalid),
      .s00_axi_rready(frame_header_decoder_axi_r_rready),

      // BlockHeaderDecoder
      .s01_axi_awid('0),
      .s01_axi_awaddr('0),
      .s01_axi_awlen('0),
      .s01_axi_awsize('0),
      .s01_axi_awburst('0),
      .s01_axi_awlock('0),
      .s01_axi_awcache('0),
      .s01_axi_awprot('0),
      .s01_axi_awqos('0),
      .s01_axi_awuser('0),
      .s01_axi_awvalid('0),
      .s01_axi_awready(),
      .s01_axi_wdata('0),
      .s01_axi_wstrb('0),
      .s01_axi_wlast('0),
      .s01_axi_wuser('0),
      .s01_axi_wvalid('0),
      .s01_axi_wready(),
      .s01_axi_bid(),
      .s01_axi_bresp(),
      .s01_axi_buser(),
      .s01_axi_bvalid(),
      .s01_axi_bready('0),
      .s01_axi_arid(block_header_decoder_axi_ar_arid),
      .s01_axi_araddr(block_header_decoder_axi_ar_araddr),
      .s01_axi_arlen(block_header_decoder_axi_ar_arlen),
      .s01_axi_arsize(block_header_decoder_axi_ar_arsize),
      .s01_axi_arburst(block_header_decoder_axi_ar_arburst),
      .s01_axi_arlock('0),
      .s01_axi_arcache(block_header_decoder_axi_ar_arcache),
      .s01_axi_arprot(block_header_decoder_axi_ar_arprot),
      .s01_axi_arqos(block_header_decoder_axi_ar_arqos),
      .s01_axi_aruser('0),
      .s01_axi_arvalid(block_header_decoder_axi_ar_arvalid),
      .s01_axi_arready(block_header_decoder_axi_ar_arready),
      .s01_axi_rid(block_header_decoder_axi_r_rid),
      .s01_axi_rdata(block_header_decoder_axi_r_rdata),
      .s01_axi_rresp(block_header_decoder_axi_r_rresp[1:0]),
      .s01_axi_rlast(block_header_decoder_axi_r_rlast),
      .s01_axi_ruser(),
      .s01_axi_rvalid(block_header_decoder_axi_r_rvalid),
      .s01_axi_rready(block_header_decoder_axi_r_rready),

      // RawBlockDecoder
      .s02_axi_awid('0),
      .s02_axi_awaddr('0),
      .s02_axi_awlen('0),
      .s02_axi_awsize('0),
      .s02_axi_awburst('0),
      .s02_axi_awlock('0),
      .s02_axi_awcache('0),
      .s02_axi_awprot('0),
      .s02_axi_awqos('0),
      .s02_axi_awuser('0),
      .s02_axi_awvalid('0),
      .s02_axi_awready(),
      .s02_axi_wdata('0),
      .s02_axi_wstrb('0),
      .s02_axi_wlast('0),
      .s02_axi_wuser('0),
      .s02_axi_wvalid('0),
      .s02_axi_wready(),
      .s02_axi_bid(),
      .s02_axi_bresp(),
      .s02_axi_buser(),
      .s02_axi_bvalid(),
      .s02_axi_bready('0),
      .s02_axi_arid(raw_block_decoder_axi_ar_arid),
      .s02_axi_araddr(raw_block_decoder_axi_ar_araddr),
      .s02_axi_arlen(raw_block_decoder_axi_ar_arlen),
      .s02_axi_arsize(raw_block_decoder_axi_ar_arsize),
      .s02_axi_arburst(raw_block_decoder_axi_ar_arburst),
      .s02_axi_arlock('0),
      .s02_axi_arcache(raw_block_decoder_axi_ar_arcache),
      .s02_axi_arprot(raw_block_decoder_axi_ar_arprot),
      .s02_axi_arqos(raw_block_decoder_axi_ar_arqos),
      .s02_axi_aruser('0),
      .s02_axi_arvalid(raw_block_decoder_axi_ar_arvalid),
      .s02_axi_arready(raw_block_decoder_axi_ar_arready),
      .s02_axi_rid(raw_block_decoder_axi_r_rid),
      .s02_axi_rdata(raw_block_decoder_axi_r_rdata),
      .s02_axi_rresp(raw_block_decoder_axi_r_rresp[1:0]),
      .s02_axi_rlast(raw_block_decoder_axi_r_rlast),
      .s02_axi_ruser(),
      .s02_axi_rvalid(raw_block_decoder_axi_r_rvalid),
      .s02_axi_rready(raw_block_decoder_axi_r_rready),

      // SequenceExecutor
      .s03_axi_awid(output_axi_aw_awid),
      .s03_axi_awaddr(output_axi_aw_awaddr),
      .s03_axi_awlen(output_axi_aw_awlen),
      .s03_axi_awsize(output_axi_aw_awsize),
      .s03_axi_awburst(output_axi_aw_awburst),
      .s03_axi_awlock('0),
      .s03_axi_awcache('0),
      .s03_axi_awprot('0),
      .s03_axi_awqos('0),
      .s03_axi_awuser('0),
      .s03_axi_awvalid(output_axi_aw_awvalid),
      .s03_axi_awready(output_axi_aw_awready),
      .s03_axi_wdata(output_axi_w_wdata),
      .s03_axi_wstrb(output_axi_w_wstrb),
      .s03_axi_wlast(output_axi_w_wlast),
      .s03_axi_wuser('0),
      .s03_axi_wvalid(output_axi_w_wvalid),
      .s03_axi_wready(output_axi_w_wready),
      .s03_axi_bid(output_axi_b_bid),
      .s03_axi_bresp(output_axi_b_bresp[1:0]),
      .s03_axi_buser(),
      .s03_axi_bvalid(output_axi_b_bvalid),
      .s03_axi_bready(output_axi_b_bready),
      .s03_axi_arid('0),
      .s03_axi_araddr('0),
      .s03_axi_arlen('0),
      .s03_axi_arsize('0),
      .s03_axi_arburst('0),
      .s03_axi_arlock('0),
      .s03_axi_arcache('0),
      .s03_axi_arprot('0),
      .s03_axi_arqos('0),
      .s03_axi_aruser('0),
      .s03_axi_arvalid('0),
      .s03_axi_arready(),
      .s03_axi_rid(),
      .s03_axi_rdata(),
      .s03_axi_rresp(),
      .s03_axi_rlast(),
      .s03_axi_ruser(),
      .s03_axi_rvalid(),
      .s03_axi_rready('0),

      // axi_ram_s__0
      .s04_axi_awid('0),
      .s04_axi_awaddr('0),
      .s04_axi_awlen('0),
      .s04_axi_awsize('0),
      .s04_axi_awburst('0),
      .s04_axi_awlock('0),
      .s04_axi_awcache('0),
      .s04_axi_awprot('0),
      .s04_axi_awqos('0),
      .s04_axi_awuser('0),
      .s04_axi_awvalid('0),
      .s04_axi_awready(),
      .s04_axi_wdata('0),
      .s04_axi_wstrb('0),
      .s04_axi_wlast('0),
      .s04_axi_wuser('0),
      .s04_axi_wvalid('0),
      .s04_axi_wready(),
      .s04_axi_bid(),
      .s04_axi_bresp(),
      .s04_axi_buser(),
      .s04_axi_bvalid(),
      .s04_axi_bready('0),
      .s04_axi_arid(axi_ram_s__0_axi_ar_arid),
      .s04_axi_araddr(axi_ram_s__0_axi_ar_araddr),
      .s04_axi_arlen(axi_ram_s__0_axi_ar_arlen),
      .s04_axi_arsize(axi_ram_s__0_axi_ar_arsize),
      .s04_axi_arburst(axi_ram_s__0_axi_ar_arburst),
      .s04_axi_arlock('0),
      .s04_axi_arcache(axi_ram_s__0_axi_ar_arcache),
      .s04_axi_arprot(axi_ram_s__0_axi_ar_arprot),
      .s04_axi_arqos(axi_ram_s__0_axi_ar_arqos),
      .s04_axi_aruser('0),
      .s04_axi_arvalid(axi_ram_s__0_axi_ar_arvalid),
      .s04_axi_arready(axi_ram_s__0_axi_ar_arready),
      .s04_axi_rid(axi_ram_s__0_axi_r_rid),
      .s04_axi_rdata(axi_ram_s__0_axi_r_rdata),
      .s04_axi_rresp(axi_ram_s__0_axi_r_rresp[1:0]),
      .s04_axi_rlast(axi_ram_s__0_axi_r_rlast),
      .s04_axi_ruser(),
      .s04_axi_rvalid(axi_ram_s__0_axi_r_rvalid),
      .s04_axi_rready(axi_ram_s__0_axi_r_rready),

      // axi_ram_s__1
      .s05_axi_awid('0),
      .s05_axi_awaddr('0),
      .s05_axi_awlen('0),
      .s05_axi_awsize('0),
      .s05_axi_awburst('0),
      .s05_axi_awlock('0),
      .s05_axi_awcache('0),
      .s05_axi_awprot('0),
      .s05_axi_awqos('0),
      .s05_axi_awuser('0),
      .s05_axi_awvalid('0),
      .s05_axi_awready(),
      .s05_axi_wdata('0),
      .s05_axi_wstrb('0),
      .s05_axi_wlast('0),
      .s05_axi_wuser('0),
      .s05_axi_wvalid('0),
      .s05_axi_wready(),
      .s05_axi_bid(),
      .s05_axi_bresp(),
      .s05_axi_buser(),
      .s05_axi_bvalid(),
      .s05_axi_bready('0),
      .s05_axi_arid(axi_ram_s__1_axi_ar_arid),
      .s05_axi_araddr(axi_ram_s__1_axi_ar_araddr),
      .s05_axi_arlen(axi_ram_s__1_axi_ar_arlen),
      .s05_axi_arsize(axi_ram_s__1_axi_ar_arsize),
      .s05_axi_arburst(axi_ram_s__1_axi_ar_arburst),
      .s05_axi_arlock('0),
      .s05_axi_arcache(axi_ram_s__1_axi_ar_arcache),
      .s05_axi_arprot(axi_ram_s__1_axi_ar_arprot),
      .s05_axi_arqos(axi_ram_s__1_axi_ar_arqos),
      .s05_axi_aruser('0),
      .s05_axi_arvalid(axi_ram_s__1_axi_ar_arvalid),
      .s05_axi_arready(axi_ram_s__1_axi_ar_arready),
      .s05_axi_rid(axi_ram_s__1_axi_r_rid),
      .s05_axi_rdata(axi_ram_s__1_axi_r_rdata),
      .s05_axi_rresp(axi_ram_s__1_axi_r_rresp[1:0]),
      .s05_axi_rlast(axi_ram_s__1_axi_r_rlast),
      .s05_axi_ruser(),
      .s05_axi_rvalid(axi_ram_s__1_axi_r_rvalid),
      .s05_axi_rready(axi_ram_s__1_axi_r_rready),

      // axi_ram_s__2
      .s06_axi_awid('0),
      .s06_axi_awaddr('0),
      .s06_axi_awlen('0),
      .s06_axi_awsize('0),
      .s06_axi_awburst('0),
      .s06_axi_awlock('0),
      .s06_axi_awcache('0),
      .s06_axi_awprot('0),
      .s06_axi_awqos('0),
      .s06_axi_awuser('0),
      .s06_axi_awvalid('0),
      .s06_axi_awready(),
      .s06_axi_wdata('0),
      .s06_axi_wstrb('0),
      .s06_axi_wlast('0),
      .s06_axi_wuser('0),
      .s06_axi_wvalid('0),
      .s06_axi_wready(),
      .s06_axi_bid(),
      .s06_axi_bresp(),
      .s06_axi_buser(),
      .s06_axi_bvalid(),
      .s06_axi_bready('0),
      .s06_axi_arid(axi_ram_s__2_axi_ar_arid),
      .s06_axi_araddr(axi_ram_s__2_axi_ar_araddr),
      .s06_axi_arlen(axi_ram_s__2_axi_ar_arlen),
      .s06_axi_arsize(axi_ram_s__2_axi_ar_arsize),
      .s06_axi_arburst(axi_ram_s__2_axi_ar_arburst),
      .s06_axi_arlock('0),
      .s06_axi_arcache(axi_ram_s__2_axi_ar_arcache),
      .s06_axi_arprot(axi_ram_s__2_axi_ar_arprot),
      .s06_axi_arqos(axi_ram_s__2_axi_ar_arqos),
      .s06_axi_aruser('0),
      .s06_axi_arvalid(axi_ram_s__2_axi_ar_arvalid),
      .s06_axi_arready(axi_ram_s__2_axi_ar_arready),
      .s06_axi_rid(axi_ram_s__2_axi_r_rid),
      .s06_axi_rdata(axi_ram_s__2_axi_r_rdata),
      .s06_axi_rresp(axi_ram_s__2_axi_r_rresp[1:0]),
      .s06_axi_rlast(axi_ram_s__2_axi_r_rlast),
      .s06_axi_ruser(),
      .s06_axi_rvalid(axi_ram_s__2_axi_r_rvalid),
      .s06_axi_rready(axi_ram_s__2_axi_r_rready),

      // axi_ram_s__3
      .s07_axi_awid('0),
      .s07_axi_awaddr('0),
      .s07_axi_awlen('0),
      .s07_axi_awsize('0),
      .s07_axi_awburst('0),
      .s07_axi_awlock('0),
      .s07_axi_awcache('0),
      .s07_axi_awprot('0),
      .s07_axi_awqos('0),
      .s07_axi_awuser('0),
      .s07_axi_awvalid('0),
      .s07_axi_awready(),
      .s07_axi_wdata('0),
      .s07_axi_wstrb('0),
      .s07_axi_wlast('0),
      .s07_axi_wuser('0),
      .s07_axi_wvalid('0),
      .s07_axi_wready(),
      .s07_axi_bid(),
      .s07_axi_bresp(),
      .s07_axi_buser(),
      .s07_axi_bvalid(),
      .s07_axi_bready('0),
      .s07_axi_arid(axi_ram_s__3_axi_ar_arid),
      .s07_axi_araddr(axi_ram_s__3_axi_ar_araddr),
      .s07_axi_arlen(axi_ram_s__3_axi_ar_arlen),
      .s07_axi_arsize(axi_ram_s__3_axi_ar_arsize),
      .s07_axi_arburst(axi_ram_s__3_axi_ar_arburst),
      .s07_axi_arlock('0),
      .s07_axi_arcache(axi_ram_s__3_axi_ar_arcache),
      .s07_axi_arprot(axi_ram_s__3_axi_ar_arprot),
      .s07_axi_arqos(axi_ram_s__3_axi_ar_arqos),
      .s07_axi_aruser('0),
      .s07_axi_arvalid(axi_ram_s__3_axi_ar_arvalid),
      .s07_axi_arready(axi_ram_s__3_axi_ar_arready),
      .s07_axi_rid(axi_ram_s__3_axi_r_rid),
      .s07_axi_rdata(axi_ram_s__3_axi_r_rdata),
      .s07_axi_rresp(axi_ram_s__3_axi_r_rresp[1:0]),
      .s07_axi_rlast(axi_ram_s__3_axi_r_rlast),
      .s07_axi_ruser(),
      .s07_axi_rvalid(axi_ram_s__3_axi_r_rvalid),
      .s07_axi_rready(axi_ram_s__3_axi_r_rready),

      // axi_ram_s__4
      .s08_axi_awid('0),
      .s08_axi_awaddr('0),
      .s08_axi_awlen('0),
      .s08_axi_awsize('0),
      .s08_axi_awburst('0),
      .s08_axi_awlock('0),
      .s08_axi_awcache('0),
      .s08_axi_awprot('0),
      .s08_axi_awqos('0),
      .s08_axi_awuser('0),
      .s08_axi_awvalid('0),
      .s08_axi_awready(),
      .s08_axi_wdata('0),
      .s08_axi_wstrb('0),
      .s08_axi_wlast('0),
      .s08_axi_wuser('0),
      .s08_axi_wvalid('0),
      .s08_axi_wready(),
      .s08_axi_bid(),
      .s08_axi_bresp(),
      .s08_axi_buser(),
      .s08_axi_bvalid(),
      .s08_axi_bready('0),
      .s08_axi_arid(axi_ram_s__4_axi_ar_arid),
      .s08_axi_araddr(axi_ram_s__4_axi_ar_araddr),
      .s08_axi_arlen(axi_ram_s__4_axi_ar_arlen),
      .s08_axi_arsize(axi_ram_s__4_axi_ar_arsize),
      .s08_axi_arburst(axi_ram_s__4_axi_ar_arburst),
      .s08_axi_arlock('0),
      .s08_axi_arcache(axi_ram_s__4_axi_ar_arcache),
      .s08_axi_arprot(axi_ram_s__4_axi_ar_arprot),
      .s08_axi_arqos(axi_ram_s__4_axi_ar_arqos),
      .s08_axi_aruser('0),
      .s08_axi_arvalid(axi_ram_s__4_axi_ar_arvalid),
      .s08_axi_arready(axi_ram_s__4_axi_ar_arready),
      .s08_axi_rid(axi_ram_s__4_axi_r_rid),
      .s08_axi_rdata(axi_ram_s__4_axi_r_rdata),
      .s08_axi_rresp(axi_ram_s__4_axi_r_rresp[1:0]),
      .s08_axi_rlast(axi_ram_s__4_axi_r_rlast),
      .s08_axi_ruser(),
      .s08_axi_rvalid(axi_ram_s__4_axi_r_rvalid),
      .s08_axi_rready(axi_ram_s__4_axi_r_rready),

      // axi_ram_s__5
      .s09_axi_awid('0),
      .s09_axi_awaddr('0),
      .s09_axi_awlen('0),
      .s09_axi_awsize('0),
      .s09_axi_awburst('0),
      .s09_axi_awlock('0),
      .s09_axi_awcache('0),
      .s09_axi_awprot('0),
      .s09_axi_awqos('0),
      .s09_axi_awuser('0),
      .s09_axi_awvalid('0),
      .s09_axi_awready(),
      .s09_axi_wdata('0),
      .s09_axi_wstrb('0),
      .s09_axi_wlast('0),
      .s09_axi_wuser('0),
      .s09_axi_wvalid('0),
      .s09_axi_wready(),
      .s09_axi_bid(),
      .s09_axi_bresp(),
      .s09_axi_buser(),
      .s09_axi_bvalid(),
      .s09_axi_bready('0),
      .s09_axi_arid(axi_ram_s__5_axi_ar_arid),
      .s09_axi_araddr(axi_ram_s__5_axi_ar_araddr),
      .s09_axi_arlen(axi_ram_s__5_axi_ar_arlen),
      .s09_axi_arsize(axi_ram_s__5_axi_ar_arsize),
      .s09_axi_arburst(axi_ram_s__5_axi_ar_arburst),
      .s09_axi_arlock('0),
      .s09_axi_arcache(axi_ram_s__5_axi_ar_arcache),
      .s09_axi_arprot(axi_ram_s__5_axi_ar_arprot),
      .s09_axi_arqos(axi_ram_s__5_axi_ar_arqos),
      .s09_axi_aruser('0),
      .s09_axi_arvalid(axi_ram_s__5_axi_ar_arvalid),
      .s09_axi_arready(axi_ram_s__5_axi_ar_arready),
      .s09_axi_rid(axi_ram_s__5_axi_r_rid),
      .s09_axi_rdata(axi_ram_s__5_axi_r_rdata),
      .s09_axi_rresp(axi_ram_s__5_axi_r_rresp[1:0]),
      .s09_axi_rlast(axi_ram_s__5_axi_r_rlast),
      .s09_axi_ruser(),
      .s09_axi_rvalid(axi_ram_s__5_axi_r_rvalid),
      .s09_axi_rready(axi_ram_s__5_axi_r_rready),

      // axi_ram_s__6
      .s10_axi_awid('0),
      .s10_axi_awaddr('0),
      .s10_axi_awlen('0),
      .s10_axi_awsize('0),
      .s10_axi_awburst('0),
      .s10_axi_awlock('0),
      .s10_axi_awcache('0),
      .s10_axi_awprot('0),
      .s10_axi_awqos('0),
      .s10_axi_awuser('0),
      .s10_axi_awvalid('0),
      .s10_axi_awready(),
      .s10_axi_wdata('0),
      .s10_axi_wstrb('0),
      .s10_axi_wlast('0),
      .s10_axi_wuser('0),
      .s10_axi_wvalid('0),
      .s10_axi_wready(),
      .s10_axi_bid(),
      .s10_axi_bresp(),
      .s10_axi_buser(),
      .s10_axi_bvalid(),
      .s10_axi_bready('0),
      .s10_axi_arid(axi_ram_s__6_axi_ar_arid),
      .s10_axi_araddr(axi_ram_s__6_axi_ar_araddr),
      .s10_axi_arlen(axi_ram_s__6_axi_ar_arlen),
      .s10_axi_arsize(axi_ram_s__6_axi_ar_arsize),
      .s10_axi_arburst(axi_ram_s__6_axi_ar_arburst),
      .s10_axi_arlock('0),
      .s10_axi_arcache(axi_ram_s__6_axi_ar_arcache),
      .s10_axi_arprot(axi_ram_s__6_axi_ar_arprot),
      .s10_axi_arqos(axi_ram_s__6_axi_ar_arqos),
      .s10_axi_aruser('0),
      .s10_axi_arvalid(axi_ram_s__6_axi_ar_arvalid),
      .s10_axi_arready(axi_ram_s__6_axi_ar_arready),
      .s10_axi_rid(axi_ram_s__6_axi_r_rid),
      .s10_axi_rdata(axi_ram_s__6_axi_r_rdata),
      .s10_axi_rresp(axi_ram_s__6_axi_r_rresp[1:0]),
      .s10_axi_rlast(axi_ram_s__6_axi_r_rlast),
      .s10_axi_ruser(),
      .s10_axi_rvalid(axi_ram_s__6_axi_r_rvalid),
      .s10_axi_rready(axi_ram_s__6_axi_r_rready),

      // axi_ram_s__7
      .s11_axi_awid('0),
      .s11_axi_awaddr('0),
      .s11_axi_awlen('0),
      .s11_axi_awsize('0),
      .s11_axi_awburst('0),
      .s11_axi_awlock('0),
      .s11_axi_awcache('0),
      .s11_axi_awprot('0),
      .s11_axi_awqos('0),
      .s11_axi_awuser('0),
      .s11_axi_awvalid('0),
      .s11_axi_awready(),
      .s11_axi_wdata('0),
      .s11_axi_wstrb('0),
      .s11_axi_wlast('0),
      .s11_axi_wuser('0),
      .s11_axi_wvalid('0),
      .s11_axi_wready(),
      .s11_axi_bid(),
      .s11_axi_bresp(),
      .s11_axi_buser(),
      .s11_axi_bvalid(),
      .s11_axi_bready('0),
      .s11_axi_arid(axi_ram_s__7_axi_ar_arid),
      .s11_axi_araddr(axi_ram_s__7_axi_ar_araddr),
      .s11_axi_arlen(axi_ram_s__7_axi_ar_arlen),
      .s11_axi_arsize(axi_ram_s__7_axi_ar_arsize),
      .s11_axi_arburst(axi_ram_s__7_axi_ar_arburst),
      .s11_axi_arlock('0),
      .s11_axi_arcache(axi_ram_s__7_axi_ar_arcache),
      .s11_axi_arprot(axi_ram_s__7_axi_ar_arprot),
      .s11_axi_arqos(axi_ram_s__7_axi_ar_arqos),
      .s11_axi_aruser('0),
      .s11_axi_arvalid(axi_ram_s__7_axi_ar_arvalid),
      .s11_axi_arready(axi_ram_s__7_axi_ar_arready),
      .s11_axi_rid(axi_ram_s__7_axi_r_rid),
      .s11_axi_rdata(axi_ram_s__7_axi_r_rdata),
      .s11_axi_rresp(axi_ram_s__7_axi_r_rresp[1:0]),
      .s11_axi_rlast(axi_ram_s__7_axi_r_rlast),
      .s11_axi_ruser(),
      .s11_axi_rvalid(axi_ram_s__7_axi_r_rvalid),
      .s11_axi_rready(axi_ram_s__7_axi_r_rready),

      // axi_ram_s__8
      .s12_axi_awid('0),
      .s12_axi_awaddr('0),
      .s12_axi_awlen('0),
      .s12_axi_awsize('0),
      .s12_axi_awburst('0),
      .s12_axi_awlock('0),
      .s12_axi_awcache('0),
      .s12_axi_awprot('0),
      .s12_axi_awqos('0),
      .s12_axi_awuser('0),
      .s12_axi_awvalid('0),
      .s12_axi_awready(),
      .s12_axi_wdata('0),
      .s12_axi_wstrb('0),
      .s12_axi_wlast('0),
      .s12_axi_wuser('0),
      .s12_axi_wvalid('0),
      .s12_axi_wready(),
      .s12_axi_bid(),
      .s12_axi_bresp(),
      .s12_axi_buser(),
      .s12_axi_bvalid(),
      .s12_axi_bready('0),
      .s12_axi_arid(axi_ram_s__8_axi_ar_arid),
      .s12_axi_araddr(axi_ram_s__8_axi_ar_araddr),
      .s12_axi_arlen(axi_ram_s__8_axi_ar_arlen),
      .s12_axi_arsize(axi_ram_s__8_axi_ar_arsize),
      .s12_axi_arburst(axi_ram_s__8_axi_ar_arburst),
      .s12_axi_arlock('0),
      .s12_axi_arcache(axi_ram_s__8_axi_ar_arcache),
      .s12_axi_arprot(axi_ram_s__8_axi_ar_arprot),
      .s12_axi_arqos(axi_ram_s__8_axi_ar_arqos),
      .s12_axi_aruser('0),
      .s12_axi_arvalid(axi_ram_s__8_axi_ar_arvalid),
      .s12_axi_arready(axi_ram_s__8_axi_ar_arready),
      .s12_axi_rid(axi_ram_s__8_axi_r_rid),
      .s12_axi_rdata(axi_ram_s__8_axi_r_rdata),
      .s12_axi_rresp(axi_ram_s__8_axi_r_rresp[1:0]),
      .s12_axi_rlast(axi_ram_s__8_axi_r_rlast),
      .s12_axi_ruser(),
      .s12_axi_rvalid(axi_ram_s__8_axi_r_rvalid),
      .s12_axi_rready(axi_ram_s__8_axi_r_rready),

      // axi_ram_s__9
      .s13_axi_awid('0),
      .s13_axi_awaddr('0),
      .s13_axi_awlen('0),
      .s13_axi_awsize('0),
      .s13_axi_awburst('0),
      .s13_axi_awlock('0),
      .s13_axi_awcache('0),
      .s13_axi_awprot('0),
      .s13_axi_awqos('0),
      .s13_axi_awuser('0),
      .s13_axi_awvalid('0),
      .s13_axi_awready(),
      .s13_axi_wdata('0),
      .s13_axi_wstrb('0),
      .s13_axi_wlast('0),
      .s13_axi_wuser('0),
      .s13_axi_wvalid('0),
      .s13_axi_wready(),
      .s13_axi_bid(),
      .s13_axi_bresp(),
      .s13_axi_buser(),
      .s13_axi_bvalid(),
      .s13_axi_bready('0),
      .s13_axi_arid(axi_ram_s__9_axi_ar_arid),
      .s13_axi_araddr(axi_ram_s__9_axi_ar_araddr),
      .s13_axi_arlen(axi_ram_s__9_axi_ar_arlen),
      .s13_axi_arsize(axi_ram_s__9_axi_ar_arsize),
      .s13_axi_arburst(axi_ram_s__9_axi_ar_arburst),
      .s13_axi_arlock('0),
      .s13_axi_arcache(axi_ram_s__9_axi_ar_arcache),
      .s13_axi_arprot(axi_ram_s__9_axi_ar_arprot),
      .s13_axi_arqos(axi_ram_s__9_axi_ar_arqos),
      .s13_axi_aruser('0),
      .s13_axi_arvalid(axi_ram_s__9_axi_ar_arvalid),
      .s13_axi_arready(axi_ram_s__9_axi_ar_arready),
      .s13_axi_rid(axi_ram_s__9_axi_r_rid),
      .s13_axi_rdata(axi_ram_s__9_axi_r_rdata),
      .s13_axi_rresp(axi_ram_s__9_axi_r_rresp[1:0]),
      .s13_axi_rlast(axi_ram_s__9_axi_r_rlast),
      .s13_axi_ruser(),
      .s13_axi_rvalid(axi_ram_s__9_axi_r_rvalid),
      .s13_axi_rready(axi_ram_s__9_axi_r_rready),

      // axi_ram_s__10
      .s14_axi_awid('0),
      .s14_axi_awaddr('0),
      .s14_axi_awlen('0),
      .s14_axi_awsize('0),
      .s14_axi_awburst('0),
      .s14_axi_awlock('0),
      .s14_axi_awcache('0),
      .s14_axi_awprot('0),
      .s14_axi_awqos('0),
      .s14_axi_awuser('0),
      .s14_axi_awvalid('0),
      .s14_axi_awready(),
      .s14_axi_wdata('0),
      .s14_axi_wstrb('0),
      .s14_axi_wlast('0),
      .s14_axi_wuser('0),
      .s14_axi_wvalid('0),
      .s14_axi_wready(),
      .s14_axi_bid(),
      .s14_axi_bresp(),
      .s14_axi_buser(),
      .s14_axi_bvalid(),
      .s14_axi_bready('0),
      .s14_axi_arid(axi_ram_s__10_axi_ar_arid),
      .s14_axi_araddr(axi_ram_s__10_axi_ar_araddr),
      .s14_axi_arlen(axi_ram_s__10_axi_ar_arlen),
      .s14_axi_arsize(axi_ram_s__10_axi_ar_arsize),
      .s14_axi_arburst(axi_ram_s__10_axi_ar_arburst),
      .s14_axi_arlock('0),
      .s14_axi_arcache(axi_ram_s__10_axi_ar_arcache),
      .s14_axi_arprot(axi_ram_s__10_axi_ar_arprot),
      .s14_axi_arqos(axi_ram_s__10_axi_ar_arqos),
      .s14_axi_aruser('0),
      .s14_axi_arvalid(axi_ram_s__10_axi_ar_arvalid),
      .s14_axi_arready(axi_ram_s__10_axi_ar_arready),
      .s14_axi_rid(axi_ram_s__10_axi_r_rid),
      .s14_axi_rdata(axi_ram_s__10_axi_r_rdata),
      .s14_axi_rresp(axi_ram_s__10_axi_r_rresp[1:0]),
      .s14_axi_rlast(axi_ram_s__10_axi_r_rlast),
      .s14_axi_ruser(),
      .s14_axi_rvalid(axi_ram_s__10_axi_r_rvalid),
      .s14_axi_rready(axi_ram_s__10_axi_r_rready),

      /*
       * AXI Manager interface
       */
      // Outside-facing AXI interface of the ZSTD Decoder
      .m00_axi_awid(memory_axi_aw_awid),
      .m00_axi_awaddr(memory_axi_aw_awaddr),
      .m00_axi_awlen(memory_axi_aw_awlen),
      .m00_axi_awsize(memory_axi_aw_awsize),
      .m00_axi_awburst(memory_axi_aw_awburst),
      .m00_axi_awlock(memory_axi_aw_awlock),
      .m00_axi_awcache(memory_axi_aw_awcache),
      .m00_axi_awprot(memory_axi_aw_awprot),
      .m00_axi_awqos(memory_axi_aw_awqos),
      .m00_axi_awregion(memory_axi_aw_awregion),
      .m00_axi_awuser(memory_axi_aw_awuser),
      .m00_axi_awvalid(memory_axi_aw_awvalid),
      .m00_axi_awready(memory_axi_aw_awready),
      .m00_axi_wdata(memory_axi_w_wdata),
      .m00_axi_wstrb(memory_axi_w_wstrb),
      .m00_axi_wlast(memory_axi_w_wlast),
      .m00_axi_wuser(memory_axi_w_wuser),
      .m00_axi_wvalid(memory_axi_w_wvalid),
      .m00_axi_wready(memory_axi_w_wready),
      .m00_axi_bid(memory_axi_b_bid),
      .m00_axi_bresp(memory_axi_b_bresp[1:0]),
      .m00_axi_buser(memory_axi_b_buser),
      .m00_axi_bvalid(memory_axi_b_bvalid),
      .m00_axi_bready(memory_axi_b_bready),
      .m00_axi_arid(memory_axi_ar_arid),
      .m00_axi_araddr(memory_axi_ar_araddr),
      .m00_axi_arlen(memory_axi_ar_arlen),
      .m00_axi_arsize(memory_axi_ar_arsize),
      .m00_axi_arburst(memory_axi_ar_arburst),
      .m00_axi_arlock(memory_axi_ar_arlock),
      .m00_axi_arcache(memory_axi_ar_arcache),
      .m00_axi_arprot(memory_axi_ar_arprot),
      .m00_axi_arqos(memory_axi_ar_arqos),
      .m00_axi_arregion(memory_axi_ar_arregion),
      .m00_axi_aruser(memory_axi_ar_aruser),
      .m00_axi_arvalid(memory_axi_ar_arvalid),
      .m00_axi_arready(memory_axi_ar_arready),
      .m00_axi_rid(memory_axi_r_rid),
      .m00_axi_rdata(memory_axi_r_rdata),
      .m00_axi_rresp(memory_axi_r_rresp[1:0]),
      .m00_axi_rlast(memory_axi_r_rlast),
      .m00_axi_ruser(memory_axi_r_ruser),
      .m00_axi_rvalid(memory_axi_r_rvalid),
      .m00_axi_rready(memory_axi_r_rready)
  );

endmodule : zstd_dec_wrapper
