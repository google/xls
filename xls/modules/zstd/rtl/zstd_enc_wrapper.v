// Copyright 2025 The XLS Authors
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

`default_nettype none

module zstd_enc_wrapper #(
    parameter AXI_DATA_W  = 32,
    parameter AXI_ADDR_W  = 32,
    parameter S_AXI_ID_W  = 4,
    parameter M_AXI_ID_W  = 6,
    parameter AXI_STRB_W  = 4,
    parameter AWUSER_WIDTH = 1,
    parameter WUSER_WIDTH = 1,
    parameter BUSER_WIDTH = 1,
    parameter ARUSER_WIDTH = 1,
    parameter RUSER_WIDTH = 1
) (
    input wire clk,
    input wire rst,
    // request
    input wire [127:0] req_r_data,
    input wire  req_r_vld,
    output wire req_r_rdy,

    // response
    output wire resp_s_data,
    output wire resp_s_rdy,
    input wire resp_s_vld,

    // memory
    output wire [M_AXI_ID_W-1:0]    memory_axi_aw_awid,
    output wire [AXI_ADDR_W-1:0]    memory_axi_aw_awaddr,
    output wire [7:0]               memory_axi_aw_awlen,
    output wire [2:0]               memory_axi_aw_awsize,
    output wire [1:0]               memory_axi_aw_awburst,
    output wire                     memory_axi_aw_awlock,
    output wire [3:0]               memory_axi_aw_awcache,
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
    output wire                     memory_axi_r_rready
);

    localparam XLS_AXI_AW_SIZE_W = 3;
    localparam XLS_AXI_AW_BURST_W = 2;
    localparam XLS_AXI_AW_W = AXI_ADDR_W + S_AXI_ID_W + XLS_AXI_AW_SIZE_W + XLS_AXI_AW_BURST_W;

    localparam XLS_AXI_W_LAST_W = 1;
    localparam XLS_AXI_W_W = AXI_DATA_W + AXI_STRB_W + XLS_AXI_W_LAST_W;

    localparam XLS_AXI_B_RESP_W = 3;
    localparam XLS_AXI_B_W = XLS_AXI_B_RESP_W + S_AXI_ID_W;

    localparam XLS_AXI_AR_PROT_W = 3;
    localparam XLS_AXI_AR_SIZE_W = 3;
    localparam XLS_AXI_AR_CACHE_W = 4;
    localparam XLS_AXI_AR_QOS_W = 4;
    localparam XLS_AXI_AR_LEN_W = 8;
    localparam XLS_AXI_AR_BURST_W = 2;
    localparam XLS_AXI_AR_REGION_W = 4;
    localparam XLS_AXI_AR_W = S_AXI_ID_W + AXI_ADDR_W + XLS_AXI_AR_CACHE_W + XLS_AXI_AR_LEN_W + XLS_AXI_AR_PROT_W + XLS_AXI_AR_BURST_W + XLS_AXI_AR_QOS_W + XLS_AXI_AR_SIZE_W + XLS_AXI_AR_REGION_W;

    localparam XLS_AXI_R_RESP_W = 3;
    localparam XLS_AXI_R_LAST_W = 1;
    localparam XLS_AXI_R_W = S_AXI_ID_W + AXI_DATA_W + XLS_AXI_R_RESP_W + XLS_AXI_R_LAST_W;

    wire [XLS_AXI_AW_W-1:0] zstd_enc__axi_aw;
    wire                    zstd_enc__axi_aw_rdy;
    wire                    zstd_enc__axi_aw_vld;
    wire [XLS_AXI_W_W-1:0]  zstd_enc__axi_w;
    wire                    zstd_enc__axi_w_rdy;
    wire                    zstd_enc__axi_w_vld;
    wire [ XLS_AXI_B_W-1:0] zstd_enc__axi_b;
    wire                    zstd_enc__axi_b_rdy;
    wire                    zstd_enc__axi_b_vld;
    wire [XLS_AXI_AR_W-1:0] zstd_enc__axi_ar;
    wire                    zstd_enc__axi_ar_rdy;
    wire                    zstd_enc__axi_ar_vld;
    wire [ XLS_AXI_R_W-1:0] zstd_enc__axi_r;
    wire                    zstd_enc__axi_r_rdy;
    wire                    zstd_enc__axi_r_vld;

    assign  {
        memory_axi_aw_awid,
        memory_axi_aw_awaddr,
        memory_axi_aw_awsize,
        memory_axi_aw_awlen,
        memory_axi_aw_awburst
    } = zstd_enc__axi_aw;
    assign  memory_axi_aw_awvalid = zstd_enc__axi_aw_vld;
    assign zstd_enc__axi_aw_rdy = memory_axi_aw_awready;
    assign {
        memory_axi_w_wdata,
        memory_axi_w_wstrb,
        memory_axi_w_wlast
        } = zstd_enc__axi_w;
    assign memory_axi_w_wvalid = zstd_enc__axi_w_vld;
    assign zstd_enc__axi_w_rdy = memory_axi_w_wready;
    assign zstd_enc__axi_b = {
        memory_axi_b_bresp,
        memory_axi_b_bid
        };
    assign zstd_enc__axi_b_vld = memory_axi_b_bvalid;
    assign memory_axi_b_bready = zstd_enc__axi_b_rdy;
    assign {
        memory_axi_ar_arid,
        memory_axi_ar_araddr,
        memory_axi_ar_arregion,
        memory_axi_ar_arlen,
        memory_axi_ar_arsize,
        memory_axi_ar_arburst,
        memory_axi_ar_arcache,
        memory_axi_ar_arprot,
        memory_axi_ar_arqos
    } = zstd_enc__axi_ar;
    assign memory_axi_ar_arvalid = zstd_enc__axi_ar_vld;
    assign zstd_enc__axi_ar_rdy = memory_axi_ar_arready;
    assign zstd_enc__axi_r = {
        memory_axi_r_rid,
        memory_axi_r_rdata,
        memory_axi_r_rresp,
        memory_axi_r_rlast
    };
    assign memory_axi_b_buser = 1'b0;
    assign memory_axi_r_ruser = 1'b0;

    assign zstd_enc__axi_r_vld = memory_axi_r_rvalid;
    assign memory_axi_r_rready = zstd_enc__axi_r_rdy;

    ZstdEncoder zstd_encoder_benchmark (
        .clk(clk),
        .rst(rst),

        // reader
        .zstd_enc__axi_ar_s(zstd_enc__axi_ar),
        .zstd_enc__axi_ar_s_vld(zstd_enc__axi_ar_vld),
        .zstd_enc__axi_ar_s_rdy(zstd_enc__axi_ar_rdy),
        .zstd_enc__axi_r_r(zstd_enc__axi_r),
        .zstd_enc__axi_r_r_vld(zstd_enc__axi_r_vld),
        .zstd_enc__axi_r_r_rdy(zstd_enc__axi_r_rdy),

        // writer
        .zstd_enc__axi_b_r(zstd_enc__axi_b),
        .zstd_enc__axi_b_r_vld(zstd_enc__axi_b_vld),
        .zstd_enc__axi_b_r_rdy(zstd_enc__axi_b_rdy),
        .zstd_enc__axi_aw_s(zstd_enc__axi_aw),
        .zstd_enc__axi_aw_s_rdy(zstd_enc__axi_aw_rdy),
        .zstd_enc__axi_aw_s_vld(zstd_enc__axi_aw_vld),
        .zstd_enc__axi_w_s(zstd_enc__axi_w),
        .zstd_enc__axi_w_s_rdy(zstd_enc__axi_w_rdy),
        .zstd_enc__axi_w_s_vld(zstd_enc__axi_w_vld),

        .zstd_enc__enc_resp_s(resp_s_data),
        .zstd_enc__enc_resp_s_rdy(resp_s_rdy),
        .zstd_enc__enc_resp_s_vld(resp_s_vld),
        .zstd_enc__enc_req_r(req_r_data),
        .zstd_enc__enc_req_r_rdy(req_r_rdy),
        .zstd_enc__enc_req_r_vld(req_r_vld)
    );

endmodule
