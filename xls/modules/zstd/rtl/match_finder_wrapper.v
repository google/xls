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

module match_finder_wrapper #(
    parameter AXI_DATA_W  = 64,
    parameter AXI_ADDR_W  = 32,
    parameter S_AXI_ID_W  = 4,
    parameter M_AXI_ID_W  = 6,
    parameter AXI_STRB_W  = 8,
    parameter AWUSER_WIDTH = 1,
    parameter WUSER_WIDTH = 1,
    parameter BUSER_WIDTH = 1,
    parameter ARUSER_WIDTH = 1,
    parameter RUSER_WIDTH = 1
) (
    input wire clk,
    input wire rst,
    // request
    input wire [137:0] req_r_data,
    input wire  req_r_vld,
    output wire req_r_rdy,

    // response
    output wire [137:0] resp_s_data,
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

    wire [XLS_AXI_AW_W-1:0] match_finder__axi_aw;
    wire                    match_finder__axi_aw_rdy;
    wire                    match_finder__axi_aw_vld;
    wire [XLS_AXI_W_W-1:0]  match_finder__axi_w;
    wire                    match_finder__axi_w_rdy;
    wire                    match_finder__axi_w_vld;
    wire [ XLS_AXI_B_W-1:0] match_finder__axi_b;
    wire                    match_finder__axi_b_rdy;
    wire                    match_finder__axi_b_vld;
    wire [XLS_AXI_AR_W-1:0] match_finder__axi_ar;
    wire                    match_finder__axi_ar_rdy;
    wire                    match_finder__axi_ar_vld;
    wire [XLS_AXI_R_W-1:0]  match_finder__axi_r;
    wire                    match_finder__axi_r_rdy;
    wire                    match_finder__axi_r_vld;

    assign  {
        memory_axi_aw_awid,
        memory_axi_aw_awaddr,
        memory_axi_aw_awsize,
        memory_axi_aw_awlen,
        memory_axi_aw_awburst
    } = match_finder__axi_aw;
    assign  memory_axi_aw_awvalid = match_finder__axi_aw_vld;
    assign match_finder__axi_aw_rdy = memory_axi_aw_awready;
    assign {
        memory_axi_w_wdata,
        memory_axi_w_wstrb,
        memory_axi_w_wlast
        } = match_finder__axi_w;
    assign memory_axi_w_wvalid = match_finder__axi_w_vld;
    assign match_finder__axi_w_rdy = memory_axi_w_wready;
    assign match_finder__axi_b = {
        memory_axi_b_bresp,
        memory_axi_b_bid
        };
    assign match_finder__axi_b_vld = memory_axi_b_bvalid;
    assign memory_axi_b_bready = match_finder__axi_b_rdy;
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
    } = match_finder__axi_ar;
    assign memory_axi_ar_arvalid = match_finder__axi_ar_vld;
    assign match_finder__axi_ar_rdy = memory_axi_ar_arready;
    assign match_finder__axi_r = {
        memory_axi_r_rid,
        memory_axi_r_rdata,
        memory_axi_r_rresp,
        memory_axi_r_rlast
    };
    assign match_finder__axi_r_vld = memory_axi_r_rvalid;
    assign memory_axi_r_rready = match_finder__axi_r_rdy;

    // History Buffer

    localparam HB_SIZE = 1024;
    localparam HB_DATA_W = 64;
    localparam HB_ADDR_W = $clog2(HB_SIZE);
    localparam HB_NUM_PARTITIONS = 8;

    wire [HB_ADDR_W-1:0]         hb_ram_rd_addr [0:7];
    wire [HB_DATA_W-1:0]         hb_ram_rd_data [0:7];
    wire                         hb_ram_rd_en   [0:7];
    wire [HB_NUM_PARTITIONS-1:0] hb_ram_rd_mask [0:7];

    wire [HB_ADDR_W-1:0]         hb_ram_wr_addr [0:7];
    wire [HB_DATA_W-1:0]         hb_ram_wr_data [0:7];
    wire                         hb_ram_wr_en   [0:7];
    wire [HB_NUM_PARTITIONS-1:0] hb_ram_wr_mask [0:7];

    genvar i;
    generate
        for (i = 0; i < 8; i = i + 1) begin : hb_loop
            ram_1r1w #(
                .DATA_WIDTH(HB_DATA_W),
                .ADDR_WIDTH(HB_ADDR_W),
                .SIZE(HB_SIZE),
                .NUM_PARTITIONS(HB_NUM_PARTITIONS)
            ) hb_ram (
                .clk(clk),
                .rst(rst),
                .rd_addr (hb_ram_rd_addr[i]),
                .rd_data (hb_ram_rd_data[i]),
                .rd_en   (hb_ram_rd_en[i]),
                .rd_mask (hb_ram_rd_mask[i]),
                .wr_data (hb_ram_wr_data[i]),
                .wr_addr (hb_ram_wr_addr[i]),
                .wr_en   (hb_ram_wr_en[i]),
                .wr_mask (hb_ram_wr_mask[i])
            );
        end
    endgenerate

    // Hash Table
    localparam HT_KEY_W = 64;
    localparam HT_SIZE = 512;
    localparam HT_ADDR_W = $clog2(HT_SIZE);
    localparam HT_ENTRY_W = HT_KEY_W + 32;
    localparam HT_DATA_W = HT_ENTRY_W + 1;  // 64 (KEY_WIDTH) + 32 (ADDR_W) + 1 (match or not);
    localparam HT_NUM_PARTITIONS = 1;

    wire [HT_DATA_W-1:0]         ht_ram_rd_data;
    wire [HT_ADDR_W-1:0]         ht_ram_rd_addr;
    wire                         ht_ram_rd_en;
    wire [HT_DATA_W-1:0]         ht_ram_rd_mask;

    wire [HT_DATA_W-1:0]         ht_ram_wr_data;
    wire [HT_ADDR_W-1:0]         ht_ram_wr_addr;
    wire                         ht_ram_wr_en;
    wire [HT_DATA_W-1:0]         ht_ram_wr_mask;

    ram_1r1w #(
        .DATA_WIDTH(HT_DATA_W),
        .ADDR_WIDTH(HT_ADDR_W),
        .SIZE(HT_SIZE),
        .NUM_PARTITIONS(HT_NUM_PARTITIONS)
    ) ht_ram (
        .clk(clk),
        .rst(rst),
        .rd_addr (ht_ram_rd_addr),
        .rd_data (ht_ram_rd_data),
        .rd_en   (ht_ram_rd_en),
        .rd_mask (ht_ram_rd_mask),
        .wr_addr (ht_ram_wr_addr),
        .wr_data (ht_ram_wr_data),
        .wr_en   (ht_ram_wr_en),
        .wr_mask (ht_ram_wr_mask)
    );

    MatchFinder match_finder_benchmark (
        .clk(clk),
        .rst(rst),
        // reader
        .match_finder__axi_ar_s(match_finder__axi_ar),
        .match_finder__axi_ar_s_vld(match_finder__axi_ar_vld),
        .match_finder__axi_ar_s_rdy(match_finder__axi_ar_rdy),
        .match_finder__axi_r_r(match_finder__axi_r),
        .match_finder__axi_r_r_vld(match_finder__axi_r_vld),
        .match_finder__axi_r_r_rdy(match_finder__axi_r_rdy),

        // writer
        .match_finder__axi_b_r(match_finder__axi_b),
        .match_finder__axi_b_r_vld(match_finder__axi_b_vld),
        .match_finder__axi_b_r_rdy(match_finder__axi_b_rdy),
        .match_finder__axi_aw_s(match_finder__axi_aw),
        .match_finder__axi_aw_s_rdy(match_finder__axi_aw_rdy),
        .match_finder__axi_aw_s_vld(match_finder__axi_aw_vld),
        .match_finder__axi_w_s(match_finder__axi_w),
        .match_finder__axi_w_s_rdy(match_finder__axi_w_rdy),
        .match_finder__axi_w_s_vld(match_finder__axi_w_vld),

        .match_finder__resp_s(resp_s_data),
        .match_finder__resp_s_rdy(resp_s_rdy),
        .match_finder__resp_s_vld(resp_s_vld),
        .match_finder__req_r(req_r_data),
        .match_finder__req_r_rdy(req_r_rdy),
        .match_finder__req_r_vld(req_r_vld),

        .hb_ram0_rd_addr(hb_ram_rd_addr[0]),
        .hb_ram0_rd_data(hb_ram_rd_data[0]),
        .hb_ram0_rd_en  (hb_ram_rd_en[0]),
        .hb_ram0_rd_mask(hb_ram_rd_mask[0]),
        .hb_ram0_wr_addr(hb_ram_wr_addr[0]),
        .hb_ram0_wr_data(hb_ram_wr_data[0]),
        .hb_ram0_wr_en  (hb_ram_wr_en[0]),
        .hb_ram0_wr_mask(hb_ram_wr_mask[0]),

        .hb_ram1_rd_addr(hb_ram_rd_addr[1]),
        .hb_ram1_rd_data(hb_ram_rd_data[1]),
        .hb_ram1_rd_en  (hb_ram_rd_en[1]),
        .hb_ram1_rd_mask(hb_ram_rd_mask[1]),
        .hb_ram1_wr_addr(hb_ram_wr_addr[1]),
        .hb_ram1_wr_data(hb_ram_wr_data[1]),
        .hb_ram1_wr_en  (hb_ram_wr_en[1]),
        .hb_ram1_wr_mask(hb_ram_wr_mask[1]),

        .hb_ram2_rd_addr(hb_ram_rd_addr[2]),
        .hb_ram2_rd_data(hb_ram_rd_data[2]),
        .hb_ram2_rd_en  (hb_ram_rd_en[2]),
        .hb_ram2_rd_mask(hb_ram_rd_mask[2]),
        .hb_ram2_wr_addr(hb_ram_wr_addr[2]),
        .hb_ram2_wr_data(hb_ram_wr_data[2]),
        .hb_ram2_wr_en  (hb_ram_wr_en[2]),
        .hb_ram2_wr_mask(hb_ram_wr_mask[2]),

        .hb_ram3_rd_addr(hb_ram_rd_addr[3]),
        .hb_ram3_rd_data(hb_ram_rd_data[3]),
        .hb_ram3_rd_en  (hb_ram_rd_en[3]),
        .hb_ram3_rd_mask(hb_ram_rd_mask[3]),
        .hb_ram3_wr_addr(hb_ram_wr_addr[3]),
        .hb_ram3_wr_data(hb_ram_wr_data[3]),
        .hb_ram3_wr_en  (hb_ram_wr_en[3]),
        .hb_ram3_wr_mask(hb_ram_wr_mask[3]),

        .hb_ram4_rd_addr(hb_ram_rd_addr[4]),
        .hb_ram4_rd_data(hb_ram_rd_data[4]),
        .hb_ram4_rd_en  (hb_ram_rd_en[4]),
        .hb_ram4_rd_mask(hb_ram_rd_mask[4]),
        .hb_ram4_wr_addr(hb_ram_wr_addr[4]),
        .hb_ram4_wr_data(hb_ram_wr_data[4]),
        .hb_ram4_wr_en  (hb_ram_wr_en[4]),
        .hb_ram4_wr_mask(hb_ram_wr_mask[4]),

        .hb_ram5_rd_addr(hb_ram_rd_addr[5]),
        .hb_ram5_rd_data(hb_ram_rd_data[5]),
        .hb_ram5_rd_en  (hb_ram_rd_en[5]),
        .hb_ram5_rd_mask(hb_ram_rd_mask[5]),
        .hb_ram5_wr_addr(hb_ram_wr_addr[5]),
        .hb_ram5_wr_data(hb_ram_wr_data[5]),
        .hb_ram5_wr_en  (hb_ram_wr_en[5]),
        .hb_ram5_wr_mask(hb_ram_wr_mask[5]),

        .hb_ram6_rd_addr(hb_ram_rd_addr[6]),
        .hb_ram6_rd_data(hb_ram_rd_data[6]),
        .hb_ram6_rd_en  (hb_ram_rd_en[6]),
        .hb_ram6_rd_mask(hb_ram_rd_mask[6]),
        .hb_ram6_wr_addr(hb_ram_wr_addr[6]),
        .hb_ram6_wr_data(hb_ram_wr_data[6]),
        .hb_ram6_wr_en  (hb_ram_wr_en[6]),
        .hb_ram6_wr_mask(hb_ram_wr_mask[6]),

        .hb_ram7_rd_addr(hb_ram_rd_addr[7]),
        .hb_ram7_rd_data(hb_ram_rd_data[7]),
        .hb_ram7_rd_en  (hb_ram_rd_en[7]),
        .hb_ram7_rd_mask(hb_ram_rd_mask[7]),
        .hb_ram7_wr_addr(hb_ram_wr_addr[7]),
        .hb_ram7_wr_data(hb_ram_wr_data[7]),
        .hb_ram7_wr_en  (hb_ram_wr_en[7]),
        .hb_ram7_wr_mask(hb_ram_wr_mask[7]),

        .ht_ram_rd_addr(ht_ram_rd_addr),
        .ht_ram_rd_data(ht_ram_rd_data),
        .ht_ram_rd_en  (ht_ram_rd_en),
        .ht_ram_rd_mask(ht_ram_rd_mask),
        .ht_ram_wr_addr(ht_ram_wr_addr),
        .ht_ram_wr_data(ht_ram_wr_data),
        .ht_ram_wr_en  (ht_ram_wr_en),
        .ht_ram_wr_mask(ht_ram_wr_mask)
    );

endmodule
