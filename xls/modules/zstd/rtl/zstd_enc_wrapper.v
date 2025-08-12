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
    parameter RUSER_WIDTH = 1,
    parameter CONF_W = 2,
    parameter REQ_W = AXI_DATA_W + AXI_ADDR_W + AXI_ADDR_W + AXI_DATA_W + CONF_W
) (
    input wire clk,
    input wire rst,
    // request
    input wire [REQ_W-1:0] req_r_data,
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

    localparam HB_SIZE = 1024;
    localparam HB_RAM_NUM = 8;
    localparam HB_DATA_W = 8;
    localparam HB_RAM_SIZE = HB_SIZE / HB_RAM_NUM;
    localparam HB_ADDR_W = $clog2(HB_RAM_SIZE);
    localparam HB_NUM_PARTITIONS = 1;
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

    localparam HT_KEY_W = 32;
    localparam HT_SIZE = 512;
    localparam HT_ADDR_W = $clog2(HT_SIZE);
    localparam HT_ENTRY_W = HT_KEY_W + 32;
    localparam HT_DATA_W = HT_ENTRY_W + 1;  // 32 (KEY_WIDTH) + 32 (ADDR_W) + 1 (match or not);
    localparam HT_NUM_PARTITIONS = HT_DATA_W;
    wire [HT_DATA_W-1:0]         ht_ram_rd_data;
    wire [HT_ADDR_W-1:0]         ht_ram_rd_addr;
    wire                         ht_ram_rd_en;
    wire [HT_NUM_PARTITIONS-1:0]         ht_ram_rd_mask;
    wire [HT_DATA_W-1:0]         ht_ram_wr_data;
    wire [HT_ADDR_W-1:0]         ht_ram_wr_addr;
    wire                         ht_ram_wr_en;
    wire [HT_NUM_PARTITIONS-1:0] ht_ram_wr_mask;

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

    // fse table buffers

    localparam FSE_TABLE_RAM_ADDR_W = 32;
    localparam FSE_CTABLE_RAM_DATA_W = 16;
    localparam FSE_TTABLE_RAM_DATA_W = 64;
    localparam FSE_CTABLE_NUM_PARTITIONS = 2;
    localparam FSE_TTABLE_NUM_PARTITIONS = 4;

    localparam FSE_MAX_ACCURACY_LOG = 13;
    localparam FSE_RAM_SIZE = 1 << FSE_MAX_ACCURACY_LOG;

    wire [FSE_CTABLE_RAM_DATA_W-1:0]        ll_ctable_ram_rd_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         ll_ctable_ram_rd_addr;
    wire                                    ll_ctable_ram_rd_en;
    wire [FSE_CTABLE_NUM_PARTITIONS-1:0]    ll_ctable_ram_rd_mask;
    wire [FSE_CTABLE_RAM_DATA_W-1:0]        ll_ctable_ram_wr_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         ll_ctable_ram_wr_addr;
    wire                                    ll_ctable_ram_wr_en;
    wire [FSE_CTABLE_NUM_PARTITIONS-1:0]    ll_ctable_ram_wr_mask;

    wire [FSE_CTABLE_RAM_DATA_W-1:0]        ml_ctable_ram_rd_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         ml_ctable_ram_rd_addr;
    wire                                    ml_ctable_ram_rd_en;
    wire [FSE_CTABLE_NUM_PARTITIONS-1:0]    ml_ctable_ram_rd_mask;
    wire [FSE_CTABLE_RAM_DATA_W-1:0]        ml_ctable_ram_wr_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         ml_ctable_ram_wr_addr;
    wire                                    ml_ctable_ram_wr_en;
    wire [FSE_CTABLE_NUM_PARTITIONS-1:0]    ml_ctable_ram_wr_mask;

    wire [FSE_CTABLE_RAM_DATA_W-1:0]        of_ctable_ram_rd_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         of_ctable_ram_rd_addr;
    wire                                    of_ctable_ram_rd_en;
    wire [FSE_CTABLE_NUM_PARTITIONS-1:0]    of_ctable_ram_rd_mask;
    wire [FSE_CTABLE_RAM_DATA_W-1:0]        of_ctable_ram_wr_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         of_ctable_ram_wr_addr;
    wire                                    of_ctable_ram_wr_en;
    wire [FSE_CTABLE_NUM_PARTITIONS-1:0]    of_ctable_ram_wr_mask;

    wire [FSE_TTABLE_RAM_DATA_W-1:0]        ll_ttable_ram_rd_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         ll_ttable_ram_rd_addr;
    wire                                    ll_ttable_ram_rd_en;
    wire [FSE_TTABLE_NUM_PARTITIONS-1:0]    ll_ttable_ram_rd_mask;
    wire [FSE_TTABLE_RAM_DATA_W-1:0]        ll_ttable_ram_wr_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         ll_ttable_ram_wr_addr;
    wire                                    ll_ttable_ram_wr_en;
    wire [FSE_TTABLE_NUM_PARTITIONS-1:0]    ll_ttable_ram_wr_mask;

    wire [FSE_TTABLE_RAM_DATA_W-1:0]        ml_ttable_ram_rd_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         ml_ttable_ram_rd_addr;
    wire                                    ml_ttable_ram_rd_en;
    wire [FSE_TTABLE_NUM_PARTITIONS-1:0]    ml_ttable_ram_rd_mask;
    wire [FSE_TTABLE_RAM_DATA_W-1:0]        ml_ttable_ram_wr_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         ml_ttable_ram_wr_addr;
    wire                                    ml_ttable_ram_wr_en;
    wire [FSE_TTABLE_NUM_PARTITIONS-1:0]    ml_ttable_ram_wr_mask;

    wire [FSE_TTABLE_RAM_DATA_W-1:0]        of_ttable_ram_rd_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         of_ttable_ram_rd_addr;
    wire                                    of_ttable_ram_rd_en;
    wire [FSE_TTABLE_NUM_PARTITIONS-1:0]    of_ttable_ram_rd_mask;
    wire [FSE_TTABLE_RAM_DATA_W-1:0]        of_ttable_ram_wr_data;
    wire [FSE_TABLE_RAM_ADDR_W-1:0]         of_ttable_ram_wr_addr;
    wire                                    of_ttable_ram_wr_en;
    wire [FSE_TTABLE_NUM_PARTITIONS-1:0]    of_ttable_ram_wr_mask;

    ram_1r1w #(
        .DATA_WIDTH(FSE_CTABLE_RAM_DATA_W),
        .ADDR_WIDTH(FSE_TABLE_RAM_ADDR_W),
        .SIZE(FSE_RAM_SIZE),
        .NUM_PARTITIONS(FSE_CTABLE_NUM_PARTITIONS),
        .INIT_FILE("../../xls/modules/zstd/zstd_enc_ll_ctable_default.mem")
    ) ll_ctable_ram (
        .clk(clk),
        .rst(rst),
        .rd_addr (ll_ctable_ram_rd_addr),
        .rd_data (ll_ctable_ram_rd_data),
        .rd_en   (ll_ctable_ram_rd_en),
        .rd_mask (ll_ctable_ram_rd_mask),
        .wr_addr (ll_ctable_ram_wr_addr),
        .wr_data (ll_ctable_ram_wr_data),
        .wr_en   (ll_ctable_ram_wr_en),
        .wr_mask (ll_ctable_ram_wr_mask)
    );

    ram_1r1w #(
        .DATA_WIDTH(FSE_CTABLE_RAM_DATA_W),
        .ADDR_WIDTH(FSE_TABLE_RAM_ADDR_W),
        .SIZE(FSE_RAM_SIZE),
        .NUM_PARTITIONS(FSE_CTABLE_NUM_PARTITIONS),
        .INIT_FILE("../../xls/modules/zstd/zstd_enc_ml_ctable_default.mem")
    ) ml_ctable_ram (
        .clk(clk),
        .rst(rst),
        .rd_addr (ml_ctable_ram_rd_addr),
        .rd_data (ml_ctable_ram_rd_data),
        .rd_en   (ml_ctable_ram_rd_en),
        .rd_mask (ml_ctable_ram_rd_mask),
        .wr_addr (ml_ctable_ram_wr_addr),
        .wr_data (ml_ctable_ram_wr_data),
        .wr_en   (ml_ctable_ram_wr_en),
        .wr_mask (ml_ctable_ram_wr_mask)
    );

    ram_1r1w #(
        .DATA_WIDTH(FSE_CTABLE_RAM_DATA_W),
        .ADDR_WIDTH(FSE_TABLE_RAM_ADDR_W),
        .SIZE(FSE_RAM_SIZE),
        .NUM_PARTITIONS(FSE_CTABLE_NUM_PARTITIONS),
        .INIT_FILE("../../xls/modules/zstd/zstd_enc_of_ctable_default.mem")
    ) ol_ctable_ram (
        .clk(clk),
        .rst(rst),
        .rd_addr (of_ctable_ram_rd_addr),
        .rd_data (of_ctable_ram_rd_data),
        .rd_en   (of_ctable_ram_rd_en),
        .rd_mask (of_ctable_ram_rd_mask),
        .wr_addr (of_ctable_ram_wr_addr),
        .wr_data (of_ctable_ram_wr_data),
        .wr_en   (of_ctable_ram_wr_en),
        .wr_mask (of_ctable_ram_wr_mask)
    );

    ram_1r1w #(
        .DATA_WIDTH(FSE_TTABLE_RAM_DATA_W),
        .ADDR_WIDTH(FSE_TABLE_RAM_ADDR_W),
        .SIZE(FSE_RAM_SIZE),
        .NUM_PARTITIONS(FSE_TTABLE_NUM_PARTITIONS),
        .INIT_FILE("../../xls/modules/zstd/zstd_enc_ll_ttable_default.mem")
    ) ll_ttable_ram (
        .clk(clk),
        .rst(rst),
        .rd_addr (ll_ttable_ram_rd_addr),
        .rd_data (ll_ttable_ram_rd_data),
        .rd_en   (ll_ttable_ram_rd_en),
        .rd_mask (ll_ttable_ram_rd_mask),
        .wr_addr (ll_ttable_ram_wr_addr),
        .wr_data (ll_ttable_ram_wr_data),
        .wr_en   (ll_ttable_ram_wr_en),
        .wr_mask (ll_ttable_ram_wr_mask)
    );

    ram_1r1w #(
    .DATA_WIDTH(FSE_TTABLE_RAM_DATA_W),
        .ADDR_WIDTH(FSE_TABLE_RAM_ADDR_W),
        .SIZE(FSE_RAM_SIZE),
        .NUM_PARTITIONS(FSE_TTABLE_NUM_PARTITIONS),
        .INIT_FILE("../../xls/modules/zstd/zstd_enc_ml_ttable_default.mem")
    ) ml_ttable_ram (
        .clk(clk),
        .rst(rst),
        .rd_addr (ml_ttable_ram_rd_addr),
        .rd_data (ml_ttable_ram_rd_data),
        .rd_en   (ml_ttable_ram_rd_en),
        .rd_mask (ml_ttable_ram_rd_mask),
        .wr_addr (ml_ttable_ram_wr_addr),
        .wr_data (ml_ttable_ram_wr_data),
        .wr_en   (ml_ttable_ram_wr_en),
        .wr_mask (ml_ttable_ram_wr_mask)
    );

    ram_1r1w #(
        .DATA_WIDTH(FSE_TTABLE_RAM_DATA_W),
        .ADDR_WIDTH(FSE_TABLE_RAM_ADDR_W),
        .SIZE(FSE_RAM_SIZE),
        .NUM_PARTITIONS(FSE_TTABLE_NUM_PARTITIONS),
        .INIT_FILE("../../xls/modules/zstd/zstd_enc_of_ttable_default.mem")
    ) of_ttable_ram (
        .clk(clk),
        .rst(rst),
        .rd_addr (of_ttable_ram_rd_addr),
        .rd_data (of_ttable_ram_rd_data),
        .rd_en   (of_ttable_ram_rd_en),
        .rd_mask (of_ttable_ram_rd_mask),
        .wr_addr (of_ttable_ram_wr_addr),
        .wr_data (of_ttable_ram_wr_data),
        .wr_en   (of_ttable_ram_wr_en),
        .wr_mask (of_ttable_ram_wr_mask)
    );

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
        .zstd_enc__enc_req_r_vld(req_r_vld),

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
        .ht_ram_wr_mask(ht_ram_wr_mask),

        .ll_ctable_ram_rd_addr(ll_ctable_ram_rd_addr),
        .ll_ctable_ram_rd_data(ll_ctable_ram_rd_data),
        .ll_ctable_ram_rd_en  (ll_ctable_ram_rd_en),
        .ll_ctable_ram_rd_mask(ll_ctable_ram_rd_mask),
        .ll_ctable_ram_wr_addr(ll_ctable_ram_wr_addr),
        .ll_ctable_ram_wr_data(ll_ctable_ram_wr_data),
        .ll_ctable_ram_wr_en  (ll_ctable_ram_wr_en),
        .ll_ctable_ram_wr_mask(ll_ctable_ram_wr_mask),

        .ml_ctable_ram_rd_addr(ml_ctable_ram_rd_addr),
        .ml_ctable_ram_rd_data(ml_ctable_ram_rd_data),
        .ml_ctable_ram_rd_en  (ml_ctable_ram_rd_en),
        .ml_ctable_ram_rd_mask(ml_ctable_ram_rd_mask),
        .ml_ctable_ram_wr_addr(ml_ctable_ram_wr_addr),
        .ml_ctable_ram_wr_data(ml_ctable_ram_wr_data),
        .ml_ctable_ram_wr_en  (ml_ctable_ram_wr_en),
        .ml_ctable_ram_wr_mask(ml_ctable_ram_wr_mask),

        .of_ctable_ram_rd_addr(of_ctable_ram_rd_addr),
        .of_ctable_ram_rd_data(of_ctable_ram_rd_data),
        .of_ctable_ram_rd_en  (of_ctable_ram_rd_en),
        .of_ctable_ram_rd_mask(of_ctable_ram_rd_mask),
        .of_ctable_ram_wr_addr(of_ctable_ram_wr_addr),
        .of_ctable_ram_wr_data(of_ctable_ram_wr_data),
        .of_ctable_ram_wr_en  (of_ctable_ram_wr_en),
        .of_ctable_ram_wr_mask(of_ctable_ram_wr_mask),

        .ll_ttable_ram_rd_addr(ll_ttable_ram_rd_addr),
        .ll_ttable_ram_rd_data(ll_ttable_ram_rd_data),
        .ll_ttable_ram_rd_en  (ll_ttable_ram_rd_en),
        .ll_ttable_ram_rd_mask(ll_ttable_ram_rd_mask),
        .ll_ttable_ram_wr_addr(ll_ttable_ram_wr_addr),
        .ll_ttable_ram_wr_data(ll_ttable_ram_wr_data),
        .ll_ttable_ram_wr_en  (ll_ttable_ram_wr_en),
        .ll_ttable_ram_wr_mask(ll_ttable_ram_wr_mask),

        .ml_ttable_ram_rd_addr(ml_ttable_ram_rd_addr),
        .ml_ttable_ram_rd_data(ml_ttable_ram_rd_data),
        .ml_ttable_ram_rd_en  (ml_ttable_ram_rd_en),
        .ml_ttable_ram_rd_mask(ml_ttable_ram_rd_mask),
        .ml_ttable_ram_wr_addr(ml_ttable_ram_wr_addr),
        .ml_ttable_ram_wr_data(ml_ttable_ram_wr_data),
        .ml_ttable_ram_wr_en  (ml_ttable_ram_wr_en),
        .ml_ttable_ram_wr_mask(ml_ttable_ram_wr_mask),

        .of_ttable_ram_rd_addr(of_ttable_ram_rd_addr),
        .of_ttable_ram_rd_data(of_ttable_ram_rd_data),
        .of_ttable_ram_rd_en  (of_ttable_ram_rd_en),
        .of_ttable_ram_rd_mask(of_ttable_ram_rd_mask),
        .of_ttable_ram_wr_addr(of_ttable_ram_wr_addr),
        .of_ttable_ram_wr_data(of_ttable_ram_wr_data),
        .of_ttable_ram_wr_en  (of_ttable_ram_wr_en),
        .of_ttable_ram_wr_mask(of_ttable_ram_wr_mask)
    );

endmodule
