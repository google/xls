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

module ram_demux_wrapper #(
    parameter DATA_WIDTH = 64,
    parameter SIZE = 1024,
    parameter ADDR_WIDTH = $clog2(SIZE),
    parameter NUM_PARTITIONS = 64
) (
    input wire clk,
    input wire rst,

    input wire                      ram_demux__sel_req_r_data,
    input wire                      ram_demux__sel_req_r_vld,
    output wire                     ram_demux__sel_req_r_rdy,

    input wire                      ram_demux__sel_resp_s_rdy,
    output wire                     ram_demux__sel_resp_s_vld,

    input wire [ NUM_PARTITIONS + ADDR_WIDTH - 1:0 ] ram_demux__rd_req_r_data,
    input wire                                       ram_demux__rd_req_r_vld,
    output wire                                      ram_demux__rd_req_r_rdy,

    output wire [ DATA_WIDTH -1:0 ] ram_demux__rd_resp_s_data,
    output wire                     ram_demux__rd_resp_s_vld,
    input wire                      ram_demux__rd_resp_s_rdy,

    input wire [ DATA_WIDTH + NUM_PARTITIONS + ADDR_WIDTH -1:0 ] ram_demux__wr_req_r_data,
    input wire                                                   ram_demux__wr_req_r_vld,
    output wire                                                  ram_demux__wr_req_r_rdy,

    input wire                      ram_demux__wr_resp_s_rdy,
    output wire                     ram_demux__wr_resp_s_vld
);

wire [ DATA_WIDTH     -1:0 ]   ram0_wr_data;
wire [ ADDR_WIDTH     -1:0 ]   ram0_wr_addr;
wire                           ram0_wr_en;
wire [ NUM_PARTITIONS -1:0 ]   ram0_wr_mask;

wire [ DATA_WIDTH     -1:0 ]   ram0_rd_data;
wire [ ADDR_WIDTH     -1:0 ]   ram0_rd_addr;
wire                           ram0_rd_en;
wire [ NUM_PARTITIONS -1:0 ]   ram0_rd_mask;

ram_1r1w # (
    .DATA_WIDTH(DATA_WIDTH),
    .SIZE(SIZE),
    .NUM_PARTITIONS(NUM_PARTITIONS),
    .ADDR_WIDTH(ADDR_WIDTH)
) ram0 (
    .clk     (clk),
    .rst     (rst),

    .wr_data (ram0_wr_data),
    .wr_addr (ram0_wr_addr),
    .wr_en   (ram0_wr_en),
    .wr_mask (ram0_wr_mask),

    .rd_data (ram0_rd_data),
    .rd_addr (ram0_rd_addr),
    .rd_en   (ram0_rd_en),
    .rd_mask (ram0_rd_mask)
);

wire [ DATA_WIDTH     -1:0 ]   ram1_wr_data;
wire [ ADDR_WIDTH     -1:0 ]   ram1_wr_addr;
wire                           ram1_wr_en;
wire [ NUM_PARTITIONS -1:0 ]   ram1_wr_mask;

wire [ DATA_WIDTH     -1:0 ]   ram1_rd_data;
wire [ ADDR_WIDTH     -1:0 ]   ram1_rd_addr;
wire                           ram1_rd_en;
wire [ NUM_PARTITIONS -1:0 ]   ram1_rd_mask;


ram_1r1w # (
    .DATA_WIDTH(DATA_WIDTH),
    .SIZE(SIZE),
    .NUM_PARTITIONS(NUM_PARTITIONS),
    .ADDR_WIDTH(ADDR_WIDTH)
) ram1 (
    .clk     (clk),
    .rst     (rst),

    .wr_data (ram1_wr_data),
    .wr_addr (ram1_wr_addr),
    .wr_en   (ram1_wr_en),
    .wr_mask (ram1_wr_mask),

    .rd_data (ram1_rd_data),
    .rd_addr (ram1_rd_addr),
    .rd_en   (ram1_rd_en),
    .rd_mask (ram1_rd_mask)
);

RamDemux demux (
    .clk(clk),
    .rst(rst),

    .ram_demux__rd_req_r_data(ram_demux__rd_req_r_data),
    .ram_demux__rd_req_r_vld(ram_demux__rd_req_r_vld),
    .ram_demux__rd_req_r_rdy(ram_demux__rd_req_r_rdy),

    .ram_demux__sel_req_r_data(ram_demux__sel_req_r_data),
    .ram_demux__sel_req_r_rdy(ram_demux__sel_req_r_rdy),
    .ram_demux__sel_req_r_vld(ram_demux__sel_req_r_vld),

    .ram_demux__sel_resp_s_rdy(ram_demux__sel_resp_s_rdy),
    .ram_demux__sel_resp_s_vld(ram_demux__sel_resp_s_vld),

    .ram_demux__wr_req_r_vld(ram_demux__wr_req_r_vld),
    .ram_demux__wr_req_r_rdy(ram_demux__wr_req_r_rdy),
    .ram_demux__wr_req_r_data(ram_demux__wr_req_r_data),

    .ram_demux__wr_resp_s_rdy(ram_demux__wr_resp_s_rdy),
    .ram_demux__wr_resp_s_vld(ram_demux__wr_resp_s_vld),

    .ram_demux__rd_resp_s_rdy(ram_demux__rd_resp_s_rdy),
    .ram_demux__rd_resp_s_data(ram_demux__rd_resp_s_data),
    .ram_demux__rd_resp_s_vld(ram_demux__rd_resp_s_vld),

    .ram0_rd_data (ram0_rd_data),
    .ram0_rd_addr (ram0_rd_addr),
    .ram0_rd_mask (ram0_rd_mask),
    .ram0_rd_en   (ram0_rd_en),

    .ram0_wr_addr (ram0_wr_addr),
    .ram0_wr_data (ram0_wr_data),
    .ram0_wr_mask (ram0_wr_mask),
    .ram0_wr_en   (ram0_wr_en),

    .ram1_rd_data (ram1_rd_data),
    .ram1_rd_addr (ram1_rd_addr),
    .ram1_rd_mask (ram1_rd_mask),
    .ram1_rd_en   (ram1_rd_en),

    .ram1_wr_addr (ram1_wr_addr),
    .ram1_wr_data (ram1_wr_data),
    .ram1_wr_mask (ram1_wr_mask),
    .ram1_wr_en   (ram1_wr_en)
);

endmodule
