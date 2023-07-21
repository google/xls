// Copyright 2023 The XLS Authors
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


module lz4_decoder_wrap (
    input clk,
    input rst,
    // tokens input
    input [45:0] i_token_data,
    input i_token_vld,
    output i_token_rdy,
    // data output
    output [12:0] o_data_data,
    output o_data_vld,
    input o_data_rdy
);

    localparam RAM_HB_ADDR_WIDTH = 16;
    localparam RAM_HB_DATA_WIDTH = 8;
    localparam RAM_HB_NUM_PART = 1;

    // HB RAM bus
    wire [RAM_HB_ADDR_WIDTH+RAM_HB_DATA_WIDTH+RAM_HB_NUM_PART-1:0]
        ram_hb_wr_req_data;
    wire ram_hb_wr_req_vld;
    wire ram_hb_wr_req_rdy;
    wire ram_hb_wr_resp_vld;
    wire ram_hb_wr_resp_rdy;
    wire [RAM_HB_ADDR_WIDTH+RAM_HB_NUM_PART-1:0] ram_hb_rd_req_data;
    wire ram_hb_rd_req_vld;
    wire ram_hb_rd_req_rdy;
    wire [RAM_HB_DATA_WIDTH-1:0] ram_hb_rd_resp_data;
    wire ram_hb_rd_resp_vld;
    wire ram_hb_rd_resp_rdy;

    // HB RAM
    sdpram_xls_chan #(
        .DATA_WIDTH(RAM_HB_DATA_WIDTH),
        .ADDR_WIDTH(RAM_HB_ADDR_WIDTH),
        .NUM_PARTITIONS(RAM_HB_NUM_PART)
    ) ram_hb (
        .clk(clk),
        .rst(rst),
        .wr_req_data(ram_hb_wr_req_data),
        .wr_req_vld(ram_hb_wr_req_vld),
        .wr_req_rdy(ram_hb_wr_req_rdy),
        .wr_resp_vld(ram_hb_wr_resp_vld),
        .wr_resp_rdy(ram_hb_wr_resp_rdy),
        .rd_req_data(ram_hb_rd_req_data),
        .rd_req_vld(ram_hb_rd_req_vld),
        .rd_req_rdy(ram_hb_rd_req_rdy),
        .rd_resp_data(ram_hb_rd_resp_data),
        .rd_resp_vld(ram_hb_rd_resp_vld),
        .rd_resp_rdy(ram_hb_rd_resp_rdy)
    );

    // Decoder engine
    dbe_lz4_decoder dut (
        .clk(clk),
        .rst(rst),
        .lz4_decoder__i_encoded_data(i_token_data),
        .lz4_decoder__i_encoded_vld(i_token_vld),
        .lz4_decoder__i_encoded_rdy(i_token_rdy), 
        .lz4_decoder__o_data_data(o_data_data),
        .lz4_decoder__o_data_vld(o_data_vld),
        .lz4_decoder__o_data_rdy(o_data_rdy),
        // RAM
        .lz4_decoder__o_ram_hb_wr_req_data(ram_hb_wr_req_data),
        .lz4_decoder__o_ram_hb_wr_req_vld(ram_hb_wr_req_vld),
        .lz4_decoder__o_ram_hb_wr_req_rdy(ram_hb_wr_req_rdy),
        .lz4_decoder__i_ram_hb_wr_resp_vld(ram_hb_wr_resp_vld),
        .lz4_decoder__i_ram_hb_wr_resp_rdy(ram_hb_wr_resp_rdy),
        .lz4_decoder__o_ram_hb_rd_req_data(ram_hb_rd_req_data),
        .lz4_decoder__o_ram_hb_rd_req_vld(ram_hb_rd_req_vld),
        .lz4_decoder__o_ram_hb_rd_req_rdy(ram_hb_rd_req_rdy),
        .lz4_decoder__i_ram_hb_rd_resp_data(ram_hb_rd_resp_data),
        .lz4_decoder__i_ram_hb_rd_resp_vld(ram_hb_rd_resp_vld),
        .lz4_decoder__i_ram_hb_rd_resp_rdy(ram_hb_rd_resp_rdy)
    );

    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, lz4_decoder_wrap);
    end

endmodule