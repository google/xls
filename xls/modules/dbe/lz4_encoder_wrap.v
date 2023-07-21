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


module lz4_encoder_wrap (
    input clk,
    input rst,
    // data input
    // [u1 is_mark, u8 data, u4 mark]
    input [12:0] i_data_data,
    input i_data_vld,
    output i_data_rdy,
    // tokens output
    // [u2 kind, u8 lt_sym, u16 cp_off, u16 cp_cnt, u4 mark]
    output [45:0] o_token_data,
    output o_token_vld,
    input o_token_rdy
);

    localparam RAM_HB_ADDR_WIDTH = 16;
    localparam RAM_HB_DATA_WIDTH = 8;

    localparam RAM_HT_ADDR_WIDTH = 13;
    localparam RAM_HT_DATA_WIDTH = 16;

    // HB RAM bus
    wire [RAM_HB_ADDR_WIDTH+RAM_HB_DATA_WIDTH+1:0] ram_hb_req_data;
    wire ram_hb_req_vld;
    wire ram_hb_req_rdy;
    wire [RAM_HB_DATA_WIDTH-1:0] ram_hb_resp_data;
    wire ram_hb_resp_vld;
    wire ram_hb_resp_rdy;
    wire ram_hb_wr_comp_vld;
    wire ram_hb_wr_comp_rdy;

    // HT RAM bus
    wire [RAM_HT_ADDR_WIDTH+RAM_HT_DATA_WIDTH+1:0] ram_ht_req_data;
    wire ram_ht_req_vld;
    wire ram_ht_req_rdy;
    wire [RAM_HT_DATA_WIDTH-1:0] ram_ht_resp_data;
    wire ram_ht_resp_vld;
    wire ram_ht_resp_rdy;
    wire ram_ht_wr_comp_vld;
    wire ram_ht_wr_comp_rdy;

    // HB RAM
    spram_xls_chan #(
        .DATA_WIDTH(RAM_HB_DATA_WIDTH),
        .ADDR_WIDTH(RAM_HB_ADDR_WIDTH)
    ) ram_hb (
        .clk(clk),
        .rst(rst),
        .req_data(ram_hb_req_data),
        .req_vld(ram_hb_req_vld),
        .req_rdy(ram_hb_req_rdy),
        .resp_data(ram_hb_resp_data),
        .resp_vld(ram_hb_resp_vld),
        .resp_rdy(ram_hb_resp_rdy),
        .wr_comp_vld(ram_hb_wr_comp_vld),
        .wr_comp_rdy(ram_hb_wr_comp_rdy)
    );

    // HT RAM
    spram_xls_chan #(
        .DATA_WIDTH(RAM_HT_DATA_WIDTH),
        .ADDR_WIDTH(RAM_HT_ADDR_WIDTH)
    ) ram_ht (
        .clk(clk),
        .rst(rst),
        .req_data(ram_ht_req_data),
        .req_vld(ram_ht_req_vld),
        .req_rdy(ram_ht_req_rdy),
        .resp_data(ram_ht_resp_data),
        .resp_vld(ram_ht_resp_vld),
        .resp_rdy(ram_ht_resp_rdy),
        .wr_comp_vld(ram_ht_wr_comp_vld),
        .wr_comp_rdy(ram_ht_wr_comp_rdy)
    );

    // Encoder engine
    dbe_lz4_encoder dut (
        .clk(clk),
        .rst(rst),
        .lz4_encoder__i_data_data(i_data_data),
        .lz4_encoder__i_data_vld(i_data_vld),
        .lz4_encoder__i_data_rdy(i_data_rdy),
        .lz4_encoder__o_encoded_data(o_token_data),
        .lz4_encoder__o_encoded_vld(o_token_vld),
        .lz4_encoder__o_encoded_rdy(o_token_rdy), 
        // HB RAM
        .lz4_encoder__o_ram_hb_req_data(ram_hb_req_data),
        .lz4_encoder__o_ram_hb_req_vld(ram_hb_req_vld),
        .lz4_encoder__o_ram_hb_req_rdy(ram_hb_req_rdy),
        .lz4_encoder__i_ram_hb_resp_data(ram_hb_resp_data),
        .lz4_encoder__i_ram_hb_resp_vld(ram_hb_resp_vld),
        .lz4_encoder__i_ram_hb_resp_rdy(ram_hb_resp_rdy),
        .lz4_encoder__i_ram_hb_wr_comp_vld(ram_hb_wr_comp_vld),
        .lz4_encoder__i_ram_hb_wr_comp_rdy(ram_hb_wr_comp_rdy),
        // HT RAM
        .lz4_encoder__o_ram_ht_req_data(ram_ht_req_data),
        .lz4_encoder__o_ram_ht_req_vld(ram_ht_req_vld),
        .lz4_encoder__o_ram_ht_req_rdy(ram_ht_req_rdy),
        .lz4_encoder__i_ram_ht_resp_data(ram_ht_resp_data),
        .lz4_encoder__i_ram_ht_resp_vld(ram_ht_resp_vld),
        .lz4_encoder__i_ram_ht_resp_rdy(ram_ht_resp_rdy),
        .lz4_encoder__i_ram_ht_wr_comp_vld(ram_ht_wr_comp_vld),
        .lz4_encoder__i_ram_ht_wr_comp_rdy(ram_ht_wr_comp_rdy)
    );

    initial begin
        $dumpfile("dump.vcd");
        $dumpvars(0, lz4_encoder_wrap);
    end

endmodule