/*

Copyright (c) 2020 Alex Forencich

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

*/

// Language: Verilog 2001

`resetall
`timescale 1ns / 1ps
`default_nettype none

/*
 * AXI4 4x1 crossbar (wrapper)
 */
module axi_crossbar_wrapper #
(
    // Width of data bus in bits
    parameter DATA_WIDTH = 32,
    // Width of address bus in bits
    parameter ADDR_WIDTH = 32,
    // Width of wstrb (width of data bus in words)
    parameter STRB_WIDTH = (DATA_WIDTH/8),
    // Input ID field width (from AXI masters)
    parameter S_ID_WIDTH = 8,
    // Output ID field width (towards AXI slaves)
    // Additional bits required for response routing
    parameter M_ID_WIDTH = S_ID_WIDTH+$clog2(S_COUNT),
    // Propagate awuser signal
    parameter AWUSER_ENABLE = 0,
    // Width of awuser signal
    parameter AWUSER_WIDTH = 1,
    // Propagate wuser signal
    parameter WUSER_ENABLE = 0,
    // Width of wuser signal
    parameter WUSER_WIDTH = 1,
    // Propagate buser signal
    parameter BUSER_ENABLE = 0,
    // Width of buser signal
    parameter BUSER_WIDTH = 1,
    // Propagate aruser signal
    parameter ARUSER_ENABLE = 0,
    // Width of aruser signal
    parameter ARUSER_WIDTH = 1,
    // Propagate ruser signal
    parameter RUSER_ENABLE = 0,
    // Width of ruser signal
    parameter RUSER_WIDTH = 1,
    // Number of concurrent unique IDs
    parameter S00_THREADS = 2,
    // Number of concurrent operations
    parameter S00_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S01_THREADS = 2,
    // Number of concurrent operations
    parameter S01_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S02_THREADS = 2,
    // Number of concurrent operations
    parameter S02_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S03_THREADS = 2,
    // Number of concurrent operations
    parameter S03_ACCEPT = 16,
    // Number of regions per master interface
    parameter M_REGIONS = 1,
    // Master interface base addresses
    // M_REGIONS concatenated fields of ADDR_WIDTH bits
    parameter M00_BASE_ADDR = 0,
    // Master interface address widths
    // M_REGIONS concatenated fields of 32 bits
    parameter M00_ADDR_WIDTH = {M_REGIONS{32'd24}},
    // Read connections between interfaces
    // S_COUNT bits
    parameter M00_CONNECT_READ = 4'b1111,
    // Write connections between interfaces
    // S_COUNT bits
    parameter M00_CONNECT_WRITE = 4'b1111,
    // Number of concurrent operations for each master interface
    parameter M00_ISSUE = 4,
    // Secure master (fail operations based on awprot/arprot)
    parameter M00_SECURE = 0,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S00_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S00_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S00_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S00_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S00_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S01_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S01_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S01_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S01_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S01_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S02_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S02_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S02_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S02_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S02_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S03_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S03_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S03_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S03_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S03_R_REG_TYPE = 2,
    // Master interface AW channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter M00_AW_REG_TYPE = 1,
    // Master interface W channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter M00_W_REG_TYPE = 2,
    // Master interface B channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter M00_B_REG_TYPE = 0,
    // Master interface AR channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter M00_AR_REG_TYPE = 1,
    // Master interface R channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter M00_R_REG_TYPE = 0
)
(
    input  wire                     clk,
    input  wire                     rst,

    /*
     * AXI slave interface
     */
    input  wire [S_ID_WIDTH-1:0]    s00_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s00_axi_awaddr,
    input  wire [7:0]               s00_axi_awlen,
    input  wire [2:0]               s00_axi_awsize,
    input  wire [1:0]               s00_axi_awburst,
    input  wire                     s00_axi_awlock,
    input  wire [3:0]               s00_axi_awcache,
    input  wire [2:0]               s00_axi_awprot,
    input  wire [3:0]               s00_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s00_axi_awuser,
    input  wire                     s00_axi_awvalid,
    output wire                     s00_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s00_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s00_axi_wstrb,
    input  wire                     s00_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s00_axi_wuser,
    input  wire                     s00_axi_wvalid,
    output wire                     s00_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s00_axi_bid,
    output wire [1:0]               s00_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s00_axi_buser,
    output wire                     s00_axi_bvalid,
    input  wire                     s00_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s00_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s00_axi_araddr,
    input  wire [7:0]               s00_axi_arlen,
    input  wire [2:0]               s00_axi_arsize,
    input  wire [1:0]               s00_axi_arburst,
    input  wire                     s00_axi_arlock,
    input  wire [3:0]               s00_axi_arcache,
    input  wire [2:0]               s00_axi_arprot,
    input  wire [3:0]               s00_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s00_axi_aruser,
    input  wire                     s00_axi_arvalid,
    output wire                     s00_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s00_axi_rid,
    output wire [DATA_WIDTH-1:0]    s00_axi_rdata,
    output wire [1:0]               s00_axi_rresp,
    output wire                     s00_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s00_axi_ruser,
    output wire                     s00_axi_rvalid,
    input  wire                     s00_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s01_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s01_axi_awaddr,
    input  wire [7:0]               s01_axi_awlen,
    input  wire [2:0]               s01_axi_awsize,
    input  wire [1:0]               s01_axi_awburst,
    input  wire                     s01_axi_awlock,
    input  wire [3:0]               s01_axi_awcache,
    input  wire [2:0]               s01_axi_awprot,
    input  wire [3:0]               s01_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s01_axi_awuser,
    input  wire                     s01_axi_awvalid,
    output wire                     s01_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s01_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s01_axi_wstrb,
    input  wire                     s01_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s01_axi_wuser,
    input  wire                     s01_axi_wvalid,
    output wire                     s01_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s01_axi_bid,
    output wire [1:0]               s01_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s01_axi_buser,
    output wire                     s01_axi_bvalid,
    input  wire                     s01_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s01_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s01_axi_araddr,
    input  wire [7:0]               s01_axi_arlen,
    input  wire [2:0]               s01_axi_arsize,
    input  wire [1:0]               s01_axi_arburst,
    input  wire                     s01_axi_arlock,
    input  wire [3:0]               s01_axi_arcache,
    input  wire [2:0]               s01_axi_arprot,
    input  wire [3:0]               s01_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s01_axi_aruser,
    input  wire                     s01_axi_arvalid,
    output wire                     s01_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s01_axi_rid,
    output wire [DATA_WIDTH-1:0]    s01_axi_rdata,
    output wire [1:0]               s01_axi_rresp,
    output wire                     s01_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s01_axi_ruser,
    output wire                     s01_axi_rvalid,
    input  wire                     s01_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s02_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s02_axi_awaddr,
    input  wire [7:0]               s02_axi_awlen,
    input  wire [2:0]               s02_axi_awsize,
    input  wire [1:0]               s02_axi_awburst,
    input  wire                     s02_axi_awlock,
    input  wire [3:0]               s02_axi_awcache,
    input  wire [2:0]               s02_axi_awprot,
    input  wire [3:0]               s02_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s02_axi_awuser,
    input  wire                     s02_axi_awvalid,
    output wire                     s02_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s02_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s02_axi_wstrb,
    input  wire                     s02_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s02_axi_wuser,
    input  wire                     s02_axi_wvalid,
    output wire                     s02_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s02_axi_bid,
    output wire [1:0]               s02_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s02_axi_buser,
    output wire                     s02_axi_bvalid,
    input  wire                     s02_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s02_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s02_axi_araddr,
    input  wire [7:0]               s02_axi_arlen,
    input  wire [2:0]               s02_axi_arsize,
    input  wire [1:0]               s02_axi_arburst,
    input  wire                     s02_axi_arlock,
    input  wire [3:0]               s02_axi_arcache,
    input  wire [2:0]               s02_axi_arprot,
    input  wire [3:0]               s02_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s02_axi_aruser,
    input  wire                     s02_axi_arvalid,
    output wire                     s02_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s02_axi_rid,
    output wire [DATA_WIDTH-1:0]    s02_axi_rdata,
    output wire [1:0]               s02_axi_rresp,
    output wire                     s02_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s02_axi_ruser,
    output wire                     s02_axi_rvalid,
    input  wire                     s02_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s03_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s03_axi_awaddr,
    input  wire [7:0]               s03_axi_awlen,
    input  wire [2:0]               s03_axi_awsize,
    input  wire [1:0]               s03_axi_awburst,
    input  wire                     s03_axi_awlock,
    input  wire [3:0]               s03_axi_awcache,
    input  wire [2:0]               s03_axi_awprot,
    input  wire [3:0]               s03_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s03_axi_awuser,
    input  wire                     s03_axi_awvalid,
    output wire                     s03_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s03_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s03_axi_wstrb,
    input  wire                     s03_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s03_axi_wuser,
    input  wire                     s03_axi_wvalid,
    output wire                     s03_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s03_axi_bid,
    output wire [1:0]               s03_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s03_axi_buser,
    output wire                     s03_axi_bvalid,
    input  wire                     s03_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s03_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s03_axi_araddr,
    input  wire [7:0]               s03_axi_arlen,
    input  wire [2:0]               s03_axi_arsize,
    input  wire [1:0]               s03_axi_arburst,
    input  wire                     s03_axi_arlock,
    input  wire [3:0]               s03_axi_arcache,
    input  wire [2:0]               s03_axi_arprot,
    input  wire [3:0]               s03_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s03_axi_aruser,
    input  wire                     s03_axi_arvalid,
    output wire                     s03_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s03_axi_rid,
    output wire [DATA_WIDTH-1:0]    s03_axi_rdata,
    output wire [1:0]               s03_axi_rresp,
    output wire                     s03_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s03_axi_ruser,
    output wire                     s03_axi_rvalid,
    input  wire                     s03_axi_rready,

    /*
     * AXI master interface
     */
    output wire [M_ID_WIDTH-1:0]    m00_axi_awid,
    output wire [ADDR_WIDTH-1:0]    m00_axi_awaddr,
    output wire [7:0]               m00_axi_awlen,
    output wire [2:0]               m00_axi_awsize,
    output wire [1:0]               m00_axi_awburst,
    output wire                     m00_axi_awlock,
    output wire [3:0]               m00_axi_awcache,
    output wire [2:0]               m00_axi_awprot,
    output wire [3:0]               m00_axi_awqos,
    output wire [3:0]               m00_axi_awregion,
    output wire [AWUSER_WIDTH-1:0]  m00_axi_awuser,
    output wire                     m00_axi_awvalid,
    input  wire                     m00_axi_awready,
    output wire [DATA_WIDTH-1:0]    m00_axi_wdata,
    output wire [STRB_WIDTH-1:0]    m00_axi_wstrb,
    output wire                     m00_axi_wlast,
    output wire [WUSER_WIDTH-1:0]   m00_axi_wuser,
    output wire                     m00_axi_wvalid,
    input  wire                     m00_axi_wready,
    input  wire [M_ID_WIDTH-1:0]    m00_axi_bid,
    input  wire [1:0]               m00_axi_bresp,
    input  wire [BUSER_WIDTH-1:0]   m00_axi_buser,
    input  wire                     m00_axi_bvalid,
    output wire                     m00_axi_bready,
    output wire [M_ID_WIDTH-1:0]    m00_axi_arid,
    output wire [ADDR_WIDTH-1:0]    m00_axi_araddr,
    output wire [7:0]               m00_axi_arlen,
    output wire [2:0]               m00_axi_arsize,
    output wire [1:0]               m00_axi_arburst,
    output wire                     m00_axi_arlock,
    output wire [3:0]               m00_axi_arcache,
    output wire [2:0]               m00_axi_arprot,
    output wire [3:0]               m00_axi_arqos,
    output wire [3:0]               m00_axi_arregion,
    output wire [ARUSER_WIDTH-1:0]  m00_axi_aruser,
    output wire                     m00_axi_arvalid,
    input  wire                     m00_axi_arready,
    input  wire [M_ID_WIDTH-1:0]    m00_axi_rid,
    input  wire [DATA_WIDTH-1:0]    m00_axi_rdata,
    input  wire [1:0]               m00_axi_rresp,
    input  wire                     m00_axi_rlast,
    input  wire [RUSER_WIDTH-1:0]   m00_axi_ruser,
    input  wire                     m00_axi_rvalid,
    output wire                     m00_axi_rready
);

localparam S_COUNT = 4;
localparam M_COUNT = 1;

// parameter sizing helpers
function [ADDR_WIDTH*M_REGIONS-1:0] w_a_r(input [ADDR_WIDTH*M_REGIONS-1:0] val);
    w_a_r = val;
endfunction

function [32*M_REGIONS-1:0] w_32_r(input [32*M_REGIONS-1:0] val);
    w_32_r = val;
endfunction

function [S_COUNT-1:0] w_s(input [S_COUNT-1:0] val);
    w_s = val;
endfunction

function [31:0] w_32(input [31:0] val);
    w_32 = val;
endfunction

function [1:0] w_2(input [1:0] val);
    w_2 = val;
endfunction

function w_1(input val);
    w_1 = val;
endfunction

axi_crossbar #(
    .S_COUNT(S_COUNT),
    .M_COUNT(M_COUNT),
    .DATA_WIDTH(DATA_WIDTH),
    .ADDR_WIDTH(ADDR_WIDTH),
    .STRB_WIDTH(STRB_WIDTH),
    .S_ID_WIDTH(S_ID_WIDTH),
    .M_ID_WIDTH(M_ID_WIDTH),
    .AWUSER_ENABLE(AWUSER_ENABLE),
    .AWUSER_WIDTH(AWUSER_WIDTH),
    .WUSER_ENABLE(WUSER_ENABLE),
    .WUSER_WIDTH(WUSER_WIDTH),
    .BUSER_ENABLE(BUSER_ENABLE),
    .BUSER_WIDTH(BUSER_WIDTH),
    .ARUSER_ENABLE(ARUSER_ENABLE),
    .ARUSER_WIDTH(ARUSER_WIDTH),
    .RUSER_ENABLE(RUSER_ENABLE),
    .RUSER_WIDTH(RUSER_WIDTH),
    .S_THREADS({ w_32(S03_THREADS), w_32(S02_THREADS), w_32(S01_THREADS), w_32(S00_THREADS) }),
    .S_ACCEPT({ w_32(S03_ACCEPT), w_32(S02_ACCEPT), w_32(S01_ACCEPT), w_32(S00_ACCEPT) }),
    .M_REGIONS(M_REGIONS),
    .M_BASE_ADDR({ w_a_r(M00_BASE_ADDR) }),
    .M_ADDR_WIDTH({ w_32_r(M00_ADDR_WIDTH) }),
    .M_CONNECT_READ({ w_s(M00_CONNECT_READ) }),
    .M_CONNECT_WRITE({ w_s(M00_CONNECT_WRITE) }),
    .M_ISSUE({ w_32(M00_ISSUE) }),
    .M_SECURE({ w_1(M00_SECURE) }),
    .S_AR_REG_TYPE({ w_2(S03_AR_REG_TYPE), w_2(S02_AR_REG_TYPE), w_2(S01_AR_REG_TYPE), w_2(S00_AR_REG_TYPE) }),
    .S_R_REG_TYPE({ w_2(S03_R_REG_TYPE), w_2(S02_R_REG_TYPE), w_2(S01_R_REG_TYPE), w_2(S00_R_REG_TYPE) }),
    .S_AW_REG_TYPE({ w_2(S03_AW_REG_TYPE), w_2(S02_AW_REG_TYPE), w_2(S01_AW_REG_TYPE), w_2(S00_AW_REG_TYPE) }),
    .S_W_REG_TYPE({ w_2(S03_W_REG_TYPE), w_2(S02_W_REG_TYPE), w_2(S01_W_REG_TYPE), w_2(S00_W_REG_TYPE) }),
    .S_B_REG_TYPE({ w_2(S03_B_REG_TYPE), w_2(S02_B_REG_TYPE), w_2(S01_B_REG_TYPE), w_2(S00_B_REG_TYPE) }),
    .M_AR_REG_TYPE({ w_2(M00_AR_REG_TYPE) }),
    .M_R_REG_TYPE({ w_2(M00_R_REG_TYPE) }),
    .M_AW_REG_TYPE({ w_2(M00_AW_REG_TYPE) }),
    .M_W_REG_TYPE({ w_2(M00_W_REG_TYPE) }),
    .M_B_REG_TYPE({ w_2(M00_B_REG_TYPE) })
)
axi_crossbar_inst (
    .clk(clk),
    .rst(rst),
    .s_axi_awid({ s03_axi_awid, s02_axi_awid, s01_axi_awid, s00_axi_awid }),
    .s_axi_awaddr({ s03_axi_awaddr, s02_axi_awaddr, s01_axi_awaddr, s00_axi_awaddr }),
    .s_axi_awlen({ s03_axi_awlen, s02_axi_awlen, s01_axi_awlen, s00_axi_awlen }),
    .s_axi_awsize({ s03_axi_awsize, s02_axi_awsize, s01_axi_awsize, s00_axi_awsize }),
    .s_axi_awburst({ s03_axi_awburst, s02_axi_awburst, s01_axi_awburst, s00_axi_awburst }),
    .s_axi_awlock({ s03_axi_awlock, s02_axi_awlock, s01_axi_awlock, s00_axi_awlock }),
    .s_axi_awcache({ s03_axi_awcache, s02_axi_awcache, s01_axi_awcache, s00_axi_awcache }),
    .s_axi_awprot({ s03_axi_awprot, s02_axi_awprot, s01_axi_awprot, s00_axi_awprot }),
    .s_axi_awqos({ s03_axi_awqos, s02_axi_awqos, s01_axi_awqos, s00_axi_awqos }),
    .s_axi_awuser({ s03_axi_awuser, s02_axi_awuser, s01_axi_awuser, s00_axi_awuser }),
    .s_axi_awvalid({ s03_axi_awvalid, s02_axi_awvalid, s01_axi_awvalid, s00_axi_awvalid }),
    .s_axi_awready({ s03_axi_awready, s02_axi_awready, s01_axi_awready, s00_axi_awready }),
    .s_axi_wdata({ s03_axi_wdata, s02_axi_wdata, s01_axi_wdata, s00_axi_wdata }),
    .s_axi_wstrb({ s03_axi_wstrb, s02_axi_wstrb, s01_axi_wstrb, s00_axi_wstrb }),
    .s_axi_wlast({ s03_axi_wlast, s02_axi_wlast, s01_axi_wlast, s00_axi_wlast }),
    .s_axi_wuser({ s03_axi_wuser, s02_axi_wuser, s01_axi_wuser, s00_axi_wuser }),
    .s_axi_wvalid({ s03_axi_wvalid, s02_axi_wvalid, s01_axi_wvalid, s00_axi_wvalid }),
    .s_axi_wready({ s03_axi_wready, s02_axi_wready, s01_axi_wready, s00_axi_wready }),
    .s_axi_bid({ s03_axi_bid, s02_axi_bid, s01_axi_bid, s00_axi_bid }),
    .s_axi_bresp({ s03_axi_bresp, s02_axi_bresp, s01_axi_bresp, s00_axi_bresp }),
    .s_axi_buser({ s03_axi_buser, s02_axi_buser, s01_axi_buser, s00_axi_buser }),
    .s_axi_bvalid({ s03_axi_bvalid, s02_axi_bvalid, s01_axi_bvalid, s00_axi_bvalid }),
    .s_axi_bready({ s03_axi_bready, s02_axi_bready, s01_axi_bready, s00_axi_bready }),
    .s_axi_arid({ s03_axi_arid, s02_axi_arid, s01_axi_arid, s00_axi_arid }),
    .s_axi_araddr({ s03_axi_araddr, s02_axi_araddr, s01_axi_araddr, s00_axi_araddr }),
    .s_axi_arlen({ s03_axi_arlen, s02_axi_arlen, s01_axi_arlen, s00_axi_arlen }),
    .s_axi_arsize({ s03_axi_arsize, s02_axi_arsize, s01_axi_arsize, s00_axi_arsize }),
    .s_axi_arburst({ s03_axi_arburst, s02_axi_arburst, s01_axi_arburst, s00_axi_arburst }),
    .s_axi_arlock({ s03_axi_arlock, s02_axi_arlock, s01_axi_arlock, s00_axi_arlock }),
    .s_axi_arcache({ s03_axi_arcache, s02_axi_arcache, s01_axi_arcache, s00_axi_arcache }),
    .s_axi_arprot({ s03_axi_arprot, s02_axi_arprot, s01_axi_arprot, s00_axi_arprot }),
    .s_axi_arqos({ s03_axi_arqos, s02_axi_arqos, s01_axi_arqos, s00_axi_arqos }),
    .s_axi_aruser({ s03_axi_aruser, s02_axi_aruser, s01_axi_aruser, s00_axi_aruser }),
    .s_axi_arvalid({ s03_axi_arvalid, s02_axi_arvalid, s01_axi_arvalid, s00_axi_arvalid }),
    .s_axi_arready({ s03_axi_arready, s02_axi_arready, s01_axi_arready, s00_axi_arready }),
    .s_axi_rid({ s03_axi_rid, s02_axi_rid, s01_axi_rid, s00_axi_rid }),
    .s_axi_rdata({ s03_axi_rdata, s02_axi_rdata, s01_axi_rdata, s00_axi_rdata }),
    .s_axi_rresp({ s03_axi_rresp, s02_axi_rresp, s01_axi_rresp, s00_axi_rresp }),
    .s_axi_rlast({ s03_axi_rlast, s02_axi_rlast, s01_axi_rlast, s00_axi_rlast }),
    .s_axi_ruser({ s03_axi_ruser, s02_axi_ruser, s01_axi_ruser, s00_axi_ruser }),
    .s_axi_rvalid({ s03_axi_rvalid, s02_axi_rvalid, s01_axi_rvalid, s00_axi_rvalid }),
    .s_axi_rready({ s03_axi_rready, s02_axi_rready, s01_axi_rready, s00_axi_rready }),
    .m_axi_awid({ m00_axi_awid }),
    .m_axi_awaddr({ m00_axi_awaddr }),
    .m_axi_awlen({ m00_axi_awlen }),
    .m_axi_awsize({ m00_axi_awsize }),
    .m_axi_awburst({ m00_axi_awburst }),
    .m_axi_awlock({ m00_axi_awlock }),
    .m_axi_awcache({ m00_axi_awcache }),
    .m_axi_awprot({ m00_axi_awprot }),
    .m_axi_awqos({ m00_axi_awqos }),
    .m_axi_awregion({ m00_axi_awregion }),
    .m_axi_awuser({ m00_axi_awuser }),
    .m_axi_awvalid({ m00_axi_awvalid }),
    .m_axi_awready({ m00_axi_awready }),
    .m_axi_wdata({ m00_axi_wdata }),
    .m_axi_wstrb({ m00_axi_wstrb }),
    .m_axi_wlast({ m00_axi_wlast }),
    .m_axi_wuser({ m00_axi_wuser }),
    .m_axi_wvalid({ m00_axi_wvalid }),
    .m_axi_wready({ m00_axi_wready }),
    .m_axi_bid({ m00_axi_bid }),
    .m_axi_bresp({ m00_axi_bresp }),
    .m_axi_buser({ m00_axi_buser }),
    .m_axi_bvalid({ m00_axi_bvalid }),
    .m_axi_bready({ m00_axi_bready }),
    .m_axi_arid({ m00_axi_arid }),
    .m_axi_araddr({ m00_axi_araddr }),
    .m_axi_arlen({ m00_axi_arlen }),
    .m_axi_arsize({ m00_axi_arsize }),
    .m_axi_arburst({ m00_axi_arburst }),
    .m_axi_arlock({ m00_axi_arlock }),
    .m_axi_arcache({ m00_axi_arcache }),
    .m_axi_arprot({ m00_axi_arprot }),
    .m_axi_arqos({ m00_axi_arqos }),
    .m_axi_arregion({ m00_axi_arregion }),
    .m_axi_aruser({ m00_axi_aruser }),
    .m_axi_arvalid({ m00_axi_arvalid }),
    .m_axi_arready({ m00_axi_arready }),
    .m_axi_rid({ m00_axi_rid }),
    .m_axi_rdata({ m00_axi_rdata }),
    .m_axi_rresp({ m00_axi_rresp }),
    .m_axi_rlast({ m00_axi_rlast }),
    .m_axi_ruser({ m00_axi_ruser }),
    .m_axi_rvalid({ m00_axi_rvalid }),
    .m_axi_rready({ m00_axi_rready })
);

endmodule

`resetall
