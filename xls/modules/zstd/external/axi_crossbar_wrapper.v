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
 * AXI4 15x1 crossbar (wrapper)
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
    // Number of concurrent unique IDs
    parameter S04_THREADS = 2,
    // Number of concurrent operations
    parameter S04_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S05_THREADS = 2,
    // Number of concurrent operations
    parameter S05_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S06_THREADS = 2,
    // Number of concurrent operations
    parameter S06_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S07_THREADS = 2,
    // Number of concurrent operations
    parameter S07_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S08_THREADS = 2,
    // Number of concurrent operations
    parameter S08_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S09_THREADS = 2,
    // Number of concurrent operations
    parameter S09_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S10_THREADS = 2,
    // Number of concurrent operations
    parameter S10_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S11_THREADS = 2,
    // Number of concurrent operations
    parameter S11_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S12_THREADS = 2,
    // Number of concurrent operations
    parameter S12_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S13_THREADS = 2,
    // Number of concurrent operations
    parameter S13_ACCEPT = 16,
    // Number of concurrent unique IDs
    parameter S14_THREADS = 2,
    // Number of concurrent operations
    parameter S14_ACCEPT = 16,
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
    parameter M00_CONNECT_READ = 15'b111111111111111,
    // Write connections between interfaces
    // S_COUNT bits
    parameter M00_CONNECT_WRITE = 15'b111111111111111,
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
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S04_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S04_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S04_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S04_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S04_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S05_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S05_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S05_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S05_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S05_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S06_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S06_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S06_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S06_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S06_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S07_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S07_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S07_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S07_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S07_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S08_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S08_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S08_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S08_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S08_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S09_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S09_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S09_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S09_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S09_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S10_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S10_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S10_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S10_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S10_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S11_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S11_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S11_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S11_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S11_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S12_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S12_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S12_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S12_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S12_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S13_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S13_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S13_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S13_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S13_R_REG_TYPE = 2,
    // Slave interface AW channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S14_AW_REG_TYPE = 0,
    // Slave interface W channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S14_W_REG_TYPE = 0,
    // Slave interface B channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S14_B_REG_TYPE = 1,
    // Slave interface AR channel register type (input)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S14_AR_REG_TYPE = 0,
    // Slave interface R channel register type (output)
    // 0 to bypass, 1 for simple buffer, 2 for skid buffer
    parameter S14_R_REG_TYPE = 2,
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

    input  wire [S_ID_WIDTH-1:0]    s04_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s04_axi_awaddr,
    input  wire [7:0]               s04_axi_awlen,
    input  wire [2:0]               s04_axi_awsize,
    input  wire [1:0]               s04_axi_awburst,
    input  wire                     s04_axi_awlock,
    input  wire [3:0]               s04_axi_awcache,
    input  wire [2:0]               s04_axi_awprot,
    input  wire [3:0]               s04_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s04_axi_awuser,
    input  wire                     s04_axi_awvalid,
    output wire                     s04_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s04_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s04_axi_wstrb,
    input  wire                     s04_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s04_axi_wuser,
    input  wire                     s04_axi_wvalid,
    output wire                     s04_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s04_axi_bid,
    output wire [1:0]               s04_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s04_axi_buser,
    output wire                     s04_axi_bvalid,
    input  wire                     s04_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s04_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s04_axi_araddr,
    input  wire [7:0]               s04_axi_arlen,
    input  wire [2:0]               s04_axi_arsize,
    input  wire [1:0]               s04_axi_arburst,
    input  wire                     s04_axi_arlock,
    input  wire [3:0]               s04_axi_arcache,
    input  wire [2:0]               s04_axi_arprot,
    input  wire [3:0]               s04_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s04_axi_aruser,
    input  wire                     s04_axi_arvalid,
    output wire                     s04_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s04_axi_rid,
    output wire [DATA_WIDTH-1:0]    s04_axi_rdata,
    output wire [1:0]               s04_axi_rresp,
    output wire                     s04_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s04_axi_ruser,
    output wire                     s04_axi_rvalid,
    input  wire                     s04_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s05_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s05_axi_awaddr,
    input  wire [7:0]               s05_axi_awlen,
    input  wire [2:0]               s05_axi_awsize,
    input  wire [1:0]               s05_axi_awburst,
    input  wire                     s05_axi_awlock,
    input  wire [3:0]               s05_axi_awcache,
    input  wire [2:0]               s05_axi_awprot,
    input  wire [3:0]               s05_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s05_axi_awuser,
    input  wire                     s05_axi_awvalid,
    output wire                     s05_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s05_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s05_axi_wstrb,
    input  wire                     s05_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s05_axi_wuser,
    input  wire                     s05_axi_wvalid,
    output wire                     s05_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s05_axi_bid,
    output wire [1:0]               s05_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s05_axi_buser,
    output wire                     s05_axi_bvalid,
    input  wire                     s05_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s05_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s05_axi_araddr,
    input  wire [7:0]               s05_axi_arlen,
    input  wire [2:0]               s05_axi_arsize,
    input  wire [1:0]               s05_axi_arburst,
    input  wire                     s05_axi_arlock,
    input  wire [3:0]               s05_axi_arcache,
    input  wire [2:0]               s05_axi_arprot,
    input  wire [3:0]               s05_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s05_axi_aruser,
    input  wire                     s05_axi_arvalid,
    output wire                     s05_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s05_axi_rid,
    output wire [DATA_WIDTH-1:0]    s05_axi_rdata,
    output wire [1:0]               s05_axi_rresp,
    output wire                     s05_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s05_axi_ruser,
    output wire                     s05_axi_rvalid,
    input  wire                     s05_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s06_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s06_axi_awaddr,
    input  wire [7:0]               s06_axi_awlen,
    input  wire [2:0]               s06_axi_awsize,
    input  wire [1:0]               s06_axi_awburst,
    input  wire                     s06_axi_awlock,
    input  wire [3:0]               s06_axi_awcache,
    input  wire [2:0]               s06_axi_awprot,
    input  wire [3:0]               s06_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s06_axi_awuser,
    input  wire                     s06_axi_awvalid,
    output wire                     s06_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s06_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s06_axi_wstrb,
    input  wire                     s06_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s06_axi_wuser,
    input  wire                     s06_axi_wvalid,
    output wire                     s06_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s06_axi_bid,
    output wire [1:0]               s06_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s06_axi_buser,
    output wire                     s06_axi_bvalid,
    input  wire                     s06_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s06_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s06_axi_araddr,
    input  wire [7:0]               s06_axi_arlen,
    input  wire [2:0]               s06_axi_arsize,
    input  wire [1:0]               s06_axi_arburst,
    input  wire                     s06_axi_arlock,
    input  wire [3:0]               s06_axi_arcache,
    input  wire [2:0]               s06_axi_arprot,
    input  wire [3:0]               s06_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s06_axi_aruser,
    input  wire                     s06_axi_arvalid,
    output wire                     s06_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s06_axi_rid,
    output wire [DATA_WIDTH-1:0]    s06_axi_rdata,
    output wire [1:0]               s06_axi_rresp,
    output wire                     s06_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s06_axi_ruser,
    output wire                     s06_axi_rvalid,
    input  wire                     s06_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s07_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s07_axi_awaddr,
    input  wire [7:0]               s07_axi_awlen,
    input  wire [2:0]               s07_axi_awsize,
    input  wire [1:0]               s07_axi_awburst,
    input  wire                     s07_axi_awlock,
    input  wire [3:0]               s07_axi_awcache,
    input  wire [2:0]               s07_axi_awprot,
    input  wire [3:0]               s07_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s07_axi_awuser,
    input  wire                     s07_axi_awvalid,
    output wire                     s07_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s07_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s07_axi_wstrb,
    input  wire                     s07_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s07_axi_wuser,
    input  wire                     s07_axi_wvalid,
    output wire                     s07_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s07_axi_bid,
    output wire [1:0]               s07_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s07_axi_buser,
    output wire                     s07_axi_bvalid,
    input  wire                     s07_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s07_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s07_axi_araddr,
    input  wire [7:0]               s07_axi_arlen,
    input  wire [2:0]               s07_axi_arsize,
    input  wire [1:0]               s07_axi_arburst,
    input  wire                     s07_axi_arlock,
    input  wire [3:0]               s07_axi_arcache,
    input  wire [2:0]               s07_axi_arprot,
    input  wire [3:0]               s07_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s07_axi_aruser,
    input  wire                     s07_axi_arvalid,
    output wire                     s07_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s07_axi_rid,
    output wire [DATA_WIDTH-1:0]    s07_axi_rdata,
    output wire [1:0]               s07_axi_rresp,
    output wire                     s07_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s07_axi_ruser,
    output wire                     s07_axi_rvalid,
    input  wire                     s07_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s08_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s08_axi_awaddr,
    input  wire [7:0]               s08_axi_awlen,
    input  wire [2:0]               s08_axi_awsize,
    input  wire [1:0]               s08_axi_awburst,
    input  wire                     s08_axi_awlock,
    input  wire [3:0]               s08_axi_awcache,
    input  wire [2:0]               s08_axi_awprot,
    input  wire [3:0]               s08_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s08_axi_awuser,
    input  wire                     s08_axi_awvalid,
    output wire                     s08_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s08_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s08_axi_wstrb,
    input  wire                     s08_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s08_axi_wuser,
    input  wire                     s08_axi_wvalid,
    output wire                     s08_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s08_axi_bid,
    output wire [1:0]               s08_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s08_axi_buser,
    output wire                     s08_axi_bvalid,
    input  wire                     s08_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s08_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s08_axi_araddr,
    input  wire [7:0]               s08_axi_arlen,
    input  wire [2:0]               s08_axi_arsize,
    input  wire [1:0]               s08_axi_arburst,
    input  wire                     s08_axi_arlock,
    input  wire [3:0]               s08_axi_arcache,
    input  wire [2:0]               s08_axi_arprot,
    input  wire [3:0]               s08_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s08_axi_aruser,
    input  wire                     s08_axi_arvalid,
    output wire                     s08_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s08_axi_rid,
    output wire [DATA_WIDTH-1:0]    s08_axi_rdata,
    output wire [1:0]               s08_axi_rresp,
    output wire                     s08_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s08_axi_ruser,
    output wire                     s08_axi_rvalid,
    input  wire                     s08_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s09_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s09_axi_awaddr,
    input  wire [7:0]               s09_axi_awlen,
    input  wire [2:0]               s09_axi_awsize,
    input  wire [1:0]               s09_axi_awburst,
    input  wire                     s09_axi_awlock,
    input  wire [3:0]               s09_axi_awcache,
    input  wire [2:0]               s09_axi_awprot,
    input  wire [3:0]               s09_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s09_axi_awuser,
    input  wire                     s09_axi_awvalid,
    output wire                     s09_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s09_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s09_axi_wstrb,
    input  wire                     s09_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s09_axi_wuser,
    input  wire                     s09_axi_wvalid,
    output wire                     s09_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s09_axi_bid,
    output wire [1:0]               s09_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s09_axi_buser,
    output wire                     s09_axi_bvalid,
    input  wire                     s09_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s09_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s09_axi_araddr,
    input  wire [7:0]               s09_axi_arlen,
    input  wire [2:0]               s09_axi_arsize,
    input  wire [1:0]               s09_axi_arburst,
    input  wire                     s09_axi_arlock,
    input  wire [3:0]               s09_axi_arcache,
    input  wire [2:0]               s09_axi_arprot,
    input  wire [3:0]               s09_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s09_axi_aruser,
    input  wire                     s09_axi_arvalid,
    output wire                     s09_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s09_axi_rid,
    output wire [DATA_WIDTH-1:0]    s09_axi_rdata,
    output wire [1:0]               s09_axi_rresp,
    output wire                     s09_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s09_axi_ruser,
    output wire                     s09_axi_rvalid,
    input  wire                     s09_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s10_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s10_axi_awaddr,
    input  wire [7:0]               s10_axi_awlen,
    input  wire [2:0]               s10_axi_awsize,
    input  wire [1:0]               s10_axi_awburst,
    input  wire                     s10_axi_awlock,
    input  wire [3:0]               s10_axi_awcache,
    input  wire [2:0]               s10_axi_awprot,
    input  wire [3:0]               s10_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s10_axi_awuser,
    input  wire                     s10_axi_awvalid,
    output wire                     s10_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s10_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s10_axi_wstrb,
    input  wire                     s10_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s10_axi_wuser,
    input  wire                     s10_axi_wvalid,
    output wire                     s10_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s10_axi_bid,
    output wire [1:0]               s10_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s10_axi_buser,
    output wire                     s10_axi_bvalid,
    input  wire                     s10_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s10_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s10_axi_araddr,
    input  wire [7:0]               s10_axi_arlen,
    input  wire [2:0]               s10_axi_arsize,
    input  wire [1:0]               s10_axi_arburst,
    input  wire                     s10_axi_arlock,
    input  wire [3:0]               s10_axi_arcache,
    input  wire [2:0]               s10_axi_arprot,
    input  wire [3:0]               s10_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s10_axi_aruser,
    input  wire                     s10_axi_arvalid,
    output wire                     s10_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s10_axi_rid,
    output wire [DATA_WIDTH-1:0]    s10_axi_rdata,
    output wire [1:0]               s10_axi_rresp,
    output wire                     s10_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s10_axi_ruser,
    output wire                     s10_axi_rvalid,
    input  wire                     s10_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s11_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s11_axi_awaddr,
    input  wire [7:0]               s11_axi_awlen,
    input  wire [2:0]               s11_axi_awsize,
    input  wire [1:0]               s11_axi_awburst,
    input  wire                     s11_axi_awlock,
    input  wire [3:0]               s11_axi_awcache,
    input  wire [2:0]               s11_axi_awprot,
    input  wire [3:0]               s11_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s11_axi_awuser,
    input  wire                     s11_axi_awvalid,
    output wire                     s11_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s11_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s11_axi_wstrb,
    input  wire                     s11_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s11_axi_wuser,
    input  wire                     s11_axi_wvalid,
    output wire                     s11_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s11_axi_bid,
    output wire [1:0]               s11_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s11_axi_buser,
    output wire                     s11_axi_bvalid,
    input  wire                     s11_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s11_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s11_axi_araddr,
    input  wire [7:0]               s11_axi_arlen,
    input  wire [2:0]               s11_axi_arsize,
    input  wire [1:0]               s11_axi_arburst,
    input  wire                     s11_axi_arlock,
    input  wire [3:0]               s11_axi_arcache,
    input  wire [2:0]               s11_axi_arprot,
    input  wire [3:0]               s11_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s11_axi_aruser,
    input  wire                     s11_axi_arvalid,
    output wire                     s11_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s11_axi_rid,
    output wire [DATA_WIDTH-1:0]    s11_axi_rdata,
    output wire [1:0]               s11_axi_rresp,
    output wire                     s11_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s11_axi_ruser,
    output wire                     s11_axi_rvalid,
    input  wire                     s11_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s12_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s12_axi_awaddr,
    input  wire [7:0]               s12_axi_awlen,
    input  wire [2:0]               s12_axi_awsize,
    input  wire [1:0]               s12_axi_awburst,
    input  wire                     s12_axi_awlock,
    input  wire [3:0]               s12_axi_awcache,
    input  wire [2:0]               s12_axi_awprot,
    input  wire [3:0]               s12_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s12_axi_awuser,
    input  wire                     s12_axi_awvalid,
    output wire                     s12_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s12_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s12_axi_wstrb,
    input  wire                     s12_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s12_axi_wuser,
    input  wire                     s12_axi_wvalid,
    output wire                     s12_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s12_axi_bid,
    output wire [1:0]               s12_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s12_axi_buser,
    output wire                     s12_axi_bvalid,
    input  wire                     s12_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s12_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s12_axi_araddr,
    input  wire [7:0]               s12_axi_arlen,
    input  wire [2:0]               s12_axi_arsize,
    input  wire [1:0]               s12_axi_arburst,
    input  wire                     s12_axi_arlock,
    input  wire [3:0]               s12_axi_arcache,
    input  wire [2:0]               s12_axi_arprot,
    input  wire [3:0]               s12_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s12_axi_aruser,
    input  wire                     s12_axi_arvalid,
    output wire                     s12_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s12_axi_rid,
    output wire [DATA_WIDTH-1:0]    s12_axi_rdata,
    output wire [1:0]               s12_axi_rresp,
    output wire                     s12_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s12_axi_ruser,
    output wire                     s12_axi_rvalid,
    input  wire                     s12_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s13_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s13_axi_awaddr,
    input  wire [7:0]               s13_axi_awlen,
    input  wire [2:0]               s13_axi_awsize,
    input  wire [1:0]               s13_axi_awburst,
    input  wire                     s13_axi_awlock,
    input  wire [3:0]               s13_axi_awcache,
    input  wire [2:0]               s13_axi_awprot,
    input  wire [3:0]               s13_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s13_axi_awuser,
    input  wire                     s13_axi_awvalid,
    output wire                     s13_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s13_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s13_axi_wstrb,
    input  wire                     s13_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s13_axi_wuser,
    input  wire                     s13_axi_wvalid,
    output wire                     s13_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s13_axi_bid,
    output wire [1:0]               s13_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s13_axi_buser,
    output wire                     s13_axi_bvalid,
    input  wire                     s13_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s13_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s13_axi_araddr,
    input  wire [7:0]               s13_axi_arlen,
    input  wire [2:0]               s13_axi_arsize,
    input  wire [1:0]               s13_axi_arburst,
    input  wire                     s13_axi_arlock,
    input  wire [3:0]               s13_axi_arcache,
    input  wire [2:0]               s13_axi_arprot,
    input  wire [3:0]               s13_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s13_axi_aruser,
    input  wire                     s13_axi_arvalid,
    output wire                     s13_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s13_axi_rid,
    output wire [DATA_WIDTH-1:0]    s13_axi_rdata,
    output wire [1:0]               s13_axi_rresp,
    output wire                     s13_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s13_axi_ruser,
    output wire                     s13_axi_rvalid,
    input  wire                     s13_axi_rready,

    input  wire [S_ID_WIDTH-1:0]    s14_axi_awid,
    input  wire [ADDR_WIDTH-1:0]    s14_axi_awaddr,
    input  wire [7:0]               s14_axi_awlen,
    input  wire [2:0]               s14_axi_awsize,
    input  wire [1:0]               s14_axi_awburst,
    input  wire                     s14_axi_awlock,
    input  wire [3:0]               s14_axi_awcache,
    input  wire [2:0]               s14_axi_awprot,
    input  wire [3:0]               s14_axi_awqos,
    input  wire [AWUSER_WIDTH-1:0]  s14_axi_awuser,
    input  wire                     s14_axi_awvalid,
    output wire                     s14_axi_awready,
    input  wire [DATA_WIDTH-1:0]    s14_axi_wdata,
    input  wire [STRB_WIDTH-1:0]    s14_axi_wstrb,
    input  wire                     s14_axi_wlast,
    input  wire [WUSER_WIDTH-1:0]   s14_axi_wuser,
    input  wire                     s14_axi_wvalid,
    output wire                     s14_axi_wready,
    output wire [S_ID_WIDTH-1:0]    s14_axi_bid,
    output wire [1:0]               s14_axi_bresp,
    output wire [BUSER_WIDTH-1:0]   s14_axi_buser,
    output wire                     s14_axi_bvalid,
    input  wire                     s14_axi_bready,
    input  wire [S_ID_WIDTH-1:0]    s14_axi_arid,
    input  wire [ADDR_WIDTH-1:0]    s14_axi_araddr,
    input  wire [7:0]               s14_axi_arlen,
    input  wire [2:0]               s14_axi_arsize,
    input  wire [1:0]               s14_axi_arburst,
    input  wire                     s14_axi_arlock,
    input  wire [3:0]               s14_axi_arcache,
    input  wire [2:0]               s14_axi_arprot,
    input  wire [3:0]               s14_axi_arqos,
    input  wire [ARUSER_WIDTH-1:0]  s14_axi_aruser,
    input  wire                     s14_axi_arvalid,
    output wire                     s14_axi_arready,
    output wire [S_ID_WIDTH-1:0]    s14_axi_rid,
    output wire [DATA_WIDTH-1:0]    s14_axi_rdata,
    output wire [1:0]               s14_axi_rresp,
    output wire                     s14_axi_rlast,
    output wire [RUSER_WIDTH-1:0]   s14_axi_ruser,
    output wire                     s14_axi_rvalid,
    input  wire                     s14_axi_rready,

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

localparam S_COUNT = 15;
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
    .S_THREADS({ w_32(S14_THREADS), w_32(S13_THREADS), w_32(S12_THREADS), w_32(S11_THREADS), w_32(S10_THREADS), w_32(S09_THREADS), w_32(S08_THREADS), w_32(S07_THREADS), w_32(S06_THREADS), w_32(S05_THREADS), w_32(S04_THREADS), w_32(S03_THREADS), w_32(S02_THREADS), w_32(S01_THREADS), w_32(S00_THREADS) }),
    .S_ACCEPT({ w_32(S14_ACCEPT), w_32(S13_ACCEPT), w_32(S12_ACCEPT), w_32(S11_ACCEPT), w_32(S10_ACCEPT), w_32(S09_ACCEPT), w_32(S08_ACCEPT), w_32(S07_ACCEPT), w_32(S06_ACCEPT), w_32(S05_ACCEPT), w_32(S04_ACCEPT), w_32(S03_ACCEPT), w_32(S02_ACCEPT), w_32(S01_ACCEPT), w_32(S00_ACCEPT) }),
    .M_REGIONS(M_REGIONS),
    .M_BASE_ADDR({ w_a_r(M00_BASE_ADDR) }),
    .M_ADDR_WIDTH({ w_32_r(M00_ADDR_WIDTH) }),
    .M_CONNECT_READ({ w_s(M00_CONNECT_READ) }),
    .M_CONNECT_WRITE({ w_s(M00_CONNECT_WRITE) }),
    .M_ISSUE({ w_32(M00_ISSUE) }),
    .M_SECURE({ w_1(M00_SECURE) }),
    .S_AR_REG_TYPE({ w_2(S14_AR_REG_TYPE), w_2(S13_AR_REG_TYPE), w_2(S12_AR_REG_TYPE), w_2(S11_AR_REG_TYPE), w_2(S10_AR_REG_TYPE), w_2(S09_AR_REG_TYPE), w_2(S08_AR_REG_TYPE), w_2(S07_AR_REG_TYPE), w_2(S06_AR_REG_TYPE), w_2(S05_AR_REG_TYPE), w_2(S04_AR_REG_TYPE), w_2(S03_AR_REG_TYPE), w_2(S02_AR_REG_TYPE), w_2(S01_AR_REG_TYPE), w_2(S00_AR_REG_TYPE) }),
    .S_R_REG_TYPE({ w_2(S14_R_REG_TYPE), w_2(S13_R_REG_TYPE), w_2(S12_R_REG_TYPE), w_2(S11_R_REG_TYPE), w_2(S10_R_REG_TYPE), w_2(S09_R_REG_TYPE), w_2(S08_R_REG_TYPE), w_2(S07_R_REG_TYPE), w_2(S06_R_REG_TYPE), w_2(S05_R_REG_TYPE), w_2(S04_R_REG_TYPE), w_2(S03_R_REG_TYPE), w_2(S02_R_REG_TYPE), w_2(S01_R_REG_TYPE), w_2(S00_R_REG_TYPE) }),
    .S_AW_REG_TYPE({ w_2(S14_AW_REG_TYPE), w_2(S13_AW_REG_TYPE), w_2(S12_AW_REG_TYPE), w_2(S11_AW_REG_TYPE), w_2(S10_AW_REG_TYPE), w_2(S09_AW_REG_TYPE), w_2(S08_AW_REG_TYPE), w_2(S07_AW_REG_TYPE), w_2(S06_AW_REG_TYPE), w_2(S05_AW_REG_TYPE), w_2(S04_AW_REG_TYPE), w_2(S03_AW_REG_TYPE), w_2(S02_AW_REG_TYPE), w_2(S01_AW_REG_TYPE), w_2(S00_AW_REG_TYPE) }),
    .S_W_REG_TYPE({ w_2(S14_W_REG_TYPE), w_2(S13_W_REG_TYPE), w_2(S12_W_REG_TYPE), w_2(S11_W_REG_TYPE), w_2(S10_W_REG_TYPE), w_2(S09_W_REG_TYPE), w_2(S08_W_REG_TYPE), w_2(S07_W_REG_TYPE), w_2(S06_W_REG_TYPE), w_2(S05_W_REG_TYPE), w_2(S04_W_REG_TYPE), w_2(S03_W_REG_TYPE), w_2(S02_W_REG_TYPE), w_2(S01_W_REG_TYPE), w_2(S00_W_REG_TYPE) }),
    .S_B_REG_TYPE({ w_2(S14_B_REG_TYPE), w_2(S13_B_REG_TYPE), w_2(S12_B_REG_TYPE), w_2(S11_B_REG_TYPE), w_2(S10_B_REG_TYPE), w_2(S09_B_REG_TYPE), w_2(S08_B_REG_TYPE), w_2(S07_B_REG_TYPE), w_2(S06_B_REG_TYPE), w_2(S05_B_REG_TYPE), w_2(S04_B_REG_TYPE), w_2(S03_B_REG_TYPE), w_2(S02_B_REG_TYPE), w_2(S01_B_REG_TYPE), w_2(S00_B_REG_TYPE) }),
    .M_AR_REG_TYPE({ w_2(M00_AR_REG_TYPE) }),
    .M_R_REG_TYPE({ w_2(M00_R_REG_TYPE) }),
    .M_AW_REG_TYPE({ w_2(M00_AW_REG_TYPE) }),
    .M_W_REG_TYPE({ w_2(M00_W_REG_TYPE) }),
    .M_B_REG_TYPE({ w_2(M00_B_REG_TYPE) })
)
axi_crossbar_inst (
    .clk(clk),
    .rst(rst),
    .s_axi_awid({ s14_axi_awid, s13_axi_awid, s12_axi_awid, s11_axi_awid, s10_axi_awid, s09_axi_awid, s08_axi_awid, s07_axi_awid, s06_axi_awid, s05_axi_awid, s04_axi_awid, s03_axi_awid, s02_axi_awid, s01_axi_awid, s00_axi_awid }),
    .s_axi_awaddr({ s14_axi_awaddr, s13_axi_awaddr, s12_axi_awaddr, s11_axi_awaddr, s10_axi_awaddr, s09_axi_awaddr, s08_axi_awaddr, s07_axi_awaddr, s06_axi_awaddr, s05_axi_awaddr, s04_axi_awaddr, s03_axi_awaddr, s02_axi_awaddr, s01_axi_awaddr, s00_axi_awaddr }),
    .s_axi_awlen({ s14_axi_awlen, s13_axi_awlen, s12_axi_awlen, s11_axi_awlen, s10_axi_awlen, s09_axi_awlen, s08_axi_awlen, s07_axi_awlen, s06_axi_awlen, s05_axi_awlen, s04_axi_awlen, s03_axi_awlen, s02_axi_awlen, s01_axi_awlen, s00_axi_awlen }),
    .s_axi_awsize({ s14_axi_awsize, s13_axi_awsize, s12_axi_awsize, s11_axi_awsize, s10_axi_awsize, s09_axi_awsize, s08_axi_awsize, s07_axi_awsize, s06_axi_awsize, s05_axi_awsize, s04_axi_awsize, s03_axi_awsize, s02_axi_awsize, s01_axi_awsize, s00_axi_awsize }),
    .s_axi_awburst({ s14_axi_awburst, s13_axi_awburst, s12_axi_awburst, s11_axi_awburst, s10_axi_awburst, s09_axi_awburst, s08_axi_awburst, s07_axi_awburst, s06_axi_awburst, s05_axi_awburst, s04_axi_awburst, s03_axi_awburst, s02_axi_awburst, s01_axi_awburst, s00_axi_awburst }),
    .s_axi_awlock({ s14_axi_awlock, s13_axi_awlock, s12_axi_awlock, s11_axi_awlock, s10_axi_awlock, s09_axi_awlock, s08_axi_awlock, s07_axi_awlock, s06_axi_awlock, s05_axi_awlock, s04_axi_awlock, s03_axi_awlock, s02_axi_awlock, s01_axi_awlock, s00_axi_awlock }),
    .s_axi_awcache({ s14_axi_awcache, s13_axi_awcache, s12_axi_awcache, s11_axi_awcache, s10_axi_awcache, s09_axi_awcache, s08_axi_awcache, s07_axi_awcache, s06_axi_awcache, s05_axi_awcache, s04_axi_awcache, s03_axi_awcache, s02_axi_awcache, s01_axi_awcache, s00_axi_awcache }),
    .s_axi_awprot({ s14_axi_awprot, s13_axi_awprot, s12_axi_awprot, s11_axi_awprot, s10_axi_awprot, s09_axi_awprot, s08_axi_awprot, s07_axi_awprot, s06_axi_awprot, s05_axi_awprot, s04_axi_awprot, s03_axi_awprot, s02_axi_awprot, s01_axi_awprot, s00_axi_awprot }),
    .s_axi_awqos({ s14_axi_awqos, s13_axi_awqos, s12_axi_awqos, s11_axi_awqos, s10_axi_awqos, s09_axi_awqos, s08_axi_awqos, s07_axi_awqos, s06_axi_awqos, s05_axi_awqos, s04_axi_awqos, s03_axi_awqos, s02_axi_awqos, s01_axi_awqos, s00_axi_awqos }),
    .s_axi_awuser({ s14_axi_awuser, s13_axi_awuser, s12_axi_awuser, s11_axi_awuser, s10_axi_awuser, s09_axi_awuser, s08_axi_awuser, s07_axi_awuser, s06_axi_awuser, s05_axi_awuser, s04_axi_awuser, s03_axi_awuser, s02_axi_awuser, s01_axi_awuser, s00_axi_awuser }),
    .s_axi_awvalid({ s14_axi_awvalid, s13_axi_awvalid, s12_axi_awvalid, s11_axi_awvalid, s10_axi_awvalid, s09_axi_awvalid, s08_axi_awvalid, s07_axi_awvalid, s06_axi_awvalid, s05_axi_awvalid, s04_axi_awvalid, s03_axi_awvalid, s02_axi_awvalid, s01_axi_awvalid, s00_axi_awvalid }),
    .s_axi_awready({ s14_axi_awready, s13_axi_awready, s12_axi_awready, s11_axi_awready, s10_axi_awready, s09_axi_awready, s08_axi_awready, s07_axi_awready, s06_axi_awready, s05_axi_awready, s04_axi_awready, s03_axi_awready, s02_axi_awready, s01_axi_awready, s00_axi_awready }),
    .s_axi_wdata({ s14_axi_wdata, s13_axi_wdata, s12_axi_wdata, s11_axi_wdata, s10_axi_wdata, s09_axi_wdata, s08_axi_wdata, s07_axi_wdata, s06_axi_wdata, s05_axi_wdata, s04_axi_wdata, s03_axi_wdata, s02_axi_wdata, s01_axi_wdata, s00_axi_wdata }),
    .s_axi_wstrb({ s14_axi_wstrb, s13_axi_wstrb, s12_axi_wstrb, s11_axi_wstrb, s10_axi_wstrb, s09_axi_wstrb, s08_axi_wstrb, s07_axi_wstrb, s06_axi_wstrb, s05_axi_wstrb, s04_axi_wstrb, s03_axi_wstrb, s02_axi_wstrb, s01_axi_wstrb, s00_axi_wstrb }),
    .s_axi_wlast({ s14_axi_wlast, s13_axi_wlast, s12_axi_wlast, s11_axi_wlast, s10_axi_wlast, s09_axi_wlast, s08_axi_wlast, s07_axi_wlast, s06_axi_wlast, s05_axi_wlast, s04_axi_wlast, s03_axi_wlast, s02_axi_wlast, s01_axi_wlast, s00_axi_wlast }),
    .s_axi_wuser({ s14_axi_wuser, s13_axi_wuser, s12_axi_wuser, s11_axi_wuser, s10_axi_wuser, s09_axi_wuser, s08_axi_wuser, s07_axi_wuser, s06_axi_wuser, s05_axi_wuser, s04_axi_wuser, s03_axi_wuser, s02_axi_wuser, s01_axi_wuser, s00_axi_wuser }),
    .s_axi_wvalid({ s14_axi_wvalid, s13_axi_wvalid, s12_axi_wvalid, s11_axi_wvalid, s10_axi_wvalid, s09_axi_wvalid, s08_axi_wvalid, s07_axi_wvalid, s06_axi_wvalid, s05_axi_wvalid, s04_axi_wvalid, s03_axi_wvalid, s02_axi_wvalid, s01_axi_wvalid, s00_axi_wvalid }),
    .s_axi_wready({ s14_axi_wready, s13_axi_wready, s12_axi_wready, s11_axi_wready, s10_axi_wready, s09_axi_wready, s08_axi_wready, s07_axi_wready, s06_axi_wready, s05_axi_wready, s04_axi_wready, s03_axi_wready, s02_axi_wready, s01_axi_wready, s00_axi_wready }),
    .s_axi_bid({ s14_axi_bid, s13_axi_bid, s12_axi_bid, s11_axi_bid, s10_axi_bid, s09_axi_bid, s08_axi_bid, s07_axi_bid, s06_axi_bid, s05_axi_bid, s04_axi_bid, s03_axi_bid, s02_axi_bid, s01_axi_bid, s00_axi_bid }),
    .s_axi_bresp({ s14_axi_bresp, s13_axi_bresp, s12_axi_bresp, s11_axi_bresp, s10_axi_bresp, s09_axi_bresp, s08_axi_bresp, s07_axi_bresp, s06_axi_bresp, s05_axi_bresp, s04_axi_bresp, s03_axi_bresp, s02_axi_bresp, s01_axi_bresp, s00_axi_bresp }),
    .s_axi_buser({ s14_axi_buser, s13_axi_buser, s12_axi_buser, s11_axi_buser, s10_axi_buser, s09_axi_buser, s08_axi_buser, s07_axi_buser, s06_axi_buser, s05_axi_buser, s04_axi_buser, s03_axi_buser, s02_axi_buser, s01_axi_buser, s00_axi_buser }),
    .s_axi_bvalid({ s14_axi_bvalid, s13_axi_bvalid, s12_axi_bvalid, s11_axi_bvalid, s10_axi_bvalid, s09_axi_bvalid, s08_axi_bvalid, s07_axi_bvalid, s06_axi_bvalid, s05_axi_bvalid, s04_axi_bvalid, s03_axi_bvalid, s02_axi_bvalid, s01_axi_bvalid, s00_axi_bvalid }),
    .s_axi_bready({ s14_axi_bready, s13_axi_bready, s12_axi_bready, s11_axi_bready, s10_axi_bready, s09_axi_bready, s08_axi_bready, s07_axi_bready, s06_axi_bready, s05_axi_bready, s04_axi_bready, s03_axi_bready, s02_axi_bready, s01_axi_bready, s00_axi_bready }),
    .s_axi_arid({ s14_axi_arid, s13_axi_arid, s12_axi_arid, s11_axi_arid, s10_axi_arid, s09_axi_arid, s08_axi_arid, s07_axi_arid, s06_axi_arid, s05_axi_arid, s04_axi_arid, s03_axi_arid, s02_axi_arid, s01_axi_arid, s00_axi_arid }),
    .s_axi_araddr({ s14_axi_araddr, s13_axi_araddr, s12_axi_araddr, s11_axi_araddr, s10_axi_araddr, s09_axi_araddr, s08_axi_araddr, s07_axi_araddr, s06_axi_araddr, s05_axi_araddr, s04_axi_araddr, s03_axi_araddr, s02_axi_araddr, s01_axi_araddr, s00_axi_araddr }),
    .s_axi_arlen({ s14_axi_arlen, s13_axi_arlen, s12_axi_arlen, s11_axi_arlen, s10_axi_arlen, s09_axi_arlen, s08_axi_arlen, s07_axi_arlen, s06_axi_arlen, s05_axi_arlen, s04_axi_arlen, s03_axi_arlen, s02_axi_arlen, s01_axi_arlen, s00_axi_arlen }),
    .s_axi_arsize({ s14_axi_arsize, s13_axi_arsize, s12_axi_arsize, s11_axi_arsize, s10_axi_arsize, s09_axi_arsize, s08_axi_arsize, s07_axi_arsize, s06_axi_arsize, s05_axi_arsize, s04_axi_arsize, s03_axi_arsize, s02_axi_arsize, s01_axi_arsize, s00_axi_arsize }),
    .s_axi_arburst({ s14_axi_arburst, s13_axi_arburst, s12_axi_arburst, s11_axi_arburst, s10_axi_arburst, s09_axi_arburst, s08_axi_arburst, s07_axi_arburst, s06_axi_arburst, s05_axi_arburst, s04_axi_arburst, s03_axi_arburst, s02_axi_arburst, s01_axi_arburst, s00_axi_arburst }),
    .s_axi_arlock({ s14_axi_arlock, s13_axi_arlock, s12_axi_arlock, s11_axi_arlock, s10_axi_arlock, s09_axi_arlock, s08_axi_arlock, s07_axi_arlock, s06_axi_arlock, s05_axi_arlock, s04_axi_arlock, s03_axi_arlock, s02_axi_arlock, s01_axi_arlock, s00_axi_arlock }),
    .s_axi_arcache({ s14_axi_arcache, s13_axi_arcache, s12_axi_arcache, s11_axi_arcache, s10_axi_arcache, s09_axi_arcache, s08_axi_arcache, s07_axi_arcache, s06_axi_arcache, s05_axi_arcache, s04_axi_arcache, s03_axi_arcache, s02_axi_arcache, s01_axi_arcache, s00_axi_arcache }),
    .s_axi_arprot({ s14_axi_arprot, s13_axi_arprot, s12_axi_arprot, s11_axi_arprot, s10_axi_arprot, s09_axi_arprot, s08_axi_arprot, s07_axi_arprot, s06_axi_arprot, s05_axi_arprot, s04_axi_arprot, s03_axi_arprot, s02_axi_arprot, s01_axi_arprot, s00_axi_arprot }),
    .s_axi_arqos({ s14_axi_arqos, s13_axi_arqos, s12_axi_arqos, s11_axi_arqos, s10_axi_arqos, s09_axi_arqos, s08_axi_arqos, s07_axi_arqos, s06_axi_arqos, s05_axi_arqos, s04_axi_arqos, s03_axi_arqos, s02_axi_arqos, s01_axi_arqos, s00_axi_arqos }),
    .s_axi_aruser({ s14_axi_aruser, s13_axi_aruser, s12_axi_aruser, s11_axi_aruser, s10_axi_aruser, s09_axi_aruser, s08_axi_aruser, s07_axi_aruser, s06_axi_aruser, s05_axi_aruser, s04_axi_aruser, s03_axi_aruser, s02_axi_aruser, s01_axi_aruser, s00_axi_aruser }),
    .s_axi_arvalid({ s14_axi_arvalid, s13_axi_arvalid, s12_axi_arvalid, s11_axi_arvalid, s10_axi_arvalid, s09_axi_arvalid, s08_axi_arvalid, s07_axi_arvalid, s06_axi_arvalid, s05_axi_arvalid, s04_axi_arvalid, s03_axi_arvalid, s02_axi_arvalid, s01_axi_arvalid, s00_axi_arvalid }),
    .s_axi_arready({ s14_axi_arready, s13_axi_arready, s12_axi_arready, s11_axi_arready, s10_axi_arready, s09_axi_arready, s08_axi_arready, s07_axi_arready, s06_axi_arready, s05_axi_arready, s04_axi_arready, s03_axi_arready, s02_axi_arready, s01_axi_arready, s00_axi_arready }),
    .s_axi_rid({ s14_axi_rid, s13_axi_rid, s12_axi_rid, s11_axi_rid, s10_axi_rid, s09_axi_rid, s08_axi_rid, s07_axi_rid, s06_axi_rid, s05_axi_rid, s04_axi_rid, s03_axi_rid, s02_axi_rid, s01_axi_rid, s00_axi_rid }),
    .s_axi_rdata({ s14_axi_rdata, s13_axi_rdata, s12_axi_rdata, s11_axi_rdata, s10_axi_rdata, s09_axi_rdata, s08_axi_rdata, s07_axi_rdata, s06_axi_rdata, s05_axi_rdata, s04_axi_rdata, s03_axi_rdata, s02_axi_rdata, s01_axi_rdata, s00_axi_rdata }),
    .s_axi_rresp({ s14_axi_rresp, s13_axi_rresp, s12_axi_rresp, s11_axi_rresp, s10_axi_rresp, s09_axi_rresp, s08_axi_rresp, s07_axi_rresp, s06_axi_rresp, s05_axi_rresp, s04_axi_rresp, s03_axi_rresp, s02_axi_rresp, s01_axi_rresp, s00_axi_rresp }),
    .s_axi_rlast({ s14_axi_rlast, s13_axi_rlast, s12_axi_rlast, s11_axi_rlast, s10_axi_rlast, s09_axi_rlast, s08_axi_rlast, s07_axi_rlast, s06_axi_rlast, s05_axi_rlast, s04_axi_rlast, s03_axi_rlast, s02_axi_rlast, s01_axi_rlast, s00_axi_rlast }),
    .s_axi_ruser({ s14_axi_ruser, s13_axi_ruser, s12_axi_ruser, s11_axi_ruser, s10_axi_ruser, s09_axi_ruser, s08_axi_ruser, s07_axi_ruser, s06_axi_ruser, s05_axi_ruser, s04_axi_ruser, s03_axi_ruser, s02_axi_ruser, s01_axi_ruser, s00_axi_ruser }),
    .s_axi_rvalid({ s14_axi_rvalid, s13_axi_rvalid, s12_axi_rvalid, s11_axi_rvalid, s10_axi_rvalid, s09_axi_rvalid, s08_axi_rvalid, s07_axi_rvalid, s06_axi_rvalid, s05_axi_rvalid, s04_axi_rvalid, s03_axi_rvalid, s02_axi_rvalid, s01_axi_rvalid, s00_axi_rvalid }),
    .s_axi_rready({ s14_axi_rready, s13_axi_rready, s12_axi_rready, s11_axi_rready, s10_axi_rready, s09_axi_rready, s08_axi_rready, s07_axi_rready, s06_axi_rready, s05_axi_rready, s04_axi_rready, s03_axi_rready, s02_axi_rready, s01_axi_rready, s00_axi_rready }),
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
