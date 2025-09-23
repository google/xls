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

module ram_1r1w # (
    parameter DATA_WIDTH = 4,
    parameter SIZE = 32,
    parameter NUM_PARTITIONS = 1,
    parameter ADDR_WIDTH = $clog(SIZE),
    parameter INIT_FILE = ""
) (
    input  wire clk,
    input  wire rst,

    input  wire [ DATA_WIDTH     -1:0 ]   wr_data,
    input  wire [ ADDR_WIDTH     -1:0 ]   wr_addr,
    input  wire                           wr_en,
    input  wire [ NUM_PARTITIONS     -1:0 ]   wr_mask,

    output reg  [ DATA_WIDTH     -1:0 ]   rd_data,
    input  wire [ ADDR_WIDTH     -1:0 ]   rd_addr,
    input  wire                           rd_en,
    input  wire [ NUM_PARTITIONS -1:0 ]   rd_mask
);

localparam PARTITION_WIDTH = (DATA_WIDTH / NUM_PARTITIONS);

reg [DATA_WIDTH-1:0] mem [SIZE];

integer i;

always @(rst) begin
    if (INIT_FILE != "") begin
          $readmemh(INIT_FILE, mem);
    end
    else begin
        for (i = 0; i < SIZE; i++) begin
            mem[i] <= {DATA_WIDTH{1'b0}};
        end
    end
end

reg [DATA_WIDTH-1:0] wr_exp_mask;
reg [DATA_WIDTH-1:0] rd_exp_mask;

genvar j;
generate
for (j = 0; j < NUM_PARTITIONS; j = j + 1) begin
    always @(*) begin
        if (wr_mask[j]) begin
            wr_exp_mask[((j+1)*PARTITION_WIDTH)-1 : j*PARTITION_WIDTH] <= {PARTITION_WIDTH{1'b1}};
        end else begin
            wr_exp_mask[((j+1)*PARTITION_WIDTH)-1 : j*PARTITION_WIDTH] <= {PARTITION_WIDTH{1'b0}};
        end

        if (rd_mask[j]) begin
            rd_exp_mask[((j+1)*PARTITION_WIDTH)-1:j*PARTITION_WIDTH] <= {PARTITION_WIDTH{1'b1}};
        end else begin
            rd_exp_mask[((j+1)*PARTITION_WIDTH)-1:j*PARTITION_WIDTH] <= {PARTITION_WIDTH{1'b0}};
        end
    end
end
endgenerate

wire [DATA_WIDTH-1:0] rd_data_masked;
wire [DATA_WIDTH-1:0] wr_data_masked;

assign rd_data_masked = mem[rd_addr] & rd_exp_mask;
assign wr_data_masked = (mem[wr_addr] & ~wr_exp_mask) | (wr_data & wr_exp_mask);

always @(posedge clk) begin
    if (rd_en) begin
        rd_data <= rd_data_masked;
    end else if (wr_en) begin
        mem[wr_addr] <= wr_data_masked;
    end
end
endmodule
