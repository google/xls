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

module ram_2rw # (
    parameter DATA_WIDTH = 4,
    parameter SIZE = 32,
    parameter ADDR_WIDTH = $clog2(SIZE),
    parameter INIT_FILE = ""
) (
    input wire clk,
    input wire rst,

    input wire [DATA_WIDTH - 1:0]     data_0,
    input wire [ADDR_WIDTH - 1:0]     addr_0,
    input wire                        wr_en_0,
    input wire                        rd_en_0,
    input wire [DATA_WIDTH - 1:0]     data_1,
    input wire [ADDR_WIDTH - 1:0]     addr_1,
    input wire                        wr_en_1,
    input wire                        rd_en_1,

    output reg [DATA_WIDTH - 1:0]     rd_data_0,
    output reg [DATA_WIDTH - 1:0]     rd_data_1
);

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

integer j;

always @(posedge clk) begin
    if (rd_en_0) begin
        rd_data_0 <= mem[addr_0];
    end
    if (rd_en_1) begin
        rd_data_1 <= mem[addr_1];
    end

    if (wr_en_0) begin
        mem[addr_0] <= data_0;
    end
    if (wr_en_1) begin
        mem[addr_1] <= data_1;
    end
end
endmodule
