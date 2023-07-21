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


// Semi-dual-port RAM with XLS-native channel interface
// - Can be used with Verilog modules generated without using
//   `ram_configurations` codegen option
// - Simultaneous write & read behavior: reads new value
module sdpram_xls_chan #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 16,
    parameter NUM_PARTITIONS = 1
) (
    input clk,
    input rst,
    // write request
    input [DATA_WIDTH+ADDR_WIDTH+NUM_PARTITIONS-1:0] wr_req_data,
    input wr_req_vld,
    output wr_req_rdy,
    // write response (no data)
    output wr_resp_vld,
    input wr_resp_rdy,
    // read request
    input [ADDR_WIDTH+NUM_PARTITIONS-1:0] rd_req_data,
    input rd_req_vld,
    output rd_req_rdy,
    // read response
    output [DATA_WIDTH-1:0] rd_resp_data,
    output rd_resp_vld,
    input rd_resp_rdy
);

    localparam SIZE = 32'd1 << ADDR_WIDTH;

    // RAM array
    reg [DATA_WIDTH-1:0] mem [SIZE];

    // Disassemble requests
    reg [DATA_WIDTH-1:0] r_wr_req_f_data;
    reg [ADDR_WIDTH-1:0] r_wr_req_f_addr;
    reg [NUM_PARTITIONS-1:0] r_wr_req_f_mask;
    reg [ADDR_WIDTH-1:0] r_rd_req_f_addr;
    reg [NUM_PARTITIONS-1:0] r_rd_req_f_mask;
    
    always @* begin
        {
            r_wr_req_f_addr,
            r_wr_req_f_data,
            r_wr_req_f_mask
        } = wr_req_data;
        {
            r_rd_req_f_addr,
            r_rd_req_f_mask
        } = rd_req_data;
    end

    // Assemble response
    reg [DATA_WIDTH-1:0] r_rd_resp_f_data;
    assign rd_resp_data = r_rd_resp_f_data;

    // Reset RAM on reset (simulator only)
    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < SIZE; i = i + 1) begin
                mem[i] <= {DATA_WIDTH{1'bx}};
            end
        end
    end

    // RAM-write state machine
    reg r_wr_req_canhandle;
    reg r_wr_req_ack;
    reg r_wr_resp_pending;
    reg r_wr_resp_ack;

    always @* begin
        r_wr_req_ack = wr_req_vld && wr_req_rdy;
        r_wr_resp_ack = wr_resp_vld && wr_resp_rdy;
        // We're accepting new request only if there is no pending response
        // or that response is gonna be acknowledged in this cycle
        r_wr_req_canhandle = !r_wr_resp_pending || r_wr_resp_ack;
    end

    assign wr_req_rdy = r_wr_req_canhandle;
    assign wr_resp_vld = r_wr_resp_pending;

    always @(posedge clk) begin
        if (rst) begin
            r_wr_resp_pending <= 1'b0;
        end else begin
            if (r_wr_resp_ack) begin
                r_wr_resp_pending <= 1'b0;
            end
            if (r_wr_req_ack) begin
                // write to RAM and signal response availability
                mem[r_wr_req_f_addr] <= r_wr_req_f_data;
                r_wr_resp_pending <= 1'b1;
            end
        end
    end

    // RAM-read state machine
    reg r_rd_req_canhandle;
    reg r_rd_req_ack;
    reg r_rd_resp_pending;
    reg r_rd_resp_ack;

    always @* begin
        r_rd_req_ack = rd_req_vld && rd_req_rdy;
        r_rd_resp_ack = rd_resp_vld && rd_resp_rdy;
        // We're accepting new request only if there is no pending response
        // or that response is gonna be acknowledged in this cycle
        r_rd_req_canhandle = !r_rd_resp_pending || r_rd_resp_ack;
    end

    assign rd_req_rdy = r_rd_req_canhandle;
    assign rd_resp_vld = r_rd_resp_pending;

    always @(posedge clk) begin
        if (rst) begin
            r_rd_resp_pending <= 1'b0;
            r_rd_resp_f_data <= {DATA_WIDTH{1'bx}};
        end else begin
            if (r_rd_resp_ack) begin
                r_rd_resp_pending <= 1'b0;
                r_rd_resp_f_data <= {DATA_WIDTH{1'bx}};
            end
            if (r_rd_req_ack) begin
                // read from RAM and signal response availability
                if (r_wr_req_ack && r_rd_req_f_addr == r_wr_req_f_addr) begin
                    // write-before-read logic
                    r_rd_resp_f_data <= r_wr_req_f_data;
                end else begin
                    // normal read
                    r_rd_resp_f_data <= mem[r_rd_req_f_addr];
                end
                r_rd_resp_pending <= 1'b1;
            end
        end
    end

endmodule

// Single-port with XLS-native channel interface
// - Can be used with Verilog modules generated without using
//   `ram_configurations` codegen option
module spram_xls_chan #(
    parameter DATA_WIDTH = 8,
    parameter ADDR_WIDTH = 16
) (
    input clk,
    input rst,
    // request
    input [ADDR_WIDTH+DATA_WIDTH+1:0] req_data,
    input req_vld,
    output req_rdy,
    // write comletion response (no data)
    output wr_comp_vld,
    input wr_comp_rdy,
    // read response
    output [DATA_WIDTH-1:0] resp_data,
    output resp_vld,
    input resp_rdy
);

    localparam SIZE = 32'd1 << ADDR_WIDTH;

    // RAM array
    reg [DATA_WIDTH-1:0] mem [SIZE];

    // Disassemble requests
    reg [ADDR_WIDTH-1:0] r_req_f_addr;
    reg [DATA_WIDTH-1:0] r_req_f_data;
    reg r_req_f_we;
    reg r_req_f_re;
    
    always @* begin
        {
            r_req_f_addr,
            r_req_f_data,
            r_req_f_we,
            r_req_f_re
        } = req_data;
    end

    // Assemble response
    reg [DATA_WIDTH-1:0] r_resp_f_data;
    assign resp_data = r_resp_f_data;

    // Reset RAM on reset (simulator only)
    integer i;
    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < SIZE; i = i + 1) begin
                mem[i] <= {DATA_WIDTH{1'bx}};
            end
        end
    end

    // State machine
    reg r_req_canhandle;
    reg r_req_ack;
    reg r_resp_pending;
    reg r_resp_ack;
    reg r_wr_comp_pending;
    reg r_wr_comp_ack;

    always @* begin
        r_req_ack = req_vld && req_rdy;
        r_resp_ack = resp_vld && resp_rdy;
        r_wr_comp_ack = wr_comp_vld && wr_comp_rdy;
        // We're accepting new request only if there is no pending response
        // or that response is gonna be acknowledged in this cycle
        r_req_canhandle =
            (!r_resp_pending || r_resp_ack)
            && (!r_wr_comp_pending || r_wr_comp_ack);
    end

    assign req_rdy = r_req_canhandle;
    assign resp_vld = r_resp_pending;
    assign wr_comp_vld = r_wr_comp_pending;

    always @(posedge clk) begin
        if (rst) begin
            r_resp_pending <= 1'b0;
            r_wr_comp_pending <= 1'b0;
        end else begin
            if (r_resp_ack) begin
                r_resp_pending <= 1'b0;
            end
            if (r_wr_comp_ack) begin
                r_wr_comp_pending <= 1'b0;
            end
            if (r_req_ack) begin
                // SinglePortRamModel has WE prioritized over RE.
                // If neither WE nor RE are set, no response is generated.
                if (r_req_f_we) begin
                    mem[r_req_f_addr] <= r_req_f_data;
                    r_wr_comp_pending <= 1'b1;
                end else if (r_req_f_re) begin
                    r_resp_f_data <= mem[r_req_f_addr];
                    r_resp_pending <= 1'b1;
                end
            end
        end
    end

endmodule
