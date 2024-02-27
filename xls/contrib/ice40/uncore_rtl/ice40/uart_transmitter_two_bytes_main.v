// Copyright 2020 The XLS Authors
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

// Writes "ABC" onto the terminal via UART.

`include "xls/contrib/ice40/uncore_rtl/ice40/uart_transmitter.v"

module top(
  input wire clk,
  output wire tx_out,
  output wire led_left_out,
  output wire led_center_out
);
  parameter ClocksPerBaud = `DEFAULT_CLOCKS_PER_BAUD;
  localparam
    Start = 'd0,
    TransmitByte0 = 'd1,
    TransmitByte1 = 'd2,
    Done = 'd3;
  localparam StateBits = 2;

  reg [StateBits-1:0] state = Start;
  reg [7:0] tx_byte = 'hff;
  reg tx_byte_valid = 0;

  // No external reset pin on IceStick?
  reg rst_n = 1;

  reg [StateBits-1:0] state_next;
  reg [7:0] tx_byte_next;
  reg tx_byte_valid_next;

  wire tx_byte_done;

  assign {led_left_out, led_center_out} = state;

  uart_transmitter #(
    .ClocksPerBaud(ClocksPerBaud)
  ) transmitter(
    .clk             (clk),
    .rst_n           (rst_n),
    .tx_byte         (tx_byte),
    .tx_byte_valid   (tx_byte_valid),
    .tx_byte_done_out(tx_byte_done),
    .tx_out          (tx_out)
  );

  // State manipulation.
  always @(*) begin  // verilog_lint: waive always-comb b/72410891
    state_next = state;
    case (state)
      Start: begin
        if (tx_byte_done == 0) begin
          state_next = TransmitByte0;
        end
      end
      TransmitByte0: begin
        if (tx_byte_done && !tx_byte_valid) begin
          state_next = TransmitByte1;
        end
      end
      TransmitByte1: begin
        if (tx_byte_done && !tx_byte_valid) begin
          state_next = Done;
        end
      end
      Done: begin
        // Final state.
      end
      default: begin
        state_next = 1'bX;
      end
    endcase
  end

  // Non-state updates.
  always @(*) begin  // verilog_lint: waive always-comb b/72410891
    tx_byte_next       = tx_byte;
    tx_byte_valid_next = tx_byte_valid;
    case (state)
      Start: begin
        tx_byte_next       = 'h41;  // 'A'
        tx_byte_valid_next = 'b1;
      end
      TransmitByte0: begin
        if (tx_byte_done && !tx_byte_valid) begin
          tx_byte_next       = 'h42;  // 'B'
          tx_byte_valid_next = 'b1;
        end else begin
          tx_byte_valid_next = 'b0;
        end
      end
      TransmitByte1: begin
        if (tx_byte_done && !tx_byte_valid) begin
          tx_byte_next       = 'h43;  // 'C'
          tx_byte_valid_next = 'b1;
        end else begin
          tx_byte_valid_next = 'b0;
        end
      end
      Done: begin
        tx_byte_valid_next = 'b0;
      end
      default: begin
        tx_byte_next       = 'hXX;
        tx_byte_valid_next = 'bX;
      end
    endcase
  end

  always @ (posedge clk) begin
    state         <= state_next;
    tx_byte       <= tx_byte_next;
    tx_byte_valid <= tx_byte_valid_next;
  end

endmodule
