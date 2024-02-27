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

// Implements the "identity" (often abbreviated as ident) function for 8-bit
// inputs. That is, buffers up four bytes and then sends them back.

`include "xls/contrib/ice40/uncore_rtl/ice40/uart_receiver.v"
`include "xls/contrib/ice40/uncore_rtl/ice40/uart_transmitter.v"

module top(
  input wire clk,
  input wire rx_in,
  output wire tx_out,
  output wire clear_to_send_out_n
);
  parameter ClocksPerBaud = `DEFAULT_CLOCKS_PER_BAUD;

  localparam
    StateIdle    = 2'd0,
    StateGotByte = 2'd1,
    StateError   = 2'd2;
  localparam StateBits = 2;

  wire rst_n;
  assign rst_n = 1;

  reg rx_byte_done = 0;
  reg rx_byte_done_next;

  reg [StateBits-1:0] state = StateIdle;
  reg [StateBits-1:0] state_next;

  reg [7:0] tx_byte = 8'hff;
  reg [7:0] tx_byte_next;

  reg tx_byte_valid = 0;
  reg tx_byte_valid_next;

  wire tx_byte_done;
  wire [7:0] rx_byte;
  wire rx_byte_valid;

  wire clear_to_send;
  assign clear_to_send_out_n = ~clear_to_send;

  uart_receiver #(
    .ClocksPerBaud    (ClocksPerBaud)
  ) rx(
    .clk              (clk),
    .rst_n            (rst_n),
    .rx               (rx_in),
    .rx_byte_out      (rx_byte),
    .rx_byte_valid_out(rx_byte_valid),
    .rx_byte_done     (rx_byte_done),
    .clear_to_send_out(clear_to_send)
  );
  uart_transmitter #(
    .ClocksPerBaud   (ClocksPerBaud)
  ) tx(
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
      StateIdle: begin
        if (rx_byte_valid) begin
          state_next = tx_byte_done ? StateGotByte : StateError;
        end
      end
      StateGotByte: begin
        state_next = StateIdle;
      end
    endcase
  end

  // Non-state updates.
  always @(*) begin  // verilog_lint: waive always-comb b/72410891
    rx_byte_done_next  = rx_byte_done;
    tx_byte_next       = tx_byte;
    tx_byte_valid_next = tx_byte_valid;

    case (state)
      StateIdle: begin
        tx_byte_valid_next = 0;
      end
      StateGotByte: begin
        tx_byte_next       = rx_byte;
        tx_byte_valid_next = 1;
        rx_byte_done_next  = 1;
      end
    endcase
  end

  // Note: our version of iverilog has no support for always_ff.
  always @ (posedge clk) begin
    rx_byte_done  <= rx_byte_done_next;
    state         <= state_next;
    tx_byte       <= tx_byte_next;
    tx_byte_valid <= tx_byte_valid_next;
  end

endmodule
