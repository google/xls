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

// Implements a UART transmitter with a single start bit, a single stop bit,
// and 8 bits of data payload. Intended for use with the Lattice ICE40.

// The number of clock ticks within a single baud transfer time.
//
// Default value is 12MHz/9600 = 1250
//
// We define this with a `define so that wrapping modules can refer to it
// without duplication (it does not appear possible to refer to a module's
// default parameter value).
`ifndef DEFAULT_CLOCKS_PER_BAUD
 `define DEFAULT_CLOCKS_PER_BAUD 1250
`endif

// Interacting with the transmitter:
//
// * tx_byte_valid signals that the contents of tx_byte should be used for the
//   transfer.
// * After the user asserts tx_byte_valid, then tx_byte must be held constant
//   until tx_byte_done is asserted by the transmitter.
// * Once tx_byte_done has been asserted by the transmitter, the user only
//   needs to signal the new tx_byte (along with tx_byte_valid) for a single
//   cycle.
// * Implementation note: internally the FSM can transition straight from
//   sending the stop bit for the previous byte into sending the next tx_byte.
module uart_transmitter(
  input wire       clk,
  input wire       rst_n,
  input wire [7:0] tx_byte,
  input wire       tx_byte_valid,
  output wire      tx_byte_done_out,
  output wire      tx_out
);
  parameter ClocksPerBaud = `DEFAULT_CLOCKS_PER_BAUD;

  // Before we transition to a new state we prime the "countdown" (in
  // tx_countdown) to stay in that state until the next transition action
  // should occur.
  localparam CountdownBits = $clog2(ClocksPerBaud);
  localparam CountdownStart = ClocksPerBaud-1;
  localparam
      StateIdle  = 'd0,
      StateStart = 'd1,
      StateFrame = 'd2,
      StateStop  = 'd3;
  localparam StateBits = 2;

  reg                     tx = 1;
  reg                     tx_byte_done = 1;
  reg [StateBits-1:0]     tx_state = StateIdle;
  reg [CountdownBits-1:0] tx_countdown = 0;
  reg [2:0]               tx_bitno = 0;

  reg                     tx_next;
  reg                     tx_byte_done_next;
  reg [StateBits-1:0]     tx_state_next;
  reg [CountdownBits-1:0] tx_countdown_next;
  reg [2:0]               tx_bitno_next;

  assign tx_out           = tx;
  assign tx_byte_done_out = tx_byte_done;

  // State manipulation.
  always @(*) begin  // verilog_lint: waive always-comb b/72410891
    tx_state_next = tx_state;

    case (tx_state)
      StateIdle: begin
        if (tx_byte_valid || !tx_byte_done) begin
          tx_state_next = StateStart;
        end
      end
      StateStart: begin
        if (tx_countdown == 0) begin
          tx_state_next = StateFrame;
        end
      end
      StateFrame: begin
        if (tx_countdown == 0 && tx_bitno == 7) begin
          tx_state_next = StateStop;
        end
      end
      StateStop: begin
        if (tx_countdown == 0) begin
          // We permit a transition directly from "stop" to "start" if the
          // outside world presented a byte to transmit during the stop bit.
          tx_state_next = tx_byte_done ? StateIdle : StateStart;
        end
      end
      default: begin
        tx_state_next     = 1'bX;
      end
    endcase
  end

  // Non-state updates.
  always @(*) begin  // verilog_lint: waive always-comb b/72410891
    tx_next           = tx;
    tx_byte_done_next = tx_byte_done;
    tx_bitno_next     = tx_bitno;
    tx_countdown_next = tx_countdown;

    case (tx_state)
      StateIdle: begin
        tx_next           = 1;
        tx_bitno_next     = 0;
        tx_countdown_next = CountdownStart;
        tx_byte_done_next = !tx_byte_valid;
      end
      StateStart: begin
        tx_next           = 0;  // Start bit signal.
        tx_byte_done_next = 0;
        tx_bitno_next     = 0;
        tx_countdown_next = tx_countdown == 0 ? CountdownStart : tx_countdown - 1;
      end
      StateFrame: begin
        tx_next           = tx_byte[tx_bitno];
        tx_byte_done_next = tx_countdown == 0 && tx_bitno == 7;
        tx_bitno_next     = tx_countdown == 0 ? tx_bitno + 1 : tx_bitno;
        tx_countdown_next = tx_countdown == 0 ? CountdownStart : tx_countdown - 1;
      end
      StateStop: begin
        tx_next           = 1;  // Stop bit signal.
        // The byte is done, unless a new valid byte has been presented.
        // When a new byte is presented the state is sticky so the outside
        // world doesn't need to assert tx_byte_valid for longer than a cycle.
        tx_byte_done_next = (tx_byte_done == 0 || tx_byte_valid == 1) ? 0 : 1;
        tx_countdown_next = tx_countdown == 0 ? CountdownStart : tx_countdown - 1;
      end
      default: begin
        tx_next           = 1'bX;
        tx_byte_done_next = 1'bX;
        tx_bitno_next     = 1'bX;
        tx_countdown_next = 1'bX;
      end
    endcase
  end

  // Note: our version of iverilog has no support for always_ff.
  always @ (posedge clk) begin
    if (rst_n == 0) begin
      tx           <= 1;
      tx_byte_done <= 1;
      tx_state     <= StateIdle;
      tx_countdown <= 0;
      tx_bitno     <= 0;
    end else begin
      tx           <= tx_next;
      tx_byte_done <= tx_byte_done_next;
      tx_state     <= tx_state_next;
      tx_countdown <= tx_countdown_next;
      tx_bitno     <= tx_bitno_next;
    end
  end

endmodule
