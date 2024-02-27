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

`timescale 1 ns / 1 ps
`include "xls/contrib/ice40/uncore_rtl/ice40/uart_transmitter.v"

`ifndef CLOCKS_PER_BAUD
  `define CLOCKS_PER_BAUD 2
`endif

module uart_transmitter_two_bytes_test;

  reg clk;
  reg rst_n;

  reg [7:0] tx_byte;
  reg tx_byte_valid;

  wire tx_byte_done;
  wire tx;

  integer i, start_time, first_byte_done_time, second_byte_done_time;

  // #TicksPerClock waits for a single clock cycle.
  localparam TicksPerClock = 2;

  // Number of clock cycles that consitutes the time of a single baud.
  localparam ClocksPerBaud = `CLOCKS_PER_BAUD;

  uart_transmitter #(
    .ClocksPerBaud(ClocksPerBaud)
  ) transmitter(
    .clk             (clk),
    .rst_n           (rst_n),
    .tx_byte         (tx_byte),
    .tx_byte_valid   (tx_byte_valid),
    .tx_byte_done_out(tx_byte_done),
    .tx_out          (tx)
  );

  initial begin
    #1 clk = 0;

    forever #1 clk = !clk;
  end

  `include "xls/contrib/ice40/uncore_rtl/ice40/xls_assertions.inc"

  // Make sure we finish after some reasonable amount of time.
  initial begin
    #1024 begin
      $display("ERROR: timeout, simulation ran too long");
      $finish;
    end
  end

  initial begin
    //$dumpfile("/tmp/uart_transmitter_two_bytes_test.vcd");
    //$dumpvars(0, clk, rst_n, tx_byte, tx_byte_valid, tx_byte_done, tx,
    //          transmitter.tx_bitno, transmitter.tx_state,
    //          transmitter.tx_state_next, transmitter.tx_byte_done_next);
    $display("Starting...\n");
    $monitor("%t tx: %b tx_byte_done: %b", $time, tx, tx_byte_done);

    rst_n <= 0;
    tx_byte_valid <= 0;
    tx_byte <= 'hff;

    // Come out of reset after a few cycles.
    #4 rst_n <= 1;

    $dumpon;

    #TicksPerClock;
    for (i = 0; i < 10; i = i + 1) begin
      xls_assert(1, tx, "idle");
      #1;
    end

    // Present a byte to transmit after those few cycles of non-reset idle
    // activity.
    tx_byte <= 'h55;
    tx_byte_valid <= 1;

    #TicksPerClock tx_byte_valid <= 0;

    #0.1 xls_assert(tx_byte_done, 0, "tx_byte_done 'h55 start");

    // Wait for the start bit to show up.
    wait (tx == 0);

    // Note the start time!
    start_time = $time;

    // Check we wiggle 1-0 four times for the data.
    repeat (4) begin
      wait (tx == 1);
      wait (tx == 0);
    end

    wait (transmitter.tx_state_next == 'b11);  // Going to stop.
    $display("About to stop @ %t", $time);

    `ifdef PRESENT_BIT_EARLY

    $display("Presenting bit early...");

    // Present the next byte to transmit for one cycle while the stop bit is
    // tranmitting.
    #TicksPerClock;
    $display("Presenting second byte @ %t", $time);
    tx_byte <= 'haa;
    tx_byte_valid <= 1;
    #TicksPerClock tx_byte_valid <= 0;

    `else

    $display("Presenting bit late...");

    wait (tx == 1);  // stop bit

    #((ClocksPerBaud-2)*TicksPerClock);

    // Present the transfer request on the last cycle.
    // Note we already observed one cycle of stop bit above.
    tx_byte <= 'haa;
    tx_byte_valid <= 1;
    #(TicksPerClock+0.3) tx_byte_valid <= 0;

    `endif

    // Wait for it to say it's processing that second byte.
    wait (transmitter.tx_state != 'b11);

    first_byte_done_time = $time;

    // Check we wiggle 0-1 four times for the data.
    repeat (4) begin
      wait (tx == 0);
      wait (tx == 1);
    end

    // Wait for that second byte to be done.
    wait (tx_byte_done == 1);

    // Note when this second byte is done!
    second_byte_done_time = $time;

    $display("start: %t first_byte_done: %t second_byte_done: %t",
             start_time, first_byte_done_time, second_byte_done_time);

    xls_assert_int_eq(
      10*ClocksPerBaud-1,
      (first_byte_done_time-start_time)/2, "first byte cycles");

    xls_assert_int_eq(
      `ifdef PRESENT_BIT_EARLY
      9*ClocksPerBaud,
      `else
      9*ClocksPerBaud+1,
      `endif
      (second_byte_done_time-first_byte_done_time)/2,
      "second byte cycles");

    #256 $finish;
  end

endmodule
