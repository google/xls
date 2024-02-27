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

// Sends a single byte (if TEST_SINGLE_BYTE) or two bytes to the UART
// receiver and checks it is properly received / obeying its interface
// protocol.

`include "xls/contrib/ice40/uncore_rtl/ice40/uart_receiver.v"

module uart_receiver_test;
  localparam TicksPerClock = 2;
  localparam ClocksPerBaud = 8;

  integer i;

  reg clk;
  reg rst_n;

  reg rx;
  reg rx_byte_done;

  wire [7:0] rx_byte;
  wire rx_byte_valid;
  wire clear_to_send;

  integer stop_bit_begin_time;
  integer stop_bit_done = 0;

  uart_receiver #(
    .ClocksPerBaud(ClocksPerBaud)
  ) receiver (
    .clk              (clk),
    .rst_n            (rst_n),
    .rx               (rx),
    .clear_to_send_out(clear_to_send),
    .rx_byte_out      (rx_byte),
    .rx_byte_valid_out(rx_byte_valid),
    .rx_byte_done     (rx_byte_done)
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
    //$dumpfile("/tmp/uart_receiver_test.vcd");
    //$dumpvars(0, clk, rst_n, rx_byte, rx_byte_valid, rx, rx_byte_done,
    //  clear_to_send, receiver.state, receiver.state_next,
    //  receiver.rx_countdown, receiver.samples, receiver.sample_count,
    //  receiver.data_bitno, receiver.rx_byte_valid_next);
    $display("Starting...\n");
    $monitor("%t rx: %b rx_byte_valid: %b", $time, rx, rx_byte_valid);

    rst_n <= 0;
    rx <= 1;
    rx_byte_done <= 0;

    // Come out of reset after a few cycles.
    #4 rst_n <= 1;

    #TicksPerClock;

    xls_assert(receiver.state, 0, "receiver state should be idle");
    xls_assert(clear_to_send, 1, "should be clear to send when idle");

    // Start bit.
    rx <= 0;
    #(TicksPerClock*ClocksPerBaud);

    // Send toggling bits, starting with 'b1 to make 'h55.
    for (i = 0; i < 8; i = i + 1) begin
      xls_assert(clear_to_send, 0, "transmitting from testbench");
      rx <= (i % 2 == 0);
      #(TicksPerClock*ClocksPerBaud);
    end

    stop_bit_begin_time = $time;

    // Stop bit / idle.
    rx <= 1;

    #TicksPerClock;

    // Byte should be valid before we receive the stop bit.
    #1;
    xls_assert(1, rx_byte_valid, "valid during stop bit");
    xls_assert_int_eq(8'h55, rx_byte, "byte payload during stop bit");

    `ifdef TEST_SINGLE_BYTE

    // Wait to transition back to idle.
    wait (receiver.state == 'd0);

    // Byte should be valid and the same after we're idle.
    xls_assert(1, rx_byte_valid, "valid when idle");
    xls_assert_int_eq(8'h55, rx_byte, "byte payload when idle");
    xls_assert(clear_to_send, 0, "byte is valid, should not be clear to send");

    `else

    // Discard the byte immediately now that we've checked it.
    rx_byte_done <= 1;

    // Check that subsequently we say it's ok to send.
    #TicksPerClock;
    xls_assert(1, clear_to_send, "clear to send once byte is done");

    // Only note that the RX byte is done for a single cycle.
    rx_byte_done <= 0;

    // Wait until we've sent the stop bit for a full baud of time.
    #(TicksPerClock*ClocksPerBaud-($time-stop_bit_begin_time));

    $display("Starting second byte.");

    // Then send a start bit and the next byte.
    rx <= 0;
    #(ClocksPerBaud*TicksPerClock);

    for (i = 0; i < 8; i = i + 1) begin
      rx <= (8'h01 >> i) & 1'b1;
      #(ClocksPerBaud*TicksPerClock);
    end

    // Stop bit / idle.
    rx <= 1;

    #TicksPerClock;

    // Byte should be valid before we receive the stop bit.
    #1;
    xls_assert(1, rx_byte_valid, "valid during stop bit");
    xls_assert_int_eq(8'h01, rx_byte, "byte payload during stop bit");

    `endif

    // Pad a little time before end of sim.
    #(8*TicksPerClock);
    $finish;
  end

endmodule
