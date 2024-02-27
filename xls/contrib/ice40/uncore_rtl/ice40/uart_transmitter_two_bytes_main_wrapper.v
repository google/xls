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

// Wraps the ICE40 top-level with monitors/ wave dumping for debugging
// purposes.

`include "xls/contrib/ice40/uncore_rtl/ice40/uart_transmitter_two_bytes_main.v"

module top_wrapper;
  reg clk = 0;
  wire led_left;
  wire led_center;
  wire tx;

  top #(
    .ClocksPerBaud(2)
  ) t(
    .clk           (clk),
    .tx_out        (tx),
    .led_left_out  (led_left),
    .led_center_out(led_center)
  );

  initial begin
    clk = 0;
    forever #2 clk = !clk;
  end

  initial begin
    $monitor("%t: clk: %b tx: %b state: %b next_state: %b leds: {%b, %b}; tx_byte_done: %b",
             $time, clk, tx, t.state, t.state_next, led_left, led_center, t.tx_byte_done);
    //$dumpfile("/tmp/uart_transmitter_two_bytes_main_wrapper.vcd");
    //$dumpvars(0, clk, tx, t.state, t.state_next, t.tx_byte_done, t.tx_byte, t.tx_byte_valid,
    //          t.transmitter.tx_state);

    #1024 $finish;
  end

endmodule
