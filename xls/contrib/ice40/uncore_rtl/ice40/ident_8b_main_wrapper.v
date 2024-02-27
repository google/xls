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

`include "xls/contrib/ice40/uncore_rtl/ice40/ident_8b_main.v"

module top_wrapper;
  reg clk = 0;
  reg rx = 1;
  wire clear_to_send;
  wire tx;
  integer i;

  localparam ClocksPerBaud = 8;
  localparam TicksPerClock = 2;

  top #(
    .ClocksPerBaud      (ClocksPerBaud)
  ) t(
    .clk                (clk),
    .rx_in              (rx),
    .tx_out             (tx),
    .clear_to_send_out_n(clear_to_send)
  );

  initial begin
    clk = 0;
    #1;
    forever #1 clk = !clk;
  end

  initial begin
    //$dumpfile("/tmp/ident_8b_main_wrapper.vcd");
    //$dumpvars(0, clk, rx, tx, t.state, t.state_next, t.rx.state,
    //  t.rx.sample_count, t.rx.samples, t.rx.rx_countdown, t.rx_byte_done);

    #8;
    rx = 0;
    #(ClocksPerBaud*TicksPerClock);
    for (i = 0; i < 8; i = i + 1) begin
      rx = ((i % 2) == 0);
      #(ClocksPerBaud*TicksPerClock);
    end
    rx = 1;

    #1024 $finish;
  end

endmodule
