// Copyright 2024 The XLS Authors
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

#ifndef XLS_CODEGEN_TEST_FIFOS_H_
#define XLS_CODEGEN_TEST_FIFOS_H_

#include <string>

#include "xls/ir/channel.h"

namespace xls {
namespace verilog {

// Library of FIFOs for use in testing. Includes FifoConfig and RTL text.

struct FifoWithConfig {
  FifoConfig config;
  std::string rtl;
};

inline FifoWithConfig kDepth1Fifo{
    .config = FifoConfig(/*depth=*/1, /*bypass=*/false,
                         /*register_push_outputs=*/true,
                         /*register_pop_outputs=*/false),
    .rtl = R"(// simple fifo depth-1 implementation
module xls_fifo_wrapper (
clk, rst,
push_ready, push_data, push_valid,
pop_ready,  pop_data,  pop_valid);
  parameter Width = 32,
            Depth = 32,
            EnableBypass = 0,
            RegisterPushOutputs = 1,
            RegisterPopOutputs = 1;
  localparam AddrWidth = $clog2(Depth) + 1;
  input  wire             clk;
  input  wire             rst;
  output wire             push_ready;
  input  wire [Width-1:0] push_data;
  input  wire             push_valid;
  input  wire             pop_ready;
  output wire [Width-1:0] pop_data;
  output wire             pop_valid;

  // Require depth be 1 and bypass disabled.
  initial begin
    if (EnableBypass || Depth != 1 || !RegisterPushOutputs || RegisterPopOutputs) begin
      // FIFO configuration not supported.
      $fatal(1);
    end
  end


  reg [Width-1:0] mem;
  reg full;

  assign push_ready = !full;
  assign pop_valid = full;
  assign pop_data = mem;

  always @(posedge clk) begin
    if (rst == 1'b1) begin
      full <= 1'b0;
    end else begin
      if (push_valid && push_ready) begin
        mem <= push_data;
        full <= 1'b1;
      end else if (pop_valid && pop_ready) begin
        mem <= mem;
        full <= 1'b0;
      end else begin
        mem <= mem;
        full <= full;
      end
    end
  end
endmodule
)"};

inline FifoWithConfig kDepth1NoDataFifo{
    .config = FifoConfig(/*depth=*/1, /*bypass=*/false,
                         /*register_push_outputs=*/true,
                         /*register_pop_outputs=*/false),
    .rtl =
        R"(// simple nodata fifo depth-1 implementation
module xls_nodata_fifo_wrapper (
clk, rst,
push_ready, push_valid,
pop_ready,  pop_valid);
  parameter Depth = 32,
            EnableBypass = 0,
            RegisterPushOutputs = 1,
            RegisterPopOutputs = 1;
  localparam AddrWidth = $clog2(Depth) + 1;
  input  wire             clk;
  input  wire             rst;
  output wire             push_ready;
  input  wire             push_valid;
  input  wire             pop_ready;
  output wire             pop_valid;

  // Require depth be 1 and bypass disabled.
  initial begin
    if (EnableBypass || Depth != 1 || !RegisterPushOutputs || RegisterPopOutputs) begin
      // FIFO configuration not supported.
      $fatal(1);
    end
  end

  reg full;

  assign push_ready = !full;
  assign pop_valid = full;

  always @(posedge clk) begin
    if (rst == 1'b1) begin
      full <= 1'b0;
    end else begin
      if (push_valid && push_ready) begin
        full <= 1'b1;
      end else if (pop_valid && pop_ready) begin
        full <= 1'b0;
      end else begin
        full <= full;
      end
    end
  end
endmodule
)"};

inline FifoWithConfig kDepth0Fifo{
    .config = FifoConfig(/*depth=*/0, /*bypass=*/true,
                         /*register_push_outputs=*/false,
                         /*register_pop_outputs=*/false),
    .rtl = R"(// simple fifo depth-1 implementation
module xls_fifo_wrapper (
clk, rst,
push_ready, push_data, push_valid,
pop_ready,  pop_data,  pop_valid);
  parameter Width = 32,
            Depth = 32,
            EnableBypass = 0,
            RegisterPushOutputs = 1,
            RegisterPopOutputs = 1;
  localparam AddrWidth = $clog2(Depth) + 1;
  input  wire             clk;
  input  wire             rst;
  output wire             push_ready;
  input  wire [Width-1:0] push_data;
  input  wire             push_valid;
  input  wire             pop_ready;
  output wire [Width-1:0] pop_data;
  output wire             pop_valid;

  // Require depth be 1 and bypass disabled.
  initial begin
    if (EnableBypass != 1 || Depth != 0 || RegisterPushOutputs || RegisterPopOutputs) begin
      // FIFO configuration not supported.
      $fatal(1);
    end
  end


  assign push_ready = pop_ready;
  assign pop_valid = push_valid;
  assign pop_data = push_data;

endmodule
)"};

inline FifoWithConfig kDepth0NoDataFifo{
    .config = FifoConfig(/*depth=*/0, /*bypass=*/true,
                         /*register_push_outputs=*/false,
                         /*register_pop_outputs=*/false),
    .rtl =
        R"(// simple nodata fifo depth-1 implementation
module xls_nodata_fifo_wrapper (
clk, rst,
push_ready, push_valid,
pop_ready,  pop_valid);
  parameter Depth = 32,
            EnableBypass = 0,
            RegisterPushOutputs = 1,
            RegisterPopOutputs = 1;
  localparam AddrWidth = $clog2(Depth) + 1;
  input  wire             clk;
  input  wire             rst;
  output wire             push_ready;
  input  wire             push_valid;
  input  wire             pop_ready;
  output wire             pop_valid;

  // Require depth be 1 and bypass disabled.
  initial begin
    if (EnableBypass != 1 || Depth != 0 || RegisterPushOutputs || RegisterPopOutputs) begin
      // FIFO configuration not supported.
      $fatal(1);
    end
  end

  assign push_ready = pop_ready;
  assign pop_valid = push_valid;

endmodule
)"};

}  // namespace verilog
}  // namespace xls

#endif  // XLS_CODEGEN_TEST_FIFOS_H_
