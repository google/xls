// Copyright 2021 The XLS Authors
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

// Module which passes the data and valid signals of an input ready-valid
// channel out on an output channel.
module input_channel_monitor(
  input wire        rst,

  output wire       input_ready,
  input wire        input_valid,
  input wire [7:0]  input_data,

  input wire        monitor_ready,
  output wire       monitor_valid,
  output wire [9:0] monitor_data
);
  assign input_ready = 1'h1;
  assign monitor_valid = 1'h1;
  assign monitor_data = {input_valid, input_data};
endmodule
