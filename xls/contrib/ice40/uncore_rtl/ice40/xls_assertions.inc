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

// Assertions to be `included into Verilog testbench modules for testing.
//
// TODO(leary): Move to xls/contrib/ice40/uncore_rtl/common/xls_assertions.inc
// when we have a way to package up verilog modules for iverilog usage.

task automatic xls_assert(
  input want,
  input got,
  input reg [32*8:0] message
);
  if (want != got) begin
    $display("ERROR: assertion failure @ %t; want %b got %b; %s\n", $time,
             want, got, message);
  end
endtask

task automatic xls_assert_int_eq(
  input integer want,
  input integer got,
  input reg [32 * 8:0] message);
  if (want != got) begin
    $display("ERROR: assertion failure @ %t; want %x got %x; %s\n", $time,
             want, got, message);
  end
endtask
