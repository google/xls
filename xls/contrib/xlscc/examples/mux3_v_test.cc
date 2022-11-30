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

#include <cstdio>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/get_runfile_path.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xlscc {
namespace {

using ::testing::HasSubstr;

void IOSignalsPresent(std::string_view verilog_text) {
  EXPECT_THAT(verilog_text, HasSubstr("input wire [1:0] csrs"));

  EXPECT_THAT(verilog_text, HasSubstr("input wire [7:0] mux_in0"));
  EXPECT_THAT(verilog_text, HasSubstr("input wire mux_in0_vld"));
  EXPECT_THAT(verilog_text, HasSubstr("output wire mux_in0_rdy"));

  EXPECT_THAT(verilog_text, HasSubstr("input wire [7:0] mux_in1"));
  EXPECT_THAT(verilog_text, HasSubstr("input wire mux_in1_vld"));
  EXPECT_THAT(verilog_text, HasSubstr("output wire mux_in1_rdy"));

  EXPECT_THAT(verilog_text, HasSubstr("input wire [7:0] mux_in2"));
  EXPECT_THAT(verilog_text, HasSubstr("input wire mux_in2_vld"));
  EXPECT_THAT(verilog_text, HasSubstr("output wire mux_in2_rdy"));

  EXPECT_THAT(verilog_text, HasSubstr("output wire [7:0] mux_out"));
  EXPECT_THAT(verilog_text, HasSubstr("output wire mux_out_vld"));
  EXPECT_THAT(verilog_text, HasSubstr("input wire mux_out_rdy"));
}

TEST(Mux3VerilogTest, BasicCombSignaturePresent) {
  constexpr const char kVerPath[] = "xls/contrib/xlscc/examples/mux3_comb_v.sv";

  {
    SCOPED_TRACE("BasicCombSignaturePresent");

    XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path ir_path,
                             xls::GetXlsRunfilePath(kVerPath));
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog_text,
                             xls::GetFileContents(ir_path));
    XLS_VLOG_LINES(2, verilog_text);

    IOSignalsPresent(verilog_text);
    EXPECT_THAT(verilog_text, HasSubstr("module mux3_comb("));
  }
}

TEST(Mux3VerilogTest, BasicPipelineSignaturePresent) {
  constexpr const char kVerPath[] =
      "xls/contrib/xlscc/examples/mux3_stages_5_v.sv";

  {
    SCOPED_TRACE("BasicPipelineSignaturePresent");

    XLS_ASSERT_OK_AND_ASSIGN(std::filesystem::path ir_path,
                             xls::GetXlsRunfilePath(kVerPath));
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog_text,
                             xls::GetFileContents(ir_path));
    XLS_VLOG_LINES(2, verilog_text);

    IOSignalsPresent(verilog_text);

    EXPECT_THAT(verilog_text, HasSubstr("module mux3_stages_5("));
    EXPECT_THAT(verilog_text, HasSubstr("input wire clk"));
    EXPECT_THAT(verilog_text, HasSubstr("input wire rst"));
    EXPECT_THAT(verilog_text, HasSubstr("always_ff @ (posedge clk)"));

    // Has 5 pipeline stages
    //  x | p0 | p1 | p2 | p3
    EXPECT_THAT(verilog_text, HasSubstr("p0_valid"));
    EXPECT_THAT(verilog_text, HasSubstr("p1_valid"));
    EXPECT_THAT(verilog_text, HasSubstr("p2_valid"));
    EXPECT_THAT(verilog_text, HasSubstr("p3_valid"));
    EXPECT_THAT(verilog_text, Not(HasSubstr("p4_valid")));
  }
}

}  // namespace
}  // namespace xlscc
