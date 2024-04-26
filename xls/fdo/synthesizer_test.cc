// Copyright 2023 The XLS Authors
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

#include "xls/fdo/synthesizer.h"

#include <cstdint>
#include <string>
#include <string_view>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"

namespace xls {
namespace {

class FakeSynthesizer : public synthesis::Synthesizer {
 public:
  FakeSynthesizer() : synthesis::Synthesizer("FakeSynthesizer") {}

  absl::StatusOr<int64_t> SynthesizeVerilogAndGetDelay(
      std::string_view verilog_text,
      std::string_view top_module_name) const override {
    return 0;
  }
};

class SynthesizerTest : public IrTestBase {
 public:
  FakeSynthesizer synthesizer_;
};

TEST_F(SynthesizerTest, FunctionBaseToVerilogSimple) {
  const std::string ir_text = R"(
package p

fn test(i0: bits[3], i1: bits[3]) -> bits[3] {
  ret add.3: bits[3] = add(i0, i1, id=3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("test"));

  const std::string expected_verilog_text = R"(module test(
  input wire [2:0] i0,
  input wire [2:0] i1,
  output wire [2:0] out
);
  wire [2:0] add_6;
  assign add_6 = i0 + i1;
  assign out = add_6;
endmodule
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string actual_verilog_text,
      synthesizer_.FunctionBaseToVerilog(function,
                                         /*flop_inputs_outputs=*/false));
  EXPECT_EQ(actual_verilog_text, expected_verilog_text);

  const std::string expected_ff_verilog_text = R"(module test(
  input wire clk,
  input wire [2:0] i0,
  input wire [2:0] i1,
  output wire [2:0] out
);
  reg [2:0] p0_i0;
  reg [2:0] p0_i1;
  reg [2:0] p1_add_14;
  wire [2:0] add_14;
  assign add_14 = p0_i0 + p0_i1;
  always @ (posedge clk) begin
    p0_i0 <= i0;
    p0_i1 <= i1;
    p1_add_14 <= add_14;
  end
  assign out = p1_add_14;
endmodule
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string actual_ff_verilog_text,
      synthesizer_.FunctionBaseToVerilog(function,
                                         /*flop_inputs_outputs=*/true));
  EXPECT_EQ(actual_ff_verilog_text, expected_ff_verilog_text);
}

TEST_F(SynthesizerTest, FunctionBaseToVerilogWithLiveIn) {
  std::string ir_text = R"(
package p

fn test(add_1: bits[3], i1: bits[3]) -> bits[3] {
  ret sub.3: bits[3] = sub(add_1, i1, id=3)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("test"));

  const std::string expected_verilog_text = R"(module test(
  input wire [2:0] add_1,
  input wire [2:0] i1,
  output wire [2:0] out
);
  wire [2:0] sub_6;
  assign sub_6 = add_1 - i1;
  assign out = sub_6;
endmodule
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string actual_verilog_text,
      synthesizer_.FunctionBaseToVerilog(function,
                                         /*flop_inputs_outputs=*/false));
  EXPECT_EQ(actual_verilog_text, expected_verilog_text);
}

TEST_F(SynthesizerTest, FunctionBaseToVerilogWithLiveOut) {
  std::string ir_text = R"(
package p

fn test(i0: bits[3], i1: bits[3]) -> (bits[3], bits[3]) {
  add.3: bits[3] = add(i0, i1, id=3)
  sub.4: bits[3] = sub(add.3, i1, id=4)
  ret tuple.5: (bits[3], bits[3]) = tuple(add.3, sub.4, id=5)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("test"));

  const std::string expected_verilog_text = R"(module test(
  input wire [2:0] i0,
  input wire [2:0] i1,
  output wire [5:0] out
);
  wire [2:0] add_8;
  wire [2:0] sub_9;
  assign add_8 = i0 + i1;
  assign sub_9 = add_8 - i1;
  assign out = {add_8, sub_9};
endmodule
)";
  XLS_ASSERT_OK_AND_ASSIGN(
      std::string actual_verilog_text,
      synthesizer_.FunctionBaseToVerilog(function,
                                         /*flop_inputs_outputs=*/false));
  EXPECT_EQ(actual_verilog_text, expected_verilog_text);
}

}  // namespace
}  // namespace xls
