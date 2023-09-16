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

#include "xls/fdo/extract_nodes.h"

#include <string>

#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"

namespace xls {
namespace {

class ExtractNodesTest : public IrTestBase {};

TEST_F(ExtractNodesTest, SimpleExtraction) {
  std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  ret add.1: bits[3] = add(i0, i1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));

  absl::flat_hash_set<Node*> nodes({FindNode("i0", function),
                                    FindNode("i1", function),
                                    FindNode("add.1", function)});

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string verilog_text,
      ExtractNodesAndGetVerilog(nodes, "test", /*flop_inputs_outputs=*/false));
  std::string expected_verilog_text = R"(module test(
  input wire [2:0] i0,
  input wire [2:0] i1,
  output wire [2:0] out
);
  wire [2:0] add_6;
  assign add_6 = i0 + i1;
  assign out = add_6;
endmodule
)";
  EXPECT_EQ(verilog_text, expected_verilog_text);

  XLS_ASSERT_OK_AND_ASSIGN(std::string ff_verilog_text,
                           ExtractNodesAndGetVerilog(
                               nodes, "ff_test", /*flop_inputs_outputs=*/true));
  std::string expected_ff_verilog_text = R"(module ff_test(
  input wire clk,
  input wire [2:0] i0,
  input wire [2:0] i1,
  output wire [2:0] out
);
  reg [2:0] p0_i0;
  reg [2:0] p0_i1;
  reg [2:0] p1_add_10;
  wire [2:0] add_10;
  assign add_10 = p0_i0 + p0_i1;
  always @ (posedge clk) begin
    p0_i0 <= i0;
    p0_i1 <= i1;
    p1_add_10 <= add_10;
  end
  assign out = p1_add_10;
endmodule
)";
  EXPECT_EQ(ff_verilog_text, expected_ff_verilog_text);
}

TEST_F(ExtractNodesTest, ExtractionWithLivein) {
  std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  add.1: bits[3] = add(i0, i1)
  ret sub.2: bits[3] = sub(add.1, i1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));

  absl::flat_hash_set<Node*> nodes({FindNode("sub.2", function)});

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string verilog_text,
      ExtractNodesAndGetVerilog(nodes, "test", /*flop_inputs_outputs=*/false));
  std::string expected_verilog_text = R"(module test(
  input wire [2:0] add_1,
  input wire [2:0] i1,
  output wire [2:0] out
);
  wire [2:0] sub_6;
  assign sub_6 = add_1 - i1;
  assign out = sub_6;
endmodule
)";
  EXPECT_EQ(verilog_text, expected_verilog_text);
}

TEST_F(ExtractNodesTest, ExtractionWithLiveout) {
  std::string ir_text = R"(
package p

fn main(i0: bits[3], i1: bits[3]) -> bits[3] {
  add.1: bits[3] = add(i0, i1)
  sub.2: bits[3] = sub(add.1, i1)
  ret or.3: bits[3] = or(sub.2, add.1)
}
)";

  XLS_ASSERT_OK_AND_ASSIGN(auto package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Function * function, package->GetFunction("main"));

  absl::flat_hash_set<Node*> nodes(
      {FindNode("add.1", function), FindNode("sub.2", function)});

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string verilog_text,
      ExtractNodesAndGetVerilog(nodes, "test", /*flop_inputs_outputs=*/false));
  std::string expected_verilog_text = R"(module test(
  input wire [2:0] i0,
  input wire [2:0] i1,
  output wire [2:0] out
);
  wire [2:0] add_7;
  wire [2:0] sub_8;
  assign add_7 = i0 + i1;
  assign sub_8 = add_7 - i1;
  assign out = sub_8;
endmodule
)";
  EXPECT_EQ(verilog_text, expected_verilog_text);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string all_liveouts_verilog_text,
      ExtractNodesAndGetVerilog(nodes, "test", /*flop_inputs_outputs=*/false,
                                /*return_all_liveouts=*/true));
  std::string expected_all_liveouts_verilog_text = R"(module test(
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
  EXPECT_EQ(all_liveouts_verilog_text, expected_all_liveouts_verilog_text);
}

}  // namespace
}  // namespace xls
