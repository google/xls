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

#include "xls/simulation/module_simulator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/simulation/verilog_simulators.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Pair;

// A test for ModuleSimulator which uses bare Verilog.
class ModuleSimulatorTest : public VerilogTestBase {
 protected:
  // Returns a Verilog module with a fixed latency interface as a pair of
  // Verilog text and Module signature. Output is the sum of the input and a
  // delayed version of the input. For a correct result the input must be driven
  // for the duration of the computation.
  absl::StatusOr<std::pair<std::string, ModuleSignature>>
  MakeFixedLatencyModule() const {
    const std::string text = R"(
module fixed_latency_3(
  input wire clk,
  input wire [7:0] x,
  output wire [7:0] out
);

  reg [7:0] x_0;
  reg [7:0] x_1;
  reg [7:0] x_2;
  assign out = x_2 + x;

  always @ (posedge clk) begin
    x_0 <= x;
    x_1 <= x_0;
    x_2 <= x_1;
  end

endmodule
)";

    ModuleSignatureBuilder b("fixed_latency_3");
    b.WithClock("clk").WithFixedLatencyInterface(3);
    b.AddDataInputAsBits("x", 8);
    b.AddDataOutputAsBits("out", 8);
    XLS_ASSIGN_OR_RETURN(ModuleSignature signature, b.Build());
    return std::make_pair(text, signature);
  }

  // Returns a combinatorial Verilog module as a pair of Verilog text and Module
  // signature. Output is the difference between the two inputs.
  absl::StatusOr<std::pair<std::string, ModuleSignature>>
  MakeCombinationalModule() const {
    const std::string text =
        R"(
module comb_diff(
  input wire clk,
  input wire [7:0] x,
  input wire [7:0] y,
  output wire [7:0] out
);

  assign out = x - y;

endmodule
)";

    ModuleSignatureBuilder b("comb_diff");
    b.WithCombinationalInterface();
    b.AddDataInputAsBits("x", 8);
    b.AddDataInputAsBits("y", 8);
    b.AddDataOutputAsBits("out", 8);
    XLS_ASSIGN_OR_RETURN(ModuleSignature signature, b.Build());
    return std::make_pair(text, signature);
  }

  // Returns a Verilog module with a ready-valid interface as a pair of Verilog
  // text and Module signature. Output is the difference between the two inputs.
  absl::StatusOr<std::pair<std::string, ModuleSignature>> MakeReadyValidModule()
      const {
    const std::string text =
        R"(
module comb_diff(
  input wire clk,
  input wire [7:0] x,
  input wire [7:0] y,
  output wire [7:0] out
);

  assign out = x - y;

endmodule
)";

    ModuleSignatureBuilder b("comb_diff");
    b.WithCombinationalInterface();
    b.AddDataInputAsBits("x", 8);
    b.AddDataInputAsBits("y", 8);
    b.AddDataOutputAsBits("out", 8);
    XLS_ASSIGN_OR_RETURN(ModuleSignature signature, b.Build());
    return std::make_pair(text, signature);
  }
};

TEST_P(ModuleSimulatorTest, FixedLatency) {
  XLS_ASSERT_OK_AND_ASSIGN(auto verilog_signature, MakeFixedLatencyModule());
  ModuleSimulator simulator =
      NewModuleSimulator(verilog_signature.first, verilog_signature.second);
  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(42, 8);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.RunFunction(inputs));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(84, 8));
}

TEST_P(ModuleSimulatorTest, FixedLatencyBatched) {
  XLS_ASSERT_OK_AND_ASSIGN(auto verilog_signature, MakeFixedLatencyModule());
  ModuleSimulator simulator =
      NewModuleSimulator(verilog_signature.first, verilog_signature.second);

  // Test using bits
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<ModuleSimulator::BitsMap> outputs,
                           simulator.RunBatched({{{"x", UBits(44, 8)}},
                                                 {{"x", UBits(123, 8)}},
                                                 {{"x", UBits(7, 8)}}}));

  EXPECT_EQ(outputs.size(), 3);
  EXPECT_THAT(outputs[0], ElementsAre(Pair("out", UBits(88, 8))));
  EXPECT_THAT(outputs[1], ElementsAre(Pair("out", UBits(246, 8))));
  EXPECT_THAT(outputs[2], ElementsAre(Pair("out", UBits(14, 8))));
}

TEST_P(ModuleSimulatorTest, CombinationalBatched) {
  XLS_ASSERT_OK_AND_ASSIGN(auto verilog_signature, MakeCombinationalModule());
  ModuleSimulator simulator =
      NewModuleSimulator(verilog_signature.first, verilog_signature.second);

  // Test using bits
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ModuleSimulator::BitsMap> outputs,
      simulator.RunBatched({{{"x", UBits(99, 8)}, {"y", UBits(12, 8)}},
                            {{"x", UBits(100, 8)}, {"y", UBits(25, 8)}},
                            {{"x", UBits(255, 8)}, {"y", UBits(155, 8)}}}));

  EXPECT_EQ(outputs.size(), 3);
  EXPECT_THAT(outputs[0], ElementsAre(Pair("out", UBits(87, 8))));
  EXPECT_THAT(outputs[1], ElementsAre(Pair("out", UBits(75, 8))));
  EXPECT_THAT(outputs[2], ElementsAre(Pair("out", UBits(100, 8))));
}

TEST_P(ModuleSimulatorTest, MultipleOutputs) {
  const std::string text = R"(
module delay_3(
  input wire the_clk,
  input wire [7:0] x,
  output wire [7:0] out1,
  output wire [7:0] out2
);

  assign out1 = x + 1;
  assign out2 = x + 123;

endmodule
)";

  ModuleSignatureBuilder b("delay_3");
  b.WithClock("the_clk").WithFixedLatencyInterface(3);
  b.AddDataInputAsBits("x", 8);
  b.AddDataOutputAsBits("out1", 8);
  b.AddDataOutputAsBits("out2", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  ModuleSimulator simulator = NewModuleSimulator(text, signature);
  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(7, 8);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.RunFunction(inputs));

  EXPECT_EQ(outputs.size(), 2);
  ASSERT_TRUE(outputs.contains("out1"));
  ASSERT_TRUE(outputs.contains("out2"));
  EXPECT_EQ(outputs.at("out1"), UBits(8, 8));
  EXPECT_EQ(outputs.at("out2"), UBits(130, 8));
}

TEST_P(ModuleSimulatorTest, BadInputs) {
  const std::string text = R"(
module delay_3(
  input wire clk,
  input wire [7:0] x,
  input wire [7:0] y,
  output wire [7:0] out,
);

  assign out = x + y;

endmodule
)";

  ModuleSignatureBuilder b("delay_3");
  b.WithClock("the_clk").WithFixedLatencyInterface(3);
  b.AddDataInputAsBits("x", 8);
  b.AddDataInputAsBits("y", 8);
  b.AddDataOutputAsBits("out", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  ModuleSimulator simulator = NewModuleSimulator(text, signature);

  {
    absl::flat_hash_map<std::string, Bits> inputs;
    inputs["x"] = UBits(7, 8);
    EXPECT_THAT(simulator.RunFunction(inputs),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Input 'y' was not passed as an argument")));
  }

  {
    absl::flat_hash_map<std::string, Bits> inputs;
    inputs["x"] = UBits(7, 8);
    inputs["y"] = UBits(7, 32);
    EXPECT_THAT(
        simulator.RunFunction(inputs),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("Expected input 'y' to have width 8, has width 32")));
  }

  {
    absl::flat_hash_map<std::string, Bits> inputs;
    inputs["x"] = UBits(1, 8);
    inputs["y"] = UBits(2, 8);
    inputs["z"] = UBits(3, 8);
    EXPECT_THAT(simulator.RunFunction(inputs),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Unexpected input value named 'z'")));
  }
}

TEST_P(ModuleSimulatorTest, RunInputSeriesProcCombinational) {
  const std::string text = R"(
module proc_adder(
  input wire [31:0] operand_0,
  input wire operand_0_vld,
  input wire [31:0] operand_1,
  input wire operand_1_vld,
  input wire result_rdy,
  output wire [31:0] result,
  output wire result_vld,
  output wire operand_0_rdy,
  output wire operand_1_rdy
);
  wire [31:0] result_val;
  assign result_val = operand_0 + operand_1;
  assign result = result_val;
  assign result_vld = operand_0_vld & operand_1_vld;
  assign operand_0_rdy = result_rdy;
  assign operand_1_rdy = result_rdy;
endmodule

)";

  ModuleSignatureBuilder b("proc_adder");
  b.WithCombinationalInterface();
  b.AddDataInputAsBits("operand_0", 32);
  b.AddDataInputAsBits("operand_0_vld", 1);
  b.AddDataInputAsBits("operand_1", 32);
  b.AddDataInputAsBits("operand_1_vld", 1);
  b.AddDataInputAsBits("result_rdy", 1);
  b.AddDataOutputAsBits("result", 32);
  b.AddDataOutputAsBits("result_vld", 1);
  b.AddDataOutputAsBits("operand_0_rdy", 1);
  b.AddDataOutputAsBits("operand_1_rdy", 1);
  b.AddStreamingChannel("operand_0", ChannelOps::kReceiveOnly,
                        FlowControl::kReadyValid, /*fifo_depth=*/42,
                        "operand_0", "operand_0_vld", "operand_0_rdy");
  b.AddStreamingChannel("operand_1", ChannelOps::kReceiveOnly,
                        FlowControl::kReadyValid, /*fifo_depth=*/42,
                        "operand_1", "operand_1_vld", "operand_1_rdy");
  b.AddStreamingChannel("result", ChannelOps::kSendOnly,
                        FlowControl::kReadyValid, /*fifo_depth=*/42, "result",
                        "result_vld", "result_rdy");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  ModuleSimulator simulator = NewModuleSimulator(text, signature);

  absl::flat_hash_map<std::string, int64_t> output_channel_counts;
  output_channel_counts["result"] = 2;
  {
    // Test with Bits.
    absl::flat_hash_map<std::string, std::vector<Bits>> input_values;
    input_values["operand_0"] = {UBits(41, 32), UBits(32, 32)};
    input_values["operand_1"] = {UBits(1, 32), UBits(32, 32)};

    absl::flat_hash_map<std::string, std::vector<Bits>> result_values;
    result_values["result"] = {UBits(42, 32), UBits(64, 32)};
    EXPECT_THAT(
        simulator.RunInputSeriesProc(input_values, output_channel_counts),
        IsOkAndHolds(result_values));
  }
  {
    // Test with Values.
    absl::flat_hash_map<std::string, std::vector<Value>> input_values;
    input_values["operand_0"] = {Value(UBits(41, 32)), Value(UBits(32, 32))};
    input_values["operand_1"] = {Value(UBits(1, 32)), Value(UBits(32, 32))};

    absl::flat_hash_map<std::string, std::vector<Value>> result_values;
    result_values["result"] = {Value(UBits(42, 32)), Value(UBits(64, 32))};
    EXPECT_THAT(
        simulator.RunInputSeriesProc(input_values, output_channel_counts),
        IsOkAndHolds(result_values));
  }
}

TEST_P(ModuleSimulatorTest, RunInputSeriesProcPipeline) {
  const std::string text = R"(
module proc_adder_pipeline(
  input wire clk,
  input wire rst,
  input wire [31:0] operand_0,
  input wire operand_0_vld,
  input wire [31:0] operand_1,
  input wire operand_1_vld,
  input wire result_rdy,
  output wire [31:0] result,
  output wire result_vld,
  output wire operand_0_rdy,
  output wire operand_1_rdy
);
  reg [31:0] p0_operand_0_val;
  reg [31:0] p0_operand_1_val;
  reg p0_valid;
  reg [31:0] __operand_0_reg;
  reg __operand_0_valid_reg;
  reg [31:0] __operand_1_reg;
  reg __operand_1_valid_reg;
  reg [31:0] __result_reg;
  reg __result_valid_reg;
  wire result_valid_inv;
  wire __result_vld_buf;
  wire result_valid_load_en;
  wire result_load_en;
  wire p1_not_valid;
  wire p0_enable;
  wire and_53;
  wire p0_data_enable;
  wire p0_load_en;
  wire operand_0_valid_inv;
  wire operand_1_valid_inv;
  wire operand_0_valid_load_en;
  wire operand_1_valid_load_en;
  wire operand_0_load_en;
  wire operand_1_load_en;
  wire [31:0] result_val;
  assign result_valid_inv = ~__result_valid_reg;
  assign __result_vld_buf = p0_valid;
  assign result_valid_load_en = result_rdy | result_valid_inv;
  assign result_load_en = __result_vld_buf & result_valid_load_en;
  assign p1_not_valid = ~p0_valid;
  assign p0_enable = p0_valid & result_load_en | p1_not_valid;
  assign and_53 = __operand_0_valid_reg & __operand_1_valid_reg;
  assign p0_data_enable = p0_enable & and_53;
  assign p0_load_en = p0_data_enable | rst;
  assign operand_0_valid_inv = ~__operand_0_valid_reg;
  assign operand_1_valid_inv = ~__operand_1_valid_reg;
  assign operand_0_valid_load_en = p0_load_en | operand_0_valid_inv;
  assign operand_1_valid_load_en = p0_load_en | operand_1_valid_inv;
  assign operand_0_load_en = operand_0_vld & operand_0_valid_load_en;
  assign operand_1_load_en = operand_1_vld & operand_1_valid_load_en;
  assign result_val = p0_operand_0_val + p0_operand_1_val;
  always @ (posedge clk) begin
    p0_operand_0_val <= p0_load_en ? __operand_0_reg : p0_operand_0_val;
    p0_operand_1_val <= p0_load_en ? __operand_1_reg : p0_operand_1_val;
  end
  always @ (posedge clk) begin
    if (rst) begin
      p0_valid <= 1'h0;
      __operand_0_reg <= 32'h0000_0000;
      __operand_0_valid_reg <= 1'h0;
      __operand_1_reg <= 32'h0000_0000;
      __operand_1_valid_reg <= 1'h0;
      __result_reg <= 32'h0000_0000;
      __result_valid_reg <= 1'h0;
    end else begin
      p0_valid <= p0_enable ? and_53 : p0_valid;
      __operand_0_reg <= operand_0_load_en ? operand_0 : __operand_0_reg;
      __operand_0_valid_reg <= operand_0_valid_load_en ? operand_0_vld : __operand_0_valid_reg;
      __operand_1_reg <= operand_1_load_en ? operand_1 : __operand_1_reg;
      __operand_1_valid_reg <= operand_1_valid_load_en ? operand_1_vld : __operand_1_valid_reg;
      __result_reg <= result_load_en ? result_val : __result_reg;
      __result_valid_reg <= result_valid_load_en ? __result_vld_buf : __result_valid_reg;
    end
  end
  assign result = __result_reg;
  assign result_vld = __result_valid_reg;
  assign operand_0_rdy = operand_0_load_en;
  assign operand_1_rdy = operand_1_load_en;
endmodule

)";

  ModuleSignatureBuilder b("proc_adder_pipeline");
  b.WithClock("clk");
  b.WithReset("rst", /*asynchronous=*/false, /*active_low=*/false);
  b.WithPipelineInterface(2, 1);
  b.AddDataInputAsBits("operand_0", 32);
  b.AddDataInputAsBits("operand_0_vld", 1);
  b.AddDataInputAsBits("operand_1", 32);
  b.AddDataInputAsBits("operand_1_vld", 1);
  b.AddDataInputAsBits("result_rdy", 1);
  b.AddDataOutputAsBits("result", 32);
  b.AddDataOutputAsBits("result_vld", 1);
  b.AddDataOutputAsBits("operand_0_rdy", 1);
  b.AddDataOutputAsBits("operand_1_rdy", 1);
  b.AddStreamingChannel("operand_0", ChannelOps::kReceiveOnly,
                        FlowControl::kReadyValid, /*fifo_depth=*/42,
                        "operand_0", "operand_0_vld", "operand_0_rdy");
  b.AddStreamingChannel("operand_1", ChannelOps::kReceiveOnly,
                        FlowControl::kReadyValid, /*fifo_depth=*/42,
                        "operand_1", "operand_1_vld", "operand_1_rdy");
  b.AddStreamingChannel("result", ChannelOps::kSendOnly,
                        FlowControl::kReadyValid, /*fifo_depth=*/42, "result",
                        "result_vld", "result_rdy");
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  ModuleSimulator simulator = NewModuleSimulator(text, signature);
  absl::flat_hash_map<std::string, int64_t> output_channel_counts;
  output_channel_counts["result"] = 2;
  {
    // Test with Bits.
    absl::flat_hash_map<std::string, std::vector<Bits>> input_values;
    input_values["operand_0"] = {UBits(41, 32), UBits(32, 32)};
    input_values["operand_1"] = {UBits(1, 32), UBits(32, 32)};

    absl::flat_hash_map<std::string, std::vector<Bits>> result_values;
    result_values["result"] = {UBits(42, 32), UBits(64, 32)};
    EXPECT_THAT(
        simulator.RunInputSeriesProc(input_values, output_channel_counts),
        IsOkAndHolds(result_values));
  }
  {
    // Test with Values.
    absl::flat_hash_map<std::string, std::vector<Value>> input_values;
    input_values["operand_0"] = {Value(UBits(41, 32)), Value(UBits(32, 32))};
    input_values["operand_1"] = {Value(UBits(1, 32)), Value(UBits(32, 32))};

    absl::flat_hash_map<std::string, std::vector<Value>> result_values;
    result_values["result"] = {Value(UBits(42, 32)), Value(UBits(64, 32))};
    EXPECT_THAT(
        simulator.RunInputSeriesProc(input_values, output_channel_counts),
        IsOkAndHolds(result_values));
  }
}

INSTANTIATE_TEST_SUITE_P(ModuleSimulatorTestInstantiation, ModuleSimulatorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<ModuleSimulatorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
