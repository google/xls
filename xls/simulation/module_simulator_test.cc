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
    b.AddDataInput("x", 8);
    b.AddDataOutput("out", 8);
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
    b.AddDataInput("x", 8);
    b.AddDataInput("y", 8);
    b.AddDataOutput("out", 8);
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
    b.AddDataInput("x", 8);
    b.AddDataInput("y", 8);
    b.AddDataOutput("out", 8);
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
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

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
  b.AddDataInput("x", 8);
  b.AddDataOutput("out1", 8);
  b.AddDataOutput("out2", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  ModuleSimulator simulator = NewModuleSimulator(text, signature);
  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(7, 8);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

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
  b.AddDataInput("x", 8);
  b.AddDataInput("y", 8);
  b.AddDataOutput("out", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  ModuleSimulator simulator = NewModuleSimulator(text, signature);

  {
    absl::flat_hash_map<std::string, Bits> inputs;
    inputs["x"] = UBits(7, 8);
    EXPECT_THAT(simulator.Run(inputs),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Input 'y' was not passed as an argument")));
  }

  {
    absl::flat_hash_map<std::string, Bits> inputs;
    inputs["x"] = UBits(7, 8);
    inputs["y"] = UBits(7, 32);
    EXPECT_THAT(
        simulator.Run(inputs),
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            HasSubstr("Expected input 'y' to have width 8, has width 32")));
  }

  {
    absl::flat_hash_map<std::string, Bits> inputs;
    inputs["x"] = UBits(1, 8);
    inputs["y"] = UBits(2, 8);
    inputs["z"] = UBits(3, 8);
    EXPECT_THAT(simulator.Run(inputs),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("Unexpected input value named 'z'")));
  }
}

INSTANTIATE_TEST_SUITE_P(ModuleSimulatorTestInstantiation, ModuleSimulatorTest,
                         testing::ValuesIn(kVerilogOnlySimulationTargets),
                         ParameterizedTestName<ModuleSimulatorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
