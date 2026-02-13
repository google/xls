// Copyright 2025 The XLS Authors
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

#include "xls/codegen_v_1_5/function_io_lowering_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_finalization_pass.h"
#include "xls/codegen_v_1_5/flow_control_insertion_pass.h"
#include "xls/codegen_v_1_5/pipeline_register_insertion_pass.h"
#include "xls/codegen_v_1_5/scheduled_block_conversion_pass.h"
#include "xls/codegen_v_1_5/scheduling_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/interpreter/function_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/schedule.h"

namespace xls::codegen {
namespace {

using absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Property;

struct FunctionalTestOptions {
  std::optional<std::string> input_valid = std::nullopt;
  std::optional<std::string> output_valid = std::nullopt;
  int64_t input_valid_delay = 0;
};

class FunctionIOLoweringPassTest : public IrTestBase {
 protected:
  FunctionIOLoweringPassTest() = default;

  absl::StatusOr<BlockConversionPassOptions> CreateBlockConversionPassOptions(
      Package* p, int64_t pipeline_stages,
      ::xls::verilog::CodegenOptions codegen_options =
          ::xls::verilog::CodegenOptions().clock_name("clk")) {
    TestDelayEstimator delay_estimator;
    XLS_ASSIGN_OR_RETURN(
        SchedulingResult scheduling_result,
        Schedule(
            p,
            SchedulingOptions().opt_level(0).pipeline_stages(pipeline_stages),
            &delay_estimator));
    return BlockConversionPassOptions{
        .codegen_options = std::move(codegen_options),
        .package_schedule = std::move(scheduling_result.package_schedule),
    };
  }

  absl::StatusOr<ScheduledBlock*> CreateScheduledBlock(
      Package* p, std::string_view function_name,
      const BlockConversionPassOptions& options) {
    SchedulingPass scheduling_pass;
    PassResults scheduling_results;
    XLS_RETURN_IF_ERROR(
        scheduling_pass.Run(p, options, &scheduling_results).status());

    PassResults results;
    XLS_RETURN_IF_ERROR(
        ScheduledBlockConversionPass().Run(p, options, &results).status());
    XLS_ASSIGN_OR_RETURN(Block * block, p->GetBlock(function_name));
    return absl::down_cast<ScheduledBlock*>(block);
  }

  // Runs PipelineRegisterInsertionPass and BlockFinalizationPass, then
  // verify & return the output of block interpretation.
  absl::StatusOr<Value> RunFunctionalTest(
      Package* p, Block* block, const BlockConversionPassOptions& pass_options,
      absl::Span<const Value> inputs, int64_t expected_latency,
      FunctionalTestOptions options = {}) {
    std::string block_name = block->name();
    PassResults pass_results;
    XLS_RETURN_IF_ERROR(PipelineRegisterInsertionPass()
                            .Run(p, pass_options, &pass_results)
                            .status());
    XLS_RETURN_IF_ERROR(FlowControlInsertionPass()
                            .Run(p, pass_options, &pass_results)
                            .status());
    XLS_RETURN_IF_ERROR(
        BlockFinalizationPass().Run(p, pass_options, &pass_results).status());
    XLS_ASSIGN_OR_RETURN(Block * sim_block, p->GetBlock(block_name));

    std::vector<absl::flat_hash_map<std::string, Value>> in_values;
    const int64_t num_cycles = options.input_valid_delay + expected_latency +
                               (options.output_valid.has_value() ? 1 : 0);
    const int64_t output_cycle =
        options.input_valid_delay + expected_latency - 1;
    in_values.resize(num_cycles);
    const bool has_reset = pass_options.codegen_options.reset().has_value();
    const bool has_input_valid = options.input_valid.has_value();
    const int64_t value_offset =
        (has_reset ? 1 : 0) + (has_input_valid ? 1 : 0);
    XLS_RET_CHECK_EQ(sim_block->GetInputPorts().size(),
                     value_offset + inputs.size());
    for (int64_t i = 0; i < sim_block->GetInputPorts().size(); ++i) {
      std::string_view input_name = sim_block->GetInputPorts()[i]->name();
      if (has_reset && i == 0) {
        Value reset_disabled = Value(UBits(
            pass_options.codegen_options.reset()->active_low() ? 1 : 0, 1));
        for (int64_t j = 0; j < num_cycles; ++j) {
          in_values[j].emplace(input_name, reset_disabled);
        }
        continue;
      }

      Value input_value;
      if (i < value_offset) {
        XLS_RET_CHECK(has_input_valid);
        XLS_RET_CHECK_EQ(i, value_offset - 1);
        // input_valid should be 1 whenever we have a valid input.
        input_value = Value(UBits(1, 1));
      } else {
        input_value = inputs[i - value_offset];
      }
      for (int64_t j = 0; j <= options.input_valid_delay; ++j) {
        if (i == value_offset - 1 && j < options.input_valid_delay) {
          in_values[j].emplace(input_name, Value(UBits(0, 1)));
        } else {
          in_values[j].emplace(input_name, input_value);
        }
      }
      for (int64_t j = options.input_valid_delay + 1; j < num_cycles; ++j) {
        in_values[j].emplace(
            input_name, ZeroOfType(sim_block->GetInputPorts()[i]->port_type()));
      }
    }

    std::vector<absl::flat_hash_map<std::string, Value>> results;
    XLS_ASSIGN_OR_RETURN(results,
                         InterpretSequentialBlock(sim_block, in_values));

    if (results.size() != num_cycles) {
      return absl::InternalError(absl::StrFormat(
          "Expected %d cycles of results, got %d", num_cycles, results.size()));
    }
    if (options.output_valid.has_value()) {
      // We should see a single output-valid pulse at the end of the latency.
      for (int64_t cycle = 0; cycle < num_cycles; ++cycle) {
        if (cycle == output_cycle) {
          if (results[cycle].at(*options.output_valid) != Value(UBits(1, 1))) {
            return absl::InternalError(absl::StrFormat(
                "Expected output-valid signal at cycle %d", cycle));
          }
        } else {
          if (results[cycle].at(*options.output_valid) == Value(UBits(1, 1))) {
            return absl::InternalError(absl::StrFormat(
                "Expected no output-valid signal at cycle %d (%s return)",
                cycle, (cycle < output_cycle) ? "before" : "after"));
          }
        }
      }
    }
    return results[output_cycle].at(
        pass_options.codegen_options.output_port_name());
  }
};

TEST_F(FunctionIOLoweringPassTest, SingleStage) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32]) -> bits[32] {
   ret identity.1: bits[32] = identity(x)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(42, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockConversionPassOptions options,
      CreateBlockConversionPassOptions(p.get(), /*pipeline_stages=*/1));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "x")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 1);

  EXPECT_THAT(RunFunctionalTest(p.get(), sb, options, inputs,
                                /*expected_latency=*/1),
              IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, SingleStageWithInputValid) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32]) -> bits[32] {
   ret identity.1: bits[32] = identity(x)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(42, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockConversionPassOptions options,
      CreateBlockConversionPassOptions(
          p.get(), /*pipeline_stages=*/1,
          ::xls::verilog::CodegenOptions().clock_name("clk").valid_control(
              "x_valid", std::nullopt)));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "x_valid"),
                          Property(&PortNode::GetName, "x")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 1);

  EXPECT_THAT(
      RunFunctionalTest(p.get(), sb, options, inputs,
                        /*expected_latency=*/1, {.input_valid = "x_valid"}),
      IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, SingleStageWithIOValid) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32]) -> bits[32] {
   ret identity.1: bits[32] = identity(x)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(42, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(BlockConversionPassOptions options,
                           CreateBlockConversionPassOptions(
                               p.get(), /*pipeline_stages=*/1,
                               ::xls::verilog::CodegenOptions()
                                   .clock_name("clk")
                                   .reset("rst", false, false, false)
                                   .valid_control("x_valid", "out_valid")));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "rst"),
                          Property(&PortNode::GetName, "x_valid"),
                          Property(&PortNode::GetName, "x")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out_valid"),
                          Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 1);

  EXPECT_THAT(RunFunctionalTest(
                  p.get(), sb, options, inputs,
                  /*expected_latency=*/1,
                  {.input_valid = "x_valid", .output_valid = "out_valid"}),
              IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, MultiStage) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32]) -> bits[32] {
   literal.1: bits[32] = literal(value=1)
   shll.2: bits[32] = shll(x, literal.1)
   shrl.3: bits[32] = shrl(shll.2, literal.1)
   ret identity.4: bits[32] = identity(shrl.3)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(42, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockConversionPassOptions options,
      CreateBlockConversionPassOptions(p.get(), /*pipeline_stages=*/3));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "x")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 3);

  EXPECT_THAT(RunFunctionalTest(p.get(), sb, options, inputs,
                                /*expected_latency=*/3),
              IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, MultiStageWithInputValid) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32]) -> bits[32] {
   literal.1: bits[32] = literal(value=1)
   shll.2: bits[32] = shll(x, literal.1)
   shrl.3: bits[32] = shrl(shll.2, literal.1)
   ret identity.4: bits[32] = identity(shrl.3)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(42, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockConversionPassOptions options,
      CreateBlockConversionPassOptions(
          p.get(), /*pipeline_stages=*/3,
          ::xls::verilog::CodegenOptions().clock_name("clk").valid_control(
              "x_valid", std::nullopt)));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "x_valid"),
                          Property(&PortNode::GetName, "x")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 3);

  EXPECT_THAT(
      RunFunctionalTest(p.get(), sb, options, inputs,
                        /*expected_latency=*/3, {.input_valid = "x_valid"}),
      IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, MultiStageWithIOValidDelayed) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32]) -> bits[32] {
   literal.1: bits[32] = literal(value=1)
   shll.2: bits[32] = shll(x, literal.1)
   shrl.3: bits[32] = shrl(shll.2, literal.1)
   ret identity.4: bits[32] = identity(shrl.3)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(42, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(BlockConversionPassOptions options,
                           CreateBlockConversionPassOptions(
                               p.get(), /*pipeline_stages=*/3,
                               ::xls::verilog::CodegenOptions()
                                   .clock_name("clk")
                                   .reset("rst", false, false, false)
                                   .valid_control("x_valid", "out_valid")));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "rst"),
                          Property(&PortNode::GetName, "x_valid"),
                          Property(&PortNode::GetName, "x")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out_valid"),
                          Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 3);

  EXPECT_THAT(RunFunctionalTest(p.get(), sb, options, inputs,
                                /*expected_latency=*/3,
                                {.input_valid = "x_valid",
                                 .output_valid = "out_valid",
                                 .input_valid_delay = 2}),
              IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, MultiStageWithIOValid) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32]) -> bits[32] {
   literal.1: bits[32] = literal(value=1)
   shll.2: bits[32] = shll(x, literal.1)
   shrl.3: bits[32] = shrl(shll.2, literal.1)
   ret identity.4: bits[32] = identity(shrl.3)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(42, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(BlockConversionPassOptions options,
                           CreateBlockConversionPassOptions(
                               p.get(), /*pipeline_stages=*/3,
                               ::xls::verilog::CodegenOptions()
                                   .clock_name("clk")
                                   .reset("rst", false, false, false)
                                   .valid_control("x_valid", "out_valid")));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "rst"),
                          Property(&PortNode::GetName, "x_valid"),
                          Property(&PortNode::GetName, "x")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out_valid"),
                          Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 3);

  EXPECT_THAT(RunFunctionalTest(
                  p.get(), sb, options, inputs,
                  /*expected_latency=*/3,
                  {.input_valid = "x_valid", .output_valid = "out_valid"}),
              IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, MultiStageMultiInputWithIOValid) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32], y: bits[32]) -> bits[32] {
   literal.1: bits[32] = literal(value=1)
   shll.2: bits[32] = shll(x, literal.1)
   shrl.3: bits[32] = shrl(shll.2, literal.1)
   ret add.4: bits[32] = add(shrl.3, y)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32)), Value(UBits(21, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(63, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(BlockConversionPassOptions options,
                           CreateBlockConversionPassOptions(
                               p.get(), /*pipeline_stages=*/3,
                               ::xls::verilog::CodegenOptions()
                                   .clock_name("clk")
                                   .reset("rst", false, false, false)
                                   .valid_control("in_valid", "out_valid")));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "rst"),
                          Property(&PortNode::GetName, "in_valid"),
                          Property(&PortNode::GetName, "x"),
                          Property(&PortNode::GetName, "y")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out_valid"),
                          Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 3);

  EXPECT_THAT(RunFunctionalTest(
                  p.get(), sb, options, inputs,
                  /*expected_latency=*/3,
                  {.input_valid = "x_valid", .output_valid = "out_valid"}),
              IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, MultiStageMultiInputWithFloppedInputs) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32], y: bits[32]) -> bits[32] {
   literal.1: bits[32] = literal(value=1)
   shll.2: bits[32] = shll(x, literal.1)
   shrl.3: bits[32] = shrl(shll.2, literal.1)
   ret add.4: bits[32] = add(shrl.3, y)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32)), Value(UBits(21, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(63, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockConversionPassOptions options,
      CreateBlockConversionPassOptions(
          p.get(), /*pipeline_stages=*/3,
          ::xls::verilog::CodegenOptions().clock_name("clk").flop_inputs(
              true)));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "x"),
                          Property(&PortNode::GetName, "y")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 3);

  EXPECT_THAT(RunFunctionalTest(p.get(), sb, options, inputs,
                                /*expected_latency=*/4),
              IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, MultiStageMultiInputWithFloppedOutputs) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32], y: bits[32]) -> bits[32] {
   literal.1: bits[32] = literal(value=1)
   shll.2: bits[32] = shll(x, literal.1)
   shrl.3: bits[32] = shrl(shll.2, literal.1)
   ret add.4: bits[32] = add(shrl.3, y)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32)), Value(UBits(21, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(63, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockConversionPassOptions options,
      CreateBlockConversionPassOptions(
          p.get(), /*pipeline_stages=*/3,
          ::xls::verilog::CodegenOptions().clock_name("clk").flop_outputs(
              true)));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "x"),
                          Property(&PortNode::GetName, "y")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 3);

  EXPECT_THAT(RunFunctionalTest(p.get(), sb, options, inputs,
                                /*expected_latency=*/4),
              IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, MultiStageMultiInputWithFloppedIO) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32], y: bits[32]) -> bits[32] {
   literal.1: bits[32] = literal(value=1)
   shll.2: bits[32] = shll(x, literal.1)
   shrl.3: bits[32] = shrl(shll.2, literal.1)
   ret add.4: bits[32] = add(shrl.3, y)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32)), Value(UBits(21, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(63, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockConversionPassOptions options,
      CreateBlockConversionPassOptions(p.get(), /*pipeline_stages=*/3,
                                       ::xls::verilog::CodegenOptions()
                                           .clock_name("clk")
                                           .flop_inputs(true)
                                           .flop_outputs(true)));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "x"),
                          Property(&PortNode::GetName, "y")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 3);

  EXPECT_THAT(RunFunctionalTest(p.get(), sb, options, inputs,
                                /*expected_latency=*/5),
              IsOkAndHolds(expected_output));
}

TEST_F(FunctionIOLoweringPassTest, MultiStageWithFloppedInputsAndValid) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
 fn id_func(x: bits[32], y: bits[32]) -> bits[32] {
   literal.1: bits[32] = literal(value=1)
   shll.2: bits[32] = shll(x, literal.1)
   shrl.3: bits[32] = shrl(shll.2, literal.1)
   ret add.4: bits[32] = add(shrl.3, y)
 }
  )",
                                                       p.get()));
  XLS_ASSERT_OK(p->SetTopByName("id_func"));

  std::vector<Value> inputs = {Value(UBits(42, 32)), Value(UBits(21, 32))};
  XLS_ASSERT_OK_AND_ASSIGN(InterpreterResult<Value> result,
                           InterpretFunction(f, inputs));
  XLS_ASSERT_OK_AND_ASSIGN(Value expected_output,
                           InterpreterResultToStatusOrValue(result));
  ASSERT_EQ(expected_output, Value(UBits(63, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(BlockConversionPassOptions options,
                           CreateBlockConversionPassOptions(
                               p.get(), /*pipeline_stages=*/3,
                               ::xls::verilog::CodegenOptions()
                                   .clock_name("clk")
                                   .reset("rst", false, false, false)
                                   .flop_inputs(true)
                                   .valid_control("in_valid", "out_valid")));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb,
                           CreateScheduledBlock(p.get(), "id_func", options));

  PassResults results;
  ASSERT_THAT(FunctionIOLoweringPass().Run(p.get(), options, &results),
              IsOkAndHolds(true));

  EXPECT_THAT(sb->GetInputPorts(),
              ElementsAre(Property(&PortNode::GetName, "rst"),
                          Property(&PortNode::GetName, "in_valid"),
                          Property(&PortNode::GetName, "x"),
                          Property(&PortNode::GetName, "y")));
  EXPECT_THAT(sb->GetOutputPorts(),
              ElementsAre(Property(&PortNode::GetName, "out_valid"),
                          Property(&PortNode::GetName, "out")));
  EXPECT_EQ(sb->stages().size(), 3);

  EXPECT_THAT(RunFunctionalTest(
                  p.get(), sb, options, inputs,
                  /*expected_latency=*/4,
                  {.input_valid = "in_valid", .output_valid = "out_valid"}),
              IsOkAndHolds(expected_output));
}

}  // namespace
}  // namespace xls::codegen
