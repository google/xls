// Copyright 2020 Google LLC
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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/status/statusor.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

class TestDelayEstimator : public DelayEstimator {
 public:
  xabsl::StatusOr<int64> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case OP_PARAM:
      case OP_LITERAL:
      case OP_BIT_SLICE:
      case OP_CONCAT:
        return 0;
      default:
        return 1;
    }
  }
};

// A test for ModuleSimulator which uses generated Verilog.
class ModuleSimulatorCodegenTest : public VerilogTestBase {};

TEST_P(ModuleSimulatorCodegenTest, PassThroughPipeline) {
  Package package(TestName());
  FunctionBuilder fb("pass_through", &package);
  fb.Param("x", package.GetBitsType(8));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          PipelineOptions().use_system_verilog(UseSystemVerilog())));
  ASSERT_EQ(result.signature.proto().pipeline().latency(), 2);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(42, 8);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(42, 8));
}

TEST_P(ModuleSimulatorCodegenTest, PassThroughPipelineBatched) {
  Package package(TestName());
  FunctionBuilder fb("pass_through", &package);
  fb.Param("x", package.GetBitsType(8));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          PipelineOptions().use_system_verilog(UseSystemVerilog())));
  ASSERT_EQ(result.signature.proto().pipeline().latency(), 2);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  // Run various size batches through the module.
  for (int64 batch_size = 0; batch_size < 4; ++batch_size) {
    std::vector<absl::flat_hash_map<std::string, Bits>> input_batches(
        batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      input_batches[i]["x"] = UBits(42 + i, 8);
    }
    std::vector<absl::flat_hash_map<std::string, Bits>> outputs;
    XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.RunBatched(input_batches));

    EXPECT_EQ(outputs.size(), batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      const absl::flat_hash_map<std::string, Bits>& output = outputs[i];
      ASSERT_TRUE(output.contains("out"));
      EXPECT_EQ(output.at("out"), UBits(42 + i, 8));
    }
  }
}

TEST_P(ModuleSimulatorCodegenTest, SingleNegatePipeline) {
  Package package(TestName());
  FunctionBuilder fb("negate", &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  fb.Negate(x);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          PipelineOptions().use_system_verilog(UseSystemVerilog())));
  ASSERT_EQ(result.signature.proto().pipeline().latency(), 2);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(42, 8);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(214, 8));
}

TEST_P(ModuleSimulatorCodegenTest, TripleNegatePipelineBatched) {
  Package package(TestName());
  FunctionBuilder fb("negate", &package);
  auto x = fb.Param("x", package.GetBitsType(8));
  fb.Negate(fb.Negate(fb.Negate(x)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          PipelineOptions().use_system_verilog(UseSystemVerilog())));
  ASSERT_EQ(result.signature.proto().pipeline().latency(), 4);

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  // Run various size batches through the module up to and beyond the length of
  // the pipeline.
  for (int64 batch_size = 0; batch_size < 6; ++batch_size) {
    std::vector<absl::flat_hash_map<std::string, Bits>> input_batches(
        batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      input_batches[i]["x"] = UBits(100 + i, 8);
    }
    std::vector<absl::flat_hash_map<std::string, Bits>> outputs;
    XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.RunBatched(input_batches));

    EXPECT_EQ(outputs.size(), batch_size);
    for (int64 i = 0; i < batch_size; ++i) {
      const absl::flat_hash_map<std::string, Bits>& output = outputs[i];
      ASSERT_TRUE(output.contains("out"));
      EXPECT_EQ(output.at("out"), UBits((-(100 + i)) & 0xff, 8))
          << "Batch size = " << batch_size << ", set " << i;
    }
  }
}

TEST_P(ModuleSimulatorCodegenTest, AddsWithSharedResource) {
  Package package(TestName());
  FunctionBuilder fb("x_plus_y_plus_z_plus_x", &package);
  Type* u32 = package.GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  auto out = x + y + z + x;

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.BuildWithReturnValue(out));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(40)));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          PipelineOptions().use_system_verilog(UseSystemVerilog())));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(42, 32);
  inputs["y"] = UBits(123, 32);
  inputs["z"] = UBits(3, 32);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(210, 32));
}

TEST_P(ModuleSimulatorCodegenTest, PipelinedAdds) {
  Package package(TestName());
  FunctionBuilder fb("x_plus_y_plus_z_plus_x", &package);
  Type* u32 = package.GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  auto out = x + y + z + x;

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.BuildWithReturnValue(out));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(40)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          PipelineOptions().use_system_verilog(UseSystemVerilog())));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(42, 32);
  inputs["y"] = UBits(123, 32);
  inputs["z"] = UBits(3, 32);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(210, 32));
}

TEST_P(ModuleSimulatorCodegenTest, PipelinedAddWithValid) {
  Package package(TestName());
  FunctionBuilder fb("x_plus_y_plus_z_plus_x", &package);
  Type* u32 = package.GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  auto out = x + y + z + x;

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.BuildWithReturnValue(out));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(5)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           PipelineOptions()
                               .valid_control("valid_in", "valid_out")
                               .use_system_verilog(UseSystemVerilog())));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(42, 32);
  inputs["y"] = UBits(123, 32);
  inputs["z"] = UBits(3, 32);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(210, 32));
}

TEST_P(ModuleSimulatorCodegenTest, PipelinedAddWithManualPipelineControl) {
  Package package(TestName());
  FunctionBuilder fb("x_plus_y_plus_z_plus_x", &package);
  Type* u32 = package.GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  auto out = x + y + z + x;

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.BuildWithReturnValue(out));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(5)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, func,
                           PipelineOptions()
                               .manual_control("pipeline_le")
                               .use_system_verilog(UseSystemVerilog())));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(42, 32);
  inputs["y"] = UBits(123, 32);
  inputs["z"] = UBits(3, 32);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(210, 32));
}

TEST_P(ModuleSimulatorCodegenTest, AddTwoTupleElements) {
  Package package(TestName());
  FunctionBuilder fb(TestName(), &package);
  Type* u8 = package.GetBitsType(8);
  auto in = fb.Param("in", package.GetTupleType({u8, u8}));
  auto out = fb.TupleIndex(in, 0) + fb.TupleIndex(in, 1);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.BuildWithReturnValue(out));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(40)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, func,
          PipelineOptions().use_system_verilog(UseSystemVerilog())));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());

  // Run with flat inputs.
  EXPECT_THAT(simulator.RunAndReturnSingleOutput({{"in", UBits(0x1234, 16)}}),
              IsOkAndHolds(UBits(0x46, 8)));

  // Run with wrong width flat inputs.
  EXPECT_THAT(
      simulator.RunAndReturnSingleOutput({{"in", UBits(0x1234, 17)}}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Expected input 'in' to have width 16, has width 17")));

  // Run with Value inputs.
  EXPECT_THAT(simulator.Run({{"in", Value::Tuple({Value(UBits(0x11, 8)),
                                                  Value(UBits(0x78, 8))})}}),
              IsOkAndHolds(Value(UBits(0x89, 8))));

  // Run with wrong-type Value inputs (tuple element wrong width).
  EXPECT_THAT(
      simulator.Run({{"in", Value::Tuple({Value(UBits(0x11, 8)),
                                          Value(UBits(0x78, 9))})}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Input value 'in' is wrong type. Expected '(bits[8], "
                         "bits[8])', got '(bits[8], bits[9])'")));

  // Run with wrong-type Value inputs where the input is the right flattened
  // width.
  EXPECT_THAT(simulator.Run({{"in", Value(UBits(0x1234, 16))}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Input value 'in' is wrong type. Expected "
                                 "'(bits[8], bits[8])', got 'bits[16]'")));
}

TEST_P(ModuleSimulatorCodegenTest, CombinationalModule) {
  Package package(TestName());
  FunctionBuilder fb(TestName(), &package);
  fb.Add(fb.Param("x", package.GetBitsType(8)),
         fb.Param("y", package.GetBitsType(8)));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToCombinationalModuleText(func,
                                /*use_system_verilog=*/UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(42, 8);
  inputs["y"] = UBits(100, 8);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(142, 8));
}

TEST_P(ModuleSimulatorCodegenTest, ReturnLiteral) {
  Package package(TestName());
  FunctionBuilder fb(TestName(), &package);
  fb.Literal(UBits(42, 8));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToCombinationalModuleText(func,
                                /*use_system_verilog=*/UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(
      outputs, simulator.Run(absl::flat_hash_map<std::string, Bits>()));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(42, 8));
}

TEST_P(ModuleSimulatorCodegenTest, ReturnParameter) {
  Package package(TestName());
  FunctionBuilder fb(TestName(), &package);
  fb.Param("x", package.GetBitsType(8));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToCombinationalModuleText(func,
                                /*use_system_verilog=*/UseSystemVerilog()));

  ModuleSimulator simulator(result.signature, result.verilog_text,
                            GetSimulator());
  absl::flat_hash_map<std::string, Bits> inputs;
  inputs["x"] = UBits(42, 8);
  absl::flat_hash_map<std::string, Bits> outputs;
  XLS_ASSERT_OK_AND_ASSIGN(outputs, simulator.Run(inputs));

  EXPECT_EQ(outputs.size(), 1);
  ASSERT_TRUE(outputs.contains("out"));
  EXPECT_EQ(outputs.at("out"), UBits(42, 8));
}

INSTANTIATE_TEST_SUITE_P(ModuleSimulatorCodegenTestInstantiation,
                         ModuleSimulatorCodegenTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<ModuleSimulatorCodegenTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
