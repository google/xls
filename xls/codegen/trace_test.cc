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

#include <memory>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/combinational_generator.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/simulation/module_testbench.h"
#include "xls/simulation/module_testbench_thread.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

constexpr char kTestName[] = "trace_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class TraceTest : public VerilogTestBase {};

constexpr char kSimpleTraceText[] = R"(
package SimpleTrace
top fn main(tkn: token, cond: bits[1]) -> token {
  ret trace.1: token = trace(tkn, cond, format="This is a simple trace.", data_operands=[], id=1)
}
)";

TEST_P(TraceTest, CombinationalSimpleTrace) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kSimpleTraceText));
  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  FunctionBase* entry = top.value();
  CodegenOptions options;
  options.use_system_verilog(UseSystemVerilog());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           GenerateCombinationalModule(entry, options));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      NewModuleTestbench(result.verilog_text, result.signature));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", /*initial_value=*/ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();

  // The combinational module doesn't a connected clock, but the clock can still
  // be used to sequence events in time.
  seq.NextCycle().Set("cond", 0);
  tbt->ExpectTrace("This is a simple trace.");
  EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kNotFound,
                                  HasSubstr("This is a simple trace.")));

  seq.NextCycle().Set("cond", 1);
  XLS_ASSERT_OK(tb->Run());

  // Expect a second trace output
  tbt->ExpectTrace("This is a simple trace.");
  EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kNotFound,
                                  HasSubstr("This is a simple trace.")));

  // Trigger a second output by changing cond
  seq.NextCycle().Set("cond", 0);
  seq.NextCycle().Set("cond", 1);
  XLS_ASSERT_OK(tb->Run());

  // Expect a third trace output
  tbt->ExpectTrace("This is a simple trace.");
  seq.NextCycle();

  // Fail to find the third trace output because cond did not change.
  EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kNotFound,
                                  HasSubstr("This is a simple trace.")));
}

// This is just a basic test to ensure that traces in clocked modules generate
// output. See side_effect_condition_pass_test.cc for coverage of more
// interesting scenarios.
TEST_P(TraceTest, ClockedSimpleTraceTest) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kSimpleTraceText));
  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  FunctionBase* entry = top.value();

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(entry, *delay_estimator,
                          SchedulingOptions().pipeline_stages(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(
          schedule, entry,
          BuildPipelineOptions().use_system_verilog(UseSystemVerilog())));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      NewModuleTestbench(result.verilog_text, result.signature));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", /*initial_value=*/ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();

  seq.NextCycle().Set("cond", 0);
  tbt->ExpectTrace("This is a simple trace.");
  EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kNotFound,
                                  HasSubstr("This is a simple trace.")));

  seq.NextCycle().Set("cond", 1);
  // Advance a second cycle so that cond makes it through the pipeline to
  // trigger the trace.
  seq.NextCycle();
  XLS_ASSERT_OK(tb->Run());

  // Expect a second trace output
  tbt->ExpectTrace("This is a simple trace.");
  // Fail to find the second trace because we haven't advanced the clock.
  EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kNotFound,
                                  HasSubstr("This is a simple trace.")));

  // Trigger a second output by advancing the clock even though cond is 0.
  seq.NextCycle().Set("cond", 0);
  XLS_ASSERT_OK(tb->Run());

  // Expect a third trace output
  tbt->ExpectTrace("This is a simple trace.");

  // Fail to find it after advancing the clock because cond was 0 in the
  // previous cycle.
  EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kNotFound,
                                  HasSubstr("This is a simple trace.")));
}

TEST_P(TraceTest, ClockedSimpleTraceTestWithInvertedSimulationMacro) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(kSimpleTraceText));
  std::optional<FunctionBase*> top = package->GetTop();
  ASSERT_TRUE(top.has_value());
  FunctionBase* entry = top.value();

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(entry, *delay_estimator,
                          SchedulingOptions().pipeline_stages(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleGeneratorResult result,
      ToPipelineModuleText(schedule, entry,
                           BuildPipelineOptions()
                               .use_system_verilog(UseSystemVerilog())
                               .set_simulation_macro_name("!SYNTHESIS")));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

INSTANTIATE_TEST_SUITE_P(TraceTestInstantiation, TraceTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<TraceTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
