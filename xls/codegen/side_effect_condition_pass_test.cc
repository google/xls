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

#include "xls/codegen/side_effect_condition_pass.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_pass_pipeline.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/scheduling/scheduling_pass_pipeline.h"

namespace m = xls::op_matchers;

namespace xls::verilog {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

CodegenPass* DefaultCodegenPassPipeline() {
  static absl::NoDestructor<std::unique_ptr<CodegenCompoundPass>> singleton(
      CreateCodegenPassPipeline());
  return singleton->get();
}

CodegenPass* SideEffectConditionPassOnly() {
  static absl::NoDestructor<SideEffectConditionPass> singleton;
  return singleton.get();
}

std::string_view CodegenPassName(CodegenPass const* pass) {
  if (pass == DefaultCodegenPassPipeline()) {
    return "DefaultCodegenPassPipeline";
  }
  if (pass == SideEffectConditionPassOnly()) {
    return "SideEffectConditionPassOnly";
  }
  // We're seeing an unknown codegen pass, so error
  LOG(FATAL) << "Unknown codegen pass!";
  return "";
}

static CodegenOptions kDefaultCodegenOptions =
    CodegenOptions().clock_name("clk").reset("rst", /*asynchronous=*/false,
                                             /*active_low=*/false,
                                             /*reset_data_path=*/false);
static SchedulingOptions kDefaultSchedulingOptions =
    SchedulingOptions().clock_period_ps(2);

class SideEffectConditionPassTest
    : public testing::TestWithParam<CodegenPass*> {
 protected:
  static std::string_view PackageName() {
    return std::vector<std::string_view>(
               absl::StrSplit(::testing::UnitTest::GetInstance()
                                  ->current_test_info()
                                  ->test_suite_name(),
                              '/'))
        .back();
  }
  static absl::StatusOr<bool> Run(
      Package* p, CodegenOptions codegen_options = kDefaultCodegenOptions,
      SchedulingOptions scheduling_options = kDefaultSchedulingOptions) {
    // First, schedule.
    std::unique_ptr<SchedulingPass> scheduling_pipeline =
        CreateSchedulingPassPipeline();
    XLS_RET_CHECK(p->GetTop().has_value());
    FunctionBase* top = p->GetTop().value();
    auto scheduling_unit =
        SchedulingUnit::CreateForSingleFunction(p->GetTop().value());
    SchedulingPassOptions scheduling_pass_options{
        .scheduling_options = std::move(scheduling_options),
        .delay_estimator = GetDelayEstimator("unit").value()};
    SchedulingPassResults scheduling_results;
    XLS_RETURN_IF_ERROR(scheduling_pipeline
                            ->Run(&scheduling_unit, scheduling_pass_options,
                                  &scheduling_results)
                            .status());
    const PipelineSchedule& schedule = scheduling_unit.schedules().at(top);
    XLS_ASSIGN_OR_RETURN(
        CodegenPassUnit unit,
        FunctionBaseToPipelinedBlock(schedule, codegen_options, top));
    CodegenPassResults results;
    CodegenPassOptions codegen_pass_option{.codegen_options = codegen_options,
                                           .schedule = schedule};
    return GetParam()->Run(&unit, codegen_pass_option, &results);
  }
  static absl::StatusOr<std::vector<std::string>> RunInterpreterWithEvents(
      Block* block,
      absl::Span<absl::flat_hash_map<std::string, Value> const> inputs) {
    InterpreterBlockEvaluator evaluator;
    XLS_ASSIGN_OR_RETURN(std::unique_ptr<BlockContinuation> continuation,
                         evaluator.NewContinuation(block));
    std::vector<std::string> traces;

    std::vector<absl::flat_hash_map<std::string, Value>> outputs;
    for (const absl::flat_hash_map<std::string, Value>& input_set : inputs) {
      XLS_RETURN_IF_ERROR(continuation->RunOneCycle(input_set));
      XLS_RETURN_IF_ERROR(InterpreterEventsToStatus(continuation->events()));
      for (const TraceMessage& trace : continuation->events().trace_msgs) {
        traces.push_back(trace.message);
      }
    }
    return traces;
  }
};

TEST_P(SideEffectConditionPassTest, UnchangedWithNoSideEffects) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  FunctionBuilder fb("f", &package);
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  XLS_ASSERT_OK_AND_ASSIGN(Function * top,
                           fb.BuildWithReturnValue(fb.Add(x, y)));
  XLS_ASSERT_OK(package.SetTop(top));

  // Side-effect condition pass should leave unchanged, but other passes will
  // change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(Run(&package, CodegenOptions()
                                .valid_control("in_vld", "out_vld")
                                .clock_name("clk")),
              IsOkAndHolds(should_change));
}

TEST_F(SideEffectConditionPassTest, UnchangedIfCombinationalFunction) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  Type* token = package.GetTokenType();
  FunctionBuilder fb("f", &package);
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue tkn = fb.Param("tkn", token);
  BValue x_gt_y = fb.UGt(x, y);
  BValue assertion = fb.Assert(tkn, x_gt_y, "foo", "bar");
  BValue sum = fb.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * top,
                           fb.BuildWithReturnValue(fb.Tuple({assertion, sum})));
  ASSERT_NE(top, nullptr);
  CodegenOptions codegen_options;
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenPassUnit unit,
      FunctionBaseToCombinationalBlock(top, codegen_options));
  CodegenPassResults results;
  CodegenPassOptions codegen_pass_options{.codegen_options = codegen_options};
  EXPECT_THAT(
      SideEffectConditionPassOnly()->Run(&unit, codegen_pass_options, &results),
      IsOkAndHolds(false));
}

TEST_P(SideEffectConditionPassTest, CombinationalProc) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb("g", "tok", &package);
  BValue recv_data = pb.Receive(in, pb.AfterAll({}));
  recv_data = pb.TupleIndex(recv_data, 1);
  BValue xy_plus_1 =
      pb.Add(pb.UMul(recv_data, recv_data), pb.Literal(UBits(1, 32)),
             SourceInfo(), "xy_plus_1");
  BValue send_tok = pb.Send(out, xy_plus_1);
  pb.Assert(send_tok,
            pb.UGt(xy_plus_1, pb.Literal(UBits(4, 32)), SourceInfo(),
                   "xy_plus_1_gt_4"),
            /*message=*/"bar", /*label=*/"foo", SourceInfo(), "assertion");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build());
  XLS_ASSERT_OK(package.SetTop(top));

  CodegenOptions codegen_options;
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenPassUnit unit,
      FunctionBaseToCombinationalBlock(top, codegen_options));
  CodegenPassResults results;
  CodegenPassOptions codegen_pass_options{.codegen_options = codegen_options};
  EXPECT_THAT(GetParam()->Run(&unit, codegen_pass_options, &results),
              IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("g"));

  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition,
              m::Or(m::Not(m::Name("in_vld")), m::Name("xy_plus_1_gt_4")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"tkn", Value::Token()},
                      {"in", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                      {"out_rdy", Value(UBits(1, 1))},
                  });
  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[6]["in"] = Value(UBits(4, 32));
  inputs[6]["in_vld"] = Value(UBits(1, 1));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[7]["in"] = Value(UBits(0, 32));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[7]["in_vld"] = Value(UBits(1, 1));

  EXPECT_THAT(RunInterpreterWithEvents(block, inputs),
              StatusIs(absl::StatusCode::kAborted, HasSubstr("bar")));
}

TEST_P(SideEffectConditionPassTest, FunctionAssertionWorks) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  FunctionBuilder fb("f", &package);
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue tkn = fb.Param("tkn", package.GetTokenType());
  BValue x_gt_y = fb.UGt(x, y, SourceInfo(), "x_gt_y");
  BValue assertion_bval = fb.Assert(tkn, x_gt_y, /*message=*/"bar",
                                    /*label=*/"foo", SourceInfo(), "assertion");
  BValue sum = fb.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * top, fb.BuildWithReturnValue(fb.Tuple({assertion_bval, sum})));
  XLS_ASSERT_OK(package.SetTop(top));

  // Without setting valid_control, there are no valid signals and assertions
  // won't be updated.
  // Side-effect condition pass should leave unchanged, but other passes will
  // change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(Run(&package), IsOkAndHolds(should_change));

  // Re-run with valid_control set so the assertions can be rewritten.
  // First, remove the previous block.
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK(package.RemoveBlock(block));
  block = nullptr;
  XLS_ASSERT_OK(package.SetTop(top));
  EXPECT_THAT(
      Run(&package, CodegenOptions()
                        .clock_name("clk")
                        .reset("rst", /*asynchronous=*/false,
                               /*active_low=*/false, /*reset_data_path=*/false)
                        .valid_control("in_vld", "out_vld")),
      IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name(HasSubstr("_vld"))),
                               m::Name("x_gt_y"), m::Name("rst")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"tkn", Value::Token()},
                      {"rst", Value(UBits(1, 1))},
                      {"x", Value(UBits(0, 32))},
                      {"y", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                  });
  for (int64_t cycle = 0; cycle < 5; ++cycle) {
    // During reset cycles, put in inputs that would otherwise trigger the
    // assertion.
    inputs[cycle]["x"] = Value(UBits(1, 32));
    inputs[cycle]["y"] = Value(UBits(3, 32));
    inputs[cycle]["in_vld"] = Value(UBits(1, 1));
  }
  for (int64_t cycle = 5; cycle < kNumCycles; ++cycle) {
    inputs[cycle]["rst"] = Value(UBits(0, 1));
  }

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[6]["x"] = Value(UBits(3, 32));
  inputs[6]["y"] = Value(UBits(1, 32));
  inputs[6]["in_vld"] = Value(UBits(1, 1));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[7]["x"] = Value(UBits(1, 32));
  inputs[7]["y"] = Value(UBits(3, 32));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[7]["in_vld"] = Value(UBits(1, 1));
  EXPECT_THAT(RunInterpreterWithEvents(block, inputs),
              StatusIs(absl::StatusCode::kAborted, HasSubstr("bar")));
}

TEST_P(SideEffectConditionPassTest, FunctionTraceWorks) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  FunctionBuilder fb("f", &package);
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue tkn = fb.Param("tkn", package.GetTokenType());
  BValue not_x_gt_y = fb.ULe(x, y, SourceInfo(), "not_x_gt_y");
  BValue trace_bval = fb.Trace(tkn, not_x_gt_y, {x}, "x = {}", /*verbosity=*/0,
                               SourceInfo(), "trace");
  BValue sum = fb.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * top, fb.BuildWithReturnValue(fb.Tuple({trace_bval, sum})));
  XLS_ASSERT_OK(package.SetTop(top));

  // Without setting valid_control, there are no valid signals and traces won't
  // be updated. Side-effect condition pass should leave unchanged, but other
  // passes will change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(Run(&package), IsOkAndHolds(should_change));

  // Re-run with valid_control set so the assertions can be rewritten.
  // First, remove the previous block.
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK(package.RemoveBlock(block));
  block = nullptr;
  EXPECT_THAT(Run(&package,
                  CodegenOptions()
                      .valid_control("in_vld", "out_vld")
                      .clock_name("clk")
                      .reset("rst", /*asynchronous=*/false,
                             /*active_low=*/false, /*reset_data_path=*/false)),
              IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * trace, block->GetNode("trace"));
  ASSERT_NE(trace, nullptr);
  Node* condition = trace->As<xls::Trace>()->condition();
  EXPECT_THAT(condition, m::And(m::Name(HasSubstr("_vld")),
                                m::Name("not_x_gt_y"), m::Not(m::Name("rst"))));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"tkn", Value::Token()},
                      {"rst", Value(UBits(1, 1))},
                      {"x", Value(UBits(0, 32))},
                      {"y", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                  });
  for (int64_t cycle = 0; cycle < 5; ++cycle) {
    // During reset cycles, put in inputs that would otherwise trigger the
    // trace.
    inputs[cycle]["x"] = Value(UBits(1, 32));
    inputs[cycle]["y"] = Value(UBits(3, 32));
    inputs[cycle]["in_vld"] = Value(UBits(1, 1));
  }
  for (int64_t cycle = 5; cycle < kNumCycles; ++cycle) {
    inputs[cycle]["rst"] = Value(UBits(0, 1));
  }

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<std::string> traces,
                           RunInterpreterWithEvents(block, inputs));
  EXPECT_THAT(traces, IsEmpty());

  inputs[6]["x"] = Value(UBits(3, 32));
  inputs[6]["y"] = Value(UBits(1, 32));
  inputs[6]["in_vld"] = Value(UBits(1, 1));

  XLS_ASSERT_OK_AND_ASSIGN(traces, RunInterpreterWithEvents(block, inputs));
  EXPECT_THAT(traces, IsEmpty());

  inputs[7]["x"] = Value(UBits(1, 32));
  inputs[7]["y"] = Value(UBits(3, 32));

  XLS_ASSERT_OK_AND_ASSIGN(traces, RunInterpreterWithEvents(block, inputs));
  EXPECT_THAT(traces, IsEmpty());

  inputs[7]["in_vld"] = Value(UBits(1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(traces, RunInterpreterWithEvents(block, inputs));
  EXPECT_THAT(traces, ElementsAre("x = 1"));
}

TEST_P(SideEffectConditionPassTest, FunctionCoverWorks) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  FunctionBuilder fb("f", &package);
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue not_x_gt_y = fb.ULe(x, y, SourceInfo(), "not_x_gt_y");
  fb.Cover(/*condition=*/not_x_gt_y, /*label=*/"not_x_gt_y", SourceInfo(),
           "cover_");
  XLS_ASSERT_OK_AND_ASSIGN(Function * top,
                           fb.BuildWithReturnValue(fb.Add(x, y)));
  XLS_ASSERT_OK(package.SetTop(top));

  // Without setting valid_control, there are no valid signals and traces won't
  // be updated. Side-effect condition pass should leave unchanged, but other
  // passes will change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(Run(&package), IsOkAndHolds(should_change));

  // Re-run with valid_control set so the assertions can be rewritten.
  // First, remove the previous block.
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK(package.RemoveBlock(block));
  block = nullptr;
  EXPECT_THAT(Run(&package,
                  CodegenOptions()
                      .valid_control("in_vld", "out_vld")
                      .clock_name("clk")
                      .reset("rst", /*asynchronous=*/false,
                             /*active_low=*/false, /*reset_data_path=*/false)),
              IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * cover, block->GetNode("cover_"));
  ASSERT_NE(cover, nullptr);
  Node* condition = cover->As<xls::Cover>()->condition();
  EXPECT_THAT(condition, m::And(m::Name(HasSubstr("_vld")),
                                m::Name("not_x_gt_y"), m::Not(m::Name("rst"))));

  // TODO(google/xls#1126): We don't currently have a good way to get cover
  // events out of the interpreter, so stop testing here.
}

TEST_P(SideEffectConditionPassTest, SingleStageProc) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));
  ProcBuilder pb("f", &package);
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y_recv = pb.Receive(in, pb.AfterAll({}));
  BValue y_token = pb.TupleIndex(y_recv, 0);
  BValue y = pb.TupleIndex(y_recv, 1);
  BValue x_lt_y = pb.ULe(x, y, SourceInfo(), "x_lt_y");
  BValue assertion_bval = pb.Assert(y_token, x_lt_y, /*message=*/"bar",
                                    /*label=*/"foo", SourceInfo(), "assertion");
  BValue sum = pb.Add(x, y);
  pb.Send(out, assertion_bval, sum);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({sum}));
  XLS_ASSERT_OK(package.SetTop(top));

  EXPECT_THAT(
      Run(&package, /*codegen_options=*/kDefaultCodegenOptions,
          /*scheduling_options=*/SchedulingOptions().pipeline_stages(1)),
      IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name(HasSubstr("_stage_done"))),
                               m::Name("x_lt_y"), m::Name("rst")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"rst", Value(UBits(1, 1))},
                      {"in", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                      {"out_rdy", Value(UBits(1, 1))},
                  });
  for (int64_t cycle = 0; cycle < 5; ++cycle) {
    // During reset cycles, put in inputs that would otherwise trigger the
    // assertion.
    inputs[cycle]["in"] = Value(UBits(0, 32));
    inputs[cycle]["in_vld"] = Value(UBits(1, 1));
  }
  for (int64_t cycle = 5; cycle < kNumCycles; ++cycle) {
    inputs[cycle]["rst"] = Value(UBits(0, 1));
  }
  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[6]["in"] = Value(UBits(3, 32));
  inputs[6]["in_vld"] = Value(UBits(1, 1));
  inputs[7]["in"] = Value(UBits(1, 32));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[7]["in_vld"] = Value(UBits(1, 1));
  EXPECT_THAT(RunInterpreterWithEvents(block, inputs),
              StatusIs(absl::StatusCode::kAborted, HasSubstr("bar")));
}

TEST_P(SideEffectConditionPassTest, AssertionInLastStageOfFunction) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  FunctionBuilder fb("f", &package);
  BValue tkn = fb.Param("tkn", package.GetTokenType());
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  BValue xy_plus_1 = fb.Add(fb.UMul(x, y), fb.Literal(UBits(1, 32)),
                            SourceInfo(), "xy_plus_1");
  BValue xy_plus_1_gt_4 = fb.UGt(xy_plus_1, fb.Literal(UBits(4, 32)),
                                 SourceInfo(), "xy_plus_1_gt_4");
  BValue assertion_bval = fb.Assert(tkn, xy_plus_1_gt_4, /*message=*/"bar",
                                    /*label=*/"foo", SourceInfo(), "assertion");
  XLS_ASSERT_OK_AND_ASSIGN(Function * top,
                           fb.BuildWithReturnValue(fb.Tuple(
                               {assertion_bval, xy_plus_1, xy_plus_1_gt_4})));
  XLS_ASSERT_OK(package.SetTop(top));

  // Without setting valid_control, there are no valid signals and assertions
  // won't be updated.
  // Side-effect condition pass should leave unchanged, but other passes will
  // change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(Run(&package), IsOkAndHolds(should_change));

  // Re-run with valid_control set so the assertions can be rewritten.
  // First, remove the previous block.
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK(package.RemoveBlock(block));
  block = nullptr;
  EXPECT_THAT(Run(&package,
                  CodegenOptions()
                      .valid_control("in_vld", "out_vld")
                      .clock_name("clk")
                      .reset("rst", /*asynchronous=*/false,
                             /*active_low=*/false, /*reset_data_path=*/false)),
              IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::RegisterRead(HasSubstr("_valid"))),
                               m::Name("xy_plus_1_gt_4"), m::Name("rst")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"tkn", Value::Token()},
                      {"rst", Value(UBits(1, 1))},
                      {"x", Value(UBits(0, 32))},
                      {"y", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                  });
  for (int64_t cycle = 0; cycle < 5; ++cycle) {
    // During reset cycles, put in inputs that would otherwise trigger the
    // assertion.
    inputs[cycle]["x"] = Value(UBits(2, 32));
    inputs[cycle]["y"] = Value(UBits(1, 32));
    inputs[cycle]["in_vld"] = Value(UBits(1, 1));
  }
  for (int64_t cycle = 5; cycle < kNumCycles; ++cycle) {
    inputs[cycle]["rst"] = Value(UBits(0, 1));
  }

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[6]["x"] = Value(UBits(4, 32));
  inputs[6]["y"] = Value(UBits(4, 32));
  inputs[6]["in_vld"] = Value(UBits(1, 1));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[7]["x"] = Value(UBits(2, 32));
  inputs[7]["y"] = Value(UBits(1, 32));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[7]["in_vld"] = Value(UBits(1, 1));

  EXPECT_THAT(RunInterpreterWithEvents(block, inputs),
              StatusIs(absl::StatusCode::kAborted, HasSubstr("bar")));
}

TEST_P(SideEffectConditionPassTest, AssertionInLastStageOfProc) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb("g", &package);
  BValue x = pb.StateElement("x", Value(UBits(4, 32)));
  BValue recv = pb.Receive(in, pb.AfterAll({}));
  BValue recv_token = pb.TupleIndex(recv, 0);
  BValue recv_data = pb.TupleIndex(recv, 1);
  BValue xy = pb.UMul(x, recv_data);
  BValue xy_plus_1 =
      pb.Add(xy, pb.Literal(UBits(1, 32)), SourceInfo(), "xy_plus_1");
  BValue xy_plus_1_gt_4 = pb.UGt(xy_plus_1, pb.Literal(UBits(4, 32)),
                                 SourceInfo(), "xy_plus_1_gt_4");
  pb.Assert(recv_token, xy_plus_1_gt_4, /*message=*/"bar",
            /*label=*/"foo", SourceInfo(), "assertion");
  pb.Send(out, recv_token, xy_plus_1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({pb.Literal(UBits(1, 32))}));
  XLS_ASSERT_OK(package.SetTop(top));
  EXPECT_THAT(Run(&package), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("g"));

  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition,
              m::Or(m::Not(m::Name(HasSubstr(
                        "_stage_"))),  // can be stage_done or stage_valid
                                       // depending on passes.
                    m::Name(HasSubstr("xy_plus_1_gt_4")), m::Name("rst")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"tkn", Value::Token()},
                      {"rst", Value(UBits(1, 1))},
                      {"in", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                      {"out_rdy", Value(UBits(1, 1))},
                  });
  for (int64_t cycle = 0; cycle < 5; ++cycle) {
    // During reset cycles, put in inputs that would otherwise trigger the
    // assertion.
    inputs[cycle]["in"] = Value(UBits(4, 32));
    inputs[cycle]["in_vld"] = Value(UBits(1, 1));
  }
  for (int64_t cycle = 5; cycle < kNumCycles; ++cycle) {
    inputs[cycle]["rst"] = Value(UBits(0, 1));
  }

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[6]["in"] = Value(UBits(4, 32));
  inputs[6]["in_vld"] = Value(UBits(1, 1));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[7]["in"] = Value(UBits(0, 32));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[7]["in_vld"] = Value(UBits(1, 1));

  EXPECT_THAT(RunInterpreterWithEvents(block, inputs),
              StatusIs(absl::StatusCode::kAborted, HasSubstr("bar")));
}

TEST_P(SideEffectConditionPassTest, IIGreaterThanOne) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_out,
      package.CreateStreamingChannel("in_out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb("ii_greater_than_one", &package);
  pb.proc()->SetInitiationInterval(2);

  BValue x = pb.StateElement("st", Value(UBits(0, 32)));
  BValue send0_token = pb.Send(out, pb.Literal(Value::Token()), x);
  BValue min_delay_token = pb.MinDelay(send0_token, /*delay=*/1);
  BValue recv_tuple = pb.Receive(in, min_delay_token);
  BValue recv_token = pb.TupleIndex(recv_tuple, 0);
  BValue recv_data = pb.TupleIndex(recv_tuple, 1);
  BValue recv_data_lt_5 = pb.ULt(recv_data, pb.Literal(UBits(5, 32)),
                                 SourceInfo(), "receive_data_lt_5");
  BValue assertion_bval = pb.Assert(recv_token, recv_data_lt_5,
                                    /*message=*/"bar",
                                    /*label=*/"foo", SourceInfo(), "assertion");
  pb.Send(in_out, assertion_bval, recv_data);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({recv_data}));
  XLS_ASSERT_OK(package.SetTop(top));

  EXPECT_THAT(Run(&package), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           package.GetBlock("ii_greater_than_one"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name(HasSubstr("_stage_done"))),
                               m::Name("receive_data_lt_5"), m::Name("rst")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {{"rst", Value(UBits(1, 1))},
                   {"in", Value(UBits(0, 32))},
                   {"in_vld", Value(UBits(0, 1))},
                   {"out_rdy", Value(UBits(1, 1))},
                   {"in_out_rdy", Value(UBits(1, 1))}});
  for (int64_t cycle = 0; cycle < 5; ++cycle) {
    // During reset cycles, put in inputs that would otherwise trigger the
    // assertion.
    inputs[cycle]["in"] = Value(UBits(8, 32));
    inputs[cycle]["in_vld"] = Value(UBits(1, 1));
  }
  for (int64_t cycle = 5; cycle < kNumCycles; ++cycle) {
    inputs[cycle]["rst"] = Value(UBits(0, 1));
  }
  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[6]["in"] = Value(UBits(3, 32));
  inputs[6]["in_vld"] = Value(UBits(1, 1));
  inputs[7]["in"] = Value(UBits(10, 32));
  inputs[8]["in"] = Value(UBits(10, 32));

  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[8]["in_vld"] = Value(UBits(1, 1));
  EXPECT_THAT(RunInterpreterWithEvents(block, inputs),
              StatusIs(absl::StatusCode::kAborted, HasSubstr("bar")));
}

TEST_P(SideEffectConditionPassTest, FunctionWithActiveLowReset) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  FunctionBuilder fb("f", &package);
  BValue x = fb.Param("x", u32);
  BValue y = fb.Param("y", u32);
  fb.Cover(/*condition=*/fb.ULt(x, y), /*label=*/"x_lt_y", SourceInfo(),
           /*name=*/"cover_");
  fb.Trace(fb.AfterAll({}), fb.Eq(x, y), {}, "x == y", /*verbosity=*/0,
           SourceInfo(), /*name=*/"trace");
  fb.Assert(fb.AfterAll({}), fb.ULe(x, y), "Saw x > y", "assert_x_le_y",
            SourceInfo(), /*name=*/"assertion");
  XLS_ASSERT_OK_AND_ASSIGN(Function * top,
                           fb.BuildWithReturnValue(fb.Add(x, y)));
  XLS_ASSERT_OK(package.SetTop(top));

  EXPECT_THAT(
      Run(&package, kDefaultCodegenOptions.valid_control("in_vld", "out_vld")
                        .reset("rst", /*asynchronous=*/false,
                               /*active_low=*/true,
                               /*reset_data_path=*/false)),
      IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));

  XLS_ASSERT_OK_AND_ASSIGN(Node * cover, block->GetNode("cover_"));
  ASSERT_NE(cover, nullptr);
  EXPECT_THAT(
      cover->As<xls::Cover>()->condition(),
      m::And(m::Name("in_vld"), m::ULt(m::InputPort("x"), m::InputPort("y")),
             m::InputPort("rst")));

  XLS_ASSERT_OK_AND_ASSIGN(Node * trace, block->GetNode("trace"));
  ASSERT_NE(trace, nullptr);
  EXPECT_THAT(
      trace->As<xls::Trace>()->condition(),
      m::And(m::InputPort("in_vld"),
             m::Eq(m::InputPort("x"), m::InputPort("y")), m::InputPort("rst")));

  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  EXPECT_THAT(assertion->As<xls::Assert>()->condition(),
              m::Or(m::Not(m::InputPort("in_vld")),
                    m::ULe(m::InputPort("x"), m::InputPort("y")),
                    m::Not(m::InputPort("rst"))));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {{"rst", Value(UBits(0, 1))},
                   {"x", Value(UBits(0, 32))},
                   {"y", Value(UBits(1, 32))},  // x < y will trigger the cover
                   {"in_vld", Value(UBits(0, 1))}});
  for (int64_t cycle = 0; cycle < 5; ++cycle) {
    // During reset cycles, put in inputs that would otherwise trigger each of
    // the assertion, trace, and cover.
    inputs[cycle]["x"] = Value(UBits(3, 32));
    inputs[cycle]["y"] = Value(UBits(cycle, 32));
    inputs[cycle]["in_vld"] = Value(UBits(1, 1));
  }
  for (int64_t cycle = 5; cycle < kNumCycles; ++cycle) {
    inputs[cycle]["rst"] =
        Value(UBits(1, 1));  // deassert reset from cycle 5 on
  }
  XLS_EXPECT_OK(RunInterpreterWithEvents(block, inputs).status());

  inputs[6]["x"] = Value(UBits(3, 32));
  inputs[6]["y"] = Value(UBits(3, 32));  // trigger the trace
  inputs[6]["in_vld"] = Value(UBits(1, 1));

  EXPECT_THAT(RunInterpreterWithEvents(block, inputs),
              IsOkAndHolds(UnorderedElementsAre("x == y")));

  inputs[6]["x"] = Value(UBits(4, 32));
  inputs[6]["y"] = Value(UBits(3, 32));  // trigger the assert
  inputs[6]["in_vld"] = Value(UBits(1, 1));

  EXPECT_THAT(RunInterpreterWithEvents(block, inputs),
              StatusIs(absl::StatusCode::kAborted, HasSubstr("x > y")));
}

INSTANTIATE_TEST_SUITE_P(SideEffectConditionPassTestInstantiation,
                         SideEffectConditionPassTest,

                         testing::Values(DefaultCodegenPassPipeline(),
                                         SideEffectConditionPassOnly()),
                         [](const testing::TestParamInfo<CodegenPass*>& info) {
                           return std::string(CodegenPassName(info.param));
                         });

}  // namespace
}  // namespace xls::verilog
