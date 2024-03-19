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
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/events.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/scheduling/scheduling_pass_pipeline.h"

namespace m = xls::op_matchers;

namespace xls::verilog {
namespace {
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

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

static CodegenOptions kDefaultCodegenOptions = CodegenOptions();
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
    codegen_options.clock_name("clk");
    codegen_options.reset("rst", /*asynchronous=*/false, /*active_low=*/false,
                          /*reset_data_path=*/false);
    const PipelineSchedule& schedule = scheduling_unit.schedules().at(top);
    XLS_ASSIGN_OR_RETURN(
        CodegenPassUnit unit,
        FunctionBaseToPipelinedBlock(schedule, codegen_options, top));
    PassResults results;
    CodegenPassOptions codegen_pass_option{.codegen_options = codegen_options,
                                           .schedule = schedule};
    return GetParam()->Run(&unit, codegen_pass_option, &results);
  }
  static absl::StatusOr<std::vector<std::string>> RunInterpreterWithEvents(
      Block* block,
      absl::Span<absl::flat_hash_map<std::string, Value> const> inputs) {
    std::vector<std::string> traces;
    // Initial register state is zero for all registers.
    absl::flat_hash_map<std::string, Value> reg_state;
    for (Register* reg : block->GetRegisters()) {
      reg_state[reg->name()] = ZeroOfType(reg->type());
    }

    std::vector<absl::flat_hash_map<std::string, Value>> outputs;
    for (const absl::flat_hash_map<std::string, Value>& input_set : inputs) {
      XLS_ASSIGN_OR_RETURN(BlockRunResult result,
                           BlockRun(input_set, reg_state, block));
      XLS_RETURN_IF_ERROR(InterpreterEventsToStatus(result.interpreter_events));
      for (TraceMessage& trace : result.interpreter_events.trace_msgs) {
        traces.push_back(std::move(trace.message));
      }
      reg_state = std::move(result.reg_state);
    }
    return traces;
  }
};

TEST_P(SideEffectConditionPassTest, UnchangedWithNoSideEffects) {
  constexpr std::string_view ir_text = R"(package test
top fn f(x: bits[32], y: bits[32]) -> bits[32] {
  ret add.1: bits[32] = add(x, y)
}
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  // Side-effect condition pass should leave unchanged, but other passes will
  // change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(
      Run(package.get(), CodegenOptions().valid_control("in_vld", "out_vld")),
      IsOkAndHolds(should_change));
}

TEST_F(SideEffectConditionPassTest, UnchangedIfCombinationalFunction) {
  constexpr std::string_view ir_text = R"(package test
top fn f(tkn: token, x: bits[32], y: bits[32]) -> (token, bits[32]) {
  x_gt_y: bits[1] = ugt(x, y)
  assertion: token = assert(tkn, x_gt_y, label="foo", message="bar")
  sum: bits[32] = add(x, y)
  ret out: (token, bits[32]) = tuple(assertion, sum)
}
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));
  FunctionBase* top = package->GetTop().value_or(nullptr);
  ASSERT_NE(top, nullptr);
  CodegenOptions codegen_options;
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenPassUnit unit,
      FunctionBaseToCombinationalBlock(top, codegen_options));
  PassResults results;
  CodegenPassOptions codegen_pass_options{.codegen_options = codegen_options};
  EXPECT_THAT(
      SideEffectConditionPassOnly()->Run(&unit, codegen_pass_options, &results),
      IsOkAndHolds(false));
}

TEST_P(SideEffectConditionPassTest, CombinationalProc) {
  constexpr std::string_view ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

top proc g(tok: token, x: (), init={()}) {
  recv_tuple: (token, bits[32]) = receive(tok, channel=in)
  recv_token: token = tuple_index(recv_tuple, index=0)
  recv_data: bits[32] = tuple_index(recv_tuple, index=1)
  xy: bits[32] = umul(recv_data, recv_data)
  literal1: bits[32] = literal(value=1)
  xy_plus_1: bits[32] = add(xy, literal1)
  send: token = send(recv_token, xy_plus_1, channel=out)
  literal4: bits[32] = literal(value=4)
  xy_plus_1_gt_4: bits[1] = ugt(xy_plus_1, literal4)
  assertion: token = assert(send, xy_plus_1_gt_4, label="foo", message="bar")
  next (assertion, x)
}
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));
  FunctionBase* top = package->GetTop().value_or(nullptr);
  ASSERT_NE(top, nullptr);
  CodegenOptions codegen_options;
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenPassUnit unit,
      FunctionBaseToCombinationalBlock(top, codegen_options));
  PassResults results;
  CodegenPassOptions codegen_pass_options{.codegen_options = codegen_options};
  EXPECT_THAT(GetParam()->Run(&unit, codegen_pass_options, &results),
              IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock("g"));

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
  constexpr std::string_view ir_text = R"(package test
top fn f(tkn: token, x: bits[32], y: bits[32]) -> (token, bits[32]) {
  x_gt_y: bits[1] = ugt(x, y)
  assertion: token = assert(tkn, x_gt_y, label="foo", message="bar")
  sum: bits[32] = add(x, y)
  ret out: (token, bits[32]) = tuple(assertion, sum)
}
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  // Without setting valid_control, there are no valid signals and assertions
  // won't be updated.
  // Side-effect condition pass should leave unchanged, but other passes will
  // change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(Run(package.get()), IsOkAndHolds(should_change));

  // Re-run with valid_control set so the assertions can be rewritten.
  XLS_ASSERT_OK_AND_ASSIGN(package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      Run(package.get(), CodegenOptions().valid_control("in_vld", "out_vld")),
      IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition,
              m::Or(m::Not(m::Name(HasSubstr("_vld"))), m::Name("x_gt_y")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"tkn", Value::Token()},
                      {"rst", Value(UBits(1, 1))},
                      {"x", Value(UBits(0, 32))},
                      {"y", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                  });
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
  constexpr std::string_view ir_text = R"(package test
top fn f(tkn: token, x: bits[32], y: bits[32]) -> (token, bits[32]) {
  not_x_gt_y: bits[1] = ule(x, y)
  trace: token = trace(tkn, not_x_gt_y, format="x = {}", data_operands=[x])
  sum: bits[32] = add(x, y)
  ret out: (token, bits[32]) = tuple(trace, sum)
}
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  // Without setting valid_control, there are no valid signals and traces won't
  // be updated. Side-effect condition pass should leave unchanged, but other
  // passes will change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(Run(package.get()), IsOkAndHolds(should_change));

  // Re-run with valid_control set so the assertions can be rewritten.
  XLS_ASSERT_OK_AND_ASSIGN(package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      Run(package.get(), CodegenOptions().valid_control("in_vld", "out_vld")),
      IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * trace, block->GetNode("trace"));
  ASSERT_NE(trace, nullptr);
  Node* condition = trace->As<xls::Trace>()->condition();
  EXPECT_THAT(condition,
              m::And(m::Name(HasSubstr("_vld")), m::Name("not_x_gt_y")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"tkn", Value::Token()},
                      {"rst", Value(UBits(1, 1))},
                      {"x", Value(UBits(0, 32))},
                      {"y", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                  });
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
  constexpr std::string_view ir_text = R"(package test
top fn f(tkn: token, x: bits[32], y: bits[32]) -> (token, bits[32]) {
  not_x_gt_y: bits[1] = ule(x, y)
  cover: token = cover(tkn, not_x_gt_y, label="not_x_gt_y")
  sum: bits[32] = add(x, y)
  ret out: (token, bits[32]) = tuple(cover, sum)
}
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  // Without setting valid_control, there are no valid signals and traces won't
  // be updated. Side-effect condition pass should leave unchanged, but other
  // passes will change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(Run(package.get()), IsOkAndHolds(should_change));

  // Re-run with valid_control set so the assertions can be rewritten.
  XLS_ASSERT_OK_AND_ASSIGN(package, Parser::ParsePackage(ir_text));
  EXPECT_THAT(
      Run(package.get(), CodegenOptions().valid_control("in_vld", "out_vld")),
      IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * cover, block->GetNode("cover"));
  ASSERT_NE(cover, nullptr);
  Node* condition = cover->As<xls::Cover>()->condition();
  EXPECT_THAT(condition,
              m::And(m::Name(HasSubstr("_vld")), m::Name("not_x_gt_y")));

  // TODO(google/xls#1126): We don't currently have a good way to get cover
  // events out of the interpreter, so stop testing here.
}

TEST_P(SideEffectConditionPassTest, SingleStageProc) {
  constexpr std::string_view ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

top proc f(tkn: token, x: bits[32], init={0}) {
  y_recv: (token, bits[32]) = receive(tkn, channel=in)
  y_token: token = tuple_index(y_recv, index=0)
  y: bits[32] = tuple_index(y_recv, index=1)
  x_lt_y: bits[1] = ult(x, y)
  assertion: token = assert(y_token, x_lt_y, label="foo", message="bar")
  sum: bits[32] = add(x, y)
  send_tok: token = send(assertion, sum, channel=out)
  next (send_tok, sum)
}
    )";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  EXPECT_THAT(
      Run(package.get(), /*codegen_options=*/kDefaultCodegenOptions,
          /*scheduling_options=*/SchedulingOptions().pipeline_stages(1)),
      IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name(HasSubstr("_stage_done"))),
                               m::Name("x_lt_y")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"rst", Value(UBits(1, 1))},
                      {"in", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                      {"out_rdy", Value(UBits(1, 1))},
                  });
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
  constexpr std::string_view ir_text = R"(package test
fn f(tkn: token, x: bits[32], y: bits[32]) -> (token, bits[32]) {
  xy: bits[32] = umul(x, y)
  literal1: bits[32] = literal(value=1)
  xy_plus_1: bits[32] = add(xy, literal1)
  literal4: bits[32] = literal(value=4)
  xy_plus_1_gt_4: bits[1] = ugt(xy_plus_1, literal4)
  assertion: token = assert(tkn, xy_plus_1_gt_4, label="foo", message="bar")
  ret out: (token, bits[32]) = tuple(assertion, xy_plus_1)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  // First, codegen the function 'f'. We'll test proc 'g' afterwards.
  XLS_ASSERT_OK(package->SetTop(package->GetFunctionBaseByName("f").value()));

  // Without setting valid_control, there are no valid signals and assertions
  // won't be updated.
  // Side-effect condition pass should leave unchanged, but other passes will
  // change.
  bool should_change = GetParam() != SideEffectConditionPassOnly();
  EXPECT_THAT(Run(package.get()), IsOkAndHolds(should_change));

  // Re-run with valid_control set so the assertions can be rewritten.
  XLS_ASSERT_OK_AND_ASSIGN(package, Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK(package->SetTop(package->GetFunctionBaseByName("f").value()));
  EXPECT_THAT(
      Run(package.get(), CodegenOptions().valid_control("in_vld", "out_vld")),
      IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::RegisterRead(HasSubstr("_valid"))),
                               m::Name("xy_plus_1_gt_4")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"tkn", Value::Token()},
                      {"rst", Value(UBits(1, 1))},
                      {"x", Value(UBits(0, 32))},
                      {"y", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                  });
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
  constexpr std::string_view ir_text = R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

proc g(tok: token, x: bits[32], init={4}) {
  recv_tuple: (token, bits[32]) = receive(tok, channel=in)
  recv_token: token = tuple_index(recv_tuple, index=0)
  recv_data: bits[32] = tuple_index(recv_tuple, index=1)
  xy: bits[32] = umul(x, recv_data)
  literal1: bits[32] = literal(value=1)
  xy_plus_1: bits[32] = add(xy, literal1)
  send: token = send(recv_token, xy_plus_1, channel=out)
  literal4: bits[32] = literal(value=4)
  xy_plus_1_gt_4: bits[1] = ugt(xy_plus_1, literal4)
  assertion: token = assert(send, xy_plus_1_gt_4, label="foo", message="bar")
  next (assertion, literal1)
}
    )";
  // Now test proc 'g'.
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK(package->SetTop(package->GetFunctionBaseByName("g").value()));
  EXPECT_THAT(Run(package.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package->GetBlock("g"));

  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name(HasSubstr("_stage_done"))),
                               m::Name("xy_plus_1_gt_4")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {
                      {"tkn", Value::Token()},
                      {"rst", Value(UBits(1, 1))},
                      {"in", Value(UBits(0, 32))},
                      {"in_vld", Value(UBits(0, 1))},
                      {"out_rdy", Value(UBits(1, 1))},
                  });
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
  constexpr std::string_view ir_text =
      R"(package test
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="")
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")
chan in_out(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

#[initiation_interval(2)]
top proc ii_greater_than_one(tkn: token, st: bits[32], init={0}) {
  send0_token: token = send(tkn, st, channel=out)
  min_delay_token: token = min_delay(send0_token, delay=1)
  receive_tuple: (token, bits[32]) = receive(min_delay_token, channel=in)
  receive_token: token = tuple_index(receive_tuple, index=0)
  receive_data: bits[32] = tuple_index(receive_tuple, index=1)
  literal5: bits[32] = literal(value=5)
  receive_data_lt_5: bits[1] = ult(receive_data, literal5)
  assertion: token = assert(receive_token, receive_data_lt_5, label="foo", message="bar")
  send1_token: token = send(assertion, receive_data, channel=in_out)
  next (send1_token, receive_data)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           Parser::ParsePackage(ir_text));

  EXPECT_THAT(Run(package.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           package->GetBlock("ii_greater_than_one"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name(HasSubstr("_stage_done"))),
                               m::Name("receive_data_lt_5")));

  constexpr int64_t kNumCycles = 10;
  std::vector<absl::flat_hash_map<std::string, Value>> inputs(
      kNumCycles, {{"rst", Value(UBits(1, 1))},
                   {"in", Value(UBits(0, 32))},
                   {"in_vld", Value(UBits(0, 1))},
                   {"out_rdy", Value(UBits(1, 1))},
                   {"in_out_rdy", Value(UBits(1, 1))}});
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

INSTANTIATE_TEST_SUITE_P(SideEffectConditionPassTestInstantiation,
                         SideEffectConditionPassTest,

                         testing::Values(DefaultCodegenPassPipeline(),
                                         SideEffectConditionPassOnly()),
                         [](const testing::TestParamInfo<CodegenPass*>& info) {
                           return std::string(CodegenPassName(info.param));
                         });

}  // namespace
}  // namespace xls::verilog
