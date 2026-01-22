// Copyright 2026 The XLS Authors
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

#include "xls/codegen_v_1_5/side_effect_condition_pass.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_conversion_pass_pipeline.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = xls::op_matchers;

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

enum class CodegenPassType {
  kDefault,
  kSideEffectConditionPassOnly,
};

std::unique_ptr<BlockConversionPass> GetCodegenPass(
    CodegenPassType type, OptimizationContext& context) {
  switch (type) {
    case CodegenPassType::kDefault:
      return CreateBlockConversionPassPipeline(context);
    case CodegenPassType::kSideEffectConditionPassOnly:
      return std::make_unique<SideEffectConditionPass>();
  }
}

std::string_view CodegenPassName(CodegenPassType type) {
  switch (type) {
    case CodegenPassType::kDefault:
      return "DefaultCodegenPassPipeline";
    case CodegenPassType::kSideEffectConditionPassOnly:
      return "SideEffectConditionPassOnly";
  }
  // We're seeing an unknown codegen pass, so error
  LOG(FATAL) << "Unknown codegen pass!";
  return "";
}

static const verilog::CodegenOptions kDefaultCodegenOptions =
    verilog::CodegenOptions().clock_name("clk").reset(
        "rst", /*asynchronous=*/false,
        /*active_low=*/false,
        /*reset_data_path=*/false);

class SideEffectConditionPassTest
    : public testing::TestWithParam<CodegenPassType> {
 protected:
  static std::string_view PackageName() {
    return std::vector<std::string_view>(
               absl::StrSplit(::testing::UnitTest::GetInstance()
                                  ->current_test_info()
                                  ->test_suite_name(),
                              '/'))
        .back();
  }

  static absl::StatusOr<bool> Run(Package* p,
                                  BlockConversionPassOptions options) {
    OptimizationContext optimization_context;
    PassResults results;
    return GetCodegenPass(GetParam(), optimization_context)
        ->Run(p, options, &results);
  }

  static BlockConversionPassOptions CreateBlockConversionPassOptions(
      ::xls::verilog::CodegenOptions codegen_options =
          ::xls::verilog::CodegenOptions()) {
    if (codegen_options.clock_name().value_or("").empty()) {
      codegen_options.clock_name("clk");
    }
    return BlockConversionPassOptions{
        .codegen_options = std::move(codegen_options),
        .package_schedule = {},
    };
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
      const std::vector<std::string> msgs =
          continuation->events().GetTraceMessageStrings();
      traces.insert(traces.end(), msgs.begin(), msgs.end());
    }
    return traces;
  }
};

TEST_P(SideEffectConditionPassTest, UnchangedWithNoSideEffects) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  ScheduledBlockBuilder bb("f", &package);
  bb.ResetPort("rst", {.asynchronous = false, .active_low = false});
  Function* source;
  {
    auto owned_source = std::make_unique<Function>("__f_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->MakeNodeWithName<Param>(SourceInfo(), u32, "x"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_y, source->MakeNodeWithName<Param>(SourceInfo(), u32, "y"));
  BValue x = bb.SourceNode(source_x);
  BValue y = bb.SourceNode(source_y);
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  BValue p1_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p1_input_valid, bb.Literal(UBits(1, 1)));
  bb.SetSourceReturnValue(bb.Add(x, y).node());
  BValue p1_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p1_aiv, bb.And(p1_input_valid, p1_aiv));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * top, bb.Build());
  XLS_ASSERT_OK(package.SetTop(top));

  // Side-effect condition pass should leave unchanged, but other passes will
  // change.
  bool should_change =
      GetParam() != CodegenPassType::kSideEffectConditionPassOnly;
  BlockConversionPassOptions options = {
      .codegen_options = verilog::CodegenOptions()
                             .valid_control("in_vld", "out_vld")
                             .clock_name("clk")
                             .reset("rst", false, false, false),
  };
  EXPECT_THAT(Run(&package, options), IsOkAndHolds(should_change));
}

TEST_F(SideEffectConditionPassTest, UnchangedIfCombinationalFunction) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  Type* token = package.GetTokenType();
  ScheduledBlockBuilder bb("f", &package);
  Function* source;
  {
    auto owned_source = std::make_unique<Function>("__f_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->MakeNodeWithName<Param>(SourceInfo(), u32, "x"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_y, source->MakeNodeWithName<Param>(SourceInfo(), u32, "y"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_tkn, source->MakeNodeWithName<Param>(
                                                  SourceInfo(), token, "tkn"));
  BValue x = bb.SourceNode(source_x);
  BValue y = bb.SourceNode(source_y);
  BValue tkn = bb.SourceNode(source_tkn);
  BValue x_gt_y = bb.UGt(x, y);
  BValue assertion = bb.Assert(tkn, x_gt_y, "foo", "bar");
  BValue sum = bb.Add(x, y);
  bb.SetSourceReturnValue(bb.Tuple({assertion, sum}).node());
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * top, bb.Build());
  ASSERT_NE(top, nullptr);

  BlockConversionPassOptions options = CreateBlockConversionPassOptions(
      verilog::CodegenOptions().clock_name("clk").reset("rst", false, false,
                                                        false));
  PassResults results;
  OptimizationContext opt_context;
  EXPECT_THAT(
      GetCodegenPass(CodegenPassType::kSideEffectConditionPassOnly, opt_context)
          ->Run(&package, options, &results),
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

  ScheduledBlockBuilder bb("g", &package);
  Proc* source;
  {
    auto owned_source = std::make_unique<Proc>("__g_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  BValue recv = bb.Receive(in, bb.AfterAll({}));
  BValue recv_tkn = bb.TupleIndex(recv, 0);
  BValue recv_data = bb.TupleIndex(recv, 1);
  BValue xy_plus_1 =
      bb.Add(bb.UMul(recv_data, recv_data), bb.Literal(UBits(1, 32)),
             SourceInfo(), "xy_plus_1");
  BValue send_tok = bb.Send(out, recv_tkn, xy_plus_1);
  bb.Assert(send_tok,
            bb.UGt(xy_plus_1, bb.Literal(UBits(4, 32)), SourceInfo(),
                   "xy_plus_1_gt_4"),
            /*message=*/"bar", /*label=*/"foo", SourceInfo(), "assertion");
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  XLS_ASSERT_OK(bb.Build().status());

  BlockConversionPassOptions options = CreateBlockConversionPassOptions();
  options.codegen_options.generate_combinational(true);
  PassResults results;
  OptimizationContext opt_context;
  EXPECT_THAT(
      GetCodegenPass(GetParam(), opt_context)->Run(&package, options, &results),
      IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("g"));

  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name("p0_all_inputs_valid")),
                               m::Name("xy_plus_1_gt_4")));

  if (GetParam() == CodegenPassType::kDefault) {
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
}

TEST_P(SideEffectConditionPassTest, FunctionAssertionWorks) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  ScheduledBlockBuilder bb("f", &package);
  bb.ResetPort("rst", {.asynchronous = false, .active_low = false});
  Function* source;
  {
    auto owned_source = std::make_unique<Function>("__f_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->MakeNodeWithName<Param>(SourceInfo(), u32, "x"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_y, source->MakeNodeWithName<Param>(SourceInfo(), u32, "y"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_tkn,
                           source->MakeNodeWithName<Param>(
                               SourceInfo(), package.GetTokenType(), "tkn"));
  BValue x = bb.SourceNode(source_x);
  BValue y = bb.SourceNode(source_y);
  BValue tkn = bb.SourceNode(source_tkn);
  BValue x_gt_y = bb.UGt(x, y, SourceInfo(), "x_gt_y");
  BValue assertion_bval = bb.Assert(tkn, x_gt_y, /*message=*/"bar",
                                    /*label=*/"foo", SourceInfo(), "assertion");
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  BValue p1_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p1_input_valid, bb.Literal(UBits(1, 1)));
  BValue sum = bb.Add(x, y);
  bb.SetSourceReturnValue(bb.Tuple({assertion_bval, sum}).node());
  BValue p1_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p1_aiv, bb.And(p1_input_valid, p1_aiv));

  XLS_ASSERT_OK(bb.Build().status());

  BlockConversionPassOptions options =
      CreateBlockConversionPassOptions(verilog::CodegenOptions()
                                           .clock_name("clk")
                                           .reset("rst", /*asynchronous=*/false,
                                                  /*active_low=*/false,
                                                  /*reset_data_path=*/false)
                                           .valid_control("in_vld", "out_vld"));
  EXPECT_THAT(Run(&package, options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name(HasSubstr("_all_inputs_valid"))),
                               m::Name("x_gt_y"), m::Name("rst")));

  if (GetParam() == CodegenPassType::kDefault) {
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
}

TEST_P(SideEffectConditionPassTest, FunctionTraceWorks) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  ScheduledBlockBuilder bb("f", &package);
  bb.ResetPort("rst", {.asynchronous = false, .active_low = false});
  Function* source;
  {
    auto owned_source = std::make_unique<Function>("__f_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->MakeNodeWithName<Param>(SourceInfo(), u32, "x"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_y, source->MakeNodeWithName<Param>(SourceInfo(), u32, "y"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_tkn,
                           source->MakeNodeWithName<Param>(
                               SourceInfo(), package.GetTokenType(), "tkn"));
  BValue x = bb.SourceNode(source_x);
  BValue y = bb.SourceNode(source_y);
  BValue tkn = bb.SourceNode(source_tkn);
  BValue not_x_gt_y = bb.ULe(x, y, SourceInfo(), "not_x_gt_y");
  BValue trace_bval = bb.Trace(tkn, not_x_gt_y, {x}, "x = {}",
                               /*verbosity=*/0, SourceInfo(), "trace");
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  BValue p1_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p1_input_valid, bb.Literal(UBits(1, 1)));
  BValue sum = bb.Add(x, y);
  bb.SetSourceReturnValue(bb.Tuple({trace_bval, sum}).node());
  BValue p1_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p1_aiv, bb.And(p1_input_valid, p1_aiv));

  XLS_ASSERT_OK(bb.Build().status());

  BlockConversionPassOptions options =
      CreateBlockConversionPassOptions(verilog::CodegenOptions()
                                           .clock_name("clk")
                                           .reset("rst", /*asynchronous=*/false,
                                                  /*active_low=*/false,
                                                  /*reset_data_path=*/false)
                                           .valid_control("in_vld", "out_vld"));
  EXPECT_THAT(Run(&package, options), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * trace, block->GetNode("trace"));
  ASSERT_NE(trace, nullptr);
  Node* condition = trace->As<xls::Trace>()->condition();
  EXPECT_THAT(condition, m::And(m::Name(HasSubstr("_all_inputs_valid")),
                                m::Name("not_x_gt_y"), m::Not(m::Name("rst"))));

  if (GetParam() == CodegenPassType::kDefault) {
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
}

TEST_P(SideEffectConditionPassTest, FunctionCoverWorks) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  ScheduledBlockBuilder bb("f", &package);
  bb.ResetPort("rst", {.asynchronous = false, .active_low = false});
  Function* source;
  {
    auto owned_source = std::make_unique<Function>("__f_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->MakeNodeWithName<Param>(SourceInfo(), u32, "x"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_y, source->MakeNodeWithName<Param>(SourceInfo(), u32, "y"));
  BValue x = bb.SourceNode(source_x);
  BValue y = bb.SourceNode(source_y);
  BValue not_x_gt_y = bb.ULe(x, y, SourceInfo(), "not_x_gt_y");
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  BValue p1_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p1_input_valid, bb.Literal(UBits(1, 1)));
  bb.Cover(/*condition=*/not_x_gt_y, /*label=*/"not_x_gt_y", SourceInfo(),
           "cover_");
  bb.SetSourceReturnValue(bb.Add(x, y).node());
  BValue p1_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p1_aiv, bb.And(p1_input_valid, p1_aiv));

  XLS_ASSERT_OK(bb.Build().status());

  BlockConversionPassOptions options =
      CreateBlockConversionPassOptions(verilog::CodegenOptions()
                                           .clock_name("clk")
                                           .reset("rst", /*asynchronous=*/false,
                                                  /*active_low=*/false,
                                                  /*reset_data_path=*/false)
                                           .valid_control("in_vld", "out_vld"));
  EXPECT_THAT(Run(&package, options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * cover, block->GetNode("cover_"));
  ASSERT_NE(cover, nullptr);
  Node* condition = cover->As<xls::Cover>()->condition();
  EXPECT_THAT(condition, m::And(m::Name(HasSubstr("_all_inputs_valid")),
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
  ScheduledBlockBuilder bb("f", &package);
  bb.ResetPort("rst", {.asynchronous = false, .active_low = false});
  Proc* source;
  {
    auto owned_source = std::make_unique<Proc>("__f_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->AppendStateElement("x", Value(UBits(0, 32))));
  BValue x = bb.SourceNode(source_x);
  BValue y_recv = bb.Receive(in, bb.AfterAll({}));
  BValue y_token = bb.TupleIndex(y_recv, 0);
  BValue y = bb.TupleIndex(y_recv, 1);
  BValue x_lt_y = bb.ULe(x, y, SourceInfo(), "x_lt_y");
  BValue assertion_bval = bb.Assert(y_token, x_lt_y, /*message=*/"bar",
                                    /*label=*/"foo", SourceInfo(), "assertion");
  BValue sum = bb.Add(x, y);
  bb.Send(out, assertion_bval, sum);
  bb.Next(x, sum);
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  XLS_ASSERT_OK(bb.Build().status());

  BlockConversionPassOptions options =
      CreateBlockConversionPassOptions(kDefaultCodegenOptions);
  EXPECT_THAT(Run(&package, options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name("p0_stage_done")),
                               m::Name("x_lt_y"), m::Name("rst")));

  if (GetParam() == CodegenPassType::kDefault) {
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
}

TEST_P(SideEffectConditionPassTest, AssertionInLastStageOfFunction) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  ScheduledBlockBuilder bb("f", &package);
  bb.ResetPort("rst", {.asynchronous = false, .active_low = false});
  Function* source;
  {
    auto owned_source = std::make_unique<Function>("__f_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Node * source_tkn,
                           source->MakeNodeWithName<Param>(
                               SourceInfo(), package.GetTokenType(), "tkn"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->MakeNodeWithName<Param>(SourceInfo(), u32, "x"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_y, source->MakeNodeWithName<Param>(SourceInfo(), u32, "y"));
  BValue tkn = bb.SourceNode(source_tkn);
  BValue x = bb.SourceNode(source_x);
  BValue y = bb.SourceNode(source_y);
  BValue xy_plus_1 = bb.Add(bb.UMul(x, y), bb.Literal(UBits(1, 32)),
                            SourceInfo(), "xy_plus_1");
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  BValue p1_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p1_input_valid, bb.Literal(UBits(1, 1)));
  BValue xy_plus_1_gt_4 = bb.UGt(xy_plus_1, bb.Literal(UBits(4, 32)),
                                 SourceInfo(), "xy_plus_1_gt_4");
  BValue assertion_bval = bb.Assert(tkn, xy_plus_1_gt_4, /*message=*/"bar",
                                    /*label=*/"foo", SourceInfo(), "assertion");
  bb.SetSourceReturnValue(bb.Tuple({assertion_bval, xy_plus_1}).node());
  BValue p1_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p1_aiv, bb.And(p1_input_valid, p1_aiv));

  XLS_ASSERT_OK(bb.Build().status());

  BlockConversionPassOptions options =
      CreateBlockConversionPassOptions(verilog::CodegenOptions()
                                           .valid_control("in_vld", "out_vld")
                                           .clock_name("clk")
                                           .reset("rst", /*asynchronous=*/false,
                                                  /*active_low=*/false,
                                                  /*reset_data_path=*/false));
  EXPECT_THAT(Run(&package, options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition,
              m::Or(m::Not(m::Name(HasSubstr("p1_all_inputs_valid"))),
                    m::Name("xy_plus_1_gt_4"), m::Name("rst")));

  if (GetParam() == CodegenPassType::kDefault) {
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

  ScheduledBlockBuilder bb("g", &package);
  bb.ResetPort("rst", {.asynchronous = false, .active_low = false});
  Proc* source;
  {
    auto owned_source = std::make_unique<Proc>("__g_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->AppendStateElement("x", Value(UBits(4, 32))));
  BValue x = bb.SourceNode(source_x);
  BValue recv = bb.Receive(in, bb.AfterAll({}));
  BValue recv_token = bb.TupleIndex(recv, 0);
  BValue recv_data = bb.TupleIndex(recv, 1);
  BValue xy = bb.UMul(x, recv_data);
  BValue xy_plus_1 =
      bb.Add(xy, bb.Literal(UBits(1, 32)), SourceInfo(), "xy_plus_1");
  bb.Next(x, bb.Literal(UBits(1, 32)));
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  BValue p1_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p1_input_valid, bb.Literal(UBits(1, 1)));
  BValue xy_plus_1_gt_4 = bb.UGt(xy_plus_1, bb.Literal(UBits(4, 32)),
                                 SourceInfo(), "xy_plus_1_gt_4");
  bb.Assert(recv_token, xy_plus_1_gt_4, /*message=*/"bar",
            /*label=*/"foo", SourceInfo(), "assertion");
  bb.Send(out, recv_token, xy_plus_1);
  BValue p1_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p1_aiv, bb.And(p1_input_valid, p1_aiv));

  XLS_ASSERT_OK(bb.Build().status());

  BlockConversionPassOptions options =
      CreateBlockConversionPassOptions(kDefaultCodegenOptions);
  EXPECT_THAT(Run(&package, options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("g"));

  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition,
              m::Or(m::Not(m::Name(HasSubstr("p1_stage_done"))),
                    m::Name(HasSubstr("xy_plus_1_gt_4")), m::Name("rst")));

  if (GetParam() == CodegenPassType::kDefault) {
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

  ScheduledBlockBuilder bb("ii_greater_than_one", &package);
  bb.ResetPort("rst", {.asynchronous = false, .active_low = false});
  Proc* source;
  {
    auto owned_source =
        std::make_unique<Proc>("__ii_greater_than_one_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }
  source->SetInitiationInterval(2);

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->AppendStateElement("st", Value(UBits(0, 32))));
  BValue x = bb.SourceNode(source_x);
  BValue send0_token = bb.Send(out, bb.Literal(Value::Token()), x);
  BValue min_delay_token = bb.MinDelay(send0_token, /*delay=*/1);
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  BValue p1_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p1_input_valid, bb.Literal(UBits(1, 1)));
  BValue recv_tuple = bb.Receive(in, min_delay_token);
  BValue recv_token = bb.TupleIndex(recv_tuple, 0);
  BValue recv_data = bb.TupleIndex(recv_tuple, 1);
  BValue recv_data_lt_5 = bb.ULt(recv_data, bb.Literal(UBits(5, 32)),
                                 SourceInfo(), "receive_data_lt_5");
  BValue assertion_bval = bb.Assert(recv_token, recv_data_lt_5,
                                    /*message=*/"bar",
                                    /*label=*/"foo", SourceInfo(), "assertion");
  bb.Send(in_out, assertion_bval, recv_data);
  bb.Next(x, recv_data);
  BValue p1_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p1_aiv, bb.And(p1_input_valid, p1_aiv));

  XLS_ASSERT_OK(bb.Build().status());

  BlockConversionPassOptions options =
      CreateBlockConversionPassOptions(kDefaultCodegenOptions);
  EXPECT_THAT(Run(&package, options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           package.GetBlock("ii_greater_than_one"));
  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  Node* condition = assertion->As<xls::Assert>()->condition();
  EXPECT_THAT(condition, m::Or(m::Not(m::Name(HasSubstr("p1_stage_done"))),
                               m::Name("receive_data_lt_5"), m::Name("rst")));

  if (GetParam() == CodegenPassType::kDefault) {
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
}

TEST_P(SideEffectConditionPassTest, FunctionWithActiveLowReset) {
  Package package("test_package");
  Type* u32 = package.GetBitsType(32);
  ScheduledBlockBuilder bb("f", &package);
  bb.ResetPort("rst", {.asynchronous = false, .active_low = true});
  Function* source;
  {
    auto owned_source = std::make_unique<Function>("__f_source", &package);
    source = owned_source.get();
    bb.SetSource(std::move(owned_source));
  }

  BValue p0_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p0_input_valid, bb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_x, source->MakeNodeWithName<Param>(SourceInfo(), u32, "x"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * source_y, source->MakeNodeWithName<Param>(SourceInfo(), u32, "y"));
  BValue x = bb.SourceNode(source_x);
  BValue y = bb.SourceNode(source_y);
  bb.Cover(/*condition=*/bb.ULt(x, y), /*label=*/"x_lt_y", SourceInfo(),
           /*name=*/"cover_");
  bb.Trace(bb.AfterAll({}), bb.Eq(x, y), {}, "x == y", /*verbosity=*/0,
           SourceInfo(), /*name=*/"trace");
  bb.Assert(bb.AfterAll({}), bb.ULe(x, y), "Saw x > y", "assert_x_le_y",
            SourceInfo(), /*name=*/"assertion");
  BValue p0_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p0_aiv, bb.And(p0_input_valid, p0_aiv));

  BValue p1_input_valid = bb.Literal(UBits(1, 1));
  bb.StartStage(p1_input_valid, bb.Literal(UBits(1, 1)));
  bb.SetSourceReturnValue(bb.Add(x, y).node());
  BValue p1_aiv = bb.Literal(UBits(1, 1));
  bb.EndStage(p1_aiv, bb.And(p1_input_valid, p1_aiv));

  XLS_ASSERT_OK(bb.Build().status());

  BlockConversionPassOptions options =
      CreateBlockConversionPassOptions(verilog::CodegenOptions()
                                           .valid_control("in_vld", "out_vld")
                                           .reset("rst", /*asynchronous=*/false,
                                                  /*active_low=*/true,
                                                  /*reset_data_path=*/false));
  EXPECT_THAT(Run(&package, options), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("f"));

  XLS_ASSERT_OK_AND_ASSIGN(Node * cover, block->GetNode("cover_"));
  ASSERT_NE(cover, nullptr);
  EXPECT_THAT(cover->As<xls::Cover>()->condition(),
              m::And(m::Name("p0_all_inputs_valid"),
                     m::ULt(m::Name("x"), m::Name("y")), m::InputPort("rst")));

  XLS_ASSERT_OK_AND_ASSIGN(Node * trace, block->GetNode("trace"));
  ASSERT_NE(trace, nullptr);
  EXPECT_THAT(trace->As<xls::Trace>()->condition(),
              m::And(m::Name("p0_all_inputs_valid"),
                     m::Eq(m::Name("x"), m::Name("y")), m::InputPort("rst")));

  XLS_ASSERT_OK_AND_ASSIGN(Node * assertion, block->GetNode("assertion"));
  ASSERT_NE(assertion, nullptr);
  EXPECT_THAT(
      assertion->As<xls::Assert>()->condition(),
      m::Or(m::Not(m::Name("p0_all_inputs_valid")),
            m::ULe(m::Name("x"), m::Name("y")), m::Not(m::InputPort("rst"))));

  if (GetParam() == CodegenPassType::kDefault) {
    constexpr int64_t kNumCycles = 10;
    std::vector<absl::flat_hash_map<std::string, Value>> inputs(
        kNumCycles,
        {{"rst", Value(UBits(0, 1))},
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
}

INSTANTIATE_TEST_SUITE_P(
    SideEffectConditionPassTestInstantiation, SideEffectConditionPassTest,

    testing::Values(CodegenPassType::kDefault,
                    CodegenPassType::kSideEffectConditionPassOnly),
    [](const testing::TestParamInfo<CodegenPassType>& info) {
      return std::string(CodegenPassName(info.param));
    });

}  // namespace
}  // namespace xls::codegen
