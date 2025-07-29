// Copyright 2024 The XLS Authors
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

#include "xls/codegen/register_combining_pass.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace m = xls::op_matchers;
namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::AnyOf;
using ::testing::MatchesRegex;
using ::testing::UnorderedElementsAre;

MATCHER_P(Reg, name,
          absl::StrCat("Register should be named: ",
                       testing::DescribeMatcher<std::string>(name, negation))) {
  const Register* reg = arg;
  return testing::ExplainMatchResult(name, reg->name(), result_listener);
}

// TODO(allight): Writing tests for this is hampered by the fact we rely on
// block-converter invariants and metadata that aren't really represented well
// or at all in the block-ir for this pass to function. This makes it difficult
// to create really robust tests since we are basically forced to depend on some
// implementation details of the block-converter to avoid having 1000s of lines
// of test setup that is equally fragile to any changes to block-converter.
class RegisterCombiningPassTest : public IrTestBase {
 public:
  absl::StatusOr<bool> Run(Package* package, CodegenContext& context) {
    RegisterCombiningPass rcp;
    PassResults res;
    return rcp.Run(
        package,
        CodegenPassOptions{
            .codegen_options = CodegenOptions().register_merge_strategy(
                CodegenOptions::RegisterMergeStrategy::kIdentityOnly)},
        &res, context);
  }

  std::string NodeName(BValue n) {
    // TODO(allight): Recreates part of node.h
    if (n.node()->HasAssignedName()) {
      return n.GetName();
    }
    return absl::StrFormat("%s_[0-9]+", OpToString(n.node()->op()));
  }

  auto StateToRegMatcher(BValue st) {
    // TODO(allight): Recreates a block-conversion function.
    EXPECT_THAT(st.node(), m::StateRead());
    StateElement* state_element = st.node()->As<StateRead>()->state_element();
    return Reg(state_element->name());
  }
  auto StateToRegFullMatcher(BValue st) {
    // TODO(allight): Recreates a block-conversion function.
    EXPECT_THAT(st.node(), m::StateRead());
    StateElement* state_element = st.node()->As<StateRead>()->state_element();
    return Reg(absl::StrFormat("%s_full", state_element->name()));
  }
  auto StageValidMatcher(Stage s) {
    return Reg(absl::StrFormat("p%d_valid", s));
  }
  auto NodeToRegMatcher(BValue v, Stage s) {
    // TODO(allight): Recreates a block-conversion function.
    if (v.node()->Is<StateRead>()) {
      StateElement* state_element = v.node()->As<StateRead>()->state_element();
      return Reg(MatchesRegex(
          absl::StrFormat("p%d_%s__[0-9]+", s, state_element->name())));
    }
    return Reg(MatchesRegex(absl::StrFormat("p%d_%s", s, NodeName(v))));
  }
};

TEST_F(RegisterCombiningPassTest, CombineBasic) {
  // TODO(allight): Just this 12 stage pipeline with 6 operation nodes takes
  // ~100 nodes to write out manually and all the codegen-pass-unit metadata
  // too. Just use the proc-builder and block-converter instead.
  auto p = CreatePackage();
  TokenlessProcBuilder pb(TestName(), "tok", p.get());
  auto tok = pb.InitialToken();
  auto st = pb.StateElement("foo", UBits(1, 32));
  auto lit_1 = pb.Literal(UBits(1, 32));
  auto lit_2 = pb.Literal(UBits(2, 32));
  auto add_1 = pb.Add(st, lit_1);
  auto mul_2 = pb.UMul(add_1, lit_2);
  auto nxt = pb.Next(st, mul_2);

  XLS_ASSERT_OK_AND_ASSIGN(auto proc, pb.Build());
  PipelineSchedule sched(proc,
                         {
                             {tok.node(), 0},
                             {st.node(), 0},
                             {lit_1.node(), 4},
                             {add_1.node(), 4},
                             {lit_2.node(), 8},
                             {mul_2.node(), 8},
                             {nxt.node(), 12},
                         },
                         13);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto context,
      FunctionBaseToPipelinedBlock(sched,
                                   CodegenOptions()
                                       .emit_as_pipeline(true)
                                       .module_name("foobar")
                                       .clock_name("clk")
                                       .reset("rst", false, false, false)
                                       .streaming_channel_data_suffix("_d")
                                       .streaming_channel_ready_suffix("_r")
                                       .streaming_channel_valid_suffix("_v"),
                                   proc));
  RecordProperty("ir", p->DumpIr());
  // Just make sure that changes to block-conversion haven't broken us.
  ASSERT_THAT(
      context.top_block()->GetRegisters(),
      UnorderedElementsAre(
          StateToRegMatcher(st), StateToRegFullMatcher(st),
          NodeToRegMatcher(st, 0), NodeToRegMatcher(st, 1),
          NodeToRegMatcher(st, 2), NodeToRegMatcher(st, 3),
          NodeToRegMatcher(add_1, 4), NodeToRegMatcher(add_1, 5),
          NodeToRegMatcher(add_1, 6), NodeToRegMatcher(add_1, 7),
          NodeToRegMatcher(mul_2, 8), NodeToRegMatcher(mul_2, 9),
          NodeToRegMatcher(mul_2, 10), NodeToRegMatcher(mul_2, 11),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11)))
      << "Register names changed. Test needs to be updated for "
         "block-conversion changes";

  EXPECT_THAT(Run(p.get(), context), IsOkAndHolds(true));
  RecordProperty("result", p->DumpIr());

  EXPECT_THAT(
      context.top_block()->GetRegisters(),

      UnorderedElementsAre(
          StateToRegMatcher(st), StateToRegFullMatcher(st),
          NodeToRegMatcher(add_1, 4), NodeToRegMatcher(mul_2, 8),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11)));
}

TEST_F(RegisterCombiningPassTest, CombineOverlap) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(TestName(), "tok", p.get());
  auto tok = pb.InitialToken();
  auto st1 = pb.StateElement("foo", UBits(1, 32));
  auto st2 = pb.StateElement("bar", UBits(1, 32));
  auto lit_1 = pb.Literal(UBits(1, 32), SourceInfo(), "lit_1");
  auto lit_2 = pb.Literal(UBits(2, 32), SourceInfo(), "lit_2");
  auto lit_3 = pb.Literal(UBits(3, 32), SourceInfo(), "lit_3");
  auto lit_4 = pb.Literal(UBits(4, 32), SourceInfo(), "lit_4");
  auto add_1 = pb.Add(st1, lit_1, SourceInfo(), "add_1");
  auto mul_2 = pb.UMul(add_1, lit_2, SourceInfo(), "mul_2");
  auto mul_1 = pb.UMul(st2, lit_3, SourceInfo(), "mul_1");
  auto add_2 = pb.Add(mul_1, lit_4, SourceInfo(), "add_2");
  auto nxt1 = pb.Next(st1, mul_2);
  auto nxt2 = pb.Next(st2, add_2);

  XLS_ASSERT_OK_AND_ASSIGN(auto proc, pb.Build());
  PipelineSchedule sched(proc,
                         {
                             {tok.node(), 0},
                             {st1.node(), 0},
                             {lit_1.node(), 3},
                             {add_1.node(), 3},
                             {lit_2.node(), 6},
                             {mul_2.node(), 6},
                             {nxt1.node(), 9},
                             {st2.node(), 5},
                             {lit_3.node(), 8},
                             {mul_1.node(), 8},
                             {lit_4.node(), 11},
                             {add_2.node(), 11},
                             {nxt2.node(), 14},
                         },
                         15);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto context,
      FunctionBaseToPipelinedBlock(sched,
                                   CodegenOptions()
                                       .emit_as_pipeline(true)
                                       .module_name("foobar")
                                       .clock_name("clk")
                                       .reset("rst", false, false, false)
                                       .streaming_channel_data_suffix("_d")
                                       .streaming_channel_ready_suffix("_r")
                                       .streaming_channel_valid_suffix("_v"),
                                   proc));
  RecordProperty("ir", p->DumpIr());
  // Just make sure that changes to block-conversion haven't broken us.
  ASSERT_THAT(
      context.top_block()->GetRegisters(),
      UnorderedElementsAre(
          StateToRegMatcher(st1), StateToRegFullMatcher(st1),
          NodeToRegMatcher(st1, 0), NodeToRegMatcher(st1, 1),
          NodeToRegMatcher(st1, 2), NodeToRegMatcher(add_1, 3),
          NodeToRegMatcher(add_1, 4), NodeToRegMatcher(add_1, 5),
          NodeToRegMatcher(mul_2, 6), NodeToRegMatcher(mul_2, 7),
          NodeToRegMatcher(mul_2, 8), StateToRegMatcher(st2),
          StateToRegFullMatcher(st2), NodeToRegMatcher(st2, 5),
          NodeToRegMatcher(st2, 6), NodeToRegMatcher(st2, 7),
          NodeToRegMatcher(mul_1, 8), NodeToRegMatcher(mul_1, 9),
          NodeToRegMatcher(mul_1, 10), NodeToRegMatcher(add_2, 11),
          NodeToRegMatcher(add_2, 12), NodeToRegMatcher(add_2, 13),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11),
          StageValidMatcher(12), StageValidMatcher(13)))
      << "Register names changed. Test needs to be updated for "
         "block-conversion changes";

  EXPECT_THAT(Run(p.get(), context), IsOkAndHolds(true));
  RecordProperty("result", p->DumpIr());

  EXPECT_THAT(
      context.top_block()->GetRegisters(),

      UnorderedElementsAre(
          StateToRegMatcher(st1), StateToRegFullMatcher(st1),
          NodeToRegMatcher(add_1, 3), NodeToRegMatcher(mul_2, 6),
          StateToRegMatcher(st2), StateToRegFullMatcher(st2),
          NodeToRegMatcher(mul_1, 8), NodeToRegMatcher(add_2, 11),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11),
          StageValidMatcher(12), StageValidMatcher(13)));
}

TEST_F(RegisterCombiningPassTest, CombineWithRegisterSwap) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(TestName(), "tok", p.get());
  auto tok = pb.InitialToken();
  auto st1 = pb.StateElement("foo", UBits(1, 32));
  auto st2 = pb.StateElement("bar", UBits(1, 32));
  auto lit_1 = pb.Literal(UBits(1, 32), SourceInfo(), "lit_1");
  auto lit_2 = pb.Literal(UBits(2, 32), SourceInfo(), "lit_2");
  auto lit_3 = pb.Literal(UBits(3, 32), SourceInfo(), "lit_3");
  auto lit_4 = pb.Literal(UBits(4, 32), SourceInfo(), "lit_4");
  auto add_1 = pb.Add(st1, lit_1, SourceInfo(), "add_1");
  auto mul_2 = pb.UMul(add_1, lit_2, SourceInfo(), "mul_2");
  auto mul_1 = pb.UMul(st2, lit_3, SourceInfo(), "mul_1");
  auto add_2 = pb.Add(mul_1, lit_4, SourceInfo(), "add_2");
  auto add_3 = pb.Add(add_2, add_1, SourceInfo(), "add_3");
  auto nxt1 = pb.Next(st1, mul_2);
  auto nxt2 = pb.Next(st2, add_3);

  XLS_ASSERT_OK_AND_ASSIGN(auto proc, pb.Build());
  PipelineSchedule sched(proc,
                         {
                             {tok.node(), 0},
                             {st1.node(), 0},
                             {lit_1.node(), 3},
                             {add_1.node(), 3},
                             {lit_2.node(), 6},
                             {mul_2.node(), 6},
                             {nxt1.node(), 9},
                             {st2.node(), 5},
                             {lit_3.node(), 8},
                             {mul_1.node(), 8},
                             {lit_4.node(), 11},
                             {add_2.node(), 11},
                             {add_3.node(), 13},
                             {nxt2.node(), 15},
                         },
                         16);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto context,
      FunctionBaseToPipelinedBlock(sched,
                                   CodegenOptions()
                                       .emit_as_pipeline(true)
                                       .module_name("foobar")
                                       .clock_name("clk")
                                       .reset("rst", false, false, false)
                                       .streaming_channel_data_suffix("_d")
                                       .streaming_channel_ready_suffix("_r")
                                       .streaming_channel_valid_suffix("_v"),
                                   proc));
  RecordProperty("ir", p->DumpIr());
  // Just make sure that changes to block-conversion haven't broken us.
  ASSERT_THAT(
      context.top_block()->GetRegisters(),
      UnorderedElementsAre(
          StateToRegMatcher(st1), StateToRegFullMatcher(st1),
          NodeToRegMatcher(st1, 0), NodeToRegMatcher(st1, 1),
          NodeToRegMatcher(st1, 2), NodeToRegMatcher(add_1, 3),
          NodeToRegMatcher(add_1, 4), NodeToRegMatcher(add_1, 5),
          NodeToRegMatcher(add_1, 6), NodeToRegMatcher(add_1, 7),
          NodeToRegMatcher(add_1, 8), NodeToRegMatcher(add_1, 9),
          NodeToRegMatcher(add_1, 10), NodeToRegMatcher(add_1, 11),
          NodeToRegMatcher(add_1, 12), NodeToRegMatcher(mul_2, 6),
          NodeToRegMatcher(mul_2, 7), NodeToRegMatcher(mul_2, 8),
          StateToRegMatcher(st2), StateToRegFullMatcher(st2),
          NodeToRegMatcher(st2, 5), NodeToRegMatcher(st2, 6),
          NodeToRegMatcher(st2, 7), NodeToRegMatcher(mul_1, 8),
          NodeToRegMatcher(mul_1, 9), NodeToRegMatcher(mul_1, 10),
          NodeToRegMatcher(add_2, 11), NodeToRegMatcher(add_2, 12),
          NodeToRegMatcher(add_3, 13), NodeToRegMatcher(add_3, 14),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11),
          StageValidMatcher(12), StageValidMatcher(13), StageValidMatcher(14)))
      << "Register names changed. Test needs to be updated for "
         "block-conversion changes";

  EXPECT_THAT(Run(p.get(), context), IsOkAndHolds(true));
  RecordProperty("result", p->DumpIr());

  EXPECT_THAT(
      context.top_block()->GetRegisters(),

      UnorderedElementsAre(
          StateToRegMatcher(st1), StateToRegFullMatcher(st1),
          NodeToRegMatcher(add_1, 3),
          AnyOf(NodeToRegMatcher(add_1, 5), NodeToRegMatcher(add_1, 6),
                NodeToRegMatcher(add_1, 7), NodeToRegMatcher(add_1, 8),
                NodeToRegMatcher(add_1, 9)),
          NodeToRegMatcher(mul_2, 6), StateToRegMatcher(st2),
          StateToRegFullMatcher(st2), NodeToRegMatcher(mul_1, 8),
          NodeToRegMatcher(add_2, 11), NodeToRegMatcher(add_3, 13),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11),
          StageValidMatcher(12), StageValidMatcher(13), StageValidMatcher(14)));
}

TEST_F(RegisterCombiningPassTest, AppliesToPredicatedWrites) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(TestName(), "tok", p.get());
  auto tok = pb.InitialToken();
  auto st = pb.StateElement("foo", UBits(1, 32));
  auto lit_1 = pb.Literal(UBits(1, 32));
  auto lit_2 = pb.Literal(UBits(2, 32));
  auto add_1 = pb.Add(st, lit_1);
  auto mul_2 = pb.UMul(add_1, lit_2);
  auto nxt_pred = pb.Literal(UBits(0, 1));
  auto nxt = pb.Next(st, mul_2, /*pred=*/nxt_pred);

  XLS_ASSERT_OK_AND_ASSIGN(auto proc, pb.Build());
  PipelineSchedule sched(proc,
                         {
                             {tok.node(), 0},
                             {st.node(), 0},
                             {lit_1.node(), 4},
                             {add_1.node(), 4},
                             {lit_2.node(), 8},
                             {mul_2.node(), 8},
                             {nxt_pred.node(), 12},
                             {nxt.node(), 12},
                         },
                         13);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto context,
      FunctionBaseToPipelinedBlock(sched,
                                   CodegenOptions()
                                       .emit_as_pipeline(true)
                                       .module_name("foobar")
                                       .clock_name("clk")
                                       .reset("rst", false, false, false)
                                       .streaming_channel_data_suffix("_d")
                                       .streaming_channel_ready_suffix("_r")
                                       .streaming_channel_valid_suffix("_v"),
                                   proc));
  RecordProperty("ir", p->DumpIr());
  // Just make sure that changes to block-conversion haven't broken us.
  ASSERT_THAT(
      context.top_block()->GetRegisters(),
      UnorderedElementsAre(
          StateToRegMatcher(st), StateToRegFullMatcher(st),
          NodeToRegMatcher(st, 0), NodeToRegMatcher(st, 1),
          NodeToRegMatcher(st, 2), NodeToRegMatcher(st, 3),
          NodeToRegMatcher(add_1, 4), NodeToRegMatcher(add_1, 5),
          NodeToRegMatcher(add_1, 6), NodeToRegMatcher(add_1, 7),
          NodeToRegMatcher(mul_2, 8), NodeToRegMatcher(mul_2, 9),
          NodeToRegMatcher(mul_2, 10), NodeToRegMatcher(mul_2, 11),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11)))
      << "Register names changed. Test needs to be updated for "
         "block-conversion changes";

  EXPECT_THAT(Run(p.get(), context), IsOkAndHolds(true));
  RecordProperty("result", p->DumpIr());

  EXPECT_THAT(
      context.top_block()->GetRegisters(),

      UnorderedElementsAre(
          StateToRegMatcher(st), StateToRegFullMatcher(st),
          NodeToRegMatcher(add_1, 4), NodeToRegMatcher(mul_2, 8),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11)));
}

TEST_F(RegisterCombiningPassTest, DoesntApplyToPredicatedReads) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(TestName(), "tok", p.get());
  auto tok = pb.InitialToken();
  auto always_false = pb.Literal(UBits(0, 1));
  auto st = pb.StateElement("foo", UBits(1, 32),
                            /*read_predicate=*/always_false);
  auto lit_1 = pb.Literal(UBits(1, 32));
  auto always_false_2 = pb.Literal(UBits(0, 1));
  auto st_v = pb.Select(always_false_2, /*cases=*/{lit_1, st});
  auto add_1 = pb.Add(st_v, lit_1);
  auto lit_2 = pb.Literal(UBits(2, 32));
  auto mul_2 = pb.UMul(add_1, lit_2);
  auto always_false_3 = pb.Literal(UBits(0, 1));
  auto nxt = pb.Next(st, mul_2, /*pred=*/always_false_3);

  XLS_ASSERT_OK_AND_ASSIGN(auto proc, pb.Build());
  PipelineSchedule sched(proc,
                         {
                             {tok.node(), 0},
                             {always_false.node(), 0},
                             {st.node(), 0},
                             {lit_1.node(), 4},
                             {always_false_2.node(), 4},
                             {st_v.node(), 4},
                             {add_1.node(), 4},
                             {lit_2.node(), 8},
                             {mul_2.node(), 8},
                             {always_false_3.node(), 12},
                             {nxt.node(), 12},
                         },
                         13);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto context,
      FunctionBaseToPipelinedBlock(sched,
                                   CodegenOptions()
                                       .emit_as_pipeline(true)
                                       .module_name("foobar")
                                       .clock_name("clk")
                                       .reset("rst", false, false, false)
                                       .streaming_channel_data_suffix("_d")
                                       .streaming_channel_ready_suffix("_r")
                                       .streaming_channel_valid_suffix("_v"),
                                   proc));
  RecordProperty("ir", p->DumpIr());
  // Just make sure that changes to block-conversion haven't broken us.
  ASSERT_THAT(
      context.top_block()->GetRegisters(),
      UnorderedElementsAre(
          StateToRegMatcher(st), StateToRegFullMatcher(st),
          NodeToRegMatcher(st, 0), NodeToRegMatcher(st, 1),
          NodeToRegMatcher(st, 2), NodeToRegMatcher(st, 3),
          NodeToRegMatcher(add_1, 4), NodeToRegMatcher(add_1, 5),
          NodeToRegMatcher(add_1, 6), NodeToRegMatcher(add_1, 7),
          NodeToRegMatcher(mul_2, 8), NodeToRegMatcher(mul_2, 9),
          NodeToRegMatcher(mul_2, 10), NodeToRegMatcher(mul_2, 11),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11)))
      << "Register names changed. Test needs to be updated for "
         "block-conversion changes";

  EXPECT_THAT(Run(p.get(), context), IsOkAndHolds(false));
  RecordProperty("result", p->DumpIr());

  EXPECT_THAT(
      context.top_block()->GetRegisters(),
      UnorderedElementsAre(
          StateToRegMatcher(st), StateToRegFullMatcher(st),
          NodeToRegMatcher(st, 0), NodeToRegMatcher(st, 1),
          NodeToRegMatcher(st, 2), NodeToRegMatcher(st, 3),
          NodeToRegMatcher(add_1, 4), NodeToRegMatcher(add_1, 5),
          NodeToRegMatcher(add_1, 6), NodeToRegMatcher(add_1, 7),
          NodeToRegMatcher(mul_2, 8), NodeToRegMatcher(mul_2, 9),
          NodeToRegMatcher(mul_2, 10), NodeToRegMatcher(mul_2, 11),
          StageValidMatcher(0), StageValidMatcher(1), StageValidMatcher(2),
          StageValidMatcher(3), StageValidMatcher(4), StageValidMatcher(5),
          StageValidMatcher(6), StageValidMatcher(7), StageValidMatcher(8),
          StageValidMatcher(9), StageValidMatcher(10), StageValidMatcher(11)));
}

}  // namespace
}  // namespace xls::verilog
