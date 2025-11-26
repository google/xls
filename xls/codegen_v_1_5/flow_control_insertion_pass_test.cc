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

#include "xls/codegen_v_1_5/flow_control_insertion_pass.h"

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/pipeline_register_insertion_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/register.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/ir/value.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;

class FlowControlInsertionPassTest : public IrTestBase {
 protected:
  FlowControlInsertionPassTest() = default;

  absl::StatusOr<bool> Run(Package* p,
                           const BlockConversionPassOptions& options =
                               BlockConversionPassOptions()) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         FlowControlInsertionPass().Run(p, options, &results));

    // Run pipeline register insertion pass to add pipeline registers for easy
    // simulation.
    XLS_RETURN_IF_ERROR(
        PipelineRegisterInsertionPass().Run(p, options, &results).status());

    return changed;
  }
};

TEST_F(FlowControlInsertionPassTest, SingleStagePipeline) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  BValue in = sbb.InputPort("in", p->GetBitsType(32));
  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue out = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.OutputPort("out", out);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());
  ASSERT_EQ(sb->stages().size(), 1);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_TRUE(sb->stages()[0].inputs_valid()->Is<Literal>());
  EXPECT_TRUE(sb->stages()[0].outputs_ready()->Is<Literal>());
  EXPECT_EQ(sb->stages()[0].inputs_valid()->As<Literal>()->value(),
            Value(UBits(1, 1)));
  EXPECT_EQ(sb->stages()[0].outputs_ready()->As<Literal>()->value(),
            Value(UBits(1, 1)));
}

TEST_F(FlowControlInsertionPassTest, MultiStagePipelineFullThroughput) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue p0 = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue out = sbb.UMul(p0, sbb.Literal(UBits(2, 32)));
  sbb.OutputPort("out", out);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK(sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));

  std::vector<absl::flat_hash_map<std::string, Value>> block_results;
  XLS_ASSERT_OK_AND_ASSIGN(
      block_results,
      InterpretSequentialBlock(block, {{{"in", Value(UBits(10, 32))}},
                                       {{"in", Value(UBits(20, 32))}},
                                       {{"in", Value(UBits(30, 32))}}}));
  EXPECT_THAT(
      block_results,
      ElementsAre(UnorderedElementsAre(Pair("out", Value(UBits(0, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(22, 32)))),
                  UnorderedElementsAre(Pair("out", Value(UBits(42, 32))))));
}

TEST_F(FlowControlInsertionPassTest, PipelineStallDueToBackpressure) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));
  BValue s1_ov_in = sbb.InputPort("s1_ov_in", p->GetBitsType(1));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue p0 = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue out = sbb.UMul(p0, sbb.Literal(UBits(2, 32)));
  sbb.OutputPort("out", out);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), s1_ov_in);

  XLS_ASSERT_OK(sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));

  std::vector<absl::flat_hash_map<std::string, Value>> inputs = {
      {{"in", Value(UBits(10, 32))}, {"s1_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(20, 32))}, {"s1_ov_in", Value(UBits(0, 1))}},
      {{"in", Value(UBits(30, 32))}, {"s1_ov_in", Value(UBits(0, 1))}},
      {{"in", Value(UBits(40, 32))}, {"s1_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(50, 32))}, {"s1_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(60, 32))}, {"s1_ov_in", Value(UBits(1, 1))}}};
  std::vector<absl::flat_hash_map<std::string, Value>> block_results;
  XLS_ASSERT_OK_AND_ASSIGN(block_results,
                           InterpretSequentialBlock(block, inputs));
  EXPECT_THAT(
      block_results,
      ElementsAre(
          UnorderedElementsAre(
              Pair("out", Value(UBits(0, 32)))),  // C0: in=10, s1_ov=1. S1
                                                  // done. out=0. p0 gets 11.
          UnorderedElementsAre(Pair(
              "out",
              Value(UBits(22, 32)))),  // C1: in=20, s1_ov=0. S1 not done. S0
                                       // stalls. out=11*2=22. p0 retains 11.
          UnorderedElementsAre(Pair(
              "out",
              Value(UBits(22, 32)))),  // C2: in=30, s1_ov=0. S1 not done. S0
                                       // stalls. out=11*2=22. p0 retains 11.
          UnorderedElementsAre(Pair(
              "out",
              Value(UBits(22, 32)))),  // C3: in=40, s1_ov=1. S1 done with input
                                       // 11. S0 unstalls. out=11*2=22. p0 gets
                                       // 40+1=41.
          UnorderedElementsAre(Pair(
              "out", Value(UBits(82, 32)))),  // C4: in=50, s1_ov=1. S1 done
                                              // with input 41. out=41*2=82. p0
                                              // gets 50+1=51.
          UnorderedElementsAre(Pair("out", Value(UBits(102, 32))))));
}

TEST_F(FlowControlInsertionPassTest, PipelineBubblePropagation) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));
  BValue s0_ov_in = sbb.InputPort("s0_ov_in", p->GetBitsType(1));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue p0 = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), s0_ov_in);

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue out = sbb.UMul(p0, sbb.Literal(UBits(2, 32)));
  sbb.OutputPort("out", out);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK(sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));

  std::vector<absl::flat_hash_map<std::string, Value>> inputs = {
      {{"in", Value(UBits(10, 32))}, {"s0_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(20, 32))}, {"s0_ov_in", Value(UBits(0, 1))}},
      {{"in", Value(UBits(30, 32))}, {"s0_ov_in", Value(UBits(0, 1))}},
      {{"in", Value(UBits(40, 32))}, {"s0_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(50, 32))}, {"s0_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(60, 32))}, {"s0_ov_in", Value(UBits(1, 1))}}};
  std::vector<absl::flat_hash_map<std::string, Value>> block_results;
  XLS_ASSERT_OK_AND_ASSIGN(block_results,
                           InterpretSequentialBlock(block, inputs));
  EXPECT_THAT(
      block_results,
      ElementsAre(
          UnorderedElementsAre(
              Pair("out", Value(UBits(0, 32)))),  // C0: in=10, s0_ov=1. S0
                                                  // done. out=0. p0 gets 11.
          UnorderedElementsAre(Pair(
              "out",
              Value(UBits(22, 32)))),  // C1: in=20, s0_ov=0. S0 not done. S1
                                       // gets 11. out=11*2=22. p0 retains 11.
          UnorderedElementsAre(Pair(
              "out",
              Value(UBits(22, 32)))),  // C2: in=30, s0_ov=0. S0 not done. S1
                                       // gets 11. out=11*2=22. p0 retains 11.
          UnorderedElementsAre(Pair(
              "out",
              Value(UBits(22, 32)))),  // C3: in=40, s0_ov=1. S0 done. S1
                                       // gets 11. out=11*2=22. p0 gets 40+1=41.
          UnorderedElementsAre(Pair(
              "out", Value(UBits(82, 32)))),  // C4: in=50, s0_ov=1.
                                              // out=41*2=82. p0 gets 50+1=51.
          UnorderedElementsAre(Pair("out", Value(UBits(102, 32))))));
}

TEST_F(FlowControlInsertionPassTest, CorrectResetBehavior) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  XLS_ASSERT_OK(sbb.block()->AddResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false}));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue p0 = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue out = sbb.UMul(p0, sbb.Literal(UBits(2, 32)));
  sbb.OutputPort("out", out);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledBlock * sb, sbb.Build());

  ScopedRecordIr sri(p.get());
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  ASSERT_TRUE(sb->stages()[1].inputs_valid()->Is<RegisterRead>());
  Register* iv_reg =
      sb->stages()[1].inputs_valid()->As<RegisterRead>()->GetRegister();
  EXPECT_EQ(iv_reg->name(), "p1_inputs_valid");
  ASSERT_TRUE(iv_reg->reset_value().has_value());
  EXPECT_EQ(iv_reg->reset_value().value(), Value(UBits(0, 1)));
}

TEST_F(FlowControlInsertionPassTest, FillingPipelineBubbles) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  XLS_ASSERT_OK(sbb.block()->AddClockPort("clk"));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));
  BValue s0_ov_in = sbb.InputPort("s0_ov_in", p->GetBitsType(1));
  BValue s2_ov_in = sbb.InputPort("s2_ov_in", p->GetBitsType(1));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue p0 = sbb.Add(in, sbb.Literal(UBits(1, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), s0_ov_in);

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue p1 = sbb.UMul(p0, sbb.Literal(UBits(2, 32)));
  sbb.EndStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));

  sbb.StartStage(sbb.Literal(UBits(1, 1)), sbb.Literal(UBits(1, 1)));
  BValue out = sbb.Add(p1, sbb.Literal(UBits(3, 32)));
  sbb.OutputPort("out", out);
  sbb.EndStage(sbb.Literal(UBits(1, 1)), s2_ov_in);

  XLS_ASSERT_OK(sbb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));

  std::vector<absl::flat_hash_map<std::string, Value>> inputs = {
      {{"in", Value(UBits(10, 32))},
       {"s0_ov_in", Value(UBits(1, 1))},
       {"s2_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(20, 32))},
       {"s0_ov_in", Value(UBits(0, 1))},
       {"s2_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(30, 32))},
       {"s0_ov_in", Value(UBits(1, 1))},
       {"s2_ov_in", Value(UBits(0, 1))}},
      {{"in", Value(UBits(40, 32))},
       {"s0_ov_in", Value(UBits(1, 1))},
       {"s2_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(50, 32))},
       {"s0_ov_in", Value(UBits(1, 1))},
       {"s2_ov_in", Value(UBits(1, 1))}},
      {{"in", Value(UBits(60, 32))},
       {"s0_ov_in", Value(UBits(1, 1))},
       {"s2_ov_in", Value(UBits(1, 1))}},
  };
  std::vector<absl::flat_hash_map<std::string, Value>> block_results;
  XLS_ASSERT_OK_AND_ASSIGN(block_results,
                           InterpretSequentialBlock(block, inputs));
  EXPECT_THAT(
      block_results,
      ElementsAre(
          // C0: in=10,s0_ov=1,s2_ov=1. p0 gets 11. p1 gets 0. out=0+3=3.
          UnorderedElementsAre(Pair("out", Value(UBits(3, 32)))),
          // C1: in=20,s0_ov=0,s2_ov=1. S0 stalls. p1 gets 11*2=22. out=0+3=3.
          //     p0 retains 11.
          UnorderedElementsAre(Pair("out", Value(UBits(3, 32)))),
          // C2: in=30,s0_ov=1,s2_ov=0. S2 stalls -> S1 stalls.
          //     S1_iv=0 because S0 stalled in C1.
          //     S1 sees iv=0, so s0_or = !0 || s1_done = 1.
          //     S0 proceeds. p0 gets 31.
          //     S1 input is p0_reg=11, iv=0. S1 stalled by S2. p1 retains 22.
          //     S2 input is p1_reg=22 from C1. out=22+3=25.
          UnorderedElementsAre(Pair("out", Value(UBits(25, 32)))),
          // C3: in=40,s0_ov=1,s2_ov=1.
          //     S2 recovers, accepts from S1. S1 recovers.
          //     S0 ran in C2, so s1_iv=1 for C3. S1 gets p0=31.
          //     p1 gets 31*2=62.
          //     S2 input is p1_reg=22 (S1 stalled in C2), iv=0.
          //     out=22+3=25.
          UnorderedElementsAre(Pair("out", Value(UBits(25, 32)))),
          // C4: in=50,s0_ov=1,s2_ov=1.
          //     p0 gets 40+1=41.
          //     S1 input p0_reg=31, iv=1. p1 gets 41*2=82.
          //     S2 input p1_reg=62, iv=1. out=62+3=65.
          UnorderedElementsAre(Pair("out", Value(UBits(65, 32)))),
          // C5: in=60,s0_ov=1,s2_ov=1.
          //     p0 gets 50+1=51.
          //     p1 gets 41*2=82
          //     S2 input p1_reg=82, iv=1. out=82+3=85
          UnorderedElementsAre(Pair("out", Value(UBits(85, 32))))));
}

}  // namespace
}  // namespace xls::codegen
