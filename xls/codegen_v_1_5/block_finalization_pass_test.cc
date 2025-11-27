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

#include "xls/codegen_v_1_5/block_finalization_pass.h"

#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/functional/overload.h"
#include "absl/status/statusor.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/ir/source_location.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::Each;
using ::testing::IsFalse;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;

class BlockFinalizationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return BlockFinalizationPass().Run(p, BlockConversionPassOptions(),
                                       &results);
  }
};

TEST_F(BlockFinalizationPassTest, BasicScheduledBlock) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  ScheduledBlock* sblk = down_cast<ScheduledBlock*>(sbb.block());

  XLS_ASSERT_OK(sblk->AddClockPort("clk"));
  BValue in = sbb.InputPort("in", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Register * data_reg,
                           sblk->AddRegister("data_reg", p->GetBitsType(32)));

  BValue iv0 = sbb.Literal(UBits(1, 1));
  BValue or0 = sbb.Literal(UBits(1, 1));
  sbb.StartStage(iv0, or0);
  BValue data_read = sbb.RegisterRead(data_reg);
  BValue add0 = sbb.Add(in, data_read);
  BValue aiv0 = sbb.Literal(UBits(1, 1));
  BValue ov0 = sbb.Literal(UBits(1, 1));
  sbb.EndStage(aiv0, ov0);

  BValue iv1 = sbb.Literal(UBits(1, 1));
  BValue or1 = sbb.Literal(UBits(1, 1));
  sbb.StartStage(iv1, or1);
  BValue neg1 =
      sbb.Subtract(sbb.Literal(UBits(0, 32), SourceInfo(), "neg_zero"), add0);
  sbb.RegisterWrite(data_reg, neg1);
  sbb.OutputPort("out", neg1);
  BValue aiv1 = sbb.Literal(UBits(1, 1));
  BValue ov1 = sbb.Literal(UBits(1, 1));
  sbb.EndStage(aiv1, ov1);

  XLS_ASSERT_OK_AND_ASSIGN(Block * scheduled_block, sbb.Build());
  ASSERT_EQ(scheduled_block, sblk);
  ASSERT_TRUE(scheduled_block->IsScheduled());
  EXPECT_EQ(down_cast<ScheduledBlock*>(scheduled_block)->stages().size(), 2);
  int64_t node_count_before = scheduled_block->node_count();
  int64_t port_count_before = scheduled_block->GetPorts().size();
  int64_t reg_count_before = scheduled_block->GetRegisters().size();

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));
  ASSERT_FALSE(block->IsScheduled());
  EXPECT_EQ(block->GetPorts().size(), port_count_before);
  EXPECT_EQ(block->GetRegisters().size(), reg_count_before);
  EXPECT_EQ(block->node_count(), node_count_before);
  ASSERT_TRUE(block->GetClockPort().has_value());
  EXPECT_EQ(block->GetClockPort()->name, "clk");
  EXPECT_EQ(block->GetInputPort("in").value()->direction(),
            PortDirection::kInput);
  EXPECT_EQ(block->GetOutputPort("out").value()->direction(),
            PortDirection::kOutput);
  EXPECT_TRUE(block->GetRegister("data_reg").ok());

  EXPECT_THAT(p->blocks(),
              Each(Pointee(Property(&Block::IsScheduled, IsFalse()))));
}

TEST_F(BlockFinalizationPassTest, EmptyScheduledBlock) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  BValue iv0 = sbb.Literal(UBits(1, 1));
  BValue or0 = sbb.Literal(UBits(1, 1));
  sbb.StartStage(iv0, or0);
  BValue aiv0 = sbb.Literal(UBits(1, 1));
  BValue ov0 = sbb.Literal(UBits(1, 1));
  sbb.EndStage(aiv0, ov0);
  XLS_ASSERT_OK_AND_ASSIGN(Block * scheduled_block, sbb.Build());
  ASSERT_TRUE(scheduled_block->IsScheduled());
  EXPECT_EQ(scheduled_block->node_count(), 4);  // 4 literals for stage signals
  EXPECT_EQ(scheduled_block->GetPorts().size(), 0);
  EXPECT_EQ(scheduled_block->GetRegisters().size(), 0);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));
  ASSERT_FALSE(block->IsScheduled());
  EXPECT_EQ(block->node_count(), 4);
  EXPECT_EQ(block->GetPorts().size(), 0);
  EXPECT_EQ(block->GetRegisters().size(), 0);

  EXPECT_THAT(p->blocks(),
              Each(Pointee(Property(&Block::IsScheduled, IsFalse()))));
}

TEST_F(BlockFinalizationPassTest, ResetAndPortOrder) {
  auto p = CreatePackage();
  ScheduledBlockBuilder sbb(TestName(), p.get());
  ScheduledBlock* sblk = down_cast<ScheduledBlock*>(sbb.block());

  XLS_ASSERT_OK(sblk->AddClockPort("clk"));
  BValue rst = sbb.InputPort("rst", p->GetBitsType(1));
  XLS_ASSERT_OK(sblk->SetResetPort(
      rst.node()->As<InputPort>(),
      ResetBehavior{.asynchronous = false, .active_low = false}));
  BValue in = sbb.InputPort("in", p->GetBitsType(8));

  BValue iv0 = sbb.Literal(UBits(1, 1));
  BValue or0 = sbb.Literal(UBits(1, 1));
  sbb.StartStage(iv0, or0);
  BValue aiv0 = sbb.Literal(UBits(1, 1));
  BValue ov0 = sbb.Literal(UBits(1, 1));
  sbb.EndStage(aiv0, ov0);
  sbb.OutputPort("out", in);

  XLS_ASSERT_OK_AND_ASSIGN(Block * scheduled_block, sbb.Build());
  ASSERT_TRUE(scheduled_block->IsScheduled());
  ASSERT_TRUE(scheduled_block->GetResetPort().has_value());
  EXPECT_EQ((*scheduled_block->GetResetPort())->name(), "rst");
  ASSERT_TRUE(scheduled_block->GetResetBehavior().has_value());
  EXPECT_FALSE(scheduled_block->GetResetBehavior()->asynchronous);
  EXPECT_FALSE(scheduled_block->GetResetBehavior()->active_low);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock(TestName()));
  ASSERT_FALSE(block->IsScheduled());
  ASSERT_TRUE(block->GetResetPort().has_value());
  EXPECT_EQ((*block->GetResetPort())->name(), "rst");
  ASSERT_TRUE(block->GetResetBehavior().has_value());
  EXPECT_FALSE(block->GetResetBehavior()->asynchronous);
  EXPECT_FALSE(block->GetResetBehavior()->active_low);

  std::vector<std::string> port_names;
  for (const auto& port : block->GetPorts()) {
    std::visit(
        absl::Overload(
            [&](InputPort* p) { port_names.push_back(std::string(p->name())); },
            [&](OutputPort* p) {
              port_names.push_back(std::string(p->name()));
            },
            [&](Block::ClockPort* p) {
              port_names.push_back(std::string(p->name));
            }),
        port);
  }
  EXPECT_THAT(port_names, UnorderedElementsAre("clk", "rst", "in", "out"));

  EXPECT_THAT(p->blocks(),
              Each(Pointee(Property(&Block::IsScheduled, IsFalse()))));
}

TEST_F(BlockFinalizationPassTest, Idempotency) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  BValue in = bb.InputPort("in", p->GetBitsType(8));
  bb.OutputPort("out", in);
  XLS_ASSERT_OK(bb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls::codegen
