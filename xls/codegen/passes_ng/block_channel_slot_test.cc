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

#include "xls/codegen/passes_ng/block_channel_slot.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/passes_ng/passes_ng_test_fixtures.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/verifier.h"

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

TEST_F(SlotTestBase, SlotsWillPassthroughValuesTest) {
  XLS_ASSERT_OK_AND_ASSIGN(BlockAndSlots block_and_slots,
                           CreateTestProcAndAssociatedBlock(TestName()));

  Block* block = block_and_slots.block;
  ASSERT_NE(block, nullptr);
  XLS_VLOG_LINES(1, block->DumpIr());

  std::vector<uint64_t> input_sequence = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  XLS_ASSERT_OK_AND_ASSIGN(
      SimulationResults results,
      InterpretTestBlock(block, input_sequence, /*cycle_count=*/100));

  EXPECT_THAT(results.sinks[0].GetOutputSequenceAsUint64(),
              IsOkAndHolds(ElementsAre(1, 2, 3, 4, 5, 6, 7, 8, 9, 10)));
}

TEST_F(SlotTestBase, ModifyReceiveAndSendChannelValuesViaSlotsTest) {
  XLS_ASSERT_OK_AND_ASSIGN(BlockAndSlots block_and_slots,
                           CreateTestProcAndAssociatedBlock(TestName()));

  Block* block = block_and_slots.block;
  ASSERT_NE(block, nullptr);
  XLS_VLOG_LINES(1, block->DumpIr());

  // Before simulation, modify the slots so that the output is
  // y = (2*x + 1)^2
  RDVNodeGroup slot_s_upstream = block_and_slots.slot_s.GetUpstreamBufferBank();
  RDVNodeGroup slot_s_downstream =
      block_and_slots.slot_s.GetDownstreamBufferBank();
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * slot_s_upstream_square,
      block->MakeNode<ArithOp>(slot_s_upstream.data->loc(),
                               slot_s_upstream.data, slot_s_upstream.data,
                               /*width=*/32, Op::kUMul));
  XLS_ASSERT_OK(
      slot_s_downstream.data->ReplaceOperandNumber(0, slot_s_upstream_square));

  RDVNodeGroup slot_r_upstream = block_and_slots.slot_r.GetUpstreamBufferBank();
  RDVNodeGroup slot_r_downstream =
      block_and_slots.slot_r.GetDownstreamBufferBank();
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * slot_r_upstream_double,
      block->MakeNode<BinOp>(slot_r_upstream.data->loc(), slot_r_upstream.data,
                             slot_r_upstream.data, Op::kAdd));
  XLS_ASSERT_OK(
      slot_r_downstream.data->ReplaceOperandNumber(0, slot_r_upstream_double));

  std::vector<uint64_t> input_sequence = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  XLS_ASSERT_OK_AND_ASSIGN(
      SimulationResults results,
      InterpretTestBlock(block, input_sequence, /*cycle_count=*/100));

  EXPECT_THAT(
      results.sinks[0].GetOutputSequenceAsUint64(),
      IsOkAndHolds(ElementsAre(1, 9, 25, 49, 81, 121, 169, 225, 289, 361)));
}

}  // namespace
}  // namespace xls::verilog
