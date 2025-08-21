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

#include "xls/codegen/passes_ng/block_pipeline_inserter.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/codegen/passes_ng/passes_ng_test_fixtures.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/proc.h"

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAre;

TEST_F(SlotTestBase, PipelineInsertionShiftsByOneCycle) {
  XLS_ASSERT_OK_AND_ASSIGN(BlockAndSlots block_and_slots,
                           CreateTestProcAndAssociatedBlock(TestName()));

  Block* block = block_and_slots.block;
  Proc* proc = block_and_slots.proc;

  XLS_ASSERT_OK_AND_ASSIGN(ChannelInterface * send_y,
                           proc->GetSendChannelInterface("y"));

  BlockMetadata block_metadata(block);
  block_metadata.AddChannelMetadata(BlockChannelMetadata(send_y)
                                        .AddSlot(block_and_slots.slot_s)
                                        .SetIsInternalChannel());

  // Simulate before inserting the pipeline flop.
  std::vector<uint64_t> input_sequence = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  SimulationResults results;

  XLS_ASSERT_OK_AND_ASSIGN(
      results, InterpretTestBlock(block, input_sequence, /*cycle_count=*/11,
                                  /*lambda_source=*/1.0, /*lambda_sink=*/1.0));
  ASSERT_EQ(results.sinks.size(), 1);
  EXPECT_THAT(
      results.sinks[0].GetOutputCycleSequenceAsUint64(),
      IsOkAndHolds(ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, std::nullopt)));

  // After inserting the pipeline flop, the output is shifted by one cycle.
  XLS_ASSERT_OK(
      InsertPipelineIntoBlock(CodegenOptions(), block_metadata).status());

  XLS_ASSERT_OK_AND_ASSIGN(
      results, InterpretTestBlock(block, input_sequence, /*cycle_count=*/11,
                                  /*lambda_source=*/1.0, /*lambda_sink=*/1.0));
  ASSERT_EQ(results.sinks.size(), 1);
  EXPECT_THAT(
      results.sinks[0].GetOutputCycleSequenceAsUint64(),
      IsOkAndHolds(ElementsAre(std::nullopt, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)));
}

TEST_F(SlotTestBase, PipelineInsertionShiftsByOneCycleWithoutReset) {
  XLS_ASSERT_OK_AND_ASSIGN(
      BlockAndSlots block_and_slots,
      CreateTestProcAndAssociatedBlock(TestName(), /*use_non_blocking=*/false,
                                       /*insert_reset=*/false));

  Block* block = block_and_slots.block;
  Proc* proc = block_and_slots.proc;

  XLS_ASSERT_OK_AND_ASSIGN(ChannelInterface * send_y,
                           proc->GetSendChannelInterface("y"));

  BlockMetadata block_metadata(block);
  block_metadata.AddChannelMetadata(BlockChannelMetadata(send_y)
                                        .AddSlot(block_and_slots.slot_s)
                                        .SetIsInternalChannel());

  // Simulate before inserting the pipeline flop.
  std::vector<uint64_t> input_sequence = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  SimulationResults results;

  XLS_ASSERT_OK_AND_ASSIGN(
      results, InterpretTestBlock(block, input_sequence, /*cycle_count=*/11,
                                  /*lambda_source=*/1.0, /*lambda_sink=*/1.0));
  ASSERT_EQ(results.sinks.size(), 1);
  EXPECT_THAT(
      results.sinks[0].GetOutputCycleSequenceAsUint64(),
      IsOkAndHolds(ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, std::nullopt)));

  // After inserting the pipeline flop, the output is shifted by one cycle.
  XLS_ASSERT_OK(
      InsertPipelineIntoBlock(CodegenOptions(), block_metadata).status());

  XLS_ASSERT_OK_AND_ASSIGN(
      results, InterpretTestBlock(block, input_sequence, /*cycle_count=*/11,
                                  /*lambda_source=*/1.0, /*lambda_sink=*/1.0));

  ASSERT_EQ(results.sinks.size(), 1);
  EXPECT_THAT(
      results.sinks[0].GetOutputCycleSequenceAsUint64(),
      IsOkAndHolds(ElementsAre(std::nullopt, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)));
}

}  // namespace
}  // namespace xls::verilog
