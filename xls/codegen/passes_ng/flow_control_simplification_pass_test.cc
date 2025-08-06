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

#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/codegen/passes_ng/flow_control_simplification.h"
#include "xls/codegen/passes_ng/passes_ng_test_fixtures.h"
#include "xls/codegen/passes_ng/stage_to_block_conversion_metadata.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/proc.h"

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::IsSupersetOf;

TEST_F(SlotTestBase, ReadyFlowControlCanBeRemoved) {
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

  XLS_ASSERT_OK(RemoveReadyBackpressure(block_metadata));

  VLOG(1) << "After removing ready backpressure:";
  XLS_VLOG_LINES(1, block->DumpIr());

  // Simulate with a ready signal of 1.
  std::vector<uint64_t> input_sequence = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  SimulationResults results;

  XLS_ASSERT_OK_AND_ASSIGN(
      results, InterpretTestBlock(block, input_sequence, /*cycle_count=*/100,
                                  /*lambda_source=*/0.25, /*lambda_sink=*/1.0));
  ASSERT_EQ(results.sinks.size(), 1);
  EXPECT_THAT(results.sinks[0].GetOutputCycleSequenceAsUint64(),
              IsOkAndHolds(IsSupersetOf({2, 3, 4, 5, 6, 7, 8, 9, 10, 11})));

  // Simulate with a ready signal with prob 0.5, which should result in
  // backpressure and data dropped.
  XLS_ASSERT_OK_AND_ASSIGN(
      results, InterpretTestBlock(block, input_sequence, /*cycle_count=*/100,
                                  /*lambda_source=*/1.0, /*lambda_sink=*/0.5));

  XLS_ASSERT_OK_AND_ASSIGN(std::vector<std::optional<uint64_t>> output_sequence,
                           results.sinks[0].GetOutputCycleSequenceAsUint64());
  for (int64_t i = 0; i < input_sequence.size(); ++i) {
    if (output_sequence[i].has_value()) {
      EXPECT_EQ(output_sequence[i].value(), input_sequence[i] + 1);
    }
  }
}

}  // namespace
}  // namespace xls::verilog
