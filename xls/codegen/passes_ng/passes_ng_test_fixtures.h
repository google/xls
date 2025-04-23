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

#ifndef XLS_CODEGEN_PASSES_NG_PASSES_NG_TEST_FIXTURES_H_
#define XLS_CODEGEN_PASSES_NG_PASSES_NG_TEST_FIXTURES_H_

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/codegen/block_conversion_test_fixture.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/register.h"
#include "xls/ir/verifier.h"

// Defines fixtures for testing codegen ng block passes.
//
// TODO(tedhong): Merge with codegen/block_conversion_test_fixture.h

namespace xls::verilog {

// Test fixture for testing Slots and Adapters.
class SlotTestBase : public BlockConversionTestFixture {
 protected:
  SlotTestBase() : package_(CreatePackage()) {}

  struct BlockAndSlots {
    Block* block;
    BlockRDVSlot slot_r;
    BlockRDVSlot slot_s;
  };

  // Create a proc/block with the following structure:
  //           -------------------------
  //           |          [1]          |
  // x_ready   |     ---   |    ---    |  y_ready
  //        ---|<---| S |--|---| S |<--|---
  // x         |    | l |  |   | l |   |  y
  //        ---|--->| o |--+---| o |-->|---
  // x_valid   |    | t |      | t |   |  y_valid
  //        ---|--->|(r)|------|(s)|-->|---
  //           |     ---        ---    |
  //           -------------------------
  //
  // That reads from input channel x, adds one to the input, and writes to
  // output channel y.
  absl::StatusOr<BlockAndSlots> CreateTestProcAndAssociatedBlock(
      std::string_view name) {
    // Create proc.
    TokenlessProcBuilder pb(NewStyleProc(), name, "tkn", package_.get());
    XLS_ASSIGN_OR_RETURN(ReceiveChannelInterface * x_ch,
                         pb.AddInputChannel("x", package_->GetBitsType(32)));
    XLS_ASSIGN_OR_RETURN(SendChannelInterface * y_ch,
                         pb.AddOutputChannel("y", package_->GetBitsType(32)));
    BValue x_value = pb.Receive(x_ch);
    pb.Send(y_ch, pb.Add(x_value, pb.Literal(UBits(1, 32))));
    XLS_RETURN_IF_ERROR(pb.Build().status());

    // Create equivalent block.
    BlockBuilder bb(name, package_.get());
    XLS_RETURN_IF_ERROR(bb.block()->AddClockPort("clk"));
    XLS_RETURN_IF_ERROR(bb.block()
                            ->AddResetPort("rst",
                                           ResetBehavior{
                                               .asynchronous = false,
                                               .active_low = false,
                                           })
                            .status());

    BValue x = bb.InputPort("x", package_->GetBitsType(32));
    BValue y = bb.OutputPort("y", bb.Add(x, bb.Literal(UBits(1, 32))));
    BValue x_valid = bb.InputPort("x_valid", package_->GetBitsType(1));
    BValue y_valid = bb.OutputPort("y_valid", x_valid);

    BValue y_ready = bb.InputPort("y_ready", package_->GetBitsType(1));
    BValue x_ready = bb.OutputPort("x_ready", y_ready);
    XLS_ASSIGN_OR_RETURN(Block * block, bb.Build());

    // Add slots.
    XLS_ASSIGN_OR_RETURN(
        BlockRDVSlot slot_r,
        BlockRDVSlot::CreateReceiveSlot(
            "slot_r", RDVNodeGroup{x_ready.node(), x.node(), x_valid.node()},
            block));
    XLS_ASSIGN_OR_RETURN(
        BlockRDVSlot slot_s,
        BlockRDVSlot::CreateSendSlot(
            "slot_s", RDVNodeGroup{y_ready.node(), y.node(), y_valid.node()},
            block));

    return BlockAndSlots{block, slot_r, slot_s};
  }

  struct SimulationResults {
    BlockIOResultsAsUint64 io_results;
    std::vector<ChannelSource> sources;
    std::vector<ChannelSink> sinks;
  };

  // Interpret a block created with CreateTestProcAndAssociatedBlock with the
  // given input sequence.
  absl::StatusOr<SimulationResults> InterpretTestBlock(
      Block* block, absl::Span<const uint64_t> input_sequence,
      int64_t cycle_count) {
    SimulationResults results{
        .sources = {ChannelSource("x", "x_valid", "x_ready", 0.5, block)},
        .sinks = {ChannelSink("y", "y_valid", "y_ready", 0.5, block)},
    };

    XLS_RETURN_IF_ERROR(results.sources[0].SetDataSequence(input_sequence));

    std::vector<absl::flat_hash_map<std::string, uint64_t>> signals(
        100, {{"rst", 0}});

    XLS_ASSIGN_OR_RETURN(results.io_results,
                         InterpretChannelizedSequentialBlockWithUint64(
                             block, absl::MakeSpan(results.sources),
                             absl::MakeSpan(results.sinks), signals));

    // Add a cycle count for easier comparison with simulation results.
    if (VLOG_IS_ON(1)) {
      std::vector<absl::flat_hash_map<std::string, uint64_t>>& blk_inputs =
          results.io_results.inputs;
      std::vector<absl::flat_hash_map<std::string, uint64_t>> blk_outputs =
          results.io_results.outputs;
      XLS_RETURN_IF_ERROR(
          SetIncrementingSignalOverCycles(0, blk_outputs.size() - 1, "cycle", 0,
                                          blk_outputs)
              .status());

      VLOG(1) << "Signal Trace";
      XLS_RETURN_IF_ERROR(VLogTestPipelinedIO(
          std::vector<SignalSpec>{{"cycle", SignalType::kOutput},
                                  {"rst", SignalType::kInput},
                                  {"x", SignalType::kInput},
                                  {"x_valid", SignalType::kInput},
                                  {"x_ready", SignalType::kOutput},
                                  {"y", SignalType::kOutput},
                                  {"y_valid", SignalType::kOutput},
                                  {"y_ready", SignalType::kInput}},
          /*column_width=*/10, blk_inputs, blk_outputs));
    }

    return results;
  }

  std::unique_ptr<Package> package_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_PASSES_NG_TEST_FIXTURES_H_
