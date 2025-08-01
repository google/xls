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
#include <utility>
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
#include "xls/ir/source_location.h"
#include "xls/ir/verifier.h"

// Defines fixtures for testing codegen ng block passes.
//
// TODO(tedhong): Merge with codegen/block_conversion_test_fixture.h

namespace xls::verilog {

// Test fixture for testing Slots and Adapters.
class SlotTestBase : public BlockConversionTestFixture {
 protected:
  // Used to map IR nodes to Block IR nodes.
  using IrToBlockIrMap = absl::flat_hash_map<const Node*, Node*>;

  SlotTestBase() : package_(CreatePackage()) {}

  // A structure to hold the proc, block, and associated slots created by
  // the test.
  struct BlockAndSlots {
    Proc* proc;
    Block* block;
    BlockRDVSlot slot_r;
    BlockRDVSlot slot_s;
    IrToBlockIrMap node_map;
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
      std::string_view name, bool use_non_blocking = false,
      bool insert_reset = true) {
    // Create block builder with a single token that is simply reused.
    BlockBuilder bb(name, package_.get());
    BValue block_token = bb.AfterAll({});

    IrToBlockIrMap node_map;

    // Create proc and populate the node map with tokens as we go.
    TokenlessProcBuilder pb(NewStyleProc(), name, "tkn", package_.get());
    node_map[pb.CurrentToken().node()] = block_token.node();

    XLS_ASSIGN_OR_RETURN(ReceiveChannelInterface * x_ch,
                         pb.AddInputChannel("x", package_->GetBitsType(32)));
    XLS_ASSIGN_OR_RETURN(SendChannelInterface * y_ch,
                         pb.AddOutputChannel("y", package_->GetBitsType(32)));

    BValue x_value;
    if (use_non_blocking) {
      // Ignore the valid bit for purposes of this test.
      x_value =
          pb.ReceiveNonBlocking(x_ch, /*loc=*/SourceInfo(), "recv_x").first;
    } else {
      x_value = pb.Receive(x_ch, /*loc=*/SourceInfo(), "recv_x");
    }
    node_map[pb.CurrentToken().node()] = block_token.node();
    pb.Send(y_ch, pb.Add(x_value, pb.Literal(UBits(1, 32))),
            /*loc=*/SourceInfo(), "send_y");
    node_map[pb.CurrentToken().node()] = block_token.node();
    XLS_ASSIGN_OR_RETURN(Proc * proc, pb.Build());

    // Create equivalent block.
    XLS_RETURN_IF_ERROR(bb.block()->AddClockPort("clk"));
    if (insert_reset) {
      XLS_RETURN_IF_ERROR(bb.block()
                              ->AddResetPort("rst",
                                             ResetBehavior{
                                                 .asynchronous = false,
                                                 .active_low = false,
                                             })
                              .status());
    }

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

    return BlockAndSlots{proc, block, slot_r, slot_s, std::move(node_map)};
  }

  struct SimulationResults {
    BlockIOResultsAsUint64 io_results;
    std::vector<ChannelSource> sources;
    std::vector<ChannelSink> sinks;
  };

  // Interpret a block created with CreateTestProcAndAssociatedBlock with the
  // given input sequence.
  //
  // lambda is the probability that data would be placed on the input channel
  // or read from the output channel.
  absl::StatusOr<SimulationResults> InterpretTestBlock(
      Block* block, absl::Span<const uint64_t> input_sequence,
      int64_t cycle_count, double lambda = 0.5) {
    SimulationResults results{
        .sources = {ChannelSource("x", "x_valid", "x_ready", lambda, block)},
        .sinks = {ChannelSink("y", "y_valid", "y_ready", lambda, block)},
    };

    XLS_RETURN_IF_ERROR(results.sources[0].SetDataSequence(input_sequence));

    std::vector<absl::flat_hash_map<std::string, uint64_t>> signals(
        cycle_count,
        block->GetResetPort().has_value()
            ? absl::flat_hash_map<std::string, uint64_t>{{"rst", 0}}
            : absl::flat_hash_map<std::string, uint64_t>());

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
      if (block->GetResetPort().has_value()) {
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
      } else {
        XLS_RETURN_IF_ERROR(VLogTestPipelinedIO(
            std::vector<SignalSpec>{{"cycle", SignalType::kOutput},
                                    {"x", SignalType::kInput},
                                    {"x_valid", SignalType::kInput},
                                    {"x_ready", SignalType::kOutput},
                                    {"y", SignalType::kOutput},
                                    {"y_valid", SignalType::kOutput},
                                    {"y_ready", SignalType::kInput}},
            /*column_width=*/10, blk_inputs, blk_outputs));
      }
    }

    return results;
  }
  // Package created for test.
  std::unique_ptr<Package> package_;
};

// Test fixture for testing Slots and Adapters with predicates
class PredicatedSlotTestBase : public BlockConversionTestFixture {
 protected:
  // Used to map IR nodes to Block IR nodes.
  using IrToBlockIrMap = absl::flat_hash_map<const Node*, Node*>;

  PredicatedSlotTestBase() : package_(CreatePackage()) {}

  // A structure to hold the proc, block, and associated slots created by
  // the test.
  struct BlockAndSlots {
    Proc* proc;
    Block* block;
    BlockRDVSlot slot_r;
    BlockRDVSlot slot_s;
    IrToBlockIrMap node_map;
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
  //           |                       |
  // pred_rdy  |     ---               |
  //        ---|<---| S |              |
  // pred      |    | l |              |
  //        ---|--->| o |              |
  // pred_vd   |    | t |              |
  //        ---|--->|(p)|              |
  //           |     ---               |
  //           |                       |
  //           -------------------------
  //
  // That read from input channel pred, and if it is 1 then it
  // reads from input channel x, adds one to the input, and writes to
  // output channel y.
  //
  absl::StatusOr<BlockAndSlots> CreateTestProcAndAssociatedBlock(
      std::string_view name) {
    // Create block builder with a single token that is simply reused.
    BlockBuilder bb(name, package_.get());
    BValue block_token = bb.AfterAll({});

    IrToBlockIrMap node_map;

    // Create proc and populate the node map with tokens as we go.
    TokenlessProcBuilder pb(NewStyleProc(), name, "tkn", package_.get());
    node_map[pb.CurrentToken().node()] = block_token.node();

    XLS_ASSIGN_OR_RETURN(ReceiveChannelInterface * pred_ch,
                         pb.AddInputChannel("pred", package_->GetBitsType(1)));
    XLS_ASSIGN_OR_RETURN(ReceiveChannelInterface * x_ch,
                         pb.AddInputChannel("x", package_->GetBitsType(32)));
    XLS_ASSIGN_OR_RETURN(SendChannelInterface * y_ch,
                         pb.AddOutputChannel("y", package_->GetBitsType(32)));

    BValue pred_value = pb.Receive(pred_ch, /*loc=*/SourceInfo(), "recv_pred");
    node_map[pb.CurrentToken().node()] = block_token.node();
    BValue x_value =
        pb.ReceiveIf(x_ch, pred_value, /*loc=*/SourceInfo(), "recv_x");
    node_map[pb.CurrentToken().node()] = block_token.node();
    pb.SendIf(y_ch, pred_value, pb.Add(x_value, pb.Literal(UBits(1, 32))),
              /*loc=*/SourceInfo(), "send_y");
    node_map[pb.CurrentToken().node()] = block_token.node();
    XLS_ASSIGN_OR_RETURN(Proc * proc, pb.Build());

    // Create equivalent block.
    XLS_RETURN_IF_ERROR(bb.block()->AddClockPort("clk"));
    XLS_RETURN_IF_ERROR(bb.block()
                            ->AddResetPort("rst",
                                           ResetBehavior{
                                               .asynchronous = false,
                                               .active_low = false,
                                           })
                            .status());

    BValue pred = bb.InputPort("pred", package_->GetBitsType(1));
    node_map[pred_value.node()] = pred.node();
    BValue x = bb.InputPort("x", package_->GetBitsType(32));
    BValue y = bb.OutputPort("y", bb.Add(x, bb.Literal(UBits(1, 32))));

    BValue pred_valid = bb.InputPort("pred_valid", package_->GetBitsType(1));
    BValue x_valid = bb.InputPort("x_valid", package_->GetBitsType(1));
    BValue y_valid = bb.OutputPort("y_valid", bb.And(pred_valid, x_valid));

    BValue y_ready = bb.InputPort("y_ready", package_->GetBitsType(1));
    bb.OutputPort("pred_ready", y_ready);
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

    return BlockAndSlots{proc, block, slot_r, slot_s, std::move(node_map)};
  }

  // Package created for test.
  std::unique_ptr<Package> package_;
};

}  // namespace xls::verilog

#endif  // XLS_CODEGEN_PASSES_NG_PASSES_NG_TEST_FIXTURES_H_
