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

#include "xls/codegen_v_1_5/idle_insertion_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {

absl::StatusOr<bool> IdleInsertionPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  if (!options.codegen_options.add_idle_output()) {
    return false;
  }

  bool changed = false;
  for (std::unique_ptr<Block>& block : package->blocks()) {
    if (!block->IsScheduled()) {
      continue;
    }
    ScheduledBlock* scheduled_block =
        absl::down_cast<ScheduledBlock*>(block.get());
    if (scheduled_block->stages().empty()) {
      continue;
    }

    // If the block represents a function without input-valid control, then it's
    // never idle.
    const bool is_function = scheduled_block->source() != nullptr &&
                             scheduled_block->source()->IsFunction();
    if (is_function && !options.codegen_options.valid_control().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          Node * idle_signal,
          scheduled_block->MakeNode<Literal>(SourceInfo(), Value(UBits(0, 1))));
      XLS_RETURN_IF_ERROR(
          scheduled_block->AddOutputPort("idle", idle_signal).status());
      changed = true;
      continue;
    }

    std::vector<Node*> stage_idle_signals;
    stage_idle_signals.reserve(scheduled_block->stages().size());

    // The block is idle if *every* stage is effectively idle.
    // A stage is idle if it is:
    //   1. Empty (nothing coming down the pipeline),
    //   2. Blocked on active inputs, or
    //   3. All active outputs have resolved, but we're stalled due to internal
    //      backpressure (the next stage isn't ready).
    for (int64_t i = 0; i < scheduled_block->stages().size(); ++i) {
      const Stage& stage = scheduled_block->stages()[i];
      XLS_RET_CHECK(stage.IsControlled());

      // NOTE: The stage is idle iff `empty || blocked || stalled`.
      //       That's equivalent to `NAND(!empty, !blocked, !stalled)`.
      //       Since `!empty` and `!blocked` correspond to existing flow-control
      //       signals, this requires fewer operation nodes.

      // The stage is not empty (pipeline inputs are present) iff `inputs_valid`
      // is 1.
      Node* stage_not_empty = stage.inputs_valid();

      // The stage is not blocked on active inputs (all active inputs are
      // present or disabled) iff `active_inputs_valid` is 1.
      Node* stage_not_blocked_on_active_inputs = stage.active_inputs_valid();

      // The stage is stalled due to internal backpressure iff `outputs_valid`
      // is 1 and `outputs_ready` is 0.
      // NOTE: `!stalled` is `!(outputs_valid && !outputs_ready)`. That's
      //       equivalent to `(!outputs_valid || outputs_ready)`, which requires
      //       fewer operations.
      XLS_ASSIGN_OR_RETURN(Node * outputs_not_finished,
                           scheduled_block->MakeNode<UnOp>(
                               SourceInfo(), stage.outputs_valid(), Op::kNot));
      XLS_ASSIGN_OR_RETURN(Node * stage_not_stalled,
                           scheduled_block->MakeNode<NaryOp>(
                               SourceInfo(),
                               absl::MakeConstSpan({outputs_not_finished,
                                                    stage.outputs_ready()}),
                               Op::kOr));

      XLS_ASSIGN_OR_RETURN(
          Node * stage_idle,
          scheduled_block->MakeNodeWithName<NaryOp>(
              SourceInfo(),
              absl::MakeConstSpan({stage_not_empty,
                                   stage_not_blocked_on_active_inputs,
                                   stage_not_stalled}),
              Op::kNand, absl::StrFormat("stage_%d_idle", i)));
      stage_idle_signals.push_back(stage_idle);
    }

    XLS_ASSIGN_OR_RETURN(Node * pipeline_idle,
                         NaryAndIfNeeded(scheduled_block, stage_idle_signals,
                                         /*name=*/"pipeline_idle"));

    // Except... we also need to check if any I/O flops are active. The good
    // news is that they're only active in isolation when they are completing an
    // external-facing ready/valid handshake... *unless* we're flopping
    // function inputs, which gets a bit trickier.
    std::vector<Node*> idle_signals;
    idle_signals.push_back(pipeline_idle);
    if (is_function) {
      // If we have no valid control, we shouldn't be here; we should have
      // returned earlier with an always-false idle signal.
      CHECK(options.codegen_options.valid_control().has_value());

      // If we're flopping function inputs with any latency, we need to signal
      // as active the same cycle that the input becomes valid, while it's still
      // propagating through the flop.
      if (options.codegen_options.flop_inputs() &&
          options.codegen_options.flop_inputs_kind() !=
              verilog::CodegenOptions::IOKind::kZeroLatencyBuffer) {
        CHECK(options.codegen_options.flop_inputs_kind() ==
                  verilog::CodegenOptions::IOKind::kFlop ||
              options.codegen_options.flop_inputs_kind() ==
                  verilog::CodegenOptions::IOKind::kSkidBuffer);
        XLS_ASSIGN_OR_RETURN(
            InputPort * input_port,
            scheduled_block->GetInputPort(
                options.codegen_options.valid_control()->input_name()));
        XLS_ASSIGN_OR_RETURN(Node * input_is_not_valid,
                             scheduled_block->MakeNode<UnOp>(
                                 SourceInfo(), input_port, Op::kNot));
        idle_signals.push_back(input_is_not_valid);
      }

      // If we're flopping function outputs with any latency, we need to signal
      // as active any time the output is valid - which means an additional
      // cycle after our last stage is active. (We have no incoming ready signal
      // that could cause us to stall.)
      if (options.codegen_options.flop_outputs() &&
          options.codegen_options.flop_outputs_kind() !=
              verilog::CodegenOptions::IOKind::kZeroLatencyBuffer) {
        CHECK(options.codegen_options.flop_outputs_kind() ==
                  verilog::CodegenOptions::IOKind::kFlop ||
              options.codegen_options.flop_outputs_kind() ==
                  verilog::CodegenOptions::IOKind::kSkidBuffer);
        // We need to synthesize a signal that's high when the output is valid;
        // i.e., one cycle after the last stage is active.
        const std::string flop_name =
            absl::StrCat(scheduled_block->name(), "_output_active_flop");
        Node* output_valid = scheduled_block->stages().back().outputs_valid();
        XLS_ASSIGN_OR_RETURN(
            Register * output_valid_flop,
            scheduled_block->AddRegister(
                flop_name, output_valid->GetType(),
                scheduled_block->GetResetPort().has_value()
                    ? std::make_optional(ZeroOfType(output_valid->GetType()))
                    : std::nullopt));
        XLS_RETURN_IF_ERROR(scheduled_block
                                ->MakeNodeWithName<RegisterWrite>(
                                    output_valid->loc(), output_valid,
                                    /*load_enable=*/std::nullopt,
                                    /*reset=*/block->GetResetPort(),
                                    output_valid_flop,
                                    absl::StrCat(flop_name, "_write"))
                                .status());
        XLS_ASSIGN_OR_RETURN(Node * output_currently_valid,
                             scheduled_block->MakeNodeWithName<RegisterRead>(
                                 output_valid->loc(), output_valid_flop,
                                 absl::StrCat(flop_name, "_read")));
        XLS_ASSIGN_OR_RETURN(
            Node * output_is_not_valid,
            scheduled_block->MakeNode<UnOp>(SourceInfo(),
                                            output_currently_valid, Op::kNot));
        idle_signals.push_back(output_is_not_valid);
      }
    } else {
      for (const auto& [name, direction] :
           scheduled_block->GetChannelsWithMappedPorts()) {
        XLS_ASSIGN_OR_RETURN(
            ChannelPortMetadata metadata,
            scheduled_block->GetChannelPortMetadata(name, direction));

        // In case this signal is flopped (which could be due to channel
        // configuration, so we have to do this for every channel), we need to
        // suppress the idle signal any time the I/O handshake is completing.
        //
        // If the channel doesn't implement a ready/valid handshake (i.e., is a
        // single-value channel), it's presumed to never change in a way that
        // could affect idle status, regardless of flopping.
        Node* valid_signal = nullptr;
        Node* ready_signal = nullptr;
        if (metadata.valid_port.has_value() &&
            metadata.ready_port.has_value()) {
          switch (direction) {
            case ChannelDirection::kSend: {
              XLS_ASSIGN_OR_RETURN(ready_signal, scheduled_block->GetInputPort(
                                                     *metadata.ready_port));
              XLS_ASSIGN_OR_RETURN(
                  OutputPort * valid_port,
                  scheduled_block->GetOutputPort(*metadata.valid_port));
              valid_signal = valid_port->output_source();
              break;
            }
            case ChannelDirection::kReceive: {
              XLS_ASSIGN_OR_RETURN(valid_signal, scheduled_block->GetInputPort(
                                                     *metadata.valid_port));
              XLS_ASSIGN_OR_RETURN(
                  OutputPort * ready_port,
                  scheduled_block->GetOutputPort(*metadata.ready_port));
              ready_signal = ready_port->output_source();
              break;
            }
          }
        }
        if (valid_signal != nullptr && ready_signal != nullptr) {
          XLS_ASSIGN_OR_RETURN(
              Node * channel_inactive,
              scheduled_block->MakeNodeWithName<NaryOp>(
                  SourceInfo(),
                  absl::MakeConstSpan({valid_signal, ready_signal}), Op::kNand,
                  absl::StrCat("__", name, "_inactive")));
          idle_signals.push_back(channel_inactive);
        }
      }
    }

    XLS_ASSIGN_OR_RETURN(Node * idle_signal,
                         NaryAndIfNeeded(scheduled_block, idle_signals,
                                         /*name=*/"pipeline_and_io_idle"));

    XLS_RETURN_IF_ERROR(
        scheduled_block->AddOutputPort("idle", idle_signal).status());
    changed = true;
  }

  return changed;
}

}  // namespace xls::codegen
