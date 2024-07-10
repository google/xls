// Copyright 2021 The XLS Authors
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

#include "xls/codegen/block_conversion.h"

#include <algorithm>
#include <cstdint>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/codegen/bdd_io_analysis.h"
#include "xls/codegen/codegen_checker.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/codegen_wrapper_pass.h"
#include "xls/codegen/concurrent_stage_groups.h"
#include "xls/codegen/register_legalization_pass.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/casts.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/name_uniquer.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/passes/dataflow_simplification_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "re2/re2.h"

namespace xls {
namespace verilog {
namespace {

std::optional<PackageInterfaceProto::Function> FindFunctionInterface(
    const std::optional<PackageInterfaceProto>& src,
    std::string_view func_name) {
  if (!src) {
    return std::nullopt;
  }
  auto it = absl::c_find_if(src->functions(),
                            [&](const PackageInterfaceProto::Function& f) {
                              return f.base().name() == func_name;
                            });
  if (it != src->functions().end()) {
    return *it;
  }
  return std::nullopt;
}
std::optional<PackageInterfaceProto::Proc> FindProcInterface(
    const std::optional<PackageInterfaceProto>& src,
    std::string_view proc_name) {
  if (!src) {
    return std::nullopt;
  }
  auto it =
      absl::c_find_if(src->procs(), [&](const PackageInterfaceProto::Proc& f) {
        return f.base().name() == proc_name;
      });
  if (it != src->procs().end()) {
    return *it;
  }
  return std::nullopt;
}
std::optional<PackageInterfaceProto::Channel> FindChannelInterface(
    const std::optional<PackageInterfaceProto>& src,
    std::string_view chan_name) {
  if (!src) {
    return std::nullopt;
  }
  auto it = absl::c_find_if(src->channels(),
                            [&](const PackageInterfaceProto::Channel& f) {
                              return f.name() == chan_name;
                            });
  if (it != src->channels().end()) {
    return *it;
  }
  return std::nullopt;
}

// If options specify it, adds and returns an input for a reset signal.
static absl::Status MaybeAddResetPort(Block* block,
                                      const CodegenOptions& options) {
  // TODO(tedhong): 2021-09-18 Combine this with AddValidSignal
  if (options.reset().has_value()) {
    XLS_RET_CHECK_OK(block->AddResetPort(options.reset()->name()));
  }

  // Connect reset to FIFO instantiations.
  for (xls::Instantiation* instantiation : block->GetInstantiations()) {
    if (instantiation->kind() != InstantiationKind::kFifo) {
      continue;
    }
    XLS_RET_CHECK(options.reset().has_value())
        << "Fifo instantiations require reset.";
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<xls::InstantiationInput>(
                                SourceInfo(), block->GetResetPort().value(),
                                instantiation,
                                xls::FifoInstantiation::kResetPortName)
                            .status());
  }

  return absl::OkStatus();
}

// Plumb valid signal through the pipeline stages, ANDing with a valid produced
// by each stage. Gather the pipelined valid signal in a vector where the
// zero-th element is the input port and subsequent elements are the pipelined
// valid signal from each stage.
static absl::StatusOr<std::vector<std::optional<Node*>>>
MakePipelineStagesForValid(
    Node* valid_input_port,
    absl::Span<const PipelineStageRegisters> pipeline_registers,
    const std::optional<xls::Reset>& reset_behavior, Block* block) {
  Type* u1 = block->package()->GetBitsType(1);

  std::vector<std::optional<Node*>> pipelined_valids(pipeline_registers.size() +
                                                     1);
  pipelined_valids[0] = valid_input_port;

  for (int64_t stage = 0; stage < pipeline_registers.size(); ++stage) {
    // Add valid register to each pipeline stage.
    XLS_ASSIGN_OR_RETURN(Register * valid_reg,
                         block->AddRegister(PipelineSignalName("valid", stage),
                                            u1, reset_behavior));
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<RegisterWrite>(
                                /*loc=*/SourceInfo(), *pipelined_valids[stage],
                                /*load_enable=*/std::nullopt,
                                /*reset=*/block->GetResetPort(), valid_reg)
                            .status());
    XLS_ASSIGN_OR_RETURN(pipelined_valids[stage + 1],
                         block->MakeNode<RegisterRead>(
                             /*loc=*/SourceInfo(), valid_reg));
  }

  return pipelined_valids;
}

// Plumb valid signal through the pipeline stages, ANDing with a valid produced
// by each stage. Gather the pipelined valid signal in a vector where the
// zero-th element is the input port and subsequent elements are the pipelined
// valid signal from each stage.
static absl::Status MakePipelineStagesForValidIO(
    StreamingIOPipeline& streaming_io, absl::Span<Node* const> recvs_valid,
    absl::Span<Node* const> states_valid, absl::Span<Node* const> sends_ready,
    const std::optional<xls::Reset>& reset_behavior, Block* block) {
  std::vector<PipelineStageRegisters>& pipeline_registers =
      streaming_io.pipeline_registers;
  std::vector<std::optional<Node*>>& pipeline_valid =
      streaming_io.pipeline_valid;
  std::vector<std::optional<Node*>>& stage_valid = streaming_io.stage_valid;
  std::vector<std::optional<Node*>>& stage_done = streaming_io.stage_done;

  Type* u1 = block->package()->GetBitsType(1);

  // Node denoting if the specific stage's input data from the previous stage is
  // valid.
  pipeline_valid.resize(pipeline_registers.size() + 1);

  // Node denoting if all of the specific stage's input data is valid.
  stage_valid.resize(pipeline_registers.size() + 1);

  // Node denoting if the specific state is done with its computation.
  // A stage N is done if
  //   a. It's valid (stage[N] == true).
  //   b. All receives are valid (recvs_valid[N] == true).
  //   c. All sends are ready (sends_ready[N] == true).
  //   d. All states are ready (states_ready[N] == true).
  stage_done.resize(pipeline_registers.size() + 1);

  // The 0'th stage, having no previous stage, is valid unless its state is not
  // valid.
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));
  pipeline_valid[0] = literal_1;
  stage_valid[0] = states_valid[0];

  int64_t stage_count = pipeline_registers.size() + 1;
  for (int64_t stage = 0; stage < stage_count; ++stage) {
    XLS_ASSIGN_OR_RETURN(
        stage_done[stage],
        block->MakeNodeWithName<xls::NaryOp>(
            /*loc=*/SourceInfo(),
            std::vector<xls::Node*>{
                *stage_valid[stage],
                recvs_valid[stage],
                sends_ready[stage],
            },
            Op::kAnd, PipelineSignalName("stage_done", stage)));

    if (stage < stage_count - 1) {
      // Only add a valid register if it will be read from and written to.
      XLS_ASSIGN_OR_RETURN(
          Register * valid_reg,
          block->AddRegister(PipelineSignalName("valid", stage), u1,
                             reset_behavior));
      XLS_RETURN_IF_ERROR(block
                              ->MakeNode<RegisterWrite>(
                                  /*loc=*/SourceInfo(), *stage_done[stage],
                                  /*load_enable=*/std::nullopt,
                                  /*reset=*/block->GetResetPort(), valid_reg)
                              .status());
      XLS_ASSIGN_OR_RETURN(pipeline_valid[stage + 1],
                           block->MakeNode<RegisterRead>(
                               /*loc=*/SourceInfo(), valid_reg));
      XLS_ASSIGN_OR_RETURN(
          stage_valid[stage + 1],
          block->MakeNodeWithName<NaryOp>(
              SourceInfo(),
              std::vector<Node*>{states_valid[stage + 1],
                                 *pipeline_valid[stage + 1]},
              Op::kAnd, PipelineSignalName("stage_valid", stage + 1)));
    }
  }

  return absl::OkStatus();
}

// Returns or makes a node that is 1 when the block is under reset,
// if said reset signal exists.
//
//   - If no reset exists, std::nullopt is returned
//   - Active low reset signals are inverted.
//
// See also MakeOrWithResetNode()
static absl::StatusOr<std::optional<Node*>> MaybeGetOrMakeResetNode(
    const std::optional<xls::Reset>& reset_behavior, Block* block) {
  if (!block->GetResetPort().has_value()) {
    return std::nullopt;
  }

  Node* reset_node = block->GetResetPort().value();
  if (reset_behavior->active_low) {
    return block->MakeNode<UnOp>(/*loc=*/SourceInfo(), reset_node, Op::kNot);
  }

  return reset_node;
}

// Updates the state_register with a reset signal.
//  1. The state register is reset active_high or active_low
//     following the block behavior.
//  2. The state register is reset to the initial value of the proc.
//  3. The state register is reset whenever the block reset is active.
static absl::Status UpdateStateRegisterWithReset(
    const std::optional<xls::Reset>& reset_behavior,
    StateRegister& state_register, Block* block) {
  if (state_register.reg == nullptr && state_register.reg_full == nullptr) {
    // No register to update; move on.
    return absl::OkStatus();
  }

  if (state_register.reg != nullptr) {
    CHECK_NE(state_register.reg_write, nullptr)
        << "reg_write is null for " << state_register.name;
    CHECK_NE(state_register.reg_read, nullptr)
        << "reg_read is null for " << state_register.name;
  }
  if (state_register.reg_full) {
    CHECK_NE(state_register.reg_full_write, nullptr);
  }

  // Blocks containing a state register must also have a reset signal.
  if (!block->GetResetPort().has_value() || !reset_behavior.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unable to update state register %s with reset, signal as block"
        " was not created with a reset.",
        state_register.name));
  }

  if (state_register.reg) {
    // Follow the reset behavior of the valid registers except for the initial
    // value.
    xls::Reset reset_behavior_with_copied_init = *reset_behavior;
    reset_behavior_with_copied_init.reset_value = state_register.reset_value;
    // Replace the register's reset signal
    XLS_RETURN_IF_ERROR(state_register.reg_write->AddOrReplaceReset(
        block->GetResetPort().value(), reset_behavior_with_copied_init));
  }

  if (state_register.reg_full) {
    // `reg_full` should be initialized to 1; at startup, the state register has
    // an initial value and is therefore valid.
    xls::Reset reset_behavior_for_full = *reset_behavior;
    reset_behavior_for_full.reset_value = Value(UBits(1, 1));
    XLS_RETURN_IF_ERROR(state_register.reg_full_write->AddOrReplaceReset(
        block->GetResetPort().value(), reset_behavior_for_full));
  }

  return absl::OkStatus();
}

// Updates datapath pipeline registers with a reset signal.
//  1. The pipeline registers are reset active_high or active_low
//     following the block behavior.
//  2. The registers are reset to zero.
//  3. The registers are reset whenever the block reset is active.
static absl::Status UpdateDatapathRegistersWithReset(
    const std::optional<xls::Reset>& reset_behavior,
    absl::Span<PipelineStageRegisters> pipeline_data_registers, Block* block) {
  // Blocks should have reset information.
  if (!block->GetResetPort().has_value() || !reset_behavior.has_value()) {
    return absl::InvalidArgumentError(
        "Unable to update pipeline registers with reset, signal as block"
        " was not created with a reset.");
  }

  // Update each datapath register with a reset.
  Node* reset_node = block->GetResetPort().value();
  int64_t stage_count = pipeline_data_registers.size();

  for (int64_t stage = 0; stage < stage_count; ++stage) {
    for (PipelineRegister& pipeline_reg : pipeline_data_registers.at(stage)) {
      CHECK_NE(pipeline_reg.reg, nullptr);
      CHECK_NE(pipeline_reg.reg_write, nullptr);
      CHECK_NE(pipeline_reg.reg_read, nullptr);

      // Don't attempt to reset registers that will be removed later
      // (ex. tokens).
      Type* node_type = pipeline_reg.reg->type();
      if (node_type->GetFlatBitCount() == 0) {
        continue;
      }

      // Reset the register to zero of the correct type.
      xls::Reset reset_behavior_for_type = reset_behavior.value();
      reset_behavior_for_type.reset_value = ZeroOfType(node_type);

      // Replace register reset.
      XLS_RETURN_IF_ERROR(pipeline_reg.reg_write->AddOrReplaceReset(
          reset_node, reset_behavior_for_type));
    }
  }

  return absl::OkStatus();
}

// Given a node returns a node that is OR'd with the reset signal.
// if said reset signal exists.  That node can be thought of as
//     1 - If being reset or if the src_node is 1
//     0 - otherwise.
//
//   - If no reset exists, the node is returned and the graph unchanged.
//   - Active low reset signals are inverted so that the resulting signal
//      OR(src_node, NOT(reset))
//
// This is used to drive load_enable signals of pipeline valid registers.
static absl::StatusOr<Node*> MakeOrWithResetNode(
    Node* src_node, std::string_view result_name,
    const std::optional<xls::Reset>& reset_behavior, Block* block) {
  Node* result = src_node;

  XLS_ASSIGN_OR_RETURN(std::optional<Node*> maybe_reset_node,
                       MaybeGetOrMakeResetNode(reset_behavior, block));

  if (maybe_reset_node.has_value()) {
    Node* reset_node = maybe_reset_node.value();
    XLS_ASSIGN_OR_RETURN(result, block->MakeNodeWithName<NaryOp>(
                                     /*loc=*/SourceInfo(),
                                     std::vector<Node*>({result, reset_node}),
                                     Op::kOr, result_name));
  }

  return result;
}

struct BubbleFlowControl {
  std::vector<Node*> data_load_enable;
  std::vector<Node*> next_stage_open;
};

// Adds bubble flow control to the pipeline.
//
// - With bubble flow control, a pipeline stage is not stalled if
//   the next stage is either invalid or is not stalled.
// - This enabled bubbles within the pipeline to be collapsed when the
//   output block of the pipeline is not ready to accept data.
//
// Returns the ready signal output by the earliest pipeline stage.
//
static absl::StatusOr<BubbleFlowControl> UpdatePipelineWithBubbleFlowControl(
    const CodegenOptions& options,
    absl::Span<std::optional<Node*> const> pipeline_valid_nodes,
    absl::Span<std::optional<Node*> const> pipeline_done_nodes,
    absl::Span<PipelineStageRegisters> pipeline_data_registers,
    absl::Span<std::optional<StateRegister>> state_registers,
    absl::flat_hash_map<Node*, Stage>& node_to_stage_map, Block* block) {
  // Create enable signals for each pipeline stage.
  //   - The enable signal for stage N is true either
  //       a. The next stage is empty/not valid
  //         or
  //       b. The next stage will latch data and leave the stage empty
  //     enable_signal[n] = data_enable[n+1] || ! pipeline_valid[n+1]
  //   - The data enable signal for stage N is true if both
  //       a. The enable signal is true (the next stage can accept data)
  //         and
  //       b. This stage is finished with is computation
  //     data_enable[n] = enable_signal[n] && stage_done[n]
  //
  //     Note that if data registers are not reset, data path registers
  //     are transparent during reset.  In this case:
  //       data_enable[n] = (enable_signal[n] && stage_done[n]) || rst
  //
  // As the last stage has no next stage, that is equivalent to
  //   pipeline_valid[N+1] = 0 so
  //   enable_signal[N] = 1
  //   data_enable[N] = stage_done[N]
  //
  // Data registers are gated whenever data is invalid so
  //   - data_enable_signal[n-1] = (enable_signal[n-1] && valid[n-1]) || rst
  //
  // State registers are gated whenever data is invalid, but
  // are not transparent during reset
  //   - state_enable_signal = (enable_signal[0] && valid[0]).
  //
  // enable_n is the same as next_stage_open

  int64_t stage_count = pipeline_data_registers.size() + 1;

  BubbleFlowControl result;
  result.data_load_enable = std::vector<Node*>(stage_count);
  result.next_stage_open = std::vector<Node*>(stage_count);

  std::vector<Node*>& enable_n = result.next_stage_open;

  XLS_ASSIGN_OR_RETURN(
      enable_n.at(stage_count - 1),
      block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
  result.data_load_enable.at(stage_count - 1) =
      *pipeline_done_nodes.at(stage_count - 1);

  std::vector<Node*> state_enables;
  state_enables.resize(stage_count, nullptr);
  state_enables.at(stage_count - 1) = *pipeline_done_nodes.at(stage_count - 1);

  for (int64_t stage = stage_count - 2; stage >= 0; --stage) {
    // Create load enables for valid registers.
    XLS_ASSIGN_OR_RETURN(
        Node * not_valid_np1,
        block->MakeNodeWithName<UnOp>(
            /*loc=*/SourceInfo(), *pipeline_valid_nodes.at(stage + 1), Op::kNot,
            PipelineSignalName("not_valid", stage + 1)));

    XLS_ASSIGN_OR_RETURN(
        Node * enable,
        block->MakeNodeWithName<NaryOp>(
            SourceInfo(),
            std::vector<Node*>{result.data_load_enable.at(stage + 1),
                               not_valid_np1},
            Op::kOr, PipelineSignalName("enable", stage)));
    enable_n.at(stage) = enable;

    // Update valid registers with load enables.
    RegisterRead* valid_reg_read =
        pipeline_valid_nodes.at(stage + 1).value()->As<RegisterRead>();
    XLS_RET_CHECK(valid_reg_read != nullptr);
    Register* valid_reg = valid_reg_read->GetRegister();
    XLS_ASSIGN_OR_RETURN(RegisterWrite * valid_reg_write,
                         block->GetRegisterWrite(valid_reg));
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<RegisterWrite>(
                                /*loc=*/SourceInfo(), valid_reg_write->data(),
                                /*load_enable=*/enable,
                                /*reset=*/valid_reg_write->reset(), valid_reg)
                            .status());
    XLS_RETURN_IF_ERROR(block->RemoveNode(valid_reg_write));

    // Create load enables for datapath registers.
    std::vector<Node*> data_en_operands = {enable,
                                           *pipeline_done_nodes.at(stage)};
    XLS_ASSIGN_OR_RETURN(Node * data_enable,
                         block->MakeNodeWithName<NaryOp>(
                             SourceInfo(), data_en_operands, Op::kAnd,
                             PipelineSignalName("data_enable", stage)));

    state_enables.at(stage) = data_enable;

    // If datapath registers are reset, then adding reset to the
    // load enable is redundant.
    if (options.reset().has_value() && options.reset()->reset_data_path()) {
      result.data_load_enable.at(stage) = data_enable;
    } else {
      XLS_ASSIGN_OR_RETURN(
          result.data_load_enable.at(stage),
          MakeOrWithResetNode(data_enable, PipelineSignalName("load_en", stage),
                              options.ResetBehavior(), block));
    }

    // Update datapath registers with load enables.
    if (!pipeline_data_registers.at(stage).empty()) {
      for (PipelineRegister& pipeline_reg : pipeline_data_registers.at(stage)) {
        XLS_ASSIGN_OR_RETURN(
            RegisterWrite * new_reg_write,
            block->MakeNode<RegisterWrite>(
                /*loc=*/SourceInfo(), pipeline_reg.reg_write->data(),
                /*load_enable=*/result.data_load_enable.at(stage),
                /*reset=*/pipeline_reg.reg_write->reset(), pipeline_reg.reg));
        XLS_RET_CHECK(node_to_stage_map.contains(pipeline_reg.reg_write));
        Stage s = node_to_stage_map[pipeline_reg.reg_write];
        node_to_stage_map.erase(pipeline_reg.reg_write);
        XLS_RETURN_IF_ERROR(block->RemoveNode(pipeline_reg.reg_write));
        pipeline_reg.reg_write = new_reg_write;
        node_to_stage_map[new_reg_write] = s;
      }
    }
  }

  // Generate writes for state registers. This is done in a separate loop
  // because the last stage isn't included in the pipeline register loop.
  for (std::optional<StateRegister>& state_register : state_registers) {
    if (!state_register.has_value()) {
      continue;
    }
    CHECK(!state_register->next_values.empty());

    if (state_register->reg == nullptr && state_register->reg_full == nullptr) {
      // No actual contents for this state element, and no need to track it for
      // flow control; skip it.
      continue;
    }

    SourceInfo write_loc = state_register->reg_write != nullptr
                               ? state_register->reg_write->loc()
                               : state_register->reg_full_write->loc();

    std::vector<Node*> values;
    std::vector<Node*> write_conditions;
    std::vector<Node*> unchanged_conditions;
    values.reserve(state_register->next_values.size());
    write_conditions.reserve(state_register->next_values.size());
    unchanged_conditions.reserve(state_register->next_values.size());
    for (const StateRegister::NextValue& next_value :
         state_register->next_values) {
      Node* activated_predicate;
      if (next_value.predicate.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            activated_predicate,
            block->MakeNode<NaryOp>(
                write_loc,
                std::vector<Node*>{*next_value.predicate,
                                   state_enables.at(next_value.stage)},
                Op::kAnd));
      } else {
        activated_predicate = state_enables.at(next_value.stage);
      }

      if (next_value.value.has_value()) {
        values.push_back(*next_value.value);
        write_conditions.push_back(activated_predicate);
      } else {
        unchanged_conditions.push_back(activated_predicate);
      }
    }

    Node* value;
    std::optional<Node*> load_enable;
    if (values.empty()) {
      // If we never change the state element, write the current value
      // unconditionally.
      value = state_register->reg_read;
      load_enable = std::nullopt;
    } else if (values.size() == 1) {
      value = values[0];
      load_enable = write_conditions[0];
    } else {
      XLS_ASSIGN_OR_RETURN(Node * selector, block->MakeNode<xls::Concat>(
                                                write_loc, write_conditions));

      // Reverse the order of the values, so they match up to the selector.
      std::reverse(values.begin(), values.end());
      XLS_ASSIGN_OR_RETURN(
          value, block->MakeNode<OneHotSelect>(write_loc, selector, values));

      XLS_ASSIGN_OR_RETURN(
          load_enable,
          block->MakeNode<NaryOp>(write_loc, write_conditions, Op::kOr));
    }

    if (state_register->reg) {
      XLS_ASSIGN_OR_RETURN(
          RegisterWrite * new_reg_write,
          block->MakeNode<RegisterWrite>(
              /*loc=*/write_loc,
              /*data=*/value,
              /*load_enable=*/load_enable,
              /*reset=*/state_register->reg_write->reset(),
              /*reg=*/state_register->reg_write->GetRegister()));
      XLS_RET_CHECK(node_to_stage_map.contains(state_register->reg_write));
      node_to_stage_map.erase(state_register->reg_write);
      XLS_RETURN_IF_ERROR(block->RemoveNode(state_register->reg_write));
      state_register->reg_write = new_reg_write;
    }

    if (state_register->reg_full) {
      // The state register's fullness changes whenever we read or write the
      // state parameter's value - and it changes to 1 if we're writing, or 0
      // if we're only reading. In other words, the fullness register should
      // be load-enabled if we're reading or writing, and its value should be
      // set to 1 if we're writing.

      // However, it also updates to 1 if we *would have* written an unchanged
      // value (even though that lets us skip actually writing to the state
      // register), so we also need to account for those.
      Node* value_determined;
      if (unchanged_conditions.empty()) {
        // There should be an active write, since there's nothing to support
        // leaving the register unchanged.
        CHECK(load_enable.has_value());
        value_determined = *load_enable;
      } else {
        std::vector<Node*> determined_conditions = unchanged_conditions;
        if (load_enable.has_value()) {
          determined_conditions.push_back(*load_enable);
        }
        XLS_ASSIGN_OR_RETURN(
            value_determined,
            block->MakeNode<NaryOp>(write_loc, determined_conditions, Op::kOr));
      }

      XLS_ASSIGN_OR_RETURN(
          Node * reg_full_load_enable,
          block->MakeNode<NaryOp>(
              write_loc,
              std::vector<Node*>{state_enables.at(state_register->read_stage),
                                 value_determined},
              Op::kOr));
      XLS_ASSIGN_OR_RETURN(
          RegisterWrite * new_reg_full_write,
          block->MakeNode<RegisterWrite>(
              /*loc=*/write_loc,
              /*data=*/value_determined,
              /*load_enable=*/reg_full_load_enable,
              /*reset=*/state_register->reg_full_write->reset(),
              /*reg=*/state_register->reg_full_write->GetRegister()));
      XLS_RETURN_IF_ERROR(block->RemoveNode(state_register->reg_full_write));
      state_register->reg_full_write = new_reg_full_write;
    }
  }

  return result;
}

// Adds flow control when no pipeline registers are created
// (a pipeline of 1 stage).
//
// Returns a node that is true when the stage accepts a new set of
// of inputs.
//
// The stage is active if
//   a. All active inputs are valid
//   b. All active outputs are ready
//
// See UpdatePipelineWithBubbleFlowControl() for a description,
// and for supporting sharing the enable signal of the state register
// with the datapath pipeline registers.
//
// if there is a state register,
//   the state register is updated with an updated RegisterWrite node.
//
static absl::StatusOr<Node*> UpdateSingleStagePipelineWithFlowControl(
    Node* all_active_outputs_ready, Node* all_active_inputs_valid,
    absl::Span<std::optional<StateRegister>> state_registers,
    absl::flat_hash_map<Node*, Stage>& node_to_stage_map, Block* block) {
  std::vector<Node*> operands = {all_active_outputs_ready,
                                 all_active_inputs_valid};
  XLS_ASSIGN_OR_RETURN(
      Node * pipeline_enable,
      block->MakeNodeWithName<NaryOp>(SourceInfo(), operands, Op::kAnd,
                                      "pipeline_enable"));

  for (std::optional<StateRegister>& state_register : state_registers) {
    if (state_register.has_value() && state_register->reg) {
      std::vector<Node*> write_predicates;
      for (const StateRegister::NextValue& next_value :
           state_register->next_values) {
        if (next_value.value.has_value() && next_value.predicate.has_value()) {
          write_predicates.push_back(*next_value.predicate);
        }
      }

      Node* load_enable = pipeline_enable;
      if (!write_predicates.empty()) {
        XLS_ASSIGN_OR_RETURN(Node * active_write,
                             NaryOrIfNeeded(block, write_predicates));
        XLS_ASSIGN_OR_RETURN(
            load_enable,
            block->MakeNode<NaryOp>(
                state_register->reg_write->loc(),
                std::vector<Node*>{active_write, pipeline_enable}, Op::kAnd));
      }

      XLS_ASSIGN_OR_RETURN(
          RegisterWrite * new_reg_write,
          block->MakeNode<RegisterWrite>(
              /*loc=*/state_register->reg_write->loc(),
              /*data=*/state_register->reg_write->data(),
              /*load_enable=*/load_enable,
              /*reset=*/state_register->reg_write->reset(),
              /*reg=*/state_register->reg_write->GetRegister()));
      XLS_RETURN_IF_ERROR(block->RemoveNode(state_register->reg_write));
      CHECK(!node_to_stage_map.contains(state_register->reg_write));
      state_register->reg_write = new_reg_write;
    }
  }

  return pipeline_enable;
}

static absl::StatusOr<ValidPorts> AddValidSignal(
    absl::Span<PipelineStageRegisters> pipeline_registers,
    const CodegenOptions& options, Block* block,
    std::vector<std::optional<Node*>>& pipelined_valids,
    absl::flat_hash_map<Node*, Stage>& node_to_stage_map) {
  // Add valid input port.
  XLS_RET_CHECK(options.valid_control().has_value());
  if (options.valid_control()->input_name().empty()) {
    return absl::InvalidArgumentError(
        "Must specify input name of valid signal.");
  }
  Type* u1 = block->package()->GetBitsType(1);
  XLS_ASSIGN_OR_RETURN(
      InputPort * valid_input_port,
      block->AddInputPort(options.valid_control()->input_name(), u1));

  std::optional<xls::Reset> reset_behavior = options.ResetBehavior();

  // Plumb valid signal through the pipeline stages. Gather the pipelined valid
  // signal in a vector where the zero-th element is the input port and
  // subsequent elements are the pipelined valid signal from each stage.
  XLS_ASSIGN_OR_RETURN(
      pipelined_valids,
      MakePipelineStagesForValid(valid_input_port, pipeline_registers,
                                 reset_behavior, block));

  // Use the pipelined valid signal as load enable for each datapath pipeline
  // register in each stage as a power optimization.
  for (int64_t stage = 0; stage < pipeline_registers.size(); ++stage) {
    // For each (non-valid-signal) pipeline register add `valid` or `valid ||
    // reset` (if reset exists) as a load enable. The `reset` term ensures the
    // pipeline flushes when reset is enabled.
    if (!pipeline_registers.at(stage).empty()) {
      XLS_ASSIGN_OR_RETURN(
          Node * load_enable,
          MakeOrWithResetNode(*pipelined_valids[stage],
                              PipelineSignalName("load_en", stage),
                              reset_behavior, block));

      for (PipelineRegister& pipeline_reg : pipeline_registers.at(stage)) {
        XLS_ASSIGN_OR_RETURN(
            auto* new_write,
            block->MakeNode<RegisterWrite>(
                /*loc=*/SourceInfo(), pipeline_reg.reg_write->data(),
                /*load_enable=*/load_enable,
                /*reset=*/std::nullopt, pipeline_reg.reg));
        if (node_to_stage_map.contains(pipeline_reg.reg_write)) {
          Stage s = node_to_stage_map[pipeline_reg.reg_write];
          node_to_stage_map.erase(pipeline_reg.reg_write);
          node_to_stage_map[new_write] = s;
        }
        XLS_RETURN_IF_ERROR(block->RemoveNode(pipeline_reg.reg_write));
        pipeline_reg.reg_write = new_write;
      }
    }
  }

  // Add valid output port.
  OutputPort* valid_output_port = nullptr;
  if (options.valid_control().has_value() &&
      options.valid_control()->has_output_name()) {
    if (options.valid_control()->output_name().empty()) {
      return absl::InvalidArgumentError(
          "Must specify output name of valid signal.");
    }
    XLS_ASSIGN_OR_RETURN(
        valid_output_port,
        block->AddOutputPort(options.valid_control()->output_name(),
                             pipelined_valids.back().value()));
  }

  return ValidPorts{valid_input_port, valid_output_port};
}

absl::StatusOr<std::string> StreamingIOName(Node* node) {
  switch (node->op()) {
    case Op::kInputPort:
      return std::string{node->As<InputPort>()->name()};
    case Op::kOutputPort:
      return std::string{node->As<OutputPort>()->name()};
    case Op::kInstantiationInput:
      return absl::StrCat(
          node->As<InstantiationInput>()->instantiation()->name(), "_",
          node->As<InstantiationInput>()->port_name());
    case Op::kInstantiationOutput:
      return absl::StrCat(
          node->As<InstantiationOutput>()->instantiation()->name(), "_",
          node->As<InstantiationOutput>()->port_name());
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported streaming operation %s.", OpToString(node->op())));
  }
}

// Update io channel metadata with latest information from block conversion.
absl::Status UpdateChannelMetadata(const StreamingIOPipeline& io,
                                   Block* block) {
  for (const auto& inputs : io.inputs) {
    for (const StreamingInput& input : inputs) {
      CHECK_NE(*input.port, nullptr);
      CHECK_NE(input.port_valid, nullptr);
      CHECK_NE(input.port_ready, nullptr);
      CHECK_NE(input.channel, nullptr);
      // Ports are either all external IOs or all from instantiations.
      CHECK(input.IsExternal() || input.IsInstantiation());

      input.channel->SetBlockName(block->name());
      XLS_ASSIGN_OR_RETURN(std::string name, StreamingIOName(*input.port));
      input.channel->SetDataPortName(name);
      XLS_ASSIGN_OR_RETURN(name, StreamingIOName(input.port_valid));
      input.channel->SetValidPortName(name);
      XLS_ASSIGN_OR_RETURN(name, StreamingIOName(input.port_ready));
      input.channel->SetReadyPortName(name);

      CHECK(input.channel->HasCompletedBlockPortNames());
    }
  }

  for (const auto& outputs : io.outputs) {
    for (const StreamingOutput& output : outputs) {
      CHECK_NE(*output.port, nullptr);
      CHECK_NE(output.port_valid, nullptr);
      CHECK_NE(output.port_ready, nullptr);
      CHECK_NE(output.channel, nullptr);
      // Ports are either all external IOs or all from instantiations.
      CHECK(output.IsExternal() || output.IsInstantiation());

      output.channel->SetBlockName(block->name());
      XLS_ASSIGN_OR_RETURN(std::string name, StreamingIOName(*output.port));
      output.channel->SetDataPortName(name);
      XLS_ASSIGN_OR_RETURN(name, StreamingIOName(output.port_valid));
      output.channel->SetValidPortName(name);
      XLS_ASSIGN_OR_RETURN(name, StreamingIOName(output.port_ready));
      output.channel->SetReadyPortName(name);

      CHECK(output.channel->HasCompletedBlockPortNames());
    }
  }

  for (const SingleValueInput& input : io.single_value_inputs) {
    CHECK_NE(input.port, nullptr);
    CHECK_NE(input.channel, nullptr);

    input.channel->SetBlockName(block->name());
    input.channel->SetDataPortName(input.port->name());

    CHECK(input.channel->HasCompletedBlockPortNames());
  }

  for (const SingleValueOutput& output : io.single_value_outputs) {
    CHECK_NE(output.port, nullptr);
    CHECK_NE(output.channel, nullptr);

    output.channel->SetBlockName(block->name());
    output.channel->SetDataPortName(output.port->name());

    CHECK(output.channel->HasCompletedBlockPortNames());
  }

  return absl::OkStatus();
}

// For each output streaming channel add a corresponding ready port (input
// port). Combinationally combine those ready signals with their predicates to
// generate an  all_active_outputs_ready signal.
//
// Upon success returns a Node* to the all_active_inputs_valid signal.
static absl::StatusOr<std::vector<Node*>> MakeInputReadyPortsForOutputChannels(
    std::vector<std::vector<StreamingOutput>>& streaming_outputs,
    int64_t stage_count, std::string_view ready_suffix, Block* block) {
  std::vector<Node*> result;

  // Add a ready input port for each streaming output. Gather the ready signals
  // into a vector. Ready signals from streaming outputs generated from Send
  // operations are conditioned upon the optional predicate value.
  for (Stage stage = 0; stage < stage_count; ++stage) {
    std::vector<Node*> active_readys;
    for (StreamingOutput& streaming_output : streaming_outputs[stage]) {
      if (streaming_output.fifo_instantiation.has_value()) {
        // The ready signal is managed elsewhere for FIFO instantiations.
        XLS_RET_CHECK_NE(streaming_output.port_ready, nullptr);
      } else {
        XLS_ASSIGN_OR_RETURN(
            streaming_output.port_ready,
            block->AddInputPort(
                absl::StrCat(streaming_output.channel->name(), ready_suffix),
                block->package()->GetBitsType(1)));
      }

      if (streaming_output.predicate.has_value()) {
        // Logic for the active ready signal for a Send operation with a
        // predicate `pred`.
        //
        //   active = !pred | pred && ready
        //          = !pred | ready
        XLS_ASSIGN_OR_RETURN(
            Node * not_pred,
            block->MakeNode<UnOp>(
                SourceInfo(), streaming_output.predicate.value(), Op::kNot));
        // If predicate has an assigned name, let the not expression get
        // inlined. Otherwise, give a descriptive name.
        if (!streaming_output.predicate.value()->HasAssignedName()) {
          not_pred->SetName(
              absl::StrFormat("%s_not_pred", streaming_output.channel->name()));
        }
        std::vector<Node*> operands{not_pred, streaming_output.port_ready};
        XLS_ASSIGN_OR_RETURN(
            Node * active_ready,
            block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kOr));
        // not_pred will have an assigned name or be inlined, so only check
        // the ready port. If it has an assigned name, just let everything
        // inline. Otherwise, give a descriptive name.
        if (!streaming_output.port_ready->HasAssignedName()) {
          active_ready->SetName(absl::StrFormat(
              "%s_active_ready", streaming_output.channel->name()));
        }
        active_readys.push_back(active_ready);
      } else {
        active_readys.push_back(streaming_output.port_ready);
      }
    }

    // And reduce all the active ready signals. This signal is true iff all
    // active outputs are ready.
    XLS_ASSIGN_OR_RETURN(
        Node * all_active_outputs_ready,
        NaryAndIfNeeded(block, active_readys,
                        PipelineSignalName("all_active_outputs_ready", stage)));
    result.push_back(all_active_outputs_ready);
  }

  return result;
}

// Create a signal that is true if stage 0 has no active inputs.
//
// In other words, there is no receive op whose predicate is true.
static absl::StatusOr<Node*> Stage0HasNoActiveRecvs(
    std::vector<std::vector<StreamingInput>>& streaming_inputs, Block* block) {
  CHECK(!streaming_inputs.empty());

  if (streaming_inputs[0].empty()) {
    // Note that a proc with no receives at all would have this signal be a
    // literal 1.
    return block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1)));
  }

  std::vector<Node*> recv_if_preds;
  for (StreamingInput& streaming_input : streaming_inputs[0]) {
    if (streaming_input.predicate.has_value()) {
      recv_if_preds.push_back(streaming_input.predicate.value());
    } else {
      // There is an unconditional receive node, return literal 0
      return block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(0, 1)));
    }
  }

  // Stage 0 has only conditional receives, return NOR of their predicates.
  return NaryNorIfNeeded(block, recv_if_preds);
}

// For each input streaming channel add a corresponding valid port (input port).
// Combinationally combine those valid signals with their predicates
// to generate an all_active_inputs_valid signal.
//
// Upon success returns a Node* to the all_active_inputs_valid signal.
static absl::StatusOr<std::vector<Node*>> MakeInputValidPortsForInputChannels(
    std::vector<std::vector<StreamingInput>>& streaming_inputs,
    int64_t stage_count, std::string_view valid_suffix, Block* block) {
  std::vector<Node*> result;

  for (Stage stage = 0; stage < stage_count; ++stage) {
    // Add a valid input port for each streaming input. Gather the valid
    // signals into a vector. Valid signals from streaming inputs generated
    // from Receive operations are conditioned upon the optional predicate
    // value.
    std::vector<Node*> active_valids;
    for (StreamingInput& streaming_input : streaming_inputs[stage]) {
      // Input ports for input channels are already created during
      // HandleReceiveNode().
      XLS_RET_CHECK(streaming_input.signal_valid.has_value());
      Node* streaming_input_valid = *streaming_input.signal_valid;

      if (streaming_input.predicate.has_value()) {
        // Logic for the active valid signal for a Receive operation with a
        // predicate `pred`.
        //
        //   active = !pred | pred && valid
        //          = !pred | valid
        XLS_ASSIGN_OR_RETURN(
            Node * not_pred,
            block->MakeNode<UnOp>(SourceInfo(),
                                  streaming_input.predicate.value(), Op::kNot));

        // If predicate has an assigned name, let the not expression get
        // inlined. Otherwise, give a descriptive name.
        if (!streaming_input.predicate.value()->HasAssignedName()) {
          not_pred->SetName(
              absl::StrFormat("%s_not_pred", streaming_input.channel->name()));
        }
        std::vector<Node*> operands = {not_pred, streaming_input_valid};
        XLS_ASSIGN_OR_RETURN(
            Node * active_valid,
            block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kOr));
        // not_pred will have an assigned name or be inlined, so only check
        // the ready port. If it has an assigned name, just let everything
        // inline. Otherwise, give a descriptive name.
        if (!streaming_input_valid->HasAssignedName()) {
          active_valid->SetName(absl::StrFormat(
              "%s_active_valid", streaming_input.channel->name()));
        }
        active_valids.push_back(active_valid);
      } else {
        // No predicate is the same as pred = true, so
        // active = !pred | valid = !true | valid = false | valid = valid
        active_valids.push_back(streaming_input_valid);
      }
    }

    // And reduce all the active valid signals. This signal is true iff all
    // active inputs are valid.
    XLS_ASSIGN_OR_RETURN(
        Node * all_active_inputs_valid,
        NaryAndIfNeeded(block, active_valids,
                        PipelineSignalName("all_active_inputs_valid", stage)));
    result.push_back(all_active_inputs_valid);
  }

  return result;
}

// Make valid ports (output) for the output channel.
//
// A valid signal is asserted iff all active
// inputs valid signals are asserted and the predicate of the data channel (if
// any) is asserted.
static absl::Status MakeOutputValidPortsForOutputChannels(
    absl::Span<Node* const> all_active_inputs_valid,
    absl::Span<Node* const> pipelined_valids,
    absl::Span<Node* const> next_stage_open,
    std::vector<std::vector<StreamingOutput>>& streaming_outputs,
    std::string_view valid_suffix, Block* block) {
  for (Stage stage = 0; stage < streaming_outputs.size(); ++stage) {
    for (StreamingOutput& streaming_output : streaming_outputs.at(stage)) {
      std::vector<Node*> operands{all_active_inputs_valid.at(stage),
                                  pipelined_valids.at(stage),
                                  next_stage_open.at(stage)};

      if (streaming_output.predicate.has_value()) {
        operands.push_back(streaming_output.predicate.value());
      }

      XLS_ASSIGN_OR_RETURN(Node * valid, block->MakeNode<NaryOp>(
                                             SourceInfo(), operands, Op::kAnd));
      if (streaming_output.fifo_instantiation.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            streaming_output.port_valid,
            block->MakeNode<xls::InstantiationInput>(
                streaming_output.port.value()->loc(), valid,
                streaming_output.fifo_instantiation.value(), "push_valid"));
      } else {
        XLS_ASSIGN_OR_RETURN(
            streaming_output.port_valid,
            block->AddOutputPort(
                absl::StrCat(streaming_output.channel->name(), valid_suffix),
                valid));
      }
    }
  }

  return absl::OkStatus();
}

// Make ready ports (output) for each input channel.
//
// A ready signal is asserted iff all active
// output ready signals are asserted and the predicate of the data channel (if
// any) is asserted.
static absl::Status MakeOutputReadyPortsForInputChannels(
    absl::Span<Node* const> all_active_outputs_ready,
    std::vector<std::vector<StreamingInput>>& streaming_inputs,
    std::string_view ready_suffix, Block* block) {
  for (Stage stage = 0; stage < streaming_inputs.size(); ++stage) {
    for (StreamingInput& streaming_input : streaming_inputs[stage]) {
      Node* ready = all_active_outputs_ready.at(stage);
      if (streaming_input.predicate.has_value()) {
        std::vector<Node*> operands{streaming_input.predicate.value(),
                                    all_active_outputs_ready.at(stage)};
        XLS_ASSIGN_OR_RETURN(
            ready, block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kAnd));
      }
      if (streaming_input.fifo_instantiation.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            streaming_input.port_ready,
            block->MakeNode<xls::InstantiationInput>(
                streaming_input.port.value()->loc(), ready,
                streaming_input.fifo_instantiation.value(), "pop_ready"));
      } else {
        XLS_ASSIGN_OR_RETURN(
            streaming_input.port_ready,
            block->AddOutputPort(
                absl::StrCat(streaming_input.channel->name(), ready_suffix),
                ready));
      }
    }
  }

  return absl::OkStatus();
}

// For each input state element, add a corresponding check that it's valid.
// Combinationally combine those valid signals with their predicates to generate
// an all_active_states_valid signal.
//
// Upon success returns a Node* to the all_active_states_valid signal.
static absl::StatusOr<std::vector<Node*>> MakeValidNodesForInputStates(
    std::vector<std::vector<int64_t>>& input_states,
    absl::Span<std::optional<StateRegister> const> state_registers,
    int64_t stage_count, std::string_view valid_suffix, Block* block) {
  std::vector<Node*> result;
  for (Stage stage = 0; stage < stage_count; ++stage) {
    // Add a valid signal for each state element with non-trivial backedge.
    // Gather the valid signals into a vector.
    std::vector<Node*> active_valids;
    for (const int64_t index : input_states[stage]) {
      const std::optional<StateRegister>& state_register =
          state_registers[index];
      if (!state_register.has_value() || !state_register->reg_full) {
        // The state is replaced by the same cycle that reads it, so it will
        // always be valid.
        continue;
      }
      active_valids.push_back(state_register->reg_full_read);
    }

    // And reduce all the active valid signals. This signal is true iff all
    // active states are valid.
    XLS_ASSIGN_OR_RETURN(
        Node * all_active_states_valid,
        NaryAndIfNeeded(block, active_valids,
                        PipelineSignalName("all_active_states_valid", stage)));

    result.push_back(all_active_states_valid);
  }

  return result;
}

// Replace load_en for the register with the given node.
static absl::Status UpdateRegisterLoadEn(Node* load_en, Register* reg,
                                         Block* block) {
  XLS_ASSIGN_OR_RETURN(RegisterWrite * old_reg_write,
                       block->GetRegisterWrite(reg));

  XLS_RETURN_IF_ERROR(block
                          ->MakeNodeWithName<RegisterWrite>(
                              /*loc=*/old_reg_write->loc(),
                              /*data=*/old_reg_write->data(),
                              /*load_enable=*/load_en,
                              /*reset=*/old_reg_write->reset(),
                              /*reg=*/old_reg_write->GetRegister(),
                              /*name=*/old_reg_write->GetName())
                          .status());

  return block->RemoveNode(old_reg_write);
}

// Add a skid buffer after the a set of data/valid/ready signal.
//
// Logic will be inserted immediately after from_data and from node.
// Logic will be inserted immediately before from_rdy,
//   from_rdy must be a node with a single operand.
//
// Updates valid_nodes with the additional nodes associated with valid
// registers.
static absl::StatusOr<Node*> AddSkidBufferToRDVNodes(
    Node* from_data, Node* from_valid, Node* from_rdy,
    std::string_view name_prefix,
    const std::optional<xls::Reset>& reset_behavior, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  CHECK_EQ(from_rdy->operand_count(), 1);

  // Add a node for load_enables (will be removed later).
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  // Create data/valid and their skid counterparts.
  XLS_ASSIGN_OR_RETURN(RegisterRead * data_reg_read,
                       AddRegisterAfterNode(name_prefix, reset_behavior,
                                            literal_1, from_data, block));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_skid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_skid"), reset_behavior,
                           literal_1, data_reg_read, block));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_valid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_valid"), reset_behavior,
                           literal_1, from_valid, block));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_valid_skid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_valid_skid"),
                           reset_behavior, literal_1, data_valid_reg_read,
                           block));

  // If data_valid_skid_reg_read is 1, then data/valid outputs should
  // be selected from the skid set.
  XLS_ASSIGN_OR_RETURN(
      Node * to_valid,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/from_data->loc(),
          std::vector<Node*>{data_valid_reg_read, data_valid_skid_reg_read},
          Op::kOr, absl::StrCat(name_prefix, "_valid_or")));
  XLS_RETURN_IF_ERROR(data_valid_skid_reg_read->ReplaceUsesWith(to_valid));

  XLS_ASSIGN_OR_RETURN(
      Node * to_data,
      block->MakeNodeWithName<Select>(
          /*loc=*/from_data->loc(),
          /*selector=*/data_valid_skid_reg_read,
          /*cases=*/std::vector<Node*>{data_reg_read, data_skid_reg_read},
          /*default_value=*/std::nullopt,
          /*name=*/absl::StrCat(name_prefix, "_select")));
  XLS_RETURN_IF_ERROR(data_skid_reg_read->ReplaceUsesWith(to_data));

  // With a skid buffer, input can be accepted whenever the skid registers
  // are empty/invalid.
  Node* to_is_ready = from_rdy->operand(0);

  XLS_ASSIGN_OR_RETURN(
      Node * from_skid_rdy,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), data_valid_skid_reg_read, Op::kNot,
          absl::StrCat(name_prefix, "_from_skid_rdy")));
  XLS_RETURN_IF_ERROR(from_rdy->ReplaceOperandNumber(0, from_skid_rdy));

  // A. Data is set when
  //    input is read, i.e. from_valid and from_skid_rdy (skid is not valid)
  //
  // B. Valid is set when
  //     a) input is read or
  //     b) data is sent to output (valid and to_ready) and skid is not valid
  //
  // In the case of A, valid is set to 1,
  XLS_ASSIGN_OR_RETURN(
      Node * input_ready_and_valid,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(), std::vector<Node*>{from_valid, from_skid_rdy},
          Op::kAnd, absl::StrCat(name_prefix, "_data_valid_load_en")));

  XLS_ASSIGN_OR_RETURN(RegisterWrite * data_reg_write,
                       block->GetRegisterWrite(data_reg_read->GetRegister()));
  XLS_RETURN_IF_ERROR(
      data_reg_write->ReplaceExistingLoadEnable(input_ready_and_valid));

  XLS_ASSIGN_OR_RETURN(
      Node * data_is_sent_to,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{data_valid_reg_read, to_is_ready, from_skid_rdy},
          Op::kAnd, absl::StrCat(name_prefix, "_data_is_sent_to")));

  XLS_ASSIGN_OR_RETURN(
      Node * valid_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{data_is_sent_to, input_ready_and_valid}, Op::kOr,
          absl::StrCat(name_prefix, "_data_valid_load_en")));

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * data_valid_reg_write,
      block->GetRegisterWrite(data_valid_reg_read->GetRegister()));
  XLS_RETURN_IF_ERROR(
      data_valid_reg_write->ReplaceExistingLoadEnable(valid_load_en));

  // Skid is loaded from 1st stage whenever
  //   a) the input is being read (input_ready_and_valid == 1) and
  //       --> which implies that the skid is invalid
  //   b) the output is not ready (from_rdy_original_src == 0) and
  //   c) there is data in the 1st stage (data_valid_reg_read == 1)
  XLS_ASSIGN_OR_RETURN(Node * to_is_not_rdy,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), to_is_ready, Op::kNot,
                           absl::StrCat(name_prefix, "_to_is_not_rdy")));

  XLS_ASSIGN_OR_RETURN(
      Node * skid_data_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{data_valid_reg_read, input_ready_and_valid,
                             to_is_not_rdy},
          Op::kAnd, absl::StrCat(name_prefix, "_skid_data_load_en")));

  // Skid is reset (valid set to zero) to invalid whenever
  //   a) skid is valid and
  //   b) output is ready
  XLS_ASSIGN_OR_RETURN(
      Node * skid_valid_set_zero,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{data_valid_skid_reg_read, to_is_ready}, Op::kAnd,
          absl::StrCat(name_prefix, "_skid_valid_set_zero")));

  // Skid valid changes from 0 to 1 (load), or 1 to 0 (set zero).
  XLS_ASSIGN_OR_RETURN(
      Node * skid_valid_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{skid_data_load_en, skid_valid_set_zero}, Op::kOr,
          absl::StrCat(name_prefix, "_skid_valid_load_en")));

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * data_skid_reg_write,
      block->GetRegisterWrite(data_skid_reg_read->GetRegister()));
  XLS_RETURN_IF_ERROR(
      data_skid_reg_write->ReplaceExistingLoadEnable(skid_data_load_en));

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * data_valid_skid_reg_write,
      block->GetRegisterWrite(data_valid_skid_reg_read->GetRegister()));

  // If the skid valid is being set
  //   - If it's being set to 1, then the input is being read,
  //     and the prior data is being stored into the skid
  //   - If it's being set to 0, then the input is not being read
  //     and we are clearing the skid and sending the data to the output
  // this implies that
  //   skid_valid := skid_valid_load_en ? !skid_valid : skid_valid
  XLS_RETURN_IF_ERROR(
      data_valid_skid_reg_write->ReplaceExistingLoadEnable(skid_valid_load_en));
  XLS_RETURN_IF_ERROR(
      data_valid_skid_reg_write->ReplaceOperandNumber(0, from_skid_rdy));

  valid_nodes.push_back(to_valid);

  return to_data;
}

// Add flops after the data/valid of a set of three data, valid, and ready
// nodes.
//
// Updates valid_nodes with the additional nodes associated with valid
// registers.
//
// Logic will be inserted immediately after from_data and from node.
// Logic will be inserted immediately before from_rdy,
//   from_rdy must be a node with a single operand.
//
// Returns the node for the register_read of the data.
static absl::StatusOr<RegisterRead*> AddRegisterToRDVNodes(
    Node* from_data, Node* from_valid, Node* from_rdy,
    std::string_view name_prefix,
    const std::optional<xls::Reset>& reset_behavior, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  CHECK_EQ(from_rdy->operand_count(), 1);

  XLS_ASSIGN_OR_RETURN(RegisterRead * data_reg_read,
                       AddRegisterAfterNode(name_prefix, reset_behavior,
                                            std::nullopt, from_data, block));
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * valid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_valid"), reset_behavior,
                           std::nullopt, from_valid, block));

  // 2. Construct and update the ready signal.
  Node* from_rdy_src = from_rdy->operand(0);
  Register* data_reg = data_reg_read->GetRegister();
  Register* valid_reg = valid_reg_read->GetRegister();

  std::string not_valid_name = absl::StrCat(name_prefix, "_valid_inv");
  XLS_ASSIGN_OR_RETURN(
      Node * not_valid,
      block->MakeNodeWithName<UnOp>(/*loc=*/SourceInfo(), valid_reg_read,
                                    Op::kNot, not_valid_name));

  std::string valid_load_en_name = absl::StrCat(name_prefix, "_valid_load_en");
  XLS_ASSIGN_OR_RETURN(
      Node * valid_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(), std::vector<Node*>({from_rdy_src, not_valid}),
          Op::kOr, valid_load_en_name));

  std::string data_load_en_name = absl::StrCat(name_prefix, "_load_en");
  XLS_ASSIGN_OR_RETURN(
      Node * data_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(), std::vector<Node*>({from_valid, valid_load_en}),
          Op::kAnd, data_load_en_name));

  CHECK(from_rdy->ReplaceOperand(from_rdy_src, data_load_en));

  // 3. Update load enables for the data and valid registers.
  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(data_load_en, data_reg, block));
  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(valid_load_en, valid_reg, block));

  valid_nodes.push_back(valid_reg_read);

  return data_reg_read;
}

// Adds a register after the input streaming channel's data and valid.
static absl::StatusOr<Node*> AddRegisterAfterStreamingInput(
    StreamingInput& input, const CodegenOptions& options, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  const std::optional<xls::Reset> reset_behavior = options.ResetBehavior();

  XLS_ASSIGN_OR_RETURN(std::string port_name, StreamingIOName(*input.port));
  if (options.flop_inputs_kind() ==
      CodegenOptions::IOKind::kZeroLatencyBuffer) {
    return AddZeroLatencyBufferToRDVNodes(*input.port, input.port_valid,
                                          input.port_ready, port_name,
                                          reset_behavior, block, valid_nodes);
  }

  if (options.flop_inputs_kind() == CodegenOptions::IOKind::kSkidBuffer) {
    return AddSkidBufferToRDVNodes(*input.port, input.port_valid,
                                   input.port_ready, port_name, reset_behavior,
                                   block, valid_nodes);
  }

  if (options.flop_inputs_kind() == CodegenOptions::IOKind::kFlop) {
    return AddRegisterToRDVNodes(*input.port, input.port_valid,
                                 input.port_ready, port_name, reset_behavior,
                                 block, valid_nodes);
  }

  return absl::UnimplementedError(absl::StrFormat(
      "Block conversion does not support registering input with kind %d",
      options.flop_inputs_kind()));
}

// Adds a register after the input streaming channel's data and valid.
// Returns the node for the register_read of the data.
static absl::StatusOr<Node*> AddRegisterBeforeStreamingOutput(
    StreamingOutput& output, const CodegenOptions& options, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  // Add buffers before the data/valid output ports and after
  // the ready input port to serve as points where the
  // additional logic from AddRegisterToRDVNodes() can be inserted.
  XLS_ASSIGN_OR_RETURN(std::string port_name, StreamingIOName(*output.port));
  XLS_ASSIGN_OR_RETURN(std::string port_valid_name,
                       StreamingIOName(output.port_valid));
  XLS_ASSIGN_OR_RETURN(std::string port_ready_name,
                       StreamingIOName(output.port_ready));
  std::string data_buf_name = absl::StrFormat("__%s_buf", port_name);
  std::string valid_buf_name = absl::StrFormat("__%s_buf", port_valid_name);
  std::string ready_buf_name = absl::StrFormat("__%s_buf", port_ready_name);
  XLS_ASSIGN_OR_RETURN(
      Node * output_port_data_buf,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), output.port.value()->operand(0), Op::kIdentity,
          data_buf_name));
  XLS_RETURN_IF_ERROR(
      output.port.value()->ReplaceOperandNumber(0, output_port_data_buf));

  XLS_ASSIGN_OR_RETURN(Node * output_port_valid_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), output.port_valid->operand(0),
                           Op::kIdentity, valid_buf_name));
  XLS_RETURN_IF_ERROR(
      output.port_valid->ReplaceOperandNumber(0, output_port_valid_buf));

  XLS_ASSIGN_OR_RETURN(Node * output_port_ready_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), output.port_ready,
                           Op::kIdentity, ready_buf_name));

  XLS_RETURN_IF_ERROR(
      output.port_ready->ReplaceUsesWith(output_port_ready_buf));

  const std::optional<xls::Reset> reset_behavior = options.ResetBehavior();

  if (options.flop_outputs_kind() ==
      CodegenOptions::IOKind::kZeroLatencyBuffer) {
    return AddZeroLatencyBufferToRDVNodes(
        output_port_data_buf, output_port_valid_buf, output_port_ready_buf,
        port_name, reset_behavior, block, valid_nodes);
  }

  if (options.flop_outputs_kind() == CodegenOptions::IOKind::kSkidBuffer) {
    return AddSkidBufferToRDVNodes(output_port_data_buf, output_port_valid_buf,
                                   output_port_ready_buf, port_name,
                                   reset_behavior, block, valid_nodes);
  }

  if (options.flop_outputs_kind() == CodegenOptions::IOKind::kFlop) {
    return AddRegisterToRDVNodes(output_port_data_buf, output_port_valid_buf,
                                 output_port_ready_buf, port_name,
                                 reset_behavior, block, valid_nodes);
  }

  return absl::UnimplementedError(absl::StrFormat(
      "Block conversion does not support registering output with kind %d",
      options.flop_outputs_kind()));
}

// Adds an input and/or output flop and related signals.
// For streaming inputs, data/valid are registered, but the ready signal
// remains as a feed-through pass.
//
// Ready is asserted if the input is ready and either a) the flop
// is invalid; or b) the flop is valid but the pipeline will accept the data,
// on the next clock tick.
static absl::Status AddInputOutputFlops(
    const CodegenOptions& options, StreamingIOPipeline& streaming_io,
    Block* block, std::vector<std::optional<Node*>>& valid_nodes) {
  absl::flat_hash_set<Node*> handled_io_nodes;

  // Flop streaming inputs.
  for (auto& vec : streaming_io.inputs) {
    for (StreamingInput& input : vec) {
      if (options.flop_inputs()) {
        XLS_RETURN_IF_ERROR(
            AddRegisterAfterStreamingInput(input, options, block, valid_nodes)
                .status());

        handled_io_nodes.insert(*input.port);
        handled_io_nodes.insert(input.port_valid);
      }

      // ready for an output port is an output for the block,
      // record that we should not separately add a flop these inputs.
      handled_io_nodes.insert(input.port_ready);
    }
  }

  // Flop streaming outputs.
  for (auto& vec : streaming_io.outputs) {
    for (StreamingOutput& output : vec) {
      if (options.flop_outputs()) {
        XLS_RETURN_IF_ERROR(AddRegisterBeforeStreamingOutput(output, options,
                                                             block, valid_nodes)
                                .status());

        handled_io_nodes.insert(*output.port);
        handled_io_nodes.insert(output.port_valid);
      }

      // ready for an output port is an input for the block,
      // record that we should not separately add a flop these inputs.
      handled_io_nodes.insert(output.port_ready);
    }
  }

  // Flop other inputs.
  if (options.flop_inputs() && options.flop_single_value_channels()) {
    for (InputPort* port : block->GetInputPorts()) {
      // Skip input ports that
      //  a) Belong to streaming input or output channels.
      //  b) Is the reset port.
      if (handled_io_nodes.contains(port) ||
          port == block->GetResetPort().value()) {
        continue;
      }

      XLS_RETURN_IF_ERROR(AddRegisterAfterNode(port->GetName(),
                                               options.ResetBehavior(),
                                               std::nullopt, port, block)
                              .status());

      handled_io_nodes.insert(port);
    }
  }

  // Flop other outputs
  if (options.flop_outputs() && options.flop_single_value_channels()) {
    for (OutputPort* port : block->GetOutputPorts()) {
      // Skip output ports that belong to streaming input or output channels.
      if (handled_io_nodes.contains(port)) {
        continue;
      }

      XLS_RETURN_IF_ERROR(
          AddRegisterAfterNode(port->GetName(), options.ResetBehavior(),
                               std::nullopt, port->operand(0), block)
              .status());

      handled_io_nodes.insert(port);
    }
  }

  return absl::OkStatus();
}

// Add one-shot logic to the and output RDV channel.
static absl::Status AddOneShotLogicToRVNodes(
    Node* from_valid, Node* from_rdy, Node* all_active_outputs_ready,
    std::string_view name_prefix,
    const std::optional<xls::Reset>& reset_behavior, Block* block) {
  // Location for added logic is taken from from_valid.
  SourceInfo loc = from_valid->loc();

  // Add a node to serve as placeholder (will be removed later).
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  // Create a register to store whether or not channel has been sent.
  Type* u1 = block->package()->GetBitsType(1);
  std::string name = absl::StrFormat("__%s_has_been_sent_reg", name_prefix);
  XLS_ASSIGN_OR_RETURN(Register * has_been_sent_reg,
                       block->AddRegister(name, u1, reset_behavior));
  XLS_ASSIGN_OR_RETURN(RegisterWrite * has_been_sent_reg_write,
                       block->MakeNode<RegisterWrite>(
                           /*loc=*/loc, literal_1,
                           /*load_enable=*/literal_1,
                           /*reset=*/block->GetResetPort(), has_been_sent_reg));
  XLS_ASSIGN_OR_RETURN(RegisterRead * has_been_sent,
                       block->MakeNode<RegisterRead>(
                           /*loc=*/from_valid->loc(), has_been_sent_reg));

  // Regenerate ready as OR of
  // 1) output channel is ready.
  // 2) data has already been sent on the channel.
  CHECK_EQ(from_rdy->operand_count(), 1);
  Node* to_is_ready = from_rdy->operand(0);
  name = absl::StrFormat("__%s_has_sent_or_is_ready", name_prefix);
  XLS_ASSIGN_OR_RETURN(
      Node * has_sent_or_is_ready,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/loc, std::vector<Node*>({to_is_ready, has_been_sent}),
          Op::kOr, name));
  XLS_RETURN_IF_ERROR(from_rdy->ReplaceOperandNumber(0, has_sent_or_is_ready));

  // Clamp output valid if has already sent.
  name = absl::StrFormat("__%s_not_has_been_sent", name_prefix);
  XLS_ASSIGN_OR_RETURN(Node * not_has_been_sent,
                       block->MakeNodeWithName<UnOp>(/*loc=*/loc, has_been_sent,
                                                     Op::kNot, name));

  name = absl::StrFormat("__%s_valid_and_not_has_been_sent", name_prefix);
  XLS_ASSIGN_OR_RETURN(
      Node * valid_and_not_has_been_sent,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/loc, std::vector<Node*>({from_valid, not_has_been_sent}),
          Op::kAnd, name));
  XLS_RETURN_IF_ERROR(from_valid->ReplaceUsesWith(valid_and_not_has_been_sent));

  // Data is transferred whenever valid and ready
  name = absl::StrFormat("__%s_valid_and_ready_txfr", name_prefix);
  XLS_ASSIGN_OR_RETURN(
      Node * valid_and_ready_txfr,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/loc,
          std::vector<Node*>({valid_and_not_has_been_sent, to_is_ready}),
          Op::kAnd, name));

  // Now set up the has_sent register
  //  txfr = valid and ready -- data has been transferred to output
  //  load = all_active_outputs_ready && valid
  //    -- pipeline stage is now being loaded with new inputs.
  //    -- Not that && valid is not strictly needed but added to remove a case
  //       where x-propagation is a bit pessimistic.
  //
  // if load
  //   has_been_sent <= 0
  // else if triggered
  //   has_been_sent <= 1
  //
  // this can be implemented with
  //   load_enable = load + triggered
  //   data = !load
  name =
      absl::StrFormat("__%s_valid_and_all_active_outputs_ready", name_prefix);
  XLS_ASSIGN_OR_RETURN(
      Node * valid_and_all_active_outputs_ready,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/loc,
          std::vector<Node*>({from_valid, all_active_outputs_ready}), Op::kAnd,
          name));

  name = absl::StrFormat("__%s_has_been_sent_reg_load_en", name_prefix);
  XLS_ASSIGN_OR_RETURN(
      Node * load_enable,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/loc,
          std::vector<Node*>(
              {valid_and_ready_txfr, valid_and_all_active_outputs_ready}),
          Op::kOr, name));

  name = absl::StrFormat("__%s_not_stage_load", name_prefix);
  XLS_ASSIGN_OR_RETURN(
      Node * not_stage_load,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/loc, valid_and_all_active_outputs_ready, Op::kNot, name));

  XLS_RETURN_IF_ERROR(
      has_been_sent_reg_write->ReplaceOperandNumber(0, not_stage_load));

  XLS_RETURN_IF_ERROR(
      has_been_sent_reg_write->ReplaceExistingLoadEnable(load_enable));

  return absl::OkStatus();
}

// Adds logic to ensure that for multi-output blocks that
// an output is not transferred more than once.
//
// For multiple-output blocks, even if all outputs are valid
// at the same time, it may not be the case that their destinations
// are ready.  In this case, for N output sends, M<N sends
// may be completed. In  subsequent cycles, more sends may yet be completed.
//
// This logic is to ensure that in those subsequent cycles,
// sends that are already completed have valid set to zero to prevent
// sending an output twice.
static absl::Status AddOneShotOutputLogic(
    const CodegenOptions& options, const StreamingIOPipeline& streaming_io,
    absl::Span<Node* const> all_active_outputs_ready, Block* block) {
  CHECK(!all_active_outputs_ready.empty());

  for (Stage stage = 0; stage < streaming_io.outputs.size(); ++stage) {
    for (const StreamingOutput& output : streaming_io.outputs.at(stage)) {
      // Add an buffers before the valid output ports and after
      // the ready input port to serve as points where the
      // additional logic from AddRegisterToRDVNodes() can be inserted.
      XLS_ASSIGN_OR_RETURN(std::string port_name,
                           StreamingIOName(*output.port));
      XLS_ASSIGN_OR_RETURN(std::string port_valid_name,
                           StreamingIOName(output.port_valid));
      std::string valid_buf_name = absl::StrFormat("__%s_buf", port_valid_name);

      XLS_ASSIGN_OR_RETURN(
          Node * output_port_valid_buf,
          block->MakeNodeWithName<UnOp>(
              /*loc=*/SourceInfo(), output.port_valid->operand(0),
              Op::kIdentity, valid_buf_name));
      XLS_RETURN_IF_ERROR(
          output.port_valid->ReplaceOperandNumber(0, output_port_valid_buf));

      XLS_ASSIGN_OR_RETURN(Node * output_port_ready_buf,
                           block->MakeNodeWithName<UnOp>(
                               /*loc=*/SourceInfo(), output.port_ready,
                               Op::kIdentity, valid_buf_name));

      XLS_RETURN_IF_ERROR(
          output.port_ready->ReplaceUsesWith(output_port_ready_buf));

      XLS_RETURN_IF_ERROR(
          AddOneShotLogicToRVNodes(output_port_valid_buf, output_port_ready_buf,
                                   all_active_outputs_ready[stage], port_name,
                                   options.ResetBehavior(), block));
    }
  }

  return absl::OkStatus();
}

// Adds an idle signal (un-flopped) to the block.
//
// Idle is asserted if valid signals for
//  a) input streaming channels
//  b) pipeline registers and any other valid flop for saved state
//  c) output streaming channels
// are asserted.
//
// TODO(tedhong): 2022-02-01 There may be some redundancy between B and C,
// Create an optimization pass within the codegen pipeline to remove it.
static absl::Status AddIdleOutput(
    std::vector<std::optional<Node*>>& valid_nodes,
    StreamingIOPipeline& streaming_io, Block* block) {
  // The block is not idle (and is implicitly valid), if there are no
  // active recvs that can block it.
  XLS_ASSIGN_OR_RETURN(Node * stage_0_has_no_active_recvs,
                       Stage0HasNoActiveRecvs(streaming_io.inputs, block));
  valid_nodes.push_back(stage_0_has_no_active_recvs);

  for (auto& vec : streaming_io.inputs) {
    for (StreamingInput& input : vec) {
      XLS_RET_CHECK(input.signal_valid.has_value());
      valid_nodes.push_back(input.signal_valid);
    }
  }

  for (auto& vec : streaming_io.outputs) {
    for (StreamingOutput& output : vec) {
      valid_nodes.push_back(output.port_valid->operand(0));
    }
  }

  Node* idle_signal;
  {
    std::vector<Node*> valid_nodes_no_optional;
    valid_nodes_no_optional.reserve(valid_nodes.size());
    for (const std::optional<Node*>& valid_node : valid_nodes) {
      if (valid_node.has_value()) {
        valid_nodes_no_optional.push_back(*valid_node);
      }
    }
    XLS_ASSIGN_OR_RETURN(idle_signal,
                         NaryNorIfNeeded(block, valid_nodes_no_optional));
  }

  XLS_ASSIGN_OR_RETURN(streaming_io.idle_port,
                       block->AddOutputPort("idle", idle_signal));

  return absl::OkStatus();
}

// Adds ready/valid ports for each of the given streaming inputs/outputs. Also,
// adds logic which propagates ready and valid signals through the block.
//
// Returns (via reference argument) the vector of all_active_output_ready nodes.
// See MakeInputReadyPortsForOutputChannels() for more.
static absl::Status AddBubbleFlowControl(
    const CodegenOptions& options, StreamingIOPipeline& streaming_io,
    Block* block, std::vector<Node*>& all_active_outputs_ready) {
  int64_t stage_count = streaming_io.pipeline_registers.size() + 1;
  std::string_view valid_suffix = options.streaming_channel_valid_suffix();
  std::string_view ready_suffix = options.streaming_channel_ready_suffix();

  // Node in each stage that represents when all inputs channels (that are
  // predicated true) are valid.
  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> all_active_inputs_valid,
      MakeInputValidPortsForInputChannels(streaming_io.inputs, stage_count,
                                          valid_suffix, block));
  VLOG(3) << "After Inputs Valid";
  XLS_VLOG_LINES(3, block->DumpIr());

  // Node in each stage that represents when all output channels (that are
  // predicated true) are ready.
  XLS_ASSIGN_OR_RETURN(
      all_active_outputs_ready,
      MakeInputReadyPortsForOutputChannels(streaming_io.outputs, stage_count,
                                           ready_suffix, block));
  VLOG(3) << "After Outputs Ready";
  XLS_VLOG_LINES(3, block->DumpIr());

  // Node in each stage that represents when all state values are valid.
  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> all_active_states_valid,
      MakeValidNodesForInputStates(streaming_io.input_states,
                                   streaming_io.state_registers, stage_count,
                                   valid_suffix, block));
  VLOG(3) << "After States Valid";
  XLS_VLOG_LINES(3, block->DumpIr());

  std::optional<xls::Reset> reset_behavior = options.ResetBehavior();

  XLS_RETURN_IF_ERROR(MakePipelineStagesForValidIO(
      streaming_io, /*recvs_valid=*/all_active_inputs_valid,
      /*states_valid=*/all_active_states_valid,
      /*sends_ready=*/all_active_outputs_ready, reset_behavior, block));

  VLOG(3) << "After Valids";
  XLS_VLOG_LINES(3, block->DumpIr());

  for (std::optional<StateRegister>& state_register :
       streaming_io.state_registers) {
    if (!state_register.has_value()) {
      continue;
    }
    XLS_RETURN_IF_ERROR(
        UpdateStateRegisterWithReset(reset_behavior, *state_register, block));
  }

  VLOG(3) << "After State Updated";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (options.reset().has_value() && options.reset()->reset_data_path()) {
    XLS_RETURN_IF_ERROR(UpdateDatapathRegistersWithReset(
        reset_behavior, absl::MakeSpan(streaming_io.pipeline_registers),
        block));

    VLOG(3) << "After Datapath Reset Updated";
    XLS_VLOG_LINES(3, block->DumpIr());
  }

  XLS_ASSIGN_OR_RETURN(BubbleFlowControl bubble_flow_control,
                       UpdatePipelineWithBubbleFlowControl(
                           options, absl::MakeSpan(streaming_io.pipeline_valid),
                           absl::MakeSpan(streaming_io.stage_done),
                           absl::MakeSpan(streaming_io.pipeline_registers),
                           absl::MakeSpan(streaming_io.state_registers),
                           streaming_io.node_to_stage_map, block));

  VLOG(3) << "After Bubble Flow Control (pipeline)";
  XLS_VLOG_LINES(3, block->DumpIr());

  {
    // MakeOutputValidPortsForOutputChannels takes a std::vector<Node*>, not
    // std::vector<std::optional<Node*>>.
    std::vector<Node*> stage_valid_no_option;
    stage_valid_no_option.reserve(streaming_io.stage_valid.size());
    for (const std::optional<Node*>& stage_valid : streaming_io.stage_valid) {
      stage_valid_no_option.push_back(*stage_valid);
    }
    XLS_RETURN_IF_ERROR(MakeOutputValidPortsForOutputChannels(
        all_active_inputs_valid, stage_valid_no_option,
        bubble_flow_control.next_stage_open, streaming_io.outputs, valid_suffix,
        block));
  }

  VLOG(3) << "After Outputs Valid";
  XLS_VLOG_LINES(3, block->DumpIr());

  // Handle flow control for the single pipeline stage case.
  if (streaming_io.pipeline_registers.empty()) {
    XLS_ASSIGN_OR_RETURN(Node * input_stage_enable,
                         UpdateSingleStagePipelineWithFlowControl(
                             bubble_flow_control.data_load_enable.front(),
                             *streaming_io.stage_done.at(0),
                             absl::MakeSpan(streaming_io.state_registers),
                             streaming_io.node_to_stage_map, block));
    bubble_flow_control.data_load_enable = {input_stage_enable};
  }

  VLOG(3) << "After Single Stage Flow Control (state)";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_RETURN_IF_ERROR(MakeOutputReadyPortsForInputChannels(
      bubble_flow_control.data_load_enable, streaming_io.inputs, ready_suffix,
      block));

  VLOG(3) << "After Ready";
  XLS_VLOG_LINES(3, block->DumpIr());
  return absl::OkStatus();
}

// Adds ready/valid ports for each of the given streaming inputs/outputs. Also,
// adds logic which propagates ready and valid signals through the block.
static absl::Status AddCombinationalFlowControl(
    std::vector<std::vector<StreamingInput>>& streaming_inputs,
    std::vector<std::vector<StreamingOutput>>& streaming_outputs,
    std::vector<std::optional<Node*>>& stage_valid,
    const CodegenOptions& options, Block* block) {
  std::string_view valid_suffix = options.streaming_channel_valid_suffix();
  std::string_view ready_suffix = options.streaming_channel_ready_suffix();

  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> all_active_outputs_ready,
      MakeInputReadyPortsForOutputChannels(streaming_outputs, /*stage_count=*/1,
                                           ready_suffix, block));

  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> all_active_inputs_valid,
      MakeInputValidPortsForInputChannels(streaming_inputs, /*stage_count=*/1,
                                          valid_suffix, block));

  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));
  std::vector<Node*> pipelined_valids{literal_1};
  std::vector<Node*> next_stage_open{literal_1};
  XLS_RETURN_IF_ERROR(MakeOutputValidPortsForOutputChannels(
      all_active_inputs_valid, pipelined_valids, next_stage_open,
      streaming_outputs, valid_suffix, block));

  XLS_RETURN_IF_ERROR(MakeOutputReadyPortsForInputChannels(
      all_active_outputs_ready, streaming_inputs, ready_suffix, block));

  XLS_RET_CHECK(stage_valid.empty());
  XLS_RET_CHECK_EQ(all_active_inputs_valid.size(), 1);
  stage_valid.push_back(all_active_inputs_valid.front());

  return absl::OkStatus();
}

// Send/receive nodes are not cloned from the proc into the block, but the
// network of tokens connecting these send/receive nodes *is* cloned. This
// function removes the token operations.
static absl::Status RemoveDeadTokenNodes(CodegenPassUnit* unit) {
  // Receive nodes produce a tuple of a token and a data value. In the block
  // this becomes a tuple of a token and an InputPort. Run tuple simplification
  // to disentangle the tuples so DCE can do its work and eliminate the token
  // network.

  // TODO: We really shouldn't be running passes like this during block
  // conversion. These should be fully in the pipeline. This is work for the
  // future.
  CodegenPassResults pass_results;
  CodegenPassOptions pass_options;
  CodegenCompoundPass ccp("block_conversion_dead_token_removal",
                          "Dead token removal during block-conversion process");
  ccp.AddInvariantChecker<CodegenChecker>();
  ccp.Add<CodegenWrapperPass>(std::make_unique<DataflowSimplificationPass>());
  ccp.Add<CodegenWrapperPass>(std::make_unique<DeadCodeEliminationPass>());
  ccp.Add<RegisterLegalizationPass>();
  ccp.Add<CodegenWrapperPass>(std::make_unique<DeadCodeEliminationPass>());

  XLS_RETURN_IF_ERROR(ccp.Run(unit, pass_options, &pass_results).status());
  // Nodes like cover and assert have token types and will cause
  // a dangling token network remaining.
  //
  // TODO(tedhong): 2022-02-14, clean up dangling token
  // network to ensure that deleted nodes can't be accessed via normal
  // ir operations.

  return absl::OkStatus();
}

// Determine which stages are mutually exclusive with each other.
//
// Takes a map from stage number to the state elements which are read and
// written to on each stage.
//
// Since each state element defines a mutual exclusive zone lasting from its
// first read to its first write we can walk through the stage list updating the
// mutual exclusion state.
absl::StatusOr<ConcurrentStageGroups> CalculateConcurrentGroupsFromStateWrites(
    absl::Span<const std::optional<StateRegister>> state_registers,
    int64_t stage_count) {
  ConcurrentStageGroups result(stage_count);
  // Find all the mutex regions
  for (const auto& reg : state_registers) {
    if (!reg) {
      continue;
    }
    auto start = reg->read_stage;
    if (reg->next_values.empty()) {
      // If empty, absl::c_min_element()->stage will dereference the end
      // iterator. Skip instead.
      continue;
    }
    auto end = absl::c_min_element(reg->next_values,
                                   [](const StateRegister::NextValue& l,
                                      const StateRegister::NextValue& r) {
                                     return l.stage < r.stage;
                                   })
                   ->stage;
    if (start == end) {
      continue;
    }
    for (int64_t i = start; i < end; ++i) {
      // NB <= since end is inclusive.
      for (int64_t j = i + 1; j <= end; ++j) {
        result.MarkMutuallyExclusive(i, j);
      }
    }
  }
  return result;
}

// Clones every node in the given func/proc into the given block. Some nodes are
// handled specially:
//
// * Proc token parameter becomes an operandless AfterAll operation in the
//   block.
// * Proc state parameter (which must be an empty tuple) becomes a Literal
//   operation in the block.
// * Receive operations become InputPorts.
// * Send operations become OutputPorts.
// * Function parameters become InputPorts.
// * The Function return value becomes an OutputPort.
//
// GetResult() returns a StreamingIOPipeline which
//   1. Contains the InputPorts and OutputPorts created from
//      Send/Receive operations of streaming channels
//   2. Contains a list of PipelineRegisters per stage of the pipeline.
//
// Example input proc:
//
//   chan x_ch(bits[32], kind=streaming, flow_control=single_value, id=0, ...)
//   chan y_ch(bits[32], kind=streaming, flow_control=single_value, id=1, ...)
//
//   proc foo(tkn: token, st: (), init=42) {
//     rcv_x: (token, bits[32]) = receive(tkn, channel=x_ch)
//     rcv_x_token: token = tuple_index(rcv_x, index=0)
//     x: bits[32] = tuple_index(rcv_x, index=1)
//     not_x: bits[32] = not(x)
//     snd_y: token = send(rcv_x_token, not_x, channel=y_ch)
//     next (tkn, snd_y)
//   }
//
// Resulting block:
//
//  block (x: bits[32], y: bits[32]) {
//    x: bits[32] = input_port(name=x)
//    not_x: bits[32] = not(x)
//    y: bits[32] = output_port(not_x, name=x)
//  }
//
// Ready/valid flow control including inputs ports and output ports are added
// later.
class CloneNodesIntoBlockHandler {
 public:
  static absl::StatusOr<absl::flat_hash_set<int64_t>> GetLoopbackChannelIds(
      Package* p) {
    XLS_ASSIGN_OR_RETURN(auto nodes_for_channel, ChannelUsers(p));
    absl::flat_hash_set<int64_t> loopback_channel_ids;
    for (auto& [chan, nodes] : nodes_for_channel) {
      if (chan->supported_ops() != ChannelOps::kSendReceive) {
        continue;
      }
      bool saw_send = false;
      bool saw_receive = false;
      bool same_fb = true;
      FunctionBase* fb = nullptr;
      for (Node* node : nodes) {
        if (node->Is<Send>()) {
          saw_send = true;
        } else if (node->Is<Receive>()) {
          saw_receive = true;
        }
        if (fb == nullptr) {
          fb = node->function_base();
        } else {
          same_fb &= fb == node->function_base();
        }
      }
      if (saw_send && saw_receive && same_fb) {
        loopback_channel_ids.insert(chan->id());
      }
    }

    return loopback_channel_ids;
  }

  // Initialize this object with the proc/function, the block the
  // proc/function should be cloned into, and the stage_count.
  //
  // If the block is to be a combinational block, stage_count should be
  // set to 0;
  CloneNodesIntoBlockHandler(FunctionBase* proc_or_function,
                             int64_t stage_count, const CodegenOptions& options,
                             Block* block)
      : is_proc_(proc_or_function->IsProc()),
        function_base_(proc_or_function),
        options_(options),
        block_(block),
        fifo_instantiations_({}) {
    absl::StatusOr<absl::flat_hash_set<int64_t>>
        loopback_channel_ids_or_status =
            GetLoopbackChannelIds(proc_or_function->package());
    CHECK_OK(loopback_channel_ids_or_status.status());
    loopback_channel_ids_ = std::move(loopback_channel_ids_or_status.value());
    if (is_proc_) {
      Proc* proc = function_base_->AsProcOrDie();
      result_.state_registers.resize(proc->GetStateElementCount());
    }
    if (stage_count > 1) {
      result_.pipeline_registers.resize(stage_count - 1);
    }
    result_.inputs.resize(stage_count + 1);
    result_.outputs.resize(stage_count + 1);
    result_.input_states.resize(stage_count + 1);
    result_.output_states.resize(stage_count + 1);
  }

  // For a given set of sorted nodes, process and clone them into the
  // block.
  absl::Status CloneNodes(absl::Span<Node* const> sorted_nodes, int64_t stage) {
    for (Node* node : sorted_nodes) {
      Node* next_node = nullptr;
      if (node->Is<Param>()) {
        if (is_proc_) {
          XLS_ASSIGN_OR_RETURN(next_node, HandleStateParam(node, stage));
        } else {
          XLS_ASSIGN_OR_RETURN(next_node, HandleFunctionParam(node));
        }
      } else if (node->Is<Next>()) {
        XLS_RET_CHECK(is_proc_);
        XLS_RETURN_IF_ERROR(HandleNextValue(node, stage));
      } else if (IsChannelNode(node)) {
        XLS_RET_CHECK(is_proc_);
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));

        if (loopback_channel_ids_.contains(channel->id()) &&
            channel->kind() == ChannelKind::kStreaming) {
          StreamingChannel* streaming_channel =
              down_cast<StreamingChannel*>(channel);
          XLS_RET_CHECK(streaming_channel->fifo_config().has_value());

          xls::Instantiation* instantiation;
          auto [itr, inserted] =
              fifo_instantiations_.insert({channel->id(), nullptr});
          if (inserted) {
            std::string inst_name = absl::StrFormat("fifo_%s", channel->name());
            XLS_ASSIGN_OR_RETURN(
                instantiation,
                block()->AddInstantiation(
                    inst_name,
                    std::make_unique<xls::FifoInstantiation>(
                        inst_name, *streaming_channel->fifo_config(),
                        streaming_channel->type(), streaming_channel->name(),
                        block()->package())));
            itr->second = instantiation;
          } else {
            instantiation = itr->second;
          }
          xls::FifoInstantiation* fifo_instantiation =
              down_cast<xls::FifoInstantiation*>(instantiation);
          if (node->Is<Receive>()) {
            XLS_ASSIGN_OR_RETURN(
                next_node, HandleFifoReceiveNode(node->As<Receive>(), stage,
                                                 fifo_instantiation));
          } else {
            XLS_ASSIGN_OR_RETURN(next_node,
                                 HandleFifoSendNode(node->As<Send>(), stage,
                                                    fifo_instantiation));
          }
        } else {
          if (node->Is<Receive>()) {
            XLS_ASSIGN_OR_RETURN(next_node, HandleReceiveNode(node, stage));
          } else {
            XLS_ASSIGN_OR_RETURN(next_node, HandleSendNode(node, stage));
          }
        }
      } else {
        XLS_ASSIGN_OR_RETURN(next_node, HandleGeneralNode(node));
      }
      node_map_[node] = next_node;
      if (next_node != nullptr) {
        result_.node_to_stage_map[next_node] = stage;
      }
    }

    return absl::OkStatus();
  }

  // Add pipeline registers. A register is needed for each node which is
  // scheduled at or before this cycle and has a use after this cycle.
  absl::Status AddNextPipelineStage(const PipelineSchedule& schedule,
                                    int64_t stage) {
    for (Node* function_base_node : function_base_->nodes()) {
      if (schedule.IsLiveOutOfCycle(function_base_node, stage)) {
        Node* node = node_map_.at(function_base_node);

        XLS_ASSIGN_OR_RETURN(
            Node * node_after_stage,
            CreatePipelineRegistersForNode(
                PipelineSignalName(node->GetName(), stage), node, stage,
                result_.pipeline_registers.at(stage)));

        node_map_[function_base_node] = node_after_stage;
      }
    }

    return absl::OkStatus();
  }

  // If a function, create an output port for the function's return.
  absl::Status AddOutputPortsIfFunction(std::string_view output_port_name) {
    if (!is_proc_) {
      Function* function = function_base_->AsFunctionOrDie();
      XLS_ASSIGN_OR_RETURN(
          OutputPort * port,
          block()->AddOutputPort(output_port_name,
                                 node_map_.at(function->return_value())));
      if (std::optional<PackageInterfaceProto::Function> f =
              FindFunctionInterface(options_.package_interface(),
                                    function_base_->name());
          f && f->has_sv_result_type()) {
        // Record sv-type associated with this port.
        result_.output_port_sv_type[port] = f->sv_result_type();
      }
      return absl::OkStatus();
    }

    return absl::OkStatus();
  }

  // Figure out based on the reads and writes to state variables what stages are
  // mutually exclusive with one another.
  absl::Status MarkMutualExclusiveStages(int64_t stage_count) {
    if (!is_proc_) {
      // feed-forward pipelines (non-procs) can't stall in ways that make mutual
      // exclusive stages possible. All stage activation patterns are equally
      // possible.
      concurrent_stages_.reset();
      return absl::OkStatus();
    }
    XLS_ASSIGN_OR_RETURN(concurrent_stages_,
                         CalculateConcurrentGroupsFromStateWrites(
                             result_.state_registers, stage_count));
    return absl::OkStatus();
  }

  // Return structure describing streaming io ports and pipeline registers.
  StreamingIOPipeline GetResult() { return result_; }

  std::optional<ConcurrentStageGroups> GetConcurrentStages() {
    return concurrent_stages_;
  }

 private:
  // Don't clone state Param operations. Instead replace with a RegisterRead
  // operation.
  absl::StatusOr<Node*> HandleStateParam(Node* node, Stage stage) {
    CHECK_GE(stage, 0);

    Proc* proc = function_base_->AsProcOrDie();
    Param* param = node->As<Param>();
    XLS_ASSIGN_OR_RETURN(int64_t index, proc->GetStateParamIndex(param));

    Register* reg = nullptr;
    RegisterRead* reg_read = nullptr;
    if (!node->GetType()->IsToken() && node->GetType()->GetFlatBitCount() > 0) {
      // Create a temporary name as this register will later be removed
      // and updated.  That register should be created with the
      // state parameter's name.  See UpdateStateRegisterWithReset().
      std::string name =
          block()->UniquifyNodeName(absl::StrCat("__", param->name()));

      XLS_ASSIGN_OR_RETURN(reg, block()->AddRegister(name, node->GetType()));

      XLS_ASSIGN_OR_RETURN(reg_read, block()->MakeNodeWithName<RegisterRead>(
                                         node->loc(), reg,
                                         /*name=*/reg->name()));

      result_.node_to_stage_map[reg_read] = stage;
    }

    // The register write will be created later in HandleNextValue.
    result_.state_registers[index] =
        StateRegister{.name = std::string(param->name()),
                      .reset_value = proc->GetInitValueElement(index),
                      .read_stage = stage,
                      .reg = reg,
                      .reg_write = nullptr,
                      .reg_read = reg_read};

    result_.input_states[stage].push_back(index);

    if (reg_read == nullptr) {
      // Parameter has no meaningful data contents; replace with a literal. (We
      // know the parameter has flat bit-count 0, so any literal will have the
      // same result.)
      return block()->MakeNode<xls::Literal>(node->loc(),
                                             ZeroOfType(node->GetType()));
    }
    return reg_read;
  }

  // Replace function parameters with input ports.
  absl::StatusOr<Node*> HandleFunctionParam(Node* node) {
    Param* param = node->As<Param>();
    XLS_ASSIGN_OR_RETURN(InputPort * res,
                         block()->AddInputPort(param->GetName(),
                                               param->GetType(), param->loc()));
    if (std::optional<PackageInterfaceProto::Function> f =
            FindFunctionInterface(options_.package_interface(),
                                  function_base_->name())) {
      // Record sv-type associated with this port.
      auto it = absl::c_find_if(
          f->parameters(), [&](const PackageInterfaceProto::NamedValue& p) {
            return p.name() == param->name();
          });
      if (it != f->parameters().end() && it->has_sv_type()) {
        result_.input_port_sv_type[res] = it->sv_type();
      }
    }
    return res;
  }

  // Replace next values with a RegisterWrite.
  absl::Status HandleNextValue(Node* node, Stage stage) {
    Proc* proc = function_base_->AsProcOrDie();
    Next* next = node->As<Next>();
    Param* param = next->param()->As<Param>();
    XLS_ASSIGN_OR_RETURN(int64_t index, proc->GetStateParamIndex(param));

    CHECK_EQ(proc->GetNextStateElement(index), param);
    StateRegister& state_register = *result_.state_registers.at(index);
    state_register.next_values.push_back(
        {.stage = stage,
         .value = next->value() == next->param()
                      ? std::nullopt
                      : std::make_optional(node_map_.at(next->value())),
         .predicate =
             next->predicate().has_value()
                 ? std::make_optional(node_map_.at(next->predicate().value()))
                 : std::nullopt});

    bool last_next_value =
        absl::c_all_of(proc->next_values(param), [&](Next* next_value) {
          return next_value == next || node_map_.contains(next_value);
        });
    if (!last_next_value) {
      // We don't create the RegisterWrite until we're at the last `next_value`
      // for this `param`, so we've translated all the values already.
      return absl::OkStatus();
    }

    if (param->GetType()->GetFlatBitCount() > 0) {
      // We need a write for the actual value.

      // We should only create the RegisterWrite once.
      CHECK_EQ(state_register.reg_write, nullptr);

      // Make a placeholder RegisterWrite; the real one requires access to all
      // the `next_value` nodes and the control flow logic.
      XLS_ASSIGN_OR_RETURN(state_register.reg_write,
                           block()->MakeNode<RegisterWrite>(
                               next->loc(), node_map_.at(next->value()),
                               /*load_enable=*/std::nullopt,
                               /*reset=*/std::nullopt, state_register.reg));
      result_.output_states[stage].push_back(index);
      result_.node_to_stage_map[state_register.reg_write] = stage;
    } else if (!param->GetType()->IsToken() &&
               param->GetType() != proc->package()->GetTupleType({})) {
      return absl::UnimplementedError(
          absl::StrFormat("Proc has zero-width state element %d, but type is "
                          "not token or empty tuple, instead got %s.",
                          index, node->GetType()->ToString()));
    }

    // If the next state can be determined in a later cycle than the param
    // access, we have a non-trivial backedge between initiations (II>1); use a
    // "full" bit to track whether the state is currently valid.
    //
    // TODO(epastor): Consider an optimization that merges the "full" bits for
    // all states with the same read stage & matching write stages/predicates...
    // or maybe a more general optimization that merges registers with identical
    // type, input, and load-enable values.
    if (stage > state_register.read_stage) {
      XLS_ASSIGN_OR_RETURN(
          state_register.reg_full,
          block()->AddRegister(absl::StrCat("__", state_register.name, "_full"),
                               block()->package()->GetBitsType(1)));
      XLS_ASSIGN_OR_RETURN(state_register.reg_full_read,
                           block()->MakeNodeWithName<RegisterRead>(
                               next->loc(), state_register.reg_full,
                               /*name=*/state_register.reg_full->name()));
      XLS_ASSIGN_OR_RETURN(
          Node * literal_1,
          block()->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
      XLS_ASSIGN_OR_RETURN(
          state_register.reg_full_write,
          block()->MakeNode<RegisterWrite>(next->loc(), literal_1,
                                           /*load_enable=*/std::nullopt,
                                           /*reset=*/std::nullopt,
                                           state_register.reg_full));
    }

    return absl::OkStatus();
  }

  // Don't clone Receive operations. Instead replace with a tuple
  // containing the Receive's token operand and an InputPort operation.
  //
  // Both data and valid ports are created in this function.  See
  // MakeInputValidPortsForInputChannels() for additional handling of
  // the valid signal.
  //
  // In the case of handling non-blocking receives, the logic to adapt
  // data to a tuple of (data, valid) is added here.
  absl::StatusOr<Node*> HandleReceiveNode(Node* node, int64_t stage) {
    Node* next_node;

    Receive* receive = node->As<Receive>();
    XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
    std::string_view data_suffix =
        (channel->kind() == ChannelKind::kStreaming)
            ? options_.streaming_channel_data_suffix()
            : "";
    XLS_ASSIGN_OR_RETURN(
        InputPort * input_port,
        block()->AddInputPort(absl::StrCat(channel->name(), data_suffix),
                              channel->type()));

    if (std::optional<PackageInterfaceProto::Channel> c =
            FindChannelInterface(options_.package_interface(), channel->name());
        c && c->has_sv_type()) {
      result_.input_port_sv_type[input_port] = c->sv_type();
    }

    XLS_ASSIGN_OR_RETURN(
        Node * literal_1,
        block()->MakeNode<xls::Literal>(node->loc(), Value(UBits(1, 1))));

    if (channel->kind() == ChannelKind::kSingleValue) {
      if (receive->is_blocking()) {
        XLS_ASSIGN_OR_RETURN(
            next_node,
            block()->MakeNode<Tuple>(
                node->loc(), std::vector<Node*>({node_map_.at(node->operand(0)),
                                                 input_port})));
      } else {
        XLS_ASSIGN_OR_RETURN(
            next_node,
            block()->MakeNode<Tuple>(
                node->loc(), std::vector<Node*>({node_map_.at(node->operand(0)),
                                                 input_port, literal_1})));
      }

      result_.single_value_inputs.push_back(
          SingleValueInput{.port = input_port, .channel = channel});

      return next_node;
    }

    XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kStreaming);
    XLS_RET_CHECK_EQ(down_cast<StreamingChannel*>(channel)->GetFlowControl(),
                     FlowControl::kReadyValid);

    // Construct the valid port.
    std::string_view valid_suffix = options_.streaming_channel_valid_suffix();

    XLS_ASSIGN_OR_RETURN(
        InputPort * input_valid_port,
        block()->AddInputPort(absl::StrCat(channel->name(), valid_suffix),
                              block()->package()->GetBitsType(1)));

    // If blocking return a tuple of (token, data), and if non-blocking
    // return a tuple of (token, data, valid).
    if (receive->is_blocking()) {
      Node* data = input_port;
      if (receive->predicate().has_value() && options_.gate_recvs()) {
        XLS_ASSIGN_OR_RETURN(
            Node * zero_value,
            block()->MakeNode<xls::Literal>(node->loc(),
                                            ZeroOfType(input_port->GetType())));
        XLS_ASSIGN_OR_RETURN(
            Select * select,
            block()->MakeNodeWithName<Select>(
                /*loc=*/node->loc(),
                /*selector=*/node_map_.at(receive->predicate().value()),
                /*cases=*/std::vector<Node*>({zero_value, input_port}),
                /*default_value=*/std::nullopt,
                /*name=*/absl::StrCat(channel->name(), "_select")));
        data = select;
      }
      XLS_ASSIGN_OR_RETURN(
          next_node,
          block()->MakeNode<Tuple>(
              node->loc(),
              std::vector<Node*>({node_map_.at(node->operand(0)), data})));
    } else {
      XLS_ASSIGN_OR_RETURN(Node * zero_value,
                           block()->MakeNode<xls::Literal>(
                               node->loc(), ZeroOfType(input_port->GetType())));
      // Ensure that the output of the receive is zero when the data is not
      // valid or the predicate is false.
      Node* valid = input_valid_port;
      Node* data = input_port;
      if (options_.gate_recvs()) {
        if (receive->predicate().has_value()) {
          XLS_ASSIGN_OR_RETURN(
              NaryOp * and_pred,
              block()->MakeNode<NaryOp>(
                  /*loc=*/node->loc(),
                  /*args=*/
                  std::vector<Node*>(
                      {node_map_.at(receive->predicate().value()),
                       input_valid_port}),
                  /*op=*/Op::kAnd));
          valid = and_pred;
        }
        XLS_ASSIGN_OR_RETURN(
            Select * select,
            block()->MakeNodeWithName<Select>(
                /*loc=*/node->loc(), /*selector=*/valid,
                /*cases=*/std::vector<Node*>({zero_value, input_port}),
                /*default_value=*/std::nullopt,
                /*name=*/absl::StrCat(channel->name(), "_select")));
        data = select;
      }
      XLS_ASSIGN_OR_RETURN(
          next_node,
          block()->MakeNode<Tuple>(
              node->loc(), std::vector<Node*>(
                               {node_map_.at(node->operand(0)), data, valid})));
    }

    // To the rest of the logic, a non-blocking receive is always valid.
    Node* signal_valid = receive->is_blocking() ? input_valid_port : literal_1;

    StreamingInput streaming_input{.port = input_port,
                                   .port_valid = input_valid_port,
                                   .port_ready = nullptr,
                                   .signal_data = next_node,
                                   .signal_valid = signal_valid,
                                   .channel = channel};

    if (receive->predicate().has_value()) {
      streaming_input.predicate = node_map_.at(receive->predicate().value());
    }
    result_.inputs[stage].push_back(streaming_input);

    return next_node;
  }

  // Don't clone Send operations. Instead replace with an OutputPort
  // operation in the block.
  absl::StatusOr<Node*> HandleSendNode(Node* node, int64_t stage) {
    Node* next_node;

    XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
    Send* send = node->As<Send>();
    std::string_view data_suffix =
        (channel->kind() == ChannelKind::kStreaming)
            ? options_.streaming_channel_data_suffix()
            : "";
    XLS_ASSIGN_OR_RETURN(
        OutputPort * output_port,
        block()->AddOutputPort(absl::StrCat(channel->name(), data_suffix),
                               node_map_.at(send->data())));

    if (std::optional<PackageInterfaceProto::Channel> c =
            FindChannelInterface(options_.package_interface(), channel->name());
        c && c->has_sv_type()) {
      result_.output_port_sv_type[output_port] = c->sv_type();
    }
    // Map the Send node to the token operand of the Send in the
    // block.
    next_node = node_map_.at(send->token());

    XLS_ASSIGN_OR_RETURN(
        Node * token_buf,
        block()->MakeNode<UnOp>(
            /*loc=*/SourceInfo(), node_map_.at(send->token()), Op::kIdentity));
    next_node = token_buf;

    if (channel->kind() == ChannelKind::kSingleValue) {
      result_.single_value_outputs.push_back(
          SingleValueOutput{.port = output_port, .channel = channel});
      return next_node;
    }

    XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kStreaming);
    XLS_RET_CHECK_EQ(down_cast<StreamingChannel*>(channel)->GetFlowControl(),
                     FlowControl::kReadyValid);

    StreamingOutput streaming_output{.port = output_port,
                                     .port_valid = nullptr,
                                     .port_ready = nullptr,
                                     .channel = channel};

    if (send->predicate().has_value()) {
      streaming_output.predicate = node_map_.at(send->predicate().value());
    }

    result_.outputs[stage].push_back(streaming_output);

    return next_node;
  }

  absl::StatusOr<Node*> HandleFifoReceiveNode(
      Receive* receive, int64_t stage, FifoInstantiation* fifo_instantiation) {
    XLS_ASSIGN_OR_RETURN(Node * data,
                         block()->MakeNode<xls::InstantiationOutput>(
                             receive->loc(), fifo_instantiation, "pop_data"));
    XLS_ASSIGN_OR_RETURN(Node * valid,
                         block()->MakeNode<xls::InstantiationOutput>(
                             receive->loc(), fifo_instantiation, "pop_valid"));
    XLS_ASSIGN_OR_RETURN(Channel * channel, block()->package()->GetChannel(
                                                receive->channel_name()));
    Node* signal_valid;
    if (receive->is_blocking()) {
      signal_valid = valid;
    } else {
      XLS_ASSIGN_OR_RETURN(
          Node * literal_1,
          block()->MakeNode<xls::Literal>(receive->loc(), Value(UBits(1, 1))));
      signal_valid = literal_1;
    }
    StreamingInput streaming_input{.port = data,
                                   .port_valid = valid,
                                   .port_ready = nullptr,
                                   .signal_data = data,
                                   .signal_valid = signal_valid,
                                   .channel = channel,
                                   .fifo_instantiation = fifo_instantiation};

    if (receive->predicate().has_value()) {
      streaming_input.predicate = node_map_.at(receive->predicate().value());
    }
    result_.inputs[stage].push_back(streaming_input);
    Node* next_token = node_map_.at(receive->token());
    const SourceInfo& loc = receive->loc();
    Node* next_node;
    // If blocking return a tuple of (token, data), and if non-blocking
    // return a tuple of (token, data, valid).
    if (receive->is_blocking()) {
      if (receive->predicate().has_value() && options_.gate_recvs()) {
        XLS_ASSIGN_OR_RETURN(
            Node * zero_value,
            block()->MakeNode<xls::Literal>(loc, ZeroOfType(channel->type())));
        XLS_ASSIGN_OR_RETURN(
            Select * select,
            block()->MakeNodeWithName<Select>(
                /*loc=*/loc,
                /*selector=*/node_map_.at(receive->predicate().value()),
                /*cases=*/std::vector<Node*>({zero_value, data}),
                /*default_value=*/std::nullopt,
                /*name=*/absl::StrCat(channel->name(), "_select")));
        data = select;
      }
      XLS_ASSIGN_OR_RETURN(next_node,
                           block()->MakeNode<Tuple>(
                               loc, std::vector<Node*>({next_token, data})));
    } else {
      // Receive is non-blocking; we need a zero value to pass through if there
      // is no valid data.
      XLS_ASSIGN_OR_RETURN(
          Node * zero_value,
          block()->MakeNode<xls::Literal>(loc, ZeroOfType(channel->type())));
      // Ensure that the output of the receive is zero when the data is not
      // valid or the predicate is false.
      if (options_.gate_recvs()) {
        if (receive->predicate().has_value()) {
          XLS_ASSIGN_OR_RETURN(
              NaryOp * and_pred,
              block()->MakeNode<NaryOp>(
                  /*loc=*/loc,
                  /*args=*/
                  std::initializer_list<Node*>{
                      node_map_.at(receive->predicate().value()), valid},
                  /*op=*/Op::kAnd));
          valid = and_pred;
        }
        XLS_ASSIGN_OR_RETURN(
            Select * select,
            block()->MakeNodeWithName<Select>(
                /*loc=*/loc, /*selector=*/valid,
                /*cases=*/
                std::initializer_list<Node*>({zero_value, data}),
                /*default_value=*/std::nullopt,
                /*name=*/absl::StrCat(channel->name(), "_select")));
        data = select;
      }
      XLS_ASSIGN_OR_RETURN(
          next_node,
          block()->MakeNode<Tuple>(
              loc, std::initializer_list<Node*>({next_token, data, valid})));
    }
    return next_node;
  }

  absl::StatusOr<Node*> HandleFifoSendNode(
      Send* send, int64_t stage, FifoInstantiation* fifo_instantiation) {
    XLS_ASSIGN_OR_RETURN(Node * ready,
                         block()->MakeNode<xls::InstantiationOutput>(
                             send->loc(), fifo_instantiation, "push_ready"));
    XLS_ASSIGN_OR_RETURN(Channel * channel,
                         block()->package()->GetChannel(send->channel_name()));
    Node* data = node_map_.at(send->data());
    XLS_ASSIGN_OR_RETURN(
        Node * port, block()->MakeNode<xls::InstantiationInput>(
                         send->loc(), data, fifo_instantiation, "push_data"));
    StreamingOutput streaming_output{.port = port,
                                     .port_valid = nullptr,
                                     .port_ready = ready,
                                     .channel = channel,
                                     .fifo_instantiation = fifo_instantiation};

    if (send->predicate().has_value()) {
      streaming_output.predicate = node_map_.at(send->predicate().value());
    }
    result_.outputs[stage].push_back(streaming_output);
    // Map the Send node to the token operand of the Send in the block.
    XLS_ASSIGN_OR_RETURN(
        Node * token_buf,
        block()->MakeNode<UnOp>(
            /*loc=*/SourceInfo(), node_map_.at(send->token()), Op::kIdentity));
    return token_buf;
  }

  // Clone the operation from the source to the block as is.
  absl::StatusOr<Node*> HandleGeneralNode(Node* node) {
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map_.at(operand));
    }
    return node->CloneInNewFunction(new_operands, block());
  }

  // Create a pipeline register for the given node.
  //
  // Returns a PipelineRegister whose reg_read field can be used
  // to chain dependent ops to.
  absl::StatusOr<PipelineRegister> CreatePipelineRegister(std::string_view name,
                                                          Node* node,
                                                          Stage stage_write) {
    XLS_ASSIGN_OR_RETURN(Register * reg,
                         block()->AddRegister(name, node->GetType()));
    XLS_ASSIGN_OR_RETURN(
        RegisterWrite * reg_write,
        block()->MakeNode<RegisterWrite>(node->loc(), node,
                                         /*load_enable=*/std::nullopt,
                                         /*reset=*/std::nullopt, reg));
    XLS_ASSIGN_OR_RETURN(
        RegisterRead * reg_read,
        block()->MakeNodeWithName<RegisterRead>(node->loc(), reg,
                                                /*name=*/reg->name()));
    result_.node_to_stage_map[reg_write] = stage_write;
    result_.node_to_stage_map[reg_read] = stage_write + 1;
    return PipelineRegister{reg, reg_write, reg_read};
  }

  // Returns true if tuple_type has a zero width element at the top level.
  bool HasZeroWidthType(TupleType* tuple_type) {
    CHECK(tuple_type != nullptr);

    for (Type* element_type : tuple_type->element_types()) {
      if (element_type->GetFlatBitCount() == 0) {
        return true;
      }
    }

    return false;
  }

  // Creates pipeline registers for a given node.
  //
  // Depending on the type of node, multiple pipeline registers
  // may be created.
  //  1. Each pipeline register added will be added to pipeline_register_list
  //     which is passed in by reference.
  //  2. Logic may be inserted after said registers  so that a single node with
  //     the same type as the input node is returned.
  //
  absl::StatusOr<Node*> CreatePipelineRegistersForNode(
      std::string_view base_name, Node* node, Stage stage,
      std::vector<PipelineRegister>& pipeline_registers_list) {
    // As a special case, check if the node is a tuple
    // containing types that are of zero-width.  If so, separate them out so
    // that future optimization passes can remove them.
    //
    // Note that for nested tuples, only the first level will be split,
    // any nested tuple will remain as a tuple.
    Type* node_type = node->GetType();
    if (node_type->IsTuple()) {
      TupleType* tuple_type = node_type->AsTupleOrDie();

      if (HasZeroWidthType(tuple_type)) {
        std::vector<Node*> split_registers(tuple_type->size());

        // Create registers for each element.
        for (int64_t i = 0; i < split_registers.size(); ++i) {
          XLS_ASSIGN_OR_RETURN(Node * split_node, block()->MakeNode<TupleIndex>(
                                                      node->loc(), node, i));

          XLS_ASSIGN_OR_RETURN(PipelineRegister pipe_reg,
                               CreatePipelineRegister(
                                   absl::StrFormat("%s_index%d", base_name, i),
                                   split_node, stage));

          split_registers.at(i) = pipe_reg.reg_read;
          pipeline_registers_list.push_back(pipe_reg);
        }

        // Reconstruct tuple for the rest of the graph.
        XLS_ASSIGN_OR_RETURN(
            Node * merge_after_reg_read,
            block()->MakeNode<Tuple>(node->loc(), split_registers));

        return merge_after_reg_read;
      }
    }

    // Create a single register to store the node
    XLS_ASSIGN_OR_RETURN(PipelineRegister pipe_reg,
                         CreatePipelineRegister(base_name, node, stage));

    pipeline_registers_list.push_back(pipe_reg);
    return pipe_reg.reg_read;
  }

  Block* block() const { return block_; };

  bool is_proc_;
  FunctionBase* function_base_;

  const CodegenOptions& options_;

  Block* block_;
  std::optional<ConcurrentStageGroups> concurrent_stages_;
  StreamingIOPipeline result_;
  absl::flat_hash_map<Node*, Node*> node_map_;
  absl::flat_hash_set<int64_t> loopback_channel_ids_;
  absl::flat_hash_map<int64_t, xls::Instantiation*> fifo_instantiations_;
};

// Adds the nodes in the given schedule to the block. Pipeline registers are
// inserted between stages and returned as a vector indexed by cycle. The block
// should be empty prior to calling this function.
//
// Returns the resulting pipeline and concurrent stages.
// TODO google/xls#1324 and google/xls#1300: ideally this wouldn't need to
// return so much and more of this could be done later or stored directly in the
// IR.
static absl::StatusOr<
    std::tuple<StreamingIOPipeline, std::optional<ConcurrentStageGroups>>>
CloneNodesIntoPipelinedBlock(const PipelineSchedule& schedule,
                             const CodegenOptions& options, Block* block) {
  FunctionBase* function_base = schedule.function_base();
  XLS_RET_CHECK(function_base->IsProc() || function_base->IsFunction());

  CloneNodesIntoBlockHandler cloner(function_base, schedule.length(), options,
                                    block);
  for (int64_t stage = 0; stage < schedule.length(); ++stage) {
    XLS_RET_CHECK_OK(cloner.CloneNodes(schedule.nodes_in_cycle(stage), stage));
    XLS_RET_CHECK_OK(cloner.AddNextPipelineStage(schedule, stage));
  }

  XLS_RET_CHECK_OK(cloner.AddOutputPortsIfFunction(options.output_port_name()));
  XLS_RET_CHECK_OK(cloner.MarkMutualExclusiveStages(schedule.length()));

  return std::make_tuple(cloner.GetResult(), cloner.GetConcurrentStages());
}

// Clones every node in the given proc into the given block. Some nodes are
// handled specially.  See CloneNodesIntoBlockHandler for details.
static absl::StatusOr<StreamingIOPipeline> CloneProcNodesIntoBlock(
    Proc* proc, const CodegenOptions& options, Block* block) {
  CloneNodesIntoBlockHandler cloner(proc, /*stage_count=*/0, options, block);
  XLS_RET_CHECK_OK(cloner.CloneNodes(TopoSort(proc), /*stage=*/0));
  return cloner.GetResult();
}

absl::StatusOr<CodegenPassUnit> FunctionToPipelinedBlock(
    const PipelineSchedule& schedule, const CodegenOptions& options,
    Function* f, CodegenPassUnit& unit) {
  if (options.manual_control().has_value()) {
    return absl::UnimplementedError("Manual pipeline control not implemented");
  }
  if (options.split_outputs()) {
    return absl::UnimplementedError("Splitting outputs not supported.");
  }
  if (options.reset().has_value() && options.reset()->reset_data_path()) {
    return absl::UnimplementedError("Data path reset not supported");
  }
  if (options.manual_control().has_value()) {
    return absl::UnimplementedError("Manual pipeline control not implemented");
  }

  if (std::optional<int64_t> ii = f->GetInitiationInterval(); ii.has_value()) {
    unit.top_block->SetInitiationInterval(*ii);
  }

  if (!options.clock_name().has_value()) {
    return absl::InvalidArgumentError(
        "Clock name must be specified when generating a pipelined block");
  }
  XLS_RETURN_IF_ERROR(
      unit.top_block->AddClockPort(options.clock_name().value()));

  // Flopping inputs and outputs can be handled as a transformation to the
  // schedule. This makes the later code for creation of the pipeline simpler.
  // TODO(meheff): 2021/7/21 Add input/output flopping as an option to the
  // scheduler.
  XLS_ASSIGN_OR_RETURN(PipelineSchedule transformed_schedule,
                       MaybeAddInputOutputFlopsToSchedule(schedule, options));

  XLS_ASSIGN_OR_RETURN((auto [streaming_io_and_pipeline, concurrent_stages]),
                       CloneNodesIntoPipelinedBlock(transformed_schedule,
                                                    options, unit.top_block));

  XLS_RET_CHECK_OK(MaybeAddResetPort(unit.top_block, options));

  FunctionConversionMetadata function_metadata;
  if (options.valid_control().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        function_metadata.valid_ports,
        AddValidSignal(
            absl::MakeSpan(streaming_io_and_pipeline.pipeline_registers),
            options, unit.top_block, streaming_io_and_pipeline.pipeline_valid,
            streaming_io_and_pipeline.node_to_stage_map));
  }

  // Reorder the ports of the block to the following:
  //   - clk
  //   - reset (optional)
  //   - input valid (optional)
  //   - function inputs
  //   - output valid (optional)
  //   - function output
  // This is solely a cosmetic change to improve readability.
  std::vector<std::string> port_order;
  port_order.push_back(std::string{options.clock_name().value()});
  if (unit.top_block->GetResetPort().has_value()) {
    port_order.push_back(unit.top_block->GetResetPort().value()->GetName());
  }
  if (function_metadata.valid_ports.has_value()) {
    port_order.push_back(function_metadata.valid_ports->input->GetName());
  }
  for (Param* param : f->params()) {
    port_order.push_back(param->GetName());
  }
  if (function_metadata.valid_ports.has_value() &&
      function_metadata.valid_ports->output != nullptr) {
    port_order.push_back(function_metadata.valid_ports->output->GetName());
  }
  port_order.push_back(std::string{options.output_port_name()});
  XLS_RETURN_IF_ERROR(unit.top_block->ReorderPorts(port_order));

  unit.metadata[unit.top_block] = CodegenMetadata{
      .streaming_io_and_pipeline = std::move(streaming_io_and_pipeline),
      .conversion_metadata = function_metadata,
      .concurrent_stages = std::move(concurrent_stages),
  };

  return unit;
}

absl::Status SingleProcToPipelinedBlock(const PipelineSchedule& schedule,
                                        const CodegenOptions& options,
                                        CodegenPassUnit& unit, Proc* proc,
                                        absl::Nonnull<Block*> block) {
  XLS_RET_CHECK_EQ(schedule.function_base(), proc);
  if (std::optional<int64_t> ii = proc->GetInitiationInterval();
      ii.has_value()) {
    block->SetInitiationInterval(*ii);
  }

  XLS_RETURN_IF_ERROR(block->AddClockPort("clk"));
  VLOG(3) << "Schedule Used";
  XLS_VLOG_LINES(3, schedule.ToString());

  XLS_ASSIGN_OR_RETURN((auto [streaming_io_and_pipeline, concurrent_stages]),
                       CloneNodesIntoPipelinedBlock(schedule, options, block));

  VLOG(3) << "After Pipeline";
  XLS_VLOG_LINES(3, block->DumpIr());

  int64_t number_of_outputs = 0;
  for (const auto& outputs : streaming_io_and_pipeline.outputs) {
    number_of_outputs += outputs.size();
  }

  bool streaming_outputs_mutually_exclusive = true;
  if (number_of_outputs > 1) {
    // TODO: do this analysis on a per-stage basis
    XLS_ASSIGN_OR_RETURN(streaming_outputs_mutually_exclusive,
                         AreStreamingOutputsMutuallyExclusive(proc));

    if (streaming_outputs_mutually_exclusive) {
      VLOG(3) << absl::StrFormat(
          "%d streaming outputs determined to be mutually exclusive",
          streaming_io_and_pipeline.outputs.size());
    } else {
      VLOG(3) << absl::StrFormat(
          "%d streaming outputs not proven to be mutually exclusive -- "
          "assuming false",
          streaming_io_and_pipeline.outputs.size());
    }
  }

  VLOG(3) << "After Pipeline";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_RET_CHECK_OK(MaybeAddResetPort(block, options));

  // Initialize the valid flops to the pipeline registers.
  // First element is skipped as the initial stage valid flops will be
  // constructed from the input flops and/or input valid ports and will be
  // added later in AddInputFlops() and AddIdleOutput().
  ProcConversionMetadata proc_metadata;
  std::vector<Node*> all_active_outputs_ready;
  XLS_RETURN_IF_ERROR(AddBubbleFlowControl(
      options, streaming_io_and_pipeline, block,
      /*all_active_outputs_ready=*/all_active_outputs_ready));
  CHECK_GE(streaming_io_and_pipeline.stage_valid.size(), 1);
  std::copy(streaming_io_and_pipeline.stage_valid.begin() + 1,
            streaming_io_and_pipeline.stage_valid.end(),
            std::back_inserter(proc_metadata.valid_flops));

  VLOG(3) << "After Flow Control";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (!streaming_outputs_mutually_exclusive) {
    XLS_RETURN_IF_ERROR(AddOneShotOutputLogic(
        options, streaming_io_and_pipeline,
        /*all_active_outputs_ready=*/all_active_outputs_ready, block));
  }
  VLOG(3) << absl::StrFormat("After Output Triggers");
  XLS_VLOG_LINES(3, block->DumpIr());

  if (options.flop_inputs() || options.flop_outputs()) {
    XLS_RETURN_IF_ERROR(AddInputOutputFlops(options, streaming_io_and_pipeline,
                                            block, proc_metadata.valid_flops));
  }
  VLOG(3) << "After Input or Output Flops";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (options.add_idle_output()) {
    XLS_RETURN_IF_ERROR(AddIdleOutput(proc_metadata.valid_flops,
                                      streaming_io_and_pipeline, block));
  }
  VLOG(3) << "After Add Idle Output";
  XLS_VLOG_LINES(3, block->DumpIr());

  // RemoveDeadTokenNodes() mutates metadata.
  unit.metadata[block] = CodegenMetadata{
      .streaming_io_and_pipeline = std::move(streaming_io_and_pipeline),
      .conversion_metadata = std::move(proc_metadata),
      .concurrent_stages = std::move(concurrent_stages),
  };

  // TODO(tedhong): 2021-09-23 Remove and add any missing functionality to
  //                codegen pipeline.
  XLS_RETURN_IF_ERROR(RemoveDeadTokenNodes(&unit));

  VLOG(3) << "After RemoveDeadTokenNodes";
  XLS_VLOG_LINES(3, block->DumpIr());

  // TODO: add simplification pass here to remove unnecessary `1 & x`

  XLS_RETURN_IF_ERROR(UpdateChannelMetadata(
      unit.metadata[block].streaming_io_and_pipeline, block));
  VLOG(3) << "After UpdateChannelMetadata";
  XLS_VLOG_LINES(3, block->DumpIr());

  return absl::OkStatus();
}
}  // namespace

absl::StatusOr<PipelineSchedule> MaybeAddInputOutputFlopsToSchedule(
    const PipelineSchedule& schedule, const CodegenOptions& options) {
  // All function params must be scheduled in cycle 0.
  ScheduleCycleMap cycle_map;
  if (schedule.function_base()->IsFunction()) {
    for (Param* param : schedule.function_base()->params()) {
      XLS_RET_CHECK_EQ(schedule.cycle(param), 0);
      cycle_map[param] = 0;
    }
  }

  // If `flop_inputs` is true, adjust the cycle of all remaining nodes by one.
  int64_t cycle_offset = 0;
  if (options.flop_inputs()) {
    ++cycle_offset;
  }
  for (int64_t cycle = 0; cycle < schedule.length(); ++cycle) {
    for (Node* node : schedule.nodes_in_cycle(cycle)) {
      if (node->Is<Param>()) {
        continue;
      }
      cycle_map[node] = cycle + cycle_offset;
    }
  }

  // Add one more cycle to the schedule if `flop_outputs` is true. This only
  // changes the length of the schedule not the cycle that any node is placed
  // in. The final cycle is empty which effectively puts a pipeline register
  // between the nodes of the function and the output ports.
  if (options.flop_outputs()) {
    ++cycle_offset;
  }
  PipelineSchedule result(schedule.function_base(), cycle_map,
                          schedule.length() + cycle_offset);
  return std::move(result);
}

// Adds a register between the node and all its downstream users.
// Returns the new register added.
absl::StatusOr<RegisterRead*> AddRegisterAfterNode(
    std::string_view name_prefix,
    const std::optional<xls::Reset>& reset_behavior,
    std::optional<Node*> load_enable, Node* node, Block* block) {
  XLS_RET_CHECK(reset_behavior.has_value());
  XLS_RET_CHECK(block->GetResetPort().has_value());

  Type* node_type = node->GetType();

  if (!reset_behavior.has_value()) {
    return absl::InvalidArgumentError(
        "Reset signal required but not specified.");
  }

  // Update reset_behavior to have a reset value of the right type.
  xls::Reset reset_behavior_for_type = *reset_behavior;
  reset_behavior_for_type.reset_value = ZeroOfType(node_type);

  std::string name = absl::StrFormat("__%s_reg", name_prefix);

  XLS_ASSIGN_OR_RETURN(
      Register * reg,
      block->AddRegister(name, node_type, reset_behavior_for_type));

  XLS_ASSIGN_OR_RETURN(RegisterRead * reg_read,
                       block->MakeNodeWithName<RegisterRead>(
                           /*loc=*/node->loc(),
                           /*reg=*/reg,
                           /*name=*/name));

  XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(reg_read));

  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<RegisterWrite>(
                              /*loc=*/node->loc(),
                              /*data=*/node,
                              /*load_enable=*/load_enable,
                              /*reset=*/block->GetResetPort().value(),
                              /*reg=*/reg)
                          .status());

  return reg_read;
}

// Add a zero-latency buffer after a set of data/valid/ready signal.
//
// Logic will be inserted immediately after from_data and from node.
// Logic will be inserted immediately before from_rdy,
//   from_rdy must be a node with a single operand.
//
// Updates valid_nodes with the additional nodes associated with valid
// registers.
absl::StatusOr<Node*> AddZeroLatencyBufferToRDVNodes(
    Node* from_data, Node* from_valid, Node* from_rdy,
    std::string_view name_prefix,
    const std::optional<xls::Reset>& reset_behavior, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  CHECK_EQ(from_rdy->operand_count(), 1);

  // Add a node for load_enables (will be removed later).
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  // Create data/valid and their skid counterparts.
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_skid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_skid"), reset_behavior,
                           literal_1, from_data, block));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_valid_skid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_valid_skid"),
                           reset_behavior, literal_1, from_valid, block));

  // If data_valid_skid_reg_read is 1, then data/valid outputs should
  // be selected from the skid set.
  XLS_ASSIGN_OR_RETURN(
      Node * to_valid,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/from_data->loc(),
          std::vector<Node*>{from_valid, data_valid_skid_reg_read}, Op::kOr,
          absl::StrCat(name_prefix, "_valid_or")));
  XLS_RETURN_IF_ERROR(data_valid_skid_reg_read->ReplaceUsesWith(to_valid));

  XLS_ASSIGN_OR_RETURN(
      Node * to_data,
      block->MakeNodeWithName<Select>(
          /*loc=*/from_data->loc(),
          /*selector=*/data_valid_skid_reg_read,
          /*cases=*/std::vector<Node*>{from_data, data_skid_reg_read},
          /*default_value=*/std::nullopt,
          /*name=*/absl::StrCat(name_prefix, "_select")));
  XLS_RETURN_IF_ERROR(data_skid_reg_read->ReplaceUsesWith(to_data));

  // Input can be accepted whenever the skid registers
  // are empty/invalid.
  Node* to_is_ready = from_rdy->operand(0);

  XLS_ASSIGN_OR_RETURN(
      Node * from_skid_rdy,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), data_valid_skid_reg_read, Op::kNot,
          absl::StrCat(name_prefix, "_from_skid_rdy")));
  XLS_RETURN_IF_ERROR(from_rdy->ReplaceOperandNumber(0, from_skid_rdy));

  // Skid is loaded from 1st stage whenever
  //   a) the input is being read (input_ready_and_valid == 1) and
  //       --> which implies that the skid is invalid
  //   b) the output is not ready (to_is_ready == 0) and
  XLS_ASSIGN_OR_RETURN(Node * to_is_not_rdy,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), to_is_ready, Op::kNot,
                           absl::StrCat(name_prefix, "_to_is_not_rdy")));

  XLS_ASSIGN_OR_RETURN(
      Node * skid_data_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{from_valid, from_skid_rdy, to_is_not_rdy},
          Op::kAnd, absl::StrCat(name_prefix, "_skid_data_load_en")));

  // Skid is reset (valid set to zero) to invalid whenever
  //   a) skid is valid and
  //   b) output is ready
  XLS_ASSIGN_OR_RETURN(
      Node * skid_valid_set_zero,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{data_valid_skid_reg_read, to_is_ready}, Op::kAnd,
          absl::StrCat(name_prefix, "_skid_valid_set_zero")));

  // Skid valid changes from 0 to 1 (load), or 1 to 0 (set zero).
  XLS_ASSIGN_OR_RETURN(
      Node * skid_valid_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(),
          std::vector<Node*>{skid_data_load_en, skid_valid_set_zero}, Op::kOr,
          absl::StrCat(name_prefix, "_skid_valid_load_en")));

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * data_skid_reg_write,
      block->GetRegisterWrite(data_skid_reg_read->GetRegister()));
  XLS_RETURN_IF_ERROR(
      data_skid_reg_write->ReplaceExistingLoadEnable(skid_data_load_en));

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * data_valid_skid_reg_write,
      block->GetRegisterWrite(data_valid_skid_reg_read->GetRegister()));

  // If the skid valid is being set
  //   - If it's being set to 1, then the input is being read,
  //     and the prior data is being stored into the skid
  //   - If it's being set to 0, then the input is not being read
  //     and we are clearing the skid and sending the data to the output
  // this implies that
  //   skid_valid := skid_valid_load_en ? !skid_valid : skid_valid
  XLS_RETURN_IF_ERROR(
      data_valid_skid_reg_write->ReplaceExistingLoadEnable(skid_valid_load_en));
  XLS_RETURN_IF_ERROR(
      data_valid_skid_reg_write->ReplaceOperandNumber(0, from_skid_rdy));

  valid_nodes.push_back(to_valid);

  return to_data;
}

std::string PipelineSignalName(std::string_view root, int64_t stage) {
  std::string base;
  // Strip any existing pipeline prefix from the name.
  if (!RE2::PartialMatch(root, R"(^p\d+_(.+))", &base)) {
    base = root;
  }
  return absl::StrFormat("p%d_%s", stage, SanitizeIdentifier(base));
}

absl::StatusOr<CodegenPassUnit> PackageToPipelinedBlocks(
    const PackagePipelineSchedules& schedules, const CodegenOptions& options,
    Package* package) {
  XLS_RET_CHECK_GT(schedules.size(), 0);
  VLOG(3) << "Converting package to pipelined blocks:";
  XLS_VLOG_LINES(3, package->DumpIr());

  if (options.manual_control().has_value()) {
    return absl::UnimplementedError("Manual pipeline control not implemented");
  }
  if (options.split_outputs()) {
    return absl::UnimplementedError("Splitting outputs not supported.");
  }

  XLS_RET_CHECK(package->GetTop().has_value());
  FunctionBase* top = *package->GetTop();
  absl::btree_map<FunctionBase*, PipelineSchedule, FunctionBase::NameLessThan>
      sorted_schedules(schedules.begin(), schedules.end());
  // Make `unit` optional because we haven't created the top block yet. We will
  // create it on the first iteration and emplace `unit`.
  std::string module_name(
      SanitizeIdentifier(options.module_name().value_or(top->name())));
  Block* top_block =
      package->AddBlock(std::make_unique<Block>(module_name, package));
  // We use a uniquer here because the top block name comes from the codegen
  // option's `module_name` field (if set). A non-top proc could have the same
  // name, so the name uniquer will ensure that the sub-block gets a suffix if
  // needed. Note that the NameUniquer's sanitize performs a different function
  // from `SanitizeIdentifier()`, which is used to ensure that identifiers are
  // OK for RTL.
  NameUniquer block_name_uniquer("__");
  XLS_RET_CHECK_EQ(block_name_uniquer.GetSanitizedUniqueName(module_name),
                   module_name);
  CodegenPassUnit unit(package, top_block);

  for (const auto& [fb, schedule] : sorted_schedules) {
    std::string sub_block_name = block_name_uniquer.GetSanitizedUniqueName(
        SanitizeIdentifier(fb->name()));
    Block* sub_block;
    if (fb == top) {
      sub_block = top_block;
    } else {
      sub_block =
          package->AddBlock(std::make_unique<Block>(sub_block_name, package));
    }
    if (fb->IsProc()) {
      XLS_RETURN_IF_ERROR(SingleProcToPipelinedBlock(
          schedule, options, unit, fb->AsProcOrDie(), sub_block));
    } else if (fb->IsFunction()) {
      XLS_RET_CHECK_EQ(sorted_schedules.size(), 1);
      XLS_RET_CHECK_EQ(fb, top);
      XLS_RETURN_IF_ERROR(FunctionToPipelinedBlock(schedule, options,
                                                   fb->AsFunctionOrDie(), unit)
                              .status());
    } else {
      return absl::InvalidArgumentError(absl::StrFormat(
          "FunctionBase %s was not a function or proc.", fb->name()));
    }
  }

  // Avoid leaving any dangling pointers.
  unit.GcMetadata();

  return unit;
}

absl::StatusOr<CodegenPassUnit> FunctionBaseToPipelinedBlock(
    const PipelineSchedule& schedule, const CodegenOptions& options,
    FunctionBase* f) {
  PackagePipelineSchedules schedules{{f, schedule}};
  std::optional<FunctionBase*> old_top = f->package()->GetTop();
  XLS_RETURN_IF_ERROR(f->package()->SetTop(f));

  // Don't return yet if there's an error- we need to restore old_top first.
  absl::StatusOr<CodegenPassUnit> unit_or_status =
      PackageToPipelinedBlocks(schedules, options, f->package());
  XLS_RETURN_IF_ERROR(f->package()->SetTop(old_top));
  return unit_or_status;
}

absl::StatusOr<CodegenPassUnit> FunctionToCombinationalBlock(
    Function* f, const CodegenOptions& options) {
  XLS_RET_CHECK(!options.valid_control().has_value())
      << "Combinational block generator does not support valid control.";
  std::string module_name(
      options.module_name().value_or(SanitizeIdentifier(f->name())));
  Block* block = f->package()->AddBlock(
      std::make_unique<Block>(module_name, f->package()));

  // A map from the nodes in 'f' to their corresponding node in the block.
  absl::flat_hash_map<Node*, Node*> node_map;
  CodegenPassUnit unit(block->package(), block);

  // Emit the parameters first to ensure the their order is preserved in the
  // block.
  auto func_interface =
      FindFunctionInterface(options.package_interface(), f->name());
  for (Param* param : f->params()) {
    XLS_ASSIGN_OR_RETURN(
        node_map[param],
        block->AddInputPort(param->GetName(), param->GetType(), param->loc()));

    if (func_interface) {
      auto name =
          absl::c_find_if(func_interface->parameters(),
                          [&](const PackageInterfaceProto::NamedValue& p) {
                            return p.name() == param->name();
                          });
      if (name != func_interface->parameters().end() && name->has_sv_type()) {
        unit.metadata[block]
            .streaming_io_and_pipeline
            .input_port_sv_type[node_map[param]->As<InputPort>()] =
            name->sv_type();
      }
    }
  }

  for (Node* node : TopoSort(f)) {
    if (node->Is<Param>()) {
      continue;
    }

    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(Node * block_node,
                         node->CloneInNewFunction(new_operands, block));
    node_map[node] = block_node;
  }

  XLS_ASSIGN_OR_RETURN(OutputPort * output,
                       block->AddOutputPort(options.output_port_name(),
                                            node_map.at(f->return_value())));
  if (func_interface && func_interface->has_sv_result_type()) {
    unit.metadata[block].streaming_io_and_pipeline.output_port_sv_type[output] =
        func_interface->sv_result_type();
  }

  unit.metadata[block]
      .conversion_metadata.emplace<FunctionConversionMetadata>();
  unit.GcMetadata();
  return unit;
}

absl::StatusOr<CodegenPassUnit> ProcToCombinationalBlock(
    Proc* proc, const CodegenOptions& options) {
  VLOG(3) << "Converting proc to combinational block:";
  XLS_VLOG_LINES(3, proc->DumpIr());

  // In a combinational module, the proc cannot have any state to avoid
  // combinational loops. That is, the only loop state must be empty tuples.
  if (proc->GetStateElementCount() > 1 &&
      !std::all_of(proc->StateParams().begin(), proc->StateParams().end(),
                   [&](Param* p) {
                     return p->GetType() == proc->package()->GetTupleType({});
                   })) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Proc must have no state (or state type is all empty tuples) when "
        "lowering to a combinational block. Proc state type is: {%s}",
        absl::StrJoin(proc->StateParams(), ", ",
                      [](std::string* out, Param* p) {
                        absl::StrAppend(out, p->GetType()->ToString());
                      })));
  }

  std::string module_name = SanitizeIdentifier(std::string{
      options.module_name().has_value() ? options.module_name().value()
                                        : proc->name()});
  Block* block = proc->package()->AddBlock(
      std::make_unique<Block>(module_name, proc->package()));

  XLS_ASSIGN_OR_RETURN(StreamingIOPipeline streaming_io,
                       CloneProcNodesIntoBlock(proc, options, block));

  int64_t number_of_outputs = 0;
  for (const auto& outputs : streaming_io.outputs) {
    number_of_outputs += outputs.size();
  }

  if (number_of_outputs > 1) {
    // TODO: do this analysis on a per-stage basis
    XLS_ASSIGN_OR_RETURN(bool streaming_outputs_mutually_exclusive,
                         AreStreamingOutputsMutuallyExclusive(proc));

    if (streaming_outputs_mutually_exclusive) {
      VLOG(3) << absl::StrFormat(
          "%d streaming outputs determined to be mutually exclusive",
          number_of_outputs);
    } else {
      return absl::UnimplementedError(absl::StrFormat(
          "Proc combinational generator only supports streaming "
          "output channels which can be determined to be mutually "
          "exclusive, got %d output channels which were not proven "
          "to be mutually exclusive",
          number_of_outputs));
    }
  }

  XLS_RET_CHECK_EQ(streaming_io.pipeline_registers.size(), 0);

  XLS_RETURN_IF_ERROR(
      AddCombinationalFlowControl(streaming_io.inputs, streaming_io.outputs,
                                  streaming_io.stage_valid, options, block));

  // TODO(tedhong): 2021-09-23 Remove and add any missing functionality to
  //                codegen pipeline.
  CodegenPassUnit unit(block->package(), block);
  unit.metadata[unit.top_block] = CodegenMetadata{
      .streaming_io_and_pipeline = std::move(streaming_io),
      .conversion_metadata = ProcConversionMetadata(),
      .concurrent_stages = std::nullopt,
  };
  XLS_RETURN_IF_ERROR(RemoveDeadTokenNodes(&unit));
  VLOG(3) << "After RemoveDeadTokenNodes";
  XLS_VLOG_LINES(3, unit.DumpIr());

  XLS_RETURN_IF_ERROR(UpdateChannelMetadata(
      unit.metadata[unit.top_block].streaming_io_and_pipeline, unit.top_block));
  VLOG(3) << "After UpdateChannelMetadata";
  XLS_VLOG_LINES(3, unit.DumpIr());

  unit.GcMetadata();
  return unit;
}

absl::StatusOr<CodegenPassUnit> FunctionBaseToCombinationalBlock(
    FunctionBase* f, const CodegenOptions& options) {
  if (f->IsFunction()) {
    return FunctionToCombinationalBlock(f->AsFunctionOrDie(), options);
  }
  if (f->IsProc()) {
    return ProcToCombinationalBlock(f->AsProcOrDie(), options);
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "FunctionBase %s was not a function or proc.", f->name()));
}

}  // namespace verilog
}  // namespace xls
