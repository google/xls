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

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/bdd_io_analysis.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/common/casts.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

namespace {
struct BubbleFlowControl {
  std::vector<Node*> data_load_enable;
  std::vector<Node*> next_stage_open;
};

static absl::StatusOr<std::vector<Node*>> BubbleFlowControlStateEnables(
    const CodegenOptions& options,
    absl::Span<std::optional<Node*> const> pipeline_valid_nodes,
    absl::Span<std::optional<Node*> const> pipeline_done_nodes,
    absl::Span<PipelineStageRegisters> pipeline_data_registers,
    absl::flat_hash_map<Node*, Stage>& node_to_stage_map, Block* block,
    BubbleFlowControl& result) {
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
  const int64_t stage_count = pipeline_data_registers.size() + 1;

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
                         block->GetUniqueRegisterWrite(valid_reg));
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
                              block));
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
  return state_enables;
}

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

  BubbleFlowControl result;
  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> state_enables,
      BubbleFlowControlStateEnables(
          options, pipeline_valid_nodes, pipeline_done_nodes,
          pipeline_data_registers, node_to_stage_map, block, result));

  // Generate writes for state registers. This is done in a separate loop
  // because the last stage isn't included in the pipeline register loop.
  for (std::optional<StateRegister>& state_register : state_registers) {
    if (!state_register.has_value()) {
      continue;
    }
    CHECK(!state_register->next_values.empty());

    if (state_register->reg == nullptr &&
        !state_register->reg_full.has_value()) {
      // No actual contents for this state element, and no need to track it for
      // flow control; skip it.
      continue;
    }

    SourceInfo write_loc = state_register->reg_writes.empty()
                               ? state_register->reg_full->sets.front()->loc()
                               : state_register->reg_writes.front()->loc();

    if (!state_register->reg_writes.empty()) {
      XLS_RET_CHECK_EQ(state_register->next_values.size(),
                       state_register->reg_writes.size());
    }
    if (state_register->reg_full.has_value()) {
      XLS_RET_CHECK_EQ(state_register->next_values.size(),
                       state_register->reg_full->sets.size());
    }
    for (int64_t i = 0; i < state_register->next_values.size(); ++i) {
      const StateRegister::NextValue& next_value =
          state_register->next_values[i];
      std::optional<RegisterWrite*> reg_write =
          state_register->reg_writes.empty()
              ? std::nullopt
              : std::make_optional(state_register->reg_writes[i]);
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

      if (reg_write.has_value()) {
        XLS_RETURN_IF_ERROR(
            reg_write.value()->SetLoadEnable(activated_predicate));
      }
      if (state_register->reg_full.has_value()) {
        XLS_RETURN_IF_ERROR(state_register->reg_full->sets[i]->SetLoadEnable(
            activated_predicate));
      }
    }

    // Predicate the clearing of the reg full bit with the stage activation of
    // the stage containing the state read.
    if (state_register->reg_full.has_value()) {
      RegisterWrite* reg_clear = state_register->reg_full->clear;
      Node* new_predicate;
      if (reg_clear->load_enable().has_value()) {
        XLS_ASSIGN_OR_RETURN(
            new_predicate,
            block->MakeNode<NaryOp>(
                write_loc,
                std::vector<Node*>{state_enables.at(state_register->read_stage),
                                   *reg_clear->load_enable()},
                Op::kAnd));
      } else {
        new_predicate = state_enables.at(state_register->read_stage);
      }
      XLS_RETURN_IF_ERROR(reg_clear->SetLoadEnable(new_predicate));
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
    if (!state_register.has_value() || state_register->reg_writes.empty()) {
      continue;
    }
    CHECK(!state_register->next_values.empty());
    XLS_RET_CHECK_EQ(state_register->next_values.size(),
                     state_register->reg_writes.size());
    if (state_register->reg_full.has_value()) {
      XLS_RET_CHECK_EQ(state_register->reg_full->sets.size(),
                       state_register->reg_writes.size());
    }
    SourceInfo loc = state_register->reg_writes.front()->loc();
    for (int64_t i = 0; i < state_register->next_values.size(); ++i) {
      const StateRegister::NextValue& next_value =
          state_register->next_values[i];
      RegisterWrite* reg_write = state_register->reg_writes[i];
      Node* activated_predicate;
      if (next_value.predicate.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            activated_predicate,
            block->MakeNode<NaryOp>(
                loc, std::vector<Node*>{*next_value.predicate, pipeline_enable},
                Op::kAnd));
      } else {
        activated_predicate = pipeline_enable;
      }

      XLS_RETURN_IF_ERROR(reg_write->SetLoadEnable(activated_predicate));
    }
  }

  return pipeline_enable;
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
      if (!state_register.has_value() ||
          !state_register->reg_full.has_value()) {
        // The state is replaced by the same cycle that reads it, so it will
        // always be valid.
        continue;
      }

      // The state is valid if it's full, or if we know it's unread (and thus
      // not written to either) in the activation that's currently at the stage
      // that wants to read it.
      std::string state_valid_name = "";
      absl::InlinedVector<Node*, 2> state_valid_conditions(
          {state_register->reg_full->read});
      if (state_register->read_predicate.has_value()) {
        // Don't block if we're not reading the state.

        // If predicate has an assigned name, let the not expression get
        // inlined. Otherwise, give a descriptive name.
        std::string name = "";
        if (!state_register->read_predicate.value()->HasAssignedName()) {
          name = absl::StrFormat("%s_not_read", state_register->name);
        }
        XLS_ASSIGN_OR_RETURN(
            Node * unread,
            block->MakeNodeWithName<UnOp>(
                state_register->reg_full->read->loc(),
                state_register->read_predicate.value(), Op::kNot, name));

        // not_read will have an assigned name or be inlined, so only check the
        // state full register read. If it has an assigned name, just let
        // everything inline. Otherwise, give a descriptive name.
        if (!state_register->reg_full->read->HasAssignedName()) {
          state_valid_name =
              absl::StrFormat("%s_state_valid", state_register->name);
        }
        state_valid_conditions.push_back(unread);
      }

      XLS_ASSIGN_OR_RETURN(
          Node * state_valid,
          NaryOrIfNeeded(
              block, state_valid_conditions,
              /*name=*/state_valid_name,
              /*source_info=*/state_register->reg_full->read->loc()));
      active_valids.push_back(state_valid);
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

// Plumb valid signal through the pipeline stages, ANDing with a valid produced
// by each stage. Gather the pipelined valid signal in a vector where the
// zero-th element is the input port and subsequent elements are the pipelined
// valid signal from each stage.
static absl::Status MakePipelineStagesForValidIO(
    StreamingIOPipeline& streaming_io, absl::Span<Node* const> recvs_valid,
    absl::Span<Node* const> states_valid, absl::Span<Node* const> sends_ready,
    Block* block, const CodegenOptions& options) {
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
      if (!block->GetResetBehavior().has_value()) {
        return absl::InternalError(absl::StrFormat(
            "Block `%s` must have a reset for valid signal to be registered",
            block->name()));
      }
      if (!block->GetResetBehavior().has_value() &&
          options.valid_control().value().has_output_name()) {
        return absl::InternalError(absl::StrFormat(
            "Block `%s` must have a reset for valid signal to be registered",
            block->name()));
      }
      XLS_ASSIGN_OR_RETURN(Register * valid_reg,
                           block->AddRegisterWithZeroResetValue(
                               PipelineSignalName("valid", stage), u1));
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

// Updates datapath pipeline registers with a reset signal.
//  1. The pipeline registers are reset active_high or active_low
//     following the block behavior.
//  2. The registers are reset to zero.
//  3. The registers are reset whenever the block reset is active.
static absl::Status UpdateDatapathRegistersWithReset(
    absl::Span<PipelineStageRegisters> pipeline_data_registers, Block* block) {
  // Blocks should have reset information.

  std::optional<ResetBehavior> reset_behavior = block->GetResetBehavior();
  std::optional<InputPort*> reset_port = block->GetResetPort();

  if (!reset_port.has_value() || !reset_behavior.has_value()) {
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

      // Replace register reset.
      XLS_RETURN_IF_ERROR(pipeline_reg.reg_write->SetReset(reset_node));
    }
  }

  return absl::OkStatus();
}

// Add one-shot logic to the and output RDV channel.
static absl::Status AddOneShotLogicToRVNodes(Node* from_valid, Node* from_rdy,
                                             Node* all_active_outputs_ready,
                                             std::string_view name_prefix,
                                             Block* block) {
  // Location for added logic is taken from from_valid.
  SourceInfo loc = from_valid->loc();

  // Add a node to serve as placeholder (will be removed later).
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  // Create a register to store whether or not channel has been sent.
  Type* u1 = block->package()->GetBitsType(1);
  std::string name = absl::StrFormat("__%s_has_been_sent_reg", name_prefix);
  if (!block->GetResetBehavior().has_value()) {
    return absl::InternalError(absl::StrFormat(
        "Block `%s` must have a reset signal for send one-shot logic",
        block->name()));
  }
  XLS_ASSIGN_OR_RETURN(Register * has_been_sent_reg,
                       block->AddRegisterWithZeroResetValue(name, u1));
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
      // Add a buffer before the valid output ports and after
      // the ready input port to serve as points where the
      // additional logic from AddRegisterToRDVNodes() can be inserted.
      XLS_ASSIGN_OR_RETURN(std::string port_name,
                           StreamingIOName(*output.GetDataPort()));
      XLS_ASSIGN_OR_RETURN(std::string port_valid_name,
                           StreamingIOName(*output.GetValidPort()));
      std::string valid_buf_name = absl::StrFormat("__%s_buf", port_valid_name);

      XLS_ASSIGN_OR_RETURN(
          Node * output_port_valid_buf,
          block->MakeNodeWithName<UnOp>(
              /*loc=*/SourceInfo(), output.GetValidPort().value()->operand(0),
              Op::kIdentity, valid_buf_name));
      XLS_RETURN_IF_ERROR(output.GetValidPort().value()->ReplaceOperandNumber(
          0, output_port_valid_buf));

      XLS_ASSIGN_OR_RETURN(
          Node * output_port_ready_buf,
          block->MakeNodeWithName<UnOp>(
              /*loc=*/SourceInfo(), output.GetReadyPort().value(),
              Op::kIdentity, valid_buf_name));

      XLS_RETURN_IF_ERROR(output.GetReadyPort().value()->ReplaceUsesWith(
          output_port_ready_buf));

      XLS_RETURN_IF_ERROR(AddOneShotLogicToRVNodes(
          output_port_valid_buf, output_port_ready_buf,
          all_active_outputs_ready[stage], port_name, block));
    }
  }

  return absl::OkStatus();
}

// Adds ready/valid ports for each of the given streaming inputs/outputs. Also,
// adds logic which propagates ready and valid signals through the block.
//
// Returns (via reference argument) the vector of all_active_output_ready nodes.
// See MakeInputReadyPortsForOutputChannels() for more.
static absl::Status AddBubbleFlowControl(
    const CodegenOptions& options, StreamingIOPipeline& streaming_io,
    Proc* proc, absl::Span<ProcInstance* const> instances, Block* block,
    std::vector<Node*>& all_active_outputs_ready) {
  int64_t stage_count = streaming_io.pipeline_registers.size() + 1;
  std::string_view data_suffix = options.streaming_channel_data_suffix();
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

  XLS_RETURN_IF_ERROR(MakePipelineStagesForValidIO(
      streaming_io, /*recvs_valid=*/all_active_inputs_valid,
      /*states_valid=*/all_active_states_valid,
      /*sends_ready=*/all_active_outputs_ready, block, options));

  VLOG(3) << "After Valids";
  XLS_VLOG_LINES(3, block->DumpIr());

  VLOG(3) << "After State Updated";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (options.reset().has_value() && options.reset()->reset_data_path()) {
    XLS_RETURN_IF_ERROR(UpdateDatapathRegistersWithReset(
        absl::MakeSpan(streaming_io.pipeline_registers), block));

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
        proc, instances, block));
    XLS_RETURN_IF_ERROR(MakeOutputDataPortsForOutputChannels(
        all_active_inputs_valid, stage_valid_no_option,
        bubble_flow_control.next_stage_open, streaming_io.outputs, data_suffix,
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
      proc, instances, block));

  VLOG(3) << "After Ready";
  XLS_VLOG_LINES(3, block->DumpIr());
  return absl::OkStatus();
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
    std::string_view name_prefix, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  CHECK_EQ(from_rdy->operand_count(), 1);

  // Add a node for load_enables (will be removed later).
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  // Create data/valid and their skid counterparts.
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_reg_read,
      AddRegisterAfterNode(/*name_prefix=*/name_prefix,
                           /*load_enable=*/literal_1, from_data));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_skid_reg_read,
      AddRegisterAfterNode(/*name_prefix=*/absl::StrCat(name_prefix, "_skid"),
                           /*load_enable=*/literal_1, data_reg_read));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_valid_reg_read,
      AddRegisterAfterNode(/*name_prefix=*/absl::StrCat(name_prefix, "_valid"),
                           /*load_enable=*/literal_1, from_valid));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_valid_skid_reg_read,
      AddRegisterAfterNode(
          /*name_prefix=*/absl::StrCat(name_prefix, "_valid_skid"),
          /*load_enable=*/literal_1, data_valid_reg_read));

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

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * data_reg_write,
      block->GetUniqueRegisterWrite(data_reg_read->GetRegister()));
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
      block->GetUniqueRegisterWrite(data_valid_reg_read->GetRegister()));
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
      block->GetUniqueRegisterWrite(data_skid_reg_read->GetRegister()));
  XLS_RETURN_IF_ERROR(
      data_skid_reg_write->ReplaceExistingLoadEnable(skid_data_load_en));

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * data_valid_skid_reg_write,
      block->GetUniqueRegisterWrite(data_valid_skid_reg_read->GetRegister()));

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

// Replace load_en for the register with the given node.
static absl::Status UpdateRegisterLoadEn(Node* load_en, Register* reg,
                                         Block* block) {
  XLS_ASSIGN_OR_RETURN(RegisterWrite * old_reg_write,
                       block->GetUniqueRegisterWrite(reg));

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
    std::string_view name_prefix, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  CHECK_EQ(from_rdy->operand_count(), 1);

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_reg_read,
      AddRegisterAfterNode(/*name_prefix=*/name_prefix,
                           /*load_enable=*/std::nullopt, from_data));
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * valid_reg_read,
      AddRegisterAfterNode(/*name_prefix=*/absl::StrCat(name_prefix, "_valid"),
                           /*load_enable=*/std::nullopt, from_valid));

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
static absl::Status AddRegisterAfterStreamingInput(
    StreamingInput& input, FlopKind flop, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  XLS_ASSIGN_OR_RETURN(std::string port_name,
                       StreamingIOName(*input.GetDataPort()));
  switch (flop) {
    case FlopKind::kZeroLatency:
      return AddZeroLatencyBufferToRDVNodes(
                 *input.GetDataPort(), *input.GetValidPort(),
                 *input.GetReadyPort(), port_name, block, valid_nodes)
          .status();
    case FlopKind::kSkid:
      return AddSkidBufferToRDVNodes(
                 *input.GetDataPort(), *input.GetValidPort(),
                 *input.GetReadyPort(), port_name, block, valid_nodes)
          .status();
    case FlopKind::kFlop:
      return AddRegisterToRDVNodes(*input.GetDataPort(), *input.GetValidPort(),
                                   *input.GetReadyPort(), port_name, block,
                                   valid_nodes)
          .status();
    case FlopKind::kNone:
      return absl::OkStatus();
  }
}

// Adds a register after the input streaming channel's data and valid.
// Returns the node for the register_read of the data.
static absl::Status AddRegisterBeforeStreamingOutput(
    StreamingOutput& output, FlopKind flop, Block* block,
    std::vector<std::optional<Node*>>& valid_nodes) {
  if (flop == FlopKind::kNone) {
    // Non-flopped outputs need no additional buffers/logic
    return absl::OkStatus();
  }
  // Add buffers before the data/valid output ports and after
  // the ready input port to serve as points where the
  // additional logic from AddRegisterToRDVNodes() can be inserted.
  XLS_ASSIGN_OR_RETURN(std::string port_name,
                       StreamingIOName(*output.GetDataPort()));
  XLS_ASSIGN_OR_RETURN(std::string port_valid_name,
                       StreamingIOName(*output.GetValidPort()));
  XLS_ASSIGN_OR_RETURN(std::string port_ready_name,
                       StreamingIOName(*output.GetReadyPort()));
  std::string data_buf_name = absl::StrFormat("__%s_buf", port_name);
  std::string valid_buf_name = absl::StrFormat("__%s_buf", port_valid_name);
  std::string ready_buf_name = absl::StrFormat("__%s_buf", port_ready_name);
  XLS_ASSIGN_OR_RETURN(
      Node * output_port_data_buf,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), output.GetDataPort().value()->operand(0),
          Op::kIdentity, data_buf_name));
  XLS_RETURN_IF_ERROR(output.GetDataPort().value()->ReplaceOperandNumber(
      0, output_port_data_buf));

  XLS_ASSIGN_OR_RETURN(
      Node * output_port_valid_buf,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), output.GetValidPort().value()->operand(0),
          Op::kIdentity, valid_buf_name));
  XLS_RETURN_IF_ERROR(output.GetValidPort().value()->ReplaceOperandNumber(
      0, output_port_valid_buf));

  XLS_ASSIGN_OR_RETURN(Node * output_port_ready_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), *output.GetReadyPort(),
                           Op::kIdentity, ready_buf_name));

  XLS_RETURN_IF_ERROR(
      output.GetReadyPort().value()->ReplaceUsesWith(output_port_ready_buf));

  switch (flop) {
    case FlopKind::kZeroLatency:
      return AddZeroLatencyBufferToRDVNodes(
                 output_port_data_buf, output_port_valid_buf,
                 output_port_ready_buf, port_name, block, valid_nodes)
          .status();
    case FlopKind::kSkid:
      return AddSkidBufferToRDVNodes(
                 output_port_data_buf, output_port_valid_buf,
                 output_port_ready_buf, port_name, block, valid_nodes)
          .status();
    case FlopKind::kFlop:
      return AddRegisterToRDVNodes(output_port_data_buf, output_port_valid_buf,
                                   output_port_ready_buf, port_name, block,
                                   valid_nodes)
          .status();
    case FlopKind::kNone:
      LOG(FATAL)
          << "Unreachable condition. Non-flopped should have short circuited.";
  }
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
  auto maybe_mark_node_handled = [&](std::optional<Node*> n) {
    if (n.has_value()) {
      handled_io_nodes.insert(*n);
    }
  };

  // Flop streaming inputs.
  for (auto& vec : streaming_io.inputs) {
    for (StreamingInput& input : vec) {
      // TODO(https://github.com/google/xls/issues/1803): This is super hacky.
      // We really should have a different pass that configures all the channels
      // in a separate lowering step.
      FlopKind kind;
      if (std::holds_alternative<Channel*>(input.GetChannel())) {
        if (block->package()->ChannelsAreProcScoped()) {
          // With proc scoped channels, the interface holds the flop kind.
          continue;
        }
        StreamingChannel* channel = down_cast<StreamingChannel*>(
            std::get<Channel*>(input.GetChannel()));
        XLS_RET_CHECK(channel->channel_config().input_flop_kind())
            << "No input flop kind";
        kind = channel->channel_config().input_flop_kind().value();
      } else {
        kind = std::get<ChannelInterface*>(input.GetChannel())->flop_kind();
      }
      XLS_RETURN_IF_ERROR(
          AddRegisterAfterStreamingInput(input, kind, block, valid_nodes));

      // ready for an output port is an output for the block,
      // record that we should not separately add a flop these inputs.
      maybe_mark_node_handled(input.GetReadyPort());
      maybe_mark_node_handled(input.GetDataPort());
      maybe_mark_node_handled(input.GetValidPort());
    }
  }

  // Flop streaming outputs.
  for (auto& vec : streaming_io.outputs) {
    for (StreamingOutput& output : vec) {
      // TODO(https://github.com/google/xls/issues/1803): This is super hacky.
      // We really should have a different pass that configures all the channels
      // in a separate lowering step.
      FlopKind kind;
      if (std::holds_alternative<Channel*>(output.GetChannel())) {
        if (block->package()->ChannelsAreProcScoped()) {
          // With proc scoped channels, the interface holds the flop kind.
          continue;
        }
        StreamingChannel* channel = down_cast<StreamingChannel*>(
            std::get<Channel*>(output.GetChannel()));
        XLS_RET_CHECK(channel->channel_config().output_flop_kind())
            << "No output flop kind";
        kind = channel->channel_config().output_flop_kind().value();
      } else {
        kind = std::get<ChannelInterface*>(output.GetChannel())->flop_kind();
      }
      XLS_RETURN_IF_ERROR(
          AddRegisterBeforeStreamingOutput(output, kind, block, valid_nodes));

      // ready for an output port is an input for the block,
      // record that we should not separately add a flop these inputs.
      maybe_mark_node_handled(output.GetReadyPort());
      maybe_mark_node_handled(output.GetDataPort());
      maybe_mark_node_handled(output.GetValidPort());
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

      XLS_RETURN_IF_ERROR(AddRegisterAfterNode(/*name_prefix=*/port->GetName(),
                                               /*load_enable=*/std::nullopt,
                                               port)
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

      XLS_RETURN_IF_ERROR(AddRegisterAfterNode(/*name_prefix=*/port->GetName(),
                                               /*load_enable=*/std::nullopt,
                                               port->operand(0))
                              .status());

      handled_io_nodes.insert(port);
    }
  }

  return absl::OkStatus();
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
    if (streaming_input.GetPredicate().has_value()) {
      recv_if_preds.push_back(streaming_input.GetPredicate().value());
    } else {
      // There is an unconditional receive node, return literal 0
      return block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(0, 1)));
    }
  }

  // Stage 0 has only conditional receives, return NOR of their predicates.
  return NaryNorIfNeeded(block, recv_if_preds);
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
      XLS_RET_CHECK(input.GetSignalValid().has_value());
      valid_nodes.push_back(input.GetSignalValid());
    }
  }

  for (auto& vec : streaming_io.outputs) {
    for (StreamingOutput& output : vec) {
      valid_nodes.push_back(output.GetValidPort().value()->operand(0));
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
}  // namespace

// Public interface
absl::Status SingleProcToPipelinedBlock(
    const PackageSchedule& package_schedule, const CodegenOptions& options,
    CodegenContext& context, Proc* proc,
    absl::Span<ProcInstance* const> instances, Block* absl_nonnull block,
    const absl::flat_hash_map<FunctionBase*, Block*>& converted_blocks,
    std::optional<const ProcElaboration*> elab) {
  VLOG(1) << absl::StrFormat("SingleProcToPipelinedBlock(proc=`%s`, block=`%s)",
                             proc->name(), block->name());
  XLS_RET_CHECK(package_schedule.HasSchedule(proc));
  if (std::optional<int64_t> ii = proc->GetInitiationInterval();
      ii.has_value()) {
    block->SetInitiationInterval(*ii);
  }

  XLS_RET_CHECK_OK(MaybeAddResetPort(block, options));

  XLS_RETURN_IF_ERROR(block->AddClockPort("clk"));
  VLOG(3) << "Schedule Used";
  XLS_VLOG_LINES(3, package_schedule.ToString());

  XLS_ASSIGN_OR_RETURN(
      (auto [streaming_io_and_pipeline, concurrent_stages]),
      CloneNodesIntoPipelinedBlock(proc, package_schedule, options, block,
                                   converted_blocks));

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

  // Initialize the valid flops to the pipeline registers.
  // First element is skipped as the initial stage valid flops will be
  // constructed from the input flops and/or input valid ports and will be
  // added later in AddInputFlops() and AddIdleOutput().
  std::vector<std::optional<Node*>> valid_flops;
  std::vector<Node*> all_active_outputs_ready;
  XLS_RETURN_IF_ERROR(AddBubbleFlowControl(
      options, streaming_io_and_pipeline, proc, instances, block,
      /*all_active_outputs_ready=*/all_active_outputs_ready));
  CHECK_GE(streaming_io_and_pipeline.stage_valid.size(), 1);
  std::copy(streaming_io_and_pipeline.stage_valid.begin() + 1,
            streaming_io_and_pipeline.stage_valid.end(),
            std::back_inserter(valid_flops));

  VLOG(3) << "After Flow Control";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (!streaming_outputs_mutually_exclusive) {
    XLS_RETURN_IF_ERROR(AddOneShotOutputLogic(
        options, streaming_io_and_pipeline,
        /*all_active_outputs_ready=*/all_active_outputs_ready, block));
  }
  VLOG(3) << absl::StrFormat("After Output Triggers");
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_RETURN_IF_ERROR(AddInputOutputFlops(options, streaming_io_and_pipeline,
                                          block, valid_flops));
  VLOG(3) << "After Input or Output Flops";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (options.add_idle_output()) {
    XLS_RETURN_IF_ERROR(
        AddIdleOutput(valid_flops, streaming_io_and_pipeline, block));
  }
  VLOG(3) << "After Add Idle Output";
  XLS_VLOG_LINES(3, block->DumpIr());

  // RemoveDeadTokenNodes() mutates metadata.
  context.SetMetadataForBlock(
      block,
      CodegenMetadata{
          .streaming_io_and_pipeline = std::move(streaming_io_and_pipeline),
          .concurrent_stages = std::move(concurrent_stages),
      });

  // TODO(tedhong): 2021-09-23 Remove and add any missing functionality to
  //                codegen pipeline.
  XLS_RETURN_IF_ERROR(RemoveDeadTokenNodes(block, context));

  VLOG(3) << "After RemoveDeadTokenNodes";
  XLS_VLOG_LINES(3, block->DumpIr());

  // TODO: add simplification pass here to remove unnecessary `1 & x`

  return absl::OkStatus();
}
}  // namespace xls::verilog
