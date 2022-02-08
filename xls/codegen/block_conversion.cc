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

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/register_legalization_pass.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value_helpers.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/tuple_simplification_pass.h"
#include "re2/re2.h"

namespace xls {
namespace verilog {
namespace {

// Name of the output port which holds the return value of the function.
// TODO(meheff): 2021-03-01 Allow port names other than "out".
static const char kOutputPortName[] = "out";

// Contains information about the block's reset -- its input port and behavior.
struct ResetInfo {
  absl::optional<InputPort*> input_port;
  absl::optional<xls::Reset> behavior;
  bool reset_data_path = false;
};

// If options specify it, adds and returns an input for a reset signal.
static absl::StatusOr<ResetInfo> MaybeAddResetPort(
    Block* block, const CodegenOptions& options) {
  // Optional add reset port.
  ResetInfo reset_info;

  // TODO(tedhong): 2021-09-18 Combine this with AddValidSignal
  if (options.reset().has_value()) {
    XLS_ASSIGN_OR_RETURN(reset_info.input_port,
                         block->AddInputPort(options.reset()->name(),
                                             block->package()->GetBitsType(1)));
    reset_info.behavior = xls::Reset();
    reset_info.behavior->reset_value = Value(UBits(0, 1));
    reset_info.behavior->asynchronous = options.reset()->asynchronous();
    reset_info.behavior->active_low = options.reset()->active_low();

    reset_info.reset_data_path = options.reset()->reset_data_path();
  }

  return reset_info;
}

// Return a schedule based on the given schedule that adjusts the cycles of the
// nodes to introduce pipeline registers immediately after input ports or
// immediately before output ports based on the `flop_inputs` and `flop_outputs`
// options in CodegenOptions.
static absl::StatusOr<PipelineSchedule> MaybeAddInputOutputFlopsToSchedule(
    const PipelineSchedule& schedule, const CodegenOptions& options) {
  // All params must be scheduled in cycle 0.
  ScheduleCycleMap cycle_map;
  for (Param* param : schedule.function_base()->params()) {
    XLS_RET_CHECK_EQ(schedule.cycle(param), 0);
    cycle_map[param] = 0;
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

// A data structure representing a pipeline register for a single XLS IR value.
struct PipelineRegister {
  Register* reg;
  RegisterWrite* reg_write;
  RegisterRead* reg_read;
};

// A data structure representing a state register for a single XLS IR value.
struct StateRegister {
  std::string name;
  Value reset_value;
  Register* reg;
  RegisterWrite* reg_write;
  RegisterRead* reg_read;
};

// The collection of pipeline registers for a single stage.
using PipelineStageRegisters = std::vector<PipelineRegister>;

// Plumb valid signal through the pipeline stages. Gather the pipelined valid
// signal in a vector where the zero-th element is the input port and
// subsequent elements are the pipelined valid signal from each stage.
static absl::StatusOr<std::vector<Node*>> MakePipelineStagesForValid(
    Node* initial_valid_node,
    absl::Span<const PipelineStageRegisters> pipeline_registers,
    const ResetInfo& reset_info, Block* block) {
  Type* u1 = block->package()->GetBitsType(1);

  std::vector<Node*> pipelined_valids(pipeline_registers.size() + 1);
  pipelined_valids[0] = initial_valid_node;

  for (int64_t stage = 0; stage < pipeline_registers.size(); ++stage) {
    // Add valid register to each pipeline stage.
    XLS_ASSIGN_OR_RETURN(Register * valid_reg,
                         block->AddRegister(PipelineSignalName("valid", stage),
                                            u1, reset_info.behavior));
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<RegisterWrite>(
                                /*loc=*/absl::nullopt, pipelined_valids[stage],
                                /*load_enable=*/absl::nullopt,
                                /*reset=*/reset_info.input_port, valid_reg)
                            .status());
    XLS_ASSIGN_OR_RETURN(pipelined_valids[stage + 1],
                         block->MakeNode<RegisterRead>(
                             /*loc=*/absl::nullopt, valid_reg));
  }

  return pipelined_valids;
}

// Returns or makes a node that is 1 when the block is under reset,
// if said reset signal exists.
//
//   - If no reset exists, absl::nullopt is returned
//   - Active low reset signals are inverted.
//
// See also MakeOrWithResetNode()
static absl::StatusOr<absl::optional<Node*>> MaybeGetOrMakeResetNode(
    const ResetInfo& reset_info, Block* block) {
  if (!reset_info.input_port.has_value()) {
    return absl::nullopt;
  }

  Node* reset_node = reset_info.input_port.value();
  if (reset_info.behavior->active_low) {
    return block->MakeNode<UnOp>(/*loc=*/absl::nullopt, reset_node, Op::kNot);
  }

  return reset_node;
}

// Updates the state_register with a reset signal.
//  1. The state register is reset active_high or active_low
//     following the block behavior.
//  2. The state register is reset to the initial value of the proc.
//  3. The state register is reset whenever the block reset is active.
static absl::Status UpdateStateRegisterWithReset(const ResetInfo& reset_info,
                                                 StateRegister& state_register,
                                                 Block* block) {
  XLS_CHECK_NE(state_register.reg, nullptr);
  XLS_CHECK_NE(state_register.reg_write, nullptr);
  XLS_CHECK_NE(state_register.reg_read, nullptr);

  Register* old_reg = state_register.reg;
  RegisterWrite* old_reg_write = state_register.reg_write;
  RegisterRead* old_reg_read = state_register.reg_read;

  // Blocks containing a state register must also have a reset signal.
  if (!reset_info.input_port.has_value() || !reset_info.behavior.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unable to update state register %s with reset, signal as block"
        " was not created with a reset.",
        state_register.reg->name()));
  }

  // Follow the reset behavior of the valid registers except for the initial
  // value.
  xls::Reset reset_behavior = reset_info.behavior.value();
  reset_behavior.reset_value = state_register.reset_value;
  Node* reset_node = reset_info.input_port.value();

  // Clone and create a new set of Register, RegisterWrite, and RegisterRead.
  // Update the inout parameter state_register as well.
  std::string name = block->UniquifyNodeName(state_register.name);

  XLS_ASSIGN_OR_RETURN(
      state_register.reg,
      block->AddRegister(name, old_reg->type(), reset_behavior));

  XLS_ASSIGN_OR_RETURN(state_register.reg_write,
                       block->MakeNode<RegisterWrite>(
                           /*loc=*/old_reg_write->loc(),
                           /*data=*/old_reg_write->data(),
                           /*load_enable=*/absl::nullopt,
                           /*reset=*/reset_node,
                           /*reg=*/state_register.reg));

  XLS_ASSIGN_OR_RETURN(state_register.reg_read,
                       block->MakeNodeWithName<RegisterRead>(
                           /*loc=*/old_reg_read->loc(),
                           /*reg=*/state_register.reg,
                           /*name=*/name));

  // Replace old uses of RegisterRead with the new one
  XLS_RETURN_IF_ERROR(old_reg_read->ReplaceUsesWith(state_register.reg_read));
  XLS_RETURN_IF_ERROR(block->RemoveNode(old_reg_read));
  XLS_RETURN_IF_ERROR(block->RemoveNode(old_reg_write));
  XLS_RETURN_IF_ERROR(block->RemoveRegister(old_reg));

  return absl::OkStatus();
}

// Updates datapath pipeline registers with a reset signal.
//  1. The pipeline registers are reset active_high or active_low
//     following the block behavior.
//  2. The registers are reset to zero.
//  3. The registers are reset whenever the block reset is active.
static absl::Status UpdateDatapathRegistersWithReset(
    const ResetInfo& reset_info,
    absl::Span<PipelineStageRegisters> pipeline_data_registers, Block* block) {
  // Blocks should have reset information.
  if (!reset_info.input_port.has_value() || !reset_info.behavior.has_value()) {
    return absl::InvalidArgumentError(
        "Unable to update pipeline registers with reset, signal as block"
        " was not created with a reset.");
  }

  // Update each datapath register with a reset.
  Node* reset_node = reset_info.input_port.value();
  int64_t stage_count = pipeline_data_registers.size();

  for (int64_t stage = 0; stage < stage_count; ++stage) {
    for (PipelineRegister& pipeline_reg : pipeline_data_registers.at(stage)) {
      XLS_CHECK_NE(pipeline_reg.reg, nullptr);
      XLS_CHECK_NE(pipeline_reg.reg_write, nullptr);
      XLS_CHECK_NE(pipeline_reg.reg_read, nullptr);

      // Don't attempt to reset registers that will be removed later
      // (ex. tokens).
      Type* node_type = pipeline_reg.reg->type();
      if (node_type->GetFlatBitCount() == 0) {
        continue;
      }

      // Reset the register to zero of the correct type.
      xls::Reset reset_behavior = reset_info.behavior.value();
      reset_behavior.reset_value = ZeroOfType(node_type);

      // Clone and create a new set of Register, RegisterWrite, and
      // RegisterRead.
      Register* old_reg = pipeline_reg.reg;
      RegisterWrite* old_reg_write = pipeline_reg.reg_write;
      RegisterRead* old_reg_read = pipeline_reg.reg_read;

      std::string name = block->UniquifyNodeName(old_reg->name());

      XLS_ASSIGN_OR_RETURN(
          pipeline_reg.reg,
          block->AddRegister(name, old_reg->type(), reset_behavior));

      XLS_ASSIGN_OR_RETURN(pipeline_reg.reg_write,
                           block->MakeNode<RegisterWrite>(
                               /*loc=*/old_reg_write->loc(),
                               /*data=*/old_reg_write->data(),
                               /*load_enable=*/old_reg_write->load_enable(),
                               /*reset=*/reset_node,
                               /*reg=*/pipeline_reg.reg));

      XLS_ASSIGN_OR_RETURN(pipeline_reg.reg_read,
                           block->MakeNodeWithName<RegisterRead>(
                               /*loc=*/old_reg_read->loc(),
                               /*reg=*/pipeline_reg.reg,
                               /*name=*/name));

      // Replace old uses of RegisterRead with the new one
      XLS_RETURN_IF_ERROR(old_reg_read->ReplaceUsesWith(pipeline_reg.reg_read));
      XLS_RETURN_IF_ERROR(block->RemoveNode(old_reg_read));
      XLS_RETURN_IF_ERROR(block->RemoveNode(old_reg_write));
      XLS_RETURN_IF_ERROR(block->RemoveRegister(old_reg));
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
static absl::StatusOr<Node*> MakeOrWithResetNode(Node* src_node,
                                                 absl::string_view result_name,
                                                 const ResetInfo& reset_info,
                                                 Block* block) {
  Node* result = src_node;

  XLS_ASSIGN_OR_RETURN(absl::optional<Node*> maybe_reset_node,
                       MaybeGetOrMakeResetNode(reset_info, block));

  if (maybe_reset_node.has_value()) {
    Node* reset_node = maybe_reset_node.value();
    XLS_ASSIGN_OR_RETURN(result, block->MakeNodeWithName<NaryOp>(
                                     /*loc=*/absl::nullopt,
                                     std::vector<Node*>({result, reset_node}),
                                     Op::kOr, result_name));
  }

  return result;
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
static absl::StatusOr<Node*> UpdatePipelineWithBubbleFlowControl(
    Node* initial_output_ready_node, const ResetInfo& reset_info,
    absl::Span<Node*> pipeline_valid_nodes,
    absl::Span<PipelineStageRegisters> pipeline_data_registers,
    absl::optional<StateRegister>& state_register, Block* block) {
  // Create enable signals for each pipeline stage.
  //   - the last enable signal is the initial_output_ready_node.
  //     enable_signals[N] = initial_output_ready_node
  //   - enable_signal[n-1] = enable_signal[n] || ! valid[n]
  //
  // Data registers are gated whenever data is invalid so
  //   - data_enable_signal[n-1] = (enable_signal[n-1] && valid[n-1]) || rst
  //
  // State registers are gated whenever data is invalid, but
  // are not transparent during reset
  //   - state_enable_signal = (enable_signal[0] && valid[0]).

  int64_t stage_count = pipeline_data_registers.size();
  std::vector<Node*> enable_n(stage_count + 1);
  enable_n.at(stage_count) = initial_output_ready_node;

  // We initialize data_load_enable here so that this function
  // can return the data_load_enable for the first pipeline stage --
  // which is the last data_load_enable node created by this function.
  Node* data_load_enable = initial_output_ready_node;

  for (int64_t stage = stage_count - 1; stage >= 0; --stage) {
    // Create load enables for valid registers.
    XLS_ASSIGN_OR_RETURN(
        Node * not_valid_np1,
        block->MakeNodeWithName<UnOp>(
            /*loc=*/absl::nullopt, pipeline_valid_nodes.at(stage + 1), Op::kNot,
            PipelineSignalName("not_valid", stage)));

    std::vector<Node*> en_operands = {enable_n.at(stage + 1), not_valid_np1};
    XLS_ASSIGN_OR_RETURN(
        Node * enable,
        block->MakeNodeWithName<NaryOp>(absl::nullopt, en_operands, Op::kOr,
                                        PipelineSignalName("enable", stage)));
    enable_n.at(stage) = enable;

    // Update valid registers with load enables.
    RegisterRead* valid_reg_read =
        pipeline_valid_nodes.at(stage + 1)->As<RegisterRead>();
    XLS_RET_CHECK(valid_reg_read != nullptr);
    Register* valid_reg = valid_reg_read->GetRegister();
    XLS_ASSIGN_OR_RETURN(RegisterWrite * valid_reg_write,
                         block->GetRegisterWrite(valid_reg));
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<RegisterWrite>(
                                /*loc=*/absl::nullopt, valid_reg_write->data(),
                                /*load_enable=*/enable,
                                /*reset=*/valid_reg_write->reset(), valid_reg)
                            .status());
    XLS_RETURN_IF_ERROR(block->RemoveNode(valid_reg_write));

    // Create load enables for datapath registers.
    std::vector<Node*> data_en_operands = {enable,
                                           pipeline_valid_nodes.at(stage)};
    XLS_ASSIGN_OR_RETURN(Node * data_enable,
                         block->MakeNodeWithName<NaryOp>(
                             absl::nullopt, data_en_operands, Op::kAnd,
                             PipelineSignalName("data_enable", stage)));

    // If datapath registers are reset, then adding reset to the
    // load enable is redundant.
    if (reset_info.reset_data_path) {
      data_load_enable = data_enable;
    } else {
      XLS_ASSIGN_OR_RETURN(
          data_load_enable,
          MakeOrWithResetNode(data_enable, PipelineSignalName("load_en", stage),
                              reset_info, block));
    }

    // Update datapath registers with load enables.
    if (!pipeline_data_registers.at(stage).empty()) {
      for (PipelineRegister& pipeline_reg : pipeline_data_registers.at(stage)) {
        XLS_ASSIGN_OR_RETURN(
            RegisterWrite * new_reg_write,
            block->MakeNode<RegisterWrite>(
                /*loc=*/absl::nullopt, pipeline_reg.reg_write->data(),
                /*load_enable=*/data_load_enable,
                /*reset=*/pipeline_reg.reg_write->reset(), pipeline_reg.reg));
        XLS_RETURN_IF_ERROR(block->RemoveNode(pipeline_reg.reg_write));
        pipeline_reg.reg_write = new_reg_write;
      }
    }

    // Also update the state register and share the enable signal
    // with the data registers.
    if (stage == 0 && state_register.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          RegisterWrite * new_reg_write,
          block->MakeNode<RegisterWrite>(
              /*loc=*/state_register->reg_write->loc(),
              /*data=*/state_register->reg_write->data(),
              /*load_enable=*/data_enable,
              /*reset=*/state_register->reg_write->reset(),
              /*reg=*/state_register->reg_write->GetRegister()));

      XLS_RETURN_IF_ERROR(block->RemoveNode(state_register->reg_write));
      state_register->reg_write = new_reg_write;
    }
  }

  return data_load_enable;
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
    absl::optional<StateRegister>& state_register_opt, Block* block) {
  std::vector<Node*> operands = {all_active_outputs_ready,
                                 all_active_inputs_valid};
  XLS_ASSIGN_OR_RETURN(
      Node * pipeline_enable,
      block->MakeNodeWithName<NaryOp>(absl::nullopt, operands, Op::kAnd,
                                      "pipeline_enable"));

  if (state_register_opt.has_value()) {
    StateRegister& state_register = state_register_opt.value();
    XLS_ASSIGN_OR_RETURN(RegisterWrite * new_reg_write,
                         block->MakeNode<RegisterWrite>(
                             /*loc=*/state_register.reg_write->loc(),
                             /*data=*/state_register.reg_write->data(),
                             /*load_enable=*/pipeline_enable,
                             /*reset=*/state_register.reg_write->reset(),
                             /*reg=*/state_register.reg_write->GetRegister()));

    XLS_RETURN_IF_ERROR(block->RemoveNode(state_register.reg_write));
    state_register.reg_write = new_reg_write;
  }

  return pipeline_enable;
}

// Plumbs a valid signal through the block. This includes:
// (1) Add an input port for a single-bit valid signal.
// (2) Add a pipeline register for the valid signal at each pipeline stage.
// (3) Add an output port for the valid signal from the final stage of the
//     pipeline.
// (4) Use the (pipelined) valid signal as the load enable signal for other
//     pipeline registers in each stage. This is a power optimization
//     which reduces switching in the data path when the valid signal is
//     deasserted.
// TODO(meheff): 2021/08/21 This might be better performed as a codegen pass.
struct ValidPorts {
  InputPort* input;
  OutputPort* output;
};

static absl::StatusOr<ValidPorts> AddValidSignal(
    absl::Span<const PipelineStageRegisters> pipeline_registers,
    const CodegenOptions& options, ResetInfo reset_info, Block* block) {
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

  // Plumb valid signal through the pipeline stages. Gather the pipelined valid
  // signal in a vector where the zero-th element is the input port and
  // subsequent elements are the pipelined valid signal from each stage.
  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> pipelined_valids,
      MakePipelineStagesForValid(valid_input_port, pipeline_registers,
                                 reset_info, block));

  // Use the pipelined valid signal as load enable each datapath  pipeline
  // register in each stage as a power optimization.
  for (int64_t stage = 0; stage < pipeline_registers.size(); ++stage) {
    // For each (non-valid-signal) pipeline register add `valid` or `valid ||
    // reset` (if reset exists) as a load enable. The `reset` term ensures the
    // pipeline flushes when reset is enabled.
    if (!pipeline_registers.at(stage).empty()) {
      XLS_ASSIGN_OR_RETURN(
          Node * load_enable,
          MakeOrWithResetNode(pipelined_valids[stage],
                              PipelineSignalName("load_en", stage), reset_info,
                              block));

      for (const PipelineRegister& pipeline_reg :
           pipeline_registers.at(stage)) {
        XLS_RETURN_IF_ERROR(block
                                ->MakeNode<RegisterWrite>(
                                    /*loc=*/absl::nullopt,
                                    pipeline_reg.reg_write->data(),
                                    /*load_enable=*/load_enable,
                                    /*reset=*/absl::nullopt, pipeline_reg.reg)
                                .status());
        XLS_RETURN_IF_ERROR(block->RemoveNode(pipeline_reg.reg_write));
      }
    }
  }

  // Add valid output port.
  if (options.valid_control()->output_name().empty()) {
    return absl::InvalidArgumentError(
        "Must specify output name of valid signal.");
  }
  XLS_ASSIGN_OR_RETURN(
      OutputPort * valid_output_port,
      block->AddOutputPort(options.valid_control()->output_name(),
                           pipelined_valids.back()));

  return ValidPorts{valid_input_port, valid_output_port};
}

// Data structures holding the data and (optional) predicate nodes representing
// streaming inputs (receive over streaming channel) and streaming outputs (send
// over streaming channel) in the generated block.
struct StreamingInput {
  InputPort* port;
  InputPort* port_valid;
  OutputPort* port_ready;
  Channel* channel;
  absl::optional<Node*> predicate;
};

struct StreamingOutput {
  OutputPort* port;
  OutputPort* port_valid;
  InputPort* port_ready;
  Channel* channel;
  absl::optional<Node*> predicate;
};

struct StreamingIoPipeline {
  std::vector<StreamingInput> inputs;
  std::vector<StreamingOutput> outputs;
  std::vector<PipelineStageRegisters> pipeline_registers;
  absl::optional<StateRegister> state_register;
  absl::optional<OutputPort*> idle_port;
};

// For each output streaming channel add a corresponding ready port (input
// port). Combinationally combine those ready signals with their predicates to
// generate an  all_active_outputs_ready signal.
//
// Upon success returns a Node* to the all_active_inputs_valid signal.
static absl::StatusOr<Node*> MakeOutputReadyPortsForOutputChannels(
    absl::Span<StreamingOutput> streaming_outputs,
    absl::string_view ready_suffix, Block* block) {
  // Add a ready input port for each streaming output. Gather the ready signals
  // into a vector. Ready signals from streaming outputs generated from Send
  // operations are conditioned upon the optional predicate value.
  std::vector<Node*> active_readys;
  for (StreamingOutput& streaming_output : streaming_outputs) {
    XLS_ASSIGN_OR_RETURN(
        streaming_output.port_ready,
        block->AddInputPort(
            absl::StrCat(streaming_output.channel->name(), ready_suffix),
            block->package()->GetBitsType(1)));

    if (streaming_output.predicate.has_value()) {
      // Logic for the active ready signal for a Send operation with a
      // predicate `pred`.
      //
      //   active = !pred | pred && ready
      //          = !pred | ready
      XLS_ASSIGN_OR_RETURN(
          Node * not_pred,
          block->MakeNode<UnOp>(absl::nullopt,
                                streaming_output.predicate.value(), Op::kNot));
      std::vector<Node*> operands = {not_pred, streaming_output.port_ready};
      XLS_ASSIGN_OR_RETURN(
          Node * active_ready,
          block->MakeNode<NaryOp>(absl::nullopt, operands, Op::kOr));
      active_readys.push_back(active_ready);
    } else {
      active_readys.push_back(streaming_output.port_ready);
    }
  }

  // And reduce all the active ready signals. This signal is true iff all active
  // outputs are ready.
  Node* all_active_outputs_ready;
  if (active_readys.empty()) {
    XLS_ASSIGN_OR_RETURN(
        all_active_outputs_ready,
        block->MakeNode<xls::Literal>(absl::nullopt, Value(UBits(1, 1))));
  } else {
    XLS_ASSIGN_OR_RETURN(
        all_active_outputs_ready,
        block->MakeNode<NaryOp>(absl::nullopt, active_readys, Op::kAnd));
  }

  return all_active_outputs_ready;
}

// For each input streaming channel add a corresponding valid port (input port).
// Combinationally combine those valid signals with their predicates
// to generate an  all_active_inputs_valid signal.
//
// Upon success returns a Node* to the all_active_inputs_valid signal.
static absl::StatusOr<Node*> MakeInputValidPortsForInputChannels(
    absl::Span<StreamingInput> streaming_inputs, absl::string_view valid_suffix,
    Block* block) {
  // Add a valid input port for each streaming input. Gather the valid signals
  // into a vector. Valid signals from streaming inputs generated from Receive
  // operations are conditioned upon the optional predicate value.
  std::vector<Node*> active_valids;
  for (StreamingInput& streaming_input : streaming_inputs) {
    XLS_ASSIGN_OR_RETURN(
        streaming_input.port_valid,
        block->AddInputPort(
            absl::StrCat(streaming_input.channel->name(), valid_suffix),
            block->package()->GetBitsType(1)));

    if (streaming_input.predicate.has_value()) {
      // Logic for the active valid signal for a Receive operation with a
      // predicate `pred`.
      //
      //   active = !pred | pred && valid
      //          = !pred | valid
      XLS_ASSIGN_OR_RETURN(
          Node * not_pred,
          block->MakeNode<UnOp>(absl::nullopt,
                                streaming_input.predicate.value(), Op::kNot));
      std::vector<Node*> operands = {not_pred, streaming_input.port_valid};
      XLS_ASSIGN_OR_RETURN(
          Node * active_valid,
          block->MakeNode<NaryOp>(absl::nullopt, operands, Op::kOr));
      active_valids.push_back(active_valid);
    } else {
      active_valids.push_back(streaming_input.port_valid);
    }
  }

  // And reduce all the active valid signals. This signal is true iff all active
  // inputs are valid.
  Node* all_active_inputs_valid;
  if (active_valids.empty()) {
    XLS_ASSIGN_OR_RETURN(
        all_active_inputs_valid,
        block->MakeNode<xls::Literal>(absl::nullopt, Value(UBits(1, 1))));
  } else {
    XLS_ASSIGN_OR_RETURN(
        all_active_inputs_valid,
        block->MakeNode<NaryOp>(absl::nullopt, active_valids, Op::kAnd));
  }

  return all_active_inputs_valid;
}

// Make valid ports (output) for the output channel.
//
// A valid signal is asserted iff all active
// inputs valid signals are asserted and the predicate of the data channel (if
// any) is asserted.
static absl::Status MakeOutputValidPortsForOutputChannels(
    Node* all_active_inputs_valid,
    absl::Span<StreamingOutput> streaming_outputs,
    absl::string_view valid_suffix, Block* block) {
  for (StreamingOutput& streaming_output : streaming_outputs) {
    Node* valid;
    if (streaming_output.predicate.has_value()) {
      std::vector<Node*> operands = {streaming_output.predicate.value(),
                                     all_active_inputs_valid};
      XLS_ASSIGN_OR_RETURN(
          valid, block->MakeNode<NaryOp>(absl::nullopt, operands, Op::kAnd));
    } else {
      valid = all_active_inputs_valid;
    }
    XLS_ASSIGN_OR_RETURN(
        streaming_output.port_valid,
        block->AddOutputPort(
            absl::StrCat(streaming_output.channel->name(), valid_suffix),
            valid));
  }

  return absl::OkStatus();
}

// Make ready ports (output) for each input channel.
//
// A ready signal is asserted iff all active
// output ready signals are asserted and the predicate of the data channel (if
// any) is asserted.
static absl::Status MakeOutputReadyPortsForInputChannels(
    Node* all_active_outputs_ready, absl::Span<StreamingInput> streaming_inputs,
    absl::string_view ready_suffix, Block* block) {
  for (StreamingInput& streaming_input : streaming_inputs) {
    Node* ready;
    if (streaming_input.predicate.has_value()) {
      std::vector<Node*> operands = {streaming_input.predicate.value(),
                                     all_active_outputs_ready};
      XLS_ASSIGN_OR_RETURN(
          ready, block->MakeNode<NaryOp>(absl::nullopt, operands, Op::kAnd));
    } else {
      ready = all_active_outputs_ready;
    }
    XLS_ASSIGN_OR_RETURN(
        streaming_input.port_ready,
        block->AddOutputPort(
            absl::StrCat(streaming_input.channel->name(), ready_suffix),
            ready));
  }

  return absl::OkStatus();
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

// Adds a register between the node and all its downstream users.
// Returns the new register added.
static absl::StatusOr<RegisterRead*> AddRegisterAfterNode(
    absl::string_view name_prefix, const ResetInfo& reset_info,
    absl::optional<Node*> load_enable, Node* node, Block* block) {
  Type* node_type = node->GetType();

  xls::Reset reset_behavior = reset_info.behavior.value();
  reset_behavior.reset_value = ZeroOfType(node_type);
  Node* reset_node = reset_info.input_port.value();

  std::string name = absl::StrFormat("__%s_reg", name_prefix);

  XLS_ASSIGN_OR_RETURN(Register * reg,
                       block->AddRegister(name, node_type, reset_behavior));

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
                              /*reset=*/reset_node,
                              /*reg=*/reg)
                          .status());

  return reg_read;
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
    absl::string_view name_prefix, const ResetInfo& reset_info, Block* block,
    std::vector<Node*>& valid_nodes) {
  XLS_CHECK_EQ(from_rdy->operand_count(), 1);

  // Add a node for load_enables (will be removed later).
  XLS_ASSIGN_OR_RETURN(
      Node * literal_1,
      block->MakeNode<xls::Literal>(absl::nullopt, Value(UBits(1, 1))));

  // Create data/valid and their skid counterparts.
  XLS_ASSIGN_OR_RETURN(RegisterRead * data_reg_read,
                       AddRegisterAfterNode(name_prefix, reset_info, literal_1,
                                            from_data, block));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_skid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_skid"), reset_info,
                           literal_1, data_reg_read, block));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_valid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_valid"), reset_info,
                           literal_1, from_valid, block));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_valid_skid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_valid_skid"), reset_info,
                           literal_1, data_valid_reg_read, block));

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
          /*default_value=*/absl::nullopt,
          /*name=*/absl::StrCat(name_prefix, "_select")));
  XLS_RETURN_IF_ERROR(data_skid_reg_read->ReplaceUsesWith(to_data));

  // With a skid buffer, input can be accepted whenever the skid registers
  // are empty/invalid.
  Node* to_is_ready = from_rdy->operand(0);

  XLS_ASSIGN_OR_RETURN(
      Node * from_skid_rdy,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/absl::nullopt, data_valid_skid_reg_read, Op::kNot,
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
          /*loc=*/absl::nullopt, std::vector<Node*>{from_valid, from_skid_rdy},
          Op::kAnd, absl::StrCat(name_prefix, "_data_valid_load_en")));

  XLS_ASSIGN_OR_RETURN(RegisterWrite * data_reg_write,
                       block->GetRegisterWrite(data_reg_read->GetRegister()));
  XLS_RETURN_IF_ERROR(
      data_reg_write->ReplaceExistingLoadEnable(input_ready_and_valid));

  XLS_ASSIGN_OR_RETURN(
      Node * data_is_sent_to,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/absl::nullopt,
          std::vector<Node*>{data_valid_reg_read, to_is_ready, from_skid_rdy},
          Op::kAnd, absl::StrCat(name_prefix, "_data_is_sent_to")));

  XLS_ASSIGN_OR_RETURN(
      Node * valid_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/absl::nullopt,
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
                           /*loc=*/absl::nullopt, to_is_ready, Op::kNot,
                           absl::StrCat(name_prefix, "_to_is_not_rdy")));

  XLS_ASSIGN_OR_RETURN(
      Node * skid_data_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/absl::nullopt,
          std::vector<Node*>{data_valid_reg_read, input_ready_and_valid,
                             to_is_not_rdy},
          Op::kAnd, absl::StrCat(name_prefix, "_skid_data_load_en")));

  // Skid is reset (valid set to zero) to invalid whenever
  //   a) skid is valid and
  //   b) output is ready
  XLS_ASSIGN_OR_RETURN(
      Node * skid_valid_set_zero,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/absl::nullopt,
          std::vector<Node*>{data_valid_skid_reg_read, to_is_ready}, Op::kAnd,
          absl::StrCat(name_prefix, "_skid_valid_set_zero")));

  // Skid valid changes from 0 to 1 (load), or 1 to 0 (set zero).
  XLS_ASSIGN_OR_RETURN(
      Node * skid_valid_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/absl::nullopt,
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
//<
// Logic will be inserted immediately after from_data and from node.
// Logic will be inserted immediately before from_rdy,
//   from_rdy must be a node with a single operand.
//
// Returns the node for the register_read of the data.
static absl::StatusOr<RegisterRead*> AddRegisterToRDVNodes(
    Node* from_data, Node* from_valid, Node* from_rdy,
    absl::string_view name_prefix, const ResetInfo& reset_info, Block* block,
    std::vector<Node*>& valid_nodes) {
  XLS_CHECK_EQ(from_rdy->operand_count(), 1);

  XLS_ASSIGN_OR_RETURN(RegisterRead * data_reg_read,
                       AddRegisterAfterNode(name_prefix, reset_info,
                                            absl::nullopt, from_data, block));
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * valid_reg_read,
      AddRegisterAfterNode(absl::StrCat(name_prefix, "_valid"), reset_info,
                           absl::nullopt, from_valid, block));

  // 2. Construct and update the ready signal.
  Node* from_rdy_src = from_rdy->operand(0);
  Register* data_reg = data_reg_read->GetRegister();
  Register* valid_reg = valid_reg_read->GetRegister();

  std::string not_valid_name = absl::StrCat(name_prefix, "_valid_inv");
  XLS_ASSIGN_OR_RETURN(
      Node * not_valid,
      block->MakeNodeWithName<UnOp>(/*loc=*/absl::nullopt, valid_reg_read,
                                    Op::kNot, not_valid_name));

  std::string valid_load_en_name = absl::StrCat(name_prefix, "_valid_load_en");
  XLS_ASSIGN_OR_RETURN(
      Node * valid_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/absl::nullopt, std::vector<Node*>({from_rdy_src, not_valid}),
          Op::kOr, valid_load_en_name));

  std::string data_load_en_name = absl::StrCat(name_prefix, "_load_en");
  XLS_ASSIGN_OR_RETURN(Node * data_load_en,
                       block->MakeNodeWithName<NaryOp>(
                           /*loc=*/absl::nullopt,
                           std::vector<Node*>({from_valid, valid_load_en}),
                           Op::kAnd, data_load_en_name));

  XLS_CHECK_EQ(from_rdy->ReplaceOperand(from_rdy_src, data_load_en), 1);

  // 3. Update load enables for the data and valid registers.
  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(data_load_en, data_reg, block));
  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(valid_load_en, valid_reg, block));

  valid_nodes.push_back(valid_reg_read);

  return data_reg_read;
}

// Adds a register after the input streaming channel's data and valid.
static absl::StatusOr<Node*> AddRegisterAfterStreamingInput(
    StreamingInput& input, const ResetInfo& reset_info, Block* block,
    std::vector<Node*>& valid_nodes) {
  return AddRegisterToRDVNodes(input.port, input.port_valid, input.port_ready,
                               input.port->name(), reset_info, block,
                               valid_nodes);
}

// Adds a register after the input streaming channel's data and valid.
// Returns the node for the register_read of the data.
static absl::StatusOr<Node*> AddRegisterBeforeStreamingOutput(
    StreamingOutput& output, const ResetInfo& reset_info,
    const CodegenOptions& options, Block* block,
    std::vector<Node*>& valid_nodes) {
  // Add an buffers before the data/valid output ports and after
  // the ready input port to serve as points where the
  // additional logic from AddRegisterToRDVNodes() can be inserted.
  std::string data_buf_name =
      absl::StrFormat("__%s_buf", output.port_ready->name());
  std::string valid_buf_name =
      absl::StrFormat("__%s_buf", output.port_ready->name());
  std::string ready_buf_name =
      absl::StrFormat("__%s_buf", output.port_ready->name());

  XLS_ASSIGN_OR_RETURN(Node * output_port_data_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/absl::nullopt, output.port->operand(0),
                           Op::kIdentity, data_buf_name));
  XLS_RETURN_IF_ERROR(
      output.port->ReplaceOperandNumber(0, output_port_data_buf));

  XLS_ASSIGN_OR_RETURN(Node * output_port_valid_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/absl::nullopt, output.port_valid->operand(0),
                           Op::kIdentity, valid_buf_name));
  XLS_RETURN_IF_ERROR(
      output.port_valid->ReplaceOperandNumber(0, output_port_valid_buf));

  XLS_ASSIGN_OR_RETURN(Node * output_port_ready_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/absl::nullopt, output.port_ready,
                           Op::kIdentity, valid_buf_name));

  XLS_RETURN_IF_ERROR(
      output.port_ready->ReplaceUsesWith(output_port_ready_buf));

  if (options.flop_outputs_kind() == CodegenOptions::IOKind::kSkidBuffer) {
    return AddSkidBufferToRDVNodes(output_port_data_buf, output_port_valid_buf,
                                   output_port_ready_buf, output.port->name(),
                                   reset_info, block, valid_nodes);
  }

  if (options.flop_outputs_kind() == CodegenOptions::IOKind::kFlop) {
    return AddRegisterToRDVNodes(output_port_data_buf, output_port_valid_buf,
                                 output_port_ready_buf, output.port->name(),
                                 reset_info, block, valid_nodes);
  }

  return absl::UnimplementedError(absl::StrFormat(
      "Block conversion does not registering output with kind %d",
      options.flop_outputs_kind()));
}

// Adds an input and/or output flop and related signals.
// For streaming inputs, data/valid are registered, but the ready signal
// remains as a feed-through pass.
//
// Ready is asserted if the input is ready and either a) the flop
// is invalid; or b) the flop is valid but the pipeline will accept the data,
// on the next clock tick.
static absl::Status AddInputOutputFlops(const ResetInfo& reset_info,
                                        const CodegenOptions& options,
                                        StreamingIoPipeline& streaming_io,
                                        Block* block,
                                        std::vector<Node*>& valid_nodes) {
  absl::flat_hash_set<Node*> handled_io_nodes;

  // Flop streaming inputs.
  for (StreamingInput& input : streaming_io.inputs) {
    if (options.flop_inputs()) {
      XLS_RETURN_IF_ERROR(
          AddRegisterAfterStreamingInput(input, reset_info, block, valid_nodes)
              .status());

      handled_io_nodes.insert(input.port);
      handled_io_nodes.insert(input.port_valid);
    }

    // ready for an output port is an output for the block,
    // record that we should not separately add a flop these inputs.
    handled_io_nodes.insert(input.port_ready);
  }

  // Flop streaming outputs.
  for (StreamingOutput& output : streaming_io.outputs) {
    if (options.flop_outputs()) {
      XLS_RETURN_IF_ERROR(AddRegisterBeforeStreamingOutput(
                              output, reset_info, options, block, valid_nodes)
                              .status());

      handled_io_nodes.insert(output.port);
      handled_io_nodes.insert(output.port_valid);
    }

    // ready for an output port is an input for the block,
    // record that we should not separately add a flop these inputs.
    handled_io_nodes.insert(output.port_ready);
  }

  // Flop other inputs.
  if (options.flop_inputs()) {
    for (InputPort* port : block->GetInputPorts()) {
      // Skip input ports that
      //  a) Belong to streaming input or output channels.
      //  b) Is the reset port.
      if (handled_io_nodes.contains(port) ||
          port == reset_info.input_port.value()) {
        continue;
      }

      XLS_RETURN_IF_ERROR(AddRegisterAfterNode(port->GetName(), reset_info,
                                               absl::nullopt, port, block)
                              .status());

      handled_io_nodes.insert(port);
    }
  }

  // Flop other outputs
  if (options.flop_outputs()) {
    for (OutputPort* port : block->GetOutputPorts()) {
      // Skip output ports that belong to streaming input or output channels.
      if (handled_io_nodes.contains(port)) {
        continue;
      }

      XLS_RETURN_IF_ERROR(AddRegisterAfterNode(port->GetName(), reset_info,
                                               absl::nullopt, port->operand(0),
                                               block)
                              .status());

      handled_io_nodes.insert(port);
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
static absl::Status AddIdleOutput(std::vector<Node*> valid_nodes,
                                  StreamingIoPipeline& streaming_io,
                                  Block* block) {
  for (StreamingInput& input : streaming_io.inputs) {
    valid_nodes.push_back(input.port_valid);
  }

  for (StreamingOutput& output : streaming_io.outputs) {
    valid_nodes.push_back(output.port_valid->operand(0));
  }

  XLS_ASSIGN_OR_RETURN(
      Node * idle_signal,
      block->MakeNode<NaryOp>(absl::nullopt, valid_nodes, Op::kNor));

  XLS_ASSIGN_OR_RETURN(streaming_io.idle_port,
                       block->AddOutputPort("idle", idle_signal));

  return absl::OkStatus();
}

// Adds ready/valid ports for each of the given streaming inputs/outputs. Also,
// adds logic which propagates ready and valid signals through the block.
//
// Returns a vector of all valids for each stage.  ret[0] is the AND
// of all valids for used channels for the initial stage.  That is, if
// any used input channel is invalid, the node represented by ret[0] will
// be invalid (see MakeInputValidPortsForInputChannels() and
// MakePipelineStagesForValid().
static absl::StatusOr<std::vector<Node*>> AddBubbleFlowControl(
    const ResetInfo& reset_info, const CodegenOptions& options,
    StreamingIoPipeline& streaming_io, Block* block) {
  absl::string_view valid_suffix = options.streaming_channel_valid_suffix();
  absl::string_view ready_suffix = options.streaming_channel_ready_suffix();

  XLS_ASSIGN_OR_RETURN(
      Node * all_active_inputs_valid,
      MakeInputValidPortsForInputChannels(absl::MakeSpan(streaming_io.inputs),
                                          valid_suffix, block));

  XLS_VLOG(3) << "After Inputs";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> pipelined_valids,
                       MakePipelineStagesForValid(
                           all_active_inputs_valid,
                           streaming_io.pipeline_registers, reset_info, block));

  XLS_VLOG(3) << "After Valids";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (streaming_io.state_register.has_value()) {
    XLS_RETURN_IF_ERROR(UpdateStateRegisterWithReset(
        reset_info, streaming_io.state_register.value(), block));
  }

  XLS_VLOG(3) << "After State Updated";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (reset_info.reset_data_path) {
    XLS_RETURN_IF_ERROR(UpdateDatapathRegistersWithReset(
        reset_info, absl::MakeSpan(streaming_io.pipeline_registers), block));

    XLS_VLOG(3) << "After Datapath Reset Updated";
    XLS_VLOG_LINES(3, block->DumpIr());
  }

  XLS_RETURN_IF_ERROR(MakeOutputValidPortsForOutputChannels(
      pipelined_valids.back(), absl::MakeSpan(streaming_io.outputs),
      valid_suffix, block));

  XLS_VLOG(3) << "After Outputs Valid";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_ASSIGN_OR_RETURN(
      Node * all_active_outputs_ready,
      MakeOutputReadyPortsForOutputChannels(
          absl::MakeSpan(streaming_io.outputs), ready_suffix, block));

  XLS_VLOG(3) << "After Outputs Ready";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_ASSIGN_OR_RETURN(Node * input_stage_enable,
                       UpdatePipelineWithBubbleFlowControl(
                           all_active_outputs_ready, reset_info,
                           absl::MakeSpan(pipelined_valids),
                           absl::MakeSpan(streaming_io.pipeline_registers),
                           streaming_io.state_register, block));

  XLS_VLOG(3) << "After Bubble Flow Control (pipeline)";
  XLS_VLOG_LINES(3, block->DumpIr());

  // Handle flow control for the single pipeline stage case.
  if (streaming_io.pipeline_registers.empty()) {
    XLS_ASSIGN_OR_RETURN(input_stage_enable,
                         UpdateSingleStagePipelineWithFlowControl(
                             input_stage_enable, pipelined_valids.at(0),
                             streaming_io.state_register, block));
  }

  XLS_VLOG(3) << "After Single Stage Flow Control (state)";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_RETURN_IF_ERROR(MakeOutputReadyPortsForInputChannels(
      input_stage_enable, absl::MakeSpan(streaming_io.inputs), ready_suffix,
      block));

  XLS_VLOG(3) << "After Ready";
  XLS_VLOG_LINES(3, block->DumpIr());

  return pipelined_valids;
}

// Adds ready/valid ports for each of the given streaming inputs/outputs. Also,
// adds logic which propagates ready and valid signals through the block.
static absl::Status AddFlowControl(
    absl::Span<StreamingInput> streaming_inputs,
    absl::Span<StreamingOutput> streaming_outputs,
    const CodegenOptions& options, Block* block) {
  absl::string_view valid_suffix = options.streaming_channel_valid_suffix();
  absl::string_view ready_suffix = options.streaming_channel_ready_suffix();

  XLS_ASSIGN_OR_RETURN(Node * all_active_outputs_ready,
                       MakeOutputReadyPortsForOutputChannels(
                           streaming_outputs, ready_suffix, block));

  XLS_ASSIGN_OR_RETURN(Node * all_active_inputs_valid,
                       MakeInputValidPortsForInputChannels(
                           streaming_inputs, valid_suffix, block));

  XLS_RETURN_IF_ERROR(MakeOutputValidPortsForOutputChannels(
      all_active_inputs_valid, streaming_outputs, valid_suffix, block));

  XLS_RETURN_IF_ERROR(MakeOutputReadyPortsForInputChannels(
      all_active_outputs_ready, streaming_inputs, ready_suffix, block));

  return absl::OkStatus();
}

// Send/receive nodes are not cloned from the proc into the block, but the
// network of tokens connecting these send/receive nodes *is* cloned. This
// function removes the token operations.
static absl::Status RemoveDeadTokenNodes(Block* block) {
  // Receive nodes produce a tuple of a token and a data value. In the block
  // this becomes a tuple of a token and an InputPort. Run tuple simplification
  // to disentangle the tuples so DCE can do its work and eliminate the token
  // network.
  PassResults pass_results;

  XLS_RETURN_IF_ERROR(
      TupleSimplificationPass()
          .RunOnFunctionBase(block, PassOptions(), &pass_results)
          .status());

  XLS_RETURN_IF_ERROR(
      DeadCodeEliminationPass()
          .RunOnFunctionBase(block, PassOptions(), &pass_results)
          .status());

  CodegenPassUnit unit(block->package(), block);
  CodegenPassOptions pass_options;
  XLS_RETURN_IF_ERROR(RegisterLegalizationPass()
                          .Run(&unit, pass_options, &pass_results)
                          .status());
  XLS_RETURN_IF_ERROR(
      DeadCodeEliminationPass()
          .RunOnFunctionBase(block, PassOptions(), &pass_results)
          .status());

  for (Node* node : block->nodes()) {
    // Nodes like cover and assume have token types and will cause a failure
    // here. Ultimately these operations should *not*
    // have tokens and instead are handled as side-effecting operations.
    XLS_RET_CHECK(!TypeHasToken(node->GetType()));
  }

  return absl::OkStatus();
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
// GetResult() returns a StreamingIoPipeline which
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
//     rcv_x: (token, bits[32]) = receive(tkn, channel_id=0)
//     rcv_x_token: token = tuple_index(rcv_x, index=0)
//     x: bits[32] = tuple_index(rcv_x, index=1)
//     not_x: bits[32] = not(x)
//     snd_y: token = send(rcv_x_token, not_x, channel_id=1)
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
        token_param_(nullptr),
        state_param_(nullptr),
        next_state_node_(nullptr),
        options_(options),
        block_(block) {
    if (is_proc_) {
      Proc* proc = function_base_->AsProcOrDie();
      token_param_ = proc->TokenParam();
      state_param_ = proc->StateParam();
      next_state_node_ = proc->NextState();
    }

    if (stage_count > 1) {
      result_.pipeline_registers.resize(stage_count - 1);
    }
  }

  // For a given set of sorted nodes, process and clone them into the
  // block.
  absl::Status CloneNodes(absl::Span<Node* const> sorted_nodes, int64_t stage) {
    for (Node* node : sorted_nodes) {
      Node* next_node = nullptr;

      if (node->Is<Param>()) {
        if (is_proc_) {
          if (node == token_param_) {
            XLS_ASSIGN_OR_RETURN(next_node, HandleTokenParam(node));
          } else {
            XLS_RET_CHECK_EQ(stage, 0);
            XLS_RET_CHECK(node == state_param_);
            XLS_ASSIGN_OR_RETURN(next_node, HandleStateParam(node));
          }
        } else {
          XLS_RET_CHECK_EQ(stage, 0);
          XLS_ASSIGN_OR_RETURN(next_node, HandleFunctionParam(node));
        }
      } else if (node->Is<Receive>()) {
        XLS_RET_CHECK(is_proc_);
        XLS_ASSIGN_OR_RETURN(next_node, HandleReceiveNode(node));
      } else if (node->Is<Send>()) {
        XLS_RET_CHECK(is_proc_);
        XLS_ASSIGN_OR_RETURN(next_node, HandleSendNode(node));
      } else if (node == next_state_node_) {
        XLS_RET_CHECK(is_proc_);
        XLS_RET_CHECK_EQ(stage, 0);
        XLS_ASSIGN_OR_RETURN(next_node, HandleNextStateNode(node));
      } else {
        XLS_ASSIGN_OR_RETURN(next_node, HandleGeneralNode(node));
      }

      node_map_[node] = next_node;
    }

    return absl::OkStatus();
  }

  // Add pipeline registers. A register is needed for each node which is
  // scheduled at or before this cycle and has a use after this cycle.
  absl::Status AddNextPipelineStage(const PipelineSchedule& schedule,
                                    int64_t stage) {
    Function* as_func = dynamic_cast<Function*>(function_base_);

    for (Node* function_base_node : function_base_->nodes()) {
      if (schedule.cycle(function_base_node) > stage) {
        continue;
      }
      auto is_live_out_of_stage = [&](Node* n) {
        if (stage == schedule.length() - 1) {
          return false;
        }
        if (as_func && (n == as_func->return_value())) {
          return true;
        }
        for (Node* user : n->users()) {
          if (schedule.cycle(user) > stage) {
            return true;
          }
        }
        return false;
      };

      if (is_live_out_of_stage(function_base_node)) {
        Node* node = node_map_.at(function_base_node);

        XLS_ASSIGN_OR_RETURN(
            Node * node_after_stage,
            CreatePipelineRegistersForNode(
                PipelineSignalName(node->GetName(), stage), node,
                result_.pipeline_registers.at(stage), block_));

        node_map_[function_base_node] = node_after_stage;
      }
    }

    return absl::OkStatus();
  }

  // If a function, create an output port for the function's return.
  absl::Status AddOutputPortsIfFunction() {
    if (!is_proc_) {
      Function* function = function_base_->AsFunctionOrDie();
      return block_
          ->AddOutputPort(kOutputPortName,
                          node_map_.at(function->return_value()))
          .status();
    }

    return absl::OkStatus();
  }

  // Return structure describing streaming io ports and pipeline registers.
  StreamingIoPipeline GetResult() { return result_; }

 private:
  // Replace token parameter with zero operand AfterAll.
  absl::StatusOr<Node*> HandleTokenParam(Node* node) {
    return block_->MakeNode<AfterAll>(node->loc(), std::vector<Node*>());
  }

  // Replace state parameter with Literal empty tuple.
  absl::StatusOr<Node*> HandleStateParam(Node* node) {
    Proc* proc = function_base_->AsProcOrDie();

    if (node->GetType()->GetFlatBitCount() == 0) {
      if (node->GetType() != proc->package()->GetTupleType({})) {
        return absl::UnimplementedError(
            absl::StrFormat("Proc has no state, but (state type is not"
                            "empty tuple), instead got %s.",
                            node->GetType()->ToString()));
      }

      return block_->MakeNode<xls::Literal>(node->loc(), Value::Tuple({}));
    }

    // Create a temporary name as this register will later be removed
    // and updated.  That register should be created with the
    // state parameter's name.  See UpdateStateRegisterWithReset().
    std::string name = block_->UniquifyNodeName(
        absl::StrCat("__", proc->StateParam()->name()));

    XLS_ASSIGN_OR_RETURN(Register * reg,
                         block_->AddRegister(name, node->GetType()));

    // Register write will be created later in HandleNextState.

    XLS_ASSIGN_OR_RETURN(
        RegisterRead * reg_read,
        block_->MakeNodeWithName<RegisterRead>(node->loc(), reg,
                                               /*name=*/reg->name()));

    result_.state_register = StateRegister{
        std::string(proc->StateParam()->name()), proc->InitValue(), reg,
        /*reg_write=*/nullptr, reg_read};

    return reg_read;
  }

  // Replace function parameters with input ports.
  absl::StatusOr<Node*> HandleFunctionParam(Node* node) {
    Param* param = node->As<Param>();
    return block_->AddInputPort(param->GetName(), param->GetType(),
                                param->loc());
  }

  // Don't clone Receive operations. Instead replace with a tuple
  // containing the Receive's token operand and an InputPort operation.
  absl::StatusOr<Node*> HandleReceiveNode(Node* node) {
    Node* next_node;

    Receive* receive = node->As<Receive>();
    XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
    absl::string_view data_suffix =
        (channel->kind() == ChannelKind::kStreaming)
            ? options_.streaming_channel_data_suffix()
            : "";
    XLS_ASSIGN_OR_RETURN(
        InputPort * input_port,
        block_->AddInputPort(absl::StrCat(channel->name(), data_suffix),
                             channel->type()));
    XLS_ASSIGN_OR_RETURN(
        next_node,
        block_->MakeNode<Tuple>(
            node->loc(),
            std::vector<Node*>({node_map_.at(node->operand(0)), input_port})));

    if (channel->kind() == ChannelKind::kSingleValue) {
      return next_node;
    }

    XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kStreaming);
    XLS_RET_CHECK_EQ(down_cast<StreamingChannel*>(channel)->flow_control(),
                     FlowControl::kReadyValid);

    StreamingInput streaming_input;
    streaming_input.port = input_port;
    streaming_input.port_valid = nullptr;
    streaming_input.port_ready = nullptr;
    streaming_input.channel = channel;

    if (receive->predicate().has_value()) {
      streaming_input.predicate = node_map_.at(receive->predicate().value());
    }
    result_.inputs.push_back(streaming_input);

    return next_node;
  }

  // Don't clone Send operations. Instead replace with an OutputPort
  // operation in the block.
  absl::StatusOr<Node*> HandleSendNode(Node* node) {
    Node* next_node;

    XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
    Send* send = node->As<Send>();
    absl::string_view data_suffix =
        (channel->kind() == ChannelKind::kStreaming)
            ? options_.streaming_channel_data_suffix()
            : "";
    XLS_ASSIGN_OR_RETURN(
        OutputPort * output_port,
        block_->AddOutputPort(absl::StrCat(channel->name(), data_suffix),
                              node_map_.at(send->data())));
    // Map the Send node to the token operand of the Send in the
    // block.
    next_node = node_map_.at(send->token());

    if (channel->kind() == ChannelKind::kSingleValue) {
      return next_node;
    }

    XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kStreaming);
    XLS_RET_CHECK_EQ(down_cast<StreamingChannel*>(channel)->flow_control(),
                     FlowControl::kReadyValid);

    StreamingOutput streaming_output;
    streaming_output.port = output_port;
    streaming_output.port_valid = nullptr;
    streaming_output.port_ready = nullptr;
    streaming_output.channel = channel;
    if (send->predicate().has_value()) {
      streaming_output.predicate = node_map_.at(send->predicate().value());
    }

    result_.outputs.push_back(streaming_output);

    return next_node;
  }

  // Clone the operation and then write to the state register.
  absl::StatusOr<Node*> HandleNextStateNode(Node* node) {
    XLS_ASSIGN_OR_RETURN(Node * next_state, HandleGeneralNode(node));

    if (node->GetType()->GetFlatBitCount() == 0) {
      Proc* proc = function_base_->AsProcOrDie();

      if (node->GetType() != proc->package()->GetTupleType({})) {
        return absl::UnimplementedError(
            absl::StrFormat("Proc has no state, but (state type is not"
                            "empty tuple), instead got %s.",
                            node->GetType()->ToString()));
      }

      return next_state;
    }

    if (!result_.state_register.has_value()) {
      return absl::InternalError(
          absl::StrFormat("Expected next state node %s to be dependent "
                          "on state",
                          node->ToString()));
    }

    // There should only be one next state node.
    XLS_CHECK_EQ(result_.state_register->reg_write, nullptr);

    XLS_ASSIGN_OR_RETURN(
        result_.state_register->reg_write,
        block_->MakeNode<RegisterWrite>(node->loc(), next_state,
                                        /*load_enable=*/absl::nullopt,
                                        /*reset=*/absl::nullopt,
                                        result_.state_register->reg));

    // For propagation in the fanout, the next_state data in is used
    // instead of the register.
    return next_state;
  }

  // Clone the operation from the source to the block as is.
  absl::StatusOr<Node*> HandleGeneralNode(Node* node) {
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map_.at(operand));
    }
    return node->CloneInNewFunction(new_operands, block_);
  }

  // Create a pipeline register for the given node.
  //
  // Returns a PipelineRegister whose reg_read field can be used
  // to chain dependent ops to.
  absl::StatusOr<PipelineRegister> CreatePipelineRegister(
      absl::string_view name, Node* node, Block* block) {
    XLS_ASSIGN_OR_RETURN(Register * reg,
                         block_->AddRegister(name, node->GetType()));
    XLS_ASSIGN_OR_RETURN(
        RegisterWrite * reg_write,
        block_->MakeNode<RegisterWrite>(node->loc(), node,
                                        /*load_enable=*/absl::nullopt,
                                        /*reset=*/absl::nullopt, reg));
    XLS_ASSIGN_OR_RETURN(
        RegisterRead * reg_read,
        block_->MakeNodeWithName<RegisterRead>(node->loc(), reg,
                                               /*name=*/reg->name()));
    return PipelineRegister{reg, reg_write, reg_read};
  }

  // Returns true if tuple_type has a zero width element at the top level.
  bool HasZeroWidthType(TupleType* tuple_type) {
    XLS_CHECK(tuple_type != nullptr);

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
      absl::string_view base_name, Node* node,
      std::vector<PipelineRegister>& pipeline_registers_list, Block* block) {
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
          XLS_ASSIGN_OR_RETURN(Node * split_node, block_->MakeNode<TupleIndex>(
                                                      node->loc(), node, i));

          XLS_ASSIGN_OR_RETURN(PipelineRegister pipe_reg,
                               CreatePipelineRegister(
                                   absl::StrFormat("%s_index%d", base_name, i),
                                   split_node, block));

          split_registers.at(i) = pipe_reg.reg_read;
          pipeline_registers_list.push_back(pipe_reg);
        }

        // Reconstruct tuple for the rest of the graph.
        XLS_ASSIGN_OR_RETURN(
            Node * merge_after_reg_read,
            block_->MakeNode<Tuple>(node->loc(), split_registers));

        return merge_after_reg_read;
      }
    }

    // Create a single register to store the node
    XLS_ASSIGN_OR_RETURN(PipelineRegister pipe_reg,
                         CreatePipelineRegister(base_name, node, block));

    pipeline_registers_list.push_back(pipe_reg);
    return pipe_reg.reg_read;
  }

  bool is_proc_;
  FunctionBase* function_base_;
  Node* token_param_;
  Node* state_param_;
  Node* next_state_node_;

  const CodegenOptions& options_;

  Block* block_;
  StreamingIoPipeline result_;
  absl::flat_hash_map<Node*, Node*> node_map_;
};

// Adds the nodes in the given schedule to the block. Pipeline registers are
// inserted between stages and returned as a vector indexed by cycle. The block
// should be empty prior to calling this function.
static absl::StatusOr<StreamingIoPipeline> CloneNodesIntoPipelinedBlock(
    const PipelineSchedule& schedule, const CodegenOptions& options,
    Block* block) {
  FunctionBase* function_base = schedule.function_base();
  XLS_RET_CHECK(function_base->IsProc() || function_base->IsFunction());

  CloneNodesIntoBlockHandler cloner(function_base, schedule.length(), options,
                                    block);
  for (int64_t stage = 0; stage < schedule.length(); ++stage) {
    XLS_RET_CHECK_OK(cloner.CloneNodes(schedule.nodes_in_cycle(stage), stage));
    XLS_RET_CHECK_OK(cloner.AddNextPipelineStage(schedule, stage));
  }

  XLS_RET_CHECK_OK(cloner.AddOutputPortsIfFunction());

  return cloner.GetResult();
}

// Clones every node in the given proc into the given block. Some nodes are
// handled specially.  See CloneNodesIntoBlockHandler for details.
static absl::StatusOr<StreamingIoPipeline> CloneProcNodesIntoBlock(
    Proc* proc, const CodegenOptions& options, Block* block) {
  CloneNodesIntoBlockHandler cloner(proc, /*stage_count=*/0, options, block);
  XLS_RET_CHECK_OK(cloner.CloneNodes(TopoSort(proc).AsVector(), /*stage=*/0));
  return cloner.GetResult();
}

}  // namespace

std::string PipelineSignalName(absl::string_view root, int64_t stage) {
  std::string base;
  // Strip any existing pipeline prefix from the name.
  if (!RE2::PartialMatch(root, R"(^p\d+_(.+))", &base)) {
    base = root;
  }
  return absl::StrFormat("p%d_%s", stage, SanitizeIdentifier(base));
}

absl::StatusOr<Block*> FunctionToPipelinedBlock(
    const PipelineSchedule& schedule, const CodegenOptions& options,
    Function* f) {
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

  std::string block_name(
      options.module_name().value_or(SanitizeIdentifier(f->name())));

  Block* block =
      f->package()->AddBlock(std::make_unique<Block>(block_name, f->package()));

  if (!options.clock_name().has_value()) {
    return absl::InvalidArgumentError(
        "Clock name must be specified when generating a pipelined block");
  }
  XLS_RETURN_IF_ERROR(block->AddClockPort(options.clock_name().value()));

  // Flopping inputs and outputs can be handled as a transformation to the
  // schedule. This makes the later code for creation of the pipeline simpler.
  // TODO(meheff): 2021/7/21 Add input/output flopping as an option to the
  // scheduler.
  XLS_ASSIGN_OR_RETURN(PipelineSchedule transformed_schedule,
                       MaybeAddInputOutputFlopsToSchedule(schedule, options));

  XLS_ASSIGN_OR_RETURN(
      StreamingIoPipeline streaming_io_and_pipeline,
      CloneNodesIntoPipelinedBlock(transformed_schedule, options, block));

  XLS_ASSIGN_OR_RETURN(ResetInfo reset_info, MaybeAddResetPort(block, options));

  absl::optional<ValidPorts> valid_ports;
  if (options.valid_control().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        valid_ports,
        AddValidSignal(streaming_io_and_pipeline.pipeline_registers, options,
                       reset_info, block));
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
  if (reset_info.input_port.has_value()) {
    port_order.push_back(reset_info.input_port.value()->GetName());
  }
  if (valid_ports.has_value()) {
    port_order.push_back(valid_ports->input->GetName());
  }
  for (Param* param : f->params()) {
    port_order.push_back(param->GetName());
  }
  if (valid_ports.has_value()) {
    port_order.push_back(valid_ports->output->GetName());
  }
  port_order.push_back(kOutputPortName);
  XLS_RETURN_IF_ERROR(block->ReorderPorts(port_order));

  return block;
}

absl::StatusOr<Block*> ProcToPipelinedBlock(const PipelineSchedule& schedule,
                                            const CodegenOptions& options,
                                            Proc* proc) {
  XLS_VLOG(3) << "Converting proc to pipelined block:";
  XLS_VLOG_LINES(3, proc->DumpIr());

  std::string block_name(
      options.module_name().value_or(SanitizeIdentifier(proc->name())));

  if (options.manual_control().has_value()) {
    return absl::UnimplementedError("Manual pipeline control not implemented");
  }
  if (options.split_outputs()) {
    return absl::UnimplementedError("Splitting outputs not supported.");
  }
  if (options.manual_control().has_value()) {
    return absl::UnimplementedError("Manual pipeline control not implemented");
  }

  Block* block = proc->package()->AddBlock(
      std::make_unique<Block>(block_name, proc->package()));

  XLS_RETURN_IF_ERROR(block->AddClockPort("clk"));

  XLS_VLOG(3) << "Schedule Used";
  XLS_VLOG_LINES(3, schedule.ToString());

  XLS_ASSIGN_OR_RETURN(StreamingIoPipeline streaming_io_and_pipeline,
                       CloneNodesIntoPipelinedBlock(schedule, options, block));

  if (streaming_io_and_pipeline.outputs.size() != 1) {
    // TODO(tedhong): 2021-09-27 Add additional logic to ensure that
    // when output channels are ready at different times, no data is lost
    // or received twice.

    // TODO(tedhong): 2021-10-22 Add an additional check that
    // in the case of mutually exclusive streaming outputs (at most one output
    // is written at a time, then logic can be simplified.

    XLS_LOG(WARNING) << absl::StrFormat(
        "Proc pipeline generator only supports streaming "
        "output channels which are guaranteed to be mutually "
        "exclusive, got %d output channels which were not "
        "checked for this condition",
        streaming_io_and_pipeline.outputs.size());
  }

  XLS_VLOG(3) << "After Pipeline";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_ASSIGN_OR_RETURN(ResetInfo reset_info, MaybeAddResetPort(block, options));

  XLS_ASSIGN_OR_RETURN(std::vector<Node*> pipelined_valids,
                       AddBubbleFlowControl(reset_info, options,
                                            streaming_io_and_pipeline, block));

  XLS_VLOG(3) << "After Flow Control";
  XLS_VLOG_LINES(3, block->DumpIr());

  // Initialize the valid flops to the pipeline registers.
  // First element is skipped as the initial stage valid flops will be
  // constructed from the input flops and/or input valid ports and will be
  // added later in AddInputFlops() and AddIdleOutput().
  XLS_CHECK_GE(pipelined_valids.size(), 1);
  std::vector<Node*> valid_flops(pipelined_valids.begin() + 1,
                                 pipelined_valids.end());

  if (options.flop_inputs() || options.flop_outputs()) {
    XLS_RETURN_IF_ERROR(AddInputOutputFlops(
        reset_info, options, streaming_io_and_pipeline, block, valid_flops));
  }
  XLS_VLOG(3) << "After Input or Output Flops";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (options.add_idle_output()) {
    XLS_RETURN_IF_ERROR(
        AddIdleOutput(valid_flops, streaming_io_and_pipeline, block));
  }
  XLS_VLOG(3) << "After Add Idle Output";
  XLS_VLOG_LINES(3, block->DumpIr());

  // TODO(tedhong): 2021-09-23 Remove and add any missing functionality to
  //                codegen pipeline.
  XLS_RETURN_IF_ERROR(RemoveDeadTokenNodes(block));

  XLS_VLOG(3) << "After RemoveDeadTokenNodes";
  XLS_VLOG_LINES(3, block->DumpIr());

  return block;
}

absl::StatusOr<Block*> FunctionToCombinationalBlock(
    Function* f, absl::string_view block_name) {
  Block* block =
      f->package()->AddBlock(std::make_unique<Block>(block_name, f->package()));

  // A map from the nodes in 'f' to their corresponding node in the block.
  absl::flat_hash_map<Node*, Node*> node_map;

  // Emit the parameters first to ensure the their order is preserved in the
  // block.
  for (Param* param : f->params()) {
    XLS_ASSIGN_OR_RETURN(
        node_map[param],
        block->AddInputPort(param->GetName(), param->GetType(), param->loc()));
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

  XLS_RETURN_IF_ERROR(
      block->AddOutputPort(kOutputPortName, node_map.at(f->return_value()))
          .status());

  return block;
}

absl::StatusOr<Block*> ProcToCombinationalBlock(Proc* proc,
                                                absl::string_view block_name,
                                                const CodegenOptions& options) {
  XLS_VLOG(3) << "Converting proc to combinational block:";
  XLS_VLOG_LINES(3, proc->DumpIr());

  // In a combinational module, the proc cannot have any state to avoid
  // combinational loops. That is, the loop state must be an empty tuple.
  if (proc->StateType() != proc->package()->GetTupleType({})) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Proc must have no state (state type is empty tuple) "
                        "when lowering to a combinational block. "
                        "Proc state is: %s",
                        proc->StateType()->ToString()));
  }

  Block* block = proc->package()->AddBlock(
      std::make_unique<Block>(block_name, proc->package()));

  XLS_ASSIGN_OR_RETURN(StreamingIoPipeline streaming_io,
                       CloneProcNodesIntoBlock(proc, options, block));

  XLS_RET_CHECK_EQ(streaming_io.pipeline_registers.size(), 0);

  XLS_RETURN_IF_ERROR(AddFlowControl(absl::MakeSpan(streaming_io.inputs),
                                     absl::MakeSpan(streaming_io.outputs),
                                     options, block));

  // TODO(tedhong): 2021-09-23 Remove and add any missing functionality to
  //                codegen pipeline.
  XLS_RETURN_IF_ERROR(RemoveDeadTokenNodes(block));

  XLS_VLOG_LINES(3, block->DumpIr());
  return block;
}

}  // namespace verilog
}  // namespace xls
