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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "xls/codegen/bdd_io_analysis.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/register_legalization_pass.h"
#include "xls/codegen/vast.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/channel.h"
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

using Stage = int64_t;

// Name of the output port which holds the return value of the function.
// TODO(meheff): 2021-03-01 Allow port names other than "out".
static const char kOutputPortName[] = "out";

// If options specify it, adds and returns an input for a reset signal.
static absl::Status MaybeAddResetPort(Block* block,
                                      const CodegenOptions& options) {
  // TODO(tedhong): 2021-09-18 Combine this with AddValidSignal
  if (options.reset().has_value()) {
    XLS_RET_CHECK_OK(block->AddResetPort(options.reset()->name()));
  }

  return absl::OkStatus();
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
class StateRegister {
 public:
  StateRegister(std::string_view name, Value reset_value, Stage stage,
                Register* reg, RegisterWrite* reg_write, RegisterRead* reg_read)
      : name_(name),
        reset_value_(reset_value),
        stage_(stage),
        reg_(reg),
        reg_write_(reg_write),
        reg_read_(reg_read) {}

  std::string& name() { return name_; }
  Value& reset_value() { return reset_value_; }
  Stage& stage() { return stage_; }
  Register*& reg() { return reg_; }
  RegisterWrite*& reg_write() { return reg_write_; }
  RegisterRead*& reg_read() { return reg_read_; }

  std::string_view name() const { return name_; }
  const Value& reset_value() const { return reset_value_; }
  Stage stage() const { return stage_; }
  Register* reg() const { return reg_; }
  RegisterWrite* reg_write() const { return reg_write_; }
  RegisterRead* reg_read() const { return reg_read_; }

 private:
  std::string name_;
  Value reset_value_;
  Stage stage_;
  Register* reg_;
  RegisterWrite* reg_write_;
  RegisterRead* reg_read_;
};

// The collection of pipeline registers for a single stage.
using PipelineStageRegisters = std::vector<PipelineRegister>;

// Plumb valid signal through the pipeline stages, ANDing with a valid produced
// by each stage. Gather the pipelined valid signal in a vector where the
// zero-th element is the input port and subsequent elements are the pipelined
// valid signal from each stage.
static absl::StatusOr<std::vector<Node*>> MakePipelineStagesForValid(
    Node* valid_input_port,
    absl::Span<const PipelineStageRegisters> pipeline_registers,
    const std::optional<xls::Reset>& reset_behavior, Block* block) {
  Type* u1 = block->package()->GetBitsType(1);

  std::vector<Node*> pipelined_valids(pipeline_registers.size() + 1);
  pipelined_valids[0] = valid_input_port;

  for (int64_t stage = 0; stage < pipeline_registers.size(); ++stage) {
    // Add valid register to each pipeline stage.
    XLS_ASSIGN_OR_RETURN(Register * valid_reg,
                         block->AddRegister(PipelineSignalName("valid", stage),
                                            u1, reset_behavior));
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<RegisterWrite>(
                                /*loc=*/SourceInfo(), pipelined_valids[stage],
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
    absl::Span<Node* const> recvs_valid, absl::Span<Node* const> sends_ready,
    absl::Span<const PipelineStageRegisters> pipeline_registers,
    const std::optional<xls::Reset>& reset_behavior, Block* block,
    std::vector<Node*>& stage_valid, std::vector<Node*>& stage_done) {
  Type* u1 = block->package()->GetBitsType(1);

  // Node denoting if the specific stage's input data is valid.
  // The 0'th stage having no previous stage is always valid.
  stage_valid.resize(pipeline_registers.size() + 1);

  // Node denoting if the specific state is done with its computation.
  // A stage N is done if
  //   a. It's valid (stage[N] == true).
  //   b. All receives are valid (recvs_valid[N] == true).
  //   c. All sends are ready (sends_ready[N] == true).
  stage_done.resize(pipeline_registers.size() + 1);

  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));
  stage_valid[0] = literal_1;

  int64_t stage_count = pipeline_registers.size() + 1;
  for (int64_t stage = 0; stage < stage_count; ++stage) {
    XLS_ASSIGN_OR_RETURN(stage_done[stage], block->MakeNode<xls::NaryOp>(
                                                /*loc=*/SourceInfo(),
                                                std::vector<xls::Node*>{
                                                    stage_valid[stage],
                                                    recvs_valid[stage],
                                                    sends_ready[stage],
                                                },
                                                Op::kAnd));

    if (stage < stage_count - 1) {
      // Only add a valid register if it will be read from and written to.
      XLS_ASSIGN_OR_RETURN(
          Register * valid_reg,
          block->AddRegister(PipelineSignalName("valid", stage), u1,
                             reset_behavior));
      XLS_RETURN_IF_ERROR(block
                              ->MakeNode<RegisterWrite>(
                                  /*loc=*/SourceInfo(), stage_done[stage],
                                  /*load_enable=*/std::nullopt,
                                  /*reset=*/block->GetResetPort(), valid_reg)
                              .status());

      XLS_ASSIGN_OR_RETURN(stage_valid[stage + 1],
                           block->MakeNode<RegisterRead>(
                               /*loc=*/SourceInfo(), valid_reg));
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
  XLS_CHECK_NE(state_register.reg(), nullptr);
  XLS_CHECK_NE(state_register.reg_write(), nullptr);
  XLS_CHECK_NE(state_register.reg_read(), nullptr);

  // Blocks containing a state register must also have a reset signal.
  if (!block->GetResetPort().has_value() || !reset_behavior.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Unable to update state register %s with reset, signal as block"
        " was not created with a reset.",
        state_register.reg()->name()));
  }

  // Follow the reset behavior of the valid registers except for the initial
  // value.
  xls::Reset reset_behavior_with_copied_init = *reset_behavior;
  reset_behavior_with_copied_init.reset_value = state_register.reset_value();

  // Replace the register's reset signal
  return state_register.reg_write()->AddOrReplaceReset(
      block->GetResetPort().value(), reset_behavior_with_copied_init);
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
    absl::Span<Node* const> output_ready_nodes, const CodegenOptions& options,
    absl::Span<Node* const> pipeline_valid_nodes,
    absl::Span<Node* const> pipeline_done_nodes,
    absl::Span<PipelineStageRegisters> pipeline_data_registers,
    absl::Span<std::optional<StateRegister>> state_registers, Block* block) {
  // Create enable signals for each pipeline stage.
  //   - The enable signal for stage N is true either
  //       a. The next stage is empty/not valid
  //         or
  //       b. The next stage will latch data and leave the stage empty
  //     enable_signal[n] = data_enable[n+1] || ! stage_valid[n+1]
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
  //   stage_valid[N+1] = 0 so
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
      pipeline_done_nodes.at(stage_count - 1);

  std::vector<Node*> state_enables;
  state_enables.resize(stage_count, nullptr);
  state_enables.at(stage_count - 1) = pipeline_done_nodes.at(stage_count - 1);

  for (int64_t stage = stage_count - 2; stage >= 0; --stage) {
    // Create load enables for valid registers.
    XLS_ASSIGN_OR_RETURN(
        Node * not_valid_np1,
        block->MakeNodeWithName<UnOp>(
            /*loc=*/SourceInfo(), pipeline_valid_nodes.at(stage + 1), Op::kNot,
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
        pipeline_valid_nodes.at(stage + 1)->As<RegisterRead>();
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
                                           pipeline_done_nodes.at(stage)};
    XLS_ASSIGN_OR_RETURN(Node * data_enable,
                         block->MakeNodeWithName<NaryOp>(
                             SourceInfo(), data_en_operands, Op::kAnd,
                             PipelineSignalName("data_enable", stage)));

    state_enables.at(stage) = data_enable;

    // If datapath registers are reset, then adding reset to the
    // load enable is redundant.
    if (options.reset()->reset_data_path()) {
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
        XLS_RETURN_IF_ERROR(block->RemoveNode(pipeline_reg.reg_write));
        pipeline_reg.reg_write = new_reg_write;
      }
    }
  }

  // Generate writes for state registers. This is done in a separate loop
  // because the last stage isn't included in the pipeline register loop.
  for (std::optional<StateRegister>& state_register : state_registers) {
    if (state_register.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          RegisterWrite * new_reg_write,
          block->MakeNode<RegisterWrite>(
              /*loc=*/state_register->reg_write()->loc(),
              /*data=*/state_register->reg_write()->data(),
              /*load_enable=*/state_enables.at(state_register->stage()),
              /*reset=*/state_register->reg_write()->reset(),
              /*reg=*/state_register->reg_write()->GetRegister()));
      XLS_RETURN_IF_ERROR(block->RemoveNode(state_register->reg_write()));
      state_register->reg_write() = new_reg_write;
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
    absl::Span<std::optional<StateRegister>> state_registers, Block* block) {
  std::vector<Node*> operands = {all_active_outputs_ready,
                                 all_active_inputs_valid};
  XLS_ASSIGN_OR_RETURN(
      Node * pipeline_enable,
      block->MakeNodeWithName<NaryOp>(SourceInfo(), operands, Op::kAnd,
                                      "pipeline_enable"));

  for (std::optional<StateRegister>& state_register : state_registers) {
    if (state_register.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          RegisterWrite * new_reg_write,
          block->MakeNode<RegisterWrite>(
              /*loc=*/state_register->reg_write()->loc(),
              /*data=*/state_register->reg_write()->data(),
              /*load_enable=*/pipeline_enable,
              /*reset=*/state_register->reg_write()->reset(),
              /*reg=*/state_register->reg_write()->GetRegister()));
      XLS_RETURN_IF_ERROR(block->RemoveNode(state_register->reg_write()));
      state_register->reg_write() = new_reg_write;
    }
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
    const CodegenOptions& options,
    Block* block) {
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
      std::vector<Node*> pipelined_valids,
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
          MakeOrWithResetNode(pipelined_valids[stage],
                              PipelineSignalName("load_en", stage),
                              reset_behavior, block));

      for (const PipelineRegister& pipeline_reg :
           pipeline_registers.at(stage)) {
        XLS_RETURN_IF_ERROR(block
                                ->MakeNode<RegisterWrite>(
                                    /*loc=*/SourceInfo(),
                                    pipeline_reg.reg_write->data(),
                                    /*load_enable=*/load_enable,
                                    /*reset=*/std::nullopt, pipeline_reg.reg)
                                .status());
        XLS_RETURN_IF_ERROR(block->RemoveNode(pipeline_reg.reg_write));
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
                             pipelined_valids.back()));
  }

  return ValidPorts{valid_input_port, valid_output_port};
}

// Data structures holding the data and (optional) predicate nodes representing
// streaming inputs (receive over streaming channel) and streaming outputs (send
// over streaming channel) in the generated block.
struct StreamingInput {
  InputPort* port;
  InputPort* port_valid;
  OutputPort* port_ready;

  // signal_data and signal_valid respresent the internal view of the
  // streaming input.  These are used (ex. for handling non-blocking receives)
  // as additional logic are placed between the ports and the pipeline use of
  // these signals.
  //
  // Pictorially:
  //      | port_data   | port_valid   | port_ready
  //  ----|-------------|--------------|-------------
  //  |   |             |              |            |
  //  | |--------------------------|   |            |
  //  | |  Logic / Adapter         |   |            |
  //  | |                          |   |            |
  //  | |--------------------------|   |            |
  //  |   | signal_data | signal_valid |            |
  //  |   |             |              |            |
  //  |                                             |
  //  -----------------------------------------------
  Node* signal_data;
  Node* signal_valid;

  Channel* channel;
  std::optional<Node*> predicate;
};

struct StreamingOutput {
  OutputPort* port;
  OutputPort* port_valid;
  InputPort* port_ready;
  Channel* channel;
  std::optional<Node*> predicate;
};

// Data structures holding the port representing single value inputs/outputs
// in the generated block.
struct SingleValueInput {
  InputPort* port;
  Channel* channel;
};

struct SingleValueOutput {
  OutputPort* port;
  Channel* channel;
};

struct StreamingIOPipeline {
  absl::flat_hash_map<Stage, std::vector<StreamingInput>> inputs;
  absl::flat_hash_map<Stage, std::vector<StreamingOutput>> outputs;
  std::vector<SingleValueInput> single_value_inputs;
  std::vector<SingleValueOutput> single_value_outputs;
  std::vector<PipelineStageRegisters> pipeline_registers;
  // `state_registers` includes an element for each state element in the
  // proc. The vector element is nullopt if the state element is an empty tuple.
  std::vector<std::optional<StateRegister>> state_registers;
  std::optional<OutputPort*> idle_port;

  // Node in block that represents when all output channels (that
  // are predicated true) are ready.
  // See MakeInputReadyPortsForOutputChannels().
  std::vector<Node*> all_active_outputs_ready;
  std::vector<Node*> all_active_inputs_valid;
};

// Update io channel metadata with latest information from block conversion.
absl::Status UpdateChannelMetadata(const StreamingIOPipeline& io,
                                   Block* block) {
  for (const auto& [stage, inputs] : io.inputs) {
    for (const StreamingInput& input : inputs) {
      XLS_CHECK_NE(input.port, nullptr);
      XLS_CHECK_NE(input.port_valid, nullptr);
      XLS_CHECK_NE(input.port_ready, nullptr);
      XLS_CHECK_NE(input.channel, nullptr);

      input.channel->SetBlockName(block->name());
      input.channel->SetDataPortName(input.port->name());
      input.channel->SetValidPortName(input.port_valid->name());
      input.channel->SetReadyPortName(input.port_ready->name());

      XLS_CHECK(input.channel->HasCompletedBlockPortNames());
    }
  }

  for (const auto& [stage, outputs] : io.outputs) {
    for (const StreamingOutput& output : outputs) {
      XLS_CHECK_NE(output.port, nullptr);
      XLS_CHECK_NE(output.port_valid, nullptr);
      XLS_CHECK_NE(output.port_ready, nullptr);
      XLS_CHECK_NE(output.channel, nullptr);

      output.channel->SetBlockName(block->name());
      output.channel->SetDataPortName(output.port->name());
      output.channel->SetValidPortName(output.port_valid->name());
      output.channel->SetReadyPortName(output.port_ready->name());

      XLS_CHECK(output.channel->HasCompletedBlockPortNames());
    }
  }

  for (const SingleValueInput& input : io.single_value_inputs) {
    XLS_CHECK_NE(input.port, nullptr);
    XLS_CHECK_NE(input.channel, nullptr);

    input.channel->SetBlockName(block->name());
    input.channel->SetDataPortName(input.port->name());

    XLS_CHECK(input.channel->HasCompletedBlockPortNames());
  }

  for (const SingleValueOutput& output : io.single_value_outputs) {
    XLS_CHECK_NE(output.port, nullptr);
    XLS_CHECK_NE(output.channel, nullptr);

    output.channel->SetBlockName(block->name());
    output.channel->SetDataPortName(output.port->name());

    XLS_CHECK(output.channel->HasCompletedBlockPortNames());
  }

  return absl::OkStatus();
}

// For each output streaming channel add a corresponding ready port (input
// port). Combinationally combine those ready signals with their predicates to
// generate an  all_active_outputs_ready signal.
//
// Upon success returns a Node* to the all_active_inputs_valid signal.
static absl::StatusOr<std::vector<Node*>> MakeInputReadyPortsForOutputChannels(
    absl::flat_hash_map<Stage, std::vector<StreamingOutput>>& streaming_outputs,
    int64_t stage_count, std::string_view ready_suffix, Block* block) {
  std::vector<Node*> result;

  // Add a ready input port for each streaming output. Gather the ready signals
  // into a vector. Ready signals from streaming outputs generated from Send
  // operations are conditioned upon the optional predicate value.
  for (int64_t i = 0; i < stage_count; ++i) {
    std::vector<Node*> active_readys;
    if (streaming_outputs.contains(i)) {
      for (StreamingOutput& streaming_output : streaming_outputs.at(i)) {
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
              block->MakeNode<UnOp>(
                  SourceInfo(), streaming_output.predicate.value(), Op::kNot));
          std::vector<Node*> operands{not_pred, streaming_output.port_ready};
          XLS_ASSIGN_OR_RETURN(
              Node * active_ready,
              block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kOr));
          active_readys.push_back(active_ready);
        } else {
          active_readys.push_back(streaming_output.port_ready);
        }
      }
    }

    // And reduce all the active ready signals. This signal is true iff all
    // active outputs are ready.
    Node* all_active_outputs_ready;
    if (active_readys.empty()) {
      XLS_ASSIGN_OR_RETURN(
          all_active_outputs_ready,
          block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
    } else {
      XLS_ASSIGN_OR_RETURN(
          all_active_outputs_ready,
          block->MakeNode<NaryOp>(SourceInfo(), active_readys, Op::kAnd));
    }

    result.push_back(all_active_outputs_ready);
  }

  return result;
}

// For each input streaming channel add a corresponding valid port (input port).
// Combinationally combine those valid signals with their predicates
// to generate an all_active_inputs_valid signal.
//
// Upon success returns a Node* to the all_active_inputs_valid signal.
static absl::StatusOr<std::vector<Node*>> MakeInputValidPortsForInputChannels(
    absl::flat_hash_map<Stage, std::vector<StreamingInput>>& streaming_inputs,
    int64_t stage_count, std::string_view valid_suffix, Block* block) {
  std::vector<Node*> result;

  for (int64_t i = 0; i < stage_count; ++i) {
    // Add a valid input port for each streaming input. Gather the valid
    // signals into a vector. Valid signals from streaming inputs generated
    // from Receive operations are conditioned upon the optional predicate
    // value.
    std::vector<Node*> active_valids;
    if (streaming_inputs.contains(i)) {
      for (StreamingInput& streaming_input : streaming_inputs.at(i)) {
        // Input ports for input channels are already created during
        // HandleReceiveNode().
        Node* streaming_input_valid = streaming_input.signal_valid;

        if (streaming_input.predicate.has_value()) {
          // Logic for the active valid signal for a Receive operation with a
          // predicate `pred`.
          //
          //   active = !pred | pred && valid
          //          = !pred | valid
          XLS_ASSIGN_OR_RETURN(
              Node * not_pred,
              block->MakeNode<UnOp>(
                  SourceInfo(), streaming_input.predicate.value(), Op::kNot));
          std::vector<Node*> operands = {not_pred, streaming_input_valid};
          XLS_ASSIGN_OR_RETURN(
              Node * active_valid,
              block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kOr));
          active_valids.push_back(active_valid);
        } else {
          // No predicate is the same as pred = true, so
          // active = !pred | valid = !true | valid = false | valid = valid
          active_valids.push_back(streaming_input_valid);
        }
      }
    }

    // And reduce all the active valid signals. This signal is true iff all
    // active inputs are valid.
    Node* all_active_inputs_valid;
    if (active_valids.empty()) {
      XLS_ASSIGN_OR_RETURN(
          all_active_inputs_valid,
          block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
    } else {
      XLS_ASSIGN_OR_RETURN(
          all_active_inputs_valid,
          block->MakeNode<NaryOp>(SourceInfo(), active_valids, Op::kAnd));
    }

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
    absl::flat_hash_map<Stage, std::vector<StreamingOutput>>& streaming_outputs,
    std::string_view valid_suffix, Block* block) {
  for (auto& [stage, vec] : streaming_outputs) {
    for (StreamingOutput& streaming_output : vec) {
      std::vector<Node*> operands{all_active_inputs_valid.at(stage),
                                  pipelined_valids.at(stage),
                                  next_stage_open.at(stage)};

      if (streaming_output.predicate.has_value()) {
        operands.push_back(streaming_output.predicate.value());
      }

      XLS_ASSIGN_OR_RETURN(Node * valid, block->MakeNode<NaryOp>(
                                             SourceInfo(), operands, Op::kAnd));
      XLS_ASSIGN_OR_RETURN(
          streaming_output.port_valid,
          block->AddOutputPort(
              absl::StrCat(streaming_output.channel->name(), valid_suffix),
              valid));
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
    absl::flat_hash_map<Stage, std::vector<StreamingInput>>& streaming_inputs,
    std::string_view ready_suffix, Block* block) {
  for (auto& [stage, vec] : streaming_inputs) {
    for (StreamingInput& streaming_input : vec) {
      Node* ready = all_active_outputs_ready.at(stage);
      if (streaming_input.predicate.has_value()) {
        std::vector<Node*> operands{streaming_input.predicate.value(),
                                    all_active_outputs_ready.at(stage)};
        XLS_ASSIGN_OR_RETURN(
            ready, block->MakeNode<NaryOp>(SourceInfo(), operands, Op::kAnd));
      }
      XLS_ASSIGN_OR_RETURN(
          streaming_input.port_ready,
          block->AddOutputPort(
              absl::StrCat(streaming_input.channel->name(), ready_suffix),
              ready));
    }
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
    std::vector<Node*>& valid_nodes) {
  XLS_CHECK_EQ(from_rdy->operand_count(), 1);

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
    std::vector<Node*>& valid_nodes) {
  XLS_CHECK_EQ(from_rdy->operand_count(), 1);

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

  XLS_CHECK(from_rdy->ReplaceOperand(from_rdy_src, data_load_en));

  // 3. Update load enables for the data and valid registers.
  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(data_load_en, data_reg, block));
  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(valid_load_en, valid_reg, block));

  valid_nodes.push_back(valid_reg_read);

  return data_reg_read;
}

// Adds a register after the input streaming channel's data and valid.
static absl::StatusOr<Node*> AddRegisterAfterStreamingInput(
    StreamingInput& input, const CodegenOptions& options, Block* block,
    std::vector<Node*>& valid_nodes) {
  const std::optional<xls::Reset> reset_behavior = options.ResetBehavior();

  if (options.flop_inputs_kind() ==
      CodegenOptions::IOKind::kZeroLatencyBuffer) {
    return AddZeroLatencyBufferToRDVNodes(input.port, input.port_valid,
                                          input.port_ready, input.port->name(),
                                          reset_behavior, block, valid_nodes);
  }

  if (options.flop_inputs_kind() == CodegenOptions::IOKind::kSkidBuffer) {
    return AddSkidBufferToRDVNodes(input.port, input.port_valid,
                                   input.port_ready, input.port->name(),
                                   reset_behavior, block, valid_nodes);
  }

  if (options.flop_inputs_kind() == CodegenOptions::IOKind::kFlop) {
    return AddRegisterToRDVNodes(input.port, input.port_valid, input.port_ready,
                                 input.port->name(), reset_behavior, block,
                                 valid_nodes);
  }

  return absl::UnimplementedError(absl::StrFormat(
      "Block conversion does not support registering input with kind %d",
      options.flop_inputs_kind()));
}

// Adds a register after the input streaming channel's data and valid.
// Returns the node for the register_read of the data.
static absl::StatusOr<Node*> AddRegisterBeforeStreamingOutput(
    StreamingOutput& output, const CodegenOptions& options, Block* block,
    std::vector<Node*>& valid_nodes) {
  // Add buffers before the data/valid output ports and after
  // the ready input port to serve as points where the
  // additional logic from AddRegisterToRDVNodes() can be inserted.
  std::string data_buf_name = absl::StrFormat("__%s_buf", output.port->name());
  std::string valid_buf_name =
      absl::StrFormat("__%s_buf", output.port_valid->name());
  std::string ready_buf_name =
      absl::StrFormat("__%s_buf", output.port_ready->name());
  XLS_ASSIGN_OR_RETURN(Node * output_port_data_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), output.port->operand(0),
                           Op::kIdentity, data_buf_name));
  XLS_RETURN_IF_ERROR(
      output.port->ReplaceOperandNumber(0, output_port_data_buf));

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
        output.port->name(), reset_behavior, block, valid_nodes);
  }

  if (options.flop_outputs_kind() == CodegenOptions::IOKind::kSkidBuffer) {
    return AddSkidBufferToRDVNodes(output_port_data_buf, output_port_valid_buf,
                                   output_port_ready_buf, output.port->name(),
                                   reset_behavior, block, valid_nodes);
  }

  if (options.flop_outputs_kind() == CodegenOptions::IOKind::kFlop) {
    return AddRegisterToRDVNodes(output_port_data_buf, output_port_valid_buf,
                                 output_port_ready_buf, output.port->name(),
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
    Block* block, std::vector<Node*>& valid_nodes) {
  absl::flat_hash_set<Node*> handled_io_nodes;

  // Flop streaming inputs.
  for (auto& [stage, vec] : streaming_io.inputs) {
    for (StreamingInput& input : vec) {
      if (options.flop_inputs()) {
        XLS_RETURN_IF_ERROR(
            AddRegisterAfterStreamingInput(input, options,
                                           block, valid_nodes)
                .status());

        handled_io_nodes.insert(input.port);
        handled_io_nodes.insert(input.port_valid);
      }

      // ready for an output port is an output for the block,
      // record that we should not separately add a flop these inputs.
      handled_io_nodes.insert(input.port_ready);
    }
  }

  // Flop streaming outputs.
  for (auto& [stage, vec] : streaming_io.outputs) {
    for (StreamingOutput& output : vec) {
      if (options.flop_outputs()) {
        XLS_RETURN_IF_ERROR(
            AddRegisterBeforeStreamingOutput(output, options,
                                             block, valid_nodes)
                .status());

        handled_io_nodes.insert(output.port);
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
  XLS_CHECK_EQ(from_rdy->operand_count(), 1);
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

  // Data is transfered whenever valid and ready
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
static absl::Status AddOneShotOutputLogic(const CodegenOptions& options,
                                          StreamingIOPipeline& streaming_io,
                                          Block* block) {
  XLS_CHECK(!streaming_io.all_active_outputs_ready.empty());

  for (auto& [stage, vec] : streaming_io.outputs) {
    for (StreamingOutput& output : vec) {
      // Add an buffers before the valid output ports and after
      // the ready input port to serve as points where the
      // additional logic from AddRegisterToRDVNodes() can be inserted.
      std::string valid_buf_name =
          absl::StrFormat("__%s_buf", output.port_valid->name());
      std::string ready_buf_name =
          absl::StrFormat("__%s_buf", output.port_ready->name());

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

      XLS_RETURN_IF_ERROR(AddOneShotLogicToRVNodes(
          output_port_valid_buf, output_port_ready_buf,
          streaming_io.all_active_outputs_ready[stage], output.port->name(),
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
static absl::Status AddIdleOutput(std::vector<Node*> valid_nodes,
                                  StreamingIOPipeline& streaming_io,
                                  Block* block) {
  for (auto& [stage, vec] : streaming_io.inputs) {
    for (StreamingInput& input : vec) {
      valid_nodes.push_back(input.signal_valid);
    }
  }

  for (auto& [stage, vec] : streaming_io.outputs) {
    for (StreamingOutput& output : vec) {
      valid_nodes.push_back(output.port_valid->operand(0));
    }
  }

  XLS_ASSIGN_OR_RETURN(
      Node * idle_signal,
      block->MakeNode<NaryOp>(SourceInfo(), valid_nodes, Op::kNor));

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
    const CodegenOptions& options, StreamingIOPipeline& streaming_io,
    Block* block) {
  int64_t stage_count = streaming_io.pipeline_registers.size() + 1;
  std::string_view valid_suffix = options.streaming_channel_valid_suffix();
  std::string_view ready_suffix = options.streaming_channel_ready_suffix();

  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> all_active_inputs_valid,
      MakeInputValidPortsForInputChannels(streaming_io.inputs, stage_count,
                                          valid_suffix, block));
  streaming_io.all_active_inputs_valid = all_active_inputs_valid;
  XLS_VLOG(3) << "After Inputs Valid";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> all_active_outputs_ready,
      MakeInputReadyPortsForOutputChannels(streaming_io.outputs, stage_count,
                                           ready_suffix, block));
  streaming_io.all_active_outputs_ready = all_active_outputs_ready;

  XLS_VLOG(3) << "After Outputs Ready";
  XLS_VLOG_LINES(3, block->DumpIr());

  std::vector<Node*> stage_valid;
  std::vector<Node*> stage_done;

  std::optional<xls::Reset> reset_behavior = options.ResetBehavior();

  XLS_RETURN_IF_ERROR(MakePipelineStagesForValidIO(
      all_active_inputs_valid, all_active_outputs_ready,
      streaming_io.pipeline_registers, reset_behavior, block, stage_valid,
      stage_done));

  XLS_VLOG(3) << "After Valids";
  XLS_VLOG_LINES(3, block->DumpIr());

  for (std::optional<StateRegister>& state_register :
       streaming_io.state_registers) {
    if (state_register.has_value()) {
      XLS_RETURN_IF_ERROR(UpdateStateRegisterWithReset(
          reset_behavior, state_register.value(), block));
    }
  }

  XLS_VLOG(3) << "After State Updated";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (options.reset()->reset_data_path()) {
    XLS_RETURN_IF_ERROR(UpdateDatapathRegistersWithReset(
        reset_behavior, absl::MakeSpan(streaming_io.pipeline_registers),
        block));

    XLS_VLOG(3) << "After Datapath Reset Updated";
    XLS_VLOG_LINES(3, block->DumpIr());
  }

  XLS_ASSIGN_OR_RETURN(
      BubbleFlowControl bubble_flow_control,
      UpdatePipelineWithBubbleFlowControl(
          absl::MakeSpan(all_active_outputs_ready), options,
          absl::MakeSpan(stage_valid), absl::MakeSpan(stage_done),
          absl::MakeSpan(streaming_io.pipeline_registers),
          absl::MakeSpan(streaming_io.state_registers), block));

  XLS_VLOG(3) << "After Bubble Flow Control (pipeline)";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_RETURN_IF_ERROR(MakeOutputValidPortsForOutputChannels(
      all_active_inputs_valid, stage_valid, bubble_flow_control.next_stage_open,
      streaming_io.outputs, valid_suffix, block));

  XLS_VLOG(3) << "After Outputs Valid";
  XLS_VLOG_LINES(3, block->DumpIr());

  // Handle flow control for the single pipeline stage case.
  if (streaming_io.pipeline_registers.empty()) {
    XLS_ASSIGN_OR_RETURN(
        Node * input_stage_enable,
        UpdateSingleStagePipelineWithFlowControl(
            bubble_flow_control.data_load_enable.front(), stage_done.at(0),
            absl::MakeSpan(streaming_io.state_registers), block));
    bubble_flow_control.data_load_enable = {input_stage_enable};
  }

  XLS_VLOG(3) << "After Single Stage Flow Control (state)";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_RETURN_IF_ERROR(MakeOutputReadyPortsForInputChannels(
      bubble_flow_control.data_load_enable, streaming_io.inputs, ready_suffix,
      block));

  XLS_VLOG(3) << "After Ready";
  XLS_VLOG_LINES(3, block->DumpIr());

  return stage_valid;
}

// Adds ready/valid ports for each of the given streaming inputs/outputs. Also,
// adds logic which propagates ready and valid signals through the block.
static absl::Status AddCombinationalFlowControl(
    absl::flat_hash_map<Stage, std::vector<StreamingInput>>& streaming_inputs,
    absl::flat_hash_map<Stage, std::vector<StreamingOutput>>& streaming_outputs,
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

  // Nodes like cover and assert have token types and will cause
  // a dangling token network remaining.
  //
  // TODO(tedhong): 2022-02-14, clean up dangling token
  // network to ensure that deleted nodes can't be accessed via normal
  // ir operations.

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
        options_(options),
        block_(block) {
    if (is_proc_) {
      Proc* proc = function_base_->AsProcOrDie();
      token_param_ = proc->TokenParam();
      for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
        next_state_nodes_[proc->GetNextStateElement(i)].push_back(i);
      }
      result_.state_registers.resize(proc->GetStateElementCount());
    }
    if (stage_count > 1) {
      result_.pipeline_registers.resize(stage_count - 1);
    }
    for (int64_t i = 0; i < stage_count; ++i) {
      result_.inputs[i];
      result_.outputs[i];
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
            XLS_ASSIGN_OR_RETURN(
                int64_t index,
                node->function_base()->AsProcOrDie()->GetStateParamIndex(
                    node->As<Param>()));
            XLS_ASSIGN_OR_RETURN(next_node,
                                 HandleStateParam(node, stage, index));
          }
        } else {
          XLS_ASSIGN_OR_RETURN(next_node, HandleFunctionParam(node));
        }
        node_map_[node] = next_node;
      } else if (node->Is<Receive>()) {
        XLS_RET_CHECK(is_proc_);
        XLS_ASSIGN_OR_RETURN(next_node, HandleReceiveNode(node, stage));
      } else if (node->Is<Send>()) {
        XLS_RET_CHECK(is_proc_);
        XLS_ASSIGN_OR_RETURN(next_node, HandleSendNode(node, stage));
      } else {
        XLS_ASSIGN_OR_RETURN(next_node, HandleGeneralNode(node));
      }
      node_map_[node] = next_node;
    }

    // After all nodes have been cloned, handle writing of next state values
    // into state registers.
    if (is_proc_) {
      for (Node* node : sorted_nodes) {
        if (next_state_nodes_.contains(node)) {
          XLS_RETURN_IF_ERROR(
              SetNextStateNode(node_map_.at(node), next_state_nodes_.at(node)));
        }
      }
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
        if ((as_func != nullptr) && (n == as_func->return_value())) {
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
  StreamingIOPipeline GetResult() { return result_; }

 private:
  // Replace token parameter with zero operand AfterAll.
  absl::StatusOr<Node*> HandleTokenParam(Node* node) {
    return block_->MakeNode<AfterAll>(node->loc(), std::vector<Node*>());
  }

  // Replace state parameter at the given index with Literal empty tuple.
  absl::StatusOr<Node*> HandleStateParam(Node* node, Stage stage,
                                         int64_t index) {
    XLS_CHECK_GE(stage, 0);

    Proc* proc = function_base_->AsProcOrDie();

    if (node->GetType()->GetFlatBitCount() == 0) {
      if (node->GetType() != proc->package()->GetTupleType({})) {
        return absl::UnimplementedError(
            absl::StrFormat("Proc has zero-width state element %d, but type is "
                            "not empty tuple, instead got %s.",
                            index, node->GetType()->ToString()));
      }

      return block_->MakeNode<xls::Literal>(node->loc(), Value::Tuple({}));
    }

    // Create a temporary name as this register will later be removed
    // and updated.  That register should be created with the
    // state parameter's name.  See UpdateStateRegisterWithReset().
    std::string name = block_->UniquifyNodeName(
        absl::StrCat("__", proc->GetStateParam(index)->name()));

    XLS_ASSIGN_OR_RETURN(Register * reg,
                         block_->AddRegister(name, node->GetType()));

    // Register write will be created later in HandleNextState.
    XLS_ASSIGN_OR_RETURN(
        RegisterRead * reg_read,
        block_->MakeNodeWithName<RegisterRead>(node->loc(), reg,
                                               /*name=*/reg->name()));

    result_.state_registers[index] =
        StateRegister(std::string(proc->GetStateParam(index)->name()),
                      proc->GetInitValueElement(index),
                      /*stage=*/stage, reg,
                      /*reg_write=*/nullptr, reg_read);

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
        block_->AddInputPort(absl::StrCat(channel->name(), data_suffix),
                             channel->type()));

    XLS_ASSIGN_OR_RETURN(
        Node * literal_1,
        block_->MakeNode<xls::Literal>(node->loc(), Value(UBits(1, 1))));

    if (channel->kind() == ChannelKind::kSingleValue) {
      if (receive->is_blocking()) {
        XLS_ASSIGN_OR_RETURN(
            next_node,
            block_->MakeNode<Tuple>(
                node->loc(), std::vector<Node*>({node_map_.at(node->operand(0)),
                                                 input_port})));
      } else {
        XLS_ASSIGN_OR_RETURN(
            next_node,
            block_->MakeNode<Tuple>(
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
        block_->AddInputPort(absl::StrCat(channel->name(), valid_suffix),
                             block_->package()->GetBitsType(1)));

    // If blocking return a tuple of (token, data), and if non-blocking
    // return a tuple of (token, data, valid).
    if (receive->is_blocking()) {
      Node* data = input_port;
      if (receive->predicate().has_value() && options_.gate_recvs()) {
        XLS_ASSIGN_OR_RETURN(
            Node * zero_value,
            block_->MakeNode<xls::Literal>(node->loc(),
                                           ZeroOfType(input_port->GetType())));
        XLS_ASSIGN_OR_RETURN(
            Select * select,
            block_->MakeNodeWithName<Select>(
                /*loc=*/node->loc(),
                /*selector=*/node_map_.at(receive->predicate().value()),
                /*cases=*/std::vector<Node*>({zero_value, input_port}),
                /*default_value=*/std::nullopt,
                /*name=*/absl::StrCat(channel->name(), "_select")));
        data = select;
      }
      XLS_ASSIGN_OR_RETURN(
          next_node,
          block_->MakeNode<Tuple>(
              node->loc(),
              std::vector<Node*>({node_map_.at(node->operand(0)), data})));
    } else {
      XLS_ASSIGN_OR_RETURN(Node * zero_value,
                           block_->MakeNode<xls::Literal>(
                               node->loc(), ZeroOfType(input_port->GetType())));
      // Ensure that the output of the receive is zero when the data is not
      // valid or the predicate is false.
      Node* valid = input_valid_port;
      Node* data = input_port;
      if (options_.gate_recvs()) {
        if (receive->predicate().has_value()) {
          XLS_ASSIGN_OR_RETURN(
              NaryOp * and_pred,
              block_->MakeNode<NaryOp>(
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
            block_->MakeNodeWithName<Select>(
                /*loc=*/node->loc(), /*selector=*/valid,
                /*cases=*/std::vector<Node*>({zero_value, input_port}),
                /*default_value=*/std::nullopt,
                /*name=*/absl::StrCat(channel->name(), "_select")));
        data = select;
      }
      XLS_ASSIGN_OR_RETURN(
          next_node,
          block_->MakeNode<Tuple>(
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
        block_->AddOutputPort(absl::StrCat(channel->name(), data_suffix),
                              node_map_.at(send->data())));
    // Map the Send node to the token operand of the Send in the
    // block.
    next_node = node_map_.at(send->token());

    XLS_ASSIGN_OR_RETURN(
        Node * token_buf,
        block_->MakeNode<UnOp>(
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

  // Sets the next state value for the state elements with the given indices to
  // `next_state`.
  absl::Status SetNextStateNode(Node* next_state,
                                absl::Span<const int64_t> indices) {
    if (next_state->GetType()->GetFlatBitCount() == 0) {
      Proc* proc = function_base_->AsProcOrDie();

      if (next_state->GetType() != proc->package()->GetTupleType({})) {
        return absl::UnimplementedError(
            absl::StrFormat("Proc has no state, but (state type is not"
                            "empty tuple), instead got %s.",
                            next_state->GetType()->ToString()));
      }
      return absl::OkStatus();
    }

    for (int64_t index : indices) {
      if (!result_.state_registers.at(index).has_value()) {
        return absl::InternalError(
            absl::StrFormat("Expected next state node %s to be dependent "
                            "on state",
                            next_state->ToString()));
      }

      StateRegister& state_register = result_.state_registers.at(index).value();
      // There should only be one next state node.
      XLS_CHECK_EQ(state_register.reg_write(), nullptr);

      XLS_ASSIGN_OR_RETURN(state_register.reg_write(),
                           block_->MakeNode<RegisterWrite>(
                               next_state->loc(), next_state,
                               /*load_enable=*/std::nullopt,
                               /*reset=*/std::nullopt, state_register.reg()));
    }

    return absl::OkStatus();
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
      std::string_view name, Node* node, Block* block) {
    XLS_ASSIGN_OR_RETURN(Register * reg,
                         block_->AddRegister(name, node->GetType()));
    XLS_ASSIGN_OR_RETURN(
        RegisterWrite * reg_write,
        block_->MakeNode<RegisterWrite>(node->loc(), node,
                                        /*load_enable=*/std::nullopt,
                                        /*reset=*/std::nullopt, reg));
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
      std::string_view base_name, Node* node,
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
  std::vector<Node*> state_params_;
  // A map from each next state node to it's index(es) in the state element
  // vector. A vector is used because node may be a next state value for more
  // than one statue element.
  absl::flat_hash_map<Node*, std::vector<int64_t>> next_state_nodes_;

  const CodegenOptions& options_;

  Block* block_;
  StreamingIOPipeline result_;
  absl::flat_hash_map<Node*, Node*> node_map_;
};

// Adds the nodes in the given schedule to the block. Pipeline registers are
// inserted between stages and returned as a vector indexed by cycle. The block
// should be empty prior to calling this function.
static absl::StatusOr<StreamingIOPipeline> CloneNodesIntoPipelinedBlock(
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
static absl::StatusOr<StreamingIOPipeline> CloneProcNodesIntoBlock(
    Proc* proc, const CodegenOptions& options, Block* block) {
  CloneNodesIntoBlockHandler cloner(proc, /*stage_count=*/0, options, block);
  XLS_RET_CHECK_OK(cloner.CloneNodes(TopoSort(proc).AsVector(), /*stage=*/0));
  return cloner.GetResult();
}

}  // namespace

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
    std::vector<Node*>& valid_nodes) {
  XLS_CHECK_EQ(from_rdy->operand_count(), 1);

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
      StreamingIOPipeline streaming_io_and_pipeline,
      CloneNodesIntoPipelinedBlock(transformed_schedule, options, block));

  XLS_RET_CHECK_OK(MaybeAddResetPort(block, options));

  std::optional<ValidPorts> valid_ports;
  if (options.valid_control().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        valid_ports,
        AddValidSignal(streaming_io_and_pipeline.pipeline_registers, options,
                       block));
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
  if (block->GetResetPort().has_value()) {
    port_order.push_back(block->GetResetPort().value()->GetName());
  }
  if (valid_ports.has_value()) {
    port_order.push_back(valid_ports->input->GetName());
  }
  for (Param* param : f->params()) {
    port_order.push_back(param->GetName());
  }
  if (valid_ports.has_value() && valid_ports->output != nullptr) {
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

  XLS_ASSIGN_OR_RETURN(StreamingIOPipeline streaming_io_and_pipeline,
                       CloneNodesIntoPipelinedBlock(schedule, options, block));

  int64_t number_of_outputs = 0;
  for (const auto& [stage, outputs] : streaming_io_and_pipeline.outputs) {
    number_of_outputs += outputs.size();
  }

  bool streaming_outputs_mutually_exclusive = true;
  if (number_of_outputs > 1) {
    // TODO: do this analysis on a per-stage basis
    XLS_ASSIGN_OR_RETURN(streaming_outputs_mutually_exclusive,
                         AreStreamingOutputsMutuallyExclusive(proc));

    if (streaming_outputs_mutually_exclusive) {
      XLS_VLOG(3) << absl::StrFormat(
          "%d streaming outputs determined to be mutually exclusive",
          streaming_io_and_pipeline.outputs.size());
    } else {
      XLS_VLOG(3) << absl::StrFormat(
          "%d streaming outputs not proven to be mutually exclusive -- "
          "assuming false",
          streaming_io_and_pipeline.outputs.size());
    }
  }

  XLS_VLOG(3) << "After Pipeline";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_RET_CHECK_OK(MaybeAddResetPort(block, options));

  XLS_ASSIGN_OR_RETURN(
      std::vector<Node*> pipelined_valids,
      AddBubbleFlowControl(options, streaming_io_and_pipeline, block));

  XLS_VLOG(3) << "After Flow Control";
  XLS_VLOG_LINES(3, block->DumpIr());

  if (!streaming_outputs_mutually_exclusive) {
    XLS_RETURN_IF_ERROR(
        AddOneShotOutputLogic(options, streaming_io_and_pipeline, block));
  }
  XLS_VLOG(3) << absl::StrFormat("After Output Triggers");
  XLS_VLOG_LINES(3, block->DumpIr());

  // Initialize the valid flops to the pipeline registers.
  // First element is skipped as the initial stage valid flops will be
  // constructed from the input flops and/or input valid ports and will be
  // added later in AddInputFlops() and AddIdleOutput().
  XLS_CHECK_GE(pipelined_valids.size(), 1);
  std::vector<Node*> valid_flops(pipelined_valids.begin() + 1,
                                 pipelined_valids.end());

  if (options.flop_inputs() || options.flop_outputs()) {
    XLS_RETURN_IF_ERROR(AddInputOutputFlops(options,
                                            streaming_io_and_pipeline, block,
                                            valid_flops));
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

  // TODO: add simplification pass here to remove unnecessary `1 & x`

  XLS_RETURN_IF_ERROR(UpdateChannelMetadata(streaming_io_and_pipeline, block));
  XLS_VLOG(3) << "After UpdateChannelMetadata";
  XLS_VLOG_LINES(3, block->DumpIr());

  return block;
}

absl::StatusOr<Block*> FunctionToCombinationalBlock(
    Function* f, std::string_view block_name) {
  return FunctionToCombinationalBlock(f,
                                      CodegenOptions().module_name(block_name));
}

absl::StatusOr<Block*> FunctionToCombinationalBlock(
    Function* f, const CodegenOptions& options) {
  std::string module_name(
      options.module_name().value_or(SanitizeIdentifier(f->name())));
  Block* block = f->package()->AddBlock(
      std::make_unique<Block>(module_name, f->package()));

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
                                                const CodegenOptions& options) {
  XLS_VLOG(3) << "Converting proc to combinational block:";
  XLS_VLOG_LINES(3, proc->DumpIr());

  // In a combinational module, the proc cannot have any state to avoid
  // combinational loops. That is, the loop state must be an empty tuple.
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
  for (const auto& [stage, outputs] : streaming_io.outputs) {
    number_of_outputs += outputs.size();
  }

  if (number_of_outputs > 1) {
    // TODO: do this analysis on a per-stage basis
    XLS_ASSIGN_OR_RETURN(bool streaming_outputs_mutually_exclusive,
                         AreStreamingOutputsMutuallyExclusive(proc));

    if (streaming_outputs_mutually_exclusive) {
      XLS_VLOG(3) << absl::StrFormat(
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

  XLS_RETURN_IF_ERROR(AddCombinationalFlowControl(
      streaming_io.inputs, streaming_io.outputs, options, block));

  // TODO(tedhong): 2021-09-23 Remove and add any missing functionality to
  //                codegen pipeline.
  XLS_RETURN_IF_ERROR(RemoveDeadTokenNodes(block));
  XLS_VLOG(3) << "After RemoveDeadTokenNodes";
  XLS_VLOG_LINES(3, block->DumpIr());

  XLS_RETURN_IF_ERROR(UpdateChannelMetadata(streaming_io, block));
  XLS_VLOG(3) << "After UpdateChannelMetadata";
  XLS_VLOG_LINES(3, block->DumpIr());

  return block;
}

}  // namespace verilog
}  // namespace xls
