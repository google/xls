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
#include "absl/status/status.h"
#include "xls/codegen/vast.h"
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

// Suffixes for ready/valid ports for streaming channels.
static const char kReadySuffix[] = "_rdy";
static const char kValidSuffix[] = "_vld";

// Name of the output port which holds the return value of the function.
// TODO(meheff): 2021-03-01 Allow port names other than "out".
static const char kOutputPortName[] = "out";

std::string PipelineSignalName(absl::string_view root, int64_t stage) {
  std::string base;
  // Strip any existing pipeline prefix from the name.
  if (!RE2::PartialMatch(root, R"(^p\d+_(.+))", &base)) {
    base = root;
  }
  return absl::StrFormat("p%d_%s", stage, SanitizeIdentifier(base));
}

absl::StatusOr<Block*> FunctionToBlock(Function* f,
                                       absl::string_view block_name) {
  Block* block = f->package()->AddBlock(
      absl::make_unique<Block>(block_name, f->package()));

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
  // in. The final cycle is empty which effectly puts a pipeline register
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

// The collection of pipeline registers for a single stage.
using PipelineStageRegisters = std::vector<PipelineRegister>;

// Adds the nodes in the given schedule to the block. Pipeline registers are
// inserted between stages and returned as a vector indexed by cycle. The block
// should be empty prior to calling this function.
static absl::StatusOr<std::vector<PipelineStageRegisters>> CreatePipeline(
    const PipelineSchedule& schedule, Block* block) {
  Function* function = dynamic_cast<Function*>(schedule.function_base());
  XLS_RET_CHECK(function != nullptr)
      << "CreatePipeline can only be run on a function";

  // A map from the nodes in the function (which the schedule refers to) to the
  // corresponding node in the block.
  absl::flat_hash_map<Node*, Node*> node_map;

  std::vector<PipelineStageRegisters> pipeline_registers(schedule.length() - 1);

  for (int64_t stage = 0; stage < schedule.length(); ++stage) {
    for (Node* function_node : schedule.nodes_in_cycle(stage)) {
      Node* node;

      if (function_node->Is<Param>()) {
        Param* param = function_node->As<Param>();

        XLS_RET_CHECK_EQ(schedule.cycle(param), 0);
        XLS_ASSIGN_OR_RETURN(
            node, block->AddInputPort(param->GetName(), param->GetType(),
                                      param->loc()));
      } else {
        std::vector<Node*> new_operands;
        for (Node* operand : function_node->operands()) {
          new_operands.push_back(node_map.at(operand));
        }

        XLS_ASSIGN_OR_RETURN(
            node, function_node->CloneInNewFunction(new_operands, block));
      }

      node_map[function_node] = node;
    }

    // Add pipeline registers. A register is needed for each node which is
    // scheduled at or before this cycle and has a use after this cycle.
    for (Node* function_node : function->nodes()) {
      if (schedule.cycle(function_node) > stage) {
        continue;
      }
      auto is_live_out_of_stage = [&](Node* n) {
        if (stage == schedule.length() - 1) {
          return false;
        }
        if (n == function->return_value()) {
          return true;
        }
        for (Node* user : n->users()) {
          if (schedule.cycle(user) > stage) {
            return true;
          }
        }
        return false;
      };

      Node* node = node_map.at(function_node);
      if (is_live_out_of_stage(function_node)) {
        XLS_ASSIGN_OR_RETURN(
            Register * reg,
            block->AddRegister(PipelineSignalName(node->GetName(), stage),
                               node->GetType()));
        XLS_ASSIGN_OR_RETURN(
            RegisterWrite * reg_write,
            block->MakeNode<RegisterWrite>(node->loc(), node,
                                           /*load_enable=*/absl::nullopt,
                                           /*reset=*/absl::nullopt, reg));
        XLS_ASSIGN_OR_RETURN(
            RegisterRead * reg_read,
            block->MakeNodeWithName<RegisterRead>(node->loc(), reg,
                                                  /*name=*/reg->name()));
        node_map[function_node] = reg_read;
        pipeline_registers.at(stage).push_back(
            PipelineRegister{reg, reg_write, reg_read});
      }
    }
  }

  XLS_RETURN_IF_ERROR(block
                          ->AddOutputPort(kOutputPortName,
                                          node_map.at(function->return_value()))
                          .status());

  return pipeline_registers;
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
    const absl::Span<const PipelineStageRegisters>& pipeline_registers,
    const CodegenOptions& options, absl::optional<InputPort*> reset_input_port,
    Block* block) {
  absl::optional<xls::Reset> reset;
  if (reset_input_port.has_value()) {
    XLS_RET_CHECK(options.reset().has_value());
    reset = xls::Reset();
    reset->reset_value = Value(UBits(0, 1));
    reset->asynchronous = options.reset()->asynchronous();
    reset->active_low = options.reset()->active_low();
  }
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
  std::vector<Node*> pipelined_valids(pipeline_registers.size() + 1);
  pipelined_valids[0] = valid_input_port;
  for (int64_t stage = 0; stage < pipeline_registers.size(); ++stage) {
    // Add valid register to each pipeline stage.
    XLS_ASSIGN_OR_RETURN(
        Register * valid_reg,
        block->AddRegister(PipelineSignalName("valid", stage), u1, reset));
    XLS_RETURN_IF_ERROR(block
                            ->MakeNode<RegisterWrite>(
                                /*loc=*/absl::nullopt, pipelined_valids[stage],
                                /*load_enable=*/absl::nullopt,
                                /*reset=*/reset_input_port, valid_reg)
                            .status());
    XLS_ASSIGN_OR_RETURN(pipelined_valids[stage + 1],
                         block->MakeNode<RegisterRead>(
                             /*loc=*/absl::nullopt, valid_reg));
  }

  // Use the pipelined valid signal as load enable each datapath  pipeline
  // register in each stage as a power optimization.
  for (int64_t stage = 0; stage < pipeline_registers.size(); ++stage) {
    // For each (non-valid-signal) pipeline register add `valid` or `valid ||
    // reset` (if reset exists) as a load enable. The `reset` term ensures the
    // pipeline flushes when reset is enabled.
    if (!pipeline_registers.at(stage).empty()) {
      Node* load_enable = pipelined_valids[stage];
      if (reset_input_port.has_value()) {
        Node* reset_node = reset_input_port.value();
        if (reset->active_low) {
          XLS_ASSIGN_OR_RETURN(reset_node,
                               block->MakeNode<UnOp>(/*loc=*/absl::nullopt,
                                                     reset_node, Op::kNot));
        }
        XLS_ASSIGN_OR_RETURN(
            load_enable, block->MakeNodeWithName<NaryOp>(
                             /*loc=*/absl::nullopt,
                             std::vector<Node*>({load_enable, reset_node}),
                             Op::kOr, PipelineSignalName("load_en", stage)));
      }
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

  std::string block_name = options.module_name().has_value()
                               ? std::string{options.module_name().value()}
                               : SanitizeIdentifier(f->name());
  Block* block = f->package()->AddBlock(
      absl::make_unique<Block>(block_name, f->package()));

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

  XLS_ASSIGN_OR_RETURN(std::vector<PipelineStageRegisters> pipeline_registers,
                       CreatePipeline(transformed_schedule, block));

  absl::optional<InputPort*> reset_port;
  if (options.reset().has_value()) {
    XLS_ASSIGN_OR_RETURN(reset_port,
                         block->AddInputPort(options.reset()->name(),
                                             block->package()->GetBitsType(1)));
  }

  absl::optional<ValidPorts> valid_ports;
  if (options.valid_control().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        valid_ports,
        AddValidSignal(pipeline_registers, options, reset_port, block));
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
  if (reset_port.has_value()) {
    port_order.push_back(reset_port.value()->GetName());
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

// Data structures holding the data and (optional) predicate nodes representing
// streaming inputs (receive over streaming channel) and streaming outputs (send
// over streaming channel) in the generated block.
struct StreamingInput {
  InputPort* port;
  absl::optional<Node*> predicate;
};

struct StreamingOutput {
  OutputPort* port;
  absl::optional<Node*> predicate;
};

struct StreamingIo {
  std::vector<StreamingInput> inputs;
  std::vector<StreamingOutput> outputs;
};

// Clones every node in the given proc into the given block. Some nodes are
// handled specially:
//
// * Proc token parameter becomes an operandless AfterAll operation in the
//   block.
// * Proc state parameter (which must be an empty tuple) becomes a Literal
//   operation in theblock.
// * Receive operations become InputPorts.
// * Send operations become OutputPorts.
//
// Returns vectors containing the InputPorts and OutputPorts created from
// Send/Receive operations of streaming channels.
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
static absl::StatusOr<StreamingIo> CloneProcNodesIntoBlock(Proc* proc,
                                                           Block* block) {
  // Gather the inputs and outputs from streaming channels.
  StreamingIo result;

  // A map from the nodes in `proc` to their corresponding node in the block.
  absl::flat_hash_map<Node*, Node*> node_map;

  for (Node* node : TopoSort(proc)) {
    // Replace token parameter with zero operand AfterAll.
    if (node == proc->TokenParam()) {
      XLS_ASSIGN_OR_RETURN(
          node_map[node],
          block->MakeNode<AfterAll>(node->loc(), std::vector<Node*>()));
      continue;
    }

    // Replace state parameter with Literal empty tuple.
    if (node == proc->StateParam()) {
      XLS_ASSIGN_OR_RETURN(node_map[node], block->MakeNode<xls::Literal>(
                                               node->loc(), Value::Tuple({})));
      continue;
    }
    XLS_RET_CHECK(!node->Is<Param>());

    // Don't clone Receive operations. Instead replace with a tuple
    // containing the Receive's token operand and an InputPort operation.
    if (node->Is<Receive>()) {
      Receive* receive = node->As<Receive>();
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      XLS_ASSIGN_OR_RETURN(
          InputPort * input_port,
          block->AddInputPort(channel->name(), channel->type()));
      XLS_ASSIGN_OR_RETURN(
          node_map[node],
          block->MakeNode<Tuple>(
              node->loc(),
              std::vector<Node*>({node_map.at(node->operand(0)), input_port})));
      if (channel->kind() == ChannelKind::kSingleValue) {
        continue;
      }
      XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kStreaming);
      XLS_RET_CHECK_EQ(down_cast<StreamingChannel*>(channel)->flow_control(),
                       FlowControl::kReadyValid);

      StreamingInput streaming_input;
      streaming_input.port = input_port;
      if (receive->predicate().has_value()) {
        streaming_input.predicate = node_map.at(receive->predicate().value());
      }
      result.inputs.push_back(streaming_input);
      continue;
    }

    // Don't clone Send operations. Instead replace with  an OutputPort
    // operation in the block.
    if (node->Is<Send>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      Send* send = node->As<Send>();

      XLS_ASSIGN_OR_RETURN(
          OutputPort * output_port,
          block->AddOutputPort(channel->name(), node_map.at(send->data())));
      // Map the Send node to the token operand of the Send in the
      // block.
      node_map[node] = node_map.at(send->token());

      if (channel->kind() == ChannelKind::kSingleValue) {
        continue;
      }

      XLS_RET_CHECK_EQ(channel->kind(), ChannelKind::kStreaming);
      XLS_RET_CHECK_EQ(down_cast<StreamingChannel*>(channel)->flow_control(),
                       FlowControl::kReadyValid);

      StreamingOutput streaming_output;
      streaming_output.port = output_port;
      if (send->predicate().has_value()) {
        streaming_output.predicate = node_map.at(send->predicate().value());
      }
      result.outputs.push_back(streaming_output);
      continue;
    }

    // Clone the operation from the proc to the block as is.
    std::vector<Node*> new_operands;
    for (Node* operand : node->operands()) {
      new_operands.push_back(node_map.at(operand));
    }
    XLS_ASSIGN_OR_RETURN(node_map[node],
                         node->CloneInNewFunction(new_operands, block));
  }

  return result;
}

// Adds ready/valid ports for each of the given streaming inputs/outputs. Also,
// adds logic which propagates ready and valid signals through the block.
static absl::Status AddFlowControl(
    absl::Span<const StreamingInput> streaming_inputs,
    absl::Span<const StreamingOutput> streaming_outputs, Block* block) {
  // Add a ready input port for each streaming output. Gather the ready signals
  // into a vector. Ready signals from streaming outputs generated from Send
  // operations are conditioned upon the optional predicate value.
  std::vector<Node*> active_readys;
  for (const StreamingOutput& streaming_output : streaming_outputs) {
    XLS_ASSIGN_OR_RETURN(
        Node * ready,
        block->AddInputPort(
            absl::StrFormat("%s_rdy", streaming_output.port->GetName()),
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
      std::vector<Node*> operands = {not_pred, ready};
      XLS_ASSIGN_OR_RETURN(
          Node * active_ready,
          block->MakeNode<NaryOp>(absl::nullopt, operands, Op::kOr));
      active_readys.push_back(active_ready);
    } else {
      active_readys.push_back(ready);
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

  // Add a valid input port for each streaming input. Gather the valid signals
  // into a vector. Valid signals from streaming inputs generated from Receive
  // operations are conditioned upon the optional predicate value.
  std::vector<Node*> active_valids;
  for (const StreamingInput& streaming_input : streaming_inputs) {
    XLS_ASSIGN_OR_RETURN(
        Node * valid,
        block->AddInputPort(
            absl::StrFormat("%s%s", streaming_input.port->name(), kValidSuffix),
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
      std::vector<Node*> operands = {not_pred, valid};
      XLS_ASSIGN_OR_RETURN(
          Node * active_valid,
          block->MakeNode<NaryOp>(absl::nullopt, operands, Op::kOr));
      active_valids.push_back(active_valid);
    } else {
      active_valids.push_back(valid);
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

  // Make output valid output ports. A valid signal is asserted iff all active
  // inputs valid signals are asserted and the predicate of the data channel (if
  // any) is asserted.
  for (const StreamingOutput& streaming_output : streaming_outputs) {
    Node* valid;
    if (streaming_output.predicate.has_value()) {
      std::vector<Node*> operands = {streaming_output.predicate.value(),
                                     all_active_inputs_valid};
      XLS_ASSIGN_OR_RETURN(
          valid, block->MakeNode<NaryOp>(absl::nullopt, operands, Op::kAnd));
    } else {
      valid = all_active_inputs_valid;
    }
    XLS_RETURN_IF_ERROR(
        block
            ->AddOutputPort(
                absl::StrFormat("%s_vld", streaming_output.port->GetName()),
                valid)
            .status());
  }

  // Make output ready output ports. A ready signal is asserted iff all active
  // output ready signals are asserted and the predicate of the data channel (if
  // any) is asserted.
  for (const StreamingInput& streaming_input : streaming_inputs) {
    Node* ready;
    if (streaming_input.predicate.has_value()) {
      std::vector<Node*> operands = {streaming_input.predicate.value(),
                                     all_active_outputs_ready};
      XLS_ASSIGN_OR_RETURN(
          ready, block->MakeNode<NaryOp>(absl::nullopt, operands, Op::kAnd));
    } else {
      ready = all_active_outputs_ready;
    }
    XLS_RETURN_IF_ERROR(
        block
            ->AddOutputPort(
                absl::StrFormat("%s%s", streaming_input.port->GetName(),
                                kReadySuffix),
                ready)
            .status());
  }
  return absl::OkStatus();
}

// Send/receive nodes are not cloned from the proc into the block, but the
// network of tokens connecting these send/receive nodes *is* cloned. This
// function removes the token operations.
static absl::Status RemoveDeadTokenNodes(Block* block) {
  // Receive nodes produce a tuple of a token and a data value. In the block
  // this becomes a tuple of a token and an InputPort. Run tuple simplification
  // to disintangle the tuples so DCE can do its work and eliminate the token
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

  for (Node* node : block->nodes()) {
    // Nodes like cover and assume have token types and will cause a failure
    // here. Ultimately these operations should *not*
    // have tokens and instead are handled as side-effecting operations.
    XLS_RET_CHECK(!TypeHasToken(node->GetType()));
  }

  return absl::OkStatus();
}

absl::StatusOr<Block*> ProcToCombinationalBlock(Proc* proc,
                                                absl::string_view block_name) {
  XLS_VLOG(3) << "Converting proc to combinational block:";
  XLS_VLOG_LINES(3, proc->DumpIr());

  // In a combinational module, the proc cannot have any state to avoid
  // combinational loops. That is, the loop state must be an empty tuple.
  if (proc->StateType() != proc->package()->GetTupleType({})) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Proc must have no state (state type is empty tuple) "
                        "when lowering to "
                        "a combinational block. Proc state is: %s",
                        proc->StateType()->ToString()));
  }

  Block* block = proc->package()->AddBlock(
      absl::make_unique<Block>(block_name, proc->package()));

  XLS_ASSIGN_OR_RETURN(StreamingIo streaming_io,
                       CloneProcNodesIntoBlock(proc, block));
  XLS_RETURN_IF_ERROR(
      AddFlowControl(streaming_io.inputs, streaming_io.outputs, block));

  XLS_RETURN_IF_ERROR(RemoveDeadTokenNodes(block));

  XLS_VLOG_LINES(3, block->DumpIr());
  return block;
}

}  // namespace verilog
}  // namespace xls
