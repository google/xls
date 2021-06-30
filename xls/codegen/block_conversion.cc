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

namespace xls {
namespace verilog {
namespace {

// Suffixes for ready/valid ports for streaming channels.
char kReadySuffix[] = "_rdy";
char kValidSuffix[] = "_vld";

}  // namespace

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

  // TODO(meheff): 2021-03-01 Allow port names other than "out".
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort("out", node_map.at(f->return_value())).status());

  return block;
}

// Returns pipeline-stage prefixed signal name for the given node. For
// example: p3_foo.
static std::string PipelineSignalName(Node* node, int64_t stage) {
  return absl::StrFormat("p%d_%s", stage, SanitizeIdentifier(node->GetName()));
}

absl::StatusOr<Block*> FunctionToPipelinedBlock(
    const PipelineSchedule& schedule, Function* f,
    absl::string_view block_name) {
  Block* block = f->package()->AddBlock(
      absl::make_unique<Block>(block_name, f->package()));

  XLS_RETURN_IF_ERROR(block->AddClockPort("clk"));

  // A map from the nodes in 'f' to their corresponding node in the block.
  absl::flat_hash_map<Node*, Node*> node_map;

  // Emit the parameters first to ensure the their order is preserved in the
  // block.
  for (Param* param : f->params()) {
    XLS_ASSIGN_OR_RETURN(
        node_map[param],
        block->AddInputPort(param->GetName(), param->GetType(), param->loc()));
  }

  for (int64_t stage = 0; stage < schedule.length(); ++stage) {
    for (Node* function_node : schedule.nodes_in_cycle(stage)) {
      if (function_node->Is<Param>()) {
        continue;
      }
      std::vector<Node*> new_operands;
      for (Node* operand : function_node->operands()) {
        new_operands.push_back(node_map.at(operand));
      }
      XLS_ASSIGN_OR_RETURN(
          Node * node, function_node->CloneInNewFunction(new_operands, block));
      node_map[function_node] = node;
    }

    // Add pipeline registers. A register is needed for each node which is
    // scheduled at or before this cycle and has a use after this cycle.
    for (Node* function_node : f->nodes()) {
      if (schedule.cycle(function_node) > stage) {
        continue;
      }
      auto is_live_out_of_stage = [&](Node* n) {
        if (stage == schedule.length() - 1) {
          return false;
        }
        if (n == f->return_value()) {
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
        XLS_ASSIGN_OR_RETURN(Register * reg,
                             block->AddRegister(PipelineSignalName(node, stage),
                                                node->GetType()));
        XLS_RETURN_IF_ERROR(
            block
                ->MakeNode<RegisterWrite>(node->loc(), node,
                                          /*load_enable=*/absl::nullopt,
                                          /*reset=*/absl::nullopt, reg->name())
                .status());

        XLS_ASSIGN_OR_RETURN(
            node_map[function_node],
            block->MakeNode<RegisterRead>(node->loc(), reg->name()));
      }
    }
  }

  // TODO(https://github.com/google/xls/issues/448): 2021-03-01 Allow port names
  // other than "out".
  XLS_RETURN_IF_ERROR(
      block->AddOutputPort("out", node_map.at(f->return_value())).status());

  return block;
}

namespace {

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
absl::StatusOr<StreamingIo> CloneProcNodesIntoBlock(Proc* proc, Block* block) {
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
absl::Status AddFlowControl(absl::Span<const StreamingInput> streaming_inputs,
                            absl::Span<const StreamingOutput> streaming_outputs,
                            Block* block) {
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
absl::Status RemoveDeadTokenNodes(Block* block) {
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

}  // namespace

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
