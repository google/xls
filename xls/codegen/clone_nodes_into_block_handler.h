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

#ifndef XLS_CODEGEN_CLONE_NODES_INTO_BLOCK_HANDLER_H_
#define XLS_CODEGEN_CLONE_NODES_INTO_BLOCK_HANDLER_H_

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/concurrent_stage_groups.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {
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
        XLS_RET_CHECK(!is_proc_);
        XLS_ASSIGN_OR_RETURN(next_node, HandleFunctionParam(node));
      } else if (node->Is<StateRead>()) {
        XLS_RET_CHECK(is_proc_);
        XLS_ASSIGN_OR_RETURN(next_node, HandleStateRead(node, stage));
      } else if (node->Is<Next>()) {
        XLS_RET_CHECK(is_proc_);
        XLS_RETURN_IF_ERROR(HandleNextValue(node, stage));
      } else if (node->Is<ChannelNode>()) {
        XLS_RET_CHECK(is_proc_);
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));

        if (loopback_channel_ids_.contains(channel->id()) &&
            channel->kind() == ChannelKind::kStreaming) {
          StreamingChannel* streaming_channel =
              down_cast<StreamingChannel*>(channel);
          XLS_RET_CHECK(
              streaming_channel->channel_config().fifo_config().has_value())
              << absl::StreamFormat("Channel %s has no fifo config.",
                                    channel->name());

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
                        inst_name,
                        *streaming_channel->channel_config().fifo_config(),
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
  // Don't clone state read operations. Instead replace with a RegisterRead
  // operation.
  absl::StatusOr<Node*> HandleStateRead(Node* node, Stage stage) {
    CHECK_GE(stage, 0);

    Proc* proc = function_base_->AsProcOrDie();
    StateRead* state_read = node->As<StateRead>();
    StateElement* state_element = state_read->state_element();
    XLS_ASSIGN_OR_RETURN(int64_t index,
                         proc->GetStateElementIndex(state_element));

    Register* reg = nullptr;
    RegisterRead* reg_read = nullptr;
    if (!node->GetType()->IsToken() && node->GetType()->GetFlatBitCount() > 0) {
      // Create a temporary name as this register will later be removed
      // and updated.  That register should be created with the
      // state parameter's name.  See UpdateStateRegisterWithReset().
      std::string name =
          block()->UniquifyNodeName(absl::StrCat("__", state_element->name()));

      XLS_ASSIGN_OR_RETURN(reg, block()->AddRegister(name, node->GetType()));

      XLS_ASSIGN_OR_RETURN(reg_read, block()->MakeNodeWithName<RegisterRead>(
                                         node->loc(), reg,
                                         /*name=*/reg->name()));

      result_.node_to_stage_map[reg_read] = stage;
    }

    // The register write will be created later in HandleNextValue.
    result_.state_registers[index] = StateRegister{
        .name = std::string(state_element->name()),
        .reset_value = state_element->initial_value(),
        .read_stage = stage,
        .read_predicate = state_read->predicate().has_value()
                              ? node_map_.at(*state_read->predicate())
                              : nullptr,
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
    StateElement* state_element =
        next->state_read()->As<StateRead>()->state_element();
    XLS_ASSIGN_OR_RETURN(int64_t index,
                         proc->GetStateElementIndex(state_element));

    CHECK_EQ(proc->GetNextStateElement(index), next->state_read());
    StateRegister& state_register = *result_.state_registers.at(index);
    state_register.next_values.push_back(
        {.stage = stage,
         .value = next->value() == next->state_read()
                      ? std::nullopt
                      : std::make_optional(node_map_.at(next->value())),
         .predicate =
             next->predicate().has_value()
                 ? std::make_optional(node_map_.at(next->predicate().value()))
                 : std::nullopt});

    bool last_next_value = absl::c_all_of(
        proc->next_values(proc->GetStateRead(state_element)),
        [&](Next* next_value) {
          return next_value == next || node_map_.contains(next_value);
        });
    if (!last_next_value) {
      // We don't create the RegisterWrite until we're at the last `next_value`
      // for this `param`, so we've translated all the values already.
      return absl::OkStatus();
    }

    if (state_element->type()->GetFlatBitCount() > 0) {
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
    } else if (!state_element->type()->IsToken() &&
               state_element->type() != proc->package()->GetTupleType({})) {
      return absl::UnimplementedError(
          absl::StrFormat("Proc has zero-width state element %d, but type is "
                          "not token or empty tuple, instead got %s.",
                          index, node->GetType()->ToString()));
    }

    // If the next state can be determined in a later cycle than the state read,
    // we have a non-trivial backedge between initiations (II>1); use a "full"
    // bit to track whether the state is currently valid.
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

  static std::optional<PackageInterfaceProto::Channel> FindChannelInterface(
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

  // Determine which stages are mutually exclusive with each other.
  //
  // Takes a map from stage number to the state elements which are read and
  // written to on each stage.
  //
  // Since each state element defines a mutual exclusive zone lasting from its
  // first read to its first write we can walk through the stage list updating
  // the mutual exclusion state.
  static absl::StatusOr<ConcurrentStageGroups>
  CalculateConcurrentGroupsFromStateWrites(
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
}  // namespace xls::verilog

#endif  // XLS_CODEGEN_CLONE_NODES_INTO_BLOCK_HANDLER_H_
