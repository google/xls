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

#include "xls/codegen/clone_nodes_into_block_handler.h"

#include <algorithm>
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
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_instantiation.h"
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/ir/xls_ir_interface.pb.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls::verilog {

namespace {

absl::StatusOr<ChannelRef> GetChannelRefUsedByNode(ChannelNode* node) {
  if (node->package()->ChannelsAreProcScoped()) {
    return node->function_base()->AsProcOrDie()->GetChannelInterface(
        node->channel_name(), node->direction());
  }
  return GetChannelUsedByNode(node);
}

absl::StatusOr<std::vector<Channel*>> GetLoopbackChannels(Proc* proc) {
  XLS_RET_CHECK(!proc->is_new_style_proc());
  absl::flat_hash_set<Channel*> send_channels;
  absl::flat_hash_set<Channel*> receive_channels;
  for (Node* node : proc->nodes()) {
    if (node->Is<ChannelNode>()) {
      XLS_ASSIGN_OR_RETURN(Channel * channel,
                           GetChannelUsedByNode(node->As<ChannelNode>()));
      if (node->Is<Send>()) {
        send_channels.insert(channel);
      } else if (node->Is<Receive>()) {
        receive_channels.insert(channel);
      }
    }
  }
  std::vector<Channel*> loopback_channels;
  for (Channel* c : send_channels) {
    if (receive_channels.contains(c)) {
      loopback_channels.push_back(c);
    }
  }
  std::sort(loopback_channels.begin(), loopback_channels.end(),
            Channel::NameLessThan);
  return loopback_channels;
}

// Determine which stages are mutually exclusive with each other.
//
// Takes a map from stage number to the state elements which are read and
// written to on each stage.
//
// Since each state element defines a mutual exclusive zone lasting from its
// first read to its first write we can walk through the stage list updating
// the mutual exclusion state.
absl::StatusOr<ConcurrentStageGroups> CalculateConcurrentGroupsFromStateWrites(
    absl::Span<const std::optional<StateRegister>> state_registers,
    int64_t stage_count) {
  ConcurrentStageGroups result(stage_count);
  // Find all the mutex regions
  for (const auto& reg : state_registers) {
    if (!reg) {
      continue;
    }
    if (reg->read_predicate != nullptr) {
      // If the state read is predicated, then it doesn't start a mutual
      // exclusion zone.
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

struct FifoConnections {
  xls::FifoInstantiation* instantiation;
  CloneNodesIntoBlockHandler::ChannelConnection receive_connection;
  CloneNodesIntoBlockHandler::ChannelConnection send_connection;
};

// Adds a FIFO instantiation to the given block which backs the given channel.
absl::StatusOr<FifoConnections> AddFifoInstantiation(
    StreamingChannel* channel, Block* block, const CodegenOptions& options) {
  XLS_RET_CHECK(channel->channel_config().fifo_config().has_value())
      << absl::StreamFormat("Channel %s has no fifo config.", channel->name());

  std::string inst_name = absl::StrFormat("fifo_%s", channel->name());
  XLS_ASSIGN_OR_RETURN(xls::FifoInstantiation * instantiation,
                       block->AddFifoInstantiation(
                           inst_name, *channel->channel_config().fifo_config(),
                           channel->type(), channel->name()));
  XLS_RETURN_IF_ERROR(block
                          ->MakeNode<xls::InstantiationInput>(
                              SourceInfo(), block->GetResetPort().value(),
                              instantiation,
                              xls::FifoInstantiation::kResetPortName)
                          .status());
  XLS_ASSIGN_OR_RETURN(
      Node * dummy_data,
      block->MakeNode<xls::Literal>(SourceInfo(), ZeroOfType(channel->type())));
  XLS_ASSIGN_OR_RETURN(Node * one, block->MakeNode<xls::Literal>(
                                       SourceInfo(), Value(UBits(1, 1))));

  XLS_ASSIGN_OR_RETURN(
      InstantiationConnection * push_data,
      block->MakeNode<xls::InstantiationInput>(SourceInfo(), dummy_data,
                                               instantiation, "push_data"));
  XLS_ASSIGN_OR_RETURN(InstantiationConnection * push_valid,
                       block->MakeNode<xls::InstantiationInput>(
                           SourceInfo(), one, instantiation, "push_valid"));
  XLS_ASSIGN_OR_RETURN(InstantiationConnection * push_ready,
                       block->MakeNode<xls::InstantiationOutput>(
                           SourceInfo(), instantiation, "push_ready"));
  CloneNodesIntoBlockHandler::ChannelConnection push_connection{
      .channel = channel,
      .direction = ChannelDirection::kSend,
      .kind = ConnectionKind::kInternal,
      .data = push_data,
      .valid = push_valid,
      .ready = push_ready};

  XLS_ASSIGN_OR_RETURN(InstantiationConnection * pop_data,
                       block->MakeNode<xls::InstantiationOutput>(
                           SourceInfo(), instantiation, "pop_data"));
  XLS_ASSIGN_OR_RETURN(InstantiationConnection * pop_valid,
                       block->MakeNode<xls::InstantiationOutput>(
                           SourceInfo(), instantiation, "pop_valid"));
  XLS_ASSIGN_OR_RETURN(InstantiationConnection * pop_ready,
                       block->MakeNode<xls::InstantiationInput>(
                           SourceInfo(), one, instantiation, "pop_ready"));
  CloneNodesIntoBlockHandler::ChannelConnection pop_connection{
      .channel = channel,
      .direction = ChannelDirection::kReceive,
      .kind = ConnectionKind::kInternal,
      .data = pop_data,
      .valid = pop_valid,
      .ready = pop_ready};
  return FifoConnections{.instantiation = instantiation,
                         .receive_connection = pop_connection,
                         .send_connection = push_connection};
}

std::optional<std::string> GetOptionalNodeName(std::optional<Node*> n) {
  if (n.has_value()) {
    return (*n)->GetName();
  }
  return std::nullopt;
};

// Adds the ports on `block` required for receiving on `channel`. Also, adds the
// block port mapping metadata to the block.
absl::StatusOr<CloneNodesIntoBlockHandler::ChannelConnection>
AddPortsForReceive(ChannelRef channel, Block* block,
                   const CodegenOptions& options) {
  std::string_view data_suffix =
      (ChannelRefKind(channel) == ChannelKind::kStreaming)
          ? options.streaming_channel_data_suffix()
          : "";
  XLS_ASSIGN_OR_RETURN(
      Node * data,
      block->AddInputPort(absl::StrCat(ChannelRefName(channel), data_suffix),
                          ChannelRefType(channel)));
  std::optional<Node*> valid;
  std::optional<Node*> ready;
  if (ChannelRefKind(channel) == ChannelKind::kStreaming) {
    XLS_ASSIGN_OR_RETURN(
        valid, block->AddInputPort(
                   absl::StrCat(ChannelRefName(channel),
                                options.streaming_channel_valid_suffix()),
                   block->package()->GetBitsType(1)));
    XLS_ASSIGN_OR_RETURN(Node * one, block->MakeNode<xls::Literal>(
                                         SourceInfo(), Value(UBits(1, 1))));
    XLS_ASSIGN_OR_RETURN(
        ready, block->AddOutputPort(
                   absl::StrCat(ChannelRefName(channel),
                                options.streaming_channel_ready_suffix()),
                   one));
  }

  CloneNodesIntoBlockHandler::ChannelConnection connection{
      .channel = channel,
      .direction = ChannelDirection::kReceive,
      .kind = ConnectionKind::kExternal,
      .data = data,
      .valid = valid,
      .ready = ready};

  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      channel, ChannelDirection::kReceive, GetOptionalNodeName(connection.data),
      GetOptionalNodeName(connection.valid),
      GetOptionalNodeName(connection.ready)));

  return connection;
}

// Adds the ports on `block` required for sending on `channel`. Also, adds the
// block port mapping metadata to the block.
absl::StatusOr<CloneNodesIntoBlockHandler::ChannelConnection> AddPortsForSend(
    ChannelRef channel, Block* block, const CodegenOptions& options) {
  std::string_view data_suffix =
      (ChannelRefKind(channel) == ChannelKind::kStreaming)
          ? options.streaming_channel_data_suffix()
          : "";
  XLS_ASSIGN_OR_RETURN(Node * dummy_data,
                       block->MakeNode<xls::Literal>(
                           SourceInfo(), ZeroOfType(ChannelRefType(channel))));
  XLS_ASSIGN_OR_RETURN(
      Node * data,
      block->AddOutputPort(absl::StrCat(ChannelRefName(channel), data_suffix),
                           dummy_data));
  std::optional<Node*> valid;
  std::optional<Node*> ready;
  if (ChannelRefKind(channel) == ChannelKind::kStreaming) {
    XLS_ASSIGN_OR_RETURN(Node * one, block->MakeNode<xls::Literal>(
                                         SourceInfo(), Value(UBits(1, 1))));
    XLS_ASSIGN_OR_RETURN(
        valid, block->AddOutputPort(
                   absl::StrCat(ChannelRefName(channel),
                                options.streaming_channel_valid_suffix()),
                   one));
    XLS_ASSIGN_OR_RETURN(
        ready, block->AddInputPort(
                   absl::StrCat(ChannelRefName(channel),
                                options.streaming_channel_ready_suffix()),
                   block->package()->GetBitsType(1)));
  }

  CloneNodesIntoBlockHandler::ChannelConnection connection{
      .channel = channel,
      .direction = ChannelDirection::kSend,
      .kind = ConnectionKind::kExternal,
      .data = data,
      .valid = valid,
      .ready = ready};

  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      channel, ChannelDirection::kSend, GetOptionalNodeName(connection.data),
      GetOptionalNodeName(connection.valid),
      GetOptionalNodeName(connection.ready)));

  return connection;
}

absl::StatusOr<std::vector<CloneNodesIntoBlockHandler::ChannelConnection>>
AddChannelConnections(Proc* proc, Block* block, const CodegenOptions& options) {
  std::vector<CloneNodesIntoBlockHandler::ChannelConnection> connections;
  if (proc->is_new_style_proc()) {
    // Iterate through interface and add data/valid/ready ports as needed.
    for (ChannelInterface* interface : proc->interface()) {
      if (interface->direction() == ChannelDirection::kSend) {
        XLS_ASSIGN_OR_RETURN(
            CloneNodesIntoBlockHandler::ChannelConnection connection,
            AddPortsForSend(interface, block, options));

        connections.push_back(connection);
      } else {
        XLS_ASSIGN_OR_RETURN(
            CloneNodesIntoBlockHandler::ChannelConnection connection,
            AddPortsForReceive(interface, block, options));
        connections.push_back(connection);
      }
    }
    // Create a FIFO instantiation for each channel declared in the proc.
    for (Channel* channel : proc->channels()) {
      XLS_ASSIGN_OR_RETURN(
          FifoConnections fifo_connections,
          AddFifoInstantiation(down_cast<StreamingChannel*>(channel), block,
                               options));

      connections.push_back(fifo_connections.send_connection);
      connections.push_back(fifo_connections.receive_connection);
    }
  } else {
    // Add a FIFO instantiation for each loopback channel.
    XLS_ASSIGN_OR_RETURN(std::vector<Channel*> loopback_channels,
                         GetLoopbackChannels(proc));
    for (Channel* channel : loopback_channels) {
      XLS_ASSIGN_OR_RETURN(
          FifoConnections fifo_connections,
          AddFifoInstantiation(down_cast<StreamingChannel*>(channel), block,
                               options));
      connections.push_back(fifo_connections.send_connection);
      connections.push_back(fifo_connections.receive_connection);
    }

    absl::flat_hash_set<Channel*> loopback_channel_set;
    loopback_channel_set.insert(loopback_channels.begin(),
                                loopback_channels.end());

    // Iterate through nodes and add data/valid/ready ports for any non-loopback
    // channel node encountered.
    absl::flat_hash_set<std::pair<Channel*, ChannelDirection>> channels;
    for (Node* node : proc->nodes()) {
      if (node->Is<ChannelNode>()) {
        XLS_ASSIGN_OR_RETURN(Channel * channel,
                             GetChannelUsedByNode(node->As<ChannelNode>()));
        if (loopback_channel_set.contains(channel)) {
          // Loopback channels are handled above.
          continue;
        }
        if (node->Is<Receive>()) {
          channels.insert({channel, ChannelDirection::kReceive});
        } else if (node->Is<Send>()) {
          channels.insert({channel, ChannelDirection::kSend});
        }
      }
    }
    std::vector<std::pair<Channel*, ChannelDirection>> sorted_channels(
        channels.begin(), channels.end());
    std::sort(sorted_channels.begin(), sorted_channels.end(),
              [](std::pair<Channel*, ChannelDirection> a,
                 std::pair<Channel*, ChannelDirection> b) {
                return a.first->name() < b.first->name();
              });
    for (auto [channel, direction] : sorted_channels) {
      if (direction == ChannelDirection::kReceive) {
        XLS_ASSIGN_OR_RETURN(
            CloneNodesIntoBlockHandler::ChannelConnection connection,
            AddPortsForReceive(channel, block, options));
        connections.push_back(connection);
      } else {
        XLS_ASSIGN_OR_RETURN(
            CloneNodesIntoBlockHandler::ChannelConnection connection,
            AddPortsForSend(channel, block, options));
        connections.push_back(connection);
      }
    }
  }
  return connections;
}

}  // namespace

absl::Status
CloneNodesIntoBlockHandler::AddChannelPortsAndFifoInstantiations() {
  XLS_RET_CHECK(is_proc_);
  Proc* proc = function_base_->AsProcOrDie();

  XLS_ASSIGN_OR_RETURN(std::vector<ChannelConnection> connections,
                       AddChannelConnections(proc, block(), options_));
  for (const ChannelConnection& connection : connections) {
    channel_connections_[std::make_pair(
        std::string{ChannelRefName(connection.channel)},
        connection.direction)] = connection;
  }
  return absl::OkStatus();
}

absl::Status CloneNodesIntoBlockHandler::ChannelConnection::ReplaceDataSignal(
    Node* value) const {
  XLS_RET_CHECK_EQ(direction, ChannelDirection::kSend);
  if (data->Is<OutputPort>()) {
    return data->ReplaceOperandNumber(OutputPort::kOperandOperand, value);
  }
  XLS_RET_CHECK(data->Is<InstantiationInput>());
  return data->ReplaceOperandNumber(InstantiationInput::kDataOperand, value);
}

absl::Status CloneNodesIntoBlockHandler::ChannelConnection::ReplaceValidSignal(
    Node* value) const {
  XLS_RET_CHECK_EQ(direction, ChannelDirection::kSend);
  XLS_RET_CHECK(valid.has_value());
  if (valid.value()->Is<OutputPort>()) {
    return valid.value()->ReplaceOperandNumber(OutputPort::kOperandOperand,
                                               value);
  }
  XLS_RET_CHECK(valid.value()->Is<InstantiationInput>());
  return valid.value()->ReplaceOperandNumber(InstantiationInput::kDataOperand,
                                             value);
}

absl::Status CloneNodesIntoBlockHandler::ChannelConnection::ReplaceReadySignal(
    Node* value) const {
  XLS_RET_CHECK_EQ(direction, ChannelDirection::kReceive);
  XLS_RET_CHECK(ready.has_value());
  if (ready.value()->Is<OutputPort>()) {
    return ready.value()->ReplaceOperandNumber(OutputPort::kOperandOperand,
                                               value);
  }
  XLS_RET_CHECK(ready.value()->Is<InstantiationInput>());
  return ready.value()->ReplaceOperandNumber(InstantiationInput::kDataOperand,
                                             value);
}

absl::Status CloneNodesIntoBlockHandler::AddBlockInstantiations(
    const absl::flat_hash_map<FunctionBase*, Block*>& converted_blocks) {
  XLS_RET_CHECK(is_proc_);
  Proc* proc = function_base_->AsProcOrDie();
  if (!proc->is_new_style_proc()) {
    return absl::OkStatus();
  }
  for (const std::unique_ptr<ProcInstantiation>& proc_instantiation :
       proc->proc_instantiations()) {
    XLS_RET_CHECK(converted_blocks.contains(proc_instantiation->proc()))
        << absl::StrFormat(
               "No block generated for proc `%s` instantiated by proc `%s`",
               proc_instantiation->proc()->name(), proc->name());
    Block* instantiated_block = converted_blocks.at(proc_instantiation->proc());

    // Gather the names of the ports in the instantiated block to which a
    // channel is connected.
    struct ConnectionPorts {
      const ChannelConnection* connection;
      std::string data_port;
      std::optional<std::string> ready_port;
      std::optional<std::string> valid_port;
    };
    std::vector<ConnectionPorts> connection_ports;
    for (int64_t i = 0; i < proc_instantiation->channel_args().size(); ++i) {
      ChannelInterface* caller_interface =
          proc_instantiation->channel_args()[i];
      const ChannelConnection& connection =
          channel_connections_.at({std::string{caller_interface->name()},
                                   caller_interface->direction()});
      ChannelInterface* callee_interface =
          proc_instantiation->proc()->interface()[i];
      XLS_ASSIGN_OR_RETURN(
          ChannelPortMetadata callee_port_metadata,
          instantiated_block->GetChannelPortMetadata(
              callee_interface->name(), callee_interface->direction()));
      connection_ports.push_back(
          ConnectionPorts{.connection = &connection,
                          .data_port = callee_port_metadata.data_port.value(),
                          .ready_port = callee_port_metadata.ready_port,
                          .valid_port = callee_port_metadata.valid_port});
    }

    // Gather inputs to feed into the instantiation.
    absl::flat_hash_map<std::string, Node*> instantiation_inputs;
    for (const ConnectionPorts& cp : connection_ports) {
      if (cp.connection->direction == ChannelDirection::kReceive) {
        instantiation_inputs[cp.data_port] = cp.connection->data;
        if (cp.valid_port.has_value()) {
          instantiation_inputs[cp.valid_port.value()] =
              cp.connection->valid.value();
        }
      } else {
        if (cp.ready_port.has_value()) {
          instantiation_inputs[cp.ready_port.value()] =
              cp.connection->ready.value();
        }
      }
    }

    XLS_ASSIGN_OR_RETURN(Block::InstantiationAndConnections instantiation,
                         block()->AddAndConnectBlockInstantiation(
                             proc_instantiation->name(), instantiated_block,
                             instantiation_inputs));

    // Connect up the instantiation outputs.
    for (const ConnectionPorts& cp : connection_ports) {
      if (cp.connection->direction == ChannelDirection::kReceive) {
        if (cp.ready_port.has_value()) {
          XLS_RETURN_IF_ERROR(cp.connection->ReplaceReadySignal(
              instantiation.outputs.at(cp.ready_port.value())));
        }
      } else {
        XLS_RETURN_IF_ERROR(cp.connection->ReplaceDataSignal(
            instantiation.outputs.at(cp.data_port)));
        if (cp.valid_port.has_value()) {
          XLS_RETURN_IF_ERROR(cp.connection->ReplaceValidSignal(
              instantiation.outputs.at(cp.valid_port.value())));
        }
      }
    }
  }
  return absl::OkStatus();
}

CloneNodesIntoBlockHandler::CloneNodesIntoBlockHandler(
    FunctionBase* proc_or_function, int64_t stage_count,
    const CodegenOptions& options, Block* block,
    std::optional<const PackageSchedule*> schedule)
    : is_proc_(proc_or_function->IsProc()),
      function_base_(proc_or_function),
      options_(options),
      block_(block),
      schedule_(schedule) {
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

absl::Status CloneNodesIntoBlockHandler::CloneNodes(
    absl::Span<Node* const> sorted_nodes, int64_t stage) {
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
    } else if (node->Is<Receive>()) {
      Receive* receive = node->As<Receive>();
      XLS_ASSIGN_OR_RETURN(
          next_node, HandleReceiveNode(receive, stage,
                                       channel_connections_.at(
                                           {receive->channel_name(),
                                            ChannelDirection::kReceive})));
    } else if (node->Is<Send>()) {
      Send* send = node->As<Send>();
      XLS_ASSIGN_OR_RETURN(
          next_node,
          HandleSendNode(send, stage,
                         channel_connections_.at(
                             {send->channel_name(), ChannelDirection::kSend})));
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

absl::Status CloneNodesIntoBlockHandler::AddNextPipelineStage(int64_t stage) {
  for (Node* function_base_node : function_base_->nodes()) {
    if (GetSchedule().IsLiveOutOfCycle(function_base_node, stage)) {
      Node* node = node_map_.at(function_base_node);

      XLS_ASSIGN_OR_RETURN(Node * node_after_stage,
                           CreatePipelineRegistersForNode(
                               PipelineSignalName(node->GetName(), stage), node,
                               stage, result_.pipeline_registers.at(stage)));

      node_map_[function_base_node] = node_after_stage;
    }
  }

  return absl::OkStatus();
}

absl::Status CloneNodesIntoBlockHandler::AddOutputPortsIfFunction(
    std::string_view output_port_name) {
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
      port->set_system_verilog_type(f->sv_result_type());
    }
    return absl::OkStatus();
  }

  return absl::OkStatus();
}

absl::Status CloneNodesIntoBlockHandler::MarkMutualExclusiveStages(
    int64_t stage_count) {
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

absl::StatusOr<Node*> CloneNodesIntoBlockHandler::HandleStateRead(Node* node,
                                                                  Stage stage) {
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

    XLS_ASSIGN_OR_RETURN(reg,
                         block()->AddRegister(name, node->GetType(),
                                              state_element->initial_value()));

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

absl::StatusOr<Node*> CloneNodesIntoBlockHandler::HandleFunctionParam(
    Node* node) {
  Param* param = node->As<Param>();
  XLS_ASSIGN_OR_RETURN(
      InputPort * res,
      block()->AddInputPort(param->GetName(), param->GetType(), param->loc()));
  if (std::optional<PackageInterfaceProto::Function> f = FindFunctionInterface(
          options_.package_interface(), function_base_->name())) {
    // Record sv-type associated with this port.
    auto it = absl::c_find_if(f->parameters(),
                              [&](const PackageInterfaceProto::NamedValue& p) {
                                return p.name() == param->name();
                              });
    if (it != f->parameters().end() && it->has_sv_type()) {
      res->set_system_verilog_type(it->sv_type());
    }
  }
  return res;
}

absl::Status CloneNodesIntoBlockHandler::HandleNextValue(Node* node,
                                                         Stage stage) {
  Proc* proc = function_base_->AsProcOrDie();
  Next* next = node->As<Next>();
  StateElement* state_element =
      next->state_read()->As<StateRead>()->state_element();
  XLS_ASSIGN_OR_RETURN(int64_t index,
                       proc->GetStateElementIndex(state_element));

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
    XLS_ASSIGN_OR_RETURN(
        state_register.reg_write,
        block()->MakeNode<RegisterWrite>(
            next->loc(), node_map_.at(next->value()),
            /*load_enable=*/std::nullopt,
            /*reset=*/block()->GetResetPort(), state_register.reg));

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
                             block()->package()->GetBitsType(1),
                             Value(UBits(1, 1))));
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
                                         /*reset=*/block()->GetResetPort(),
                                         state_register.reg_full));
  }

  return absl::OkStatus();
}

absl::StatusOr<Node*> CloneNodesIntoBlockHandler::HandleReceiveNode(
    Receive* receive, int64_t stage, const ChannelConnection& connection) {
  Node* next_node;

  if (std::optional<PackageInterfaceProto::Channel> c = FindChannelInterface(
          options_.package_interface(), ChannelRefName(connection.channel));
      c && c->has_sv_type()) {
    if (connection.data->Is<InputPort>()) {
      connection.data->As<InputPort>()->set_system_verilog_type(c->sv_type());
    }
  }

  XLS_ASSIGN_OR_RETURN(
      Node * literal_1,
      block()->MakeNode<xls::Literal>(receive->loc(), Value(UBits(1, 1))));

  if (ChannelRefKind(connection.channel) == ChannelKind::kSingleValue) {
    XLS_RET_CHECK(!connection.valid.has_value());
    XLS_RET_CHECK(!connection.ready.has_value());
    if (receive->is_blocking()) {
      XLS_ASSIGN_OR_RETURN(
          next_node, block()->MakeNode<Tuple>(
                         receive->loc(),
                         std::vector<Node*>({node_map_.at(receive->operand(0)),
                                             connection.data})));
    } else {
      XLS_ASSIGN_OR_RETURN(
          next_node, block()->MakeNode<Tuple>(
                         receive->loc(),
                         std::vector<Node*>({node_map_.at(receive->operand(0)),
                                             connection.data, literal_1})));
    }

    result_.single_value_inputs.push_back(
        SingleValueInput(block(), ChannelRefName(connection.channel)));

    return next_node;
  }

  XLS_RET_CHECK_EQ(ChannelRefKind(connection.channel), ChannelKind::kStreaming);
  XLS_RET_CHECK_EQ(ChannelRefFlowControl(connection.channel),
                   FlowControl::kReadyValid);

  // If blocking return a tuple of (token, data), and if non-blocking
  // return a tuple of (token, data, valid).
  if (receive->is_blocking()) {
    Node* data = connection.data;
    if (receive->predicate().has_value() && options_.gate_recvs()) {
      XLS_ASSIGN_OR_RETURN(
          Node * zero_value,
          block()->MakeNode<xls::Literal>(
              receive->loc(), ZeroOfType(connection.data->GetType())));
      XLS_ASSIGN_OR_RETURN(
          Select * select,
          block()->MakeNodeWithName<Select>(
              /*loc=*/receive->loc(),
              /*selector=*/node_map_.at(receive->predicate().value()),
              /*cases=*/
              std::vector<Node*>({zero_value, connection.data}),
              /*default_value=*/std::nullopt,
              /*name=*/
              absl::StrCat(ChannelRefName(connection.channel), "_select")));
      data = select;
    }
    XLS_ASSIGN_OR_RETURN(
        next_node,
        block()->MakeNode<Tuple>(
            receive->loc(),
            std::vector<Node*>({node_map_.at(receive->operand(0)), data})));
  } else {
    XLS_ASSIGN_OR_RETURN(
        Node * zero_value,
        block()->MakeNode<xls::Literal>(
            receive->loc(), ZeroOfType(connection.data->GetType())));
    // Ensure that the output of the receive is zero when the data is not
    // valid or the predicate is false.
    Node* valid = connection.valid.value();
    Node* data = connection.data;
    if (options_.gate_recvs()) {
      if (receive->predicate().has_value()) {
        XLS_ASSIGN_OR_RETURN(
            NaryOp * and_pred,
            block()->MakeNode<NaryOp>(
                /*loc=*/receive->loc(),
                /*args=*/
                std::vector<Node*>({node_map_.at(receive->predicate().value()),
                                    connection.valid.value()}),
                /*op=*/Op::kAnd));
        valid = and_pred;
      }
      XLS_ASSIGN_OR_RETURN(
          Select * select,
          block()->MakeNodeWithName<Select>(
              /*loc=*/receive->loc(), /*selector=*/valid,
              /*cases=*/
              std::vector<Node*>({zero_value, connection.data}),
              /*default_value=*/std::nullopt,
              /*name=*/
              absl::StrCat(ChannelRefName(connection.channel), "_select")));
      data = select;
    }
    XLS_ASSIGN_OR_RETURN(
        next_node, block()->MakeNode<Tuple>(
                       receive->loc(),
                       std::vector<Node*>(
                           {node_map_.at(receive->operand(0)), data, valid})));
  }

  // To the rest of the logic, a non-blocking receive is always valid.
  Node* signal_valid =
      receive->is_blocking() ? connection.valid.value() : literal_1;

  StreamingInput streaming_input(block(), ChannelRefName(connection.channel),
                                 connection.kind);
  streaming_input.SetSignalData(next_node);
  streaming_input.SetSignalValid(signal_valid);

  if (receive->predicate().has_value()) {
    streaming_input.SetPredicate(node_map_.at(receive->predicate().value()));
  }
  result_.inputs[stage].push_back(streaming_input);

  return next_node;
}

absl::StatusOr<Node*> CloneNodesIntoBlockHandler::HandleSendNode(
    Send* send, int64_t stage, const ChannelConnection& connection) {
  if (std::optional<PackageInterfaceProto::Channel> c = FindChannelInterface(
          options_.package_interface(), ChannelRefName(connection.channel));
      c && c->has_sv_type()) {
    if (connection.data->Is<OutputPort>()) {
      connection.data->As<OutputPort>()->set_system_verilog_type(c->sv_type());
    }
  }

  // Map the Send node to the token operand of the Send in the
  // block.
  Node* next_node = node_map_.at(send->token());

  XLS_ASSIGN_OR_RETURN(
      Node * token_buf,
      block()->MakeNode<UnOp>(
          /*loc=*/SourceInfo(), node_map_.at(send->token()), Op::kIdentity));
  next_node = token_buf;

  if (ChannelRefKind(connection.channel) == ChannelKind::kSingleValue) {
    // The channel data signal is initially wired to a dummy value. Replace it
    // with the real data value.
    if (connection.data->Is<OutputPort>()) {
      XLS_RETURN_IF_ERROR(connection.data->ReplaceOperandNumber(
          OutputPort::kOperandOperand, node_map_.at(send->data())));
    } else {
      XLS_RET_CHECK(connection.data->Is<InstantiationInput>());
      XLS_RETURN_IF_ERROR(connection.data->ReplaceOperandNumber(
          InstantiationInput::kDataOperand, node_map_.at(send->data())));
    }

    result_.single_value_outputs.push_back(
        SingleValueOutput(block(), ChannelRefName(connection.channel)));
    return next_node;
  }

  XLS_RET_CHECK_EQ(ChannelRefKind(connection.channel), ChannelKind::kStreaming);
  XLS_RET_CHECK_EQ(ChannelRefFlowControl(connection.channel),
                   FlowControl::kReadyValid);

  StreamingOutput streaming_output(block(), ChannelRefName(connection.channel),
                                   connection.kind);

  if (send->predicate().has_value()) {
    streaming_output.SetPredicate(node_map_.at(send->predicate().value()));
  }
  streaming_output.SetData(node_map_.at(send->data()));

  result_.outputs[stage].push_back(streaming_output);

  return next_node;
}

absl::StatusOr<Node*> CloneNodesIntoBlockHandler::HandleGeneralNode(
    Node* node) {
  std::vector<Node*> new_operands;
  for (Node* operand : node->operands()) {
    new_operands.push_back(node_map_.at(operand));
  }
  return node->CloneInNewFunction(new_operands, block());
}

absl::StatusOr<PipelineRegister>
CloneNodesIntoBlockHandler::CreatePipelineRegister(std::string_view name,
                                                   Node* node,
                                                   Stage stage_write) {
  std::optional<Value> reset_value;
  std::optional<Node*> reset_signal;
  if (block()->GetResetPort().has_value() &&
      options_.reset()->reset_data_path()) {
    reset_value = ZeroOfType(node->GetType());
    reset_signal = block()->GetResetPort();
  }

  XLS_ASSIGN_OR_RETURN(
      Register * reg, block()->AddRegister(name, node->GetType(), reset_value));

  XLS_ASSIGN_OR_RETURN(
      RegisterWrite * reg_write,
      block()->MakeNode<RegisterWrite>(node->loc(), node,
                                       /*load_enable=*/std::nullopt,
                                       /*reset=*/reset_signal, reg));
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * reg_read,
      block()->MakeNodeWithName<RegisterRead>(node->loc(), reg,
                                              /*name=*/reg->name()));
  result_.node_to_stage_map[reg_write] = stage_write;
  result_.node_to_stage_map[reg_read] = stage_write + 1;
  return PipelineRegister{reg, reg_write, reg_read};
}

absl::StatusOr<Node*>
CloneNodesIntoBlockHandler::CreatePipelineRegistersForNode(
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

        XLS_ASSIGN_OR_RETURN(
            PipelineRegister pipe_reg,
            CreatePipelineRegister(absl::StrFormat("%s_index%d", base_name, i),
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

}  // namespace xls::verilog
