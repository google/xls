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

#include "xls/codegen_v_1_5/channel_to_port_io_lowering_pass.h"

#include <algorithm>
#include <compare>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/optimization.h"
#include "absl/container/btree_map.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/common/casts.h"
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
#include "xls/ir/register.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/pass_base.h"
#include "xls/public/function_builder.h"

namespace xls::codegen {

namespace {

using DirectedChannelRef = std::pair<ChannelRef, ChannelDirection>;

enum class ConnectionKind { kInternal, kExternal };

struct Connector {
 public:
  ChannelDirection direction;
  ConnectionKind kind;

  Node* data;
  std::optional<Node*> valid;
  std::optional<Node*> ready;

  // Used only for one-shot connections.
  std::optional<Node*> reset_one_shot;
  std::optional<std::pair<Node*, /*operand_no=*/int64_t>> incoming_valid;
  std::optional<Node*> ready_port;

  absl::Status MakeOneShot(Node* new_reset_one_shot, Node* new_outgoing_valid,
                           Node* visible_ready,
                           std::pair<Node*, int64_t> incoming_valid_operand);

  // Replaces the value driving the data/ready/valid port with the given
  // node.
  absl::Status ReplaceDataSignal(Node* value) const;
  absl::Status ReplaceValidSignal(Node* value) const;
  absl::Status ReplaceReadySignal(Node* value) const;

  // Returns the value driving the data/ready/valid port.
  Node* DataSignal() const;
  std::optional<Node*> ValidSignal() const;
  std::optional<Node*> ReadySignal() const;
};

struct DirectedNameLessThan {
  bool operator()(DirectedChannelRef a, DirectedChannelRef b) const {
    auto cmp = ChannelRefName(a.first) <=> ChannelRefName(b.first);
    if (cmp == std::strong_ordering::equal) {
      return a.second < b.second;
    }
    return cmp == std::strong_ordering::less;
  }
};

absl::StatusOr<Connector> MakeConnector(const ChannelPortMetadata& metadata,
                                        Block* block) {
  XLS_ASSIGN_OR_RETURN(Node * data, block->GetPortNode(*metadata.data_port));
  std::optional<Node*> valid;
  std::optional<Node*> ready;
  if (metadata.valid_port.has_value()) {
    XLS_ASSIGN_OR_RETURN(valid, block->GetPortNode(*metadata.valid_port));
  }
  if (metadata.ready_port.has_value()) {
    XLS_ASSIGN_OR_RETURN(ready, block->GetPortNode(*metadata.ready_port));
  }
  return Connector{
      .direction = metadata.direction,
      .kind = ConnectionKind::kExternal,
      .data = data,
      .valid = valid,
      .ready = ready,
  };
}

struct ConnectorPair {
  Connector input;
  Connector output;
};

Node* Connector::DataSignal() const {
  CHECK_EQ(direction, ChannelDirection::kSend);
  if (data->Is<OutputPort>()) {
    return data->As<OutputPort>()->output_source();
  }
  CHECK(data->Is<InstantiationInput>());
  return data->As<InstantiationInput>()->data();
}

absl::Status Connector::ReplaceDataSignal(Node* value) const {
  XLS_RET_CHECK_EQ(direction, ChannelDirection::kSend);
  if (data->Is<OutputPort>()) {
    return data->ReplaceOperandNumber(OutputPort::kOperandOperand, value);
  }
  XLS_RET_CHECK(data->Is<InstantiationInput>());
  return data->ReplaceOperandNumber(InstantiationInput::kDataOperand, value);
}

std::optional<Node*> Connector::ValidSignal() const {
  CHECK_EQ(direction, ChannelDirection::kSend);
  if (incoming_valid.has_value()) {
    // One-shot connector; the signal we see internally is the *incoming* valid
    // signal, before it's filtered by the one-shot logic.
    return incoming_valid->first->operand(incoming_valid->second);
  }
  if (!valid.has_value()) {
    return std::nullopt;
  }
  if (valid.value()->Is<OutputPort>()) {
    return valid.value()->As<OutputPort>()->output_source();
  }
  CHECK(valid.value()->Is<InstantiationInput>());
  return valid.value()->As<InstantiationInput>()->data();
}

absl::Status Connector::ReplaceValidSignal(Node* value) const {
  XLS_RET_CHECK_EQ(direction, ChannelDirection::kSend);
  XLS_RET_CHECK(valid.has_value());
  if (incoming_valid.has_value()) {
    // One-shot connector; the signal we need to replace is the *incoming* valid
    // signal, before it's filtered by the one-shot logic.
    return incoming_valid->first->ReplaceOperandNumber(incoming_valid->second,
                                                       value);
  }
  if (valid.value()->Is<OutputPort>()) {
    return valid.value()->ReplaceOperandNumber(OutputPort::kOperandOperand,
                                               value);
  }
  XLS_RET_CHECK(valid.value()->Is<InstantiationInput>());
  return valid.value()->ReplaceOperandNumber(InstantiationInput::kDataOperand,
                                             value);
}

std::optional<Node*> Connector::ReadySignal() const {
  CHECK_EQ(direction, ChannelDirection::kReceive);
  if (!ready.has_value()) {
    return std::nullopt;
  }
  if (ready.value()->Is<OutputPort>()) {
    return ready.value()->As<OutputPort>()->output_source();
  }
  CHECK(ready.value()->Is<InstantiationInput>());
  return ready.value()->As<InstantiationInput>()->data();
}

absl::Status Connector::ReplaceReadySignal(Node* value) const {
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

absl::Status Connector::MakeOneShot(
    Node* new_reset_one_shot, Node* new_outgoing_valid, Node* visible_ready,
    std::pair<Node*, int64_t> incoming_valid_operand) {
  CHECK(!reset_one_shot.has_value());
  CHECK_EQ(direction, ChannelDirection::kSend);
  CHECK(valid.has_value());

  XLS_RETURN_IF_ERROR(ReplaceValidSignal(new_outgoing_valid));

  reset_one_shot = new_reset_one_shot;
  incoming_valid = incoming_valid_operand;
  ready_port = ready;
  ready = visible_ready;

  return absl::OkStatus();
}

std::optional<std::string> GetOptionalNodeName(std::optional<Node*> n) {
  if (n.has_value()) {
    return (*n)->GetName();
  }
  return std::nullopt;
};

absl::StatusOr<absl::btree_set<Channel*, struct Channel::NameLessThan>>
GetLoopbackChannels(ScheduledBlock* block) {
  XLS_RET_CHECK_NE(block->source(), nullptr);
  XLS_RET_CHECK(block->source()->IsProc());
  XLS_RET_CHECK(!block->source()->AsProcOrDie()->is_new_style_proc());
  absl::flat_hash_set<Channel*> send_channels;
  absl::flat_hash_set<Channel*> receive_channels;
  for (Node* node : block->nodes()) {
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
  absl::btree_set<Channel*, struct Channel::NameLessThan> loopback_channels;
  for (Channel* c : send_channels) {
    if (receive_channels.contains(c)) {
      loopback_channels.insert(c);
    }
  }
  return loopback_channels;
}

absl::StatusOr<Connector> AddPortsForSend(
    ChannelRef channel, ScheduledBlock* block,
    const BlockConversionPassOptions& options) {
  std::string_view data_suffix =
      (ChannelRefKind(channel) == ChannelKind::kStreaming)
          ? options.codegen_options.streaming_channel_data_suffix()
          : "";
  XLS_ASSIGN_OR_RETURN(Node * placeholder_data,
                       block->MakeNode<xls::Literal>(
                           SourceInfo(), ZeroOfType(ChannelRefType(channel))));
  XLS_ASSIGN_OR_RETURN(
      Node * data,
      block->AddOutputPort(absl::StrCat(ChannelRefName(channel), data_suffix),
                           placeholder_data));

  if (std::optional<PackageInterfaceProto::Channel> c =
          ::xls::verilog::FindChannelInterface(
              options.codegen_options.package_interface(),
              ChannelRefName(channel));
      c && c->has_sv_type()) {
    data->As<OutputPort>()->set_system_verilog_type(c->sv_type());
  }

  std::optional<Node*> valid;
  std::optional<Node*> ready;
  if (ChannelRefKind(channel) == ChannelKind::kStreaming) {
    XLS_ASSIGN_OR_RETURN(
        Node * placeholder_valid,
        block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
    XLS_ASSIGN_OR_RETURN(
        valid,
        block->AddOutputPort(
            absl::StrCat(
                ChannelRefName(channel),
                options.codegen_options.streaming_channel_valid_suffix()),
            placeholder_valid));
    XLS_ASSIGN_OR_RETURN(
        ready,
        block->AddInputPort(
            absl::StrCat(
                ChannelRefName(channel),
                options.codegen_options.streaming_channel_ready_suffix()),
            block->package()->GetBitsType(1)));
  }

  Connector connector{.direction = ChannelDirection::kSend,
                      .kind = ConnectionKind::kExternal,
                      .data = data,
                      .valid = valid,
                      .ready = ready};

  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      channel, ChannelDirection::kSend, connector.data->GetName(),
      GetOptionalNodeName(connector.valid),
      GetOptionalNodeName(connector.ready)));

  return connector;
}

absl::StatusOr<Connector> AddPortsForReceive(
    ChannelRef channel, ScheduledBlock* block,
    const BlockConversionPassOptions& options) {
  std::string_view data_suffix =
      (ChannelRefKind(channel) == ChannelKind::kStreaming)
          ? options.codegen_options.streaming_channel_data_suffix()
          : "";

  XLS_ASSIGN_OR_RETURN(
      Node * data,
      block->AddInputPort(absl::StrCat(ChannelRefName(channel), data_suffix),
                          ChannelRefType(channel)));
  if (std::optional<PackageInterfaceProto::Channel> c =
          ::xls::verilog::FindChannelInterface(
              options.codegen_options.package_interface(),
              ChannelRefName(channel));
      c.has_value() && c->has_sv_type()) {
    data->As<InputPort>()->set_system_verilog_type(c->sv_type());
  }

  std::optional<Node*> valid;
  std::optional<Node*> ready;
  if (ChannelRefKind(channel) == ChannelKind::kStreaming) {
    XLS_ASSIGN_OR_RETURN(
        valid,
        block->AddInputPort(
            absl::StrCat(
                ChannelRefName(channel),
                options.codegen_options.streaming_channel_valid_suffix()),
            block->package()->GetBitsType(1)));
    XLS_ASSIGN_OR_RETURN(
        Node * placeholder_ready,
        block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
    XLS_ASSIGN_OR_RETURN(
        ready,
        block->AddOutputPort(
            absl::StrCat(
                ChannelRefName(channel),
                options.codegen_options.streaming_channel_ready_suffix()),
            placeholder_ready));
  }

  Connector connector{.direction = ChannelDirection::kReceive,
                      .kind = ConnectionKind::kExternal,
                      .data = data,
                      .valid = valid,
                      .ready = ready};

  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      channel, ChannelDirection::kReceive, connector.data->GetName(),
      GetOptionalNodeName(connector.valid),
      GetOptionalNodeName(connector.ready)));

  return connector;
}

// Adds a FIFO instantiation to the given block which backs the given channel.
absl::StatusOr<ConnectorPair> AddFifoInstantiation(StreamingChannel* channel,
                                                   Block* block) {
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
  Connector input{.direction = ChannelDirection::kSend,
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
  Connector output{.direction = ChannelDirection::kReceive,
                   .kind = ConnectionKind::kInternal,
                   .data = pop_data,
                   .valid = pop_valid,
                   .ready = pop_ready};

  return ConnectorPair{.input = std::move(input), .output = std::move(output)};
}

absl::StatusOr<
    std::pair<bool, absl::flat_hash_map<DirectedChannelRef, Connector>>>
LowerChannelsToConnectors(ScheduledBlock* block,
                          const BlockConversionPassOptions& options) {
  std::pair<bool, absl::flat_hash_map<DirectedChannelRef, Connector>> result =
      std::make_pair(false,
                     absl::flat_hash_map<DirectedChannelRef, Connector>());
  auto& [changed, connections] = result;

  if (block->source() == nullptr || !block->source()->IsProc()) {
    return result;
  }

  Proc* source = block->source()->AsProcOrDie();
  if (source->is_new_style_proc()) {
    // Iterate through interface and add data/valid/ready ports as needed.
    for (ChannelInterface* interface : source->interface()) {
      if (block->HasChannelPortMetadata(interface->name(),
                                        interface->direction())) {
        XLS_ASSIGN_OR_RETURN(ChannelPortMetadata metadata,
                             block->GetChannelPortMetadata(
                                 interface->name(), interface->direction()));
        XLS_ASSIGN_OR_RETURN(Connector connector,
                             MakeConnector(metadata, block));
        connections.emplace(
            DirectedChannelRef{interface, interface->direction()}, connector);
        continue;
      }
      if (interface->direction() == ChannelDirection::kSend) {
        XLS_ASSIGN_OR_RETURN(Connector connection,
                             AddPortsForSend(interface, block, options));
        changed = true;
        connections.emplace(
            DirectedChannelRef{interface, interface->direction()}, connection);
      } else {
        CHECK_EQ(interface->direction(), ChannelDirection::kReceive);
        XLS_ASSIGN_OR_RETURN(Connector connection,
                             AddPortsForReceive(interface, block, options));
        changed = true;
        connections.emplace(
            DirectedChannelRef{interface, interface->direction()}, connection);
      }
    }

    // Create a FIFO instantiation for each channel declared in the proc.
    //
    // Note: channels need to be lowered differently depending on whether or not
    // the channel is a loopback channel; if it is, we create connectors for
    // both ends and also instantiate the FIFO. We also want to avoid creating
    // connectors for the same channel twice, so we try to reuse
    // ChannelPortMetadata when available (which should also help handle if
    // we're invoked on a partially-lowered proc).
    for (Channel* channel : source->channels()) {
      if (block->HasChannelPortMetadata(channel->name(),
                                        ChannelDirection::kSend) &&
          block->HasChannelPortMetadata(channel->name(),
                                        ChannelDirection::kReceive)) {
        XLS_ASSIGN_OR_RETURN(ChannelPortMetadata input_metadata,
                             block->GetChannelPortMetadata(
                                 channel->name(), ChannelDirection::kSend));
        XLS_ASSIGN_OR_RETURN(Connector input,
                             MakeConnector(input_metadata, block));
        connections.emplace(
            DirectedChannelRef{channel, ChannelDirection::kSend}, input);

        XLS_ASSIGN_OR_RETURN(ChannelPortMetadata output_metadata,
                             block->GetChannelPortMetadata(
                                 channel->name(), ChannelDirection::kReceive));
        XLS_ASSIGN_OR_RETURN(Connector output,
                             MakeConnector(output_metadata, block));
        connections.emplace(
            DirectedChannelRef{channel, ChannelDirection::kReceive}, output);
        continue;
      }

      XLS_RET_CHECK(!block->HasChannelPortMetadata(channel->name(),
                                                   ChannelDirection::kSend))
          << "Channel " << channel->name()
          << " already has send port metadata but no receive port metadata.";
      XLS_RET_CHECK(!block->HasChannelPortMetadata(channel->name(),
                                                   ChannelDirection::kReceive))
          << "Channel " << channel->name()
          << " already has receive port metadata but no send port metadata.";

      XLS_ASSIGN_OR_RETURN(
          ConnectorPair fifo_connections,
          AddFifoInstantiation(down_cast<StreamingChannel*>(channel), block));
      changed = true;

      XLS_ASSIGN_OR_RETURN(ChannelInterface * send_interface,
                           source->GetSendChannelInterface(channel->name()));
      connections.emplace(
          DirectedChannelRef{send_interface, ChannelDirection::kSend},
          fifo_connections.input);

      XLS_ASSIGN_OR_RETURN(ChannelInterface * recv_interface,
                           source->GetReceiveChannelInterface(channel->name()));
      connections.emplace(
          DirectedChannelRef{recv_interface, ChannelDirection::kReceive},
          fifo_connections.output);
    }
  } else {
    // Add a FIFO instantiation for each loopback channel.
    XLS_ASSIGN_OR_RETURN(
        (absl::btree_set<Channel*, struct Channel::NameLessThan>
             loopback_channels),
        GetLoopbackChannels(block));
    for (Channel* channel : loopback_channels) {
      if (block->HasChannelPortMetadata(channel->name(),
                                        ChannelDirection::kSend) &&
          block->HasChannelPortMetadata(channel->name(),
                                        ChannelDirection::kReceive)) {
        XLS_ASSIGN_OR_RETURN(ChannelPortMetadata input_metadata,
                             block->GetChannelPortMetadata(
                                 channel->name(), ChannelDirection::kSend));
        XLS_ASSIGN_OR_RETURN(Connector input,
                             MakeConnector(input_metadata, block));
        connections.emplace(
            DirectedChannelRef{channel, ChannelDirection::kSend}, input);

        XLS_ASSIGN_OR_RETURN(ChannelPortMetadata output_metadata,
                             block->GetChannelPortMetadata(
                                 channel->name(), ChannelDirection::kReceive));
        XLS_ASSIGN_OR_RETURN(Connector output,
                             MakeConnector(output_metadata, block));
        connections.emplace(
            DirectedChannelRef{channel, ChannelDirection::kReceive}, output);
        continue;
      }

      XLS_ASSIGN_OR_RETURN(
          ConnectorPair fifo_connections,
          AddFifoInstantiation(down_cast<StreamingChannel*>(channel), block));
      changed = true;

      connections.emplace(DirectedChannelRef{channel, ChannelDirection::kSend},
                          fifo_connections.input);
      connections.emplace(
          DirectedChannelRef{channel, ChannelDirection::kReceive},
          fifo_connections.output);
    }

    // Iterate through nodes and add data/valid/ready ports for any non-loopback
    // channel node encountered.
    absl::btree_set<DirectedChannelRef, DirectedNameLessThan> channels;
    for (Node* node : block->nodes()) {
      if (node->Is<ChannelNode>()) {
        XLS_ASSIGN_OR_RETURN(Channel * channel,
                             GetChannelUsedByNode(node->As<ChannelNode>()));
        if (loopback_channels.contains(channel)) {
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
    for (auto [channel, direction] : channels) {
      if (direction == ChannelDirection::kReceive) {
        XLS_ASSIGN_OR_RETURN(Connector connection,
                             AddPortsForReceive(channel, block, options));
        changed = true;
        connections.emplace(DirectedChannelRef{channel, direction}, connection);
      } else {
        XLS_ASSIGN_OR_RETURN(Connector connection,
                             AddPortsForSend(channel, block, options));
        changed = true;
        connections.emplace(DirectedChannelRef{channel, direction}, connection);
      }
    }
  }
  return result;
}

absl::Status ConnectReceivesToConnector(
    absl::Span<Node* const> receives, Connector& connector,
    ScheduledBlock* block, const BlockConversionPassOptions& options) {
  XLS_RET_CHECK_EQ(connector.direction, ChannelDirection::kReceive);

  std::vector<int64_t> stage_indices;
  stage_indices.reserve(receives.size());
  for (Node* receive : receives) {
    XLS_RET_CHECK(receive->Is<Receive>());
    XLS_ASSIGN_OR_RETURN(int64_t stage_index, block->GetStageIndex(receive));
    stage_indices.push_back(stage_index);
  }

  // Connect the data & valid lines to all users (passing tokens through).
  // Collect each stage's ready signal, which we'll OR together with the
  // existing ready signal for the result.
  // In other words, we are ready to receive data from this connector if any
  // receive is active & ready.
  std::vector<Node*> ready_signals;
  if (connector.ready.has_value()) {
    ready_signals.reserve(1 + receives.size());
    ready_signals.push_back(*connector.ReadySignal());
  }
  for (const auto& [receive, stage_index] :
       iter::zip(receives, stage_indices)) {
    Stage& stage = block->stages()[stage_index];
    Node* token = receive->As<Receive>()->token();
    std::optional<Node*> predicate = receive->As<Receive>()->predicate();
    bool is_blocking = receive->As<Receive>()->is_blocking();

    if (connector.ready.has_value()) {
      // The ready signal from this receive is:
      //     (predicate AND outputs_ready AND outputs_valid)
      absl::InlinedVector<Node*, 3> recv_finishing_requirements;
      recv_finishing_requirements.push_back(stage.outputs_valid());
      recv_finishing_requirements.push_back(stage.outputs_ready());
      if (predicate.has_value()) {
        recv_finishing_requirements.push_back(*predicate);
      }
      XLS_ASSIGN_OR_RETURN(
          Node * recv_finishing,
          block->MakeNode<NaryOp>(receive->loc(), recv_finishing_requirements,
                                  Op::kAnd));
      ready_signals.push_back(recv_finishing);
    }

    if (connector.valid.has_value()) {
      // This active input is valid iff the receive is inactive (!predicate)
      // or the valid signal is asserted.
      Node* recv_valid_or_inactive = *connector.valid;
      if (predicate.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            Node * recv_inactive,
            block->MakeNodeInStage<UnOp>(stage_index, receive->loc(),
                                         *predicate, Op::kNot));
        XLS_ASSIGN_OR_RETURN(
            recv_valid_or_inactive,
            block->MakeNodeInStage<NaryOp>(
                stage_index, receive->loc(),
                absl::MakeConstSpan({*connector.valid, recv_inactive}),
                Op::kOr));
      }
      XLS_RETURN_IF_ERROR(
          ReplaceWithAnd(stage.active_inputs_valid(), recv_valid_or_inactive)
              .status());
    }

    Node* data = connector.data;
    if (options.codegen_options.gate_recvs()) {
      std::vector<Node*> gate_conditions;
      gate_conditions.reserve(2);
      if (predicate.has_value()) {
        gate_conditions.push_back(*predicate);
      }
      if (!is_blocking && connector.valid.has_value()) {
        gate_conditions.push_back(*connector.valid);
      }

      Node* gate_condition = nullptr;
      if (gate_conditions.size() == 1) {
        gate_condition = gate_conditions.front();
      } else if (gate_conditions.size() > 1) {
        XLS_ASSIGN_OR_RETURN(gate_condition, block->MakeNodeInStage<NaryOp>(
                                                 stage_index, receive->loc(),
                                                 gate_conditions, Op::kAnd));
      }

      if (gate_condition != nullptr) {
        XLS_ASSIGN_OR_RETURN(Node * zero,
                             block->MakeNodeInStage<Literal>(
                                 stage_index, receive->loc(),
                                 ZeroOfType(connector.data->GetType())));
        XLS_ASSIGN_OR_RETURN(data,
                             block->MakeNodeInStage<Select>(
                                 stage_index, receive->loc(), gate_condition,
                                 absl::MakeConstSpan({zero, connector.data}),
                                 /*default_value=*/std::nullopt));
      }
    }

    std::vector<Node*> replacement_elements = {token, data};
    if (!is_blocking) {
      if (connector.valid.has_value()) {
        replacement_elements.push_back(*connector.valid);
      } else {
        XLS_ASSIGN_OR_RETURN(
            Node * valid, block->MakeNodeInStage<Literal>(
                              stage_index, receive->loc(), Value(UBits(1, 1))));
        replacement_elements.push_back(valid);
      }
    }
    XLS_RETURN_IF_ERROR(receive
                            ->ReplaceUsesWithNewInStage<Tuple>(
                                stage_index, replacement_elements)
                            .status());
    XLS_RETURN_IF_ERROR(block->RemoveNode(receive));
  }
  if (connector.ready.has_value()) {
    if (ready_signals.size() == 1) {
      XLS_RETURN_IF_ERROR(connector.ReplaceReadySignal(ready_signals.front()));
    } else {
      XLS_ASSIGN_OR_RETURN(Node * ready_signal,
                           JoinWithOr(block, ready_signals));
      XLS_RETURN_IF_ERROR(connector.ReplaceReadySignal(ready_signal));
    }
  }

  return absl::OkStatus();
}

absl::Status ConnectSendsToConnector(
    absl::Span<Node* const> sends, Connector& connector, ScheduledBlock* block,
    const BlockConversionPassOptions& options) {
  XLS_RET_CHECK_EQ(connector.direction, ChannelDirection::kSend);

  std::vector<int64_t> stage_indices;
  stage_indices.reserve(sends.size());
  for (Node* send : sends) {
    XLS_RET_CHECK(send->Is<Send>());
    XLS_ASSIGN_OR_RETURN(int64_t stage_index, block->GetStageIndex(send));
    stage_indices.push_back(stage_index);
  }

  // We assume that at most one send is active on a channel at a time, which
  // means we don't need any conflict-resolution logic. Instead, we:
  //
  // 1. Collect the data signals for each operation.
  // 2. Collect the valid conditions from each send; we'll use them to gate the
  //    data signals, and OR them together for the valid signal (if needed).
  //    In other words, the output is valid if any of our sends is actually
  //    sending valid data.
  // 3. Connect the ready-or-done signal (with appropriate predicate control) to
  //    each stage's `outputs_valid`.
  // 4. Use the predicates and a OneHotSelect to combine the data signals.
  std::vector<Node*> data_signals;
  std::vector<Node*> valid_conditions;
  data_signals.reserve(sends.size());
  valid_conditions.reserve(sends.size());
  for (const auto& [send, stage_index] : iter::zip(sends, stage_indices)) {
    Stage& stage = block->stages()[stage_index];
    Node* token = send->As<Send>()->token();
    std::optional<Node*> predicate = send->As<Send>()->predicate();

    data_signals.push_back(send->As<Send>()->data());

    // This send is active if and only if the inputs are valid and the predicate
    // (if any) is true.
    absl::InlinedVector<Node*, 3> gate_conditions(
        {stage.inputs_valid(), stage.active_inputs_valid()});
    if (predicate.has_value()) {
      gate_conditions.push_back(*predicate);
    }
    XLS_ASSIGN_OR_RETURN(
        Node * valid_condition,
        block->MakeNode<NaryOp>(send->loc(), gate_conditions, Op::kAnd));
    valid_conditions.push_back(valid_condition);

    if (connector.ready.has_value()) {
      // The stage's output can't be valid until we've successfully sent on the
      // channel.
      Node* send_done_or_inactive = *connector.ready;
      if (predicate.has_value()) {
        XLS_ASSIGN_OR_RETURN(Node * send_inactive, block->MakeNodeInStage<UnOp>(
                                                       stage_index, send->loc(),
                                                       *predicate, Op::kNot));
        XLS_ASSIGN_OR_RETURN(
            send_done_or_inactive,
            block->MakeNodeInStage<NaryOp>(
                stage_index, send->loc(),
                absl::MakeConstSpan({*connector.ready, send_inactive}),
                Op::kOr));
      }
      XLS_RETURN_IF_ERROR(
          ReplaceWithAnd(stage.outputs_valid(), send_done_or_inactive)
              .status());
    }

    if (connector.valid.has_value()) {
      // Since at most one send is active on a channel at a time, the output
      // data is valid iff at least one operation is trying to send data.
      if (valid_conditions.size() == 1) {
        XLS_RETURN_IF_ERROR(
            connector.ReplaceValidSignal(valid_conditions.front()));
      } else {
        XLS_ASSIGN_OR_RETURN(Node * valid_condition,
                             JoinWithOr(block, valid_conditions));
        XLS_RETURN_IF_ERROR(connector.ReplaceValidSignal(valid_condition));
      }
    }

    // Lastly, we connect up the data signals from the sends. Since we assume no
    // two sends can be active on the same channel at the same time, we can use
    // OneHotSelect.
    if (data_signals.size() == 1) {
      XLS_RETURN_IF_ERROR(connector.ReplaceDataSignal(data_signals.front()));
    } else {
      XLS_RET_CHECK_GT(data_signals.size(), 1);
      // Reverse the order of the valid conditions, so LSB-to-MSB order will
      // match each condition up with its data signal.
      absl::c_reverse(valid_conditions);
      XLS_ASSIGN_OR_RETURN(
          Node * selector,
          block->MakeNode<Concat>(send->loc(),
                                  absl::MakeConstSpan(valid_conditions)));
      XLS_ASSIGN_OR_RETURN(
          Node * data,
          block->MakeNode<OneHotSelect>(send->loc(), selector, data_signals));
      XLS_RETURN_IF_ERROR(connector.ReplaceDataSignal(data));
    }

    if (connector.reset_one_shot.has_value()) {
      // Make sure to reset the connector's one-shot logic when this send
      // actually resolves.
      absl::InlinedVector<Node*, 3> finished_conditions(
          {stage.outputs_valid(), stage.outputs_ready()});
      if (predicate.has_value()) {
        finished_conditions.push_back(*predicate);
      }
      XLS_ASSIGN_OR_RETURN(Node * finished,
                           block->MakeNodeWithName<NaryOp>(
                               send->loc(), finished_conditions, Op::kAnd,
                               absl::StrCat(send->GetName(), "_finished")));
      XLS_ASSIGN_OR_RETURN(
          connector.reset_one_shot,
          ReplaceWithOr(
              *connector.reset_one_shot,
              absl::MakeConstSpan({finished, *connector.reset_one_shot})));
    }

    XLS_RETURN_IF_ERROR(send->ReplaceUsesWith(token));
    XLS_RETURN_IF_ERROR(block->RemoveNode(send));
  }

  return absl::OkStatus();
}

// Restrict BDD analysis to a subset of nodes.
//
// Currently limited to those cheap to analyze using BDDs plus
// compare ops.
bool UseNodeForMutualExclusionBDD(const Node* node) {
  if (std::all_of(node->operands().begin(), node->operands().end(),
                  IsSingleBitType) &&
      IsSingleBitType(node)) {
    return true;
  }
  return (node->Is<NaryOp>() || node->Is<UnOp>() || node->Is<BitSlice>() ||
          node->Is<ExtendOp>() || node->Is<Concat>() ||
          node->Is<BitwiseReductionOp>() || node->Is<Literal>()) ||
         node->Is<CompareOp>();
}

absl::StatusOr<bool> AreStreamingOutputsMutuallyExclusive(
    ScheduledBlock* block) {
  // Find all send nodes associated with streaming channels.
  int64_t streaming_send_count = 0;
  std::vector<Node*> send_predicates;

  for (Node* node : block->nodes()) {
    if (!node->Is<Send>()) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(ChannelRef channel, node->As<Send>()->GetChannelRef());
    if (ChannelRefKind(channel) != ChannelKind::kStreaming) {
      continue;
    }

    Send* send = node->As<Send>();
    ++streaming_send_count;

    if (send->predicate().has_value()) {
      Node* predicate = send->predicate().value();
      send_predicates.push_back(predicate);
    }
  }

  // If there is only <=1 streaming send node, outputs are mutually exclusive
  if (streaming_send_count <= 1) {
    return true;
  }

  // If there > 1 streaming send node and not all have predicates, then
  // make an assumption that the streaming channels are not exclusive.
  // TODO(tedhong): 2022-02-12 - Refine this to perform a less
  // pessimistic assumption.
  if (streaming_send_count != send_predicates.size()) {
    return false;
  }

  // Use BDD query engine to determine predicates are such that
  // if one is true, the rest are false.
  BddQueryEngine query_engine(BddQueryEngine::kDefaultPathLimit,
                              UseNodeForMutualExclusionBDD);
  XLS_RETURN_IF_ERROR(query_engine.Populate(block).status());

  return query_engine.AtMostOneNodeTrue(send_predicates);
}

// Adds logic to ensure that an output is not transferred more than once.
//
// For multiple-output blocks, even if all outputs are valid at the same time,
// it may not be the case that their destinations are ready.  In this case, for
// N output sends, M<N sends may be completed. In subsequent cycles, more sends
// may yet be completed.
//
// This logic is to ensure that in those subsequent cycles, sends that are
// already completed have valid set to zero to prevent sending an output twice.
absl::Status AddOneShotLogic(Connector& connector, ScheduledBlock* block,
                             const BlockConversionPassOptions& options,
                             std::string_view channel_name = "") {
  XLS_RET_CHECK_EQ(connector.direction, ChannelDirection::kSend);
  XLS_RET_CHECK(connector.ready.has_value());
  XLS_RET_CHECK(connector.valid.has_value());

  std::string channel_prefix =
      channel_name.empty() ? "__" : absl::StrCat("__", channel_name, "_");

  // When implementing one-shot logic for a streaming send channel...
  // 0. We add a placeholder node to represent the "reset_one_shot" signal; this
  //    will be replaced as various users connect their reset signals, but we
  //    need to reference it in the logic we generate.
  // 1. We add a 1-bit "already_done" register, with a `RegisterRead`.
  // 2. We patch the outgoing valid signal, replacing it with:
  //    `AND(incoming_valid, !already_done)`.
  //    NOTE: updates should affect the "incoming_valid" signal component.
  // 3. We patch the visible "ready" signal, replacing it with:
  //    `OR(incoming_ready, already_done)`.
  // 4. We add the `RegisterWrite` for "already_done"; this needs to start out
  //    disabled, be set to 1 when the outgoing valid & incoming ready signals
  //    are both asserted, but be set to 0 as soon as the "reset_one_shot"
  //    signal is asserted.
  //    a. Set `data` to:
  //       `AND(outgoing_valid, incoming_ready, !reset_one_shot)`.
  //    b. Set `load_enable` to:
  //       `OR(AND(outgoing_valid, incoming_ready), reset_one_shot)`.

  Node* incoming_valid = *connector.ValidSignal();
  Node* incoming_ready = *connector.ready;

  // 0. Placeholder node for the "reset_one_shot" signal.
  XLS_ASSIGN_OR_RETURN(
      Node * reset_one_shot,
      block->MakeNode<Literal>(SourceInfo(), Value(UBits(0, 1))));

  // 1. Add the "already_done" register.
  XLS_ASSIGN_OR_RETURN(
      Register * already_done_reg,
      block->AddRegister(absl::StrCat(channel_prefix, "already_done_reg"),
                         block->package()->GetBitsType(1), Value(UBits(0, 1))));
  XLS_ASSIGN_OR_RETURN(Node * already_done,
                       block->MakeNodeWithName<RegisterRead>(
                           SourceInfo(), already_done_reg,
                           absl::StrCat(channel_prefix, "_send_already_done")));
  XLS_ASSIGN_OR_RETURN(
      Node * not_done,
      block->MakeNode<UnOp>(SourceInfo(), already_done, Op::kNot));

  // 2. Patch the outgoing valid signal.
  XLS_ASSIGN_OR_RETURN(
      Node * outgoing_valid,
      block->MakeNode<NaryOp>(SourceInfo(),
                              absl::MakeConstSpan({incoming_valid, not_done}),
                              Op::kAnd));
  std::pair<Node*, int64_t> incoming_valid_location = {outgoing_valid, 0};

  // 3. Patch the visible "ready" signal.
  XLS_ASSIGN_OR_RETURN(
      Node * visible_ready,
      block->MakeNode<NaryOp>(
          SourceInfo(), absl::MakeConstSpan({incoming_ready, already_done}),
          Op::kOr));

  // 4. Add the `RegisterWrite` for "already_done".
  XLS_ASSIGN_OR_RETURN(
      Node * done,
      block->MakeNode<NaryOp>(
          SourceInfo(), absl::MakeConstSpan({outgoing_valid, incoming_ready}),
          Op::kAnd));
  XLS_ASSIGN_OR_RETURN(
      Node * not_resetting,
      block->MakeNode<UnOp>(SourceInfo(), reset_one_shot, Op::kNot));
  XLS_ASSIGN_OR_RETURN(
      Node * already_done_data,
      block->MakeNode<NaryOp>(
          SourceInfo(), absl::MakeConstSpan({done, not_resetting}), Op::kAnd));
  XLS_ASSIGN_OR_RETURN(
      Node * already_done_load_enable,
      block->MakeNode<NaryOp>(
          SourceInfo(), absl::MakeConstSpan({done, reset_one_shot}), Op::kOr));
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<RegisterWrite>(SourceInfo(), already_done_data,
                                    already_done_load_enable, reset_one_shot,
                                    already_done_reg)
          .status());

  // Actually record all of these changes in the connector, so it can correctly
  // handle these signals.
  return connector.MakeOneShot(reset_one_shot, outgoing_valid, visible_ready,
                               incoming_valid_location);
}

// Returns the set of all names of channels used for RAMs.
// These channels are handled specially by RamRewritePass and should be excluded
// from some codegen logic (especially I/O flopping).
absl::flat_hash_set<std::string> GetRamChannelNames(
    const BlockConversionPassOptions& options) {
  absl::flat_hash_set<std::string> ram_channel_names;
  for (const ::xls::verilog::RamConfiguration& ram_config :
       options.codegen_options.ram_configurations()) {
    if (std::holds_alternative<::xls::verilog::Ram1RWConfiguration>(
            ram_config)) {
      const auto& config =
          std::get<::xls::verilog::Ram1RWConfiguration>(ram_config);
      ram_channel_names.insert(
          config.rw_port_configuration().request_channel_name);
      ram_channel_names.insert(
          config.rw_port_configuration().response_channel_name);
      ram_channel_names.insert(
          config.rw_port_configuration().write_completion_channel_name);
    } else if (std::holds_alternative<::xls::verilog::Ram1R1WConfiguration>(
                   ram_config)) {
      const auto& config =
          std::get<::xls::verilog::Ram1R1WConfiguration>(ram_config);
      ram_channel_names.insert(
          config.r_port_configuration().request_channel_name);
      ram_channel_names.insert(
          config.r_port_configuration().response_channel_name);
      ram_channel_names.insert(
          config.w_port_configuration().request_channel_name);
      ram_channel_names.insert(
          config.w_port_configuration().write_completion_channel_name);
    }
  }
  return ram_channel_names;
}

absl::StatusOr<FlopKind> GetFlopKind(ChannelRef channel,
                                     ChannelDirection direction,
                                     ScheduledBlock* block) {
  if (std::holds_alternative<Channel*>(channel)) {
    XLS_RET_CHECK(!block->package()->ChannelsAreProcScoped())
        << "For proc-scoped channels, the flop kind is set on the interface.";
    StreamingChannel* streaming_channel =
        down_cast<StreamingChannel*>(std::get<Channel*>(channel));
    switch (direction) {
      case ChannelDirection::kSend: {
        XLS_RET_CHECK(streaming_channel->channel_config().output_flop_kind())
            << "No output flop kind";
        return streaming_channel->channel_config().output_flop_kind().value();
      }
      case ChannelDirection::kReceive: {
        XLS_RET_CHECK(streaming_channel->channel_config().input_flop_kind())
            << "No input flop kind";
        return streaming_channel->channel_config().input_flop_kind().value();
      }
    }
    ABSL_UNREACHABLE();
    return absl::InternalError(
        absl::StrFormat("Unknown channel direction %d", direction));
  } else {
    return std::get<ChannelInterface*>(channel)->flop_kind();
  }
}

// Adds a register between the node and all its downstream users.
// Returns the new register added.
absl::StatusOr<RegisterRead*> AddRegisterAfterNode(
    std::string_view name_prefix, std::optional<Node*> load_enable,
    Node* node) {
  Block* block = node->function_base()->AsBlockOrDie();

  Type* node_type = node->GetType();
  std::string name = absl::StrFormat("__%s_reg", name_prefix);

  XLS_ASSIGN_OR_RETURN(Register * reg,
                       block->AddRegisterWithZeroResetValue(name, node_type));

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
                              /*reset=*/block->GetResetPort(),
                              /*reg=*/reg)
                          .status());

  return reg_read;
}

// Replace load_en for the register with the given node.
absl::Status UpdateRegisterLoadEn(Node* load_en, Register* reg, Block* block) {
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

// Add a zero-latency buffer after a set of data/valid/ready signal.
//
// Latency: 0 cycles
// Capacity: 1
//
// Breaks the ready signal timing path, but allows combinational valid/data
// pass-through (zero latency).
//
// Logic will be inserted immediately after from_data and from node.
// Logic will be inserted immediately before from_rdy,
//   from_rdy must be a node with a single operand.
absl::Status AddZeroLatencyBufferToRDVNodes(Node* from_data, Node* from_valid,
                                            Node* from_rdy,
                                            std::string_view name_prefix,
                                            Block* block) {
  CHECK_EQ(from_rdy->operand_count(), 1);

  // Add a node for load_enables (will be removed later).
  XLS_ASSIGN_OR_RETURN(Node * literal_1, block->MakeNode<xls::Literal>(
                                             SourceInfo(), Value(UBits(1, 1))));

  // Create data/valid and their skid counterparts.
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_skid_reg_read,
      AddRegisterAfterNode(/*name_prefix=*/absl::StrCat(name_prefix, "_skid"),
                           /*load_enable=*/literal_1, from_data));

  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_valid_skid_reg_read,
      AddRegisterAfterNode(
          /*name_prefix=*/absl::StrCat(name_prefix, "_valid_skid"),
          /*load_enable=*/literal_1, from_valid));

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

  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(
      skid_data_load_en, data_skid_reg_read->GetRegister(), block));

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
      data_valid_skid_reg_write->ReplaceOperandNumber(0, from_skid_rdy));
  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(
      skid_valid_load_en, data_valid_skid_reg_read->GetRegister(), block));

  return absl::OkStatus();
}

// Add flops after the data/valid of a set of three data, valid, and ready
// nodes.
//
// Latency: 1 cycle
// Capacity: 1
//
// Breaks the valid/data timing paths, but leaves the ready path combinational.
// Lower area than a skid buffer, but propagates a timing path backwards.
//
// Logic will be inserted immediately after from_data and from node.
// Logic will be inserted immediately before from_rdy,
//   from_rdy must be a node with a single operand.
//
absl::Status AddRegisterToRDVNodes(Node* from_data, Node* from_valid,
                                   Node* from_rdy, std::string_view name_prefix,
                                   Block* block) {
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

  return absl::OkStatus();
}

// Add a skid buffer after the a set of data/valid/ready signal.
//
// Latency: 1 cycle
// Capacity: 2
//
// Breaks timing paths on all signals (ready, valid, and data) while maintaining
// full throughput (1 data per cycle).
//
// Logic will be inserted immediately after from_data and from node.
// Logic will be inserted immediately before from_rdy,
//   from_rdy must be a node with a single operand.
absl::Status AddSkidBufferToRDVNodes(Node* from_data, Node* from_valid,
                                     Node* from_rdy,
                                     std::string_view name_prefix,
                                     Block* block) {
  // A skid buffer is composed of a zero-latency buffer (skid) fed by a
  // simple pipeline register.
  //
  //   [Data/Valid] -> [Register] -> [Skid Buffer] -> [Output]
  //
  // Note that the order of insertion is important because AddRegister... and
  // AddZeroLatency... both insert logic *at the location* of the inputs.
  //
  // If we insert the ZeroLatency buffer first, it consumes the original inputs.
  // Then if we insert the Register buffer on the *same* original inputs, it
  // effectively places the register *upstream* of the skid buffer, which is
  // exactly what we want.
  XLS_RETURN_IF_ERROR(AddZeroLatencyBufferToRDVNodes(
      from_data, from_valid, from_rdy, name_prefix, block));

  XLS_RETURN_IF_ERROR(AddRegisterToRDVNodes(from_data, from_valid, from_rdy,
                                            name_prefix, block));

  return absl::OkStatus();
}

absl::Status AddFlopToRDVNodes(FlopKind flop_kind, Node* data, Node* valid,
                               Node* ready, std::string_view name_prefix,
                               Block* block) {
  switch (flop_kind) {
    case FlopKind::kZeroLatency:
      return AddZeroLatencyBufferToRDVNodes(data, valid, ready, name_prefix,
                                            block);
    case FlopKind::kSkid:
      return AddSkidBufferToRDVNodes(data, valid, ready, name_prefix, block);
    case FlopKind::kFlop:
      return AddRegisterToRDVNodes(data, valid, ready, name_prefix, block);
    case FlopKind::kNone:
      return absl::OkStatus();
  }
  ABSL_UNREACHABLE();
  return absl::InternalError(
      absl::StrFormat("Unknown flop kind %d", flop_kind));
}

absl::Status AddIOFlopsForReceive(Connector& connector, FlopKind flop_kind,
                                  ChannelRef channel, ScheduledBlock* block,
                                  const BlockConversionPassOptions& options) {
  // NOTE: We control the flop insertion for single-value channels globally,
  // ignoring the flop_kind parameter. This matches the behavior in codegen v1.
  if (ChannelRefKind(channel) == ChannelKind::kSingleValue) {
    if (options.codegen_options.flop_inputs() &&
        options.codegen_options.flop_single_value_channels()) {
      return AddRegisterAfterNode(/*name_prefix=*/connector.data->GetName(),
                                  /*load_enable=*/std::nullopt, connector.data)
          .status();
    }
    // No flops needed
    return absl::OkStatus();
  }
  CHECK_EQ(ChannelRefKind(channel), ChannelKind::kStreaming);

  if (flop_kind == FlopKind::kNone) {
    return absl::OkStatus();
  }
  return AddFlopToRDVNodes(flop_kind, connector.data, *connector.valid,
                           *connector.ready, ChannelRefName(channel), block);
}

absl::Status AddIOFlopsForSend(Connector& connector, FlopKind flop_kind,
                               ChannelRef channel, ScheduledBlock* block,
                               const BlockConversionPassOptions& options) {
  // NOTE: We control the flop insertion for single-value channels globally,
  // ignoring the flop_kind parameter. This matches the behavior in codegen v1.
  if (ChannelRefKind(channel) == ChannelKind::kSingleValue) {
    if (options.codegen_options.flop_outputs() &&
        options.codegen_options.flop_single_value_channels()) {
      flop_kind = FlopKind::kFlop;
    } else {
      flop_kind = FlopKind::kNone;
    }
  }

  if (flop_kind == FlopKind::kNone) {
    return absl::OkStatus();
  }

  const bool should_flop =
      ChannelRefKind(channel) == ChannelKind::kStreaming ||
      (ChannelRefKind(channel) == ChannelKind::kSingleValue &&
       options.codegen_options.flop_outputs() &&
       options.codegen_options.flop_single_value_channels());
  if (!should_flop) {
    return absl::OkStatus();
  }

  std::string data_buf_name =
      absl::StrFormat("__%s_buf", ChannelRefName(channel));
  XLS_ASSIGN_OR_RETURN(Node * output_port_data_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), connector.data->operand(0),
                           Op::kIdentity, data_buf_name));
  XLS_RETURN_IF_ERROR(
      connector.data->ReplaceOperandNumber(0, output_port_data_buf));

  if (ChannelRefKind(channel) == ChannelKind::kSingleValue) {
    // We just need to flop the outgoing data, adding a register after the data
    // signal.
    return AddRegisterAfterNode(/*name_prefix=*/connector.data->GetName(),
                                /*load_enable=*/std::nullopt,
                                output_port_data_buf)
        .status();
  }
  CHECK_EQ(ChannelRefKind(channel), ChannelKind::kStreaming);
  CHECK(connector.valid.has_value());
  CHECK(connector.ready.has_value());

  Node* ready_port = connector.ready_port.has_value() ? *connector.ready_port
                                                      : *connector.ready;

  // Re-calculate the port name for valid/ready to match what
  // StreamingIOName does (suffixing based on port type/etc), or just
  // use the name of the node.
  std::string valid_buf_name =
      absl::StrFormat("__%s_buf", (*connector.valid)->GetName());
  std::string ready_buf_name =
      absl::StrFormat("__%s_buf", ready_port->GetName());

  XLS_ASSIGN_OR_RETURN(Node * output_port_valid_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), (*connector.valid)->operand(0),
                           Op::kIdentity, valid_buf_name));
  XLS_RETURN_IF_ERROR(
      (*connector.valid)->ReplaceOperandNumber(0, output_port_valid_buf));

  XLS_ASSIGN_OR_RETURN(
      Node * output_port_ready_buf,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), ready_port, Op::kIdentity, ready_buf_name));

  XLS_RETURN_IF_ERROR(ready_port->ReplaceUsesWith(output_port_ready_buf));

  return AddFlopToRDVNodes(flop_kind, output_port_data_buf,
                           output_port_valid_buf, output_port_ready_buf,
                           ChannelRefName(channel), block);
}

absl::Status AddIOFlopsForConnector(Connector& connector, ChannelRef channel,
                                    ScheduledBlock* block,
                                    const BlockConversionPassOptions& options) {
  XLS_ASSIGN_OR_RETURN(FlopKind flop_kind,
                       GetFlopKind(channel, connector.direction, block));
  switch (connector.direction) {
    case ChannelDirection::kReceive:
      return AddIOFlopsForReceive(connector, flop_kind, channel, block,
                                  options);
    case ChannelDirection::kSend:
      return AddIOFlopsForSend(connector, flop_kind, channel, block, options);
  }
  ABSL_UNREACHABLE();
  return absl::InternalError(
      absl::StrFormat("Unknown channel direction %d", connector.direction));
}

absl::StatusOr<bool> LowerIoToPorts(
    ScheduledBlock* block,
    absl::flat_hash_map<DirectedChannelRef, Connector>& connections,
    const BlockConversionPassOptions& options) {
  absl::flat_hash_set<std::string> ram_channel_names =
      GetRamChannelNames(options);

  absl::btree_map<DirectedChannelRef, std::vector<Node*>, DirectedNameLessThan>
      io_ops;
  for (Node* node : block->nodes()) {
    if (!node->Is<ChannelNode>()) {
      continue;
    }
    ChannelNode* io_op = node->As<ChannelNode>();
    XLS_ASSIGN_OR_RETURN(ChannelRef channel, io_op->GetChannelRef());
    DirectedChannelRef directed_channel{channel, io_op->direction()};
    io_ops[directed_channel].push_back(node);
  }

  bool changed = false;

  // TODO: do this analysis on a per-stage basis, and apply it per-channel.
  bool needs_one_shot_logic = false;
  const int64_t outgoing_channel_count =
      absl::c_count_if(io_ops, [](const auto& channel_and_io_ops) {
        return channel_and_io_ops.first.second == ChannelDirection::kSend;
      });
  if (outgoing_channel_count > 1) {
    XLS_ASSIGN_OR_RETURN(bool outputs_mutually_exclusive,
                         AreStreamingOutputsMutuallyExclusive(block));
    needs_one_shot_logic = !outputs_mutually_exclusive;
  }
  if (needs_one_shot_logic) {
    for (const auto& [directed_channel, _] : io_ops) {
      if (directed_channel.second != ChannelDirection::kSend) {
        continue;
      }
      auto it = connections.find(directed_channel);
      XLS_RET_CHECK(it != connections.end());
      Connector& connector = it->second;
      XLS_RETURN_IF_ERROR(AddOneShotLogic(connector, block, options));
      changed = true;
    }
  }

  for (const auto& [directed_channel, io_ops_for_channel] : io_ops) {
    auto it = connections.find(directed_channel);
    XLS_RET_CHECK(it != connections.end())
        << "Missing connector for channel: "
        << ChannelRefName(directed_channel.first) << " ("
        << ChannelDirectionToString(directed_channel.second) << ")";
    Connector& connector = it->second;
    if (connector.direction == ChannelDirection::kSend) {
      XLS_RET_CHECK(absl::c_all_of(
          io_ops_for_channel, [](Node* io_op) { return io_op->Is<Send>(); }));
      XLS_RETURN_IF_ERROR(ConnectSendsToConnector(io_ops_for_channel, connector,
                                                  block, options));
    } else {
      XLS_RET_CHECK_EQ(connector.direction, ChannelDirection::kReceive);
      XLS_RET_CHECK(absl::c_all_of(io_ops_for_channel, [](Node* io_op) {
        return io_op->Is<Receive>();
      }));
      XLS_RETURN_IF_ERROR(ConnectReceivesToConnector(
          io_ops_for_channel, connector, block, options));
    }

    // Add any configured I/O flops.
    // However, if this is a RAM channel, then we don't want to add any I/O
    // flops, as RamRewritePass manages the appropriate buffering.
    if (!ram_channel_names.contains(ChannelRefName(directed_channel.first))) {
      XLS_RETURN_IF_ERROR(AddIOFlopsForConnector(
          connector, directed_channel.first, block, options));
    }

    // If this is a proc-scoped channel interface, remove it now that it is no
    // longer used.
    if (std::holds_alternative<ChannelInterface*>(directed_channel.first) &&
        connector.kind == ConnectionKind::kExternal) {
      XLS_RETURN_IF_ERROR(
          block->source()->AsProcOrDie()->RemoveChannelInterface(
              std::get<ChannelInterface*>(directed_channel.first)));
    }

    changed = true;
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> ChannelToPortIoLoweringPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (!block->IsScheduled()) {
      continue;
    }
    ScheduledBlock* scheduled_block = down_cast<ScheduledBlock*>(block.get());
    if (!scheduled_block->source()->IsProc()) {
      continue;
    }
    Proc* source = scheduled_block->source()->AsProcOrDie();

    XLS_ASSIGN_OR_RETURN((auto [changed_connections, connections]),
                         LowerChannelsToConnectors(scheduled_block, options));
    changed |= changed_connections;

    XLS_ASSIGN_OR_RETURN(bool changed_io,
                         LowerIoToPorts(scheduled_block, connections, options));
    changed |= changed_io;

    // Make sure to delete any remaining channels owned by the source proc.
    if (source->is_new_style_proc()) {
      for (Channel* channel : source->channels()) {
        XLS_RETURN_IF_ERROR(source->RemoveChannel(channel));
        changed = true;
      }
    }
  }

  // Make sure to remove any remaining channels in the package.
  for (Channel* channel : package->channels()) {
    XLS_RETURN_IF_ERROR(package->RemoveChannel(channel));
    changed = true;
  }

  return changed;
}

}  // namespace xls::codegen
