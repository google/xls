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
#include <bit>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
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
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/conversion_utils.h"
#include "xls/codegen/ram_configuration.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
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

// Helper to AND two nodes that might be nullptr.
//
// NOTE: If both a and b are nullptr, returns nullptr.
absl::StatusOr<Node*> MakeAnd(Block* block, Node* a, Node* b,
                              const SourceInfo& loc) {
  if (a == nullptr) {
    return b;
  }
  if (b == nullptr) {
    return a;
  }
  if (a == b) {
    return a;
  }
  return block->MakeNode<NaryOp>(loc, absl::MakeConstSpan({a, b}), Op::kAnd);
}

struct ConjunctionResult {
  Node* all;                           // AND(all inputs) including context
  Node* all_without_context;           // AND(all inputs) without common_context
  std::vector<Node*> all_except_each;  // AND(all inputs *except* the k-th node)
                                       //    [including context]
};

// Computes the full conjunction and all possible all-except-one conjunctions
// for the given `nodes` and `common_context`.
//
// Uses a Brent-Kung parallel-prefix architecture to keep the gate count low
// and the depth O(log n).
absl::StatusOr<ConjunctionResult> ComputeConjunctions(
    Block* block, absl::Span<Node* const> nodes,
    std::optional<Node*> common_context = std::nullopt,
    const SourceInfo& loc = SourceInfo()) {
  if (nodes.empty()) {
    XLS_ASSIGN_OR_RETURN(Node * one,
                         block->MakeNode<Literal>(loc, Value(UBits(1, 1))));
    return ConjunctionResult{
        .all = common_context.has_value() ? *common_context : one,
        .all_without_context = one,
        .all_except_each = std::vector<Node*>({}),
    };
  }
  if (nodes.size() == 1) {
    XLS_ASSIGN_OR_RETURN(
        Node * full_conjunction,
        MakeAnd(block, nodes.front(), common_context.value_or(nullptr), loc));
    XLS_ASSIGN_OR_RETURN(
        Node * except_node,
        common_context.has_value()
            ? absl::StatusOr<Node*>(*common_context)
            : block->MakeNode<Literal>(loc, Value(UBits(1, 1))));
    return ConjunctionResult{
        .all = full_conjunction,
        .all_without_context = nodes.front(),
        .all_except_each = std::vector<Node*>({except_node})};
  }

  const int64_t n = nodes.size() + (common_context.has_value() ? 1 : 0);
  const int64_t k = std::bit_ceil(static_cast<size_t>(n));
  std::vector<Node*> tree(2 * k, nullptr);
  std::vector<Node*> context(2 * k, nullptr);

  // Populate the leaves of the tree with the input values, the last being
  // `common_context` (if present).
  for (int64_t i = 0; i < nodes.size(); ++i) {
    tree[k + i] = nodes[i];
  }
  if (common_context.has_value()) {
    tree[k + nodes.size()] = *common_context;
  }

  // 1. Up-Sweep on `tree`: each node is assigned the AND of its children;
  //    the root (tree[1]) ends up holding the AND of all values.
  for (int64_t i = k - 1; i >= 1; --i) {
    XLS_ASSIGN_OR_RETURN(tree[i],
                         MakeAnd(block, tree[2 * i], tree[2 * i + 1], loc));
  }
  // 2. Down-Sweep on `context`; each node is assigned the AND of its parent's
  //    `context` and its sibling's `tree` value. By induction, this is the AND
  //    of all `tree` leaves not in the subtree rooted at the corresponding
  //    `tree` node. In particular, each leaf contains the AND of all inputs
  //    except its corresponding `tree` leaf.
  for (int64_t c = 2; c < k + n; ++c) {
    XLS_ASSIGN_OR_RETURN(context[c],
                         MakeAnd(block, context[c >> 1], tree[c ^ 1], loc));
  }

  // The leaves of the `context` tree hold the AND of all inputs except the one
  // at the leaf, while the root of `tree` holds the AND of all inputs.
  //
  // If `common_context` is present, the last leaf of `tree` holds the AND of
  // all inputs omitting `common_context`.
  std::vector<Node*> all_except_each(nodes.size());
  for (int64_t i = 0; i < nodes.size(); ++i) {
    all_except_each[i] = context[k + i];
  }
  return ConjunctionResult{
      .all = tree[1],
      .all_without_context =
          common_context.has_value() ? context[k + nodes.size()] : tree[1],
      .all_except_each = std::move(all_except_each),
  };
}

using DirectedChannelRef = std::pair<ChannelRef, ChannelDirection>;

enum class ConnectionKind { kInternal, kExternal };

struct Connector {
 public:
  ChannelDirection direction;
  ConnectionKind kind;

  Node* data;
  std::optional<Node*> valid;
  std::optional<Node*> ready;

  // If present, this is a connector where the output is committed at a specific
  // time. (Usually used for Sends)
  std::optional<Node*> commit;

  bool is_one_shot = false;

  // Used only for one-shot connections.
  std::optional<std::pair<Node*, /*operand_no=*/int64_t>> incoming_valid;
  std::optional<Node*> ready_port;

  absl::Status MakeOneShot(Node* new_outgoing_valid,
                           std::optional<Node*> visible_ready,
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
    Node* new_outgoing_valid, std::optional<Node*> visible_ready,
    std::pair<Node*, int64_t> incoming_valid_operand) {
  CHECK(!is_one_shot);
  CHECK_EQ(direction, ChannelDirection::kSend);
  CHECK(valid.has_value());

  XLS_RETURN_IF_ERROR(ReplaceValidSignal(new_outgoing_valid));

  is_one_shot = true;
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

FlopKind GetDefaultFlopKind(bool enabled,
                            ::xls::verilog::CodegenOptions::IOKind kind) {
  if (!enabled) {
    return FlopKind::kNone;
  }
  return ::xls::verilog::CodegenOptions::IOKindToFlopKind(kind);
}

absl::StatusOr<FlopKind> GetFlopKind(
    ChannelRef channel, ChannelDirection direction, ScheduledBlock* block,
    const BlockConversionPassOptions& options) {
  if (ChannelRefKind(channel) == ChannelKind::kSingleValue) {
    // NOTE: We control the flop insertion for single-value channels globally,
    // with no per-channel configuration. This matches the behavior in codegen
    // v1.0.
    if (!options.codegen_options.flop_single_value_channels()) {
      return FlopKind::kNone;
    }
    switch (direction) {
      case ChannelDirection::kSend: {
        if (!options.codegen_options.flop_outputs()) {
          return FlopKind::kNone;
        }
        return FlopKind::kFlop;
      }
      case ChannelDirection::kReceive: {
        if (!options.codegen_options.flop_inputs()) {
          return FlopKind::kNone;
        }
        return FlopKind::kFlop;
      }
    }
    ABSL_UNREACHABLE();
    return absl::InternalError(
        absl::StrFormat("Unknown channel direction %d", direction));
  }

  if (std::holds_alternative<Channel*>(channel)) {
    XLS_RET_CHECK(!block->package()->ChannelsAreProcScoped())
        << "For proc-scoped channels, the flop kind is set on the interface.";
    XLS_RET_CHECK_EQ(ChannelRefKind(channel), ChannelKind::kStreaming);
    StreamingChannel* streaming_channel =
        absl::down_cast<StreamingChannel*>(std::get<Channel*>(channel));
    switch (direction) {
      case ChannelDirection::kSend:
        return streaming_channel->channel_config().output_flop_kind().value_or(
            GetDefaultFlopKind(options.codegen_options.flop_outputs(),
                               options.codegen_options.flop_outputs_kind()));
      case ChannelDirection::kReceive:
        return streaming_channel->channel_config().input_flop_kind().value_or(
            GetDefaultFlopKind(options.codegen_options.flop_inputs(),
                               options.codegen_options.flop_inputs_kind()));
    }
    ABSL_UNREACHABLE();
    return absl::InternalError(
        absl::StrFormat("Unknown channel direction %d", direction));
  } else {
    return std::get<ChannelInterface*>(channel)->flop_kind();
  }
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
  std::optional<Node*> placeholder_commit;
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
    if (ChannelRefFlowControl(channel) != FlowControl::kValidData) {
      XLS_ASSIGN_OR_RETURN(
          ready,
          block->AddInputPort(
              absl::StrCat(
                  ChannelRefName(channel),
                  options.codegen_options.streaming_channel_ready_suffix()),
              block->package()->GetBitsType(1)));
    }
    XLS_ASSIGN_OR_RETURN(
        placeholder_commit,
        block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(0, 1))));
  }

  Connector connector{.direction = ChannelDirection::kSend,
                      .kind = ConnectionKind::kExternal,
                      .data = data,
                      .valid = valid,
                      .ready = ready,
                      .commit = placeholder_commit};

  XLS_ASSIGN_OR_RETURN(
      FlopKind flop_kind,
      GetFlopKind(channel, ChannelDirection::kSend, block, options));
  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      ChannelPortMetadata{.channel_name = std::string(ChannelRefName(channel)),
                          .type = ChannelRefType(channel),
                          .direction = ChannelDirection::kSend,
                          .channel_kind = ChannelRefKind(channel),
                          .flop_kind = flop_kind,
                          .data_port = data->GetName(),
                          .valid_port = GetOptionalNodeName(valid),
                          .ready_port = GetOptionalNodeName(ready)}));

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
    if (ChannelRefFlowControl(channel) != FlowControl::kValidData) {
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
  }

  Connector connector{.direction = ChannelDirection::kReceive,
                      .kind = ConnectionKind::kExternal,
                      .data = data,
                      .valid = valid,
                      .ready = ready};

  XLS_ASSIGN_OR_RETURN(
      FlopKind flop_kind,
      GetFlopKind(channel, ChannelDirection::kReceive, block, options));
  XLS_RETURN_IF_ERROR(block->AddChannelPortMetadata(
      ChannelPortMetadata{.channel_name = std::string(ChannelRefName(channel)),
                          .type = ChannelRefType(channel),
                          .direction = ChannelDirection::kReceive,
                          .channel_kind = ChannelRefKind(channel),
                          .flop_kind = flop_kind,
                          .data_port = data->GetName(),
                          .valid_port = GetOptionalNodeName(valid),
                          .ready_port = GetOptionalNodeName(ready)}));

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
          AddFifoInstantiation(absl::down_cast<StreamingChannel*>(channel),
                               block));
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
          AddFifoInstantiation(absl::down_cast<StreamingChannel*>(channel),
                               block));
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

absl::StatusOr<Node*> MakeStagedUseIfNeeded(Node* operand, int64_t stage_index,
                                            ScheduledBlock* block,
                                            SourceInfo loc = SourceInfo()) {
  if (!block->IsStaged(operand)) {
    // Unstaged nodes don't need their uses to be stages.
    return operand;
  }
  if (stage_index == *block->GetStageIndex(operand)) {
    // Operand is already staged at the correct stage.
    return operand;
  }
  return block->MakeNodeInStage<UnOp>(stage_index, loc, operand, Op::kIdentity);
}

absl::StatusOr<Node*> MakeStagedNode(Node* node, int64_t stage_index,
                                     ScheduledBlock* block,
                                     SourceInfo loc = SourceInfo()) {
  if (block->IsStaged(node) && *block->GetStageIndex(node) == stage_index) {
    return node;
  }
  return block->MakeNodeInStage<UnOp>(stage_index, loc, node, Op::kIdentity);
}

// Combines a new condition into an existing stage control signal
// (active_inputs_valid or active_outputs_ready). If existing_signal is still
// the initial unsigned one literal placeholder, it replaces it; otherwise,
// it ANDs the new condition into the existing signal.
absl::StatusOr<Node*> UpdateStageSignalWithAnd(
    Node* existing_signal, Node* new_condition, int64_t stage_index,
    ScheduledBlock* block, SourceInfo loc = SourceInfo(),
    const std::function<bool(Node*)>& filter = [](Node*) { return true; }) {
  XLS_ASSIGN_OR_RETURN(Node * staged_condition,
                       MakeStagedNode(new_condition, stage_index, block, loc));
  if (IsLiteralUnsignedOne(existing_signal)) {
    if (existing_signal->HasAssignedName() &&
        !staged_condition->HasAssignedName() &&
        !staged_condition->OpIn(
            {Op::kInputPort, Op::kOutputPort, Op::kParam})) {
      std::string name = existing_signal->GetName();
      existing_signal->ClearName();
      staged_condition->SetNameDirectly(name);
    }
    XLS_RETURN_IF_ERROR(existing_signal->ReplaceUsesWith(staged_condition));
    return staged_condition;
  }
  return ReplaceWithAnd(existing_signal, staged_condition,
                        /*combine_literals=*/false, /*name=*/"", loc, filter);
}

struct ReceiveConnection {
  Receive* receive;
  Connector* connector;
};

// Lowers all receives within a single stage to standard block operations,
// adding the ready signals contributed to each connector to the provided map,
// and directly updating the commit signal of each connector.
absl::Status ConnectReceivesForStage(
    int64_t stage_index, absl::Span<const ReceiveConnection> stage_recvs,
    ScheduledBlock* block, const BlockConversionPassOptions& options,
    absl::flat_hash_map<Connector*, std::vector<Node*>>&
        ready_signals_by_connector) {
  Stage& stage = block->stages()[stage_index];
  SourceInfo stage_loc = SourceInfo();
  std::vector<Node*> blocking_nodes;
  absl::flat_hash_map<Receive*, int64_t> blocking_node_by_receive;
  blocking_nodes.reserve(stage_recvs.size());
  blocking_node_by_receive.reserve(stage_recvs.size());

  for (const ReceiveConnection& recv_conn : stage_recvs) {
    Receive* receive = recv_conn.receive;
    stage_loc = stage_loc.Extend(receive->loc());
    Connector* connector = recv_conn.connector;

    std::optional<Node*> predicate = receive->predicate();
    if (predicate.has_value()) {
      XLS_ASSIGN_OR_RETURN(predicate,
                           MakeStagedUseIfNeeded(*predicate, stage_index, block,
                                                 receive->loc()));
    }

    if (connector->valid.has_value() && receive->is_blocking()) {
      Node* recv_valid_or_inactive = *connector->valid;
      if (predicate.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            Node * recv_inactive,
            block->MakeNodeInStage<UnOp>(stage_index, receive->loc(),
                                         *predicate, Op::kNot));
        XLS_ASSIGN_OR_RETURN(
            recv_valid_or_inactive,
            block->MakeNodeInStage<NaryOp>(
                stage_index, receive->loc(),
                absl::MakeConstSpan({*connector->valid, recv_inactive}),
                Op::kOr));
      }
      blocking_node_by_receive[receive] = blocking_nodes.size();
      blocking_nodes.push_back(recv_valid_or_inactive);
    }

    Node* data = connector->data;
    if (options.codegen_options.gate_recvs()) {
      std::vector<Node*> gate_conditions;
      if (predicate.has_value()) {
        gate_conditions.push_back(*predicate);
      }
      if (!receive->is_blocking() && connector->valid.has_value()) {
        gate_conditions.push_back(*connector->valid);
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
                                 ZeroOfType(connector->data->GetType())));
        XLS_ASSIGN_OR_RETURN(data,
                             block->MakeNodeInStage<Select>(
                                 stage_index, receive->loc(), gate_condition,
                                 absl::MakeConstSpan({zero, connector->data}),
                                 /*default_value=*/std::nullopt));
      }
    }

    std::vector<Node*> replacement_elements = {receive->token(), data};
    if (!receive->is_blocking()) {
      Node* valid = nullptr;
      if (connector->valid.has_value()) {
        valid = *connector->valid;
      } else {
        XLS_ASSIGN_OR_RETURN(
            valid, block->MakeNodeInStage<Literal>(stage_index, receive->loc(),
                                                   Value(UBits(1, 1))));
      }
      if (predicate.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            valid, block->MakeNodeInStage<NaryOp>(
                       stage_index, receive->loc(),
                       absl::MakeConstSpan({valid, *predicate}), Op::kAnd));
      }
      replacement_elements.push_back(valid);
    }

    XLS_RETURN_IF_ERROR(receive
                            ->ReplaceUsesWithNewInStage<Tuple>(
                                stage_index, replacement_elements)
                            .status());
  }

  // Compute stage common ready.
  std::vector<Node*> common_ready_terms = {stage.inputs_valid(),
                                           stage.outputs_ready()};
  if (!IsLiteralUnsignedOne(stage.active_inputs_valid())) {
    // Capture any existing `active_inputs_valid` signals (e.g., from state
    // reads).
    common_ready_terms.push_back(stage.active_inputs_valid());
  }
  if (!IsLiteralUnsignedOne(stage.active_outputs_ready())) {
    common_ready_terms.push_back(stage.active_outputs_ready());
  }
  XLS_ASSIGN_OR_RETURN(
      Node * common_ready,
      JoinWithAnd(block, common_ready_terms, /*combine_literals=*/false));

  // We need the conjunction of *all* the blocking nodes with the common ready
  // terms to signal on non-blocking receives' ready signals... but each
  // blocking receive's ready signal needs the conjunction of all *other*
  // blocking nodes with the common ready terms, and the update to our
  // `active_inputs_valid` signal needs us to leave out the common ready terms
  // entirely.
  //
  // Thankfully, it turns out it's efficient to compute all of these at once; we
  // can do it in O(n) gates with O(log n) depth.
  XLS_ASSIGN_OR_RETURN(
      ConjunctionResult conj,
      ComputeConjunctions(block, blocking_nodes, common_ready, stage_loc));

  // Update the `active_inputs_valid` signal to include the conjunction of all
  // blocking inputs' valid signals.
  if (conj.all_without_context != nullptr) {
    XLS_RETURN_IF_ERROR(
        UpdateStageSignalWithAnd(
            stage.active_inputs_valid(), conj.all_without_context, stage_index,
            block, stage_loc, [&](Node* n) { return n != common_ready; })
            .status());
  }

  XLS_ASSIGN_OR_RETURN(Node * stage_done,
                       block->GetOrCreateStageDone(stage_index));

  for (const ReceiveConnection& recv_conn : stage_recvs) {
    Receive* receive = recv_conn.receive;
    Connector* connector = recv_conn.connector;

    std::optional<Node*> predicate = receive->predicate();
    if (predicate.has_value()) {
      XLS_ASSIGN_OR_RETURN(predicate,
                           MakeStagedUseIfNeeded(*predicate, stage_index, block,
                                                 receive->loc()));
    }

    if (connector->ready.has_value()) {
      // The ready signal from this stage should be the predicated conjunction
      // of the `common_ready` signal and all blocking inputs' valid signals,
      // *except* for its own. (This removes self-loops, while respecting the
      // ready-valid protocol.)
      Node* recv_ready = conj.all;
      if (receive->is_blocking()) {
        auto it = blocking_node_by_receive.find(receive);
        CHECK(it != blocking_node_by_receive.end());
        recv_ready = conj.all_except_each[it->second];
      }
      if (predicate.has_value()) {
        XLS_ASSIGN_OR_RETURN(
            recv_ready,
            block->MakeNodeInStage<NaryOp>(
                stage_index, receive->loc(),
                absl::MakeConstSpan({recv_ready, *predicate}), Op::kAnd));
      }
      ready_signals_by_connector[connector].push_back(recv_ready);
    }

    Node* recv_commit = stage_done;
    if (predicate.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          recv_commit,
          block->MakeNodeInStage<NaryOp>(
              stage_index, receive->loc(),
              absl::MakeConstSpan({stage_done, *predicate}), Op::kAnd));
    }
    if (!connector->commit.has_value()) {
      connector->commit = recv_commit;
    } else {
      XLS_ASSIGN_OR_RETURN(*connector->commit,
                           ReplaceWithOr(*connector->commit, recv_commit));
    }

    if (!connector->ready.has_value() && connector->valid.has_value() &&
        options.codegen_options.assert_on_valid_data_not_ready()) {
      XLS_ASSIGN_OR_RETURN(Node * not_valid, block->MakeNodeInStage<UnOp>(
                                                 stage_index, receive->loc(),
                                                 *connector->valid, Op::kNot));
      XLS_ASSIGN_OR_RETURN(
          Node * ready_if_valid,
          block->MakeNodeInStage<NaryOp>(
              stage_index, receive->loc(),
              absl::MakeConstSpan({recv_commit, not_valid}), Op::kOr));
      XLS_RETURN_IF_ERROR(
          block
              ->MakeNode<Assert>(
                  recv_commit->loc(), receive->token(), ready_if_valid,
                  absl::StrCat("Unable to receive ", receive->GetName(),
                               " due to not ready signal."),
                  absl::StrCat(receive->GetName(), ".", stage_index),
                  std::nullopt)
              .status());
    }

    XLS_RETURN_IF_ERROR(block->RemoveNode(receive));
  }
  return absl::OkStatus();
}

absl::Status ConnectReceives(
    const absl::btree_map<DirectedChannelRef, std::vector<Node*>,
                          DirectedNameLessThan>& io_ops,
    absl::flat_hash_map<DirectedChannelRef, Connector>& connections,
    ScheduledBlock* block, const BlockConversionPassOptions& options) {
  // Group receives in the same stage; their signals are related by the stage's
  // control logic.
  absl::btree_map<int64_t, std::vector<ReceiveConnection>> receives_by_stage;
  for (const auto& [directed_channel, io_ops_for_channel] : io_ops) {
    if (directed_channel.second != ChannelDirection::kReceive) {
      continue;
    }
    auto it = connections.find(directed_channel);
    XLS_RET_CHECK(it != connections.end())
        << "Missing connector for channel: "
        << ChannelRefName(directed_channel.first) << " ("
        << ChannelDirectionToString(directed_channel.second) << ")";
    Connector& connector = it->second;
    for (Node* io_op : io_ops_for_channel) {
      XLS_RET_CHECK(io_op->Is<Receive>());
      Receive* receive = io_op->As<Receive>();
      XLS_ASSIGN_OR_RETURN(int64_t stage_index, block->GetStageIndex(receive));

      receives_by_stage[stage_index].push_back(ReceiveConnection{
          .receive = receive,
          .connector = &connector,
      });
    }
  }

  // Process receives stage by stage, collecting ready signals; it's easier to
  // connect them safely after we know the full set of signals for each
  // connector.
  absl::flat_hash_map<Connector*, std::vector<Node*>>
      ready_signals_by_connector;
  for (int64_t stage_index = 0; stage_index < block->stages().size();
       ++stage_index) {
    auto it = receives_by_stage.find(stage_index);
    if (it == receives_by_stage.end()) {
      continue;
    }
    XLS_RETURN_IF_ERROR(ConnectReceivesForStage(
        stage_index, it->second, block, options, ready_signals_by_connector));
  }

  // Connect ready signals to connectors. We assume that at most one receive is
  // active on a channel at a time, so we don't need any conflict-resolution
  // logic.
  //
  // NOTE: We iterate over `io_ops`, rather than `ready_signals_by_connector`,
  //       to guarantee determinism.
  for (const auto& [directed_channel, _] : io_ops) {
    Connector& connector = connections.at(directed_channel);
    auto it = ready_signals_by_connector.find(&connector);
    if (it == ready_signals_by_connector.end()) {
      continue;
    }
    const std::vector<Node*>& ready_signals = it->second;
    XLS_ASSIGN_OR_RETURN(
        Node * ready_signal,
        JoinWithOr(block, ready_signals, /*combine_literals=*/false));
    if (connector.ready.has_value()) {
      XLS_RETURN_IF_ERROR(connector.ReplaceReadySignal(ready_signal));
    }
  }

  return absl::OkStatus();
}

absl::Status ConnectSendsToConnector(
    absl::Span<Node* const> sends, Connector& connector, ScheduledBlock* block,
    const BlockConversionPassOptions& options) {
  XLS_RET_CHECK_EQ(connector.direction, ChannelDirection::kSend);
  SourceInfo loc = SourceInfo();
  for (Node* send : sends) {
    loc = loc.Extend(send->loc());
  }

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
  //    each stage's `active_outputs_ready`.
  // 4. Use the predicates and a OneHotSelect to combine the data signals.
  std::vector<Node*> data_signals;
  std::vector<Node*> valid_conditions;
  data_signals.reserve(sends.size());
  valid_conditions.reserve(sends.size());
  for (const auto& [send, stage_index] : iter::zip(sends, stage_indices)) {
    Stage& stage = block->stages()[stage_index];
    Node* token = send->As<Send>()->token();

    Node* data = send->As<Send>()->data();
    std::optional<Node*> predicate = send->As<Send>()->predicate();

    // If needed, add identity nodes to signal that the predicate & data need to
    // be available at the send's stage. (This enables pipeline register
    // insertion later.)
    XLS_ASSIGN_OR_RETURN(
        data, MakeStagedUseIfNeeded(data, stage_index, block, send->loc()));
    if (predicate.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          predicate,
          MakeStagedUseIfNeeded(*predicate, stage_index, block, send->loc()));
    }

    data_signals.push_back(data);

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
      // The stage's active outputs aren't ready until we've successfully sent
      // on the channel.
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

      XLS_RETURN_IF_ERROR(UpdateStageSignalWithAnd(stage.active_outputs_ready(),
                                                   send_done_or_inactive,
                                                   stage_index, block,
                                                   send->loc())
                              .status());
    }

    // Make sure to update the commit signal with the finished condition for
    // this send.
    XLS_ASSIGN_OR_RETURN(Node * finished,
                         block->GetOrCreateStageDone(stage_index));
    if (predicate.has_value()) {
      XLS_ASSIGN_OR_RETURN(
          finished, block->MakeNode<NaryOp>(
                        send->loc(),
                        absl::MakeConstSpan({*predicate, finished}), Op::kAnd));
    }
    if (!connector.commit.has_value()) {
      connector.commit = finished;
    } else {
      XLS_ASSIGN_OR_RETURN(*connector.commit,
                           ReplaceWithOr(*connector.commit, finished));
    }

    XLS_RETURN_IF_ERROR(send->ReplaceUsesWith(token));
    XLS_RETURN_IF_ERROR(block->RemoveNode(send));
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
    // Reverse the valid conditions, so LSB-to-MSB order will match each
    // condition up with its data signal.
    absl::c_reverse(valid_conditions);
    XLS_ASSIGN_OR_RETURN(
        Node * selector,
        block->MakeNode<Concat>(loc, absl::MakeConstSpan(valid_conditions)));
    XLS_ASSIGN_OR_RETURN(Node * data, block->MakeNode<OneHotSelect>(
                                          loc, selector, data_signals));
    XLS_RETURN_IF_ERROR(connector.ReplaceDataSignal(data));
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
  XLS_RET_CHECK(connector.valid.has_value());

  std::string channel_prefix =
      channel_name.empty() ? "__" : absl::StrCat("__", channel_name, "_");

  // When implementing one-shot logic for a streaming send channel...
  // 1. We add a 1-bit "already_done" register, with a `RegisterRead`.
  // 2. We patch the outgoing valid signal, replacing it with:
  //    `AND(incoming_valid, !already_done)`.
  //    NOTE: updates should affect the "incoming_valid" signal component.
  // 3. We patch the visible "ready" signal, replacing it with:
  //    `OR(incoming_ready, already_done)`.
  // 4. We add the `RegisterWrite` for "already_done"; this needs to start out
  //    disabled, be set to 1 when the outgoing valid & incoming ready signals
  //    are both asserted, but be reset to 0 as soon as the "commit" signal is
  //    asserted.
  //    a. Set `data` to:
  //       `AND(outgoing_valid, incoming_ready, !commit)`.
  //    b. Set `load_enable` to:
  //       `OR(AND(outgoing_valid, incoming_ready), commit)`.

  Node* incoming_valid = *connector.ValidSignal();
  std::optional<Node*> incoming_ready = connector.ready;

  // Ensure `connector.commit` is populated.
  if (!connector.commit.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        connector.commit,
        block->MakeNode<Literal>(SourceInfo(), Value(UBits(0, 1))));
  }

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
  std::optional<Node*> visible_ready = std::nullopt;
  if (incoming_ready.has_value()) {
    XLS_ASSIGN_OR_RETURN(
        visible_ready,
        block->MakeNode<NaryOp>(
            SourceInfo(), absl::MakeConstSpan({*incoming_ready, already_done}),
            Op::kOr));
  }

  // 4. Add the `RegisterWrite` for "already_done".
  std::vector<Node*> done_srcs = {outgoing_valid};
  if (incoming_ready.has_value()) {
    done_srcs.push_back(*incoming_ready);
  }
  XLS_ASSIGN_OR_RETURN(
      Node * done, block->MakeNode<NaryOp>(
                       SourceInfo(), absl::MakeConstSpan(done_srcs), Op::kAnd));
  XLS_ASSIGN_OR_RETURN(
      Node * not_resetting,
      block->MakeNode<UnOp>(SourceInfo(), *connector.commit, Op::kNot));
  XLS_ASSIGN_OR_RETURN(
      Node * already_done_data,
      block->MakeNode<NaryOp>(
          SourceInfo(), absl::MakeConstSpan({done, not_resetting}), Op::kAnd));
  XLS_ASSIGN_OR_RETURN(
      Node * already_done_load_enable,
      block->MakeNode<NaryOp>(SourceInfo(),
                              absl::MakeConstSpan({done, *connector.commit}),
                              Op::kOr));
  XLS_RETURN_IF_ERROR(
      block
          ->MakeNode<RegisterWrite>(SourceInfo(), already_done_data,
                                    already_done_load_enable,
                                    block->GetResetPort(), already_done_reg)
          .status());

  // Actually record all of these changes in the connector, so it can correctly
  // handle these signals.
  return connector.MakeOneShot(outgoing_valid, visible_ready,
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
                                            std::optional<Node*> from_rdy,
                                            std::string_view name_prefix,
                                            Block* block) {
  bool has_ready = from_rdy.has_value();
  if (has_ready) {
    CHECK_EQ((*from_rdy)->operand_count(), 1);
  }

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

  XLS_ASSIGN_OR_RETURN(
      Node * from_skid_rdy,
      block->MakeNodeWithName<UnOp>(
          /*loc=*/SourceInfo(), data_valid_skid_reg_read, Op::kNot,
          absl::StrCat(name_prefix, "_from_skid_rdy")));

  // Skid is loaded from 1st stage whenever
  //   a) the input is being read (input_ready_and_valid == 1) and
  //       --> which implies that the skid is invalid
  //   b) the output is not ready (to_is_ready == 0), if available
  std::vector<Node*> skid_data_load_en_srcs = {from_valid, from_skid_rdy};

  // Skid is reset to invalid (valid is set to zero) whenever
  //   a) skid is valid and
  //   b) output is ready, if available
  std::vector<Node*> skid_valid_set_zero_srcs = {data_valid_skid_reg_read};

  if (has_ready) {
    // Input can be accepted whenever the skid registers
    // are empty/invalid.
    Node* to_is_ready = (*from_rdy)->operand(0);
    XLS_RETURN_IF_ERROR((*from_rdy)->ReplaceOperandNumber(0, from_skid_rdy));
    XLS_ASSIGN_OR_RETURN(Node * to_is_not_rdy,
                         block->MakeNodeWithName<UnOp>(
                             /*loc=*/SourceInfo(), to_is_ready, Op::kNot,
                             absl::StrCat(name_prefix, "_to_is_not_rdy")));
    skid_data_load_en_srcs.push_back(to_is_not_rdy);
    skid_valid_set_zero_srcs.push_back(to_is_ready);
  }

  XLS_ASSIGN_OR_RETURN(
      Node * skid_data_load_en,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(), skid_data_load_en_srcs, Op::kAnd,
          absl::StrCat(name_prefix, "_skid_data_load_en")));

  XLS_ASSIGN_OR_RETURN(
      Node * skid_valid_set_zero,
      block->MakeNodeWithName<NaryOp>(
          /*loc=*/SourceInfo(), skid_valid_set_zero_srcs, Op::kAnd,
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
                                   std::optional<Node*> from_rdy,
                                   std::string_view name_prefix, Block* block) {
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * data_reg_read,
      AddRegisterAfterNode(/*name_prefix=*/name_prefix,
                           /*load_enable=*/std::nullopt, from_data));
  XLS_ASSIGN_OR_RETURN(
      RegisterRead * valid_reg_read,
      AddRegisterAfterNode(/*name_prefix=*/absl::StrCat(name_prefix, "_valid"),
                           /*load_enable=*/std::nullopt, from_valid));

  // 2. Construct and update the ready signal.
  Register* data_reg = data_reg_read->GetRegister();
  Register* valid_reg = valid_reg_read->GetRegister();

  std::string not_valid_name = absl::StrCat(name_prefix, "_valid_inv");
  XLS_ASSIGN_OR_RETURN(
      Node * not_valid,
      block->MakeNodeWithName<UnOp>(/*loc=*/SourceInfo(), valid_reg_read,
                                    Op::kNot, not_valid_name));

  Node* flop_rdy = nullptr;
  if (from_rdy.has_value()) {
    CHECK_EQ((*from_rdy)->operand_count(), 1);
    Node* from_rdy_src = (*from_rdy)->operand(0);

    // The flop is ready to receive new data if it's draining OR empty.
    XLS_ASSIGN_OR_RETURN(
        flop_rdy, block->MakeNodeWithName<NaryOp>(
                      /*loc=*/SourceInfo(),
                      absl::MakeConstSpan({from_rdy_src, not_valid}), Op::kOr,
                      absl::StrCat(name_prefix, "_flop_rdy")));
    XLS_RETURN_IF_ERROR((*from_rdy)->ReplaceOperandNumber(0, flop_rdy));
  }

  Node* data_load_en = nullptr;
  if (flop_rdy == nullptr) {
    // No backpressure, so the data just loads whenever the input is valid.
    data_load_en = from_valid;
  } else {
    // The data loads whenever the input is valid & we're ready to receive it.
    XLS_ASSIGN_OR_RETURN(
        data_load_en,
        block->MakeNodeWithName<NaryOp>(
            /*loc=*/SourceInfo(), absl::MakeConstSpan({from_valid, flop_rdy}),
            Op::kAnd, absl::StrCat(name_prefix, "_load_en")));
  }

  // 3. Update load enables for the data and valid registers.
  XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(data_load_en, data_reg, block));
  if (flop_rdy != nullptr) {
    XLS_RETURN_IF_ERROR(UpdateRegisterLoadEn(flop_rdy, valid_reg, block));
  }

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
                                     std::optional<Node*> from_rdy,
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
                               std::optional<Node*> ready,
                               std::string_view name_prefix, Block* block) {
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

  std::optional<Node*> consumer_ready = connector.ready;
  if (!connector.ready.has_value() && connector.commit.has_value()) {
    // This is a kValidData receive; the consumer is ready exactly if the
    // operation is being committed.
    //
    // We wrap the commit signal in an identity operation to make sure later
    // operations don't interfere with it by trying to patch the signal.
    XLS_ASSIGN_OR_RETURN(
        consumer_ready,
        block->MakeNodeWithName<UnOp>(
            /*loc=*/SourceInfo(), *connector.commit, Op::kIdentity,
            absl::StrCat(ChannelRefName(channel), "_consumer_ready")));
  }
  return AddFlopToRDVNodes(flop_kind, connector.data, *connector.valid,
                           consumer_ready, ChannelRefName(channel), block);
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
  // Re-calculate the port name for valid/ready to match what
  // StreamingIOName does (suffixing based on port type/etc), or just
  // use the name of the node.
  std::string valid_buf_name =
      absl::StrFormat("__%s_buf", (*connector.valid)->GetName());
  std::optional<Node*> output_port_ready_buf;
  if (ChannelRefFlowControl(channel) != FlowControl::kValidData) {
    CHECK(connector.ready.has_value());
    Node* ready_port = connector.ready_port.has_value() ? *connector.ready_port
                                                        : *connector.ready;
    std::string ready_buf_name =
        absl::StrFormat("__%s_buf", ready_port->GetName());
    XLS_ASSIGN_OR_RETURN(
        output_port_ready_buf,
        block->MakeNodeWithName<UnOp>(
            /*loc=*/SourceInfo(), ready_port, Op::kIdentity, ready_buf_name));

    XLS_RETURN_IF_ERROR(ready_port->ReplaceUsesWith(*output_port_ready_buf));
  }

  XLS_ASSIGN_OR_RETURN(Node * output_port_valid_buf,
                       block->MakeNodeWithName<UnOp>(
                           /*loc=*/SourceInfo(), (*connector.valid)->operand(0),
                           Op::kIdentity, valid_buf_name));
  XLS_RETURN_IF_ERROR(
      (*connector.valid)->ReplaceOperandNumber(0, output_port_valid_buf));

  return AddFlopToRDVNodes(flop_kind, output_port_data_buf,
                           output_port_valid_buf, output_port_ready_buf,
                           ChannelRefName(channel), block);
}

absl::Status AddIOFlopsForConnector(Connector& connector, ChannelRef channel,
                                    ScheduledBlock* block,
                                    const BlockConversionPassOptions& options) {
  XLS_ASSIGN_OR_RETURN(
      FlopKind flop_kind,
      GetFlopKind(channel, connector.direction, block, options));
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
    if (options.codegen_options.generate_combinational()) {
      return absl::UnimplementedError(absl::StrFormat(
          "Proc combinational generator only supports streaming output "
          "channels which can be determined to be mutually exclusive, got %d "
          "output channels which were not proven to be mutually exclusive",
          outgoing_channel_count));
    }
    for (const auto& [directed_channel, _] : io_ops) {
      if (directed_channel.second != ChannelDirection::kSend ||
          ChannelRefKind(directed_channel.first) != ChannelKind::kStreaming) {
        continue;
      }
      auto it = connections.find(directed_channel);
      XLS_RET_CHECK(it != connections.end());
      Connector& connector = it->second;
      XLS_RETURN_IF_ERROR(AddOneShotLogic(
          connector, block, options, ChannelRefName(directed_channel.first)));
      changed = true;
    }
  }

  for (const auto& [directed_channel, io_ops_for_channel] : io_ops) {
    if (directed_channel.second != ChannelDirection::kSend) {
      continue;
    }
    auto it = connections.find(directed_channel);
    XLS_RET_CHECK(it != connections.end())
        << "Missing connector for channel: "
        << ChannelRefName(directed_channel.first) << " ("
        << ChannelDirectionToString(directed_channel.second) << ")";
    Connector& connector = it->second;
    XLS_RET_CHECK(absl::c_all_of(
        io_ops_for_channel, [](Node* io_op) { return io_op->Is<Send>(); }));
    XLS_RETURN_IF_ERROR(
        ConnectSendsToConnector(io_ops_for_channel, connector, block, options));
  }

  XLS_RETURN_IF_ERROR(ConnectReceives(io_ops, connections, block, options));

  for (const auto& [directed_channel, _] : io_ops) {
    auto it = connections.find(directed_channel);
    XLS_RET_CHECK(it != connections.end());
    Connector& connector = it->second;
    // Add any configured I/O flops.
    // However, if this is a RAM channel, then we don't want to add any I/O
    // flops, as RamRewritePass manages the appropriate buffering.
    if (!ram_channel_names.contains(ChannelRefName(directed_channel.first))) {
      XLS_RETURN_IF_ERROR(AddIOFlopsForConnector(
          connector, directed_channel.first, block, options));
    }

    changed = true;
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> ChannelToPortIoLoweringPass::RunInternal(
    Package* package, const BlockConversionPassOptions& options,
    PassResults* results, BlockConversionContext& context) const {
  bool changed = false;
  for (const std::unique_ptr<Block>& block : package->blocks()) {
    if (!block->IsScheduled()) {
      continue;
    }
    ScheduledBlock* scheduled_block =
        absl::down_cast<ScheduledBlock*>(block.get());
    if (scheduled_block->source() == nullptr ||
        !scheduled_block->source()->IsProc()) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN((auto [changed_connections, connections]),
                         LowerChannelsToConnectors(scheduled_block, options));
    changed |= changed_connections;

    XLS_ASSIGN_OR_RETURN(bool changed_io,
                         LowerIoToPorts(scheduled_block, connections, options));
    changed |= changed_io;
  }

  return changed;
}

}  // namespace xls::codegen
