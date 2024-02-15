// Copyright 2023 The XLS Authors
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

#include "xls/passes/ram_rewrite_pass.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/common/casts.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {
// Extract and returns the elements of `node`.
//
// Raises an error if `node` is not a Tuple with element types from
// `expected_type`. If an element of `expected_type` is nullopt, it matches
// anything.
template <size_t N>
absl::StatusOr<std::array<Node*, N>> ExtractTupleElements(
    Node* node,
    const std::array<std::optional<Type* const>, N>& expected_element_types,
    std::string_view tuple_name,
    const std::array<const std::string_view, N>& element_names = {}) {
  if (!node->GetType()->IsTuple()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected node %s in %s to have tuple type, but got type %s.",
        node->ToString(), tuple_name, node->GetType()->ToString()));
  }
  TupleType* node_type = node->GetType()->AsTupleOrDie();

  std::array<Node*, N> elements;

  for (int64_t idx = 0; idx < expected_element_types.size(); ++idx) {
    if (expected_element_types[idx].has_value()) {
      if (!expected_element_types[idx].value()->IsEqualTo(
              node_type->element_type(idx))) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Expected %s (tuple %s element %d of %d) to have type %s, got %s.",
            element_names[idx], tuple_name, idx, expected_element_types.size(),
            expected_element_types[idx].value()->ToString(),
            node_type->element_type(idx)->ToString()));
      }
    }
    XLS_ASSIGN_OR_RETURN(
        elements[idx],
        node->function_base()->MakeNode<TupleIndex>(node->loc(), node, idx));
  }
  return elements;
}

// Makes channels for config ram_config with the given name prefix.
absl::StatusOr<absl::flat_hash_map<RamLogicalChannel, Channel*>> MakeChannels(
    Package* p, std::string_view name_prefix, const RamConfig& ram_config,
    Type* data_type, ChannelStrictness strictness) {
  absl::flat_hash_map<RamLogicalChannel, Channel*> channels;

  int64_t addr_width = ram_config.addr_width();
  int64_t data_width = data_type->GetFlatBitCount();

  Type* addr_type = p->GetBitsType(addr_width);
  Type* mask_type = GetMaskType(p, ram_config.mask_width(data_width));

  switch (ram_config.kind) {
    case RamKind::kAbstract: {
      Type* read_req_type = p->GetTupleType({addr_type, mask_type});
      Type* read_resp_type = p->GetTupleType({data_type});
      Type* write_req_type = p->GetTupleType({addr_type, data_type, mask_type});
      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::kAbstractReadReq],
          p->CreateStreamingChannel(absl::StrCat(name_prefix, "_read_req"),
                                    ChannelOps::kSendOnly, read_req_type,
                                    /*initial_values=*/{},
                                    /*fifo_config=*/FifoConfig{.depth = 0},
                                    /*flow_control=*/FlowControl::kReadyValid,
                                    /*strictness=*/strictness));

      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::kAbstractReadResp],
          p->CreateStreamingChannel(absl::StrCat(name_prefix, "_read_resp"),
                                    ChannelOps::kReceiveOnly, read_resp_type,
                                    /*initial_values=*/{},
                                    /*fifo_config=*/FifoConfig{.depth = 0},
                                    /*flow_control=*/FlowControl::kReadyValid,
                                    /*strictness=*/strictness));
      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::kAbstractWriteReq],
          p->CreateStreamingChannel(absl::StrCat(name_prefix, "_write_req"),
                                    ChannelOps::kSendOnly, write_req_type,
                                    /*initial_values=*/{},
                                    /*fifo_config=*/FifoConfig{.depth = 0},
                                    /*flow_control=*/FlowControl::kReadyValid,
                                    /*strictness=*/strictness));
      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::kWriteCompletion],
          p->CreateStreamingChannel(
              absl::StrCat(name_prefix, "_write_completion"),
              ChannelOps::kReceiveOnly, p->GetTupleType({}),
              /*initial_values=*/{}, /*fifo_config=*/FifoConfig{.depth = 0},
              /*flow_control=*/FlowControl::kReadyValid,
              /*strictness=*/strictness));
      break;
    }
    case RamKind::k1RW: {
      Type* bool_type = p->GetBitsType(1);
      Type* req_type = p->GetTupleType(
          {addr_type, data_type, mask_type, mask_type, bool_type, bool_type});
      Type* resp_type = p->GetTupleType({data_type});
      Type* empty_tuple_type = p->GetTupleType({});

      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::k1RWReq],
          p->CreateStreamingChannel(absl::StrCat(name_prefix, "_req"),
                                    ChannelOps::kSendOnly, req_type,
                                    /*initial_values=*/{},
                                    /*fifo_config=*/FifoConfig{.depth = 0},
                                    /*flow_control=*/FlowControl::kReadyValid,
                                    /*strictness=*/strictness));
      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::k1RWResp],
          p->CreateStreamingChannel(absl::StrCat(name_prefix, "_resp"),
                                    ChannelOps::kReceiveOnly, resp_type,
                                    /*initial_values=*/{},
                                    /*fifo_config=*/FifoConfig{.depth = 0},
                                    /*flow_control=*/FlowControl::kReadyValid,
                                    /*strictness=*/strictness));

      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::kWriteCompletion],
          p->CreateStreamingChannel(
              absl::StrCat(name_prefix, "_write_completion"),
              ChannelOps::kReceiveOnly, empty_tuple_type,
              /*initial_values=*/{}, /*fifo_config=*/FifoConfig{.depth = 0},
              /*flow_control=*/FlowControl::kReadyValid,
              /*strictness=*/strictness));
      break;
    }
    case RamKind::k1R1W: {
      Type* read_req_type = p->GetTupleType({addr_type, mask_type});
      Type* read_resp_type = p->GetTupleType({data_type});
      Type* write_req_type = p->GetTupleType({addr_type, data_type, mask_type});
      Type* empty_tuple_type = p->GetTupleType({});

      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::k1R1WReadReq],
          p->CreateStreamingChannel(absl::StrCat(name_prefix, "_read_req"),
                                    ChannelOps::kSendOnly, read_req_type,
                                    /*initial_values=*/{},
                                    /*fifo_config=*/FifoConfig{.depth = 0},
                                    /*flow_control=*/FlowControl::kReadyValid,
                                    /*strictness=*/strictness));
      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::k1R1WReadResp],
          p->CreateStreamingChannel(absl::StrCat(name_prefix, "_read_resp"),
                                    ChannelOps::kReceiveOnly, read_resp_type,
                                    /*initial_values=*/{},
                                    /*fifo_config=*/FifoConfig{.depth = 0},
                                    /*flow_control=*/FlowControl::kReadyValid,
                                    /*strictness=*/strictness));
      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::k1R1WWriteReq],
          p->CreateStreamingChannel(absl::StrCat(name_prefix, "_write_req"),
                                    ChannelOps::kSendOnly, write_req_type,
                                    /*initial_values=*/{},
                                    /*fifo_config=*/FifoConfig{.depth = 0},
                                    /*flow_control=*/FlowControl::kReadyValid,
                                    /*strictness=*/strictness));
      XLS_ASSIGN_OR_RETURN(
          channels[RamLogicalChannel::kWriteCompletion],
          p->CreateStreamingChannel(
              absl::StrCat(name_prefix, "_write_completion"),
              ChannelOps::kReceiveOnly, empty_tuple_type,
              /*initial_values=*/{}, /*fifo_config=*/FifoConfig{.depth = 0},
              /*flow_control=*/FlowControl::kReadyValid,
              /*strictness=*/strictness));
      break;
    }
    default: {
      return absl::UnimplementedError(
          absl::StrFormat("Cannot create channels for kind %s",
                          RamKindToString(ram_config.kind)));
    }
  }
  return channels;
}

// Returns a mapping from logical names to channels for a new ram with config
// ram_config.
//
// If a model builder is supplied, invoke it and add the result to the current
// package. If a model builder is not supplied, call MakeChannels() to directly
// create bare channels for the given config.
absl::StatusOr<absl::flat_hash_map<RamLogicalChannel, Channel*>>
GetChannelsForNewRam(Package* p, std::string_view name_prefix,
                     const RamConfig& ram_config, Type* data_type,
                     ChannelStrictness strictness,
                     std::optional<ram_model_builder_t> ram_model_builder) {
  if (ram_model_builder.has_value()) {
    auto [new_package, new_channel_mapping] = (*ram_model_builder)(ram_config);
    XLS_ASSIGN_OR_RETURN(auto linked_channel_mapping,
                         p->AddPackage(new_package.get()));
    absl::flat_hash_map<RamLogicalChannel, Channel*> resolved_channel_mapping;
    for (const auto& [logical_name, original_channel] : new_channel_mapping) {
      auto new_channel_it =
          linked_channel_mapping.channel_updates.find(original_channel);
      if (new_channel_it == linked_channel_mapping.channel_updates.end()) {
        return absl::InternalError(absl::StrFormat(
            "Could not find new channel for linked channel %s.", logical_name));
      }
      XLS_ASSIGN_OR_RETURN(RamLogicalChannel logical_channel,
                           RamLogicalChannelFromName(logical_name));
      XLS_ASSIGN_OR_RETURN(Channel * new_channel,
                           p->GetChannel(new_channel_it->second));
      resolved_channel_mapping.insert({logical_channel, new_channel});
    }
    return resolved_channel_mapping;
  }
  return MakeChannels(p, name_prefix, ram_config, data_type, strictness);
}

// Map channels logical names from one ram kind to another, e.g. an abstract
// RAM's "read_req" channel should be rewritten to a 1RW RAM's "req" channel.
//
// TODO(rigge): do actual scheduling instead of this simple mapping.
absl::StatusOr<RamLogicalChannel> MapChannel(RamKind from_kind,
                                             RamLogicalChannel logical_channel,
                                             RamKind to_kind) {
  switch (from_kind) {
    case RamKind::kAbstract: {
      switch (to_kind) {
        case RamKind::k1RW: {
          switch (logical_channel) {
            case RamLogicalChannel::kAbstractReadReq:
            case RamLogicalChannel::kAbstractWriteReq:
              return RamLogicalChannel::k1RWReq;
            case RamLogicalChannel::kAbstractReadResp:
              return RamLogicalChannel::k1RWResp;
            case RamLogicalChannel::kWriteCompletion:
              return RamLogicalChannel::kWriteCompletion;
            default:
              return absl::InvalidArgumentError(
                  absl::StrFormat("Invalid logical channel %s for RAM kind %s.",
                                  RamLogicalChannelName(logical_channel),
                                  RamKindToString(from_kind)));
          }
        }
        case RamKind::k1R1W: {
          switch (logical_channel) {
            case RamLogicalChannel::kAbstractReadReq:
              return RamLogicalChannel::k1R1WReadReq;
            case RamLogicalChannel::kAbstractReadResp:
              return RamLogicalChannel::k1R1WReadResp;
            case RamLogicalChannel::kAbstractWriteReq:
              return RamLogicalChannel::k1R1WWriteReq;
            case RamLogicalChannel::kWriteCompletion:
              return RamLogicalChannel::kWriteCompletion;
            default:
              return absl::InvalidArgumentError(
                  absl::StrFormat("Invalid logical channel %s for RAM kind %s.",
                                  RamLogicalChannelName(logical_channel),
                                  RamKindToString(from_kind)));
          }
        }
        default: {
          return absl::UnimplementedError(absl::StrFormat(
              "Channel scheduling not implemented from kind %s to %s.",
              RamKindToString(from_kind), RamKindToString(to_kind)));
        }
      }
    }
    default: {
      return absl::UnimplementedError(absl::StrFormat(
          "Channel scheduling not implemented from kind %s to %s.",
          RamKindToString(from_kind), RamKindToString(to_kind)));
    }
  }
}

// Repack a tuple going into a send or coming out of a receive from one ram
// config to another. For example, an abstract RAM's read_req with (addr, mask)
// will be mapped to a 1RW RAM's req with (addr, data, we=0, re=1).
// It is recommended for new kinds of RAMs that any functionality more
// complicated than repacking, adding literals, dropping entries, or very simple
// operations be implemented in a wrapper proc rather than here if at all
// possible.
absl::StatusOr<Node*> RepackPayload(FunctionBase* fb, Node* operand,
                                    RamLogicalChannel logical_channel,
                                    const RamConfig& from_config,
                                    const RamConfig& to_config,
                                    Type* data_type) {
  Type* addr_type = fb->package()->GetBitsType(from_config.addr_width());
  int64_t data_width = data_type->GetFlatBitCount();
  Type* mask_type =
      GetMaskType(fb->package(), from_config.mask_width(data_width));
  Type* token_type = fb->package()->GetTokenType();

  if (from_config.kind == RamKind::kAbstract &&
      to_config.kind == RamKind::k1RW) {
    switch (logical_channel) {
      case RamLogicalChannel::kAbstractReadReq: {
        // Abstract read_req is (addr, mask).
        XLS_ASSIGN_OR_RETURN(
            auto addr_and_read_mask,
            ExtractTupleElements<2>(operand, {addr_type, mask_type}, "read_req",
                                    {"addr", "mask"}));
        auto& [addr, read_mask] = addr_and_read_mask;

        // Data can be anything on read, make a zero literal.
        XLS_ASSIGN_OR_RETURN(
            auto* data,
            fb->MakeNode<Literal>(operand->loc(), ZeroOfType(data_type)));
        // Write mask is unused on read req, make a zero literal.
        XLS_ASSIGN_OR_RETURN(
            auto* write_mask,
            fb->MakeNode<Literal>(operand->loc(), ZeroOfType(mask_type)));
        // Make a 0 and 1 literal for we and we, respectively.
        XLS_ASSIGN_OR_RETURN(
            auto* we, fb->MakeNode<Literal>(operand->loc(),
                                            Value(UBits(0, /*bit_count=*/1))));
        XLS_ASSIGN_OR_RETURN(
            auto* re, fb->MakeNode<Literal>(operand->loc(),
                                            Value(UBits(1, /*bit_count=*/1))));
        // TODO(rigge): update when 1RW supports mask.
        return fb->MakeNode<Tuple>(
            operand->loc(),
            std::vector<Node*>{addr, data, write_mask, read_mask, we, re});
      }
      case RamLogicalChannel::kAbstractReadResp: {
        // Abstract read_req is (tok, (data)), same for abstract and 1RW.
        // We'll extract the elements to check their type, but we aren't
        // actually repacking so we'll discard them when we're done.
        XLS_ASSIGN_OR_RETURN(
            auto tok_and_data_tuple,
            ExtractTupleElements<2>(
                operand, {token_type, fb->package()->GetTupleType({data_type})},
                "read_resp", {"token", "data"}));
        auto& [tok, data_tuple] = tok_and_data_tuple;
        XLS_RETURN_IF_ERROR(fb->RemoveNode(tok));
        XLS_RETURN_IF_ERROR(fb->RemoveNode(data_tuple));

        // After we're sure the type is OK, simply return unaltered.
        return operand;
      }
      case RamLogicalChannel::kAbstractWriteReq: {
        // Abstract write_req is (addr, data, mask). First, check operand's
        // type.
        XLS_ASSIGN_OR_RETURN(
            auto addr_data_and_write_mask,
            ExtractTupleElements<3>(operand, {addr_type, data_type, mask_type},
                                    "write_req", {"addr", "data", "mask"}));
        auto& [addr, data, write_mask] = addr_data_and_write_mask;
        // Read mask is unused on read req, make a zero literal.
        XLS_ASSIGN_OR_RETURN(
            auto* read_mask,
            fb->MakeNode<Literal>(operand->loc(), ZeroOfType(mask_type)));
        XLS_ASSIGN_OR_RETURN(
            auto* we, fb->MakeNode<Literal>(operand->loc(),
                                            Value(UBits(1, /*bit_count=*/1))));
        XLS_ASSIGN_OR_RETURN(
            auto* re, fb->MakeNode<Literal>(operand->loc(),
                                            Value(UBits(0, /*bit_count=*/1))));
        // TODO(rigge): update when 1RW supports mask.
        return fb->MakeNode<Tuple>(
            operand->loc(),
            std::vector<Node*>{addr, data, write_mask, read_mask, we, re});
      }
      case RamLogicalChannel::kWriteCompletion: {
        // write_completion is (tok, ()). First, check operand's type.
        XLS_RETURN_IF_ERROR(ExtractTupleElements<2>(
                                operand,
                                {token_type, fb->package()->GetTupleType({})},
                                "write_completion", {"token", "empty_tuple"})
                                .status());
        // After we're sure the type is OK, simply return unaltered.
        return operand;
      }
      default: {
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid logical channel %s for RAM kind %s.",
                            RamLogicalChannelName(logical_channel),
                            RamKindToString(from_config.kind)));
      }
    }
  }
  if (from_config.kind == RamKind::kAbstract &&
      to_config.kind == RamKind::k1R1W) {
    switch (logical_channel) {
      case RamLogicalChannel::kAbstractReadReq:
      case RamLogicalChannel::kAbstractReadResp:
      case RamLogicalChannel::kAbstractWriteReq:
      case RamLogicalChannel::kWriteCompletion:
        return operand;
      default:
        return absl::InvalidArgumentError(
            absl::StrFormat("Invalid logical channel %s for RAM kind %s.",
                            RamLogicalChannelName(logical_channel),
                            RamKindToString(from_config.kind)));
    }
  }

  return absl::UnimplementedError(absl::StrFormat(
      "Repacking not supported for %s -> %s", RamKindToString(from_config.kind),
      RamKindToString(to_config.kind)));
}

// For sends, first we update the payload, then we update the send to use the
// new payload. The old and new send both return the same thing: a single token.
absl::Status ReplaceSend(Proc* proc, Send* old_send,
                         RamLogicalChannel logical_channel,
                         const RamConfig& from_config,
                         const RamConfig& to_config, Type* data_type,
                         std::string_view new_channel) {
  XLS_ASSIGN_OR_RETURN(Node * new_payload,
                       RepackPayload(proc, old_send->data(), logical_channel,
                                     from_config, to_config, data_type));
  XLS_RETURN_IF_ERROR(
      old_send
          ->ReplaceUsesWithNew<Send>(old_send->token(), new_payload,
                                     old_send->predicate(), new_channel)
          .status());
  XLS_RETURN_IF_ERROR(proc->RemoveNode(old_send));
  return absl::OkStatus();
}

// Receives are different than sends: only the channel_id (and hence the type)
// change. The return value is different, though, so we need to hunt down all of
// those and replace them with a repacked version. The steps are as follows:
// 1. Add new receives that use the new channel.
// 2. Update the value returned by the new receive to look like the old
// receive.
// 3. Replace the old receive with the repacked new receives.
absl::Status ReplaceReceive(Proc* proc, Receive* old_receive,
                            RamLogicalChannel logical_channel,
                            const RamConfig& from_config,
                            const RamConfig& to_config, Type* data_type,
                            std::string_view new_channel) {
  std::optional<Node*> new_receive;
  if (from_config.kind == RamKind::kAbstract &&
      (to_config.kind == RamKind::k1RW || to_config.kind == RamKind::k1R1W)) {
    switch (logical_channel) {
      case RamLogicalChannel::kAbstractReadResp: {
        XLS_ASSIGN_OR_RETURN(
            new_receive,
            proc->MakeNode<Receive>(old_receive->loc(), old_receive->token(),
                                    old_receive->predicate(), new_channel,
                                    old_receive->is_blocking()));
        break;
      }
      case RamLogicalChannel::kWriteCompletion: {
        XLS_ASSIGN_OR_RETURN(
            new_receive,
            proc->MakeNode<Receive>(old_receive->loc(), old_receive->token(),
                                    old_receive->predicate(), new_channel,
                                    old_receive->is_blocking()));
        break;
      }
      default: {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Invalid logical channel %s for receive on RAM kind %s.",
            RamLogicalChannelName(logical_channel),
            RamKindToString(from_config.kind)));
      }
    }
  } else {
    return absl::UnimplementedError(absl::StrFormat(
        "New receive not supported for %s -> %s",
        RamKindToString(from_config.kind), RamKindToString(to_config.kind)));
  }
  if (!new_receive.has_value()) {
    return absl::InternalError(
        "new_receive was not set and a more informative error should already "
        "have been returned.");
  }
  XLS_ASSIGN_OR_RETURN(Node * new_return_value,
                       RepackPayload(proc, new_receive.value(), logical_channel,
                                     from_config, to_config, data_type));

  XLS_RETURN_IF_ERROR(old_receive->ReplaceUsesWith(new_return_value));
  XLS_RETURN_IF_ERROR(proc->RemoveNode(old_receive));
  return absl::OkStatus();
}

// Replace sends and receives on old channels with sends and receives on new
// channels. This involves:
// 1. Repacking the inputs to sends.
// 2. Making a new send with the old send's predicate and token and the repacked
// input on the new channel.
// 3. Replacing usages of the old send with the new send.
// 4. Making a new receive with the same arguments as the old using the new
// channel.
// 5. Repacking the output of the new receive to look like the output of the
// previous receive.
// 6. Replacing usages of the old receive with the new repacking.
// 7. For both sends and receives, remove the old send/receive when their usages
// have been replaced.
absl::Status ReplaceChannelReferences(
    Package* p,
    const absl::flat_hash_map<RamLogicalChannel, Channel*>& from_mapping,
    const RamConfig& from_config, Type* data_type,
    const absl::flat_hash_map<RamLogicalChannel, Channel*>& to_mapping,
    const RamConfig& to_config) {
  // Make a reverse mapping from channel -> logical name.
  // Used for figuring out what kind of channel each send/recv is operating on.
  absl::flat_hash_map<Channel*, RamLogicalChannel> reverse_from_mapping;
  for (auto& [logical_channel, channel] : from_mapping) {
    if (auto [it, inserted] =
            reverse_from_mapping.try_emplace(channel, logical_channel);
        !inserted) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Channel mapping must be one-to-one, got multiple "
                          "names for channel %s.",
                          channel->name()));
    }
  }

  for (auto& proc : p->procs()) {
    for (Node* node : TopoSort(proc.get())) {
      if (node->Is<Send>()) {
        Send* old_send = node->As<Send>();
        // If this send operates on a channel in our mapping, find the new
        // channel and replace this send with a new send to the new channel.
        XLS_ASSIGN_OR_RETURN(
            Channel * old_channel,
            proc->package()->GetChannel(old_send->channel_name()));
        auto it = reverse_from_mapping.find(old_channel);
        if (it == reverse_from_mapping.end()) {
          continue;
        }
        RamLogicalChannel logical_channel = it->second;
        XLS_ASSIGN_OR_RETURN(
            RamLogicalChannel new_logical_channel,
            MapChannel(from_config.kind, logical_channel, to_config.kind));
        std::string_view new_channel =
            to_mapping.at(new_logical_channel)->name();
        XLS_RETURN_IF_ERROR(ReplaceSend(proc.get(), old_send, logical_channel,
                                        from_config, to_config, data_type,
                                        new_channel));
      } else if (node->Is<Receive>()) {
        // If this receive operates on a channel in our mapping, find the new
        // channel and replace this receive with a new receive to the new
        // channel, rewriting the output to match the old channel.
        Receive* old_receive = node->As<Receive>();
        XLS_ASSIGN_OR_RETURN(
            Channel * old_channel,
            proc->package()->GetChannel(old_receive->channel_name()));
        auto it = reverse_from_mapping.find(old_channel);
        if (it == reverse_from_mapping.end()) {
          continue;
        }
        RamLogicalChannel logical_channel = it->second;
        XLS_ASSIGN_OR_RETURN(
            RamLogicalChannel new_logical_channel,
            MapChannel(from_config.kind, logical_channel, to_config.kind));
        std::string_view new_channel =
            to_mapping.at(new_logical_channel)->name();
        XLS_RETURN_IF_ERROR(ReplaceReceive(proc.get(), old_receive,
                                           logical_channel, from_config,
                                           to_config, data_type, new_channel));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<std::optional<Type*>> DataTypeFromChannelType(
    Type* channel_type, RamLogicalChannel logical_channel) {
  switch (logical_channel) {
    case RamLogicalChannel::kAbstractReadReq: {
      return std::nullopt;
    }
    case RamLogicalChannel::kAbstractReadResp: {
      if (!channel_type->IsTuple() ||
          channel_type->AsTupleOrDie()->size() != 1) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Abstract read resp must be tuple type with 1 element, got %s",
            channel_type->ToString()));
      }
      // data is the only element
      return channel_type->AsTupleOrDie()->element_type(0);
    }
    case RamLogicalChannel::kAbstractWriteReq: {
      if (!channel_type->IsTuple() ||
          channel_type->AsTupleOrDie()->size() != 3) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Abstract write req must be tuple type with 3 elements, got %s",
            channel_type->ToString()));
      }
      // data is the second element
      return channel_type->AsTupleOrDie()->element_type(1);
    }
    case RamLogicalChannel::k1RWReq: {
      if (!channel_type->IsTuple() ||
          channel_type->AsTupleOrDie()->size() != 4) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "1RW write req must be tuple type with 4 elements, got %s",
            channel_type->ToString()));
      }
      // data is the second element
      return channel_type->AsTupleOrDie()->element_type(1);
    }
    case RamLogicalChannel::k1RWResp: {
      if (!channel_type->IsTuple() ||
          channel_type->AsTupleOrDie()->size() != 1) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "1RW read resp must be tuple type with 1 element, got %s",
            channel_type->ToString()));
      }
      // data is the only element
      return channel_type->AsTupleOrDie()->element_type(0);
    }
    case RamLogicalChannel::k1R1WReadReq: {
      return std::nullopt;
    }
    case RamLogicalChannel::k1R1WReadResp: {
      if (!channel_type->IsTuple() ||
          channel_type->AsTupleOrDie()->size() != 1) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "1R1W read resp must be tuple type with 1 element, got %s",
            channel_type->ToString()));
      }
      // data is the only element
      return channel_type->AsTupleOrDie()->element_type(0);
    }
    case RamLogicalChannel::k1R1WWriteReq: {
      // write req is (addr, data, mask)
      if (!channel_type->IsTuple() ||
          channel_type->AsTupleOrDie()->size() != 3) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "1R1W write req must be tuple type with 3 elements, got %s",
            channel_type->ToString()));
      }
      // data is the second element
      return channel_type->AsTupleOrDie()->element_type(1);
    }
    case RamLogicalChannel::kWriteCompletion: {
      return std::nullopt;
    }
  }
}

absl::StatusOr<Type*> DataTypeFromChannels(
    const absl::flat_hash_map<RamLogicalChannel, Channel*>&
        logical_to_channels) {
  std::optional<Type*> data_type;

  for (const auto& [logical_channel, channel] : logical_to_channels) {
    XLS_ASSIGN_OR_RETURN(
        std::optional<Type*> new_data_type,
        DataTypeFromChannelType(channel->type(), logical_channel));
    if (data_type.has_value()) {
      if (new_data_type.has_value() &&
          !data_type.value()->IsEqualTo(new_data_type.value())) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Multiple data types detected, %s!=%s.",
            data_type.value()->ToString(), new_data_type.value()->ToString()));
      }
    } else {
      data_type = new_data_type;
    }
  }

  if (!data_type.has_value()) {
    return absl::InvalidArgumentError("No data type found in channels.");
  }
  return data_type.value();
}
}  // namespace

absl::StatusOr<RamLogicalChannel> RamLogicalChannelFromName(
    std::string_view name) {
  if (name == "abstract_read_req") {
    return RamLogicalChannel::kAbstractReadReq;
  }
  if (name == "abstract_read_resp") {
    return RamLogicalChannel::kAbstractReadResp;
  }
  if (name == "abstract_write_req") {
    return RamLogicalChannel::kAbstractWriteReq;
  }
  if (name == "1rw_req") {
    return RamLogicalChannel::k1RWReq;
  }
  if (name == "1rw_resp") {
    return RamLogicalChannel::k1RWResp;
  }
  if (name == "1r1w_read_req") {
    return RamLogicalChannel::k1R1WReadReq;
  }
  if (name == "1r1w_read_resp") {
    return RamLogicalChannel::k1R1WReadResp;
  }
  if (name == "1r1w_write_req") {
    return RamLogicalChannel::k1R1WWriteReq;
  }
  if (name == "write_completion") {
    return RamLogicalChannel::kWriteCompletion;
  }
  return absl::InvalidArgumentError(
      absl::StrFormat("Unrecognized logical channel name for ram: %s.", name));
}

std::string_view RamLogicalChannelName(RamLogicalChannel logical_channel) {
  switch (logical_channel) {
    case RamLogicalChannel::kAbstractReadReq:
      return "abstract_read_req";
    case RamLogicalChannel::kAbstractReadResp:
      return "abstract_read_resp";
    case RamLogicalChannel::kAbstractWriteReq:
      return "abstract_write_req";
    case RamLogicalChannel::k1RWReq:
      return "1rw_req";
    case RamLogicalChannel::k1RWResp:
      return "1rw_resp";
    case RamLogicalChannel::k1R1WReadReq:
      return "1r1w_read_req";
    case RamLogicalChannel::k1R1WReadResp:
      return "1r1w_read_resp";
    case RamLogicalChannel::k1R1WWriteReq:
      return "1r1w_write_req";
    case RamLogicalChannel::kWriteCompletion:
      return "write_completion";
  }
}

Type* GetMaskType(Package* package, std::optional<int64_t> mask_width) {
  if (mask_width.has_value()) {
    return package->GetBitsType(mask_width.value());
  }
  return package->GetTupleType({});
}

absl::StatusOr<bool> RamRewritePass::RunInternal(
    Package* p, const OptimizationPassOptions& options,
    PassResults* results) const {
  // Given the mapping from logical names (e.g. read_req) to physical names
  // (e.g. channel_for_read_req_0), build a new mapping from logical names ->
  // Channel objects.
  absl::flat_hash_map<RamLogicalChannel, Channel*> old_logical_to_channels;
  for (auto& rewrite : options.ram_rewrites) {
    old_logical_to_channels.clear();
    old_logical_to_channels.reserve(
        rewrite.from_channels_logical_to_physical.size());
    for (auto& [logical_name, physical_name] :
         rewrite.from_channels_logical_to_physical) {
      XLS_ASSIGN_OR_RETURN(RamLogicalChannel logical_channel,
                           RamLogicalChannelFromName(logical_name));
      XLS_ASSIGN_OR_RETURN(Channel * resolved_channel,
                           p->GetChannel(physical_name));
      old_logical_to_channels.insert({logical_channel, resolved_channel});
    }

    XLS_ASSIGN_OR_RETURN(Type * data_type,
                         DataTypeFromChannels(old_logical_to_channels));

    std::vector<ChannelStrictness> strictnesses;
    for (auto& [_, base_channel] : old_logical_to_channels) {
      strictnesses.push_back(
          down_cast<StreamingChannel*>(base_channel)->GetStrictness());
    }
    XLS_CHECK(!strictnesses.empty());
    XLS_CHECK(std::equal(strictnesses.begin() + 1, strictnesses.end(),
                         strictnesses.begin()));
    ChannelStrictness strictness = strictnesses.front();

    XLS_ASSIGN_OR_RETURN(
        auto new_logical_to_channels,
        GetChannelsForNewRam(p, rewrite.to_name_prefix, rewrite.to_config,
                             data_type, strictness, rewrite.model_builder));
    XLS_RETURN_IF_ERROR(ReplaceChannelReferences(
        p, old_logical_to_channels, rewrite.from_config, data_type,
        new_logical_to_channels, rewrite.to_config));

    // ReplaceChannelReferences() removes old sends and receives, but the old
    // channels are still there. Remove them.
    for (auto& [logical_name, channel] : old_logical_to_channels) {
      XLS_RETURN_IF_ERROR(p->RemoveChannel(channel));
    }
  }
  return !options.ram_rewrites.empty();
}

REGISTER_OPT_PASS(RamRewritePass);

}  // namespace xls
