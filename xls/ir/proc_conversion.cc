// Copyright 2024 The XLS Authors
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

#include "xls/ir/proc_conversion.h"

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/casts.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"

namespace xls {
namespace {

// Data structure describing relationships between procs and channels.
struct ChannelProcMap {
  // Map from proc to the set of channels used by the proc.
  absl::flat_hash_map<Proc*,
                      absl::btree_set<Channel*, struct Channel::NameLessThan>>
      proc_to_channels;

  // Map from channel to the set of procs which use the channel.
  absl::flat_hash_map<Channel*,
                      absl::btree_set<Proc*, struct FunctionBase::NameLessThan>>
      channel_to_procs;

  // Map from (proc, channel) pair to the operations (send and/or receive) which
  // the proc performs on the channel.
  absl::flat_hash_map<std::pair<Proc*, Channel*>, absl::btree_set<Direction>>
      directions;

  // Returns true if the given channel is a send-receive channel which is only
  // used in a single proc.
  bool IsLoopbackChannel(Channel* channel) {
    return channel->supported_ops() == ChannelOps::kSendReceive &&
           channel_to_procs.at(channel).size() == 1;
  }
};

absl::StatusOr<ChannelProcMap> GetChannelProcMap(Package* package) {
  ChannelProcMap channel_map;
  for (Channel* channel : package->channels()) {
    // Create empty set.
    channel_map.channel_to_procs[channel];
  }
  for (const std::unique_ptr<Proc>& proc : package->procs()) {
    // Create empty set.
    channel_map.proc_to_channels[proc.get()];
  }

  for (const std::unique_ptr<Proc>& proc : package->procs()) {
    for (Node* node : proc->nodes()) {
      if (node->Is<ChannelNode>()) {
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
        channel_map.channel_to_procs[channel].insert(proc.get());
        channel_map.proc_to_channels[proc.get()].insert(channel);
        channel_map.directions[{proc.get(), channel}].insert(
            node->Is<Receive>() ? Direction::kReceive : Direction::kSend);
      }
    }
  }

  return std::move(channel_map);
}

absl::Status DeclareChannelInProc(Proc* proc, Channel* channel) {
  if (channel->kind() == ChannelKind::kStreaming) {
    return proc
        ->AddChannel(std::make_unique<StreamingChannel>(
            *down_cast<StreamingChannel*>(channel)))
        .status();
  }
  return proc
      ->AddChannel(std::make_unique<SingleValueChannel>(
          *down_cast<SingleValueChannel*>(channel)))
      .status();
}

absl::Status AddInterfaceChannel(Proc* proc, Channel* channel,
                                 Direction direction) {
  std::optional<ChannelStrictness> strictness;
  if (StreamingChannel* streaming_channel =
          dynamic_cast<StreamingChannel*>(channel)) {
    strictness = streaming_channel->GetStrictness();
  }
  std::unique_ptr<ChannelReference> channel_ref;
  if (direction == Direction::kSend) {
    channel_ref = std::make_unique<SendChannelReference>(
        channel->name(), channel->type(), channel->kind(), strictness);
  } else {
    channel_ref = std::make_unique<ReceiveChannelReference>(
        channel->name(), channel->type(), channel->kind(), strictness);
  }
  return proc->AddInterfaceChannelReference(std::move(channel_ref)).status();
}

}  // namespace

absl::Status ConvertPackageToNewStyleProcs(Package* package) {
  if (!package->GetTop().has_value() || !package->GetTop().value()->IsProc()) {
    return absl::InvalidArgumentError(
        "Package top must be proc to convert package to new style procs");
  }
  for (const std::unique_ptr<Proc>& proc : package->procs()) {
    if (proc->is_new_style_proc()) {
      return absl::InvalidArgumentError(
          "Package already contains new style proc(s)");
    }
  }
  Proc* top = package->GetTop().value()->AsProcOrDie();

  XLS_ASSIGN_OR_RETURN(ChannelProcMap channel_map, GetChannelProcMap(package));

  for (const std::unique_ptr<Proc>& proc : package->procs()) {
    // This simply sets a bit on the proc. The procs are malformed until the
    // interface and channel declarations are added.
    XLS_RETURN_IF_ERROR(proc->ConvertToNewStyle());
  }

  // Create a proc-scoped or interface channels for each global channel.
  for (Channel* channel : package->channels()) {
    if (channel->supported_ops() != ChannelOps::kSendReceive) {
      // This is a channel on the interface of the design. Add as interface
      // channels on the top proc.
      XLS_RETURN_IF_ERROR(
          AddInterfaceChannel(top, channel,
                              channel->supported_ops() == ChannelOps::kSendOnly
                                  ? Direction::kSend
                                  : Direction::kReceive));
    } else if (channel_map.IsLoopbackChannel(channel)) {
      // Loopback channels are declared in the procs in which they are used.
      XLS_RET_CHECK_EQ(channel_map.channel_to_procs[channel].size(), 1);
      Proc* proc = *channel_map.channel_to_procs[channel].begin();
      XLS_RETURN_IF_ERROR(DeclareChannelInProc(proc, channel));
    } else {
      // Channel communicating between procs. Declare as channel in top proc.
      XLS_RETURN_IF_ERROR(DeclareChannelInProc(top, channel));
    }
  }

  // Add the interface channels for the non-top procs and instantiate the proc.
  for (const std::unique_ptr<Proc>& proc : package->procs()) {
    if (proc.get() == top) {
      continue;
    }
    std::vector<ChannelReference*> instantiation_args;
    for (Channel* channel : channel_map.proc_to_channels[proc.get()]) {
      if (channel_map.IsLoopbackChannel(channel)) {
        // Loopback channels are not part of any proc interface and are declared
        // above.
        continue;
      }
      if (channel_map.directions.at({proc.get(), channel}).size() > 1) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Proc `%s` sends and receives on channel `%s` but "
                            "channel is not a loopback channel",
                            proc->name(), channel->name()));
      }
      Direction direction =
          *channel_map.directions.at({proc.get(), channel}).begin();
      XLS_RETURN_IF_ERROR(AddInterfaceChannel(proc.get(), channel, direction));
      XLS_ASSIGN_OR_RETURN(
          ChannelReference * arg,
          top->GetChannelReference(channel->name(), direction));
      instantiation_args.push_back(arg);
    }
    XLS_RETURN_IF_ERROR(
        top->AddProcInstantiation(absl::StrFormat("%s_inst", proc->name()),
                                  instantiation_args, proc.get())
            .status());
  }

  // Delete old channels.
  std::vector<Channel*> global_channels(package->channels().begin(),
                                        package->channels().end());
  for (Channel* channel : global_channels) {
    XLS_RETURN_IF_ERROR(package->RemoveChannel(channel));
  }
  return absl::OkStatus();
}

}  // namespace xls
