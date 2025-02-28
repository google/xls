// Copyright 2022 The XLS Authors
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

#include "xls/passes/useless_io_removal_pass.h"

#include <memory>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

namespace {

using ChannelToSendMap =
    absl::flat_hash_map<ChannelRef, absl::flat_hash_set<Send*>>;
using ChannelToReceiveMap =
    absl::flat_hash_map<ChannelRef, absl::flat_hash_set<Receive*>>;

struct ChannelMaps {
  ChannelToSendMap to_send;
  ChannelToReceiveMap to_receive;
};

absl::StatusOr<ChannelMaps> ComputeChannelMaps(Package* package) {
  ChannelMaps result;
  for (std::unique_ptr<Proc>& proc : package->procs()) {
    for (Node* node : proc->nodes()) {
      if (node->Is<Send>()) {
        XLS_ASSIGN_OR_RETURN(ChannelRef channel_ref,
                             node->As<Send>()->GetChannelRef());
        result.to_send[channel_ref].insert(node->As<Send>());
      }
      if (node->Is<Receive>()) {
        XLS_ASSIGN_OR_RETURN(ChannelRef channel_ref,
                             node->As<Receive>()->GetChannelRef());
        result.to_receive[channel_ref].insert(node->As<Receive>());
      }
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<bool> UselessIORemovalPass::RunInternal(
    Package* p, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  StatelessQueryEngine query_engine;

  bool changed = false;
  XLS_ASSIGN_OR_RETURN(ChannelMaps channel_maps, ComputeChannelMaps(p));
  // Remove send_if and recv_if with constant false conditions, unless they are
  // the last send/receive on that channel.
  // Replace send_if and recv_if with constant true conditions with unpredicated
  // send and recv.
  for (std::unique_ptr<Proc>& proc : p->procs()) {
    for (Node* node : context.TopoSort(proc.get())) {
      Node* replacement = nullptr;
      if (node->Is<Send>()) {
        Send* send = node->As<Send>();
        if (!send->predicate().has_value()) {
          continue;
        }
        XLS_ASSIGN_OR_RETURN(ChannelRef channel_ref, send->GetChannelRef());
        Node* predicate = send->predicate().value();
        if (query_engine.IsAllZeros(predicate) &&
            channel_maps.to_send.at(channel_ref).size() >= 2) {
          channel_maps.to_send.at(channel_ref).erase(send);
          replacement = send->token();
        } else if (query_engine.IsAllOnes(predicate)) {
          XLS_ASSIGN_OR_RETURN(
              replacement,
              proc->MakeNode<Send>(node->loc(), send->token(), send->data(),
                                   /*predicate=*/std::nullopt,
                                   send->channel_name()));
        }
      } else if (node->Is<Receive>()) {
        Receive* receive = node->As<Receive>();
        if (!receive->predicate().has_value()) {
          continue;
        }
        XLS_ASSIGN_OR_RETURN(ChannelRef channel_ref, receive->GetChannelRef());
        Node* predicate = receive->predicate().value();
        if (query_engine.IsAllZeros(predicate) &&
            channel_maps.to_receive.at(channel_ref).size() >= 2) {
          XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
          channel_maps.to_receive.at(channel_ref).erase(receive);
          XLS_ASSIGN_OR_RETURN(Literal * zero,
                               proc->MakeNode<Literal>(
                                   node->loc(), ZeroOfType(channel->type())));
          XLS_ASSIGN_OR_RETURN(
              replacement,
              proc->MakeNode<Tuple>(
                  node->loc(), std::vector<Node*>{receive->token(), zero}));
        } else if (query_engine.IsAllOnes(predicate)) {
          XLS_ASSIGN_OR_RETURN(replacement, proc->MakeNode<Receive>(
                                                node->loc(), receive->token(),
                                                /*predicate=*/std::nullopt,
                                                receive->channel_name(),
                                                receive->is_blocking()));
        }
      }
      if (replacement != nullptr) {
        XLS_RETURN_IF_ERROR(node->ReplaceUsesWith(replacement));
        XLS_RETURN_IF_ERROR(proc->RemoveNode(node));
        changed = true;
      }
    }
  }
  return changed;
}

REGISTER_OPT_PASS(UselessIORemovalPass);

}  // namespace xls
