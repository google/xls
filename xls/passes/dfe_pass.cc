// Copyright 2020 The XLS Authors
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

#include "xls/passes/dfe_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/union_find.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

void MarkReachedFunctions(FunctionBase* func,
                          absl::flat_hash_set<FunctionBase*>* reached) {
  if (reached->contains(func)) {
    return;
  }
  reached->insert(func);
  // Iterate over statements and find invocations or references.
  for (Node* node : func->nodes()) {
    switch (node->op()) {
      case Op::kCountedFor:
        MarkReachedFunctions(node->As<CountedFor>()->body(), reached);
        break;
      case Op::kDynamicCountedFor:
        MarkReachedFunctions(node->As<DynamicCountedFor>()->body(), reached);
        break;
      case Op::kInvoke:
        MarkReachedFunctions(node->As<Invoke>()->to_apply(), reached);
        break;
      case Op::kMap:
        MarkReachedFunctions(node->As<Map>()->to_apply(), reached);
        break;
      default:
        break;
    }
  }

  // If the FunctionBase is a block, add all instantiated blocks.
  if (func->IsBlock()) {
    for (Instantiation* instantiation :
         func->AsBlockOrDie()->GetInstantiations()) {
      if (auto block_instantiation =
              dynamic_cast<BlockInstantiation*>(instantiation)) {
        MarkReachedFunctions(block_instantiation->instantiated_block(),
                             reached);
      }
    }
  }
}
}  // namespace

// Starting from the return_value(s), DFS over all nodes. Unvisited
// nodes, or parameters, are dead.
absl::StatusOr<bool> DeadFunctionEliminationPass::RunInternal(
    Package* p, const OptimizationPassOptions& options,
    PassResults* results) const {
  std::optional<FunctionBase*> top = p->GetTop();
  if (!top.has_value()) {
    return false;
  }

  // Mapping from proc->channel, where channel is a representative value
  // for all the channel names in the UnionFind.
  absl::flat_hash_map<Proc*, std::string> representative_channels;
  representative_channels.reserve(p->procs().size());
  // Channels in the same proc will be union'd.
  UnionFind<std::string> channel_union;
  for (std::unique_ptr<Proc>& proc : p->procs()) {
    std::optional<std::string> representative_proc_channel;
    for (Node* node : proc->nodes()) {
      if (IsChannelNode(node)) {
        std::string channel;
        if (node->Is<Send>()) {
          channel = node->As<Send>()->channel_name();
        } else if (node->Is<Receive>()) {
          channel = node->As<Receive>()->channel_name();
        } else {
          return absl::NotFoundError(absl::StrFormat(
              "No channel associated with node %s", node->GetName()));
        }
        channel_union.Insert(channel);
        if (representative_proc_channel.has_value()) {
          channel_union.Union(representative_proc_channel.value(), channel);
        } else {
          representative_proc_channel = channel;
          representative_channels.insert({proc.get(), channel});
        }
      }
    }
  }

  absl::flat_hash_set<FunctionBase*> reached;
  MarkReachedFunctions(top.value(), &reached);
  std::optional<std::string> top_proc_representative_channel;
  if ((*top)->IsProc()) {
    auto itr = representative_channels.find(top.value()->AsProcOrDie());
    if (itr != representative_channels.end()) {
      top_proc_representative_channel = channel_union.Find(itr->second);
      for (auto [proc, representative_channel] : representative_channels) {
        if (channel_union.Find(representative_channel) ==
            *top_proc_representative_channel) {
          MarkReachedFunctions(proc, &reached);
        }
      }
    }
  }

  // Accumulate a list of FunctionBases to unlink.
  bool changed = false;
  for (FunctionBase* f : p->GetFunctionBases()) {
    if (!reached.contains(f)) {
      XLS_VLOG(2) << "Removing: " << f->name();
      XLS_RETURN_IF_ERROR(p->RemoveFunctionBase(f));
      changed = true;
    }
  }

  // Find any channels which are only used by now-removed procs.
  std::vector<std::string> channels_to_remove;
  channels_to_remove.reserve(p->channels().size());
  for (Channel* channel : p->channels()) {
    if (!top_proc_representative_channel.has_value() ||
        channel_union.Find(channel->name()) !=
            *top_proc_representative_channel) {
      channels_to_remove.push_back(channel->name());
    }
  }
  // Now remove any channels which are only used by now-removed procs.
  for (const std::string& channel_name : channels_to_remove) {
    XLS_ASSIGN_OR_RETURN(Channel * channel, p->GetChannel(channel_name));
    XLS_VLOG(2) << "Removing channel: " << channel->name();
    XLS_RETURN_IF_ERROR(p->RemoveChannel(channel));
    changed = true;
  }
  return changed;
}

}  // namespace xls
