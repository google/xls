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

  // Mapping from proc->channel_id, where channel_id is a representative value
  // for all the channel_ids in the UnionFind.
  absl::flat_hash_map<Proc*, int64_t> representative_channel_ids;
  representative_channel_ids.reserve(p->procs().size());
  // Channels in the same proc will be union'd.
  UnionFind<int64_t> channel_id_union;
  for (std::unique_ptr<Proc>& proc : p->procs()) {
    std::optional<int64_t> representative_proc_channel_id;
    for (Node* node : proc->nodes()) {
      if (IsChannelNode(node)) {
        int64_t channel_id;
        if (node->Is<Send>()) {
          channel_id = node->As<Send>()->channel_id();
        } else if (node->Is<Receive>()) {
          channel_id = node->As<Receive>()->channel_id();
        } else {
          return absl::NotFoundError(absl::StrFormat(
              "No channel associated with node %s", node->GetName()));
        }
        channel_id_union.Insert(channel_id);
        if (representative_proc_channel_id.has_value()) {
          channel_id_union.Union(representative_proc_channel_id.value(),
                                 channel_id);
        } else {
          representative_proc_channel_id = channel_id;
          representative_channel_ids.insert({proc.get(), channel_id});
        }
      }
    }
  }

  absl::flat_hash_set<FunctionBase*> reached;
  MarkReachedFunctions(top.value(), &reached);
  std::optional<int64_t> top_proc_representative_channel_id;
  if ((*top)->IsProc()) {
    auto itr = representative_channel_ids.find(top.value()->AsProcOrDie());
    if (itr != representative_channel_ids.end()) {
      top_proc_representative_channel_id = channel_id_union.Find(itr->second);
      for (auto [proc, representative_channel_id] :
           representative_channel_ids) {
        if (channel_id_union.Find(representative_channel_id) ==
            *top_proc_representative_channel_id) {
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
  std::vector<int64_t> channel_ids_to_remove;
  channel_ids_to_remove.reserve(p->channels().size());
  for (Channel* channel : p->channels()) {
    int64_t channel_id = channel->id();
    if (!top_proc_representative_channel_id.has_value() ||
        channel_id_union.Find(channel_id) !=
            *top_proc_representative_channel_id) {
      channel_ids_to_remove.push_back(channel_id);
    }
  }
  // Now remove any channels which are only used by now-removed procs.
  for (int64_t channel_id : channel_ids_to_remove) {
    XLS_ASSIGN_OR_RETURN(Channel * channel, p->GetChannel(channel_id));
    XLS_VLOG(2) << "Removing channel: " << channel->name();
    XLS_RETURN_IF_ERROR(p->RemoveChannel(channel));
    changed = true;
  }
  return changed;
}

}  // namespace xls
