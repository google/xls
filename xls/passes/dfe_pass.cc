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

#include "absl/container/btree_set.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/block.h"
#include "xls/ir/node_util.h"
#include "xls/ir/proc.h"

namespace xls {
namespace {

// Data structure indicating the proc which holds the send or receive of each
// channel.
struct ChannelProcMap {
  //  Map indices are channel ids to make indexing easier because Send and
  //  Receive node hold channel IDs not channel pointers.
  absl::flat_hash_map<int64_t, Proc*> receiving_proc;
  absl::flat_hash_map<int64_t, Proc*> sending_proc;

  // Map from proc to the IDs of channels which it uses to communicate. Use
  // btree for stable iteration order.
  absl::flat_hash_map<Proc*, absl::btree_set<int64_t>> proc_channels;
};

ChannelProcMap ComputeChannelProcMap(Package* package) {
  ChannelProcMap channel_map;
  for (std::unique_ptr<Proc>& proc : package->procs()) {
    channel_map.proc_channels[proc.get()] = {};
    for (Node* node : proc->nodes()) {
      if (node->Is<Receive>()) {
        int64_t channel_id = node->As<Receive>()->channel_id();
        channel_map.receiving_proc[channel_id] = proc.get();
        channel_map.proc_channels[proc.get()].insert(channel_id);
      } else if (node->Is<Send>()) {
        int64_t channel_id = node->As<Send>()->channel_id();
        channel_map.sending_proc[channel_id] = proc.get();
        channel_map.proc_channels[proc.get()].insert(channel_id);
      }
    }
  }
  return channel_map;
}

void MarkReachedFunctions(FunctionBase* func, const ChannelProcMap& channel_map,
                          absl::flat_hash_set<FunctionBase*>* reached) {
  if (reached->contains(func)) {
    return;
  }
  reached->insert(func);
  // Iterate over statements and find invocations or references.
  for (Node* node : func->nodes()) {
    switch (node->op()) {
      case Op::kCountedFor:
        MarkReachedFunctions(node->As<CountedFor>()->body(), channel_map,
                             reached);
        break;
      case Op::kDynamicCountedFor:
        MarkReachedFunctions(node->As<DynamicCountedFor>()->body(), channel_map,
                             reached);
        break;
      case Op::kInvoke:
        MarkReachedFunctions(node->As<Invoke>()->to_apply(), channel_map,
                             reached);
        break;
      case Op::kMap:
        MarkReachedFunctions(node->As<Map>()->to_apply(), channel_map, reached);
        break;
      case Op::kReceive: {
        int64_t channel_id = node->As<Receive>()->channel_id();
        if (channel_map.sending_proc.contains(channel_id)) {
          MarkReachedFunctions(channel_map.sending_proc.at(channel_id),
                               channel_map, reached);
        }
        break;
      }
      case Op::kSend: {
        int64_t channel_id = node->As<Send>()->channel_id();
        if (channel_map.receiving_proc.contains(channel_id)) {
          MarkReachedFunctions(channel_map.receiving_proc.at(channel_id),
                               channel_map, reached);
        }
        break;
      }
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
                             channel_map, reached);
      }
    }
  }
}

}  // namespace

// Starting from the return_value(s), DFS over all nodes. Unvisited
// nodes, or parameters, are dead.
absl::StatusOr<bool> DeadFunctionEliminationPass::RunInternal(
    Package* p, const PassOptions& options, PassResults* results) const {
  absl::flat_hash_set<FunctionBase*> reached;
  std::optional<FunctionBase*> top = p->GetTop();
  if (!top.has_value()) {
    return false;
  }
  ChannelProcMap channel_map = ComputeChannelProcMap(p);
  MarkReachedFunctions(top.value(), channel_map, &reached);

  // Accumulate a list of nodes to unlink.
  std::vector<FunctionBase*> to_unlink;
  for (FunctionBase* f : p->GetFunctionBases()) {
    if (!reached.contains(f)) {
      to_unlink.push_back(f);
    }
  }
  std::vector<Proc*> removed_procs;
  for (FunctionBase* f : to_unlink) {
    XLS_VLOG(2) << "Removing: " << f->name();
    if (f->IsProc()) {
      removed_procs.push_back(f->AsProcOrDie());
    }
    XLS_RETURN_IF_ERROR(p->RemoveFunctionBase(f));
  }

  // Now remove any channels which were used for communicating between deleted
  // procs.
  for (Proc* proc : removed_procs) {
    // The pointer `proc` is now deallocated but ok to use as a map key.
    for (int64_t channel_id : channel_map.proc_channels.at(proc)) {
      // The channel could have been deleted in an earlier iteration of this
      // loop.
      if (p->HasChannelWithId(channel_id)) {
        XLS_ASSIGN_OR_RETURN(Channel * channel, p->GetChannel(channel_id));
        XLS_VLOG(2) << "Removing channel: " << channel->name();
        XLS_RETURN_IF_ERROR(p->RemoveChannel(channel));
      }
    }
  }
  return !to_unlink.empty();
}

}  // namespace xls
