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

#include <deque>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"  // IWYU pragma: keep
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
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

// Data structure describing the liveness of global constructs in a package.
struct FunctionBaseLiveness {
  // The live roots of the package. This does not include FunctionBases which
  // are live because they are called/instantiated from other FunctionBases.
  std::vector<FunctionBase*> live_roots;

  // Set of the live global channels. Only set for old-style procs.
  absl::flat_hash_set<Channel*> live_global_channels;
};

// Analyzes the package to determine which Procs are live. For
// old-style procs, this is determined by looking for procs that use external
// channels (i.e send_only or receive_only) or procs that communicate with other
// live procs over internal channels. For new-style procs, this is determined by
// looking at the procs that are instantiated by the top proc.
//
// The top proc is always considered live.
absl::StatusOr<FunctionBaseLiveness> ProcLiveness(Proc* top) {
  if (top->is_new_style_proc()) {
    XLS_ASSIGN_OR_RETURN(ProcElaboration elab, ProcElaboration::Elaborate(top));
    return FunctionBaseLiveness{.live_roots = std::vector<FunctionBase*>(
                                    elab.procs().begin(), elab.procs().end()),
                                .live_global_channels = {}};
  }

  Package* p = top->package();

  std::deque<Proc*> worklist;
  absl::flat_hash_map<Channel*, std::vector<Proc*>> channel_to_proc;
  absl::flat_hash_map<Proc*, std::vector<Channel*>> proc_to_channel;

  worklist.push_back(top);
  for (std::unique_ptr<Proc>& proc : p->procs()) {
    auto [proc_to_channel_iter, inserted] =
        proc_to_channel.insert({proc.get(), {}});
    bool saw_channel = false;
    for (Node* node : proc->nodes()) {
      if (!node->Is<ChannelNode>()) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
      channel_to_proc[channel].push_back(proc.get());
      proc_to_channel_iter->second.push_back(channel);

      if (channel->supported_ops() == ChannelOps::kSendReceive) {
        continue;
      }
      if (!saw_channel) {
        worklist.push_back(proc.get());
        saw_channel = true;
      }
    }
  }

  FunctionBaseLiveness liveness;
  absl::flat_hash_set<Proc*> seen;
  while (!worklist.empty()) {
    Proc* proc = worklist.front();
    liveness.live_roots.push_back(proc);
    seen.insert(proc);

    for (Channel* channel : proc_to_channel.at(proc)) {
      liveness.live_global_channels.insert(channel);
      for (Proc* proc_for_channel : channel_to_proc.at(channel)) {
        if (!seen.contains(proc_for_channel)) {
          worklist.push_back(proc_for_channel);
        }
      }
    }
    worklist.pop_front();
  }

  return liveness;
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

  FunctionBaseLiveness liveness;
  if ((*top)->IsProc()) {
    XLS_ASSIGN_OR_RETURN(liveness, ProcLiveness((*top)->AsProcOrDie()));
  } else {
    liveness.live_roots = {*top};
  }

  absl::flat_hash_set<FunctionBase*> reached;
  for (FunctionBase* fb : liveness.live_roots) {
    MarkReachedFunctions(fb, &reached);
  }

  // Accumulate a list of FunctionBases to unlink.
  bool changed = false;
  for (FunctionBase* f : p->GetFunctionBases()) {
    if (!reached.contains(f)) {
      VLOG(2) << "Removing: " << f->name();
      XLS_RETURN_IF_ERROR(p->RemoveFunctionBase(f));
      changed = true;
    }
  }

  // Remove dead channels.
  std::vector<Channel*> channels_to_remove;
  channels_to_remove.reserve(p->channels().size());
  for (Channel* channel : p->channels()) {
    if (!liveness.live_global_channels.contains(channel)) {
      channels_to_remove.push_back(channel);
    }
  }
  for (Channel* channel : channels_to_remove) {
    VLOG(2) << "Removing channel: " << channel->name();
    XLS_RETURN_IF_ERROR(p->RemoveChannel(channel));
    changed = true;
  }
  return changed;
}

REGISTER_OPT_PASS(DeadFunctionEliminationPass);

}  // namespace xls
