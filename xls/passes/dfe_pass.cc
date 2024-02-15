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

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/union_find.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/elaboration.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
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

absl::StatusOr<FunctionBaseLiveness> LivenessFromTopProc(Proc* top) {
  if (top->is_new_style_proc()) {
    XLS_ASSIGN_OR_RETURN(Elaboration elab, Elaboration::Elaborate(top));
    return FunctionBaseLiveness{.live_roots = std::vector<FunctionBase*>(
                                    elab.procs().begin(), elab.procs().end()),
                                .live_global_channels = {}};
  }

  Package* p = top->package();

  // Mapping from proc to channel, where channel is a representative value for
  // all the channel names in the UnionFind. If the proc uses no channels then
  // the value will be nullopt.
  absl::flat_hash_map<Proc*, std::optional<std::string_view>>
      representative_channels;
  representative_channels.reserve(p->procs().size());
  // Channels in the same proc will be union'd.
  UnionFind<std::string_view> channel_union;
  for (const std::unique_ptr<Proc>& proc : p->procs()) {
    std::optional<std::string_view> representative_proc_channel;
    for (Node* node : proc->nodes()) {
      if (IsChannelNode(node)) {
        XLS_ASSIGN_OR_RETURN(Channel * channel, GetChannelUsedByNode(node));
        channel_union.Insert(channel->name());
        if (representative_proc_channel.has_value()) {
          channel_union.Union(representative_proc_channel.value(),
                              channel->name());
        } else {
          representative_proc_channel = channel->name();
        }
      }
    }
    representative_channels[proc.get()] = representative_proc_channel;
  }

  FunctionBaseLiveness liveness;

  // Add procs to the live set if they are connnected to `top` via channels.
  for (const std::unique_ptr<Proc>& proc : p->procs()) {
    if (proc.get() == top) {
      liveness.live_roots.push_back(proc.get());
      continue;
    }
    if (representative_channels.at(top).has_value() &&
        representative_channels.at(proc.get()) &&
        channel_union.Find(representative_channels.at(top).value()) ==
            channel_union.Find(
                representative_channels.at(proc.get()).value())) {
      liveness.live_roots.push_back(proc.get());
    }
  }

  // Add channels to the live set if they are connnected to `top`.
  if (representative_channels.at(top).has_value()) {
    for (Channel* channel : p->channels()) {
      if (channel_union.Find(channel->name()) ==
          channel_union.Find(representative_channels.at(top).value())) {
        liveness.live_global_channels.insert(channel);
      }
    }
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
    XLS_ASSIGN_OR_RETURN(liveness, LivenessFromTopProc((*top)->AsProcOrDie()));
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
      XLS_VLOG(2) << "Removing: " << f->name();
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
    XLS_VLOG(2) << "Removing channel: " << channel->name();
    XLS_RETURN_IF_ERROR(p->RemoveChannel(channel));
    changed = true;
  }
  return changed;
}

REGISTER_OPT_PASS(DeadFunctionEliminationPass);

}  // namespace xls
