// Copyright 2026 The XLS Authors
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

#include "xls/passes/name_promotion_pass.h"

#include <deque>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

absl::StatusOr<bool> NamePromotionPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_sort_nodes, context.TopoSort(f));
  bool changed = false;
  // Map from an in-use target name to the first node (in topological order)
  // waiting to reclaim it if its current holder is itself promoted.
  absl::flat_hash_map<std::string, Node*> waiting_for_name;
  std::deque<Node*> worklist;
  for (Node* node : topo_sort_nodes) {
    if (!node->HasAssignedName() || node->Is<Param>() || node->Is<PortNode>()) {
      continue;
    }
    std::optional<std::string> orig_name =
        f->GetOriginalNodeName(node->GetNameView());
    if (!orig_name.has_value()) {
      continue;
    }
    if (f->IsNodeNameAvailable(*orig_name)) {
      worklist.push_back(node);
    } else {
      waiting_for_name.try_emplace(*std::move(orig_name), node);
    }
  }

  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop_front();
    std::string old_name = node->GetName();
    XLS_ASSIGN_OR_RETURN(std::optional<std::string> promoted,
                         f->TryReclaimUnqualifiedNodeName(old_name));
    if (!promoted.has_value()) {
      continue;
    }
    node->SetNameDirectly(*promoted);
    changed = true;
    if (auto it = waiting_for_name.find(old_name);
        it != waiting_for_name.end()) {
      worklist.push_back(it->second);
      waiting_for_name.erase(it);
    }
  }
  return changed;
}

}  // namespace xls
