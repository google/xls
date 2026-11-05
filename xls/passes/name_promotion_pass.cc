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

#include <optional>
#include <string>
#include <vector>

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
  bool changed_in_iteration = true;
  while (changed_in_iteration) {
    changed_in_iteration = false;
    for (Node* node : topo_sort_nodes) {
      if (!node->HasAssignedName() || node->Is<Param>() ||
          node->Is<PortNode>()) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(
          std::optional<std::string> promoted,
          f->TryReclaimUnqualifiedNodeName(node->GetNameView()));
      if (promoted.has_value()) {
        node->SetNameDirectly(*promoted);
        changed = true;
        changed_in_iteration = true;
      }
    }
  }
  return changed;
}

}  // namespace xls
