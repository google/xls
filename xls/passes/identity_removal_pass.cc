// Copyright 2020 Google LLC
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

#include "xls/passes/identity_removal_pass.h"

#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"

namespace xls {

// Identity Removal performs one forward pass over the TopoSort'ed nodes
// and replaces identities with their respective operands.
xabsl::StatusOr<bool> IdentityRemovalPass::RunOnFunction(
    Function* f, const PassOptions& options, PassResults* results) const {
  bool changed = false;
  absl::flat_hash_map<Node*, Node*> identity_map;
  auto get_src_value = [&](Node* n) {
    return n->op() == Op::kIdentity ? identity_map.at(n) : n;
  };
  for (Node* node : TopoSort(f)) {
    if (node->op() == Op::kIdentity) {
      identity_map[node] = node->operand(0);
    }
  }
  for (Node* node : TopoSort(f)) {
    if (node->op() == Op::kIdentity) {
      identity_map[node] = get_src_value(node->operand(0));
      XLS_ASSIGN_OR_RETURN(bool node_changed,
                           node->ReplaceUsesWith(identity_map.at(node)));
      changed |= node_changed;
    }
  }
  return changed;
}

}  // namespace xls
