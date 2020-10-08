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

#include "xls/passes/cse_pass.h"

#include "absl/hash/hash.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"

namespace xls {

absl::StatusOr<bool> CsePass::RunOnFunction(Function* f,
                                            const PassOptions& options,
                                            PassResults* results) const {
  // To improve efficiency, bucket potentially common nodes together. The
  // bucketing is done via an int64 hash value which is constructed from the
  // op() of the node and the uid's of the node's operands.
  auto hasher = absl::Hash<std::vector<int64>>();
  auto node_hash = [&](Node* n) {
    std::vector<int64> values_to_hash = {static_cast<int64>(n->op())};
    for (Node* operand : n->operands()) {
      values_to_hash.push_back(operand->id());
    }
    // If this is slow because of many literals, the Literal values could be
    // combined into the hash. As is, all literals get the same hash value.
    return hasher(values_to_hash);
  };
  bool changed = false;
  absl::flat_hash_map<int64, std::vector<Node*>> node_buckets;
  node_buckets.reserve(f->node_count());
  for (Node* node : TopoSort(f)) {
    int64 hash = node_hash(node);
    if (!node_buckets.contains(hash)) {
      node_buckets[hash].push_back(node);
      continue;
    }
    bool replaced = false;
    for (Node* candidate : node_buckets.at(hash)) {
      if (node->operands() == candidate->operands() &&
          node->IsDefinitelyEqualTo(candidate)) {
        XLS_ASSIGN_OR_RETURN(bool node_changed,
                             node->ReplaceUsesWith(candidate));
        changed |= node_changed;
        replaced = true;
        break;
      }
    }
    if (!replaced) {
      node_buckets[hash].push_back(node);
    }
  }

  return changed;
}

}  // namespace xls
