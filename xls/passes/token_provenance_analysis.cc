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

#include "xls/passes/token_provenance_analysis.h"

#include <stdint.h>

#include <algorithm>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/type.h"

namespace xls {

absl::StatusOr<TokenProvenance> TokenProvenanceAnalysis(FunctionBase* f) {
  absl::flat_hash_map<Node*, LeafTypeTree<Node*>> result;

  for (Node* node : TopoSort(f)) {
    if (!TypeHasToken(node->GetType())) {
      continue;
    }
    // Insert the new element in the map and create a reference to it. This
    // avoids the use-after-free footgun of `m[a] = m.at(b)` where `m[a]`
    // invalidates the `m.at(b)` reference.
    auto [it, inserted] =
        result.insert({node, LeafTypeTree<Node*>(node->GetType(), nullptr)});
    LeafTypeTree<Node*>& ltt = it->second;

    if (node->Is<Literal>() || node->Is<Param>() || node->Is<Assert>() ||
        node->Is<Cover>() || node->Is<Trace>() || node->Is<Receive>() ||
        node->Is<Send>() || node->Is<AfterAll>()) {
      for (int64_t i = 0; i < ltt.size(); ++i) {
        if (ltt.leaf_types().at(i)->IsToken()) {
          ltt.elements().at(i) = node;
        }
      }
    } else if (node->op() == Op::kIdentity) {
      ltt = result.at(node->operand(0));
    } else if (node->Is<Tuple>()) {
      Tuple* tuple = node->As<Tuple>();
      std::vector<LeafTypeTree<Node*>> children;
      children.reserve(tuple->operands().size());
      for (Node* child : tuple->operands()) {
        children.push_back(
            result.contains(child)
                ? result.at(child)
                : LeafTypeTree<Node*>(child->GetType(), nullptr));
      }
      ltt = LeafTypeTree<Node*>(tuple->GetType(), children);
    } else if (node->Is<TupleIndex>()) {
      ltt = result.at(node->operand(0))
                .CopySubtree({node->As<TupleIndex>()->index()});
    } else {
      return absl::InternalError(absl::StrFormat(
          "Node type contains token type even though it shouldn't: %s",
          node->ToString()));
    }
  }

  return result;
}

}  // namespace xls
