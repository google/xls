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
    if (node->Is<Literal>() || node->Is<Param>() || node->Is<Assert>() ||
        node->Is<Cover>() || node->Is<Trace>() || node->Is<Receive>() ||
        node->Is<Send>() || node->Is<AfterAll>()) {
      LeafTypeTree<Node*> ltt(node->GetType(), nullptr);
      for (int64_t i = 0; i < ltt.size(); ++i) {
        if (ltt.leaf_types().at(i)->IsToken()) {
          ltt.elements().at(i) = node;
        }
      }
      result[node] = ltt;
    } else if (node->op() == Op::kIdentity) {
      if (result.contains(node->operand(0))) {
        result[node] = result.at(node->operand(0));
      }
    } else if (node->Is<Tuple>()) {
      Tuple* tuple = node->As<Tuple>();
      std::vector<LeafTypeTree<Node*>> children;
      children.reserve(tuple->operands().size());
      if (!std::any_of(tuple->operands().begin(), tuple->operands().end(),
                       [&](Node* c) -> bool { return result.contains(c); })) {
        continue;
      }
      for (Node* child : tuple->operands()) {
        children.push_back(
            result.contains(child)
                ? result.at(child)
                : LeafTypeTree<Node*>(child->GetType(), nullptr));
      }
      result[tuple] = LeafTypeTree<Node*>(tuple->GetType(), children);
    } else if (node->Is<TupleIndex>()) {
      TupleIndex* tuple_index = node->As<TupleIndex>();
      if (result.contains(tuple_index->operand(0))) {
        result[tuple_index] = result.at(tuple_index->operand(0))
                                  .CopySubtree({tuple_index->index()});
      }
    } else if (TypeHasToken(node->GetType())) {
      return absl::InternalError(absl::StrFormat(
          "Node type contains token type even though it shouldn't: %s",
          node->ToString()));
    }
  }

  return result;
}

}  // namespace xls
