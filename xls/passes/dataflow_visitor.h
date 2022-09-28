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

#ifndef XLS_PASSES_DATAFLOW_VISITOR_H_
#define XLS_PASSES_DATAFLOW_VISITOR_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/node.h"

namespace xls {

// Abstract base class which performs data flow analysis to track value elements
// (e.g., tuple elements) through tuple, tuple-index, and other operations. The
// data structure stores a LeafTypeTree<T> for each node. The visitor includes
// handlers for tuple, tuple-index, and identity operations. These handlers set
// the LeafTypeTree data elements of a node to the respective data element of
// the node's operand. For example, given:
//
//   x = tuple(a, b)
//   y = tuple_index(x, 0)
//   z = tuple_index(x, 1)
//   x_z = tuple(x, z)
//
// Where the associated value for `x` (as, say, a LeafTypeTree<int64_t> in
// DataFlowVisitor) is `(42, 123)`, then DataFlowVisitor::GetValue will return
// the following for the other nodes:
//
//   y : 42
//   z : 123
//   x_z : ((42, 123), 123)
//
// Users should define default handler and (optionally) other handlers.
template <typename T>
class DataFlowVisitor : public DfsVisitorWithDefault {
 public:
  absl::Status HandleTuple(Tuple* tuple) override {
    // Use InlinedVector to avoid std::vector<bool> abomination.
    absl::InlinedVector<T, 1> elements;
    for (Node* operand : tuple->operands()) {
      const LeafTypeTree<T>& operand_tree = map_.at(operand);
      elements.insert(elements.end(), operand_tree.elements().begin(),
                      operand_tree.elements().end());
    }
    return SetValue(
        tuple, LeafTypeTree<T>(tuple->GetType(), absl::MakeSpan(elements)));
  }

  absl::Status HandleTupleIndex(TupleIndex* tuple_index) override {
    return SetValue(tuple_index,
                    map_.at(tuple_index->operand(0))
                        .CopySubtree(/*index=*/{tuple_index->index()}));
  }

  absl::Status HandleIdentity(UnOp* identity) override {
    return SetValue(identity, map_.at(identity->operand(0)));
  }

  // Sets the leaf type tree value associated with `node`.
  absl::Status SetValue(Node* node, LeafTypeTree<T> value) {
    XLS_RET_CHECK_EQ(node->GetType(), value.type());
    map_[node] = std::move(value);
    return absl::OkStatus();
  }

  // Returns the leaf type tree value associated with `node`.
  const LeafTypeTree<T>& GetValue(Node* node) const { return map_.at(node); }

  // Returns the moved leaf type tree value associated with `node`.
  LeafTypeTree<T> ConsumeValue(Node* node) {
    LeafTypeTree<T> ltt = std::move(map_.at(node));
    // Erase the moved element from the map to avoid later access.
    map_.erase(node);
    return ltt;
  }

 private:
  absl::flat_hash_map<Node*, LeafTypeTree<T>> map_;
};

}  // namespace xls

#endif  // XLS_PASSES_DATAFLOW_VISITOR_H_
