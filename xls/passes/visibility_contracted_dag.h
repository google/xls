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

#ifndef XLS_PASSES_VISIBILITY_CONTRACTED_DAG_H_
#define XLS_PASSES_VISIBILITY_CONTRACTED_DAG_H_

#include <compare>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/node.h"

namespace xls {

// Represents a directed def-use edge `(operand, node)` in the IR.
struct OperandNode {
  Node* operand;
  Node* node;

  OperandNode(Node* operand, Node* node) : operand(operand), node(node) {}

  template <typename H>
  friend H AbslHashValue(H h, const OperandNode& op_node) {
    return H::combine(std::move(h), op_node.operand, op_node.node);
  }

  std::strong_ordering operator<=>(const OperandNode& other) const {
    if (auto cmp = operand->id() <=> other.operand->id(); cmp != 0) {
      return cmp;
    }
    return node->id() <=> other.node->id();
  }
  bool operator==(const OperandNode& other) const {
    return operand == other.operand && node == other.node;
  }

  std::string ToString() const {
    return absl::StrCat(operand->GetName(), "->", node->GetName());
  }
};

// Represents a directed edge to an adjacent user in a contracted DAG.
struct ContractedDagEdge {
  int32_t user_idx;       // Index of the user in the `ContractedDagNode` list.
  int32_t edge_idx;       // Index of the `(node, user)` edge in `edges`.
  BddNodeIndex edge_vis;  // Visibility of the `(node, user)` edge.

  bool operator==(const ContractedDagEdge& other) const {
    return user_idx == other.user_idx && edge_idx == other.edge_idx &&
           edge_vis == other.edge_vis;
  }
};

// A node in a contracted DAG. Nodes in `adjacent` are separated from this node
// by an edge in `edges`, with no path bypassing `edges`. Nodes in `joined` are
// indices of nodes reachable from this node via at least one path in the
// original DAG without passing through any of `edges`.
struct ContractedDagNode {
  std::vector<ContractedDagEdge> adjacent;  // Users separated by an edge.
  std::vector<int32_t> joined;              // Indices of unseparated users.

  bool operator==(const ContractedDagNode& other) const {
    return adjacent == other.adjacent && joined == other.joined;
  }
};

// Builds a reverse topologically sorted DAG over `one` and endpoints of `edges`
// reachable from `one`, with `one` as the last element.
std::vector<ContractedDagNode> BuildContractedVisibilityDag(
    Node* one, absl::Span<const OperandNode> edges,
    absl::FunctionRef<BddNodeIndex(Node*, Node*)> edge_vis);

}  // namespace xls

#endif  // XLS_PASSES_VISIBILITY_CONTRACTED_DAG_H_
