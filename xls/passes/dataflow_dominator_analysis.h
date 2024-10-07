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

#ifndef XLS_PASSES_DATAFLOW_DOMINATOR_ANALYSIS_H_
#define XLS_PASSES_DATAFLOW_DOMINATOR_ANALYSIS_H_

#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/node.h"

namespace xls {

// A class for dataflow dominator analysis of the IR instructions in a function.
//
// This finds all dominators of each node, accounting for potential external
// sources of data and disregarding literals.
class DataflowDominatorAnalysis {
 public:
  // Performs dataflow dominator analysis on the function and returns the
  // result.
  static absl::StatusOr<DataflowDominatorAnalysis> Run(FunctionBase* f);

  // Returns the nodes that dominate this node.
  absl::Span<Node* const> GetDominatorsOfNode(const Node* node) const {
    return dominated_node_to_dominators_ordered_by_id_.at(node);
  }
  // Returns the nodes that are dominated by this node.
  absl::Span<Node* const> GetNodesDominatedByNode(const Node* node) const {
    return dominator_to_dominated_nodes_ordered_by_id_.at(node);
  }
  // Returns true if 'node' is dominated by 'dominator'.
  bool NodeIsDominatedBy(const Node* node, const Node* dominator) const {
    return dominated_node_to_dominators_.at(node).contains(dominator);
  }
  // Returns true if 'node' dominates 'dominated'.
  bool NodeDominates(const Node* node, const Node* dominated) const {
    return dominator_to_dominated_nodes_.at(node).contains(dominated);
  }

 private:
  // Maps from a node to all nodes that dominate the node.
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>
      dominated_node_to_dominators_;
  absl::flat_hash_map<Node*, std::vector<Node*>>
      dominated_node_to_dominators_ordered_by_id_;

  // Maps from a node to all nodes that are dominated by the node.
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>
      dominator_to_dominated_nodes_;
  absl::flat_hash_map<Node*, std::vector<Node*>>
      dominator_to_dominated_nodes_ordered_by_id_;
};

}  // namespace xls

#endif  // XLS_PASSES_DATAFLOW_DOMINATOR_ANALYSIS_H_
