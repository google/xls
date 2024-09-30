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

#include "xls/passes/dataflow_dominator_analysis.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"

namespace xls {
namespace {

using NodeIndex = int64_t;

// Returns the intersection of the given sorted lists. Lists should be sorted in
// ascending order.
std::vector<NodeIndex> IntersectSortedLists(
    absl::Span<const absl::Span<const NodeIndex>> lists) {
  std::vector<NodeIndex> intersection;
  if (lists.empty()) {
    return intersection;
  }
  std::vector<int64_t> indices(lists.size(), 0);

  // Returns true if any of the indices are at the end of their respective
  // lists.
  auto at_end_of_a_list = [&]() {
    for (int64_t i = 0; i < indices.size(); ++i) {
      if (indices[i] == lists[i].size()) {
        return true;
      }
    }
    return false;
  };
  while (!at_end_of_a_list()) {
    // Find the minimum value among the list elements at their respective
    // indices.
    NodeIndex min_value = lists[0][indices[0]];
    for (int64_t i = 1; i < lists.size(); ++i) {
      NodeIndex value = lists[i][indices[i]];
      min_value = std::min(value, min_value);
    }

    // Advance all list indices which hold the minimum value.
    int64_t match_count = 0;
    for (int64_t i = 0; i < lists.size(); ++i) {
      if (lists[i][indices[i]] == min_value) {
        indices[i]++;
        match_count++;
      }
    }

    // If all lists contained the minimum value then add the value to the
    // intersection.
    if (match_count == lists.size()) {
      intersection.push_back(min_value);
    }
  }
  return intersection;
}

}  // namespace

/* static */ absl::StatusOr<DataflowDominatorAnalysis>
DataflowDominatorAnalysis::Run(FunctionBase* f) {
  DataflowDominatorAnalysis analysis;

  // A topological sort of the function nodes.
  std::vector<Node*> toposort = TopoSort(f);

  // Construct the dominators for each node. Dominators are gathered as a sorted
  // vector containing the node indices (in a toposort) of the dominator nodes;
  // nodes that don't provide variable data correspond to nullopt entries.
  absl::flat_hash_map<Node*, std::optional<std::vector<NodeIndex>>> dominators;
  for (NodeIndex i = 0; i < toposort.size(); ++i) {
    Node* node = toposort[i];
    if (node->OpIn({Op::kReceive, Op::kRegisterRead, Op::kParam, Op::kInputPort,
                    Op::kInstantiationInput})) {
      // These nodes originate (potentially) variable data; they can't be
      // dominated by anything other than themselves, but they do participate in
      // dataflow .
      dominators[node] = std::vector<NodeIndex>({i});
      continue;
    }

    std::vector<absl::Span<const NodeIndex>> operand_dominators;
    for (Node* operand : node->operands()) {
      // Disregard token dependencies, since they can't provide data.
      if (operand->GetType()->IsToken()) {
        continue;
      }
      // Ignore operands that don't provide variable data.
      if (dominators.at(operand).has_value()) {
        operand_dominators.push_back(*dominators.at(operand));
      }
    }
    if (operand_dominators.empty()) {
      // No operands provide variable data, and this node doesn't originate it.
      dominators[node] = std::nullopt;
      continue;
    }
    // The dominators of a node is the intersection of the lists of
    // dominators for its operands plus the node itself.
    dominators[node] = IntersectSortedLists(operand_dominators);
    dominators[node]->push_back(i);
  }

  for (Node* node : f->nodes()) {
    if (!dominators[node].has_value()) {
      analysis.dominated_node_to_dominators_[node];
      analysis.dominator_to_dominated_nodes_[node];
      continue;
    }
    for (NodeIndex dominator_index : *dominators[node]) {
      Node* dominator = toposort.at(dominator_index);
      analysis.dominated_node_to_dominators_[node].insert(dominator);
      analysis.dominator_to_dominated_nodes_[dominator].insert(node);
    }
  }

  // Order nodes.
  auto generate_ordered_by_id_nodes =
      [](const absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>&
             node_to_node_set,
         absl::flat_hash_map<Node*, std::vector<Node*>>* node_to_node_vect) {
        for (auto& [base_node, node_set] : node_to_node_set) {
          std::vector<Node*>& node_vect = (*node_to_node_vect)[base_node];
          node_vect.insert(node_vect.begin(), node_set.begin(), node_set.end());
          SortByNodeId(&node_vect);
        }
      };
  XLS_RET_CHECK(analysis.dominated_node_to_dominators_ordered_by_id_.empty());
  XLS_RET_CHECK(analysis.dominator_to_dominated_nodes_ordered_by_id_.empty());
  generate_ordered_by_id_nodes(
      analysis.dominated_node_to_dominators_,
      &analysis.dominated_node_to_dominators_ordered_by_id_);
  generate_ordered_by_id_nodes(
      analysis.dominator_to_dominated_nodes_,
      &analysis.dominator_to_dominated_nodes_ordered_by_id_);

  return analysis;
}

}  // namespace xls
