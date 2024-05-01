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

#include "xls/passes/post_dominator_analysis.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
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
      if (value < min_value) {
        min_value = value;
      }
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

/* static */ absl::StatusOr<std::unique_ptr<PostDominatorAnalysis>>
PostDominatorAnalysis::Run(FunctionBase* f) {
  auto analysis = std::make_unique<PostDominatorAnalysis>();

  // A reverse topological sort of the function nodes.
  std::vector<Node*> reverse_toposort = ReverseTopoSort(f);

  // Construct the postdominators for each node. Postdominators are gathered as
  // a sorted vector containing the node indices (in a reverse toposort) of the
  // post dominator nodes.
  absl::flat_hash_map<Node*, std::vector<NodeIndex>> postdominators;
  for (NodeIndex i = 0; i < reverse_toposort.size(); ++i) {
    Node* node = reverse_toposort[i];
    std::vector<absl::Span<const NodeIndex>> user_postdominators;
    for (Node* user : node->users()) {
      user_postdominators.push_back(postdominators.at(user));
    }
    // The postdominators of a node is the intersection of the lists of
    // postdominators for its users plus the node itself.
    postdominators[node] = IntersectSortedLists(user_postdominators);
    // If a node has an implicit use, then there exists an alternate path to a
    // root node other than its users, so it can't be dominated by anything
    // other than itself.
    if (f->HasImplicitUse(node)) {
      postdominators[node] = std::vector<NodeIndex>();
    }
    postdominators[node].push_back(i);
  }

  for (Node* node : f->nodes()) {
    for (NodeIndex postdominator_index : postdominators[node]) {
      Node* postdominator = reverse_toposort.at(postdominator_index);
      analysis->dominated_node_to_post_dominators_[node].insert(postdominator);
      analysis->post_dominator_to_dominated_nodes_[postdominator].insert(node);
    }
  }

  // Order nodes.
  auto generate_ordered_by_id_nodes =
      [](const absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>&
             node_to_node_set,
         absl::flat_hash_map<Node*, std::vector<Node*>>* node_to_node_vect) {
        for (auto& [base_node, node_set] : node_to_node_set) {
          auto& node_vect = (*node_to_node_vect)[base_node];
          node_vect.insert(node_vect.begin(), node_set.begin(), node_set.end());
          SortByNodeId(&node_vect);
        }
      };
  XLS_RET_CHECK(
      analysis->dominated_node_to_post_dominators_ordered_by_id_.empty());
  XLS_RET_CHECK(
      analysis->post_dominator_to_dominated_nodes_ordered_by_id_.empty());
  generate_ordered_by_id_nodes(
      analysis->dominated_node_to_post_dominators_,
      &analysis->dominated_node_to_post_dominators_ordered_by_id_);
  generate_ordered_by_id_nodes(
      analysis->post_dominator_to_dominated_nodes_,
      &analysis->post_dominator_to_dominated_nodes_ordered_by_id_);

  return std::move(analysis);
}

}  // namespace xls
