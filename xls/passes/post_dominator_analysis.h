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

#ifndef XLS_PASSES_POST_DOMINATOR_ANALYSIS_H_
#define XLS_PASSES_POST_DOMINATOR_ANALYSIS_H_

#include <memory>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/ir/function.h"
#include "xls/ir/node.h"

namespace xls {

// A class for post-dominator analysis of the IR instructions in a function.
class PostDominatorAnalysis {
 public:
  // Performs post-dominator analysis on the function and returns the result.
  static absl::StatusOr<std::unique_ptr<PostDominatorAnalysis>> Run(
      FunctionBase* f);

  // Returns the nodes that post-dominate this node.
  absl::Span<Node* const> GetPostDominatorsOfNode(const Node* node) const {
    return dominated_node_to_post_dominators_ordered_by_id_.at(node);
  }
  // Returns the nodes that are post-dominated by this node.
  absl::Span<Node* const> GetNodesPostDominatedByNode(const Node* node) const {
    return post_dominator_to_dominated_nodes_ordered_by_id_.at(node);
  }
  // Returns true if 'node' is post-dominated by 'post_dominator'.
  bool NodeIsPostDominatedBy(const Node* node,
                             const Node* post_dominator) const {
    return dominated_node_to_post_dominators_.at(node).contains(post_dominator);
  }
  // Returns true if 'node' post_dominates 'post_dominated'.
  bool NodePostDominates(const Node* node, const Node* post_dominated) const {
    return post_dominator_to_dominated_nodes_.at(node).contains(post_dominated);
  }

 private:
  // Maps from a node to all nodes that post-dominate the node.
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>
      dominated_node_to_post_dominators_;
  absl::flat_hash_map<Node*, std::vector<Node*>>
      dominated_node_to_post_dominators_ordered_by_id_;

  // Maps from a node to all nodes that are post-dominated by the node.
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>
      post_dominator_to_dominated_nodes_;
  absl::flat_hash_map<Node*, std::vector<Node*>>
      post_dominator_to_dominated_nodes_ordered_by_id_;
};

}  // namespace xls

#endif  // XLS_PASSES_POST_DOMINATOR_ANALYSIS_H_
