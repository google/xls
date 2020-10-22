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

#include <algorithm>
#include <cstdio>
#include <queue>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/memory/memory.h"
#include "absl/meta/type_traits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/ir/function.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/proc.h"

namespace xls {
namespace {

// Returns the exit nodes of the function/proc. These are the nodes which have
// implicit uses (are live out of the graph).
std::vector<Node*> GetExitNodes(FunctionBase* f) {
  return f->IsFunction()
             ? std::vector<Node*>({f->AsFunctionOrDie()->return_value()})
             : std::vector<Node*>({f->AsProcOrDie()->NextToken(),
                                   f->AsProcOrDie()->NextState()});
}

// Returns whether n is an exit node as defined above.
bool IsExitNode(Node* n) { return n->function_base()->HasImplicitUse(n); }

}  // namespace

/* static */ absl::StatusOr<std::unique_ptr<PostDominatorAnalysis>>
PostDominatorAnalysis::Run(FunctionBase* f) {
  auto analysis = absl::WrapUnique(new PostDominatorAnalysis(f));

  // Intialize data structs.
  analysis->PopulateReturnReachingNodes();
  for (Node* current_node : f->nodes()) {
    analysis->post_dominator_to_dominated_nodes_[current_node] = {};
    analysis->dominated_node_to_post_dominators_[current_node] = {};
  }
  // Find post-dominators of all nodes.
  // Note: We assume there are no backwards edges in the graph,
  // making it safe to just iterate over the nodes once in reverse
  // topological order.
  for (Node* dominated_node : ReverseTopoSort(f)) {
    absl::flat_hash_set<Node*>& node_post_dominators =
        analysis->dominated_node_to_post_dominators_.at(dominated_node);
    if (IsExitNode(dominated_node)) {
      analysis->dominated_node_to_post_dominators_[dominated_node] = {
          dominated_node};
    } else if (!dominated_node->users().empty()) {
      // Calculate post-dominators of dominated_node.
      // new_post_dominators = union(dominated_node,
      // intersection(post-dominators of dominated_node users / consumers))
      const absl::flat_hash_set<Node*>& user_post_dominators =
          analysis->dominated_node_to_post_dominators_.at(
              dominated_node->users().front());
      node_post_dominators.insert(user_post_dominators.begin(),
                                  user_post_dominators.end());

      // Note: No native intersection operation for absl::flat_hash_set
      for (const auto* user_node_itr =
               std::next(dominated_node->users().begin());
           user_node_itr != dominated_node->users().end(); ++user_node_itr) {
        absl::erase_if(node_post_dominators, [&](const Node* node) {
          return !analysis->NodeIsPostDominatedBy(*user_node_itr, node);
        });
      }
      node_post_dominators.insert(dominated_node);
    } else {
      // For userless nodes, we temporarily add all nodes as postdominaters.
      // This way, nodes which feed both the userless node and a non-userless
      // node only have their post-dominators set by the non-userless node.
      node_post_dominators.insert(f->nodes().begin(), f->nodes().end());
    }
  }

  // Handle non-return-reaching nodes.
  for (auto* node : f->nodes()) {
    if (!analysis->return_reaching_nodes_.contains(node)) {
      analysis->dominated_node_to_post_dominators_[node] = {node};
    }
  }

  // Enable look-up in both directions (dominated <-> dominator).
  for (auto& [dominated_node, all_post_dominators] :
       analysis->dominated_node_to_post_dominators_) {
    for (const Node* post_dominator : all_post_dominators) {
      analysis->post_dominator_to_dominated_nodes_.at(post_dominator)
          .insert(dominated_node);
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
          std::sort(node_vect.begin(), node_vect.end(),
                    [](Node* a, Node* b) { return a->id() < b->id(); });
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

void PostDominatorAnalysis::PopulateReturnReachingNodes() {
  for (Node* node : GetExitNodes(func_)) {
    return_reaching_nodes_.insert(node);
  }
  for (const Node* node : ReverseTopoSort(func_)) {
    if (std::any_of(node->users().begin(), node->users().end(),
                    [&](Node* user) {
                      return return_reaching_nodes_.contains(user);
                    })) {
      return_reaching_nodes_.insert(node);
    }
  }
}

}  // namespace xls
