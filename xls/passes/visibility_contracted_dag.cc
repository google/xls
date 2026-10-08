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

#include "xls/passes/visibility_contracted_dag.h"

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/node.h"

namespace xls {

namespace {

// Pushes any `nodes` not present in `visited` onto `worklist`.
// Returns true if at least one node was pushed.
template <typename SequenceT, typename VisitedContainerT>
bool PushUnvisitedToWorklist(const SequenceT& nodes,
                             const VisitedContainerT& visited,
                             std::vector<Node*>& worklist) {
  bool pushed_any = false;
  for (Node* node : nodes) {
    if (!visited.contains(node)) {
      worklist.push_back(node);
      pushed_any = true;
    }
  }
  return pushed_any;
}

// Maps from `start_node` to reachable `users` in `dag_nodes`, stopping at those
// users. Overriding behavior: returns the empty set if a terminal is reachable
// without being stopped by a node in `dag_nodes`.
const absl::flat_hash_set<Node*>& GetReachedDagNodes(
    Node* start_node, const absl::flat_hash_set<Node*>& dag_nodes,
    absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>& node_to_reached) {
  if (auto it = node_to_reached.find(start_node); it != node_to_reached.end()) {
    return it->second;
  }
  std::vector<Node*> worklist = {start_node};
  while (!worklist.empty()) {
    Node* node = worklist.back();
    if (node_to_reached.contains(node)) {
      worklist.pop_back();
      continue;
    }
    if (dag_nodes.contains(node)) {
      node_to_reached[node] = {node};
      worklist.pop_back();
      continue;
    }
    // Ensure users are processed so that `node_to_reached` is populated.
    if (PushUnvisitedToWorklist(node->users(), node_to_reached, worklist)) {
      continue;
    }
    absl::flat_hash_set<Node*> reached;
    for (Node* user : node->users()) {
      const absl::flat_hash_set<Node*>& user_reached = node_to_reached.at(user);
      // Terminal is reachable: override and return the empty set.
      if (user_reached.empty()) {
        reached.clear();
        break;
      }
      reached.insert(user_reached.begin(), user_reached.end());
    }
    node_to_reached[node] = std::move(reached);
    worklist.pop_back();
  }
  return node_to_reached.at(start_node);
}

struct AdjacentAndJoinedUsers {
  std::vector<Node*> adjacent;
  absl::flat_hash_set<Node*> joined;
};

// Partitions contracted DAG nodes reachable from `node` via its users into
// candidate `adjacent` users connected directly by an edge and `joined` users
// reachable via a path not containing any edge in `edge_to_idx`. Returns empty
// sets if `node` can reach a terminal node without passing through `dag_nodes`.
AdjacentAndJoinedUsers CollectAdjacentAndJoinedUsers(
    Node* node, const absl::flat_hash_set<Node*>& dag_nodes,
    const absl::flat_hash_map<OperandNode, int32_t>& edge_to_idx,
    absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>& reached_dag_nodes) {
  AdjacentAndJoinedUsers result;
  for (Node* user : node->users()) {
    // Preemptively add `user` as adjacent, to be filtered out if also joined.
    if (edge_to_idx.contains({node, user})) {
      result.adjacent.push_back(user);
      continue;
    }
    const absl::flat_hash_set<Node*>& user_reached =
        GetReachedDagNodes(user, dag_nodes, reached_dag_nodes);
    // Empty adjacent and joined sets because `node` reaches a terminal.
    if (user_reached.empty()) {
      return {};
    }
    result.joined.insert(user_reached.begin(), user_reached.end());
  }
  return result;
}

// Pushes any unindexed dependent users from `adjacent` (excluding those already
// in `joined`) and `joined` onto `worklist`. Returns true if at least one user
// was pushed.
bool PushUnindexedUsersToWorklist(
    absl::Span<Node* const> adjacent, const absl::flat_hash_set<Node*>& joined,
    const absl::flat_hash_map<Node*, int32_t>& node_to_idx,
    std::vector<Node*>& worklist) {
  bool pushed_any = false;
  for (Node* adj_user : adjacent) {
    if (!joined.contains(adj_user) && !node_to_idx.contains(adj_user)) {
      worklist.push_back(adj_user);
      pushed_any = true;
    }
  }
  pushed_any |= PushUnvisitedToWorklist(joined, node_to_idx, worklist);
  return pushed_any;
}

}  // namespace

std::vector<ContractedDagNode> BuildContractedVisibilityDag(
    Node* one, absl::Span<const OperandNode> edges,
    absl::FunctionRef<BddNodeIndex(Node*, Node*)> edge_vis) {
  // All nodes an edge is incident on, and `one`, belong in the DAG.
  absl::flat_hash_set<Node*> dag_nodes = {one};
  absl::flat_hash_map<OperandNode, int32_t> edge_to_idx;
  edge_to_idx.reserve(edges.size());
  for (int32_t i = 0; i < static_cast<int32_t>(edges.size()); ++i) {
    dag_nodes.insert(edges[i].operand);
    dag_nodes.insert(edges[i].node);
    edge_to_idx.emplace(edges[i], i);
  }

  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> reached_dag_nodes;
  std::vector<ContractedDagNode> contracted_nodes;
  contracted_nodes.reserve(dag_nodes.size());
  absl::flat_hash_map<Node*, int32_t> node_to_idx;
  node_to_idx.reserve(dag_nodes.size());
  std::vector<Node*> worklist = {one};

  while (!worklist.empty()) {
    Node* node = worklist.back();
    if (node_to_idx.contains(node)) {
      worklist.pop_back();
      continue;
    }
    const auto [adjacent, joined] = CollectAdjacentAndJoinedUsers(
        node, dag_nodes, edge_to_idx, reached_dag_nodes);
    if (PushUnindexedUsersToWorklist(adjacent, joined, node_to_idx, worklist)) {
      continue;
    }
    worklist.pop_back();

    ContractedDagNode dag_node;
    for (Node* adj_user : adjacent) {
      // Only nodes not joined, i.e. blocked by `edges`, belong in `adjacent`.
      if (joined.contains(adj_user)) {
        continue;
      }
      dag_node.adjacent.push_back(ContractedDagEdge{
          .user_idx = node_to_idx.at(adj_user),
          .edge_idx = edge_to_idx.at({node, adj_user}),
          .edge_vis = edge_vis(node, adj_user),
      });
    }
    dag_node.joined.reserve(joined.size());
    for (Node* joined_user : joined) {
      dag_node.joined.push_back(node_to_idx.at(joined_user));
    }
    node_to_idx.emplace(node, static_cast<int32_t>(contracted_nodes.size()));
    contracted_nodes.push_back(std::move(dag_node));
  }
  return contracted_nodes;
}

}  // namespace xls
