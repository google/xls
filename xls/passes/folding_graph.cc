// Copyright 2025 The XLS Authors
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

#include "xls/passes/folding_graph.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "ortools/graph/cliques.h"

namespace xls {

namespace {

// This function permutes the order of the elements of the array
// @array_to_permute following the permutations listed in @permutation.
//
// This function is similar to util::Permute. There are only two differences
// between this function and util::Permute:
// 1) This function uses std::move rather than relying on the copy constructor.
//    This is important when using smart pointers.
// 2) This function relies on "typeof" to find the type of the elements of the
//    array to permute.
template <class IntVector, class Array>
void Permute(const IntVector& permutation, Array* array_to_permute) {
  if (permutation.empty()) {
    return;
  }
  std::vector<std::remove_reference_t<decltype((*array_to_permute)[0])>> temp(
      permutation.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    temp[i] = std::move((*array_to_permute)[i]);
  }
  for (size_t i = 0; i < permutation.size(); ++i) {
    (*array_to_permute)[static_cast<size_t>(permutation[i])] =
        std::move(temp[i]);
  }
}

}  // namespace

FoldingGraph::FoldingGraph(
    FunctionBase* f,
    std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions)
    : f_{f} {
  // Allocate the graph
  graph_ = std::make_unique<Graph>();

  // Ensure deterministic construction of FoldingGraph
  std::sort(foldable_actions.begin(), foldable_actions.end(),
            [](const std::unique_ptr<BinaryFoldingAction>& a,
               const std::unique_ptr<BinaryFoldingAction>& b) {
              if (a->GetFrom()->id() == b->GetFrom()->id()) {
                return a->GetTo()->id() < b->GetTo()->id();
              }
              return a->GetFrom()->id() < b->GetFrom()->id();
            });

  // Add the nodes
  AddNodes(foldable_actions);

  // Add the edges
  AddEdges(std::move(foldable_actions));

  // Build the graph
  std::vector<EdgeIndex> edge_permutations;
  graph_->Build(&edge_permutations);
  Permute(edge_permutations, &edges_);

  // Print the folding graph
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "Folding graph";

    // Print the folding graph following outgoing edges
    VLOG(2) << "  Following outgoing edges:";
    for (NodeIndex node_index : graph_->AllNodes()) {
      if (graph_->OutDegree(node_index) == 0) {
        continue;
      }
      Node* from_node = nodes_[node_index];
      VLOG(2) << "    [" << node_index << "] " << from_node->ToString();
      for (EdgeIndex edge_index : graph_->OutgoingArcs(node_index)) {
        NodeIndex to_node_index = graph_->Head(edge_index);
        Node* to_node = nodes_[to_node_index];
        CHECK_EQ(edges_[edge_index]->GetFrom(), from_node);
        CHECK_EQ(edges_[edge_index]->GetTo(), to_node);
        VLOG(2) << "      -> [" << to_node_index << "] " << to_node->ToString();
      }
    }

    // Print the folding graph following incoming edges
    VLOG(2) << "  Following incoming edges:";
    for (NodeIndex node_index : graph_->AllNodes()) {
      if (graph_->InDegree(node_index) == 0) {
        continue;
      }
      Node* to_node = nodes_[node_index];
      VLOG(2) << "    [" << node_index << "] " << to_node->ToString();
      for (EdgeIndex edge_index : graph_->IncomingArcs(node_index)) {
        CHECK_EQ(graph_->Head(edge_index), node_index);
        NodeIndex from_node_index = graph_->Tail(edge_index);
        Node* from_node = nodes_[from_node_index];
        CHECK_EQ(edges_[edge_index]->GetFrom(), from_node);
        CHECK_EQ(edges_[edge_index]->GetTo(), to_node);
        VLOG(2) << "      <- [" << from_node_index << "] "
                << from_node->ToString();
      }
    }
  }
}

FunctionBase* FoldingGraph::function() const { return f_; }

void FoldingGraph::AddNodes(
    absl::Span<const std::unique_ptr<BinaryFoldingAction>> foldable_actions) {
  // Add all nodes involved in folding actions into the internal
  // representation.
  absl::flat_hash_set<Node*> already_added;
  for (const std::unique_ptr<BinaryFoldingAction>& f : foldable_actions) {
    // Add the nodes to our internal representation if they were not added
    // already
    Node* from_node = f->GetFrom();
    Node* to_node = f->GetTo();
    if (!already_added.contains(from_node)) {
      already_added.insert(from_node);
      nodes_.push_back(from_node);
    }
    if (!already_added.contains(to_node)) {
      already_added.insert(to_node);
      nodes_.push_back(to_node);
    }
  }

  // Add the mapping from Node to its index
  node_to_index_.reserve(nodes_.size());
  for (size_t i = 0; i < nodes_.size(); ++i) {
    node_to_index_[nodes_[i]] = i;
  }

  // Add the nodes to the graph
  for (size_t i = 0; i < nodes_.size(); ++i) {
    graph_->AddNode(i);
  }
}

void FoldingGraph::AddEdges(
    std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions) {
  // Add all edges to the graph
  for (std::unique_ptr<BinaryFoldingAction>& f : foldable_actions) {
    // Add a new edge into the graph to represent the current folding action
    NodeIndex from_index = node_to_index_.at(f->GetFrom());
    NodeIndex to_index = node_to_index_.at(f->GetTo());
    CHECK(graph_->IsNodeValid(from_index));
    CHECK(graph_->IsNodeValid(to_index));
    graph_->AddArc(from_index, to_index);

    // Add the current folding action to our internal representation
    edges_.push_back(std::move(f));
  }
}

void FoldingGraph::IdentifyCliques() {
  if (!this->cliques_.empty()) {
    return;
  }
  auto graph_descriptor = [this](int from_node, int to_node) -> bool {
    for (EdgeIndex outgoing_edge : this->graph_->OutgoingArcs(from_node)) {
      if (this->graph_->Head(outgoing_edge) == to_node) {
        return true;
      }
    }
    return false;
  };
  auto found_clique = [this](const std::vector<int>& clique) -> bool {
    absl::flat_hash_set<NodeIndex> clique_to_add;
    VLOG(3) << "New clique:";
    for (NodeIndex node : clique) {
      VLOG(3) << "  " << node;
      clique_to_add.insert(node);
    }
    this->cliques_.insert(clique_to_add);

    return false;
  };
  ::operations_research::FindCliques(graph_descriptor, nodes_.size(),
                                     found_clique);
}

absl::flat_hash_set<absl::flat_hash_set<BinaryFoldingAction*>>
FoldingGraph::GetEdgeCliques() {
  absl::flat_hash_set<absl::flat_hash_set<BinaryFoldingAction*>> cliques;

  // Identify the cliques within the graph
  IdentifyCliques();

  // Find all the cliques of edges
  for (const absl::flat_hash_set<NodeIndex>& node_clique : cliques_) {
    // Find all the edges within the clique
    absl::flat_hash_set<BinaryFoldingAction*> edge_clique;
    for (NodeIndex from_node_index : node_clique) {
      for (EdgeIndex outgoing_edge : graph_->OutgoingArcs(from_node_index)) {
        CHECK_LT(outgoing_edge, edges_.size());

        // Fetch the destination of the current outgoing edge
        NodeIndex to_node_index = graph_->Head(outgoing_edge);
        CHECK_NE(from_node_index, to_node_index);

        // Check if the current edge belongs to the clique
        if (!node_clique.contains(to_node_index)) {
          continue;
        }

        // We found a new edge that belongs to the clique
        BinaryFoldingAction* new_folding_action_within_clique =
            edges_[outgoing_edge].get();
        CHECK_NE(new_folding_action_within_clique, nullptr);
        edge_clique.insert(new_folding_action_within_clique);
      }
    }

    // Add the new edge clique
    cliques.insert(edge_clique);
  }

  return cliques;
}

std::vector<Node*> FoldingGraph::GetNodes() const { return nodes_; }

std::vector<BinaryFoldingAction*> FoldingGraph::GetEdges() const {
  std::vector<BinaryFoldingAction*> edges;

  // Collect all the edges
  for (auto& edge : edges_) {
    BinaryFoldingAction* edge_raw = edge.get();
    edges.push_back(edge_raw);
  }

  return edges;
}

uint64_t FoldingGraph::GetInDegree(Node* n) const {
  NodeIndex node_id = node_to_index_.at(n);
  uint64_t in_degree = graph_->InDegree(node_id);

  return in_degree;
}

uint64_t FoldingGraph::GetOutDegree(Node* n) const {
  NodeIndex node_id = node_to_index_.at(n);
  uint64_t out_degree = graph_->OutDegree(node_id);

  return out_degree;
}

std::vector<BinaryFoldingAction*> FoldingGraph::GetEdgesTo(Node* n) const {
  std::vector<BinaryFoldingAction*> edges_to_n;

  // Get the index of @n
  NodeIndex node_id = node_to_index_.at(n);

  // Get the indexes of the incoming edges of @n
  for (EdgeIndex edge_index : graph_->IncomingArcs(node_id)) {
    CHECK_EQ(graph_->Head(edge_index), node_id);

    // Get the edge
    BinaryFoldingAction* f = edges_[edge_index].get();

    // Add the current edge
    edges_to_n.push_back(f);
  }

  return edges_to_n;
}

}  // namespace xls
