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

#ifndef XLS_PASSES_FOLDING_GRAPH_H_
#define XLS_PASSES_FOLDING_GRAPH_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/passes/visibility_analysis.h"
#include "ortools/graph/graph.h"

namespace xls {

// This class represents a single folding action where the destination of the
// folding operation is a single IR node.
//
// Folding is the code transformation that re-uses a given hardware resource
// (e.g., multiplier) for multiple IR operations.
//
// Consider for example the following IR:
//   a2 = umul a1, a0
//   b2 = umul b1, b0
//   c  = sel(s, a2, b2)
//
// The code above can be rewritten into
//   lhs = sel(s, a1, b1)
//   rhs = sel(s, a0, b0)
//   c   = umul lhs, rhs
//
// The last version of the code is obtained by folding the two multiplications
// together.
//
// This class is meant to be extended with sub-classes that specify the sources
// (e.g., a single IR node, or a set of nodes) of the folding operation.
//
// Visibility is used to understand which inputs to select as the operands of
// the folded IR node. A node's visibility is an expression which evaluates to
// true whenever the node's value impacts the value of outputs from a function,
// stored in a proc's state, or sent via channels.
//
// Instead of pre-computing visibility expressions, folding actions store the
// edges in the def-use graph between operands and nodes that should be used to
// compose a visibility expression. Such an edge indicates that visibility of
// the operand through the node is conditional.
class FoldingAction {
 public:
  using VisibilityEdges =
      absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>;

  // This function returns the destination of the folding action.
  Node* GetTo() const { return to_; }

  // The (operand -> node) edges that impact visibility of the 'to' node. Each
  // edge indicates that the visibility of operand through node is conditional.
  const VisibilityEdges& GetToVisibilityEdges() const { return to_edges_; }

  // This function returns true if the operation performed by the nodes involved
  // is signed (e.g., smul), false otherwise (e.g., umul).
  bool IsSigned() const { return this->to_->op() == Op::kSMul; }

  // Return the estimate on the amount of area that will be saved if this
  // folding is performed
  double area_saved() const { return area_saved_; }

 protected:
  FoldingAction(Node* to, VisibilityEdges to_edges, double area_saved)
      : to_(to), to_edges_(to_edges), area_saved_(area_saved) {}

 private:
  Node* to_;
  VisibilityEdges to_edges_;
  double area_saved_;
};

// This class represents a single folding action from an IR node into another IR
// node.
//
// An example of such folding action is the following.
// Consider the following IR:
//   a2 = umul a1, a0
//   b2 = umul b1, b0
//   r  = priority_sel(s, cases=[a2, b2])
//
// An instance of the class BinaryFoldingAction where
// - @from is "a2"
// - @to is "b2"
// - @to_case_number is 1 and
// - @select is "r"
// is the folding that transforms the code above into the code below:
//   custom_s = bit_slice(s, start=0, width=1)
//   lhs = priority_sel(custom_s, cases=[a1], default=b1)
//   rhs = priority_sel(custom_s, cases=[a0], default=b0)
//   r   = umul lhs, rhs
class BinaryFoldingAction : public FoldingAction {
 public:
  BinaryFoldingAction(Node* from, Node* to, VisibilityEdges from_edges,
                      VisibilityEdges to_edges, double area_saved)
      : FoldingAction{to, to_edges, area_saved},
        from_{from},
        from_edges_{std::move(from_edges)} {}

  Node* GetFrom() const { return from_; }

  const VisibilityEdges& GetFromVisibilityEdges() const { return from_edges_; }

 private:
  Node* from_;
  VisibilityEdges from_edges_;
};

// This class represents a single folding action from a set of IR nodes into
// another IR node.
//
// An example of such folding action is the following.
// Consider the following IR:
//   a2 = umul a1, a0
//   b2 = umul b1, b0
//   c2 = umul c1, c0
//   d2 = umul d1, d0
//   r  = priority_sel(s, cases=[a2, b2, c2, d2])
//
// An instance of the class NaryFoldingAction where
// - @from is "<a2, 0>, <b2, 1>, <c2, 2>"
// - @to is "d2"
// - @to_case_number is "3" and
// - @select is "r"
// is the folding the transforms the code above into the code below:
//   custom_s = bit_slice(s, start=0, width=3)
//   lhs = priority_sel(custom_s, cases=[a1, b1, c1], default=d1)
//   rhs = priority_sel(custom_s, cases=[a0, b0, c0], default=d0)
//   r   = umul lhs, rhs
class NaryFoldingAction : public FoldingAction {
 public:
  NaryFoldingAction(std::vector<std::pair<Node*, VisibilityEdges>>&& froms,
                    Node* to, VisibilityEdges to_edges, double area_saved = 0.0)
      : FoldingAction{to, std::move(to_edges), area_saved},
        from_(std::move(froms)) {}

  absl::Span<const std::pair<Node*, VisibilityEdges>> GetFrom() const {
    return from_;
  }

  uint64_t GetNumberOfFroms() const { return from_.size(); }

 private:
  // holds pairs of <source, source's used expression>
  std::vector<std::pair<Node*, VisibilityEdges>> from_;
};

// This class organizes the set of binary folding given as input into a graph
// where a binary folding from node ni to the node nj is an edge from ni to nj.
class FoldingGraph {
 public:
  FoldingGraph(
      FunctionBase* f,
      std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions);

  // This function returns a set of sets of edges, one outermost set per clique
  // found in the graph.
  // Each outermost set includes the set of edges of the graph that creates a
  // clique within that graph.
  absl::flat_hash_set<absl::flat_hash_set<BinaryFoldingAction*>>
  GetEdgeCliques();

  // This function returns the IR function that this folding graph represents.
  FunctionBase* function() const;

  // This function returns all the nodes of the folding graph.
  std::vector<Node*> GetNodes() const;

  // This function returns all the edges of the folding graph.
  std::vector<BinaryFoldingAction*> GetEdges() const;

  // This function returns the in-degree of the node @n.
  uint64_t GetInDegree(Node* n) const;

  // This function returns the out-degree of the node @n.
  uint64_t GetOutDegree(Node* n) const;

  // This function returns all the edges of the folding graph that have @n as
  // destination.
  // In other words, these are edges that have @n as head.
  std::vector<BinaryFoldingAction*> GetEdgesTo(Node* n) const;

 private:
  using NodeIndex = int32_t;
  using EdgeIndex = int32_t;
  using Graph = ::util::ReverseArcStaticGraph<NodeIndex, EdgeIndex>;
  FunctionBase* f_;
  std::unique_ptr<Graph> graph_;
  std::vector<Node*> nodes_;
  absl::flat_hash_map<Node*, NodeIndex> node_to_index_;
  std::vector<std::unique_ptr<BinaryFoldingAction>> edges_;
  absl::flat_hash_set<absl::flat_hash_set<NodeIndex>> cliques_;

  void AddNodes(
      absl::Span<const std::unique_ptr<BinaryFoldingAction>> foldable_actions);
  void AddEdges(
      std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions);
  void IdentifyCliques();
};

}  // namespace xls

#endif  // XLS_PASSES_FOLDING_GRAPH_H_
