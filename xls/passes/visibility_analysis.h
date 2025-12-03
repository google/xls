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

#ifndef XLS_PASSES_VISIBILITY_ANALYSIS_H_
#define XLS_PASSES_VISIBILITY_ANALYSIS_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/lazy_node_data.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/query_engine.h"

namespace xls {

// The visibility of operands to a node's users is the BDD expression that,
// if true, indicates that the operand's value impacts the value of the node.
//
// This analysis caches results for (operand, node) pairs. LazyNodeData was not
// used because the result for a given pair does not necessarily depend on all
// the previous inputs (in this case operands), but LazyNodeData computes them
// all in advance. Additionally, one usually only needs a subset of the
// (operand, node) pairs for a given node, not all of them.
class OperandVisibilityAnalysis : public ChangeListener {
 public:
  struct OperandNode {
    Node* operand;
    Node* node;

    OperandNode(Node* operand, Node* node) : operand(operand), node(node) {}

    template <typename H>
    friend H AbslHashValue(H h, const OperandNode& op_node) {
      return H::combine(std::move(h), op_node.operand, op_node.node);
    }

    bool operator<(const OperandNode& other) const {
      if (operand->id() == other.operand->id()) {
        return node->id() < other.node->id();
      }
      return operand->id() < other.operand->id();
    }
    bool operator==(const OperandNode& other) const {
      return operand == other.operand && node == other.node;
    }

    std::string ToString() const {
      return absl::StrCat(operand->GetName(), "->", node->GetName());
    }
  };

  static constexpr int64_t kDefaultTermLimitForNodeToUserEdge = 32;

  static absl::StatusOr<OperandVisibilityAnalysis> Create(
      const NodeForwardDependencyAnalysis* nda,
      const BddQueryEngine* bdd_query_engine);
  static absl::StatusOr<OperandVisibilityAnalysis> Create(
      int64_t edge_term_limit, const NodeForwardDependencyAnalysis* nda,
      const BddQueryEngine* bdd_query_engine);

  explicit OperandVisibilityAnalysis(const NodeForwardDependencyAnalysis* nda,
                                     const BddQueryEngine* bdd_query_engine,
                                     int64_t edge_term_limit);

  ~OperandVisibilityAnalysis() override;
  OperandVisibilityAnalysis(const OperandVisibilityAnalysis& other) = delete;
  OperandVisibilityAnalysis& operator=(const OperandVisibilityAnalysis& other) =
      delete;
  OperandVisibilityAnalysis(OperandVisibilityAnalysis&& other);
  OperandVisibilityAnalysis& operator=(OperandVisibilityAnalysis&& other);

  // Bind the node data to the given function.
  absl::StatusOr<ReachedFixpoint> Attach(FunctionBase* f);
  // The function that this cache is bound on.
  FunctionBase* bound_function() const { return f_; }

  BddNodeIndex OperandVisibilityThroughNode(Node* operand, Node* node) const;
  virtual BddNodeIndex OperandVisibilityThroughNode(OperandNode& pair) const;

  absl::StatusOr<bool> IsVisibilityIndependentOf(
      Node* operand, Node* node, std::vector<Node*>& sources) const;

  void NodeAdded(Node* node) override;
  void NodeDeleted(Node* node) override;
  void OperandChanged(Node* node, Node* old_operand,
                      absl::Span<const int64_t> operand_nos) override;
  void OperandRemoved(Node* node, Node* old_operand) override;
  void OperandAdded(Node* node) override;

 protected:
  const NodeForwardDependencyAnalysis* nda_;
  const BddQueryEngine* bdd_query_engine_;
  int64_t edge_term_limit_;

  // Caches OperandVisibilityThroughNode for an (operand, node) pair
  mutable absl::flat_hash_map<OperandNode, BddNodeIndex> pair_to_op_vis_;
  FunctionBase* f_;

  BddNodeIndex GetNodeBit(Node* node, uint32_t bit_index) const;
  std::vector<BddNodeIndex> GetNodeBits(Node* node) const;
  bool IsFullyUnconstrained(Node* node) const;
  std::vector<SaturatingBddNodeIndex> GetSaturatingNodeBits(Node* node) const;

  BddNodeIndex AndAggregateSubsetThatFitsTermLimit(
      absl::Span<BddNodeIndex> operands) const;

  BddNodeIndex ConditionOfUse(Node* node, Node* user) const;
  BddNodeIndex ConditionOfUseWithPrioritySelect(Node* node,
                                                PrioritySelect* select) const;
  BddNodeIndex ConditionOfUseWithSelect(Node* node, Select* select) const;
  BddNodeIndex ConditionOnPredicate(std::optional<Node*> predicate) const;
  BddNodeIndex ConditionOfUseWithAnd(Node* node, NaryOp* and_node) const;
  BddNodeIndex ConditionOfUseWithOr(Node* node, NaryOp* or_node) const;

 private:
  // For use of GetNodeBit
  friend class SingleSelectVisibilityAnalysis;
};

// Counts the number of times a node is directly involved in the visibility of
// another node. This is a naive heuristic that looks down the def-use chain
// for specific uses, e.g is the selector in a select, is an operand in an
// `and` or `or` instruction. Used to sort edges for pruning.
class NodeImpactOnVisibilityAnalysis : public LazyNodeData<int64_t> {
 public:
  static absl::StatusOr<NodeImpactOnVisibilityAnalysis> Create(FunctionBase* f);

  explicit NodeImpactOnVisibilityAnalysis()
      : LazyNodeData<int64_t>(
            DagCacheInvalidateDirection::kInvalidatesOperands) {}

  int64_t NodeImpactOnVisibility(Node* node) const { return *GetInfo(node); }

 protected:
  int64_t ComputeInfo(
      Node* node, absl::Span<const int64_t* const> user_infos) const override;

  absl::Status MergeWithGiven(int64_t& info,
                              const int64_t& given) const override;

  // Propagate from users to operands
  absl::Span<Node* const> GetInputs(Node* const& node) const override {
    return node->users();
  }
  absl::Span<Node* const> GetUsers(Node* const& node) const override {
    return node->operands();
  }
};

// The visibility of a node is the BDD expression that, if true, indicates that
// the node's value propagates outside of the function or proc.
//
// The analysis is conservative, assuming a node is always visible to its user
// unless a cheap BDD expression can be derived for that node->user edge.
class VisibilityAnalysis : public LazyNodeData<BddNodeIndex> {
 public:
  static constexpr int64_t kDefaultMaxEdgeCountForPruning = 128;

  using OperandNode = OperandVisibilityAnalysis::OperandNode;

  static absl::StatusOr<std::unique_ptr<VisibilityAnalysis>> Create(
      const OperandVisibilityAnalysis* operand_vis,
      const BddQueryEngine* bdd_query_engine,
      const LazyPostDominatorAnalysis* post_dom_analysis,
      int64_t max_edge_count_for_pruning = kDefaultMaxEdgeCountForPruning,
      absl::flat_hash_set<OperandNode> exclusions = {});

  explicit VisibilityAnalysis(
      const OperandVisibilityAnalysis* operand_vis,
      const BddQueryEngine* bdd_query_engine,
      const LazyPostDominatorAnalysis* post_dom_analysis,
      int64_t max_edge_count_for_pruning = kDefaultMaxEdgeCountForPruning,
      absl::flat_hash_set<OperandNode> exclusions = {});

  absl::StatusOr<ReachedFixpoint> AttachWithGivens(
      FunctionBase* f,
      absl::flat_hash_map<Node*, BddNodeIndex> givens) override;

  // Two nodes are mutually exclusive if, at most, only one of them ever
  // propagates outside of the function or proc.
  bool IsMutuallyExclusive(Node* one, Node* other) const;

  // Returns the (node -> user) edges necessary to compute the visibility
  // expression 'E' for node 'one' such that:
  //   1) node 'one' is visible implies 'E' is true.
  //   2) 'E' is true implies each 'other' is NOT visible
  //
  // Assuming 'one' and all 'other' are mutually exclusive, a trivial result
  // would be all edges impacting the visibility of 'one'. Ideally, only a
  // subset of those edges are returned. The pruned edges are constraints that
  // the visibility of 'other' is not a function of which produce a conservative
  // expression for the visibility of 'one'.
  absl::StatusOr<absl::flat_hash_set<OperandNode>>
  GetEdgesForMutuallyExclusiveVisibilityExpr(Node* one,
                                             absl::Span<Node* const> others,
                                             int64_t max_edges_to_handle) const;

  BddNodeIndex VisibilityOfNearestPostDominator(Node* node) const;

 protected:
  BddNodeIndex ComputeInfo(
      Node* node,
      absl::Span<const BddNodeIndex* const> user_infos) const override;

  absl::Status MergeWithGiven(BddNodeIndex& info,
                              const BddNodeIndex& given) const override;

  // Produces a conservative visibility expression where if 'node' is visible,
  // then the result is true, while retaining as many constraints as possible.
  // Does so by pruning operand->node edges a few node->user hops from 'node'
  // which have high path count, considering up to a configured # of edges.
  BddNodeIndex ConservativeVisibilityByPruningEdges(Node* node) const;

  // Propagate from users to operands
  absl::Span<Node* const> GetInputs(Node* const& node) const override {
    return node->users();
  }
  absl::Span<Node* const> GetUsers(Node* const& node) const override {
    return node->operands();
  }

 private:
  const OperandVisibilityAnalysis* operand_visibility_;
  const BddQueryEngine* bdd_query_engine_;
  const LazyPostDominatorAnalysis* post_dom_analysis_;
  mutable NodeImpactOnVisibilityAnalysis node_impact_analysis_;
  int64_t max_edge_count_for_pruning_;
  absl::flat_hash_set<OperandNode> exclusions_;

 public:
  // It is necessary to recompute visibility whenever nodes are modified
  // because reachability may have changed.
  void NodeAdded(Node* node) override;
  void NodeDeleted(Node* node) override;
  void UserAdded(Node* node, Node* user) override;
  void UserRemoved(Node* node, Node* user) override;
};

struct SingleSelectVisibility {
  Node* source;
  PrioritySelect* select;
  BddNodeIndex visibility;

  SingleSelectVisibility() : source(nullptr), select(nullptr), visibility{-1} {}

  SingleSelectVisibility(Node* source, PrioritySelect* select,
                         BddNodeIndex visibility)
      : source(source), select(select), visibility(visibility) {}

  bool operator==(const SingleSelectVisibility& other) const {
    return source == other.source && select == other.select &&
           visibility == other.visibility;
  }
};

class SingleSelectVisibilityAnalysis
    : public LazyNodeData<SingleSelectVisibility> {
 public:
  using OperandNode = OperandVisibilityAnalysis::OperandNode;

  static absl::StatusOr<std::unique_ptr<SingleSelectVisibilityAnalysis>> Create(
      const OperandVisibilityAnalysis* operand_vis,
      const NodeForwardDependencyAnalysis* nda,
      const BddQueryEngine* bdd_query_engine);

  SingleSelectVisibilityAnalysis(const OperandVisibilityAnalysis* operand_vis,
                                 const NodeForwardDependencyAnalysis* nda,
                                 const BddQueryEngine* bdd_query_engine);

  bool IsMutuallyExclusive(Node* one, Node* other) const;

  absl::StatusOr<absl::flat_hash_set<OperandNode>> GetEdgesForVisibilityExpr(
      Node* one) const;

 protected:
  SingleSelectVisibility ComputeInfo(
      Node* node, absl::Span<const SingleSelectVisibility* const> user_infos)
      const override;

  absl::Status MergeWithGiven(
      SingleSelectVisibility& info,
      const SingleSelectVisibility& given) const override;

  absl::Span<Node* const> GetInputs(Node* const& node) const override {
    return node->users();
  }
  absl::Span<Node* const> GetUsers(Node* const& node) const override {
    return node->operands();
  }

 private:
  const OperandVisibilityAnalysis* operand_visibility_;
  const NodeForwardDependencyAnalysis* nda_;
  const BddQueryEngine* bdd_query_engine_;

 public:
  void NodeAdded(Node* node) override;
  void NodeDeleted(Node* node) override;
  void UserAdded(Node* node, Node* user) override;
  void UserRemoved(Node* node, Node* user) override;
};

}  // namespace xls

#endif  // XLS_PASSES_VISIBILITY_ANALYSIS_H_
