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
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/lazy_node_data.h"
#include "xls/passes/node_dependency_analysis.h"

namespace xls {

// The visibility of a node is the BDD expression that, if true, indicates that
// the node's value propagates outside of the function or proc.
//
// The analysis is conservative, assuming a node is always visible to its user
// unless a cheap BDD expression can be derived for that node->user edge.
class VisibilityAnalysis : public LazyNodeData<BddNodeIndex> {
 public:
  static constexpr int64_t kDefaultTermLimitForNodeToUserEdge = 32;

  explicit VisibilityAnalysis(const NodeForwardDependencyAnalysis* nda,
                              BddQueryEngine* bdd_query_engine);

  explicit VisibilityAnalysis(int64_t edge_term_limit,
                              const NodeForwardDependencyAnalysis* nda,
                              BddQueryEngine* bdd_query_engine);

  // Two nodes are mutually exclusive if, at most, only one of them ever
  // propagates outside of the function or proc.
  bool IsMutuallyExclusive(Node* one, Node* other) const {
    BinaryDecisionDiagram& bdd = bdd_query_engine_->bdd();
    return bdd.Implies(*GetInfo(one), bdd.Not(*GetInfo(other))) == bdd.one();
  }

 protected:
  BddNodeIndex ComputeInfo(
      Node* node,
      absl::Span<const BddNodeIndex* const> user_infos) const override;

  absl::Status MergeWithGiven(BddNodeIndex& info,
                              const BddNodeIndex& given) const override;

  // Propagate from users to operands
  absl::Span<Node* const> GetInputs(Node* const& node) const override {
    return node->users();
  }
  absl::Span<Node* const> GetUsers(Node* const& node) const override {
    return node->operands();
  }

 private:
  const NodeForwardDependencyAnalysis* nda_;
  BddQueryEngine* bdd_query_engine_;
  int64_t edge_term_limit_;

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

 public:
  // It is necessary to recompute visibility whenever nodes are modified
  // because reachability may have changed.
  void NodeAdded(Node* node) override;
  void NodeDeleted(Node* node) override;
  void OperandChanged(Node* node, Node* old_operand,
                      absl::Span<const int64_t> operand_nos) override;
  void OperandRemoved(Node* node, Node* old_operand) override;
  void OperandAdded(Node* node) override;
};

}  // namespace xls

#endif  // XLS_PASSES_VISIBILITY_ANALYSIS_H_
