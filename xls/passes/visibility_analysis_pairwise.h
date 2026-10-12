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

#ifndef XLS_PASSES_VISIBILITY_ANALYSIS_PAIRWISE_H_
#define XLS_PASSES_VISIBILITY_ANALYSIS_PAIRWISE_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {

// Caches and queries sorted, deduped BDD variables used in BDD expressions.
class BddVariablesAnalysis {
 public:
  explicit BddVariablesAnalysis(const BinaryDecisionDiagram* bdd);

  BddVariablesAnalysis(const BddVariablesAnalysis&) = delete;
  BddVariablesAnalysis& operator=(const BddVariablesAnalysis&) = delete;
  BddVariablesAnalysis(BddVariablesAnalysis&&) = default;
  BddVariablesAnalysis& operator=(BddVariablesAnalysis&&) = default;

  // Returns the sorted, deduplicated BDD variables referenced by `expr`.
  const std::vector<BddVariable>& Variables(BddNodeIndex expr) const;

  // Returns true if any variable present in `expr` is contained in
  // `relevant_vars`, which must be sorted.
  bool UsesVariables(BddNodeIndex expr,
                     absl::Span<const BddVariable> relevant_vars) const;

  void Clear() { bdd_cache_.clear(); }

 private:
  bool IsTerminalOrInfeasible(BddNodeIndex expr) const;

  const BinaryDecisionDiagram* bdd_;
  // Uses `std::unique_ptr` so references remain valid across map rehashes.
  mutable absl::flat_hash_map<BddNodeIndex,
                              std::unique_ptr<std::vector<BddVariable>>>
      bdd_cache_;
};

// Analyzes pairwise visibility of two nodes. Computes the intersection of BDD
// variables relevant to the visibility of both nodes, filters edges down to
// those dependent only on variables in that intersection, and evaluates
// VisibilityAnalysis given the reduced edge set to produce a minimized
// visibility expression relevant to the pair.
class PairwiseVisibilityAnalysis : public ChangeListener {
 public:
  using OperandNode = OperandVisibilityAnalysis::OperandNode;
  using NodeVisibility = VisibilityAnalysis::NodeVisibility;

  static absl::StatusOr<std::unique_ptr<PairwiseVisibilityAnalysis>> Create(
      const OperandVisibilityAnalysis& operand_visibility,
      const VisibilityAnalysis& visibility);

  explicit PairwiseVisibilityAnalysis(
      const OperandVisibilityAnalysis& operand_visibility,
      const VisibilityAnalysis& visibility);

  ~PairwiseVisibilityAnalysis() override;

  PairwiseVisibilityAnalysis(const PairwiseVisibilityAnalysis&) = delete;
  PairwiseVisibilityAnalysis& operator=(const PairwiseVisibilityAnalysis&) =
      delete;
  PairwiseVisibilityAnalysis(PairwiseVisibilityAnalysis&&) = delete;
  PairwiseVisibilityAnalysis& operator=(PairwiseVisibilityAnalysis&&) = delete;

  // Returns visibility expressions for `one` and `other` containing only BDD
  // variables in the intersection of variables in their visibility expressions.
  std::pair<BddNodeIndex, BddNodeIndex> PairwiseVisibility(Node* one,
                                                           Node* other) const;

  // Returns true if at most one of `one` and `other` propagates outside the
  // function or proc under their pairwise visibility expressions.
  bool IsMutuallyExclusive(Node* one, Node* other) const;

  // Returns the (node -> user) edges necessary to compute the pairwise
  // visibility expression for `one` mutually exclusive with `others`.
  absl::StatusOr<absl::flat_hash_set<OperandNode>>
  GetEdgesForMutuallyExclusiveVisibilityExpr(
      Node* one, absl::Span<Node* const> others,
      int64_t max_edges_to_handle = -1) const;

  // Returns the (node -> user) edges necessary to compute the pairwise
  // visibility expression for `one` relative to `other`.
  absl::StatusOr<absl::flat_hash_set<OperandNode>> GetEdgesForVisibilityExpr(
      Node* one, Node* other, int64_t max_edges_to_handle = -1) const {
    return GetEdgesForMutuallyExclusiveVisibilityExpr(one, {other},
                                                      max_edges_to_handle);
  }

  const BddVariablesAnalysis& bdd_variables() const { return bdd_variables_; }
  const VisibilityAnalysis& visibility_analysis() const { return visibility_; }

  void NodeAdded(Node* node) override;
  void NodeDeleted(Node* node) override;
  void OperandChanged(Node* node, Node* old_operand,
                      absl::Span<const int64_t> operand_nos) override;
  void OperandRemoved(Node* node, Node* old_operand) override;
  void OperandAdded(Node* node) override;
  void ClearCacheFields();

 private:
  struct NodePair {
    Node* one;
    Node* other;

    bool operator==(const NodePair& o) const {
      return one == o.one && other == o.other;
    }

    template <typename H>
    friend H AbslHashValue(H h, const NodePair& p) {
      return H::combine(std::move(h), p.one, p.other);
    }
  };

  struct BddVarEdges {
    BddVariable var;
    // Indexes into `NodeVisibility::edges`; if the bit is set, indicates the
    // edge's visibility expression depends on `var`.
    InlineBitmap dependent_edges;
    // Only one entry per variable ever exists, so we sort by variable only.
    bool operator<(const BddVarEdges& other) const { return var < other.var; }
  };

  // Per-node sorted BDD variables and `NodeVisibility::edges` involvement.
  struct NodeVarInfo {
    std::vector<BddVarEdges> bdd_var_to_edges;
  };

  // Key for caching scoped visibility for a node with a given bitmap of kept
  // edges, useful when multiple partner nodes induce the same shared edges.
  struct NodeEdgeMask {
    Node* node;
    InlineBitmap kept_edges;  // Indexed by `NodeVisibility::edges` index.

    bool operator==(const NodeEdgeMask& o) const {
      return node == o.node && kept_edges == o.kept_edges;
    }

    template <typename H>
    friend H AbslHashValue(H h, const NodeEdgeMask& p) {
      return H::combine(std::move(h), p.node, p.kept_edges);
    }
  };

  // Returns computed or cached BDD variable to edge dependencies for `node`.
  const NodeVarInfo& GetNodeVarInfo(Node* node) const;

  // Unions dependent edge bitmaps into `kept_one` and `kept_other` for
  // every BDD variable shared by `one_var_info` and `other_var_info`. Returns
  // true if the two nodes share at least one BDD variable.
  static bool AccumulateKeptEdgesForSharedVars(
      const NodeVarInfo& one_var_info, const NodeVarInfo& other_var_info,
      InlineBitmap& kept_one, InlineBitmap& kept_other);

  // Returns the subset of `edges` whose bit in `kept_edges` is false.
  absl::flat_hash_set<OperandNode> ComputeExcludedEdges(
      absl::Span<const OperandNode> edges,
      const InlineBitmap& kept_edges) const;

  // Returns a visibility expression for `node` using only the edges marked true
  // in `kept_edges`.
  BddNodeIndex ComputeVisibilityWithKeptEdges(
      Node* node, const NodeVisibility& info,
      const InlineBitmap& kept_edges) const;

  const OperandVisibilityAnalysis& operand_visibility_;
  const VisibilityAnalysis& visibility_;
  mutable BddVariablesAnalysis bdd_variables_;
  FunctionBase* f_;

  // Caches a node's BDD variables and the edges from which they originate.
  mutable absl::flat_hash_map<Node*, std::unique_ptr<NodeVarInfo>> var_cache_;
  // Caches pairwise visibility for the ID ordered canonical pair of nodes.
  mutable absl::flat_hash_map<NodePair, std::pair<BddNodeIndex, BddNodeIndex>>
      pairwise_vis_cache_;
  // Caches scoped visibility for a node with a given bitmap of kept edges.
  mutable absl::flat_hash_map<NodeEdgeMask, BddNodeIndex> scoped_vis_cache_;
};

}  // namespace xls

#endif  // XLS_PASSES_VISIBILITY_ANALYSIS_PAIRWISE_H_
