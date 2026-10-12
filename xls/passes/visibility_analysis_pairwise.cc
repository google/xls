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

#include "xls/passes/visibility_analysis_pairwise.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {

BddVariablesAnalysis::BddVariablesAnalysis(const BinaryDecisionDiagram* bdd)
    : bdd_(bdd) {
  CHECK(bdd_ != nullptr);
}

bool BddVariablesAnalysis::IsTerminalOrInfeasible(BddNodeIndex expr) const {
  return expr == bdd_->zero() || expr == bdd_->one() ||
         expr == BinaryDecisionDiagram::kInfeasible;
}

const std::vector<BddVariable>& BddVariablesAnalysis::Variables(
    BddNodeIndex expr) const {
  if (auto it = bdd_cache_.find(expr); it != bdd_cache_.end()) {
    return *it->second;
  }
  auto vars = std::make_unique<std::vector<BddVariable>>();
  absl::flat_hash_set<BddNodeIndex> visited;
  std::vector<BddNodeIndex> stack = {expr};
  while (!stack.empty()) {
    BddNodeIndex curr = stack.back();
    stack.pop_back();
    if (IsTerminalOrInfeasible(curr) || !visited.insert(curr).second) {
      continue;
    }
    if (auto it = bdd_cache_.find(curr); it != bdd_cache_.end()) {
      vars->insert(vars->end(), it->second->begin(), it->second->end());
      continue;
    }
    const BddNode& node = bdd_->GetNode(curr);
    vars->push_back(node.variable);
    stack.push_back(node.high);
    stack.push_back(node.low);
  }
  absl::c_sort(*vars);
  vars->erase(std::unique(vars->begin(), vars->end()), vars->end());
  return *bdd_cache_.emplace(expr, std::move(vars)).first->second;
}

namespace {

// Returns whether there is any overlap between two sorted containers without
// intermediate allocations and with early exit. Absl doesn't seem to have this.
template <typename T>
bool SortedRangesIntersect(absl::Span<const T> range1,
                           absl::Span<const T> range2) {
  auto first1 = range1.begin();
  auto first2 = range2.begin();
  while (first1 != range1.end() && first2 != range2.end()) {
    if (*first1 < *first2) {
      ++first1;
    } else if (*first2 < *first1) {
      ++first2;
    } else {
      return true;
    }
  }
  return false;
}

// Invokes `on_match(elem1, elem2)` for each pair of equal elements in two
// sorted, duplicate-free spans without intermediate allocations. Returns
// whether any matching pair was found.
template <typename T, typename MatchFn>
bool ForEachSortedIntersection(absl::Span<const T> range1,
                               absl::Span<const T> range2, MatchFn&& on_match) {
  if (range1.empty() || range2.empty() || range1.back() < range2.front() ||
      range2.back() < range1.front()) {
    return false;
  }
  bool has_intersection = false;
  size_t idx1 = 0;
  size_t idx2 = 0;
  while (idx1 < range1.size() && idx2 < range2.size()) {
    if (range1[idx1] < range2[idx2]) {
      ++idx1;
    } else if (range2[idx2] < range1[idx1]) {
      ++idx2;
    } else {
      has_intersection = true;
      on_match(range1[idx1], range2[idx2]);
      ++idx1;
      ++idx2;
    }
  }
  return has_intersection;
}

}  // namespace

bool BddVariablesAnalysis::UsesVariables(
    BddNodeIndex expr, absl::Span<const BddVariable> relevant_vars) const {
  if (IsTerminalOrInfeasible(expr) || relevant_vars.empty()) {
    return false;
  }
  return SortedRangesIntersect(absl::MakeSpan(Variables(expr)), relevant_vars);
}

/* static */ absl::StatusOr<std::unique_ptr<PairwiseVisibilityAnalysis>>
PairwiseVisibilityAnalysis::Create(
    const OperandVisibilityAnalysis& operand_visibility,
    const VisibilityAnalysis& visibility) {
  return std::make_unique<PairwiseVisibilityAnalysis>(operand_visibility,
                                                      visibility);
}

PairwiseVisibilityAnalysis::PairwiseVisibilityAnalysis(
    const OperandVisibilityAnalysis& operand_visibility,
    const VisibilityAnalysis& visibility)
    : operand_visibility_(operand_visibility),
      visibility_(visibility),
      bdd_variables_(&visibility.bdd_query_engine()->bdd()),
      f_(visibility.bound_function()) {
  if (f_ != nullptr) {
    f_->RegisterChangeListener(this);
  }
}

PairwiseVisibilityAnalysis::~PairwiseVisibilityAnalysis() {
  if (f_ != nullptr) {
    f_->UnregisterChangeListener(this);
  }
}

const PairwiseVisibilityAnalysis::NodeVarInfo&
PairwiseVisibilityAnalysis::GetNodeVarInfo(Node* node) const {
  if (auto it = var_cache_.find(node); it != var_cache_.end()) {
    return *it->second;
  }

  const NodeVisibility& vis_info = *visibility_.GetInfo(node);
  visibility_.PopulateContractedVisibility(node, vis_info);
  auto info = std::make_unique<NodeVarInfo>();

  // For each edge, collect the BDD variables used in its visibility expression.
  const int64_t num_edges = static_cast<int64_t>(vis_info.edges.size());
  std::vector<absl::Span<const BddVariable>> edge_vars_list(num_edges);
  std::vector<BddVariable> all_vars;
  for (int64_t edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    const OperandNode& edge = vis_info.edges[edge_idx];
    BddNodeIndex edge_vis = operand_visibility_.OperandVisibilityThroughNode(
        edge.operand, edge.node);
    const std::vector<BddVariable>& edge_vars =
        bdd_variables_.Variables(edge_vis);
    edge_vars_list[edge_idx] = edge_vars;
    all_vars.insert(all_vars.end(), edge_vars.begin(), edge_vars.end());
  }
  absl::c_sort(all_vars);
  all_vars.erase(std::unique(all_vars.begin(), all_vars.end()), all_vars.end());

  // For each variable, collect the edges whose visibility expression uses it.
  info->bdd_var_to_edges.reserve(all_vars.size());
  for (BddVariable var : all_vars) {
    info->bdd_var_to_edges.push_back(BddVarEdges{
        .var = var,
        .dependent_edges = InlineBitmap(num_edges, /*fill=*/false)});
  }
  for (int64_t edge_idx = 0; edge_idx < num_edges; ++edge_idx) {
    for (BddVariable var : edge_vars_list[edge_idx]) {
      size_t var_idx = absl::c_lower_bound(all_vars, var) - all_vars.begin();
      info->bdd_var_to_edges[var_idx].dependent_edges.Set(edge_idx, true);
    }
  }

  return *var_cache_.emplace(node, std::move(info)).first->second;
}

/* static */ bool PairwiseVisibilityAnalysis::AccumulateKeptEdgesForSharedVars(
    const NodeVarInfo& one_var_info, const NodeVarInfo& other_var_info,
    InlineBitmap& kept_one, InlineBitmap& kept_other) {
  return ForEachSortedIntersection(
      absl::MakeConstSpan(one_var_info.bdd_var_to_edges),
      absl::MakeConstSpan(other_var_info.bdd_var_to_edges),
      // For each shared variable, keep any edge that is a function of that var.
      [&](const BddVarEdges& one_entry, const BddVarEdges& other_entry) {
        kept_one.Union(one_entry.dependent_edges);
        kept_other.Union(other_entry.dependent_edges);
      });
}

absl::flat_hash_set<PairwiseVisibilityAnalysis::OperandNode>
PairwiseVisibilityAnalysis::ComputeExcludedEdges(
    absl::Span<const OperandNode> edges, const InlineBitmap& kept_edges) const {
  absl::flat_hash_set<OperandNode> exclusions;
  if (kept_edges.IsAllOnes()) {
    return exclusions;
  }
  for (int64_t edge_idx = 0; edge_idx < static_cast<int64_t>(edges.size());
       ++edge_idx) {
    if (!kept_edges.Get(edge_idx)) {
      exclusions.insert(edges[edge_idx]);
    }
  }
  return exclusions;
}

BddNodeIndex PairwiseVisibilityAnalysis::ComputeVisibilityWithKeptEdges(
    Node* node, const NodeVisibility& info,
    const InlineBitmap& kept_edges) const {
  // If all edges are kept, return the default visibility expression.
  if (kept_edges.IsAllOnes()) {
    return info.visibility;
  }
  // If all edges are excluded, visibility will be unconditionally true.
  if (kept_edges.IsAllZeroes()) {
    return visibility_.bdd_query_engine()->bdd().one();
  }
  auto [it, inserted] = scoped_vis_cache_.try_emplace(
      NodeEdgeMask{.node = node, .kept_edges = kept_edges});
  if (inserted) {
    absl::flat_hash_set<OperandNode> exclusions =
        ComputeExcludedEdges(info.edges, kept_edges);
    // Use the underlying visibility analysis to try further exclusions.
    it->second = visibility_.ConservativeVisibilityByPruningEdges(
        node, info, std::move(exclusions));
  }
  return it->second;
}

std::pair<BddNodeIndex, BddNodeIndex>
PairwiseVisibilityAnalysis::PairwiseVisibility(Node* one, Node* other) const {
  if (one == other) {
    BddNodeIndex vis = visibility_.GetInfo(one)->visibility;
    return {vis, vis};
  }
  // We only cache the pair of nodes in ID ascending order.
  bool swapped = one->id() > other->id();
  NodePair key = swapped ? NodePair{.one = other, .other = one}
                         : NodePair{.one = one, .other = other};
  if (auto it = pairwise_vis_cache_.find(key);
      it != pairwise_vis_cache_.end()) {
    return swapped ? std::make_pair(it->second.second, it->second.first)
                   : it->second;
  }

  const NodeVarInfo& one_var_info = GetNodeVarInfo(one);
  const NodeVarInfo& other_var_info = GetNodeVarInfo(other);
  const NodeVisibility& one_info = *visibility_.GetInfo(one);
  const NodeVisibility& other_info = *visibility_.GetInfo(other);

  InlineBitmap kept_one(one_info.edges.size(), /*fill=*/false);
  InlineBitmap kept_other(other_info.edges.size(), /*fill=*/false);
  BddNodeIndex vis_one;
  BddNodeIndex vis_other;
  if (!AccumulateKeptEdgesForSharedVars(one_var_info, other_var_info, kept_one,
                                        kept_other)) {
    vis_one = one_info.visibility;
    vis_other = other_info.visibility;
  } else {
    vis_one = ComputeVisibilityWithKeptEdges(one, one_info, kept_one);
    vis_other = ComputeVisibilityWithKeptEdges(other, other_info, kept_other);
  }

  std::pair<BddNodeIndex, BddNodeIndex> canonical_result =
      swapped ? std::make_pair(vis_other, vis_one)
              : std::make_pair(vis_one, vis_other);
  pairwise_vis_cache_.emplace(key, canonical_result);
  return {vis_one, vis_other};
}

bool PairwiseVisibilityAnalysis::IsMutuallyExclusive(Node* one,
                                                     Node* other) const {
  auto [vis_one, vis_other] = PairwiseVisibility(one, other);
  return visibility_.bdd_query_engine()->bdd().MutuallyExclusive(vis_one,
                                                                 vis_other);
}

absl::StatusOr<absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>>
PairwiseVisibilityAnalysis::GetEdgesForMutuallyExclusiveVisibilityExpr(
    Node* one, absl::Span<Node* const> others,
    int64_t max_edges_to_handle) const {
  const NodeVarInfo& one_var_info = GetNodeVarInfo(one);
  if (one_var_info.bdd_var_to_edges.empty() || others.empty()) {
    return visibility_.GetEdgesForMutuallyExclusiveVisibilityExpr(
        one, others, max_edges_to_handle);
  }

  const NodeVisibility& one_info = *visibility_.GetInfo(one);
  // Accumulate edges for `one` used in visibility of `one` and any of `others`.
  InlineBitmap kept_one_all(one_info.edges.size(), /*fill=*/false);
  std::vector<BddNodeIndex> others_visible;
  others_visible.reserve(others.size());

  for (Node* other : others) {
    const NodeVarInfo& other_var_info = GetNodeVarInfo(other);
    const NodeVisibility& other_info = *visibility_.GetInfo(other);
    InlineBitmap kept_other(other_info.edges.size(), /*fill=*/false);
    // If `one` and `others` share no BDD vars, defer to general analysis.
    if (!AccumulateKeptEdgesForSharedVars(one_var_info, other_var_info,
                                          kept_one_all, kept_other)) {
      return visibility_.GetEdgesForMutuallyExclusiveVisibilityExpr(
          one, others, max_edges_to_handle);
    }
    others_visible.push_back(
        ComputeVisibilityWithKeptEdges(other, other_info, kept_other));
  }

  BddNodeIndex one_visible =
      ComputeVisibilityWithKeptEdges(one, one_info, kept_one_all);
  absl::flat_hash_set<OperandNode> exclusions_one =
      ComputeExcludedEdges(one_info.edges, kept_one_all);

  // Use the underlying visibility analysis to try further exclusions.
  return visibility_.GetEdgesForMutuallyExclusiveVisibilityExpr(
      one, others, max_edges_to_handle, one_visible, others_visible,
      std::move(exclusions_one));
}

void PairwiseVisibilityAnalysis::NodeAdded(Node* node) { ClearCacheFields(); }
void PairwiseVisibilityAnalysis::NodeDeleted(Node* node) { ClearCacheFields(); }
void PairwiseVisibilityAnalysis::OperandChanged(
    Node* node, Node* old_operand, absl::Span<const int64_t> operand_nos) {
  ClearCacheFields();
}
void PairwiseVisibilityAnalysis::OperandRemoved(Node* node, Node* old_operand) {
  ClearCacheFields();
}
void PairwiseVisibilityAnalysis::OperandAdded(Node* node) {
  ClearCacheFields();
}
void PairwiseVisibilityAnalysis::ClearCacheFields() {
  pairwise_vis_cache_.clear();
  var_cache_.clear();
  scoped_vis_cache_.clear();
}

}  // namespace xls
