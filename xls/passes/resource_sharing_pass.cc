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

#include "xls/passes/resource_sharing_pass.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "cppitertools/combinations.hpp"
#include "cppitertools/zip.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/bit_provenance_analysis.h"
#include "xls/passes/critical_path_delay_analysis.h"
#include "xls/passes/folding_graph.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/visibility_analysis.h"
#include "xls/passes/visibility_expr_builder.h"
#include "xls/visualization/math_notation.h"

namespace xls {

class TimingAnalysis {
 public:
  TimingAnalysis(
      const std::vector<std::unique_ptr<NaryFoldingAction>>& folding_actions,
      const CriticalPathDelayAnalysis& node_delay);

  int64_t GetDelayIncrease(NaryFoldingAction* folding_action) const;

  double GetDelaySpread(NaryFoldingAction* folding_action) const;

 private:
  absl::flat_hash_map<NaryFoldingAction*, int64_t> delay_increase_;
  absl::flat_hash_map<NaryFoldingAction*, double> delay_spread_;
};

// GetDelayIncrease computes the maximum difference in delay between any pair
// of folded nodes, used to determine whether @next_folding_action should be
// merged into the group of @current_folding_actions.
// NOTE: It is assumed all folding actions have the same destination.
int64_t GetDelayIncrease(
    const CriticalPathDelayAnalysis& node_delay,
    std::vector<BinaryFoldingAction*>& current_folding_actions,
    BinaryFoldingAction* next_folding_action = nullptr) {
  if (current_folding_actions.empty() && !next_folding_action) {
    return 0;
  }

  int64_t min_delay = 0;
  int64_t max_delay = 0;
  // Min/max delay of 0 is not valid because the delay must be the
  // critical path delay of a node, which is likely non-zero. Therefore, we
  // initialize the min/max delay to the destination of the folding actions.
  // If next_folding_action is not provided, we need to initialize min/max
  // delay to the first element in current_folding_actions.
  if (next_folding_action) {
    int64_t to_delay = *node_delay.GetInfo(next_folding_action->GetTo());
    int64_t from_delay = *node_delay.GetInfo(next_folding_action->GetFrom());
    std::tie(min_delay, max_delay) = std::minmax(to_delay, from_delay);
  } else {
    min_delay = *node_delay.GetInfo(current_folding_actions[0]->GetTo());
    max_delay = min_delay;
  }
  for (BinaryFoldingAction* folding : current_folding_actions) {
    min_delay = std::min(min_delay, *node_delay.GetInfo(folding->GetFrom()));
    max_delay = std::max(max_delay, *node_delay.GetInfo(folding->GetFrom()));
  }
  return max_delay - min_delay;
}

// Compute the spread of the delays between the sources and the destination
//
// We use the square distance rather than the distance because we prefer the
// following folding action over the next one:
// Preferred folding action:
//   Destination at delay 1
//   Two sources both at delay 3
//   Total distance = (3 - 1) + (3 - 1) = 4
//   Total square distance = (3 - 1)^2 + (3 - 1)^2 = 8
//
// Less preferred folding action:
//   Destination at delay 1
//   Two sources at delay 2 and 4
//   Total distance = (2 - 1) + (4 - 1) = 4
//   Total square distance = (2 - 1)^2 + (4 - 1)^2 = 10
//
// This is because the last folding action (the less preferred one) often
// creates a less beneficial folding due to the higher furthest-away source
// (4 in the above example).
double GetDelaySpreadBetweenNodes(Node* one, Node* other,
                                  const CriticalPathDelayAnalysis& node_delay) {
  int64_t one_delay = *node_delay.GetInfo(one);
  int64_t other_delay = *node_delay.GetInfo(other);
  return std::pow(static_cast<double>(one_delay - other_delay), 2);
}

double GetDelaySpread(BinaryFoldingAction* folding_action,
                      const CriticalPathDelayAnalysis& node_delay) {
  return GetDelaySpreadBetweenNodes(folding_action->GetFrom(),
                                    folding_action->GetTo(), node_delay);
}

absl::Span<Node* const> GetCases(Node* select) {
  if (select->Is<PrioritySelect>()) {
    return select->As<PrioritySelect>()->cases();
  }
  if (select->Is<OneHotSelect>()) {
    return select->As<OneHotSelect>()->cases();
  }
  CHECK(select->Is<Select>());
  return select->As<Select>()->cases();
}

Node* GetSelector(Node* select) {
  if (select->Is<PrioritySelect>()) {
    return select->As<PrioritySelect>()->selector();
  }
  if (select->Is<OneHotSelect>()) {
    return select->As<OneHotSelect>()->selector();
  }
  CHECK(select->Is<Select>());
  return select->As<Select>()->selector();
}

std::optional<uint32_t> GetSelectCaseNumberOfNode(
    const NodeForwardDependencyAnalysis& nda, Node* node, Node* select_case,
    uint32_t select_case_number) {
  // Check if @node reaches the select case given as input
  if (!nda.IsDependent(node, select_case)) {
    return {};
  }

  // @node reaches the select case given as input.
  // Update the case number
  return select_case_number;
}

// This function check if @node_to_map can be folded into @folding_destination.
//
// The current analysis succeeds at declaring the two nodes can be
// folded together if their types are compatible and if the bit width of the
// @folding_destination is at least as large as that of @node_to_map.
bool CanMapOpInto(Node* node_to_map, Node* folding_destination) {
  // We only handle either nodes of the same types (e.g., umul), or they need to
  // be either add or sub. This is because sub can be mapped into add (and
  // vice versa) by negating the second operand.
  if (node_to_map->op() != folding_destination->op()) {
    bool can_map_into = false;
    constexpr std::array<Op, 2> kAddOrSub = {Op::kAdd, Op::kSub};
    constexpr std::array<Op, 2> kRightShiftOrDynamicBitslice = {
        Op::kShrl, Op::kDynamicBitSlice};
    if (node_to_map->OpIn(kAddOrSub) && folding_destination->OpIn(kAddOrSub)) {
      can_map_into = true;
    }
    if (node_to_map->OpIn(kRightShiftOrDynamicBitslice) &&
        folding_destination->OpIn(kRightShiftOrDynamicBitslice)) {
      can_map_into = true;
    }
    if (!can_map_into) {
      return false;
    }
  }

  // Check the source bit-widths are not larger than the destination bit-widths
  if (node_to_map->BitCountOrDie() > folding_destination->BitCountOrDie()) {
    return false;
  }
  for (auto [operand_from_mul, operand_to_mul] :
       iter::zip(node_to_map->operands(), folding_destination->operands())) {
    if (operand_from_mul->BitCountOrDie() > operand_to_mul->BitCountOrDie()) {
      return false;
    }
  }

  return true;
}

// Check if we are currently capable to potentially handle the node given as
// input for folding.
bool CanTarget(Node* n) {
  if (n->OpIn({Op::kUMul, Op::kSMul, Op::kAdd, Op::kSub, Op::kShrl, Op::kShra,
               Op::kShll, Op::kDynamicBitSlice})) {
    return true;
  }

  return false;
}

// Check if we should consider the node given as input for folding.
// This is part of the profitability guard of the resource sharing pass.
bool ShouldTarget(Node* n) {
  // Additions are not always worth folding
  if (n->OpIn({Op::kAdd})) {
    // They are worth folding only if their bit-width is big enough
    if (n->BitCountOrDie() >= 18) {
      return true;
    }
    return false;
  }

  // Subtractions are not always worth folding
  if (n->OpIn({Op::kSub})) {
    // They are worth folding only if their bit-width is big enough
    if (n->BitCountOrDie() >= 33) {
      return true;
    }
    return false;
  }

  // Default: we consider folding a node potentially profitable.
  return true;
}

// Check if we are currently capable to potentially handle the node given as
// input for folding.
bool CanTarget(Node* n, Node* selector,
               const NodeForwardDependencyAnalysis& nda) {
  // We currently handle only multiplications
  if (!CanTarget(n)) {
    return false;
  }

  // Check if @n reaches the selector.
  // In this case, @n cannot be considered for this select
  return !nda.IsDependent(n, selector);
}

absl::StatusOr<absl::flat_hash_set<ResourceSharingPass::MutuallyExclPair>>
ResourceSharingPass::ComputeMutualExclusionAnalysis(
    FunctionBase* f, OptimizationContext& context,
    const VisibilityAnalyses& visibility,
    const ResourceSharingPass::Config& config) {
  absl::flat_hash_set<MutuallyExclPair> mutual_exclusivity;

  std::vector<Node*> relevant_nodes;
  // Relatively few nodes are expected to satisfy CanTarget and ShouldTarget
  relevant_nodes.reserve(f->node_count() / 10);
  for (Node* node : f->nodes()) {
    if (CanTarget(node) && ShouldTarget(node)) {
      relevant_nodes.push_back(node);
    }
  }
  for (auto&& nodes : iter::combinations(relevant_nodes, 2)) {
    Node* one_node = nodes[0];
    Node* other_node = nodes[1];
    // Leverage both single select visibility analysis and the general
    // visibility analysis in case the other of the two analyses produced a
    // conservative expression which is unable to prove mutual exclusivity.
    // The general analysis produces a conservative expression should the BDD
    // saturate. The single select analysis only produces an expression where
    // the node's visibility is entirely representable by a single priority
    // select descendant of the node.
    if (visibility.single_select.IsMutuallyExclusive(one_node, other_node) ||
        visibility.general.IsMutuallyExclusive(one_node, other_node)) {
      mutual_exclusivity.insert({one_node, other_node});
    }
  }

  // Print the mutual exclusivity relation
  if (VLOG_IS_ON(5)) {
    VLOG(5) << "Mutually exclusive graph";
    absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>
        node_to_mutual_exclusive_nodes;
    for (const auto& pair : mutual_exclusivity) {
      node_to_mutual_exclusive_nodes[pair.one].insert(pair.other);
      node_to_mutual_exclusive_nodes[pair.other].insert(pair.one);
    }
    for (const auto& [one, others] : node_to_mutual_exclusive_nodes) {
      VLOG(5) << one->ToString();
      for (Node* other : others) {
        VLOG(5) << "  <-> " << other->ToString() << " ";
      }
    }
  }

  return mutual_exclusivity;
}

absl::StatusOr<std::vector<std::unique_ptr<BinaryFoldingAction>>>
ResourceSharingPass::ComputeFoldableActions(
    FunctionBase* f, absl::flat_hash_set<MutuallyExclPair>& mutual_exclusivity,
    const VisibilityAnalyses& visibility,
    const ResourceSharingPass::Config& config) {
  std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions;
  for (auto& [one_node, other_node] : mutual_exclusivity) {
    // CanTarget and ShouldTarget have already been vetted, so all that is left
    // is to see if the types and bit widths are compatible
    bool can_one_to_other = CanMapOpInto(one_node, other_node);
    bool can_other_to_one = CanMapOpInto(other_node, one_node);
    if (!can_one_to_other && !can_other_to_one) {
      continue;
    }

    absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode> one_edges;
    absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode> other_edges;
    if (visibility.single_select.IsMutuallyExclusive(one_node, other_node)) {
      XLS_ASSIGN_OR_RETURN(
          one_edges,
          visibility.single_select.GetEdgesForVisibilityExpr(one_node));
      XLS_ASSIGN_OR_RETURN(
          other_edges,
          visibility.single_select.GetEdgesForVisibilityExpr(other_node));
    } else {
      XLS_ASSIGN_OR_RETURN(
          one_edges,
          visibility.general.GetEdgesForMutuallyExclusiveVisibilityExpr(
              one_node, {other_node}, config.max_edges_to_handle));
      XLS_ASSIGN_OR_RETURN(
          other_edges,
          visibility.general.GetEdgesForMutuallyExclusiveVisibilityExpr(
              other_node, {one_node}, config.max_edges_to_handle));
    }
    if (one_edges.empty() || other_edges.empty()) {
      // This will only ever happen if visibility analysis returns no edges
      // because there were too many of them, more than the configured threshold
      // allows for. In this case, it isn't practical to fold.
      VLOG(4) << "Cannot determine visibility expression necessary to fold: "
              << one_node << " into: " << other_node;
      continue;
    }

    if (can_one_to_other) {
      VLOG(4) << "Adding folding action: " << one_node << " into "
              << other_node;
      // Only n-ary foldings are evaluated with area saving thresholds, so we
      // defer computing area saved until then
      foldable_actions.push_back(std::make_unique<BinaryFoldingAction>(
          one_node, other_node, one_edges, other_edges, 0.0));
    }
    if (can_other_to_one) {
      VLOG(4) << "Adding folding action: " << other_node << " into "
              << one_node;
      // Only n-ary foldings are evaluated with area saving thresholds, so we
      // defer computing area saved until then
      foldable_actions.push_back(std::make_unique<BinaryFoldingAction>(
          other_node, one_node, other_edges, one_edges, 0.0));
    }
  }

  // Print the folding actions found
  absl::flat_hash_set<Node*> foldables;
  foldables.reserve(foldable_actions.size() * 2);
  for (const std::unique_ptr<BinaryFoldingAction>& folding : foldable_actions) {
    foldables.insert(folding->GetFrom());
    foldables.insert(folding->GetTo());
  }
  VLOG(1) << "Instructions that can be folded: " << foldables.size();
  VLOG(1) << "Possible folding actions: " << foldable_actions.size();
  if (VLOG_IS_ON(3)) {
    for (const std::unique_ptr<BinaryFoldingAction>& folding :
         foldable_actions) {
      VLOG(3) << "  From " << folding->GetFrom()->ToString();
      VLOG(3) << "  To " << folding->GetTo()->ToString();
    }
  }

  return foldable_actions;
}

bool GreaterBitwidthComparator(BinaryFoldingAction* a0,
                               BinaryFoldingAction* a1) {
  Node* a0_source = a0->GetFrom();
  Node* a1_source = a1->GetFrom();
  int64_t a0_bitwidth = a0_source->BitCountOrDie();
  int64_t a1_bitwidth = a1_source->BitCountOrDie();
  if (a0_bitwidth == a1_bitwidth) {
    // Use ID for a deterministic tiebreaker.
    return a0_source->id() < a1_source->id();
  }
  return a0_bitwidth > a1_bitwidth;
}

absl::StatusOr<std::unique_ptr<NaryFoldingAction>>
ResourceSharingPass::MakeNaryFoldingAction(
    std::vector<BinaryFoldingAction*>& subset_of_edges_to_n, double area_saved,
    const VisibilityAnalyses& visibility,
    const ResourceSharingPass::Config& config) {
  XLS_RET_CHECK_NE(subset_of_edges_to_n.size(), 0);
  std::vector<std::pair<Node*, FoldingAction::VisibilityEdges>> froms;
  froms.reserve(subset_of_edges_to_n.size());
  for (int i = 0; i < subset_of_edges_to_n.size(); ++i) {
    BinaryFoldingAction* binary_folding = subset_of_edges_to_n[i];
    std::vector<Node*> rest_of_others;
    for (int j = i + 1; j < subset_of_edges_to_n.size(); ++j) {
      rest_of_others.push_back(subset_of_edges_to_n[j]->GetFrom());
    }
    rest_of_others.push_back(binary_folding->GetTo());

    bool is_describable_via_single_select = true;
    bool is_describable_via_full_visibility = true;
    for (Node* other : rest_of_others) {
      if (!visibility.single_select.IsMutuallyExclusive(
              binary_folding->GetFrom(), other)) {
        is_describable_via_single_select = false;
        if (!is_describable_via_full_visibility) {
          break;
        }
      }
      if (!visibility.general.IsMutuallyExclusive(binary_folding->GetFrom(),
                                                  other)) {
        is_describable_via_full_visibility = false;
        if (!is_describable_via_single_select) {
          break;
        }
      }
    }

    absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode> edges;
    if (is_describable_via_single_select) {
      edges = binary_folding->GetFromVisibilityEdges();
    } else if (is_describable_via_full_visibility) {
      XLS_ASSIGN_OR_RETURN(
          edges, visibility.general.GetEdgesForMutuallyExclusiveVisibilityExpr(
                     binary_folding->GetFrom(), rest_of_others,
                     config.max_edges_to_handle));
    }
    if (!edges.empty()) {
      froms.push_back(
          std::make_pair(binary_folding->GetFrom(), std::move(edges)));
    }
  }
  XLS_RET_CHECK_GT(froms.size(), 0);
  std::unique_ptr<NaryFoldingAction> new_action =
      std::make_unique<NaryFoldingAction>(
          std::move(froms), subset_of_edges_to_n[0]->GetTo(),
          subset_of_edges_to_n[0]->GetToVisibilityEdges(), area_saved);
  return new_action;
}

// This function implements the heuristics that selects the sub-set of legal
// folding actions to perform based on the cliques of the folding graph.
// This function is a profitability guard of the resource sharing optimization.
//
// This heuristics works particularly well when folding actions are symmetric.
// For example, when only multiplications that have the same bit-widths are
// considered.
absl::StatusOr<std::vector<std::unique_ptr<NaryFoldingAction>>>
SelectFoldingActionsBasedOnCliques(
    FoldingGraph* folding_graph,
    const ResourceSharingPass::VisibilityAnalyses& visibility,
    const ResourceSharingPass::Config& config) {
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;

  // Choose all of them matching the maximum cliques of the folding graph
  for (const absl::flat_hash_set<BinaryFoldingAction*>& clique :
       folding_graph->GetEdgeCliques()) {
    VLOG(3) << "  New clique";

    // Get the destination that is shared among all elements in the clique
    BinaryFoldingAction* first_action = *clique.begin();
    Node* to = first_action->GetTo();
    VLOG(3) << "    To: " << to->ToString();

    // Get the nodes to map into the destination
    //
    // This step requires to translate a clique into a star where the center is
    // the common destination chosen for the entire clique.
    // For now, we choose a random element of the clique as the center of the
    // start. Later, we should find a good heuristic to pick a candidate that
    // lead to a better PPA.
    std::vector<BinaryFoldingAction*> folds;
    folds.reserve(clique.size());
    double area_saved = 0;
    for (BinaryFoldingAction* binary_folding : clique) {
      if (binary_folding->GetTo() != to) {
        // We can skip this binary folding because it doesn't target the
        // destination we chose.
        continue;
      }
      VLOG(3) << "    From: " << binary_folding->GetFrom()->ToString();
      folds.push_back(binary_folding);
      area_saved += binary_folding->area_saved();
    }
    absl::c_sort(folds, GreaterBitwidthComparator);

    // Create a single n-ary folding action for the whole clique
    std::unique_ptr<NaryFoldingAction> new_action;
    XLS_ASSIGN_OR_RETURN(new_action,
                         ResourceSharingPass::MakeNaryFoldingAction(
                             folds, area_saved, visibility, config));
    folding_actions_to_perform.push_back(std::move(new_action));
  }

  // Sort the cliques based on their size
  auto size_comparator = [](std::unique_ptr<NaryFoldingAction>& f0,
                            std::unique_ptr<NaryFoldingAction>& f1) -> bool {
    return f0->GetNumberOfFroms() > f1->GetNumberOfFroms();
  };
  absl::c_sort(folding_actions_to_perform, size_comparator);

  return folding_actions_to_perform;
}

absl::StatusOr<double> EstimateAreaForSelectingASingleInput(
    BinaryFoldingAction* folding, const AreaEstimator& ae) {
  // Get the required information from the folding action
  Node* destination = folding->GetTo();

  // Get the type of the input that will need to be forwarded.
  //
  // Notice that we need to use the type of the operand of the destination of
  // the folding action because the source might have a lower bit-width.
  // In that case, the source input is extended to match the bit-width of the
  // operand of the destination of the folding action.
  Package p("area_check");
  XLS_ASSIGN_OR_RETURN(
      Type * input_type,
      p.MapTypeFromOtherPackage(destination->operand(0)->GetType()));

  // Create the IR that selects which inputs to forward to a given operand of
  // the destination of the folding action.
  FunctionBuilder fb("area_check", &p);
  BValue selector = fb.Param("sel", p.GetBitsType(1));
  BValue select =
      fb.PrioritySelect(selector, {fb.Param("selected", input_type)},
                        fb.Param("n_zero", input_type));
  XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
  XLS_RET_CHECK_EQ(select.node(), f->return_value());

  // Estimate the area to select one input of the folding action
  XLS_ASSIGN_OR_RETURN(double area_select,
                       ae.GetOperationAreaInSquareMicrons(f->return_value()));

  return area_select;
}

absl::StatusOr<double> EstimateAreaForNegatingNode(Node* n,
                                                   const AreaEstimator& ae) {
  Package p("area_check");
  FunctionBuilder fb("area_check", &p);
  XLS_ASSIGN_OR_RETURN(Type * input_type,
                       p.MapTypeFromOtherPackage(n->GetType()));
  BValue value_to_negate = fb.Param("value_to_negate", input_type);
  fb.Negate(value_to_negate);
  XLS_ASSIGN_OR_RETURN(Function * f, fb.Build());
  XLS_ASSIGN_OR_RETURN(double area_select,
                       ae.GetOperationAreaInSquareMicrons(f->return_value()));

  return area_select;
}

// CanFoldTogether evaluates whether two binary foldings can be grouped together
// into a single n-ary folding on the same destination. Note that binary
// foldings are sorted by benefit (e.g maximizing area savings) in descending
// order, meaning 'previous' is more important than 'next'.
bool CanFoldTogether(absl::flat_hash_set<ResourceSharingPass::MutuallyExclPair>&
                         mutual_exclusivity,
                     BinaryFoldingAction* next, BinaryFoldingAction* previous) {
  if (previous->GetTo() != next->GetTo()) {
    return false;
  }
  Node* prev_node = previous->GetFrom();
  Node* next_node = next->GetFrom();
  if (mutual_exclusivity.contains(
          ResourceSharingPass::MutuallyExclPair(prev_node, next_node))) {
    return true;
  }
  VLOG(4) << "        Excluding the following source because it is not "
             "mutually exclusive with the previous source:";
  VLOG(4) << "          Source excluded = " << next_node->ToString();
  VLOG(4) << "          Previous source = " << prev_node->ToString();
  return false;
}

struct NaryFoldEstimate {
  std::vector<BinaryFoldingAction*> folds;
  double area_saved;
  double delay_spread;
  uint64_t delay_increase;
};

// Select the subset of folds. Note that the 'config' fields are not used in
// favor of the explicit max* parameters.
absl::StatusOr<NaryFoldEstimate> SelectSubsetOfFolds(
    const std::vector<BinaryFoldingAction*>& possible_folds,
    const AreaEstimator& area_estimator,
    const CriticalPathDelayAnalysis& critical_path_delay,
    VisibilityEstimator* visibility_estimator, double min_area,
    int64_t max_delay_spread, uint64_t max_delay_increase,
    const ResourceSharingPass::Config& config) {
  std::vector<BinaryFoldingAction*> subset_of_folds;
  subset_of_folds.reserve(possible_folds.size());
  double area_saved = 0;
  double total_spread = 0;
  int64_t max_selector_delay_increase = 0;
  for (BinaryFoldingAction* folding : possible_folds) {
    // Get the source of the binary folding
    Node* from = folding->GetFrom();
    VLOG(4) << "        From " << from->ToString();

    // Estimate the area this folding will save
    XLS_ASSIGN_OR_RETURN(double area_of_from,
                         area_estimator.GetOperationAreaInSquareMicrons(from));
    VLOG(4) << "          Area of the source " << area_of_from;

    // Estimate the area overhead that will be paid to forward one input of
    // the current source (i.e.,
    // @from) to the destination of the folding.
    XLS_ASSIGN_OR_RETURN(
        double area_select,
        EstimateAreaForSelectingASingleInput(folding, area_estimator));
    VLOG(4) << "          Area of selecting a single input " << area_select;

    // Estimate the area the selector takes up: all instructions that did not
    // already exist in determining which source or destination is being used.
    XLS_ASSIGN_OR_RETURN(
        VisibilityEstimator::AreaDelay selector_cost,
        visibility_estimator->GetAreaAndDelayOfVisibilityExpr(
            folding->GetFrom(), folding->GetFromVisibilityEdges()));

    // Estimate the area overhead that will be paid to forward all inputs to
    // the destination of the current binary folding.
    uint32_t number_of_inputs_that_require_select = 0;
    Node* destination = folding->GetTo();
    for (uint32_t op_id = 0; op_id < destination->operand_count(); op_id++) {
      if (destination->operand(op_id) != from->operand(op_id)) {
        number_of_inputs_that_require_select++;
      }
    }
    double area_selects_overhead =
        (area_select * number_of_inputs_that_require_select) +
        selector_cost.area;

    // Add overhead if we need to compensate for having different node types
    // (e.g., folding a sub into an add) in the folding
    double area_overhead = area_selects_overhead;
    if ((from->op() != destination->op()) &&
        destination->OpIn({Op::kAdd, Op::kSub}) &&
        from->OpIn({Op::kAdd, Op::kSub})) {
      // The only case we currently handle where the node types are different
      // is when folding an add to a sub or vice-versa.
      // In both cases, we need to negate the second operand.
      //
      // Check if the negated value is already available
      Node* from_operand = from->operand(1);
      if (from_operand->op() != Op::kNeg) {
        // We actually need to negate the value
        XLS_ASSIGN_OR_RETURN(
            double negate_area,
            EstimateAreaForNegatingNode(from->operand(1), area_estimator));
        area_overhead += negate_area;
      }
    }

    // Compute the net area saved
    double area_saved_by_current_folding = area_of_from - area_overhead;
    VLOG(4) << "          Potential area savings: "
            << area_saved_by_current_folding;
    if (area_saved_by_current_folding < min_area) {
      VLOG(4) << "            Excluding the current source of the folding "
                 "action because it does not generate enough benefits";
      continue;
    }

    double delay_spread = GetDelaySpread(folding, critical_path_delay);
    if (total_spread + delay_spread > max_delay_spread) {
      VLOG(4) << "            Excluding the current source of the folding "
                 "action because delay spread is too large: "
              << total_spread + delay_spread << " > " << max_delay_spread;
      continue;
    }
    total_spread += delay_spread;

    uint64_t delay_increase =
        GetDelayIncrease(critical_path_delay, subset_of_folds, folding);
    if (delay_increase + selector_cost.delay > max_delay_increase) {
      VLOG(4) << "            Excluding the current source of the folding "
                 "action because delay increase is too large: "
              << delay_increase << " > " << max_delay_increase;
      continue;
    }

    // We select this folding
    subset_of_folds.push_back(folding);

    // Update the total area saved by the n-ary folding we are defining
    area_saved += area_saved_by_current_folding;
    max_selector_delay_increase =
        std::max(max_selector_delay_increase, selector_cost.delay);
  }

  uint64_t total_delay_increase =
      max_selector_delay_increase +
      GetDelayIncrease(critical_path_delay, subset_of_folds);
  return NaryFoldEstimate{.folds = subset_of_folds,
                          .area_saved = area_saved,
                          .delay_spread = total_spread,
                          .delay_increase = total_delay_increase};
}

// Return the list of legal n-ary folding actions that target a given node
absl::StatusOr<std::vector<std::unique_ptr<NaryFoldingAction>>>
ListOfAllFoldingActionsWithDestination(
    Node* n, FoldingGraph* folding_graph,
    absl::flat_hash_set<ResourceSharingPass::MutuallyExclPair>&
        mutual_exclusivity,
    const ResourceSharingPass::VisibilityAnalyses& visibility,
    const AreaEstimator& area_estimator,
    const CriticalPathDelayAnalysis& critical_path_delay,
    const ResourceSharingPass::Config& config,
    VisibilityEstimator* visibility_estimator) {
  std::vector<std::unique_ptr<NaryFoldingAction>>
      potential_folding_actions_to_perform;

  // Fetch the in degree of @n
  uint64_t n_in_degree = folding_graph->GetInDegree(n);
  if (n_in_degree == 0) {
    return potential_folding_actions_to_perform;
  }
  VLOG(4) << "";
  VLOG(3) << "    [In-degree=" << n_in_degree << "] " << n->ToString();

  // Get all edges that end to @n
  std::vector<BinaryFoldingAction*> edges_to_n = folding_graph->GetEdgesTo(n);
  XLS_RET_CHECK_GT(edges_to_n.size(), 0);
  XLS_RET_CHECK_EQ(edges_to_n.size(), n_in_degree);

  // Remove folding actions that are from a node that is not mutually
  // exclusive with all the others.
  // To this end, we give priority to folding actions that save
  // higher bit-width operations.
  //
  // Step 0: Sort the folding actions based on the bit-width of the source
  absl::c_sort(edges_to_n, GreaterBitwidthComparator);

  // Step 1: Remove folding actions that are from a node that is not mutually
  // exclusive with all the others.
  //
  // The implemented solution is better than an alternative approach of
  // computing the largest clique among the source nodes of the edges within
  // @edges_to_n. This is because of the following two observations:
  // - we often don't want the largest clique; this is because nodes have
  // different bit-widths and we prefer to fold a single large "add" rather than
  // multiple much smaller "add" nodes.
  // - there are multiple cliques with the largest size
  // In other words, the solution below works better because it prefers
  // solutions with (potentially) fewer sources, but higher bit-widths. This is
  // obtaining by iterating over @edges_to_n that has been sorted in descending
  // order based on the bit-width of the sources of the binary folding actions.
  std::vector<BinaryFoldingAction*> subset_of_edges_to_n;
  subset_of_edges_to_n.reserve(edges_to_n.size());
  VLOG(4) << "      Excluding the sources that would make an illegal n-ary "
             "folding";
  for (BinaryFoldingAction* a : edges_to_n) {
    auto can_fold_together = [&](BinaryFoldingAction* b) {
      return CanFoldTogether(mutual_exclusivity, a, b);
    };
    // Check if @a_source is mutually-exclusive with all other nodes already
    // confirmed (i.e., the sources of @subset_of_edges_to_n)
    bool is_a_mutually_exclusive =
        absl::c_all_of(subset_of_edges_to_n, can_fold_together);

    // Consider the current folding action only if its source is mutually
    // exclusive with the other sources
    if (is_a_mutually_exclusive) {
      subset_of_edges_to_n.push_back(a);
    }
  }
  XLS_RET_CHECK_GT(subset_of_edges_to_n.size(), 0);

  // Estimate the area saved by the n-ary folding action.
  XLS_ASSIGN_OR_RETURN(
      NaryFoldEstimate estimate,
      SelectSubsetOfFolds(subset_of_edges_to_n, area_estimator,
                          critical_path_delay, visibility_estimator,
                          -std::numeric_limits<double>::infinity(),
                          std::numeric_limits<int64_t>::max(),
                          std::numeric_limits<uint64_t>::max(), config));

  // Create the hyper-edge by merging all these edges
  // Notice this is possible because all edges in @edges_to_n are guaranteed
  // to have @n as destination.
  XLS_ASSIGN_OR_RETURN(
      std::unique_ptr<NaryFoldingAction> new_action,
      ResourceSharingPass::MakeNaryFoldingAction(
          estimate.folds, estimate.area_saved, visibility, config));
  potential_folding_actions_to_perform.push_back(std::move(new_action));

  return potential_folding_actions_to_perform;
}

// Return the list of legal and profitable n-ary folding actions that target a
// given node
absl::StatusOr<std::vector<std::unique_ptr<NaryFoldingAction>>>
ListOfFoldingActionsWithDestination(
    Node* n, FoldingGraph* folding_graph,
    absl::flat_hash_set<ResourceSharingPass::MutuallyExclPair>&
        mutual_exclusivity,
    const ResourceSharingPass::VisibilityAnalyses& visibility,
    const AreaEstimator& area_estimator,
    const CriticalPathDelayAnalysis& critical_path_delay,
    const ResourceSharingPass::Config& config,
    VisibilityEstimator* visibility_estimator) {
  std::vector<std::unique_ptr<NaryFoldingAction>>
      potential_folding_actions_to_perform;

  // Fetch the in degree of @n
  uint64_t n_in_degree = folding_graph->GetInDegree(n);
  if (n_in_degree == 0) {
    return potential_folding_actions_to_perform;
  }
  VLOG(3) << "";
  VLOG(3) << "    [In-degree=" << n_in_degree << "] " << n;

  // Get all edges that end to @n
  std::vector<BinaryFoldingAction*> edges_to_n = folding_graph->GetEdgesTo(n);
  XLS_RET_CHECK_GT(edges_to_n.size(), 0);
  XLS_RET_CHECK_EQ(edges_to_n.size(), n_in_degree);

  // Remove folding actions that are from a node that is not mutually
  // exclusive with all the others.
  // To this end, we give priority to folding actions that save
  // higher bit-width operations.
  //
  // Step 0: Sort the folding actions based on the bit-width of the source
  absl::c_sort(edges_to_n, GreaterBitwidthComparator);

  // As the number of nodes folded together increases, it becomes very likely
  // that too much delay is introduced and the folding will be discarded. Also,
  // only one n-ary fold upon a given destination node will be selected in the
  // very end.
  //
  // To bound complexity by some factor of O(folds) instead of O(folds^2) in our
  // search for the best n-ary folds, we limit the number of nodes we attempt to
  // fold together and the maximum number of alternative n-ary folds on a
  // destination we consider, leveraging the above two properties.
  constexpr int64_t max_batches = 10;
  constexpr int64_t max_batch_size = 20;
  // holds onto possible alternative sets of nodes, where for each set, each
  // instruction in the set is mutually exclusive with all other members. as
  // well as with the destination @n of that potential n-ary fold.
  std::vector<std::vector<BinaryFoldingAction*>> subsets_of_edges_to_n;
  for (auto fold : edges_to_n) {
    bool found_batch = false;
    for (auto& batch : subsets_of_edges_to_n) {
      if (batch.size() >= max_batch_size) {
        continue;
      }
      bool all_can_fold = true;
      for (auto other_fold : batch) {
        if (!CanFoldTogether(mutual_exclusivity, fold, other_fold)) {
          all_can_fold = false;
          break;
        }
      }
      if (all_can_fold) {
        batch.push_back(fold);
        found_batch = true;
        break;
      }
    }
    if (!found_batch) {
      if (subsets_of_edges_to_n.size() >= max_batches) {
        break;
      }
      subsets_of_edges_to_n.push_back({fold});
    }
  }

  // For each set of mutually exclusive nodes with @n as destination, select a
  // subset of those nodes that meet area and delay constraints. Each resulting
  // n-ary fold will be an alternative where eventually at most one is selected.
  for (int i = 0; i < subsets_of_edges_to_n.size(); ++i) {
    // Select the sub-set of the edges that will result in a profitable folding
    VLOG(4)
        << "      Select the sub-set of the possible binary folding actions, "
           "which will generate a profitable outcome";
    XLS_ASSIGN_OR_RETURN(
        NaryFoldEstimate estimate,
        SelectSubsetOfFolds(subsets_of_edges_to_n[i], area_estimator,
                            critical_path_delay, visibility_estimator,
                            config.min_area_savings,
                            config.max_delay_spread_squared,
                            config.max_delay_increase_per_fold, config));

    if (estimate.folds.empty()) {
      VLOG(4)
          << "      Excluding the current destination of the folding action "
             "because there are no binary edges with that node as "
             "destination that are profitable to consider";
      continue;
    }

    // Create the hyper-edge by merging all these edges
    // Notice this is possible because all edges in @edges_to_n are guaranteed
    // to have @n as destination.
    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<NaryFoldingAction> new_action,
        ResourceSharingPass::MakeNaryFoldingAction(
            estimate.folds, estimate.area_saved, visibility, config));
    potential_folding_actions_to_perform.push_back(std::move(new_action));
  }
  return potential_folding_actions_to_perform;
}

absl::StatusOr<std::pair<std::vector<std::unique_ptr<NaryFoldingAction>>, bool>>
ResourceSharingPass::LegalizeSequenceOfFolding(
    absl::Span<const std::unique_ptr<NaryFoldingAction>>
        potential_folding_actions_to_perform,
    absl::flat_hash_set<ResourceSharingPass::MutuallyExclPair>&
        mutual_exclusivity,
    const NodeBackwardDependencyAnalysis& nda,
    std::optional<const AreaEstimator*> area_estimator,
    const ResourceSharingPass::Config& config) {
  if (potential_folding_actions_to_perform.empty()) {
    return std::make_pair(std::vector<std::unique_ptr<NaryFoldingAction>>(),
                          false);
  }
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;
  bool modified = false;

  // Prevent cycles by tracking what nodes are folded together; paired with nda,
  // we detect when a `to` node will depend on a `from` node after foldings
  // occur, which if `to` and `from` are also folded together, creates a cycle.
  // Consider if B <- A <- X and D <- C <- Y are two def-use chains; folding
  // X into D to produce D' makes a new def-use chain B <- A <- D' <- C <- Y.
  // Folding Y into B would cause a cycle because B now depends on Y via D'.
  // NOTE: If nda could be modified to merge dependency sets of nodes that
  // will be folded together, this map would not be necessary:
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> one_to_others_folded;
  auto will_to_use_from = [&](Node* from, Node* to) -> bool {
    const absl::flat_hash_set<Node*>* nodes_using_from = nda.GetInfo(from);
    if (nodes_using_from->contains(to)) {
      return true;
    }
    for (Node* node_using_from : *nodes_using_from) {
      if (auto merged_nodes_it = one_to_others_folded.find(node_using_from);
          merged_nodes_it != one_to_others_folded.end()) {
        // Check `to` does not use any nodes that will use `from` post-foldings.
        for (Node* will_use_from : merged_nodes_it->second) {
          if (nda.IsDependent(will_use_from, to)) {
            return true;
          }
        }
      }
    }
    return false;
  };

  // Remove folding actions (via removing either a whole n-ary folding or a
  // subset of the froms of a given n-ary folding) that overlaps.
  //
  // Notice that the overlap depends on the specific order chosen between the
  // n-ary folding actions. We iterate over the n-ary folding actions in
  // descending order based on the amount of area they save to give priority to
  // those that will have a bigger positive impact.
  VLOG(3) << "  Remove overlapping folding actions";
  absl::flat_hash_map<Node*, NaryFoldingAction*>
      nodes_already_selected_as_folding_sources;
  absl::flat_hash_map<Node*, NaryFoldingAction*> prior_folding_of_destination;
  for (const std::unique_ptr<NaryFoldingAction>& folding :
       potential_folding_actions_to_perform) {
    // Print the n-ary folding
    VLOG(3) << "";
    VLOG(3) << "    To: " << folding->GetTo()->ToString();
    for (auto& [from_node, _] : folding->GetFrom()) {
      VLOG(3) << "      From: " << from_node->ToString();
    }
    std::optional<double> area_saved_without_overhead = folding->area_saved();
    if (area_saved_without_overhead.has_value()) {
      VLOG(2) << "      Area savings = " << *area_saved_without_overhead;
    }

    // Check the destination
    Node* to_node = folding->GetTo();
    auto it = nodes_already_selected_as_folding_sources.find(to_node);
    if (it != nodes_already_selected_as_folding_sources.end()) {
      // The destination of the current n-ary folding (i.e., @to_node) was used
      // as a source on a prior n-ary folding. Hence, @to_node must be mutually
      // exclusive with the destination of such prior n-ary folding for @folding
      // to be legal.
      //
      // For example, let us assume the prior n-ary folding was from n_i to n_j.
      // Let us also assume that the current n-ary folding is from n_k to n_i;
      // this current folding is legal only if n_k is mutually exclusive with
      // n_j since the previous folding will happen before the current one.
      NaryFoldingAction* prior_folding = it->second;
      XLS_RET_CHECK_NE(prior_folding, nullptr);
      Node* prior_folding_destination = prior_folding->GetTo();
      XLS_RET_CHECK_NE(prior_folding_destination, to_node);
      bool skip_current_folding = false;
      for (auto& [source, _] : folding->GetFrom()) {
        if (!mutual_exclusivity.contains({source, prior_folding_destination})) {
          VLOG(4) << "      Excluding the current n-ary folding f_i because "
                     "its destination was used by a prior n-ary folding f_j "
                     "that has a destination that is not mutually exclusive "
                     "with the following source of f_i";
          VLOG(4) << "        Problematic source of f_i   = "
                  << source->ToString();
          VLOG(4) << "        Prior folding's destination = "
                  << prior_folding_destination->ToString();
          skip_current_folding = true;
          break;
        }
        for (auto& [prior_folding_from, _] : prior_folding->GetFrom()) {
          if (!mutual_exclusivity.contains({source, prior_folding_from})) {
            skip_current_folding = true;
            break;
          }
        }
        if (skip_current_folding) {
          break;
        }
      }
      if (skip_current_folding) {
        modified = true;
        continue;
      }
    }

    // Check all sources of the current n-ary folding
    std::vector<std::pair<Node*, FoldingAction::VisibilityEdges>> legal_froms;
    legal_froms.reserve(folding->GetNumberOfFroms());
    double area_saved = area_saved_without_overhead.value_or(0.0);
    for (const auto& [source, source_edges] : folding->GetFrom()) {
      // Exclude folding sources that have already been selected as source
      // on another, previous folding action.
      if (nodes_already_selected_as_folding_sources.contains(source)) {
        VLOG(4)
            << "      Excluding the following source because it was already "
               "considered as a source of a previous n-ary folding action: ";
        VLOG(4) << "        " << source->ToString();

        if (area_estimator.has_value()) {
          // Reduce the area saved
          XLS_ASSIGN_OR_RETURN(
              double area_of_source,
              (*area_estimator)->GetOperationAreaInSquareMicrons(source));
          area_saved -= area_of_source;
        }
        continue;
      }

      // Exclude folding sources s_i that have been used as destination of
      // another, previous folding action f_j where f_j had a source that is not
      // mutually exclusive with s_i.
      //
      // For example, let us assume that a previous folding action was from
      // node_a to node_b. Also, let us assume that the current folding is from
      // node_b to node_c. Finally, let us assume that node_a is not mutually
      // exclusive with node_c; this is possible because the mutual exclusive
      // relation is not transitive. Then, node_b -> node_c folding is illegal
      // if done after node_a -> node_b.
      if (auto it = prior_folding_of_destination.find(source);
          it != prior_folding_of_destination.end()) {
        NaryFoldingAction* prior_folding = (*it).second;

        // The node @source was used as destination on a prior n-ary folding.
        //
        // For @source to be legally foldable to @to_node, all the sources of
        // the previous folding (i.e., @prior_folding) must be mutually
        // exclusive with @to_node.
        // This check is needed because the mutual exclusive relation is not
        // transitive.
        bool is_safe = true;
        for (const auto& [prior_folding_from, _] : prior_folding->GetFrom()) {
          if (!mutual_exclusivity.contains({prior_folding_from, to_node})) {
            VLOG(4) << "      Excluding the following source because it was "
                       "already used by a prior n-ary folding as its "
                       "destination and this prior folding sources are not all "
                       "mutually exclusive with the following source";
            VLOG(4) << "        Source removed = " << source->ToString();
            VLOG(4) << "        Prior folding";
            VLOG(4) << "          To " << prior_folding->GetTo()->ToString();
            for (auto& [tmp_prior_folding_from, _] : prior_folding->GetFrom()) {
              if (prior_folding_from == tmp_prior_folding_from) {
                VLOG(4) << "          From (reason) "
                        << tmp_prior_folding_from->ToString();
              } else {
                VLOG(4) << "          From "
                        << tmp_prior_folding_from->ToString();
              }
            }
            is_safe = false;
            continue;
          }
        }
        if (!is_safe) {
          modified = true;
          continue;
        }
      }

      if (will_to_use_from(source, to_node) ||
          will_to_use_from(to_node, source)) {
        VLOG(4) << "      Excluding the following source because it causes a"
                   "use-chain between source and destination to become a cycle";
        VLOG(4) << "        Source removed = " << source->ToString();
        modified = true;
        continue;
      }

      // The current from is legal
      legal_froms.push_back(std::make_pair(source, source_edges));
    }
    if (legal_froms.empty()) {
      VLOG(4) << "      Excluding the current n-ary folding because it has no "
                 "sources left";
      modified = true;
      continue;
    }

    // Even if safe, the current n-ary fold destination is also the destination
    // of a previous fold, which was more valuable since foldings have been
    // sorted by importance in descending order.
    // NOTE: performing both folds is not ideal; if it was, the current fold's
    // nodes would have been merged with the previous fold's nodes upon the
    // construction of these n-ary folding actions.
    if (prior_folding_of_destination.contains(to_node)) {
      modified = true;
      continue;
    }

    // Track what nodes are merged together for future cycle detection.
    for (auto& [from, _] : legal_froms) {
      one_to_others_folded[from].insert(to_node);
      one_to_others_folded[to_node].insert(from);
    }

    // The current n-ary folding is worth considering. Allocate a new n-ary
    // folding to capture it.
    std::unique_ptr<NaryFoldingAction> new_folding =
        std::make_unique<NaryFoldingAction>(std::move(legal_froms), to_node,
                                            folding->GetToVisibilityEdges(),
                                            area_saved);

    // Keep track of the current n-ary folding to legalize the next ones.
    prior_folding_of_destination[to_node] = new_folding.get();

    // Keep track of all the nodes chosen as source of the folding to legalize
    // the next ones.
    for (auto& [from, from_case_number] : new_folding->GetFrom()) {
      XLS_RET_CHECK(!nodes_already_selected_as_folding_sources.contains(from));
      nodes_already_selected_as_folding_sources[from] = new_folding.get();
    }

    // Add the current n-ary folding to the list of folding to perform.
    folding_actions_to_perform.push_back(std::move(new_folding));
  }

  return std::make_pair(std::move(folding_actions_to_perform), modified);
}

// This function sorts the folding actions given as input in descending order
// based on the amount of area they save.
void SortFoldingActionsInDescendingOrderOfTheirAreaSavings(
    std::vector<std::unique_ptr<NaryFoldingAction>>& folding_actions,
    TimingAnalysis& ta) {
  auto area_comparator = [&ta](std::unique_ptr<NaryFoldingAction>& f0,
                               std::unique_ptr<NaryFoldingAction>& f1) -> bool {
    // Prioritize folding actions that save more area
    double area_f0 = f0->area_saved();
    double area_f1 = f1->area_saved();
    if (area_f0 > area_f1) {
      return true;
    }
    if (area_f0 < area_f1) {
      return false;
    }
    CHECK_EQ(area_f0, area_f1);

    // The two folding actions given as input save the same amount of area.
    // Prioritize folding actions that have:
    // - a smaller square distance between the delays of the IR nodes involved
    // in them that have sources, and
    // - a smaller perturbation to the destination of the folding action.
    int64_t f0_id = f0->GetTo()->id();
    int64_t f1_id = f1->GetTo()->id();
    if (f0->GetTo()->OpIn({Op::kAdd, Op::kDynamicBitSlice}) ||
        f1->GetTo()->OpIn({Op::kAdd, Op::kDynamicBitSlice})) {
      // These nodes behave differently than the others. They tend to lead to
      // better area savings even when their folding leads to higher delay
      // spread. Because of this, we rely on the ID-based tie breaker rather
      // than delay-based metrics.
      return f0_id > f1_id;
    }
    uint64_t f0_delay_delta = ta.GetDelayIncrease(f0.get());
    uint64_t f1_delay_delta = ta.GetDelayIncrease(f1.get());
    double f0_delay_spread = ta.GetDelaySpread(f0.get());
    double f1_delay_spread = ta.GetDelaySpread(f1.get());
    const double kDelayDeltaWeight =
        1.01;  // We care more about the delay
               // increase to the destination of the
               // folding action than the delay spread.
    double f0_delay_total =
        (static_cast<double>(f0_delay_delta) * kDelayDeltaWeight) +
        f0_delay_spread;
    double f1_delay_total =
        (static_cast<double>(f1_delay_delta) * kDelayDeltaWeight) +
        f1_delay_spread;
    return std::forward_as_tuple(f0_delay_total, f0_id) <
           std::forward_as_tuple(f1_delay_total, f1_id);
  };
  absl::c_sort(folding_actions, area_comparator);
}

// This function sorts the IR nodes in @nodes based on their in-degree withing
// the folding graph.
void SortNodesInDescendingOrderOfTheirInDegree(std::vector<Node*>& nodes,
                                               FoldingGraph* folding_graph) {
  auto node_degree_comparator = [folding_graph](Node* n0, Node* n1) {
    // Nodes with higher in-degree comes first
    uint64_t n0_in_degree = folding_graph->GetInDegree(n0);
    uint64_t n1_in_degree = folding_graph->GetInDegree(n1);
    if (n0_in_degree > n1_in_degree) {
      return true;
    }
    if (n0_in_degree < n1_in_degree) {
      return false;
    }

    // Sort the nodes with the same in-degree based on their out-degrees: nodes
    // with lower out-degree comes first.
    uint64_t n0_out_degree = folding_graph->GetInDegree(n0);
    uint64_t n1_out_degree = folding_graph->GetInDegree(n1);
    if (n0_out_degree < n1_out_degree) {
      return true;
    }
    if (n0_out_degree > n1_out_degree) {
      return false;
    }

    // Sort the nodes with the same in-degree and out-degree based on their IDs.
    // This will make the sorting deterministic
    return n0->id() < n1->id();
  };
  absl::c_sort(nodes, node_degree_comparator);
}

// This function implements the heuristic that selects the sub-set of legal
// folding actions to perform based on the in-degree of the nodes of the
// folding graph.
// This function is a profitability guard of the resource sharing optimization.
//
// This heuristics works particularly well when folding actions are asymmetric
// and when there is a high correlation between bit-widths of a node and the
// number of compatible nodes that are mutually exclusive with it.
//
// This situation occurs when when multiplications with different bit-widths are
// considered and when many nodes can be folded into the few one that have high
// bit-widths.
absl::StatusOr<std::vector<std::unique_ptr<NaryFoldingAction>>>
SelectFoldingActionsBasedOnInDegree(
    OptimizationContext& context, FoldingGraph* folding_graph,
    absl::flat_hash_set<ResourceSharingPass::MutuallyExclPair>&
        mutual_exclusivity,
    const ResourceSharingPass::VisibilityAnalyses& visibility,
    const NodeBackwardDependencyAnalysis& nda,
    const AreaEstimator& area_estimator,
    const CriticalPathDelayAnalysis& critical_path_delay,
    const ResourceSharingPass::Config& config,
    VisibilityEstimator* visibility_estimator) {
  // Get the nodes of the folding graph
  std::vector<Node*> nodes = folding_graph->GetNodes();

  // Prioritize folding actions where the target is a node with higher
  // in-degree.
  //
  // To do so, the next code sorts the nodes based on their in-degree.
  SortNodesInDescendingOrderOfTheirInDegree(nodes, folding_graph);

  // Generate a list of all n-ary folding actions starting from targeting as
  // destination the node with the highest in-degree and go through the rest of
  // the nodes (as next potential folding destinations) in descending order of
  // their in-degree.
  //
  // The next code assumes @nodes is sorted in descending order based on the
  // node's in-degree within the folding graph.
  VLOG(3) << "  Generate a list of possible n-ary folding actions to perform";
  std::vector<std::unique_ptr<NaryFoldingAction>>
      potential_folding_actions_to_perform;
  for (Node* n : nodes) {
    // Generate the list of profitable n-ary folding actions that have @n as
    // destination.
    XLS_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<NaryFoldingAction>>
            foldings_with_n_as_destination,
        ListOfFoldingActionsWithDestination(
            n, folding_graph, mutual_exclusivity, visibility, area_estimator,
            critical_path_delay, config, visibility_estimator));

    // Append such list to the list of all profitable n-ary folding actions.
    for (std::unique_ptr<NaryFoldingAction>& folding :
         foldings_with_n_as_destination) {
      if (folding->area_saved() >= config.min_area_savings) {
        potential_folding_actions_to_perform.push_back(std::move(folding));
      }
    }
  }

  // Perform timing analysis, which will be used to decide which folding actions
  // to perform to maximize area while minimizing the risk of making the
  // overall critical path unacceptably worst.
  TimingAnalysis ta{potential_folding_actions_to_perform, critical_path_delay};

  // Filter out folding actions that are likely to generate timing problems
  std::vector<std::unique_ptr<NaryFoldingAction>>
      potential_folding_actions_to_perform_without_timing_problems;
  potential_folding_actions_to_perform_without_timing_problems.reserve(
      potential_folding_actions_to_perform.size());
  for (std::unique_ptr<NaryFoldingAction>& folding :
       potential_folding_actions_to_perform) {
    if (ta.GetDelaySpread(folding.get()) <= config.max_delay_spread_squared) {
      potential_folding_actions_to_perform_without_timing_problems.push_back(
          std::move(folding));
    }
  }

  // Sort the list of n-ary folding actions to give priority to those that will
  // save more area
  SortFoldingActionsInDescendingOrderOfTheirAreaSavings(
      potential_folding_actions_to_perform_without_timing_problems, ta);
  if (VLOG_IS_ON(5)) {
    VLOG(3) << "  List of all possible n-ary folding actions";
    for (const std::unique_ptr<NaryFoldingAction>& folding :
         potential_folding_actions_to_perform_without_timing_problems) {
      VLOG(5) << "";
      VLOG(5) << "    To [delay="
              << *critical_path_delay.GetInfo(folding->GetTo()) << "] "
              << folding->GetTo()->ToString();
      for (auto& [from_node, _] : folding->GetFrom()) {
        VLOG(5) << "      From [delay="
                << *critical_path_delay.GetInfo(from_node) << "] "
                << from_node->ToString();
      }
      VLOG(5) << "      Area savings = " << folding->area_saved();
      VLOG(5) << "      Time analysis = " << ta.GetDelaySpread(folding.get())
              << "," << ta.GetDelayIncrease(folding.get());
    }
  }

  // Make the current sequence of n-ary folding legal.
  //
  // In more detail, at this point, every n-ary folding action is legal in
  // isolation.
  // However, a given n-ary folding action might be illegal if another one run
  // before.
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;
  XLS_ASSIGN_OR_RETURN(
      std::tie(folding_actions_to_perform, std::ignore),
      ResourceSharingPass::LegalizeSequenceOfFolding(
          std::move(
              potential_folding_actions_to_perform_without_timing_problems),
          mutual_exclusivity, nda, &area_estimator, config));

  return folding_actions_to_perform;
}

// This function selects all legal n-ary folding actions.
//
// Notice that there is not a single longest sequence of legal n-ary folding
// actions. This is because we cannot have overlapping between such actions and
// therefore multiple longest sequences are possible. This is because there are
// multiple maximum cliques in the folding graph. This function prioritizes the
// longest sequence that is the closest to the one returned by
// @SelectFoldingActionsBasedOnInDegree (i.e., the default profitability guard
// of the resource sharing pass). In more detail,
// @SelectFoldingActionsBasedOnInDegree will always return a sub-sequence of the
// sequence returned by this function.
absl::StatusOr<std::vector<std::unique_ptr<NaryFoldingAction>>>
SelectAllFoldingActions(
    OptimizationContext& context, FoldingGraph* folding_graph,
    absl::flat_hash_set<ResourceSharingPass::MutuallyExclPair>&
        mutual_exclusivity,
    const ResourceSharingPass::VisibilityAnalyses& visibility,
    const NodeBackwardDependencyAnalysis& nda,
    const AreaEstimator& area_estimator,
    const CriticalPathDelayAnalysis& critical_path_delay,
    const ResourceSharingPass::Config& config,
    VisibilityEstimator* visibility_estimator) {
  // Get the nodes of the folding graph
  std::vector<Node*> nodes = folding_graph->GetNodes();

  // Prioritize folding actions where the target is a node with higher
  // in-degree.
  //
  // To do so, the next code sorts the nodes based on their in-degree.
  SortNodesInDescendingOrderOfTheirInDegree(nodes, folding_graph);

  // Generate a list of all n-ary folding actions starting from targeting as
  // destination the node with the highest in-degree and go through the rest of
  // the nodes (as next potential folding destinations) in descending order of
  // their in-degree.
  //
  // The next code assumes @nodes is sorted in descending order based on the
  // node's in-degree within the folding graph.
  VLOG(3) << "  Generate a list of possible n-ary folding actions to perform";
  std::vector<std::unique_ptr<NaryFoldingAction>>
      potential_folding_actions_to_perform;
  for (Node* n : nodes) {
    // Generate the list of profitable n-ary folding actions that have @n as
    // destination.
    XLS_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<NaryFoldingAction>>
            foldings_with_n_as_destination,
        ListOfAllFoldingActionsWithDestination(
            n, folding_graph, mutual_exclusivity, visibility, area_estimator,
            critical_path_delay, config, visibility_estimator));

    // Append such list to the list of all profitable n-ary folding actions.
    for (std::unique_ptr<NaryFoldingAction>& folding :
         foldings_with_n_as_destination) {
      potential_folding_actions_to_perform.push_back(std::move(folding));
    }
  }
  if (VLOG_IS_ON(5)) {
    VLOG(3) << "  List of all possible n-ary folding actions";
    for (const std::unique_ptr<NaryFoldingAction>& folding :
         potential_folding_actions_to_perform) {
      VLOG(5) << "";
      VLOG(5) << "    To: " << folding->GetTo()->ToString();
      for (auto& [from_node, _] : folding->GetFrom()) {
        VLOG(5) << "      From: " << from_node->ToString();
      }
      VLOG(5) << "      Area savings = " << folding->area_saved();
    }
  }

  // Sort the list of n-ary folding actions to give priority to those that will
  // save more area
  TimingAnalysis ta{potential_folding_actions_to_perform, critical_path_delay};
  SortFoldingActionsInDescendingOrderOfTheirAreaSavings(
      potential_folding_actions_to_perform, ta);

  // Make the current sequence of n-ary folding legal.
  //
  // In more detail, at this point, every n-ary folding action is legal in
  // isolation.
  // However, a given n-ary folding action might be illegal if another one run
  // before.
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;
  XLS_ASSIGN_OR_RETURN(std::tie(folding_actions_to_perform, std::ignore),
                       ResourceSharingPass::LegalizeSequenceOfFolding(
                           std::move(potential_folding_actions_to_perform),
                           mutual_exclusivity, nda, &area_estimator, config));

  return folding_actions_to_perform;
}

// This function implements the heuristic that randomly selects the sub-set of
// legal folding actions to perform. This function is a profitability guard of
// the resource sharing optimization.
absl::StatusOr<std::vector<std::unique_ptr<NaryFoldingAction>>>
SelectRandomlyFoldingActions(
    FoldingGraph* folding_graph,
    absl::flat_hash_set<ResourceSharingPass::MutuallyExclPair>&
        mutual_exclusivity,
    const ResourceSharingPass::Config& config,
    const ResourceSharingPass::VisibilityAnalyses& visibility) {
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;

  // Get all edges of the folding graph
  std::vector<BinaryFoldingAction*> edges = folding_graph->GetEdges();

  // Compute the number of edges we will choose
  uint64_t target_edges = edges.size() * 0.1;

  // Create the PRVG that we will use to select the edges.
  absl::BitGen prvg;

  // Select the sub-set of the edges chosen randomly
  absl::flat_hash_map<Node*, std::vector<uint64_t>> indexes_of_selected_edges;
  for (uint64_t i = 0; i < target_edges; i++) {
    // Choose a new edge
    //
    // Because we want all edges to have equal probability to be chosen, we use
    // the uniform distribution for the PRVG.
    uint64_t index = absl::Uniform(prvg, 0u, edges.size());
    XLS_RET_CHECK_LT(index, edges.size());
    BinaryFoldingAction* edge = edges[index];

    // Keep track of the current edge
    indexes_of_selected_edges[edge->GetTo()].push_back(index);
  }

  // Merge chosen binary folding actions that have the same destination
  std::vector<BinaryFoldingAction*> folds;
  folds.reserve(indexes_of_selected_edges.size());
  for (auto& [destination, indexes] : indexes_of_selected_edges) {
    XLS_RET_CHECK_GT(indexes.size(), 0);

    double area_saved = 0.0;
    for (uint64_t index : indexes) {
      // Fetch the edge
      BinaryFoldingAction* edge = edges[index];
      if (!absl::c_all_of(folds, std::bind(CanFoldTogether, mutual_exclusivity,
                                           edge, std::placeholders::_1))) {
        continue;
      }
      folds.push_back(edge);
      area_saved += edge->area_saved();
    }
    XLS_RET_CHECK_NE(folds.size(), 0);

    // Create a single n-ary folding action
    std::unique_ptr<NaryFoldingAction> new_action;
    XLS_ASSIGN_OR_RETURN(new_action,
                         ResourceSharingPass::MakeNaryFoldingAction(
                             folds, area_saved, visibility, config));

    // Add the new n-ary folding action to the list of actions to perform
    folding_actions_to_perform.push_back(std::move(new_action));
  }

  return folding_actions_to_perform;
}

// This function chooses the subset of foldable actions to perform and decide
// their total order to perform them.
// This is part of the profitability guard of the resource sharing pass.
absl::StatusOr<std::vector<std::unique_ptr<NaryFoldingAction>>>
SelectFoldingActions(OptimizationContext& context, FoldingGraph* folding_graph,
                     ResourceSharingPass::ProfitabilityGuard heuristics,
                     absl::flat_hash_set<ResourceSharingPass::MutuallyExclPair>&
                         mutual_exclusivity,
                     const ResourceSharingPass::VisibilityAnalyses& visibility,
                     const NodeBackwardDependencyAnalysis& nda,
                     const AreaEstimator& area_estimator,
                     const CriticalPathDelayAnalysis& critical_path_delay,
                     const ResourceSharingPass::Config& config,
                     VisibilityEstimator* visibility_estimator) {
  std::vector<std::unique_ptr<NaryFoldingAction>> folding_actions_to_perform;
  VLOG(3) << "Choosing the best folding actions";

  // Decide the sub-set of legal folding actions to perform
  switch (heuristics) {
    case ResourceSharingPass::ProfitabilityGuard::kInDegree: {
      XLS_ASSIGN_OR_RETURN(
          folding_actions_to_perform,
          SelectFoldingActionsBasedOnInDegree(
              context, folding_graph, mutual_exclusivity, visibility, nda,
              area_estimator, critical_path_delay, config,
              visibility_estimator));
      break;
    }

    case ResourceSharingPass::ProfitabilityGuard::kCliques: {
      XLS_ASSIGN_OR_RETURN(folding_actions_to_perform,
                           SelectFoldingActionsBasedOnCliques(
                               folding_graph, visibility, config));
      break;
    }

    case ResourceSharingPass::ProfitabilityGuard::kRandom: {
      XLS_ASSIGN_OR_RETURN(
          folding_actions_to_perform,
          SelectRandomlyFoldingActions(folding_graph, mutual_exclusivity,
                                       config, visibility));
      break;
    }

    case ResourceSharingPass::ProfitabilityGuard::kAlways: {
      XLS_ASSIGN_OR_RETURN(
          folding_actions_to_perform,
          SelectAllFoldingActions(context, folding_graph, mutual_exclusivity,
                                  visibility, nda, area_estimator,
                                  critical_path_delay, config,
                                  visibility_estimator));
      break;
    }
  }

  TimingAnalysis ta{folding_actions_to_perform, critical_path_delay};
  std::vector<std::unique_ptr<NaryFoldingAction>>
      foldings_within_delay_increase;
  uint64_t total_delay_increase = 0;
  for (auto& folding : folding_actions_to_perform) {
    uint64_t delay_increase = ta.GetDelayIncrease(folding.get());
    if (total_delay_increase + delay_increase > config.max_delay_increase) {
      continue;
    }
    total_delay_increase += delay_increase;
    foldings_within_delay_increase.push_back(std::move(folding));
  }
  folding_actions_to_perform = std::move(foldings_within_delay_increase);

  // Print the folding actions we selected
  if (VLOG_IS_ON(2)) {
    VLOG(2) << "  We selected " << folding_actions_to_perform.size()
            << " folding actions to perform";
    for (const std::unique_ptr<NaryFoldingAction>& folding :
         folding_actions_to_perform) {
      VLOG(2) << "    To: " << folding->GetTo()->ToString();
      for (auto& [from_node, _] : folding->GetFrom()) {
        VLOG(2) << "      From: " << from_node->ToString();
      }
      std::optional<double> area_saved = folding->area_saved();
      if (area_saved.has_value()) {
        VLOG(2) << "      Area saved (estimate): " << *area_saved;
      }
      VLOG(2) << "      Time analysis = " << ta.GetDelaySpread(folding.get())
              << "," << ta.GetDelayIncrease(folding.get());
    }
  }

  return folding_actions_to_perform;
}

absl::StatusOr<bool> ResourceSharingPass::PerformFoldingActions(
    FunctionBase* f, int64_t next_node_id,
    VisibilityBuilder* visibility_builder,
    const std::vector<std::unique_ptr<NaryFoldingAction>>&
        folding_actions_to_perform) {
  bool modified = !folding_actions_to_perform.empty();
  VLOG(2) << "There are " << folding_actions_to_perform.size()
          << " folding actions to perform";

  absl::flat_hash_map<Node*, Node*> renaming_done_by_previous_folding;
  std::function<Node*(Node*)> final_renamed_node = [&](Node* initial) -> Node* {
    if (auto it = renaming_done_by_previous_folding.find(initial);
        it != renaming_done_by_previous_folding.end()) {
      return final_renamed_node(it->second);
    }
    return initial;
  };
  auto renamed_edge = [&](OperandVisibilityAnalysis::OperandNode edge)
      -> OperandVisibilityAnalysis::OperandNode {
    return OperandVisibilityAnalysis::OperandNode{
        /*operand=*/final_renamed_node(edge.operand),
        /*node=*/final_renamed_node(edge.node)};
  };

  // Perform the folding actions specified
  for (const std::unique_ptr<NaryFoldingAction>& folding :
       folding_actions_to_perform) {
    // Get the destination of the folding.
    // This might have been renamed by previous folding actions already
    // performed.
    Node* to_node = folding->GetTo();

    // Disable folding already folded sources / destinations because current
    // delay estimates are invalidated and do not take this into account.
    if (renaming_done_by_previous_folding.contains(to_node)) {
      continue;
    }
    bool renamed = false;
    for (const auto& from : folding->GetFrom()) {
      if (renaming_done_by_previous_folding.contains(from.first)) {
        renamed = true;
        break;
      }
    }
    if (renamed) {
      continue;
    }

    to_node = final_renamed_node(to_node);

    // Fetch the nodes to fold that have not been folded already
    std::vector<std::pair<Node*, FoldingAction::VisibilityEdges>> froms_to_use;
    for (auto& [from_node, from_edges] : folding->GetFrom()) {
      Node* renamed_node = final_renamed_node(from_node);
      absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>
          replaced_edges;
      for (auto edge : from_edges) {
        replaced_edges.insert(renamed_edge(edge));
      }

      // Register we can fold the current source to the destination of the
      // folding
      froms_to_use.push_back(std::make_pair(renamed_node, replaced_edges));

      // Keep track of the renaming of the nodes that we are about to perform
      // (from the sources to the destination).
      // This is to enable additional folding that involves the same nodes (only
      // if they are all related to the same select).
      renaming_done_by_previous_folding[renamed_node] = to_node;
    }
    XLS_RET_CHECK(!froms_to_use.empty());

    // Print the folding we are about to perform
    VLOG(2) << "  Next folding to perform:\n";
    VLOG(2) << "    To: " << to_node->ToString();
    for (auto& [from_node, _] : froms_to_use) {
      VLOG(2) << "      From: " << from_node->ToString();
    }

    // Fold
    //
    // - Step 0: Get the subset of the bits of the selector that are relevant.
    //
    // At the moment, we assume every source is selected by a single bit from
    // the selector of the select that made the sources and destination mutually
    // exclusive.
    VLOG(3) << "      Step 0: generate the new selector";
    std::vector<Node*> from_used_expressions;
    from_used_expressions.reserve(froms_to_use.size());
    for (const auto& [from_node, from_edges] : froms_to_use) {
      VLOG(4) << "        Source: " << from_node->ToString();
      XLS_ASSIGN_OR_RETURN(
          Node * from_used,
          visibility_builder->BuildVisibilityIRExpr(f, from_node, from_edges));
      from_used_expressions.push_back(from_used);
      VLOG(4) << "        From used: "
              << ToMathNotation(from_used, [&](const Node* n) -> bool {
                   return n->id() < next_node_id;
                 });
    }
    Node* new_selector = nullptr;
    if (from_used_expressions.size() > 1) {
      absl::c_reverse(from_used_expressions);
      XLS_ASSIGN_OR_RETURN(
          new_selector,
          f->MakeNode<Concat>(to_node->loc(), from_used_expressions));
    } else {
      new_selector = from_used_expressions.front();
    }
    XLS_RET_CHECK_NE(new_selector, nullptr);
    VLOG(3) << "        New selector: " << new_selector->ToString();

    // - Step 1: Create a new select for each input
    VLOG(3) << "      Step 1: generate the priority selects, one per input of "
               "the folding target";
    std::vector<Node*> new_operands;
    Op extension_op = folding->IsSigned() ? Op::kSignExt : Op::kZeroExt;
    for (uint32_t op_id = 0; op_id < to_node->operand_count(); op_id++) {
      VLOG(4) << "        Operand " << op_id;

      // Fetch the current operand for the target of the folding action.
      Node* to_operand = to_node->operand(op_id);
      int64_t to_operand_bitwidth = to_operand->BitCountOrDie();

      // Check if all sources have the same operand of the destination.
      // In this case, we do not to select which one to forward.
      bool we_need_to_select_operand = false;
      for (const auto& [from_node, _] : froms_to_use) {
        if (from_node->operand(op_id) != to_operand) {
          we_need_to_select_operand = true;
          break;
        }
      }
      if (!we_need_to_select_operand) {
        VLOG(4) << "          No need for a select node";

        // The current operand is in common between all sources and the
        // destination of the folding.
        // Hence, we do not need to select between them.
        new_operands.push_back(to_operand);
        continue;
      }

      // The current operand is different between the sources and the
      // destination of the folding. Hence, we need to select which one to
      // forward to the shared operation (i.e., the one performed by the
      // destination of the folding).
      //
      // Generate all select cases, one for each source of the folding action.
      std::vector<Node*> operand_select_cases;
      for (const auto& [from_node, _] : froms_to_use) {
        VLOG(4) << "          Source from " << from_node->ToString();

        // Fetch the operand of the current source of the folding action
        Node* from_operand = from_node->operand(op_id);
        XLS_RET_CHECK_LE(from_operand->BitCountOrDie(),
                         to_operand->BitCountOrDie())
            << "Illegal bit widths for folding: " << from_node->ToString()
            << " into: " << to_node->ToString();

        // Check if we need to negate it
        Node* from_operand_negated = from_operand;
        if ((to_node->op() != from_node->op()) &&
            to_node->OpIn({Op::kAdd, Op::kSub}) &&
            from_node->OpIn({Op::kAdd, Op::kSub}) && (op_id == 1)) {
          VLOG(4) << "            It needs to be negated";

          // Negate the input operand because
          //   sub(op0, op1)
          // needs to be mapped to
          //   add(op0, -op1)
          //
          //  or
          //   add(opX, opY)
          // needs to be mapped to
          //   sub(opX, -opY)
          //
          // Check if the negated value is already available
          if (from_operand->op() == Op::kNeg) {
            VLOG(4) << "            The negated value is already available";
            from_operand_negated = from_operand->operand(0);

          } else {
            VLOG(4) << "            Added the negated value";
            XLS_ASSIGN_OR_RETURN(
                from_operand_negated,
                f->MakeNode<UnOp>(to_node->loc(), from_operand_negated,
                                  Op::kNeg));
          }
        }

        // Check if we need to cast it
        Node* from_operand_casted = from_operand_negated;
        if (from_operand->BitCountOrDie() < to_operand->BitCountOrDie()) {
          VLOG(4) << "            It needs to be casted to "
                  << to_operand_bitwidth << " bits";

          // Cast the operand to the bit-width of the related operand of the
          // target of the folding action
          XLS_ASSIGN_OR_RETURN(
              from_operand_casted,
              f->MakeNode<ExtendOp>(to_node->loc(), from_operand,
                                    to_operand_bitwidth, extension_op));
        }

        // Append the current operand of the current source of the folding
        // action
        operand_select_cases.push_back(from_operand_casted);
      }

      // Generate a select between the sources of the folding
      XLS_ASSIGN_OR_RETURN(
          Node * operand_select,
          f->MakeNode<PrioritySelect>(to_node->loc(), new_selector,
                                      operand_select_cases, to_operand));
      new_operands.push_back(operand_select);
      VLOG(3) << "          " << operand_select->ToString();
    }
    XLS_RET_CHECK_EQ(new_operands.size(), 2);

    // - Step 2: Replace the operands of the @to_node to use the results of the
    //           new selectors computed at Step 1.
    VLOG(3) << "      Step 2: update the target of the folding transformation";
    for (int64_t op_id = int64_t{0}; op_id < to_node->operand_count();
         op_id++) {
      if (to_node->operand(op_id) == new_operands[op_id]) {
        continue;
      }
      XLS_RETURN_IF_ERROR(
          to_node->ReplaceOperandNumber(op_id, new_operands[op_id], true));
    }
    VLOG(3) << "        " << to_node->ToString();

    // - Step 3: Replace every source of the folding action with the new
    // @to_node
    VLOG(3)
        << "      Step 3: update the def-use chains to use the new folded node";
    for (const auto& [from_node, _] : froms_to_use) {
      XLS_RET_CHECK_LE(from_node->BitCountOrDie(), to_node->BitCountOrDie());

      // Check if we need to take a slice of the result
      Node* to_node_to_use = to_node;
      if (from_node->BitCountOrDie() < to_node->BitCountOrDie()) {
        // Take a slice of the result of the target of the folding action
        XLS_ASSIGN_OR_RETURN(to_node_to_use,
                             f->MakeNode<BitSlice>(from_node->loc(), to_node, 0,
                                                   from_node->BitCountOrDie()));
      }

      // Replace
      XLS_RETURN_IF_ERROR(from_node->ReplaceUsesWith(to_node_to_use));
    }

    // - Step 4: Remove all the sources of the folding action as they are now
    //           dead
    VLOG(3) << "      Step 4: remove the sources of the folding transformation";
    for (const auto& [from_node, _] : froms_to_use) {
      XLS_RETURN_IF_ERROR(f->RemoveNode(from_node));
    }
    VLOG(3) << "      Folding completed";
  }

  return modified;
}

// This function computes the resource sharing optimization for multiplication
// instructions. In more detail, this function folds a multiplication
// instruction into another multiplication instruction that has the same
// bit width for all operands as well as for the result.
//
// This folding operation is performed for all multiplication instructions that
// allow it (i.e., the transformation is legal).
absl::StatusOr<bool> ResourceSharingPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  // Check if the pass is enabled
  if (!options.enable_resource_sharing) {
    return false;
  }
  // Check if we have an area estimator
  if (options.area_estimator == nullptr) {
    return absl::InvalidArgumentError(
        "Enabling resource sharing requires an area estimator");
  }
  if (options.delay_estimator == nullptr) {
    return absl::InvalidArgumentError(
        "Enabling resource sharing requires a delay estimator");
  }
  bool modified = false;
  VLOG(2) << "Running resource sharing with the area model \""
          << options.area_estimator->name() << "\" and delay model \""
          << options.delay_estimator->name() << "\"";

  auto critical_path_delay =
      std::make_unique<CriticalPathDelayAnalysis>(options.delay_estimator);
  XLS_RETURN_IF_ERROR(critical_path_delay->Attach(f).status());

  NodeForwardDependencyAnalysis nda;
  NodeBackwardDependencyAnalysis nda_backwards;
  XLS_RETURN_IF_ERROR(nda.Attach(f).status());
  XLS_RETURN_IF_ERROR(nda_backwards.Attach(f).status());

  LazyPostDominatorAnalysis post_dom;
  XLS_RETURN_IF_ERROR(post_dom.Attach(f).status());

  // Run the BDD analysis
  std::unique_ptr<BddQueryEngine> bdd_engine = std::make_unique<BddQueryEngine>(
      config_.max_path_count_for_bdd_engine, IsCheapForBdds);
  XLS_RETURN_IF_ERROR(bdd_engine->Populate(f).status());

  XLS_ASSIGN_OR_RETURN(
      auto op_visibility,
      OperandVisibilityAnalysis::Create(
          config_.max_path_count_for_edge_in_general_visibility_analysis, &nda,
          bdd_engine.get()));
  XLS_ASSIGN_OR_RETURN(
      auto visibility,
      VisibilityAnalysis::Create(&op_visibility, bdd_engine.get(), &post_dom));

  XLS_ASSIGN_OR_RETURN(auto op_vis_large,
                       OperandVisibilityAnalysis::Create(
                           bdd_engine->path_limit(), &nda, bdd_engine.get()));
  XLS_ASSIGN_OR_RETURN(auto single_select_visibility,
                       SingleSelectVisibilityAnalysis::Create(
                           &op_vis_large, &nda, bdd_engine.get()));

  VisibilityAnalyses visibilities = {
      .general = *visibility,
      .single_select = *single_select_visibility,
  };

  int64_t next_node_id = 0;
  for (auto node : f->nodes()) {
    next_node_id = std::max(next_node_id, node->id());
  }
  next_node_id++;

  // Compute the mutually exclusive binary relation between IR instructions
  absl::flat_hash_set<MutuallyExclPair> mutual_exclusivity;
  XLS_ASSIGN_OR_RETURN(
      mutual_exclusivity,
      ComputeMutualExclusionAnalysis(f, context, visibilities, config_));

  BitProvenanceAnalysis bpa;
  VisibilityEstimator visibility_estimator(next_node_id - 1, bdd_engine.get(),
                                           nda, bpa, options.area_estimator,
                                           options.delay_estimator);

  // Identify the set of legal folding actions
  XLS_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<BinaryFoldingAction>> foldable_actions,
      ComputeFoldableActions(f, mutual_exclusivity, visibilities, config_));

  // Organize the folding actions into a graph
  FoldingGraph folding_graph{f, std::move(foldable_actions)};

  // Select the folding actions to perform
  ResourceSharingPass::ProfitabilityGuard selection_heuristic =
      options.force_resource_sharing ? ProfitabilityGuard::kAlways
                                     : profitability_guard_;
  XLS_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<NaryFoldingAction>>
          folding_actions_to_perform,
      SelectFoldingActions(context, &folding_graph, selection_heuristic,
                           mutual_exclusivity, visibilities, nda_backwards,
                           *options.area_estimator, *critical_path_delay,
                           config_, &visibility_estimator));

  // Perform the folding
  XLS_ASSIGN_OR_RETURN(
      modified, PerformFoldingActions(f, next_node_id, &visibility_estimator,
                                      folding_actions_to_perform));

  return modified;
}

TimingAnalysis::TimingAnalysis(
    const std::vector<std::unique_ptr<NaryFoldingAction>>& folding_actions,
    const CriticalPathDelayAnalysis& node_delay) {
  for (const std::unique_ptr<NaryFoldingAction>& folding : folding_actions) {
    // Get the information about the destination of the folding
    Node* to = folding->GetTo();
    int64_t to_delay = *node_delay.GetInfo(to);

    double delta_spread = 0.0;
    for (auto& [from, _] : folding->GetFrom()) {
      delta_spread += GetDelaySpreadBetweenNodes(from, to, node_delay);
    }
    delay_spread_[folding.get()] = delta_spread;

    // Estimate the delay increase for the destination of the folding
    int64_t min_delay = to_delay;
    int64_t max_delay = to_delay;
    for (auto& [from, _] : folding->GetFrom()) {
      min_delay = std::min(min_delay, *node_delay.GetInfo(from));
      max_delay = std::max(max_delay, *node_delay.GetInfo(from));
    }
    delay_increase_[folding.get()] = max_delay - min_delay;
  }
}

double TimingAnalysis::GetDelaySpread(NaryFoldingAction* folding_action) const {
  return delay_spread_.at(folding_action);
}

int64_t TimingAnalysis::GetDelayIncrease(
    NaryFoldingAction* folding_action) const {
  return delay_increase_.at(folding_action);
}

}  // namespace xls
