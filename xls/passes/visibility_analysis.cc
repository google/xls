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

#include "xls/passes/visibility_analysis.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/reversed.hpp"
#include "cppitertools/zip.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/lazy_node_data.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/post_dominator_analysis.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/visibility_contracted_dag.h"

namespace xls {

/* static */ absl::StatusOr<NodeImpactOnVisibilityAnalysis>
NodeImpactOnVisibilityAnalysis::Create(FunctionBase* f) {
  NodeImpactOnVisibilityAnalysis node_impact;
  XLS_RETURN_IF_ERROR(node_impact.Attach(f).status());
  return node_impact;
}

// Returns the number of nodes in the function that have visibility expressions
// that are dependent on `node`. This amplifies impact of more granular mask and
// comparison operations, e.g. a bunch of individual equality comparisons vs. a
// single inequality.
int64_t NodeImpactOnVisibilityAnalysis::ComputeInfo(
    Node* node, absl::Span<const int64_t* const> user_infos) const {
  int64_t impact = 0;
  auto users = node->users();
  for (int i = 0; i < user_infos.size(); ++i) {
    Node* user = users[i];
    if (user->OpIn({Op::kBitSlice, Op::kDynamicBitSlice, Op::kConcat,
                    Op::kSignExt, Op::kZeroExt}) ||
        user->Is<CompareOp>()) {
      // Aggregate tally of user's impact
      impact += *user_infos[i];
    } else if (user->Is<Select>()) {
      auto select = user->As<Select>();
      if (select->selector() == node) {
        impact += select->cases().size() +
                  (select->default_value().has_value() ? 1 : 0);
      }
    } else if (user->Is<PrioritySelect>()) {
      auto select = user->As<PrioritySelect>();
      if (select->selector() == node) {
        impact += select->cases().size() + 1;
      }
    } else if (user->OpIn({Op::kAnd, Op::kOr, Op::kNand, Op::kNor})) {
      // Does not count the node itself towards impact
      impact += user->operands().size() - 1;
    } else if (absl::StatusOr<std::optional<Node*>> predicate =
                   GetPredicateUsedByNode(user);
               predicate.ok() && predicate.value_or(nullptr) == node) {
      impact += 1;
    } else if (user->Is<Gate>() && user->As<Gate>()->condition() == node) {
      impact += 1;
    }
  }
  return impact;
}

absl::Status NodeImpactOnVisibilityAnalysis::MergeWithGiven(
    int64_t& info, const int64_t& given) const {
  info = std::max(info, given);
  return absl::OkStatus();
}

/* static */ absl::StatusOr<OperandVisibilityAnalysis>
OperandVisibilityAnalysis::Create(const NodeForwardDependencyAnalysis* nda,
                                  const BddQueryEngine* bdd_query_engine) {
  return Create(kDefaultTermLimitForNodeToUserEdge, nda, bdd_query_engine);
}
/* static */ absl::StatusOr<OperandVisibilityAnalysis>
OperandVisibilityAnalysis::Create(int64_t edge_term_limit,
                                  const NodeForwardDependencyAnalysis* nda,
                                  const BddQueryEngine* bdd_query_engine) {
  FunctionBase* f = nda->bound_function();
  XLS_RET_CHECK_EQ(f, bdd_query_engine->info().bound_function());
  OperandVisibilityAnalysis op_vis(nda, bdd_query_engine, edge_term_limit);
  XLS_RETURN_IF_ERROR(op_vis.Attach(f).status());
  return op_vis;
}

OperandVisibilityAnalysis::~OperandVisibilityAnalysis() {
  if (f_ != nullptr) {
    f_->UnregisterChangeListener(this);
  }
  f_ = nullptr;
  pair_to_op_vis_ = {};
}

OperandVisibilityAnalysis::OperandVisibilityAnalysis(
    const NodeForwardDependencyAnalysis* nda,
    const BddQueryEngine* bdd_query_engine, int64_t edge_term_limit)
    : nda_(nda),
      bdd_query_engine_(bdd_query_engine),
      edge_term_limit_(edge_term_limit),
      pair_to_op_vis_(),
      f_(nullptr) {
  CHECK(bdd_query_engine_ != nullptr);
}

OperandVisibilityAnalysis::OperandVisibilityAnalysis(
    OperandVisibilityAnalysis&& other)
    : nda_(other.nda_),
      bdd_query_engine_(other.bdd_query_engine_),
      edge_term_limit_(other.edge_term_limit_),
      pair_to_op_vis_(std::move(other.pair_to_op_vis_)),
      f_(other.f_) {
  if (f_ != nullptr) {
    f_->RegisterChangeListener(this);
    f_->UnregisterChangeListener(&other);
  }
}

OperandVisibilityAnalysis& OperandVisibilityAnalysis::operator=(
    OperandVisibilityAnalysis&& other) {
  if (f_ != nullptr) {
    f_->UnregisterChangeListener(this);
  }
  f_ = other.f_;
  if (other.f_ != nullptr) {
    other.f_->UnregisterChangeListener(&other);
    other.f_ = nullptr;
    f_->RegisterChangeListener(this);
  }
  nda_ = other.nda_;
  bdd_query_engine_ = other.bdd_query_engine_;
  edge_term_limit_ = other.edge_term_limit_;
  pair_to_op_vis_ = std::move(other.pair_to_op_vis_);
  return *this;
}

/* static */ absl::StatusOr<std::unique_ptr<VisibilityAnalysis>>
VisibilityAnalysis::Create(const OperandVisibilityAnalysis* operand_vis,
                           const BddQueryEngine* bdd_query_engine,
                           const LazyPostDominatorAnalysis* post_dom_analysis,
                           int64_t max_edge_count_for_pruning,
                           absl::flat_hash_set<OperandNode> exclusions) {
  FunctionBase* f = operand_vis->bound_function();
  XLS_RET_CHECK_EQ(f, bdd_query_engine->info().bound_function());
  std::unique_ptr<VisibilityAnalysis> visibility =
      std::make_unique<VisibilityAnalysis>(
          operand_vis, bdd_query_engine, post_dom_analysis,
          max_edge_count_for_pruning, std::move(exclusions));
  XLS_RETURN_IF_ERROR(visibility->Attach(f).status());
  return std::move(visibility);
}

VisibilityAnalysis::VisibilityAnalysis(
    const OperandVisibilityAnalysis* operand_vis,
    const BddQueryEngine* bdd_query_engine,
    const LazyPostDominatorAnalysis* post_dom_analysis,
    int64_t max_edge_count_for_pruning,
    absl::flat_hash_set<OperandNode> exclusions)
    : LazyNodeData<NodeVisibility>(
          DagCacheInvalidateDirection::kInvalidatesBoth),
      operand_visibility_(operand_vis),
      bdd_query_engine_(bdd_query_engine),
      post_dom_analysis_(post_dom_analysis),
      node_impact_analysis_(),
      max_edge_count_for_pruning_(max_edge_count_for_pruning),
      exclusions_(exclusions) {
  CHECK(operand_visibility_ != nullptr);
  CHECK(bdd_query_engine_ != nullptr);
}

absl::StatusOr<ReachedFixpoint> VisibilityAnalysis::AttachWithGivens(
    FunctionBase* f, absl::flat_hash_map<Node*, NodeVisibility> givens) {
  XLS_ASSIGN_OR_RETURN(ReachedFixpoint rf, node_impact_analysis_.Attach(f));
  XLS_ASSIGN_OR_RETURN(
      ReachedFixpoint rf2,
      LazyNodeData<NodeVisibility>::AttachWithGivens(f, std::move(givens)));
  return rf == ReachedFixpoint::Changed || rf2 == ReachedFixpoint::Changed
             ? ReachedFixpoint::Changed
             : ReachedFixpoint::Unchanged;
}

absl::StatusOr<ReachedFixpoint> OperandVisibilityAnalysis::Attach(
    FunctionBase* f) {
  ReachedFixpoint rf = ReachedFixpoint::Unchanged;
  if (f_ != f) {
    if (f_ != nullptr) {
      f_->UnregisterChangeListener(this);
      pair_to_op_vis_.clear();
      rf = ReachedFixpoint::Changed;
    }

    if (f != nullptr) {
      f_ = f;
      f_->RegisterChangeListener(this);
      rf = ReachedFixpoint::Changed;
    }
  }
  return rf;
}

Node* TerminalPredicate(Node* node) {
  if (node->Is<Send>()) {
    return node->As<Send>()->predicate().value_or(nullptr);
  }
  if (node->Is<Next>()) {
    return node->As<Next>()->predicate().value_or(nullptr);
  }
  return nullptr;
}

BddNodeIndex OperandVisibilityAnalysis::GetNodeBit(Node* node,
                                                   uint32_t bit_index) const {
  std::optional<BddNodeIndex> bit =
      bdd_query_engine_->GetBddNodeOrVariable(TreeBitLocation(node, bit_index));
  return bit.has_value() ? *bit : bdd_query_engine_->bdd().NewVariable();
}

std::vector<BddNodeIndex> OperandVisibilityAnalysis::GetNodeBits(
    Node* node) const {
  std::vector<BddNodeIndex> bits;
  bits.reserve(node->BitCountOrDie());
  for (int i = 0; i < node->BitCountOrDie(); ++i) {
    bits.push_back(GetNodeBit(node, i));
  }
  return bits;
}

std::vector<SaturatingBddNodeIndex>
OperandVisibilityAnalysis::GetSaturatingNodeBits(Node* node) const {
  std::vector<SaturatingBddNodeIndex> bits;
  bits.reserve(node->BitCountOrDie());
  for (int i = 0; i < node->BitCountOrDie(); ++i) {
    bits.push_back(GetNodeBit(node, i));
  }
  return bits;
}

bool OperandVisibilityAnalysis::IsFullyUnconstrained(Node* node) const {
  if (node->Is<Param>()) {
    return false;
  }
  return bdd_query_engine_->IsFullyUnconstrained(node);
}

BddNodeIndex OrAggregate(absl::Span<const BddNodeIndex> operands,
                         const BddQueryEngine* bdd_query_engine) {
  if (operands.empty()) {
    return BinaryDecisionDiagram::kInfeasible;
  }
  BddNodeIndex result = operands[0];
  for (int i = 1; i < operands.size(); ++i) {
    if (operands[i] == BinaryDecisionDiagram::kInfeasible) {
      return BinaryDecisionDiagram::kInfeasible;
    }
    result = bdd_query_engine->bdd().Or(result, operands[i]);
  }
  return result;
}

// Removes too expensive terms, relying on the fact that a subset of the terms
// expresses a conservative and valid claim on visibility.
BddNodeIndex OperandVisibilityAnalysis::AndAggregateSubsetThatFitsTermLimit(
    absl::Span<BddNodeIndex> operands) const {
  if (operands.empty()) {
    return BinaryDecisionDiagram::kInfeasible;
  }

  // Sort from least to most expensive to maximize the number of terms kept.
  absl::c_sort(operands, [&](BddNodeIndex a, BddNodeIndex b) {
    return bdd_query_engine_->bdd().path_count(a) <
           bdd_query_engine_->bdd().path_count(b);
  });

  BddNodeIndex result = operands[0];
  for (int i = 1; i < operands.size(); ++i) {
    if (operands[i] == BinaryDecisionDiagram::kInfeasible) {
      continue;
    }
    BddNodeIndex next_result =
        bdd_query_engine_->bdd().And(result, operands[i]);
    if (bdd_query_engine_->bdd().path_count(next_result) > edge_term_limit_) {
      break;
    }
    result = next_result;
  }
  return result;
}

BddNodeIndex OperandVisibilityAnalysis::ConditionOfUseWithPrioritySelect(
    Node* node, PrioritySelect* select) const {
  BddNodeIndex always_used = bdd_query_engine_->bdd().one();
  Node* selector = select->selector();
  // If the selector uses the node, then the node is always used.
  if (nda_->IsDependent(node, selector)) {
    return always_used;
  }

  // Collect cases that use this node
  std::vector<BddNodeIndex> or_cases;
  absl::Span<Node* const> cases = select->cases();
  BddNodeIndex no_prev_case = bdd_query_engine_->bdd().one();
  for (int i = 0; i < cases.size(); ++i) {
    if (i > 0) {
      no_prev_case = bdd_query_engine_->bdd().And(
          no_prev_case,
          bdd_query_engine_->bdd().Not(GetNodeBit(selector, i - 1)));
    }
    if (cases[i] != node) {
      continue;
    }
    or_cases.push_back(
        bdd_query_engine_->bdd().And(no_prev_case, GetNodeBit(selector, i)));
  }

  // Collect AND of all not-ed selector bits if default uses node
  if (select->default_value() == node) {
    no_prev_case = bdd_query_engine_->bdd().And(
        no_prev_case,
        bdd_query_engine_->bdd().Not(GetNodeBit(selector, cases.size() - 1)));
    or_cases.push_back(no_prev_case);
  }

  return OrAggregate(or_cases, bdd_query_engine_);
}

BddNodeIndex OperandVisibilityAnalysis::ConditionOfUseWithSelect(
    Node* node, Select* select) const {
  auto& evaluator = bdd_query_engine_->evaluator();
  BddNodeIndex always_used = bdd_query_engine_->bdd().one();
  Node* selector = select->selector();
  // If the selector uses the node, then the node is always used.
  if (nda_->IsDependent(node, selector)) {
    return always_used;
  }

  std::vector<SaturatingBddNodeIndex> selector_bits =
      GetSaturatingNodeBits(selector);
  std::vector<BddNodeIndex> or_cases;
  absl::Span<Node* const> cases = select->cases();
  for (int i = 0; i < cases.size(); ++i) {
    if (cases[i] != node) {
      continue;
    }
    auto value_if_chose_case = UBits(i, selector->BitCountOrDie());
    SaturatingBddNodeIndex is_case = evaluator.Equals(
        selector_bits, evaluator.BitsToVector(value_if_chose_case));
    if (HasTooManyPaths(is_case)) {
      return always_used;
    }
    or_cases.push_back(ToBddNode(is_case));
  }

  if (select->default_value() == node) {
    auto value_gt_if_default =
        UBits(select->cases().size() - 1, selector->BitCountOrDie());
    SaturatingBddNodeIndex gt_all_cases = evaluator.UGreaterThan(
        selector_bits, evaluator.BitsToVector(value_gt_if_default));
    if (HasTooManyPaths(gt_all_cases)) {
      return always_used;
    }
    or_cases.push_back(ToBddNode(gt_all_cases));
  }

  return OrAggregate(or_cases, bdd_query_engine_);
}

BddNodeIndex OperandVisibilityAnalysis::ConditionOnNextUse(Next* next,
                                                           Node* node) const {
  return ConditionOnPredicate(node, next->predicate());
}

BddNodeIndex OperandVisibilityAnalysis::ConditionOnPredicate(
    Node* node, std::optional<Node*> predicate) const {
  // If there is no predicate, we conservatively assume the node is always used
  // by the predicated instruction. If the predicate uses the node, the node is
  // always used.
  if (!predicate.has_value() || nda_->IsDependent(node, *predicate)) {
    return bdd_query_engine_->bdd().one();
  }
  if (predicate.has_value() && predicate.value()->BitCountOrDie() == 1) {
    return GetNodeBit(predicate.value(), 0);
  }
  return bdd_query_engine_->bdd().one();
}

BddNodeIndex OperandVisibilityAnalysis::ConditionOfUseWithAnd(
    Node* node, NaryOp* and_node) const {
  std::vector<BddNodeIndex> other_ops_not_zero;
  std::vector<std::vector<BddNodeIndex>> bits_not_zero_exprs(
      and_node->BitCountOrDie(), std::vector<BddNodeIndex>{});
  for (Node* operand : and_node->operands()) {
    if (nda_->IsDependent(node, operand)) {
      continue;
    }

    // Avoid aggregating unconstrained nodes in the visibility expression even
    // if they are treated as variables which has a small path count (2 per var)
    if (IsFullyUnconstrained(operand)) {
      continue;
    }

    // Approach 1: breakdown by bit in an effort to prove visibility separately
    // for each bit. This is more specific but more likely to saturate when
    // or-aggregated with other operands.
    std::vector<BddNodeIndex> operand_bits = GetNodeBits(operand);
    for (int64_t bit_index = 0; bit_index < operand_bits.size(); ++bit_index) {
      bits_not_zero_exprs[bit_index].push_back(operand_bits[bit_index]);
    }

    // Approach 2: aggregate node's bits into one term. This is less specific
    // but less likely to saturate because we toss out operands that are too
    // complicated earlier on.
    BddNodeIndex is_not_zero = OrAggregate(operand_bits, bdd_query_engine_);
    if (HasTooManyPaths(is_not_zero)) {
      continue;
    }
    other_ops_not_zero.push_back(ToBddNode(is_not_zero));
  }

  std::vector<BddNodeIndex> bit_not_zero_aggregates(and_node->BitCountOrDie(),
                                                    BddNodeIndex(-1));
  for (int64_t i = 0; i < and_node->BitCountOrDie(); ++i) {
    bit_not_zero_aggregates[i] = AndAggregateSubsetThatFitsTermLimit(
        absl::MakeSpan(bits_not_zero_exprs[i]));
  }
  BddNodeIndex bitwise_aggregate =
      OrAggregate(bit_not_zero_aggregates, bdd_query_engine_);
  BddNodeIndex opwise_aggregate =
      AndAggregateSubsetThatFitsTermLimit(absl::MakeSpan(other_ops_not_zero));
  return bdd_query_engine_->bdd().path_count(bitwise_aggregate) <
                 bdd_query_engine_->bdd().path_count(opwise_aggregate)
             ? bitwise_aggregate
             : opwise_aggregate;
}

BddNodeIndex OperandVisibilityAnalysis::ConditionOfUseWithOr(
    Node* node, NaryOp* or_node) const {
  auto& bdd = bdd_query_engine_->bdd();
  auto& evaluator = bdd_query_engine_->evaluator();
  std::vector<BddNodeIndex> other_ops_not_ones;
  std::vector<std::vector<BddNodeIndex>> bits_not_ones_exprs(
      or_node->BitCountOrDie(), std::vector<BddNodeIndex>{});
  for (Node* operand : or_node->operands()) {
    if (nda_->IsDependent(node, operand)) {
      continue;
    }

    // Avoid aggregating unknowns in the visibility expression even if they are
    // treated as variables which has low path count
    if (IsFullyUnconstrained(operand)) {
      continue;
    }

    // Approach 1: breakdown by bit in an effort to prove visibility separately
    // for each bit. This is more specific but more likely to saturate when
    // or-aggregated with other operands.
    std::vector<BddNodeIndex> operand_bits = GetNodeBits(operand);
    for (int64_t bit_index = 0; bit_index < operand_bits.size(); ++bit_index) {
      bits_not_ones_exprs[bit_index].push_back(
          bdd.Not(operand_bits[bit_index]));
    }

    // Approach 2: aggregate node's bits into one term. This is less specific
    // but less likely to saturate because we toss out operands that are too
    // complicated earlier on.
    auto all_ones =
        evaluator.BitsToVector(Bits::AllOnes(operand->BitCountOrDie()));
    SaturatingBddNodeIndex is_not_ones = evaluator.Not(
        evaluator.Equals(GetSaturatingNodeBits(operand), all_ones));
    if (HasTooManyPaths(is_not_ones)) {
      continue;
    }
    other_ops_not_ones.push_back(ToBddNode(is_not_ones));
  }

  std::vector<BddNodeIndex> bit_not_ones_aggregates(or_node->BitCountOrDie(),
                                                    BddNodeIndex(-1));
  for (int64_t i = 0; i < or_node->BitCountOrDie(); ++i) {
    bit_not_ones_aggregates[i] = AndAggregateSubsetThatFitsTermLimit(
        absl::MakeSpan(bits_not_ones_exprs[i]));
  }
  BddNodeIndex bitwise_aggregate =
      OrAggregate(bit_not_ones_aggregates, bdd_query_engine_);
  BddNodeIndex opwise_aggregate =
      AndAggregateSubsetThatFitsTermLimit(absl::MakeSpan(other_ops_not_ones));
  return bdd_query_engine_->bdd().path_count(bitwise_aggregate) <
                 bdd_query_engine_->bdd().path_count(opwise_aggregate)
             ? bitwise_aggregate
             : opwise_aggregate;
}

BddNodeIndex OperandVisibilityAnalysis::ConditionOfUse(Node* node,
                                                       Node* user) const {
  if (user->Is<PrioritySelect>()) {
    return ConditionOfUseWithPrioritySelect(node, user->As<PrioritySelect>());
  } else if (user->Is<Select>()) {
    return ConditionOfUseWithSelect(node, user->As<Select>());
  } else if (user->Is<Send>()) {
    return ConditionOnPredicate(node, user->As<Send>()->predicate());
  } else if (user->Is<Next>()) {
    return ConditionOnNextUse(user->As<Next>(), node);
  } else if (user->Is<Gate>()) {
    return ConditionOnPredicate(node, user->As<Gate>()->condition());
  } else if (user->OpIn({Op::kAnd, Op::kNand})) {
    return ConditionOfUseWithAnd(node, user->As<NaryOp>());
  } else if (user->OpIn({Op::kOr, Op::kNor})) {
    return ConditionOfUseWithOr(node, user->As<NaryOp>());
  }

  // Conservatively assume the user always uses the node.
  return bdd_query_engine_->bdd().one();
}

BddNodeIndex OperandVisibilityAnalysis::OperandVisibilityThroughNode(
    Node* operand, Node* node) const {
  OperandNode cache_key{operand, node};
  return OperandVisibilityThroughNode(cache_key);
}

BddNodeIndex OperandVisibilityAnalysis::OperandVisibilityThroughNode(
    OperandNode& pair) const {
  if (auto it = pair_to_op_vis_.find(pair); it != pair_to_op_vis_.end()) {
    return it->second;
  }
  BddNodeIndex node_uses_operand = ConditionOfUse(pair.operand, pair.node);
  if (bdd_query_engine_->bdd().path_count(node_uses_operand) >
      edge_term_limit_) {
    node_uses_operand = bdd_query_engine_->bdd().one();
  }
  pair_to_op_vis_[pair] = node_uses_operand;
  return node_uses_operand;
}

void OperandVisibilityAnalysis::NodeAdded(Node* node) {
  // A new node has no users
}

void OperandVisibilityAnalysis::NodeDeleted(Node* node) {
  // A deleted node has no users
}

void OperandVisibilityAnalysis::OperandChanged(
    Node* node, Node* old_operand, absl::Span<const int64_t> operand_nos) {
  if (node->users().empty()) {
    for (auto operand : node->operands()) {
      pair_to_op_vis_.erase({operand, node});
    }
    return;
  }
  pair_to_op_vis_.clear();
}

void OperandVisibilityAnalysis::OperandRemoved(Node* node, Node* old_operand) {
  if (node->users().empty()) {
    for (auto operand : node->operands()) {
      pair_to_op_vis_.erase({operand, node});
    }
    return;
  }
  pair_to_op_vis_.clear();
}

void OperandVisibilityAnalysis::OperandAdded(Node* node) {
  if (node->users().empty()) {
    for (auto operand : node->operands()) {
      pair_to_op_vis_.erase({operand, node});
    }
    return;
  }
  pair_to_op_vis_.clear();
}

VisibilityAnalysis::SaturatingVisibility VisibilityAnalysis::ComputeAnyUsesNode(
    Node* node, absl::Span<const NodeVisibility* const> user_infos) const {
  BinaryDecisionDiagram& bdd = bdd_query_engine_->bdd();
  if (user_infos.empty()) {
    return {.visibility = bdd.one(), .saturated = false};
  }

  absl::Span<Node* const> users = node->users();
  std::vector<BddNodeIndex> user_conditions;
  user_conditions.reserve(users.size());
  for (const auto& [user, info] : iter::zip(users, user_infos)) {
    if (exclusions_.contains({node, user})) {
      user_conditions.push_back(info->visibility);
      continue;
    }
    BddNodeIndex user_uses_node =
        operand_visibility_->OperandVisibilityThroughNode(node, user);
    user_conditions.push_back(bdd.And(info->visibility, user_uses_node));
  }
  BddNodeIndex any_uses_node = OrAggregate(user_conditions, bdd_query_engine_);
  if (bdd.path_count(any_uses_node) > bdd_query_engine_->path_limit()) {
    return {.visibility = bdd.one(), .saturated = true};
  }
  return {.visibility = any_uses_node, .saturated = false};
}

VisibilityAnalysis::NodeVisibility VisibilityAnalysis::ComputeInfo(
    Node* node, absl::Span<const NodeVisibility* const> user_infos) const {
  NodeVisibility result;
  SaturatingVisibility any_uses_node = ComputeAnyUsesNode(node, user_infos);
  if (any_uses_node.saturated) {
    // Fall back to pruning edges on the contracted DAG.
    BddNodeIndex conservative_vis =
        ConservativeVisibilityByPruningEdges(node, result);
    if (conservative_vis != bdd_query_engine_->bdd().one()) {
      result.visibility = conservative_vis;
      return result;
    }
    // Fall back to the nearest post-dominator's visibility.
    result.visibility = VisibilityOfNearestPostDominator(node);
    return result;
  }
  result.visibility = any_uses_node.visibility;
  return result;
}

void VisibilityAnalysis::PopulateContractedVisibility(
    Node* node, const NodeVisibility& info) const {
  // If empty, construct the contracted DAG. If not empty, then it is up to date
  // because if node's data is invalidated, all of NodeVisibility is dropped.
  if (!info.contracted_nodes.empty()) {
    return;
  }
  info.edges.clear();
  std::queue<Node*> worklist;
  worklist.push(node);
  absl::flat_hash_set<Node*> visited = {node};
  while (!worklist.empty()) {
    Node* curr = worklist.front();
    worklist.pop();
    for (Node* user : curr->users()) {
      OperandNode edge{curr, user};
      if (!exclusions_.contains(edge) &&
          operand_visibility_->OperandVisibilityThroughNode(curr, user) !=
              bdd_query_engine_->bdd().one()) {
        info.edges.push_back(edge);
      }
      if (visited.insert(user).second) {
        worklist.push(user);
      }
    }
  }
  SortEdgesForPruning(absl::MakeSpan(info.edges));
  info.contracted_nodes = BuildContractedVisibilityDag(
      node, info.edges, [&](Node* operand, Node* user) {
        return operand_visibility_->OperandVisibilityThroughNode(operand, user);
      });
}

void VisibilityAnalysis::SortEdgesForPruning(
    absl::Span<OperandNode> edges) const {
  absl::c_sort(edges, [&](OperandNode a, OperandNode b) {
    BddNodeIndex a_vis =
        operand_visibility_->OperandVisibilityThroughNode(a.operand, a.node);
    BddNodeIndex b_vis =
        operand_visibility_->OperandVisibilityThroughNode(b.operand, b.node);
    std::optional<TreeBitLocation> a_bit =
        bdd_query_engine_->GetTreeBitLocation(a_vis);
    bool a_unconstrained =
        a_bit.has_value() &&
        operand_visibility_->IsFullyUnconstrained(a_bit->node());
    std::optional<TreeBitLocation> b_bit =
        bdd_query_engine_->GetTreeBitLocation(b_vis);
    bool b_unconstrained =
        b_bit.has_value() &&
        operand_visibility_->IsFullyUnconstrained(b_bit->node());
    // If both are unconstrained, prune the one less impactful to visibility.
    if (a_unconstrained && b_unconstrained) {
      int64_t a_impact =
          node_impact_analysis_.NodeImpactOnVisibility(a_bit->node());
      int64_t b_impact =
          node_impact_analysis_.NodeImpactOnVisibility(b_bit->node());
      if (a_impact != b_impact) {
        return a_impact < b_impact;
      }
    }
    // If one is constrained, prefer pruning the unconstrained one.
    if (a_unconstrained != b_unconstrained) {
      return a_unconstrained;
    }
    int64_t a_path = bdd_query_engine_->bdd().path_count(a_vis);
    int64_t b_path = bdd_query_engine_->bdd().path_count(b_vis);
    if (a_path != b_path) {
      return a_path > b_path;
    }
    return a < b;
  });
}

BddNodeIndex VisibilityAnalysis::PruneEdgesOnContractedDag(
    const NodeVisibility& info,
    absl::Span<const int32_t> sorted_candidate_edge_idxs,
    absl::Span<bool> is_excluded_edge) const {
  BinaryDecisionDiagram& bdd = bdd_query_engine_->bdd();
  SaturatingVisibility eval =
      ComputeInfoOnContractedDag(info, is_excluded_edge);
  if (eval.visibility != bdd.one() || !eval.saturated) {
    return eval.visibility;
  }
  absl::InlinedVector<bool, kEdgeInlineVecSize> is_excluded(
      is_excluded_edge.begin(), is_excluded_edge.end());
  for (int32_t expensive_edge_idx : sorted_candidate_edge_idxs) {
    is_excluded[expensive_edge_idx] = true;
    eval = ComputeInfoOnContractedDag(info, is_excluded);
    if (eval.visibility != bdd.one()) {
      absl::c_copy(is_excluded, is_excluded_edge.begin());
      return eval.visibility;
    }
    if (!eval.saturated) {
      return bdd.one();
    }
  }
  return bdd.one();
}

BddNodeIndex VisibilityAnalysis::ConservativeVisibilityByPruningEdges(
    Node* node, const NodeVisibility& info,
    absl::flat_hash_set<OperandNode> exclusions) const {
  PopulateContractedVisibility(node, info);
  exclusions.insert(exclusions_.begin(), exclusions_.end());
  absl::InlinedVector<bool, kEdgeInlineVecSize> is_excluded_edge(
      info.edges.size(), false);
  absl::InlinedVector<int32_t, kEdgeInlineVecSize> candidate_edge_idxs;
  candidate_edge_idxs.reserve(
      std::min<int64_t>(info.edges.size(), max_edge_count_for_pruning_));
  for (int32_t i = 0; i < static_cast<int32_t>(info.edges.size()); ++i) {
    if (exclusions.contains(info.edges[i])) {
      is_excluded_edge[i] = true;
    } else if (candidate_edge_idxs.size() < max_edge_count_for_pruning_) {
      candidate_edge_idxs.push_back(i);
    }
  }
  return PruneEdgesOnContractedDag(info, candidate_edge_idxs,
                                   absl::MakeSpan(is_excluded_edge));
}

VisibilityAnnotator VisibilityAnalysis::annotator() const {
  return VisibilityAnnotator(this);
}

Annotation VisibilityAnnotator::NodeAnnotation(Node* node) const {
  return Annotation{
      .suffix = absl::StrFormat("visible[%s]",
                                vis_->bdd_query_engine()->bdd().ToStringDnf(
                                    vis_->GetInfo(node)->visibility))};
}

BddNodeIndex VisibilityAnalysis::VisibilityOfNearestPostDominator(
    Node* node) const {
  // Find the nearest post dominator that constrains visibility. Post dominators
  // are sorted bottom up, so we start at the end of the list.
  auto post_doms = post_dom_analysis_->GetPostDominators(node);
  for (Node* post_dom : iter::reversed(post_doms)) {
    // Ignore the node itself; avoids recursing infinitely asking for visibility
    if (post_dom == node) {
      continue;
    }
    BddNodeIndex post_dom_visibility = GetInfo(post_dom)->visibility;
    if (post_dom_visibility != bdd_query_engine_->bdd().one()) {
      return post_dom_visibility;
    }
  }
  return bdd_query_engine_->bdd().one();
}

absl::Status VisibilityAnalysis::MergeWithGiven(
    NodeVisibility& info, const NodeVisibility& given) const {
  if (given.visibility != BinaryDecisionDiagram::kInfeasible) {
    info = given;
  }
  return absl::OkStatus();
}

bool VisibilityAnalysis::IsMutuallyExclusive(Node* one, Node* other) const {
  BinaryDecisionDiagram& bdd = bdd_query_engine_->bdd();
  return bdd.MutuallyExclusive(GetInfo(one)->visibility,
                               GetInfo(other)->visibility);
}

namespace {

absl::StatusOr<std::vector<Node*>> GetVisibilityControlConditions(
    const Node* operand, Node* node) {
  std::vector<Node*> conditions;
  if (auto gs = GenericSelect::TryFrom(node); gs.has_value()) {
    conditions.push_back(gs->selector());
  } else if (auto predicate = GetPredicateUsedByNode(node); predicate.ok()) {
    if (predicate->has_value()) {
      conditions.push_back(**predicate);
    }
  } else if (node->Is<Gate>()) {
    conditions.push_back(node->As<Gate>()->condition());
  } else if (node->OpIn({Op::kAnd, Op::kOr, Op::kNand, Op::kNor})) {
    for (Node* other_op : node->operands()) {
      if (other_op != operand) {
        conditions.push_back(other_op);
      }
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrFormat("Unsupported node type for visibility expression: %s",
                        node->ToString()));
  }
  return conditions;
}

}  // namespace

absl::StatusOr<bool> IsVisibilityIndependentOf(
    const NodeForwardDependencyAnalysis& nda, Node* operand, Node* node,
    absl::Span<Node* const> sources) {
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> conditions,
                       GetVisibilityControlConditions(operand, node));

  for (Node* condition : conditions) {
    for (Node* source : sources) {
      if (nda.IsDependent(source, condition)) {
        return false;
      }
    }
  }
  return true;
}

VisibilityAnalysis::SaturatingVisibility
VisibilityAnalysis::ComputeInfoOnContractedDag(
    const NodeVisibility& info, absl::Span<const bool> is_excluded_edge) const {
  BinaryDecisionDiagram& bdd = bdd_query_engine_->bdd();
  absl::InlinedVector<BddNodeIndex, kNodeInlineVecSize> cache(
      info.contracted_nodes.size());
  absl::InlinedVector<BddNodeIndex, kEdgeInlineVecSize> adj_conditions;
  absl::InlinedVector<BddNodeIndex, kNodeInlineVecSize> user_conditions;
  bool any_saturated = false;

  for (size_t i = 0; i < info.contracted_nodes.size(); ++i) {
    const ContractedDagNode& dag_node = info.contracted_nodes[i];
    if (dag_node.adjacent.empty() && dag_node.joined.empty()) {
      cache[i] = bdd.one();
      continue;
    }
    user_conditions.clear();
    if (!dag_node.adjacent.empty()) {
      adj_conditions.clear();
      for (const ContractedDagEdge& adj : dag_node.adjacent) {
        BddNodeIndex user_vis = cache[adj.user_idx];
        if (!is_excluded_edge[adj.edge_idx] && adj.edge_vis != bdd.one()) {
          user_vis = bdd.And(user_vis, adj.edge_vis);
        }
        adj_conditions.push_back(user_vis);
      }
      BddNodeIndex adj_vis = OrAggregate(adj_conditions, bdd_query_engine_);
      if (bdd.path_count(adj_vis) > bdd_query_engine_->path_limit()) {
        adj_vis = bdd.one();
        any_saturated = true;
      }
      user_conditions.push_back(adj_vis);
    }
    for (int32_t joined_idx : dag_node.joined) {
      user_conditions.push_back(cache[joined_idx]);
    }
    BddNodeIndex any_uses_node =
        OrAggregate(user_conditions, bdd_query_engine_);
    if (bdd.path_count(any_uses_node) > bdd_query_engine_->path_limit()) {
      any_saturated = true;
      any_uses_node = bdd.one();
    }
    cache[i] = any_uses_node;
  }
  return {.visibility = cache.back(), .saturated = any_saturated};
}

// Determines if a simplified visibility expression for `one` can still
// determine that none of `others` are visible while ensuring that if `one` is
// visible, the expression must be true.
bool VisibilityAnalysis::IsVisUsefulForMutualExclusivity(
    BddNodeIndex one_visible, BddNodeIndex one_visible_simplified,
    absl::Span<const BddNodeIndex> others_visible) const {
  BinaryDecisionDiagram& bdd = bdd_query_engine_->bdd();
  if (!bdd.DoesImply(one_visible, one_visible_simplified)) {
    return false;
  }
  for (BddNodeIndex other_visible : others_visible) {
    if (!bdd.DoesImply(one_visible_simplified, bdd.Not(other_visible))) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<absl::flat_hash_set<OperandNode>>
VisibilityAnalysis::GetEdgesForMutuallyExclusiveVisibilityExpr(
    Node* one, absl::Span<Node* const> others,
    int64_t max_edges_to_handle) const {
  const NodeVisibility& one_info = *GetInfo(one);
  BddNodeIndex one_visible = one_info.visibility;
  std::vector<BddNodeIndex> others_visible;
  others_visible.reserve(others.size());
  for (Node* other : others) {
    others_visible.push_back(GetInfo(other)->visibility);
  }
  return GetEdgesForMutuallyExclusiveVisibilityExpr(
      one, others, max_edges_to_handle, one_visible, others_visible,
      exclusions_);
}

absl::StatusOr<absl::flat_hash_set<OperandNode>>
VisibilityAnalysis::GetEdgesForMutuallyExclusiveVisibilityExpr(
    Node* one, absl::Span<Node* const> others, int64_t max_edges_to_handle,
    BddNodeIndex one_visible, absl::Span<const BddNodeIndex> others_visible,
    absl::flat_hash_set<OperandNode> exclusions) const {
  const NodeVisibility& one_info = *GetInfo(one);
  std::vector<Node*> sources;
  sources.reserve(others.size() + 1);
  sources.push_back(one);
  for (Node* other : others) {
    sources.push_back(other);
  }

  // This set contains edges that are not to be used in constructing the
  // visibility expression for `one` because of either:
  //   1. The edge was already excluded in this VisibilityAnalysis instance.
  //   2. The expression for the edge depends on a value from `others`; if this
  //      was used in resource sharing's foldings, it would produce a cycle.
  //   3. The edge is not needed to ensure the resulting visibility expression
  //      is true if `one` is visible and NOT true when any `other` is visible.
  PopulateContractedVisibility(one, one_info);
  exclusions.insert(exclusions_.begin(), exclusions_.end());

  const int32_t num_edges = static_cast<int32_t>(one_info.edges.size());
  absl::InlinedVector<bool, kEdgeInlineVecSize> is_excluded_edge(num_edges,
                                                                 false);
  absl::InlinedVector<int32_t, kEdgeInlineVecSize> candidate_edge_idxs;
  candidate_edge_idxs.reserve(num_edges);
  for (int32_t i = 0; i < num_edges; ++i) {
    const OperandNode& edge = one_info.edges[i];
    if (exclusions.contains(edge)) {
      is_excluded_edge[i] = true;
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        bool is_independent,
        IsVisibilityIndependentOf(operand_visibility_->nda(), edge.operand,
                                  edge.node, sources));
    if (is_independent) {
      candidate_edge_idxs.push_back(i);
    } else {
      // The visibility defined by this edge is dependent on a value from
      // `others`; we cannot use it in producing the visibility IR expression
      // for `one` without risking a cycle.
      is_excluded_edge[i] = true;
    }
  }

  // If there are more edges than the given threshold, drop all but the cheapest
  // `max_edges_to_handle` number of edges. `one_info.edges` is already sorted
  // in pruning order (more complex and unconstrained edges first).
  if (max_edges_to_handle >= 0 &&
      candidate_edge_idxs.size() > max_edges_to_handle) {
    int64_t num_to_drop = candidate_edge_idxs.size() - max_edges_to_handle;
    for (int64_t i = 0; i < num_to_drop; ++i) {
      is_excluded_edge[candidate_edge_idxs[i]] = true;
    }
    candidate_edge_idxs.erase(candidate_edge_idxs.begin(),
                              candidate_edge_idxs.begin() + num_to_drop);
  }

  // Ensure that with all required exclusions so far, the visibility expression
  // on the contracted DAG does not saturate and remains useful for proving
  // mutual exclusivity.
  BddNodeIndex baseline_vis = PruneEdgesOnContractedDag(
      one_info, candidate_edge_idxs, absl::MakeSpan(is_excluded_edge));
  if (!IsVisUsefulForMutualExclusivity(one_visible, baseline_vis,
                                       others_visible)) {
    return absl::flat_hash_set<OperandNode>{};
  }
  absl::erase_if(candidate_edge_idxs,
                 [&](int32_t edge_idx) { return is_excluded_edge[edge_idx]; });

  if (others.empty() || (others.size() == 1 && others[0] == one)) {
    absl::flat_hash_set<OperandNode> kept_edges;
    kept_edges.reserve(candidate_edge_idxs.size());
    for (int32_t idx : candidate_edge_idxs) {
      kept_edges.insert(one_info.edges[idx]);
    }
    return kept_edges;
  }

  absl::flat_hash_set<OperandNode> kept_edges;
  for (int32_t edge_idx : candidate_edge_idxs) {
    is_excluded_edge[edge_idx] = true;
    BddNodeIndex simplified_vis =
        ComputeInfoOnContractedDag(one_info, is_excluded_edge).visibility;
    if (IsVisUsefulForMutualExclusivity(one_visible, simplified_vis,
                                        others_visible)) {
      continue;
    }

    // The edge is needed to ensure the visibility expression is true if `one`
    // is visible and NOT true when any `other` is visible.
    is_excluded_edge[edge_idx] = false;
    kept_edges.insert(one_info.edges[edge_idx]);
  }
  return kept_edges;
}

absl::StatusOr<absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>>
VisibilityAnalysis::GetEdgesForConservativeVisibilityExpr(
    Node* one, absl::AnyInvocable<bool(Node*) const> is_live_source,
    int64_t max_edges_to_handle) const {
  absl::flat_hash_set<OperandNode> kept_edges;
  std::queue<Node*> worklist;
  worklist.push(one);
  absl::flat_hash_set<Node*> visited = {one};

  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop();

    for (Node* user : node->users()) {
      // Whether or not we want to keep this edge, we should add the user to
      // the worklist if it hasn't been visited yet.
      if (auto [_, inserted] = visited.insert(user); inserted) {
        worklist.push(user);
      }

      BddNodeIndex visibility =
          operand_visibility_->OperandVisibilityThroughNode(node, user);
      if (visibility == bdd_query_engine_->bdd().one()) {
        continue;
      }

      XLS_ASSIGN_OR_RETURN(std::vector<Node*> conditions,
                           GetVisibilityControlConditions(node, user));

      if (!conditions.empty() &&
          absl::c_none_of(conditions, [&](Node* condition) {
            return is_live_source(condition);
          })) {
        // There are conditions, but none of them are live; we have to consider
        // this edge always active.
        continue;
      }

      kept_edges.insert({node, user});
    }
  }

  if (max_edges_to_handle >= 0 && kept_edges.size() > max_edges_to_handle) {
    return absl::flat_hash_set<OperandNode>{};
  }
  return kept_edges;
}

/* static */ absl::StatusOr<std::unique_ptr<SingleSelectVisibilityAnalysis>>
SingleSelectVisibilityAnalysis::Create(
    const OperandVisibilityAnalysis* operand_vis,
    const NodeForwardDependencyAnalysis* nda,
    const BddQueryEngine* bdd_query_engine) {
  std::unique_ptr<SingleSelectVisibilityAnalysis> analysis =
      std::make_unique<SingleSelectVisibilityAnalysis>(operand_vis, nda,
                                                       bdd_query_engine);
  XLS_RETURN_IF_ERROR(analysis->Attach(operand_vis->bound_function()).status());
  return std::move(analysis);
}

SingleSelectVisibilityAnalysis::SingleSelectVisibilityAnalysis(
    const OperandVisibilityAnalysis* operand_vis,
    const NodeForwardDependencyAnalysis* nda,
    const BddQueryEngine* bdd_query_engine)
    : LazyNodeData<SingleSelectVisibility>(
          DagCacheInvalidateDirection::kInvalidatesBoth),
      operand_visibility_(operand_vis),
      nda_(nda),
      bdd_query_engine_(bdd_query_engine) {}

bool SingleSelectVisibilityAnalysis::IsMutuallyExclusive(Node* one,
                                                         Node* other) const {
  BinaryDecisionDiagram& bdd = bdd_query_engine_->bdd();
  const SingleSelectVisibility* one_info = GetInfo(one);
  const SingleSelectVisibility* other_info = GetInfo(other);
  if (!other_info->source || !one_info->source) {
    return false;
  }
  return bdd.MutuallyExclusive(one_info->visibility, other_info->visibility);
}

SingleSelectVisibility SingleSelectVisibilityAnalysis::ComputeInfo(
    Node* node,
    absl::Span<const SingleSelectVisibility* const> user_infos) const {
  PrioritySelect* single_select = nullptr;
  for (int i = 0; i < user_infos.size(); ++i) {
    if (user_infos[i]->select) {
      if (nda_->IsDependent(node, user_infos[i]->select->selector())) {
        continue;
      }
      single_select = user_infos[i]->select;
      break;
    }
  }
  absl::Span<Node* const> users = node->users();
  if (!single_select) {
    for (int i = 0; i < users.size(); ++i) {
      if (users[i]->Is<PrioritySelect>()) {
        auto select = users[i]->As<PrioritySelect>();
        if (nda_->IsDependent(node, select->selector())) {
          continue;
        }
        single_select = select;
        break;
      }
    }
  }
  // No user is a priority select or has a descendant that is.
  if (!single_select) {
    return SingleSelectVisibility();
  }

  BinaryDecisionDiagram& bdd = bdd_query_engine_->bdd();
  BddNodeIndex no_prev_case = bdd_query_engine_->bdd().one();
  BddNodeIndex source_visible = bdd.zero();
  Node* selector = single_select->selector();
  auto cases = single_select->cases();
  for (int i = 0; i < cases.size(); ++i) {
    if (i > 0) {
      no_prev_case =
          bdd.And(no_prev_case,
                  bdd.Not(operand_visibility_->GetNodeBit(selector, i - 1)));
    }
    if (nda_->IsDependent(node, cases[i])) {
      source_visible = bdd.Or(
          source_visible,
          bdd.And(no_prev_case, operand_visibility_->GetNodeBit(selector, i)));
    }
  }
  if (!cases.empty()) {
    no_prev_case = bdd.And(
        no_prev_case,
        bdd.Not(operand_visibility_->GetNodeBit(selector, cases.size() - 1)));
  }
  if (nda_->IsDependent(node, single_select->default_value())) {
    source_visible = bdd.Or(source_visible, no_prev_case);
  }
  SingleSelectVisibility single_select_vis(node, single_select, source_visible);

  // Now check that all other users are never visible if the node is not visible
  // to the select; if so, the select produces a sufficiently conservative
  // visibility expression.
  std::queue<OperandNode> worklist;
  absl::flat_hash_set<OperandNode> visited;
  for (int i = 0; i < users.size(); ++i) {
    worklist.push({node, users[i]});
    visited.insert({node, users[i]});
  }
  while (!worklist.empty()) {
    OperandNode curr_edge = worklist.front();
    worklist.pop();
    if (curr_edge.node == single_select_vis.select) {
      // This def-use chain does not escape the select.
      continue;
    }
    BddNodeIndex curr_edge_condition =
        operand_visibility_->OperandVisibilityThroughNode(curr_edge.operand,
                                                          curr_edge.node);
    // For the select edge to be representative, the source must be visible to
    // the select when it is visible on this edge.
    if (bdd.DoesImply(curr_edge_condition, single_select_vis.visibility)) {
      continue;
    }
    if (curr_edge.node->users().empty()) {
      // The user is terminal; there is a def-use path where visibility is not
      // constrained by the single select edge's visibility expression.
      return SingleSelectVisibility();
    }
    for (Node* user : curr_edge.node->users()) {
      OperandNode next_edge = {curr_edge.node, user};
      if (!visited.contains(next_edge)) {
        visited.insert(next_edge);
        worklist.push(next_edge);
      }
    }
  }
  return single_select_vis;
}

absl::Status SingleSelectVisibilityAnalysis::MergeWithGiven(
    SingleSelectVisibility& info, const SingleSelectVisibility& given) const {
  if (given.select) {
    info = given;
  }
  return absl::OkStatus();
}

absl::StatusOr<absl::flat_hash_set<OperandNode>>
SingleSelectVisibilityAnalysis::GetEdgesForVisibilityExpr(Node* one) const {
  absl::flat_hash_set<OperandNode> edges;
  auto info = GetInfo(one);
  for (auto select_case : info->select->cases()) {
    if (nda_->IsDependent(one, select_case)) {
      edges.insert({select_case, info->select});
    }
  }
  if (info->select->default_value() &&
      nda_->IsDependent(one, info->select->default_value())) {
    edges.insert({info->select->default_value(), info->select});
  }
  return edges;
}

void VisibilityAnalysis::NodeAdded(Node* node) {
  LazyNodeData<NodeVisibility>::NodeAdded(node);
  // On adding a node, normal invalidation is enough; dependency analysis does
  // not impact visibility by modifying a terminal node.
}
void VisibilityAnalysis::NodeDeleted(Node* node) {
  LazyNodeData<NodeVisibility>::NodeDeleted(node);
}

void VisibilityAnalysis::UserAdded(Node* node, Node* user) {
  LazyNodeData<NodeVisibility>::UserAdded(node, user);
  if (user->users().empty()) {
    // On modifying a terminal node, normal invalidation is enough; dependency
    // analysis does not impact visibility by modifying a terminal node.
    return;
  }
  ClearCache();
}

void VisibilityAnalysis::UserRemoved(Node* node, Node* user) {
  LazyNodeData<NodeVisibility>::UserRemoved(node, user);
  if (user->users().empty()) {
    // On modifying a terminal node, normal invalidation is enough; dependency
    // analysis does not impact visibility by modifying a terminal node.
    return;
  }
  ClearCache();
}

void SingleSelectVisibilityAnalysis::NodeAdded(Node* node) {
  LazyNodeData<SingleSelectVisibility>::NodeAdded(node);
}
void SingleSelectVisibilityAnalysis::NodeDeleted(Node* node) {
  LazyNodeData<SingleSelectVisibility>::NodeDeleted(node);
}

void SingleSelectVisibilityAnalysis::UserAdded(Node* node, Node* user) {
  LazyNodeData<SingleSelectVisibility>::UserAdded(node, user);
  if (user->users().empty()) {
    return;
  }
  ClearCache();
}

void SingleSelectVisibilityAnalysis::UserRemoved(Node* node, Node* user) {
  LazyNodeData<SingleSelectVisibility>::UserRemoved(node, user);
  if (user->users().empty()) {
    return;
  }
  ClearCache();
}

bool VisibilityDependsOn(
    const NodeForwardDependencyAnalysis& nda,
    const absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>&
        conditional_edges,
    Node* target) {
  const FunctionBase* f = nda.bound_function();
  if (!f->Contains(target)) {
    return false;
  }
  for (const auto& edge : conditional_edges) {
    Node* node = edge.node;
    Node* operand = edge.operand;
    if (!f->Contains(node)) {
      continue;
    }
    if (auto gs = GenericSelect::From(node); gs.ok()) {
      if (f->Contains(gs->selector()) &&
          nda.IsDependent(target, gs->selector())) {
        return true;
      }
    } else if (auto predicate = GetPredicateUsedByNode(node);
               predicate.ok() && predicate->has_value()) {
      if (f->Contains(**predicate) && nda.IsDependent(target, **predicate)) {
        return true;
      }
    } else if (node->OpIn({Op::kAnd, Op::kOr, Op::kNand, Op::kNor})) {
      for (Node* other_op : node->operands()) {
        if (other_op != operand && f->Contains(other_op) &&
            nda.IsDependent(target, other_op)) {
          return true;
        }
      }
    }
  }
  return false;
}
}  // namespace xls
