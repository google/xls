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

#include <cstdint>
#include <memory>
#include <optional>
#include <queue>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/lazy_dag_cache.h"
#include "xls/passes/lazy_node_data.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/query_engine.h"

namespace xls {

namespace {

using OperandNode = OperandVisibilityAnalysis::OperandNode;

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
  pair_to_op_vis_.clear();
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
  }
}

OperandVisibilityAnalysis& OperandVisibilityAnalysis::operator=(
    OperandVisibilityAnalysis&& other) {
  if (f_ != other.f_) {
    if (f_ != nullptr) {
      f_->UnregisterChangeListener(this);
    }
    f_ = other.f_;
    if (other.f_ != nullptr) {
      other.f_->UnregisterChangeListener(&other);
      other.f_ = nullptr;
      f_->RegisterChangeListener(this);
    }
  }
  nda_ = other.nda_;
  bdd_query_engine_ = other.bdd_query_engine_;
  edge_term_limit_ = other.edge_term_limit_;
  pair_to_op_vis_ = std::move(other.pair_to_op_vis_);
  return *this;
}

/* static */ absl::StatusOr<std::unique_ptr<VisibilityAnalysis>>
VisibilityAnalysis::Create(const OperandVisibilityAnalysis* operand_vis,
                           const BddQueryEngine* bdd_query_engine) {
  return VisibilityAnalysis::Create(operand_vis, bdd_query_engine, {});
}

/* static */ absl::StatusOr<std::unique_ptr<VisibilityAnalysis>>
VisibilityAnalysis::Create(const OperandVisibilityAnalysis* operand_vis,
                           const BddQueryEngine* bdd_query_engine,
                           absl::flat_hash_set<OperandNode> exclusions) {
  FunctionBase* f = operand_vis->bound_function();
  XLS_RET_CHECK_EQ(f, bdd_query_engine->info().bound_function());
  std::unique_ptr<VisibilityAnalysis> visibility =
      std::make_unique<VisibilityAnalysis>(operand_vis, bdd_query_engine,
                                           exclusions);
  XLS_RETURN_IF_ERROR(visibility->Attach(f).status());
  return std::move(visibility);
}

VisibilityAnalysis::VisibilityAnalysis(
    const OperandVisibilityAnalysis* operand_vis,
    const BddQueryEngine* bdd_query_engine)
    : VisibilityAnalysis(operand_vis, bdd_query_engine, {}) {}

VisibilityAnalysis::VisibilityAnalysis(
    const OperandVisibilityAnalysis* operand_vis,
    const BddQueryEngine* bdd_query_engine,
    absl::flat_hash_set<OperandNode> exclusions)
    : LazyNodeData<BddNodeIndex>(DagCacheInvalidateDirection::kInvalidatesBoth),
      operand_visibility_(operand_vis),
      bdd_query_engine_(bdd_query_engine),
      exclusions_(exclusions) {
  CHECK(operand_visibility_ != nullptr);
  CHECK(bdd_query_engine_ != nullptr);
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
  return bdd_query_engine_
      ->GetBddNodeOrVariable(TreeBitLocation(node, bit_index))
      .value_or(bdd_query_engine_->bdd().NewVariable());
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

BddNodeIndex OperandVisibilityAnalysis::ConditionOnPredicate(
    std::optional<Node*> predicate) const {
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
    return ConditionOnPredicate(user->As<Send>()->predicate());
  } else if (user->Is<Next>()) {
    return ConditionOnPredicate(user->As<Next>()->predicate());
  } else if (user->op() == Op::kAnd) {
    return ConditionOfUseWithAnd(node, user->As<NaryOp>());
  } else if (user->op() == Op::kOr) {
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

BddNodeIndex VisibilityAnalysis::ComputeInfo(
    Node* node, absl::Span<const BddNodeIndex* const> user_infos) const {
  if (user_infos.empty()) {
    if (Node* predicate = TerminalPredicate(node);
        predicate && predicate->BitCountOrDie() == 1) {
      TreeBitLocation predicate_bit(predicate, 0);
      auto bdd_node = bdd_query_engine_->GetBddNode(predicate_bit);
      if (bdd_node.has_value()) {
        return *bdd_node;
      }
    }
    return bdd_query_engine_->bdd().one();
  }

  absl::Span<Node* const> users = node->users();
  int64_t sum_of_user_path_counts = 0;
  for (int i = 0; i < users.size(); ++i) {
    sum_of_user_path_counts +=
        bdd_query_engine_->bdd().path_count(*user_infos[i]);
  }
  if (sum_of_user_path_counts > bdd_query_engine_->path_limit()) {
    return bdd_query_engine_->bdd().one();
  }

  std::vector<BddNodeIndex> user_conditions;
  for (int i = 0; i < users.size(); ++i) {
    if (exclusions_.contains({node, users[i]})) {
      user_conditions.push_back(*user_infos[i]);
      continue;
    }
    BddNodeIndex user_uses_node =
        operand_visibility_->OperandVisibilityThroughNode(node, users[i]);
    if (user_uses_node == bdd_query_engine_->bdd().one()) {
      user_conditions.push_back(*user_infos[i]);
      continue;
    }
    user_conditions.push_back(
        bdd_query_engine_->bdd().And(*user_infos[i], user_uses_node));
  }
  BddNodeIndex any_uses_node = OrAggregate(user_conditions, bdd_query_engine_);
  if (bdd_query_engine_->bdd().path_count(any_uses_node) >
      bdd_query_engine_->path_limit()) {
    return bdd_query_engine_->bdd().one();
  }
  return any_uses_node;
}

absl::Status VisibilityAnalysis::MergeWithGiven(
    BddNodeIndex& info, const BddNodeIndex& given) const {
  if (given != BinaryDecisionDiagram::kInfeasible) {
    info = given;
  }
  return absl::OkStatus();
}

void VisibilityAnalysis::NodeAdded(Node* node) {
  LazyNodeData<BddNodeIndex>::NodeAdded(node);
  // On adding a node, normal invalidation is enough; dependency analysis does
  // not impact visibility by modifying a terminal node.
}
void VisibilityAnalysis::NodeDeleted(Node* node) {
  LazyNodeData<BddNodeIndex>::NodeDeleted(node);
  // On removing a node, normal invalidation is enough; dependency analysis does
  // not impact visibility by modifying a terminal node.
}

void VisibilityAnalysis::UserAdded(Node* node, Node* user) {
  LazyNodeData<BddNodeIndex>::UserAdded(node, user);
  if (user->users().empty()) {
    // On modifying a terminal node, normal invalidation is enough; dependency
    // analysis does not impact visibility by modifying a terminal node.
    return;
  }
  ClearCache();
}

void VisibilityAnalysis::UserRemoved(Node* node, Node* user) {
  LazyNodeData<BddNodeIndex>::UserRemoved(node, user);
  if (user->users().empty()) {
    // On modifying a terminal node, normal invalidation is enough; dependency
    // analysis does not impact visibility by modifying a terminal node.
    return;
  }
  ClearCache();
}

bool OperandVisibilityAnalysis::IsVisibilityIndependentOf(
    Node* operand, Node* node, std::vector<Node*>& sources) const {
  std::vector<Node*> conditions;
  if (node->Is<PrioritySelect>()) {
    conditions.push_back(node->As<PrioritySelect>()->selector());
  } else if (node->Is<Select>()) {
    conditions.push_back(node->As<Select>()->selector());
  } else if (node->Is<Send>() && node->As<Send>()->predicate().has_value()) {
    conditions.push_back(*node->As<Send>()->predicate());
  } else if (node->Is<Next>() && node->As<Next>()->predicate().has_value()) {
    conditions.push_back(*node->As<Next>()->predicate());
  } else if (node->op() == Op::kAnd || node->op() == Op::kOr) {
    for (Node* other_op : node->operands()) {
      if (other_op != operand) {
        conditions.push_back(other_op);
      }
    }
  } else {
    return true;
  }

  for (Node* condition : conditions) {
    for (Node* source : sources) {
      if (nda_->IsDependent(source, condition)) {
        return false;
      }
    }
  }
  return true;
}

absl::StatusOr<absl::flat_hash_set<OperandNode>>
VisibilityAnalysis::GetEdgesForMutuallyExclusiveVisibilityExpr(
    Node* one, absl::Span<Node* const> others) const {
  std::vector<Node*> sources;
  sources.reserve(others.size() + 1);
  sources.push_back(one);
  for (Node* other : others) {
    sources.push_back(other);
  }

  // Populate edges by computing visibility
  BddNodeIndex one_visible = *GetInfo(one);
  std::queue<Node*> worklist;
  worklist.push(one);
  absl::flat_hash_set<Node*> visited = {one};
  std::vector<OperandNode> edges;
  while (!worklist.empty()) {
    Node* node = worklist.front();
    worklist.pop();
    for (Node* user : node->users()) {
      if (operand_visibility_->OperandVisibilityThroughNode(node, user) !=
          bdd_query_engine_->bdd().one()) {
        if (operand_visibility_->IsVisibilityIndependentOf(node, user,
                                                           sources)) {
          edges.push_back({node, user});
        }
      }
      if (visited.contains(user)) {
        continue;
      }
      visited.insert(user);
      worklist.push(user);
    }
  }

  BinaryDecisionDiagram& bdd = bdd_query_engine_->bdd();
  // Heuristic: prefer pruning more complex edges first to reduce the chance
  // they are disqualified later.
  absl::c_sort(edges, [&](OperandNode a, OperandNode b) {
    int64_t a_path =
        bdd.path_count(operand_visibility_->OperandVisibilityThroughNode(a));
    int64_t b_path =
        bdd.path_count(operand_visibility_->OperandVisibilityThroughNode(b));
    if (a_path == b_path) {
      return a < b;
    }
    return a_path > b_path;
  });

  absl::flat_hash_set<OperandNode> kept_edges;
  absl::flat_hash_set<OperandNode> exclusions;
  std::vector<BddNodeIndex> others_visible;
  others_visible.reserve(others.size());
  for (Node* other : others) {
    others_visible.push_back(*GetInfo(other));
  }

  for (auto& edge : edges) {
    exclusions.insert(edge);
    XLS_ASSIGN_OR_RETURN(
        auto simplified_vis,
        VisibilityAnalysis::Create(operand_visibility_, bdd_query_engine_,
                                   exclusions));
    BddNodeIndex vis_expr = *simplified_vis->GetInfo(one);
    if (bdd.Implies(one_visible, vis_expr) == bdd.one()) {
      bool implies_others_invisible = true;
      for (BddNodeIndex other_visible : others_visible) {
        if (bdd.Implies(vis_expr, bdd.Not(other_visible)) != bdd.one()) {
          implies_others_invisible = false;
          break;
        }
      }
      if (implies_others_invisible) {
        continue;
      }
    }

    // The edge is needed to ensure the visibility expression is true if 'one'
    // is visible and NOT true when any 'other' is visible.
    exclusions.erase(edge);
    kept_edges.insert(edge);
  }
  return kept_edges;
}

}  // namespace xls
