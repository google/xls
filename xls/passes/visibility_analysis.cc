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
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/passes/bdd_evaluator.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/lazy_node_data.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/query_engine.h"

namespace xls {

VisibilityAnalysis::VisibilityAnalysis(const NodeForwardDependencyAnalysis* nda,
                                       BddQueryEngine* bdd_query_engine)
    : nda_(nda),
      bdd_query_engine_(bdd_query_engine),
      edge_term_limit_(kDefaultTermLimitForNodeToUserEdge) {
  CHECK(bdd_query_engine_ != nullptr);
}

VisibilityAnalysis::VisibilityAnalysis(int64_t edge_term_limit,
                                       const NodeForwardDependencyAnalysis* nda,
                                       BddQueryEngine* bdd_query_engine)
    : nda_(nda),
      bdd_query_engine_(bdd_query_engine),
      edge_term_limit_(edge_term_limit) {
  CHECK(bdd_query_engine_ != nullptr);
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

BddNodeIndex VisibilityAnalysis::GetNodeBit(Node* node,
                                            uint32_t bit_index) const {
  return bdd_query_engine_
      ->GetBddNodeOrVariable(TreeBitLocation(node, bit_index))
      .value_or(bdd_query_engine_->bdd().NewVariable());
}

std::vector<BddNodeIndex> VisibilityAnalysis::GetNodeBits(Node* node) const {
  std::vector<BddNodeIndex> bits;
  bits.reserve(node->BitCountOrDie());
  for (int i = 0; i < node->BitCountOrDie(); ++i) {
    bits.push_back(GetNodeBit(node, i));
  }
  return bits;
}

std::vector<SaturatingBddNodeIndex> VisibilityAnalysis::GetSaturatingNodeBits(
    Node* node) const {
  std::vector<SaturatingBddNodeIndex> bits;
  bits.reserve(node->BitCountOrDie());
  for (int i = 0; i < node->BitCountOrDie(); ++i) {
    bits.push_back(GetNodeBit(node, i));
  }
  return bits;
}

bool VisibilityAnalysis::IsFullyUnconstrained(Node* node) const {
  if (node->Is<Param>()) {
    return false;
  }
  return bdd_query_engine_->IsFullyUnconstrained(node);
}

BddNodeIndex OrAggregate(absl::Span<const BddNodeIndex> operands,
                         BddQueryEngine* bdd_query_engine) {
  if (operands.empty()) {
    return BddNodeIndex(-1);
  }
  BddNodeIndex result = operands[0];
  for (int i = 1; i < operands.size(); ++i) {
    result = bdd_query_engine->bdd().Or(result, operands[i]);
  }
  return result;
}

// Removes too expensive terms, relying on the fact that a subset of the terms
// expresses a conservative and valid claim on visibility.
BddNodeIndex VisibilityAnalysis::AndAggregateSubsetThatFitsTermLimit(
    absl::Span<BddNodeIndex> operands) const {
  if (operands.empty()) {
    return BddNodeIndex(-1);
  }

  // Sort from least to most expensive to maximize the number of terms kept.
  absl::c_sort(operands, [&](BddNodeIndex a, BddNodeIndex b) {
    return bdd_query_engine_->bdd().path_count(a) <
           bdd_query_engine_->bdd().path_count(b);
  });

  BddNodeIndex result = operands[0];
  for (int i = 1; i < operands.size(); ++i) {
    BddNodeIndex next_result =
        bdd_query_engine_->bdd().And(result, operands[i]);
    if (bdd_query_engine_->bdd().path_count(next_result) > edge_term_limit_) {
      break;
    }
    result = next_result;
  }
  return result;
}

BddNodeIndex VisibilityAnalysis::ConditionOfUseWithPrioritySelect(
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

BddNodeIndex VisibilityAnalysis::ConditionOfUseWithSelect(
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

BddNodeIndex VisibilityAnalysis::ConditionOnPredicate(
    std::optional<Node*> predicate) const {
  if (predicate.has_value() && predicate.value()->BitCountOrDie() == 1) {
    return GetNodeBit(predicate.value(), 0);
  }
  return bdd_query_engine_->bdd().one();
}

BddNodeIndex VisibilityAnalysis::ConditionOfUseWithAnd(Node* node,
                                                       NaryOp* and_node) const {
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
  if (bdd_query_engine_->bdd().path_count(bitwise_aggregate) <=
      edge_term_limit_) {
    return bitwise_aggregate;
  }
  return AndAggregateSubsetThatFitsTermLimit(
      absl::MakeSpan(other_ops_not_zero));
}

BddNodeIndex VisibilityAnalysis::ConditionOfUseWithOr(Node* node,
                                                      NaryOp* or_node) const {
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
  if (bdd_query_engine_->bdd().path_count(bitwise_aggregate) <=
      edge_term_limit_) {
    return bitwise_aggregate;
  }
  return AndAggregateSubsetThatFitsTermLimit(
      absl::MakeSpan(other_ops_not_ones));
}

BddNodeIndex VisibilityAnalysis::ConditionOfUse(Node* node, Node* user) const {
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

BddNodeIndex VisibilityAnalysis::ComputeInfo(
    Node* node, absl::Span<const BddNodeIndex* const> user_infos) const {
  if (user_infos.empty()) {
    Node* predicate = TerminalPredicate(node);
    if (predicate && predicate->BitCountOrDie() == 1) {
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
    BddNodeIndex user_uses_node = ConditionOfUse(node, users[i]);
    if (bdd_query_engine_->bdd().path_count(user_uses_node) >
        edge_term_limit_) {
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
  if (info != bdd_query_engine_->bdd().zero() &&
      info != bdd_query_engine_->bdd().one()) {
    info = given;
  }
  return absl::OkStatus();
}

void VisibilityAnalysis::NodeAdded(Node* node) {
  LazyNodeData<BddNodeIndex>::NodeAdded(node);
  ClearCache();
}
void VisibilityAnalysis::NodeDeleted(Node* node) {
  LazyNodeData<BddNodeIndex>::NodeDeleted(node);
  ClearCache();
}

void VisibilityAnalysis::OperandChanged(Node* node, Node* old_operand,
                                        absl::Span<const int64_t> operand_nos) {
  LazyNodeData<BddNodeIndex>::OperandChanged(node, old_operand, operand_nos);
  ClearCache();
}

void VisibilityAnalysis::OperandRemoved(Node* node, Node* old_operand) {
  LazyNodeData<BddNodeIndex>::OperandRemoved(node, old_operand);
  ClearCache();
}

void VisibilityAnalysis::OperandAdded(Node* node) {
  LazyNodeData<BddNodeIndex>::OperandAdded(node);
  ClearCache();
}

}  // namespace xls
