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

#include "xls/passes/visibility_expr_builder.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/algorithm.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/bit_provenance_analysis.h"
#include "xls/passes/node_dependency_analysis.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {

absl::StatusOr<Node*> VisibilityBuilder::GetSelectorIfIndependent(
    Node* node, Node* select, Node* source, FunctionBase* func) {
  Node* selector = nullptr;
  if (select->Is<Select>()) {
    selector = select->As<Select>()->selector();
  } else if (select->Is<PrioritySelect>()) {
    selector = select->As<PrioritySelect>()->selector();
  }
  if (!selector || nda_.IsDependent(source, selector) ||
      !is_live_source_(selector)) {
    return nullptr;
  }
  return MakeParamIfTmpFunc(selector, func);
}

bool VisibilityBuilder::DoesCaseImplyNoPrevCase(PrioritySelect* select,
                                                int64_t case_index) {
  if (case_index == 0) {
    return true;
  }
  if (case_index > kMaxCasesToCheckImplyNoPrevCase) {
    return false;
  }
  BinaryDecisionDiagram& bdd = bdd_engine_->bdd();
  Node* selector = select->selector();
  auto case_bit =
      bdd_engine_->GetBddNode(TreeBitLocation(selector, case_index));
  if (!case_bit.has_value()) {
    return false;
  }
  for (int64_t i = 0; i < case_index; ++i) {
    auto prev_case_bit = bdd_engine_->GetBddNode(TreeBitLocation(selector, i));
    if (!prev_case_bit.has_value()) {
      return false;
    }
    if (bdd.Implies(*case_bit, bdd.Not(*prev_case_bit)) != bdd.one()) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<Node*> VisibilityBuilder::GetVisibilityExprForPrioritySelect(
    Node* node, PrioritySelect* select, Node* source, FunctionBase* func) {
  XLS_ASSIGN_OR_RETURN(Node * selector,
                       GetSelectorIfIndependent(node, select, source, func));
  if (!selector) {
    return nullptr;
  }
  absl::Span<Node* const> cases = select->cases();
  std::vector<Node*> or_cases;
  bool is_concat_on_single_bits =
      selector->Is<Concat>() &&
      selector->As<Concat>()->operands().size() == cases.size();
  for (int64_t i = 0; i < cases.size(); ++i) {
    if (node == cases[i]) {
      Node* selector_bit_i = nullptr;
      bool no_prev_case_can_be_true = DoesCaseImplyNoPrevCase(select, i);
      if (no_prev_case_can_be_true) {
        if (is_concat_on_single_bits) {
          Concat* concat = selector->As<Concat>();
          selector_bit_i = concat->operand(concat->operand_count() - i - 1);
        } else {
          XLS_ASSIGN_OR_RETURN(selector_bit_i,
                               FindOrMakeBitSlice(selector, i, 1));
        }
      } else {
        XLS_ASSIGN_OR_RETURN(Node * bits_to_i,
                             FindOrMakeBitSlice(selector, 0, i + 1));
        XLS_ASSIGN_OR_RETURN(
            Node * one, func->MakeNode<Literal>(select->loc(),
                                                Value(UBits(1 << i, i + 1))));
        XLS_ASSIGN_OR_RETURN(
            selector_bit_i,
            func->MakeNode<CompareOp>(select->loc(), bits_to_i, one, Op::kEq));
      }
      or_cases.push_back(selector_bit_i);
    }
  }
  if (node == select->default_value()) {
    XLS_ASSIGN_OR_RETURN(
        Node * zero,
        func->MakeNode<Literal>(
            select->loc(),
            Value(UBits(0, selector->GetType()->GetFlatBitCount()))));
    XLS_ASSIGN_OR_RETURN(
        Node * selector_is_zero,
        func->MakeNode<CompareOp>(select->loc(), selector, zero, Op::kEq));
    or_cases.push_back(selector_is_zero);
  }

  XLS_ASSIGN_OR_RETURN(Node * result, OrOperands(or_cases));
  if (get_remaining_delay_(result) < 0) {
    // This would push the condition over the acceptable delay.
    return nullptr;
  }
  return result;
}

absl::StatusOr<Node*> VisibilityBuilder::GetVisibilityExprForSelect(
    Node* node, Select* select, Node* source, FunctionBase* func) {
  XLS_ASSIGN_OR_RETURN(Node * selector,
                       GetSelectorIfIndependent(node, select, source, func));
  if (!selector) {
    return nullptr;
  }
  absl::Span<Node* const> cases = select->cases();
  std::vector<Node*> or_cases;
  for (int64_t i = 0; i < cases.size(); ++i) {
    if (node == cases[i]) {
      XLS_ASSIGN_OR_RETURN(
          Node * value_i,
          func->MakeNode<Literal>(
              select->loc(),
              Value(UBits(i, selector->GetType()->GetFlatBitCount()))));
      XLS_ASSIGN_OR_RETURN(
          Node * is_case,
          func->MakeNode<CompareOp>(select->loc(), selector, value_i, Op::kEq));
      or_cases.push_back(is_case);
    }
  }
  if (node == select->default_value()) {
    XLS_ASSIGN_OR_RETURN(
        Node * value_num_cases,
        func->MakeNode<Literal>(
            select->loc(),
            Value(
                UBits(cases.size(), selector->GetType()->GetFlatBitCount()))));
    XLS_ASSIGN_OR_RETURN(Node * is_default,
                         func->MakeNode<CompareOp>(select->loc(), selector,
                                                   value_num_cases, Op::kUGe));
    or_cases.push_back(is_default);
  }
  XLS_ASSIGN_OR_RETURN(Node * result, OrOperands(or_cases));
  if (get_remaining_delay_(result) < 0) {
    // This would push the condition over the acceptable delay.
    return nullptr;
  }
  return result;
}

// Find the source bits of operand, and if unknown, use the operand itself.
absl::StatusOr<Node*> VisibilityBuilder::GetNonRepeatedSourceOf(
    Node* operand, FunctionBase* func) {
  XLS_ASSIGN_OR_RETURN(auto bit_sources_tree, bpa_.GetBitSources(operand));
  LeafTypeTree<TreeBitSources> trimmed_bit_sources_tree =
      BitProvenanceAnalysis::TrimRepeatedSourceBits(
          std::move(bit_sources_tree));
  const auto& source_ranges = trimmed_bit_sources_tree.Get({}).ranges();
  if (source_ranges.size() == 1) {
    const TreeBitSources::BitRange& single_range = source_ranges[0];
    Node* source = single_range.source_node();
    if (!is_live_source_(source)) {
      return operand;
    }
    // Clone the source node if building expressions in a temp function.
    XLS_ASSIGN_OR_RETURN(source, MakeParamIfTmpFunc(source, func));
    // If derived from a single range of contiguous bits, find or create a bit
    // slice (skip this if the range covers the entire source node)
    if (single_range.source_bit_index_low() != 0 ||
        source->GetType()->GetFlatBitCount() != single_range.bit_width()) {
      XLS_ASSIGN_OR_RETURN(
          source,
          FindOrMakeBitSlice(source, single_range.source_bit_index_low(),
                             single_range.bit_width()));
    }
    return source;
  }

  // Default to the operand itself without further knowledge
  // NOTE: We could concat together the multiple source ranges found instead of
  // defaulting to the operand itself, but it isn't clear if this would help.
  // Clone the operand if building expressions in a temp function.
  return MakeParamIfTmpFunc(operand, func);
}

namespace {

absl::StatusOr<Node*> FilterByDelay(
    FunctionBase* func, const SourceInfo& loc, std::vector<Node*>& operands,
    const std::function<int64_t(Node*)>& get_remaining_delay,
    const std::function<absl::StatusOr<Node*>(const SourceInfo&)>& build_empty,
    const std::function<absl::StatusOr<Node*>(std::vector<Node*>&)>& build_fn) {
  XLS_ASSIGN_OR_RETURN(Node * result, build_fn(operands));
  if (get_remaining_delay(result) < 0) {
    SortByKey(operands, get_remaining_delay, std::greater<>{});
    while (!operands.empty() && get_remaining_delay(result) < 0) {
      operands.pop_back();
      if (operands.empty()) {
        XLS_ASSIGN_OR_RETURN(result, build_empty(loc));
        break;
      }
      XLS_ASSIGN_OR_RETURN(result, build_fn(operands));
    }
  }
  return result;
}

}  // namespace

absl::StatusOr<Node*> VisibilityBuilder::GetVisibilityExprForAnd(
    Node* node, NaryOp* and_node, Node* source, FunctionBase* func,
    Literal* always_visible) {
  std::vector<Node*> others_not_zero;
  for (Node* operand : and_node->operands()) {
    if (nda_.IsDependent(source, operand)) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(Node * compare_val,
                         GetNonRepeatedSourceOf(operand, func));
    if (!is_live_source_(compare_val)) {
      // Skipping this non-live source will result in a conservative
      // visibility expression; as we're building an AND of these not being
      // zero, dropping it will only ever make the expression more often true.
      continue;
    }

    if (compare_val->GetType()->GetFlatBitCount() == 1) {
      others_not_zero.push_back(compare_val);
      continue;
    }

    XLS_ASSIGN_OR_RETURN(
        Node * value_zero,
        func->MakeNode<Literal>(
            and_node->loc(),
            Value(UBits(0, compare_val->GetType()->GetFlatBitCount()))));
    XLS_ASSIGN_OR_RETURN(Node * is_not_zero,
                         func->MakeNode<CompareOp>(and_node->loc(), compare_val,
                                                   value_zero, Op::kNe));
    others_not_zero.push_back(is_not_zero);
  }

  if (others_not_zero.empty()) {
    return AndOperands(others_not_zero);
  }
  if (others_not_zero.size() == 1) {
    return others_not_zero[0];
  }

  return FilterByDelay(
      func, and_node->loc(), others_not_zero, get_remaining_delay_,
      /*build_empty=*/
      [&](const SourceInfo& loc) -> absl::StatusOr<Node*> {
        return always_visible;
      },
      /*build_fn=*/
      [&](std::vector<Node*>& ops) -> absl::StatusOr<Node*> {
        return AndOperands(ops);
      });
}

absl::StatusOr<Node*> VisibilityBuilder::GetVisibilityExprForOr(
    Node* node, NaryOp* or_node, Node* source, FunctionBase* func,
    Literal* always_visible) {
  std::vector<Node*> others_not_ones;
  for (Node* operand : or_node->operands()) {
    if (nda_.IsDependent(source, operand)) {
      continue;
    }

    XLS_ASSIGN_OR_RETURN(Node * compare_val,
                         GetNonRepeatedSourceOf(operand, func));
    if (!is_live_source_(compare_val)) {
      // Skipping this non-live source will result in a conservative
      // visibility expression; as we're building an AND of these not being
      // all-ones, dropping it will only ever make the expression more often
      // true.
      continue;
    }

    if (compare_val->GetType()->GetFlatBitCount() == 1) {
      XLS_ASSIGN_OR_RETURN(
          Node * not_single_bit_value,
          func->MakeNode<UnOp>(compare_val->loc(), compare_val, Op::kNot));
      others_not_ones.push_back(not_single_bit_value);
      continue;
    }

    XLS_ASSIGN_OR_RETURN(
        Node * value_ones,
        func->MakeNode<Literal>(
            or_node->loc(),
            Value(Bits::AllOnes(compare_val->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(Node * is_not_ones,
                         func->MakeNode<CompareOp>(or_node->loc(), compare_val,
                                                   value_ones, Op::kNe));
    others_not_ones.push_back(is_not_ones);
  }

  if (others_not_ones.empty()) {
    return AndOperands(others_not_ones);
  }

  return FilterByDelay(
      func, or_node->loc(), others_not_ones, get_remaining_delay_,
      /*build_empty=*/
      [&](const SourceInfo& loc) -> absl::StatusOr<Node*> {
        return always_visible;
      },
      /*build_fn=*/
      [&](std::vector<Node*>& ops) -> absl::StatusOr<Node*> {
        return AndOperands(ops);
      });
}

absl::StatusOr<Node*> VisibilityBuilder::GetVisibilityExprForPredicate(
    std::optional<Node*> predicate, Node* source, FunctionBase* func) {
  if (!predicate.has_value() || nda_.IsDependent(source, *predicate) ||
      !is_live_source_(*predicate)) {
    return nullptr;
  }
  return MakeParamIfTmpFunc(*predicate, func);
}

absl::StatusOr<Node*> VisibilityBuilder::BuildVisibilityExprHelper(
    Node* node, Node* user, Node* source, FunctionBase* func,
    Literal* always_visible) {
  if (user->Is<Select>()) {
    return GetVisibilityExprForSelect(node, user->As<Select>(), source, func);
  } else if (user->Is<PrioritySelect>()) {
    return GetVisibilityExprForPrioritySelect(node, user->As<PrioritySelect>(),
                                              source, func);
  } else if (user->Is<Send>()) {
    return GetVisibilityExprForPredicate(user->As<Send>()->predicate(), source,
                                         func);
  } else if (user->Is<Next>()) {
    return GetVisibilityExprForPredicate(user->As<Next>()->predicate(), source,
                                         func);
  } else if (user->Is<Gate>()) {
    return GetVisibilityExprForPredicate(user->As<Gate>()->condition(), source,
                                         func);
  } else if (user->OpIn({Op::kAnd, Op::kNand})) {
    return GetVisibilityExprForAnd(node, user->As<NaryOp>(), source, func,
                                   always_visible);
  } else if (user->OpIn({Op::kOr, Op::kNor})) {
    return GetVisibilityExprForOr(node, user->As<NaryOp>(), source, func,
                                  always_visible);
  }
  return nullptr;
}

// Builds predicate for node `u` being used by `v` on `func`.
absl::StatusOr<Node*> VisibilityBuilder::BuildVisibilityExpr(
    Node* node, Node* user, Node* source, FunctionBase* func,
    Literal* always_visible) {
  auto cache_key = std::make_tuple(node, user, func);
  if (auto it = visibility_expr_cache_.find(cache_key);
      it != visibility_expr_cache_.end()) {
    return it->second;
  }
  XLS_ASSIGN_OR_RETURN(
      Node * visibility,
      BuildVisibilityExprHelper(node, user, source, func, always_visible));
  return visibility_expr_cache_[cache_key] = visibility;
}

absl::StatusOr<Node*> VisibilityBuilder::BuildNodeAndUserVisibleExpr(
    FunctionBase* func, Node* user_uses_node, Node* user_is_used,
    Literal* always_visible) {
  if (IsLiteralAllOnes(user_uses_node)) {
    return user_is_used;
  }
  if (IsLiteralAllOnes(user_is_used)) {
    return user_uses_node;
  }
  XLS_ASSIGN_OR_RETURN(
      Node * and_node,
      func->MakeNode<NaryOp>(user_uses_node->loc(),
                             std::vector<Node*>{user_uses_node, user_is_used},
                             Op::kAnd));
  return and_node;
}

absl::StatusOr<Node*> VisibilityBuilder::BuildVisibilityIRExprFromEdges(
    FunctionBase* func, Node* node, Node* source,
    const absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>&
        conditional_edges,
    absl::flat_hash_map<Node*, Node*>& node_to_visibility_ir_cache,
    Literal* always_visible, const absl::flat_hash_set<Node*>& sinks) {
  if (!is_live_source_(node)) {
    return always_visible;
  }
  if (node->users().empty() || sinks.contains(node)) {
    return always_visible;
  }
  if (auto it = node_to_visibility_ir_cache.find(node);
      it != node_to_visibility_ir_cache.end()) {
    return it->second;
  }

  absl::btree_set<Node*, Node::NodeIdLessThan> user_visibilities;
  for (Node* user : node->users()) {
    if (user->id() > prior_existing_id_) {
      continue;
    }
    bool sink_depends_on_user = absl::c_any_of(
        sinks, [&](Node* sink) { return nda_.IsDependent(user, sink); });
    if (!sinks.empty() && !sink_depends_on_user) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        Node * user_is_used,
        BuildVisibilityIRExprFromEdges(func, user, source, conditional_edges,
                                       node_to_visibility_ir_cache,
                                       always_visible, sinks));
    Node* user_uses_node = always_visible;
    if (conditional_edges.contains({node, user})) {
      XLS_ASSIGN_OR_RETURN(
          user_uses_node,
          BuildVisibilityExpr(node, user, source, func, always_visible));
      if (!user_uses_node) {
        user_uses_node = always_visible;
      }
    }
    if (IsLiteralAllOnes(user_uses_node) && IsLiteralAllOnes(user_is_used)) {
      return node_to_visibility_ir_cache[node] = always_visible;
    }

    std::vector<Node*> and_operands;
    if (!IsLiteralAllOnes(user_uses_node)) {
      if (user_uses_node->op() == Op::kAnd) {
        and_operands.insert(and_operands.end(),
                            user_uses_node->operands().begin(),
                            user_uses_node->operands().end());
      } else {
        and_operands.push_back(user_uses_node);
      }
    }
    if (!IsLiteralAllOnes(user_is_used)) {
      if (user_is_used->op() == Op::kAnd) {
        and_operands.insert(and_operands.end(),
                            user_is_used->operands().begin(),
                            user_is_used->operands().end());
      } else {
        and_operands.push_back(user_is_used);
      }
    }

    Node* node_and_user_visible;
    if (and_operands.empty()) {
      node_and_user_visible = always_visible;
    } else if (and_operands.size() == 1) {
      node_and_user_visible = and_operands[0];
    } else {
      XLS_ASSIGN_OR_RETURN(
          node_and_user_visible,
          func->MakeNode<NaryOp>(SourceInfo(), and_operands, Op::kAnd));
      if (get_remaining_delay_(node_and_user_visible) < 0) {
        SortByKey(and_operands, get_remaining_delay_, std::greater<>{});
        while (!and_operands.empty() &&
               get_remaining_delay_(node_and_user_visible) < 0) {
          and_operands.pop_back();
          if (and_operands.empty()) {
            node_and_user_visible = always_visible;
            break;
          }
          if (and_operands.size() == 1) {
            node_and_user_visible = and_operands[0];
          } else {
            XLS_ASSIGN_OR_RETURN(
                node_and_user_visible,
                func->MakeNode<NaryOp>(SourceInfo(), and_operands, Op::kAnd));
          }
        }
      }
    }
    user_visibilities.insert(node_and_user_visible);
  }

  if (user_visibilities.empty()) {
    return node_to_visibility_ir_cache[node] = always_visible;
  }
  if (user_visibilities.size() == 1) {
    return node_to_visibility_ir_cache[node] = *user_visibilities.begin();
  }
  std::vector<Node*> user_vis_vec;
  for (Node* n : user_visibilities) {
    if (n->op() == Op::kOr) {
      user_vis_vec.insert(user_vis_vec.end(), n->operands().begin(),
                          n->operands().end());
    } else {
      user_vis_vec.push_back(n);
    }
  }

  Node* any_user_visible;
  if (absl::c_any_of(user_vis_vec,
                     [&](Node* n) { return IsLiteralAllOnes(n); })) {
    any_user_visible = always_visible;
  } else {
    XLS_ASSIGN_OR_RETURN(
        any_user_visible,
        NaryOrIfNeeded(func, user_vis_vec, /*name=*/"", SourceInfo(),
                       /*drop_literal_zero_operands=*/true));
    if (get_remaining_delay_(any_user_visible) < 0) {
      any_user_visible = always_visible;
    }
  }
  return node_to_visibility_ir_cache[node] = any_user_visible;
}

absl::StatusOr<Node*> VisibilityBuilder::BuildVisibilityIRExpr(
    FunctionBase* func, Node* node,
    const absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>&
        conditional_edges,
    const absl::flat_hash_set<Node*>& sinks,
    std::optional<int64_t> target_stage) {
  int64_t max_id_before = 0;
  if (target_stage.has_value()) {
    for (Node* n : func->nodes()) {
      max_id_before = std::max(max_id_before, n->id());
    }
  }

  XLS_ASSIGN_OR_RETURN(
      Literal * always_visible,
      func->MakeNode<Literal>(SourceInfo(), Value(UBits(1, 1))));

  absl::flat_hash_map<Node*, Node*> node_to_visibility_ir_cache;
  absl::flat_hash_map<std::tuple<Op, Node*, Node*>, Node*> binary_op_cache;
  XLS_ASSIGN_OR_RETURN(Node * result_expr,
                       BuildVisibilityIRExprFromEdges(
                           func, node, node, conditional_edges,
                           node_to_visibility_ir_cache, always_visible, sinks));

  if (target_stage.has_value()) {
    for (Node* n : func->nodes()) {
      if (n->id() > max_id_before) {
        XLS_RETURN_IF_ERROR(func->AddNodeToStage(*target_stage, n).status());
      }
    }
  }

  return result_expr;
}

absl::Status VisibilityBuilder::CleanUpUnusedNodes(FunctionBase* fb) {
  std::vector<Node*> worklist;
  absl::flat_hash_set<Node*> dead_nodes;

  for (Node* n : fb->nodes()) {
    if (n->id() > prior_existing_id_ && n->IsDead()) {
      dead_nodes.insert(n);
      worklist.push_back(n);
    }
  }

  while (!worklist.empty()) {
    Node* n = worklist.back();
    worklist.pop_back();

    std::vector<Node*> operands(n->operands().begin(), n->operands().end());

    CHECK_GT(dead_nodes.erase(n), 0);
    XLS_RETURN_IF_ERROR(fb->RemoveNode(n));

    for (Node* operand : operands) {
      if (operand->id() > prior_existing_id_ && operand->IsDead()) {
        if (auto [_, inserted] = dead_nodes.insert(operand); inserted) {
          worklist.push_back(operand);
        }
      }
    }
  }

  return absl::OkStatus();
}

absl::StatusOr<VisibilityEstimator::AreaDelay>
VisibilityEstimator::GetAreaAndDelayOfVisibilityExpr(
    Node* node,
    const absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>&
        conditional_edges) {
  XLS_ASSIGN_OR_RETURN(Node * visibility_expr,
                       BuildVisibilityIRExpr(TmpFunc(), node, conditional_edges,
                                             /*sinks=*/{}));
  double area = area_analysis_.GetAreaThroughToNode(visibility_expr);
  int64_t delay = *delay_analysis_.GetInfo(visibility_expr);
  return VisibilityEstimator::AreaDelay{area, delay};
}

}  // namespace xls
