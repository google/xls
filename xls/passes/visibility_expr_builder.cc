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

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/binary_decision_diagram.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/bdd_query_engine.h"
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
  if (!selector || nda_.IsDependent(source, selector)) {
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
  return OrOperands(or_cases);
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
  return OrOperands(or_cases);
}

absl::StatusOr<Node*> VisibilityBuilder::GetVisibilityExprForAnd(
    Node* node, NaryOp* and_node, Node* source, FunctionBase* func) {
  std::vector<Node*> others_not_zero;
  for (Node* operand : and_node->operands()) {
    if (nda_.IsDependent(source, operand)) {
      continue;
    }

    Node* single_bit_value = operand;
    if (operand->op() == Op::kSignExt) {
      single_bit_value = operand->operand(0);
    }
    if (single_bit_value->GetType()->GetFlatBitCount() == 1) {
      XLS_ASSIGN_OR_RETURN(single_bit_value,
                           MakeParamIfTmpFunc(single_bit_value, func));
      others_not_zero.push_back(single_bit_value);
      continue;
    }

    XLS_ASSIGN_OR_RETURN(Node * compare_val, MakeParamIfTmpFunc(operand, func));
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
  return AndOperands(others_not_zero);
}

absl::StatusOr<Node*> VisibilityBuilder::GetVisibilityExprForOr(
    Node* node, NaryOp* or_node, Node* source, FunctionBase* func) {
  std::vector<Node*> others_not_ones;
  for (Node* operand : or_node->operands()) {
    if (nda_.IsDependent(source, operand)) {
      continue;
    }

    Node* single_bit_value = operand;
    if (operand->op() == Op::kSignExt) {
      single_bit_value = operand->operand(0);
    }
    if (single_bit_value->GetType()->GetFlatBitCount() == 1) {
      XLS_ASSIGN_OR_RETURN(single_bit_value,
                           MakeParamIfTmpFunc(single_bit_value, func));
      XLS_ASSIGN_OR_RETURN(Node * not_single_bit_value,
                           func->MakeNode<UnOp>(single_bit_value->loc(),
                                                single_bit_value, Op::kNot));
      others_not_ones.push_back(not_single_bit_value);
      continue;
    }

    XLS_ASSIGN_OR_RETURN(Node * compare_val, MakeParamIfTmpFunc(operand, func));
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
  return AndOperands(others_not_ones);
}

absl::StatusOr<Node*> VisibilityBuilder::GetVisibilityExprForPredicate(
    std::optional<Node*> predicate, Node* source, FunctionBase* func) {
  if (!predicate.has_value() || nda_.IsDependent(source, *predicate)) {
    return nullptr;
  }
  return MakeParamIfTmpFunc(*predicate, func);
}

absl::StatusOr<Node*> VisibilityBuilder::BuildVisibilityExprHelper(
    Node* node, Node* user, Node* source, FunctionBase* func) {
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
  } else if (user->OpIn({Op::kAnd, Op::kNand})) {
    return GetVisibilityExprForAnd(node, user->As<NaryOp>(), source, func);
  } else if (user->OpIn({Op::kOr, Op::kNor})) {
    return GetVisibilityExprForOr(node, user->As<NaryOp>(), source, func);
  }
  return nullptr;
}

// Builds predicate for node `u` being used by `v` on `func`.
absl::StatusOr<Node*> VisibilityBuilder::BuildVisibilityExpr(
    Node* node, Node* user, Node* source, FunctionBase* func) {
  auto cache_key = std::make_tuple(node, user, func);
  if (auto it = visibility_expr_cache_.find(cache_key);
      it != visibility_expr_cache_.end()) {
    return it->second;
  }
  XLS_ASSIGN_OR_RETURN(Node * visibility,
                       BuildVisibilityExprHelper(node, user, source, func));
  return visibility_expr_cache_[cache_key] = visibility;
}

absl::StatusOr<Node*> VisibilityBuilder::BuildNodeAndUserVisibleExpr(
    FunctionBase* func, Node* user_uses_node, Node* user_is_used,
    Literal* always_visible) {
  if (user_uses_node == always_visible) {
    return user_is_used;
  }
  if (user_is_used == always_visible) {
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
    Literal* always_visible) {
  if (node->users().empty()) {
    return always_visible;
  }
  if (auto it = node_to_visibility_ir_cache.find(node);
      it != node_to_visibility_ir_cache.end()) {
    return it->second;
  }

  absl::flat_hash_set<Node*> user_visibilities;
  for (Node* user : node->users()) {
    if (user->id() > prior_existing_id_) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(Node * user_is_used,
                         BuildVisibilityIRExprFromEdges(
                             func, user, source, conditional_edges,
                             node_to_visibility_ir_cache, always_visible));
    Node* user_uses_node = always_visible;
    if (conditional_edges.contains({node, user})) {
      XLS_ASSIGN_OR_RETURN(user_uses_node,
                           BuildVisibilityExpr(node, user, source, func));
      if (!user_uses_node) {
        return absl::InternalError(absl::StrCat(
            "conditional edge exists but visibility expression could NOT be "
            "made between ",
            node->GetName(), " and ", user->GetName()));
      }
    }
    if (user_uses_node == always_visible && user_is_used == always_visible) {
      return node_to_visibility_ir_cache[node] = always_visible;
    }

    Node* node_and_user_visible;
    if (user_uses_node == always_visible) {
      node_and_user_visible = user_is_used;
    } else if (user_is_used == always_visible) {
      node_and_user_visible = user_uses_node;
    } else {
      XLS_ASSIGN_OR_RETURN(
          node_and_user_visible,
          FindOrMakeBinaryNode(Op::kAnd, user_uses_node, user_is_used));
    }
    user_visibilities.insert(node_and_user_visible);
  }

  if (user_visibilities.empty()) {
    return node_to_visibility_ir_cache[node] = always_visible;
  }
  if (user_visibilities.size() == 1) {
    return node_to_visibility_ir_cache[node] = *user_visibilities.begin();
  }
  std::vector<Node*> user_vis_vec{user_visibilities.begin(),
                                  user_visibilities.end()};
  absl::c_sort(user_vis_vec,
               [&](Node* a, Node* b) { return a->id() < b->id(); });
  Node* any_user_visible = user_vis_vec[0];
  for (int64_t i = 1; i < user_vis_vec.size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        any_user_visible,
        FindOrMakeBinaryNode(Op::kOr, any_user_visible, user_vis_vec[i]));
  }
  return node_to_visibility_ir_cache[node] = any_user_visible;
}

absl::StatusOr<Node*> VisibilityBuilder::BuildVisibilityIRExpr(
    FunctionBase* func, Node* node,
    const absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>&
        conditional_edges) {
  XLS_ASSIGN_OR_RETURN(
      Literal * always_visible,
      func->MakeNode<Literal>(SourceInfo(), Value(UBits(1, 1))));
  if (conditional_edges.size() == 1) {
    XLS_ASSIGN_OR_RETURN(
        Node * user_uses_node,
        BuildVisibilityExpr(conditional_edges.begin()->operand,
                            conditional_edges.begin()->node, node, func));
    return user_uses_node ? user_uses_node : always_visible;
  }
  absl::flat_hash_map<Node*, Node*> node_to_visibility_ir_cache;
  absl::flat_hash_map<std::tuple<Op, Node*, Node*>, Node*> binary_op_cache;
  return BuildVisibilityIRExprFromEdges(func, node, node, conditional_edges,
                                        node_to_visibility_ir_cache,
                                        always_visible);
}

absl::StatusOr<VisibilityEstimator::AreaDelay>
VisibilityEstimator::GetAreaAndDelayOfVisibilityExpr(
    Node* node,
    const absl::flat_hash_set<OperandVisibilityAnalysis::OperandNode>&
        conditional_edges) {
  XLS_ASSIGN_OR_RETURN(
      Node * visibility_expr,
      BuildVisibilityIRExpr(TmpFunc(), node, conditional_edges));
  double area = area_analysis_.GetAreaThroughToNode(visibility_expr);
  int64_t delay = *delay_analysis_.GetInfo(visibility_expr);
  return VisibilityEstimator::AreaDelay{area, delay};
}

}  // namespace xls
