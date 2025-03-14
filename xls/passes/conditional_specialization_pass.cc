// Copyright 2021 The XLS Authors
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

#include "xls/passes/conditional_specialization_pass.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/module_initializer.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/transitive_closure.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/change_listener.h"
#include "xls/ir/function_base.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/source_location.h"
#include "xls/ir/ternary.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/bdd_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/pass_pipeline.pb.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"
#include "xls/passes/visibility_analysis.h"

namespace xls {
namespace {

// Returns the value for node logically implied by the given conditions if a
// value can be implied. Returns std::nullopt otherwise.
std::optional<Bits> ImpliedNodeValue(const ConditionSet& condition_set,
                                     Node* node,
                                     const QueryEngine& query_engine) {
  if (!node->GetType()->IsBits()) {
    return std::nullopt;
  }
  if (condition_set.impossible()) {
    VLOG(4) << absl::StreamFormat(
        "%s is impossible, so %s can be anything; we chose zero",
        condition_set.ToString(), node->GetName());
    return ZeroOfType(node->GetType()).bits();
  }

  if (std::optional<Condition> condition = condition_set.condition(node);
      condition.has_value() && condition->partial.Ternary().has_value() &&
      ternary_ops::IsFullyKnown(*condition->partial.Ternary())) {
    VLOG(4) << absl::StreamFormat("%s trivially implies %s==%s",
                                  condition_set.ToString(), node->GetName(),
                                  xls::ToString(*condition->partial.Ternary()));
    return ternary_ops::ToKnownBitsValues(*condition->partial.Ternary());
  }

  std::vector<std::pair<TreeBitLocation, bool>> predicates =
      condition_set.GetPredicates();
  std::optional<Bits> implied_value =
      query_engine.ImpliedNodeValue(predicates, node);

  if (implied_value.has_value()) {
    VLOG(4) << absl::StreamFormat("%s implies %s==%v", condition_set.ToString(),
                                  node->GetName(), implied_value.value());
  }
  return implied_value;
}

// Returns the value for node logically implied by the given conditions if a
// value can be implied. Returns std::nullopt otherwise.
std::optional<TernaryVector> ImpliedNodeTernary(
    const ConditionSet& condition_set, Node* node,
    const QueryEngine& query_engine) {
  if (!node->GetType()->IsBits()) {
    return std::nullopt;
  }
  if (condition_set.impossible()) {
    // This is impossible, as the conditions contradict each other. For
    // now, we can't do anything about this; it might be worth finding a
    // way to propagate this information.
    VLOG(1) << "This condition is impossible: " << condition_set.ToString();
    return std::nullopt;
  }

  PartialInformation partial =
      PartialInformation::Unconstrained(node->BitCountOrDie());
  std::optional<Condition> condition = condition_set.condition(node);
  if (condition.has_value() && !condition->partial.IsUnconstrained()) {
    VLOG(4) << absl::StreamFormat("%s trivially implies %s==%s",
                                  condition_set.ToString(), node->GetName(),
                                  condition->partial.ToString());
    partial = condition->partial;
  }
  if (partial.Ternary().has_value() &&
      ternary_ops::IsFullyKnown(*partial.Ternary())) {
    return std::move(partial).Ternary();
  }

  std::vector<std::pair<TreeBitLocation, bool>> predicates =
      condition_set.GetPredicates();
  std::optional<TernaryVector> implied_ternary =
      query_engine.ImpliedNodeTernary(predicates, node);
  if (implied_ternary.has_value()) {
    VLOG(4) << absl::StreamFormat("%s implies %s==%s", condition_set.ToString(),
                                  node->GetName(),
                                  xls::ToString(*implied_ternary));
    partial.JoinWith(PartialInformation(*implied_ternary));
    if (partial.IsImpossible()) {
      // This is impossible, as the conditions contradict each other. For
      // now, we can't do anything about this; it might be worth finding a
      // way to propagate this information.
      VLOG(1) << "Proved this condition is impossible: "
              << condition_set.ToString();
      return std::nullopt;
    }
  }

  return std::move(partial).Ternary();
}

// Returns the case arm node of the given select which is selected when the
// selector has the given value.
Node* GetSelectedCase(Select* select, const Bits& selector_value) {
  if (bits_ops::UGreaterThanOrEqual(selector_value, select->cases().size())) {
    return select->default_value().value();
  }
  // It is safe to convert to uint64_t because of the above check against
  // cases size.
  return select->get_case(selector_value.ToUint64().value());
}

std::optional<Node*> GetSelectedCase(PrioritySelect* select,
                                     const TernaryVector& selector_value) {
  for (int64_t i = 0; i < select->cases().size(); ++i) {
    if (selector_value[i] == TernaryValue::kUnknown) {
      // We can't be sure which case is selected.
      return std::nullopt;
    }
    if (selector_value[i] == TernaryValue::kKnownOne) {
      return select->get_case(i);
    }
  }
  // All bits of the selector are zero.
  return select->default_value();
}

struct ZeroValue : std::monostate {};
std::optional<std::variant<Node*, ZeroValue>> GetSelectedCase(
    OneHotSelect* ohs, const TernaryVector& selector_value) {
  if (!ternary_ops::IsFullyKnown(selector_value)) {
    // We can't be sure which case is selected.
    return std::nullopt;
  }
  Bits selector_bits = ternary_ops::ToKnownBitsValues(selector_value);
  if (selector_bits.PopCount() > 1) {
    // We aren't selecting just one state.
    return std::nullopt;
  }
  for (int64_t i = 0; i < selector_value.size(); ++i) {
    if (selector_bits.Get(i)) {
      return ohs->get_case(i);
    }
  }
  // All bits of the selector are zero.
  return ZeroValue{};
}

absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> AffectedBy(
    FunctionBase* f, OptimizationContext& context) {
  absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>> affected_by;
  for (Node* node : context.TopoSort(f)) {
    for (Node* operand : node->operands()) {
      affected_by[operand].insert(node);
    }
  }
  return TransitiveClosure(affected_by);
}

absl::StatusOr<std::optional<Node*>> CheckMatch(Node* node,
                                                PartialInformation partial,
                                                Node* user) {
  if (partial.IsImpossible()) {
    // Matching is impossible. Return a literal 0.
    return node->function_base()->MakeNode<Literal>(user->loc(),
                                                    Value(UBits(0, 1)));
  }
  if (partial.IsUnconstrained()) {
    return std::nullopt;
  }
  if (std::optional<Bits> precise_value = partial.GetPreciseValue();
      precise_value.has_value()) {
    if (precise_value->bit_count() == 1) {
      if (precise_value->IsOne()) {
        return node;
      } else {
        XLS_ASSIGN_OR_RETURN(Node * negated,
                             node->function_base()->MakeNode<UnOp>(
                                 SourceInfo(), node, Op::kNot));
        return negated;
      }
    }
    XLS_ASSIGN_OR_RETURN(Node * matched_value,
                         node->function_base()->MakeNode<Literal>(
                             SourceInfo(), Value(*precise_value)));
    XLS_ASSIGN_OR_RETURN(Node * matched_value_check,
                         node->function_base()->MakeNode<CompareOp>(
                             SourceInfo(), node, matched_value, Op::kEq));
    return matched_value_check;
  }
  if (std::optional<Bits> punctured_value = partial.GetPuncturedValue();
      punctured_value.has_value()) {
    XLS_ASSIGN_OR_RETURN(Node * punctured_literal,
                         node->function_base()->MakeNode<Literal>(
                             SourceInfo(), Value(*punctured_value)));
    XLS_ASSIGN_OR_RETURN(Node * punctured_value_check,
                         node->function_base()->MakeNode<CompareOp>(
                             SourceInfo(), node, punctured_literal, Op::kNe));
    return punctured_value_check;
  }

  bool should_match_ternary =
      partial.BitCount() > 0 && partial.Ternary().has_value();
  bool should_match_range =
      partial.BitCount() > 0 && partial.Range().has_value();
  if (should_match_ternary && should_match_range) {
    if (partial.Range() == interval_ops::FromTernary(*partial.Ternary())) {
      // The range is redundant; just use the ternary check, which is cheaper.
      should_match_range = false;
    } else if (partial.Ternary() ==
               interval_ops::ExtractTernaryVector(*partial.Range())) {
      // The ternary is redundant; just use the range check.
      should_match_ternary = false;
    }
  }

  std::vector<Node*> match_conditions;
  match_conditions.reserve(2);

  if (should_match_ternary) {
    XLS_ASSIGN_OR_RETURN(
        Node * bits_to_check,
        GatherBits(node, ternary_ops::ToKnownBits(*partial.Ternary())));

    InlineBitmap target_bitmap(bits_to_check->BitCountOrDie());
    int64_t target_index = 0;
    for (TernaryValue value : *partial.Ternary()) {
      if (value == TernaryValue::kUnknown) {
        continue;
      }
      target_bitmap.Set(target_index++, value == TernaryValue::kKnownOne);
    }
    XLS_ASSIGN_OR_RETURN(
        Node * target,
        node->function_base()->MakeNode<Literal>(
            user->loc(), Value(Bits::FromBitmap(std::move(target_bitmap)))));
    XLS_ASSIGN_OR_RETURN(Node * check_ternary,
                         node->function_base()->MakeNode<CompareOp>(
                             user->loc(), bits_to_check, target, Op::kEq));
    match_conditions.push_back(std::move(check_ternary));
  }

  if (should_match_range) {
    std::vector<Node*> interval_checks;
    interval_checks.reserve(partial.Range()->Intervals().size());
    for (const Interval& interval : partial.Range()->Intervals()) {
      if (interval.IsPrecise()) {
        XLS_ASSIGN_OR_RETURN(
            Node * value, node->function_base()->MakeNode<Literal>(
                              user->loc(), Value(*interval.GetPreciseValue())));
        XLS_ASSIGN_OR_RETURN(Node * equals_value,
                             node->function_base()->MakeNode<CompareOp>(
                                 user->loc(), node, value, Op::kEq));
        interval_checks.push_back(equals_value);
        continue;
      }

      std::optional<Node*> interval_check;
      if (!interval.LowerBound().IsZero()) {
        XLS_ASSIGN_OR_RETURN(Node * lb,
                             node->function_base()->MakeNode<Literal>(
                                 user->loc(), Value(interval.LowerBound())));
        XLS_ASSIGN_OR_RETURN(interval_check,
                             node->function_base()->MakeNode<CompareOp>(
                                 user->loc(), node, lb, Op::kUGe));
      }
      if (!interval.UpperBound().IsAllOnes()) {
        XLS_ASSIGN_OR_RETURN(Node * ub,
                             node->function_base()->MakeNode<Literal>(
                                 user->loc(), Value(interval.UpperBound())));
        XLS_ASSIGN_OR_RETURN(Node * ub_check,
                             node->function_base()->MakeNode<CompareOp>(
                                 user->loc(), node, ub, Op::kULe));
        if (interval_check.has_value()) {
          XLS_ASSIGN_OR_RETURN(
              interval_check,
              node->function_base()->MakeNode<NaryOp>(
                  user->loc(), absl::MakeConstSpan({*interval_check, ub_check}),
                  Op::kAnd));
        } else {
          interval_check = ub_check;
        }
      }

      if (interval_check.has_value()) {
        interval_checks.push_back(*interval_check);
      }
    }
    XLS_ASSIGN_OR_RETURN(
        Node * check_range,
        NaryOrIfNeeded(node->function_base(), interval_checks));
    match_conditions.push_back(std::move(check_range));
  };

  return NaryAndIfNeeded(node->function_base(), match_conditions);
}

absl::StatusOr<bool> EliminateNoopNext(FunctionBase* f) {
  std::vector<Next*> to_remove;
  for (Node* n : f->nodes()) {
    if (!n->Is<Next>()) {
      continue;
    }
    Next* next = n->As<Next>();
    if (next->state_read() == next->value()) {
      to_remove.push_back(next);
    }
  }
  for (Next* next : to_remove) {
    XLS_RETURN_IF_ERROR(
        next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
    XLS_RETURN_IF_ERROR(f->RemoveNode(next));
  }
  return !to_remove.empty();
}

}  // namespace

absl::StatusOr<bool> ConditionalSpecializationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results, OptimizationContext& context) const {
  bool changed = false;
  if (options.eliminate_noop_next) {
    XLS_ASSIGN_OR_RETURN(changed, EliminateNoopNext(f));
  }

  std::vector<std::unique_ptr<QueryEngine>> owned_query_engines;
  std::vector<QueryEngine*> unowned_query_engines;
  owned_query_engines.push_back(std::make_unique<StatelessQueryEngine>());
  if (use_bdd_) {
    unowned_query_engines.push_back(
        context.SharedQueryEngine<BddQueryEngine>(f));
  }

  UnionQueryEngine query_engine(std::move(owned_query_engines),
                                std::move(unowned_query_engines));
  XLS_RETURN_IF_ERROR(query_engine.Populate(f).status());

  // NOTE: Since this is not a lazy analysis, it can go stale!
  XLS_ASSIGN_OR_RETURN(VisibilityConditions analysis,
                       RunVisibilityAnalysis(f, context, use_bdd_));

  std::optional<absl::flat_hash_map<Node*, absl::flat_hash_set<Node*>>>
      affected_by;

  // Iterate backwards through the graph because we add conditions at the case
  // arm operands of selects and propagate them upwards through the expressions
  // which compute the case arm.
  for (Node* node : context.ReverseTopoSort(f)) {
    VLOG(4) << absl::StreamFormat("Considering node %s", node->GetName());

    // Specialize the node itself based on the conditions on its access.
    if (options.optimize_for_best_case_throughput && node->Is<StateRead>() &&
        !analysis.GetVisibilityConditions(node).empty()) {
      const ConditionSet& set = analysis.GetVisibilityConditions(node);

      StateRead* state_read = node->As<StateRead>();
      if (state_read->predicate().has_value()) {
        // For now, avoid specializing the predicate of an already-conditional
        // state read. This keeps us from getting into an infinite loop.
        continue;
      }

      // Record that this node is unused (including by next_value nodes) when
      // the condition set is not met.
      absl::Span<const Condition> accessed_when = set.conditions();

      std::vector<Node*> access_conditions;
      if (!affected_by.has_value()) {
        affected_by = AffectedBy(f, context);
      }
      for (auto& [src, given] : accessed_when) {
        if ((*affected_by)[node].contains(src)) {
          // The value of `src` depends on the value of `node`, so it's not
          // possible to specialize on `src` without creating a cycle.
          continue;
        }
        XLS_ASSIGN_OR_RETURN(std::optional<Node*> access_condition,
                             CheckMatch(src, given, node));
        if (access_condition.has_value()) {
          access_conditions.push_back(*access_condition);
        }
      }
      if (access_conditions.empty()) {
        continue;
      }

      VLOG(2) << absl::StreamFormat(
          "Specializing previously-unconditional state read %s; only accessed "
          "when: %s",
          node->GetName(), set.ToString());
      XLS_ASSIGN_OR_RETURN(Node * new_predicate,
                           NaryAndIfNeeded(f, access_conditions));
      XLS_RETURN_IF_ERROR(state_read->SetPredicate(new_predicate));
      changed = true;
    }

    // Now specialize any operands (if possible) based on the conditions.
    if (node->Is<StateRead>()) {
      // We don't want to specialize the predicate of a state read; this can
      // reduce throughput.
      continue;
    }
    for (int64_t operand_no = 0; operand_no < node->operand_count();
         ++operand_no) {
      Node* operand = node->operand(operand_no);

      if (operand->Is<Literal>()) {
        // Operand is already a literal. Nothing to do.
        continue;
      }

      const ConditionSet& edge_set =
          analysis.GetVisibilityConditionsForEdge(node, operand_no);
      VLOG(4) << absl::StrFormat(
          "Conditions on edge %s -> %s (as operand %d): %s", operand->GetName(),
          node->GetName(), operand_no, edge_set.ToString());
      if (edge_set.empty()) {
        continue;
      }

      if (node->Is<Next>() && operand_no == Next::kStateReadOperand) {
        // No point in specializing the state read, and it would make the node
        // invalid anyway; this is just a pointer to the state element.
        continue;
      }

      std::unique_ptr<QueryEngine> specialized_query_engine =
          query_engine.SpecializeGiven(edge_set.GetAsGivens());

      // First check to see if the condition set directly implies a value for
      // the operand. If so replace with the implied value.
      if (std::optional<Bits> implied_value =
              ImpliedNodeValue(edge_set, operand, *specialized_query_engine);
          implied_value.has_value()) {
        VLOG(3) << absl::StreamFormat("Replacing operand %d of %s with %v",
                                      operand_no, node->GetName(),
                                      implied_value.value());
        XLS_ASSIGN_OR_RETURN(
            Literal * literal,
            f->MakeNode<Literal>(operand->loc(), Value(implied_value.value())));
        XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(operand_no, literal));
        changed = true;
        continue;
      }

      // If `operand` is a select and any condition set of `node` implies the
      // selector value then we can wire the respective implied case directly to
      // that user. For example:
      //
      //         a   b                 a     b
      //          \ /                   \   / \
      //  s|t ->  sel0          s|t ->  sel0    \
      //           | \     =>            |      |
      //        c  |  \                  d   c  |
      //         \ |   d                      \ |
      //  s   ->  sel1                   s -> sel1
      //           |                            |
      //
      // This pattern is not handled elsewhere because `sel0` has other uses
      // than `sel1` so `sel0` does not inherit the selector condition `s==1`.
      //
      // It may be possible to bypass multiple selects so walk the edge up the
      // graph as far as possible. For example, in the diagram above `b` may
      // also be a select with a selector whose value is implied by `s`.
      //
      // This also applies to ANDs, ORs, and XORs, if the condition set implies
      // that all but one operand is the identity for the operation.
      if (operand->OpIn({Op::kSel, Op::kPrioritySel, Op::kOneHotSel, Op::kAnd,
                         Op::kOr, Op::kXor})) {
        std::optional<Node*> replacement;
        Node* src = operand;
        while (src->OpIn({Op::kSel, Op::kPrioritySel, Op::kOneHotSel, Op::kAnd,
                          Op::kOr, Op::kXor})) {
          if (src->Is<Select>()) {
            Select* select = src->As<Select>();
            if (select->selector()->Is<Literal>()) {
              break;
            }
            std::optional<Bits> implied_selector = ImpliedNodeValue(
                edge_set, select->selector(), *specialized_query_engine);
            if (!implied_selector.has_value()) {
              break;
            }
            Node* implied_case =
                GetSelectedCase(select, implied_selector.value());
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply selector %s of select %s "
                "has value %v",
                operand->GetName(), node->GetName(),
                select->selector()->GetName(), select->GetName(),
                implied_selector.value());
            replacement = implied_case;
            src = implied_case;
          } else if (src->Is<PrioritySelect>()) {
            PrioritySelect* select = src->As<PrioritySelect>();
            if (select->selector()->Is<Literal>()) {
              break;
            }
            std::optional<TernaryVector> implied_selector = ImpliedNodeTernary(
                edge_set, select->selector(), *specialized_query_engine);
            if (!implied_selector.has_value()) {
              break;
            }
            std::optional<Node*> implied_case =
                GetSelectedCase(select, *implied_selector);
            if (!implied_case.has_value()) {
              break;
            }
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply selector %s of select %s "
                "has value %s",
                operand->GetName(), node->GetName(),
                select->selector()->GetName(), select->GetName(),
                xls::ToString(*implied_selector));
            src = *implied_case;
            replacement = src;
          } else if (src->Is<OneHotSelect>()) {
            XLS_RET_CHECK(src->Is<OneHotSelect>());
            OneHotSelect* ohs = src->As<OneHotSelect>();
            if (ohs->selector()->Is<Literal>()) {
              break;
            }
            std::optional<TernaryVector> implied_selector = ImpliedNodeTernary(
                edge_set, ohs->selector(), *specialized_query_engine);
            if (!implied_selector.has_value()) {
              break;
            }
            for (int64_t case_no = 0; case_no < ohs->cases().size();
                 ++case_no) {
              if (implied_selector.value()[case_no] ==
                  TernaryValue::kKnownZero) {
                continue;
              }

              // This case could be selected - but if it's definitely zero when
              // selected, then we can ignore it.
              std::optional<Bits> implied_case = ImpliedNodeValue(
                  analysis.GetVisibilityConditionsForEdge(ohs, case_no + 1),
                  ohs->cases()[case_no], *specialized_query_engine);
              if (implied_case.has_value() && implied_case->IsZero()) {
                implied_selector.value()[case_no] = TernaryValue::kKnownZero;
              }
            }
            std::optional<std::variant<Node*, ZeroValue>> implied_case =
                GetSelectedCase(ohs, *implied_selector);
            if (!implied_case.has_value()) {
              break;
            }
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply selector %s of select %s "
                "has value %s",
                operand->GetName(), node->GetName(), ohs->selector()->GetName(),
                ohs->GetName(), xls::ToString(*implied_selector));
            if (std::holds_alternative<Node*>(*implied_case)) {
              src = std::get<Node*>(*implied_case);
            } else {
              XLS_RET_CHECK(std::holds_alternative<ZeroValue>(*implied_case));
              XLS_ASSIGN_OR_RETURN(
                  src,
                  f->MakeNode<Literal>(src->loc(), ZeroOfType(src->GetType())));
            }
            replacement = src;
          } else {
            XLS_RET_CHECK(src->OpIn({Op::kAnd, Op::kOr, Op::kXor}));
            auto is_identity = [&](const Bits& b) {
              if (src->op() == Op::kAnd) {
                return b.IsAllOnes();
              }
              return b.IsZero();
            };
            NaryOp* bitwise_op = src->As<NaryOp>();
            std::optional<Node*> nonidentity_operand = std::nullopt;
            for (Node* potential_src : bitwise_op->operands()) {
              XLS_RET_CHECK(potential_src->GetType()->IsBits());
              std::optional<Bits> implied_src = ImpliedNodeValue(
                  edge_set, potential_src, *specialized_query_engine);
              if (implied_src.has_value() && is_identity(*implied_src)) {
                continue;
              }
              if (nonidentity_operand.has_value()) {
                // There's more than one potentially-non-zero operand; we're
                // done, there's nothing to do.
                nonidentity_operand = std::nullopt;
                break;
              }
              nonidentity_operand = potential_src;
            }
            if (!nonidentity_operand.has_value()) {
              break;
            }
            VLOG(3) << absl::StreamFormat(
                "Conditions for edge (%s, %s) imply that bitwise operation "
                "%s has only one non-identity operand: %s",
                operand->GetName(), node->GetName(), bitwise_op->GetName(),
                nonidentity_operand.value()->GetName());
            src = *nonidentity_operand;
            replacement = src;
          }
        }
        if (replacement.has_value()) {
          VLOG(3) << absl::StreamFormat(
              "Replacing operand %d of %s with %s due to implied selector "
              "value(s)",
              operand_no, node->GetName(), replacement.value()->GetName());
          XLS_RETURN_IF_ERROR(
              node->ReplaceOperandNumber(operand_no, replacement.value()));
          changed = true;
        }
      }
    }
  }

  return changed;
}
absl::StatusOr<PassPipelineProto::Element>
ConditionalSpecializationPass::ToProto() const {
  // TODO(allight): This is not very elegant. Ideally the registry could handle
  // this? Doing it there would probably be even more weird though.
  PassPipelineProto::Element e;
  *e.mutable_pass_name() = use_bdd_ ? "cond_spec(Bdd)" : "cond_spec(noBdd)";
  return e;
}

XLS_REGISTER_MODULE_INITIALIZER(cond_spec, {
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(true)", true));
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(Bdd)", true));
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(false)", false));
  CHECK_OK(RegisterOptimizationPass<ConditionalSpecializationPass>(
      "cond_spec(noBdd)", false));
});

}  // namespace xls
