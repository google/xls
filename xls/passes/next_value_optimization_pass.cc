// Copyright 2024 The XLS Authors
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

#include "xls/passes/next_value_optimization_pass.h"

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

absl::StatusOr<bool> ModernizeNextValues(Proc* proc) {
  CHECK(proc->next_values().empty());

  for (int64_t index = 0; index < proc->GetStateElementCount(); ++index) {
    Param* param = proc->GetStateParam(index);
    Node* next_value = proc->GetNextStateElement(index);
    XLS_RETURN_IF_ERROR(
        proc->MakeNodeWithName<Next>(param->loc(), /*param=*/param,
                                     /*value=*/next_value,
                                     /*predicate=*/std::nullopt,
                                     absl::StrCat(param->name(), "_next"))
            .status());

    if (next_value != static_cast<Node*>(param)) {
      // Nontrivial next-state element; remove it so we pass verification.
      XLS_RETURN_IF_ERROR(proc->SetNextStateElement(index, param));
    }
  }

  return proc->GetStateElementCount() > 0;
}

absl::Status RemoveNextValue(Proc* proc, Next* next) {
  XLS_RETURN_IF_ERROR(
      next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
  return proc->RemoveNode(next);
}

absl::StatusOr<std::optional<std::vector<Next*>>> RemoveLiteralPredicate(
    Proc* proc, Next* next) {
  if (!next->predicate().has_value()) {
    return std::nullopt;
  }
  Node* predicate = *next->predicate();
  if (!predicate->Is<Literal>()) {
    return std::nullopt;
  }

  Literal* literal_predicate = predicate->As<Literal>();
  if (literal_predicate->value().IsAllZeros()) {
    XLS_VLOG(2) << "Identified node as dead due to zero predicate; removing: "
                << *next;
    XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next));
    return std::vector<Next*>();
  }
  XLS_VLOG(2) << "Identified node as always live; removing predicate: "
              << *next;
  XLS_ASSIGN_OR_RETURN(Next * new_next, next->ReplaceUsesWithNew<Next>(
                                            /*param=*/next->param(),
                                            /*value=*/next->value(),
                                            /*predicate=*/std::nullopt));
  new_next->SetLoc(next->loc());
  if (next->HasAssignedName()) {
    new_next->SetName(next->GetName());
  }
  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next));
  return std::vector<Next*>({new_next});
}

absl::StatusOr<std::optional<std::vector<Next*>>> SplitSmallSelect(
    Proc* proc, Next* next, const OptimizationPassOptions& options) {
  if (!options.split_next_value_selects.has_value()) {
    return std::nullopt;
  }

  if (!next->value()->Is<Select>()) {
    return std::nullopt;
  }

  Select* selected_value = next->value()->As<Select>();
  if (selected_value->cases().size() > *options.split_next_value_selects) {
    return std::nullopt;
  }

  std::vector<Next*> new_next_values;
  for (int64_t i = 0; i < selected_value->cases().size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Literal * index,
        proc->MakeNode<Literal>(
            SourceInfo(),
            Value(UBits(i, selected_value->selector()->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(
        Node * predicate,
        proc->MakeNode<CompareOp>(SourceInfo(), selected_value->selector(),
                                  index, Op::kEq));
    if (next->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          predicate,
          proc->MakeNode<NaryOp>(
              SourceInfo(), std::vector<Node*>{*next->predicate(), predicate},
              Op::kAnd));
    }

    std::string name;
    if (next->HasAssignedName()) {
      name = absl::StrCat(next->GetName(), "_case_", i);
    }
    XLS_ASSIGN_OR_RETURN(
        Next * new_next,
        proc->MakeNodeWithName<Next>(next->loc(),
                                     /*param=*/next->param(),
                                     /*value=*/selected_value->cases()[i],
                                     predicate, name));
    new_next_values.push_back(new_next);
  }

  if (selected_value->default_value().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        Literal * max_index,
        proc->MakeNode<Literal>(
            SourceInfo(),
            Value(UBits(selected_value->cases().size() - 1,
                        selected_value->selector()->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(
        Node * predicate,
        proc->MakeNode<CompareOp>(SourceInfo(), selected_value->selector(),
                                  max_index, Op::kUGt));
    if (next->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          predicate,
          proc->MakeNode<NaryOp>(
              SourceInfo(), std::vector<Node*>{*next->predicate(), predicate},
              Op::kAnd));
    }

    std::string name;
    if (next->HasAssignedName()) {
      name = absl::StrCat(next->GetName(), "_default_case");
    }
    XLS_ASSIGN_OR_RETURN(
        Next * new_next,
        proc->MakeNodeWithName<Next>(next->loc(),
                                     /*param=*/next->param(),
                                     /*value=*/*selected_value->default_value(),
                                     predicate, name));
    new_next_values.push_back(new_next);
  }

  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next));
  return new_next_values;
}

absl::StatusOr<std::optional<std::vector<Next*>>> SplitPrioritySelect(
    Proc* proc, Next* next) {
  if (!next->value()->Is<PrioritySelect>()) {
    return std::nullopt;
  }
  PrioritySelect* selected_value = next->value()->As<PrioritySelect>();

  std::vector<Next*> new_next_values;
  for (int64_t i = 0; i < selected_value->cases().size(); ++i) {
    absl::InlinedVector<Node*, 3> all_clauses;
    XLS_ASSIGN_OR_RETURN(
        Node * case_active,
        proc->MakeNode<BitSlice>(SourceInfo(), selected_value->selector(),
                                 /*start=*/i, /*width=*/1));
    all_clauses.push_back(case_active);
    if (next->predicate().has_value()) {
      all_clauses.push_back(*next->predicate());
    }
    if (i > 0) {
      XLS_ASSIGN_OR_RETURN(Node * higher_priority_cases_inactive,
                           NorReduceTrailing(selected_value->selector(), i));
      all_clauses.push_back(higher_priority_cases_inactive);
    }
    XLS_ASSIGN_OR_RETURN(Node * case_predicate,
                         NaryAndIfNeeded(proc, all_clauses));

    std::string name;
    if (next->HasAssignedName()) {
      name = absl::StrCat(next->GetName(), "_case_", i);
    }
    XLS_ASSIGN_OR_RETURN(
        Next * new_next,
        proc->MakeNodeWithName<Next>(next->loc(),
                                     /*param=*/next->param(),
                                     /*value=*/selected_value->get_case(i),
                                     /*predicate=*/case_predicate, name));
    new_next_values.push_back(new_next);
  }

  // Default case; if all bits of the input are zero, `priority_sel` returns
  // zero.
  absl::InlinedVector<Node*, 2> all_default_clauses;
  XLS_ASSIGN_OR_RETURN(
      Literal * zero_selector,
      proc->MakeNode<Literal>(
          SourceInfo(), ZeroOfType(selected_value->selector()->GetType())));
  XLS_ASSIGN_OR_RETURN(
      Node * all_cases_inactive,
      proc->MakeNode<CompareOp>(SourceInfo(), selected_value->selector(),
                                zero_selector, Op::kEq));
  all_default_clauses.push_back(all_cases_inactive);
  if (next->predicate().has_value()) {
    all_default_clauses.push_back(*next->predicate());
  }
  XLS_ASSIGN_OR_RETURN(Node * default_predicate,
                       NaryAndIfNeeded(proc, all_default_clauses));
  XLS_ASSIGN_OR_RETURN(
      Node * default_value,
      proc->MakeNode<Literal>(SourceInfo(),
                              ZeroOfType(selected_value->GetType())));

  std::string name;
  if (next->HasAssignedName()) {
    name = absl::StrCat(next->GetName(), "_case_default");
  }
  XLS_ASSIGN_OR_RETURN(
      Next * new_next,
      proc->MakeNodeWithName<Next>(next->loc(),
                                   /*param=*/next->param(),
                                   /*value=*/default_value,
                                   /*predicate=*/default_predicate, name));
  new_next_values.push_back(new_next);

  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next));
  return new_next_values;
}

absl::StatusOr<std::optional<std::vector<Next*>>> SplitSafeOneHotSelect(
    Proc* proc, Next* next) {
  if (!next->value()->Is<OneHotSelect>()) {
    return std::nullopt;
  }
  OneHotSelect* selected_value = next->value()->As<OneHotSelect>();
  if (!selected_value->selector()->Is<OneHot>()) {
    // Not safe to use for `next_value`; actual value could be the OR of
    // multiple cases.
    return std::nullopt;
  }

  std::vector<Next*> new_next_values;
  for (int64_t i = 0; i < selected_value->cases().size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * case_predicate,
        proc->MakeNode<BitSlice>(SourceInfo(), selected_value->selector(),
                                 /*start=*/i, /*width=*/1));
    if (next->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          case_predicate,
          proc->MakeNode<NaryOp>(
              SourceInfo(),
              std::vector<Node*>{*next->predicate(), case_predicate},
              Op::kAnd));
    }

    std::string name;
    if (next->HasAssignedName()) {
      name = absl::StrCat(next->GetName(), "_case_", i);
    }
    XLS_ASSIGN_OR_RETURN(
        Next * new_next,
        proc->MakeNodeWithName<Next>(next->loc(),
                                     /*param=*/next->param(),
                                     /*value=*/selected_value->get_case(i),
                                     /*predicate=*/case_predicate, name));
    new_next_values.push_back(new_next);
  }
  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next));
  return new_next_values;
}

}  // namespace

absl::StatusOr<bool> NextValueOptimizationPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;

  if (proc->next_values().empty()) {
    XLS_ASSIGN_OR_RETURN(bool modernize_changed, ModernizeNextValues(proc));
    changed = changed || modernize_changed;
  }

  std::deque<Next*> worklist(proc->next_values().begin(),
                             proc->next_values().end());
  while (!worklist.empty()) {
    Next* next = worklist.front();
    worklist.pop_front();

    XLS_ASSIGN_OR_RETURN(
        std::optional<std::vector<Next*>> literal_predicate_next_values,
        RemoveLiteralPredicate(proc, next));
    if (literal_predicate_next_values.has_value()) {
      changed = true;
      worklist.insert(worklist.end(), literal_predicate_next_values->begin(),
                      literal_predicate_next_values->end());
      continue;
    }

    XLS_ASSIGN_OR_RETURN(
        std::optional<std::vector<Next*>> split_select_next_values,
        SplitSmallSelect(proc, next, options));
    if (split_select_next_values.has_value()) {
      changed = true;
      worklist.insert(worklist.end(), split_select_next_values->begin(),
                      split_select_next_values->end());
      continue;
    }

    XLS_ASSIGN_OR_RETURN(
        std::optional<std::vector<Next*>> split_priority_select_next_values,
        SplitPrioritySelect(proc, next));
    if (split_priority_select_next_values.has_value()) {
      changed = true;
      worklist.insert(worklist.end(),
                      split_priority_select_next_values->begin(),
                      split_priority_select_next_values->end());
      continue;
    }

    XLS_ASSIGN_OR_RETURN(
        std::optional<std::vector<Next*>> split_one_hot_select_next_values,
        SplitSafeOneHotSelect(proc, next));
    if (split_one_hot_select_next_values.has_value()) {
      changed = true;
      worklist.insert(worklist.end(), split_one_hot_select_next_values->begin(),
                      split_one_hot_select_next_values->end());
      continue;
    }
  }

  return changed;
}

REGISTER_OPT_PASS(NextValueOptimizationPass);

}  // namespace xls
