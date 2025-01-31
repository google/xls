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

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

namespace {

absl::Status RemoveNextValue(Proc* proc, Next* next,
                             absl::flat_hash_map<Next*, int64_t>& split_depth) {
  XLS_RETURN_IF_ERROR(
      next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
  if (auto it = split_depth.find(next); it != split_depth.end()) {
    split_depth.erase(it);
  }
  return proc->RemoveNode(next);
}

absl::StatusOr<std::optional<std::vector<Next*>>> RemoveConstantPredicate(
    Proc* proc, Next* next, absl::flat_hash_map<Next*, int64_t>& split_depth,
    const QueryEngine& query_engine) {
  if (!next->predicate().has_value()) {
    return std::nullopt;
  }
  std::optional<Bits> constant_predicate =
      query_engine.KnownValueAsBits(*next->predicate());
  if (!constant_predicate.has_value()) {
    return std::nullopt;
  }

  if (constant_predicate->IsZero()) {
    VLOG(2) << "Identified node as dead due to zero predicate; removing: "
            << *next;
    XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
    return std::vector<Next*>();
  }
  VLOG(2) << "Identified node as always live; removing predicate: " << *next;
  XLS_ASSIGN_OR_RETURN(Next * new_next, next->ReplaceUsesWithNew<Next>(
                                            /*state_read=*/next->state_read(),
                                            /*value=*/next->value(),
                                            /*predicate=*/std::nullopt));
  new_next->SetLoc(next->loc());
  if (next->HasAssignedName()) {
    new_next->SetName(next->GetName());
  }
  if (split_depth.contains(next)) {
    split_depth[new_next] = split_depth[next];
  }
  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
  return std::vector<Next*>({new_next});
}

absl::StatusOr<std::optional<std::vector<Next*>>> SplitSmallSelect(
    Proc* proc, Next* next, const OptimizationPassOptions& options,
    absl::flat_hash_map<Next*, int64_t>& split_depth) {
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

  int64_t depth = 1;
  if (auto it = split_depth.find(next); it != split_depth.end()) {
    depth = it->second + 1;
  }

  std::vector<Next*> new_next_values;
  for (int64_t i = 0; i < selected_value->cases().size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Literal * index,
        proc->MakeNode<Literal>(
            selected_value->selector()->loc(),
            Value(UBits(i, selected_value->selector()->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(
        Node * predicate,
        proc->MakeNode<CompareOp>(selected_value->selector()->loc(),
                                  selected_value->selector(), index, Op::kEq));
    if (next->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          predicate,
          proc->MakeNode<NaryOp>(
              selected_value->selector()->loc(),
              std::vector<Node*>{*next->predicate(), predicate}, Op::kAnd));
    }

    std::string name;
    if (next->HasAssignedName()) {
      name = absl::StrCat(next->GetName(), "_case_", i);
    }
    XLS_ASSIGN_OR_RETURN(
        Next * new_next,
        proc->MakeNodeWithName<Next>(next->loc(),
                                     /*state_read=*/next->state_read(),
                                     /*value=*/selected_value->cases()[i],
                                     predicate, name));
    new_next_values.push_back(new_next);
    split_depth[new_next] = depth;
  }

  if (selected_value->default_value().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        Literal * max_index,
        proc->MakeNode<Literal>(
            selected_value->selector()->loc(),
            Value(UBits(selected_value->cases().size() - 1,
                        selected_value->selector()->BitCountOrDie()))));
    XLS_ASSIGN_OR_RETURN(Node * predicate,
                         proc->MakeNode<CompareOp>(
                             selected_value->selector()->loc(),
                             selected_value->selector(), max_index, Op::kUGt));
    if (next->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          predicate,
          proc->MakeNode<NaryOp>(
              selected_value->selector()->loc(),
              std::vector<Node*>{*next->predicate(), predicate}, Op::kAnd));
    }

    std::string name;
    if (next->HasAssignedName()) {
      name = absl::StrCat(next->GetName(), "_default_case");
    }
    XLS_ASSIGN_OR_RETURN(
        Next * new_next,
        proc->MakeNodeWithName<Next>(next->loc(),
                                     /*state_read=*/next->state_read(),
                                     /*value=*/*selected_value->default_value(),
                                     predicate, name));
    new_next_values.push_back(new_next);
    split_depth[new_next] = depth;
  }

  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
  return new_next_values;
}

absl::StatusOr<std::optional<std::vector<Next*>>> SplitPrioritySelect(
    Proc* proc, Next* next, absl::flat_hash_map<Next*, int64_t>& split_depth) {
  if (!next->value()->Is<PrioritySelect>()) {
    return std::nullopt;
  }
  PrioritySelect* selected_value = next->value()->As<PrioritySelect>();

  VLOG(2) << "Splitting next value over priority select: " << *selected_value;
  int64_t depth = 1;
  if (auto it = split_depth.find(next); it != split_depth.end()) {
    depth = it->second + CeilOfLog2(selected_value->cases().size() + 1);
  }

  std::vector<Next*> new_next_values;
  for (int64_t i = 0; i < selected_value->cases().size(); ++i) {
    absl::InlinedVector<Node*, 3> all_clauses;
    XLS_ASSIGN_OR_RETURN(
        Node * case_active,
        proc->MakeNode<BitSlice>(selected_value->selector()->loc(),
                                 selected_value->selector(),
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
                                     /*state_read=*/next->state_read(),
                                     /*value=*/selected_value->get_case(i),
                                     /*predicate=*/case_predicate, name));
    new_next_values.push_back(new_next);
    split_depth[new_next] = depth;
  }

  // Default case; if all bits of the input are zero, `priority_sel` returns
  // zero.
  absl::InlinedVector<Node*, 2> all_default_clauses;
  XLS_ASSIGN_OR_RETURN(Literal * zero_selector,
                       proc->MakeNode<Literal>(
                           selected_value->selector()->loc(),
                           ZeroOfType(selected_value->selector()->GetType())));
  XLS_ASSIGN_OR_RETURN(Node * all_cases_inactive,
                       proc->MakeNode<CompareOp>(
                           selected_value->selector()->loc(),
                           selected_value->selector(), zero_selector, Op::kEq));
  all_default_clauses.push_back(all_cases_inactive);
  if (next->predicate().has_value()) {
    all_default_clauses.push_back(*next->predicate());
  }
  XLS_ASSIGN_OR_RETURN(Node * default_predicate,
                       NaryAndIfNeeded(proc, all_default_clauses));
  Node* default_value = selected_value->default_value();

  std::string name;
  if (next->HasAssignedName()) {
    name = absl::StrCat(next->GetName(), "_case_default");
  }
  XLS_ASSIGN_OR_RETURN(
      Next * new_next,
      proc->MakeNodeWithName<Next>(next->loc(),
                                   /*state_read=*/next->state_read(),
                                   /*value=*/default_value,
                                   /*predicate=*/default_predicate, name));
  new_next_values.push_back(new_next);
  split_depth[new_next] = depth;

  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
  return new_next_values;
}

absl::StatusOr<std::optional<std::vector<Next*>>> SplitSafeOneHotSelect(
    Proc* proc, Next* next, absl::flat_hash_map<Next*, int64_t>& split_depth,
    const QueryEngine& query_engine) {
  if (!next->value()->Is<OneHotSelect>()) {
    return std::nullopt;
  }
  OneHotSelect* selected_value = next->value()->As<OneHotSelect>();
  if (!query_engine.ExactlyOneBitTrue(selected_value->selector())) {
    // Not safe to use for `next_value`; actual value could be the OR of
    // multiple cases.
    return std::nullopt;
  }

  int64_t depth = 1;
  if (auto it = split_depth.find(next); it != split_depth.end()) {
    depth = it->second + 1;
  }

  std::vector<Next*> new_next_values;
  for (int64_t i = 0; i < selected_value->cases().size(); ++i) {
    XLS_ASSIGN_OR_RETURN(
        Node * case_predicate,
        proc->MakeNode<BitSlice>(selected_value->selector()->loc(),
                                 selected_value->selector(),
                                 /*start=*/i, /*width=*/1));
    if (next->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          case_predicate,
          proc->MakeNode<NaryOp>(
              selected_value->selector()->loc(),
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
                                     /*state_read=*/next->state_read(),
                                     /*value=*/selected_value->get_case(i),
                                     /*predicate=*/case_predicate, name));
    new_next_values.push_back(new_next);
    split_depth[new_next] = depth;
  }
  XLS_RETURN_IF_ERROR(RemoveNextValue(proc, next, split_depth));
  return new_next_values;
}

}  // namespace

absl::StatusOr<bool> NextValueOptimizationPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;

  StatelessQueryEngine query_engine;

  std::deque<Next*> worklist(proc->next_values().begin(),
                             proc->next_values().end());
  absl::flat_hash_map<Next*, int64_t> split_depth;
  while (!worklist.empty()) {
    Next* next = worklist.front();
    worklist.pop_front();

    XLS_ASSIGN_OR_RETURN(
        std::optional<std::vector<Next*>> literal_predicate_next_values,
        RemoveConstantPredicate(proc, next, split_depth, query_engine));
    if (literal_predicate_next_values.has_value()) {
      changed = true;
      worklist.insert(worklist.end(), literal_predicate_next_values->begin(),
                      literal_predicate_next_values->end());
      continue;
    }

    if (auto it = split_depth.find(next);
        options.splits_enabled() && max_split_depth_ > 0 &&
        (it == split_depth.end() || it->second < max_split_depth_)) {
      XLS_ASSIGN_OR_RETURN(
          std::optional<std::vector<Next*>> split_select_next_values,
          SplitSmallSelect(proc, next, options, split_depth));
      if (split_select_next_values.has_value()) {
        changed = true;
        worklist.insert(worklist.end(), split_select_next_values->begin(),
                        split_select_next_values->end());
        continue;
      }

      XLS_ASSIGN_OR_RETURN(
          std::optional<std::vector<Next*>> split_priority_select_next_values,
          SplitPrioritySelect(proc, next, split_depth));
      if (split_priority_select_next_values.has_value()) {
        changed = true;
        worklist.insert(worklist.end(),
                        split_priority_select_next_values->begin(),
                        split_priority_select_next_values->end());
        continue;
      }

      XLS_ASSIGN_OR_RETURN(
          std::optional<std::vector<Next*>> split_one_hot_select_next_values,
          SplitSafeOneHotSelect(proc, next, split_depth, query_engine));
      if (split_one_hot_select_next_values.has_value()) {
        changed = true;
        worklist.insert(worklist.end(),
                        split_one_hot_select_next_values->begin(),
                        split_one_hot_select_next_values->end());
        continue;
      }
    }
  }

  return changed;
}

REGISTER_OPT_PASS(NextValueOptimizationPass);

}  // namespace xls
