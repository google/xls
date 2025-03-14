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

#include "xls/scheduling/proc_state_legalization_pass.h"

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/value.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/solvers/z3_ir_translator.h"

namespace xls {

namespace {

// Ensure that `state_read` is either unconditional or has a predicate that is
// true whenever any of its corresponding `next_value`s are active.
absl::StatusOr<bool> LegalizeStateReadPredicate(
    Proc* proc, StateElement* state_element,
    const SchedulingPassOptions& options) {
  StateRead* state_read = proc->GetStateRead(state_element);
  const absl::btree_set<Next*, Node::NodeIdLessThan>& next_values =
      proc->next_values(state_read);
  if (!state_read->predicate().has_value() || next_values.empty()) {
    // Already unconditional, or no explicit `next_value`s; nothing to do.
    return false;
  }

  std::vector<Node*> predicates;
  absl::flat_hash_set<Node*> predicates_set;
  predicates.reserve(1 + next_values.size());
  predicates_set.reserve(next_values.size());
  for (Next* next : next_values) {
    if (next->state_read() == next->value()) {
      // This is a no-op next_value; we will narrow it to the case where the
      // state read is active instead.
      continue;
    }
    if (!next->predicate().has_value()) {
      // Unconditional next_value; just remove the predicate.
      XLS_RETURN_IF_ERROR(state_read->RemovePredicate());
      return true;
    }

    // We can't just add the next-value predicate to the state read predicate
    // directly, because it may create a cycle; the state read may be part of
    // what decides whether the next-value is active. To fix this, we remove the
    // state read from the predicate (erring on the side of reading too often).
    XLS_ASSIGN_OR_RETURN(
        Node * safe_predicate,
        RemoveNodeFromBooleanExpression(state_read, *next->predicate(),
                                        /*favored_outcome=*/true));
    predicates.push_back(safe_predicate);
    predicates_set.insert(safe_predicate);
  }
  if (predicates.empty()) {
    // There are no non-trivial next_value nodes; any predicate is fine.
    return false;
  }

  Node* state_read_predicate = *state_read->predicate();
  if (state_read_predicate->op() == Op::kOr &&
      predicates_set ==
          absl::flat_hash_set<Node*>(predicates.begin(), predicates.end())) {
    // The predicate is already trivially correct; nothing to do.
    return false;
  }
  if (predicates_set.size() == 1 &&
      predicates.front() == state_read_predicate) {
    // The predicate is already trivially correct; nothing to do.
    return false;
  }

  predicates.insert(predicates.begin(), state_read_predicate);
  XLS_ASSIGN_OR_RETURN(
      Node * new_predicate,
      NaryOrIfNeeded(proc, predicates, /*name=*/"", state_read->loc()));
  XLS_RETURN_IF_ERROR(state_read->ReplaceOperandNumber(
      *state_read->predicate_operand_number(), new_predicate));
  return true;
}

absl::StatusOr<bool> LegalizeStateReadPredicates(
    Proc* proc, const SchedulingPassOptions& options) {
  bool changed = false;

  for (StateElement* state_element : proc->StateElements()) {
    XLS_ASSIGN_OR_RETURN(
        bool state_read_changed,
        LegalizeStateReadPredicate(proc, state_element, options));
    if (state_read_changed) {
      VLOG(4) << "Generalized read predicate for state element: "
              << state_element->name();
      changed = true;
    }
  }

  return changed;
}

absl::StatusOr<bool> AddMutualExclusionAssert(
    Proc* proc, StateElement* state_element,
    const SchedulingPassOptions& options) {
  const absl::btree_set<Next*, Node::NodeIdLessThan>& next_values =
      proc->next_values(proc->GetStateRead(state_element));
  if (next_values.size() < 2) {
    return false;
  }

  std::string label = absl::StrCat("__", state_element->name(),
                                   "__at_most_one_next_value_assert");
  if (proc->HasNode(label)) {
    return absl::InternalError(absl::StrFormat(
        "Mutual exclusion assert already exists for state "
        "element '%s'; was this pass run twice? assert label: %s",
        state_element->name(), label));
  }

  std::vector<Node*> predicate_list;
  for (Next* next : next_values) {
    XLS_RET_CHECK(next->predicate().has_value());
    predicate_list.push_back(*next->predicate());
  }
  XLS_ASSIGN_OR_RETURN(
      Node * predicates,
      proc->MakeNodeWithName<Concat>(SourceInfo(), predicate_list,
                                     absl::StrCat("__", state_element->name(),
                                                  "__next_value_predicates")));

  // Get a version of `predicates` with at most one bit set, by taking the
  // one-hot value and slicing off the all-zero bit.
  XLS_ASSIGN_OR_RETURN(
      Node * one_hot_predicates,
      proc->MakeNode<OneHot>(SourceInfo(), predicates, LsbOrMsb::kLsb));
  XLS_ASSIGN_OR_RETURN(Node * at_most_one_predicate,
                       proc->MakeNode<BitSlice>(
                           SourceInfo(), one_hot_predicates, /*start=*/0,
                           /*width=*/one_hot_predicates->BitCountOrDie() - 1));

  XLS_ASSIGN_OR_RETURN(
      Node * at_most_one_next_value,
      proc->MakeNodeWithName<CompareOp>(
          SourceInfo(), predicates, at_most_one_predicate, Op::kEq,
          absl::StrCat("__", state_element->name(),
                       "__at_most_one_next_value")));

  XLS_ASSIGN_OR_RETURN(Node * tkn,
                       proc->MakeNode<Literal>(SourceInfo(), Value::Token()));
  XLS_RETURN_IF_ERROR(
      proc->MakeNodeWithName<Assert>(
              SourceInfo(), tkn,
              /*condition=*/at_most_one_next_value,
              /*message=*/
              absl::StrCat("More than one next_value fired for state element: ",
                           state_element->name()),
              /*label=*/label,
              /*original_label=*/std::nullopt,
              /*name=*/label)
          .status());
  return true;
}

absl::StatusOr<bool> AddMutualExclusionAsserts(
    Proc* proc, const SchedulingPassOptions& options) {
  bool changed = false;

  for (StateElement* state_element : proc->StateElements()) {
    XLS_ASSIGN_OR_RETURN(bool assert_added, AddMutualExclusionAssert(
                                                proc, state_element, options));
    if (assert_added) {
      VLOG(4) << "Added mutual exclusion assert for state element: "
              << state_element->name();
      changed = true;
    }
  }

  return changed;
}

absl::StatusOr<bool> AddWriteWithoutReadAsserts(
    Proc* proc, StateElement* state_element,
    const SchedulingPassOptions& options) {
  StateRead* state_read = proc->GetStateRead(state_element);
  if (!state_read->predicate().has_value()) {
    return false;
  }

  const absl::btree_set<Next*, Node::NodeIdLessThan>& next_values =
      proc->next_values(proc->GetStateRead(state_element));
  if (next_values.empty()) {
    return false;
  }

  std::vector<Node*> predicate_list;
  for (Next* next : next_values) {
    XLS_RET_CHECK(next->predicate().has_value());
    XLS_ASSIGN_OR_RETURN(
        Node * next_not_triggered,
        proc->MakeNodeWithName<UnOp>(
            SourceInfo(), *next->predicate(), Op::kNot,
            absl::StrCat("__", state_element->name(), "__next_", next->id(),
                         "_not_triggered")));
    XLS_ASSIGN_OR_RETURN(
        Node * no_write_without_read,
        proc->MakeNodeWithName<NaryOp>(
            SourceInfo(),
            absl::MakeConstSpan({*state_read->predicate(), next_not_triggered}),
            Op::kOr,
            absl::StrCat("__", state_element->name(), "__no_next_", next->id(),
                         "_without_read")));
    std::string label = absl::StrCat("__", state_element->name(), "__next_",
                                     next->id(), "_without_read_assert");
    if (proc->HasNode(label)) {
      return absl::InternalError(absl::StrFormat(
          "Write-without-read assert already exists for next_value node '%s'; "
          "was this pass run twice? assert label: %s",
          next->GetName(), label));
    }

    XLS_ASSIGN_OR_RETURN(Node * tkn,
                         proc->MakeNode<Literal>(SourceInfo(), Value::Token()));
    XLS_RETURN_IF_ERROR(
        proc->MakeNodeWithName<Assert>(
                SourceInfo(), tkn,
                /*condition=*/no_write_without_read,
                /*message=*/
                absl::StrCat(next->GetName(),
                             " fired while read disabled for state element: ",
                             state_element->name()),
                /*label=*/label,
                /*original_label=*/std::nullopt,
                /*name=*/label)
            .status());
  }
  return true;
}

absl::StatusOr<bool> AddWriteWithoutReadAsserts(
    Proc* proc, const SchedulingPassOptions& options) {
  bool changed = false;

  for (StateElement* state_element : proc->StateElements()) {
    XLS_ASSIGN_OR_RETURN(bool assert_added, AddWriteWithoutReadAsserts(
                                                proc, state_element, options));
    if (assert_added) {
      VLOG(4) << "Added write-without-read assert for state element: "
              << state_element->name();
      changed = true;
    }
  }

  return changed;
}

absl::StatusOr<bool> AddDefaultNextValue(Proc* proc,
                                         StateElement* state_element,
                                         const SchedulingPassOptions& options) {
  absl::btree_set<Node*, Node::NodeIdLessThan> predicates;
  StateRead* state_read = proc->GetStateRead(state_element);
  for (Next* next : proc->next_values(state_read)) {
    if (next->predicate().has_value()) {
      predicates.insert(*next->predicate());
    } else {
      // Unconditional next_value; no default next_value needed.
      return false;
    }
  }

  if (predicates.empty()) {
    // No explicit `next_value` node; leave the state element unchanged by
    // default.
    XLS_RETURN_IF_ERROR(proc->MakeNodeWithName<Next>(
                                state_read->loc(), /*state_read=*/state_read,
                                /*value=*/state_read,
                                /*predicate=*/state_read->predicate(),
                                absl::StrCat(state_element->name(), "_default"))
                            .status());
    return true;
  }

  // Check if we already have an explicit "if nothing else fires" `next_value`
  // node, which keeps things cleaner and makes sure this pass is idempotent.
  for (Next* next : proc->next_values(state_read)) {
    Node* predicate = *next->predicate();

    absl::btree_set<Node*, Node::NodeIdLessThan> other_conditions = predicates;
    other_conditions.erase(predicate);

    if (other_conditions.empty()) {
      continue;
    }

    if (state_read->predicate().has_value() && predicate->OpIn({Op::kAnd}) &&
        predicate->operands().size() == 2) {
      // Check to see if this is just an `and` with the state read predicate. If
      // so, take the other operand & see if it's a not/nor of the other
      // conditions.
      if (predicate->operand(0) == *state_read->predicate()) {
        predicate = predicate->operand(1);
      } else if (predicate->operand(1) == *state_read->predicate()) {
        predicate = predicate->operand(0);
      } else {
        // It's not, so we can't trivially recognize it as being of the right
        // form.
        continue;
      }
    }

    if (!predicate->OpIn({Op::kNot, Op::kNor})) {
      continue;
    }

    absl::btree_set<Node*, Node::NodeIdLessThan> excluded_conditions(
        predicate->operands().begin(), predicate->operands().end());
    if (excluded_conditions == other_conditions) {
      // The default case is explicitly handled in a way we can recognize; no
      // change needed. (If we can't recognize it, no harm done; we just might
      // add a dead next_value node that can be eliminated in later passes.)
      return false;
    }
  }

  if (std::optional<int64_t> default_next_value_z3_rlimit =
          options.scheduling_options.default_next_value_z3_rlimit();
      default_next_value_z3_rlimit.has_value()) {
    XLS_RET_CHECK_GE(*default_next_value_z3_rlimit, 0);

    // Try to prove that at least one of our predicates must be true at all
    // times; if we can prove this, we don't need a default.
    std::vector<solvers::z3::PredicateOfNode> z3_predicates;
    for (Node* predicate : predicates) {
      z3_predicates.push_back({
          .subject = predicate,
          .p = solvers::z3::Predicate::NotEqualToZero(),
      });
    }

    absl::StatusOr<solvers::z3::ProverResult> no_default_needed =
        solvers::z3::TryProveDisjunction(
            proc, z3_predicates,
            /*rlimit=*/*default_next_value_z3_rlimit,
            /*allow_unsupported=*/true);
    if (no_default_needed.ok() &&
        std::holds_alternative<solvers::z3::ProvenTrue>(*no_default_needed)) {
      return false;
    }
  }

  // Explicitly mark the state element as unchanged when no other `next_value`
  // node is active.
  XLS_ASSIGN_OR_RETURN(
      Node * default_predicate,
      NaryNorIfNeeded(proc, std::vector(predicates.begin(), predicates.end()),
                      /*name=*/"", state_read->loc()));
  if (state_read->predicate().has_value()) {
    XLS_ASSIGN_OR_RETURN(
        default_predicate,
        proc->MakeNode<NaryOp>(
            state_read->loc(),
            absl::MakeConstSpan({*state_read->predicate(), default_predicate}),
            Op::kAnd));
  }
  XLS_RETURN_IF_ERROR(proc->MakeNodeWithName<Next>(
                              state_read->loc(), /*state_read=*/state_read,
                              /*value=*/state_read,
                              /*predicate=*/default_predicate,
                              absl::StrCat(state_element->name(), "_default"))
                          .status());
  return true;
}

absl::StatusOr<bool> AddDefaultNextValues(
    Proc* proc, const SchedulingPassOptions& options) {
  bool changed = false;

  for (StateElement* state_element : proc->StateElements()) {
    XLS_ASSIGN_OR_RETURN(bool state_changed,
                         AddDefaultNextValue(proc, state_element, options));
    if (state_changed) {
      VLOG(4) << "Added default next_value for state element: "
              << state_element->name();
      changed = true;
    }
  }

  return changed;
}

}  // namespace

absl::StatusOr<bool> ProcStateLegalizationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, SchedulingUnit* s, const SchedulingPassOptions& options,
    SchedulingPassResults* results) const {
  if (!f->IsProc()) {
    // Not a proc; no change needed.
    return false;
  }
  Proc* proc = f->AsProcOrDie();

  bool changed = false;

  XLS_ASSIGN_OR_RETURN(bool read_predicates_changed,
                       LegalizeStateReadPredicates(proc, options));
  if (read_predicates_changed) {
    changed = true;
  }

  XLS_ASSIGN_OR_RETURN(bool mutex_asserts_added,
                       AddMutualExclusionAsserts(proc, options));
  if (mutex_asserts_added) {
    changed = true;
  }

  XLS_ASSIGN_OR_RETURN(bool write_without_read_asserts_added,
                       AddWriteWithoutReadAsserts(proc, options));
  if (write_without_read_asserts_added) {
    changed = true;
  }

  XLS_ASSIGN_OR_RETURN(bool defaults_added,
                       AddDefaultNextValues(proc, options));
  if (defaults_added) {
    changed = true;
  }

  return changed;
}

}  // namespace xls
