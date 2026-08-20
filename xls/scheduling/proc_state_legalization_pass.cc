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
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
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
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/solvers/solver.h"

namespace xls {

namespace {

absl::StatusOr<bool> AddNextValueMutualExclusionAssert(
    Proc* proc, StateElement* state_element,
    const SchedulingPassOptions& options) {
  const absl::btree_set<Next*, Node::NodeIdLessThan>& next_values =
      proc->next_values(state_element);
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

absl::StatusOr<bool> AddStateReadMutualExclusionAssert(
    Proc* proc, StateElement* state_element,
    const SchedulingPassOptions& options) {
  absl::Span<StateRead* const> state_reads =
      proc->GetStateReadsByStateElement(state_element);
  if (state_reads.size() < 2) {
    return false;
  }
  std::string label =
      absl::StrCat("__", state_element->name(), "__at_most_one_read_assert");
  if (proc->HasNode(label)) {
    return absl::InternalError(absl::StrFormat(
        "Read mutual exclusion assert already exists for state "
        "element '%s'; was this pass run twice? assert label: %s",
        state_element->name(), label));
  }
  std::vector<Node*> predicate_list;
  Node* true_lit = nullptr;
  for (StateRead* state_read : state_reads) {
    if (state_read->predicate().has_value()) {
      predicate_list.push_back(*state_read->predicate());
    } else {
      if (true_lit == nullptr) {
        XLS_ASSIGN_OR_RETURN(
            true_lit, proc->MakeNode<Literal>(SourceInfo(), Value::Bool(true)));
      }
      predicate_list.push_back(true_lit);
    }
  }
  XLS_ASSIGN_OR_RETURN(
      Node * predicates,
      proc->MakeNodeWithName<Concat>(
          SourceInfo(), predicate_list,
          absl::StrCat("__", state_element->name(), "__read_predicates")));
  XLS_ASSIGN_OR_RETURN(
      Node * one_hot_predicates,
      proc->MakeNode<OneHot>(SourceInfo(), predicates, LsbOrMsb::kLsb));
  XLS_ASSIGN_OR_RETURN(Node * at_most_one_predicate,
                       proc->MakeNode<BitSlice>(
                           SourceInfo(), one_hot_predicates, /*start=*/0,
                           /*width=*/one_hot_predicates->BitCountOrDie() - 1));
  XLS_ASSIGN_OR_RETURN(
      Node * at_most_one_read,
      proc->MakeNodeWithName<CompareOp>(
          SourceInfo(), predicates, at_most_one_predicate, Op::kEq,
          absl::StrCat("__", state_element->name(), "__at_most_one_read")));
  XLS_ASSIGN_OR_RETURN(Node * tkn,
                       proc->MakeNode<Literal>(SourceInfo(), Value::Token()));
  XLS_RETURN_IF_ERROR(
      proc->MakeNodeWithName<Assert>(
              SourceInfo(), tkn,
              /*condition=*/at_most_one_read,
              /*message=*/
              absl::StrCat("More than one StateRead active for state element: ",
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
    XLS_ASSIGN_OR_RETURN(
        bool write_assert_added,
        AddNextValueMutualExclusionAssert(proc, state_element, options));
    if (write_assert_added) {
      VLOG(4) << "Added next_value mutual exclusion assert for state element: "
              << state_element->name();
      changed = true;
    }
    XLS_ASSIGN_OR_RETURN(
        bool read_assert_added,
        AddStateReadMutualExclusionAssert(proc, state_element, options));
    if (read_assert_added) {
      VLOG(4) << "Added state_read mutual exclusion assert for state element: "
              << state_element->name();
      changed = true;
    }
  }

  return changed;
}

absl::StatusOr<bool> AddWriteWithoutReadAsserts(
    Proc* proc, StateElement* state_element,
    const SchedulingPassOptions& options) {
  absl::Span<StateRead* const> state_reads =
      proc->GetStateReadsByStateElement(state_element);

  const absl::btree_set<Next*, Node::NodeIdLessThan>& next_values =
      proc->next_values(state_element);
  if (next_values.empty()) {
    return false;
  }
  std::vector<Node*> state_read_predicates;
  for (StateRead* state_read : state_reads) {
    if (state_read->predicate().has_value()) {
      state_read_predicates.push_back(*state_read->predicate());
    } else {
      // State read is unconditional, so no write-without-read assert needed.
      return false;
    }
  }
  // If there are multiple state reads, we need to OR their predicates together
  // to see if any of them are active.
  XLS_ASSIGN_OR_RETURN(
      Node * any_read_active,
      NaryOrIfNeeded(proc, state_read_predicates,
                     absl::StrCat("__", state_element->name(),
                                  "__state_read_predicates_nary_or"),
                     SourceInfo()));

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
            absl::MakeConstSpan({any_read_active, next_not_triggered}), Op::kOr,
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
  for (Next* next : proc->next_values(state_element)) {
    if (next->predicate().has_value()) {
      predicates.insert(*next->predicate());
    } else {
      // Unconditional next_value; no default next_value needed.
      return false;
    }
  }
  absl::Span<StateRead* const> state_reads =
      proc->GetStateReadsByStateElement(state_element);
  // No explicit `next_value` node; leave the state element unchanged by
  // default.
  if (predicates.empty()) {
    for (StateRead* state_read : state_reads) {
      XLS_RETURN_IF_ERROR(
          proc->MakeNodeWithName<Next>(
                  state_read->loc(), state_element,
                  /*value=*/state_read,
                  /*predicate=*/state_read->predicate(),
                  /*label=*/std::nullopt,
                  absl::StrCat(state_element->name(), "_default_",
                               state_read->GetName()))
              .status());
    }
    return true;
  }

  auto get_underlying_default_predicate = [](Next* next) -> Node* {
    StateRead* state_read = next->value()->As<StateRead>();
    Node* pred = *next->predicate();
    if (state_read->predicate().has_value() && pred->OpIn({Op::kAnd}) &&
        pred->operands().size() == 2) {
      if (pred->operand(0) == *state_read->predicate()) {
        return pred->operand(1);
      }
      if (pred->operand(1) == *state_read->predicate()) {
        return pred->operand(0);
      }
    }
    return pred;
  };

  // Check if we already have an explicit "if nothing else fires" `next_value`
  // node, which keeps things cleaner and makes sure this pass is idempotent.
  for (Next* next : proc->next_values(state_element)) {
    if (!IsNoOpNext(next) || !next->predicate().has_value()) {
      continue;
    }

    Node* predicate = get_underlying_default_predicate(next);
    if (!predicate->OpIn({Op::kNot, Op::kNor})) {
      continue;
    }

    // Remove all next_value predicates that use this default condition from
    // `other_conditions`, leaving only the explicit write conditions.
    absl::btree_set<Node*, Node::NodeIdLessThan> other_conditions = predicates;
    for (Next* other_next : proc->next_values(state_element)) {
      if (!IsNoOpNext(other_next) || !other_next->predicate().has_value()) {
        continue;
      }
      if (get_underlying_default_predicate(other_next) == predicate) {
        other_conditions.erase(*other_next->predicate());
      }
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
    std::vector<solvers::PredicateOfNode> solver_predicates;
    solver_predicates.reserve(predicates.size());
    for (Node* predicate : predicates) {
      solver_predicates.push_back({
          .subject = predicate,
          .p = solvers::Predicate::NotEqualToZero(),
      });
    }

    XLS_ASSIGN_OR_RETURN(
        std::unique_ptr<solvers::Solver> solver,
        solvers::CreateSolver(options.scheduling_options.solver_kind()));
    solvers::SolverLimit solver_limit;
    solver_limit.deterministic_limit = *default_next_value_z3_rlimit;

    absl::StatusOr<solvers::ProverResult> no_default_needed =
        solver->TryProveCombination(proc, solver_predicates,
                                    solvers::PredicateCombination::kDisjunction,
                                    solver_limit, /*allow_unsupported=*/true);
    if (no_default_needed.ok() &&
        std::holds_alternative<solvers::ProvenTrue>(*no_default_needed)) {
      return false;
    }
  }

  // Explicitly mark the state element as unchanged when no other `next_value`
  // node is active.
  XLS_ASSIGN_OR_RETURN(
      Node * no_explicit_next_active,
      NaryNorIfNeeded(proc, std::vector(predicates.begin(), predicates.end()),
                      absl::StrCat("__", state_element->name(),
                                   "__no_explicit_next_active"),
                      SourceInfo()));
  for (StateRead* state_read : state_reads) {
    Node* default_predicate = no_explicit_next_active;
    if (state_read->predicate().has_value()) {
      XLS_ASSIGN_OR_RETURN(
          default_predicate,
          proc->MakeNode<NaryOp>(state_read->loc(),
                                 absl::MakeConstSpan({*state_read->predicate(),
                                                      no_explicit_next_active}),
                                 Op::kAnd));
    }
    XLS_RETURN_IF_ERROR(proc->MakeNodeWithName<Next>(
                                state_read->loc(), state_element,
                                /*value=*/state_read,
                                /*predicate=*/default_predicate,
                                /*label=*/std::nullopt,
                                absl::StrCat(state_element->name(), "_default_",
                                             state_read->GetName()))
                            .status());
  }
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
    FunctionBase* f, const SchedulingPassOptions& options, PassResults* results,
    SchedulingContext& context) const {
  if (!f->IsProc()) {
    // Not a proc; no change needed.
    return false;
  }
  Proc* proc = f->AsProcOrDie();

  bool changed = false;

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
