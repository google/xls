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
#include <variant>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/scheduling/scheduling_pass.h"
#include "xls/solvers/z3_ir_translator.h"

namespace xls {

namespace {

absl::StatusOr<bool> ModernizeNextValues(Proc* proc) {
  CHECK(proc->next_values().empty());

  for (int64_t index = 0; index < proc->GetStateElementCount(); ++index) {
    StateRead* state_read = proc->GetStateRead(index);
    Node* next_value = proc->GetNextStateElement(index);
    XLS_RETURN_IF_ERROR(
        proc->MakeNodeWithName<Next>(
                state_read->loc(), /*state_read=*/state_read,
                /*value=*/next_value,
                /*predicate=*/std::nullopt,
                absl::StrCat(state_read->state_element()->name(), "_next"))
            .status());

    if (next_value != static_cast<Node*>(state_read)) {
      // Nontrivial next-state element; remove it so we pass verification.
      XLS_RETURN_IF_ERROR(proc->SetNextStateElement(index, state_read));
    }
  }

  return proc->GetStateElementCount() > 0;
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
    // No explicit `next_value` node; leave the state parameter unchanged by
    // default.
    XLS_RETURN_IF_ERROR(proc->MakeNodeWithName<Next>(
                                state_read->loc(), /*state_read=*/state_read,
                                /*value=*/state_read,
                                /*predicate=*/std::nullopt,
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

  // Explicitly mark the param as unchanged when no other `next_value` node is
  // active.
  XLS_ASSIGN_OR_RETURN(
      Node * all_predicates_false,
      NaryNorIfNeeded(proc, std::vector(predicates.begin(), predicates.end()),
                      /*name=*/"", state_read->loc()));
  XLS_RETURN_IF_ERROR(proc->MakeNodeWithName<Next>(
                              state_read->loc(), /*state_read=*/state_read,
                              /*value=*/state_read,
                              /*predicate=*/all_predicates_false,
                              absl::StrCat(state_element->name(), "_default"))
                          .status());
  return true;
}

absl::StatusOr<bool> AddDefaultNextValues(
    Proc* proc, const SchedulingPassOptions& options) {
  bool changed = false;

  for (StateElement* state_element : proc->StateElements()) {
    XLS_ASSIGN_OR_RETURN(bool param_changed,
                         AddDefaultNextValue(proc, state_element, options));
    if (param_changed) {
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

  // Convert old-style `next (...)` value handling into explicit nodes.
  // TODO(epastor): Clean up after removing support for `next (...)` values.
  if (proc->next_values().empty()) {
    // No need to legalize; all of the nodes will be unconditional.
    return ModernizeNextValues(proc);
  }

  return AddDefaultNextValues(proc, options);
}

}  // namespace xls
