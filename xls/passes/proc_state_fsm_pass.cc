// Copyright 2022 The XLS Authors
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

#include "xls/passes/proc_state_fsm_pass.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

// If there's a sequence of state elements `c` with length `k` such that
//     next[c[i + 1]] ≡ state_read[c[i]] and next[c[0]] is a constant
// where `≡` denotes semantic equivalence, then this function will convert all
// of those state elements into a single state element of size ⌈log₂(k)⌉ bits,
// unless the constant value is equal to init_value[c[0]], in which case the
// state will be eliminated entirely.
//
// The reason this takes a chain as input rather than a single state element
// with constant input (and then run to fixed point) is because the latter would
// result in a one-hot encoding of the state rather than binary.
//
// TODO: 2022-08-31 this could be modified to handle arbitrary DAGs where each
// included next function doesn't have any receives (i.e.: a DAG consisting of
// arithmetic/logic operations, constants, and registers).
absl::Status ConstantChainToStateMachine(Proc* proc,
                                         absl::Span<const int64_t> chain,
                                         const QueryEngine& query_engine) {
  CHECK(!chain.empty());

  std::string state_machine_name = "state_machine";
  bool non_synthesizable = false;
  for (int64_t state_index : chain) {
    absl::StrAppend(&state_machine_name, "_",
                    proc->GetStateElement(state_index)->name());
    if (proc->GetStateElement(state_index)->non_synthesizable()) {
      non_synthesizable = true;
    }
  }

  int64_t state_machine_width = CeilOfLog2(chain.size()) + 1;
  Type* state_machine_type = proc->package()->GetBitsType(state_machine_width);
  XLS_ASSIGN_OR_RETURN(StateRead * state_machine_read,
                       proc->AppendStateElement(
                           state_machine_name, ZeroOfType(state_machine_type),
                           /*read_predicate=*/std::nullopt,
                           /*next_state=*/std::nullopt, non_synthesizable));

  {
    XLS_ASSIGN_OR_RETURN(
        Node * one, proc->MakeNode<Literal>(
                        SourceInfo(), Value(UBits(1, state_machine_width))));
    XLS_ASSIGN_OR_RETURN(
        Node * max,
        proc->MakeNode<Literal>(
            SourceInfo(), Value(UBits(chain.size() - 1, state_machine_width))));
    XLS_ASSIGN_OR_RETURN(
        Node * machine_plus_one,
        proc->MakeNode<BinOp>(SourceInfo(), state_machine_read, one, Op::kAdd));
    XLS_ASSIGN_OR_RETURN(Node * machine_too_large,
                         proc->MakeNode<CompareOp>(
                             SourceInfo(), state_machine_read, max, Op::kUGt));
    XLS_ASSIGN_OR_RETURN(
        Node * sel,
        proc->MakeNode<Select>(
            SourceInfo(), machine_too_large,
            std::vector<Node*>({machine_plus_one, state_machine_read}),
            std::nullopt));
    XLS_RETURN_IF_ERROR(
        proc->MakeNode<Next>(SourceInfo(), state_machine_read->state_element(),
                             /*value=*/sel,
                             /*predicate=*/std::nullopt, /*label=*/std::nullopt)
            .status());
  }

  std::vector<Node*> initial_state_literals;
  initial_state_literals.reserve(chain.size());
  for (int64_t state_index : chain) {
    XLS_ASSIGN_OR_RETURN(
        Node * init,
        proc->MakeNode<Literal>(
            SourceInfo(), proc->GetStateElement(state_index)->initial_value()));
    initial_state_literals.push_back(init);
  }

  CHECK_EQ(proc->next_values(proc->GetStateElement(chain.front())).size(), 1);
  Next* next_value =
      *proc->next_values(proc->GetStateElement(chain.front())).begin();
  CHECK(next_value->predicate() == std::nullopt &&
        query_engine.IsFullyKnown(next_value->value()));
  Node* chain_constant = next_value->value();
  CHECK(chain_constant != nullptr && query_engine.IsFullyKnown(chain_constant));
  XLS_ASSIGN_OR_RETURN(
      Literal * chain_literal,
      proc->MakeNode<Literal>(chain_constant->loc(),
                              *query_engine.KnownValue(chain_constant)));

  absl::btree_set<int64_t, std::greater<int64_t>> indices_to_remove;
  for (int64_t chain_index = 0; chain_index < chain.size(); ++chain_index) {
    int64_t state_index = chain.at(chain_index);
    std::vector<Node*> cases = initial_state_literals;
    CHECK_GE(cases.size(), chain_index);
    cases.resize(chain_index + 1);
    std::reverse(cases.begin(), cases.end());
    absl::btree_set<Next*, Node::NodeIdLessThan> next_values =
        proc->next_values(proc->GetStateElement(state_index));
    for (Next* next : next_values) {
      XLS_RETURN_IF_ERROR(
          next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      XLS_RETURN_IF_ERROR(proc->RemoveNode(next));
    }
    XLS_RETURN_IF_ERROR(proc->GetStateRead(state_index)
                            ->ReplaceUsesWithNew<Select>(state_machine_read,
                                                         cases, chain_literal)
                            .status());
    indices_to_remove.insert(state_index);
  }
  for (int64_t state_index : indices_to_remove) {
    VLOG(4) << "Removing state element " << proc->StateElements()[state_index]
            << " for being converted to state machine.";
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(state_index));
  }

  return absl::OkStatus();
}

// Convert all chains in the state element graph (as described in the docs for
// `ConstantChainToStateMachine`) into state machines with `⌈log₂(k)⌉` bits of
// state where `k` is the length of the chain.
//
// TODO: 2022-08-31 this currently only handles chains of length 1 with
// syntactic equivalence
absl::StatusOr<bool> ConvertConstantChainsToStateMachines(
    Proc* proc, QueryEngine& query_engine) {
  bool changed = false;
  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    const absl::btree_set<Next*, Node::NodeIdLessThan>& next_values =
        proc->next_values(proc->GetStateElement(i));
    if (next_values.size() != 1) {
      continue;
    }
    Next* next_value = *next_values.begin();
    if (next_value->predicate() == std::nullopt &&
        query_engine.IsFullyKnown(next_value->value())) {
      XLS_RETURN_IF_ERROR(ConstantChainToStateMachine(proc, {i}, query_engine));
      changed = true;

      // Repopulate the query engine in case we need to use it again.
      XLS_RETURN_IF_ERROR(query_engine.Populate(proc).status());

      continue;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> ProcStateFSMPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  auto query_engine = UnionQueryEngine::Of(
      StatelessQueryEngine(),
      GetSharedQueryEngine<LazyTernaryQueryEngine>(context, proc));

  XLS_ASSIGN_OR_RETURN(
      bool constant_chains_changed,
      ConvertConstantChainsToStateMachines(proc, query_engine));
  return constant_chains_changed;
}

}  // namespace xls
