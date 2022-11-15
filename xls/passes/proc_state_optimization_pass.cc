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

#include "xls/passes/proc_state_optimization_pass.h"

#include "absl/strings/str_join.h"
#include "xls/common/logging/logging.h"
#include "xls/common/math_util.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/union_find.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/node_util.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value_helpers.h"
#include "xls/passes/dataflow_visitor.h"

namespace xls {
namespace {

absl::StatusOr<bool> RemoveZeroWidthStateElements(Proc* proc) {
  std::vector<int64_t> to_remove;
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    if (proc->GetStateElementType(i)->GetFlatBitCount() == 0) {
      to_remove.push_back(i);
    }
  }
  if (to_remove.empty()) {
    return false;
  }
  for (int64_t i : to_remove) {
    XLS_VLOG(2) << "Removing zero-width state element: "
                << proc->GetStateParam(i)->GetName();
    XLS_RETURN_IF_ERROR(proc->GetStateParam(i)
                            ->ReplaceUsesWithNew<Literal>(
                                ZeroOfType(proc->GetStateElementType(i)))
                            .status());
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }
  return true;
}

// A visitor which computes which state elements each node is dependent
// upon. Dependence is represented using an N-bit bit-vector where the i-th bit
// set indicates that the corresponding node is dependent upon the i-th state
// parameter. Dependence is tracked an a per leaf element basis using
// LeafTypeTrees.
class StateDependencyVisitor : public DataFlowVisitor<InlineBitmap> {
 public:
  explicit StateDependencyVisitor(Proc* proc) : proc_(proc) {}

  absl::Status DefaultHandler(Node* node) override {
    // By default, conservatively assume that each element in `node` is
    // dependent upon all of the state elements which appear in the operands of
    // `node`.
    return SetValue(node, LeafTypeTree<InlineBitmap>(
                              node->GetType(), FlattenOperandBitmaps(node)));
  }

  absl::Status HandleParam(Param* param) override {
    if (param == proc_->TokenParam()) {
      return DefaultHandler(param);
    }
    // A state parameter is only dependent upon itself.
    XLS_ASSIGN_OR_RETURN(int64_t index, proc_->GetStateParamIndex(param));
    InlineBitmap bitmap(proc_->GetStateElementCount());
    bitmap.Set(index, true);
    return SetValue(param,
                    LeafTypeTree<InlineBitmap>(param->GetType(), bitmap));
  }

  // Returns the union of all of the bitmaps in the LeafTypeTree for all of the
  // operands of `node`.
  InlineBitmap FlattenOperandBitmaps(Node* node) {
    InlineBitmap result(proc_->GetStateElementCount());
    for (Node* operand : node->operands()) {
      for (const InlineBitmap& bitmap : GetValue(operand).elements()) {
        result.Union(bitmap);
      }
    }
    return result;
  }

  // Returns the union of all of the bitmaps in the LeafTypeTree for `node`.
  InlineBitmap FlattenNodeBitmaps(Node* node) {
    InlineBitmap result(proc_->GetStateElementCount());
    for (const InlineBitmap& bitmap : GetValue(node).elements()) {
      result.Union(bitmap);
    }
    return result;
  }

 private:
  Proc* proc_;
};

// Computes which state elements each node is dependent upon. Dependence is
// represented as a bit-vector with one bit per state element in the proc.
// Dependencies are only computed in a single forward pass so dependencies
// through the proc back edge are not considered.
absl::StatusOr<absl::flat_hash_map<Node*, InlineBitmap>>
ComputeStateDependencies(Proc* proc) {
  StateDependencyVisitor visitor(proc);
  XLS_RETURN_IF_ERROR(proc->Accept(&visitor));
  absl::flat_hash_map<Node*, InlineBitmap> state_dependencies;
  for (Node* node : proc->nodes()) {
    state_dependencies.insert({node, visitor.FlattenNodeBitmaps(node)});
  }
  if (XLS_VLOG_IS_ON(3)) {
    XLS_VLOG(3) << "State dependencies (** side-effecting operation):";
    for (Node* node : TopoSort(proc)) {
      std::vector<std::string> dependent_elements;
      for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
        if (state_dependencies.at(node).Get(i)) {
          dependent_elements.push_back(proc->GetStateParam(i)->GetName());
        }
      }
      XLS_VLOG(3) << absl::StrFormat("  %s : {%s}%s", node->GetName(),
                                     absl::StrJoin(dependent_elements, ", "),
                                     OpIsSideEffecting(node->op()) ? "**" : "");
    }
  }
  return std::move(state_dependencies);
}

// Removes unobservable state elements. A state element X is observable if:
//   (1) a side-effecting operation depends on X, OR
//   (2) the next-state value of an observable state element depends on X.
absl::StatusOr<bool> RemoveUnobservableStateElements(Proc* proc) {
  absl::flat_hash_map<Node*, InlineBitmap> state_dependencies;
  XLS_ASSIGN_OR_RETURN(state_dependencies, ComputeStateDependencies(proc));

  // Map from node to the state element indices for which the node is the
  // next-state value.
  absl::flat_hash_map<Node*, std::vector<int64_t>> next_state_indices;
  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    next_state_indices[proc->GetNextStateElement(i)].push_back(i);
  }

  // The equivalence classes of state element indices. State element X is in the
  // same class as Y if the next-state value of X depends on Y or vice versa.
  UnionFind<int64_t> state_components;
  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    state_components.Insert(i);
  }

  // At the end, the union-find data structure will have one equivalence class
  // corresponding to the set of all observable state indices. This value is
  // always either `std::nullopt` or an element of that equivalence class. We
  // won't have a way to represent the equivalence class until it contains at
  // least one value, so we use `std::optional`.
  std::optional<int64_t> observable_state_index;

  // Merge state elements which depend on each other and identify observable
  // state indices.
  for (Node* node : proc->nodes()) {
    if (OpIsSideEffecting(node->op()) && !node->Is<Param>()) {
      // `node` is side-effecting. All state elements that `node` is dependent
      // on are observable.
      for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
        if (state_dependencies.at(node).Get(i)) {
          XLS_VLOG(4) << absl::StreamFormat(
              "State element `%s` (%d) is observable because side-effecting "
              "node `%s` depends on it",
              proc->GetStateParam(i)->GetName(), i, node->GetName());
          if (!observable_state_index.has_value()) {
            observable_state_index = i;
          } else {
            state_components.Union(i, observable_state_index.value());
          }
        }
      }
    }
    if (next_state_indices.contains(node)) {
      for (int64_t next_state_index : next_state_indices.at(node)) {
        // `node` is the next state node for state element with index
        // `next_state_index`. Union `next_state_index` with each state index
        // that `node` is dependent on.
        for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
          if (state_dependencies.at(node).Get(i)) {
            XLS_VLOG(4) << absl::StreamFormat(
                "Unioning state elements `%s` (%d) and `%s` (%d) because next "
                "state of `%s` (node `%s`) depends on `%s`",
                proc->GetStateParam(next_state_index)->GetName(),
                next_state_index, proc->GetStateParam(i)->GetName(), i,
                proc->GetStateParam(next_state_index)->GetName(),
                node->GetName(), proc->GetStateParam(i)->GetName());
            state_components.Union(i, next_state_index);
          }
        }
      }
    }
  }
  if (observable_state_index.has_value()) {
    // Set to the representative value of the union-find data structure.
    observable_state_index =
        state_components.Find(observable_state_index.value());
  }

  // Gather unobservable state element indices into `to_remove`.
  std::vector<int64_t> to_remove;
  to_remove.reserve(proc->GetStateElementCount());
  XLS_VLOG(3) << "Observability of state elements:";
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    if (!observable_state_index.has_value() ||
        state_components.Find(i) != observable_state_index.value()) {
      to_remove.push_back(i);
      XLS_VLOG(3) << absl::StrFormat("  %s (%d) : NOT observable",
                                     proc->GetStateParam(i)->GetName(), i);
    } else {
      XLS_VLOG(3) << absl::StrFormat("  %s (%d) : observable",
                                     proc->GetStateParam(i)->GetName(), i);
    }
  }
  if (to_remove.empty()) {
    return false;
  }

  // Replace uses of to-be-removed state parameters with a zero-valued literal.
  for (int64_t i : to_remove) {
    Param* state_param = proc->GetStateParam(i);
    if (!state_param->IsDead()) {
      XLS_RETURN_IF_ERROR(
          state_param
              ->ReplaceUsesWithNew<Literal>(ZeroOfType(state_param->GetType()))
              .status());
    }
  }

  for (int64_t i : to_remove) {
    XLS_VLOG(2) << absl::StreamFormat(
        "Removing dead state element %s of type %s",
        proc->GetStateParam(i)->GetName(),
        proc->GetStateParam(i)->GetType()->ToString());
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }
  return true;
}

// If there's a sequence of state elements `c` with length `k` such that
//     next[c[i + 1]] ≡ param[c[i]] and next[c[0]] is a literal
// where `≡` denotes semantic equivalence, then this function will convert all
// of those state elements into a single state element of size ⌈log₂(k)⌉ bits,
// unless the literal value is equal to init_value[c[0]], in which case the
// state will be eliminated entirely.
//
// The reason this takes a chain as input rather than a single state element
// with literal input (and then run to fixed point) is because the latter would
// result in a one-hot encoding of the state rather than binary.
//
// TODO: 2022-08-31 this could be modified to handle arbitrary DAGs where each
// included next function doesn't have any receives (i.e.: a DAG consisting of
// arithmetic/logic operations, literals, and registers).
absl::Status LiteralChainToStateMachine(Proc* proc,
                                        absl::Span<const int64_t> chain) {
  XLS_CHECK(!chain.empty());

  std::string state_machine_name = "state_machine";
  for (int64_t param_index : chain) {
    absl::StrAppend(&state_machine_name, "_",
                    proc->GetStateParam(param_index)->GetName());
  }

  int64_t state_machine_width = CeilOfLog2(chain.size()) + 1;
  Type* state_machine_type = proc->package()->GetBitsType(state_machine_width);
  XLS_ASSIGN_OR_RETURN(Param * state_machine_param,
                       proc->AppendStateElement(
                           state_machine_name, ZeroOfType(state_machine_type)));
  XLS_ASSIGN_OR_RETURN(int64_t state_machine_index,
                       proc->GetStateParamIndex(state_machine_param));

  {
    XLS_ASSIGN_OR_RETURN(
        Node * one, proc->MakeNode<Literal>(
                        SourceInfo(), Value(UBits(1, state_machine_width))));
    XLS_ASSIGN_OR_RETURN(
        Node * max,
        proc->MakeNode<Literal>(
            SourceInfo(), Value(UBits(chain.size() - 1, state_machine_width))));
    XLS_ASSIGN_OR_RETURN(Node * machine_plus_one,
                         proc->MakeNode<BinOp>(
                             SourceInfo(), state_machine_param, one, Op::kAdd));
    XLS_ASSIGN_OR_RETURN(Node * machine_too_large,
                         proc->MakeNode<CompareOp>(
                             SourceInfo(), state_machine_param, max, Op::kUGt));
    XLS_ASSIGN_OR_RETURN(
        Node * sel,
        proc->MakeNode<Select>(
            SourceInfo(), machine_too_large,
            std::vector<Node*>({machine_plus_one, state_machine_param}),
            std::nullopt));
    XLS_RETURN_IF_ERROR(proc->SetNextStateElement(state_machine_index, sel));
  }

  std::vector<Node*> initial_state_literals;
  initial_state_literals.reserve(chain.size());
  for (int64_t param_index : chain) {
    XLS_ASSIGN_OR_RETURN(
        Node * init, proc->MakeNode<Literal>(
                         SourceInfo(), proc->GetInitValueElement(param_index)));
    initial_state_literals.push_back(init);
  }

  Node* chain_literal = proc->GetNextStateElement(chain.at(0));
  XLS_CHECK(chain_literal->Is<Literal>());

  for (int64_t chain_index = 0; chain_index < chain.size(); ++chain_index) {
    int64_t param_index = chain.at(chain_index);
    std::vector<Node*> cases = initial_state_literals;
    XLS_CHECK_GE(cases.size(), chain_index);
    cases.resize(chain_index + 1);
    std::reverse(cases.begin(), cases.end());
    XLS_RETURN_IF_ERROR(proc->GetStateParam(param_index)
                            ->ReplaceUsesWithNew<Select>(state_machine_param,
                                                         cases, chain_literal)
                            .status());
  }

  return absl::OkStatus();
}

// Convert all chains in the state element graph (as described in the docs for
// `LiteralChainToStateMachine`) into state machines with `⌈log₂(k)⌉` bits of
// state where `k` is the length of the chain.
//
// TODO: 2022-08-31 this currently only handles chains of length 1 with
// syntactic equivalence
absl::StatusOr<bool> ConvertLiteralChainsToStateMachines(Proc* proc) {
  bool changed = false;
  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    if (proc->GetNextStateElement(i)->Is<Literal>()) {
      XLS_RETURN_IF_ERROR(LiteralChainToStateMachine(proc, {i}));
      changed = true;
    }
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> ProcStateOptimizationPass::RunOnProcInternal(
    Proc* proc, const PassOptions& options, PassResults* results) const {
  bool changed = false;

  XLS_ASSIGN_OR_RETURN(bool zero_width_changed,
                       RemoveZeroWidthStateElements(proc));
  changed |= zero_width_changed;

  XLS_ASSIGN_OR_RETURN(bool literal_chains_changed,
                       ConvertLiteralChainsToStateMachines(proc));
  changed |= literal_chains_changed;

  XLS_ASSIGN_OR_RETURN(bool unobservable_changed,
                       RemoveUnobservableStateElements(proc));
  changed |= unobservable_changed;

  return changed;
}

}  // namespace xls
