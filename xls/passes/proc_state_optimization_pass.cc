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

#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/data_structures/union_find.h"
#include "xls/ir/bits.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

absl::StatusOr<bool> RemoveZeroWidthStateElements(Proc* proc) {
  std::vector<int64_t> to_remove;
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    if (proc->GetStateElementType(i)->GetFlatBitCount() == 0 &&
        !TypeHasToken(proc->GetStateElementType(i))) {
      to_remove.push_back(i);
    }
  }
  if (to_remove.empty()) {
    return false;
  }
  for (int64_t i : to_remove) {
    StateElement* state_element = proc->GetStateElement(i);
    VLOG(2) << "Removing zero-width state element: "
            << proc->GetStateElement(i)->name();
    StateRead* state_read = proc->GetStateRead(state_element);
    std::vector<Next*> next_values(proc->next_values(state_read).begin(),
                                   proc->next_values(state_read).end());
    for (Next* next : next_values) {
      XLS_RETURN_IF_ERROR(
          next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      XLS_RETURN_IF_ERROR(proc->RemoveNode(next));
    }
    XLS_RETURN_IF_ERROR(
        state_read->ReplaceUsesWithNew<Literal>(state_element->initial_value())
            .status());
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }
  return true;
}

absl::StatusOr<bool> RemoveConstantStateElements(Proc* proc,
                                                 QueryEngine& query_engine) {
  std::vector<int64_t> to_remove;
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    StateElement* state_element = proc->GetStateElement(i);
    StateRead* state_read = proc->GetStateRead(state_element);
    const Value& initial_value = state_element->initial_value();

    // TODO(epastor): Remove this once we no longer use next-state elements.
    if (proc->next_values(state_read).empty()) {
      Node* next_state = proc->GetNextStateElement(i);
      if (next_state == state_read) {
        // The state element never changes, so it's definitely constant.
        to_remove.push_back(i);
        continue;
      }

      std::optional<Value> next_state_value =
          query_engine.KnownValue(next_state);
      if (next_state_value.has_value() && *next_state_value == initial_value) {
        // We know the state never changes.
        to_remove.push_back(i);
      }
      continue;
    }

    bool never_changes = true;
    for (Next* next : proc->next_values(state_read)) {
      if (next->value() == state_read) {
        continue;
      }
      std::optional<Value> next_value = query_engine.KnownValue(next->value());
      if (!next_value.has_value() || *next_value != initial_value) {
        never_changes = false;
        break;
      }
    }
    if (never_changes) {
      to_remove.push_back(i);
    }
  }
  if (to_remove.empty()) {
    return false;
  }
  for (int64_t i : to_remove) {
    StateElement* state_element = proc->GetStateElement(i);
    Value value = state_element->initial_value();
    VLOG(2) << "Removing constant state element: " << state_element->name()
            << " (value: " << value.ToString() << ")";
    StateRead* state_read = proc->GetStateRead(state_element);
    std::vector<Next*> next_values(proc->next_values(state_read).begin(),
                                   proc->next_values(state_read).end());
    for (Next* next : next_values) {
      XLS_RETURN_IF_ERROR(
          next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      XLS_RETURN_IF_ERROR(proc->RemoveNode(next));
    }
    XLS_RETURN_IF_ERROR(
        state_read->ReplaceUsesWithNew<Literal>(value).status());
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }
  return true;
}

// A visitor which computes which state elements each node is dependent
// upon. Dependence is represented using an N-bit bit-vector where the i-th bit
// set indicates that the corresponding node is dependent upon the i-th state
// element. Dependence is tracked an a per leaf element basis using
// LeafTypeTrees.
class StateDependencyVisitor : public DataflowVisitor<InlineBitmap> {
 public:
  explicit StateDependencyVisitor(Proc* proc) : proc_(proc) {}

  absl::Status DefaultHandler(Node* node) override {
    // By default, conservatively assume that each element in `node` is
    // dependent upon all of the state elements which appear in the operands of
    // `node`.
    return SetValue(node, LeafTypeTree<InlineBitmap>(
                              node->GetType(), FlattenOperandBitmaps(node)));
  }

  absl::Status HandleStateRead(StateRead* state_read) override {
    // A state read is only dependent upon itself.
    XLS_ASSIGN_OR_RETURN(int64_t index, proc_->GetStateElementIndex(
                                            state_read->state_element()));
    InlineBitmap bitmap(proc_->GetStateElementCount());
    bitmap.Set(index, true);
    return SetValue(state_read,
                    LeafTypeTree<InlineBitmap>(state_read->GetType(), bitmap));
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

 protected:
  // We are interested in tracking the dependencies of the state elements so
  // union together all inputs (data and control sources) which represent which
  // state elements this node depends on.
  absl::StatusOr<InlineBitmap> JoinElements(
      Type* element_type, absl::Span<const InlineBitmap* const> data_sources,
      absl::Span<const LeafTypeTreeView<InlineBitmap>> control_sources,
      Node* node, absl::Span<const int64_t> index) const override {
    InlineBitmap element = *data_sources.front();
    for (const InlineBitmap* data_source : data_sources.subspan(1)) {
      element.Union(*data_source);
    }
    for (const LeafTypeTreeView<InlineBitmap>& control_source :
         control_sources) {
      XLS_RET_CHECK(IsLeafType(control_source.type()));
      element.Union(control_source.elements().front());
    }
    return std::move(element);
  }

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
  if (VLOG_IS_ON(3)) {
    VLOG(3) << "State dependencies (** side-effecting operation):";
    for (Node* node : TopoSort(proc)) {
      std::vector<std::string> dependent_elements;
      for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
        if (state_dependencies.at(node).Get(i)) {
          dependent_elements.push_back(proc->GetStateRead(i)->GetName());
        }
      }
      VLOG(3) << absl::StrFormat("  %s : {%s}%s", node->GetName(),
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

  // Map from node to the state element indices for which the node can affect
  // the next state value.
  absl::flat_hash_map<Node*, absl::flat_hash_set<int64_t>> next_state_indices;
  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    next_state_indices[proc->GetNextStateElement(i)].insert(i);
    for (Next* next : proc->next_values(proc->GetStateRead(i))) {
      next_state_indices[next->value()].insert(i);
      if (next->predicate().has_value()) {
        next_state_indices[*next->predicate()].insert(i);
      }
    }
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
    if (OpIsSideEffecting(node->op()) && !node->Is<StateRead>()) {
      // `node` is side-effecting. All state elements that `node` is dependent
      // on are observable, except if the only side effect is to change the
      // state element.
      for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
        if (!state_dependencies.at(node).Get(i)) {
          continue;
        }
        if (node->Is<Next>() &&
            node->As<Next>()->state_read() == proc->GetStateRead(i)) {
          // The only side-effect is to change this state element, so this
          // doesn't make the element observable.
          continue;
        }
        VLOG(4) << absl::StreamFormat(
            "State element `%s` (%d) is observable because side-effecting node "
            "`%s` depends on it",
            proc->GetStateRead(i)->GetName(), i, node->GetName());
        if (!observable_state_index.has_value()) {
          observable_state_index = i;
        } else {
          state_components.Union(i, observable_state_index.value());
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
            VLOG(4) << absl::StreamFormat(
                "Unioning state elements `%s` (%d) and `%s` (%d) because next "
                "state of `%s` (node `%s`) depends on `%s`",
                proc->GetStateElement(next_state_index)->name(),
                next_state_index, proc->GetStateElement(i)->name(), i,
                proc->GetStateElement(next_state_index)->name(),
                node->GetName(), proc->GetStateElement(i)->name());
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
  VLOG(3) << "Observability of state elements:";
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    if (!observable_state_index.has_value() ||
        state_components.Find(i) != observable_state_index.value()) {
      to_remove.push_back(i);
      VLOG(3) << absl::StrFormat("  %s (%d) : NOT observable",
                                 proc->GetStateElement(i)->name(), i);
    } else {
      VLOG(3) << absl::StrFormat("  %s (%d) : observable",
                                 proc->GetStateElement(i)->name(), i);
    }
  }
  if (to_remove.empty()) {
    return false;
  }

  // Replace uses of to-be-removed state elements with a zero-valued literal,
  // and remove their next_value nodes.
  for (int64_t i : to_remove) {
    StateRead* state_read = proc->GetStateRead(i);
    absl::btree_set<Next*, Node::NodeIdLessThan> next_values =
        proc->next_values(state_read);
    for (Next* next : next_values) {
      XLS_RETURN_IF_ERROR(
          next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      XLS_RETURN_IF_ERROR(proc->RemoveNode(next));
    }
    if (!state_read->IsDead()) {
      XLS_RETURN_IF_ERROR(
          state_read
              ->ReplaceUsesWithNew<Literal>(ZeroOfType(state_read->GetType()))
              .status());
    }
  }

  for (int64_t i : to_remove) {
    VLOG(2) << absl::StreamFormat("Removing dead state element %s of type %s",
                                  proc->GetStateElement(i)->name(),
                                  proc->GetStateElement(i)->type()->ToString());
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }
  return true;
}

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
  for (int64_t state_index : chain) {
    absl::StrAppend(&state_machine_name, "_",
                    proc->GetStateElement(state_index)->name());
  }

  int64_t state_machine_width = CeilOfLog2(chain.size()) + 1;
  Type* state_machine_type = proc->package()->GetBitsType(state_machine_width);
  XLS_ASSIGN_OR_RETURN(StateRead * state_machine_read,
                       proc->AppendStateElement(
                           state_machine_name, ZeroOfType(state_machine_type)));
  XLS_ASSIGN_OR_RETURN(
      int64_t state_machine_index,
      proc->GetStateElementIndex(state_machine_read->state_element()));

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
    // TODO(epastor): Clean this up once we no longer use next-state elements.
    if (proc->next_values().empty()) {
      XLS_RETURN_IF_ERROR(proc->SetNextStateElement(state_machine_index, sel));
    } else {
      XLS_RETURN_IF_ERROR(
          proc->MakeNode<Next>(SourceInfo(), /*state_read=*/state_machine_read,
                               /*value=*/sel, /*predicate=*/std::nullopt)
              .status());
    }
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

  Node* chain_constant;
  if (proc->next_values(proc->GetStateRead(chain.front())).empty()) {
    chain_constant = proc->GetNextStateElement(chain.front());
  } else {
    CHECK_EQ(proc->next_values(proc->GetStateRead(chain.front())).size(), 1);
    Next* next_value =
        *proc->next_values(proc->GetStateRead(chain.front())).begin();
    CHECK(next_value->predicate() == std::nullopt &&
          query_engine.IsFullyKnown(next_value->value()));
    chain_constant = next_value->value();
  }
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
        proc->next_values(proc->GetStateRead(state_index));
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
    if (query_engine.IsFullyKnown(proc->GetNextStateElement(i))) {
      XLS_RETURN_IF_ERROR(ConstantChainToStateMachine(proc, {i}, query_engine));
      changed = true;

      // Repopulate the query engine in case we need to use it again.
      XLS_RETURN_IF_ERROR(query_engine.Populate(proc).status());

      continue;
    }

    const absl::btree_set<Next*, Node::NodeIdLessThan>& next_values =
        proc->next_values(proc->GetStateRead(i));
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

absl::StatusOr<bool> ProcStateOptimizationPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;

  XLS_ASSIGN_OR_RETURN(bool zero_width_changed,
                       RemoveZeroWidthStateElements(proc));
  changed = changed || zero_width_changed;

  std::vector<std::unique_ptr<QueryEngine>> query_engines;
  query_engines.push_back(std::make_unique<StatelessQueryEngine>());
  query_engines.push_back(std::make_unique<TernaryQueryEngine>());
  UnionQueryEngine query_engine(std::move(query_engines));
  XLS_RETURN_IF_ERROR(query_engine.Populate(proc).status());

  // Run constant state-element removal to fixed point; should usually take just
  // one additional pass to verify, except for chains like next_s1 := s1,
  // next_s2 := f(s1), next_s3 := g(s1, s2), ..., etc., where the results all
  // match the state elements' initial values.
  bool constant_changed = false;
  do {
    XLS_ASSIGN_OR_RETURN(constant_changed,
                         RemoveConstantStateElements(proc, query_engine));
    if (constant_changed) {
      XLS_RETURN_IF_ERROR(query_engine.Populate(proc).status());
    }
    changed = changed || constant_changed;
  } while (constant_changed);

  XLS_ASSIGN_OR_RETURN(
      bool constant_chains_changed,
      ConvertConstantChainsToStateMachines(proc, query_engine));
  changed = changed || constant_chains_changed;

  XLS_ASSIGN_OR_RETURN(bool unobservable_changed,
                       RemoveUnobservableStateElements(proc));
  changed = changed || unobservable_changed;

  return changed;
}

REGISTER_OPT_PASS(ProcStateOptimizationPass);

}  // namespace xls
