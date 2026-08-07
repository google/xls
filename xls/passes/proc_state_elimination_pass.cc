// Copyright 2026 The XLS Authors
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

#include "xls/passes/proc_state_elimination_pass.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "cppitertools/zip.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/data_structures/transitive_closure.h"
#include "xls/ir/bits.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/dataflow_visitor.h"
#include "xls/passes/lazy_ternary_query_engine.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

bool IsSideEffectingOrInvoke(Node* node) {
  return OpIsSideEffecting(node->op()) || node->OpIn({Op::kInvoke});
};
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
    absl::Span<StateRead* const> state_reads =
        proc->GetStateReadsByStateElement(state_element);
    std::vector<Next*> next_values(proc->next_values(state_element).begin(),
                                   proc->next_values(state_element).end());
    for (Next* next : next_values) {
      XLS_RETURN_IF_ERROR(
          next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      XLS_RETURN_IF_ERROR(proc->RemoveNode(next));
    }
    for (StateRead* state_read : state_reads) {
      XLS_RETURN_IF_ERROR(
          state_read
              ->ReplaceUsesWithNew<Literal>(state_element->initial_value())
              .status());
    }
    VLOG(4) << "Removing state element " << proc->StateElements()[i]
            << " for being zero width.";
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }
  return true;
}

absl::StatusOr<bool> RemoveConstantStateElements(Proc* proc,
                                                 QueryEngine& query_engine) {
  std::vector<int64_t> to_remove;
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    StateElement* state_element = proc->GetStateElement(i);
    absl::Span<StateRead* const> state_reads =
        proc->GetStateReadsByStateElement(state_element);
    const Value& initial_value = state_element->initial_value();

    bool never_changes = true;
    for (Next* next : proc->next_values(state_element)) {
      if (absl::c_linear_search(state_reads, next->value())) {
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
    absl::Span<StateRead* const> state_reads =
        proc->GetStateReadsByStateElement(state_element);
    std::vector<Next*> next_values(proc->next_values(state_element).begin(),
                                   proc->next_values(state_element).end());
    for (Next* next : next_values) {
      XLS_RETURN_IF_ERROR(
          next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      XLS_RETURN_IF_ERROR(proc->RemoveNode(next));
    }
    for (StateRead* state_read : state_reads) {
      XLS_RETURN_IF_ERROR(
          state_read->ReplaceUsesWithNew<Literal>(value).status());
    }
    VLOG(4) << "Removing state element " << proc->StateElements()[i]
            << " for being constant.";
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
    return SetUnifiedValue(node, FlattenOperandBitmaps(node));
  }

  absl::Status HandleStateRead(StateRead* state_read) override {
    // A state read is only dependent upon itself.
    XLS_ASSIGN_OR_RETURN(int64_t index, proc_->GetStateElementIndex(
                                            state_read->state_element()));
    InlineBitmap bitmap(proc_->GetStateElementCount());
    bitmap.Set(index, true);
    return SetUnifiedValue(state_read, std::move(bitmap));
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
    if (node_values_.contains(node)) {
      return node_values_.at(node);
    }
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
      Node* node, absl::Span<const int64_t> index) override {
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

 private:
  absl::Status SetUnifiedValue(Node* node, InlineBitmap&& bitmap) {
    XLS_RETURN_IF_ERROR(
        SetValue(node, LeafTypeTree<InlineBitmap>(node->GetType(), bitmap)));
    node_values_.emplace(node, std::move(bitmap));
    return absl::OkStatus();
  }

  Proc* proc_;
  // The value of the node as a whole. Since even return-less nodes can have
  // values we want to track them separately.
  absl::flat_hash_map<Node*, InlineBitmap> node_values_;
};

// Computes which state elements each node is dependent upon. Dependence is
// represented as a bit-vector with one bit per state element in the proc.
// Dependencies are only computed in a single forward pass so dependencies
// through the proc back edge are not considered.
absl::StatusOr<absl::flat_hash_map<Node*, InlineBitmap>>
ComputeStateDependencies(Proc* proc, OptimizationContext& context) {
  StateDependencyVisitor visitor(proc);
  XLS_RETURN_IF_ERROR(proc->Accept(&visitor));
  absl::flat_hash_map<Node*, InlineBitmap> state_dependencies;
  // NB We can't just take node values because things like tuple/next nodes
  // won't be included.
  for (Node* node : proc->nodes()) {
    state_dependencies.insert({node, visitor.FlattenNodeBitmaps(node)});
    VLOG(5) << "Got " << node << " -> "
            << Bits::FromBitmap(state_dependencies.at(node)).ToDebugString();
  }
  if (VLOG_IS_ON(5)) {
    VLOG(5) << "State dependencies (** side-effecting operation):";
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> topo_sort_nodes,
                         context.TopoSort(proc));
    for (Node* node : topo_sort_nodes) {
      std::vector<std::string> dependent_elements;
      for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
        if (state_dependencies.at(node).Get(i)) {
          dependent_elements.push_back(proc->GetStateRead(i)->GetName());
        }
      }
      VLOG(5) << absl::StrFormat("  %s : {%s}%s", node->GetName(),
                                 absl::StrJoin(dependent_elements, ", "),
                                 IsSideEffectingOrInvoke(node) ? "**" : "");
    }
  }
  return std::move(state_dependencies);
}

// Removes unobservable state elements. A state element X is observable if:
//   (1) a side-effecting operation depends on X, OR
//   (2) the next-state value of an observable state element depends on X.
absl::StatusOr<bool> RemoveUnobservableStateElements(
    Proc* proc, OptimizationContext& context) {
  if (proc->GetStateElementCount() == 0) {
    return false;
  }
  absl::flat_hash_map<Node*, InlineBitmap> state_dependencies;
  XLS_ASSIGN_OR_RETURN(state_dependencies,
                       ComputeStateDependencies(proc, context));

  // Compute an adjacency matrix for which state elements affect each other.
  std::vector<InlineBitmap> state_dependencies_matrix(
      proc->GetStateElementCount(), InlineBitmap(proc->GetStateElementCount()));
  for (auto [elem, adj] :
       iter::zip(proc->StateElements(), state_dependencies_matrix)) {
    for (Next* next : proc->next_values(elem)) {
      adj.Union(state_dependencies.at(next->value()));
      if (next->predicate().has_value()) {
        adj.Union(state_dependencies.at(*next->predicate()));
      }
    }
  }

  // Union all the side-effecting node dependencies to find the starting state
  // elements.
  InlineBitmap side_effecting_deps(proc->GetStateElementCount());
  for (Node* node : proc->nodes()) {
    if (!IsSideEffectingOrInvoke(node) ||
        node->OpIn({Op::kStateRead, Op::kNext, Op::kGate})) {
      continue;
    }
    if (node->op() == Op::kInvoke) {
      VLOG(4) << "Unioning " << node;
    }
    side_effecting_deps.Union(state_dependencies.at(node));
  }

  // Figure out which state elements are observable, by computing the set of
  // nodes reachable from side-effecting nodes in the state-dependency graph.
  const InlineBitmap observed =
      ReachableFrom(side_effecting_deps, state_dependencies_matrix);

  // Gather unobservable state element indices into `to_remove`.
  std::vector<int64_t> to_remove;
  to_remove.reserve(proc->GetStateElementCount());

  VLOG(3) << "Observability of state elements:";
  for (int64_t i = proc->GetStateElementCount() - 1; i >= 0; --i) {
    if (!observed.Get(i)) {
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
    StateElement* state_element = proc->GetStateElement(i);
    absl::Span<StateRead* const> state_reads =
        proc->GetStateReadsByStateElement(state_element);
    absl::btree_set<Next*, Node::NodeIdLessThan> next_values =
        proc->next_values(state_element);
    for (Next* next : next_values) {
      XLS_RETURN_IF_ERROR(
          next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      XLS_RETURN_IF_ERROR(proc->RemoveNode(next));
    }
    for (StateRead* state_read : state_reads) {
      if (!state_read->IsDead()) {
        XLS_RETURN_IF_ERROR(
            state_read
                ->ReplaceUsesWithNew<Literal>(ZeroOfType(state_read->GetType()))
                .status());
      }
    }
  }

  for (int64_t i : to_remove) {
    VLOG(2) << absl::StreamFormat("Removing dead state element %s of type %s",
                                  proc->GetStateElement(i)->name(),
                                  proc->GetStateElement(i)->type()->ToString());
    VLOG(4) << "Removing state element " << proc->StateElements()[i]
            << " for being unobservable.";
    XLS_RETURN_IF_ERROR(proc->RemoveStateElement(i));
  }
  return true;
}
}  // namespace
absl::StatusOr<bool> ProcStateEliminationPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  bool changed = false;

  XLS_ASSIGN_OR_RETURN(bool zero_width_changed,
                       RemoveZeroWidthStateElements(proc));
  changed = changed || zero_width_changed;

  auto query_engine = UnionQueryEngine::Of(
      StatelessQueryEngine(),
      GetSharedQueryEngine<LazyTernaryQueryEngine>(context, proc));

  // Run constant state-element removal to fixed point; should usually take just
  // one additional pass to verify, except for chains like next_s1 := s1,
  // next_s2 := f(s1), next_s3 := g(s1, s2), ..., etc., where the results all
  // match the state elements' initial values.
  bool constant_changed = false;
  do {
    XLS_ASSIGN_OR_RETURN(constant_changed,
                         RemoveConstantStateElements(proc, query_engine));
    changed = changed || constant_changed;
  } while (constant_changed);

  XLS_ASSIGN_OR_RETURN(bool unobservable_changed,
                       RemoveUnobservableStateElements(proc, context));
  changed = changed || unobservable_changed;

  return changed;
}
}  // namespace xls
