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

#include "xls/passes/array_untuple_pass.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "cppitertools/count.hpp"
#include "cppitertools/enumerate.hpp"
#include "cppitertools/range.hpp"
#include "cppitertools/zip.hpp"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/union_find.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/function.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {
// Group all nodes together based on whether they need to be updated together.
UnionFind<Node*> FindUntupleGroups(FunctionBase* f) {
  UnionFind<Node*> array_groups;
  // To make things simpler every node is put in the union-find.
  for (Node* n : TopoSort(f)) {
    array_groups.Insert(n);
    if (n->Is<ArrayUpdate>()) {  // Array modification
      array_groups.Union(n->As<ArrayUpdate>()->array_to_update(), n);
    } else if (n->Is<ArraySlice>()) {
      array_groups.Union(n->As<ArraySlice>()->array(), n);
    } else if (n->Is<ArrayConcat>()) {
      // We don't want to have to reconstruct all the other concat arguments if
      // one of them is external.
      for (Node* arg : n->operands()) {
        array_groups.Union(arg, n);
      }
    } else if (n->OpIn({Op::kEq, Op::kNe})) {  // Comparison
      // Eq/ne force the arguments into the same equivalence class so we don't
      // need to reconstruct the array to compare it to a param or something.
      array_groups.Union(n->operand(0), n->operand(1));
    } else if (n->OpIn({
                   Op::kSel,
                   Op::kPrioritySel,
                   Op::kOneHotSel,
               })) {  // select
      for (Node* operand : n->operands().subspan(1)) {
        array_groups.Union(operand, n);
      }
    } else if (n->Is<Gate>()) {  // gate.
      array_groups.Union(n->As<Gate>()->data(), n);
    } else if (n->Is<Next>()) {
      // Next needs both sides to be represented the same.
      array_groups.Union(n->As<Next>()->param(), n->As<Next>()->value());
    }
  }
  // For non-next procs the param needs to have the same repr as the node which
  // updates it.
  if (f->IsProc()) {
    for (const auto& [param, update_value] : iter::zip(
             f->AsProcOrDie()->StateParams(), f->AsProcOrDie()->NextState())) {
      array_groups.Union(param, update_value);
    }
  }
  return array_groups;
}

// Find any instructions which make an array fully visible 'externally'. These
// uses prevent untuple-ing since we'd need to reconstruct the original before
// sending the full array out.
absl::StatusOr<absl::flat_hash_set<Node*>> FindExternalGroups(
    FunctionBase* f, UnionFind<Node*>& groups) {
  absl::flat_hash_set<Node*> excluded;
  // Inputs/outputs are always excluded.
  if (f->IsFunction()) {
    for (Node* inp : f->AsFunctionOrDie()->params()) {
      if (inp->GetType()->IsArray()) {
        excluded.insert(groups.Find(inp));
      }
    }
    Node* ret_val = f->AsFunctionOrDie()->return_value();
    if (ret_val->GetType()->IsArray()) {
      excluded.insert(groups.Find(ret_val));
    }
  } else {
    XLS_RET_CHECK(f->IsProc());
    // Don't mess with params that are only used in identity updates. Would
    // infinite loop otherwise since we don't remove these very often.
    if (f->AsProcOrDie()->next_values().empty()) {
      for (const auto& [param, update] :
           iter::zip(f->AsProcOrDie()->StateParams(),
                     f->AsProcOrDie()->NextState())) {
        if (param == update) {
          excluded.insert(groups.Find(param));
        }
      }
    } else {
      for (Param* p : f->AsProcOrDie()->StateParams()) {
        if (absl::c_all_of(p->users(), [&](Node* n) -> bool {
              if (n->Is<Next>()) {
                Next* nxt = n->As<Next>();
                return nxt->param() == nxt->value() && nxt->param() == p;
              }
              return false;
            })) {
          excluded.insert(groups.Find(p));
        }
      }
    }
  }
  for (Node* n : f->nodes()) {
    // Proc output, nb proc input is captured by other use below.
    if (n->Is<Send>()) {
      if (n->As<Send>()->data()->GetType()->IsArray()) {
        VLOG(2) << "Unable to untuple data for " << n
                << " (in group: " << groups.Find(n->As<Send>()->data())
                << ") due to external visibility";
        excluded.insert(groups.Find(n->As<Send>()->data()));
      }
    }
    // Arrays of arrays aren't handled. We could unwrap them too but that
    // doesn't seem worthwhile.
    if (n->Is<Array>() &&
        n->GetType()->AsArrayOrDie()->element_type()->IsArray() &&
        n->GetType()
            ->AsArrayOrDie()
            ->element_type()
            ->AsArrayOrDie()
            ->element_type()
            ->IsTuple()) {
      for (Node* op : n->operands()) {
        excluded.insert(groups.Find(op));
      }
    }
    if (n->Is<ArrayUpdate>()) {
      // Can't untuple the value being updated.
      excluded.insert(groups.Find(n->As<ArrayUpdate>()->update_value()));
    }
    // TODO(allight): We can exclude Trace here too and rewrite it but that's
    // rather complicated.
    if (n->OpIn({Op::kArray, Op::kArrayConcat, Op::kArraySlice,
                 Op::kArrayUpdate, Op::kParam, Op::kSel, Op::kPrioritySel,
                 Op::kOneHotSel, Op::kGate, Op::kLiteral, Op::kEq, Op::kNe,
                 Op::kNext})) {
      continue;
    }
    // Anything that is an array here can't be handled since we'd need to
    // manually unwrap them. Pretty much just tuple-index from things like
    // recvs.
    if (n->GetType()->IsArray() &&
        n->GetType()->AsArrayOrDie()->element_type()->IsTuple()) {
      // Don't know how to untuple-ify this instruction.
      VLOG(2) << "Unable to untuple " << n << " (in group: " << groups.Find(n)
              << ")";
      excluded.insert(groups.Find(n));
      continue;
    }
    // We need to exclude this if the result is an array but we don't need to
    // exclude its operands.
    if (n->Is<ArrayIndex>()) {
      continue;
    }
    // Any arguments would keep the old use around so we need to exclude them.
    for (Node* op : n->operands()) {
      if (op->GetType()->IsArray() &&
          op->GetType()->AsArrayOrDie()->element_type()->IsTuple()) {
        if (VLOG_IS_ON(2) && !excluded.contains(groups.Find(op))) {
          VLOG(2) << "Unable to untuple " << op << " (in group "
                  << groups.Find(op) << ") due to use in unhandled operation "
                  << n;
        }
        excluded.insert(groups.Find(op));
      }
    }
  }
  return excluded;
}

absl::StatusOr<Value> GetElementArray(const Value& base, int64_t idx) {
  XLS_RET_CHECK(base.IsArray());
  std::vector<Value> vals;
  vals.reserve(base.size());
  for (const Value& v : base.elements()) {
    XLS_RET_CHECK(v.IsTuple());
    XLS_RET_CHECK_GT(v.size(), idx);
    vals.push_back(v.element(idx));
  }
  return Value::Array(vals);
}

class UntupleVisitor : public DfsVisitorWithDefault {
 public:
  explicit UntupleVisitor(UnionFind<Node*>& groups,
                          const absl::flat_hash_set<Node*>& excluded_groups)
      : groups_(groups), excluded_groups_(excluded_groups) {}
  ~UntupleVisitor() override = default;

  bool changed() const { return changed_; }
  absl::Status DefaultHandler(Node* n) override { return absl::OkStatus(); }

  absl::Status FixupImplicitUses(Proc* proc, int64_t inital_param_count) {
    // For each of the original state params.
    XLS_RET_CHECK(changed_) << "Called without changes.";
    for (const auto& [idx, param, update_value] :
         iter::zip(iter::range(inital_param_count), proc->StateParams(),
                   proc->NextState())) {
      if (!CanUntuple(param)) {
        continue;
      }
      XLS_RET_CHECK(CanUntuple(update_value));
      // Link up all the new params.
      XLS_RET_CHECK(components_.contains(param));
      XLS_RET_CHECK(components_.contains(update_value));
      for (const auto& [new_param, new_update_value] :
           iter::zip(components_.at(param), components_.at(update_value))) {
        XLS_RET_CHECK(new_param->Is<Param>());
        XLS_ASSIGN_OR_RETURN(int64_t new_param_idx,
                             proc->GetStateParamIndex(new_param->As<Param>()));
        XLS_RETURN_IF_ERROR(
            proc->SetNextStateElement(new_param_idx, new_update_value));
      }
      // Set the old param to identity.
      XLS_RETURN_IF_ERROR(proc->SetNextStateElement(idx, param));
    }
    return absl::OkStatus();
  }

  absl::Status HandleEq(CompareOp* eq) override {
    if (!CanUntuple(eq->operand(0))) {
      // NB we force the two sides of this to be in the same equiv class so we
      // only need to check one.
      return DefaultHandler(eq);
    }
    VLOG(2) << "Untuple-ing eq " << eq;
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> comps, DistributeCompare(eq));
    XLS_RETURN_IF_ERROR(
        eq->ReplaceUsesWithNew<NaryOp>(comps, Op::kAnd).status());
    return absl::OkStatus();
  }
  absl::Status HandleNe(CompareOp* ne) override {
    if (!CanUntuple(ne->operand(0))) {
      // NB we force the two sides of this to be in the same equiv class so we
      // only need to check one.
      return DefaultHandler(ne);
    }
    VLOG(2) << "Untuple-ing ne " << ne;
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> comps, DistributeCompare(ne));
    XLS_RETURN_IF_ERROR(
        ne->ReplaceUsesWithNew<NaryOp>(comps, Op::kOr).status());
    return absl::OkStatus();
  }
  absl::Status HandleLiteral(Literal* lit) override {
    if (!CanUntuple(lit)) {
      return DefaultHandler(lit);
    }
    VLOG(2) << "Untuple-ing lit " << lit;
    std::vector<Node*> elements;
    int64_t tup_size =
        lit->GetType()->AsArrayOrDie()->element_type()->AsTupleOrDie()->size();
    elements.reserve(tup_size);
    for (int64_t i = 0; i < tup_size; ++i) {
      XLS_ASSIGN_OR_RETURN(Value comp, GetElementArray(lit->value(), i));
      XLS_ASSIGN_OR_RETURN(*std::back_inserter(elements),
                           lit->function_base()->MakeNodeWithName<Literal>(
                               lit->loc(), comp, IdxName(lit, i)));
    }
    return RecordUntuple(lit, std::move(elements));
  }

  absl::Status HandleParam(Param* p) override {
    if (!CanUntuple(p)) {
      return DefaultHandler(p);
    }
    XLS_RET_CHECK(p->function_base()->IsProc())
        << "Attempting to untuple argument of function: " << p;
    VLOG(2) << "Untuple-ing param " << p;
    Proc* proc = p->function_base()->AsProcOrDie();
    XLS_ASSIGN_OR_RETURN(Value init, proc->GetInitValue(p));
    TupleType* element =
        p->GetType()->AsArrayOrDie()->element_type()->AsTupleOrDie();
    std::vector<Node*> res;
    res.reserve(element->size());
    for (int64_t i = 0; i < element->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(Value element_array, GetElementArray(init, i));
      XLS_ASSIGN_OR_RETURN(
          *std::back_inserter(res),
          proc->AppendStateElement(
              absl::StrFormat("%s_tuple_element_%d", p->GetName(), i),
              std::move(element_array)));
    }
    return RecordUntuple(p, std::move(res));
  }

  absl::Status HandleNext(Next* n) override {
    if (!CanUntuple(n->param())) {
      return DefaultHandler(n);
    }
    VLOG(2) << "Untuple-ing next " << n;
    changed_ = true;
    Proc* proc = n->function_base()->AsProcOrDie();
    XLS_RET_CHECK(CanUntuple(n->value()))
        << "Unable to untuple both param and value.";
    XLS_RET_CHECK(components_.contains(n->param()));
    XLS_RET_CHECK(components_.contains(n->value()));
    absl::Span<Node* const> param_values = components_.at(n->param());
    absl::Span<Node* const> update_values = components_.at(n->value());
    for (const auto& [idx, param_node, value] :
         iter::zip(iter::count(), param_values, update_values)) {
      XLS_RET_CHECK(param_node->Is<Param>());
      Param* param = param_node->As<Param>();
      XLS_RETURN_IF_ERROR(proc->MakeNodeWithName<Next>(n->loc(), param, value,
                                                       n->predicate(),
                                                       IdxName(n, idx))
                              .status());
    }
    // Remove this next from consideration.
    if (n->value() != n->param()) {
      XLS_RET_CHECK(n->ReplaceOperand(n->value(), n->param()));
    }
    XLS_RET_CHECK(n->users().empty())
        << "Something is using the empty-tuple value of a next node: " << n;
    return absl::OkStatus();
  }

  absl::Status HandleArrayIndex(ArrayIndex* array_index) override {
    if (!CanUntuple(array_index->array())) {
      return DefaultHandler(array_index);
    }
    VLOG(2) << "Untupling array index " << array_index;
    changed_ = true;
    std::vector<Node*> elements;
    XLS_RET_CHECK(components_.contains(array_index->array())) << array_index;
    absl::Span<Node* const> elem_arrays = components_.at(array_index->array());
    elements.reserve(elem_arrays.size());
    int64_t cnt = 0;
    for (Node* elem_arr : elem_arrays) {
      XLS_ASSIGN_OR_RETURN(
          *std::back_inserter(elements),
          array_index->function_base()->MakeNodeWithName<ArrayIndex>(
              array_index->loc(), elem_arr, array_index->indices(),
              IdxName(array_index, cnt)));
      cnt++;
    }
    XLS_RETURN_IF_ERROR(
        array_index->ReplaceUsesWithNew<Tuple>(elements).status());
    return absl::OkStatus();
  }

  absl::Status HandleArray(Array* arr) override {
    if (!CanUntuple(arr)) {
      return DefaultHandler(arr);
    }
    VLOG(2) << "Untuple-ing array " << arr;
    std::vector<std::vector<Node*>> element_arrays(
        arr->element_type()->AsTupleOrDie()->size(), std::vector<Node*>{});
    for (auto v : element_arrays) {
      v.reserve(arr->size());
    }
    for (int64_t i = 0; i < element_arrays.size(); ++i) {
      for (int64_t arr_idx = 0; arr_idx < arr->size(); ++arr_idx) {
        XLS_ASSIGN_OR_RETURN(*std::back_inserter(element_arrays[i]),
                             GetNodeAtIndex(arr->operand(arr_idx), {i}));
      }
    }
    std::vector<Node*> elements;
    elements.reserve(element_arrays.size());
    for (const auto& [idx, e] : iter::enumerate(element_arrays)) {
      XLS_ASSIGN_OR_RETURN(
          *std::back_inserter(elements),
          arr->function_base()->MakeNodeWithName<Array>(
              arr->loc(), e,
              arr->element_type()->AsTupleOrDie()->element_type(idx),
              IdxName(arr, idx)));
    }
    return RecordUntuple(arr, std::move(elements));
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* update) override {
    if (!CanUntuple(update)) {
      return DefaultHandler(update);
    }
    VLOG(2) << "Untuple-ing array-update " << update;
    std::vector<Node*> results;
    XLS_RET_CHECK(components_.contains(update->array_to_update()))
        << "updating " << update->array_to_update();
    absl::Span<Node* const> prev_value =
        components_.at(update->array_to_update());
    int64_t tup_size = prev_value.size();
    results.reserve(tup_size);
    for (int64_t i = 0; i < tup_size; ++i) {
      XLS_ASSIGN_OR_RETURN(Node * to_update_part,
                           GetNodeAtIndex(update->update_value(), {i}));
      XLS_ASSIGN_OR_RETURN(
          *std::back_inserter(results),
          update->function_base()->MakeNodeWithName<ArrayUpdate>(
              update->loc(), prev_value[i], to_update_part, update->indices(),
              IdxName(update, i)));
    }
    return RecordUntuple(update, std::move(results));
  }

  absl::Status HandleArraySlice(ArraySlice* slice) override {
    if (!CanUntuple(slice)) {
      return DefaultHandler(slice);
    }
    VLOG(2) << "Untuple-ing array-slice " << slice;
    XLS_RET_CHECK(components_.contains(slice->array())) << slice;
    std::vector<Node*> res;
    res.reserve(components_.at(slice->array()).size());
    for (const auto& [idx, e] :
         iter::enumerate(components_.at(slice->array()))) {
      XLS_ASSIGN_OR_RETURN(*std::back_inserter(res),
                           slice->function_base()->MakeNodeWithName<ArraySlice>(
                               slice->loc(), e, slice->start(), slice->width(),
                               IdxName(slice, idx)));
    }
    return RecordUntuple(slice, std::move(res));
  }

  absl::Status HandleArrayConcat(ArrayConcat* concat) override {
    if (!CanUntuple(concat)) {
      return DefaultHandler(concat);
    }
    VLOG(2) << "Untuple-ing array-concat " << concat;
    XLS_RET_CHECK(absl::c_all_of(concat->operands(), [&](Node* op) {
      return components_.contains(op);
    })) << concat;
    std::vector<Node*> res;
    int64_t tup_size = concat->GetType()
                           ->AsArrayOrDie()
                           ->element_type()
                           ->AsTupleOrDie()
                           ->size();
    res.reserve(tup_size);
    for (int64_t i = 0; i < tup_size; ++i) {
      std::vector<Node*> element;
      element.reserve(concat->operand_count());
      absl::c_transform(concat->operands(), std::back_inserter(element),
                        [&](Node* n) { return components_.at(n).at(i); });
      XLS_ASSIGN_OR_RETURN(
          *std::back_inserter(res),
          concat->function_base()->MakeNodeWithName<ArrayConcat>(
              concat->loc(), element, IdxName(concat, i)));
    }
    return RecordUntuple(concat, std::move(res));
  }

  absl::Status HandleGate(Gate* gate) override {
    if (!CanUntuple(gate)) {
      return DefaultHandler(gate);
    }
    VLOG(2) << "Untuple-ing gate " << gate;
    XLS_RET_CHECK(components_.contains(gate->data())) << gate;
    std::vector<Node*> res;
    res.reserve(components_.at(gate->data()).size());
    for (const auto& [idx, e] : iter::enumerate(components_.at(gate->data()))) {
      XLS_ASSIGN_OR_RETURN(
          *std::back_inserter(res),
          gate->function_base()->MakeNodeWithName<Gate>(
              gate->loc(), gate->condition(), e, IdxName(gate, idx)));
    }
    return RecordUntuple(gate, std::move(res));
  }

  absl::Status HandleSel(Select* sel) override {
    if (!CanUntuple(sel)) {
      return DefaultHandler(sel);
    }
    return HandleSelectLike(
        sel, sel->selector(), sel->cases(), sel->default_value(),
        [](SourceInfo loc, std::string_view name, Node* selector,
           absl::Span<Node* const> cases,
           std::optional<Node*> def) -> absl::StatusOr<Node*> {
          return selector->function_base()->MakeNodeWithName<Select>(
              loc, selector, cases, def, name);
        });
  }
  absl::Status HandlePrioritySel(PrioritySelect* sel) override {
    if (!CanUntuple(sel)) {
      return DefaultHandler(sel);
    }
    return HandleSelectLike(
        sel, sel->selector(), sel->cases(), sel->default_value(),
        [](SourceInfo loc, std::string_view name, Node* selector,
           absl::Span<Node* const> cases,
           std::optional<Node*> def) -> absl::StatusOr<Node*> {
          XLS_RET_CHECK(def) << "Default required";
          return selector->function_base()->MakeNodeWithName<PrioritySelect>(
              loc, selector, cases, *def, name);
        });
  }
  absl::Status HandleOneHotSel(OneHotSelect* sel) override {
    if (!CanUntuple(sel)) {
      return DefaultHandler(sel);
    }
    return HandleSelectLike(
        sel, sel->selector(), sel->cases(), /*default_value=*/std::nullopt,
        [](SourceInfo loc, std::string_view name, Node* selector,
           absl::Span<Node* const> cases,
           std::optional<Node*> def) -> absl::StatusOr<Node*> {
          XLS_RET_CHECK(!def) << "OneHotSelect cannot have a default.";
          return selector->function_base()->MakeNodeWithName<OneHotSelect>(
              loc, selector, cases, name);
        });
  }

 private:
  // Perform a generic select on all the tuple elements.
  template <typename MakeSel>
    requires(std::is_invocable_r_v<
             absl::StatusOr<Node*>, MakeSel, SourceInfo, std::string_view,
             Node*, absl::Span<Node* const>, std::optional<Node*>>)
  absl::Status HandleSelectLike(Node* sel, Node* selector,
                                absl::Span<Node* const> cases,
                                std::optional<Node*> default_value,
                                MakeSel make_sel) {
    VLOG(2) << "Untupling " << sel->op() << " " << sel;
    std::vector<Node*> elements;
    int64_t element_count =
        sel->GetType()->AsArrayOrDie()->element_type()->AsTupleOrDie()->size();
    elements.reserve(element_count);
    for (int64_t i = 0; i < element_count; ++i) {
      std::vector<Node*> idx_cases;
      idx_cases.reserve(cases.size());
      for (Node* n : cases) {
        XLS_RET_CHECK(components_.contains(n)) << n << " in " << sel;
        idx_cases.push_back(components_.at(n).at(i));
      }
      std::optional<Node*> idx_default;
      if (default_value) {
        XLS_RET_CHECK(components_.contains(*default_value))
            << *default_value << " in " << sel;
        idx_default = components_.at(*default_value).at(i);
      }
      XLS_ASSIGN_OR_RETURN(*std::back_inserter(elements),
                           make_sel(sel->loc(), IdxName(sel, i), selector,
                                    idx_cases, idx_default));
    }
    return RecordUntuple(sel, std::move(elements));
  }

  // Run the compare operation on each tuple element of the given op.
  absl::StatusOr<std::vector<Node*>> DistributeCompare(CompareOp* op) {
    changed_ = true;
    XLS_RET_CHECK(components_.contains(op->operand(0))) << op;
    XLS_RET_CHECK(components_.contains(op->operand(1))) << op;
    std::vector<Node*> component_vals;
    int64_t component_count = components_.at(op->operand(0)).size();
    component_vals.reserve(component_count);
    auto add_end = std::back_inserter(component_vals);
    for (const auto& [idx, lhs, rhs] :
         iter::zip(iter::count(), components_.at(op->operand(0)),
                   components_.at(op->operand(1)))) {
      XLS_ASSIGN_OR_RETURN(
          *add_end, op->function_base()->MakeNodeWithName<CompareOp>(
                        op->loc(), lhs, rhs, op->op(), IdxName(op, idx)));
    }
    return component_vals;
  }
  // Give a name for the untupled values.
  std::string IdxName(Node* n, int64_t idx) const {
    return n->HasAssignedName()
               ? absl::StrFormat("%s_tuple_idx_%d", n->GetName(), idx)
               : "";
  }
  // Check if the node is eligible for and not excluded from untuple-ing
  bool CanUntuple(Node* n) {
    return n->GetType()->IsArray() &&
           n->GetType()->AsArrayOrDie()->element_type()->IsTuple() &&
           !excluded_groups_.contains(groups_.Find(n)) &&
           (!n->users().empty() || n->function_base()->HasImplicitUse(n));
  }
  // Set the value of the untuple'd node.
  absl::Status RecordUntuple(Node* n, std::vector<Node*>&& elements) {
    components_[n] = std::move(elements);
    changed_ = true;
    return absl::OkStatus();
  }
  UnionFind<Node*>& groups_;
  const absl::flat_hash_set<Node*>& excluded_groups_;
  absl::flat_hash_map<Node*, std::vector<Node*>> components_;
  bool changed_ = false;
};
}  // namespace

absl::StatusOr<bool> ArrayUntuplePass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  if (!f->IsFunction() && !f->IsProc()) {
    // Don't mess with blocks.
    return false;
  }
  UnionFind<Node*> groups = FindUntupleGroups(f);
  // Get the set of representative elements which are in groups with external
  // uses and so cannot be (profitably) array-of-structs-ified
  XLS_ASSIGN_OR_RETURN(absl::flat_hash_set<Node*> excluded,
                       FindExternalGroups(f, groups));
  UntupleVisitor vis(groups, excluded);
  int64_t initial_state_count =
      f->IsProc() ? f->AsProcOrDie()->StateParams().size() : -1;
  for (Node* n : TopoSort(f)) {
    XLS_RETURN_IF_ERROR(n->VisitSingleNode(&vis)) << n;
  }
  if (vis.changed() && f->IsProc() && f->AsProcOrDie()->next_values().empty()) {
    // Fixup old-style-next procs.
    // TODO(allight): Remove once fully moved over to next-value operations.  It
    // would be pretty simple to make the pass only affect params that are
    // updated through next-values but ordering of passes makes doing it this
    // way incompatible with also letting other proc-state opts get a look at
    // this.
    XLS_RETURN_IF_ERROR(
        vis.FixupImplicitUses(f->AsProcOrDie(), initial_state_count));
  }
  // XLS_RETURN_IF_ERROR(f->Accept(&vis));
  return vis.changed();
}

REGISTER_OPT_PASS(ArrayUntuplePass);
}  // namespace xls
