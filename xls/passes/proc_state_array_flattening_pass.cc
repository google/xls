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

#include "xls/passes/proc_state_array_flattening_pass.h"

#include <cstdint>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/state_element.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

// Returns true if the state element should be flattened into individual
// elements.
absl::StatusOr<bool> ShouldFlattenStateElement(StateElement* state_element) {
  if (!state_element->type()->IsArray()) {
    // Don't flatten non-array types.
    return false;
  }
  // Unconditionally flatten small arrays.
  return state_element->type()->AsArrayOrDie()->size() <= 2;
}

// Make an array_index() for each element of an array and stuff them into a
// tuple().
absl::StatusOr<Node*> ConvertArrayToTuple(Node* node) {
  XLS_RET_CHECK(node->GetType()->IsArray());
  int64_t size = node->GetType()->AsArrayOrDie()->size();
  std::vector<Node*> tuple_elements;
  tuple_elements.reserve(size);
  int64_t index_size = CeilOfLog2(size);
  // We don't want to introduce bits[0], which are sometimes problematic later.
  if (index_size == 0) {
    index_size = 1;
  }
  for (int64_t i = 0; i < size; ++i) {
    XLS_ASSIGN_OR_RETURN(Literal * index,
                         node->function_base()->MakeNode<Literal>(
                             node->loc(), Value(UBits(i, index_size))));
    XLS_ASSIGN_OR_RETURN(Node * element,
                         node->function_base()->MakeNode<ArrayIndex>(
                             node->loc(), node, std::vector<Node*>{index}));
    tuple_elements.push_back(element);
  }
  return node->function_base()->MakeNodeWithName<Tuple>(
      node->loc(), tuple_elements, absl::StrCat(node->GetName(), "_as_tuple"));
}

// Transformer to convert an array-typed proc state element into a tuple-typed
// element. Each array element will occupy the same position in the tuple.
// We convert to a tuple for convenience and to avoid replicating proc state
// tuple flattening's functionality.
class ArrayToTupleStateTransformer : public Proc::StateElementTransformer {
 public:
  explicit ArrayToTupleStateTransformer() = default;

  // Make a tuple_index() for each tuple element and stuff them into an array().
  absl::StatusOr<Node*> TransformStateRead(Proc* proc,
                                           StateRead* new_state_read,
                                           StateRead* old_state_read) final {
    VLOG(3) << "Transforming state read";
    XLS_RET_CHECK(new_state_read->GetType()->IsTuple());
    XLS_RET_CHECK(old_state_read->GetType()->IsArray());
    int64_t new_size = new_state_read->GetType()->AsTupleOrDie()->size();
    int64_t old_size = old_state_read->GetType()->AsArrayOrDie()->size();
    XLS_RET_CHECK_EQ(new_size, old_size);
    std::vector<Node*> tuple_elements;
    tuple_elements.reserve(new_size);
    for (int64_t i = 0; i < new_size; ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * element,
          proc->MakeNode<TupleIndex>(new_state_read->loc(), new_state_read, i));
      tuple_elements.push_back(element);
    }
    return proc->MakeNodeWithName<Array>(
        new_state_read->loc(), tuple_elements,
        old_state_read->GetType()->AsArrayOrDie()->element_type(),
        absl::StrCat(old_state_read->GetName(), "_as_array"));
  }
  absl::StatusOr<Node*> TransformNextValue(Proc* proc,
                                           StateRead* new_state_read,
                                           Next* old_next) final {
    VLOG(3) << "Transforming next value";
    XLS_RET_CHECK(new_state_read->GetType()->IsTuple());
    return ConvertArrayToTuple(old_next->value());
  }
};

// If our heuristic says we should flatten, replace a proc state element of
// array type with a tuple. Later optimizations will flatten the tuple.
absl::StatusOr<bool> SimplifyProcState(Proc* proc,
                                       StateElement* state_element) {
  if (!proc->IsOwned(state_element) || !state_element->type()->IsArray()) {
    return false;
  }
  VLOG(3) << "Simplifying proc state " << state_element->ToString();
  XLS_ASSIGN_OR_RETURN(bool should_flatten,
                       ShouldFlattenStateElement(state_element));
  if (!should_flatten) {
    VLOG(3) << "Not flattening proc state" << state_element->ToString();
    return false;
  }
  XLS_ASSIGN_OR_RETURN(int64_t old_state_index,
                       proc->GetStateElementIndex(state_element));
  StateRead* old_state_read = proc->GetStateRead(state_element);
  const Value& old_init_value = state_element->initial_value();
  Value new_init_value = Value::Tuple(old_init_value.elements());

  ArrayToTupleStateTransformer transformer;
  XLS_RETURN_IF_ERROR(
      proc->TransformStateElement(proc->GetStateRead(state_element),
                                  new_init_value, transformer)
          .status());

  std::vector<Next*> old_next_values(proc->next_values(old_state_read).begin(),
                                     proc->next_values(old_state_read).end());
  for (Next* old_next_value : old_next_values) {
    XLS_RETURN_IF_ERROR(proc->RemoveNode(old_next_value));
  }
  XLS_RETURN_IF_ERROR(proc->RemoveStateElement(old_state_index));
  return true;
}

}  // namespace

absl::StatusOr<bool> ProcStateArrayFlatteningPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options, PassResults* results,
    OptimizationContext& context) const {
  bool changed = false;
  // Iterate over state via index as SimplifyProcState() may invalidate proc
  // state.
  for (int64_t state_index = 0; state_index < proc->GetStateElementCount();
       ++state_index) {
    StateElement* state_element = proc->GetStateElement(state_index);
    XLS_ASSIGN_OR_RETURN(bool element_changed,
                         SimplifyProcState(proc, state_element));
    changed = changed || element_changed;
  }

  return changed;
}

}  // namespace xls
