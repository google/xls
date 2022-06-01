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

#include "xls/passes/proc_state_flattening_pass.h"

#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace {

bool HasTupleStateElement(Proc* proc) {
  for (Param* param : proc->StateParams()) {
    if (param->GetType()->IsTuple()) {
      return true;
    }
  }
  return false;
}

void DecomposeValueHelper(const Value& value, std::vector<Value>& elements) {
  if (value.IsTuple()) {
    for (const Value& element : value.elements()) {
      DecomposeValueHelper(element, elements);
    }
  } else {
    elements.push_back(value);
  }
}

// Decomposes the tuple structure of the given Value and returns the
// corresponding leaf elements. Array elements are *not* decomposed.
std::vector<Value> DecomposeValue(const Value& value) {
  std::vector<Value> elements;
  DecomposeValueHelper(value, elements);
  return elements;
}

absl::Status DecomposeNodeHelper(Node* node, std::vector<Node*>& elements) {
  if (node->GetType()->IsTuple()) {
    for (int64_t i = 0; i < node->GetType()->AsTupleOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * tuple_index,
          node->function_base()->MakeNode<TupleIndex>(SourceInfo(), node, i));
      XLS_RETURN_IF_ERROR(DecomposeNodeHelper(tuple_index, elements));
    }
  } else {
    elements.push_back(node);
  }
  return absl::OkStatus();
}

// Recursively decomposes the tuple structure of the given node and returns
// the elements as a vector. The elements are extracted using TupleIndex
// operations which are added to the graph as necessary. Example vectors
// returned for different types:
//
//   x: bits[32] ->
//         {x}
//   x: (bits[32], bits[32], bits[32]) ->
//         {TupleIndex(x, 0), TupleIndex(x, 1), TupleIndex(x, 2)}
//   x: (bits[32], (bits[32])) ->
//         {TupleIndex(x, 0), TupleIndex(TupleIndex(x, 1), 0)}
absl::StatusOr<std::vector<Node*>> DecomposeNode(Node* node) {
  std::vector<Node*> elements;
  XLS_RETURN_IF_ERROR(DecomposeNodeHelper(node, elements));
  return std::move(elements);
}

absl::StatusOr<Node*> ComposeNodeHelper(Type* type,
                                        absl::Span<Node* const> elements,
                                        int64_t& linear_index,
                                        FunctionBase* f) {
  if (type->IsTuple()) {
    std::vector<Node*> tuple_elements;
    for (int64_t i = 0; i < type->AsTupleOrDie()->size(); ++i) {
      XLS_ASSIGN_OR_RETURN(
          Node * element,
          ComposeNodeHelper(type->AsTupleOrDie()->element_type(i), elements,
                            linear_index, f));
      tuple_elements.push_back(element);
    }
    return f->MakeNode<Tuple>(SourceInfo(), tuple_elements);
  }
  return elements[linear_index++];
}

// Constructs a node of the given type using the given tuple leaf elements.
// Tuple operations are added to the graph as necessary. Example expressions
// returned for given type and leaf_elements vector:
//
//   bits[32] {x} -> x
//   (bits[32], bits[32]) {x, y} -> Tuple(x, y)
//   (bits[32], (bits[32])) {x, y} -> Tuple(x, Tuple(y))
absl::StatusOr<Node*> ComposeNode(Type* type, absl::Span<Node* const> elements,
                                  FunctionBase* f) {
  int64_t linear_index = 0;
  return ComposeNodeHelper(type, elements, linear_index, f);
}

// Abstraction representing a flattened state element.
struct StateElement {
  std::string name;
  Value initial_value;
  Node* placeholder;
  Node* next;
};

// Replaces the state of the given proc with the given state elements. The
// existing state elements should have no uses.
absl::Status ReplaceProcState(Proc* proc,
                              absl::Span<const StateElement> elements) {
  std::vector<std::string> names;
  std::vector<Value> init_values;
  std::vector<Node*> nexts;
  names.reserve(elements.size());
  init_values.reserve(elements.size());
  nexts.reserve(elements.size());
  for (const StateElement& element : elements) {
    names.push_back(element.name);
    init_values.push_back(element.initial_value);
    nexts.push_back(element.next);
  }
  XLS_RETURN_IF_ERROR(proc->ReplaceState(names, init_values, nexts));
  for (int64_t i = 0; i < elements.size(); ++i) {
    XLS_RETURN_IF_ERROR(
        elements[i].placeholder->ReplaceUsesWith(proc->GetStateParam(i)));
  }
  return absl::OkStatus();
}

// Flattens the state of the given proc. Tuple typed state elements (but not
// array typed elements) in the proc state are flattened into their constituent
// components.
absl::Status FlattenState(Proc* proc) {
  std::vector<Node*> identities;
  std::vector<StateElement> elements;

  for (int64_t i = 0; i < proc->GetStateElementCount(); ++i) {
    Param* state_param = proc->GetStateParam(i);

    // Create a copy of the next state node. This is necessary because the next
    // state node may be a state parameter which we will be removing via
    // Proc::ReplaceState before adding and connecting the newly created state
    // params. The copy then serves as a placeholder after the old state param
    // has been deleted.
    XLS_ASSIGN_OR_RETURN(
        Node * next_state_copy,
        proc->MakeNode<UnOp>(state_param->loc(), proc->GetNextStateElement(i),
                             Op::kIdentity));
    identities.push_back(next_state_copy);

    // Gather the flattened intial values and next state elements.
    std::vector<Value> init_values =
        DecomposeValue(proc->GetInitValueElement(i));
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> next_state,
                         DecomposeNode(next_state_copy));
    XLS_RET_CHECK_EQ(init_values.size(), next_state.size());

    // Construct a StateElement for each component of the flattened state
    // element.
    std::vector<Node*> placeholders;
    for (int64_t i = 0; i < init_values.size(); ++i) {
      StateElement element;
      // The name of the new state param is the same as the corresponding old
      // one if the old state param decomposes into a single element (eg., its
      // a bits type). Otherwise append a numeric suffix.
      element.name = (init_values.size() == 1)
                         ? state_param->GetName()
                         : absl::StrFormat("%s_%d", state_param->GetName(), i);
      element.initial_value = init_values[i];
      XLS_ASSIGN_OR_RETURN(
          element.placeholder,
          proc->MakeNode<Literal>(state_param->loc(), init_values[i]));
      element.next = next_state[i];

      placeholders.push_back(element.placeholder);
      elements.push_back(std::move(element));
    }
    // Create a node of the same type as the old state param but constructed
    // from the new (decomposed) state params placeholders.
    XLS_ASSIGN_OR_RETURN(
        Node * old_param_replacement,
        ComposeNode(state_param->GetType(), placeholders, proc));
    XLS_RETURN_IF_ERROR(state_param->ReplaceUsesWith(old_param_replacement));
  }

  XLS_RETURN_IF_ERROR(ReplaceProcState(proc, elements));

  // Now remove the indentities we inserted into the graph. They should have no
  // uses.
  for (Node* identity : identities) {
    XLS_RETURN_IF_ERROR(identity->ReplaceUsesWith(identity->operand(0)));
    XLS_RETURN_IF_ERROR(proc->RemoveNode(identity));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ProcStateFlatteningPass::RunOnProcInternal(
    Proc* proc, const PassOptions& options, PassResults* results) const {
  if (!HasTupleStateElement(proc)) {
    return false;
  }
  XLS_RETURN_IF_ERROR(FlattenState(proc));
  return true;
}

}  // namespace xls
