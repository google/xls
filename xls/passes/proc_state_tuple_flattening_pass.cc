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

#include "xls/passes/proc_state_tuple_flattening_pass.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/btree_set.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/function_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/state_element.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

bool HasTupleStateElement(Proc* proc) {
  for (StateElement* state_element : proc->StateElements()) {
    if (state_element->type()->IsTuple()) {
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
struct NextValue {
  std::string name;
  SourceInfo loc;
  Node* value;
  std::optional<Node*> predicate;
};
struct AbstractStateElement {
  std::string name;
  Value initial_value;
  Node* placeholder;
  std::optional<Node*> read_predicate;
  Node* next;
  std::vector<NextValue> next_values;
};

// Replaces the state of the given proc with the given state elements. The
// existing state elements should have no uses.
absl::Status ReplaceProcState(Proc* proc,
                              absl::Span<const AbstractStateElement> elements) {
  std::vector<std::string> names;
  std::vector<Value> init_values;
  std::vector<std::optional<Node*>> read_predicates;
  std::vector<Node*> nexts;
  names.reserve(elements.size());
  init_values.reserve(elements.size());
  read_predicates.reserve(elements.size());
  nexts.reserve(elements.size());
  for (const AbstractStateElement& element : elements) {
    names.push_back(element.name);
    init_values.push_back(element.initial_value);
    read_predicates.push_back(element.read_predicate);
    nexts.push_back(element.next);
  }
  if (absl::c_all_of(elements, [](const AbstractStateElement& element) {
        return element.next_values.empty();
      })) {
    XLS_RETURN_IF_ERROR(
        proc->ReplaceState(names, init_values, read_predicates, nexts));
  } else {
    XLS_RETURN_IF_ERROR(proc->ReplaceState(names, init_values));
    for (int64_t i = 0; i < elements.size(); ++i) {
      const AbstractStateElement& element = elements.at(i);
      if (element.next != element.placeholder) {
        XLS_RETURN_IF_ERROR(
            proc->MakeNode<Next>(
                    SourceInfo(),
                    /*state_read=*/proc->GetStateRead(proc->GetStateElement(i)),
                    /*value=*/element.next,
                    /*predicate=*/std::nullopt)
                .status());
      }
    }
  }
  for (int64_t i = 0; i < elements.size(); ++i) {
    for (const NextValue& next_value : elements[i].next_values) {
      XLS_RETURN_IF_ERROR(
          proc->MakeNodeWithName<Next>(next_value.loc,
                                       /*state_read=*/proc->GetStateRead(i),
                                       /*value=*/next_value.value,
                                       /*predicate=*/next_value.predicate,
                                       next_value.name)
              .status());
    }
    XLS_RETURN_IF_ERROR(
        elements[i].placeholder->ReplaceUsesWith(proc->GetStateRead(i)));
  }
  return absl::OkStatus();
}

// Flattens the state of the given proc. Tuple typed state elements (but not
// array typed elements) in the proc state are flattened into their constituent
// components.
absl::Status FlattenState(Proc* proc) {
  std::vector<Node*> identities;
  std::vector<AbstractStateElement> elements;

  for (int64_t state_index = 0; state_index < proc->GetStateElementCount();
       ++state_index) {
    StateElement* state_element = proc->GetStateElement(state_index);
    StateRead* state_read = proc->GetStateRead(state_element);

    // Gather the flattened initial values and next state elements.
    std::vector<Value> init_values =
        DecomposeValue(state_element->initial_value());
    const int64_t num_components = init_values.size();

    std::vector<Node*> next_state;
    if (proc->GetNextStateElement(state_index) != state_read) {
      // Create a copy of the next state node. This is necessary because the
      // next state node may be a state parameter which we will be removing via
      // Proc::ReplaceState before adding and connecting the newly created state
      // params. The copy then serves as a placeholder after the old state param
      // has been deleted.
      XLS_ASSIGN_OR_RETURN(
          Node * next_state_copy,
          proc->MakeNode<UnOp>(state_read->loc(),
                               proc->GetNextStateElement(state_index),
                               Op::kIdentity));
      identities.push_back(next_state_copy);

      XLS_ASSIGN_OR_RETURN(next_state, DecomposeNode(next_state_copy));

      XLS_RET_CHECK_EQ(next_state.size(), num_components);
    }
    // Otherwise, the empty next_state vector will signal that the param uses
    // itself as its next state element.

    // Construct a StateElement for each component of the flattened state
    // element.
    std::vector<Node*> placeholders;
    int64_t elements_offset = elements.size();
    auto component_suffix = [num_components](int64_t c) {
      return (num_components == 1) ? "" : absl::StrCat("_", c);
    };
    for (int64_t i = 0; i < num_components; ++i) {
      AbstractStateElement element;
      // The name of the new state param is the same as the corresponding old
      // one if the old state param decomposes into a single element (eg., its
      // a bits type). Otherwise append a numeric suffix.
      element.name = absl::StrCat(state_element->name(), component_suffix(i));
      element.initial_value = init_values[i];
      XLS_ASSIGN_OR_RETURN(
          element.placeholder,
          proc->MakeNode<Literal>(state_read->loc(), init_values[i]));
      element.read_predicate = state_read->predicate();
      if (next_state.empty()) {
        // The next element for this param is just itself; we preserve that.
        element.next = element.placeholder;
      } else {
        element.next = next_state[i];
      }

      placeholders.push_back(element.placeholder);
      elements.push_back(std::move(element));
    }

    // Construct NextValues for each next value & component of the flattened
    // state element.
    for (Next* next : proc->next_values(state_read)) {
      XLS_ASSIGN_OR_RETURN(std::vector<Node*> value_components,
                           DecomposeNode(next->value()));
      XLS_RET_CHECK_EQ(value_components.size(), num_components);
      for (int64_t i = 0; i < value_components.size(); ++i) {
        Node*& value_component = value_components[i];
        AbstractStateElement& element = elements[elements_offset + i];

        if (value_component == state_read) {
          // The next value for this param is just itself; we preserve that by
          // referencing the placeholder, which will eventually be replaced by
          // the new param node.
          value_component = element.placeholder;
        }

        std::optional<Node*> predicate = next->predicate();
        if (predicate.has_value() && (*predicate)->Is<StateRead>()) {
          // If the predicate is itself a state param, it will be invalid after
          // ReplaceProcState(). Insert an identity so the predicate will still
          // be valid.
          XLS_ASSIGN_OR_RETURN(Node * predicate_identity,
                               proc->MakeNode<UnOp>((*predicate)->loc(),
                                                    *predicate, Op::kIdentity));
          identities.push_back(predicate_identity);
          predicate = predicate_identity;
        }
        element.next_values.push_back(NextValue{
            .name = absl::StrCat(next->GetName(), component_suffix(i)),
            .loc = next->loc(),
            .value = value_component,
            .predicate = predicate,
        });
      }
    }

    // Remove the next_value nodes for the old state param
    absl::btree_set<Next*, Node::NodeIdLessThan> next_values =
        proc->next_values(state_read);
    for (Next* next : next_values) {
      XLS_RETURN_IF_ERROR(
          next->ReplaceUsesWithNew<Literal>(Value::Tuple({})).status());
      XLS_RETURN_IF_ERROR(proc->RemoveNode(next));
    }

    // Create a node of the same type as the old state param but constructed
    // from the new (decomposed) state params placeholders.
    XLS_ASSIGN_OR_RETURN(
        Node * old_param_replacement,
        ComposeNode(state_read->GetType(), placeholders, proc));
    XLS_RETURN_IF_ERROR(state_read->ReplaceUsesWith(old_param_replacement));
  }

  XLS_RETURN_IF_ERROR(ReplaceProcState(proc, elements));

  // Now remove the identities we inserted into the graph. They should have no
  // uses.
  for (Node* identity : identities) {
    XLS_RETURN_IF_ERROR(identity->ReplaceUsesWith(identity->operand(0)));
    XLS_RETURN_IF_ERROR(proc->RemoveNode(identity));
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> ProcStateTupleFlatteningPass::RunOnProcInternal(
    Proc* proc, const OptimizationPassOptions& options,
    PassResults* results) const {
  if (!HasTupleStateElement(proc)) {
    return false;
  }
  XLS_RETURN_IF_ERROR(FlattenState(proc));
  return true;
}

REGISTER_OPT_PASS(ProcStateTupleFlatteningPass);

}  // namespace xls
