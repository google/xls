// Copyright 2020 The XLS Authors
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

#include "xls/passes/tuple_simplification_pass.h"

#include "absl/status/status.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"

namespace xls {

absl::StatusOr<bool> TupleSimplificationPass::RunOnFunctionBase(
    FunctionBase* f, const PassOptions& options, PassResults* results) const {
  // Replace TupleIndex(Tuple(i{0}, i{1}, ..., i{N}), index=k) with i{k}
  bool changed = false;
  std::deque<absl::variant<TupleIndex*, ArrayIndex*>> worklist;
  for (Node* node : f->nodes()) {
    if (node->Is<TupleIndex>()) {
      worklist.push_back(node->As<TupleIndex>());
    } else if (node->Is<ArrayIndex>()) {
      worklist.push_back(node->As<ArrayIndex>());
    }
  }
  while (!worklist.empty()) {
    auto index = worklist.front();
    worklist.pop_front();
    if (absl::holds_alternative<TupleIndex*>(index)) {
      TupleIndex* tuple_index = absl::get<TupleIndex*>(index);
      // Note: lhs of tuple index may not be a tuple *instruction*.
      if (!tuple_index->operand(0)->Is<Tuple>()) {
        continue;
      }
      Node* tuple_element =
          tuple_index->operand(0)->operand(tuple_index->index());
      XLS_ASSIGN_OR_RETURN(bool node_changed,
                           tuple_index->ReplaceUsesWith(tuple_element));
      changed |= node_changed;

      // Simplifying this tuple index instruction may expose opportunities for
      // more simplifications.
      if (tuple_element->Is<Tuple>()) {
        for (Node* user : tuple_element->users()) {
          if (user->Is<TupleIndex>()) {
            worklist.push_back(user->As<TupleIndex>());
          }
        }
      }
    } else if (absl::holds_alternative<ArrayIndex*>(index)) {
      ArrayIndex* array_index = absl::get<ArrayIndex*>(index);
      if (!array_index->operand(1)->Is<Literal>()) {
        continue;
      }
      Literal* rhs = array_index->operand(1)->As<Literal>();
      if (!rhs->value().bits().FitsInUint64()) {
        continue;
      }
      XLS_ASSIGN_OR_RETURN(uint64 index, rhs->value().bits().ToUint64());
      if (index >= array_index->operand(0)->GetType()->AsArrayOrDie()->size()) {
        // Punt on optimizing OOB accesses.
        continue;
      }
      if (array_index->operand(0)->Is<Array>()) {
        Array* array = array_index->operand(0)->As<Array>();
        Node* array_element = array->operand(index);
        XLS_ASSIGN_OR_RETURN(bool node_changed,
                             array_index->ReplaceUsesWith(array_element));
        changed |= node_changed;
      } else if (array_index->operand(0)->Is<Literal>()) {
        Literal* array = array_index->operand(0)->As<Literal>();
        XLS_RET_CHECK(array->GetType()->IsArray());
        XLS_RET_CHECK_LT(index, array->value().size());
        const Value& element = array->value().element(index);
        XLS_RETURN_IF_ERROR(
            array_index->ReplaceUsesWithNew<Literal>(element).status());
        changed = true;
      }
    } else {
      return absl::InternalError("Unknown index type in worklist.");
    }
  }

  return changed;
}

}  // namespace xls
