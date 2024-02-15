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

#include <deque>
#include <variant>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"

namespace xls {

// Returns a pointer to a node for the replacement. Otherwise,
// if no replacement is possible, returns nullptr.
//
// Replace:
// tuple_index.0(bar, 0)
// tuple_index.1(bar, 1)
// ...
// x = tuple(tuple_index.0, tuple_index.1, ...)
// with:
// x = bar
static Node* FindEquivalentTuple(Tuple* tuple) {
  Node* common_subject = nullptr;
  for (int64_t operand_number = 0; operand_number < tuple->operand_count();
       operand_number++) {
    Node* node = tuple->operand(operand_number);
    if (
        // Each of its operands is a TupleIndex operation.
        !node->Is<TupleIndex>() ||
        // The index of each TupleIndex operand matches the operand number.
        node->As<TupleIndex>()->index() != operand_number) {
      return nullptr;
    }
    Node* subject = node->As<TupleIndex>()->operand(0);
    if (common_subject == nullptr) {
      // Tuples should be of same size.
      if (subject->GetType() != tuple->GetType()) {
        return nullptr;
      }
      common_subject = subject;
    } else if (common_subject != subject) {
      // The operand of all the TupleIndex operands is the same
      // (nodes should be equivalent).
      return nullptr;
    }
  }
  return common_subject;
}

absl::StatusOr<bool> TupleSimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* f, const OptimizationPassOptions& options,
    PassResults* results) const {
  // Replace TupleIndex(Tuple(i{0}, i{1}, ..., i{N}), index=k) with i{k}
  bool changed = false;
  std::deque<std::variant<TupleIndex*, Tuple*>> worklist;
  for (Node* node : f->nodes()) {
    if (node->Is<TupleIndex>()) {
      worklist.push_back(node->As<TupleIndex>());
    } else if (node->Is<Tuple>()) {
      worklist.push_back(node->As<Tuple>());
    }
  }
  while (!worklist.empty()) {
    auto index = worklist.front();
    worklist.pop_front();
    if (std::holds_alternative<TupleIndex*>(index)) {
      TupleIndex* tuple_index = std::get<TupleIndex*>(index);
      // Note: lhs of tuple index may not be a tuple *instruction*.
      if (!tuple_index->operand(0)->Is<Tuple>()) {
        continue;
      }
      Node* tuple_element =
          tuple_index->operand(0)->operand(tuple_index->index());
      XLS_RETURN_IF_ERROR(tuple_index->ReplaceUsesWith(tuple_element));
      changed = true;

      // Simplifying this tuple index instruction may expose opportunities for
      // more simplifications.
      if (tuple_element->Is<Tuple>()) {
        for (Node* user : tuple_element->users()) {
          if (user->Is<TupleIndex>()) {
            worklist.push_back(user->As<TupleIndex>());
          }
        }
      }
    } else if (std::holds_alternative<Tuple*>(index)) {
      Tuple* tuple = std::get<Tuple*>(index);
      Node* common_subject = FindEquivalentTuple(tuple);
      if (common_subject != nullptr) {
        XLS_RETURN_IF_ERROR(tuple->ReplaceUsesWith(common_subject));
        changed = true;
      }
    } else {
      return absl::InternalError("Unknown index type in worklist.");
    }
  }
  return changed;
}

REGISTER_OPT_PASS(TupleSimplificationPass);

}  // namespace xls
