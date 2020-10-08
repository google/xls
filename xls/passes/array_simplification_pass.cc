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

#include "xls/passes/array_simplification_pass.h"

#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value_helpers.h"

namespace xls {
namespace {

// Try to simplify the given array index by replacing it with a simpler/earlier
// expression. Specifically, if the index is a literal and the element at that
// index can be determined statically, the ArrayIndex operation is replaced with
// that value. Returns true if the IR was changed.
absl::StatusOr<bool> SimplifyArrayIndex(ArrayIndex* node) {
  if (!node->operand(1)->Is<Literal>()) {
    return false;
  }

  const Bits& bits_index = node->operand(1)->As<Literal>()->value().bits();
  Node* source_array = node->operand(0);

  int64 int_index;
  // If index is out of bounds, clamp to maximum index value.
  if (bits_ops::UGreaterThanOrEqual(
          bits_index, source_array->GetType()->AsArrayOrDie()->size())) {
    int_index = source_array->GetType()->AsArrayOrDie()->size() - 1;
  } else {
    // The index should fit in 64-bits because the the array index is in bounds.
    XLS_ASSIGN_OR_RETURN(int_index, bits_index.ToUint64());
  }

  // Walk the chain of the array-typed operands backwards to try to find the
  // source of the element being indexed by the ArrayIndex operation.
  while (true) {
    if (source_array->Is<Literal>()) {
      // Source array is a literal. Replace the array-index operation with a
      // literal equal to the indexed element.
      return node->ReplaceUsesWithNew<Literal>(
          source_array->As<Literal>()->value().element(int_index));
    }

    if (source_array->Is<Array>()) {
      // Source array is a kArray operation. Replace the array-index operation
      // with the index-th operand of the kArray operation with is the index-th
      // element of the array.
      return node->ReplaceUsesWith(source_array->operand(int_index));
    }

    if (source_array->Is<ArrayUpdate>() &&
        source_array->operand(1)->Is<Literal>()) {
      // Source array is an ArrayUpdate operation with a literal index. Compare
      // the ArrayUpdate index value with the ArrayIndex index value.
      const Bits& update_index =
          source_array->operand(1)->As<Literal>()->value().bits();
      if (bits_ops::UEqual(update_index, int_index)) {
        // Element indexed by ArrayIndex is updated by ArrayUpdate. Replace
        // ArrayIndex operation with the update value from ArrayUpdate.
        return node->ReplaceUsesWith(source_array->operand(2));
      }
      // Element indexed by ArrayIndex is not updated by ArrayUpdate. Set
      // source array to array operand of ArrayUpdate and keep walking.
      source_array = source_array->operand(0);
      continue;
    }

    // Can't determine the source of the array element so no transformation is
    // possible. Fall through and return.
    break;
  }
  return false;
}
}  // namespace

absl::StatusOr<bool> ArraySimplificationPass::RunOnFunction(
    Function* func, const PassOptions& options, PassResults* results) const {
  XLS_VLOG(2) << "Running array simplifier on function " << func->name();
  XLS_VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, func->DumpIr());

  bool changed = false;
  for (Node* node : TopoSort(func)) {
    if (node->Is<ArrayIndex>()) {
      XLS_ASSIGN_OR_RETURN(bool node_changed,
                           SimplifyArrayIndex(node->As<ArrayIndex>()));
      changed = changed | node_changed;
    }
  }

  XLS_VLOG(3) << "After:";
  XLS_VLOG_LINES(3, func->DumpIr());

  return changed;
}

}  // namespace xls
