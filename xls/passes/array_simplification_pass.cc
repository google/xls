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

#include "absl/status/statusor.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value_helpers.h"
#include "xls/passes/ternary_query_engine.h"

namespace xls {
namespace {

// Try to simplify the given array index node with a literal index by replacing
// it with a simpler/earlier expression. Specifically, if the index is a literal
// and the element at that index can be determined statically, the ArrayIndex
// operation is replaced with that value. Returns true if the IR was changed.
absl::StatusOr<bool> SimplifyArrayIndexWithLiteralOperand(ArrayIndex* node) {
  XLS_RET_CHECK(node->index()->Is<Literal>());
  const Bits& bits_index = node->index()->As<Literal>()->value().bits();
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

// Try to simplify the given array index operation. Returns true if the node was
// changed.
absl::StatusOr<bool> SimplifyArrayIndex(ArrayIndex* node,
                                        const QueryEngine& query_engine) {
  if (node->index()->Is<Literal>()) {
    XLS_ASSIGN_OR_RETURN(bool changed,
                         SimplifyArrayIndexWithLiteralOperand(node));
    if (changed) {
      return true;
    }
  }

  // If this array index operation indexes into the output of an array update
  // operation it may be possible to bypass the array update operation in the
  // graph if the index of the array index is known to be different than the
  // array update. That is, the following:
  //
  //             A        value
  //              \      /
  //  index_0 -> array_update
  //                  |
  //  index_1 -> array_index
  //
  // maybe transformed into the following if index_0 != index_1:
  //
  //                  A
  //                  |
  //  index_1 -> array_index
  //
  // The transformation may be applied iteratively to bypass multiple
  // consecutive array_update operations.

  // The transformation cannot be done in the index might be out of bounds (or
  // at least it is more complicated to do correctly).
  if (bits_ops::UGreaterThanOrEqual(
          query_engine.MaxUnsignedValue(node->index()),
          node->array()->GetType()->AsArrayOrDie()->size())) {
    // Index might be out of bounds.
    return false;
  }

  Node* source_array = node->array();
  while (source_array->Is<ArrayUpdate>() &&
         query_engine.NodesKnownUnsignedNotEquals(
             source_array->As<ArrayUpdate>()->index(), node->index())) {
    source_array = source_array->As<ArrayUpdate>()->array_to_update();
  }
  if (source_array != node->array()) {
    XLS_RETURN_IF_ERROR(node->ReplaceOperandNumber(0, source_array));
    return true;
  }
  return false;
}

// Returns the literal index of the given array_update operation as a uint64. If
// the array update does not have a literal index or it does not fit in a uint64
// then nullopt is returned.
absl::optional<uint64> MaybeGetLiteralIndex(ArrayUpdate* array_update) {
  if (!array_update->index()->Is<Literal>()) {
    return absl::nullopt;
  }
  const Bits& index_bits = array_update->index()->As<Literal>()->value().bits();
  if (!index_bits.FitsInUint64()) {
    return absl::nullopt;
  }
  return index_bits.ToUint64().value();
}

// Try to simplify the given array update operation. Returns true if successful.
absl::StatusOr<bool> SimplifyArrayUpdate(ArrayUpdate* array_update) {
  // Try to simplify a kArray operation followed by an ArrayUpdate operation
  // into a single kArray operation which uses the updated value. For example:
  //
  //     a  b  c
  //      \ | /
  //      Array
  //  idx   |   Value
  //     \  |  /
  //   ArrayUpdate
  //        |
  //
  // Might be transformed into:
  //
  //     a  b  Value
  //      \ | /
  //      Array
  //        |
  //
  // Assuming the index 'idx' corresponds to element 'c'. The single user
  // requirement is because 'Value' might be derived from 'Array' otherwise and
  // the transformation would introduce a loop in the grpah.
  if (array_update->array_to_update()->Is<Array>() &&
      array_update->array_to_update()->users().size() == 1 &&
      array_update->index()->Is<Literal>()) {
    int64 index;
    if (bits_ops::UGreaterThanOrEqual(
            array_update->index()->As<Literal>()->value().bits(),
            array_update->GetType()->AsArrayOrDie()->size())) {
      index = array_update->GetType()->AsArrayOrDie()->size() - 1;
    } else {
      index = array_update->index()
                  ->As<Literal>()
                  ->value()
                  .bits()
                  .ToUint64()
                  .value();
    }
    Array* array = array_update->array_to_update()->As<Array>();
    XLS_RETURN_IF_ERROR(
        array->ReplaceOperandNumber(index, array_update->update_value()));
    return array_update->ReplaceUsesWith(array);
  }

  // Try to optimize multiple consecutive array updates.
  if (array_update->array_to_update()->Is<ArrayUpdate>()) {
    ArrayUpdate* prev_array_update =
        array_update->array_to_update()->As<ArrayUpdate>();
    auto updates_same_index = [](ArrayUpdate* a, ArrayUpdate* b) {
      if (a->index() == b->index()) {
        return true;
      }
      if (a->index()->Is<Literal>() && b->index()->Is<Literal>()) {
        return bits_ops::UEqual(a->index()->As<Literal>()->value().bits(),
                                b->index()->As<Literal>()->value().bits());
      }
      return false;
    };
    if (updates_same_index(array_update, prev_array_update)) {
      // The seqeuential array update operations are updating the same
      // element. The earlier array update (prev_array_update) has no effect on
      // the result of this array update, so skip the earlier array update by
      // setting the array operand of this array update to the array prior to
      // the two array updates.
      // TODO(meheff): This could be generalized to the case where the are
      // intermediate updates at arbitrary indices in between the two updates at
      // the same index.
      Node* original_array = prev_array_update->array_to_update();
      XLS_RETURN_IF_ERROR(
          array_update->ReplaceOperandNumber(0, original_array));
      return true;
    }
  }

  // Identify cases where an array is constructed via a sequence of array update
  // operations and replace with a flat kArray operation gathering all the array
  // values.
  absl::flat_hash_map<uint64, Node*> index_to_element;

  // Walk up the chain of array updates.
  ArrayUpdate* current = array_update;
  while (true) {
    absl::optional<uint64> index = MaybeGetLiteralIndex(current);
    if (!index.has_value()) {
      break;
    }
    // If this element is already in the map, then it has already been set by a
    // later array update, this operation (current) can be ignored.
    if (!index_to_element.contains(index.value())) {
      index_to_element[index.value()] = current->update_value();
    }
    if (!current->array_to_update()->Is<ArrayUpdate>()) {
      break;
    }
    current = current->array_to_update()->As<ArrayUpdate>();
  }

  // If the map of indices is dense from 0 to array_size-1, then we can replace
  // the sequence of array-updates with a single kArray instruction which
  // gathers the array elements.
  auto indices_are_dense = [](const absl::flat_hash_map<uint64, Node*> m) {
    for (int64 i = 0; i < m.size(); ++i) {
      if (!m.contains(i)) {
        return false;
      }
    }
    return true;
  };
  if (index_to_element.size() ==
          array_update->GetType()->AsArrayOrDie()->size() &&
      indices_are_dense(index_to_element)) {
    std::vector<Node*> array_elements;
    for (int64 i = 0; i < index_to_element.size(); ++i) {
      array_elements.push_back(index_to_element.at(i));
    }
    XLS_RETURN_IF_ERROR(
        array_update
            ->ReplaceUsesWithNew<Array>(array_elements,
                                        array_elements.front()->GetType())
            .status());
    return true;
  }

  return false;
}

// Try to simplify the given array operation. Returns true if successful.
absl::StatusOr<bool> SimplifyArray(Array* array) {
  // Simplify a subgraph which simply decomposes an array into it's elements and
  // recomposes them into the same array. For example, the following expression
  // is equivalent to A if A has n elements:
  //
  //  Array(A[0], A[1], .... , A[n-1])
  //
  // "origin_array" is the he original array ("A" in the above example).
  Node* origin_array = nullptr;
  for (int64 i = 0; i < array->operand_count(); ++i) {
    if (!array->operand(i)->Is<ArrayIndex>()) {
      return false;
    }
    ArrayIndex* array_index = array->operand(i)->As<ArrayIndex>();
    // Returns true if the given array index node's index operand has the value
    // v.
    auto index_has_value = [](ArrayIndex* n, uint64 v) {
      if (!n->index()->Is<Literal>()) {
        return false;
      }
      const Bits& index = n->index()->As<Literal>()->value().bits();
      return index.FitsInUint64() && index.ToUint64().value() == v;
    };

    if (!index_has_value(array_index, i)) {
      return false;
    }
    if (origin_array == nullptr) {
      origin_array = array_index->array();
    } else if (origin_array != array_index->array()) {
      return false;
    }
  }
  if (origin_array->GetType() == array->GetType()) {
    return array->ReplaceUsesWith(origin_array);
  }
  return false;
}

}  // namespace

absl::StatusOr<bool> ArraySimplificationPass::RunOnFunctionBase(
    FunctionBase* func, const PassOptions& options,
    PassResults* results) const {
  XLS_VLOG(2) << "Running array simplifier on function " << func->name();
  XLS_VLOG(3) << "Before:";
  XLS_VLOG_LINES(3, func->DumpIr());

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<TernaryQueryEngine> query_engine,
                       TernaryQueryEngine::Run(func));

  bool changed = false;
  for (Node* node : TopoSort(func)) {
    if (node->Is<ArrayIndex>()) {
      ArrayIndex* array_index = node->As<ArrayIndex>();
      XLS_ASSIGN_OR_RETURN(bool node_changed,
                           SimplifyArrayIndex(array_index, *query_engine));
      changed = changed | node_changed;
    } else if (node->Is<Array>()) {
      XLS_ASSIGN_OR_RETURN(bool node_changed, SimplifyArray(node->As<Array>()));
      changed = changed | node_changed;
    }
  }
  for (Node* node : ReverseTopoSort(func)) {
    if (node->Is<ArrayUpdate>()) {
      XLS_ASSIGN_OR_RETURN(bool node_changed,
                           SimplifyArrayUpdate(node->As<ArrayUpdate>()));
      changed = changed | node_changed;
    }
  }

  XLS_VLOG(3) << "After:";
  XLS_VLOG_LINES(3, func->DumpIr());

  return changed;
}

}  // namespace xls
