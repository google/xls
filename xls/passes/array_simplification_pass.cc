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

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_util.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/topo_sort.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/optimization_pass_registry.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/query_engine.h"
#include "xls/passes/stateless_query_engine.h"
#include "xls/passes/ternary_query_engine.h"
#include "xls/passes/union_query_engine.h"

namespace xls {
namespace {

// Returns true if the given index value is definitely out of bounds for the
// given array type.
bool IndexIsDefinitelyOutOfBounds(Node* index, ArrayType* array_type,
                                  const QueryEngine& query_engine) {
  return bits_ops::UGreaterThanOrEqual(query_engine.MinUnsignedValue(index),
                                       array_type->size());
}

// Returns true if the given index is definitely in bounds for the given array
// type.
bool IndexIsDefinitelyInBounds(Node* index, ArrayType* array_type,
                               const QueryEngine& query_engine) {
  return bits_ops::ULessThan(query_engine.MaxUnsignedValue(index),
                             array_type->size());
}

// Returns true if the given (multidimensional) indices are definitely in bounds
// for the given (multidimensional) array type. A multidimensional index is in
// bounds iff *every one* of the indices are in bounds.
bool IndicesAreDefinitelyInBounds(absl::Span<Node* const> indices, Type* type,
                                  const QueryEngine& query_engine) {
  Type* subtype = type;
  for (Node* index : indices) {
    if (!IndexIsDefinitelyInBounds(index, subtype->AsArrayOrDie(),
                                   query_engine)) {
      return false;
    }
    subtype = subtype->AsArrayOrDie()->element_type();
  }
  return true;
}

// Returns true if the given sequences of nodes necessarily have the same
// values. I.e., a[0] == b[0] and a[1] == b[1], etc.
bool IndicesAreDefinitelyEqual(absl::Span<Node* const> a,
                               absl::Span<Node* const> b,
                               const QueryEngine& query_engine) {
  if (a.size() != b.size()) {
    return false;
  }
  for (int64_t i = 0; i < a.size(); ++i) {
    if (!query_engine.NodesKnownUnsignedEquals(a[i], b[i])) {
      return false;
    }
  }
  return true;
}

// Returns true iff the given sequences of nodes necessarily are not all equal.
// That is, a[i] is definitely not equal to b[i] for some i.
bool IndicesDefinitelyNotEqual(absl::Span<Node* const> a,
                               absl::Span<Node* const> b,
                               const QueryEngine& query_engine) {
  for (int64_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    if (query_engine.NodesKnownUnsignedNotEquals(a[i], b[i])) {
      return true;
    }
  }
  return false;
}

// Returns true the if indices 'prefix' is necessarily equal to the first
// prefix.size() elements of indices.
bool IndicesAreDefinitelyPrefixOf(absl::Span<Node* const> prefix,
                                  absl::Span<Node* const> indices,
                                  const QueryEngine& query_engine) {
  if (prefix.size() > indices.size()) {
    return false;
  }
  return IndicesAreDefinitelyEqual(prefix, indices.subspan(0, prefix.size()),
                                   query_engine);
}

// Clamp any known OOB indices in ArrayIndex ops. In this case the index is
// replaced with a literal value equal to the maximum in-bounds index value
// (size of array minus one). Only known-OOB are clamped. Maybe OOB indices
// cannot be replaced because the index might be a different in-bounds value.
absl::StatusOr<bool> ClampArrayIndexIndices(FunctionBase* func,
                                            const QueryEngine& query_engine) {
  bool changed = false;
  for (Node* node : TopoSort(func)) {
    if (node->Is<ArrayIndex>()) {
      ArrayIndex* array_index = node->As<ArrayIndex>();
      Type* subtype = array_index->array()->GetType();
      for (int64_t i = 0; i < array_index->indices().size(); ++i) {
        Node* index = array_index->indices()[i];
        ArrayType* array_type = subtype->AsArrayOrDie();
        if (IndexIsDefinitelyOutOfBounds(index, array_type, query_engine)) {
          VLOG(2) << absl::StrFormat("Clamping array index: %s",
                                     node->ToString());
          XLS_ASSIGN_OR_RETURN(
              Literal * new_index,
              func->MakeNode<Literal>(index->loc(),
                                      Value(UBits(array_type->size() - 1,
                                                  index->BitCountOrDie()))));
          // Index operands start at one so operand number is i + 1.
          XLS_RETURN_IF_ERROR(
              array_index->ReplaceOperandNumber(i + 1, new_index));
          changed = true;
        }
        subtype = array_type->element_type();
      }
    }
  }
  return changed;
}

// Result of a simplification transformation.
struct SimplifyResult {
  // Whether the graph was changed.
  bool changed;
  // Nodes which should be added back to the worklist.
  std::vector<Node*> new_worklist_nodes;

  static SimplifyResult Unchanged() {
    return SimplifyResult{.changed = false, .new_worklist_nodes = {}};
  }

  static SimplifyResult Changed(absl::Span<Node* const> new_worklist_nodes) {
    return SimplifyResult{
        .changed = true,
        .new_worklist_nodes = std::vector<Node*>(new_worklist_nodes.begin(),
                                                 new_worklist_nodes.end())};
  }
};

// Try to simplify the given array index operation.
absl::StatusOr<SimplifyResult> SimplifyArrayIndex(
    ArrayIndex* array_index, const QueryEngine& query_engine) {
  // An array index with a nil index (no index operands) can be replaced by the
  // array operand:
  //
  //   array_index(A, {}) => A
  //
  if (array_index->indices().empty()) {
    VLOG(2) << absl::StrFormat("Array-index with degenerate index: %s",
                               array_index->ToString());
    XLS_RETURN_IF_ERROR(array_index->ReplaceUsesWith(array_index->array()));
    return SimplifyResult::Changed({array_index->array()});
  }

  // An array index which indexes into a kArray operation and whose first
  // index element is a constant can be simplified by bypassing the kArray
  // operation:
  //
  //   array_index(array(a, b, c, d), {2, i, j, k, ...}
  //     => array_index(c, {i, j, k, ...})
  //
  if (array_index->array()->Is<Array>() && !array_index->indices().empty() &&
      query_engine.IsFullyKnown(array_index->indices().front())) {
    Array* array = array_index->array()->As<Array>();
    Node* first_index = array_index->indices().front();
    if (IndexIsDefinitelyInBounds(first_index, array->GetType()->AsArrayOrDie(),
                                  query_engine)) {
      // Indices are always interpreted as unsigned numbers.
      XLS_ASSIGN_OR_RETURN(
          uint64_t operand_no,
          query_engine.KnownValueAsBits(first_index)->ToUint64());
      VLOG(2) << absl::StrFormat(
          "Array-index of array operation with constant index: %s",
          array_index->ToString());
      XLS_ASSIGN_OR_RETURN(
          ArrayIndex * new_array_index,
          array_index->ReplaceUsesWithNew<ArrayIndex>(
              array->operand(operand_no), array_index->indices().subspan(1)));
      return SimplifyResult::Changed({new_array_index});
    }
  }

  // An array index which indexes into a kArrayConcat operation and whose first
  // index element is a constant can be simplified by bypassing the kArrayConcat
  // operation:
  //
  //   array_index(array_concat(A, B, C), {20, i, j, k, ...}
  //     => array_index(B, {10, i, j, k, ...})
  //
  // This assumes array A has a size 10, and B has a size of greater than 10.
  if (array_index->array()->Is<ArrayConcat>() &&
      !array_index->indices().empty() &&
      query_engine.IsFullyKnown(array_index->indices().front())) {
    Node* first_index = array_index->indices().front();
    if (IndexIsDefinitelyInBounds(
            first_index, array_index->array()->GetType()->AsArrayOrDie(),
            query_engine)) {
      XLS_ASSIGN_OR_RETURN(
          int64_t index,
          query_engine.KnownValueAsBits(array_index->indices().front())
              ->ToUint64());
      Node* indexed_operand = nullptr;
      for (Node* operand : array_index->array()->operands()) {
        XLS_RET_CHECK(operand->GetType()->IsArray());
        int64_t array_size = operand->GetType()->AsArrayOrDie()->size();
        if (index < array_size) {
          indexed_operand = operand;
          break;
        }
        index = index - array_size;
      }
      XLS_RET_CHECK(indexed_operand != nullptr);
      VLOG(2) << absl::StrFormat(
          "Array-index of array concat with constant index: %s",
          array_index->ToString());

      std::vector<Node*> new_indices(array_index->indices().begin(),
                                     array_index->indices().end());
      XLS_ASSIGN_OR_RETURN(
          new_indices[0],
          array_index->function_base()->MakeNode<Literal>(
              array_index->loc(),
              Value(UBits(index,
                          array_index->indices().front()->BitCountOrDie()))));

      XLS_ASSIGN_OR_RETURN(ArrayIndex * new_array_index,
                           array_index->ReplaceUsesWithNew<ArrayIndex>(
                               indexed_operand, new_indices));
      return SimplifyResult::Changed({new_array_index});
    }
  }

  // Consecutive array index operations can be combined. For example:
  //
  //   array_index(array_index(A, {a, b}), {c, d})
  //     => array_index(A, {a, b, c, d})
  //
  if (array_index->array()->Is<ArrayIndex>()) {
    ArrayIndex* operand = array_index->array()->As<ArrayIndex>();
    std::vector<Node*> combined_indices(operand->indices().begin(),
                                        operand->indices().end());
    combined_indices.insert(combined_indices.end(),
                            array_index->indices().begin(),
                            array_index->indices().end());
    VLOG(2) << absl::StrFormat("Consecutive array-index operations: %s",
                               array_index->ToString());
    XLS_ASSIGN_OR_RETURN(ArrayIndex * new_array_index,
                         array_index->ReplaceUsesWithNew<ArrayIndex>(
                             operand->array(), combined_indices));
    return SimplifyResult::Changed({new_array_index});
  }

  // Convert an array_index of a select into a select of array_indexes:
  //
  //   array_index(select(p, cases=[A0, A1]), {idx})
  //     => select(p, array_index(A0, {idx}), array_index(A1, {idx}))
  //
  // This reduces the width of the resulting mux.
  //
  // Only perform this optimization if the array_index is the only user.
  // Otherwise the array index(es) are duplicated which can outweigh the benefit
  // of selecting the smaller element.
  // TODO(meheff): Consider cases where selects with multiple users are still
  // advantageous to transform.
  if ((array_index->array()->Is<Select>() ||
       array_index->array()->Is<PrioritySelect>()) &&
      HasSingleUse(array_index->array())) {
    VLOG(2) << absl::StrFormat(
        "Replacing array-index of select with select of array-indexes: %s",
        array_index->ToString());
    absl::Span<Node* const> original_cases;
    std::optional<Node*> original_default_value;
    if (array_index->array()->Is<Select>()) {
      Select* select = array_index->array()->As<Select>();
      original_cases = select->cases();
      original_default_value = select->default_value();
    } else {
      XLS_RET_CHECK(array_index->array()->Is<PrioritySelect>());
      PrioritySelect* select = array_index->array()->As<PrioritySelect>();
      original_cases = select->cases();
      original_default_value = select->default_value();
    }

    std::vector<Node*> cases;
    cases.reserve(original_cases.size());
    for (Node* case_value : original_cases) {
      XLS_ASSIGN_OR_RETURN(
          ArrayIndex * case_array_index,
          array_index->function_base()->MakeNode<ArrayIndex>(
              array_index->loc(), case_value, array_index->indices()));
      cases.push_back(case_array_index);
    }

    std::optional<Node*> default_value;
    if (original_default_value.has_value()) {
      XLS_ASSIGN_OR_RETURN(default_value,
                           array_index->function_base()->MakeNode<ArrayIndex>(
                               array_index->loc(), *original_default_value,
                               array_index->indices()));
    }

    Node* new_select;
    if (array_index->array()->Is<Select>()) {
      XLS_ASSIGN_OR_RETURN(
          new_select, array_index->ReplaceUsesWithNew<Select>(
                          array_index->array()->As<Select>()->selector(), cases,
                          default_value));
    } else {
      XLS_RET_CHECK(array_index->array()->Is<PrioritySelect>());
      XLS_RET_CHECK(default_value.has_value());
      XLS_ASSIGN_OR_RETURN(
          new_select,
          array_index->ReplaceUsesWithNew<PrioritySelect>(
              array_index->array()->As<PrioritySelect>()->selector(), cases,
              *default_value));
    }

    std::vector<Node*> changed = std::move(cases);
    if (default_value.has_value()) {
      changed.push_back(*default_value);
    }
    changed.push_back(new_select);
    return SimplifyResult::Changed(changed);
  }

  // If this array index operation indexes into the output of an array update
  // operation it may be possible to bypass the array update operation in the
  // graph if the index of the array index is known to be the same or
  // known to be different than the array update. That is, consider the
  // following:
  //
  //                           A        value
  //                            \      /
  //  index {i_0, ... i_m} -> array_update
  //                               |
  //  index {j_0, ... j_n} -> array_index
  //
  // This might be transformed into the one of the following if both sets of
  // indices are definitely inbounds.
  //
  //  (1) {i_0, ... i_m} is a prefix of {j_0, ... j_n}, i.e.:
  //
  //        i_0 == j_0 && i_1 == j_1 && ... && i_m == j_m
  //
  //      Where n >= m. The array index necessarily indexes into the subarray
  //      updated with value so the index can directly index into value.
  //
  //                                     value
  //                                       |
  //        index {j_m+1, ... j_n} -> array_index
  //
  //
  //  (2) {i_0, ..., i_m} is definitely *not* a prefix of {j_0, ..., j_n}. The
  //      array index does *not* index into the updated part of the array. The
  //      original array can be indexed directly:
  //
  //                                       A
  //                                       |
  //        index {j_0, ... j_n} -> array_index
  //
  if (!array_index->array()->Is<ArrayUpdate>()) {
    return SimplifyResult::Unchanged();
  }
  auto* array_update = array_index->array()->As<ArrayUpdate>();

  // Consider case (1) above, where the array_update indices are a prefix of the
  // array_index indices. If the update is in-bounds (so takes effect), we can
  // index directly into the update value.
  if (IndicesAreDefinitelyPrefixOf(array_update->indices(),
                                   array_index->indices(), query_engine) &&
      IndicesAreDefinitelyInBounds(array_update->indices(),
                                   array_update->array_to_update()->GetType(),
                                   query_engine)) {
    // Remove the matching prefix from the
    // front of the array-index indices because the new array-index indexes into
    // the lower dimensional update value. so
    VLOG(2) << absl::StrFormat("Array-index of array-update (case 1): %s",
                               array_index->ToString());
    XLS_ASSIGN_OR_RETURN(
        ArrayIndex * new_array_index,
        array_index->ReplaceUsesWithNew<ArrayIndex>(
            array_update->update_value(),
            array_index->indices().subspan(array_update->indices().size())));
    return SimplifyResult::Changed({new_array_index});
  }

  // Consider case (2) above, where the array_index indices are definitely not
  // an extension of the array_update indices. If the array_index is in-bounds,
  // then we know it references an unchanged value, so we can index directly
  // into the unchanged array. (If it's out-of-bounds, we might be clamped back
  // to a location that's affected by the array_update.)
  if (IndicesDefinitelyNotEqual(array_update->indices(), array_index->indices(),
                                query_engine) &&
      IndicesAreDefinitelyInBounds(array_index->indices(),
                                   array_index->array()->GetType(),
                                   query_engine)) {
    VLOG(2) << absl::StrFormat("Array-index of array-update (case 2): %s",
                               array_index->ToString());
    XLS_RETURN_IF_ERROR(
        array_index->ReplaceOperandNumber(0, array_update->array_to_update()));
    return SimplifyResult::Changed({array_index});
  }

  return SimplifyResult::Unchanged();
}

// Try to simplify the given array update operation.
absl::StatusOr<SimplifyResult> SimplifyArrayUpdate(
    ArrayUpdate* array_update, const QueryEngine& query_engine) {
  FunctionBase* func = array_update->function_base();

  // An array update with a nil index (no index operands) can be replaced by the
  // the update value (the "array" operand is unused).
  //
  //   array_update(A, v, {}) => v
  //
  if (array_update->indices().empty()) {
    VLOG(2) << absl::StrFormat("Array-update with degenerate index: %s",
                               array_update->ToString());
    XLS_RETURN_IF_ERROR(
        array_update->ReplaceUsesWith(array_update->update_value()));
    return SimplifyResult::Changed({array_update});
  }

  // Try to simplify a kArray operation followed by an ArrayUpdate operation
  // into a single kArray operation which uses the updated value. For example:
  //
  //           a  b  c
  //            \ | /
  //            Array
  // {idx, ...}   |   Value
  //          \   |  /
  //          ArrayUpdate
  //              |
  //
  // Might be transformed into:
  //
  //          {...}  c   Value
  //             \   |   /
  //     a  b  ArrayUpdate
  //      \ | /
  //      Array
  //        |
  //
  // Assuming the index 'idx' is a constant corresponding to element 'c'. The
  // advantage is the array update is operating on a smaller array, and if the
  // index is empty after this operation the array update can be removed.
  //
  // The single user requirement on Array is because 'Value' or '{idx, ...}'
  // might be derived from 'Array' otherwise and the transformation would
  // introduce a loop in the graph.
  if (array_update->array_to_update()->Is<Array>() &&
      HasSingleUse(array_update->array_to_update()) &&
      !array_update->indices().empty() &&
      query_engine.IsFullyKnown(array_update->indices().front())) {
    Node* idx = array_update->indices().front();
    if (IndexIsDefinitelyInBounds(idx, array_update->GetType()->AsArrayOrDie(),
                                  query_engine)) {
      VLOG(2) << absl::StrFormat("Hoist array update above array: %s",
                                 array_update->ToString());

      // Indices are always interpreted as unsigned numbers.
      XLS_ASSIGN_OR_RETURN(uint64_t operand_no,
                           query_engine.KnownValueAsBits(idx)->ToUint64());
      Array* array = array_update->array_to_update()->As<Array>();

      Node* replacement_array_operand;
      if (array_update->indices().size() == 1) {
        // idx was the only element of the index. The array-update operation can
        // be elided.
        replacement_array_operand = array_update->update_value();
      } else {
        XLS_ASSIGN_OR_RETURN(
            replacement_array_operand,
            func->MakeNode<ArrayUpdate>(array_update->loc(),
                                        array->operand(operand_no),
                                        array_update->update_value(),
                                        array_update->indices().subspan(1)));
      }
      XLS_RETURN_IF_ERROR(
          array->ReplaceOperandNumber(operand_no, replacement_array_operand));
      XLS_RETURN_IF_ERROR(array_update->ReplaceUsesWith(array));

      return SimplifyResult::Changed({array, replacement_array_operand});
    }
  }

  // Replace sequential updates which effectively update one element with a
  // single array update.
  //
  //  array_update(A, array_update(array_index(A, {i}), v, {j}), {i}) =>
  //     array_update(A, v, {i, j})
  //
  if (array_update->update_value()->Is<ArrayUpdate>()) {
    ArrayUpdate* subupdate = array_update->update_value()->As<ArrayUpdate>();
    if (subupdate->array_to_update()->Is<ArrayIndex>()) {
      ArrayIndex* subindex = subupdate->array_to_update()->As<ArrayIndex>();
      if (subindex->array() == array_update->array_to_update() &&
          IndicesAreDefinitelyInBounds(subindex->indices(),
                                       subindex->array()->GetType(),
                                       query_engine) &&
          IndicesAreDefinitelyEqual(subindex->indices(),
                                    array_update->indices(), query_engine)) {
        std::vector<Node*> new_update_indices(subindex->indices().begin(),
                                              subindex->indices().end());
        new_update_indices.insert(new_update_indices.end(),
                                  subupdate->indices().begin(),
                                  subupdate->indices().end());
        VLOG(2) << absl::StrFormat("Sequential array update above array: %s",
                                   array_update->ToString());
        XLS_ASSIGN_OR_RETURN(
            ArrayUpdate * new_array_update,
            array_update->ReplaceUsesWithNew<ArrayUpdate>(
                array_update->array_to_update(), subupdate->update_value(),
                new_update_indices));
        return SimplifyResult::Changed({new_array_update});
      }
    }
  }

  // Try to optimize a chain of single-use consecutive array updates. If the
  // later array update necessarily overwrites elements updated by the earlier
  // array update, then the earlier array update may be elided.
  {
    ArrayUpdate* current = array_update;
    while (current->array_to_update()->Is<ArrayUpdate>()) {
      if (current != array_update && current->users().size() > 1) {
        break;
      }
      ArrayUpdate* prev = current->array_to_update()->As<ArrayUpdate>();
      if (IndicesAreDefinitelyPrefixOf(array_update->indices(), prev->indices(),
                                       query_engine)) {
        // array_update necessarily overwrites the values updated in the
        // array update 'prev'. 'prev' update can be elided.
        VLOG(3) << absl::StreamFormat(
            "In chain of updates starting at %s, an index is written more than "
            "once, skipping the first write of this index (%s)",
            array_update->GetName(), prev->GetName());
        VLOG(2) << absl::StrFormat("Chain of single-use array updates: %s",
                                   array_update->ToString());
        XLS_RETURN_IF_ERROR(
            current->ReplaceOperandNumber(0, prev->array_to_update()));

        return SimplifyResult::Changed({array_update, current});
      }
      current = prev;
    }
  }

  // If the array to update is a constant and the first index is a constant, the
  // array update can be replaced with a kArray operation which assembles the
  // updated value with element literals. Example:
  //
  //          [A, B, C, D]
  //               |
  //  {idx, ...}   |    value
  //           \   |   /
  //        array_update
  //
  // If idx is 2, this might be transformed into:
  //
  //      {...}  C    value
  //         \   |   /
  //         array_update
  //             |
  //         A B | D
  //          \ \|/
  //           Array
  //
  //
  // The advantage is that a smaller array is updated. This can result in
  // elimination of the array update entirely as well if the index is empty
  // after the transformation.
  if (query_engine.IsFullyKnown(array_update->array_to_update()) &&
      !array_update->indices().empty() &&
      query_engine.IsFullyKnown(array_update->indices().front())) {
    Node* idx = array_update->indices().front();
    if (IndexIsDefinitelyInBounds(idx, array_update->GetType()->AsArrayOrDie(),
                                  query_engine)) {
      VLOG(2) << absl::StrFormat(
          "Array-update of constant with constant index: %s",
          array_update->ToString());

      // Indices are always interpreted as unsigned numbers.
      XLS_ASSIGN_OR_RETURN(uint64_t operand_no,
                           query_engine.KnownValueAsBits(idx)->ToUint64());
      Value array_constant =
          *query_engine.KnownValue(array_update->array_to_update());
      XLS_RET_CHECK_LT(operand_no, array_constant.size());

      std::vector<Node*> array_operands;
      ArrayUpdate* new_array_update = nullptr;
      for (int64_t i = 0; i < array_constant.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(
            Literal * array_element,
            func->MakeNode<Literal>(array_update->loc(),
                                    array_constant.element(i)));
        if (i == operand_no) {
          XLS_ASSIGN_OR_RETURN(
              new_array_update,
              func->MakeNode<ArrayUpdate>(array_update->loc(), array_element,
                                          array_update->update_value(),
                                          array_update->indices().subspan(1)));
          array_operands.push_back(new_array_update);
        } else {
          array_operands.push_back(array_element);
        }
      }
      XLS_ASSIGN_OR_RETURN(
          Array * new_array,
          array_update->ReplaceUsesWithNew<Array>(
              array_operands,
              /*element_type=*/array_operands.front()->GetType()));
      return SimplifyResult::Changed({new_array, new_array_update});
    }
  }
  return SimplifyResult::Unchanged();
}

// Tries to flatten a chain of consecutive array update operations into a single
// kArray op which gathers the elements written into the array. Returns a vector
// of the optimized away update operations or nullopt is no optimization was
// performed.
absl::StatusOr<std::optional<std::vector<ArrayUpdate*>>>
FlattenArrayUpdateChain(ArrayUpdate* array_update,
                        const QueryEngine& query_engine) {
  // Identify cases where an array is constructed via a sequence of array update
  // operations and replace with a flat kArray operation gathering all the array
  // values.
  absl::flat_hash_map<uint64_t, Node*> index_to_element;

  if (array_update->indices().empty()) {
    return std::nullopt;
  }

  XLS_ASSIGN_OR_RETURN(
      Type * subarray_type,
      GetIndexedElementType(array_update->GetType(),
                            array_update->indices().size() - 1));
  int64_t subarray_size = subarray_type->AsArrayOrDie()->size();

  // Walk up the chain of array updates.
  ArrayUpdate* current = array_update;
  Node* source_array = nullptr;
  std::optional<absl::Span<Node* const>> common_index_prefix;
  std::vector<ArrayUpdate*> update_chain;
  while (true) {
    if (!query_engine.IsFullyKnown(current->indices().back())) {
      break;
    }

    Bits index_bits = *query_engine.KnownValueAsBits(current->indices().back());
    if (bits_ops::UGreaterThanOrEqual(index_bits, subarray_size)) {
      // Index is out of bound
      break;
    }
    uint64_t index = index_bits.ToUint64().value();

    absl::Span<Node* const> index_prefix =
        current->indices().subspan(0, current->indices().size() - 1);
    if (common_index_prefix.has_value()) {
      if (common_index_prefix.value() != index_prefix) {
        break;
      }
    } else {
      common_index_prefix = index_prefix;
    }

    // If this element is already in the map, then it has already been set by a
    // later array update, this operation (current) can be ignored.
    if (!index_to_element.contains(index)) {
      index_to_element[index] = current->update_value();
    }
    update_chain.push_back(current);
    source_array = current->array_to_update();
    if (!source_array->Is<ArrayUpdate>()) {
      break;
    }
    current = source_array->As<ArrayUpdate>();
  }

  if (!common_index_prefix.has_value()) {
    // No transformation possible.
    return std::nullopt;
  }
  XLS_RET_CHECK(source_array != nullptr);
  XLS_RET_CHECK(!update_chain.empty());

  // TODO(meheff): If at least half (>=) of the values are set then replace.
  if (index_to_element.size() < (subarray_size + 1) / 2) {
    return std::nullopt;
  }

  VLOG(2) << absl::StrFormat("Flattening chain of array-updates: %s",
                             array_update->ToString());

  std::vector<Node*> array_elements;
  for (int64_t i = 0; i < subarray_size; ++i) {
    if (index_to_element.contains(i)) {
      array_elements.push_back(index_to_element.at(i));
    } else {
      XLS_ASSIGN_OR_RETURN(Literal * literal_index,
                           array_update->function_base()->MakeNode<Literal>(
                               array_update->loc(), Value(UBits(i, 64))));
      std::vector<Node*> indices(common_index_prefix.value().begin(),
                                 common_index_prefix.value().end());
      indices.push_back(literal_index);
      XLS_ASSIGN_OR_RETURN(ArrayIndex * array_index,
                           array_update->function_base()->MakeNode<ArrayIndex>(
                               array_update->loc(), source_array, indices));
      array_elements.push_back(array_index);
    }
  }
  XLS_ASSIGN_OR_RETURN(Array * array,
                       array_update->function_base()->MakeNode<Array>(
                           array_update->loc(), array_elements,
                           array_elements.front()->GetType()));

  if (common_index_prefix->empty()) {
    XLS_RETURN_IF_ERROR(array_update->ReplaceUsesWith(array));
  } else {
    XLS_RETURN_IF_ERROR(
        array_update
            ->ReplaceUsesWithNew<ArrayUpdate>(source_array, array,
                                              common_index_prefix.value())
            .status());
  }

  return update_chain;
}

// Walk the function and replace chains of sequential array updates with kArray
// operations with gather the update values.
absl::StatusOr<bool> FlattenSequentialUpdates(FunctionBase* func,
                                              const QueryEngine& query_engine) {
  absl::flat_hash_set<ArrayUpdate*> flattened_updates;
  bool changed = false;
  // Perform this optimization in reverse topo sort order because we are looking
  // for a sequence of array update operations and the search progress upwards
  // (toward parameters).
  for (Node* node : ReverseTopoSort(func)) {
    if (!node->Is<ArrayUpdate>()) {
      continue;
    }
    ArrayUpdate* array_update = node->As<ArrayUpdate>();
    if (flattened_updates.contains(array_update)) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(std::optional<std::vector<ArrayUpdate*>> flattened_vec,
                         FlattenArrayUpdateChain(array_update, query_engine));
    if (flattened_vec.has_value()) {
      changed = true;
      flattened_updates.insert(flattened_vec->begin(), flattened_vec->end());
    }
  }
  return changed;
}

// Try to simplify the given array operation.
absl::StatusOr<SimplifyResult> SimplifyArray(Array* array,
                                             const QueryEngine& query_engine) {
  // Simplify a subgraph which simply decomposes an array into it's elements and
  // recomposes them into the same array. For example, the following
  // transformation can be performed if A has N elements:
  //
  //   Array(ArrayIndex(A, {i, j, 0}),
  //         ArrayIndex(A, {i, j, 1}),
  //         ...
  //         ArrayIndex(A, {i, j, N}))
  //
  //     =>
  //
  //   ArrayIndex(A, {i, j})
  //
  Node* origin_array = nullptr;
  std::optional<std::vector<Node*>> common_index_prefix;
  for (int64_t i = 0; i < array->operand_count(); ++i) {
    if (!array->operand(i)->Is<ArrayIndex>()) {
      return SimplifyResult::Unchanged();
    }
    ArrayIndex* array_index = array->operand(i)->As<ArrayIndex>();
    if (array_index->indices().empty()) {
      return SimplifyResult::Unchanged();
    }

    // Extract the last element of the index as a uint64_t.
    Node* last_index_node = array_index->indices().back();
    if (!query_engine.IsFullyKnown(last_index_node)) {
      return SimplifyResult::Unchanged();
    }
    Bits last_index_bits = *query_engine.KnownValueAsBits(last_index_node);
    if (!last_index_bits.FitsInUint64()) {
      return SimplifyResult::Unchanged();
    }
    uint64_t last_index = last_index_bits.ToUint64().value();

    // The final index element (0 .. N in the example above) must be sequential.
    if (last_index != i) {
      return SimplifyResult::Unchanged();
    }

    // The prefix of the index ({i, j} in the example above must match the
    // prefixes of the other array index operations.
    absl::Span<Node* const> prefix_span =
        array_index->indices().subspan(0, array_index->indices().size() - 1);
    if (common_index_prefix.has_value()) {
      if (!IndicesAreDefinitelyEqual(common_index_prefix.value(), prefix_span,
                                     query_engine)) {
        return SimplifyResult::Unchanged();
      }
    } else {
      common_index_prefix =
          std::vector<Node*>(prefix_span.begin(), prefix_span.end());
    }

    // The array index operations must all index into the same array ("A" in
    // the example above).
    if (origin_array == nullptr) {
      origin_array = array_index->array();
    } else if (origin_array != array_index->array()) {
      return SimplifyResult::Unchanged();
    }
  }

  XLS_RET_CHECK(common_index_prefix.has_value());
  XLS_ASSIGN_OR_RETURN(Type * origin_array_subtype,
                       GetIndexedElementType(origin_array->GetType(),
                                             common_index_prefix->size()));
  if (array->GetType() == origin_array_subtype) {
    VLOG(2) << absl::StrFormat("Replace array of array-indexes with array: %s",
                               array->ToString());
    if (common_index_prefix->empty()) {
      VLOG(3) << absl::StrFormat("  Replacing with original array %s",
                                 origin_array->ToString());
      XLS_RETURN_IF_ERROR(array->ReplaceUsesWith(origin_array));
      return SimplifyResult::Changed({origin_array});
    }
    VLOG(3) << absl::StrFormat("  Replacing with index on origin array %s",
                               origin_array->ToString());
    XLS_ASSIGN_OR_RETURN(ArrayIndex * array_index,
                         array->ReplaceUsesWithNew<ArrayIndex>(
                             origin_array, common_index_prefix.value()));
    return SimplifyResult::Changed({array_index});
  }
  return SimplifyResult::Unchanged();
}

// Simplify the conditional updating of an array element of the following form:
//
//   Select(p, cases=[A, array_update(A, v, {idx})])
//
// This pattern is replaced with:
//
//   array_update(A, select(p, cases=[array_index(A, {idx}), v]), {idx})
//
// The advantage is the select (mux) is only selecting an array element rather
// than the entire array.
absl::StatusOr<SimplifyResult> SimplifyConditionalAssign(Node* select) {
  absl::Span<Node* const> original_cases;
  std::optional<Node*> original_default_value;
  if (select->Is<Select>()) {
    original_cases = select->As<Select>()->cases();
    original_default_value = select->As<Select>()->default_value();
  } else {
    XLS_RET_CHECK(select->Is<PrioritySelect>());
    original_cases = select->As<PrioritySelect>()->cases();
    original_default_value = select->As<PrioritySelect>()->default_value();
  }

  struct IdentityValue : std::monostate {};
  Node* array_to_update = nullptr;
  std::optional<absl::Span<Node* const>> idx = std::nullopt;
  std::vector<std::variant<Node*, IdentityValue>> cases;
  cases.reserve(original_cases.size());
  std::optional<std::variant<Node*, IdentityValue>> default_case = std::nullopt;
  auto extract_case =
      [&](Node* sel_case) -> std::optional<std::variant<Node*, IdentityValue>> {
    if (!sel_case->Is<ArrayUpdate>()) {
      if (array_to_update == nullptr) {
        array_to_update = sel_case;
      } else if (array_to_update != sel_case) {
        // Not just a select of array updates to the same array.
        return std::nullopt;
      }
      return IdentityValue();
    }

    ArrayUpdate* update = sel_case->As<ArrayUpdate>();
    if (!HasSingleUse(update)) {
      // Multiple uses for the array update; we don't want to replace just one.
      return std::nullopt;
    }
    if (array_to_update == nullptr) {
      array_to_update = update->array_to_update();
    } else if (array_to_update != update->array_to_update()) {
      // Not just a select of array updates to the same array.
      return std::nullopt;
    }
    if (idx == std::nullopt) {
      idx = update->indices();
    } else if (*idx != update->indices()) {
      // Not just a select of array updates to the same index.
      return std::nullopt;
    }
    return update->update_value();
  };
  for (Node* sel_case : original_cases) {
    std::optional<std::variant<Node*, IdentityValue>> case_value =
        extract_case(sel_case);
    if (!case_value.has_value()) {
      // Doesn't match the pattern; this simplification doesn't apply.
      return SimplifyResult::Unchanged();
    }
    cases.push_back(*case_value);
  }
  if (original_default_value.has_value()) {
    std::optional<std::variant<Node*, IdentityValue>> default_case_value =
        extract_case(*original_default_value);
    if (!default_case_value.has_value()) {
      // Doesn't match the pattern; this simplification doesn't apply.
      return SimplifyResult::Unchanged();
    }
    default_case = *default_case_value;
  }
  if (array_to_update == nullptr || !idx.has_value()) {
    // Doesn't match the pattern; this simplification doesn't apply.
    return SimplifyResult::Unchanged();
  }

  VLOG(2) << absl::StrFormat("Hoist select above array-update(s): %s",
                             select->ToString());

  std::vector<Node*> case_values;
  std::optional<Node*> default_value;
  std::optional<Node*> shared_original_value;
  auto get_original_value = [&]() -> absl::StatusOr<Node*> {
    if (!shared_original_value.has_value()) {
      XLS_ASSIGN_OR_RETURN(shared_original_value,
                           select->function_base()->MakeNode<ArrayIndex>(
                               select->loc(), array_to_update, *idx));
    }
    return *shared_original_value;
  };
  for (std::variant<Node*, IdentityValue> c : cases) {
    if (std::holds_alternative<Node*>(c)) {
      case_values.push_back(std::get<Node*>(c));
      continue;
    }
    XLS_RET_CHECK(std::holds_alternative<IdentityValue>(c));
    XLS_ASSIGN_OR_RETURN(Node * original_value, get_original_value());
    case_values.push_back(original_value);
  }
  if (default_case.has_value()) {
    if (std::holds_alternative<Node*>(*default_case)) {
      default_value = std::get<Node*>(*default_case);
    } else {
      XLS_RET_CHECK(std::holds_alternative<IdentityValue>(*default_case));
      XLS_ASSIGN_OR_RETURN(Node * original_value, get_original_value());
      default_value = original_value;
    }
  }

  Node* selected_value;
  if (select->Is<Select>()) {
    XLS_ASSIGN_OR_RETURN(selected_value,
                         select->function_base()->MakeNode<Select>(
                             select->loc(), select->As<Select>()->selector(),
                             /*cases=*/
                             case_values,
                             /*default_value=*/default_value));
  } else {
    XLS_RET_CHECK(select->Is<PrioritySelect>());
    XLS_RET_CHECK(default_value.has_value());
    XLS_ASSIGN_OR_RETURN(
        selected_value,
        select->function_base()->MakeNode<PrioritySelect>(
            select->loc(), select->As<PrioritySelect>()->selector(),
            /*cases=*/
            case_values,
            /*default_value=*/*default_value));
  }

  XLS_ASSIGN_OR_RETURN(ArrayUpdate * overall_update,
                       select->ReplaceUsesWithNew<ArrayUpdate>(
                           array_to_update, selected_value, *idx));
  if (shared_original_value.has_value()) {
    return SimplifyResult::Changed(
        {*shared_original_value, selected_value, overall_update});
  }
  return SimplifyResult::Changed({selected_value, overall_update});
}

// Simplify a select of arrays to an array of select.
//
//   Sel(p, cases=[Array(), Array()]) => Array(Sel(p, ...), Sel(p, ...))
//
// The advantage is that hoisting the selects may be open opportunities for
// further optimization.  On the other hand, this can replicate the select
// logic, which can be expensive in area, so we limit by the number of cases.
absl::StatusOr<SimplifyResult> SimplifySelectOfArrays(Node* select) {
  absl::Span<Node* const> original_cases;
  std::optional<Node*> original_default_value;
  if (select->Is<Select>()) {
    original_cases = select->As<Select>()->cases();
    original_default_value = select->As<Select>()->default_value();
  } else {
    XLS_RET_CHECK(select->Is<PrioritySelect>());
    original_cases = select->As<PrioritySelect>()->cases();
    original_default_value = select->As<PrioritySelect>()->default_value();
  }

  for (Node* sel_case : original_cases) {
    if (!sel_case->Is<Array>()) {
      return SimplifyResult::Unchanged();
    }
  }
  if (original_default_value.has_value() &&
      !original_default_value.value()->Is<Array>()) {
    return SimplifyResult::Unchanged();
  }

  constexpr int64_t kMaxSelectCopies = 2;
  const int64_t num_cases =
      original_cases.size() + (original_default_value.has_value() ? 1 : 0);
  if (num_cases > kMaxSelectCopies) {
    return SimplifyResult::Unchanged();
  }

  VLOG(2) << absl::StrFormat(
      "Convert select of arrays to arrays of selects: %s", select->ToString());

  ArrayType* array_type = select->GetType()->AsArrayOrDie();
  std::vector<Node*> selected_elements;
  for (int64_t i = 0; i < array_type->size(); ++i) {
    std::vector<Node*> elements;
    std::optional<Node*> default_element;
    for (Node* sel_case : original_cases) {
      elements.push_back(sel_case->operand(i));
    }
    if (original_default_value.has_value()) {
      default_element = original_default_value.value()->operand(i);
    }
    Node* selected_element;
    if (select->Is<Select>()) {
      XLS_ASSIGN_OR_RETURN(
          selected_element,
          select->function_base()->MakeNode<Select>(
              select->loc(), select->As<Select>()->selector(),
              /*cases=*/elements, /*default=*/default_element));
    } else {
      XLS_RET_CHECK(select->Is<PrioritySelect>());
      XLS_RET_CHECK(default_element.has_value());
      XLS_ASSIGN_OR_RETURN(
          selected_element,
          select->function_base()->MakeNode<PrioritySelect>(
              select->loc(), select->As<PrioritySelect>()->selector(),
              /*cases=*/elements, /*default=*/*default_element));
    }
    selected_elements.push_back(selected_element);
  }
  XLS_ASSIGN_OR_RETURN(Array * new_array,
                       select->ReplaceUsesWithNew<Array>(
                           selected_elements, array_type->element_type()));
  // The nodes to add to the worklist are all the selected elements and the
  // newly created array node. Reuse selected_elements for this purpose.
  selected_elements.push_back(new_array);
  return SimplifyResult::Changed(selected_elements);
}

// Simplify various forms of a select of array-typed values.
absl::StatusOr<SimplifyResult> SimplifySelect(Node* select,
                                              const QueryEngine& query_engine) {
  XLS_RET_CHECK(select->Is<Select>() || select->Is<PrioritySelect>());

  XLS_ASSIGN_OR_RETURN(SimplifyResult conditional_assign_result,
                       SimplifyConditionalAssign(select));
  if (conditional_assign_result.changed) {
    return conditional_assign_result;
  }

  XLS_ASSIGN_OR_RETURN(SimplifyResult select_of_arrays_result,
                       SimplifySelectOfArrays(select));
  if (select_of_arrays_result.changed) {
    return select_of_arrays_result;
  }

  return SimplifyResult::Unchanged();
}

}  // namespace

absl::StatusOr<bool> ArraySimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const OptimizationPassOptions& options,
    PassResults* results) const {
  bool changed = false;

  std::vector<std::unique_ptr<QueryEngine>> query_engines;
  query_engines.push_back(std::make_unique<StatelessQueryEngine>());
  query_engines.push_back(std::make_unique<TernaryQueryEngine>());

  UnionQueryEngine query_engine(std::move(query_engines));
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());

  // Replace known OOB indicates with clamped value. This helps later
  // optimizations.
  XLS_ASSIGN_OR_RETURN(bool clamp_changed,
                       ClampArrayIndexIndices(func, query_engine));
  changed = changed || clamp_changed;

  // Clamping the array indices results in new nodes that we *do* want to be
  // able to fully optimize, so we re-populate the query engine.
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());

  // Before the worklist-driven optimization look and replace "macro" patterns
  // such as constructing an entire array with array update operations,
  // transforming selects of array to array of selects, etc.
  XLS_ASSIGN_OR_RETURN(bool flatten_changed,
                       FlattenSequentialUpdates(func, query_engine));
  changed = changed || flatten_changed;

  std::deque<Node*> worklist;
  absl::flat_hash_set<Node*> worklist_set;
  auto add_to_worklist = [&](Node* n, bool front) {
    if (worklist_set.contains(n)) {
      return;
    }
    worklist_set.insert(n);
    if (front) {
      worklist.push_front(n);
    } else {
      worklist.push_back(n);
    }
  };

  auto remove_from_worklist = [&]() {
    Node* n = worklist.front();
    worklist.pop_front();
    worklist_set.erase(n);
    return n;
  };

  // Seed the worklist with all Array, ArrayIndex, ArrayUpdate, Select, and
  // PrioritySelect operations. Inserting them in a topo sort has a dramatic
  // effect on compile time due to interactions between the optimization which
  // replaces array-indexes of select with selects of array-indexes, and other
  // optimizations.
  for (Node* node : TopoSort(func)) {
    if (!node->IsDead() &&
        node->OpIn({Op::kArray, Op::kArrayIndex, Op::kArrayUpdate, Op::kSel,
                    Op::kPrioritySel})) {
      add_to_worklist(node, false);
    }
  }

  while (!worklist.empty()) {
    Node* node = remove_from_worklist();

    if (node->IsDead()) {
      continue;
    }

    SimplifyResult result = {.changed = false, .new_worklist_nodes = {}};
    if (node->Is<ArrayIndex>()) {
      ArrayIndex* array_index = node->As<ArrayIndex>();
      XLS_ASSIGN_OR_RETURN(result,
                           SimplifyArrayIndex(array_index, query_engine));
    } else if (node->Is<ArrayUpdate>()) {
      XLS_ASSIGN_OR_RETURN(
          result, SimplifyArrayUpdate(node->As<ArrayUpdate>(), query_engine));
    } else if (node->Is<Array>()) {
      XLS_ASSIGN_OR_RETURN(result,
                           SimplifyArray(node->As<Array>(), query_engine));
    } else if (node->Is<Select>() || node->Is<PrioritySelect>()) {
      XLS_ASSIGN_OR_RETURN(result, SimplifySelect(node, query_engine));
    }

    // Add newly changed nodes to the worklist.
    if (!result.new_worklist_nodes.empty()) {
      for (Node* n : result.new_worklist_nodes) {
        add_to_worklist(n, true);
      }
    }
    changed = changed || result.changed;
  }

  return changed;
}

REGISTER_OPT_PASS(ArraySimplificationPass, pass_config::kOptLevel);

}  // namespace xls
