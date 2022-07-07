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
#include "xls/common/logging/logging.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/node.h"
#include "xls/ir/node_iterator.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/ir/value_helpers.h"
#include "xls/passes/ternary_query_engine.h"

namespace xls {
namespace {

// Returns true if the given node is a binary select (two cases, no default).
bool IsBinarySelect(Node* node) {
  if (!node->Is<Select>()) {
    return false;
  }
  Select* sel = node->As<Select>();
  return sel->cases().size() == 2 && !sel->default_value().has_value();
}

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
absl::StatusOr<bool> ClampArrayIndexIndices(FunctionBase* func) {
  // This transformation may add nodes to the graph which invalidates the query
  // engine for later use, so create a private engine for exclusive use of this
  // transformation.
  TernaryQueryEngine query_engine;
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());
  bool changed = false;
  for (Node* node : TopoSort(func)) {
    if (node->Is<ArrayIndex>()) {
      ArrayIndex* array_index = node->As<ArrayIndex>();
      Type* subtype = array_index->array()->GetType();
      for (int64_t i = 0; i < array_index->indices().size(); ++i) {
        Node* index = array_index->indices()[i];
        ArrayType* array_type = subtype->AsArrayOrDie();
        if (IndexIsDefinitelyOutOfBounds(index, array_type, query_engine)) {
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

// Try to simplify the given array index operation. Returns true if the node was
// changed.
absl::StatusOr<bool> SimplifyArrayIndex(ArrayIndex* array_index,
                                        const QueryEngine& query_engine) {
  // An array index with a nil index (no index operands) can be replaced by the
  // array operand:
  //
  //   array_index(A, {}) => A
  //
  if (array_index->indices().empty()) {
    XLS_RETURN_IF_ERROR(array_index->ReplaceUsesWith(array_index->array()));
    return true;
  }

  // An array index which indexes into a kArray operation and whose first
  // index element is a literal can be simplified by bypassing the kArray
  // operation:
  //
  //   array_index(array(a, b, c, d), {2, i, j, k, ...}
  //     => array_index(c, {i, j, k, ...})
  //
  if (array_index->array()->Is<Array>() && !array_index->indices().empty() &&
      array_index->indices().front()->Is<Literal>()) {
    Array* array = array_index->array()->As<Array>();
    Node* first_index = array_index->indices().front();
    if (IndexIsDefinitelyInBounds(first_index, array->GetType()->AsArrayOrDie(),
                                  query_engine)) {
      // Indices are always interpreted as unsigned numbers.
      XLS_ASSIGN_OR_RETURN(
          uint64_t operand_no,
          first_index->As<Literal>()->value().bits().ToUint64());
      XLS_RETURN_IF_ERROR(
          array_index
              ->ReplaceUsesWithNew<ArrayIndex>(
                  array->operand(operand_no), array_index->indices().subspan(1))
              .status());
      return true;
    }
  }

  // An array index which indexes into a kArrayConcat operation and whose first
  // index element is a literal can be simplified by bypassing the kArrayConcat
  // operation:
  //
  //   array_index(array_concat(A, B, C), {20, i, j, k, ...}
  //     => array_index(B, {10, i, j, k, ...})
  //
  // This assumes array A has a size 10, and B has a size of greater than 10.
  if (array_index->array()->Is<ArrayConcat>() &&
      !array_index->indices().empty() &&
      array_index->indices().front()->Is<Literal>()) {
    Node* first_index = array_index->indices().front();
    if (IndexIsDefinitelyInBounds(
            first_index, array_index->array()->GetType()->AsArrayOrDie(),
            query_engine)) {
      const Value& orig_first_index_value = first_index->As<Literal>()->value();
      XLS_ASSIGN_OR_RETURN(int64_t index,
                           orig_first_index_value.bits().ToUint64());
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
      std::vector<Node*> new_indices(array_index->indices().begin(),
                                     array_index->indices().end());
      XLS_ASSIGN_OR_RETURN(
          new_indices[0],
          array_index->function_base()->MakeNode<Literal>(
              array_index->loc(),
              Value(UBits(index, orig_first_index_value.bits().bit_count()))));

      XLS_RETURN_IF_ERROR(
          array_index
              ->ReplaceUsesWithNew<ArrayIndex>(indexed_operand, new_indices)
              .status());
      return true;
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
    XLS_RETURN_IF_ERROR(
        array_index
            ->ReplaceUsesWithNew<ArrayIndex>(operand->array(), combined_indices)
            .status());
    return true;
  }

  // Convert an array_index of a select into a select of array_indexes:
  //
  //   array_index(select(p, cases=[A0, A1]), {idx})
  //     => select(p, array_index(A0, {idx}), array_index(A1, {idx}))
  //
  // This reduces the width of the resulting mux.
  // TODO(meheff): generalize to arbitrary selects.
  if (IsBinarySelect(array_index->array())) {
    Select* select = array_index->array()->As<Select>();
    XLS_ASSIGN_OR_RETURN(
        ArrayIndex * on_false_index,
        array_index->function_base()->MakeNode<ArrayIndex>(
            select->loc(), select->get_case(0), array_index->indices()));
    XLS_ASSIGN_OR_RETURN(
        ArrayIndex * on_true_index,
        array_index->function_base()->MakeNode<ArrayIndex>(
            select->loc(), select->get_case(1), array_index->indices()));
    XLS_RETURN_IF_ERROR(
        array_index
            ->ReplaceUsesWithNew<Select>(
                select->selector(),
                /*cases=*/std::vector<Node*>({on_false_index, on_true_index}),
                /*default=*/absl::nullopt)
            .status());
    return true;
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
    return false;
  }
  auto* array_update = array_index->array()->As<ArrayUpdate>();

  // The transformation cannot be done if the indices might be out of bounds (or
  // at least it is more complicated to do correctly).
  if (!IndicesAreDefinitelyInBounds(array_index->indices(),
                                    array_index->array()->GetType(),
                                    query_engine) ||
      !IndicesAreDefinitelyInBounds(array_update->indices(),
                                    array_update->array_to_update()->GetType(),
                                    query_engine)) {
    return false;
  }

  // Consider case (1) above, where the array_update indices are a prefix of the
  // array_index indices.
  if (IndicesAreDefinitelyPrefixOf(array_update->indices(),
                                   array_index->indices(), query_engine)) {
    // Index directly in the update value. Remove the matching prefix from the
    // front of the array-index indices because the new array-index indexes into
    // the lower dimensional update value. so
    XLS_RETURN_IF_ERROR(
        array_index
            ->ReplaceUsesWithNew<ArrayIndex>(
                array_update->update_value(),
                array_index->indices().subspan(array_update->indices().size()))
            .status());
    return true;
  }

  if (IndicesDefinitelyNotEqual(array_update->indices(), array_index->indices(),
                                query_engine)) {
    XLS_RETURN_IF_ERROR(
        array_index->ReplaceOperandNumber(0, array_update->array_to_update()));
    return true;
  }

  return false;
}

// Try to simplify the given array update operation. Returns true if successful.
absl::StatusOr<bool> SimplifyArrayUpdate(ArrayUpdate* array_update,
                                         const QueryEngine& query_engine) {
  FunctionBase* func = array_update->function_base();

  // An array update with a nil index (no index operands) can be replaced by the
  // the update value (the "array" operand is unused).
  //
  //   array_update(A, v, {}) => v
  //
  if (array_update->indices().empty()) {
    XLS_RETURN_IF_ERROR(
        array_update->ReplaceUsesWith(array_update->update_value()));
    return true;
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
  // Assuming the index 'idx' a literal corresponding to element 'c'. The
  // advantage is the array update is operating on a smaller array, and if the
  // index is empty after this operation the array update can be removed.
  //
  // The single user requirement on Array is because 'Value' or '{idx, ...}'
  // might be derived from 'Array' otherwise and the transformation would
  // introduce a loop in the graph.
  if (array_update->array_to_update()->Is<Array>() &&
      array_update->array_to_update()->users().size() == 1 &&
      !array_update->indices().empty() &&
      array_update->indices().front()->Is<Literal>()) {
    Node* idx = array_update->indices().front();
    if (IndexIsDefinitelyInBounds(idx, array_update->GetType()->AsArrayOrDie(),
                                  query_engine)) {
      // Indices are always interpreted as unsigned numbers.
      XLS_ASSIGN_OR_RETURN(uint64_t operand_no,
                           idx->As<Literal>()->value().bits().ToUint64());
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
      return true;
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
        XLS_RETURN_IF_ERROR(array_update
                                ->ReplaceUsesWithNew<ArrayUpdate>(
                                    array_update->array_to_update(),
                                    subupdate->update_value(),
                                    new_update_indices)
                                .status());
        return true;
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
        XLS_VLOG(3) << absl::StreamFormat(
            "In chain of updates starting at %s, an index is written more than "
            "once, skipping the first write of this index (%s)",
            array_update->GetName(), prev->GetName());
        XLS_RETURN_IF_ERROR(
            current->ReplaceOperandNumber(0, prev->array_to_update()));
        return true;
      }
      current = prev;
    }
  }

  // If the array to update is a literal and the first index is a literal, the
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
  if (array_update->array_to_update()->Is<Literal>() &&
      !array_update->indices().empty() &&
      array_update->indices().front()->Is<Literal>()) {
    Node* idx = array_update->indices().front();
    if (IndexIsDefinitelyInBounds(idx, array_update->GetType()->AsArrayOrDie(),
                                  query_engine)) {
      // Indices are always interpreted as unsigned numbers.
      XLS_ASSIGN_OR_RETURN(uint64_t operand_no,
                           idx->As<Literal>()->value().bits().ToUint64());
      const Value& array_literal =
          array_update->array_to_update()->As<Literal>()->value();
      XLS_RET_CHECK_LT(operand_no, array_literal.size());

      std::vector<Node*> array_operands;
      for (int64_t i = 0; i < array_literal.size(); ++i) {
        XLS_ASSIGN_OR_RETURN(Literal * array_element,
                             func->MakeNode<Literal>(array_update->loc(),
                                                     array_literal.element(i)));
        if (i == operand_no) {
          XLS_ASSIGN_OR_RETURN(
              ArrayUpdate * new_array_update,
              func->MakeNode<ArrayUpdate>(array_update->loc(), array_element,
                                          array_update->update_value(),
                                          array_update->indices().subspan(1)));
          array_operands.push_back(new_array_update);
        } else {
          array_operands.push_back(array_element);
        }
      }
      XLS_RETURN_IF_ERROR(
          array_update
              ->ReplaceUsesWithNew<Array>(
                  array_operands,
                  /*element_type=*/array_operands.front()->GetType())
              .status());
      return true;
    }
  }
  return false;
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
    return absl::nullopt;
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
    if (!current->indices().back()->Is<Literal>()) {
      break;
    }

    const Bits& index_bits =
        current->indices().back()->As<Literal>()->value().bits();
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
    return absl::nullopt;
  }
  XLS_RET_CHECK(source_array != nullptr);
  XLS_RET_CHECK(!update_chain.empty());

  // TODO(meheff): If at least half (>=) of the values are set then replace.
  if (index_to_element.size() < (subarray_size + 1) / 2) {
    return absl::nullopt;
  }

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
absl::StatusOr<bool> FlattenSequentialUpdates(FunctionBase* func) {
  TernaryQueryEngine query_engine;
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());
  absl::flat_hash_set<ArrayUpdate*> flattened_updates;
  bool changed = false;
  // Perform this optimization in reverse topo sort order because we are looking
  // for a seqence of array update operations and the search progress upwards
  // (toward parameters).
  for (Node* node : ReverseTopoSort(func)) {
    if (!node->Is<ArrayUpdate>()) {
      continue;
    }
    ArrayUpdate* array_update = node->As<ArrayUpdate>();
    if (flattened_updates.contains(array_update)) {
      continue;
    }
    XLS_ASSIGN_OR_RETURN(
        std::optional<std::vector<ArrayUpdate*>> flattened_vec,
        FlattenArrayUpdateChain(array_update, query_engine));
    if (flattened_vec.has_value()) {
      changed = true;
      flattened_updates.insert(flattened_vec->begin(), flattened_vec->end());
    }
  }
  return changed;
}

// Try to simplify the given array operation. Returns true if successful.
absl::StatusOr<bool> SimplifyArray(Array* array,
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
      return false;
    }
    ArrayIndex* array_index = array->operand(i)->As<ArrayIndex>();
    if (array_index->indices().empty()) {
      return false;
    }

    // Extract the last element of the index as a uint64_t.
    Node* last_index_node = array_index->indices().back();
    if (!last_index_node->Is<Literal>()) {
      return false;
    }
    const Bits& last_index_bits =
        last_index_node->As<Literal>()->value().bits();
    if (!last_index_bits.FitsInUint64()) {
      return false;
    }
    uint64_t last_index = last_index_bits.ToUint64().value();

    // The final index element (0 .. N in the example above) must be sequential.
    if (last_index != i) {
      return false;
    }

    // The prefix of the index ({i, j} in the example above must match the
    // prefixes of the other array index operations.
    absl::Span<Node* const> prefix_span =
        array_index->indices().subspan(0, array_index->indices().size() - 1);
    if (common_index_prefix.has_value()) {
      if (!IndicesAreDefinitelyEqual(common_index_prefix.value(), prefix_span,
                                     query_engine)) {
        return false;
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
      return false;
    }
  }

  XLS_RET_CHECK(common_index_prefix.has_value());
  XLS_ASSIGN_OR_RETURN(Type * origin_array_subtype,
                       GetIndexedElementType(origin_array->GetType(),
                                             common_index_prefix->size()));
  if (array->GetType() == origin_array_subtype) {
    if (common_index_prefix->empty()) {
      XLS_RETURN_IF_ERROR(array->ReplaceUsesWith(origin_array));
      return true;
    }
    XLS_RETURN_IF_ERROR(array
                            ->ReplaceUsesWithNew<ArrayIndex>(
                                origin_array, common_index_prefix.value())
                            .status());
    return true;
  }
  return false;
}

// Simplify the conditional updating of an array element of the following form:
//
//   Select(p, cases=[A, array_update(A, v {idx})])
//
// This pattern is replaced with:
//
//   array_update(A, select(p, cases=[array_index(A, {idx}), v]), {idx})
//
// The advantage is the select (mux) is only selecting an array element rather
// than the entire array.
absl::StatusOr<bool> SimplifyConditionalAssign(Select* select) {
  XLS_RET_CHECK(IsBinarySelect(select));
  bool update_on_true;
  ArrayUpdate* array_update;
  if (select->get_case(0)->Is<ArrayUpdate>() &&
      select->get_case(0)->As<ArrayUpdate>()->array_to_update() ==
          select->get_case(1)) {
    array_update = select->get_case(0)->As<ArrayUpdate>();
    update_on_true = false;
  } else if (select->get_case(1)->Is<ArrayUpdate>() &&
             select->get_case(1)->As<ArrayUpdate>()->array_to_update() ==
                 select->get_case(0)) {
    array_update = select->get_case(1)->As<ArrayUpdate>();
    update_on_true = true;
  } else {
    return false;
  }
  if (array_update->users().size() != 1) {
    return false;
  }

  XLS_ASSIGN_OR_RETURN(ArrayIndex * original_value,
                       select->function_base()->MakeNode<ArrayIndex>(
                           array_update->loc(), array_update->array_to_update(),
                           array_update->indices()));
  XLS_ASSIGN_OR_RETURN(
      Select * selected_value,
      select->function_base()->MakeNode<Select>(
          select->loc(), select->selector(),
          /*cases=*/
          std::vector<Node*>(
              {update_on_true ? original_value : array_update->update_value(),
               update_on_true ? array_update->update_value() : original_value}),
          /*default_value=*/absl::nullopt));

  XLS_RETURN_IF_ERROR(array_update->ReplaceOperandNumber(1, selected_value));
  XLS_RETURN_IF_ERROR(select->ReplaceUsesWith(array_update));
  return true;
}

// Simplify a select of arrays to an array of select.
//
//   Sel(p, cases=[Array(), Array()]) => Array(Sel(p, ...), Sel(p, ...))
//
// The advantage is that hoisting the selects may be open opportunities for
// further optimization.
absl::StatusOr<bool> SimplifySelectOfArrays(Select* select) {
  for (Node* sel_case : select->cases()) {
    if (!sel_case->Is<Array>()) {
      return false;
    }
  }
  // TODO(meheff): Handle default.
  if (select->default_value().has_value()) {
    return false;
  }

  ArrayType* array_type = select->GetType()->AsArrayOrDie();
  std::vector<Node*> selected_elements;
  for (int64_t i = 0; i < array_type->size(); ++i) {
    std::vector<Node*> elements;
    for (Node* sel_case : select->cases()) {
      elements.push_back(sel_case->operand(i));
    }
    XLS_ASSIGN_OR_RETURN(Node * selected_element,
                         select->function_base()->MakeNode<Select>(
                             select->loc(), select->selector(),
                             /*cases=*/elements, /*default=*/absl::nullopt));
    selected_elements.push_back(selected_element);
  }
  XLS_RETURN_IF_ERROR(select
                          ->ReplaceUsesWithNew<Array>(
                              selected_elements, array_type->element_type())
                          .status());
  return true;
}

// Simplify various forms of a binary select of array-typed values.
absl::StatusOr<bool> SimplifyBinarySelect(Select* select,
                                          const QueryEngine& query_engine) {
  XLS_RET_CHECK(IsBinarySelect(select));
  bool changed = false;
  XLS_ASSIGN_OR_RETURN(changed, SimplifyConditionalAssign(select));
  if (changed) {
    return true;
  }

  XLS_ASSIGN_OR_RETURN(changed, SimplifySelectOfArrays(select));
  if (changed) {
    return true;
  }

  // Simplify a select between two array update operations which update the same
  // index of the same array:
  //
  //   Sel(p, {ArrayUpdate(A, v0, {idx}), ArrayUpdate(A, v1, {idx})})
  //
  //     => ArrayUpdate(A, Select(p, {v0, v1}), {idx})
  if (!select->get_case(0)->Is<ArrayUpdate>() ||
      !select->get_case(1)->Is<ArrayUpdate>()) {
    return false;
  }
  ArrayUpdate* false_update = select->get_case(0)->As<ArrayUpdate>();
  ArrayUpdate* true_update = select->get_case(1)->As<ArrayUpdate>();
  if (false_update->array_to_update() != true_update->array_to_update() ||
      !IndicesAreDefinitelyEqual(false_update->indices(),
                                 true_update->indices(), query_engine)) {
    return false;
  }

  XLS_ASSIGN_OR_RETURN(Select * selected_value,
                       select->function_base()->MakeNode<Select>(
                           select->loc(), select->selector(),
                           /*cases=*/
                           std::vector<Node*>({false_update->update_value(),
                                               true_update->update_value()}),
                           /*default_value=*/absl::nullopt));

  XLS_RETURN_IF_ERROR(select
                          ->ReplaceUsesWithNew<ArrayUpdate>(
                              false_update->array_to_update(), selected_value,
                              false_update->indices())
                          .status());
  return true;
}

}  // namespace

absl::StatusOr<bool> ArraySimplificationPass::RunOnFunctionBaseInternal(
    FunctionBase* func, const PassOptions& options,
    PassResults* results) const {
  bool changed = false;

  XLS_ASSIGN_OR_RETURN(bool clamp_changed, ClampArrayIndexIndices(func));
  changed |= clamp_changed;

  TernaryQueryEngine query_engine;
  XLS_RETURN_IF_ERROR(query_engine.Populate(func).status());

  for (Node* node : TopoSort(func)) {
    if (node->Is<ArrayIndex>()) {
      ArrayIndex* array_index = node->As<ArrayIndex>();
      XLS_ASSIGN_OR_RETURN(bool node_changed,
                           SimplifyArrayIndex(array_index, query_engine));
      changed = changed | node_changed;
    } else if (node->Is<ArrayUpdate>()) {
      XLS_ASSIGN_OR_RETURN(
          bool node_changed,
          SimplifyArrayUpdate(node->As<ArrayUpdate>(), query_engine));
      changed = changed | node_changed;
    } else if (node->Is<Array>()) {
      XLS_ASSIGN_OR_RETURN(bool node_changed,
                           SimplifyArray(node->As<Array>(), query_engine));
      changed = changed | node_changed;
    } else if (IsBinarySelect(node)) {
      XLS_ASSIGN_OR_RETURN(
          bool node_changed,
          SimplifyBinarySelect(node->As<Select>(), query_engine));
      changed = changed | node_changed;
    }
  }

  XLS_ASSIGN_OR_RETURN(bool flatten_changed, FlattenSequentialUpdates(func));
  changed = changed | flatten_changed;
  return changed;
}

}  // namespace xls
