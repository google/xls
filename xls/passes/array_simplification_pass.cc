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
  for (int64 i = 0; i < a.size(); ++i) {
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
  for (int64 i = 0; i < std::min(a.size(), b.size()); ++i) {
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

// Clamp any known OOB indices in MultiArrayIndex ops. In this case the index is
// replaced with a literal value equal to the maximum in-bounds index value
// (size of array minus one). Only known-OOB are clamped. Maybe OOB indices
// cannot be replaced because the index might be a different in-bounds value.
absl::StatusOr<bool> ClampMultiArrayIndexIndices(FunctionBase* func) {
  // This transformation may add nodes to the graph which invalidates the query
  // engine for later use, so create a private engine for exclusive use of this
  // transformation.
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<TernaryQueryEngine> query_engine,
                       TernaryQueryEngine::Run(func));
  bool changed = false;
  for (Node* node : TopoSort(func)) {
    if (node->Is<MultiArrayIndex>()) {
      MultiArrayIndex* array_index = node->As<MultiArrayIndex>();
      Type* subtype = array_index->array()->GetType();
      for (int64 i = 0; i < array_index->indices().size(); ++i) {
        Node* index = array_index->indices()[i];
        ArrayType* array_type = subtype->AsArrayOrDie();
        if (IndexIsDefinitelyOutOfBounds(index, array_type, *query_engine)) {
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
absl::StatusOr<bool> SimplifyMultiArrayIndex(MultiArrayIndex* array_index,
                                             const QueryEngine& query_engine) {
  // An array index with a nil index (no index operands) can be replaced by the
  // array operand:
  //
  //   array_index(A, {}) => A
  //
  if (array_index->indices().empty()) {
    return array_index->ReplaceUsesWith(array_index->array());
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
          uint64 operand_no,
          first_index->As<Literal>()->value().bits().ToUint64());
      XLS_RETURN_IF_ERROR(
          array_index
              ->ReplaceUsesWithNew<MultiArrayIndex>(
                  array->operand(operand_no), array_index->indices().subspan(1))
              .status());
      return true;
    }
  }

  // Consecutive multiarray index operations can be combined. For example:
  //
  //   array_index(array_index(A, {a, b}), {c, d})
  //     => array_index(A, {a, b, c, d})
  //
  if (array_index->array()->Is<MultiArrayIndex>()) {
    MultiArrayIndex* operand = array_index->array()->As<MultiArrayIndex>();
    std::vector<Node*> combined_indices(operand->indices().begin(),
                                        operand->indices().end());
    combined_indices.insert(combined_indices.end(),
                            array_index->indices().begin(),
                            array_index->indices().end());
    XLS_RETURN_IF_ERROR(array_index
                            ->ReplaceUsesWithNew<MultiArrayIndex>(
                                operand->array(), combined_indices)
                            .status());
    return true;
  }

  // Convert a select of arrays to an array of selects:
  //
  //   array_index(select(p, cases=[A0, A1]), {idx})
  //     => select(p, array_index(A0, {idx}), array_index(A1, {idx}))
  //
  // This reduces the width of the resulting mux.
  // TODO(meheff): generalize to arbitrary selects.
  if (IsBinarySelect(array_index->array())) {
    Select* select = array_index->array()->As<Select>();
    XLS_ASSIGN_OR_RETURN(
        MultiArrayIndex * on_false_index,
        array_index->function_base()->MakeNode<MultiArrayIndex>(
            select->loc(), select->get_case(0), array_index->indices()));
    XLS_ASSIGN_OR_RETURN(
        MultiArrayIndex * on_true_index,
        array_index->function_base()->MakeNode<MultiArrayIndex>(
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
  //  index {i_0, ... i_m} -> multiarray_update
  //                               |
  //  index {j_0, ... j_n} -> multiarray_index
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
  //        index {j_m+1, ... j_n} -> multiarray_index
  //
  //
  //  (2) {i_0, ..., i_m} is definitely *not* a prefix of {j_0, ..., j_n}. The
  //      array index does *not* index into the updated part of the array. The
  //      original array can be indexed directly:
  //
  //                                       A
  //                                       |
  //        index {j_0, ... j_n} -> multiarray_index
  //
  if (!array_index->array()->Is<MultiArrayUpdate>()) {
    return false;
  }
  auto* array_update = array_index->array()->As<MultiArrayUpdate>();

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
            ->ReplaceUsesWithNew<MultiArrayIndex>(
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
  absl::optional<std::vector<Node*>> common_index_prefix;
  for (int64 i = 0; i < array->operand_count(); ++i) {
    if (!array->operand(i)->Is<MultiArrayIndex>()) {
      return false;
    }
    MultiArrayIndex* array_index = array->operand(i)->As<MultiArrayIndex>();
    if (array_index->indices().empty()) {
      return false;
    }

    // Extract the last element of the index as a uint64.
    Node* last_index_node = array_index->indices().back();
    if (!last_index_node->Is<Literal>()) {
      return false;
    }
    const Bits& last_index_bits =
        last_index_node->As<Literal>()->value().bits();
    if (!last_index_bits.FitsInUint64()) {
      return false;
    }
    uint64 last_index = last_index_bits.ToUint64().value();

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
      return array->ReplaceUsesWith(origin_array);
    }
    XLS_RETURN_IF_ERROR(array
                            ->ReplaceUsesWithNew<MultiArrayIndex>(
                                origin_array, common_index_prefix.value())
                            .status());
    return true;
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

  bool changed = false;

  // Convert array index operations into multiarray index operations.
  for (Node* node : TopoSort(func)) {
    if (node->Is<ArrayIndex>()) {
      ArrayIndex* array_index = node->As<ArrayIndex>();
      XLS_RETURN_IF_ERROR(node->ReplaceUsesWithNew<MultiArrayIndex>(
                                  array_index->array(),
                                  std::vector<Node*>({array_index->index()}))
                              .status());
      changed = true;
    }
  }

  XLS_ASSIGN_OR_RETURN(bool clamp_changed, ClampMultiArrayIndexIndices(func));
  changed |= clamp_changed;

  XLS_ASSIGN_OR_RETURN(std::unique_ptr<TernaryQueryEngine> query_engine,
                       TernaryQueryEngine::Run(func));

  for (Node* node : TopoSort(func)) {
    if (node->Is<MultiArrayIndex>()) {
      MultiArrayIndex* array_index = node->As<MultiArrayIndex>();
      XLS_ASSIGN_OR_RETURN(bool node_changed,
                           SimplifyMultiArrayIndex(array_index, *query_engine));
      changed = changed | node_changed;
    } else if (node->Is<Array>()) {
      XLS_ASSIGN_OR_RETURN(bool node_changed,
                           SimplifyArray(node->As<Array>(), *query_engine));
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
