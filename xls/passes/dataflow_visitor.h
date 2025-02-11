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

#ifndef XLS_PASSES_DATAFLOW_VISITOR_H_
#define XLS_PASSES_DATAFLOW_VISITOR_H_

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/data_structures/leaf_type_tree.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/dfs_visitor.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/type.h"
#include "xls/passes/stateless_query_engine.h"

namespace xls {

// Abstract base class which performs dataflow analysis of a function base.
// The analysis flows a lattice through the graph with user defined join
// operations. The join operations are used, for example, to join possible
// selected cases in select operations.  The analysis can be used to track
// value elements (e.g., tuple elements) through tuple, tuple-index, array,
// and other operations. The data structure stores a LeafTypeTree<T> for each
// node. For example, given:
//
//   x = tuple(a, b)
//   y = tuple_index(x, 0)
//   z = tuple_index(x, 1)
//   x_z = tuple(x, z)
//
// Where the associated value for `x` (as, say, a LeafTypeTree<int64_t> in
// DataflowVisitor) is `(42, 123)`, then DataflowVisitor::GetValue will return
// the following for the other nodes:
//
//   y : 42
//   z : 123
//   x_z : ((42, 123), 123)
//
// Users must define the following methods:
//
//   * DefaultHandler: the base class includes handlers for tuple, array,
//     select, and identity operations. The default handler is used for all
//     other operations.
//
//   * JoinElements: an operation for joining data sources together for an
//      operation. This operation is used, for example, to produce the value
//      for a select operation by joining the possible selected cases.
//
// Other handlers can be optionally defined or overridden.
//
// Users may also override the following methods:
//
//   * IndexMightBeEquivalent: used to determine if a given index might be
//     equivalent to a given concrete index. This is used to determine which
//     elements of an array might be relevant to a given array operation.
//
//   * IndexIsEquivalent: used to determine if a given index is definitely
//     equivalent to a given concrete set of indices. This is used to determine
//     which elements of an array might be relevant to a given array operation.
template <typename LeafT>
class DataflowVisitor : public DfsVisitorWithDefault {
 public:
  absl::Status HandleArray(Array* array) override {
    std::vector<LeafTypeTreeView<LeafT>> elements;
    for (Node* operand : array->operands()) {
      elements.push_back(GetValue(operand));
    }
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafT> result,
                         leaf_type_tree::CreateArray<LeafT>(
                             array->GetType()->AsArrayOrDie(), elements));
    return SetValue(array, std::move(result));
  }

  absl::Status HandleArrayConcat(ArrayConcat* array_concat) override {
    // All leaf values of an array-concat operation are statically determined,
    // no need to join values from operands.
    typename LeafTypeTree<LeafT>::DataContainerT leaves;
    for (Node* operand : array_concat->operands()) {
      LeafTypeTreeView<LeafT> operand_value = GetValue(operand);
      leaves.insert(leaves.end(), operand_value.elements().begin(),
                    operand_value.elements().end());
    }
    return SetValueFromLeafElements(array_concat, std::move(leaves));
  }

  absl::Status HandleArrayIndex(ArrayIndex* array_index) override {
    // The value for an array-index operation is the join of the possibly
    // indexed values in the input array. no need to join values from
    // operands.
    LeafTypeTreeView<LeafT> array_value = GetValue(array_index->array());
    std::vector<int64_t> bounds =
        GetArrayBounds(array_index->array()->GetType());
    std::vector<LeafTypeTreeView<LeafT>> data_sources;
    XLS_RETURN_IF_ERROR(leaf_type_tree::ForEachSubArray<LeafT>(
        array_value.AsView(), array_index->indices().size(),
        [&](LeafTypeTreeView<LeafT> element_view,
            absl::Span<const int64_t> index) {
          if (IndicesMightBeEquivalent(array_index->indices(), index, bounds,
                                       /*indices_clamped=*/true)) {
            data_sources.push_back(element_view);
          }
          return absl::OkStatus();
        }));
    std::vector<LeafTypeTreeView<LeafT>> control_sources;
    for (Node* index : array_index->indices()) {
      XLS_RET_CHECK(IsLeafType(index->GetType()));
      control_sources.push_back(GetValue(index));
    }
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafT> result,
                         Join(data_sources, control_sources, array_index));
    return SetValue(array_index, std::move(result));
  }

  absl::Status HandleArraySlice(ArraySlice* array_slice) override {
    // The value for each element in an array-slice operation is the join of the
    // possibly sliced values in the input array.
    std::vector<SharedLeafTypeTree<LeafT>> elements;
    elements.reserve(array_slice->width());
    LeafTypeTreeView<LeafT> array_value = GetValue(array_slice->array());
    int64_t array_size =
        array_slice->array()->GetType()->AsArrayOrDie()->size();
    if (std::optional<Bits> start_bits =
            query_engine_.KnownValueAsBits(array_slice->start());
        start_bits.has_value()) {
      int64_t start = std::numeric_limits<int64_t>::max();
      if (start_bits->FitsInNBitsUnsigned(63)) {
        start = static_cast<int64_t>(*start_bits->ToUint64());
      }
      XLS_ASSIGN_OR_RETURN(
          LeafTypeTree<LeafT> slice_value,
          leaf_type_tree::SliceArray(array_slice->GetType()->AsArrayOrDie(),
                                     array_value, start));
      return SetValue(array_slice, std::move(slice_value));
    }

    std::vector<LeafTypeTree<LeafT>> possible_slices;
    possible_slices.reserve(array_size);
    for (int64_t start = 0; start < array_size; ++start) {
      if (IndexMightBeEquivalent(array_slice->start(), start, array_size,
                                 /*indices_clamped=*/true)) {
        XLS_ASSIGN_OR_RETURN(
            LeafTypeTree<LeafT> possible_slice,
            leaf_type_tree::SliceArray(array_slice->GetType()->AsArrayOrDie(),
                                       array_value, start));
        possible_slices.push_back(std::move(possible_slice));
      }
    }

    std::vector<LeafTypeTreeView<LeafT>> data_sources;
    data_sources.reserve(possible_slices.size());
    for (const LeafTypeTree<LeafT>& possible_slice : possible_slices) {
      data_sources.push_back(possible_slice.AsView());
    }
    std::vector<LeafTypeTreeView<LeafT>> control_sources = {
        GetValue(array_slice->start())};
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafT> slice_value,
                         Join(data_sources, control_sources, array_slice));
    return SetValue(array_slice, std::move(slice_value));
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* array_update) override {
    // The value for an array-update operation is the value of the input
    // array. Each element which could be updated is joined with the update
    // value.  indexed values in the input array. no need to join values from
    // operands.
    LeafTypeTree<LeafT> result =
        leaf_type_tree::Clone(GetValue(array_update->array_to_update()));
    MutableLeafTypeTreeView<LeafT> result_view = result.AsMutableView();

    std::vector<int64_t> bounds =
        GetArrayBounds(array_update->array_to_update()->GetType());
    LeafTypeTreeView<LeafT> update_value =
        GetValue(array_update->update_value());
    std::vector<LeafTypeTreeView<LeafT>> control_sources;
    for (Node* index : array_update->indices()) {
      XLS_RET_CHECK(IsLeafType(index->GetType()));
      control_sources.push_back(GetValue(index));
    }
    XLS_RETURN_IF_ERROR(leaf_type_tree::ForEachSubArray<LeafT>(
        result_view, array_update->indices().size(),
        [&](MutableLeafTypeTreeView<LeafT> element_view,
            absl::Span<const int64_t> index) -> absl::Status {
          if (IndicesAreEquivalent(array_update->indices(), index, bounds,
                                   /*indices_clamped=*/false)) {
            // This is confirmed to be the updated position; just replace it.
            return leaf_type_tree::ReplaceElements(element_view, update_value);
          }
          if (IndicesMightBeEquivalent(array_update->indices(), index, bounds,
                                       /*indices_clamped=*/false)) {
            return leaf_type_tree::UpdateFrom<LeafT, LeafT>(
                element_view, update_value,
                [&](Type* leaf_type, LeafT& element, const LeafT& other_element,
                    absl::Span<const int64_t> index) -> absl::Status {
                  XLS_ASSIGN_OR_RETURN(
                      element,
                      JoinElements(leaf_type, {&element, &other_element},
                                   control_sources, array_update, index));
                  return absl::OkStatus();
                },
                /*index_prefix=*/index);
          }
          return absl::OkStatus();
        }));
    return SetValue(array_update, std::move(result));
  }

  absl::Status HandleIdentity(UnOp* identity) override {
    return SetValue(identity, GetValue(identity->operand(0)));
  }

  absl::Status HandleOneHotSel(OneHotSelect* sel) override {
    // If the selector is not one-hot then the cases may be or-ed together or
    // the result may be zero which does not fit in the lattice domain so
    // bail.
    if (!query_engine_.ExactlyOneBitTrue(sel->selector())) {
      return DefaultHandler(sel);
    }

    std::vector<LeafTypeTreeView<LeafT>> cases;
    for (Node* c : sel->cases()) {
      cases.push_back(GetValue(c));
    }
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafT> result,
                         Join(cases, {GetValue(sel->selector())}, sel));
    return SetValue(sel, std::move(result));
  }

  absl::Status HandlePrioritySel(PrioritySelect* sel) override {
    std::vector<LeafTypeTreeView<LeafT>> cases;
    for (Node* c : sel->cases()) {
      cases.push_back(GetValue(c));
    }
    cases.push_back(GetValue(sel->default_value()));
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafT> result,
                         Join(cases, {GetValue(sel->selector())}, sel));
    return SetValue(sel, std::move(result));
  }

  absl::Status HandleSel(Select* sel) override {
    std::vector<LeafTypeTreeView<LeafT>> cases;
    for (Node* c : sel->cases()) {
      cases.push_back(GetValue(c));
    }
    if (sel->default_value().has_value()) {
      cases.push_back(GetValue(sel->default_value().value()));
    }
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafT> result,
                         Join(cases, {GetValue(sel->selector())}, sel));
    return SetValue(sel, std::move(result));
  }

  absl::Status HandleNext(Next* next) override {
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafT> result,
                         leaf_type_tree::CreateTuple<LeafT>(
                             next->GetType()->AsTupleOrDie(), {}));
    return SetValue(next, std::move(result));
  }

  absl::Status HandleTuple(Tuple* tuple) override {
    std::vector<LeafTypeTreeView<LeafT>> elements;
    for (Node* operand : tuple->operands()) {
      elements.push_back(GetValue(operand));
    }
    XLS_ASSIGN_OR_RETURN(LeafTypeTree<LeafT> result,
                         leaf_type_tree::CreateTuple<LeafT>(
                             tuple->GetType()->AsTupleOrDie(), elements));
    return SetValue(tuple, std::move(result));
  }

  absl::Status HandleTupleIndex(TupleIndex* tuple_index) override {
    return SetValue(
        tuple_index,
        map_.at(tuple_index->operand(0))->AsView({tuple_index->index()}));
  }

  // Returns the leaf type tree value associated with `node`.
  LeafTypeTreeView<LeafT> GetValue(Node* node) const {
    return map_.at(node)->AsView();
  }

  absl::flat_hash_map<Node*, std::unique_ptr<SharedLeafTypeTree<LeafT>>>
  ToStoredValues() && {
    return std::move(map_);
  }

 protected:
  // Joins the elements of `data_sources` together and returns the result.
  // This operation is used, for example, to join the leaf values of possible
  // selected cases of a select operation to produce the value of the
  // select. `node` is the node being analyzed, `index` is the type index of
  // these elements. `control_sources` is the set of control inputs which
  // affect which data element is actually selected. These include the
  // index(es) in an array index operation or the selector of a select
  // operation.
  virtual absl::StatusOr<LeafT> JoinElements(
      Type* element_type, absl::Span<const LeafT* const> data_sources,
      absl::Span<const LeafTypeTreeView<LeafT>> control_sources, Node* node,
      absl::Span<const int64_t> index) = 0;

  // Inplace join of `other` into `element`.
  absl::Status JoinSubtreeInPlace(
      MutableLeafTypeTreeView<LeafT> tree, LeafTypeTreeView<LeafT> other,
      absl::Span<const LeafTypeTreeView<LeafT>> control_sources, Node* node,
      absl::Span<const int64_t> index_of_root) {
    XLS_RETURN_IF_ERROR(leaf_type_tree::ForEach(
        tree,
        [&](Type* element_type, LeafT& element,
            absl::Span<const int64_t> full_index) {
          absl::Span<const int64_t> subindex =
              full_index.subspan(index_of_root.size());
          JoinElementInPlace(element, other.Get(subindex), control_sources,
                             node, full_index);
          return absl::OkStatus();
        },
        index_of_root));
    return absl::OkStatus();
  }

  // Joins the given data sources with the given control sources and returns
  // the joined value. Elements at the same type index in each data source are
  // joined.
  absl::StatusOr<LeafTypeTree<LeafT>> Join(
      absl::Span<const LeafTypeTreeView<LeafT>> data_sources,
      absl::Span<const LeafTypeTreeView<LeafT>> control_sources, Node* node) {
    return leaf_type_tree::ZipIndex<LeafT, LeafT>(
        data_sources,
        [&](Type* leaf_type, absl::Span<const LeafT* const> elements,
            absl::Span<const int64_t> index) -> absl::StatusOr<LeafT> {
          return JoinElements(leaf_type, elements, control_sources, node,
                              index);
        });
  }

  // Sets the leaf type tree value associated with `node`.
  absl::Status SetValue(Node* node, LeafTypeTreeView<LeafT> value) {
    XLS_RET_CHECK_EQ(node->GetType(), value.type());
    map_.insert_or_assign(
        node, std::make_unique<SharedLeafTypeTree<LeafT>>(value.AsShared()));
    return absl::OkStatus();
  }
  absl::Status SetValue(Node* node, LeafTypeTree<LeafT>&& value) {
    XLS_RET_CHECK_EQ(node->GetType(), value.type());
    map_.insert_or_assign(node, std::make_unique<SharedLeafTypeTree<LeafT>>(
                                    std::move(value).AsShared()));
    return absl::OkStatus();
  }
  absl::Status SetValue(Node* node, SharedLeafTypeTree<LeafT>&& value) {
    XLS_RET_CHECK_EQ(node->GetType(), value.type());
    map_.insert_or_assign(
        node, std::make_unique<SharedLeafTypeTree<LeafT>>(std::move(value)));
    return absl::OkStatus();
  }

  // Sets the leaf type tree value associated with `node`.
  absl::Status SetValueFromLeafElements(
      Node* node, LeafTypeTree<LeafT>::DataContainerT&& elements) {
    return SetValue(node, LeafTypeTree<LeafT>::CreateFromVector(
                              node->GetType(), std::move(elements)));
  }

  // Returns true if `index` is definitely equivalent to `concrete_index`. If
  // `index_clamped` is true then the value of `index` is clamped when it
  // equals or exceeds `bound`.
  bool IndexIsEquivalent(const Bits& bits_index, int64_t concrete_index,
                         int64_t bound, bool index_clamped) const {
    CHECK_LT(concrete_index, bound);
    if (index_clamped && concrete_index == (bound - 1)) {
      return bits_ops::UGreaterThanOrEqual(bits_index, concrete_index);
    }
    return bits_ops::UEqual(bits_index, concrete_index);
  }
  virtual bool IndexIsEquivalent(const LeafT& index, int64_t concrete_index,
                                 int64_t bound, bool index_clamped) const {
    return false;
  }
  bool IndexIsEquivalent(Node* index, int64_t concrete_index, int64_t bound,
                         bool index_clamped) const {
    CHECK(IsLeafType(index->GetType()));
    if (std::optional<Bits> known_value = query_engine_.KnownValueAsBits(index);
        known_value.has_value()) {
      return IndexIsEquivalent(*known_value, concrete_index, bound,
                               index_clamped);
    }
    return IndexIsEquivalent(GetValue(index).Get({}), concrete_index, bound,
                             index_clamped);
  }

  // Returns true if `index` could be equivalent to `concrete_index`. If
  // `index_clamped` is true then the value of `index` is clamped when it
  // equals or exceeds `bound`.
  bool IndexMightBeEquivalent(const Bits& bits_index, int64_t concrete_index,
                              int64_t bound, bool index_clamped) const {
    return IndexIsEquivalent(bits_index, concrete_index, bound, index_clamped);
  }
  virtual bool IndexMightBeEquivalent(const LeafT& index,
                                      int64_t concrete_index, int64_t bound,
                                      bool index_clamped) const {
    return true;
  }
  bool IndexMightBeEquivalent(Node* index, int64_t concrete_index,
                              int64_t bound, bool index_clamped) const {
    CHECK(IsLeafType(index->GetType()));
    if (std::optional<Bits> known_value = query_engine_.KnownValueAsBits(index);
        known_value.has_value()) {
      return IndexMightBeEquivalent(*known_value, concrete_index, bound,
                                    index_clamped);
    }
    return IndexMightBeEquivalent(GetValue(index).Get({}), concrete_index,
                                  bound, index_clamped);
  }

  // Returns true if all known `indices` could be equivalent to the given
  // `concrete_indices`, ignoring all unknown `indices`. If `index_clamped` is
  // true then the `indices` are each clamped when they equal or exceed their
  // corresponding `bounds`.
  bool IndicesMightBeEquivalent(absl::Span<Node* const> indices,
                                absl::Span<const int64_t> concrete_indices,
                                absl::Span<const int64_t> bounds,
                                bool indices_clamped) const {
    CHECK_EQ(indices.size(), concrete_indices.size());
    for (int64_t i = 0; i < indices.size(); ++i) {
      if (!IndexMightBeEquivalent(indices[i], concrete_indices[i], bounds[i],
                                  indices_clamped)) {
        return false;
      }
    }
    return true;
  }

  // Returns true if all known `indices` are known equivalent to the given
  // `concrete_indices`. If `index_clamped` is true then the `indices` are each
  // clamped when they equal or exceed their corresponding `bounds`.
  bool IndicesAreEquivalent(absl::Span<Node* const> indices,
                            absl::Span<const int64_t> concrete_indices,
                            absl::Span<const int64_t> bounds,
                            bool indices_clamped) const {
    CHECK_EQ(indices.size(), concrete_indices.size());
    for (int64_t i = 0; i < indices.size(); ++i) {
      if (!IndexIsEquivalent(indices[i], concrete_indices[i], bounds[i],
                             indices_clamped)) {
        return false;
      }
    }
    return true;
  }

  // Returns the multi-dimensional array bounds of `type`.
  std::vector<int64_t> GetArrayBounds(Type* type) {
    Type* subtype = type;
    std::vector<int64_t> bounds;
    while (subtype->IsArray()) {
      ArrayType* array_type = subtype->AsArrayOrDie();
      bounds.push_back(array_type->size());
      subtype = array_type->element_type();
    }
    return bounds;
  }

  StatelessQueryEngine query_engine_;

  // Storage for the leaf type tree values associated with each node; must be
  // pointer-stable so that values can be shared (by populating some values as
  // Views of others).
  absl::flat_hash_map<Node*, std::unique_ptr<SharedLeafTypeTree<LeafT>>> map_;
};

}  // namespace xls

#endif  // XLS_PASSES_DATAFLOW_VISITOR_H_
