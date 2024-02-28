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

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
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

namespace xls {

// Abstract base class which performs dataflow analysis of a function base. The
// analysis flows a lattice through the graph with user defined join
// operations. The join operations are used, for example, to join possible
// selected cases in select operations.  The analysis can be used to track value
// elements (e.g., tuple elements) through tuple, tuple-index, array, and other
// operations. The data structure stores a LeafTypeTree<T> for each node. For
// example, given:
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
//   * DefaultHandler: the base class includes handlers for tuple, array, select
//     and identity operations. The default handler is used for all other
//     operations.
//
//   * AccumulateDataElement: an operation for joining multiple potential
//   selected
//     data sources for an operation. This operation is used, for example, to
//     produce the value for a select operation by joining the possible selected
//     cases.
//
//   * AccumulateControlElement: an operation for joining a control operand
//   value
//     (selector of a select operation, index of an array index operation, etc.)
//     with the value produced by the operation (select, array_index, etc).
//
// Other handlers can optionally defined or overridden.
template <typename LeafT>
class DataflowVisitor : public DfsVisitorWithDefault {
 public:
  absl::Status HandleArray(Array* array) override {
    // All leaf values of an array operation are statically determined, no need
    // to join values from operands.
    absl::InlinedVector<LeafT, 1> leaves;
    for (Node* operand : array->operands()) {
      const LeafTypeTree<LeafT>& operand_value = GetValue(operand);
      leaves.insert(leaves.end(), operand_value.elements().begin(),
                    operand_value.elements().end());
    }
    return SetValueFromLeafElements(array, absl::MakeSpan(leaves));
  }

  absl::Status HandleArrayConcat(ArrayConcat* array_concat) override {
    // All leaf values of an array-concat operation are statically determined,
    // no need to join values from operands.
    absl::InlinedVector<LeafT, 1> leaves;
    for (Node* operand : array_concat->operands()) {
      const LeafTypeTree<LeafT>& operand_value = GetValue(operand);
      leaves.insert(leaves.end(), operand_value.elements().begin(),
                    operand_value.elements().end());
    }
    return SetValueFromLeafElements(array_concat, absl::MakeSpan(leaves));
  }

  absl::Status HandleArrayIndex(ArrayIndex* array_index) override {
    // The value for an array-index operation is the join of the possibly
    // indexed values in the input array. no need to join values from operands.
    const LeafTypeTree<LeafT>& array_value = GetValue(array_index->array());
    std::vector<int64_t> bounds =
        GetArrayBounds(array_index->array()->GetType());
    std::optional<LeafTypeTree<LeafT>> result;
    XLS_RETURN_IF_ERROR(array_value.ForEachSubArray(
        array_index->indices().size(),
        [&](Type* subtype, absl::Span<const LeafT> elements,
            absl::Span<const int64_t> index) {
          if (IndicesMightBeEqual(array_index->indices(), index, bounds,
                                  /*indices_clamped=*/true)) {
            if (result.has_value()) {
              JoinLeafDataElementsWithValue(elements, array_index, *result);
            } else {
              result = LeafTypeTree<LeafT>(array_index->GetType(), elements);
            }
          }
          return absl::OkStatus();
        }));
    // Join the index operands as control values.
    for (Node* index : array_index->indices()) {
      XLS_RETURN_IF_ERROR(
          JoinControlValue(GetValue(index), array_index, *result));
    }
    return SetValue(array_index, *result);
  }

  absl::Status HandleArrayUpdate(ArrayUpdate* array_update) override {
    // The value for an array-update operation is the value of the input
    // array. Each element which could be updated is joined with the update
    // value.  indexed values in the input array. no need to join values from
    // operands.
    LeafTypeTree<LeafT> result = GetValue(array_update->array_to_update());
    std::vector<int64_t> bounds =
        GetArrayBounds(array_update->array_to_update()->GetType());
    const LeafTypeTree<LeafT>& update_value =
        GetValue(array_update->update_value());
    XLS_RETURN_IF_ERROR(result.ForEachSubArray(
        array_update->indices().size(),
        [&](Type* subtype, absl::Span<LeafT> elements,
            absl::Span<const int64_t> index) {
          if (IndicesAreEqual(array_update->indices(), index, bounds,
                              /*indices_clamped=*/false)) {
            for (int64_t i = 0; i < elements.size(); ++i) {
              elements[i] = update_value.elements()[i];
            }
          } else if (IndicesMightBeEqual(array_update->indices(), index, bounds,
                                         /*indices_clamped=*/false)) {
            JoinLeafDataElementsWithValue(update_value.elements(), array_update,
                                          result, index);
          }
          return absl::OkStatus();
        }));
    // Join the index operands as control values.
    for (Node* index : array_update->indices()) {
      XLS_RETURN_IF_ERROR(
          JoinControlValue(GetValue(index), array_update, result));
    }
    return SetValue(array_update, result);
  }

  absl::Status HandleIdentity(UnOp* identity) override {
    return SetValue(identity, GetValue(identity->operand(0)));
  }

  absl::Status HandleOneHotSel(OneHotSelect* sel) override {
    // If the selector is not one-hot then the cases may be or-ed together or
    // the result may be zero which does not fit in the lattice domain so bail.
    if (!sel->selector()->Is<OneHot>()) {
      return DefaultHandler(sel);
    }

    LeafTypeTree<LeafT> result = GetValue(sel->cases().front());
    for (Node* c : sel->cases().subspan(1)) {
      XLS_RETURN_IF_ERROR(JoinDataValue(GetValue(c), sel, result));
    }
    XLS_RETURN_IF_ERROR(
        JoinControlValue(GetValue(sel->selector()), sel, result));
    return SetValue(sel, result);
  }

  absl::Status HandleSel(Select* sel) override {
    std::optional<LeafTypeTree<LeafT>> result;
    auto set_or_merge = [&](const LeafTypeTree<LeafT>& other) {
      if (result.has_value()) {
        return JoinDataValue(other, sel, result.value());
      }
      result = other;
      return absl::OkStatus();
    };
    for (Node* c : sel->cases()) {
      XLS_RETURN_IF_ERROR(set_or_merge(GetValue(c)));
    }
    if (sel->default_value().has_value()) {
      XLS_RETURN_IF_ERROR(set_or_merge(GetValue(sel->default_value().value())));
    }
    XLS_RETURN_IF_ERROR(
        JoinControlValue(GetValue(sel->selector()), sel, *result));
    return SetValue(sel, *result);
  }

  absl::Status HandleTuple(Tuple* tuple) override {
    // Use InlinedVector to avoid std::vector<bool> abomination.
    absl::InlinedVector<LeafT, 1> leaves;
    for (Node* operand : tuple->operands()) {
      const LeafTypeTree<LeafT>& operand_tree = map_.at(operand);
      leaves.insert(leaves.end(), operand_tree.elements().begin(),
                    operand_tree.elements().end());
    }
    return SetValueFromLeafElements(tuple, absl::MakeSpan(leaves));
  }

  absl::Status HandleTupleIndex(TupleIndex* tuple_index) override {
    return SetValue(tuple_index,
                    map_.at(tuple_index->operand(0))
                        .CopySubtree(/*index=*/{tuple_index->index()}));
  }

  // Returns the leaf type tree value associated with `node`.
  const LeafTypeTree<LeafT>& GetValue(Node* node) const {
    return map_.at(node);
  }

  // Returns the moved leaf type tree value associated with `node`.
  LeafTypeTree<LeafT> ConsumeValue(Node* node) {
    LeafTypeTree<LeafT> ltt = std::move(map_.at(node));
    // Erase the moved element from the map to avoid later access.
    map_.erase(node);
    return ltt;
  }

 protected:
  // Inplace join of `data_element` into `element`. Used to join potential data
  // sources for a leaf element. This operation is used, for example, to join
  // the possible selected cases of a select operation to produce the value of
  // the select. `node` is the node being analyzed, `index` is the tree index of
  // these elements.
  //
  // For example, for the operation `x = select(p, {a, b, c})` where x, a, b,
  // and c are u32[42] types, AccumulateDataElement will be called as follows
  // to compute the dataflow value for x:
  //
  // X: LeafTypeTree<LeafT>
  // for i in range(0, 41):
  //   X[i] = GetValue(a)[i]
  //   AccumulateDataElement(GetValue(b)[i], x, {i}, X[i])
  //   AccumulateDataElement(GetValue(c)[i], x, {i}, X[i])
  virtual absl::Status AccumulateDataElement(const LeafT& data_element,
                                             Node* node,
                                             absl::Span<const int64_t> index,
                                             LeafT& element) const = 0;

  // Inplace join of `control_element` into `element`. Used to join a control
  // source for a leaf element. This operation is used, for example, to join the
  // selector from a select operation to value of the select. `node` is the node
  // being analyzed, `index` is the tree index of `element` (not
  // `control_element).
  //
  // For example, consider the operation `x = array_index(a, {i, j})`. Let `X`
  // be the dataflow value computed for x using AccumulateDataElement.
  // AccumulateControlElement will be called to accumulate the dataflow values
  // for indices i and j into each element of X.
  //
  // TODO(meheff): Consider consolidating the join/accumulate methods into a
  // single method which looks like:
  //   absl::StatusOr<LeafTypeTree<LeafT>> Join(
  //      Node* node,
  //      absl::Span<const LeafTypeTree<LeafT>*> data_inputs,
  //      absl::Span<const LeafTypeTree<LeafT>*> control_inputs);
  // Currently this requires too much copying.
  virtual absl::Status AccumulateControlElement(const LeafT& control_element,
                                                Node* node,
                                                absl::Span<const int64_t> index,
                                                LeafT& element) const = 0;

  // Inplace data join of the LeafTypeTree value `data_value` with `value`.
  absl::Status JoinDataValue(const LeafTypeTree<LeafT>& data_value, Node* node,
                             LeafTypeTree<LeafT>& value) {
    return value.ForEach([&](Type* t, LeafT& element,
                             absl::Span<const int64_t> index) {
      return AccumulateDataElement(data_value.Get(index), node, index, element);
    });
  }

  // Inplace control join of the LeafTypeTree value `control_value` with
  // `value`. `control_value` must be bits-typed. The single leaf element in
  // `control_value` is joined with all leaf elements in `value`.
  absl::Status JoinControlValue(const LeafTypeTree<LeafT>& control_value,
                                Node* node, LeafTypeTree<LeafT>& value) {
    XLS_RET_CHECK(control_value.type()->IsBits());
    const LeafT control_element = control_value.elements().front();
    return value.ForEach([&](Type* t, LeafT& element,
                             absl::Span<const int64_t> index) {
      return AccumulateControlElement(control_element, node, index, element);
    });
  }

  // Inplace data join of the LeafTypeTree using a span of
  // elements. `index_prefix` can be used to limit the join to the subtree of
  // elements with that prefix.
  absl::Status JoinLeafDataElementsWithValue(
      absl::Span<const LeafT> elements, Node* node, LeafTypeTree<LeafT>& value,
      absl::Span<const int64_t> index_prefix = {}) {
    int64_t linear_index = 0;
    return value.ForEach(
        [&](Type* t, LeafT& element, absl::Span<const int64_t> index) {
          AccumulateDataElement(elements[linear_index], node, index, element);
          ++linear_index;
          return absl::OkStatus();
        },
        index_prefix);
  }

  // Sets the leaf type tree value associated with `node`.
  absl::Status SetValue(Node* node, LeafTypeTree<LeafT> value) {
    XLS_RET_CHECK_EQ(node->GetType(), value.type());
    map_[node] = std::move(value);
    return absl::OkStatus();
  }

  // Sets the leaf type tree value associated with `node`.
  absl::Status SetValueFromLeafElements(Node* node,
                                        absl::Span<LeafT> elements) {
    return SetValue(node, LeafTypeTree<LeafT>(node->GetType(), elements));
  }

  // Returns true if `index` is definitely equal to `concrete_index`. If
  // `index_clamped` is true then the value of `index` is clamped when it equals
  // or exceeds `bound`.
  bool IndexIsEqual(Node* index, int64_t concrete_index, int64_t bound,
                    bool index_clamped) const {
    CHECK_LT(concrete_index, bound);
    if (!index->Is<Literal>()) {
      return false;
    }
    const Bits& bits_index = index->As<Literal>()->value().bits();
    return bits_ops::UEqual(bits_index, concrete_index) ||
           (index_clamped && bits_ops::UGreaterThanOrEqual(bits_index, bound) &&
            (concrete_index == bound - 1));
  }

  // Returns true if the type tree index value `index` is might equal to
  // `concrete_index`. If `index_clamped` is true then the value of `index` is
  // clamped when it equals or exceeds `bound`.
  bool IndexMightBeEqual(Node* index, int64_t concrete_index, int64_t bound,
                         bool index_clamped) const {
    CHECK_LT(concrete_index, bound);
    if (!index->Is<Literal>()) {
      return true;
    }
    const Bits& bits_index = index->As<Literal>()->value().bits();
    return bits_ops::UEqual(bits_index, concrete_index) ||
           (index_clamped && bits_ops::UGreaterThanOrEqual(bits_index, bound) &&
            (concrete_index == bound - 1));
  }

  // Returns true if the type tree index value `index` is might equal to
  // `concrete_index`. If `index_clamped` is true then the value of `index` is
  // clamped when it equals or exceeds `bound`.
  bool IndicesMightBeEqual(absl::Span<Node* const> indices,
                           absl::Span<const int64_t> concrete_indices,
                           absl::Span<const int64_t> bounds,
                           bool indices_clamped) const {
    CHECK_EQ(indices.size(), concrete_indices.size());
    for (int64_t i = 0; i < indices.size(); ++i) {
      if (!IndexMightBeEqual(indices[i], concrete_indices[i], bounds[i],
                             indices_clamped)) {
        return false;
      }
    }
    return true;
  }

  // Returns true if the type tree index value `index` is definitely equal to
  // `concrete_index`. If `index_clamped` is true then the value of `index` is
  // clamped when it equals or exceeds `bound`.
  bool IndicesAreEqual(absl::Span<Node* const> indices,
                       absl::Span<const int64_t> concrete_indices,
                       absl::Span<const int64_t> bounds,
                       bool indices_clamped) const {
    CHECK_EQ(indices.size(), concrete_indices.size());
    for (int64_t i = 0; i < indices.size(); ++i) {
      if (!IndexIsEqual(indices[i], concrete_indices[i], bounds[i],
                        indices_clamped)) {
        return false;
      }
    }
    return true;
  }

  // Returns the multi-dimenstional array bounds of `type`.
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

  absl::flat_hash_map<Node*, LeafTypeTree<LeafT>> map_;
};

}  // namespace xls

#endif  // XLS_PASSES_DATAFLOW_VISITOR_H_
