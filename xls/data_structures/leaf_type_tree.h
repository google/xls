// Copyright 2020 Google LLC
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

#ifndef XLS_DATA_STRUCTURES_LEAF_TYPE_TREE_H_
#define XLS_DATA_STRUCTURES_LEAF_TYPE_TREE_H_

#include <vector>

#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/ir/type.h"

namespace xls {

// A container which stores values of an arbitrary type T, one value for each
// leaf element (Bits value) of a potentially-recursive XLS type. Values are
// stored in a flat vector which provides fast iteration, but indexing through
// tuple types is O(#elements in tuple).
//
// Example usage where T is an int64:
//
//   Type* t = ...; /* (bits[42], bits[55], (bits[123], bits[64])) */
//   LeafTypeTree<int64> tree(t);
//
//   // Store a value at the element associated with the bits[64] type.
//   tree.Set({2, 1}, 333);
//
//   // Store a value at the element associated with the bits[55] type.
//   tree.Set({1}, 444);
//
//   // Get the int64 values as a flat vector. All values are
//   // value-initialized (set to zero in the case of in64).
//   ... = tree.elements(); /* { 0, 444, 0, 333 } */
//
//   // Get the value associated with the bits[44] type.
//   ... = tree.Get({1});
//
//   // INVALID: try to access a value at an index which does not refer to a
//   // leaf in the XLS type. In this example, index {2} refers to the tuple
//   // type (bits[123], bits[64]).
//   ... = tree.Get{2}; /* WRONG: will CHECK fail. */
template <typename T>
class LeafTypeTree {
 public:
  LeafTypeTree() : type_(nullptr) {}
  LeafTypeTree(const LeafTypeTree<T>& other) = default;
  LeafTypeTree& operator=(const LeafTypeTree<T>& other) = default;

  explicit LeafTypeTree(Type* type)
      : type_(type), elements_(type->leaf_count()) {
    MakeLeafTypes(type);
  }
  LeafTypeTree(Type* type, const T& init_value)
      : type_(type), elements_(type->leaf_count(), init_value) {
    MakeLeafTypes(type);
  }

  // Constructor for tuples/arrays where members are provided as a span..
  LeafTypeTree(Type* type, absl::Span<LeafTypeTree<T> const> init_values)
      : type_(type) {
    // Sanity check types of given values.
    if (type->IsArray()) {
      XLS_CHECK_EQ(type->AsArrayOrDie()->size(), init_values.size());
      for (auto& init_value : init_values) {
        XLS_CHECK_EQ(type->AsArrayOrDie()->element_type(), init_value.type());
      }
    } else if (type->IsTuple()) {
      XLS_CHECK_EQ(type->AsTupleOrDie()->size(), init_values.size());
      for (int64 i = 0; i < init_values.size(); ++i) {
        XLS_CHECK_EQ(type->AsTupleOrDie()->element_type(i),
                     init_values[i].type());
      }
    } else {
      XLS_LOG(FATAL) << "Invalid constructor for bits types";
    }

    MakeLeafTypes(type);
    for (int64 i = 0; i < init_values.size(); ++i) {
      for (int64 j = 0; j < init_values[i].size(); ++j) {
        const T& leaf = init_values[i].elements()[j];
        elements_.push_back(leaf);
      }
    }
  }

  LeafTypeTree(Type* type, absl::Span<const T> elements)
      : type_(type), elements_(elements.begin(), elements.end()) {
    MakeLeafTypes(type_);
  }

  Type* type() const { return type_; }

  // Returns the number of values in the container (equivalently number of
  // leaves of the type).
  int64 size() const { return elements_.size(); }

  // Returns the element at the given Type index.  The Type index defines a
  // recursive traversal through the object's XLS type. The Type index must
  // correspond to a leaf Bits-type element in the object's XLS type.
  T& Get(absl::Span<int64 const> index) {
    std::pair<Type*, int64> type_offset =
        GetSubtypeAndOffset(type_, index, /*offset=*/0);
    // The index must refer to a leaf node (bits type).
    XLS_CHECK(type_offset.first->IsBits());
    return elements_[type_offset.second];
  }
  const T& Get(absl::Span<int64 const> index) const {
    return const_cast<LeafTypeTree*>(this)->Get(index);
  }

  // Sets the element at the given Type index to the given value.
  void Set(absl::Span<int64 const> index, const T& value) {
    std::pair<Type*, int64> type_offset =
        GetSubtypeAndOffset(type_, index, /*offset=*/0);
    // The index must refer to a leaf node (bits type).
    XLS_CHECK(type_offset.first->IsBits());
    elements_[type_offset.second] = value;
  }

  // Returns the values stored in this container.
  absl::Span<T> elements() { return absl::Span<T>(elements_); }
  absl::Span<T const> elements() const {
    return absl::Span<T const>(elements_);
  }

  // Returns the types of each leaf in the XLS type of this object. The order of
  // these types corresponds to the order of elements().
  absl::Span<BitsType* const> leaf_types() const { return leaf_types_; }

  // Copies and returns the subtree rooted at the given type index as a
  // LeafTypeTree.
  LeafTypeTree<T> CopySubtree(absl::Span<int64 const> const index) const {
    std::pair<Type*, int64> type_offset =
        GetSubtypeAndOffset(type_, index, /*offset=*/0);
    Type* subtype = type_offset.first;
    LeafTypeTree subtree(subtype);
    for (int64 i = 0; i < subtype->leaf_count(); ++i) {
      subtree.elements_[i] = elements_[type_offset.second + i];
    }
    return subtree;
  }

 private:
  // Creates the vector of leaf types.
  void MakeLeafTypes(Type* t) {
    if (t->IsBits()) {
      leaf_types_.push_back(t->AsBitsOrDie());
      return;
    }
    if (t->IsArray()) {
      for (int64 i = 0; i < t->AsArrayOrDie()->size(); ++i) {
        MakeLeafTypes(t->AsArrayOrDie()->element_type());
      }
      return;
    }
    XLS_CHECK(t->IsTuple());
    for (int64 i = 0; i < t->AsTupleOrDie()->size(); ++i) {
      MakeLeafTypes(t->AsTupleOrDie()->element_type(i));
    }
  }

  // Returns a pair containing the Type and element offset for the given type
  // index.
  std::pair<Type*, int64> GetSubtypeAndOffset(Type* t,
                                              absl::Span<int64 const> index,
                                              int64 offset) const {
    if (index.empty()) {
      return {t, offset};
    }
    if (t->IsArray()) {
      XLS_CHECK(!index.empty());
      XLS_CHECK_LT(index[0], t->AsArrayOrDie()->size());
      Type* element_type = t->AsArrayOrDie()->element_type();
      return GetSubtypeAndOffset(
          element_type, index.subspan(1),
          offset + index[0] * element_type->leaf_count());
    }
    XLS_CHECK(t->IsTuple());
    TupleType* tuple_type = t->AsTupleOrDie();
    XLS_CHECK_LT(index[0], tuple_type->size());
    int64 element_offset = 0;
    for (int64 i = 0; i < index[0]; ++i) {
      element_offset += tuple_type->element_type(i)->leaf_count();
    }
    return GetSubtypeAndOffset(tuple_type->element_type(index[0]),
                               index.subspan(1), offset + element_offset);
  }

  Type* type_;
  std::vector<T> elements_;
  std::vector<BitsType*> leaf_types_;
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_LEAF_TYPE_TREE_H_
