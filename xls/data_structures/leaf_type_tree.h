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

#ifndef XLS_DATA_STRUCTURES_LEAF_TYPE_TREE_H_
#define XLS_DATA_STRUCTURES_LEAF_TYPE_TREE_H_

#include <cstdint>
#include <functional>
#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/attributes.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/visitor.h"
#include "xls/ir/type.h"

namespace xls {

// Returns true if `t` is a leaf type (i.e. not an aggregate type).
inline bool IsLeafType(Type* t) { return t->IsBits() || t->IsToken(); }

// forward decl
template <typename T>
class SharedLeafTypeTree;

namespace leaf_type_tree_internal {

// Returns a pair containing the type and linear element offset (within the
// flattened vector of leaf elements) for the given type index within
// `type`. The index can be the index of a leaf element or interior node in the
// tree (i.e., an aggregate type).
std::pair<Type*, int64_t> GetSubtypeAndOffset(Type* t,
                                              absl::Span<int64_t const> index);

// Returns the leaf types for the type `t`. Returned as an inlined vector to
// match internal storage of LeafTypeTree.
absl::InlinedVector<Type*, 1> GetLeafTypes(Type* t);

// Returns the leaf element linear offset in the flattened representation of
// type `t` for the given type-index `index`. CHECK fails if the index is not
// the index of a leaf element.
int64_t GetLeafTypeOffset(Type* t, absl::Span<const int64_t> index);

// Returns a string representation of a type tree of the given type with the
// given flattened representation of elements. See ToString method of
// LeafTypeTree for details about output.
std::string ToString(Type* t, absl::Span<const std::string> elements,
                     bool multiline);

// Maps the given function across `elements` and returns the resulting vector.
template <typename T>
std::vector<std::string> ElementsToStrings(
    absl::Span<const T> elements,
    const std::function<std::string(const T&)>& f) {
  std::vector<std::string> element_strs;
  element_strs.reserve(elements.size());
  for (const T& element : elements) {
    element_strs.push_back(f(element));
  }
  return element_strs;
}

}  // namespace leaf_type_tree_internal

// An immutable view of a LeafTypeTree. Internal data members are stored as
// spans referring to the vectors in the original LeafTypeTree. This type should
// be used on interfaces which handle LeafTypeTree. This enables flexible and
// efficient slicing and aggregating of LeafTypeTree-like values without
// copying.
template <typename T>
class LeafTypeTreeView {
 public:
  using DataT = const T;
  using DataContainerT = absl::Span<const T>;
  using TypeContainerT = absl::Span<Type* const>;

  LeafTypeTreeView(Type* type, absl::Span<const T> elements,
                   absl::Span<Type* const> leaf_types)
      : type_(type), elements_(elements), leaf_types_(leaf_types) {}
  LeafTypeTreeView(const LeafTypeTreeView<T>& other) = default;
  LeafTypeTreeView& operator=(const LeafTypeTreeView<T>& other) = default;
  LeafTypeTreeView(LeafTypeTreeView<T>&& other) = default;
  LeafTypeTreeView& operator=(LeafTypeTreeView<T>&& other) = default;
  friend bool operator==(const LeafTypeTreeView<T>& lhs,
                         const LeafTypeTreeView<T>& rhs) = default;

  SharedLeafTypeTree<T> AsShared() const;

  // Factory function for creating a view (or view of a subtree) of a
  // LeafTypeTree. This should only be used internally.
  //
  // Args:
  //    t: top-level type
  //    elements: flattened leaf data element for type `t`
  //    leaf_types: leaf_types of type `t`
  //    index: type index of the subtree for which the view is created. If
  //        empty, a view is created for the entire tree.
  static LeafTypeTreeView<T> CreateFromSpans(Type* t,
                                             absl::Span<const T> elements,
                                             absl::Span<Type* const> leaf_types,
                                             absl::Span<const int64_t> index) {
    CHECK_EQ(elements.size(), leaf_types.size());
    auto [subtype, linear_offset] =
        leaf_type_tree_internal::GetSubtypeAndOffset(t, index);
    return LeafTypeTreeView<T>(
        subtype, elements.subspan(linear_offset, subtype->leaf_count()),
        leaf_types.subspan(linear_offset, subtype->leaf_count()));
  }

  // These methods are mirrors of those on LeafTypeTree. See LeafTypeTree for
  // descriptions.
  Type* type() const { return type_; }
  int64_t size() const { return elements_.size(); }
  const T& Get(absl::Span<int64_t const> index) const {
    return elements_[leaf_type_tree_internal::GetLeafTypeOffset(type(), index)];
  }
  absl::Span<T const> elements() const { return elements_; }
  absl::Span<Type* const> leaf_types() const { return leaf_types_; }
  LeafTypeTreeView<T> AsView(absl::Span<const int64_t> index = {}) const {
    return CreateFromSpans(type(), elements(), leaf_types(), index);
  }
  std::string ToString(const std::function<std::string(const T&)>& f) const {
    return leaf_type_tree_internal::ToString(
        type(), leaf_type_tree_internal::ElementsToStrings<T>(elements(), f),
        /*multiline=*/false);
  }
  std::string ToString() const {
    return ToString([](const T& element) { return absl::StrCat(element); });
  }
  std::string ToMultilineString(
      const std::function<std::string(const T&)>& f) const {
    return leaf_type_tree_internal::ToString(
        type(), leaf_type_tree_internal::ElementsToStrings<T>(elements(), f),
        /*multiline=*/true);
  }
  std::string ToMultilineString() const {
    return ToMultilineString(
        [](const T& element) { return absl::StrCat(element); });
  }

  template <typename H>
  friend H AbslHashValue(H h, const LeafTypeTreeView<T>& ltt) {
    return H::combine(std::move(h), ltt.type_, ltt.elements());
  }

 private:
  Type* type_;
  absl::Span<const T> elements_;
  absl::Span<Type* const> leaf_types_;
};

// A mutable view of a LeafTypeTree. Leaf data members are mutable but the
// interior structure is not. Internal data members are stored as spans
// referring to the vectors in the original LeafTypeTree. This type should be on
// interfaces which handle LeafTypeTree. This enables flexible and efficient
// slicing and aggregating of LeafTypeTree-like values without copying.
template <typename T>
class MutableLeafTypeTreeView {
 public:
  using DataT = T;
  using DataContainerT = absl::Span<T>;
  using TypeContainerT = absl::Span<Type* const>;

  MutableLeafTypeTreeView(const MutableLeafTypeTreeView<T>& other) = default;
  MutableLeafTypeTreeView& operator=(const MutableLeafTypeTreeView<T>& other) =
      default;
  MutableLeafTypeTreeView(MutableLeafTypeTreeView<T>&& other) = default;
  MutableLeafTypeTreeView& operator=(MutableLeafTypeTreeView<T>&& other) =
      default;
  MutableLeafTypeTreeView(Type* type, absl::Span<T> elements,
                          absl::Span<Type* const> leaf_types)
      : type_(type), elements_(elements), leaf_types_(leaf_types) {}

  // Factory function for creating a view (or view of a subtree) of a
  // LeafTypeTree. This should only be used internally.
  //
  // Args:
  //    t: top-level type
  //    elements: flattened leaf data element for type `t`
  //    leaf_types: leaf_types of type `t`
  //    index: type index of the subtree for which the view is created. If
  //        empty, a view is created for the entire tree.
  static MutableLeafTypeTreeView<T> CreateFromSpans(
      Type* t, absl::Span<T> elements, absl::Span<Type* const> leaf_types,
      absl::Span<const int64_t> index) {
    auto [subtype, linear_offset] =
        leaf_type_tree_internal::GetSubtypeAndOffset(t, index);
    return MutableLeafTypeTreeView<T>(
        subtype,
        absl::MakeSpan(elements).subspan(linear_offset, subtype->leaf_count()),
        leaf_types.subspan(linear_offset, subtype->leaf_count()));
  }

  // These methods mirror methods on LeafTypeTree. See LeafTypeTree for
  // description of these methods.
  Type* type() const { return type_; }
  int64_t size() const { return elements_.size(); }
  T& Get(absl::Span<int64_t const> index) const {
    return elements_[leaf_type_tree_internal::GetLeafTypeOffset(type(), index)];
  }
  void Set(absl::Span<int64_t const> index, const T& value) const {
    elements_[leaf_type_tree_internal::GetLeafTypeOffset(type(), index)] =
        value;
  }
  absl::Span<T> elements() const { return elements_; }
  absl::Span<Type* const> leaf_types() const { return leaf_types_; }
  LeafTypeTreeView<T> AsView(absl::Span<const int64_t> index = {}) const {
    return LeafTypeTreeView<T>::CreateFromSpans(type(), elements(),
                                                leaf_types(), index);
  }
  MutableLeafTypeTreeView<T> AsMutableView(
      absl::Span<const int64_t> index = {}) const {
    return CreateFromSpans(type(), elements(), leaf_types(), index);
  }
  std::string ToString(const std::function<std::string(const T&)>& f) const {
    return leaf_type_tree_internal::ToString(
        type(), leaf_type_tree_internal::ElementsToStrings(elements(), f),
        /*multiline=*/false);
  }
  std::string ToString() const {
    return ToString([](const T& element) { return absl::StrCat(element); });
  }

  template <typename H>
  friend H AbslHashValue(H h, const MutableLeafTypeTreeView<T>& ltt) {
    return H::combine(std::move(h), ltt.type_, ltt.elements());
  }

 private:
  Type* type_;
  absl::Span<T> elements_;
  absl::Span<Type* const> leaf_types_;
};

namespace leaf_type_tree_internal {

// A data structure for iterating through the index space of a type as
// determined by any tuple or array components in the type.Example index spaces:
//
//   <type>          : <set of multidimensional type-indices in index space>
//   u32             : {}
//   u32[3]          : {0} {1} {2}
//   (u32, u16)      : {0} {1}
//   (u32, u16)[2]   : {0,0} {0,1}
//   (u32, (u16, u8) : {0} {0,0} {0,1}
//   ()              : <empty>
//
// The iterator starts at the first type-index and advances up to the end
// position.
class LeafTypeTreeIterator {
 public:
  // Create a type iterator for the given type. `index_prefix` is a sequence of
  // indices which prefixes the indices returned by `type_index`; it does not
  // affect the size of the iteration space.
  explicit LeafTypeTreeIterator(Type* type,
                                absl::Span<const int64_t> index_prefix = {});

  // Returns the type which defines the space this iterator iterates over.
  Type* root_type() const { return root_type_; }

  // Returns the type of the leaf element at the current point. Iterator
  // must not be at end of space.
  Type* leaf_type() const {
    CHECK(!AtEnd());
    return leaf_type_.value();
  }

  // Returns the linear index of the current point into a flattened
  // representation of the type. Iterator must not be at end of space.
  int64_t linear_index() const {
    CHECK(!AtEnd());
    return linear_index_;
  }

  // Returns the multi-dimensional type index of the current point. Iterator
  // must not be at end of space.
  absl::Span<const int64_t> type_index() const {
    CHECK(!AtEnd());
    return type_index_;
  }

  // Returns true if the iterator is at the end of the space.
  bool AtEnd() const { return !leaf_type_.has_value(); }

  // Advances the iterator to the next leaf index. Returns true if the iterator
  // reached the end of the type. Iterator must not be at end of space.
  bool Advance();

  std::string ToString() const;

 private:
  Type* root_type_;
  std::vector<int64_t> type_index_;
  // Number of elements in the index_prefix.
  int64_t prefix_size_;
  int64_t linear_index_;
  // Leaf type at the current position. If nullopt then the iterator is at the
  // end.
  std::optional<Type*> leaf_type_;
};

}  // namespace leaf_type_tree_internal

// A container which stores values of an arbitrary type T, one value for each
// leaf element (Bits value) of a potentially-recursive XLS type. Values are
// stored in a flat vector which provides fast iteration, but indexing through
// tuple types is O(#elements in tuple).
//
// Example usage where T is an int64_t:
//
//   Type* t = ...; /* (bits[42], bits[55], (bits[123], bits[64])) */
//   LeafTypeTree<int64_t> tree(t);
//
//   // Store a value at the element associated with the bits[64] type.
//   tree.Set({2, 1}, 333);
//
//   // Store a value at the element associated with the bits[55] type.
//   tree.Set({1}, 444);
//
//   // Get the int64_t values as a flat vector. All values are
//   // value-initialized (set to zero in the case of in64).
//   ... = tree.elements(); /* { 0, 444, 0, 333 } */
//
//   // Get the value associated with the bits[55] type.
//   ... = tree.Get({1});
//
//   // INVALID: try to access a value at an index which does not refer to a
//   // leaf in the XLS type. In this example, index {2} refers to the tuple
//   // type (bits[123], bits[64]).
//   ... = tree.Get{2}; /* WRONG: will CHECK fail. */
template <typename T>
class LeafTypeTree {
 public:
  using DataContainerT = absl::InlinedVector<T, 1>;
  using TypeContainerT = absl::InlinedVector<Type*, 1>;

  LeafTypeTree() : type_(nullptr) {}
  LeafTypeTree(const LeafTypeTree<T>& other) = default;
  LeafTypeTree& operator=(const LeafTypeTree<T>& other) = default;
  LeafTypeTree(LeafTypeTree<T>&& other) = default;
  LeafTypeTree& operator=(LeafTypeTree<T>&& other) = default;
  friend bool operator==(const LeafTypeTree<T>& lhs,
                         const LeafTypeTree<T>& rhs) = default;

  // Creates a leaf type tree in which each data member is default constructed.
  explicit LeafTypeTree(Type* type)
      : type_(type),
        elements_(type->leaf_count()),
        leaf_types_(leaf_type_tree_internal::GetLeafTypes(type)) {}

  // Creates a leaf type tree in which each data member set to `init_value`.
  LeafTypeTree(Type* type, const T& init_value)
      : type_(type),
        elements_(type->leaf_count(), init_value),
        leaf_types_(leaf_type_tree_internal::GetLeafTypes(type)) {}

  // Constructor which takes a flattened representation of the leaf elements.
  LeafTypeTree(Type* type, absl::Span<const T> elements)
      : type_(type),
        elements_(elements.begin(), elements.end()),
        leaf_types_(leaf_type_tree_internal::GetLeafTypes(type)) {
    CHECK_EQ(elements_.size(), leaf_types_.size());
  }

  SharedLeafTypeTree<T> AsShared() &&;

  // Factory for efficiently constructing a LeafTypeTree by moving in the vector
  // of data elements.
  static LeafTypeTree<T> CreateFromVector(Type* type,
                                          DataContainerT&& elements) {
    CHECK_EQ(elements.size(), type->leaf_count());
    LeafTypeTree<T> ltt;
    ltt.type_ = type;
    ltt.elements_ = std::move(elements);
    ltt.leaf_types_ = leaf_type_tree_internal::GetLeafTypes(type);
    return ltt;
  }

  // Factory for efficiently constructing a LeafTypeTree with a single element.
  static LeafTypeTree<T> CreateSingleElementTree(Type* type, T element) {
    CHECK_EQ(type->leaf_count(), 1);
    LeafTypeTree<T> ltt;
    ltt.type_ = type;
    ltt.elements_ = {element};
    ltt.leaf_types_ = leaf_type_tree_internal::GetLeafTypes(type);
    return ltt;
  }

  // Creates a leaf type tree in which each data member is initialized to the
  // value returned by the given function `f`. `f` takes the type of the leaf
  // element and (optionally) the index of the element as an argument.
  static absl::StatusOr<LeafTypeTree<T>> CreateFromFunction(
      Type* type, std::function<absl::StatusOr<T>(Type* leaf_type)> f) {
    LeafTypeTree<T> ltt;
    ltt.type_ = type;
    ltt.leaf_types_ = leaf_type_tree_internal::GetLeafTypes(type);
    ltt.elements_.reserve(ltt.leaf_types_.size());
    for (Type* leaf_type : ltt.leaf_types_) {
      XLS_ASSIGN_OR_RETURN(T value, f(leaf_type));
      ltt.elements_.push_back(std::move(value));
    }
    return ltt;
  }
  static absl::StatusOr<LeafTypeTree<T>> CreateFromFunction(
      Type* type,
      std::function<absl::StatusOr<T>(Type* leaf_type,
                                      absl::Span<const int64_t> index)>
          f) {
    leaf_type_tree_internal::LeafTypeTreeIterator it(type);
    DataContainerT elements;
    elements.reserve(type->leaf_count());
    while (!it.AtEnd()) {
      XLS_ASSIGN_OR_RETURN(T value, f(it.leaf_type(), it.type_index()));
      elements.push_back(std::move(value));
      it.Advance();
    }
    return CreateFromVector(type, std::move(elements));
  }

  Type* type() const { return type_; }

  // Returns the number of values in the container (equivalently number of
  // leaves of the type).
  int64_t size() const { return elements_.size(); }

  // Returns the element at the given Type index.  The Type index defines a
  // recursive traversal through the object's XLS type. The Type index must
  // correspond to a leaf Bits-type element in the object's XLS type.
  T& Get(absl::Span<int64_t const> index) {
    return elements_[leaf_type_tree_internal::GetLeafTypeOffset(type(), index)];
  }
  const T& Get(absl::Span<int64_t const> index) const {
    return elements_[leaf_type_tree_internal::GetLeafTypeOffset(type(), index)];
  }

  // Sets the element at the given Type index to the given value.
  void Set(absl::Span<int64_t const> index, const T& value) {
    elements_[leaf_type_tree_internal::GetLeafTypeOffset(type(), index)] =
        value;
  }

  // Returns the values stored in this container.
  absl::Span<T> elements() { return absl::Span<T>(elements_); }
  absl::Span<T const> elements() const {
    return absl::Span<T const>(elements_);
  }

  // Returns the types of each leaf in the XLS type of this object. The order
  // of these types corresponds to the order of elements().
  absl::Span<Type* const> leaf_types() const { return leaf_types_; }

  // Returns an immutable view of the LeafTypeTree.
  LeafTypeTreeView<T> AsView(absl::Span<const int64_t> index = {}) const
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return LeafTypeTreeView<T>::CreateFromSpans(type(), elements(),
                                                leaf_types(), index);
  }

  // Returns a mutable view of the LeafTypeTree.
  MutableLeafTypeTreeView<T> AsMutableView(absl::Span<const int64_t> index = {})
      ABSL_ATTRIBUTE_LIFETIME_BOUND {
    return MutableLeafTypeTreeView<T>::CreateFromSpans(type(), elements(),
                                                       leaf_types(), index);
  }

  // Returns the stringified elements of the LeafTypeTree in a structured
  // form. Examples for a LeaftypeTree of integers:
  //   bits/token type: 42
  //   tuple type:      (1, 2)
  //   array type:      [10, 20, 30]
  //   compound type:   (1, (), ([42], (10, 20)))
  std::string ToString(const std::function<std::string(const T&)>& f) const {
    return leaf_type_tree_internal::ToString(
        type(), leaf_type_tree_internal::ElementsToStrings<T>(elements(), f),
        /*multiline=*/false);
  }

  // Overload which uses StrCat to stringify the elements. This only works if
  // StrCat supports the type `T`.
  std::string ToString() const {
    return ToString([](const T& element) { return absl::StrCat(element); });
  }

  // ToString variants which emit a single element per line with
  // indentation. Example for a tuple-containing-an-array type:
  //
  //   (
  //     1,
  //     [
  //       2,
  //       3,
  //     ]
  //   )
  std::string ToMultilineString(
      const std::function<std::string(const T&)>& f) const {
    return leaf_type_tree_internal::ToString(
        type(), leaf_type_tree_internal::ElementsToStrings<T>(elements(), f),
        /*multiline=*/true);
  }
  std::string ToMultilineString() const {
    return ToMultilineString(
        [](const T& element) { return absl::StrCat(element); });
  }

  template <typename H>
  friend H AbslHashValue(H h, const LeafTypeTree<T>& ltt) {
    return H::combine(std::move(h), ltt.type_, ltt.elements());
  }

 protected:
  Type* type_;
  DataContainerT elements_;
  TypeContainerT leaf_types_;
};

namespace leaf_type_tree_internal {

// Increment a multi-dimensional array index assuming the given array bounds.
// The last element of `array_index` is incremented and if it equals the
// respective `bounds` element then it is set to zero and one is added to the
// next index element, etc. Returns true if the entire index overflowed.
bool IncrementArrayIndex(absl::Span<const int64_t> bounds,
                         std::vector<int64_t>* array_index);

// Returns information about the subarray at a particular index depth in
// the given type. The fields in the returned struct correspond to the type
// remaining after peeling `index_depth` outer dimensions from `type`.
struct SubArraySize {
  // The type of the subarray.
  Type* type;
  // The bounds of the subarray.
  std::vector<int64_t> bounds;
  // The number of leaf elements in the subarray.
  int64_t element_count;
};
absl::StatusOr<SubArraySize> GetSubArraySize(Type* type, int64_t index_depth);

template <typename T, typename ViewT>
absl::Status ForEachSubArrayHelper(
    Type* type, absl::Span<T> elements, absl::Span<Type* const> leaf_types,
    int64_t index_depth,
    const std::function<absl::Status(ViewT view,
                                     absl::Span<const int64_t> index)>& f) {
  XLS_ASSIGN_OR_RETURN(
      leaf_type_tree_internal::SubArraySize subarray_size,
      leaf_type_tree_internal::GetSubArraySize(type, index_depth));
  int64_t linear_index = 0;
  std::vector<int64_t> array_index(index_depth, 0);
  do {
    XLS_RETURN_IF_ERROR(
        f(ViewT(subarray_size.type,
                elements.subspan(linear_index, subarray_size.element_count),
                leaf_types.subspan(linear_index, subarray_size.element_count)),
          array_index));
    linear_index += subarray_size.element_count;
  } while (!leaf_type_tree_internal::IncrementArrayIndex(subarray_size.bounds,
                                                         &array_index));
  return absl::OkStatus();
}

}  // namespace leaf_type_tree_internal

// An immutable shared view of a LeafTypeTree. This might or might not own the
// underlying data. This can be used for (eg) the QueryEngine functions which
// might either return a pre-calculated LTT or generate one on the fly depending
// on what sort of query-engine/node is being used. Using this a copy of
// pre-existing ones is not needed and no significant work by the users is
// needed to check for this situation.
template <typename T>
class SharedLeafTypeTree {
 public:
  using DataT = const T;
  using DataContainerT = absl::Span<const T>;
  using TypeContainerT = absl::Span<Type* const>;

  // If you want to have an owned copy use ToOwned. To get an unowned view use
  // AsView. Doing an implicit copy is probably not what you ever want.
  SharedLeafTypeTree(const SharedLeafTypeTree<T>& other) = delete;
  SharedLeafTypeTree& operator=(const SharedLeafTypeTree<T>& other) = delete;

  SharedLeafTypeTree(SharedLeafTypeTree<T>&& other) = default;
  SharedLeafTypeTree& operator=(SharedLeafTypeTree<T>&& other) = default;

  friend bool operator==(const SharedLeafTypeTree<T>& lhs,
                         const SharedLeafTypeTree<T>& rhs) = default;

  bool IsOwned() const {
    return std::holds_alternative<LeafTypeTree<T>>(inner_);
  }

  // Make an owned LTT out of this shared tree.
  LeafTypeTree<T> ToOwned() && {
    if (std::holds_alternative<LeafTypeTree<T>>(inner_)) {
      return std::get<LeafTypeTree<T>>(std::move(inner_));
    }
    return LeafTypeTree<T>(type(), elements());
  }

  // Make an owned LTT out of this shared tree. This may cause a copy.
  LeafTypeTree<T> ToOwned() const& {
    return std::visit(
        Visitor{[](const LeafTypeTree<T>& t) -> LeafTypeTree<T> { return t; },
                [](const LeafTypeTreeView<T>& t) -> LeafTypeTree<T> {
                  return LeafTypeTree<T>(t.type(), t.elements());
                }},
        inner_);
  }

  // These methods are mirrors of those on LeafTypeTree. See LeafTypeTree for
  // descriptions.
  Type* type() const { return AsView().type(); }
  int64_t size() const { return AsView().size(); }
  const T& Get(absl::Span<int64_t const> index) const {
    return AsView().Get(index);
  }
  absl::Span<T const> elements() const { return AsView().elements(); }
  absl::Span<Type* const> leaf_types() const { return AsView().leaf_types(); }
  LeafTypeTreeView<T> AsView(absl::Span<const int64_t> index = {}) const {
    return std::visit(
               Visitor{[](LeafTypeTreeView<T> t) -> LeafTypeTreeView<T> {
                         return t;
                       },
                       [](const LeafTypeTree<T>& t) -> LeafTypeTreeView<T> {
                         return t.AsView();
                       }},
               inner_)
        .AsView(index);
  }
  std::string ToString(const std::function<std::string(const T&)>& f) const {
    return leaf_type_tree_internal::ToString(
        type(), leaf_type_tree_internal::ElementsToStrings<T>(elements(), f),
        /*multiline=*/false);
  }
  std::string ToString() const {
    return ToString([](const T& element) { return absl::StrCat(element); });
  }
  std::string ToMultilineString(
      const std::function<std::string(const T&)>& f) const {
    return leaf_type_tree_internal::ToString(
        type(), leaf_type_tree_internal::ElementsToStrings<T>(elements(), f),
        /*multiline=*/true);
  }
  std::string ToMultilineString() const {
    return ToMultilineString(
        [](const T& element) { return absl::StrCat(element); });
  }

  template <typename H>
  friend H AbslHashValue(H h, const SharedLeafTypeTree<T>& ltt) {
    return AbslHashValue(h, ltt.AsView());
  }

 private:
  std::variant<LeafTypeTreeView<T>, LeafTypeTree<T>> inner_;

  SharedLeafTypeTree(std::variant<LeafTypeTreeView<T>, LeafTypeTree<T>>&& inner)
      : inner_(std::move(inner)) {}

  friend class LeafTypeTree<T>;
  friend class LeafTypeTreeView<T>;
};

namespace leaf_type_tree {

// Create an array-typed LeafTypeTree where the data values of the array
// elements is given by `elements`.
template <typename T>
absl::StatusOr<LeafTypeTree<T>> CreateArray(
    ArrayType* array_type, absl::Span<const LeafTypeTreeView<T>> elements) {
  // Validate types of given values.
  XLS_RET_CHECK_EQ(array_type->size(), elements.size());
  for (auto& element : elements) {
    XLS_RET_CHECK_EQ(array_type->element_type(), element.type());
  }
  typename LeafTypeTree<T>::DataContainerT data_vector;
  data_vector.reserve(elements.size());
  for (int64_t i = 0; i < elements.size(); ++i) {
    for (int64_t j = 0; j < elements[i].size(); ++j) {
      const T& leaf = elements[i].elements()[j];
      data_vector.push_back(leaf);
    }
  }
  return LeafTypeTree<T>::CreateFromVector(array_type, std::move(data_vector));
}

// Create an array-typed LeafTypeTree where the data values of the array
// elements is the concatenation of the given array-typed LTT in `elements`.
template <typename T>
absl::StatusOr<LeafTypeTree<T>> ConcatArray(
    ArrayType* result_array_type,
    absl::Span<const LeafTypeTreeView<T>> elements) {
  auto check_all_elements = [&](auto f) {
    return absl::c_accumulate(
        elements, absl::OkStatus(),
        [&](absl::Status&& s, const auto& v) -> absl::Status {
          absl::Status res = std::move(s);
          res.Update(f(v));
          return res;
        });
  };
  XLS_RETURN_IF_ERROR(
      check_all_elements([&](const LeafTypeTreeView<T>& v) -> absl::Status {
        XLS_ASSIGN_OR_RETURN(ArrayType * vt, v.type()->AsArray());
        XLS_RET_CHECK(
            result_array_type->element_type()->IsEqualTo(vt->element_type()))
            << "Incompatible array element types: "
            << result_array_type->element_type() << " vs "
            << vt->element_type();
        return absl::OkStatus();
      }));
  // NB Since we've already verified that the element-type matches on all
  // entries if the leaf-count sum of the elements matches the leaf-count sum of
  // the result then the number of array elements must also match.
  XLS_RET_CHECK_EQ(
      result_array_type->leaf_count(),
      absl::c_accumulate(elements, int64_t{0},
                         [](int64_t acc, const LeafTypeTreeView<T>& ltt) {
                           return acc + ltt.type()->leaf_count();
                         }))
      << "Leaf count mismatch.";
  typename LeafTypeTree<T>::DataContainerT result;
  result.reserve(result_array_type->leaf_count());
  for (const LeafTypeTreeView<T>& ltt : elements) {
    absl::c_copy(ltt.elements(), std::back_inserter(result));
  }
  return LeafTypeTree<T>::CreateFromVector(result_array_type,
                                           std::move(result));
}

// Create an array-typed LeafTypeTree where the data values of the array
// elements is the slice of `source` starting at `start` and extending to the
// size of `result_array_type`. If the slice would run off the end of `source`
// it repeats the last element of `source`. This matches the behavior of
// array-index.
template <typename T>
absl::StatusOr<LeafTypeTree<T>> SliceArray(ArrayType* result_array_type,
                                           const LeafTypeTreeView<T> source,
                                           int64_t start) {
  XLS_RET_CHECK_EQ(result_array_type->element_type(),
                   source.type()->AsArrayOrDie()->element_type());
  XLS_RET_CHECK_GE(start, 0);
  XLS_RET_CHECK_GT(source.type()->AsArrayOrDie()->size(), 0);
  typename LeafTypeTree<T>::DataContainerT result;
  result.reserve(source.AsView({0}).size() * result_array_type->size());
  auto add_all = [&](LeafTypeTreeView<T> v) {
    absl::c_copy(v.elements(), std::back_inserter(result));
  };
  int64_t source_size = source.type()->AsArrayOrDie()->size();
  for (int64_t i = 0; i < result_array_type->size(); ++i) {
    int64_t source_off = SaturatingAdd<int64_t>(i, start).result;
    if (source_off < source_size) {
      add_all(source.AsView({source_off}));
    } else {
      // Repeat of last element
      add_all(source.AsView({source.type()->AsArrayOrDie()->size() - 1}));
    }
  }
  return LeafTypeTree<T>::CreateFromVector(result_array_type,
                                           std::move(result));
}

// Create a tuple-typed LeafTypeTree where the data values of the array
// elements is given by `elements`.
template <typename T>
absl::StatusOr<LeafTypeTree<T>> CreateTuple(
    TupleType* tuple_type, absl::Span<const LeafTypeTreeView<T>> elements) {
  // Validate types of given values.
  XLS_RET_CHECK_EQ(tuple_type->size(), elements.size());
  for (int64_t i = 0; i < elements.size(); ++i) {
    XLS_RET_CHECK_EQ(tuple_type->element_type(i), elements[i].type());
  }
  typename LeafTypeTree<T>::DataContainerT data_vector;
  data_vector.reserve(elements.size());
  for (int64_t i = 0; i < elements.size(); ++i) {
    for (int64_t j = 0; j < elements[i].size(); ++j) {
      const T& leaf = elements[i].elements()[j];
      data_vector.push_back(leaf);
    }
  }
  return LeafTypeTree<T>::CreateFromVector(tuple_type, std::move(data_vector));
}

// Copy the data elements of `source` to `dest`. Both LeafTypeTrees must be the
// same type.
template <typename T>
absl::Status ReplaceElements(MutableLeafTypeTreeView<T> dest,
                             LeafTypeTreeView<T> source) {
  XLS_RET_CHECK_EQ(dest.type(), source.type());
  for (int64_t i = 0; i < dest.size(); ++i) {
    dest.elements()[i] = source.elements()[i];
  }
  return absl::OkStatus();
}

// Use the given function to combine each corresponding leaf element in the
// given `LeafTypeTree` inputs. Returns an error if the given `LeafTypeTree`s
// are not generated from the same type.
template <typename T, typename A>
absl::StatusOr<LeafTypeTree<T>> ZipIndex(
    absl::Span<const LeafTypeTreeView<A>> inputs,
    std::function<absl::StatusOr<T>(Type* element_type,
                                    absl::Span<const A* const> elements,
                                    absl::Span<const int64_t> index)>
        f) {
  XLS_RET_CHECK(!inputs.empty());
  Type* type = inputs.front().type();
  int64_t size = inputs.front().size();
  for (const LeafTypeTreeView<A>& input : inputs.subspan(1)) {
    XLS_RET_CHECK_EQ(type, input.type());
  }

  typename LeafTypeTree<T>::DataContainerT new_elements;
  new_elements.reserve(size);
  leaf_type_tree_internal::LeafTypeTreeIterator it(type);
  std::vector<const A*> input_elements(inputs.size());
  while (!it.AtEnd()) {
    for (int64_t i = 0; i < inputs.size(); ++i) {
      input_elements[i] = &inputs[i].elements()[it.linear_index()];
    }
    XLS_ASSIGN_OR_RETURN(T value,
                         f(it.leaf_type(), input_elements, it.type_index()));
    new_elements.push_back(std::move(value));
    it.Advance();
  }
  return LeafTypeTree<T>::CreateFromVector(type, std::move(new_elements));
}

// Simple form of zip which accepts only two inputs and does not include type
// and index arguments nor status return value.
template <typename T, typename A>
LeafTypeTree<T> Zip(LeafTypeTreeView<A> a, LeafTypeTreeView<A> b,
                    std::function<T(const A&, const A&)> f) {
  absl::StatusOr<LeafTypeTree<T>> result = ZipIndex<T, A>(
      {a, b},
      [&](Type* element_type, absl::Span<const A* const> elements,
          absl::Span<const int64_t> index) -> absl::StatusOr<T> {
        return f(*elements[0], *elements[1]);
      });
  CHECK_OK(result);
  return result.value();
}

// Produce a new `LeafTypeTree` from this one `LeafTypeTreeView` with a
// different leaf type by way of a function.
template <typename T, typename R>
absl::StatusOr<LeafTypeTree<T>> MapIndex(
    LeafTypeTreeView<R> ltt,
    std::function<absl::StatusOr<T>(Type* element_type, const R& element,
                                    absl::Span<const int64_t> index)>
        function) {
  return ZipIndex<T, R>(
      {ltt}, [&](Type* element_type, absl::Span<const R* const> elements,
                 absl::Span<const int64_t> index) {
        return function(element_type, *elements.front(), index);
      });
}

// Simple form of map which includes only data element arguments.
template <typename T, typename R>
LeafTypeTree<T> Map(LeafTypeTreeView<R> ltt,
                    std::function<T(const R& element)> function) {
  return MapIndex<T, R>(
             {ltt},
             [&](Type* element_type, const R& element,
                 absl::Span<const int64_t> index) -> absl::StatusOr<T> {
               return function(element);
             })
      .value();
}

// Use the given function to update each leaf element in this `LeafTypeTree`
// using the corresponding element in the `other`. Return an error if the given
// `LeafTypeTree`s are not generated from the same type.
template <typename T, typename U>
absl::Status UpdateFrom(
    MutableLeafTypeTreeView<T> ltt, LeafTypeTreeView<U> other,
    std::function<absl::Status(Type* element_type, T& element,
                               const U& other_element,
                               absl::Span<const int64_t> index)>
        update,
    absl::Span<const int64_t> index_prefix = {}) {
  leaf_type_tree_internal::LeafTypeTreeIterator it(ltt.type(), index_prefix);
  while (!it.AtEnd()) {
    XLS_RETURN_IF_ERROR(
        update(it.leaf_type(), ltt.elements()[it.linear_index()],
               other.elements()[it.linear_index()], it.type_index()));
    it.Advance();
  }
  return absl::OkStatus();
}

// Simple form of UpdateFromIndex which includes only data element arguments.
template <typename T, typename U>
void SimpleUpdateFrom(
    MutableLeafTypeTreeView<T> ltt, LeafTypeTreeView<U> other,
    std::function<void(T& element, const U& other_element)> update) {
  CHECK_EQ(ltt.type(), other.type());
  for (int64_t i = 0; i < ltt.elements().size(); ++i) {
    update(ltt.elements()[i], other.elements()[i]);
  }
}

// Clones the given view into a separate LeafTypeTree object.
template <typename T>
LeafTypeTree<T> Clone(LeafTypeTreeView<T> ltt) {
  return LeafTypeTree<T>(ltt.type(), ltt.elements());
}

// Calls the given function on each leaf element. The element type, element
// data, and element index is passed to the function. The elements are iterated
// in lexicographic order of the indices. `index_prefix` is a prefix which is
// added to the type index passed to `f`.
template <typename T>
absl::Status ForEachIndex(
    MutableLeafTypeTreeView<T> ltt,
    const std::function<absl::Status(
        Type* element_type, typename MutableLeafTypeTreeView<T>::DataT& element,
        absl::Span<const int64_t> index)>& f,
    absl::Span<const int64_t> index_prefix = {}) {
  leaf_type_tree_internal::LeafTypeTreeIterator it(ltt.type(), index_prefix);
  while (!it.AtEnd()) {
    XLS_RETURN_IF_ERROR(
        f(it.leaf_type(), ltt.elements()[it.linear_index()], it.type_index()));
    it.Advance();
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status ForEachIndex(
    LeafTypeTreeView<T> ltt,
    const std::function<absl::Status(
        Type* element_type, typename LeafTypeTreeView<T>::DataT& element,
        absl::Span<const int64_t> index)>& f,
    absl::Span<const int64_t> index_prefix = {}) {
  leaf_type_tree_internal::LeafTypeTreeIterator it(ltt.type(), index_prefix);
  while (!it.AtEnd()) {
    XLS_RETURN_IF_ERROR(
        f(it.leaf_type(), ltt.elements()[it.linear_index()], it.type_index()));
    it.Advance();
  }
  return absl::OkStatus();
}

// Simple form of ForEachIndex which includes only data element arguments.
template <typename T>
void ForEach(MutableLeafTypeTreeView<T> ltt,
             const std::function<void(
                 typename MutableLeafTypeTreeView<T>::DataT& element)>& f) {
  for (T& x : ltt.elements()) {
    f(x);
  }
}

template <typename T>
void ForEach(
    LeafTypeTreeView<T> ltt,
    const std::function<void(typename LeafTypeTreeView<T>::DataT& element)>&
        f) {
  for (const T& x : ltt.elements()) {
    f(x);
  }
}

// Calls the given function on each sub-array/element of the type.
// `index_depth` determines the iteration space, specifically this is the
// number of outer dimensions to iterate over. For example, for a 2D array
// u32[3][4]:
//
//   `index_depth` == 0: `f` is called once on the entire array (u32[3][4]).
//
//   `index_depth` == 1: `f` is called once for each of the 4 1D subarrays
//   (u32[3]).
//
//   `index_depth` == 2: `f` is called once for each of the 12 leaf elements
//   (u32).
template <typename T>
absl::Status ForEachSubArray(
    MutableLeafTypeTreeView<T> ltt, int64_t index_depth,
    const std::function<absl::Status(MutableLeafTypeTreeView<T> view,
                                     absl::Span<const int64_t> index)>& f) {
  return leaf_type_tree_internal::ForEachSubArrayHelper<
      T, MutableLeafTypeTreeView<T>>(ltt.type(), ltt.elements(),
                                     ltt.leaf_types(), index_depth, f);
}

template <typename T>
absl::Status ForEachSubArray(
    LeafTypeTreeView<T> ltt, int64_t index_depth,
    const std::function<absl::Status(LeafTypeTreeView<T> view,
                                     absl::Span<const int64_t> index)>& f) {
  return leaf_type_tree_internal::ForEachSubArrayHelper<const T,
                                                        LeafTypeTreeView<T>>(
      ltt.type(), ltt.elements(), ltt.leaf_types(), index_depth, f);
}

}  // namespace leaf_type_tree

template <typename T>
SharedLeafTypeTree<T> LeafTypeTree<T>::AsShared() && {
  return SharedLeafTypeTree<T>(std::move(*this));
}
template <typename T>
SharedLeafTypeTree<T> LeafTypeTreeView<T>::AsShared() const {
  return SharedLeafTypeTree<T>(*this);
}
}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_LEAF_TYPE_TREE_H_
