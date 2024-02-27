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
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/type.h"

namespace xls {

namespace internal {

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

}  // namespace internal

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
  LeafTypeTree(LeafTypeTree<T>&& other) = default;
  LeafTypeTree& operator=(LeafTypeTree<T>&& other) = default;

  explicit LeafTypeTree(Type* type)
      : type_(type), elements_(type->leaf_count()) {
    MakeLeafTypes(type);
  }
  LeafTypeTree(Type* type, const T& init_value)
      : type_(type), elements_(type->leaf_count(), init_value) {
    MakeLeafTypes(type);
  }
  LeafTypeTree(Type* type, std::function<T(Type*)> init_function)
      : type_(type) {
    elements_.reserve(type->leaf_count());
    MakeLeafTypes(type);
    for (Type* leaf_type : leaf_types_) {
      elements_.push_back(init_function(leaf_type));
    }
  }

  // Constructor for tuples/arrays where members are provided as a span.
  LeafTypeTree(Type* type, absl::Span<LeafTypeTree<T> const> init_values)
      : type_(type) {
    // Validate types of given values.
    if (type->IsArray()) {
      XLS_CHECK_EQ(type->AsArrayOrDie()->size(), init_values.size());
      for (auto& init_value : init_values) {
        XLS_CHECK_EQ(type->AsArrayOrDie()->element_type(), init_value.type());
      }
    } else if (type->IsTuple()) {
      XLS_CHECK_EQ(type->AsTupleOrDie()->size(), init_values.size());
      for (int64_t i = 0; i < init_values.size(); ++i) {
        XLS_CHECK_EQ(type->AsTupleOrDie()->element_type(i),
                     init_values[i].type());
      }
    } else {
      XLS_LOG(FATAL) << "Invalid constructor for bits types";
    }

    MakeLeafTypes(type);
    for (int64_t i = 0; i < init_values.size(); ++i) {
      for (int64_t j = 0; j < init_values[i].size(); ++j) {
        const T& leaf = init_values[i].elements()[j];
        elements_.push_back(leaf);
      }
    }
  }

  LeafTypeTree(Type* type, absl::Span<const T> elements)
      : type_(type), elements_(elements.begin(), elements.end()) {
    MakeLeafTypes(type_);
    XLS_CHECK_EQ(leaf_types_.size(), elements_.size());
  }

  // Constructor which avoids copying by moving elements one-by-one.
  LeafTypeTree(Type* type, absl::Span<T> elements) : type_(type) {
    MakeLeafTypes(type_);
    XLS_CHECK_EQ(leaf_types_.size(), elements.size());
    elements_.reserve(elements.size());
    for (T& element : elements) {
      elements_.push_back(std::move(element));
    }
  }

  Type* type() const { return type_; }

  // Returns the number of values in the container (equivalently number of
  // leaves of the type).
  int64_t size() const { return elements_.size(); }

  // Returns the element at the given Type index.  The Type index defines a
  // recursive traversal through the object's XLS type. The Type index must
  // correspond to a leaf Bits-type element in the object's XLS type.
  T& Get(absl::Span<int64_t const> index) {
    std::pair<Type*, int64_t> type_offset = GetSubtypeAndOffset(type_, index);
    // The index must refer to a leaf node (bits or token type).
    XLS_CHECK(IsLeafType(type_offset.first));
    return elements_[type_offset.second];
  }
  const T& Get(absl::Span<int64_t const> index) const {
    return const_cast<LeafTypeTree*>(this)->Get(index);
  }

  // Sets the element at the given Type index to the given value.
  void Set(absl::Span<int64_t const> index, const T& value) {
    std::pair<Type*, int64_t> type_offset = GetSubtypeAndOffset(type_, index);
    // The index must refer to a leaf node (bits or token type).
    XLS_CHECK(IsLeafType(type_offset.first));
    elements_[type_offset.second] = value;
  }

  // Returns the values stored in this container.
  absl::Span<T> elements() { return absl::Span<T>(elements_); }
  absl::Span<T const> elements() const {
    return absl::Span<T const>(elements_);
  }

  // Returns the values corresponding to the subtree rooted at the given index.
  absl::Span<T> GetSubelements(absl::Span<const int64_t> index) {
    std::pair<Type*, int64_t> type_offset = GetSubtypeAndOffset(type_, index);
    return type_offset.first->leaf_count() == 0
               ? absl::Span<T>()
               : absl::Span<T>(&elements_[type_offset.second],
                               type_offset.first->leaf_count());
  }
  absl::Span<T const> GetSubelements(absl::Span<const int64_t> index) const {
    std::pair<Type*, int64_t> type_offset = GetSubtypeAndOffset(type_, index);
    return type_offset.first->leaf_count() == 0
               ? absl::Span<const T>()
               : absl::Span<const T>(&elements_[type_offset.second],
                                     type_offset.first->leaf_count());
  }

  // Returns the types of each leaf in the XLS type of this object. The order of
  // these types corresponds to the order of elements().
  absl::Span<Type* const> leaf_types() const { return leaf_types_; }

  // Copies and returns the subtree rooted at the given type index as a
  // LeafTypeTree.
  LeafTypeTree<T> CopySubtree(absl::Span<const int64_t> index) const {
    std::pair<Type*, int64_t> type_offset = GetSubtypeAndOffset(type_, index);
    // To avoid indexing into an empty vector, handle the empty elements()
    // case specially.
    return LeafTypeTree<T>(
        type_offset.first,
        type_offset.first->leaf_count() == 0
            ? absl::Span<const T>()
            : absl::Span<const T>(&elements_[type_offset.second],
                                  type_offset.first->leaf_count()));
  }

  // Produce a new `LeafTypeTree` from this one `LeafTypeTree` with a different
  // leaf type by way of a function.
  template <typename R>
  LeafTypeTree<R> Map(std::function<R(const T&)> function) const {
    absl::InlinedVector<R, 1> new_elements;
    new_elements.reserve(size());
    for (int32_t i = 0; i < size(); ++i) {
      new_elements.push_back(function(elements()[i]));
    }
    return LeafTypeTree<R>(type(), new_elements);
  }

  // Use the given function to combine each corresponding leaf element in the
  // two given `LeafTypeTree`s. CHECK fails if the given `LeafTypeTree`s are not
  // generated from the same type.
  template <typename A, typename B>
  static LeafTypeTree<T> Zip(std::function<T(const A&, const B&)> function,
                             const LeafTypeTree<A>& lhs,
                             const LeafTypeTree<B>& rhs) {
    XLS_CHECK(lhs.type()->IsEqualTo(rhs.type()));
    XLS_CHECK_EQ(lhs.size(), rhs.size());

    absl::InlinedVector<T, 1> new_elements;
    new_elements.reserve(lhs.size());
    for (int32_t i = 0; i < lhs.size(); ++i) {
      new_elements.push_back(function(lhs.elements()[i], rhs.elements()[i]));
    }

    return LeafTypeTree<T>(lhs.type(), new_elements);
  }

  // Use the given function to update each leaf element in this `LeafTypeTree`
  // using the corresponding element in the `other`. CHECK fails if the given
  // `LeafTypeTree`s are not generated from the same type.
  template <typename U>
  void UpdateFrom(const LeafTypeTree<U>& other,
                  std::function<void(T&, const U&)> update) {
    XLS_CHECK(type_->IsEqualTo(other.type()));
    XLS_CHECK_EQ(elements_.size(), other.size());

    for (int64_t i = 0; i < size(); ++i) {
      update(elements_[i], other.elements()[i]);
    }
  }

  friend bool operator==(const LeafTypeTree<T>& lhs,
                         const LeafTypeTree<T>& rhs) {
    if (lhs.type_ != rhs.type_) {
      return false;
    }
    XLS_CHECK_EQ(lhs.leaf_types_.size(), rhs.leaf_types_.size());
    return lhs.elements_ == rhs.elements_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const LeafTypeTree<T>& ltt) {
    return H::combine(std::move(h), ltt.type_, ltt.elements());
  }

  // Calls the given function on each leaf element. The element type, element
  // data, and element index is passed to the function. The elements are
  // iterated in lexicographic order of the indices. `index_prefix` can be used
  // to limit the iteration to only those indices whose first elements match
  // `index_prefix`.
  absl::Status ForEach(
      const std::function<absl::Status(Type* element_type, T& element,
                                       absl::Span<const int64_t> index)>& f,
      absl::Span<const int64_t> index_prefix = {}) {
    std::vector<int64_t> type_index(index_prefix.begin(), index_prefix.end());
    auto [subtype, linear_index] = GetSubtypeAndOffset(type(), index_prefix);
    return ForEachHelper(subtype, f, linear_index, type_index);
  }

  // Const overload of ForEach.
  absl::Status ForEach(
      const std::function<absl::Status(Type* element_type, const T& element,
                                       absl::Span<const int64_t> index)>& f,
      absl::Span<const int64_t> index_prefix = {}) const {
    return const_cast<LeafTypeTree<T>*>(this)->ForEach(
        [&](Type* type, T& data, absl::Span<const int64_t> index) {
          return f(type, data, index);
        });
  }

  // Calls the given function on each sub-array/element of the
  // type. `index_depth` determines the iteration space, specifically this is
  // the number of outer dimensions to iterate over. For example, for a 2D array
  // u32[3][4]:
  //
  //   `index_depth` == 0: `f` is called once on the entire array (u32[3][4]).
  //
  //   `index_depth` == 1: `f` is called once for each of the 4 1D subarrays
  //   (u32[3]).
  //
  //   `index_depth` == 2: `f` is called once for each of the 12 leaf elements
  //   (u32).
  absl::Status ForEachSubArray(
      int64_t index_depth,
      const std::function<absl::Status(Type* subtype, absl::Span<T> elements,
                                       absl::Span<const int64_t> index)>& f) {
    XLS_ASSIGN_OR_RETURN(internal::SubArraySize subarray_size,
                         internal::GetSubArraySize(type(), index_depth));
    int64_t linear_index = 0;
    std::vector<int64_t> array_index(index_depth, 0);
    do {
      XLS_RETURN_IF_ERROR(f(subarray_size.type,
                            absl::MakeSpan(elements_).subspan(
                                linear_index, subarray_size.element_count),
                            array_index));
      linear_index += subarray_size.element_count;
    } while (
        !internal::IncrementArrayIndex(subarray_size.bounds, &array_index));
    return absl::OkStatus();
  }

  absl::Status ForEachSubArray(int64_t index_depth,
                               const std::function<absl::Status(
                                   Type* subtype, absl::Span<const T> elements,
                                   absl::Span<const int64_t> index)>& f) const {
    XLS_ASSIGN_OR_RETURN(internal::SubArraySize subarray_size,
                         internal::GetSubArraySize(type(), index_depth));
    int64_t linear_index = 0;
    std::vector<int64_t> array_index(index_depth, 0);
    do {
      XLS_RETURN_IF_ERROR(f(subarray_size.type,
                            absl::MakeConstSpan(elements_).subspan(
                                linear_index, subarray_size.element_count),
                            array_index));
      linear_index += subarray_size.element_count;
    } while (
        !internal::IncrementArrayIndex(subarray_size.bounds, &array_index));
    return absl::OkStatus();
  }

  // Returns the stringified elements of the LeafTypeTree in a structured
  // form. Examples for a LeaftypeTree of integers:
  //   bits/token type: 42
  //   tuple type:      (1, 2)
  //   array type:      [10, 20, 30]
  //   compound type:   (1, (), ([42], (10, 20)))
  std::string ToString(const std::function<std::string(const T&)>& f) const {
    int64_t linear_index = 0;
    return ToStringHelper(f, type(), /*multiline=*/false, 0, linear_index);
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
    int64_t linear_index = 0;
    return ToStringHelper(f, type(), /*multiline=*/true, 0, linear_index);
  }
  std::string ToMultilineString() const {
    return ToMultilineString(
        [](const T& element) { return absl::StrCat(element); });
  }

 private:
  static bool IsLeafType(Type* t) { return t->IsBits() || t->IsToken(); }

  std::string ToStringHelper(const std::function<std::string(const T&)>& f,
                             Type* subtype, bool multiline, int64_t indent,
                             int64_t& linear_index) const {
    std::string indentation(indent, ' ');
    if (subtype->IsArray()) {
      std::vector<std::string> pieces;
      for (int64_t i = 0; i < subtype->AsArrayOrDie()->size(); ++i) {
        pieces.push_back(ToStringHelper(f,
                                        subtype->AsArrayOrDie()->element_type(),
                                        multiline, indent + 2, linear_index));
      }
      if (multiline) {
        if (pieces.empty()) {
          return absl::StrFormat("%s[]", indentation);
        }
        return absl::StrFormat("%s[\n%s\n%s]", indentation,
                               absl::StrJoin(pieces, ",\n"), indentation);
      }
      return absl::StrFormat("[%s]", absl::StrJoin(pieces, ", "));
    }
    if (subtype->IsTuple()) {
      std::vector<std::string> pieces;
      for (int64_t i = 0; i < subtype->AsTupleOrDie()->size(); ++i) {
        pieces.push_back(
            ToStringHelper(f, subtype->AsTupleOrDie()->element_type(i),
                           multiline, indent + 2, linear_index));
      }
      if (multiline) {
        if (pieces.empty()) {
          return absl::StrFormat("%s()", indentation);
        }
        return absl::StrFormat("%s(\n%s\n%s)", indentation,
                               absl::StrJoin(pieces, ",\n"), indentation);
      }
      return absl::StrFormat("(%s)", absl::StrJoin(pieces, ", "));
    }
    if (multiline) {
      return absl::StrFormat("%s%s", indentation,
                             f(elements().at(linear_index++)));
    }
    return f(elements().at(linear_index++));
  }

  // Creates the vector of leaf types.
  void MakeLeafTypes(Type* t) {
    if (IsLeafType(t)) {
      leaf_types_.push_back(t);
      return;
    }
    if (t->IsArray()) {
      for (int64_t i = 0; i < t->AsArrayOrDie()->size(); ++i) {
        MakeLeafTypes(t->AsArrayOrDie()->element_type());
      }
      return;
    }
    XLS_CHECK(t->IsTuple());
    for (int64_t i = 0; i < t->AsTupleOrDie()->size(); ++i) {
      MakeLeafTypes(t->AsTupleOrDie()->element_type(i));
    }
  }

  // Returns a pair containing the Type and element offset for the given type
  // index.
  std::pair<Type*, int64_t> GetSubtypeAndOffset(Type* t,
                                                absl::Span<int64_t const> index,
                                                int64_t offset = 0) const {
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
    int64_t element_offset = 0;
    for (int64_t i = 0; i < index[0]; ++i) {
      element_offset += tuple_type->element_type(i)->leaf_count();
    }
    return GetSubtypeAndOffset(tuple_type->element_type(index[0]),
                               index.subspan(1), offset + element_offset);
  }

  absl::Status ForEachHelper(
      Type* subtype,
      const std::function<absl::Status(Type*, T&, absl::Span<const int64_t>)>&
          f,
      int64_t& linear_index, std::vector<int64_t>& type_index) {
    if (subtype->IsArray()) {
      for (int64_t i = 0; i < subtype->AsArrayOrDie()->size(); ++i) {
        type_index.push_back(i);
        XLS_RETURN_IF_ERROR(
            ForEachHelper(subtype->AsArrayOrDie()->element_type(), f,
                          linear_index, type_index));
        type_index.pop_back();
      }
      return absl::OkStatus();
    }
    if (subtype->IsTuple()) {
      for (int64_t i = 0; i < subtype->AsTupleOrDie()->size(); ++i) {
        type_index.push_back(i);
        XLS_RETURN_IF_ERROR(
            ForEachHelper(subtype->AsTupleOrDie()->element_type(i), f,
                          linear_index, type_index));
        type_index.pop_back();
      }
      return absl::OkStatus();
    }
    return f(subtype, elements_[linear_index++], type_index);
  }

  Type* type_;
  absl::InlinedVector<T, 1> elements_;
  absl::InlinedVector<Type*, 1> leaf_types_;
};

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_LEAF_TYPE_TREE_H_
