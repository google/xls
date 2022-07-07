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

#ifndef XLS_COMMON_PYTHON_ABSL_CASTERS_H_
#define XLS_COMMON_PYTHON_ABSL_CASTERS_H_

#include <pybind11/cast.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <exception>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <typeinfo>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"

namespace pybind11 {
namespace detail {

// Convert between absl::Span and python sequence types.
template <typename T>
struct type_caster<absl::Span<const T>> {
  // The vector caster does not work with pointers, so neither does this.
  // Making this work should be possible- it would just require writing a
  // vector converter which keeps it's element casters alive (currently they
  // are local variables and are discarded before the value can be used).
  static_assert(!std::is_pointer<T>::value,
                "Spans of pointers are not supported.");

  type_caster() : vector_converter_(), value_(get_vector()) {}
  // Copy and Move constructors need to ensure the span points to the copied
  // or moved vector, not the original one.
  type_caster(const type_caster<absl::Span<const T>>& other)
      : vector_converter_(other.vector_converter_), value_(get_vector()) {}
  type_caster(type_caster<absl::Span<const T>>&& other)
      : vector_converter_(std::move(other.vector_converter_)),
        value_(get_vector()) {}

  static constexpr auto name = _("Span[") + make_caster<T>::name + _("]");

  // We do not allow moving because 1) spans are super lightweight, so there's
  // no advantage to moving and 2) the span cannot exist without the caster,
  // so moving leaves an implicit dependency (while a reference or pointer
  // make that dependency explicit).
  operator absl::Span<const T>*() { return &value_; }
  operator absl::Span<const T>&() { return value_; }
  template <typename T_>
  using cast_op_type = cast_op_type<T_>;

  bool load(handle src, bool convert) {
    if (!vector_converter_.load(src, convert)) return false;
    // std::vector implicitly converted to absl::Span.
    value_ = get_vector();
    return true;
  }

  template <typename CType>
  static handle cast(CType&& src, return_value_policy policy, handle parent) {
    return VectorConverter::cast(src, policy, parent);
  }

 private:
  std::vector<T>& get_vector() {
    return static_cast<std::vector<T>&>(vector_converter_);
  }
  using VectorConverter = make_caster<std::vector<T>>;
  VectorConverter vector_converter_;
  absl::Span<const T> value_;
};

// Convert between absl::flat_hash_map and python dict.
template <typename Key, typename Value, typename Hash, typename Equal,
          typename Alloc>
struct type_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>>
    : map_caster<absl::flat_hash_map<Key, Value, Hash, Equal, Alloc>, Key,
                 Value> {};

// Convert between absl::flat_hash_set and python set.
template <typename Key, typename Hash, typename Equal, typename Alloc>
struct type_caster<absl::flat_hash_set<Key, Hash, Equal, Alloc>>
    : set_caster<absl::flat_hash_set<Key, Hash, Equal, Alloc>, Key> {};

// Convert between absl::string_view and python.
//
// pybind11 supports std::string_view, and absl::string_view is meant to be a
// drop-in replacement for std::string_view, so we can just use the built in
// implementation. This is only needed until absl::string_view becomes an alias
// for std::string_view.
#ifndef ABSL_USES_STD_STRING_VIEW
template <>
struct type_caster<absl::string_view> : string_caster<absl::string_view, true> {
};
#endif

// Convert between std::optional and python.
//
// pybind11 supports std::optional, and std::optional is meant to be a
// drop-in replacement for std::optional, so we can just use the built in
// implementation.
#ifndef ABSL_USES_STD_OPTIONAL
template <typename T>
struct type_caster<std::optional<T>>
    : public optional_caster<std::optional<T>> {};
template <>
struct type_caster<absl::nullopt_t> : public void_caster<absl::nullopt_t> {};
#endif

}  // namespace detail
}  // namespace pybind11

#endif  // XLS_COMMON_PYTHON_ABSL_CASTERS_H_
