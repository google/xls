// Copyright 2026 The XLS Authors
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

#ifndef XLS_COMMON_SORTED_VECTOR_H_
#define XLS_COMMON_SORTED_VECTOR_H_

#include <algorithm>
#include <compare>
#include <concepts>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/types/span.h"

namespace xabsl {

// An immutable vector that is guaranteed to be sorted.
template <typename T>
class SortedVector final : public absl::Span<T> {
  std::vector<T> storage_;
  std::function<bool(const T&, const T&)> comp_;

  SortedVector(std::vector<T>&& other,
               std::function<bool(const T&, const T&)> comp)
      // This relies on the guarantee that the move ctor of std::vector not
      // causing a memcopy, i.e. what is pointed to by data() being the same,
      // and it just transferring over the pointer and releasing ownership. See
      // also https://en.cppreference.com/cpp/container/vector/vector, "Notes":
      // After container move construction (overload (8)), references, pointers,
      // and iterators that originally refer to elements in other remain valid,
      // but refer to elements that are now in *this. The current standard makes
      // this guarantee via the blanket statement in [container.reqmts]/67, and
      // a more direct guarantee is under consideration via LWG issue 2321
      : absl::Span<T>(other.data(), other.size()),
        storage_(std::move(other)),
        comp_(std::move(comp)) {}

 public:
  SortedVector() : absl::Span<T>(), comp_() {}
  SortedVector(SortedVector&&) = default;
  SortedVector& operator=(SortedVector&&) = default;

  // The copy ctor/operator could be implemented if we have a motivating use
  // case. Right now, we don't, and the benefit of its absence is compile-time
  // detection of attempts to pass by value (which the relation to Span may
  // suggest as cheap - but they are not).
  SortedVector(const SortedVector&) = delete;
  SortedVector& operator=(const SortedVector&) = delete;

  bool contains(const T& value) const {
    return std::binary_search(storage_.begin(), storage_.end(), value, comp_);
  }

  template <typename Compare = std::compare_three_way>
    requires std::same_as<std::invoke_result_t<Compare, const T&, const T&>,
                          std::strong_ordering>
  static SortedVector create(std::vector<T>&& other, Compare comp = Compare{}) {
    auto bool_comp = [comp = std::move(comp)](const T& a, const T& b) {
      return comp(a, b) == std::strong_ordering::less;
    };
    std::sort(other.begin(), other.end(), bool_comp);
    return SortedVector(std::move(other), std::move(bool_comp));
  }
};
}  // namespace xabsl

#endif  // XLS_COMMON_SORTED_VECTOR_H_
