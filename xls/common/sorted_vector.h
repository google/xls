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

namespace xabsl {

// An immutable vector that is guaranteed to be sorted.
template <typename T>
class SortedVector final {
 private:
  std::vector<T> storage_;
  std::function<bool(const T&, const T&)> comp_;

  SortedVector(std::vector<T> other,
               std::function<bool(const T&, const T&)> comp)
      : storage_(std::move(other)), comp_(std::move(comp)) {}

 public:
  using value_type = T;
  using const_pointer = typename std::vector<T>::const_pointer;
  using const_reference = typename std::vector<T>::const_reference;
  using const_iterator = typename std::vector<T>::const_iterator;
  using size_type = typename std::vector<T>::size_type;
  using difference_type = typename std::vector<T>::difference_type;

  // Default-constructible as empty; the comparator doesn't matter.
  SortedVector()
      : storage_(), comp_([](const T& a, const T& b) { return false; }) {}

  SortedVector(SortedVector&&) = default;
  SortedVector& operator=(SortedVector&&) = default;
  SortedVector(const SortedVector& other) {
    storage_ = other.storage_;
    comp_ = other.comp_;
  }
  SortedVector& operator=(const SortedVector& other) {
    storage_ = other.storage_;
    comp_ = other.comp_;
    return *this;
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

  constexpr auto data() const { return storage_.data(); }

  constexpr auto size() const { return storage_.size(); }
  constexpr auto length() const { return storage_.length(); }

  constexpr auto empty() const { return storage_.empty(); }

  constexpr const auto& operator[](size_type i) const { return storage_[i]; }
  constexpr auto& at(size_type i) const { return storage_.at(i); }

  constexpr auto& front() const { return storage_.front(); }
  constexpr auto& back() const { return storage_.back(); }

  constexpr auto begin() const { return storage_.begin(); }
  constexpr auto cbegin() const { return storage_.cbegin(); }
  constexpr auto end() const { return storage_.end(); }
  constexpr auto cend() const { return storage_.cend(); }

  constexpr auto rbegin() const { return storage_.rbegin(); }
  constexpr auto crbegin() const { return storage_.crbegin(); }
  constexpr auto rend() const { return storage_.rend(); }
  constexpr auto crend() const { return storage_.crend(); }

  bool contains(const T& value) const {
    return std::binary_search(storage_.begin(), storage_.end(), value, comp_);
  }

  auto lower_bound(const T& value) const {
    return std::lower_bound(storage_.begin(), storage_.end(), value, comp_);
  }
  auto upper_bound(const T& value) const {
    return std::upper_bound(storage_.begin(), storage_.end(), value, comp_);
  }

  auto find(const T& value) const {
    auto it = lower_bound(value);
    if (it == storage_.end()) {
      // The entry would be past the end.
      return it;
    }
    if (comp_(*it, value)) {
      // The value is not present, since it's larger than its lower bound.
      return storage_.end();
    }
    return it;
  }
};
}  // namespace xabsl

#endif  // XLS_COMMON_SORTED_VECTOR_H_
