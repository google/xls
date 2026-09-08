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

#ifndef XLS_DATA_STRUCTURES_ALGORITHM_H_
#define XLS_DATA_STRUCTURES_ALGORITHM_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/types/span.h"

namespace xls {

// Returns the sorted indices at which the given function evaluates to true for
// the respective element in 'sequence'.
template <typename T>
std::vector<int64_t> IndicesWhere(absl::Span<const T> sequence,
                                  absl::FunctionRef<bool(const T&)> f) {
  std::vector<int64_t> result;
  for (int64_t i = 0; i < sequence.size(); ++i) {
    if (f(sequence[i])) {
      result.push_back(i);
    }
  }
  return result;
}

// Returns the sorted indices at which the given function evaluates to false for
// the respective element in 'sequence'.
template <typename T>
std::vector<int64_t> IndicesWhereNot(absl::Span<const T> sequence,
                                     absl::FunctionRef<bool(const T&)> f) {
  std::vector<int64_t> result;
  for (int64_t i = 0; i < sequence.size(); ++i) {
    if (!f(sequence[i])) {
      result.push_back(i);
    }
  }
  return result;
}

// Returns the elements of 'sequence' at the given indices.
template <typename T>
std::vector<T> GatherFromSequence(absl::Span<const T> sequence,
                                  absl::Span<const int64_t> indices) {
  std::vector<T> result;
  for (int64_t index : indices) {
    result.push_back(sequence[index]);
  }
  return result;
}

namespace internal {

template <typename T, typename Func>
inline void ForEachUniqueLinearScan(absl::Span<const T> values, Func func) {
  for (int64_t i = 0; i < values.size(); ++i) {
    bool already_seen = false;
    for (int64_t j = 0; j < i; ++j) {
      if (values[j] == values[i]) {
        already_seen = true;
        break;
      }
    }
    if (already_seen) {
      continue;
    }
    func(values[i]);
  }
}

template <typename T, typename Func>
inline void ForEachUniqueHashSet(absl::Span<const T> values, Func func,
                                 absl::flat_hash_set<T>& scratch_set) {
  scratch_set.reserve(values.size());
  for (const T& value : values) {
    if (scratch_set.insert(value).second) {
      func(value);
    }
  }
}

}  // namespace internal

// Calls `func` on each unique value in `values`.
//
// If provided, `scratch_set` will be used as scratch space if the size of
// `values` is above `kSmallSetThreshold`, potentially avoiding an allocation.
template <typename T, size_t kSmallSetThreshold = 16, typename Func>
void ForEachUnique(absl::Span<const T> values, Func func,
                   absl::flat_hash_set<T>& scratch_set) {
  if (values.size() <= kSmallSetThreshold) {
    internal::ForEachUniqueLinearScan(values, func);
    return;
  }

  scratch_set.clear();
  internal::ForEachUniqueHashSet(values, func, scratch_set);
}
template <typename T, size_t kSmallSetThreshold = 16, typename Func>
void ForEachUnique(absl::Span<const T> values, Func func) {
  if (values.size() <= kSmallSetThreshold) {
    internal::ForEachUniqueLinearScan(values, func);
    return;
  }

  absl::flat_hash_set<T> scratch_set;
  internal::ForEachUniqueHashSet(values, func, scratch_set);
}

namespace internal {

template <typename T>
inline std::vector<T> DeduplicateToVectorLinearScan(
    absl::Span<const T> values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (int64_t i = 0; i < values.size(); ++i) {
    bool already_seen = false;
    for (int64_t j = 0; j < i; ++j) {
      if (values[j] == values[i]) {
        already_seen = true;
        break;
      }
    }
    if (already_seen) {
      continue;
    }
    result.push_back(values[i]);
  }
  return result;
}

template <typename T>
inline std::vector<T> DeduplicateToVectorHashSet(
    absl::Span<const T> values, absl::flat_hash_set<T>& scratch_set) {
  scratch_set.reserve(values.size());
  std::vector<T> result;
  result.reserve(values.size());
  for (const T& value : values) {
    if (scratch_set.insert(value).second) {
      result.push_back(value);
    }
  }
  return result;
}

}  // namespace internal

// Returns a vector of the unique values in `values`.
//
// If provided, `scratch_set` will be used as scratch space if the size of
// `values` is above `kSmallSetThreshold`, potentially avoiding an allocation.
template <typename T, size_t kSmallSetThreshold = 16>
std::vector<T> DeduplicateToVector(absl::Span<const T> values,
                                   absl::flat_hash_set<T>& scratch_set) {
  if (values.size() <= kSmallSetThreshold) {
    return internal::DeduplicateToVectorLinearScan(values);
  }

  scratch_set.clear();
  return internal::DeduplicateToVectorHashSet(values, scratch_set);
}
template <typename T, size_t kSmallSetThreshold = 16>
std::vector<T> DeduplicateToVector(absl::Span<const T> values) {
  if (values.size() <= kSmallSetThreshold) {
    return internal::DeduplicateToVectorLinearScan(values);
  }

  absl::flat_hash_set<T> scratch_set;
  return internal::DeduplicateToVectorHashSet(values, scratch_set);
}

namespace internal {

constexpr size_t kSortByKeyInlineStorageCount = 16;

// Sorts `container` by moving elements into a (Key, Element) paired buffer,
// sorting the buffer, and then moving the elements back.
//
// Ideal for small types that are cheap to move.
template <typename Container, typename KeyFunc, typename Compare>
void SortByKeyAuxiliary(Container& container, KeyFunc&& key_func,
                        Compare&& comp) {
  using Element = typename Container::value_type;
  using Key = std::decay_t<std::invoke_result_t<KeyFunc, const Element&>>;

  absl::InlinedVector<std::pair<Key, Element>, kSortByKeyInlineStorageCount>
      keyed;
  keyed.reserve(container.size());
  for (auto& elem : container) {
    auto key = key_func(elem);
    keyed.emplace_back(std::move(key), std::move(elem));
  }
  absl::c_sort(keyed, [&](const auto& a, const auto& b) {
    return comp(a.first, b.first);
  });
  for (size_t i = 0; i < container.size(); ++i) {
    container[i] = std::move(keyed[i].second);
  }
}

// Sorts `container` by building a (Key, Index) buffer, sorting it, and using
// the result to permute `container` in-place.
//
// Ideal for large types that might not be cheap to move.
template <typename Container, typename KeyFunc, typename Compare>
void SortByKeyInPlace(Container& container, KeyFunc&& key_func,
                      Compare&& comp) {
  using Element = typename Container::value_type;
  using Key = std::decay_t<std::invoke_result_t<KeyFunc, const Element&>>;

  absl::InlinedVector<std::pair<Key, size_t>, kSortByKeyInlineStorageCount>
      keyed_indices;
  keyed_indices.reserve(container.size());
  for (size_t i = 0; i < container.size(); ++i) {
    keyed_indices.emplace_back(key_func(container[i]), i);
  }
  absl::c_sort(keyed_indices, [&](const auto& a, const auto& b) {
    return comp(a.first, b.first);
  });

  // Permute container to match the sorted order of the keys.
  //
  // For efficiency, we move the elements in cycles, letting us do this in O(n)
  // time and with O(1) extra space.
  for (size_t i = 0; i < container.size(); ++i) {
    if (keyed_indices[i].second == i) {
      continue;
    }
    Element tmp = std::move(container[i]);
    size_t cur = i;
    while (keyed_indices[cur].second != i) {
      size_t prev = keyed_indices[cur].second;
      container[cur] = std::move(container[prev]);
      keyed_indices[cur].second = cur;
      cur = prev;
    }
    container[cur] = std::move(tmp);
    keyed_indices[cur].second = cur;
  }
}

}  // namespace internal

// Sorts `container` by evaluating `key_func` on each element.
// `key_func` is guaranteed to be evaluated exactly once per element.
template <typename Container, typename KeyFunc, typename Compare = std::less<>>
void SortByKey(Container& container, KeyFunc&& key_func, Compare&& comp = {}) {
  using Element = typename Container::value_type;
  constexpr bool kUseDirectCopy = sizeof(Element) <= 16;

  if constexpr (kUseDirectCopy) {
    internal::SortByKeyAuxiliary(container, std::forward<KeyFunc>(key_func),
                                 std::forward<Compare>(comp));
  } else {
    internal::SortByKeyInPlace(container, std::forward<KeyFunc>(key_func),
                               std::forward<Compare>(comp));
  }
}

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_ALGORITHM_H_
