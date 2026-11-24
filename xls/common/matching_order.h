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

#ifndef XLS_COMMON_MATCHING_ORDER_H_
#define XLS_COMMON_MATCHING_ORDER_H_

#include <array>
#include <functional>
#include <optional>
#include <type_traits>
#include <utility>

#include "absl/types/span.h"

namespace xls {
namespace matching_order_internal {

template <typename T>
bool MatchRemaining(absl::Span<T> remaining) {
  return remaining.empty();
}

template <typename T, typename Predicate, typename... Rest>
bool MatchRemaining(absl::Span<T> remaining, Predicate& predicate,
                    Rest&... rest) {
  for (T& candidate : remaining) {
    const bool matches = std::invoke(predicate, std::as_const(candidate));
    if (!matches) {
      continue;
    }
    // Let argument-dependent lookup find T's swap, falling back to std::swap.
    using std::swap;
    swap(remaining.front(), candidate);
    if (MatchRemaining(remaining.subspan(1), rest...)) {
      return true;
    }
    swap(remaining.front(), candidate);
  }
  return false;
}

}  // namespace matching_order_internal

// Returns a permutation of `elements` whose ith element satisfies the ith
// predicate, or std::nullopt if no such permutation exists. Predicates may be
// called more than once. Intended for small arrays, since overlapping
// predicates can require trying all permutations.
template <typename T, typename... Predicates>
  requires(std::is_swappable_v<T> &&
           std::is_move_constructible_v<std::array<T, sizeof...(Predicates)>> &&
           (std::is_invocable_r_v<bool, Predicates&, const T&> && ...))
std::optional<std::array<T, sizeof...(Predicates)>> MatchingOrder(
    std::array<T, sizeof...(Predicates)> elements, Predicates&&... predicates) {
  if (matching_order_internal::MatchRemaining(absl::MakeSpan(elements),
                                              predicates...)) {
    return elements;
  }
  return std::nullopt;
}

}  // namespace xls

#endif  // XLS_COMMON_MATCHING_ORDER_H_
