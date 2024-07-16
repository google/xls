// Copyright 2024 The XLS Authors
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

#ifndef XLS_COMMON_ITER_UTIL_H_
#define XLS_COMMON_ITER_UTIL_H_

#include <cstdint>
#include <iterator>
#include <type_traits>
#include <vector>

#include "absl/types/span.h"

namespace xls {

// Calls the given function on every possible mixed-radix iterator of the given
// range.
//
// If the given function returns `true`, the iteration ends early and we return
// `true`. Otherwise, `false` is returned.
template <typename Iter, typename Func>
  requires std::is_invocable_r_v<
      bool, Func, absl::Span<typename Iter::const_iterator const>>
bool IteratorProduct(absl::Span<Iter const> iters, Func f) {
  if (iters.empty()) {
    // Nothing to do.
    return false;
  }
  std::vector<typename Iter::const_iterator> number;
  number.reserve(iters.size());
  for (const Iter& i : iters) {
    number.push_back(std::cbegin(i));
    if (std::cbegin(i) == std::cend(i)) {
      // An empty iterator is present. There is nothing we can do.
      // TODO(allight): Should we maybe CHECK fail here?
      return false;
    }
  }

  // Incr. Return true if the first iter resets.
  auto incr = [&]() -> bool {
    int64_t i = 0;
    while (i < number.size() && ++number[i] == std::cend(iters[i])) {
      number[i] = std::cbegin(iters[i]);
      ++i;
    }
    return (i == number.size());
  };
  do {
    if (f(number)) {
      return true;
    }
  } while (!incr());
  return false;
}

}  // namespace xls

#endif  // XLS_COMMON_ITER_UTIL_H_
