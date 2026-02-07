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

// This provides a very simple, boring adaptor for a begin and end iterator
// into a range type. This should be used to build range views that work well
// with range based for loops and range based constructors.

#ifndef XLS_COMMON_ITERATOR_RANGE_H_
#define XLS_COMMON_ITERATOR_RANGE_H_

#include <utility>

#include "absl/base/macros.h"
#include "absl/iterator/range.h"

namespace xabsl {

// A range adaptor for a pair of iterators.
//
// This just wraps two iterators into a range-compatible interface. Nothing
// fancy at all.
template <typename IteratorT>
using iterator_range ABSL_DEPRECATE_AND_INLINE() =
    absl::iterator_range<IteratorT>;

// Convenience function for iterating over sub-ranges.
//
// This provides a bit of syntactic sugar to make using sub-ranges
// in for loops a bit easier. Analogous to std::make_pair().
template <typename T>
ABSL_DEPRECATE_AND_INLINE()
iterator_range<T> make_range(T x, T y) {
  return absl::make_range(std::move(x), std::move(y));
}

// Converts std::pair<Iter,Iter> to iterator_range<Iter>. E.g.:
//   for (const auto& e : make_range(m.equal_range(k))) ...
template <typename T>
ABSL_DEPRECATE_AND_INLINE()
iterator_range<T> make_range(std::pair<T, T> p) {
  return absl::make_range(std::move(p));
}

}  // namespace xabsl

#endif  // XLS_COMMON_ITERATOR_RANGE_H_
