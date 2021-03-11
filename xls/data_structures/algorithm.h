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

#include <cstdint>
#include <vector>

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

}  // namespace xls

#endif  // XLS_DATA_STRUCTURES_ALGORITHM_H_
