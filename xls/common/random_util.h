// Copyright 2023 The XLS Authors
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

#ifndef XLS_COMMON_RANDOM_UTIL_H_
#define XLS_COMMON_RANDOM_UTIL_H_

#include <algorithm>
#include <iterator>

#include "absl/log/check.h"
#include "absl/random/bit_gen_ref.h"
#include "xls/common/logging/logging.h"

namespace xls {

// Chooses one element of `choices` at random, using the provided uniform random
// bit generator.
template <typename T>
typename T::value_type RandomChoice(const T& choices, absl::BitGenRef bit_gen) {
  CHECK(!choices.empty());
  typename T::value_type result;
  std::sample(std::begin(choices), std::end(choices), &result, 1, bit_gen);
  return result;
}

}  // namespace xls

#endif  // XLS_COMMON_RANDOM_UTIL_H_
