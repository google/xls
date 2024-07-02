// Copyright 2022 The XLS Authors
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

#include "xls/data_structures/submodular.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "absl/container/btree_set.h"

namespace xls {
namespace {

TEST(SubmodularTest, ModularFunction) {
  SubmodularFunction<int32_t> f(
      {-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8},
      [](const absl::btree_set<int32_t>& subset) -> double {
        double result = 0.0;
        for (int32_t i : subset) {
          result += static_cast<double>(20 - i * i);
        }
        return result;
      });
  absl::btree_set<int32_t> expected{-8, -7, -6, -5, 5, 6, 7, 8};
  EXPECT_EQ(f.ApproxMinimize({MinimizeMode::Alternating, 1234, 1}), expected);
}

// Iwata's test function, as described in:
// https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.365.4665
//
// The optimum value is always 0
TEST(SubmodularTest, Iwata) {
  const int32_t n = 500;
  absl::btree_set<int32_t> universe;
  for (int32_t i = 1; i < n; ++i) {
    universe.insert(i);
  }
  SubmodularFunction<int32_t> f(
      universe, [&](const absl::btree_set<int32_t>& subset) -> double {
        int64_t complement_size = 0;
        for (int32_t element : universe) {
          if (!subset.contains(element)) {
            ++complement_size;
          }
        }
        int64_t sum = 0;
        for (int32_t element : subset) {
          sum += (5 * element) - (2 * n);
        }

        // Cast whatever size() type is to int64_t.
        int64_t subset_size = static_cast<int64_t>(subset.size());

        return static_cast<double>((subset_size * complement_size) - sum);
      });
  absl::btree_set<int32_t> result =
      f.ApproxMinimize({MinimizeMode::Alternating, 1234, 10});
  EXPECT_EQ(f.Call(result), -166167);
}

}  // namespace
}  // namespace xls
