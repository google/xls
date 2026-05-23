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

#include "xls/scheduling/min_cut_scheduler.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

using ::testing::ElementsAre;

TEST(MinCutSchedulerTest, MinCutCycleOrders) {
  EXPECT_THAT(GetMinCutCycleOrders(0), ElementsAre(std::vector<int64_t>()));
  EXPECT_THAT(GetMinCutCycleOrders(1), ElementsAre(std::vector<int64_t>({0})));
  EXPECT_THAT(
      GetMinCutCycleOrders(2),
      ElementsAre(std::vector<int64_t>({0, 1}), std::vector<int64_t>({1, 0})));
  EXPECT_THAT(GetMinCutCycleOrders(3),
              ElementsAre(std::vector<int64_t>({0, 1, 2}),
                          std::vector<int64_t>({2, 1, 0}),
                          std::vector<int64_t>({1, 0, 2})));
  EXPECT_THAT(GetMinCutCycleOrders(4),
              ElementsAre(std::vector<int64_t>({0, 1, 2, 3}),
                          std::vector<int64_t>({3, 2, 1, 0}),
                          std::vector<int64_t>({1, 0, 2, 3})));
  EXPECT_THAT(GetMinCutCycleOrders(5),
              ElementsAre(std::vector<int64_t>({0, 1, 2, 3, 4}),
                          std::vector<int64_t>({4, 3, 2, 1, 0}),
                          std::vector<int64_t>({2, 0, 1, 3, 4})));
  EXPECT_THAT(GetMinCutCycleOrders(8),
              ElementsAre(std::vector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7}),
                          std::vector<int64_t>({7, 6, 5, 4, 3, 2, 1, 0}),
                          std::vector<int64_t>({3, 1, 0, 2, 5, 4, 6, 7})));
}

}  // namespace
}  // namespace xls
