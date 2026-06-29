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

#include "xls/common/sorted_vector.h"

#include <compare>
#include <type_traits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xabsl {
namespace {

using ::testing::ElementsAre;

TEST(SortedVectorTest, Properties) {
  static_assert(!std::is_copy_constructible_v<SortedVector<int>>);
  static_assert(std::is_move_constructible_v<SortedVector<int>>);
  static_assert(!std::is_copy_assignable_v<SortedVector<int>>);
  static_assert(std::is_move_assignable_v<SortedVector<int>>);
}

TEST(SortedVectorTest, BasicUsage_SortsByDefault) {
  auto sv = SortedVector<int>::create({3, 1, 2});
  EXPECT_EQ(sv.size(), 3);
  EXPECT_THAT(sv, ElementsAre(1, 2, 3));
  EXPECT_TRUE(sv.contains(1));
  EXPECT_TRUE(sv.contains(2));
  EXPECT_TRUE(sv.contains(3));
  EXPECT_FALSE(sv.contains(0));
  EXPECT_FALSE(sv.contains(4));
}

TEST(SortedVectorTest, CustomComparator) {
  struct Descending {
    std::strong_ordering operator()(int a, int b) const { return b <=> a; }
  };

  auto sv = SortedVector<int>::create<Descending>({3, 1, 2});
  EXPECT_EQ(sv.size(), 3);
  EXPECT_THAT(sv, ElementsAre(3, 2, 1));
  EXPECT_TRUE(sv.contains(1));
  EXPECT_TRUE(sv.contains(2));
  EXPECT_TRUE(sv.contains(3));
  EXPECT_FALSE(sv.contains(0));
  EXPECT_FALSE(sv.contains(4));
}

}  // namespace
}  // namespace xabsl
