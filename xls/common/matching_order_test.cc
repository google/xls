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

#include "xls/common/matching_order.h"

#include <array>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::Optional;

TEST(MatchingOrderTest, ReturnsElementsInPredicateOrder) {
  std::array elements = {1, 2, 3};
  EXPECT_THAT(MatchingOrder(
                  elements, [](int n) { return n == 3; },
                  [](int n) { return n == 1; }, [](int n) { return n == 2; }),
              Optional(ElementsAre(3, 1, 2)));
  EXPECT_THAT(elements, ElementsAre(1, 2, 3));
}

TEST(MatchingOrderTest, ReturnsNulloptWhenNoElementMatches) {
  EXPECT_EQ(
      MatchingOrder<int>(
          {1, 2}, [](int n) { return n == 3; }, [](int n) { return n == 2; }),
      std::nullopt);
}

TEST(MatchingOrderTest, PredicatesCannotReuseAnElement) {
  auto is_even = [](int n) { return n % 2 == 0; };
  EXPECT_EQ(MatchingOrder(std::array{1, 2}, is_even, is_even), std::nullopt);
}

TEST(MatchingOrderTest, BacktracksWhenTheFirstMatchIsNeededLater) {
  EXPECT_THAT(MatchingOrder(
                  std::array{1, 2, 3}, [](int n) { return n <= 2; },
                  [](int n) { return n == 1; }, [](int n) { return n == 3; }),
              Optional(ElementsAre(2, 1, 3)));
}

TEST(MatchingOrderTest, EmptyArrayMatchesNoPredicates) {
  EXPECT_THAT(MatchingOrder(std::array<int, 0>{}), Optional(IsEmpty()));
}

}  // namespace
}  // namespace xls
