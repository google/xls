// Copyright 2025 The XLS Authors
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

#include "xls/ir/partial_ops.h"

#include "gtest/gtest.h"
#include "xls/ir/interval_set_test_utils.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/ternary.h"

namespace xls {
namespace {

TEST(PartialOpsTest, Join) {
  PartialInformation a(10, *StringToTernaryVector("0b11_01X1_X001"),
                       FromRanges({{0, 8}, {10, 18}, {20, 28}}, 10));
  PartialInformation b(10, *StringToTernaryVector("0b00_00XX_X01X"),
                       FromRanges({{0, 8}, {10, 18}, {20, 28}}, 10));
  PartialInformation c(10, *StringToTernaryVector("0b00_00XX_10XX"),
                       FromRanges({{0, 20}}, 10));
  PartialInformation d(10, *StringToTernaryVector("0bXX_XXXX_1XXX"),
                       FromRanges({{0, 1023}}, 10));

  EXPECT_EQ(partial_ops::Join(a, a), a);
  EXPECT_EQ(partial_ops::Join(a, PartialInformation::Impossible(10)),
            PartialInformation::Impossible(10));
  EXPECT_EQ(partial_ops::Join(a, PartialInformation::Unconstrained(10)), a);

  EXPECT_EQ(partial_ops::Join(b, b), b);
  EXPECT_EQ(partial_ops::Join(b, PartialInformation::Impossible(10)),
            PartialInformation::Impossible(10));
  EXPECT_EQ(partial_ops::Join(b, PartialInformation::Unconstrained(10)), b);

  EXPECT_EQ(partial_ops::Join(a, b), PartialInformation::Impossible(10));

  EXPECT_EQ(partial_ops::Join(b, c),
            PartialInformation(*StringToTernaryVector("0b00_0000_101X"),
                               FromRanges({{10, 11}}, 10)));

  EXPECT_EQ(partial_ops::Join(a, d), PartialInformation::Impossible(10));
}

TEST(PartialOpsTest, Meet) {
  PartialInformation a(10, *StringToTernaryVector("0b11_01X1_X001"),
                       FromRanges({{0, 8}, {10, 18}, {20, 28}}, 10));
  PartialInformation b(10, *StringToTernaryVector("0b00_00XX_X01X"),
                       FromRanges({{0, 8}, {10, 18}, {20, 28}}, 10));
  PartialInformation c(10, *StringToTernaryVector("0b00_00XX_10XX"),
                       FromRanges({{0, 20}}, 10));
  PartialInformation d(10, *StringToTernaryVector("0bXX_XXXX_1XXX"),
                       FromRanges({{0, 1023}}, 10));

  EXPECT_EQ(partial_ops::Meet(a, a), a);
  EXPECT_EQ(partial_ops::Meet(a, PartialInformation::Impossible(10)), a);
  EXPECT_EQ(partial_ops::Meet(a, PartialInformation::Unconstrained(10)),
            PartialInformation::Unconstrained(10));

  EXPECT_EQ(partial_ops::Meet(b, b), b);
  EXPECT_EQ(partial_ops::Meet(b, PartialInformation::Impossible(10)), b);
  EXPECT_EQ(partial_ops::Meet(b, PartialInformation::Unconstrained(10)),
            PartialInformation::Unconstrained(10));

  EXPECT_EQ(partial_ops::Meet(a, b),
            PartialInformation(
                *StringToTernaryVector("0b00_000X_X01X"),
                FromRanges({{2, 3}, {10, 11}, {18, 18}, {26, 27}}, 10)));

  EXPECT_EQ(partial_ops::Meet(b, c),
            PartialInformation(
                *StringToTernaryVector("0b00_000X_X0XX"),
                FromRanges({{2, 3}, {8, 11}, {18, 18}, {26, 27}}, 10)));

  EXPECT_EQ(partial_ops::Meet(a, d), PartialInformation::Unconstrained(10));
}

}  // namespace
}  // namespace xls
