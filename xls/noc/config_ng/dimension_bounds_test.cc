// Copyright 2021 The XLS Authors
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

#include "xls/noc/config_ng/dimension_bounds.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace xls::noc {
namespace {

TEST(DimensionBoundsTest, AccessFields) {
  DimensionBounds dimension = {0, 2, 3};
  EXPECT_EQ(dimension.GetDimensionCount(), 3);
  EXPECT_EQ(dimension.GetDimensionBound(0), 0);
  dimension.SetDimensionBound(0, 1);
  EXPECT_EQ(dimension.GetDimensionBound(0), 1);
}

TEST(DimensionBoundsTest, GetDimensionBounds) {
  DimensionBounds dimension(3, 3);
  EXPECT_EQ(dimension.GetDimensionCount(), 3);
  for (int64_t bound : dimension.GetDimensionBounds()) {
    EXPECT_EQ(bound, 3);
  }
}

TEST(DimensionBoundsTest, IsNumberOfDimensionsEqual) {
  DimensionBounds dimension = {1, 2, 3};
  EXPECT_TRUE(dimension.IsDimensionCountEqual(dimension));
  EXPECT_FALSE(dimension.IsDimensionCountEqual({1, 1}));
}

TEST(DimensionBoundsTest, HasZeroDimensions) {
  EXPECT_TRUE(DimensionBounds({}).HasZeroDimensions());
  EXPECT_FALSE(DimensionBounds({1, 2, 3}).HasZeroDimensions());
}

TEST(DimensionBoundsTest, EqualOperator) {
  EXPECT_TRUE(DimensionBounds({}) == DimensionBounds({}));
  EXPECT_TRUE(DimensionBounds({2, 3, 4}) == DimensionBounds({2, 3, 4}));
  EXPECT_FALSE(DimensionBounds({1, 2, 3}) == DimensionBounds({2, 3, 4}));
}

TEST(DimensionBoundsTest, NotEqualOperator) {
  EXPECT_FALSE(DimensionBounds({}) != DimensionBounds({}));
  EXPECT_FALSE(DimensionBounds({2, 3, 4}) != DimensionBounds({2, 3, 4}));
  EXPECT_TRUE(DimensionBounds({1, 2, 3}) != DimensionBounds({2, 3, 4}));
}

}  // namespace
}  // namespace xls::noc
