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

#include "xls/noc/config_ng/coordinate.h"

#include <cstdint>
#include <optional>

#include "gtest/gtest.h"

namespace xls::noc {
namespace {

TEST(CoordinateTest, AccessFields) {
  Coordinate coordinate = {0, 2, 3};
  EXPECT_EQ(coordinate.GetDimensionCount(), 3);
  EXPECT_EQ(coordinate.GetCoordinate(0), 0);
  coordinate.SetCoordinate(0, 1);
  EXPECT_EQ(coordinate.GetCoordinate(0), 1);
}

TEST(CoordinateTest, GetDimensionBounds) {
  Coordinate coordinate(3, 3);
  EXPECT_EQ(coordinate.GetDimensionCount(), 3);
  for (int64_t coor : coordinate.GetCoordinates()) {
    EXPECT_EQ(coor, 3);
  }
}

TEST(CoordinateTest, HasZeroDimensions) {
  EXPECT_TRUE(Coordinate({}).HasZeroDimensions());
  EXPECT_FALSE(Coordinate({1, 2, 3}).HasZeroDimensions());
}

TEST(CoordinateTest, IsDimensionCountEqual) {
  Coordinate coordinate_a = {0, 2, 3};
  Coordinate coordinate_b = {0, 2, 3};
  // Test with same number of dimensions.
  EXPECT_TRUE(coordinate_a.IsDimensionCountEqual(coordinate_b));
  // Test without same number of dimensions.
  EXPECT_FALSE(coordinate_a.IsDimensionCountEqual(Coordinate({0, 9})));
  EXPECT_FALSE(coordinate_a.IsDimensionCountEqual(DimensionBounds({0})));
}

TEST(CoordinateTest, IsInDimensionalSpace) {
  DimensionBounds space = {1, 2, 3};
  // Test with coordinate within dimension.
  EXPECT_TRUE(Coordinate({0, 1, 2}).IsInDimensionalSpace(space));
  // Test with coordinate not within dimension.
  EXPECT_FALSE(Coordinate({3, 3}).IsInDimensionalSpace(space));
  EXPECT_FALSE(Coordinate({2, 3, 3}).IsInDimensionalSpace(space));
}

TEST(CoordinateTest, GetDifferentDimensionLocationsWith) {
  // Test with no difference between dimensions.
  InlineBitmap no_diff = Coordinate({0, 1, 2})
                             .GetDifferentDimensionLocationsWith({0, 1, 2})
                             .value();
  EXPECT_EQ(no_diff.bit_count(), 3);
  EXPECT_FALSE(no_diff.Get(0));
  EXPECT_FALSE(no_diff.Get(1));
  EXPECT_FALSE(no_diff.Get(2));

  // Test with two differences between dimensions.
  InlineBitmap diff = Coordinate({0, 1, 2})
                          .GetDifferentDimensionLocationsWith({2, 0, 2})
                          .value();
  EXPECT_EQ(diff.bit_count(), 3);
  EXPECT_TRUE(diff.Get(0));
  EXPECT_TRUE(diff.Get(1));
  EXPECT_FALSE(diff.Get(2));

  // Test with invalid dimension count
  EXPECT_EQ(Coordinate({0, 1, 2}).GetDifferentDimensionLocationsWith({2, 0}),
            std::nullopt);
}

TEST(CoordinateTest, GetNumDifferentDimensionLocationsWith) {
  // Test with no difference between dimensions.
  EXPECT_EQ(Coordinate({0, 1, 2})
                .GetNumDifferentDimensionLocationsWith({0, 1, 2})
                .value(),
            0);
  // Test with two differences between dimensions.
  EXPECT_EQ(Coordinate({0, 1, 2})
                .GetNumDifferentDimensionLocationsWith({2, 0, 2})
                .value(),
            2);
  // Test with invalid dimension count
  EXPECT_EQ(Coordinate({0, 1, 2}).GetDifferentDimensionLocationsWith({2, 0}),
            std::nullopt);
}

TEST(CoordinateTest, GetUniqueDifferentDimensionIndex) {
  // Test with a difference at dimension index 2.
  EXPECT_EQ(
      Coordinate({0, 1, 2}).GetUniqueDifferentDimensionIndex({0, 1, 1}).value(),
      2);
  // Test with more than one difference at the dimensions.
  EXPECT_EQ(Coordinate({0, 1, 2}).GetUniqueDifferentDimensionIndex({2, 0, 2}),
            std::nullopt);
  // Test with no dimensions that differ.
  EXPECT_EQ(Coordinate({2, 0, 2}).GetUniqueDifferentDimensionIndex({2, 0, 2}),
            std::nullopt);
}

TEST(CoordinateTest, EqualOperator) {
  EXPECT_TRUE(Coordinate({}) == Coordinate({}));
  EXPECT_TRUE(Coordinate({2, 3, 4}) == Coordinate({2, 3, 4}));
  EXPECT_FALSE(Coordinate({1, 2, 3}) == Coordinate({2, 3, 4}));
}

TEST(CoordinateTest, NotEqualOperator) {
  EXPECT_FALSE(Coordinate({}) != Coordinate({}));
  EXPECT_FALSE(Coordinate({2, 3, 4}) != Coordinate({2, 3, 4}));
  EXPECT_TRUE(Coordinate({1, 2, 3}) != Coordinate({2, 3, 4}));
}

}  // namespace
}  // namespace xls::noc
