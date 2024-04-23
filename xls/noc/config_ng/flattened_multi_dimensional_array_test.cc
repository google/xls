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

#include "xls/noc/config_ng/flattened_multi_dimensional_array.h"

#include <cstdint>

#include "gtest/gtest.h"

namespace xls::noc {
namespace {

TEST(FlattenedMultiDimensionalArrayTest, GetElementCount) {
  // Test with a dimensional space of (3,4): 12 elements.
  FlattenedMultiDimensionalArray<int64_t> array({3, 4});
  EXPECT_EQ(array.GetElementCount(), 12);
  // Test with zero dimensions (no dimensions).
  FlattenedMultiDimensionalArray<int64_t> zero_array({});
  EXPECT_EQ(zero_array.GetElementCount(), 1);
}

TEST(FlattenedMultiDimensionalArrayTest, iterators) {
  // Test with a dimensional space of (3,4): 12 elements.
  FlattenedMultiDimensionalArray<int64_t> array({3, 4});
  int64_t count = 0;
  for (int64_t element : array) {
    element++;
    count++;
  }
  EXPECT_EQ(array.GetElementCount(), count);
  // Test with zero dimensions (no dimensions).
  FlattenedMultiDimensionalArray<int64_t> zero_array({});
  count = 0;
  for (int64_t element : zero_array) {
    element++;
    count++;
  }
  EXPECT_EQ(zero_array.GetElementCount(), count);
}

TEST(FlattenedMultiDimensionalArrayTest, GetDimensions) {
  // Test with a dimensional space of (3,4): 12 elements.
  FlattenedMultiDimensionalArray<int64_t> array({3, 4});
  const DimensionBounds& dimensions = array.GetDimensions();
  EXPECT_EQ(dimensions.GetDimensionCount(), 2);
  EXPECT_EQ(dimensions.GetDimensionBound(0), 3);
  EXPECT_EQ(dimensions.GetDimensionBound(1), 4);
  // Test with zero dimensions (no dimensions).
  FlattenedMultiDimensionalArray<int64_t> zero_array({});
  const DimensionBounds& zero_dimensions = zero_array.GetDimensions();
  EXPECT_EQ(zero_dimensions.GetDimensionCount(), 0);
}

TEST(FlattenedMultiDimensionalArrayTest, SetValueCoordinate) {
  // Test setting every element of a (2,2) dimensional space referenced by
  // coordinate.
  FlattenedMultiDimensionalArray<int64_t> array({2, 2});
  for (int64_t dimension_1 = 0; dimension_1 < 2; dimension_1++) {
    for (int64_t dimension_0 = 0; dimension_0 < 2; dimension_0++) {
      array.SetValue({dimension_0, dimension_1}, 42);
    }
  }
  // Test with zero dimensions (no dimensions).
  FlattenedMultiDimensionalArray<int64_t> zero_array({});
  zero_array.SetValue({}, 42);
}

TEST(FlattenedMultiDimensionalArrayTest, SetValueIndex) {
  // Test setting every element of a (2,2) dimensional space referenced by
  // index.
  FlattenedMultiDimensionalArray<int64_t> array({2, 2});
  for (int64_t count = 0; count < 4; count++) {
    array.SetValue(count, 42);
  }
  // Test with zero dimensions (no dimensions).
  FlattenedMultiDimensionalArray<int64_t> zero_array({});
  zero_array.SetValue(0, 42);
}

TEST(FlattenedMultiDimensionalArrayTest, GetValueCoordinate) {
  // Test getting every element of a (2,2) dimensional space referenced by
  // coordinate.
  FlattenedMultiDimensionalArray<int64_t> array({2, 2});
  int value = 0;
  for (int64_t dimension_1 = 0; dimension_1 < 2; dimension_1++) {
    for (int64_t dimension_0 = 0; dimension_0 < 2; dimension_0++) {
      array.SetValue({dimension_0, dimension_1}, value++);
    }
  }
  value = 0;
  for (int64_t dimension_1 = 0; dimension_1 < 2; dimension_1++) {
    for (int64_t dimension_0 = 0; dimension_0 < 2; dimension_0++) {
      int64_t result = array.GetValue({dimension_0, dimension_1});
      EXPECT_EQ(result, value++);
    }
  }
  // Test with zero dimensions (no dimensions).
  FlattenedMultiDimensionalArray<int64_t> zero_array({});
  zero_array.SetValue({}, 42);
  int64_t result = zero_array.GetValue({});
  EXPECT_EQ(result, 42);
}

TEST(FlattenedMultiDimensionalArrayTest, GetValueIndex) {
  // Test getting every element of a (2,2) dimensional space referenced by
  // index.
  FlattenedMultiDimensionalArray<int64_t> array({2, 2});
  int value = 0;
  for (int64_t count = 0; count < 4; count++) {
    array.SetValue(count, value++);
  }
  value = 0;
  for (int64_t count = 0; count < 4; count++) {
    int64_t result = array.GetValue(count);
    EXPECT_EQ(result, value++);
  }
  // Test with zero dimensions (no dimensions).
  FlattenedMultiDimensionalArray<int64_t> zero_array({});
  zero_array.SetValue(0, 42);
  int64_t result = zero_array.GetValue(0);
  EXPECT_EQ(result, 42);
}

TEST(FlattenedMultiDimensionalArrayTest, GetCoordinateOfIndex) {
  // Test setting every element of a (2,3) dimensional space referenced by
  // coordinate.
  FlattenedMultiDimensionalArray<int64_t> array({2, 3});
  int64_t index = 0;
  for (int64_t i = 0; i < 3; i++) {
    for (int64_t j = 0; j < 2; j++) {
      Coordinate coordinate = array.GetCoordinateOfIndex(index).value();
      EXPECT_EQ(coordinate.GetDimensionCount(), 2);
      EXPECT_EQ(coordinate.GetCoordinate(0), j);
      EXPECT_EQ(coordinate.GetCoordinate(1), i);
      index++;
    }
  }
  EXPECT_EQ(array.GetCoordinateOfIndex(index), std::nullopt);
  // Test with zero dimensions (no dimensions).
  FlattenedMultiDimensionalArray<int64_t> zero_array({});
  Coordinate coordinate = zero_array.GetCoordinateOfIndex(0).value();
  EXPECT_EQ(coordinate.GetDimensionCount(), 0);
  EXPECT_EQ(zero_array.GetCoordinateOfIndex(1), std::nullopt);
}

TEST(FlattenedMultiDimensionalArrayIteratorTest, IsAtBoundsCheck) {
  // Test iterating over a (2,3) dimensional space until the iterator reaches
  // the bounds.
  using array_t = FlattenedMultiDimensionalArray<int64_t>;
  array_t array({2, 3});
  array_t::DimensionalSpaceIterator iterator =
      array.GetDimensionalSpaceIterator({0, 1});
  EXPECT_FALSE(iterator.IsAtBounds());
  int64_t count = 0;
  while (!iterator.IsAtBounds()) {
    ++iterator;
    count++;
  }
  EXPECT_EQ(count, 6);
  // Test with zero dimensions (no dimensions).
  array_t zero_array({});
  array_t::DimensionalSpaceIterator zero_iterator =
      zero_array.GetDimensionalSpaceIterator({});
  EXPECT_FALSE(zero_iterator.IsAtBounds());
  count = 0;
  while (!zero_iterator.IsAtBounds()) {
    ++zero_iterator;
    count++;
  }
  EXPECT_EQ(count, 1);
}

TEST(FlattenedMultiDimensionalArrayIteratorTest, DistanceFromEnd) {
  // Test iterating over a (2,3) dimensional space and validating the distance
  // from the end for each dimension.
  using array_t = FlattenedMultiDimensionalArray<int64_t>;
  array_t array({2, 3});
  array_t::DimensionalSpaceIterator iterator =
      array.GetDimensionalSpaceIterator({0, 1});
  for (int64_t i = 0; i < 3; i++) {
    for (int64_t j = 0; j < 2; j++) {
      EXPECT_EQ(iterator.DistanceFromEnd(0), 2 - j);
      EXPECT_EQ(iterator.DistanceFromEnd(1), 3 - i);
      ++iterator;
    }
  }
  EXPECT_EQ(iterator.DistanceFromEnd(0), 0);
  EXPECT_EQ(iterator.DistanceFromEnd(1), 0);
  /// Test with zero dimensions (no dimensions).
  array_t zero_array({});
  array_t::DimensionalSpaceIterator zero_iterator =
      zero_array.GetDimensionalSpaceIterator({});
  EXPECT_EQ(zero_iterator.DistanceFromEnd(std::nullopt), 1);
  ++zero_iterator;
  EXPECT_EQ(zero_iterator.DistanceFromEnd(std::nullopt), 0);
}

TEST(FlattenedMultiDimensionalArrayIteratorTest, GetDimensions) {
  // Test for a (2,3) dimensional space.
  using array_t = FlattenedMultiDimensionalArray<int64_t>;
  array_t array({2, 3});
  array_t::DimensionalSpaceIterator iterator =
      array.GetDimensionalSpaceIterator({0, 1});
  EXPECT_EQ(iterator.GetDimensions(), array.GetDimensions());
  // Test with zero dimensions (no dimensions).
  array_t zero_array({});
  array_t::DimensionalSpaceIterator zero_iterator =
      zero_array.GetDimensionalSpaceIterator({});
  EXPECT_EQ(zero_iterator.GetDimensions(), zero_array.GetDimensions());
}

TEST(FlattenedMultiDimensionalArrayIteratorTest, GetCoordinate) {
  // Test iterating over a (2,3) dimensional space and validating the coordinate
  // at each iteration.
  using array_t = FlattenedMultiDimensionalArray<int64_t>;
  array_t array({2, 3});
  array_t::DimensionalSpaceIterator iterator =
      array.GetDimensionalSpaceIterator({0, 1});
  for (int64_t i = 0; i < 3; i++) {
    for (int64_t j = 0; j < 2; j++) {
      EXPECT_EQ(iterator.GetCoordinate(), Coordinate({j, i}));
      ++iterator;
    }
  }
  EXPECT_EQ(iterator.GetCoordinate(), std::nullopt);
  // Test with a zero dimension.
  array_t zero_array({});
  array_t::DimensionalSpaceIterator zero_iterator =
      zero_array.GetDimensionalSpaceIterator({});
  EXPECT_EQ(zero_iterator.GetCoordinate(), Coordinate({}));
  ++zero_iterator;
  EXPECT_EQ(zero_iterator.GetCoordinate(), std::nullopt);
}

TEST(FlattenedMultiDimensionalArrayIteratorTest, EqualAndNotEqualOperator) {
  // Test iterating over a (2,3) dimensional space with two iterators and
  // validating the iterator equivalence.
  using array_t = FlattenedMultiDimensionalArray<int64_t>;
  array_t array({2, 3});
  array_t::DimensionalSpaceIterator iterator0 =
      array.GetDimensionalSpaceIterator({0, 1});
  array_t::DimensionalSpaceIterator iterator1 =
      array.GetDimensionalSpaceIterator({0, 1});
  for (; !iterator0.IsAtBounds(); ++iterator0, ++iterator1) {
    EXPECT_TRUE(iterator0 == iterator1);
    EXPECT_FALSE(iterator0 != iterator1);
  }
  EXPECT_TRUE(iterator0 == iterator1);
  EXPECT_FALSE(iterator0 != iterator1);
  // Test with a zero dimension.
  array_t zero_array({});
  array_t::DimensionalSpaceIterator zero_iterator0 =
      zero_array.GetDimensionalSpaceIterator({});
  array_t::DimensionalSpaceIterator zero_iterator1 =
      zero_array.GetDimensionalSpaceIterator({});
  for (; !iterator0.IsAtBounds(); ++zero_iterator0, ++zero_iterator1) {
    EXPECT_TRUE(zero_iterator0 == zero_iterator1);
    EXPECT_FALSE(zero_iterator0 != zero_iterator1);
  }
  EXPECT_TRUE(zero_iterator0 == zero_iterator1);
  EXPECT_FALSE(zero_iterator0 != zero_iterator1);
}

TEST(FlattenedMultiDimensionalArrayIteratorTest, PostfixAddition) {
  // Test iterating over a (2,3) dimensional space.
  using array_t = FlattenedMultiDimensionalArray<int64_t>;
  array_t array({2, 3});
  array_t::DimensionalSpaceIterator iterator =
      array.GetDimensionalSpaceIterator({0, 1});
  array_t::DimensionalSpaceIterator iterator_copy(iterator);
  array_t::DimensionalSpaceIterator iterator_postfix = iterator++;
  EXPECT_EQ(iterator_copy, iterator_postfix);
  EXPECT_NE(iterator, iterator_postfix);
  // Test with a zero dimension.
  array_t zero_array({});
  array_t::DimensionalSpaceIterator zero_iterator =
      zero_array.GetDimensionalSpaceIterator({});
  array_t::DimensionalSpaceIterator zero_iterator_copy(zero_iterator);
  array_t::DimensionalSpaceIterator zero_iterator_postfix = zero_iterator++;
  EXPECT_EQ(zero_iterator_copy, zero_iterator_postfix);
  EXPECT_NE(zero_iterator, zero_iterator_postfix);
}

TEST(FlattenedMultiDimensionalArrayIteratorTest, ElementAccess) {
  // Test modify values of a (2,3) dimensional space using an iterator.
  using array_t = FlattenedMultiDimensionalArray<int64_t>;
  array_t array({2, 3});
  array_t::DimensionalSpaceIterator iterator =
      array.GetDimensionalSpaceIterator({0, 1});
  int64_t count = 0;
  while (!iterator.IsAtBounds()) {
    *iterator++ = count++;
  }
  for (int64_t index = 0; index < 6; index++) {
    EXPECT_EQ(array.GetValue(index), index);
  }
  // Test with a zero dimension.
  array_t zero_array({});
  array_t::DimensionalSpaceIterator zero_iterator =
      zero_array.GetDimensionalSpaceIterator({});
  *zero_iterator++ = 42;
  EXPECT_EQ(zero_array.GetValue(0), 42);
}

}  // namespace
}  // namespace xls::noc
