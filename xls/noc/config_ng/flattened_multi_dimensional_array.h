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

#ifndef XLS_NOC_CONFIG_NG_FLATTENED_MULTI_DIMENSIONAL_ARRAY_H_
#define XLS_NOC_CONFIG_NG_FLATTENED_MULTI_DIMENSIONAL_ARRAY_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/noc/config_ng/coordinate.h"
#include "xls/noc/config_ng/dimension_bounds.h"

namespace xls::noc {

// A flattened representation of a multi dimensional array.
template <typename T>
class FlattenedMultiDimensionalArray {
 public:
  // A flattened multi-dimensional array.
  // dimensions: the dimensions of the array. The array is stored hierarchically
  // with the highest dimension first (highest dimension major). The index of an
  // entry represents the dimension index. The value of the entry represents the
  // number of elements in the dimension.
  // A [2, 3] dimension is flattened to:
  // [0,0], [0,1], [1,0], [1,1], [2,0], [2,1].
  //
  // A zero dimension is treated as a point. For a zero dimension, the
  // coordinate does not have coordinate values, and the index of the element is
  // zero.
  explicit FlattenedMultiDimensionalArray(const DimensionBounds& dimensions)
      : dimensions_offset_(dimensions.GetDimensionCount()),
        dimensions_(dimensions.GetDimensionCount(), 0) {
    int64_t size = 1;
    for (int64_t index = 0; index < dimensions.GetDimensionCount(); index++) {
      dimensions_offset_[index] = size;
      dimensions_.SetDimensionBound(index, dimensions.GetDimensionBound(index));
      size *= dimensions_.GetDimensionBound(index);
    }
    elements_.resize(size);
  }

  // Returns the element at the coordinate. The coordinate must be valid.
  const T& GetValue(const Coordinate& coordinate) const {
    return GetValueByCoordinate(coordinate);
  }
  T& GetValue(const Coordinate& coordinate) {
    // const_cast required for mutable instance.
    return const_cast<T&>(GetValueByCoordinate(coordinate));
  }

  // Returns the element at the index of the flattened array. The index must be
  // valid.
  const T& GetValue(int64_t index) const { return GetValueByIndex(index); }
  T& GetValue(int64_t index) {
    // const_cast required for mutable instance.
    return const_cast<T&>(GetValueByIndex(index));
  }

  // Set the entry at the coordinate to the element. The coordinate must be
  // valid.
  void SetValue(const Coordinate& coordinate, const T& element) {
    if (dimensions_.HasZeroDimensions()) {
      CHECK(coordinate.HasZeroDimensions());
      elements_[0] = element;
    }
    std::optional<int64_t> index = GetIndexOfCoordinate(coordinate);
    CHECK(index.has_value()) << "The coordinate did not resolve to an index";
    elements_[index.value()] = element;
  }

  // Set the entry at the index of the flattened array to the element. The index
  // must be valid.
  void SetValue(int64_t index, const T& element) {
    CHECK(IsIndexValid(index));
    elements_[index] = element;
  }

  // Gets the coordinate from an index in the flattened array. If the index is
  // not valid, returns an nullopt.
  std::optional<Coordinate> GetCoordinateOfIndex(int64_t index) const {
    if (!IsIndexValid(index)) {
      return std::nullopt;
    }
    if (dimensions_.HasZeroDimensions()) {
      return Coordinate({});
    }
    Coordinate coordinate(dimensions_.GetDimensionCount(), 0);
    for (int64_t count = dimensions_offset_.size() - 1; count >= 0 && index > 0;
         count--) {
      coordinate.SetCoordinate(count, index / dimensions_offset_[count]);
      index = index % dimensions_offset_[count];
    }
    return coordinate;
  }

  // Get the number of elements in the array.
  int64_t GetElementCount() const { return elements_.size(); }

  // Get the dimensions in the array.
  const DimensionBounds& GetDimensions() const { return dimensions_; }

  // Expose elements through iterators.
  typename std::vector<T>::iterator begin() { return elements_.begin(); }
  typename std::vector<T>::iterator end() { return elements_.end(); }
  typename std::vector<T>::const_iterator begin() const {
    return elements_.begin();
  }
  typename std::vector<T>::const_iterator end() const {
    return elements_.end();
  }

  // A Dimensional Space Iterator traverses a dimensional space defined by a
  // dimension bounds.
  class DimensionalSpaceIterator {
   public:
    // Returns true if the instance is at the dimensional bounds (end).
    bool IsAtBounds() const {
      if (coordinate_.HasZeroDimensions()) {
        return traversed_zero_dimension_;
      }
      bool at_end = true;
      // TODO (vmirian) 04-05-21 should be sufficient to check a single
      // dimension.
      for (int64_t count = 0; count < coordinate_.GetDimensionCount();
           count++) {
        at_end = at_end && coordinate_.GetCoordinate(count) ==
                               dimensions_.GetDimensionBound(count);
      }
      return at_end;
    }

    // Returns the distance from the end for a dimension_index.
    // dimension_index: the index of the dimension. For a non-zero dimensional
    // space, the value must be equal or greater to zero and less than the
    // number of dimensions. For a zero dimension, the dimension index is
    // nullopt. The result for a zero dimension is 0 when the instance is at the
    // dimensional bounds, otherwise the result is 1.
    int64_t DistanceFromEnd(std::optional<int64_t> dimension_index) const {
      if (coordinate_.HasZeroDimensions()) {
        CHECK(dimension_index == std::nullopt)
            << "For a zero-dimension, the dimension index must be nullopt.";
        return traversed_zero_dimension_ ? 0 : 1;
      }
      CHECK(dimension_index.has_value())
          << "Dimension index must have a value.";
      int64_t dimension_index_value = dimension_index.value();
      CHECK_GE(dimension_index_value, 0)
          << "Dimension index must be greater than zero.";
      CHECK_LT(dimension_index_value, dimensions_.GetDimensionCount())
          << "Dimension index is out of range: received "
          << dimension_index_value << ", maximum value is "
          << dimensions_.GetDimensionCount() << ".";
      return dimensions_.GetDimensionBound(dimension_index_value) -
             coordinate_.GetCoordinate(dimension_index_value);
    }

    const DimensionBounds& GetDimensions() const { return dimensions_; }

    // Returns the coordinate value of the iterator. If the iterator is outside
    // of the dimensional space, std::nullopt is returned.
    std::optional<Coordinate> GetCoordinate() const {
      if (IsAtBounds()) {
        return std::nullopt;
      }
      return coordinate_;
    }

    // Returns true if dimension bounds are equal. Otherwise, returns false.
    // Equal dimension bounds have an equal number of dimensions and values for
    // their dimensions.
    bool operator==(const DimensionalSpaceIterator& rhs) const {
      return dimensions_ == rhs.dimensions_ && coordinate_ == rhs.coordinate_ &&
             dimension_index_order_ == rhs.dimension_index_order_ &&
             traversed_zero_dimension_ == rhs.traversed_zero_dimension_;
    }

    // Returns the negation of the equal operator (==).
    bool operator!=(const DimensionalSpaceIterator& rhs) const {
      return !((*this) == rhs);
    }

    // Prefix increment operator.
    DimensionalSpaceIterator& operator++() {
      AdvanceByOne();
      return *this;
    }
    // Postfix increment operator.
    DimensionalSpaceIterator operator++(int) {
      DimensionalSpaceIterator temp(*this);
      AdvanceByOne();
      return temp;
    }

    // Operator*. If the iterator is outside of the dimensional space, a runtime
    // error is thrown.
    const T& operator*() const { return data_.GetValue(coordinate_); }
    T& operator*() { return data_.GetValue(coordinate_); }

   private:
    // data: the flattened multi dimensional array to iterate over. Does not
    // take ownership of the data instance. The data must refer to a valid
    // object that outlives this object.
    // space: the dimensional space.
    // dimension_index_order: the order of the dimension indices to traverse
    // starting from index zero.
    // Requires the size of space and dimension_index_order are equivalent.
    // Requires the values of the dimension_index_order are sequential starting
    // from zero.
    friend class FlattenedMultiDimensionalArray;
    DimensionalSpaceIterator(FlattenedMultiDimensionalArray<T>& data,
                             const DimensionBounds& space,
                             absl::Span<const int64_t> dimension_index_order)
        : dimensions_(space),
          coordinate_(space.GetDimensionCount(), 0),
          dimension_index_order_(dimension_index_order.size(), 0),
          data_(data),
          traversed_zero_dimension_(false) {
      QCHECK_EQ(space.GetDimensionCount(), dimension_index_order.size())
          << "The size of space and dimension_index_order must be equivalent.";
      int64_t total = 0;
      int64_t total_golden = 0;
      for (int64_t count = 0; count < dimension_index_order.size(); count++) {
        // TODO (vmirian) 04-13-21 Add common function to enable the xor
        // sequence evaluation
        total = total ^ dimension_index_order[count];
        total_golden = total_golden ^ count;
      }
      QCHECK_EQ(total, total_golden)
          << "The values of the dimension_index_order must be sequential "
             "starting from zero.";
      std::copy(dimension_index_order.begin(), dimension_index_order.end(),
                dimension_index_order_.begin());
    }

    // Advances by one.
    void AdvanceByOne() {
      if (coordinate_.HasZeroDimensions()) {
        traversed_zero_dimension_ = true;
        return;
      }
      // Already at the end. Do not increment.
      if (IsAtBounds()) {
        return;
      }
      bool carry = false;
      for (int64_t count : dimension_index_order_) {
        coordinate_.SetCoordinate(count, coordinate_.GetCoordinate(count) + 1);
        carry = coordinate_.GetCoordinate(count) ==
                dimensions_.GetDimensionBound(count);
        if (carry) {
          coordinate_.SetCoordinate(count, 0);
        } else {
          break;
        }
      }
      if (carry) {
        for (int index = 0; index < dimensions_.GetDimensionCount(); index++) {
          coordinate_.SetCoordinate(index,
                                    dimensions_.GetDimensionBound(index));
        }
      }
    }

    DimensionBounds dimensions_;
    Coordinate coordinate_;
    absl::InlinedVector<int64_t, 16> dimension_index_order_;
    FlattenedMultiDimensionalArray<T>& data_;
    bool traversed_zero_dimension_;
  };

  DimensionalSpaceIterator GetDimensionalSpaceIterator(
      absl::Span<const int64_t> dimension_index_order) {
    return DimensionalSpaceIterator(*this, dimensions_, dimension_index_order);
  }

 private:
  const T& GetValueByCoordinate(const Coordinate& coordinate) const {
    if (dimensions_.HasZeroDimensions()) {
      CHECK(coordinate.HasZeroDimensions());
      return elements_[0];
    }
    std::optional<int64_t> index = GetIndexOfCoordinate(coordinate);
    CHECK(index.has_value());
    return elements_[index.value()];
  }

  const T& GetValueByIndex(int64_t index) const {
    CHECK(IsIndexValid(index));
    return elements_[index];
  }

  // Returns true if the index is within range. Otherwise, returns false.
  bool IsIndexValid(int64_t index) const {
    return (index >= 0 && index < elements_.size());
  }

  // Returns true if the coordinate is valid. Otherwise, returns false.
  // For a one+ dimensions, the coordinate must be in the dimensional space.
  // For a zero dimension, the coordinate does not have coordinate values.
  bool IsCoordinateValid(const Coordinate& coordinate) const {
    if (dimensions_.HasZeroDimensions()) {
      return coordinate.HasZeroDimensions();
    }
    return coordinate.IsInDimensionalSpace(dimensions_);
  }

  // Gets the index of the coordinate in the flattened array. If the coordinate
  // is not valid, returns an nullopt.
  std::optional<int64_t> GetIndexOfCoordinate(
      const Coordinate& coordinate) const {
    if (!IsCoordinateValid(coordinate)) {
      return std::nullopt;
    }
    if (dimensions_.HasZeroDimensions()) {
      return 0;
    }
    int64_t index = 0;
    for (int64_t count = 0; count < dimensions_offset_.size(); count++) {
      index += coordinate.GetCoordinate(count) * dimensions_offset_[count];
    }
    return index;
  }

  // For the use cases predicted for this class, the number of dimensions will
  // most likely not exceed eight. Enabling an efficient implementation for an
  // eight dimension instance.
  absl::InlinedVector<int64_t, 8> dimensions_offset_;
  DimensionBounds dimensions_;
  std::vector<T> elements_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_FLATTENED_MULTI_DIMENSIONAL_ARRAY_H_
