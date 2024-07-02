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

#ifndef XLS_NOC_CONFIG_NG_COORDINATE_H_
#define XLS_NOC_CONFIG_NG_COORDINATE_H_

#include <cstdint>
#include <initializer_list>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/noc/config_ng/dimension_bounds.h"

namespace xls::noc {

// A coordinate representation in a dimensional space. It is composed of a
// coordinate value for each dimension. The dimension count and each coordinate
// value is greater than or equal to zero.
class Coordinate {
 public:
  Coordinate(std::initializer_list<int64_t> coordinates);
  Coordinate(int64_t dimension_count, int64_t value);
  absl::Span<const int64_t> GetCoordinates() const;
  int64_t GetCoordinate(int64_t dimension_index) const;
  void SetCoordinate(int64_t dimension_index, int64_t value);
  int64_t GetDimensionCount() const;

  // Returns true if the dimension count of the instance is equal to zero.
  // Otherwise, returns false.
  bool HasZeroDimensions() const;

  // Returns true if the dimension count between the instance and the
  // coordinate is equal. Otherwise, returns false.
  bool IsDimensionCountEqual(const Coordinate& coordinate) const;

  // Returns true if the dimension count between the instance and the
  // limit is equal. Otherwise, returns false.
  bool IsDimensionCountEqual(const DimensionBounds& dimensional_space) const;

  // Returns true if the coordinate has the same dimension count as the
  // dimensional space and resides in the dimensional space described by
  // dimension. Otherwise, returns false.
  bool IsInDimensionalSpace(const DimensionBounds& dimensional_space) const;

  // Returns a bit map corresponding to the dimension indices that differ
  // between the instance and another coordinate. A dimension index that differs
  // between the instance and another coordinate signifies that the location of
  // the instance and another coordinate differ along a dimension. For example,
  // coordinates [0, 2, 0] and [1, 2, 0], returns [true, false, false]. If the
  // number of dimension differ between the instance and the other coordinate,
  // nullopt is returned.
  std::optional<InlineBitmap> GetDifferentDimensionLocationsWith(
      const Coordinate& coordinate) const;

  // Returns the sum of dimension indices that differ between the instance and
  // another coordinate. A dimension index that differs between the instance and
  // another coordinate signifies that the location of the instance and the
  // other coordinate differ along a dimension. For example, coordinates [0, 2,
  // 0] and [1, 2, 0], returns 1. If the number of dimension differ between the
  // instance and the other coordinate, nullopt is returned.
  std::optional<int64_t> GetNumDifferentDimensionLocationsWith(
      const Coordinate& coordinate) const;

  // Returns the index of the dimension that differs between the instance and
  // another coordinate if one dimension index differs between the
  // instance and the other coordinate. If one dimension index does not differ
  // between the instance and the other coordinate or if the dimension count
  // differs between the instance and the other coordinate, nullopt is returned.
  std::optional<int64_t> GetUniqueDifferentDimensionIndex(
      const Coordinate& coordinate) const;

  // Returns true if the coordinates are equal. Otherwise, returns false.
  // Equal coordinates have an equal number of dimensions and values for their
  // dimensions.
  bool operator==(const Coordinate& rhs) const;

  // Returns the negation of the equal operator (==).
  bool operator!=(const Coordinate& rhs) const;

 private:
  // The order of the dimensions is in ascending order of the indices in the
  // list. Index 0 of the list is the first dimension (lowest dimension).
  std::vector<int64_t> coordinate_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_COORDINATE_H_
