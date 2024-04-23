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
#include <initializer_list>
#include <optional>

#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"

namespace xls::noc {

Coordinate::Coordinate(std::initializer_list<int64_t> coordinates) {
  for (int64_t coordinate : coordinates) {
    CHECK_GE(coordinate, 0);
  }
  coordinate_ = coordinates;
}

Coordinate::Coordinate(const int64_t dimension_count, const int64_t value)
    : coordinate_(internal::CheckGe(dimension_count, int64_t{0}),
                  internal::CheckGe(value, int64_t{0})) {}

absl::Span<const int64_t> Coordinate::GetCoordinates() const {
  return coordinate_;
}

int64_t Coordinate::GetCoordinate(const int64_t dimension_index) const {
  CHECK(dimension_index >= 0 && dimension_index < GetDimensionCount());
  return coordinate_[dimension_index];
}

void Coordinate::SetCoordinate(const int64_t dimension_index,
                               const int64_t value) {
  CHECK_GE(value, 0);
  CHECK(dimension_index >= 0 && dimension_index < GetDimensionCount());
  coordinate_[dimension_index] = value;
}

int64_t Coordinate::GetDimensionCount() const { return coordinate_.size(); }

bool Coordinate::HasZeroDimensions() const { return GetDimensionCount() == 0; }

bool Coordinate::IsDimensionCountEqual(const Coordinate& coordinate) const {
  return GetDimensionCount() == coordinate.GetDimensionCount();
}

bool Coordinate::IsDimensionCountEqual(
    const DimensionBounds& dimensional_space) const {
  return GetDimensionCount() == dimensional_space.GetDimensionCount();
}

bool Coordinate::IsInDimensionalSpace(
    const DimensionBounds& dimensional_space) const {
  if (IsDimensionCountEqual(dimensional_space)) {
    for (int64_t count = 0; count < dimensional_space.GetDimensionCount();
         count++) {
      int64_t coordinate_value = GetCoordinate(count);
      int64_t dimension_value = dimensional_space.GetDimensionBound(count);
      if (coordinate_value < 0 || coordinate_value >= dimension_value) {
        return false;
      }
    }
    return true;
  }
  return false;
}

std::optional<InlineBitmap> Coordinate::GetDifferentDimensionLocationsWith(
    const Coordinate& coordinate) const {
  if (IsDimensionCountEqual(coordinate)) {
    InlineBitmap rtn(GetDimensionCount());
    for (int64_t index = 0; index < rtn.bit_count(); index++) {
      if (GetCoordinate(index) != coordinate.GetCoordinate(index)) {
        rtn.Set(index, true);
      }
    }
    return rtn;
  }
  return std::nullopt;
}

std::optional<int64_t> Coordinate::GetNumDifferentDimensionLocationsWith(
    const Coordinate& coordinate) const {
  std::optional<InlineBitmap> result =
      GetDifferentDimensionLocationsWith(coordinate);
  if (result.has_value()) {
    InlineBitmap& dimensions = result.value();
    int64_t number_differences = 0;
    for (int64_t index = 0; index < dimensions.bit_count(); index++) {
      number_differences += dimensions.Get(index) ? 1 : 0;
    }
    return number_differences;
  }
  return std::nullopt;
}

std::optional<int64_t> Coordinate::GetUniqueDifferentDimensionIndex(
    const Coordinate& coordinate) const {
  int64_t dimension_index = -1;
  std::optional<InlineBitmap> result =
      GetDifferentDimensionLocationsWith(coordinate);
  if (result.has_value()) {
    InlineBitmap& dimensions = result.value();
    for (int64_t index = 0; index < dimensions.bit_count(); index++) {
      if (dimensions.Get(index)) {
        if (dimension_index >= 0) {
          return std::nullopt;
        }
        dimension_index = index;
      }
    }
  }
  return dimension_index == -1 ? std::nullopt
                               : std::optional(dimension_index);
}

bool Coordinate::operator==(const Coordinate& rhs) const {
  bool equal = IsDimensionCountEqual(rhs) && coordinate_ == rhs.coordinate_;
  return equal;
}

bool Coordinate::operator!=(const Coordinate& rhs) const {
  return !(*this == rhs);
}

}  // namespace xls::noc
