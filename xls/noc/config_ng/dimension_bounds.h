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

#ifndef XLS_NOC_CONFIG_NG_DIMENSION_BOUNDS_H_
#define XLS_NOC_CONFIG_NG_DIMENSION_BOUNDS_H_

#include <cstdint>
#include <initializer_list>
#include <vector>

#include "absl/types/span.h"

namespace xls::noc {

// The dimension bounds of a dimensional space. It is composed of a
// dimension-bound value for each dimension. The dimension count and each
// dimension-bound value is greater than or equal to zero.
class DimensionBounds {
 public:
  DimensionBounds(std::initializer_list<int64_t> dimension_bounds);
  DimensionBounds(int64_t dimension_count, int64_t value);
  absl::Span<const int64_t> GetDimensionBounds() const;
  int64_t GetDimensionBound(int64_t dimension_index) const;
  void SetDimensionBound(int64_t dimension_index, int64_t value);
  int64_t GetDimensionCount() const;

  // Returns true if the dimension count between the instance and the
  // dimension is equal. Otherwise, returns false.
  bool IsDimensionCountEqual(const DimensionBounds& dimension) const;

  // Returns true if the dimension count of the instance is equal to zero.
  // Otherwise, returns false.
  bool HasZeroDimensions() const;

  // Returns true if dimension bounds are equal. Otherwise, returns false.
  // Equal dimension bounds have an equal number of dimensions and values for
  // their dimensions.
  bool operator==(const DimensionBounds& rhs) const;

  // Returns the negation of the equal operator (==).
  bool operator!=(const DimensionBounds& rhs) const;

 private:
  // The order of the dimensions is in ascending order of the indices in the
  // list. Index 0 of the list is the first dimension (lowest dimension).
  std::vector<int64_t> dimension_bounds_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_NG_DIMENSION_BOUNDS_H_
