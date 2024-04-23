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
#include <initializer_list>

#include "absl/log/check.h"
#include "xls/ir/bits.h"

namespace xls::noc {

DimensionBounds::DimensionBounds(std::initializer_list<int64_t> bounds) {
  for (int64_t bound : bounds) {
    CHECK_GE(bound, 0);
  }
  dimension_bounds_ = bounds;
}

DimensionBounds::DimensionBounds(const int64_t dimension_count,
                                 const int64_t value)
    : dimension_bounds_(internal::CheckGe(dimension_count, int64_t{0}),
                        internal::CheckGe(value, int64_t{0})) {}

absl::Span<const int64_t> DimensionBounds::GetDimensionBounds() const {
  return dimension_bounds_;
}

int64_t DimensionBounds::GetDimensionBound(
    const int64_t dimension_index) const {
  CHECK(dimension_index >= 0 && dimension_index < GetDimensionCount());
  return dimension_bounds_[dimension_index];
}

void DimensionBounds::SetDimensionBound(const int64_t dimension_index,
                                        const int64_t value) {
  CHECK_GE(value, 0);
  CHECK(dimension_index >= 0 && dimension_index < GetDimensionCount());
  dimension_bounds_[dimension_index] = value;
}

int64_t DimensionBounds::GetDimensionCount() const {
  return dimension_bounds_.size();
}

bool DimensionBounds::IsDimensionCountEqual(
    const DimensionBounds& dimension) const {
  return GetDimensionCount() == dimension.GetDimensionCount();
}

bool DimensionBounds::HasZeroDimensions() const {
  return GetDimensionCount() == 0;
}

bool DimensionBounds::operator==(const DimensionBounds& rhs) const {
  bool equal =
      IsDimensionCountEqual(rhs) && dimension_bounds_ == rhs.dimension_bounds_;
  return equal;
}

bool DimensionBounds::operator!=(const DimensionBounds& rhs) const {
  return !(*this == rhs);
}

}  // namespace xls::noc
