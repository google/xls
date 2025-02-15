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

#include "xls/dslx/exhaustiveness/nd_region.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/dslx/exhaustiveness/interp_value_interval.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

// Returns the tightest lower bound for the two given intervals.
InterpValue TightestLowerBound(const InterpValueInterval& a,
                               const InterpValueInterval& b) {
  return a.min().Max(b.min()).value();
}

// Returns the tightest upper bound for the two given intervals.
InterpValue TightestUpperBound(const InterpValueInterval& a,
                               const InterpValueInterval& b) {
  return a.max().Min(b.max()).value();
}

}  // namespace

/* static */ NdInterval NdInterval::MakePoint(
    absl::Span<const InterpValue> point) {
  std::vector<InterpValueInterval> dims;
  for (const InterpValue& point_value : point) {
    dims.push_back(InterpValueInterval(point_value, point_value));
  }
  return NdInterval(dims);
}

/* static */ NdInterval NdInterval::MakeContiguous(
    absl::Span<const InterpValue> start_point,
    absl::Span<const InterpValue> end_point) {
  std::vector<InterpValueInterval> dims;
  dims.reserve(start_point.size());
  for (int64_t i = 0; i < start_point.size(); ++i) {
    dims.push_back(InterpValueInterval(start_point[i], end_point[i]));
  }
  return NdInterval(dims);
}

std::optional<NdInterval> NdIntervalWithEmpty::ToNonEmpty() const {
  std::vector<InterpValueInterval> non_empty_dims;
  for (const auto& dim : dims_) {
    if (dim.has_value()) {
      non_empty_dims.push_back(dim.value());
    } else {
      return std::nullopt;
    }
  }
  return NdInterval(std::move(non_empty_dims));
}

std::string NdIntervalWithEmpty::ToString(bool show_types) const {
  std::vector<std::string> dims_str;
  for (const std::optional<InterpValueInterval>& dim : dims_) {
    if (dim.has_value()) {
      dims_str.push_back(dim.value().ToString(show_types));
    } else {
      dims_str.push_back("<empty>");
    }
  }
  return absl::StrCat("[", absl::StrJoin(dims_str, ", "), "]");
}

NdInterval::NdInterval(std::vector<InterpValueInterval> dims)
    : dims_(std::move(dims)) {}

bool NdInterval::Covers(const NdInterval& other) const {
  for (int64_t i = 0; i < dims_.size(); ++i) {
    if (!dims_[i].Covers(other.dims_[i])) {
      return false;
    }
  }
  return true;
}

bool NdInterval::Intersects(const NdInterval& other) const {
  CHECK_EQ(dims_.size(), other.dims_.size())
      << "Cannot intersect intervals with different numbers of dimensions: "
      << ToString(/*show_types=*/false) << " and "
      << other.ToString(/*show_types=*/false);
  for (int64_t i = 0; i < dims_.size(); ++i) {
    if (!dims_[i].Intersects(other.dims_[i])) {
      return false;
    }
  }
  return true;
}

std::vector<NdInterval> NdInterval::SubtractInterval(
    const NdIntervalWithEmpty& other_with_empty) const {
  std::optional<NdInterval> non_empty = other_with_empty.ToNonEmpty();
  // If there's zero volume, subtraction is trivial.
  if (!non_empty.has_value()) {
    return {*this};
  }
  return SubtractInterval(non_empty.value());
}

std::vector<NdInterval> NdInterval::SubtractInterval(
    const NdInterval& other) const {
  CHECK_EQ(dims_.size(), other.dims_.size())
      << "Cannot subtract intervals with different numbers of dimensions: "
      << ToString(/*show_types=*/false) << " and "
      << other.ToString(/*show_types=*/false);
  if (!other.Intersects(*this)) {
    return {*this};
  }
  if (other.Covers(*this)) {
    return {};
  }

  VLOG(5) << "SubtractInterval; subtracting: "
          << other.ToString(/*show_types=*/false)
          << " from: " << ToString(/*show_types=*/false);

  std::vector<NdInterval> pieces;

  // We'll work with a copy of our interval dimensions,
  // "remaining" will be gradually shrunk to the (to-be-removed) intersection.
  std::vector<InterpValueInterval> remaining = dims_;
  for (int64_t dim_idx = 0; dim_idx < remaining.size(); ++dim_idx) {
    const InterpValueInterval& remaining_interval = remaining[dim_idx];

    // Compute the intersection in dimension `dim_idx`:
    // `lower` is the maximum of our lower bound and the other's lower bound.
    // `upper` is the minimum of our upper bound and the other's upper bound.
    InterpValue lower =
        TightestLowerBound(remaining[dim_idx], other.dims_[dim_idx]);
    InterpValue upper =
        TightestUpperBound(remaining[dim_idx], other.dims_[dim_idx]);

    // If there is a gap on the lower side, "peel off" that slice.
    if (remaining_interval.min() < lower) {
      VLOG(5) << absl::StreamFormat(
          "remaining lower for dimension %d is %s which is < %s", dim_idx,
          remaining_interval.min().ToString(/*show_types=*/true),
          lower.ToString(/*show_types=*/true));
      std::vector<InterpValueInterval> new_dims = remaining;
      new_dims[dim_idx] = InterpValueInterval(remaining_interval.min(),
                                              lower.Decrement().value());
      pieces.push_back(NdInterval(new_dims));

      // Update the remaining interval so that it now starts at L.
      remaining[dim_idx] = InterpValueInterval(lower, remaining_interval.max());
    }

    // If there is a gap on the upper side, "peel off" that slice.
    if (upper < remaining_interval.max()) {
      std::vector<InterpValueInterval> new_dims = remaining;
      new_dims[dim_idx] = InterpValueInterval(upper.Increment().value(),
                                              remaining_interval.max());
      pieces.push_back(NdInterval(new_dims));

      // Update the remaining interval so that it now ends at U.
      remaining[dim_idx] = InterpValueInterval(remaining_interval.min(), upper);
    }
  }

  VLOG(5) << "Resulting pieces: "
          << absl::StrJoin(pieces, ", ",
                           [&](std::string* out, const NdInterval& interval) {
                             absl::StrAppend(
                                 out, interval.ToString(/*show_types=*/false));
                           });
  return pieces;
}

std::string NdInterval::ToString(bool show_types) const {
  std::vector<std::string> lower_components;
  std::vector<std::string> upper_components;
  for (const auto& dim : dims_) {
    // Assume that each 'dim' (of type InterpValueInterval) exposes:
    //   - a method lower() that returns the lower bound (an InterpValue)
    //   - a method upper() that returns the upper bound (an InterpValue)
    // and that InterpValue has a ToString(bool) method.
    lower_components.push_back(dim.min().ToString(/*humanize=*/!show_types));
    upper_components.push_back(dim.max().ToString(/*humanize=*/!show_types));
  }
  // Build lower and upper corner strings.
  std::string lower_str =
      absl::StrCat("[", absl::StrJoin(lower_components, ", "), "]");
  std::string upper_str =
      absl::StrCat("[", absl::StrJoin(upper_components, ", "), "]");
  // Final string: a two-element list [lower_corner, upper_corner].
  return absl::StrCat("[", lower_str, ", ", upper_str, "]");
}

std::string NdRegion::ToString(bool show_types) const {
  std::string guts = absl::StrJoin(
      disjoint_, ", ", [&](std::string* out, const NdInterval& interval) {
        absl::StrAppend(out, interval.ToString(show_types));
      });
  return absl::StrCat("{", guts, "}");
}

ABSL_MUST_USE_RESULT NdRegion
NdRegion::SubtractInterval(const NdIntervalWithEmpty& other) const {
  std::vector<NdInterval> new_disjoint;
  for (const NdInterval& old_interval : disjoint_) {
    std::vector<NdInterval> new_interval = old_interval.SubtractInterval(other);
    new_disjoint.insert(new_disjoint.end(), new_interval.begin(),
                        new_interval.end());
    VLOG(5) << "SubtractInverval; subtracting "
            << other.ToString(/*show_types=*/false) << " from old interval "
            << old_interval.ToString(/*show_types=*/false) << " produced "
            << absl::StrJoin(new_interval, ", ",
                             [&](std::string* out, const NdInterval& interval) {
                               absl::StrAppend(out, interval.ToString(
                                                        /*show_types=*/false));
                             });
  }
  NdRegion result(dim_extents_, new_disjoint);
  VLOG(5) << "SubtractInterval; region-level result; subtracting "
          << other.ToString(/*show_types=*/false) << " from "
          << ToString(/*show_types=*/false) << " produced "
          << result.ToString(/*show_types=*/false);
  return result;
}

}  // namespace xls::dslx
