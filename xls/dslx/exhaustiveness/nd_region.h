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
#ifndef XLS_DSLX_EXHAUSTIVENESS_ND_REGION_H_
#define XLS_DSLX_EXHAUSTIVENESS_ND_REGION_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/types/span.h"
#include "xls/dslx/exhaustiveness/interp_value_interval.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {

class NdIntervalWithEmpty;

// Represents a contiguous interval in n-dimensional space.
//
// This is also called a "hyper-rectangle" because it's a contiguous rectangular
// space in >= 1D.
//
// It is lower bounded (inclusive) by the minimum values of each dimension and
// upper bounded (inclusive) by the maximum values of each dimension. We prefer
// inclusive bounds so we can easily represent the maximum value in a given type
// without worrying about whether the exclusive limit requires an extra bit.
class NdInterval {
 public:
  // Note: we do not provide a `MakeFull` factory on NdInterval because
  // we expect that the upper layers will determine the full range for types
  // that are sparse within some bit range, like enums. We don't want to provide
  // a method that could be used to accidentally create an over-wide range for a
  // type such as an enum because we think we need to create a full range for
  // its underlying bit space. (i.e. imagine an enum with 3 values in a u2
  // space).

  static NdInterval MakePoint(absl::Span<const InterpValue> point);

  // Note that end_point is included.
  static NdInterval MakeContiguous(absl::Span<const InterpValue> start_point,
                                   absl::Span<const InterpValue> end_point);

  explicit NdInterval(std::vector<InterpValueInterval> dims);

  bool Covers(const NdInterval& other) const;

  bool Intersects(const NdInterval& other) const;

  // Subtracts `other` from this contiguous n-dimensional interval space.
  // This works by "peeling off" slices along each dimension where
  // our interval extends beyond the subtracting interval.
  // (For example, in 1D subtracting [3,6] from [0,9] produces [0,2] on the left
  // and [7,9] on the right.)
  std::vector<NdInterval> SubtractInterval(
      const NdIntervalWithEmpty& other) const;
  std::vector<NdInterval> SubtractInterval(const NdInterval& other) const;

  std::string ToString(bool show_types) const;

  // Note: not safe to hold across a mutating operation.
  absl::Span<const InterpValueInterval> dims() const { return dims_; }

 private:
  std::vector<InterpValueInterval> dims_;
};

// Variation on NdInterval above that allows any given dimension to have an
// interval with no volume.
//
// This is useful because ranges can be zero volume or enums can have zero
// definitions.
class NdIntervalWithEmpty {
 public:
  explicit NdIntervalWithEmpty(
      std::vector<std::optional<InterpValueInterval>> dims)
      : dims_(std::move(dims)) {}

  std::optional<NdInterval> ToNonEmpty() const;

  std::string ToString(bool show_types) const;

 private:
  std::vector<std::optional<InterpValueInterval>> dims_;
};

class NdRegion {
 public:
  // Note: prefer this function when creating a "Full" region. For more sparse
  // types such as enums the full range is not the same as the underlying bit
  // space.
  static NdRegion MakeFromNdInterval(const NdInterval& interval,
                                     std::vector<InterpValue> dim_extents) {
    return NdRegion(std::move(dim_extents), {interval});
  }

  static NdRegion MakeEmpty(std::vector<InterpValue> dim_extents) {
    NdRegion region(std::move(dim_extents), {});
    return region;
  }

  ABSL_MUST_USE_RESULT NdRegion
  SubtractInterval(const NdIntervalWithEmpty& other) const;

  bool IsEmpty() const { return disjoint_.empty(); }

  std::string ToString(bool show_types) const;

  // Note: not safe to hold across a mutating operation.
  absl::Span<const NdInterval> disjoint() const { return disjoint_; }

 private:
  explicit NdRegion(std::vector<InterpValue> dim_extents,
                    std::vector<NdInterval> disjoint)
      : dim_extents_(std::move(dim_extents)), disjoint_(std::move(disjoint)) {}

  // The extents (i.e. limits) in all the dimensions of the regions.
  std::vector<InterpValue> dim_extents_;

  // Disjoint intervals that describe what is filled in within the region.
  std::vector<NdInterval> disjoint_;
};

}  // namespace xls::dslx

#endif  // XLS_DSLX_EXHAUSTIVENESS_ND_REGION_H_
