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

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/types/span.h"
#include "xls/dslx/exhaustiveness/interp_value_interval.h"
#include "xls/dslx/interp_value.h"

namespace xls::dslx {
namespace {

// Helper for making a full NdRegion according to a full bit value space, with
// each dimension going [0, dim_extent] (inclusive).
//
// We don't use this in the production code because we don't want to make
// it easy to accidentally misinterpret types like enum which may not populate
// the full underlying bit space but rather some sparse subset of it.
NdInterval MakeFullNdInterval(absl::Span<const InterpValue> dim_extents) {
  std::vector<InterpValueInterval> intervals;
  for (const InterpValue& dim_extent : dim_extents) {
    bool is_signed = dim_extent.IsSigned();
    int64_t bit_count = dim_extent.GetBitCount().value();
    InterpValue min = InterpValue::MakeZeroValue(is_signed, bit_count);
    InterpValue max = dim_extent;
    intervals.push_back(InterpValueInterval(min, max));
  }
  return NdInterval(std::move(intervals));
}

}  // namespace

TEST(NdRegionTest, SimpleRegionIntersectsSelf) {
  std::vector<InterpValue> dim_extents = {
      InterpValue::MakeUBits(1, 1),
      InterpValue::MakeUBits(1, 1),
  };
  const NdInterval u1_full = MakeFullNdInterval(dim_extents);
  EXPECT_TRUE(u1_full.Intersects(u1_full));
  EXPECT_EQ(u1_full.ToString(/*show_types=*/false), "[[0, 0], [1, 1]]");
}

TEST(NdRegionTest, Rectangle2DSubtractCorner) {
  std::vector<InterpValue> dim_extents = {
      InterpValue::MakeUBits(1, 1),
      InterpValue::MakeUBits(1, 1),
  };
  const NdInterval u1_full = MakeFullNdInterval(dim_extents);
  const NdInterval u1_corner =
      NdInterval::MakePoint({dim_extents[0], dim_extents[1]});
  std::vector<NdInterval> pieces = u1_full.SubtractInterval(u1_corner);
  EXPECT_EQ(pieces.size(), 2);
  // original space is:
  // [0,0] [0,1] [1,0] [1,1]
  // we subtracted the last one, so the resulting space is:
  // [0,0] [0,1] [1,0]
  // which can be represented as [[0,0]..[0,1]] union [[1,0]..[1,0]]
  EXPECT_EQ(pieces[0].ToString(/*show_types=*/false), "[[0, 0], [0, 1]]");
  EXPECT_EQ(pieces[1].ToString(/*show_types=*/false), "[[1, 0], [1, 0]]");
}

TEST(NdRegionTest, Rectangle2DSubtractAllWhereXEquals1) {
  std::vector<InterpValue> dim_extents = {
      InterpValue::MakeUBits(1, 1),
      InterpValue::MakeUBits(1, 1),
  };
  const NdInterval u1_full = MakeFullNdInterval(dim_extents);
  EXPECT_EQ(u1_full.ToString(/*show_types=*/false), "[[0, 0], [1, 1]]");

  // Make a slice that represents all the points where x = 1.
  const NdInterval u1_x1 = NdInterval::MakeContiguous(
      {
          InterpValue::MakeUBits(1, 1),  // x = 1
          InterpValue::MakeUBits(1, 0),  // y = 0
      },
      dim_extents);
  EXPECT_EQ(u1_x1.ToString(/*show_types=*/false), "[[1, 0], [1, 1]]");

  std::vector<NdInterval> pieces = u1_full.SubtractInterval(u1_x1);
  EXPECT_EQ(pieces.size(), 1);
  EXPECT_EQ(pieces[0].ToString(/*show_types=*/false), "[[0, 0], [0, 1]]");
}

TEST(NdRegionTest, Rectangle2DSubtractSelf) {
  std::vector<InterpValue> dim_extents = {
      InterpValue::MakeUBits(1, 1),
      InterpValue::MakeUBits(1, 1),
  };
  const NdInterval u1_full = MakeFullNdInterval(dim_extents);
  std::vector<NdInterval> pieces = u1_full.SubtractInterval(u1_full);
  EXPECT_TRUE(pieces.empty());
}

// From the following rectangle:
// +-------+-------+-------+-------+
// | (0,0) | (1,0) | (2,0) | (3,0) |
// +-------+-------+-------+-------+
// | (0,1) | (1,1) | (2,1) | (3,1) | <
// +-------+-------+-------+-------+ < slice these two rows out
// | (0,2) | (1,2) | (2,2) | (3,2) | <
// +-------+-------+-------+-------+
// | (0,3) | (1,3) | (2,3) | (3,3) |
// +-------+-------+-------+-------+
TEST(NdRegionTest, Rectangle2DSubtractCenterSliceInLeadingDim) {
  // Make a rectangle from point (0, 0) to point (3, 3) inclusive on both ends.
  std::vector<InterpValue> dim_extents = {
      InterpValue::MakeUBits(3, 3),
      InterpValue::MakeUBits(3, 3),
  };
  const NdInterval full = MakeFullNdInterval(dim_extents);

  const NdInterval center_slice = NdInterval::MakeContiguous(
      {
          InterpValue::MakeUBits(3, 0),
          InterpValue::MakeUBits(3, 1),
      },
      {
          InterpValue::MakeUBits(3, 3),
          InterpValue::MakeUBits(3, 2),
      });

  // The remainder should be two slices: from point (0, 0) to (0, 3) and from
  // point (3, 0) to (3, 3).
  std::vector<NdInterval> pieces = full.SubtractInterval(center_slice);
  EXPECT_EQ(pieces.size(), 2);
  EXPECT_EQ(pieces[0].ToString(/*show_types=*/false), "[[0, 0], [3, 0]]");
  EXPECT_EQ(pieces[1].ToString(/*show_types=*/false), "[[0, 3], [3, 3]]");
}

// From the following rectangle:
// +-------+-------+-------+-------+
// | (0,0) | (1,0) | (2,0) | (3,0) |
// +-------+-------+-------+-------+
// | (0,1) | (1,1) | (2,1) | (3,1) |
// +-------+-------+-------+-------+
// | (0,2) | (1,2) | (2,2) | (3,2) |
// +-------+-------+-------+-------+
// | (0,3) | (1,3) | (2,3) | (3,3) |
// +-------+-------+-------+-------+
//          ^^^^^^^^^^^^^^^ sub these two columns out
TEST(NdRegionTest, Rectangle2DSubtractCenterSliceInTrailingDim) {
  // Make a rectangle from point (0, 0) to point (3, 3) inclusive on both ends.
  std::vector<InterpValue> dim_extents = {
      InterpValue::MakeUBits(3, 3),
      InterpValue::MakeUBits(3, 3),
  };
  const NdInterval full = MakeFullNdInterval(dim_extents);

  // Slice out the center two values of the rectangle, i.e. (0, 1) to (3, 2) --
  // not we take all the y values and two of the x values.
  const NdInterval center_slice = NdInterval::MakeContiguous(
      {
          InterpValue::MakeUBits(3, 1),
          InterpValue::MakeUBits(3, 0),
      },
      {
          InterpValue::MakeUBits(3, 2),
          InterpValue::MakeUBits(3, 3),
      });

  // The remainder should be two slices: from point (0, 0) to (0, 3) and from
  // point (3, 0) to (3, 3).
  std::vector<NdInterval> pieces = full.SubtractInterval(center_slice);
  EXPECT_EQ(pieces.size(), 2);
  EXPECT_EQ(pieces[0].ToString(/*show_types=*/false), "[[0, 0], [0, 3]]");
  EXPECT_EQ(pieces[1].ToString(/*show_types=*/false), "[[3, 0], [3, 3]]");
}

TEST(NdRegionTest, NdIntervalIntersectsProperly) {
  // Construct a 2D interval A from (0,0) to (5,10).
  std::vector<InterpValue> startA = {
      InterpValue::MakeU32(0),  // x start
      InterpValue::MakeU32(0)   // y start
  };
  std::vector<InterpValue> endA = {
      InterpValue::MakeU32(5),  // x end
      InterpValue::MakeU32(10)  // y end
  };
  NdInterval intervalA = NdInterval::MakeContiguous(startA, endA);

  // Construct a 2D interval B from (3,2) to (7,12) that overlaps with A.
  std::vector<InterpValue> startB = {
      InterpValue::MakeU32(3),  // x start (overlaps with A: 3 <= x <= 5)
      InterpValue::MakeU32(2)   // y start (overlaps with A: 2 <= y <= 10)
  };
  std::vector<InterpValue> endB = {
      InterpValue::MakeU32(7),  // x end
      InterpValue::MakeU32(12)  // y end
  };
  NdInterval intervalB = NdInterval::MakeContiguous(startB, endB);

  // These intervals should intersect.
  EXPECT_TRUE(intervalA.Intersects(intervalB));
  EXPECT_TRUE(intervalB.Intersects(intervalA));

  // Construct a 2D interval C from (6,0) to (10,10) that is disjoint
  // from A along the x-dimension (since A is [0,5]).
  std::vector<InterpValue> startC = {
      InterpValue::MakeU32(6),  // x start (outside A's x range)
      InterpValue::MakeU32(0)   // y start
  };
  std::vector<InterpValue> endC = {
      InterpValue::MakeU32(10),  // x end
      InterpValue::MakeU32(10)   // y end (overlaps with A in y)
  };
  NdInterval intervalC = NdInterval::MakeContiguous(startC, endC);

  // These intervals should not intersect because intervalC's x-dimension is
  // disjoint.
  EXPECT_FALSE(intervalA.Intersects(intervalC));
  EXPECT_FALSE(intervalC.Intersects(intervalA));
}

// From the following rectangle:
// +-------+-------+-------+-------+
// | (0,0) | (1,0) | (2,0) | (3,0) |
// +-------+-------+-------+-------+
// | (0,1) |*(1,1)*|*(2,1)*| (3,1) |
// +-------+-------+-------+-------+
// | (0,2) |*(1,2)*|*(2,2)*| (3,2) |
// +-------+-------+-------+-------+
// | (0,3) | (1,3) | (2,3) | (3,3) |
// +-------+-------+-------+-------+
TEST(NdRegionTest, Rectangle2DSubtractCentralRectangle) {
  // Create a full rectangle from (0,0) to (3,3)
  std::vector<InterpValue> dim_extents = {InterpValue::MakeUBits(3, 3),
                                          InterpValue::MakeUBits(3, 3)};
  const NdInterval full = MakeFullNdInterval(dim_extents);

  // Create a central sub-rectangle from (1,1) to (2,2)
  NdInterval central = NdInterval::MakeContiguous(
      {InterpValue::MakeUBits(3, 1), InterpValue::MakeUBits(3, 1)},
      {InterpValue::MakeUBits(3, 2), InterpValue::MakeUBits(3, 2)});

  // We should end up with:
  // * (0,0) to (0,3) -- left slice in above diagram
  // * (1,0) to (2,0) -- upper slice in above diagram
  // * (1,3) to (2,3) -- lower slice in above diagram
  // * (3,0) to (3,3) -- right slice in above diagram
  std::vector<NdInterval> pieces = full.SubtractInterval(central);

  std::vector<std::string> expected = {"[[0, 0], [0, 3]]", "[[1, 0], [2, 0]]",
                                       "[[1, 3], [2, 3]]", "[[3, 0], [3, 3]]"};
  std::vector<std::string> actual;
  actual.reserve(pieces.size());
  for (const NdInterval& interval : pieces) {
    actual.push_back(interval.ToString(false));
  }
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(actual, expected);
}

TEST(NdRegionTest, OneDimensionalSubtractMiddle) {
  // Create a 1D full interval: [0, 9]
  std::vector<InterpValue> full_extent = {InterpValue::MakeU32(9)};
  NdInterval full = MakeFullNdInterval(full_extent);

  // Define a subtraction interval in the middle: [3, 6]
  NdInterval sub = NdInterval::MakeContiguous({InterpValue::MakeU32(3)},
                                              {InterpValue::MakeU32(6)});
  std::vector<NdInterval> pieces = full.SubtractInterval(sub);

  // Expect two pieces:
  //   left piece: [0] ... [2]  (since 3 - 1 = 2)
  //   right piece: [7] ... [9] (since 6 + 1 = 7)
  EXPECT_EQ(pieces.size(), 2);
  std::vector<std::string> expected = {"[[0], [2]]", "[[7], [9]]"};
  std::vector<std::string> actual;
  actual.reserve(pieces.size());
  for (const NdInterval& piece : pieces) {
    actual.push_back(piece.ToString(/*show_types=*/false));
  }
  std::sort(expected.begin(), expected.end());
  std::sort(actual.begin(), actual.end());
  EXPECT_EQ(actual, expected);
}

TEST(NdRegionTest, Rectangle2DSubtractNonIntersecting) {
  // Create a 2D full region: from (0,0) to (3,3)
  std::vector<InterpValue> dim_extents = {
      InterpValue::MakeUBits(3, 3),
      InterpValue::MakeUBits(3, 3),
  };
  const NdInterval full = MakeFullNdInterval(dim_extents);

  // Create a subtraction interval (a point) that lies completely outside
  // the full region. (Here, the point (4,4) is outside since the full region
  // ends at (3,3)).
  NdInterval non_intersecting = NdInterval::MakePoint(
      {InterpValue::MakeUBits(3, 4), InterpValue::MakeUBits(3, 4)});

  std::vector<NdInterval> pieces = full.SubtractInterval(non_intersecting);
  // Since there is no intersection, expect the original region to be returned.
  EXPECT_EQ(pieces.size(), 1);
  EXPECT_EQ(pieces[0].ToString(/*show_types=*/false), "[[0, 0], [3, 3]]");
}

}  // namespace xls::dslx
