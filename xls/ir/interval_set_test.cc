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

#include "xls/ir/interval_set.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"

using ::testing::Optional;

namespace xls {
namespace {

Interval MakeInterval(uint64_t start, uint64_t end, int64_t width) {
  return Interval(UBits(start, width), UBits(end, width));
}

TEST(IntervalTest, Normalize) {
  IntervalSet is_sorted(32);
  is_sorted.AddInterval(MakeInterval(100, 150, 32));
  is_sorted.AddInterval(MakeInterval(5, 20, 32));
  is_sorted.Normalize();
  EXPECT_EQ(is_sorted.Intervals(),
            (std::vector<Interval>{MakeInterval(5, 20, 32),
                                   MakeInterval(100, 150, 32)}));

  IntervalSet overlapping(32);
  overlapping.AddInterval(MakeInterval(5, 20, 32));
  overlapping.AddInterval(MakeInterval(15, 30, 32));
  overlapping.Normalize();
  EXPECT_EQ(overlapping.Intervals(),
            (std::vector<Interval>{MakeInterval(5, 30, 32)}));

  IntervalSet abutting(32);
  abutting.AddInterval(MakeInterval(5, 20, 32));
  abutting.AddInterval(MakeInterval(21, 30, 32));
  abutting.Normalize();
  EXPECT_EQ(abutting.Intervals(),
            (std::vector<Interval>{MakeInterval(5, 30, 32)}));

  IntervalSet improper(32);
  improper.AddInterval(MakeInterval(20, 10, 32));
  improper.Normalize();
  EXPECT_EQ(improper.Intervals(),
            (std::vector<Interval>{
                MakeInterval(0, 10, 32),
                MakeInterval(20, std::numeric_limits<uint32_t>::max(), 32)}));
}

TEST(IntervalTest, ConvexHull) {
  IntervalSet example(32);
  example.AddInterval(MakeInterval(10, 20, 32));
  example.AddInterval(MakeInterval(15, 30, 32));  // overlapping
  example.AddInterval(MakeInterval(31, 40, 32));  // abutting
  example.AddInterval(MakeInterval(70, 90, 32));  // separate
  example.AddInterval(MakeInterval(80, 85, 32));  // not sorted, landlocked
  example.AddInterval(MakeInterval(50, 55, 32));  // not sorted, separate
  EXPECT_EQ(example.ConvexHull(), MakeInterval(10, 90, 32));

  IntervalSet empty(32);
  EXPECT_EQ(empty.ConvexHull(), std::nullopt);
}

TEST(IntervalTest, ForEachElement) {
  IntervalSet example(32);
  example.AddInterval(MakeInterval(10, 40, 32));
  example.AddInterval(MakeInterval(50, 55, 32));
  example.AddInterval(MakeInterval(70, 90, 32));
  example.Normalize();

  absl::flat_hash_set<Bits> visited;
  example.ForEachElement([&](const Bits& bits) -> bool {
    visited.insert(bits);
    return false;
  });
  EXPECT_EQ(visited.size(), example.Size().value());

  for (const Interval& interval : example.Intervals()) {
    interval.ForEachElement([&](const Bits& bits) -> bool {
      EXPECT_TRUE(visited.contains(bits)) << bits.ToString();
      visited.erase(bits);
      return false;
    });
  }

  EXPECT_TRUE(visited.empty());
}

TEST(IntervalTest, Combine) {
  IntervalSet x(32);
  x.AddInterval(MakeInterval(5, 10, 32));
  x.AddInterval(MakeInterval(8, 15, 32));

  IntervalSet y(32);
  y.AddInterval(MakeInterval(25, 30, 32));
  y.AddInterval(MakeInterval(31, 35, 32));

  EXPECT_EQ(IntervalSet::Combine(x, y).Intervals(),
            (std::vector<Interval>{MakeInterval(5, 15, 32),
                                   MakeInterval(25, 35, 32)}));
}

TEST(IntervalTest, Intersect) {
  // Manually tested with 1,000 random seeds;
  int64_t seed = 815303902;
  for (int64_t i = 0; i < 30; ++i) {
    IntervalSet lhs = IntervalSet::Random(seed, 12, 5);
    IntervalSet rhs = IntervalSet::Random(seed + 1, 12, 5);
    seed = seed + 2;
    absl::flat_hash_set<Bits> intersection_set;

    {
      absl::flat_hash_set<Bits> lhs_set;
      lhs.ForEachElement([&](const Bits& bits) -> bool {
        lhs_set.insert(bits);
        return false;
      });
      rhs.ForEachElement([&](const Bits& bits) -> bool {
        if (lhs_set.contains(bits)) {
          intersection_set.insert(bits);
        }
        return false;
      });
    }

    // If A ⊆ B and |A| = |B| then A = B
    IntervalSet intersection = IntervalSet::Intersect(lhs, rhs);
    EXPECT_EQ(intersection_set.size(), intersection.Size().value());
    intersection.ForEachElement([&](const Bits& bits) -> bool {
      EXPECT_TRUE(intersection_set.contains(bits));
      return false;
    });
  }

  {
    IntervalSet empty(0);
    EXPECT_EQ(IntervalSet::Intersect(empty, empty).ToString(), "[]");
  }

  {
    IntervalSet lhs(32);
    lhs.AddInterval(MakeInterval(100, 200, 32));
    lhs.AddInterval(MakeInterval(500, 600, 32));
    lhs.AddInterval(MakeInterval(800, 1000, 32));
    lhs.Normalize();

    IntervalSet rhs(32);
    rhs.AddInterval(MakeInterval(100, 200, 32));
    rhs.AddInterval(MakeInterval(450, 600, 32));
    rhs.AddInterval(MakeInterval(900, 950, 32));
    rhs.Normalize();

    EXPECT_EQ(IntervalSet::Intersect(lhs, rhs).ToString(),
              "[[100, 200], [500, 600], [900, 950]]");
  }

  {
    IntervalSet singleton_lhs(32);
    singleton_lhs.AddInterval(MakeInterval(100, 200, 32));
    singleton_lhs.Normalize();

    IntervalSet singleton_rhs(32);
    singleton_rhs.AddInterval(MakeInterval(150, 300, 32));
    singleton_rhs.Normalize();

    IntervalSet disjoint(32);
    disjoint.AddInterval(MakeInterval(1000, 2000, 32));
    disjoint.Normalize();

    EXPECT_EQ(IntervalSet::Intersect(singleton_lhs, singleton_lhs).ToString(),
              "[[100, 200]]");
    EXPECT_EQ(IntervalSet::Intersect(singleton_rhs, singleton_rhs).ToString(),
              "[[150, 300]]");
    EXPECT_EQ(IntervalSet::Intersect(singleton_lhs, singleton_rhs).ToString(),
              "[[150, 200]]");
    EXPECT_EQ(IntervalSet::Intersect(singleton_lhs, disjoint).ToString(), "[]");
    EXPECT_EQ(IntervalSet::Intersect(singleton_rhs, disjoint).ToString(), "[]");
    EXPECT_EQ(IntervalSet::Intersect(disjoint, singleton_lhs).ToString(), "[]");
    EXPECT_EQ(IntervalSet::Intersect(disjoint, singleton_rhs).ToString(), "[]");
  }
}

TEST(IntervalTest, Size) {
  IntervalSet example(32);
  example.AddInterval(MakeInterval(5, 10, 32));
  example.AddInterval(MakeInterval(12, 14, 32));
  example.Normalize();
  EXPECT_EQ(example.Size(), 9);

  IntervalSet too_big(80);
  too_big.AddInterval(MakeInterval(10, 5, 80));
  too_big.Normalize();
  EXPECT_EQ(too_big.Size(), std::nullopt);
}

TEST(IntervalTest, IsTrueWhenMaskWith) {
  IntervalSet example(3);
  example.AddInterval(MakeInterval(0, 0, 3));
  for (int64_t value = 0; value < 8; ++value) {
    EXPECT_FALSE(example.IsTrueWhenMaskWith(UBits(value, 3)));
  }
  example.AddInterval(MakeInterval(2, 4, 3));
  EXPECT_FALSE(example.IsTrueWhenMaskWith(UBits(0, 3)));
  EXPECT_FALSE(example.IsTrueWhenMaskWith(UBits(1, 3)));
  for (int64_t value = 2; value < 8; ++value) {
    EXPECT_TRUE(example.IsTrueWhenMaskWith(UBits(value, 3)));
  }
}

TEST(IntervalTest, Covers) {
  IntervalSet example(4);
  example.AddInterval(MakeInterval(4, 10, 4));
  EXPECT_FALSE(example.Covers(UBits(0, 4)));
  EXPECT_FALSE(example.Covers(UBits(1, 4)));
  EXPECT_FALSE(example.Covers(UBits(2, 4)));
  EXPECT_FALSE(example.Covers(UBits(3, 4)));
  EXPECT_TRUE(example.Covers(UBits(4, 4)));
  EXPECT_TRUE(example.Covers(UBits(5, 4)));
  EXPECT_TRUE(example.Covers(UBits(6, 4)));
  EXPECT_TRUE(example.Covers(UBits(7, 4)));
  EXPECT_TRUE(example.Covers(UBits(8, 4)));
  EXPECT_TRUE(example.Covers(UBits(9, 4)));
  EXPECT_TRUE(example.Covers(UBits(10, 4)));
  EXPECT_FALSE(example.Covers(UBits(11, 4)));
  EXPECT_FALSE(example.Covers(UBits(12, 4)));
  EXPECT_FALSE(example.Covers(UBits(13, 4)));
  EXPECT_FALSE(example.Covers(UBits(14, 4)));
  EXPECT_FALSE(example.Covers(UBits(15, 4)));
}

TEST(IntervalTest, IsPrecise) {
  IntervalSet is_precise(10);
  is_precise.AddInterval(MakeInterval(5, 5, 10));
  EXPECT_TRUE(is_precise.IsPrecise());
  is_precise.Normalize();
  EXPECT_THAT(is_precise.GetPreciseValue(), Optional(UBits(5, 10)));

  IntervalSet is_not_precise(10);
  is_not_precise.AddInterval(MakeInterval(5, 5, 10));
  is_not_precise.AddInterval(MakeInterval(12, 12, 10));
  EXPECT_FALSE(is_not_precise.IsPrecise());
  is_not_precise.Normalize();
  EXPECT_THAT(is_not_precise.GetPreciseValue(), std::nullopt);
}

TEST(IntervalTest, IsMaximal) {
  IntervalSet is_maximal(10);
  is_maximal.AddInterval(MakeInterval(0, 1023, 10));
  is_maximal.Normalize();
  EXPECT_TRUE(is_maximal.IsMaximal());

  IntervalSet is_not_maximal(10);
  is_not_maximal.AddInterval(MakeInterval(5, 100, 10));
  is_not_maximal.AddInterval(MakeInterval(150, 500, 10));
  is_not_maximal.Normalize();
  EXPECT_FALSE(is_not_maximal.IsMaximal());
}

TEST(IntervalTest, ToString) {
  IntervalSet example(32);
  example.AddInterval(MakeInterval(10, 20, 32));
  example.AddInterval(MakeInterval(15, 30, 32));  // overlapping
  example.AddInterval(MakeInterval(31, 40, 32));  // abutting
  example.AddInterval(MakeInterval(70, 90, 32));  // separate
  example.AddInterval(MakeInterval(80, 85, 32));  // not sorted, landlocked
  example.AddInterval(MakeInterval(50, 55, 32));  // not sorted, separate

  EXPECT_EQ(example.ToString(),
            "[[10, 20], [15, 30], [31, 40], [70, 90], [80, 85], [50, 55]]");
}

TEST(IntervalTest, Complement) {
  EXPECT_EQ(IntervalSet::Complement(IntervalSet(0)), IntervalSet::Maximal(0));

  EXPECT_EQ(IntervalSet::Complement(IntervalSet(32)), IntervalSet::Maximal(32));

  EXPECT_EQ(IntervalSet::Complement(IntervalSet::Maximal(32)), IntervalSet(32));

  IntervalSet set1(32);
  set1.AddInterval(MakeInterval(100, 150, 32));
  set1.AddInterval(MakeInterval(5, 20, 32));
  set1.Normalize();
  EXPECT_EQ(IntervalSet::Complement(IntervalSet::Complement(set1)), set1);

  IntervalSet complement_set1(32);
  complement_set1.AddInterval(MakeInterval(0, 4, 32));
  complement_set1.AddInterval(MakeInterval(21, 99, 32));
  complement_set1.AddInterval(
      MakeInterval(151, std::numeric_limits<uint32_t>::max(), 32));
  complement_set1.Normalize();
  EXPECT_EQ(IntervalSet::Complement(set1), complement_set1);
}

TEST(IntervalTest, IsEmpty) {
  EXPECT_TRUE(IntervalSet(32).IsEmpty());
  EXPECT_TRUE(IntervalSet(0).IsEmpty());

  IntervalSet set1(32);
  set1.AddInterval(MakeInterval(100, 150, 32));
  set1.AddInterval(MakeInterval(5, 20, 32));
  set1.Normalize();
  EXPECT_FALSE(set1.IsEmpty());
}

TEST(IntervalTest, Bounds) {
  IntervalSet empty(32);
  EXPECT_FALSE(empty.LowerBound().has_value());
  EXPECT_FALSE(empty.UpperBound().has_value());

  IntervalSet precise(10);
  precise.AddInterval(MakeInterval(5, 5, 10));
  EXPECT_THAT(precise.LowerBound(), Optional(UBits(5, 10)));
  EXPECT_THAT(precise.UpperBound(), Optional(UBits(5, 10)));

  IntervalSet a(64);
  a.AddInterval(MakeInterval(10, 123, 64));
  EXPECT_THAT(a.LowerBound(), Optional(UBits(10, 64)));
  EXPECT_THAT(a.UpperBound(), Optional(UBits(123, 64)));

  IntervalSet b(64);
  b.AddInterval(MakeInterval(0, 42, 64));
  b.AddInterval(MakeInterval(100, 200, 64));
  EXPECT_THAT(b.LowerBound(), Optional(UBits(0, 64)));
  EXPECT_THAT(b.UpperBound(), Optional(UBits(200, 64)));
}

TEST(IntervalTest, Index) {
  IntervalSet set(32);
  set.AddInterval(MakeInterval(100, 150, 32));
  set.AddInterval(MakeInterval(200, 220, 32));
  set.AddInterval(MakeInterval(300, 370, 32));
  set.Normalize();
  EXPECT_EQ(set.Index(UBits(0, 32)), UBits(100, 32));
  EXPECT_EQ(set.Index(UBits(1, 32)), UBits(101, 32));
  EXPECT_EQ(set.Index(UBits(10, 32)), UBits(110, 32));
  EXPECT_EQ(set.Index(UBits(49, 32)), UBits(149, 32));
  EXPECT_EQ(set.Index(UBits(50, 32)), UBits(150, 32));
  EXPECT_EQ(set.Index(UBits(51, 32)), UBits(200, 32));
  EXPECT_EQ(set.Index(UBits(60, 32)), UBits(209, 32));
  EXPECT_EQ(set.Index(UBits(71, 32)), UBits(220, 32));
  EXPECT_EQ(set.Index(UBits(72, 32)), UBits(300, 32));
  EXPECT_EQ(set.Index(UBits(112, 32)), UBits(340, 32));
  EXPECT_EQ(set.Index(UBits(142, 32)), UBits(370, 32));
  EXPECT_EQ(set.Index(UBits(143, 32)), std::nullopt);
  EXPECT_EQ(set.Index(UBits(200, 32)), std::nullopt);
}

TEST(IntervalTest, ZeroExtend) {
  IntervalSet set(32);
  set.AddInterval(MakeInterval(100, 150, 32));
  set.AddInterval(MakeInterval(200, 220, 32));
  set.AddInterval(MakeInterval(300, 370, 32));
  set.Normalize();

  IntervalSet extended = set.ZeroExtend(60);
  EXPECT_EQ(extended.ToString(), "[[100, 150], [200, 220], [300, 370]]");
  EXPECT_EQ(extended.BitCount(), 60);
}

}  // namespace
}  // namespace xls
