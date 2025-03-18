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
#include <limits>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set_test_utils.h"

using ::testing::ExplainMatchResult;
using ::testing::IsTrue;
using ::testing::Not;
using ::testing::Optional;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAreArray;

namespace xls {
namespace {

Interval MakeInterval(uint64_t start, uint64_t end, int64_t width) {
  return Interval(UBits(start, width), UBits(end, width));
}
Interval MakeSignedInterval(uint64_t start, uint64_t end, int64_t width) {
  return Interval(SBits(start, width), SBits(end, width));
}
MATCHER_P3(IsInterval, low, high, bits,
           absl::StrFormat("Matches interval [%d, %d] (width: %d)", low, high,
                           bits)) {
  return testing::ExplainMatchResult(MakeInterval(low, high, bits), arg,
                                     result_listener);
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

  IntervalSet abutting_reversed(32);
  abutting_reversed.AddInterval(MakeInterval(21, 30, 32));
  abutting_reversed.AddInterval(MakeInterval(5, 20, 32));
  abutting_reversed.Normalize();
  EXPECT_EQ(abutting_reversed.Intervals(),
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
      EXPECT_TRUE(visited.contains(bits)) << bits;
      visited.erase(bits);
      return false;
    });
  }

  EXPECT_TRUE(visited.empty());
}

TEST(IntervalTest, ForEachElementAborts) {
  IntervalSet example(32);
  example.AddInterval(MakeInterval(50, 55, 32));
  example.AddInterval(MakeInterval(10, 40, 32));
  example.AddInterval(MakeInterval(70, 90, 32));
  example.Normalize();

  absl::flat_hash_set<Bits> visited;
  EXPECT_TRUE(example.ForEachElement([&](const Bits& bits) -> bool {
    if (*bits.ToUint64() >= 52) {
      visited.insert(bits);
      return true;
    }
    visited.insert(bits);
    return false;
  }));
  EXPECT_THAT(visited, SizeIs(31 + 3));  // [10, 40] and [50, 52]

  visited.clear();
  EXPECT_TRUE(example.ForEachElement([&](const Bits& bits) -> bool {
    if (*bits.ToUint64() >= 55) {
      visited.insert(bits);
      return true;
    }
    visited.insert(bits);
    return false;
  }));
  EXPECT_THAT(visited, SizeIs(31 + 6));  // [10, 40] and [50, 55]

  visited.clear();
  EXPECT_TRUE(example.ForEachElement([&](const Bits& bits) -> bool {
    if (*bits.ToUint64() >= 50) {
      visited.insert(bits);
      return true;
    }
    visited.insert(bits);
    return false;
  }));
  EXPECT_THAT(visited, SizeIs(31 + 1));  // [10, 40] and [50, 50]
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

void IntersectMatchesSetIntersection(const IntervalSet& lhs,
                                     const IntervalSet& rhs) {
  IntervalSet intersection = IntervalSet::Intersect(lhs, rhs);

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

  EXPECT_EQ(intersection_set.size(), intersection.Size().value());
  std::vector<Bits> intersection_vector;
  intersection.ForEachElement([&](const Bits& bits) -> bool {
    intersection_vector.push_back(bits);
    return false;
  });
  EXPECT_THAT(intersection_vector, UnorderedElementsAreArray(intersection_set));
}
FUZZ_TEST(IntervalFuzzTest, IntersectMatchesSetIntersection)
    .WithDomains(ArbitraryNormalizedIntervalSet(12),
                 ArbitraryNormalizedIntervalSet(12));

TEST(IntervalTest, Intersect) {
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
  IntervalSet example(4);
  example.AddInterval(MakeInterval(0, 0, 4));
  for (int64_t value = 0; value < 16; ++value) {
    EXPECT_FALSE(example.IsTrueWhenMaskWith(UBits(value, 4)));
  }
  example.AddInterval(MakeInterval(2, 4, 4));
  EXPECT_FALSE(example.IsTrueWhenMaskWith(UBits(0, 4)));
  for (int64_t value = 1; value < 8; ++value) {
    EXPECT_TRUE(example.IsTrueWhenMaskWith(UBits(value, 4)));
  }
  EXPECT_FALSE(example.IsTrueWhenMaskWith(UBits(8, 4)));
  for (int64_t value = 9; value < 16; ++value) {
    EXPECT_TRUE(example.IsTrueWhenMaskWith(UBits(value, 4)));
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
  b.AddInterval(MakeInterval(100, 200, 64));
  b.AddInterval(MakeInterval(0, 42, 64));
  EXPECT_THAT(b.LowerBound(), Optional(UBits(0, 64)));
  EXPECT_THAT(b.UpperBound(), Optional(UBits(200, 64)));

  IntervalSet c(64);
  c.AddInterval(MakeInterval(21, 200, 64));
  c.AddInterval(MakeInterval(0, 42, 64));
  EXPECT_THAT(c.LowerBound(), Optional(UBits(0, 64)));
  EXPECT_THAT(c.UpperBound(), Optional(UBits(200, 64)));
  c.Normalize();
  EXPECT_THAT(c.LowerBound(), Optional(UBits(0, 64)));
  EXPECT_THAT(c.UpperBound(), Optional(UBits(200, 64)));
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

IntervalSet SignedIntervals(absl::Span<std::pair<int64_t, int64_t> const> v,
                            int64_t bit_count = 32) {
  IntervalSet set(bit_count);
  for (const auto& [l, h] : v) {
    set.AddInterval(MakeSignedInterval(l, h, bit_count));
  }
  set.Normalize();
  return set;
}
IntervalSet Intervals(absl::Span<std::pair<int64_t, int64_t> const> v,
                      int64_t bit_count = 32) {
  IntervalSet set(bit_count);
  for (const auto& [l, h] : v) {
    set.AddInterval(MakeInterval(l, h, bit_count));
  }
  set.Normalize();
  return set;
}

MATCHER_P(DisjointWith, other,
          absl::StrFormat("Is %sdisjoint with %s", negation ? "not " : "",
                          other.ToString())) {
  const IntervalSet& rhs = other;
  const IntervalSet& lhs = arg;
  return ExplainMatchResult(IsTrue(), IntervalSet::Disjoint(lhs, rhs),
                            result_listener);
}

TEST(IntervalSetTest, Disjoint) {
  {
    IntervalSet low = Intervals({{1, 1}});
    IntervalSet high = Intervals({{10, 10}});
    EXPECT_THAT(low, DisjointWith(high));
    EXPECT_THAT(high, DisjointWith(low));
  }
  {
    IntervalSet l = Intervals({{1, 10}});
    IntervalSet r = Intervals({{10, 10}});
    EXPECT_THAT(l, Not(DisjointWith(r)));
  }
  {
    IntervalSet l = Intervals({{1, 1}, {3, 3}, {5, 5}});
    IntervalSet r = Intervals({{2, 2}, {4, 4}, {6, 6}});
    EXPECT_THAT(l, DisjointWith(r));
  }
}

TEST(IntervalSetTest, SignedIterator) {
  {
    IntervalSet all_positive = Intervals({{0, 4}, {8, 12}, {18, 127}}, 8);
    auto rng = all_positive.SignedIntervals();
    auto it = rng.begin();
    EXPECT_THAT(*it, IsInterval(0, 4, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(8, 12, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(18, 127, 8));
    ++it;
    EXPECT_TRUE(it == rng.end());
  }
  {
    IntervalSet all_neg = Intervals({{128, 133}, {138, 144}, {155, 255}}, 8);
    auto rng = all_neg.SignedIntervals();
    auto it = rng.begin();
    EXPECT_THAT(*it, IsInterval(128, 133, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(138, 144, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(155, 255, 8));
    ++it;
    EXPECT_TRUE(it == rng.end());
  }
  {
    IntervalSet cov =
        Intervals({{0, 4}, {8, 12}, {18, 133}, {138, 144}, {155, 255}}, 8);
    auto rng = cov.SignedIntervals();
    auto it = rng.begin();
    EXPECT_THAT(*it, IsInterval(0, 4, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(8, 12, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(18, 127, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(128, 133, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(138, 144, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(155, 255, 8));
    ++it;
    EXPECT_TRUE(it == rng.end());
  }

  {
    IntervalSet unbound = Intervals({{0, 255}}, 8);
    auto rng = unbound.SignedIntervals();
    auto it = rng.begin();
    EXPECT_THAT(*it, IsInterval(0, 127, 8));
    ++it;
    EXPECT_THAT(*it, IsInterval(128, 255, 8));
    ++it;
    EXPECT_TRUE(it == rng.end());
  }
}

TEST(IntervalSetTest, PositiveIntervals) {
  {
    IntervalSet all_positive = SignedIntervals({{0, 4}, {8, 12}, {18, 127}}, 8);
    EXPECT_EQ(all_positive, all_positive.PositiveIntervals(/*with_zero=*/true));
  }
  {
    IntervalSet all_positive = SignedIntervals({{0, 4}, {8, 12}, {18, 127}}, 8);
    EXPECT_EQ(Intervals({{1, 4}, {8, 12}, {18, 127}}, 8),
              all_positive.PositiveIntervals(/*with_zero=*/false));
  }
  {
    IntervalSet pos_and_neg =
        SignedIntervals({{0, 4}, {8, 12}, {18, 127}, {-12, -1}}, 8);
    EXPECT_EQ(Intervals({{0, 4}, {8, 12}, {18, 127}}, 8),
              pos_and_neg.PositiveIntervals(/*with_zero=*/true));
  }
  {
    IntervalSet neg = SignedIntervals({{-127, -22}, {-12, -1}}, 8);
    EXPECT_EQ(Intervals({}, 8), neg.PositiveIntervals(/*with_zero=*/true));
  }
  {
    IntervalSet tiny_range = SignedIntervals({{-1, 0}}, 1);
    EXPECT_EQ(Intervals({{0, 0}}, 1),
              tiny_range.PositiveIntervals(/*with_zero=*/true));
    EXPECT_EQ(Intervals({}, 1),
              tiny_range.PositiveIntervals(/*with_zero=*/false));
  }
}

TEST(IntervalSetTest, NegativeIntervals) {
  {
    IntervalSet all_negative = SignedIntervals({{-12, -8}, {-127, -18}}, 8);
    EXPECT_EQ(Intervals({{8, 12}, {18, 127}}, 8),
              all_negative.NegativeAbsoluteIntervals());
  }
  {
    IntervalSet pos_and_neg =
        SignedIntervals({{0, 4}, {8, 12}, {18, 127}, {-12, -1}}, 8);
    EXPECT_EQ(Intervals({{1, 12}}, 8), pos_and_neg.NegativeAbsoluteIntervals());
  }
  {
    IntervalSet tiny_range = SignedIntervals({{-1, 0}}, 1);
    EXPECT_EQ(Intervals({{1, 1}}, 1), tiny_range.NegativeAbsoluteIntervals());
  }
}

TEST(IntervalSetTest, IterateWithEndAndStartAdjacent) {
  IntervalSet set = Intervals({{0, 7}, {15, 15}}, 4);
  auto it = set.Values().begin();
  EXPECT_THAT(*it, UBits(0, 4));
  ++it;
  EXPECT_THAT(*it, UBits(1, 4));
  ++it;
  EXPECT_THAT(*it, UBits(2, 4));
  ++it;
  EXPECT_THAT(*it, UBits(3, 4));
  ++it;
  EXPECT_THAT(*it, UBits(4, 4));
  ++it;
  EXPECT_THAT(*it, UBits(5, 4));
  ++it;
  EXPECT_THAT(*it, UBits(6, 4));
  ++it;
  EXPECT_THAT(*it, UBits(7, 4));
  ++it;
  EXPECT_THAT(*it, UBits(15, 4));
  ++it;
  EXPECT_TRUE(it == set.Values().end());
}

void IntersectionIsSmaller(const IntervalSet& lhs, const IntervalSet& rhs) {
  IntervalSet intersection = IntervalSet::Intersect(lhs, rhs);
  std::optional<uint64_t> intersection_size = intersection.Size();
  ASSERT_NE(intersection_size, std::nullopt);
  uint64_t lhs_size = *lhs.Size();
  uint64_t rhs_size = *rhs.Size();
  EXPECT_LE(*intersection_size, lhs_size);
  EXPECT_LE(*intersection_size, rhs_size);
  if (*intersection_size == lhs_size) {
    EXPECT_EQ(intersection, lhs);
  }
  if (*intersection_size == rhs_size) {
    EXPECT_EQ(intersection, rhs);
  }
}
FUZZ_TEST(IntervalFuzzTest, IntersectionIsSmaller)
    .WithDomains(ArbitraryNormalizedIntervalSet(32),
                 ArbitraryNormalizedIntervalSet(32));

void UnionIsLarger(const IntervalSet& lhs, const IntervalSet& rhs) {
  IntervalSet union_set = IntervalSet::Combine(lhs, rhs);
  std::optional<uint64_t> union_size = union_set.Size();
  ASSERT_NE(union_size, std::nullopt);
  uint64_t lhs_size = *lhs.Size();
  uint64_t rhs_size = *rhs.Size();
  EXPECT_GE(*union_size, lhs_size);
  EXPECT_GE(*union_size, rhs_size);
  EXPECT_LE(*union_size, lhs_size + rhs_size);
  if (*union_size == lhs_size) {
    EXPECT_EQ(union_set, lhs);
  }
  if (*union_size == rhs_size) {
    EXPECT_EQ(union_set, rhs);
  }
}
FUZZ_TEST(IntervalFuzzTest, UnionIsLarger)
    .WithDomains(ArbitraryNormalizedIntervalSet(32),
                 ArbitraryNormalizedIntervalSet(32));

void DisjointEquivalentToEmptyIntersection(const IntervalSet& lhs,
                                           const IntervalSet& rhs) {
  IntervalSet intersection = IntervalSet::Intersect(lhs, rhs);
  EXPECT_EQ(IntervalSet::Disjoint(lhs, rhs), intersection.IsEmpty());
}

FUZZ_TEST(IntervalFuzzTest, DisjointEquivalentToEmptyIntersection)
    .WithDomains(ArbitraryNormalizedIntervalSet(32),
                 ArbitraryNormalizedIntervalSet(32));

}  // namespace
}  // namespace xls
