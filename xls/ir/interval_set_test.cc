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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"

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
  EXPECT_EQ(empty.ConvexHull(), absl::nullopt);
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

TEST(IntervalTest, Size) {
  IntervalSet example(32);
  example.AddInterval(MakeInterval(5, 10, 32));
  example.AddInterval(MakeInterval(12, 14, 32));
  example.Normalize();
  EXPECT_EQ(example.Size(), 9);

  IntervalSet too_big(80);
  too_big.AddInterval(MakeInterval(10, 5, 80));
  too_big.Normalize();
  EXPECT_EQ(too_big.Size(), absl::nullopt);
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

  IntervalSet is_not_precise(10);
  is_not_precise.AddInterval(MakeInterval(5, 5, 10));
  is_not_precise.AddInterval(MakeInterval(12, 12, 10));
  EXPECT_FALSE(is_not_precise.IsPrecise());
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

}  // namespace
}  // namespace xls
