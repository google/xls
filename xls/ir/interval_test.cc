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

#include "xls/ir/interval.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"

using ::testing::ElementsAre;
using ::testing::Optional;

namespace xls {
namespace {

Bits SumOf(absl::Span<Bits const> vec) {
  EXPECT_TRUE(!vec.empty());
  Bits result = Bits(vec[0].bit_count());
  for (const Bits& bits : vec) {
    EXPECT_EQ(result.bit_count(), bits.bit_count());
    result = bits_ops::Add(result, bits);
  }
  return result;
}

Interval SimpleInterval(uint32_t lower, uint32_t upper) {
  return Interval(UBits(lower, 32), UBits(upper, 32));
}

TEST(IntervalTest, BitCount) {
  Interval size_zero(Bits(0), Bits(0));
  EXPECT_EQ(size_zero.BitCount(), 0);
  Interval size_one(Bits(1), Bits(1));
  EXPECT_EQ(size_one.BitCount(), 1);
  Interval size_twenty(Bits(20), Bits(20));
  EXPECT_EQ(size_twenty.BitCount(), 20);
}

TEST(IntervalTest, OverlapsAndDisjoint) {
  Interval x(Bits::PowerOfTwo(3, 20), Bits::PowerOfTwo(10, 20));
  Interval y(Bits::PowerOfTwo(9, 20), Bits::PowerOfTwo(12, 20));
  Interval z(Bits::PowerOfTwo(12, 20), Bits::PowerOfTwo(18, 20));

  EXPECT_TRUE(Interval::Overlaps(x, y));
  EXPECT_TRUE(Interval::Overlaps(y, z));
  EXPECT_FALSE(Interval::Overlaps(x, z));
  // Flipped versions of the above
  EXPECT_TRUE(Interval::Overlaps(y, x));
  EXPECT_TRUE(Interval::Overlaps(z, y));
  EXPECT_FALSE(Interval::Overlaps(z, x));

  EXPECT_FALSE(Interval::Disjoint(x, y));
  EXPECT_FALSE(Interval::Disjoint(y, z));
  EXPECT_TRUE(Interval::Disjoint(x, z));
  // Flipped versions of the above
  EXPECT_FALSE(Interval::Disjoint(y, x));
  EXPECT_FALSE(Interval::Disjoint(z, y));
  EXPECT_TRUE(Interval::Disjoint(z, x));

  // The zero-width interval overlaps with itself
  EXPECT_TRUE(
      Interval::Overlaps(Interval(Bits(), Bits()), Interval(Bits(), Bits())));
  EXPECT_FALSE(
      Interval::Disjoint(Interval(Bits(), Bits()), Interval(Bits(), Bits())));
}

TEST(IntervalTest, Abuts) {
  Bits fifty_three = UBits(53, 6);
  Bits fifty_four = UBits(54, 6);
  Interval zero_to_fifty_three(UBits(0, 6), fifty_three);
  Interval fifty_four_to_max(fifty_four, Bits::AllOnes(6));
  Interval fifty_three_point(fifty_three, fifty_three);
  Interval fifty_four_point(fifty_four, fifty_four);
  Interval everything = Interval::Maximal(6);
  EXPECT_TRUE(Interval::Abuts(zero_to_fifty_three, fifty_four_to_max));
  EXPECT_TRUE(Interval::Abuts(fifty_three_point, fifty_four_to_max));
  EXPECT_TRUE(Interval::Abuts(zero_to_fifty_three, fifty_four_point));
  EXPECT_TRUE(Interval::Abuts(fifty_three_point, fifty_four_point));
  EXPECT_FALSE(Interval::Abuts(zero_to_fifty_three, everything));
  EXPECT_FALSE(Interval::Abuts(fifty_four_to_max, everything));
  EXPECT_FALSE(Interval::Abuts(fifty_three_point, everything));
  EXPECT_FALSE(Interval::Abuts(fifty_four_point, everything));
  EXPECT_FALSE(Interval::Abuts(everything, everything));

  // The zero-width interval does not abut itself
  EXPECT_FALSE(
      Interval::Abuts(Interval(Bits(), Bits()), Interval(Bits(), Bits())));
}

TEST(IntervalTest, ConvexHull) {
  Bits sixteen = Bits::PowerOfTwo(4, 6);
  Bits fifty_three = UBits(53, 6);
  EXPECT_EQ(Interval::ConvexHull(Interval(sixteen, sixteen),
                                 Interval(fifty_three, fifty_three)),
            Interval(sixteen, fifty_three));
  EXPECT_EQ(Interval::ConvexHull(Interval(fifty_three, fifty_three),
                                 Interval(sixteen, sixteen)),
            Interval(sixteen, fifty_three));
  EXPECT_EQ(
      Interval::ConvexHull(Interval(Bits(), Bits()), Interval(Bits(), Bits())),
      Interval(Bits(), Bits()));
}

TEST(IntervalSet, Intersect) {
  EXPECT_EQ(Interval::Intersect(SimpleInterval(20, 30), SimpleInterval(40, 50)),
            absl::nullopt);
  EXPECT_EQ(Interval::Intersect(SimpleInterval(20, 30), SimpleInterval(30, 40)),
            SimpleInterval(30, 30));
  EXPECT_EQ(Interval::Intersect(SimpleInterval(20, 30), SimpleInterval(25, 40)),
            SimpleInterval(25, 30));
  EXPECT_EQ(Interval::Intersect(SimpleInterval(25, 30), SimpleInterval(20, 40)),
            SimpleInterval(25, 30));
  EXPECT_EQ(Interval::Intersect(SimpleInterval(40, 50), SimpleInterval(20, 30)),
            absl::nullopt);
  EXPECT_EQ(Interval::Intersect(SimpleInterval(30, 40), SimpleInterval(20, 30)),
            SimpleInterval(30, 30));
  EXPECT_EQ(Interval::Intersect(SimpleInterval(25, 40), SimpleInterval(20, 30)),
            SimpleInterval(25, 30));
  EXPECT_EQ(Interval::Intersect(SimpleInterval(20, 40), SimpleInterval(25, 30)),
            SimpleInterval(25, 30));
}

TEST(IntervalSet, Difference) {
  EXPECT_EQ(
      Interval::Difference(SimpleInterval(20, 30), SimpleInterval(40, 50)),
      std::vector<Interval>{SimpleInterval(20, 30)});
  EXPECT_EQ(
      Interval::Difference(SimpleInterval(20, 30), SimpleInterval(20, 30)),
      std::vector<Interval>{});
  EXPECT_EQ(
      Interval::Difference(SimpleInterval(20, 30), SimpleInterval(25, 35)),
      std::vector<Interval>{SimpleInterval(20, 24)});
  EXPECT_EQ(
      Interval::Difference(SimpleInterval(20, 30), SimpleInterval(15, 25)),
      std::vector<Interval>{SimpleInterval(26, 30)});
  EXPECT_EQ(
      Interval::Difference(SimpleInterval(20, 30), SimpleInterval(24, 26)),
      (std::vector<Interval>{SimpleInterval(20, 23), SimpleInterval(27, 30)}));
}

TEST(IntervalSet, IsSubsetOf) {
  EXPECT_FALSE(
      Interval::IsSubsetOf(SimpleInterval(20, 30), SimpleInterval(40, 50)));
  EXPECT_FALSE(
      Interval::IsSubsetOf(SimpleInterval(20, 30), SimpleInterval(25, 50)));
  EXPECT_FALSE(
      Interval::IsSubsetOf(SimpleInterval(20, 50), SimpleInterval(30, 40)));
  EXPECT_TRUE(
      Interval::IsSubsetOf(SimpleInterval(20, 30), SimpleInterval(15, 50)));
}

TEST(IntervalTest, Elements) {
  Bits one = Bits::PowerOfTwo(0, 6);
  Bits four = Bits::PowerOfTwo(2, 6);
  Bits eight = Bits::PowerOfTwo(3, 6);

  std::vector<Bits> simple = Interval(four, eight).Elements();
  std::vector<Bits> simple_result{UBits(4, 6), UBits(5, 6), UBits(6, 6),
                                  UBits(7, 6), UBits(8, 6)};
  EXPECT_EQ(simple, simple_result);

  std::vector<Bits> zero_width = Interval(Bits(), Bits()).Elements();
  std::vector<Bits> zero_width_result{Bits()};
  EXPECT_EQ(zero_width, zero_width_result);

  std::vector<Bits> improper = Interval(UBits(62, 6), four).Elements();
  std::vector<Bits> improper_result{UBits(62, 6), UBits(63, 6), UBits(0, 6),
                                    UBits(1, 6),  UBits(2, 6),  UBits(3, 6),
                                    UBits(4, 6)};
  EXPECT_EQ(improper, improper_result);

  std::vector<Bits> early_return;
  EXPECT_TRUE(
      Interval(four, eight).ForEachElement([&](const Bits& bits) -> bool {
        if (bits_ops::UGreaterThan(bits, SumOf({four, one, one}))) {
          return true;
        }
        early_return.push_back(bits);
        return false;
      }));
  std::vector<Bits> early_return_result{
      four,
      SumOf({four, one}),
      SumOf({four, one, one}),
  };
  EXPECT_EQ(early_return, early_return_result);
}

TEST(IntervalTest, Size) {
  Bits one = Bits::PowerOfTwo(0, 160);
  Bits two = Bits::PowerOfTwo(1, 160);
  Bits four = Bits::PowerOfTwo(2, 160);
  Bits eight = Bits::PowerOfTwo(3, 160);
  Bits two_to_the_63 = Bits::PowerOfTwo(63, 160);
  Bits two_to_the_64 = Bits::PowerOfTwo(64, 160);
  Bits two_to_the_65 = Bits::PowerOfTwo(65, 160);

  EXPECT_EQ(Interval(Bits(), Bits()).Size(), 1);
  EXPECT_EQ(Interval(four, four).Size(), 1);
  EXPECT_EQ(Interval(eight, eight).Size(), 1);
  EXPECT_EQ(Interval(two_to_the_64, two_to_the_64).Size(), 1);
  EXPECT_EQ(Interval(two_to_the_65, two_to_the_65).Size(), 1);

  EXPECT_EQ(Interval(two_to_the_63, two_to_the_64).Size(), absl::nullopt);
  EXPECT_EQ(Interval(two_to_the_63, bits_ops::Sub(two_to_the_64, one)).Size(),
            absl::nullopt);
  EXPECT_EQ(Interval(two_to_the_63, bits_ops::Sub(two_to_the_64, two)).Size(),
            std::numeric_limits<int64_t>::max());

  EXPECT_EQ(Interval(UBits(8, 30), UBits(7, 30)).Size(), 1073741824);

  EXPECT_EQ(Interval(Bits::PowerOfTwo(3, 6), Bits::PowerOfTwo(2, 6)).Size(),
            61);
}

TEST(IntervalTest, IsImproper) {
  EXPECT_FALSE(Interval(Bits(), Bits()).IsImproper());
  EXPECT_FALSE(
      Interval(Bits::PowerOfTwo(2, 6), Bits::PowerOfTwo(3, 6)).IsImproper());
  EXPECT_TRUE(
      Interval(Bits::PowerOfTwo(3, 6), Bits::PowerOfTwo(2, 6)).IsImproper());
}

TEST(IntervalTest, IsPrecise) {
  Interval empty = Interval(Bits(), Bits());
  EXPECT_TRUE(empty.IsPrecise());
  EXPECT_THAT(empty.GetPreciseValue(), Optional(Bits()));

  Interval interval1 = Interval(Bits::PowerOfTwo(2, 6), Bits::PowerOfTwo(2, 6));
  EXPECT_TRUE(interval1.IsPrecise());
  EXPECT_THAT(interval1.GetPreciseValue(), Optional(Bits::PowerOfTwo(2, 6)));

  Interval interval2 = Interval(Bits::PowerOfTwo(2, 6), Bits::PowerOfTwo(3, 6));
  EXPECT_FALSE(interval2.IsPrecise());
  EXPECT_EQ(interval2.GetPreciseValue(), absl::nullopt);

  EXPECT_FALSE(
      Interval(Bits::PowerOfTwo(3, 6), Bits::PowerOfTwo(2, 6)).IsPrecise());
}

TEST(IntervalTest, IsMaximal) {
  EXPECT_TRUE(Interval(Bits(), Bits()).IsMaximal());
  EXPECT_FALSE(
      Interval(Bits::PowerOfTwo(2, 6), Bits::PowerOfTwo(2, 6)).IsMaximal());
  EXPECT_FALSE(
      Interval(Bits::PowerOfTwo(2, 6), Bits::PowerOfTwo(3, 6)).IsMaximal());
  EXPECT_FALSE(
      Interval(Bits::PowerOfTwo(3, 6), Bits::PowerOfTwo(2, 6)).IsMaximal());
  EXPECT_TRUE(Interval::Maximal(1).IsMaximal());
  EXPECT_TRUE(Interval::Maximal(2).IsMaximal());
  EXPECT_TRUE(Interval::Maximal(3).IsMaximal());
  EXPECT_TRUE(Interval::Maximal(6).IsMaximal());
  EXPECT_TRUE(Interval::Maximal(100).IsMaximal());
  EXPECT_TRUE(Interval::Maximal(1000).IsMaximal());
}

// Test the IsTrueWhenMaskWith with an interval starting at non-zero.
TEST(IntervalTest, NonZeroStartingValueIsTrueWhenMaskWith) {
  Interval interval(UBits(1, 4), UBits(4, 4));

  EXPECT_FALSE(interval.IsTrueWhenAndWith(UBits(0, 4)));
  for (int64_t value = 1; value < 8; ++value) {
    EXPECT_TRUE(interval.IsTrueWhenAndWith(UBits(value, 4)));
  }
  EXPECT_FALSE(interval.IsTrueWhenAndWith(UBits(8, 4)));
  for (int64_t value = 9; value < 16; ++value) {
    EXPECT_TRUE(interval.IsTrueWhenAndWith(UBits(value, 4)));
  }
}

// Test the IsTrueWhenMaskWith with an interval starting at zero.
TEST(IntervalTest, ZeroStartingValueIsTrueWhenMaskWith) {
  Interval interval(UBits(0, 3), UBits(4, 3));

  EXPECT_FALSE(interval.IsTrueWhenAndWith(UBits(0, 3)));
  EXPECT_FALSE(interval.IsTrueWhenAndWith(UBits(1, 3)));
  EXPECT_FALSE(interval.IsTrueWhenAndWith(UBits(2, 3)));
  EXPECT_FALSE(interval.IsTrueWhenAndWith(UBits(3, 3)));
  EXPECT_TRUE(interval.IsTrueWhenAndWith(UBits(4, 3)));
  EXPECT_TRUE(interval.IsTrueWhenAndWith(UBits(5, 3)));
  EXPECT_TRUE(interval.IsTrueWhenAndWith(UBits(6, 3)));
  EXPECT_TRUE(interval.IsTrueWhenAndWith(UBits(7, 3)));
}

// Test the IsTrueWhenMaskWith with an interval that does not overlap.
TEST(IntervalTest, NoOverlappingIntervalIsTrueWhenMaskWith) {
  Interval interval(UBits(4, 3), UBits(0, 3));

  for (int64_t value = 0; value < 8; ++value) {
    EXPECT_FALSE(interval.IsTrueWhenAndWith(UBits(value, 3)));
  }
}

// Test the IsTrueWhenMaskWith with an interval containing overlapping bits for
// its range.
TEST(IntervalTest, OverlappingBitsIsTrueWhenMaskWith) {
  Interval interval(UBits(2, 3), UBits(7, 3));

  EXPECT_FALSE(interval.IsTrueWhenAndWith(UBits(0, 3)));
  for (int64_t value = 1; value < 8; ++value) {
    EXPECT_TRUE(interval.IsTrueWhenAndWith(UBits(value, 3)));
  }
}

TEST(IntervalTest, Covers) {
  Bits thirty_two = Bits::PowerOfTwo(5, 12);
  Bits sixty_four = Bits::PowerOfTwo(6, 12);
  Interval interval(thirty_two, sixty_four);

  absl::flat_hash_set<Bits> covered_elements;
  interval.ForEachElement([&](const Bits& bits) -> bool {
    EXPECT_FALSE(covered_elements.contains(bits));
    covered_elements.insert(bits);
    return false;
  });

  absl::flat_hash_set<Bits> noncovered_elements;
  Interval(sixty_four, thirty_two)
      .ForEachElement([&](const Bits& bits) -> bool {
        if ((bits != thirty_two) && (bits != sixty_four)) {
          EXPECT_FALSE(noncovered_elements.contains(bits));
          noncovered_elements.insert(bits);
        }
        return false;
      });

  for (const Bits& element : covered_elements) {
    EXPECT_TRUE(interval.Covers(element));
  }

  for (const Bits& element : noncovered_elements) {
    EXPECT_FALSE(interval.Covers(element));
  }

  EXPECT_TRUE(Interval(Bits(), Bits()).Covers(Bits()));
}

TEST(IntervalTest, ToString) {
  EXPECT_EQ(Interval(UBits(4, 6), UBits(16, 6)).ToString(), "[4, 16]");
  EXPECT_EQ(Interval(Bits(), Bits()).ToString(), "[0, 0]");
}

TEST(IntervalTest, Complement) {
  EXPECT_THAT(
      Interval::Complement(SimpleInterval(0, 0)),
      ElementsAre(SimpleInterval(1, std::numeric_limits<uint32_t>::max())));
  EXPECT_THAT(
      Interval::Complement(SimpleInterval(1, 2)),
      ElementsAre(SimpleInterval(0, 0),
                  SimpleInterval(3, std::numeric_limits<uint32_t>::max())));
  EXPECT_THAT(
      Interval::Complement(SimpleInterval(0, 42)),
      ElementsAre(SimpleInterval(43, std::numeric_limits<uint32_t>::max())));
  EXPECT_THAT(Interval::Complement(
                  SimpleInterval(1234, std::numeric_limits<uint32_t>::max())),
              ElementsAre(SimpleInterval(0, 1233)));
  EXPECT_THAT(Interval::Complement(
                  SimpleInterval(0, std::numeric_limits<uint32_t>::max())),
              ElementsAre());
}

TEST(IntervalTest, ZeroExtend) {
  Interval original(UBits(4, 6), UBits(16, 6));
  Interval extended = original.ZeroExtend(30);
  EXPECT_EQ(original.ToString(), "[4, 16]");
  EXPECT_EQ(extended.ToString(), "[4, 16]");
  EXPECT_EQ(extended.BitCount(), 30);
}

}  // namespace
}  // namespace xls
