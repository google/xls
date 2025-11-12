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

#include "xls/ir/partial_information.h"

#include <cstdint>
#include <optional>
#include <ostream>
#include <tuple>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_ops.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/interval_set_test_utils.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/ternary.h"

namespace xls {
namespace {

using ::testing::Optional;
using ::testing::ResultOf;

TEST(PartialInformationTest, FromTernary) {
  PartialInformation info(*StringToTernaryVector("0b11_01X1_X001"));
  ASSERT_THAT(info.Ternary(),
              Optional(ResultOf([](TernarySpan t) { return xls::ToString(t); },
                                "0b11_01X1_X001")));
}

void PartialInfoFromTernaryIsCorrect(TernarySpan ternary) {
  PartialInformation info(ternary);
  if (absl::c_all_of(ternary, [](TernaryValue v) {
        return v == TernaryValue::kUnknown;
      })) {
    ASSERT_THAT(info.Ternary(), std::nullopt);
  } else {
    ASSERT_THAT(info.Ternary(), Optional(ternary));
  }
  ASSERT_FALSE(info.IsImpossible());
}
FUZZ_TEST(PartialInformationFuzzTest, PartialInfoFromTernaryIsCorrect)
    .WithDomains(fuzztest::VectorOf(fuzztest::ElementOf({
        TernaryValue::kKnownZero,
        TernaryValue::kKnownOne,
        TernaryValue::kUnknown,
    })));

TEST(PartialInformationTest, FromIntervals) {
  PartialInformation info(FromRanges({{0, 1}, {2, 3}}, 4));
  ASSERT_THAT(info.Range(), Optional(FromRanges({{0, 1}, {2, 3}}, 4)));
}

void PartialInfoFromIntervalsIsCorrect(const IntervalSet& intervals) {
  PartialInformation info(intervals);

  IntervalSet expected_intervals = intervals;
  expected_intervals.Normalize();
  if (expected_intervals.IsMaximal()) {
    ASSERT_THAT(info.Range(), std::nullopt);
  } else {
    ASSERT_THAT(info.Range(), Optional(expected_intervals));
  }
  ASSERT_EQ(expected_intervals.IsEmpty(), info.IsImpossible());
}
FUZZ_TEST(PartialInformationFuzzTest, PartialInfoFromIntervalsIsCorrect)
    .WithDomains(ArbitraryIntervalSet(32));

TEST(PartialInformationTest, Mixed) {
  PartialInformation contradiction(
      *StringToTernaryVector("0b11_01X1_X001"),
      FromRanges({{0, 10}, {10, 20}, {20, 30}, {30, 40}}, 10));
  EXPECT_TRUE(contradiction.IsImpossible());
  EXPECT_THAT(contradiction.Ternary(), std::nullopt);
  EXPECT_THAT(contradiction.Range(), Optional(IntervalSet(10)));

  PartialInformation info(*StringToTernaryVector("0b00_00XX_X01X"),
                          FromRanges({{0, 10}, {10, 20}, {20, 30}}, 10));
  EXPECT_FALSE(info.IsImpossible());
  EXPECT_THAT(info.Ternary(),
              Optional(ResultOf([](TernarySpan t) { return xls::ToString(t); },
                                "0b00_000X_X01X")));
  EXPECT_THAT(info.Range(),
              Optional(FromRanges({{2, 3}, {10, 11}, {18, 19}, {26, 27}}, 10)));
}

inline auto ArbitraryTernaryAndNormalizedIntervalSet() {
  return fuzztest::FlatMap(
      [](int64_t bit_count) {
        return fuzztest::TupleOf(
            fuzztest::OptionalOf(
                fuzztest::VectorOf(
                    fuzztest::ElementOf({TernaryValue::kKnownZero,
                                         TernaryValue::kKnownOne,
                                         TernaryValue::kUnknown}))
                    .WithSize(bit_count)),
            fuzztest::OptionalOf(ArbitraryNormalizedIntervalSet(bit_count)),
            fuzztest::ElementOf({bit_count}));
      },
      fuzztest::InRange(0, 1000));
}

void ImpossibleAndUnconstrained(std::tuple<std::optional<TernaryVector>,
                                           std::optional<IntervalSet>, int64_t>
                                    inputs) {
  const auto& [ternary, range, bit_count] = inputs;
  PartialInformation info(bit_count, ternary, range);
  if (range.has_value() && ternary.has_value()) {
    EXPECT_EQ(info.IsImpossible(),
              !interval_ops::CoversTernary(*range, *ternary))
        << "Ternary: " << xls::ToString(*ternary)
        << ", range: " << range->ToString()
        << ", impossible: " << info.IsImpossible();
  } else if (range.has_value()) {
    EXPECT_EQ(info.IsImpossible(), range->IsEmpty());
  } else if (ternary.has_value()) {
    EXPECT_FALSE(info.IsImpossible());
  }
  EXPECT_EQ(info.IsUnconstrained(),
            (!range.has_value() || range->IsMaximal()) &&
                (!ternary.has_value() || ternary_ops::AllUnknown(*ternary)));
}
FUZZ_TEST(PartialInformationFuzzTest, ImpossibleAndUnconstrained)
    .WithDomains(ArbitraryTernaryAndNormalizedIntervalSet());

void LeadingBitsIsCorrect(std::tuple<std::optional<TernaryVector>,
                                     std::optional<IntervalSet>, int64_t>
                              inputs) {
  const auto& [ternary, range, bit_count] = inputs;
  PartialInformation info(bit_count, ternary, range);
  if (info.IsImpossible()) {
    EXPECT_EQ(info.KnownLeadingOnes(), 0);
    EXPECT_EQ(info.KnownLeadingZeros(), 0);
    EXPECT_EQ(info.KnownLeadingSignBits(), 0);
  } else if (info.IsUnconstrained()) {
    EXPECT_EQ(info.KnownLeadingOnes(), 0);
    EXPECT_EQ(info.KnownLeadingZeros(), 0);
    EXPECT_EQ(info.KnownLeadingSignBits(), 1);
  } else if (info.KnownLeadingZeros() != 0) {
    EXPECT_EQ(info.KnownLeadingSignBits(), info.KnownLeadingZeros());
    EXPECT_EQ(info.KnownLeadingOnes(), 0);
    ASSERT_TRUE(info.Ternary().has_value())
        << info << ": Failed to unify even though some bits are known";
    EXPECT_EQ(
        info.KnownLeadingZeros(),
        ternary_ops::ToKnownBitsValues(*info.Ternary(), /*default_set=*/true)
            .CountLeadingZeros())
        << info;
  } else if (info.KnownLeadingOnes()) {
    EXPECT_EQ(info.KnownLeadingSignBits(), info.KnownLeadingOnes());
    EXPECT_EQ(info.KnownLeadingZeros(), 0);
    ASSERT_TRUE(info.Ternary().has_value())
        << info << ": Failed to unify even though some bits are known";
    EXPECT_EQ(
        info.KnownLeadingOnes(),
        ternary_ops::ToKnownBitsValues(*info.Ternary(), /*default_set=*/false)
            .CountLeadingOnes())
        << info;
  }
}

FUZZ_TEST(PartialInformationFuzzTest, LeadingBitsIsCorrect)
    .WithDomains(ArbitraryTernaryAndNormalizedIntervalSet());

// Detected by fuzzing; this regression test is a combination of a ternary and
// an interval set that is extremely close to being compatible, to the point
// where our original reconciliation logic failed due to imprecision in
// converting ternaries to intervals.
TEST(PartialInformationTest, ExtremelyCloseImpossibleCombination) {
  XLS_ASSERT_OK_AND_ASSIGN(
      TernaryVector ternary,
      StringToTernaryVector(
          "0b1_1X11_X0XX_XX11_X000_X110_XX1X_X001_1XX0_1XX1_X1X0_10X1_X01X_"
          "1001_XX0X_X101_XX1X_0XX1_1XXX_X0XX_X011_00XX_X1XX_XX0X_1X1X_10X0_"
          "000X_XX10_1000_0111_X000_1XXX_00X1_X100_0011_XX10_0X10_11X0_101X_"
          "X1X1_1X01_0110_1101_101X_110X_XXX1_01X0_010X_1X11_000X_0110_X0X1_"
          "11XX_X110_X01X_0011_X1X1_XX01_0XXX_100X_X1X0_X00X_XX11_1011_1XXX_"
          "X11X_0X10_XX11_XX01_X01X_10XX_01X1_0100_0101_010X_X111_1XX1_0XX1_"
          "X011_XX01_10X1_0X00_X1XX_X011_000X_0010_XX10_XXX1_0X0X_1X10_X1X1_"
          "XX01_11XX_1X00_X1X1_1X00_0X01_XXX1_0XX0_1010_X0XX_0X1X_0101_0101_"
          "1011_XX01_XXX0_0100_01XX_X101_1001_01XX_1100_1X11_1110_0X00_XX01_"
          "1100_11X0_01X1_10X0_111X_0100_X1X1_0X11_0X0X_0X11_010X_10XX_1X10_"
          "00XX_XX01_XX10_XX1X_1X00_X000_10X1_10X1_XXXX_0011_X110_XX1X_X101_"
          "100X_0X00_101X_X111_0X10_X001_1X11_0000_0010_1010_11X0_X001"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits lb,
      ParseNumber(
          "0xb3_d583_cb84_d899_fc0a_311f_28c6_472f_f87f_6689_ca89_1d26_88ff_"
          "81a0_26b1_21b9_ae5a_389b_04aa_ab09_0f7c_9d56_e7d6_6406_7b69_e491_"
          "f7d8_e132_bc36_047f_97e6_0d59_58ba_c2ed_5436_fc83_f529_75d3_e54e"));
  lb = bits_ops::ZeroExtend(lb, ternary.size());
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits ub,
      ParseNumber(
          "0x1b0_222b_b1c2_27b3_b2c4_055a_d9ad_d275_6ee9_1cc1_f80b_0215_8b91_"
          "f067_f2b8_2539_7d04_abe1_983a_2ddb_7cfb_cf36_d415_93ba_02d8_5fb2_"
          "9b9e_76f1_38fd_92bb_8ad5_edec_58bf_1a04_8a0c_9265_c70d_516c_a6e4"));
  ub = bits_ops::ZeroExtend(ub, ternary.size());
  IntervalSet range = IntervalSet::Of({Interval(lb, ub)});

  EXPECT_FALSE(interval_ops::CoversTernary(range, ternary));

  PartialInformation info(ternary, range);
  EXPECT_TRUE(info.IsImpossible());
}

// Found by fuzzing; shifting left by an unknown amount bounded away from zero
// should identify that some trailing bits are zero, while shifting right by an
// unknown amount bounded away from zero should correctly identify that some
// leading bits are zero. (Previous bug caused us to instead claim that some
// trailing bits were zero in the right-shift case too.)
TEST(PartialInformationTest, Shifts) {
  PartialInformation x = PartialInformation::Precise(Bits::AllOnes(128));
  PartialInformation y(IntervalSet::Of({Interval(UBits(2, 8), UBits(255, 8))}));

  TernaryVector expected_shll(128, TernaryValue::kUnknown);
  expected_shll[0] = TernaryValue::kKnownZero;
  expected_shll[1] = TernaryValue::kKnownZero;
  EXPECT_EQ(x.Shll(y), PartialInformation(expected_shll));

  TernaryVector expected_shrl(128, TernaryValue::kUnknown);
  expected_shrl[127] = TernaryValue::kKnownZero;
  expected_shrl[126] = TernaryValue::kKnownZero;
  EXPECT_EQ(x.Shrl(y), PartialInformation(expected_shrl));
}

TEST(PartialInformationTest, LeadingOnes) {
  PartialInformation x = PartialInformation(
      IntervalSet::Of({Interval(UBits(0b11110000, 8), UBits(0b11111000, 8)),
                       Interval(UBits(0b11100000, 8), UBits(0b11100011, 8))}));
  EXPECT_EQ(x.KnownLeadingSignBits(), 3);
  EXPECT_EQ(x.KnownLeadingZeros(), 0);
  EXPECT_EQ(x.KnownLeadingOnes(), 3);
}

TEST(PartialInformationTest, LeadingZero) {
  PartialInformation x = PartialInformation(
      IntervalSet::Of({Interval(UBits(0b00000111, 8), UBits(0b00001111, 8)),
                       Interval(UBits(0b00011100, 8), UBits(0b00011111, 8))}));
  EXPECT_EQ(x.KnownLeadingSignBits(), 3);
  EXPECT_EQ(x.KnownLeadingZeros(), 3);
  EXPECT_EQ(x.KnownLeadingOnes(), 0);
}

TEST(PartialInformationTest, LeadingSign) {
  PartialInformation x = PartialInformation(
      IntervalSet::Of({Interval(UBits(0b00000111, 8), UBits(0b00001111, 8)),
                       Interval(UBits(0b00011100, 8), UBits(0b00011111, 8)),
                       Interval(UBits(0b11110000, 8), UBits(0b11111000, 8)),
                       Interval(UBits(0b11100000, 8), UBits(0b11100011, 8))}));
  EXPECT_EQ(x.KnownLeadingSignBits(), 3);
  EXPECT_EQ(x.KnownLeadingZeros(), 0);
  EXPECT_EQ(x.KnownLeadingOnes(), 0);
}

}  // namespace
}  // namespace xls
