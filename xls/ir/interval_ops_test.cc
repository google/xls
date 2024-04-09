// Copyright 2023 The XLS Authors
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

#include "xls/ir/interval_ops.h"

#include <cstdint>
#include <string_view>
#include <utility>

#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/interval.h"
#include "xls/ir/interval_set.h"
#include "xls/ir/ternary.h"

namespace xls::interval_ops {

namespace {

IntervalSet SetOf(absl::Span<const Interval> intervals) {
  IntervalSet is(intervals.front().BitCount());
  absl::c_for_each(intervals, [&](auto v) { is.AddInterval(v); });
  is.Normalize();
  return is;
}

TEST(IntervalOpsTest, BitsPrecise) {
  IntervalSet is = SetOf({Interval::Precise(UBits(21, 8))});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, Bits::AllOnes(8));
  EXPECT_EQ(known.known_bit_values, UBits(21, 8));
}

TEST(IntervalOpsTest, BitsMaximal) {
  IntervalSet is = SetOf({Interval::Maximal(8)});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, Bits(8));
  EXPECT_EQ(known.known_bit_values, Bits(8));
}

TEST(IntervalOpsTest, BitsHalfFull) {
  IntervalSet is = SetOf({Interval::Maximal(4).ZeroExtend(8)});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, UBits(0xf0, 8));
  EXPECT_EQ(known.known_bit_values, Bits(8));
}

TEST(IntervalOpsTest, MiddleOut) {
  IntervalSet is = SetOf({Interval(UBits(0, 8), UBits(0x4, 8)),
                          Interval(UBits(0x10, 8), UBits(0x14, 8))});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, UBits(0xe8, 8));
  EXPECT_EQ(known.known_bit_values, Bits(8));
}

TEST(IntervalOpsTest, MiddleOutHigh) {
  IntervalSet is = SetOf({Interval(UBits(0xe0, 8), UBits(0xe4, 8)),
                          Interval(UBits(0xf0, 8), UBits(0xf4, 8))});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, UBits(0xe8, 8));
  EXPECT_EQ(known.known_bit_values, UBits(0xe0, 8));
}

TEST(IntervalOpsTest, MiddleOutTernary) {
  IntervalSet is = SetOf({Interval(UBits(0, 8), UBits(0x4, 8)),
                          Interval(UBits(0x10, 8), UBits(0x14, 8))});
  auto known = ExtractTernaryVector(is);
  TernaryVector expected{
      TernaryValue::kUnknown,   TernaryValue::kUnknown,
      TernaryValue::kUnknown,   TernaryValue::kKnownZero,
      TernaryValue::kUnknown,   TernaryValue::kKnownZero,
      TernaryValue::kKnownZero, TernaryValue::kKnownZero,
  };
  EXPECT_EQ(known, expected);
}

IntervalSet FromRanges(absl::Span<std::pair<int64_t, int64_t> const> ranges,
                       int64_t bits) {
  IntervalSet res(bits);
  for (const auto& [l, h] : ranges) {
    res.AddInterval(Interval::Closed(UBits(l, bits), UBits(h, bits)));
  }
  res.Normalize();
  return res;
}

IntervalSet FromTernaryString(std::string_view sv,
                              int64_t max_unknown_bits = 4) {
  auto tern_status = StringToTernaryVector(sv);
  if (!tern_status.ok()) {
    ADD_FAILURE() << "Unable to parse ternary string " << sv << "\n"
                  << tern_status.status();
    return IntervalSet(1);
  }
  return interval_ops::FromTernary(tern_status.value(), max_unknown_bits);
}

TEST(IntervalOpsTest, FromTernaryExact) {
  EXPECT_EQ(FromTernaryString("0b111000"),
            IntervalSet::Precise(UBits(0b111000, 6)));
}

TEST(IntervalOpsTest, FromTernaryAllUnknown) {
  EXPECT_EQ(FromTernaryString("0bXXXXXX"), IntervalSet::Maximal(6));
}

TEST(IntervalOpsTest, FromTernaryUnknownTrailing) {
  EXPECT_EQ(FromTernaryString("0b1010XXX"),
            FromRanges({{0b1010000, 0b1010111}}, 7));
}

TEST(IntervalOpsTest, FromTernarySegments) {
  EXPECT_EQ(FromTernaryString("0bXX1010XXX"),
            FromRanges({{0b001010000, 0b001010111},
                        {0b011010000, 0b011010111},
                        {0b101010000, 0b101010111},
                        {0b111010000, 0b111010111}},
                       9));
  EXPECT_EQ(
      FromTernaryString("0b0X1010XXX"),
      FromRanges({{0b001010000, 0b001010111}, {0b011010000, 0b011010111}}, 9));
  EXPECT_EQ(FromTernaryString("0b1X0X1010XXX"),
            FromRanges({{0b10001010000, 0b10001010111},
                        {0b10011010000, 0b10011010111},
                        {0b11001010000, 0b11001010111},
                        {0b11011010000, 0b11011010111}},
                       11));
}
TEST(IntervalOpsTest, FromTernaryPreciseSegments) {
  XLS_ASSERT_OK_AND_ASSIGN(auto tern, StringToTernaryVector("0b1X0X1X0X1"));
  IntervalSet expected(tern.size());
  for (const Bits& v : ternary_ops::AllBitsValues(tern)) {
    expected.AddInterval(Interval::Precise(v));
  }
  expected.Normalize();
  EXPECT_EQ(interval_ops::FromTernary(tern, /*max_interval_bits=*/4), expected);
}
TEST(IntervalOpsTest, FromTernaryPreciseSegmentsBig) {
  XLS_ASSERT_OK_AND_ASSIGN(auto tern,
                           StringToTernaryVector("0b1X0X1XXXXXXXXX0X1"));
  IntervalSet expected(tern.size());
  for (const Bits& v : ternary_ops::AllBitsValues(tern)) {
    expected.AddInterval(Interval::Precise(v));
  }
  expected.Normalize();
  EXPECT_EQ(interval_ops::FromTernary(tern, /*max_interval_bits=*/12),
            expected);
}

TEST(IntervalOpsTest, FromTernarySegmentsExtended) {
  // Only allow 4 segments so first 5 bits are all considered unknown
  EXPECT_EQ(FromTernaryString("0bXX_1X_0X0X", /*max_unknown_bits=*/2),
            FromRanges({{0b00100000, 0b00111111},
                        {0b01100000, 0b01111111},
                        {0b10100000, 0b10111111},
                        {0b11100000, 0b11111111}},
                       8));
  // Only allow 2 segments
  EXPECT_EQ(
      FromTernaryString("0b1X1_X1X0X", /*max_unknown_bits=*/1),
      FromRanges({{0b10100000, 0b10111111}, {0b11100000, 0b11111111}}, 8));
  // Only allow 1 segment
  EXPECT_EQ(FromTernaryString("0b1X1_X1X0X", /*max_unknown_bits=*/0),
            FromRanges({{0b10000000, 0b11111111}}, 8));
}

}  // namespace
}  // namespace xls::interval_ops
