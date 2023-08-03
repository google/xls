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

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/types/span.h"
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
  EXPECT_EQ(known.known_bits, UBits(0xe0, 8));
  EXPECT_EQ(known.known_bit_values, Bits(8));
}
TEST(IntervalOpsTest, MiddleOutHigh) {
  IntervalSet is = SetOf({Interval(UBits(0xe0, 8), UBits(0xe4, 8)),
                          Interval(UBits(0xf0, 8), UBits(0xf4, 8))});
  auto known = ExtractKnownBits(is);
  EXPECT_EQ(known.known_bits, UBits(0xe0, 8));
  EXPECT_EQ(known.known_bit_values, UBits(0xe0, 8));
}
TEST(IntervalOpsTest, MiddleOutTernary) {
  IntervalSet is = SetOf({Interval(UBits(0, 8), UBits(0x4, 8)),
                          Interval(UBits(0x10, 8), UBits(0x14, 8))});
  auto known = ExtractTernaryVector(is);
  TernaryVector expected{
      TernaryValue::kUnknown,   TernaryValue::kUnknown,
      TernaryValue::kUnknown,   TernaryValue::kUnknown,
      TernaryValue::kUnknown,   TernaryValue::kKnownZero,
      TernaryValue::kKnownZero, TernaryValue::kKnownZero,
  };
  EXPECT_EQ(known, expected);
}

}  // namespace
}  // namespace xls::interval_ops
