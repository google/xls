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

#include "xls/ir/partial_ops.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xls/common/math_util.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/bits_test_utils.h"
#include "xls/ir/interval_set_test_utils.h"
#include "xls/ir/partial_information.h"
#include "xls/ir/partial_information_test_utils.h"
#include "xls/ir/ternary.h"

namespace xls::partial_ops {
namespace {

PartialInformation Partial(std::string ternary) {
  return PartialInformation(*StringToTernaryVector(ternary));
}

PartialInformation Partial(
    std::string ternary, absl::Span<const std::pair<int64_t, int64_t>> ranges) {
  TernaryVector t = *StringToTernaryVector(ternary);
  return PartialInformation(t, FromRanges(ranges, t.size()));
}

PartialInformation Partial(absl::Span<const std::pair<int64_t, int64_t>> ranges,
                           int64_t bit_count) {
  return PartialInformation(FromRanges(ranges, bit_count));
}

TEST(PartialOpsTest, Join) {
  PartialInformation a =
      Partial("0b11_01X1_X001", {{0, 8}, {10, 18}, {20, 28}});
  PartialInformation b =
      Partial("0b00_00XX_X01X", {{0, 8}, {10, 18}, {20, 28}});
  PartialInformation c = Partial("0b00_00XX_10XX", {{0, 20}});
  PartialInformation d = Partial("0bXX_XXXX_1XXX", {{0, 1023}});

  EXPECT_EQ(Join(a, a), a);
  EXPECT_EQ(Join(a, PartialInformation::Impossible(10)),
            PartialInformation::Impossible(10));
  EXPECT_EQ(Join(a, PartialInformation::Unconstrained(10)), a);

  EXPECT_EQ(Join(b, b), b);
  EXPECT_EQ(Join(b, PartialInformation::Impossible(10)),
            PartialInformation::Impossible(10));
  EXPECT_EQ(Join(b, PartialInformation::Unconstrained(10)), b);

  EXPECT_EQ(Join(a, b), PartialInformation::Impossible(10));

  EXPECT_EQ(Join(b, c), Partial("0b00_0000_101X", {{10, 11}}));

  EXPECT_EQ(Join(a, d), PartialInformation::Impossible(10));
}

void JoinOfCompatibleInformationIsCompatible(const Bits& bits,
                                             PartialInformation a,
                                             PartialInformation b) {
  PartialInformation point = PartialInformation::Precise(bits);
  EXPECT_TRUE(
      Join(a.MeetWith(point), b.MeetWith(point)).IsCompatibleWith(bits));
}
FUZZ_TEST(PartialOpsFuzzTest, JoinOfCompatibleInformationIsCompatible)
    .WithDomains(ArbitraryBits(8), ArbitraryPartialInformation(8),
                 ArbitraryPartialInformation(8));

TEST(PartialOpsTest, Meet) {
  PartialInformation a =
      Partial("0b11_01X1_X001", {{0, 8}, {10, 18}, {20, 28}});
  PartialInformation b =
      Partial("0b00_00XX_X01X", {{0, 8}, {10, 18}, {20, 28}});
  PartialInformation c = Partial("0b00_00XX_10XX", {{0, 20}});

  EXPECT_EQ(Meet(a, a), a);
  EXPECT_EQ(Meet(a, PartialInformation::Impossible(10)), a);
  EXPECT_EQ(Meet(a, PartialInformation::Unconstrained(10)),
            PartialInformation::Unconstrained(10));

  EXPECT_EQ(Meet(b, b), b);
  EXPECT_EQ(Meet(b, PartialInformation::Impossible(10)), b);
  EXPECT_EQ(Meet(b, PartialInformation::Unconstrained(10)),
            PartialInformation::Unconstrained(10));

  EXPECT_EQ(Meet(a, b),
            Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 18}, {26, 27}}));

  EXPECT_EQ(Meet(b, c),
            Partial("0b00_000X_X0XX", {{2, 3}, {8, 11}, {18, 18}, {26, 27}}));

  EXPECT_EQ(Meet(Partial("0b0X_XXXX_XXXX", {{0, 513}}),
                 Partial("0b1X_XXXX_XXXX", {{257, 1023}})),
            PartialInformation::Unconstrained(10));
}

void MeetIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b) {
  EXPECT_TRUE(Meet(a.partial, b.partial).IsCompatibleWith(a.bits));
  EXPECT_TRUE(Meet(a.partial, b.partial).IsCompatibleWith(b.bits));
}
FUZZ_TEST(PartialOpsFuzzTest, MeetIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

TEST(PartialOpsTest, Not) {
  EXPECT_EQ(Not(PartialInformation::Impossible(10)),
            PartialInformation::Impossible(10));
  EXPECT_EQ(Not(PartialInformation::Unconstrained(10)),
            PartialInformation::Unconstrained(10));

  EXPECT_EQ(
      Not(Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 18}, {26, 27}})),
      Partial("0b11_111X_X10X",
              {{996, 997}, {1004, 1005}, {1012, 1013}, {1020, 1021}}));
  EXPECT_EQ(Not(Partial("0b00_000X_10XX", {{8, 11}})),
            Partial("0b11_1111_01XX", {{1012, 1015}}));
  EXPECT_EQ(Not(Partial("0bXX_XXXX_1XXX")), Partial("0bXX_XXXX_0XXX"));
}

void NotIsCompatible(const PartiallyDescribedBits& a) {
  EXPECT_TRUE(Not(a.partial).IsCompatibleWith(bits_ops::Not(a.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, NotIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8));

TEST(PartialOpsTest, And) {
  PartialInformation a =
      Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 18}, {26, 27}});
  PartialInformation b = Partial("0b00_000X_10XX", {{8, 11}});
  PartialInformation c = Partial("0bXX_XXXX_1XXX");
  PartialInformation d = PartialInformation::Impossible(10);
  PartialInformation e = PartialInformation::Unconstrained(10);

  EXPECT_EQ(And(a, a),
            Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 19}, {26, 27}}));
  EXPECT_EQ(And(b, b), b);
  EXPECT_EQ(And(c, c), c);
  EXPECT_EQ(And(d, d), d);
  EXPECT_EQ(And(e, e), e);

  EXPECT_EQ(And(a, b), Partial("0b00_0000_X0XX", {{0, 3}, {8, 11}}));
  EXPECT_EQ(And(a, c),
            Partial("0b00_000X_X0XX", {{0, 3}, {8, 11}, {16, 19}, {24, 27}}));
  EXPECT_EQ(And(a, d), PartialInformation::Impossible(10));
  EXPECT_EQ(And(a, e),
            Partial("0b00_000X_X0XX", {{0, 3}, {8, 11}, {16, 19}, {24, 27}}));

  EXPECT_EQ(And(b, c), b);
  EXPECT_EQ(And(b, d), PartialInformation::Impossible(10));
  EXPECT_EQ(And(b, e), Partial("0b00_0000_X0XX", {{0, 3}, {8, 11}}));

  EXPECT_EQ(And(c, d), PartialInformation::Impossible(10));
  EXPECT_EQ(And(c, e), PartialInformation::Unconstrained(10));

  EXPECT_EQ(And(d, e), PartialInformation::Impossible(10));
}

void AndIsCompatible(const PartiallyDescribedBits& a,
                     const PartiallyDescribedBits& b) {
  EXPECT_TRUE(And(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::And(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, AndIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

TEST(PartialOpsTest, Or) {
  PartialInformation a =
      Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 18}, {26, 27}});
  PartialInformation b = Partial("0b00_000X_10XX", {{8, 11}});
  PartialInformation c = Partial("0bXX_XXXX_1XXX");
  PartialInformation d = PartialInformation::Impossible(10);
  PartialInformation e = PartialInformation::Unconstrained(10);

  EXPECT_EQ(Or(a, a),
            Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 19}, {26, 27}}));
  EXPECT_EQ(Or(b, b), b);
  EXPECT_EQ(Or(c, c), c);
  EXPECT_EQ(Or(d, d), d);
  EXPECT_EQ(Or(e, e), e);

  EXPECT_EQ(Or(a, b), Partial("0b00_000X_101X", {{10, 11}, {26, 27}}));
  EXPECT_EQ(Or(a, c), Partial("0bXX_XXXX_1X1X"));
  EXPECT_EQ(Or(a, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Or(a, e), Partial("0bXX_XXXX_XX1X"));

  EXPECT_EQ(Or(b, c), Partial("0bXX_XXXX_1XXX"));
  EXPECT_EQ(Or(b, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Or(b, e), Partial("0bXX_XXXX_1XXX"));

  EXPECT_EQ(Or(c, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Or(c, e), Partial("0bXX_XXXX_1XXX"));

  EXPECT_EQ(Or(d, e), PartialInformation::Impossible(10));
}

void OrIsCompatible(const PartiallyDescribedBits& a,
                    const PartiallyDescribedBits& b) {
  EXPECT_TRUE(
      Or(a.partial, b.partial).IsCompatibleWith(bits_ops::Or(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, OrIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

TEST(PartialOpsTest, Xor) {
  PartialInformation a =
      Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 18}, {26, 27}});
  PartialInformation b = Partial("0b00_000X_10XX", {{8, 11}});
  PartialInformation c = Partial("0bXX_XXXX_1XXX");
  PartialInformation d = PartialInformation::Impossible(10);
  PartialInformation e = PartialInformation::Unconstrained(10);

  EXPECT_EQ(Xor(a, a),
            Partial("0b00_000X_X00X", {{0, 1}, {8, 9}, {16, 17}, {24, 25}}));
  EXPECT_EQ(Xor(b, b), Partial("0b00_0000_00XX", {{0, 3}}));
  EXPECT_EQ(Xor(c, c), Partial("0bXX_XXXX_0XXX"));
  EXPECT_EQ(Xor(d, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Xor(e, e), PartialInformation::Unconstrained(10));

  EXPECT_EQ(Xor(a, b),
            Partial("0b00_000X_X0XX", {{0, 3}, {8, 11}, {16, 19}, {24, 27}}));
  EXPECT_EQ(Xor(a, c), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Xor(a, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Xor(a, e), PartialInformation::Unconstrained(10));

  EXPECT_EQ(Xor(b, c), Partial("0bXX_XXXX_0XXX"));
  EXPECT_EQ(Xor(b, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Xor(b, e), PartialInformation::Unconstrained(10));

  EXPECT_EQ(Xor(c, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Xor(c, e), PartialInformation::Unconstrained(10));

  EXPECT_EQ(Xor(d, e), PartialInformation::Impossible(10));
}

void XorIsCompatible(const PartiallyDescribedBits& a,
                     const PartiallyDescribedBits& b) {
  EXPECT_TRUE(Xor(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::Xor(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, XorIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void NandIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b) {
  EXPECT_TRUE(Nand(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::Nand(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, NandIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void NorIsCompatible(const PartiallyDescribedBits& a,
                     const PartiallyDescribedBits& b) {
  EXPECT_TRUE(Nor(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::Nor(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, NorIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void AndReduceIsCompatible(const PartiallyDescribedBits& a) {
  EXPECT_TRUE(
      AndReduce(a.partial).IsCompatibleWith(bits_ops::AndReduce(a.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, AndReduceIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8));

void OrReduceIsCompatible(const PartiallyDescribedBits& a) {
  EXPECT_TRUE(OrReduce(a.partial).IsCompatibleWith(bits_ops::OrReduce(a.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, OrReduceIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8));

void XorReduceIsCompatible(const PartiallyDescribedBits& a) {
  EXPECT_TRUE(
      XorReduce(a.partial).IsCompatibleWith(bits_ops::XorReduce(a.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, XorReduceIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8));

void ConcatIsCompatible(std::vector<PartiallyDescribedBits> values) {
  std::vector<PartialInformation> partials;
  std::vector<Bits> bits;
  for (PartiallyDescribedBits& value : values) {
    partials.push_back(std::move(value.partial));
    bits.push_back(std::move(value.bits));
  }
  EXPECT_TRUE(Concat(partials).IsCompatibleWith(bits_ops::Concat(bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, ConcatIsCompatible)
    .WithDomains(fuzztest::VectorOf(ArbitraryPartiallyDescribedBits(8)));

void SignExtendIsCompatible(const PartiallyDescribedBits& a,
                            int64_t additional_width) {
  int64_t new_bit_count = a.bits.bit_count() + additional_width;
  EXPECT_TRUE(
      SignExtend(a.partial, new_bit_count)
          .IsCompatibleWith(bits_ops::SignExtend(a.bits, new_bit_count)));
}
FUZZ_TEST(PartialOpsFuzzTest, SignExtendIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 fuzztest::InRange<int64_t>(0, 1000));

void ZeroExtendIsCompatible(const PartiallyDescribedBits& a,
                            int64_t additional_width) {
  int64_t new_bit_count = a.bits.bit_count() + additional_width;
  EXPECT_TRUE(
      ZeroExtend(a.partial, new_bit_count)
          .IsCompatibleWith(bits_ops::ZeroExtend(a.bits, new_bit_count)));
}
FUZZ_TEST(PartialOpsFuzzTest, ZeroExtendIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 fuzztest::InRange<int64_t>(0, 1000));

void TruncateIsCompatible(const PartiallyDescribedBits& a, int64_t width) {
  width = std::min(width, a.bits.bit_count());
  EXPECT_TRUE(Truncate(a.partial, width)
                  .IsCompatibleWith(bits_ops::Truncate(a.bits, width)));
}
FUZZ_TEST(PartialOpsFuzzTest, TruncateIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 fuzztest::InRange<int64_t>(0, 8));

void BitSliceIsCompatible(const PartiallyDescribedBits& a, int64_t start,
                          int64_t width) {
  start = std::min(start, a.bits.bit_count() - 1);
  width = std::min(width, a.bits.bit_count() - start);
  EXPECT_TRUE(BitSlice(a.partial, start, width)
                  .IsCompatibleWith(a.bits.Slice(start, width)));
}
FUZZ_TEST(PartialOpsFuzzTest, BitSliceIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 fuzztest::InRange<int64_t>(0, 8),
                 fuzztest::InRange<int64_t>(0, 8));

void DynamicBitSliceIsCompatible(const PartiallyDescribedBits& a,
                                 const PartiallyDescribedBits& start,
                                 int64_t width) {
  int64_t start_pos = std::min(
      a.bits.bit_count(), bits_ops::UnsignedBitsToSaturatedInt64(start.bits));
  int64_t slice_width = std::min(width, a.bits.bit_count() - start_pos);
  EXPECT_TRUE(DynamicBitSlice(a.partial, start.partial, width)
                  .IsCompatibleWith(bits_ops::ZeroExtend(
                      a.bits.Slice(start_pos, slice_width), width)));
}
FUZZ_TEST(PartialOpsFuzzTest, DynamicBitSliceIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8),
                 fuzztest::InRange<int64_t>(0, 1000));

void BitSliceUpdateIsCompatible(const PartiallyDescribedBits& to_update,
                                const PartiallyDescribedBits& start,
                                const PartiallyDescribedBits& update_value) {
  EXPECT_TRUE(
      BitSliceUpdate(to_update.partial, start.partial, update_value.partial)
          .IsCompatibleWith(bits_ops::BitSliceUpdate(
              to_update.bits,
              bits_ops::UnsignedBitsToSaturatedInt64(start.bits),
              update_value.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, BitSliceUpdateIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void ReverseIsCompatible(const PartiallyDescribedBits& p) {
  EXPECT_TRUE(Reverse(p.partial).IsCompatibleWith(bits_ops::Reverse(p.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, ReverseIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8));

void DecodeIsCompatible(const PartiallyDescribedBits& a, int64_t width) {
  Bits decoded(width);
  if (int64_t index = bits_ops::UnsignedBitsToSaturatedInt64(a.bits);
      index < width) {
    decoded = Bits::PowerOfTwo(/*set_bit_index=*/index, width);
  }
  EXPECT_TRUE(Decode(a.partial, width).IsCompatibleWith(decoded))
      << "partial: " << Decode(a.partial, width).ToString()
      << ", bits: " << decoded.ToDebugString();
}
FUZZ_TEST(PartialOpsFuzzTest, DecodeIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 fuzztest::InRange<int64_t>(0, 1000));

void EncodeIsCompatible(const PartiallyDescribedBits& a) {
  const int64_t output_width = CeilOfLog2(a.bits.bit_count());
  Bits result(output_width);
  for (int64_t i = 0; i < a.bits.bit_count(); ++i) {
    if (a.bits.Get(i)) {
      result = bits_ops::Or(result, UBits(i, output_width));
    }
  }
  EXPECT_TRUE(Encode(a.partial).IsCompatibleWith(result));
}
FUZZ_TEST(PartialOpsFuzzTest, EncodeIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8));

void OneHotLsbToMsbIsCompatible(const PartiallyDescribedBits& a) {
  EXPECT_TRUE(OneHotLsbToMsb(a.partial).IsCompatibleWith(
      bits_ops::OneHotLsbToMsb(a.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, OneHotLsbToMsbIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8));

void OneHotMsbToLsbIsCompatible(const PartiallyDescribedBits& a) {
  EXPECT_TRUE(OneHotMsbToLsb(a.partial).IsCompatibleWith(
      bits_ops::OneHotMsbToLsb(a.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, OneHotMsbToLsbIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8));

void OneHotSelectIsCompatible(const PartiallyDescribedBits& selector,
                              absl::Span<const PartiallyDescribedBits> cases,
                              bool selector_can_be_zero) {
  selector_can_be_zero |=
      selector.bits.IsZero() ||
      selector.partial.IsCompatibleWith(Bits(selector.bits.bit_count()));
  std::vector<PartialInformation> partial_cases;
  for (const PartiallyDescribedBits& c : cases) {
    partial_cases.push_back(c.partial);
  }

  Bits result(cases.front().bits.bit_count());
  for (int64_t i = 0; i < cases.size(); ++i) {
    if (selector.bits.Get(i)) {
      result = bits_ops::Or(result, cases[i].bits);
    }
  }
  EXPECT_TRUE(
      OneHotSelect(selector.partial, partial_cases, selector_can_be_zero)
          .IsCompatibleWith(result));
}
FUZZ_TEST(PartialOpsFuzzTest, OneHotSelectIsCompatible)
    .WithDomains(
        ArbitraryPartiallyDescribedBits(16),
        fuzztest::VectorOf(ArbitraryPartiallyDescribedBits(8)).WithSize(16),
        fuzztest::Arbitrary<bool>());

void GateIsCompatible(const PartiallyDescribedBits& control,
                      const PartiallyDescribedBits& input) {
  EXPECT_TRUE(Gate(control.partial, input.partial)
                  .IsCompatibleWith(control.bits.IsOne()
                                        ? input.bits
                                        : Bits(input.bits.bit_count())));
}
FUZZ_TEST(PartialOpsFuzzTest, GateIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(1),
                 ArbitraryPartiallyDescribedBits(8));

TEST(PartialOpsTest, Neg) {
  PartialInformation a =
      Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 18}, {26, 27}});
  PartialInformation b = Partial("0b00_000X_10XX", {{8, 11}});
  PartialInformation c = Partial("0bXX_XXXX_1XXX");
  PartialInformation d = PartialInformation::Impossible(10);
  PartialInformation e = PartialInformation::Unconstrained(10);
  PartialInformation f = Partial("0b0X_XXXX_X11X", {{6, 511}});

  EXPECT_EQ(Neg(a),
            Partial("0b11_111X_X1XX",
                    {{997, 998}, {1006, 1006}, {1013, 1014}, {1021, 1022}}));
  EXPECT_EQ(Neg(b), Partial("0b11_1111_XXXX", {{1013, 1016}}));
  EXPECT_EQ(Neg(c), Partial({{1, 1016}}, 10));
  EXPECT_EQ(Neg(d), PartialInformation::Impossible(10));
  EXPECT_EQ(Neg(e), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Neg(f), Partial("0b1X_XXXX_X0XX", {{513, 1018}}));
}

void NegIsCompatible(const PartiallyDescribedBits& a) {
  EXPECT_TRUE(Neg(a.partial).IsCompatibleWith(bits_ops::Negate(a.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, NegIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8));

TEST(PartialOpsTest, Add) {
  PartialInformation a =
      Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 18}, {26, 27}});
  PartialInformation b = Partial("0b00_000X_10XX", {{8, 11}});
  PartialInformation c = Partial("0bXX_XXXX_1XXX");
  PartialInformation d = PartialInformation::Impossible(10);
  PartialInformation e = PartialInformation::Unconstrained(10);
  PartialInformation f = Partial("0bXX_XXXX_00XX");

  EXPECT_EQ(Add(a, a), Partial("0b00_00XX_X1XX", {{4, 6},
                                                  {12, 14},
                                                  {20, 22},
                                                  {28, 30},
                                                  {36, 38},
                                                  {44, 45},
                                                  {52, 54}}));
  EXPECT_EQ(Add(b, b), Partial("0b00_0001_0XXX", {{16, 22}}));
  EXPECT_EQ(Add(c, c), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Add(d, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Add(e, e), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Add(f, f), Partial("0bXX_XXXX_0XXX"));

  EXPECT_EQ(Add(a, b), Partial("0b00_00XX_XXXX",
                               {{10, 14}, {18, 22}, {26, 29}, {34, 38}}));
  EXPECT_EQ(Add(a, c), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Add(a, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Add(a, e), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Add(a, f), PartialInformation::Unconstrained(10));

  EXPECT_EQ(Add(b, c), Partial({{0, 10}, {16, 1023}}, 10));
  EXPECT_EQ(Add(b, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Add(b, e), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Add(b, f), Partial("0bXX_XXXX_1XXX", {{8, 1022}}));

  EXPECT_EQ(Add(c, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Add(c, e), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Add(c, f), PartialInformation::Unconstrained(10));

  EXPECT_EQ(Add(d, e), PartialInformation::Impossible(10));
  EXPECT_EQ(Add(d, f), PartialInformation::Impossible(10));

  EXPECT_EQ(Add(e, f), PartialInformation::Unconstrained(10));
}

void AddIsCompatible(const PartiallyDescribedBits& a,
                     const PartiallyDescribedBits& b) {
  EXPECT_TRUE(Add(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::Add(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, AddIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

TEST(PartialOpsTest, Sub) {
  PartialInformation a =
      Partial("0b00_000X_X01X", {{2, 3}, {10, 11}, {18, 18}, {26, 27}});
  PartialInformation b = Partial("0b00_000X_10XX", {{8, 11}});
  PartialInformation c = Partial("0bXX_XXXX_1XXX");
  PartialInformation d = PartialInformation::Impossible(10);
  PartialInformation e = PartialInformation::Unconstrained(10);
  PartialInformation f = Partial("0bXX_XXXX_00XX");

  EXPECT_EQ(Sub(a, a), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Sub(b, b), Partial({{0, 3}, {1021, 1023}}, 10));
  EXPECT_EQ(Sub(c, c), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Sub(d, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Sub(e, e), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Sub(f, f), PartialInformation::Unconstrained(10));

  EXPECT_EQ(Sub(a, b), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Sub(a, c), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Sub(a, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Sub(a, e), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Sub(a, f), PartialInformation::Unconstrained(10));

  EXPECT_EQ(Sub(b, c), Partial({{0, 3}, {9, 1023}}, 10));
  EXPECT_EQ(Sub(b, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Sub(b, e), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Sub(b, f), Partial({{0, 11}, {21, 1023}}, 10));

  EXPECT_EQ(Sub(c, d), PartialInformation::Impossible(10));
  EXPECT_EQ(Sub(c, e), PartialInformation::Unconstrained(10));
  EXPECT_EQ(Sub(c, f), PartialInformation::Unconstrained(10));

  EXPECT_EQ(Sub(d, e), PartialInformation::Impossible(10));
  EXPECT_EQ(Sub(d, f), PartialInformation::Impossible(10));

  EXPECT_EQ(Sub(e, f), PartialInformation::Unconstrained(10));
}

void SubIsCompatible(const PartiallyDescribedBits& a,
                     const PartiallyDescribedBits& b) {
  EXPECT_TRUE(Sub(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::Sub(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, SubIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

Bits NarrowOrExtend(Bits b, int64_t width, bool as_signed) {
  if (b.bit_count() == width) {
    return b;
  } else if (b.bit_count() < width) {
    if (as_signed) {
      return bits_ops::SignExtend(b, width);
    } else {
      return bits_ops::ZeroExtend(b, width);
    }
  } else {
    return bits_ops::Truncate(b, width);
  }
}

TEST(PartialOpsTest, UMul) {
  EXPECT_EQ(
      UMul(PartialInformation::Unconstrained(8), Partial("0b0000_0010"), 0),
      Partial("0b"));
}

void UMulIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b, int64_t width) {
  EXPECT_TRUE(
      UMul(a.partial, b.partial, width)
          .IsCompatibleWith(NarrowOrExtend(bits_ops::UMul(a.bits, b.bits),
                                           width, /*as_signed=*/false)));
}
FUZZ_TEST(PartialOpsFuzzTest, UMulIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8),
                 fuzztest::InRange<int64_t>(0, 1000));

TEST(PartialOpsTest, UDiv) {
  // If we divide a value with no known ternary information by a value with
  // ternary information that could be zero, the result should be compatible
  // with the all-ones value (which results from dividing by zero).
  PartialInformation a = Partial({{56, 56}, {192, 232}}, 8);
  ASSERT_TRUE(a.IsCompatibleWith(UBits(56, 8)));
  PartialInformation b = Partial("0b0XXX_0XX0");
  ASSERT_TRUE(b.IsCompatibleWith(UBits(0, 8)));
  EXPECT_TRUE(UDiv(a, b).IsCompatibleWith(Bits::AllOnes(8)));
}

void UDivIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b) {
  EXPECT_TRUE(UDiv(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::UDiv(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, UDivIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void UModIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b) {
  EXPECT_TRUE(UMod(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::UMod(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, UModIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void SMulIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b, int64_t width) {
  EXPECT_TRUE(
      SMul(a.partial, b.partial, width)
          .IsCompatibleWith(NarrowOrExtend(bits_ops::SMul(a.bits, b.bits),
                                           width, /*as_signed=*/true)));
}
FUZZ_TEST(PartialOpsFuzzTest, SMulIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8),
                 fuzztest::InRange<int64_t>(0, 1000));

void SDivIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b) {
  EXPECT_TRUE(SDiv(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::SDiv(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, SDivIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void SModIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b) {
  EXPECT_TRUE(SMod(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::SMod(a.bits, b.bits)));
}
FUZZ_TEST(PartialOpsFuzzTest, SModIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void ShllIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b) {
  EXPECT_TRUE(Shll(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::ShiftLeftLogical(
                      a.bits, bits_ops::UnsignedBitsToSaturatedInt64(b.bits))));
}
FUZZ_TEST(PartialOpsFuzzTest, ShllIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void ShraIsCompatible(const PartiallyDescribedBits& a,
                      const PartiallyDescribedBits& b) {
  EXPECT_TRUE(Shra(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::ShiftRightArith(
                      a.bits, bits_ops::UnsignedBitsToSaturatedInt64(b.bits))));
}
FUZZ_TEST(PartialOpsFuzzTest, ShraIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void EqIsCompatible(const PartiallyDescribedBits& a,
                    const PartiallyDescribedBits& b) {
  EXPECT_TRUE(
      Eq(a.partial, b.partial)
          .IsCompatibleWith(a.bits == b.bits ? Bits::AllOnes(1) : Bits(1)));
}
FUZZ_TEST(PartialOpsFuzzTest, EqIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void NeIsCompatible(const PartiallyDescribedBits& a,
                    const PartiallyDescribedBits& b) {
  EXPECT_TRUE(
      Ne(a.partial, b.partial)
          .IsCompatibleWith(a.bits != b.bits ? Bits::AllOnes(1) : Bits(1)));
}
FUZZ_TEST(PartialOpsFuzzTest, NeIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void ULtIsCompatible(const PartiallyDescribedBits& a,
                     const PartiallyDescribedBits& b) {
  EXPECT_TRUE(ULt(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::ULessThan(a.bits, b.bits)
                                        ? Bits::AllOnes(1)
                                        : Bits(1)));
}
FUZZ_TEST(PartialOpsFuzzTest, ULtIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

void SLtIsCompatible(const PartiallyDescribedBits& a,
                     const PartiallyDescribedBits& b) {
  EXPECT_TRUE(SLt(a.partial, b.partial)
                  .IsCompatibleWith(bits_ops::SLessThan(a.bits, b.bits)
                                        ? Bits::AllOnes(1)
                                        : Bits(1)));
}
FUZZ_TEST(PartialOpsFuzzTest, SLtIsCompatible)
    .WithDomains(ArbitraryPartiallyDescribedBits(8),
                 ArbitraryPartiallyDescribedBits(8));

}  // namespace
}  // namespace xls::partial_ops
