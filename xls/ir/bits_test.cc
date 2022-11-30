// Copyright 2020 The XLS Authors
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

#include "xls/ir/bits.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "rapidcheck/gtest.h"
#include "rapidcheck.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

// Create a Bits of the given bit count with the prime number index bits set to
// one.
Bits PrimeBits(int64_t bit_count) {
  auto is_prime = [](int64_t n) {
    if (n < 2) {
      return false;
    }
    for (int64_t i = 2; i * i < n; ++i) {
      if (n % i == 0) {
        return false;
      }
    }
    return true;
  };

  std::vector<uint8_t> bytes(CeilOfRatio(bit_count, int64_t{8}), 0);
  for (int64_t i = 0; i < bit_count; ++i) {
    if (is_prime(i)) {
      bytes[i / 8] |= 1 << (i % 8);
    }
  }
  return Bits::FromBytes(bytes, bit_count);
}

TEST(BitsTest, BitsConstructor) {
  Bits empty(0);
  EXPECT_THAT(empty.ToInt64(), IsOkAndHolds(0));
  EXPECT_THAT(empty.ToUint64(), IsOkAndHolds(0));

  Bits b0(15);
  EXPECT_THAT(b0.ToInt64(), IsOkAndHolds(0));
  EXPECT_THAT(b0.ToUint64(), IsOkAndHolds(0));

  Bits b1(1234);
  for (int64_t i = 0; i < 1234; ++i) {
    EXPECT_EQ(b1.Get(i), 0);
  }
}

TEST(BitsTest, BitsVectorConstructor) {
  EXPECT_EQ(Bits::FromBytes({}, 0), Bits());

  EXPECT_EQ(Bits::FromBytes({42}, 6), UBits(42, 6));
  EXPECT_EQ(Bits::FromBytes({42}, 8), UBits(42, 8));
  EXPECT_EQ(Bits::FromBytes({42, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 80),
            UBits(42, 80));
  EXPECT_EQ(Bits::FromBytes({42, 0, 0, 0, 0, 0, 0, 0, 0, 0}, 73),
            UBits(42, 73));

  EXPECT_EQ(Bits::FromBytes({0xef, 0xbe, 0xad, 0xde}, 32),
            UBits(0xdeadbeefULL, 32));

  std::vector<uint8_t> bytes = {0x1,  0xab, 0xcd, 0xef, 0x12, 0x34,
                                0x56, 0x78, 0x90, 0xfe, 0xdc, 0xba};
  std::reverse(bytes.begin(), bytes.end());
  EXPECT_EQ(Bits::FromBytes(bytes, 89).ToString(FormatPreference::kHex),
            "0x1ab_cdef_1234_5678_90fe_dcba");
}

TEST(BitsTest, SetRange) {
  Bits bits = Bits::FromBytes({0}, 6);
  EXPECT_TRUE(bits.IsZero());

  bits.SetRange(1, 3);

  EXPECT_EQ(bits, Bits({false, true, true, false, false, false}));
}

TEST(BitsTest, BitsToBytes) {
  EXPECT_TRUE(Bits().ToBytes().empty());
  EXPECT_THAT(UBits(42, 6).ToBytes(), ElementsAre(42));
  EXPECT_THAT(UBits(123, 16).ToBytes(), ElementsAre(123, 0));
  EXPECT_THAT(UBits(42, 80).ToBytes(),
              ElementsAre(42, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  EXPECT_THAT(UBits(42, 77).ToBytes(),
              ElementsAre(42, 0, 0, 0, 0, 0, 0, 0, 0, 0));
  std::vector<uint8_t> bytes = {0x1,  0xab, 0xcd, 0xef, 0x12, 0x34,
                                0x56, 0x78, 0x90, 0xfe, 0xdc, 0xba};
  std::reverse(bytes.begin(), bytes.end());
  EXPECT_THAT(Bits::FromBytes(bytes, 89).ToBytes(),
              ElementsAre(0xba, 0xdc, 0xfe, 0x90, 0x78, 0x56, 0x34, 0x12, 0xef,
                          0xcd, 0xab, 0x1));
  EXPECT_THAT(
      Bits::AllOnes(65).ToBytes(),
      ElementsAre(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01));
  {
    Bits wide_bits = Bits::AllOnes(75);
    std::vector<uint8_t> bytes(CeilOfRatio(wide_bits.bit_count(), int64_t{8}));
    wide_bits.ToBytes(absl::MakeSpan(bytes));
    EXPECT_THAT(bytes, ElementsAre(0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
                                   0xff, 0xff, 0x07));
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        Bits wide_bits,
        ParseNumber("0x4ff_ffff_0000_1111_2222_3333_4444_ffff_0000_cc31"));
    std::vector<uint8_t> bytes(CeilOfRatio(wide_bits.bit_count(), int64_t{8}));
    wide_bits.ToBytes(absl::MakeSpan(bytes));
    EXPECT_THAT(bytes, ElementsAre(0x31, 0xcc, 0x00, 0x00, 0xff, 0xff, 0x44,
                                   0x44, 0x33, 0x33, 0x22, 0x22, 0x11, 0x11,
                                   0x00, 0x00, 0xff, 0xff, 0xff, 0x4));
  }
}

TEST(BitsTest, Msb) {
  EXPECT_EQ(Bits().msb(), 0);
  EXPECT_EQ(UBits(1, 1).msb(), 1);
  EXPECT_EQ(UBits(1, 2).msb(), 0);
  EXPECT_EQ(UBits(0x80, 8).msb(), 1);
  EXPECT_EQ(UBits(0x80, 800).msb(), 0);
  EXPECT_EQ(SBits(-1, 800).msb(), 1);
  EXPECT_EQ(SBits(-1, 48).msb(), 1);
  EXPECT_EQ(SBits(-1, 63).msb(), 1);
  EXPECT_EQ(SBits(-1, 64).msb(), 1);
  EXPECT_EQ(SBits(-1, 65).msb(), 1);
}

TEST(BitsTest, IsOne) {
  EXPECT_FALSE(Bits().IsOne());
  EXPECT_TRUE(UBits(1, 1).IsOne());
  EXPECT_TRUE(UBits(1, 10).IsOne());
  EXPECT_TRUE(UBits(1, 10000000).IsOne());
  EXPECT_FALSE(UBits(0b10, 10).IsOne());
  EXPECT_FALSE(UBits(0b111, 16).IsOne());
}

TEST(BitsTest, PopCount) {
  EXPECT_EQ(Bits().PopCount(), 0);
  EXPECT_EQ(UBits(0b1, 1).PopCount(), 1);
  EXPECT_EQ(UBits(0b1, 100).PopCount(), 1);
  EXPECT_EQ(UBits(0b10101010101, 16).PopCount(), 6);
  EXPECT_EQ(UBits(0, 10000).PopCount(), 0);
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_bits,
      ParseNumber("0xffff_ffff_0000_1111_2222_3333_4444_ffff_0000_cccc"));
  EXPECT_EQ(wide_bits.PopCount(), 76);
}

TEST(BitsTest, CountLeandingZeros) {
  EXPECT_EQ(Bits().CountLeadingZeros(), 0);
  EXPECT_EQ(UBits(0b1, 1).CountLeadingZeros(), 0);
  EXPECT_EQ(UBits(0b1, 100).CountLeadingZeros(), 99);
  EXPECT_EQ(UBits(0b10101010101, 16).CountLeadingZeros(), 5);
  EXPECT_EQ(UBits(0, 10000).CountLeadingZeros(), 10000);
}

TEST(BitsTest, CountLeandingTrailingOnes) {
  EXPECT_EQ(Bits().CountLeadingOnes(), 0);
  EXPECT_EQ(UBits(0b1, 1).CountLeadingOnes(), 1);
  EXPECT_EQ(UBits(0b1, 100).CountLeadingOnes(), 0);
  EXPECT_EQ(UBits(0b11110000, 8).CountLeadingOnes(), 4);
  EXPECT_EQ(UBits(0b11110000, 9).CountLeadingOnes(), 0);

  EXPECT_EQ(Bits().CountTrailingOnes(), 0);
  EXPECT_EQ(UBits(0b1, 1).CountTrailingOnes(), 1);
  EXPECT_EQ(UBits(0b1, 100).CountTrailingOnes(), 1);
  EXPECT_EQ(UBits(0b11110000, 8).CountTrailingOnes(), 0);
  EXPECT_EQ(UBits(0x3ff, 12345).CountTrailingOnes(), 10);
  EXPECT_EQ(UBits(0x3ff0, 12345).CountTrailingOnes(), 0);
}

TEST(BitsTest, FitsIn) {
  EXPECT_TRUE(Bits().FitsInNBitsUnsigned(0));
  EXPECT_TRUE(Bits().FitsInNBitsUnsigned(1));
  EXPECT_TRUE(Bits().FitsInNBitsUnsigned(16));
  EXPECT_TRUE(Bits().FitsInNBitsSigned(0));
  EXPECT_TRUE(Bits().FitsInNBitsSigned(1));
  EXPECT_TRUE(Bits().FitsInNBitsSigned(16));

  EXPECT_FALSE(UBits(1, 1).FitsInNBitsUnsigned(0));
  EXPECT_TRUE(UBits(1, 1).FitsInNBitsUnsigned(1));
  EXPECT_TRUE(UBits(1, 1).FitsInNBitsUnsigned(16));
  EXPECT_FALSE(UBits(1, 1).FitsInNBitsSigned(0));
  EXPECT_TRUE(UBits(1, 1).FitsInNBitsSigned(1));
  EXPECT_TRUE(UBits(1, 1).FitsInNBitsSigned(16));

  EXPECT_FALSE(UBits(0xff, 8).FitsInNBitsUnsigned(1));
  EXPECT_FALSE(UBits(0xff, 8).FitsInNBitsUnsigned(4));
  EXPECT_TRUE(UBits(0xff, 8).FitsInNBitsUnsigned(8));
  EXPECT_TRUE(UBits(0xff, 8).FitsInNBitsUnsigned(16));
  // 0xff in 8-bits is -1 so it should fit in any signed width >= 1.
  EXPECT_TRUE(UBits(0xff, 8).FitsInNBitsSigned(1));
  EXPECT_TRUE(UBits(0xff, 8).FitsInNBitsSigned(4));
  EXPECT_TRUE(UBits(0xff, 8).FitsInNBitsSigned(8));
  EXPECT_TRUE(UBits(0xff, 8).FitsInNBitsSigned(16));

  EXPECT_FALSE(UBits(0xff, 16).FitsInNBitsUnsigned(1));
  EXPECT_FALSE(UBits(0xff, 16).FitsInNBitsUnsigned(4));
  EXPECT_TRUE(UBits(0xff, 16).FitsInNBitsUnsigned(8));
  EXPECT_TRUE(UBits(0xff, 16).FitsInNBitsUnsigned(16));
  EXPECT_FALSE(UBits(0xff, 16).FitsInNBitsSigned(1));
  EXPECT_FALSE(UBits(0xff, 16).FitsInNBitsSigned(4));
  EXPECT_FALSE(UBits(0xff, 16).FitsInNBitsSigned(8));
  EXPECT_TRUE(UBits(0xff, 16).FitsInNBitsSigned(16));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_bits,
      ParseNumber("0xffff_ffff_0000_1111_2222_3333_4444_ffff_0000_cccc"));
  EXPECT_FALSE(wide_bits.FitsInNBitsUnsigned(64));
  EXPECT_FALSE(wide_bits.FitsInNBitsSigned(64));
  EXPECT_FALSE(wide_bits.FitsInUint64());
  EXPECT_FALSE(wide_bits.FitsInInt64());
  EXPECT_TRUE(wide_bits.FitsInNBitsUnsigned(160));
  EXPECT_TRUE(wide_bits.FitsInNBitsSigned(160));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_minus_2,
      ParseNumber(
          "0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_fffe"));
  EXPECT_FALSE(wide_minus_2.FitsInNBitsUnsigned(64));
  EXPECT_TRUE(wide_minus_2.FitsInNBitsSigned(64));
}

TEST(BitsTest, ToUint64OrInt64) {
  EXPECT_THAT(Bits().ToUint64(), IsOkAndHolds(0));
  EXPECT_THAT(Bits().ToInt64(), IsOkAndHolds(0));

  EXPECT_THAT(UBits(0xff, 8).ToUint64(), IsOkAndHolds(0xff));
  EXPECT_THAT(UBits(0xff, 8).ToInt64(), IsOkAndHolds(-1));

  EXPECT_THAT(UBits(0xffffffffffffffffULL, 64).ToUint64(),
              IsOkAndHolds(0xffffffffffffffffULL));
  EXPECT_THAT(UBits(0xffffffffffffffffULL, 64).ToInt64(), IsOkAndHolds(-1));

  EXPECT_THAT(UBits(0xffffffff00000000ULL, 64).ToUint64(),
              IsOkAndHolds(0xffffffff00000000ULL));
  EXPECT_THAT(UBits(0xffffffff00000000ULL, 64).ToInt64(),
              IsOkAndHolds(-4294967296));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_bits,
      ParseNumber("0xffff_ffff_0000_1111_2222_3333_4444_ffff_0000_cccc"));
  EXPECT_THAT(
      wide_bits.ToUint64(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Bits value cannot be represented as an unsigned 64-bit value")));
  EXPECT_THAT(
      wide_bits.ToInt64(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Bits value cannot be represented as a signed 64-bit value")));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_minus_2,
      ParseNumber(
          "0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_fffe"));
  EXPECT_THAT(
      wide_minus_2.ToUint64(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Bits value cannot be represented as an unsigned 64-bit value")));
  EXPECT_THAT(wide_minus_2.ToInt64(), IsOkAndHolds(-2));
}

// Remove 0x or 0b prefix and separators to get plain output.
std::string ToPlainString(std::string s) {
  return absl::StrJoin(absl::StrSplit(s.substr(2), '_'), "");
}

TEST(BitsTest, ToString) {
  auto test_binary = [](Bits b, std::string expected) {
    EXPECT_EQ(b.ToString(FormatPreference::kBinary), expected);
    EXPECT_EQ(b.ToString(FormatPreference::kPlainBinary),
              ToPlainString(expected));
  };

  auto test_hex = [](Bits b, std::string expected) {
    EXPECT_EQ(b.ToString(FormatPreference::kHex), expected);
    EXPECT_EQ(b.ToString(FormatPreference::kPlainHex), ToPlainString(expected));
  };

  Bits empty_bits(0);
  EXPECT_EQ(empty_bits.ToString(FormatPreference::kUnsignedDecimal), "0");
  EXPECT_EQ(empty_bits.ToString(FormatPreference::kSignedDecimal), "0");
  test_hex(empty_bits, "0x0");
  test_binary(empty_bits, "0b0");
  EXPECT_EQ(empty_bits.ToString(FormatPreference::kUnsignedDecimal,
                                /*include_bit_count=*/true),
            "0 [0 bits]");
  EXPECT_EQ(empty_bits.ToString(FormatPreference::kSignedDecimal,
                                /*include_bit_count=*/true),
            "0 [0 bits]");

  Bits b1 = UBits(1, 1);
  EXPECT_EQ(b1.ToString(FormatPreference::kUnsignedDecimal), "1");
  EXPECT_EQ(b1.ToString(FormatPreference::kSignedDecimal), "-1");
  test_hex(b1, "0x1");
  test_binary(b1, "0b1");

  test_binary(UBits(1, 16), "0b1");
  test_hex(UBits(1, 16), "0x1");

  Bits b42 = UBits(42, 7);
  EXPECT_EQ(b42.ToString(FormatPreference::kUnsignedDecimal), "42");
  EXPECT_EQ(b42.ToString(FormatPreference::kSignedDecimal), "42");
  test_hex(b42, "0x2a");
  test_binary(b42, "0b10_1010");

  Bits prime64 = PrimeBits(64);
  EXPECT_EQ(prime64.ToString(FormatPreference::kUnsignedDecimal),
            "2892025783495830204");
  EXPECT_EQ(prime64.ToString(FormatPreference::kSignedDecimal),
            "2892025783495830204");
  test_hex(prime64, "0x2822_8a20_a28a_2abc");
  test_binary(
      prime64,
      "0b10_1000_0010_0010_1000_1010_0010_0000_1010_0010_1000_1010_0010_"
      "1010_1011_1100");

  // Test widths wider than 64. Decimal output for wide bit counts is not
  // supported.
  test_hex(PrimeBits(65), "0x2822_8a20_a28a_2abc");
  test_binary(
      PrimeBits(65),
      "0b10_1000_0010_0010_1000_1010_0010_0000_1010_0010_1000_1010_0010_"
      "1010_1011_1100");

  test_hex(PrimeBits(96), "0x208_8288_2822_8a20_a28a_2abc");
  test_binary(
      PrimeBits(96),
      "0b10_0000_1000_1000_0010_1000_1000_0010_1000_0010_0010_1000_1010_"
      "0010_0000_1010_0010_1000_1010_0010_1010_1011_1100");
}

TEST(BitsTest, ToRawString) {
  Bits empty_bits(0);
  EXPECT_EQ(empty_bits.ToRawDigits(FormatPreference::kUnsignedDecimal), "0");
  EXPECT_EQ(empty_bits.ToRawDigits(FormatPreference::kSignedDecimal), "0");
  EXPECT_EQ(empty_bits.ToRawDigits(FormatPreference::kHex), "0");
  EXPECT_EQ(empty_bits.ToRawDigits(FormatPreference::kBinary), "0");
  EXPECT_EQ(empty_bits.ToRawDigits(FormatPreference::kHex,
                                   /*emit_leading_zeros=*/true),
            "0");
  EXPECT_EQ(empty_bits.ToRawDigits(FormatPreference::kBinary,
                                   /*emit_leading_zeros=*/true),
            "0");

  EXPECT_EQ(UBits(1, 16).ToRawDigits(FormatPreference::kBinary), "1");
  EXPECT_EQ(UBits(1, 16).ToRawDigits(FormatPreference::kHex), "1");
  EXPECT_EQ(UBits(1, 16).ToRawDigits(FormatPreference::kBinary,
                                     /*emit_leading_zeros=*/true),
            "0000_0000_0000_0001");
  EXPECT_EQ(UBits(1, 16).ToRawDigits(FormatPreference::kPlainBinary,
                                     /*emit_leading_zeros=*/true),
            "0000000000000001");
  EXPECT_EQ(UBits(1, 16).ToRawDigits(FormatPreference::kHex,
                                     /*emit_leading_zeros=*/true),
            "0001");
  EXPECT_EQ(UBits(1, 16).ToRawDigits(FormatPreference::kPlainHex,
                                     /*emit_leading_zeros=*/true),
            "0001");

  EXPECT_EQ(UBits(0x1b, 13).ToRawDigits(FormatPreference::kBinary), "1_1011");
  EXPECT_EQ(UBits(0x1b, 13).ToRawDigits(FormatPreference::kHex), "1b");
  EXPECT_EQ(UBits(0x1b, 13).ToRawDigits(FormatPreference::kBinary,
                                        /*emit_leading_zeros=*/true),
            "0_0000_0001_1011");
  EXPECT_EQ(UBits(0x1b, 13).ToRawDigits(FormatPreference::kPlainBinary,
                                        /*emit_leading_zeros=*/true),
            "0000000011011");
  EXPECT_EQ(UBits(0x1b, 13).ToRawDigits(FormatPreference::kHex,
                                        /*emit_leading_zeros=*/true),
            "001b");
  EXPECT_EQ(UBits(0x1b, 13).ToRawDigits(FormatPreference::kPlainHex,
                                        /*emit_leading_zeros=*/true),
            "001b");

  EXPECT_EQ(UBits(0x55, 17).ToRawDigits(FormatPreference::kBinary,
                                        /*emit_leading_zeros=*/true),
            "0_0000_0000_0101_0101");
  EXPECT_EQ(UBits(0x55, 17).ToRawDigits(FormatPreference::kPlainBinary,
                                        /*emit_leading_zeros=*/true),
            "00000000001010101");
  EXPECT_EQ(UBits(0x55, 17).ToRawDigits(FormatPreference::kHex,
                                        /*emit_leading_zeros=*/true),
            "0_0055");
  EXPECT_EQ(UBits(0x55, 17).ToRawDigits(FormatPreference::kPlainHex,
                                        /*emit_leading_zeros=*/true),
            "00055");
}

TEST(BitsTest, UBitsFactory) {
  Bits b0 = UBits(0, 1);
  EXPECT_THAT(b0.ToInt64(), IsOkAndHolds(0));
  EXPECT_THAT(b0.ToUint64(), IsOkAndHolds(0));

  Bits b1 = UBits(1, 1);
  // 0b1 as a one-bit twos complement number is -1.
  EXPECT_THAT(b1.ToInt64(), IsOkAndHolds(-1));
  EXPECT_THAT(b1.ToUint64(), IsOkAndHolds(1));

  Bits b2 = UBits(0, 4);
  EXPECT_THAT(b2.ToInt64(), IsOkAndHolds(0));
  EXPECT_THAT(b2.ToUint64(), IsOkAndHolds(0));

  Bits b3 = UBits(0b1111, 4);
  EXPECT_THAT(b3.ToInt64(), IsOkAndHolds(-1));
  EXPECT_THAT(b3.ToUint64(), IsOkAndHolds(15));

  // Verify that 1 in the MSB constructs properly.
  Bits b4 = UBits(1ull << 63, 64);
  EXPECT_THAT(b4.ToInt64(), IsOkAndHolds(std::numeric_limits<int64_t>::min()));
  EXPECT_THAT(b4.ToUint64(), IsOkAndHolds(0x8000000000000000ULL));
}

TEST(BitsTest, SBitsFactory) {
  Bits b0 = SBits(-1, 1);
  EXPECT_THAT(b0.ToInt64(), IsOkAndHolds(-1));
  EXPECT_THAT(b0.ToUint64(), IsOkAndHolds(1));

  Bits b1 = SBits(-4, 3);
  EXPECT_THAT(b1.ToInt64(), IsOkAndHolds(-4));
  EXPECT_THAT(b1.ToUint64(), IsOkAndHolds(4));

  Bits b2 = SBits(-5, 7);
  EXPECT_THAT(b2.ToInt64(), IsOkAndHolds(-5));
  EXPECT_THAT(b2.ToUint64(), IsOkAndHolds(123));

  Bits b3 = SBits(7, 4);
  EXPECT_THAT(b3.ToInt64(), IsOkAndHolds(7));
  EXPECT_THAT(b3.ToUint64(), IsOkAndHolds(7));

  Bits b4 = SBits(123456, 64);
  EXPECT_THAT(b4.ToInt64(), IsOkAndHolds(123456));
  EXPECT_THAT(b4.ToUint64(), IsOkAndHolds(123456));

  Bits b5 = SBits(-987654321, 64);
  EXPECT_THAT(b5.ToInt64(), IsOkAndHolds(-987654321));
  EXPECT_THAT(b5.ToUint64(), IsOkAndHolds(18446744072721897295ULL));

  Bits b6 = SBits(std::numeric_limits<int64_t>::min(), 64);
  EXPECT_THAT(b6.ToInt64(), IsOkAndHolds(std::numeric_limits<int64_t>::min()));
  EXPECT_THAT(
      b6.ToUint64(),
      IsOkAndHolds(static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) +
                   1));

  Bits b7 = SBits(std::numeric_limits<int64_t>::max(), 64);
  EXPECT_THAT(b7.ToInt64(), IsOkAndHolds(std::numeric_limits<int64_t>::max()));
  EXPECT_THAT(b7.ToUint64(), IsOkAndHolds(std::numeric_limits<int64_t>::max()));

  const int64_t kNLargeBits = 345;
  Bits b8 = SBits(-4, kNLargeBits);
  for (int64_t i = 0; i < kNLargeBits; ++i) {
    // All except the two LSb's should be set.
    if (i < 2) {
      EXPECT_EQ(b8.Get(i), 0) << "Bit " << i << " should be 0: " << b8;
    } else {
      EXPECT_EQ(b8.Get(i), 1) << "Bit " << i << " should be 1: " << b8;
    }
  }

  Bits b9 = SBits(7, kNLargeBits);
  for (int64_t i = 0; i < kNLargeBits; ++i) {
    // Only the three LSb's should be set.
    if (i < 3) {
      EXPECT_EQ(b9.Get(i), 1) << "Bit " << i << " should be 1: " << b9;
    } else {
      EXPECT_EQ(b9.Get(i), 0) << "Bit " << i << " should be 0: " << b9;
    }
  }
}

TEST(BitsTest, PowerOfTwo) {
  EXPECT_THAT(Bits::PowerOfTwo(0, 1).ToUint64(), IsOkAndHolds(1));
  EXPECT_THAT(Bits::PowerOfTwo(1, 5).ToUint64(), IsOkAndHolds(2));
  EXPECT_THAT(Bits::PowerOfTwo(6, 1024).ToUint64(), IsOkAndHolds(64));
  EXPECT_THAT(Bits::PowerOfTwo(63, 1024).ToUint64(), IsOkAndHolds(1ULL << 63));

  Bits big_bits = Bits::PowerOfTwo(1234, 5000);
  for (int64_t i = 0; i < 5000; ++i) {
    EXPECT_EQ(big_bits.Get(i), i == 1234);
  }
}

TEST(BitsTest, MinimumBits) {
  EXPECT_EQ(Bits::MinBitCountUnsigned(0), 0);
  EXPECT_EQ(Bits::MinBitCountSigned(0), 0);

  EXPECT_EQ(Bits::MinBitCountUnsigned(1), 1);
  EXPECT_EQ(Bits::MinBitCountSigned(1), 2);

  EXPECT_EQ(Bits::MinBitCountUnsigned(2), 2);
  EXPECT_EQ(Bits::MinBitCountSigned(2), 3);

  EXPECT_EQ(Bits::MinBitCountUnsigned(255), 8);
  EXPECT_EQ(Bits::MinBitCountSigned(255), 9);

  EXPECT_EQ(Bits::MinBitCountUnsigned(std::numeric_limits<int64_t>::max()), 63);
  EXPECT_EQ(Bits::MinBitCountSigned(std::numeric_limits<int64_t>::max()), 64);

  EXPECT_EQ(Bits::MinBitCountUnsigned(std::numeric_limits<uint64_t>::max()),
            64);

  EXPECT_EQ(Bits::MinBitCountSigned(-1), 1);
  EXPECT_EQ(Bits::MinBitCountSigned(-2), 2);
  EXPECT_EQ(Bits::MinBitCountSigned(-3), 3);
  EXPECT_EQ(Bits::MinBitCountSigned(-4), 3);
  EXPECT_EQ(Bits::MinBitCountSigned(-5), 4);
  EXPECT_EQ(Bits::MinBitCountSigned(-128), 8);
  EXPECT_EQ(Bits::MinBitCountSigned(-129), 9);
  EXPECT_EQ(Bits::MinBitCountSigned(std::numeric_limits<int64_t>::min()), 64);
}

TEST(BitsTest, Equality) {
  EXPECT_TRUE(Bits(0) == Bits(0));
  EXPECT_FALSE(Bits(0) != Bits(0));

  EXPECT_TRUE(UBits(0, 5) == UBits(0, 5));
  EXPECT_FALSE(UBits(0, 5) == UBits(0, 3));
  EXPECT_FALSE(UBits(3, 5) == UBits(0, 5));
  EXPECT_TRUE(UBits(123456, 444) == UBits(123456, 444));
  EXPECT_FALSE(UBits(123456, 444) == UBits(123456, 445));
  EXPECT_TRUE(PrimeBits(12345) == PrimeBits(12345));
  EXPECT_FALSE(PrimeBits(12345) == PrimeBits(12346));
}

TEST(BitsTest, AllZerosOrOnes) {
  Bits empty_bits(0);
  EXPECT_TRUE(empty_bits.IsZero());
  EXPECT_TRUE(empty_bits.IsAllOnes());

  Bits b0 = UBits(0, 1);
  EXPECT_TRUE(b0.IsZero());
  EXPECT_FALSE(b0.IsAllOnes());

  Bits b1 = UBits(1, 1);
  EXPECT_FALSE(b1.IsZero());
  EXPECT_TRUE(b1.IsAllOnes());

  EXPECT_TRUE(UBits(0xffff, 16).IsAllOnes());
  EXPECT_FALSE(UBits(0xffef, 16).IsAllOnes());

  EXPECT_TRUE(UBits(0, 16).IsZero());
  EXPECT_FALSE(UBits(0x800, 16).IsZero());

  EXPECT_TRUE(Bits(1234).IsZero());
  EXPECT_FALSE(PrimeBits(1234).IsZero());

  EXPECT_TRUE(Bits::AllOnes(0).IsAllOnes());
  EXPECT_EQ(Bits::AllOnes(0).bit_count(), 0);
  EXPECT_TRUE(Bits::AllOnes(1).IsAllOnes());
  EXPECT_EQ(Bits::AllOnes(1).bit_count(), 1);

  EXPECT_TRUE(Bits::AllOnes(32).IsAllOnes());
  EXPECT_EQ(Bits::AllOnes(32).bit_count(), 32);
  EXPECT_TRUE(Bits::AllOnes(1234).IsAllOnes());
  EXPECT_EQ(Bits::AllOnes(1234).bit_count(), 1234);
}

TEST(BitsTest, Slice) {
  Bits empty_bits(0);
  Bits b0 = UBits(0, 1);
  Bits b1 = UBits(1, 1);

  EXPECT_EQ(b0, b0.Slice(0, 1));
  EXPECT_EQ(b1, b1.Slice(0, 1));
  EXPECT_EQ(empty_bits, b0.Slice(0, 0));

  Bits deadbeef = UBits(0xdeadbeef12345678ULL, 64);
  EXPECT_EQ(deadbeef, deadbeef.Slice(0, 64));
  EXPECT_EQ(UBits(0xdead, 16), deadbeef.Slice(48, 16));
  EXPECT_EQ(UBits(0x78, 8), deadbeef.Slice(0, 8));
  EXPECT_EQ(UBits(7716, 13), deadbeef.Slice(23, 13));

  Bits big_prime = PrimeBits(12345);
  EXPECT_EQ(big_prime, big_prime.Slice(0, big_prime.bit_count()));
  // Random primes in some range: 2377 2381 2383 2389 2393
  // Slice out this range and verify the appropriate 5 bits are set.
  EXPECT_EQ(big_prime.Slice(2377, 17).ToString(FormatPreference::kBinary),
            "0b1_0001_0000_0101_0001");
}

TEST(BitsTest, ValueTest) {
  Value b0 = Value(UBits(2, 4));
  EXPECT_EQ(b0.bits(), UBits(2, 4));

  Value b1 = Value::Tuple({
      Value(UBits(2, 4)),
      Value(UBits(100, 8)),
  });
  absl::Span<const Value> tuple_values = b1.elements();
  EXPECT_EQ(tuple_values[0].bits(), UBits(2, 4));
  EXPECT_EQ(tuple_values[1].bits(), UBits(100, 8));
}

TEST(BitsTest, ValueTupleEquality) {
  Value b1 = Value::Tuple({
      Value(UBits(2, 4)),
      Value(UBits(100, 8)),
  });
  Value b2 = Value::Tuple({
      Value(UBits(2, 4)),
      Value(UBits(100, 8)),
  });
  EXPECT_EQ(b1, b2);

  // Different size.
  Value b3 = Value::Tuple({Value(UBits(2, 4))});
  EXPECT_NE(b1, b3);

  // Different value.
  Value b4 = Value::Tuple({
      Value(UBits(3, 4)),
      Value(UBits(100, 8)),
  });
  EXPECT_NE(b1, b4);

  // Different bitwidth.
  Value b5 = Value::Tuple({
      Value(UBits(2, 4)),
      Value(UBits(100, 7)),
  });
  EXPECT_NE(b1, b5);
}

TEST(BitsTest, ToBitVectorAndBack) {
  EXPECT_TRUE(Bits().ToBitVector().empty());
  EXPECT_THAT(UBits(0, 1).ToBitVector(), ElementsAre(false));
  EXPECT_THAT(UBits(1, 1).ToBitVector(), ElementsAre(true));
  EXPECT_THAT(
      UBits(0, 8).ToBitVector(),
      ElementsAre(false, false, false, false, false, false, false, false));
  EXPECT_THAT(UBits(0b11001, 5).ToBitVector(),
              ElementsAre(true, false, false, true, true));

  EXPECT_EQ(Bits(Bits().ToBitVector()), Bits());
  EXPECT_EQ(Bits(UBits(0, 1).ToBitVector()), UBits(0, 1));
  EXPECT_EQ(Bits(UBits(1, 1).ToBitVector()), UBits(1, 1));
  EXPECT_EQ(Bits(UBits(0b11001, 5).ToBitVector()), UBits(0b11001, 5));
  EXPECT_EQ(Bits(UBits(0b11001, 1234).ToBitVector()), UBits(0b11001, 1234));
}

static std::string BytesToHexString(absl::Span<const uint8_t> bytes) {
  return "0x" + absl::StrJoin(bytes, "", [](std::string* out, uint8_t b) {
           absl::StrAppendFormat(out, "%02x", b);
         });
}

RC_GTEST_PROP(BitsRapidcheck, RoundTripByte, (uint8_t byte)) {
  std::vector<uint8_t> orig = {byte};
  int64_t bit_count = *rc::gen::inRange(8 * 0 + 1, 8 * 1 + 1).as("bit_count");
  const Bits b = Bits::FromBytes(orig, bit_count);
  std::vector<uint8_t> exported = b.ToBytes();

  RC_ASSERT(exported.size() == 1);
  RC_ASSERT(exported.at(0) == (byte & Mask(bit_count)));
}

RC_GTEST_PROP(BitsRapidcheck, RoundTripTwoBytes, (uint8_t b0, uint8_t b1)) {
  std::vector<uint8_t> orig = {b0, b1};
  int64_t bit_count = *rc::gen::inRange(8 * 1 + 1, 8 * 2 + 1).as("bit_count");
  const Bits b = Bits::FromBytes(orig, bit_count);
  std::vector<uint8_t> exported = b.ToBytes();

  XLS_VLOG(1) << absl::StreamFormat("b0: 0x%02x b1: 0x%02x", b0, b1);
  XLS_VLOG(1) << "exported: " << BytesToHexString(exported);

  // Note: the way we import/export bytes to words, b1 becomes the most
  // significant byte in the word, so it is the byte that gets masked.
  uint8_t mask = Mask(bit_count % 8 == 0 ? 8 : bit_count % 8);
  RC_ASSERT(exported.size() == 2);
  RC_ASSERT(exported.at(0) == b0);
  RC_ASSERT(exported.at(1) == (b1 & mask));
}

RC_GTEST_PROP(BitsRapidcheck, RoundTripThreeBytes,
              (uint8_t b0, uint8_t b1, uint8_t b2)) {
  std::vector<uint8_t> orig = {b0, b1, b2};
  int64_t bit_count = *rc::gen::inRange(8 * 2 + 1, 8 * 3 + 1).as("bit_count");
  const Bits b = Bits::FromBytes(orig, bit_count);
  std::vector<uint8_t> exported = b.ToBytes();

  // Note: the way we import/export bytes to words, b2 becomes the most
  // significant byte in the word, so it is the byte that gets masked.
  uint8_t mask = Mask(bit_count % 8 == 0 ? 8 : bit_count % 8);
  RC_ASSERT(exported.size() == 3);
  RC_ASSERT(exported.at(0) == b0);
  RC_ASSERT(exported.at(1) == b1);
  RC_ASSERT(exported.at(2) == (b2 & mask));
}

RC_GTEST_PROP(BitsRapidcheck, RoundTripNBytes,
              (const std::vector<uint8_t>& orig)) {
  if (orig.empty()) {
    return;
  }
  int64_t orig_size = orig.size();
  int64_t bit_count =
      *rc::gen::inRange(8 * (orig_size - 1) + 1, 8 * orig_size + 1)
           .as("bit_count");
  const Bits b = Bits::FromBytes(orig, bit_count);
  std::vector<uint8_t> exported = b.ToBytes();

  RC_ASSERT(exported.size() == orig.size());

  // Note: the way we import/export bytes to words, the last byte always becomes
  // the most significant byte in the last word, so it is the byte that gets
  // masked.
  uint8_t mask = Mask(bit_count % 8 == 0 ? 8 : bit_count % 8);
  for (size_t i = 0; i < orig.size() - 1; ++i) {
    RC_ASSERT(exported.at(i) == orig.at(i));
  }
  RC_ASSERT(exported.back() == (orig.back() & mask));
}

// Populates a bitmap, converts that to bits, exports it back out, and checks it
// is the same as the original.
RC_GTEST_PROP(BitsRapidcheck, RoundTripFromBitmapToBitmap,
              (const std::vector<bool>& orig)) {
  // Construct the input bitmap from our bool vector.
  InlineBitmap bitmap(orig.size());
  for (size_t i = 0; i < orig.size(); ++i) {
    bitmap.Set(i, orig.at(i));
  }

  // Make a bits object from that input bitmap.
  const Bits b = Bits::FromBitmap(bitmap);

  // Now export it to a bit vector and check it has the same contents as our
  // original bool vector.
  absl::InlinedVector<bool, 1> bit_vector = b.ToBitVector();
  RC_ASSERT(bit_vector.size() == orig.size());
  for (size_t i = 0; i < orig.size(); ++i) {
    RC_ASSERT(orig.at(i) == bit_vector.at(i));
  }
}

}  // namespace
}  // namespace xls
