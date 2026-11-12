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

#include "xls/ir/number_parser.h"

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/bits_test_utils.h"
#include "xls/ir/format_preference.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;

TEST(NumberParserTest, ParseNumbersAsUint64) {
  auto expect_uint64_value = [](std::string_view s, uint64_t expected) {
    XLS_ASSERT_OK_AND_ASSIGN(uint64_t value, ParseNumberAsUint64(s));
    EXPECT_EQ(value, expected);
  };
  expect_uint64_value("0", 0);
  expect_uint64_value("-0", 0);
  expect_uint64_value("3", 3);
  expect_uint64_value("12_34_567", 1234567);
  expect_uint64_value("0b0", 0);
  expect_uint64_value("0b1", 1);
  expect_uint64_value("0b1100110011010101010010101011010", 1718265178UL);
  expect_uint64_value("0x0", 0);
  expect_uint64_value("0x1", 1);
  expect_uint64_value("0xdeadbeef", 3735928559UL);
  expect_uint64_value("0xdeadbeefdeadbeef", 0xdeadbeefdeadbeefULL);
  expect_uint64_value("0xffffffffffffffff",
                      std::numeric_limits<uint64_t>::max());
}

TEST(NumberParserTest, ParseNumbersAsInt64) {
  auto expect_int64_value = [](std::string_view s, int64_t expected) {
    XLS_ASSERT_OK_AND_ASSIGN(int64_t value, ParseNumberAsInt64(s));
    EXPECT_EQ(value, expected);
  };
  expect_int64_value("0", 0);
  expect_int64_value("-0", 0);
  expect_int64_value("3", 3);
  expect_int64_value("1234567", 1234567);
  expect_int64_value("0b0", 0);
  expect_int64_value("0b1", 1);
  expect_int64_value("0b1100110011010101010010101011010", 1718265178UL);
  expect_int64_value("0x0", 0);
  expect_int64_value("0x1", 1);
  expect_int64_value("0xdead_beef", 3735928559UL);
  expect_int64_value("-1", -1);
  expect_int64_value("-2", -2);
  expect_int64_value("-0xf00", -0xf00);
  expect_int64_value("-0b1010", -10);
  expect_int64_value("0x7fff_ffff_ffff_ffff",
                     std::numeric_limits<int64_t>::max());
  expect_int64_value("-0x8000_0000_0000_0000",
                     std::numeric_limits<int64_t>::min());
}

TEST(NumberParserTest, ParseNumbersAsBits) {
  auto expect_bits_value = [](std::string_view s, const Bits& expected) {
    XLS_ASSERT_OK_AND_ASSIGN(Bits value, ParseNumber(s));
    EXPECT_EQ(value, expected);
  };
  // Verifies that width of the Bits type is the minimum required to represent
  // the number.
  expect_bits_value("0", UBits(0, 0));
  expect_bits_value("-0", UBits(0, 0));
  // UBits(1, 1) is -1 in 1-bit twos complement.
  expect_bits_value("-1", UBits(1, 1));
  expect_bits_value("-0x00000001", UBits(1, 1));
  expect_bits_value("-4", UBits(4, 3));
  expect_bits_value("5", UBits(5, 3));
  expect_bits_value("-5", SBits(-5, 4));
  expect_bits_value("127", UBits(127, 7));
  expect_bits_value("128", UBits(128, 8));
  expect_bits_value("-128", SBits(-128, 8));
  expect_bits_value("-129", SBits(-129, 9));
  expect_bits_value("-0x8000_0000_0000_0000",
                    SBits(std::numeric_limits<int64_t>::min(), 64));

  expect_bits_value("0xbeef_abcd_fab2_3456_7890_1010_101a_cbde_fadd_fff",
                    bits_ops::Concat({UBits(0xbeefabcdfab23456ULL, 64),
                                      UBits(0x78901010101acbdeULL, 64),
                                      UBits(0xfaddfffULL, 28)}));
  expect_bits_value(
      "0b011"
      "__1010_0010_1011_1010_1010_1011_0101_0111"
      "__0101_0101_0100_1110_1101_0101_1000_1111"
      "__1001_1101_0111_0111_0001_0010_1101_0101"
      "__0101_1001_1101_1010_1111_0110_1011_0111",
      bits_ops::Concat({UBits(3, 2), UBits(0xa2baab57554ed58fULL, 64),
                        UBits(0x9d7712d559daf6b7ULL, 64)}));
}

TEST(NumberParserTest, ParseNumbersWithoutPrefix) {
  auto expect_bits_value = [](std::string_view s, FormatPreference format,
                              const Bits& expected) {
    XLS_ASSERT_OK_AND_ASSIGN(Bits value,
                             ParseUnsignedNumberWithoutPrefix(s, format));
    EXPECT_EQ(value, expected);
  };
  expect_bits_value("0", FormatPreference::kUnsignedDecimal, UBits(0, 0));
  expect_bits_value("0", FormatPreference::kBinary, UBits(0, 0));
  expect_bits_value("0", FormatPreference::kHex, UBits(0, 0));
  expect_bits_value("1", FormatPreference::kUnsignedDecimal, UBits(1, 1));
  expect_bits_value("1", FormatPreference::kBinary, UBits(1, 1));
  expect_bits_value("1", FormatPreference::kHex, UBits(1, 1));

  expect_bits_value("42", FormatPreference::kUnsignedDecimal, UBits(42, 6));
  expect_bits_value("1234567890", FormatPreference::kUnsignedDecimal,
                    UBits(1234567890ULL, 31));
  expect_bits_value("12345678901234567890", FormatPreference::kUnsignedDecimal,
                    UBits(12345678901234567890ULL, 64));

  expect_bits_value("1010_1011_1100", FormatPreference::kBinary,
                    UBits(0xabc, 12));
  expect_bits_value("abcdef", FormatPreference::kHex, UBits(0xabcdef, 24));
}

TEST(NumberParserTest, ParseNumberErrors) {
  EXPECT_THAT(
      ParseNumberAsUint64("").status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("Cannot parse empty string as a number")));
  EXPECT_THAT(ParseNumberAsUint64("123b").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Could not convert 123b to 32-bit decimal number")));
  EXPECT_THAT(
      ParseNumberAsUint64("0b").status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("Could not convert 0b to a number")));
  EXPECT_THAT(
      ParseNumberAsUint64("0b__").status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("Could not convert 0b__ to a number")));
  EXPECT_THAT(ParseNumberAsUint64("0b02").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Could not convert 0b02 to binary number")));
  EXPECT_THAT(ParseNumberAsUint64("0x0fr").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Could not convert 0x0fr to hexadecimal number")));
  EXPECT_THAT(ParseNumberAsUint64("-1").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Value is not representable as an uint64_t")));
  EXPECT_THAT(ParseNumberAsUint64("-0xdeadbeef").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Value is not representable as an uint64_t")));
  EXPECT_THAT(ParseNumberAsUint64("0xdeadbeefdeadbeef000").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Value is not representable as an uint64_t")));

  EXPECT_THAT(ParseNumberAsInt64("0xdeadbeefdeadbeef").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Value is not representable as an int64_t")));
  EXPECT_THAT(ParseNumberAsInt64("-0x8000_0000_0000_0001").status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       ::testing::HasSubstr(
                           "Value is not representable as an int64_t")));
}

// Decimal numbers are parsed in chunks of up to 19 digits; exercise the chunk
// boundaries and values around 2^64 explicitly, cross-checked against the
// (independent) hexadecimal parser.
TEST(NumberParserTest, ParseWideDecimalNumbers) {
  auto expect_same_as_hex = [](std::string_view decimal, std::string_view hex) {
    XLS_ASSERT_OK_AND_ASSIGN(Bits from_decimal, ParseNumber(decimal));
    XLS_ASSERT_OK_AND_ASSIGN(Bits from_hex, ParseNumber(hex));
    EXPECT_EQ(from_decimal, from_hex) << decimal << " vs " << hex;
  };
  // 19 digits (the largest single chunk), 10^19 - 1.
  expect_same_as_hex("9999999999999999999", "0x8ac7230489e7ffff");
  // 20 digits; 10^19 and the extremes around 2^64.
  expect_same_as_hex("10000000000000000000", "0x8ac7230489e80000");
  expect_same_as_hex("18446744073709551615", "0xffffffffffffffff");
  expect_same_as_hex("18446744073709551616", "0x10000000000000000");
  // 2^128 - 1 (39 digits) and 2^128 (39 digits).
  expect_same_as_hex("340282366920938463463374607431768211455",
                     "0xffffffffffffffffffffffffffffffff");
  expect_same_as_hex("340282366920938463463374607431768211456",
                     "0x100000000000000000000000000000000");
  // 39 digits: 10^38.
  expect_same_as_hex("100000000000000000000000000000000000000",
                     "0x4b3b4ca85a86c47a098a224000000000");
  // 38 digits (exactly two chunks) whose second chunk is all zeros (10^37),
  // and whose second chunk has leading zeros (10^37 + 1).
  expect_same_as_hex("10000000000000000000000000000000000000",
                     "0x785ee10d5da46d900f436a000000000");
  expect_same_as_hex("10000000000000000000000000000000000001",
                     "0x785ee10d5da46d900f436a000000001");
  // Underscores are ignored wherever they appear.
  expect_same_as_hex("18_446_744_073_709_551_616", "0x10000000000000000");
}

// Round trip: any value, formatted as decimal, parses back to the same value
// (narrowed to the minimum width). This covers arbitrary numbers of chunks.
void DecimalRoundTripsThroughParser(const Bits& bits) {
  std::string decimal = BitsToString(bits, FormatPreference::kUnsignedDecimal);
  XLS_ASSERT_OK_AND_ASSIGN(Bits parsed, ParseNumber(decimal));
  EXPECT_EQ(parsed, bits_ops::DropLeadingZeroes(bits)) << decimal;
}
FUZZ_TEST(NumberParserFuzzTest, DecimalRoundTripsThroughParser)
    .WithDomains(ArbitraryBits());

void Uint64DecimalRoundTripsThroughParser(uint64_t value) {
  std::string decimal = absl::StrCat(value);
  XLS_ASSERT_OK_AND_ASSIGN(uint64_t parsed, ParseNumberAsUint64(decimal));
  EXPECT_EQ(parsed, value);
  XLS_ASSERT_OK_AND_ASSIGN(Bits parsed_bits, ParseNumber(decimal));
  EXPECT_EQ(parsed_bits, UBits(value, Bits::MinBitCountUnsigned(value)));
}
FUZZ_TEST(NumberParserFuzzTest, Uint64DecimalRoundTripsThroughParser)
    .WithDomains(fuzztest::Arbitrary<uint64_t>());

}  // namespace
}  // namespace xls
