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

#include "xls/ir/big_int.h"

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/number_parser.h"

namespace xls {
namespace {

using status_testing::StatusIs;

class BigIntTest : public ::testing::Test {
 protected:
  BigInt MakeBigInt(int64_t value) {
    return BigInt::MakeSigned(SBits(value, 64));
  }

  BigInt MakeBigInt(absl::string_view hex_string) {
    std::pair<bool, Bits> sign_magnitude =
        GetSignAndMagnitude(hex_string).value();
    if (sign_magnitude.first) {
      // Negative number (leading '-').
      return BigInt::Negate(BigInt::MakeUnsigned(sign_magnitude.second));
    }
    return BigInt::MakeUnsigned(sign_magnitude.second);
  }
};

TEST_F(BigIntTest, EqualityOperator) {
  auto one = BigInt::MakeUnsigned(UBits(1, 64));
  auto two = BigInt::MakeUnsigned(UBits(2, 64));
  EXPECT_TRUE((one == one));
  EXPECT_FALSE((one == two));
  EXPECT_FALSE((one != one));
  EXPECT_TRUE((one != two));
}

TEST_F(BigIntTest, CopyConstructor) {
  auto original = BigInt::MakeUnsigned(UBits(1, 64));
  auto copy = BigInt(original);

  EXPECT_EQ(original, BigInt::MakeUnsigned(UBits(1, 64)));
  EXPECT_EQ(original, copy);
}

TEST_F(BigIntTest, MoveConstructor) {
  auto original = BigInt::MakeUnsigned(UBits(1, 64));
  auto moved = BigInt(std::move(original));

  EXPECT_EQ(moved, BigInt::MakeUnsigned(UBits(1, 64)));
}

TEST_F(BigIntTest, CopyAssignment) {
  auto original = BigInt::MakeUnsigned(UBits(1, 64));
  auto copy = BigInt();
  copy = original;

  EXPECT_EQ(original, BigInt::MakeUnsigned(UBits(1, 64)));
  EXPECT_EQ(original, copy);
}

TEST_F(BigIntTest, MoveAssignment) {
  auto original = BigInt::MakeUnsigned(UBits(1, 64));
  auto moved = BigInt();
  moved = std::move(original);

  EXPECT_EQ(moved, BigInt::MakeUnsigned(UBits(1, 64)));
}

TEST_F(BigIntTest, MakeUnsigned) {
  BigInt zero = BigInt::MakeUnsigned(UBits(0, 0));
  EXPECT_EQ(zero.UnsignedBitCount(), 0);
  EXPECT_EQ(zero.ToUnsignedBits(), UBits(0, 0));

  // Round-trip Bits values through BigInt and verify value and bit-width (which
  // should be the minimum to hold the value).
  EXPECT_EQ(BigInt::MakeUnsigned(UBits(1, 64)).ToUnsignedBits(), UBits(1, 1));
  EXPECT_EQ(BigInt::MakeUnsigned(UBits(2, 64)).ToUnsignedBits(), UBits(2, 2));
  EXPECT_EQ(BigInt::MakeUnsigned(UBits(3, 64)).ToUnsignedBits(), UBits(3, 2));
  EXPECT_EQ(BigInt::MakeUnsigned(UBits(4, 64)).ToUnsignedBits(), UBits(4, 3));
  EXPECT_EQ(BigInt::MakeUnsigned(UBits(5, 64)).ToUnsignedBits(), UBits(5, 3));
  EXPECT_EQ(BigInt::MakeUnsigned(UBits(6, 64)).ToUnsignedBits(), UBits(6, 3));
  EXPECT_EQ(BigInt::MakeUnsigned(UBits(7, 64)).ToUnsignedBits(), UBits(7, 3));
  EXPECT_EQ(BigInt::MakeUnsigned(UBits(8, 64)).ToUnsignedBits(), UBits(8, 4));

  EXPECT_EQ(BigInt::MakeUnsigned(UBits(1234567, 64)).ToUnsignedBits(),
            UBits(1234567, 21));

  std::string wide_input = "0xabcd_abcd_1234_1234_aaaa_bbbb_cccc_dddd";
  XLS_ASSERT_OK_AND_ASSIGN(Bits wide_bits, ParseNumber(wide_input));
  EXPECT_EQ(BigInt::MakeUnsigned(wide_bits).ToUnsignedBits().ToString(
                FormatPreference::kHex),
            wide_input);
}

TEST_F(BigIntTest, MakeSigned) {
  BigInt zero = BigInt::MakeSigned(UBits(0, 0));
  EXPECT_EQ(zero.SignedBitCount(), 0);
  EXPECT_EQ(zero.ToSignedBits(), UBits(0, 0));

  // Round-trip a signed Bits values through BigInt and verify value and
  // bit-width (which should be the minimum to hold the value in twos-complement
  // representation).
  EXPECT_EQ(BigInt::MakeSigned(SBits(0, 1)).ToSignedBits(), SBits(0, 0));
  EXPECT_EQ(BigInt::MakeSigned(SBits(1, 2)).ToSignedBits(), SBits(1, 2));
  EXPECT_EQ(BigInt::MakeSigned(SBits(1, 3)).ToSignedBits(), SBits(1, 2));
  EXPECT_EQ(BigInt::MakeSigned(SBits(1, 4)).ToSignedBits(), SBits(1, 2));
  EXPECT_EQ(BigInt::MakeSigned(SBits(1, 5)).ToSignedBits(), SBits(1, 2));
  EXPECT_EQ(BigInt::MakeSigned(SBits(1, 6)).ToSignedBits(), SBits(1, 2));
  EXPECT_EQ(BigInt::MakeSigned(SBits(1, 7)).ToSignedBits(), SBits(1, 2));
  EXPECT_EQ(BigInt::MakeSigned(SBits(1, 8)).ToSignedBits(), SBits(1, 2));
  EXPECT_EQ(BigInt::MakeSigned(SBits(1, 64)).ToSignedBits(), SBits(1, 2));
  EXPECT_EQ(BigInt::MakeSigned(SBits(2, 64)).ToSignedBits(), SBits(2, 3));
  EXPECT_EQ(BigInt::MakeSigned(SBits(3, 64)).ToSignedBits(), SBits(3, 3));
  EXPECT_EQ(BigInt::MakeSigned(SBits(4, 64)).ToSignedBits(), SBits(4, 4));
  EXPECT_EQ(BigInt::MakeSigned(SBits(5, 64)).ToSignedBits(), SBits(5, 4));
  EXPECT_EQ(BigInt::MakeSigned(SBits(6, 64)).ToSignedBits(), SBits(6, 4));
  EXPECT_EQ(BigInt::MakeSigned(SBits(7, 64)).ToSignedBits(), SBits(7, 4));
  EXPECT_EQ(BigInt::MakeSigned(SBits(8, 64)).ToSignedBits(), SBits(8, 5));
  EXPECT_EQ(BigInt::MakeSigned(SBits(127, 8)).ToSignedBits(), SBits(127, 8));

  EXPECT_EQ(BigInt::MakeSigned(SBits(-1, 1)).ToSignedBits(), SBits(-1, 1));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-1, 2)).ToSignedBits(), SBits(-1, 1));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-1, 3)).ToSignedBits(), SBits(-1, 1));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-1, 4)).ToSignedBits(), SBits(-1, 1));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-1, 5)).ToSignedBits(), SBits(-1, 1));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-1, 6)).ToSignedBits(), SBits(-1, 1));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-1, 7)).ToSignedBits(), SBits(-1, 1));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-1, 8)).ToSignedBits(), SBits(-1, 1));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-1, 64)).ToSignedBits(), SBits(-1, 1));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-2, 64)).ToSignedBits(), SBits(-2, 2));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-3, 64)).ToSignedBits(), SBits(-3, 3));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-4, 64)).ToSignedBits(), SBits(-4, 3));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-5, 64)).ToSignedBits(), SBits(-5, 4));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-6, 64)).ToSignedBits(), SBits(-6, 4));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-7, 64)).ToSignedBits(), SBits(-7, 4));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-8, 64)).ToSignedBits(), SBits(-8, 4));
  EXPECT_EQ(BigInt::MakeSigned(SBits(-128, 8)).ToSignedBits(), SBits(-128, 8));

  EXPECT_EQ(BigInt::MakeSigned(UBits(1234567, 64)).ToSignedBits(),
            UBits(1234567, 22));

  std::string wide_input = "0xabcd_abcd_1234_1234_aaaa_bbbb_cccc_dddd";
  XLS_ASSERT_OK_AND_ASSIGN(Bits wide_bits, ParseNumber(wide_input));
  EXPECT_EQ(BigInt::MakeSigned(wide_bits).ToSignedBits().ToString(
                FormatPreference::kHex),
            wide_input);
}

TEST_F(BigIntTest, ToSignedBitsWithBitCount) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto small_positive,
      BigInt::MakeSigned(SBits(42, 64)).ToSignedBitsWithBitCount(32));
  EXPECT_EQ(small_positive, SBits(42, 32));

  EXPECT_THAT(BigInt::MakeSigned(SBits(3e9, 64)).ToSignedBitsWithBitCount(32),
              StatusIs(absl::StatusCode::kInvalidArgument));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto small_negative,
      BigInt::MakeSigned(SBits(-42, 64)).ToSignedBitsWithBitCount(32));
  EXPECT_EQ(small_negative, SBits(-42, 32));

  EXPECT_THAT(BigInt::MakeSigned(SBits(-3e9, 64)).ToSignedBitsWithBitCount(32),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BigIntTest, ToUnsignedBitsWithBitCount) {
  XLS_ASSERT_OK_AND_ASSIGN(
      auto small_positive,
      BigInt::MakeUnsigned(UBits(42, 64)).ToUnsignedBitsWithBitCount(32));
  EXPECT_EQ(small_positive, UBits(42, 32));

  EXPECT_THAT(
      BigInt::MakeUnsigned(UBits(6e9, 64)).ToUnsignedBitsWithBitCount(32),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BigIntTest, SignedBitCount) {
  EXPECT_EQ(MakeBigInt(0).SignedBitCount(), 0);
  EXPECT_EQ(MakeBigInt(-1).SignedBitCount(), 1);
  EXPECT_EQ(MakeBigInt(-2).SignedBitCount(), 2);
  EXPECT_EQ(MakeBigInt(-3).SignedBitCount(), 3);
  EXPECT_EQ(MakeBigInt(-4).SignedBitCount(), 3);
  EXPECT_EQ(MakeBigInt(1).SignedBitCount(), 2);
  EXPECT_EQ(MakeBigInt(127).SignedBitCount(), 8);
  EXPECT_EQ(MakeBigInt(128).SignedBitCount(), 9);
  EXPECT_EQ(MakeBigInt(-128).SignedBitCount(), 8);
  EXPECT_EQ(MakeBigInt(-129).SignedBitCount(), 9);
}

TEST_F(BigIntTest, UnignedBitCount) {
  EXPECT_EQ(MakeBigInt(0).UnsignedBitCount(), 0);
  EXPECT_EQ(MakeBigInt(1).UnsignedBitCount(), 1);
  EXPECT_EQ(MakeBigInt(127).UnsignedBitCount(), 7);
  EXPECT_EQ(MakeBigInt(128).UnsignedBitCount(), 8);
  EXPECT_EQ(MakeBigInt(129).UnsignedBitCount(), 8);
}

TEST_F(BigIntTest, Add) {
  EXPECT_EQ(BigInt::Add(MakeBigInt(0), MakeBigInt(0)), MakeBigInt(0));
  EXPECT_EQ(BigInt::Add(MakeBigInt(42), MakeBigInt(123)), MakeBigInt(165));
  EXPECT_EQ(
      BigInt::Add(MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"),
                  MakeBigInt("0x1")),
      MakeBigInt("0x1_0000_0000_0000_0000_0000_0000_0000_0000_0000"));
  EXPECT_EQ(
      BigInt::Add(
          MakeBigInt("0xabcd_ffff_1234_aaaa_0101_3333_4444_1328_abcd_0042"),
          MakeBigInt("0xffff_1234_aaaa_0101_3333_4444_1328_abcd_0042_0033")),
      MakeBigInt("0x1_abcd_1233_bcde_abab_3434_7777_576c_bef5_ac0f_0075"));
}

TEST_F(BigIntTest, Sub) {
  EXPECT_EQ(BigInt::Sub(MakeBigInt(42), MakeBigInt(123)), MakeBigInt(-81));
  EXPECT_EQ(
      BigInt::Sub(
          MakeBigInt("0xabcd_ffff_1234_aaaa_0101_3333_4444_1328_abcd_0042"),
          MakeBigInt("0xffff_1234_aaaa_0101_3333_4444_1328_abcd_0042_0033")),
      MakeBigInt("-0x5431_1235_9875_5657_3232_1110_cee4_98a4_5474_fff1"));
}

TEST_F(BigIntTest, Negate) {
  EXPECT_EQ(BigInt::Negate(MakeBigInt(0)), MakeBigInt(0));
  EXPECT_EQ(BigInt::Negate(MakeBigInt(123456)), MakeBigInt(-123456));
  EXPECT_EQ(BigInt::Negate(
                MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff")),
            MakeBigInt("-0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"));
}

TEST_F(BigIntTest, Multiply) {
  EXPECT_EQ(BigInt::Mul(MakeBigInt(0), MakeBigInt(0)), MakeBigInt(0));
  EXPECT_EQ(BigInt::Mul(MakeBigInt(123456), MakeBigInt(0)), MakeBigInt(0));
  EXPECT_EQ(BigInt::Mul(MakeBigInt(123456), MakeBigInt(1)), MakeBigInt(123456));

  EXPECT_EQ(BigInt::Mul(MakeBigInt(1000), MakeBigInt(42)), MakeBigInt(42000));
  EXPECT_EQ(BigInt::Mul(MakeBigInt(-1000), MakeBigInt(42)), MakeBigInt(-42000));
  EXPECT_EQ(BigInt::Mul(MakeBigInt(-1000), MakeBigInt(-42)), MakeBigInt(42000));

  EXPECT_EQ(
      BigInt::Mul(MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"),
                  MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff")),
      MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_fffe_0000_0000_"
                 "0000_0000_0000_0000_0000_0000_0001"));
}

TEST_F(BigIntTest, Divide) {
  EXPECT_EQ(BigInt::Div(MakeBigInt(0), MakeBigInt(10)), MakeBigInt(0));

  // Rounding should match C semantics.
  EXPECT_EQ(BigInt::Div(MakeBigInt(10), MakeBigInt(2)), MakeBigInt(10 / 2));
  EXPECT_EQ(BigInt::Div(MakeBigInt(10), MakeBigInt(3)), MakeBigInt(10 / 3));
  EXPECT_EQ(BigInt::Div(MakeBigInt(10), MakeBigInt(4)), MakeBigInt(10 / 4));

  EXPECT_EQ(BigInt::Div(MakeBigInt(-10), MakeBigInt(2)), MakeBigInt(-10 / 2));
  EXPECT_EQ(BigInt::Div(MakeBigInt(-10), MakeBigInt(3)), MakeBigInt(-10 / 3));
  EXPECT_EQ(BigInt::Div(MakeBigInt(-10), MakeBigInt(4)), MakeBigInt(-10 / 4));

  EXPECT_EQ(BigInt::Div(MakeBigInt(10), MakeBigInt(-2)), MakeBigInt(10 / -2));
  EXPECT_EQ(BigInt::Div(MakeBigInt(10), MakeBigInt(-3)), MakeBigInt(10 / -3));
  EXPECT_EQ(BigInt::Div(MakeBigInt(10), MakeBigInt(-4)), MakeBigInt(10 / -4));

  EXPECT_EQ(BigInt::Div(MakeBigInt(-10), MakeBigInt(-2)), MakeBigInt(-10 / -2));
  EXPECT_EQ(BigInt::Div(MakeBigInt(-10), MakeBigInt(-3)), MakeBigInt(-10 / -3));
  EXPECT_EQ(BigInt::Div(MakeBigInt(-10), MakeBigInt(-4)), MakeBigInt(-10 / -4));

  EXPECT_EQ(
      BigInt::Div(
          MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"),
          MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff")),
      MakeBigInt(1));

  EXPECT_EQ(
      BigInt::Div(
          MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"),
          MakeBigInt("0xffff_ffff")),
      MakeBigInt("0x1_0000_0001_0000_0001_0000_0001_0000_0001"));
}

TEST_F(BigIntTest, Modulus) {
  EXPECT_EQ(BigInt::Mod(MakeBigInt(0), MakeBigInt(10)), MakeBigInt(0));

  // Result should match C semantics.
  EXPECT_EQ(BigInt::Mod(MakeBigInt(10), MakeBigInt(2)), MakeBigInt(10 % 2));
  EXPECT_EQ(BigInt::Mod(MakeBigInt(10), MakeBigInt(3)), MakeBigInt(10 % 3));
  EXPECT_EQ(BigInt::Mod(MakeBigInt(10), MakeBigInt(4)), MakeBigInt(10 % 4));

  EXPECT_EQ(BigInt::Mod(MakeBigInt(-10), MakeBigInt(2)), MakeBigInt(-10 % 2));
  EXPECT_EQ(BigInt::Mod(MakeBigInt(-10), MakeBigInt(3)), MakeBigInt(-10 % 3));
  EXPECT_EQ(BigInt::Mod(MakeBigInt(-10), MakeBigInt(4)), MakeBigInt(-10 % 4));

  EXPECT_EQ(BigInt::Mod(MakeBigInt(10), MakeBigInt(-2)), MakeBigInt(10 % -2));
  EXPECT_EQ(BigInt::Mod(MakeBigInt(10), MakeBigInt(-3)), MakeBigInt(10 % -3));
  EXPECT_EQ(BigInt::Mod(MakeBigInt(10), MakeBigInt(-4)), MakeBigInt(10 % -4));

  EXPECT_EQ(BigInt::Mod(MakeBigInt(-10), MakeBigInt(-2)), MakeBigInt(-10 % -2));
  EXPECT_EQ(BigInt::Mod(MakeBigInt(-10), MakeBigInt(-3)), MakeBigInt(-10 % -3));
  EXPECT_EQ(BigInt::Mod(MakeBigInt(-10), MakeBigInt(-4)), MakeBigInt(-10 % -4));

  EXPECT_EQ(
      BigInt::Mod(
          MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"),
          MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff")),
      MakeBigInt(0));

  EXPECT_EQ(
      BigInt::Mod(
          MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"),
          MakeBigInt("0xfff_ffff")),
      MakeBigInt("0xf_ffff"));
}

TEST_F(BigIntTest, LessThan) {
  EXPECT_TRUE(BigInt::LessThan(MakeBigInt(2), MakeBigInt(10)));
  EXPECT_FALSE(BigInt::LessThan(MakeBigInt(10), MakeBigInt(10)));
  EXPECT_FALSE(BigInt::LessThan(MakeBigInt(12), MakeBigInt(10)));
}
}  // namespace
}  // namespace xls
