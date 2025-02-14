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

#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;

class BigIntTest : public ::testing::Test {
 protected:
  BigInt MakeBigInt(int64_t value) {
    return BigInt::MakeSigned(SBits(value, 64));
  }

  BigInt MakeBigInt(std::string_view hex_string) {
    auto [sign, magnitude] = GetSignAndMagnitude(hex_string).value();
    if (sign == Sign::kNegative) {
      // Negative number (leading '-').
      return BigInt::Negate(BigInt::MakeUnsigned(magnitude));
    }
    return BigInt::MakeUnsigned(magnitude);
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

// This is a regression test for https://github.com/google/xls/issues/759
// `operator=(BigInt&& other)` used to leak memory
TEST_F(BigIntTest, CopyAssignNoLeak) {
  {
    BigInt x = BigInt::Exp2(1);

    // tests operator=(BigInt&& other)
    x = x / x;

    EXPECT_EQ(x, BigInt::One());
  }
  {
    BigInt x = BigInt::Exp2(1);
    BigInt one = x / x;

    // tests operator=(const BigInt& other)
    x = one;

    EXPECT_EQ(x, BigInt::One());
    EXPECT_EQ(one, BigInt::One());
  }
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
  EXPECT_EQ(BitsToString(BigInt::MakeUnsigned(wide_bits).ToUnsignedBits(),
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
  EXPECT_EQ(BitsToString(BigInt::MakeSigned(wide_bits).ToSignedBits(),
                         FormatPreference::kHex),
            wide_input);
}

TEST_F(BigIntTest, Zero) {
  BigInt zero = BigInt::Zero();
  EXPECT_EQ(zero.UnsignedBitCount(), 0);
  EXPECT_EQ(zero.ToUnsignedBits(), UBits(0, 0));
}

TEST_F(BigIntTest, One) {
  BigInt one = BigInt::One();
  EXPECT_EQ(one.UnsignedBitCount(), 1);
  EXPECT_EQ(one.ToUnsignedBits(), UBits(1, 1));
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

TEST_F(BigIntTest, Exp2) {
  constexpr int many_bits = 33;
  EXPECT_EQ(BigInt::Exp2(0), BigInt::One());
  EXPECT_EQ(BigInt::Exp2(1), BigInt::MakeUnsigned(UBits(2, many_bits)));
  EXPECT_EQ(BigInt::Exp2(2), BigInt::MakeUnsigned(UBits(4, many_bits)));
  EXPECT_EQ(BigInt::Exp2(32),
            BigInt::MakeUnsigned(UBits(4294967296ul, many_bits)));
}

TEST_F(BigIntTest, Negate) {
  EXPECT_EQ(BigInt::Negate(MakeBigInt(0)), MakeBigInt(0));
  EXPECT_EQ(BigInt::Negate(MakeBigInt(123456)), MakeBigInt(-123456));
  EXPECT_EQ(BigInt::Negate(
                MakeBigInt("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff")),
            MakeBigInt("-0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff"));
}

TEST_F(BigIntTest, Absolute) {
  EXPECT_EQ(BigInt::Absolute(MakeBigInt(0)), MakeBigInt(0));

  EXPECT_EQ(BigInt::Absolute(MakeBigInt(-1)), MakeBigInt(1));
  EXPECT_EQ(BigInt::Absolute(MakeBigInt(1)), MakeBigInt(1));

  EXPECT_EQ(BigInt::Absolute(MakeBigInt(-37)), MakeBigInt(37));
  EXPECT_EQ(BigInt::Absolute(MakeBigInt(37)), MakeBigInt(37));

  EXPECT_EQ(BigInt::Absolute(MakeBigInt(-9223372036854775807ll)),
            MakeBigInt(9223372036854775807ll));
  EXPECT_EQ(BigInt::Absolute(MakeBigInt(9223372036854775807ll)),
            MakeBigInt(9223372036854775807ll));
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

  EXPECT_TRUE(MakeBigInt(2) < MakeBigInt(10));
  EXPECT_FALSE(MakeBigInt(10) < MakeBigInt(10));
  EXPECT_FALSE(MakeBigInt(12) < MakeBigInt(10));
}

TEST_F(BigIntTest, GreaterThan) {
  EXPECT_FALSE(BigInt::GreaterThan(MakeBigInt(2), MakeBigInt(10)));
  EXPECT_FALSE(BigInt::GreaterThan(MakeBigInt(10), MakeBigInt(10)));
  EXPECT_TRUE(BigInt::GreaterThan(MakeBigInt(12), MakeBigInt(10)));

  EXPECT_FALSE(MakeBigInt(2) > MakeBigInt(10));
  EXPECT_FALSE(MakeBigInt(10) > MakeBigInt(10));
  EXPECT_TRUE(MakeBigInt(12) > MakeBigInt(10));
}

TEST_F(BigIntTest, LessThanEqual) {
  EXPECT_TRUE(MakeBigInt(2) <= MakeBigInt(10));
  EXPECT_TRUE(MakeBigInt(10) <= MakeBigInt(10));
  EXPECT_FALSE(MakeBigInt(12) <= MakeBigInt(10));
}

TEST_F(BigIntTest, GreaterThanEqual) {
  EXPECT_FALSE(MakeBigInt(2) >= MakeBigInt(10));
  EXPECT_TRUE(MakeBigInt(10) >= MakeBigInt(10));
  EXPECT_TRUE(MakeBigInt(12) >= MakeBigInt(10));
}

TEST_F(BigIntTest, IsEven) {
  EXPECT_TRUE(BigInt::IsEven(MakeBigInt(-4)));
  EXPECT_FALSE(BigInt::IsEven(MakeBigInt(-3)));
  EXPECT_TRUE(BigInt::IsEven(MakeBigInt(-2)));
  EXPECT_FALSE(BigInt::IsEven(MakeBigInt(-1)));
  EXPECT_TRUE(BigInt::IsEven(MakeBigInt(0)));
  EXPECT_FALSE(BigInt::IsEven(MakeBigInt(1)));
  EXPECT_TRUE(BigInt::IsEven(MakeBigInt(2)));
  EXPECT_FALSE(BigInt::IsEven(MakeBigInt(3)));
  EXPECT_TRUE(BigInt::IsEven(MakeBigInt(4)));
}

TEST_F(BigIntTest, IsPowerOfTwo) {
  EXPECT_FALSE(BigInt::IsPowerOfTwo(MakeBigInt(-2)));
  EXPECT_FALSE(BigInt::IsPowerOfTwo(MakeBigInt(-1)));
  EXPECT_FALSE(BigInt::IsPowerOfTwo(MakeBigInt(0)));

  EXPECT_FALSE(BigInt::IsPowerOfTwo(BigInt::Exp2(129) - BigInt::One()));
  EXPECT_FALSE(BigInt::IsPowerOfTwo(BigInt::Exp2(129) + BigInt::One()));

  EXPECT_TRUE(BigInt::IsPowerOfTwo(MakeBigInt(1)));
  EXPECT_TRUE(BigInt::IsPowerOfTwo(MakeBigInt(2)));
  EXPECT_TRUE(BigInt::IsPowerOfTwo(MakeBigInt(4)));
  EXPECT_TRUE(BigInt::IsPowerOfTwo(BigInt::Exp2(129)));
}

TEST_F(BigIntTest, CeilingLog2) {
  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(0)),
            std::numeric_limits<int64_t>::min());

  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(1)), 0);

  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(2)), 1);

  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(3)), 2);
  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(4)), 2);

  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(5)), 3);
  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(6)), 3);
  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(7)), 3);
  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(8)), 3);

  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(9)), 4);
  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(15)), 4);

  EXPECT_EQ(BigInt::CeilingLog2(MakeBigInt(4611686018427387903ul)), 62);
  EXPECT_EQ(BigInt::CeilingLog2(BigInt::Exp2(62) - BigInt::One()), 62);

  EXPECT_EQ(BigInt::CeilingLog2(BigInt::Exp2(128)), 128);
  EXPECT_EQ(BigInt::CeilingLog2(BigInt::Exp2(128) - BigInt::One()), 128);

  EXPECT_EQ(BigInt::CeilingLog2(BigInt::Exp2(128) + BigInt::One()), 129);
  EXPECT_EQ(BigInt::CeilingLog2(BigInt::Exp2(129)), 129);
}

TEST_F(BigIntTest, FactorizePowerOfTwo) {
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(0));
    EXPECT_EQ(odd, MakeBigInt(0));
    EXPECT_EQ(exponent, 0);
  }
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(1));
    EXPECT_EQ(odd, MakeBigInt(1));
    EXPECT_EQ(exponent, 0);
  }
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(2));
    EXPECT_EQ(odd, MakeBigInt(1));
    EXPECT_EQ(exponent, 1);
  }
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(3));
    EXPECT_EQ(odd, MakeBigInt(3));
    EXPECT_EQ(exponent, 0);
  }
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(4));
    EXPECT_EQ(odd, MakeBigInt(1));
    EXPECT_EQ(exponent, 2);
  }
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(6));
    EXPECT_EQ(odd, MakeBigInt(3));
    EXPECT_EQ(exponent, 1);
  }
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(40));
    EXPECT_EQ(odd, MakeBigInt(5));
    EXPECT_EQ(exponent, 3);
  }
  {
    auto [odd, exponent] =
        BigInt::FactorizePowerOfTwo(MakeBigInt(7) * BigInt::Exp2(129));
    EXPECT_EQ(odd, MakeBigInt(7));
    EXPECT_EQ(exponent, 129);
  }

  // negative
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(-1));
    EXPECT_EQ(odd, MakeBigInt(-1));
    EXPECT_EQ(exponent, 0);
  }
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(-2));
    EXPECT_EQ(odd, MakeBigInt(-1));
    EXPECT_EQ(exponent, 1);
  }
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(-3));
    EXPECT_EQ(odd, MakeBigInt(-3));
    EXPECT_EQ(exponent, 0);
  }
  {
    auto [odd, exponent] = BigInt::FactorizePowerOfTwo(MakeBigInt(-4));
    EXPECT_EQ(odd, MakeBigInt(-1));
    EXPECT_EQ(exponent, 2);
  }
}

}  // namespace
}  // namespace xls
