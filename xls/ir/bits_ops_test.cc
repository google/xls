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

#include "xls/ir/bits_ops.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/math_util.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/number_parser.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

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
      bytes[bytes.size() - 1 - i / 8] |= 1 << (i % 8);
    }
  }
  return Bits::FromBytes(bytes, bit_count);
}

TEST(BitsOpsTest, LogicalOps) {
  Bits empty_bits(0);
  EXPECT_EQ(empty_bits, bits_ops::And(empty_bits, empty_bits));
  EXPECT_EQ(empty_bits, bits_ops::Or(empty_bits, empty_bits));
  EXPECT_EQ(empty_bits, bits_ops::Xor(empty_bits, empty_bits));
  EXPECT_EQ(empty_bits, bits_ops::Not(empty_bits));

  Bits b0 = UBits(0, 1);
  Bits b1 = UBits(1, 1);

  EXPECT_EQ(b0, bits_ops::And(b0, b1));
  EXPECT_EQ(b1, bits_ops::And(b1, b1));
  EXPECT_EQ(b0, bits_ops::Or(b0, b0));
  EXPECT_EQ(b1, bits_ops::Or(b0, b1));
  EXPECT_EQ(b1, bits_ops::Or(b1, b1));
  EXPECT_EQ(b1, bits_ops::Xor(b0, b1));
  EXPECT_EQ(b1, bits_ops::Xor(b1, b0));
  EXPECT_EQ(b0, bits_ops::Xor(b0, b0));
  EXPECT_EQ(b1, bits_ops::Not(b0));
  EXPECT_EQ(b0, bits_ops::Not(b1));

  Bits deadbeef2 = UBits(0xdeadbeefdeadbeefULL, 64);
  Bits fofo = UBits(0xf0f0f0f0f0f0f0f0ULL, 64);
  EXPECT_EQ(UBits(0xf0f0f0f0f0f0f0fULL, 64), bits_ops::Not(fofo));
  EXPECT_EQ(UBits(0xe0d0e0f0e0d0e0fULL, 64),
            bits_ops::And(deadbeef2, bits_ops::Not(fofo)));
  EXPECT_EQ(UBits(0xfefdfefffefdfeffULL, 64), bits_ops::Or(deadbeef2, fofo));

  Bits wide_bits = PrimeBits(12345);
  EXPECT_EQ(wide_bits, bits_ops::And(wide_bits, wide_bits));
  EXPECT_EQ(wide_bits, bits_ops::Or(wide_bits, wide_bits));
  EXPECT_TRUE(bits_ops::Xor(wide_bits, wide_bits).IsZero());
  EXPECT_TRUE(bits_ops::Xor(wide_bits, bits_ops::Not(wide_bits)).IsAllOnes());
}

TEST(BitsOpsTest, Concat) {
  Bits empty_bits(0);
  EXPECT_EQ(empty_bits, bits_ops::Concat({}));
  EXPECT_EQ(empty_bits, bits_ops::Concat({empty_bits}));
  EXPECT_EQ(empty_bits, bits_ops::Concat({empty_bits, empty_bits}));

  Bits b0 = UBits(0, 1);
  Bits b1 = UBits(1, 1);

  EXPECT_EQ(bits_ops::Concat({b0, b1}), UBits(1, 2));
  EXPECT_EQ(bits_ops::Concat({empty_bits, b0, empty_bits, b1, empty_bits}),
            UBits(1, 2));
  EXPECT_EQ(bits_ops::Concat({b1, b1, b0, b1, b0, b1}), UBits(0x35, 6));

  EXPECT_EQ(bits_ops::Concat({UBits(0xab, 8), UBits(0xcd, 8), UBits(0xef, 8)}),
            UBits(0xabcdef, 24));

  Bits deadbeef2 = UBits(0xdeadbeefdeadbeefULL, 64);
  Bits fofo = UBits(0xf0f0f0f0f0f0f0f0ULL, 64);
  EXPECT_EQ(
      bits_ops::Concat({deadbeef2, fofo}).ToString(FormatPreference::kHex),
      "0xdead_beef_dead_beef_f0f0_f0f0_f0f0_f0f0");
}

TEST(BitsOpsTest, Add) {
  EXPECT_EQ(bits_ops::Add(Bits(), Bits()), Bits());
  EXPECT_EQ(bits_ops::Add(UBits(23, 64), UBits(42, 64)), UBits(65, 64));

  // Test overflow conditions.
  EXPECT_EQ(bits_ops::Add(UBits(10, 4), UBits(13, 4)), UBits(7, 4));
  EXPECT_EQ(bits_ops::Add(UBits(0xffffffffffffffffUL, 64), UBits(1, 64)),
            UBits(0, 64));

  // Test wide values.
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_lhs,
      ParseNumber("0x1fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0042"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_rhs,
      ParseNumber("0x1000_0000_0000_0000_0001_1234_4444_3333_0000_1234"));
  EXPECT_EQ(bits_ops::Add(wide_lhs, wide_rhs).ToString(FormatPreference::kHex),
            "0x1000_0000_0000_0000_0000_1234_4444_3333_0000_1276");
}

TEST(BitsOpsTest, Sub) {
  EXPECT_EQ(bits_ops::Sub(Bits(), Bits()), Bits());
  EXPECT_EQ(bits_ops::Sub(UBits(55, 64), UBits(12, 64)), UBits(43, 64));

  // Test overflow conditions.
  EXPECT_EQ(bits_ops::Sub(UBits(9, 4), UBits(12, 4)), UBits(13, 4));
  EXPECT_EQ(bits_ops::Sub(UBits(0, 64), UBits(1, 64)),
            UBits(0xffffffffffffffffUL, 64));
  EXPECT_EQ(bits_ops::Sub(SBits(-6, 4), SBits(5, 4)), SBits(5, 4));

  // Test wide values.
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        Bits wide_lhs,
        ParseNumber("0x1fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0042"));
    XLS_ASSERT_OK_AND_ASSIGN(
        Bits wide_rhs,
        ParseNumber("0x1000_0000_0000_0000_0001_1234_4444_3333_0000_1234"));
    EXPECT_EQ(
        bits_ops::Sub(wide_lhs, wide_rhs).ToString(FormatPreference::kHex),
        "0xfff_ffff_ffff_ffff_fffd_edcb_bbbb_cccc_ffff_ee0e");
  }
  {
    // Test an underflow case.
    XLS_ASSERT_OK_AND_ASSIGN(
        Bits wide_lhs,
        ParseNumber("0x1000_0000_0000_0000_0000_0000_0000_0000_0000"));
    Bits wide_rhs = UBits(42, wide_lhs.bit_count());
    EXPECT_EQ(
        bits_ops::Sub(wide_lhs, wide_rhs).ToString(FormatPreference::kHex),
        "0xfff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffd6");
  }
}

TEST(BitsOpsTest, UMul) {
  EXPECT_EQ(bits_ops::UMul(Bits(), Bits()), Bits());
  EXPECT_EQ(bits_ops::UMul(UBits(100, 24), UBits(55, 22)), UBits(5500, 46));
  EXPECT_EQ(bits_ops::UMul(UBits(100, 64), UBits(55, 64)), UBits(5500, 128));
  EXPECT_EQ(bits_ops::UMul(UBits(100, 7), UBits(3, 2)), UBits(300, 9));

  // Test wide values.
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_lhs,
      ParseNumber("0x1fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0042"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_rhs,
      ParseNumber("0x1000_0000_0000_0000_0001_1234_4444_3333_0000_1234"));
  EXPECT_EQ(bits_ops::UMul(wide_lhs, wide_rhs).ToString(FormatPreference::kHex),
            "0x200_0000_0000_0000_0000_1246_8888_8666_6000_0249_8dcb_bbbb_cccc_"
            "ffff_ee12_b179_9995_3326_0004_b168");

  Bits result = bits_ops::UMul(Bits::AllOnes(65), Bits::AllOnes(65));
  EXPECT_EQ(result.bit_count(), 130);
  EXPECT_EQ(result.ToString(FormatPreference::kHex),
            "0x3_ffff_ffff_ffff_fffc_0000_0000_0000_0001");
}

TEST(BitsOpsTest, SMul) {
  EXPECT_EQ(bits_ops::SMul(Bits(), Bits()), Bits());
  EXPECT_EQ(bits_ops::SMul(SBits(100, 64), SBits(55, 64)), SBits(5500, 128));
  EXPECT_EQ(bits_ops::SMul(SBits(100, 64), SBits(-3, 64)), SBits(-300, 128));
  EXPECT_EQ(bits_ops::SMul(SBits(50, 7), SBits(-1, 2)), SBits(-50, 9));
  EXPECT_EQ(bits_ops::SMul(SBits(-50, 7), SBits(-1, 2)), SBits(50, 9));

  // Test wide values.
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_lhs,
      ParseNumber("0x1fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0042"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_rhs,
      ParseNumber("0x1000_0000_0000_0000_0001_1234_4444_3333_0000_1234"));
  EXPECT_EQ(bits_ops::SMul(wide_lhs, wide_rhs).ToString(FormatPreference::kHex),
            "0xfff_ffff_ffff_ffff_fffa_cdcb_bbbb_cccc_ffff_ee12_b179_9995_3326_"
            "0004_b168");
}

TEST(BitsOpsTest, UDiv) {
  EXPECT_EQ(bits_ops::UDiv(UBits(100, 64), UBits(5, 64)), UBits(20, 64));
  EXPECT_EQ(bits_ops::UDiv(UBits(100, 32), UBits(7, 32)), UBits(14, 32));
  EXPECT_EQ(bits_ops::UDiv(UBits(100, 7), UBits(7, 4)), UBits(14, 7));
  EXPECT_EQ(bits_ops::UDiv(UBits(0, 64), UBits(7, 32)), UBits(0, 64));

  // Divide by zero.
  EXPECT_EQ(bits_ops::UDiv(Bits(), Bits()), Bits());
  EXPECT_EQ(bits_ops::UDiv(UBits(10, 4), UBits(0, 4)), UBits(15, 4));
  EXPECT_EQ(bits_ops::UDiv(UBits(123456, 64), UBits(0, 20)), Bits::AllOnes(64));

  // Test wide values.
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_lhs,
      ParseNumber("0xffff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits wide_rhs, ParseNumber("0x1000_0000_0000_0000"));
  EXPECT_EQ(bits_ops::UDiv(wide_lhs,
                           bits_ops::ZeroExtend(wide_rhs, wide_lhs.bit_count()))
                .ToString(FormatPreference::kHex),
            "0xf_ffff_ffff_ffff_ffff_fff0_0000");
}

TEST(BitsOpsTest, UMod) {
  EXPECT_EQ(bits_ops::UMod(UBits(100, 64), UBits(5, 64)), UBits(0, 64));
  EXPECT_EQ(bits_ops::UMod(UBits(100, 32), UBits(7, 32)), UBits(2, 32));
  EXPECT_EQ(bits_ops::UMod(UBits(100, 7), UBits(7, 4)), UBits(2, 4));
  EXPECT_EQ(bits_ops::UMod(UBits(0, 64), UBits(7, 32)), UBits(0, 32));

  // Zero right-hand side should always produce zero.
  EXPECT_EQ(bits_ops::UMod(Bits(), Bits()), Bits());
  EXPECT_EQ(bits_ops::UMod(UBits(10, 4), UBits(0, 4)), Bits(4));
  EXPECT_EQ(bits_ops::UMod(UBits(123456, 64), UBits(0, 20)), Bits(20));

  // Test wide values.
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_lhs,
      ParseNumber("0xffff_ffff_ffff_ffff_ffff_0000_0000_1002_3004_5006"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits wide_rhs, ParseNumber("0x1000_0000_0000_0000"));
  EXPECT_EQ(bits_ops::UMod(wide_lhs,
                           bits_ops::ZeroExtend(wide_rhs, wide_lhs.bit_count()))
                .ToString(FormatPreference::kHex),
            "0x1002_3004_5006");
}

TEST(BitsOpsTest, SDiv) {
  EXPECT_EQ(bits_ops::SDiv(SBits(100, 64), SBits(5, 64)), SBits(20, 64));
  EXPECT_EQ(bits_ops::SDiv(SBits(100, 64), SBits(-5, 64)), SBits(-20, 64));
  EXPECT_EQ(bits_ops::SDiv(SBits(-100, 64), SBits(-5, 64)), SBits(20, 64));
  EXPECT_EQ(bits_ops::SDiv(SBits(-100, 64), SBits(5, 64)), SBits(-20, 64));

  EXPECT_EQ(bits_ops::SDiv(SBits(100, 32), SBits(7, 32)), SBits(14, 32));
  EXPECT_EQ(bits_ops::SDiv(SBits(-100, 32), SBits(7, 32)), SBits(-14, 32));
  EXPECT_EQ(bits_ops::SDiv(SBits(100, 32), SBits(-7, 32)), SBits(-14, 32));
  EXPECT_EQ(bits_ops::SDiv(SBits(-100, 32), SBits(-7, 32)), SBits(14, 32));

  EXPECT_EQ(bits_ops::SDiv(SBits(100, 8), SBits(7, 4)), SBits(14, 8));
  EXPECT_EQ(bits_ops::SDiv(SBits(-100, 8), SBits(7, 4)), SBits(-14, 8));
  EXPECT_EQ(bits_ops::SDiv(SBits(100, 8), SBits(-7, 4)), SBits(-14, 8));
  EXPECT_EQ(bits_ops::SDiv(SBits(-100, 8), SBits(-7, 4)), SBits(14, 8));

  EXPECT_EQ(bits_ops::SDiv(SBits(0, 64), SBits(7, 32)), SBits(0, 64));
  EXPECT_EQ(bits_ops::SDiv(SBits(0, 64), SBits(-7, 32)), SBits(0, 64));

  // Divide by zero.
  EXPECT_EQ(bits_ops::SDiv(Bits(), Bits()), Bits());
  EXPECT_EQ(bits_ops::SDiv(SBits(5, 4), SBits(0, 4)), SBits(7, 4));
  EXPECT_EQ(bits_ops::SDiv(SBits(-5, 4), SBits(0, 4)), SBits(-8, 4));
  EXPECT_EQ(bits_ops::SDiv(SBits(123456, 64), SBits(0, 20)),
            SBits(std::numeric_limits<int64_t>::max(), 64));
  EXPECT_EQ(bits_ops::SDiv(SBits(-123456, 64), SBits(0, 20)),
            SBits(std::numeric_limits<int64_t>::min(), 64));

  // Test wide values.
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_lhs,
      ParseNumber("0xffff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits wide_rhs, ParseNumber("0x1000_0000_0000_0000"));
  EXPECT_EQ(bits_ops::SDiv(wide_lhs,
                           bits_ops::ZeroExtend(wide_rhs, wide_lhs.bit_count()))
                .ToString(FormatPreference::kHex),
            "0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_fff0_0000");
}

TEST(BitsOpsTest, SMod) {
  EXPECT_EQ(bits_ops::SMod(SBits(100, 64), SBits(5, 64)), SBits(0, 64));
  EXPECT_EQ(bits_ops::SMod(SBits(100, 64), SBits(-5, 64)), SBits(0, 64));
  EXPECT_EQ(bits_ops::SMod(SBits(-100, 64), SBits(-5, 64)), SBits(0, 64));
  EXPECT_EQ(bits_ops::SMod(SBits(-100, 64), SBits(5, 64)), SBits(0, 64));

  EXPECT_EQ(bits_ops::SMod(SBits(100, 32), SBits(7, 32)), SBits(2, 32));
  EXPECT_EQ(bits_ops::SMod(SBits(-100, 32), SBits(7, 32)), SBits(-2, 32));
  EXPECT_EQ(bits_ops::SMod(SBits(100, 32), SBits(-7, 32)), SBits(2, 32));
  EXPECT_EQ(bits_ops::SMod(SBits(-100, 32), SBits(-7, 32)), SBits(-2, 32));

  EXPECT_EQ(bits_ops::SMod(SBits(100, 8), SBits(7, 4)), SBits(2, 4));
  EXPECT_EQ(bits_ops::SMod(SBits(-100, 8), SBits(7, 4)), SBits(-2, 4));
  EXPECT_EQ(bits_ops::SMod(SBits(100, 8), SBits(-7, 4)), SBits(2, 4));
  EXPECT_EQ(bits_ops::SMod(SBits(-100, 8), SBits(-7, 4)), SBits(-2, 4));

  EXPECT_EQ(bits_ops::SMod(SBits(0, 64), SBits(7, 32)), SBits(0, 32));
  EXPECT_EQ(bits_ops::SMod(SBits(0, 64), SBits(-7, 32)), SBits(0, 32));

  // Zero right hand side.
  EXPECT_EQ(bits_ops::SMod(Bits(), Bits()), Bits());
  EXPECT_EQ(bits_ops::SMod(SBits(5, 4), SBits(0, 4)), Bits(4));
  EXPECT_EQ(bits_ops::SMod(SBits(-5, 4), SBits(0, 4)), Bits(4));

  // Test wide values.
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_lhs,
      ParseNumber("0xffff_ffff_ffff_ffff_ffff_0000_0000_a000_b000_c000"));
  XLS_ASSERT_OK_AND_ASSIGN(Bits wide_rhs, ParseNumber("0x1000_0000_0000_0000"));
  EXPECT_EQ(bits_ops::SMod(wide_lhs,
                           bits_ops::ZeroExtend(wide_rhs, wide_lhs.bit_count()))
                .ToString(FormatPreference::kHex),
            "0xffff_ffff_ffff_ffff_ffff_ffff_f000_a000_b000_c000");
}

TEST(BitsOpsTest, UnsignedComparisons) {
  Bits b42 = UBits(42, 64);
  Bits b77 = UBits(77, 64);
  Bits b123 = UBits(123, 444);

  EXPECT_TRUE(bits_ops::UGreaterThanOrEqual(b42, b42));
  EXPECT_FALSE(bits_ops::UGreaterThanOrEqual(b42, b77));
  EXPECT_TRUE(bits_ops::UGreaterThanOrEqual(b77, b42));
  EXPECT_TRUE(bits_ops::UGreaterThanOrEqual(b123, b42));

  EXPECT_FALSE(bits_ops::UGreaterThan(b42, b42));
  EXPECT_FALSE(bits_ops::UGreaterThan(b42, b77));
  EXPECT_TRUE(bits_ops::UGreaterThan(b77, b42));
  EXPECT_FALSE(bits_ops::UGreaterThanOrEqual(b42, b123));

  EXPECT_TRUE(bits_ops::ULessThanOrEqual(b42, b42));
  EXPECT_TRUE(bits_ops::ULessThanOrEqual(b42, b77));
  EXPECT_FALSE(bits_ops::ULessThanOrEqual(b77, b42));
  EXPECT_TRUE(bits_ops::UGreaterThanOrEqual(b123, b77));
  EXPECT_FALSE(bits_ops::UGreaterThanOrEqual(b77, b123));
  EXPECT_TRUE(bits_ops::ULessThanOrEqual(b77, b123));
  EXPECT_TRUE(bits_ops::ULessThanOrEqual(b77, b77));

  EXPECT_FALSE(bits_ops::ULessThan(b42, b42));
  EXPECT_TRUE(bits_ops::ULessThan(b42, b77));
  EXPECT_FALSE(bits_ops::ULessThan(b77, b42));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits huge,
      ParseNumber(
          "0x2fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000_1234_5555"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits huger,
      ParseNumber(
          "0x3234_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000_1111_3333"));
  EXPECT_TRUE(bits_ops::UGreaterThanOrEqual(huger, huge));
  EXPECT_TRUE(
      bits_ops::UGreaterThanOrEqual(bits_ops::ZeroExtend(huger, 1234), huge));
  EXPECT_FALSE(bits_ops::ULessThan(huger, huge));
  EXPECT_TRUE(bits_ops::ULessThan(huge, huger));
  EXPECT_TRUE(bits_ops::ULessThan(bits_ops::ZeroExtend(huge, 10000), huger));
}

TEST(BitsOpsTest, UnsignedEqualComparisons) {
  EXPECT_TRUE(bits_ops::UEqual(Bits(), UBits(0, 42)));
  EXPECT_TRUE(bits_ops::UEqual(UBits(0, 42), UBits(0, 42)));
  EXPECT_FALSE(bits_ops::UEqual(UBits(0, 42), UBits(1, 42)));
  EXPECT_FALSE(bits_ops::UEqual(UBits(0, 42000), UBits(1, 42000)));
  EXPECT_TRUE(bits_ops::UEqual(UBits(333, 256), UBits(333, 512)));
  EXPECT_FALSE(bits_ops::UEqual(UBits(333, 256), UBits(444, 256)));

  EXPECT_TRUE(bits_ops::UEqual(Bits(), 0));
  EXPECT_TRUE(bits_ops::UEqual(UBits(0, 32), 0));
  EXPECT_TRUE(bits_ops::UEqual(UBits(44, 14), 44));
  EXPECT_TRUE(bits_ops::UEqual(Bits::AllOnes(32), 0xffffffffL));
  EXPECT_TRUE(bits_ops::UEqual(UBits(333, 256), 333));
  EXPECT_FALSE(bits_ops::UEqual(UBits(333, 256), 334));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits huge,
      ParseNumber(
          "0x2fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000_1234_5555"));
  EXPECT_TRUE(bits_ops::UEqual(huge, huge));
  EXPECT_TRUE(bits_ops::UEqual(huge, bits_ops::ZeroExtend(huge, 10000)));
}

TEST(BitsOpsTest, SignedComparisons) {
  Bits b42 = UBits(42, 100);
  Bits bminus_77 = SBits(-77, 64);
  Bits b123 = UBits(123, 16);

  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(b42, b42));
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(b42, bminus_77));
  EXPECT_FALSE(bits_ops::SGreaterThanOrEqual(bminus_77, b42));
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(b123, b42));

  EXPECT_FALSE(bits_ops::SGreaterThan(b42, b42));
  EXPECT_TRUE(bits_ops::SGreaterThan(b42, bminus_77));
  EXPECT_FALSE(bits_ops::SGreaterThan(bminus_77, b42));
  EXPECT_FALSE(bits_ops::SGreaterThanOrEqual(b42, b123));

  EXPECT_TRUE(bits_ops::SLessThanOrEqual(b42, b42));
  EXPECT_FALSE(bits_ops::SLessThanOrEqual(b42, bminus_77));
  EXPECT_TRUE(bits_ops::SLessThanOrEqual(bminus_77, b42));
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(b123, bminus_77));
  EXPECT_FALSE(bits_ops::SGreaterThanOrEqual(bminus_77, b123));
  EXPECT_TRUE(bits_ops::SLessThanOrEqual(bminus_77, b123));
  EXPECT_TRUE(bits_ops::SLessThanOrEqual(bminus_77, bminus_77));

  EXPECT_FALSE(bits_ops::SLessThan(b42, b42));
  EXPECT_FALSE(bits_ops::SLessThan(b42, bminus_77));
  EXPECT_TRUE(bits_ops::SLessThan(bminus_77, b42));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_negative,
      ParseNumber(
          "-0x2fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000_1234_5555"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_positive,
      ParseNumber(
          "0x3234_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000_1111_3333"));
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(wide_positive, wide_negative));
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(
      bits_ops::ZeroExtend(wide_positive, 1234), wide_negative));
  EXPECT_FALSE(bits_ops::SLessThan(wide_positive, wide_negative));
  EXPECT_TRUE(bits_ops::SLessThan(wide_negative, wide_positive));
  EXPECT_TRUE(bits_ops::SLessThan(bits_ops::SignExtend(wide_negative, 10000),
                                  wide_positive));
  EXPECT_FALSE(bits_ops::SLessThan(bits_ops::ZeroExtend(wide_negative, 10000),
                                   wide_positive));
}

TEST(BitsOpsTest, SignedEqualComparisons) {
  EXPECT_TRUE(bits_ops::SEqual(Bits(), SBits(0, 42)));
  EXPECT_TRUE(bits_ops::SEqual(SBits(0, 42), SBits(0, 42)));
  EXPECT_FALSE(bits_ops::SEqual(SBits(0, 42), SBits(1, 42)));
  EXPECT_TRUE(bits_ops::SEqual(SBits(-1000, 42), SBits(-1000, 42)));
  EXPECT_TRUE(bits_ops::SEqual(SBits(-1000, 42), SBits(-1000, 555)));
  EXPECT_TRUE(bits_ops::SEqual(UBits(10000, 42000), UBits(10000, 42000)));
  EXPECT_TRUE(bits_ops::SEqual(UBits(10000, 42000), UBits(10000, 123456)));
  EXPECT_TRUE(bits_ops::SEqual(UBits(-10000, 42000), UBits(-10000, 42000)));
  EXPECT_TRUE(bits_ops::SEqual(UBits(-10000, 42000), UBits(-10000, 100)));
  EXPECT_FALSE(bits_ops::SEqual(UBits(10000, 42000), UBits(-10000, 42000)));
  EXPECT_TRUE(bits_ops::SEqual(SBits(-1, 256), SBits(-1, 4)));

  EXPECT_TRUE(bits_ops::SEqual(Bits(), 0));
  EXPECT_TRUE(bits_ops::SEqual(UBits(0, 32), 0));
  EXPECT_TRUE(bits_ops::SEqual(UBits(44, 14), 44));
  EXPECT_TRUE(bits_ops::SEqual(Bits::AllOnes(1), -1));
  EXPECT_TRUE(bits_ops::SEqual(Bits::AllOnes(32), -1));
  EXPECT_TRUE(bits_ops::SEqual(Bits::AllOnes(3200), -1));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits huge,
      ParseNumber(
          "-0x2fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000_1234_5555"));
  EXPECT_TRUE(bits_ops::SEqual(huge, huge));
  EXPECT_FALSE(bits_ops::SEqual(huge, bits_ops::ZeroExtend(huge, 10000)));
  EXPECT_TRUE(bits_ops::SEqual(huge, bits_ops::SignExtend(huge, 10000)));
}

TEST(BitsOpsTest, Int64UnsignedComparisons) {
  Bits b42 = UBits(42, 64);
  EXPECT_TRUE(bits_ops::UGreaterThanOrEqual(b42, 42));
  EXPECT_FALSE(bits_ops::UGreaterThanOrEqual(b42, 123));
  EXPECT_TRUE(bits_ops::UGreaterThanOrEqual(b42, 7));

  EXPECT_FALSE(bits_ops::UGreaterThan(b42, 42));
  EXPECT_FALSE(bits_ops::UGreaterThan(b42, 100));
  EXPECT_TRUE(bits_ops::UGreaterThan(b42, 1));

  EXPECT_TRUE(bits_ops::ULessThanOrEqual(b42, 42));
  EXPECT_TRUE(bits_ops::ULessThanOrEqual(b42, 77));
  EXPECT_FALSE(bits_ops::ULessThanOrEqual(b42, 33));

  EXPECT_FALSE(bits_ops::ULessThan(b42, 42));
  EXPECT_TRUE(bits_ops::ULessThan(b42, 77));
  EXPECT_FALSE(bits_ops::ULessThan(b42, 2));

  EXPECT_TRUE(bits_ops::ULessThan(Bits(), 200));
  EXPECT_FALSE(bits_ops::UGreaterThan(Bits(), 200));
  EXPECT_FALSE(bits_ops::UGreaterThanOrEqual(UBits(42, 6), 200));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits huge,
      ParseNumber(
          "0x2fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000_1234_5555"));
  EXPECT_TRUE(bits_ops::UGreaterThanOrEqual(huge, 0));
  EXPECT_TRUE(
      bits_ops::UGreaterThanOrEqual(huge, std::numeric_limits<int64_t>::max()));
  EXPECT_TRUE(bits_ops::UGreaterThan(huge, 1234567));
  EXPECT_FALSE(
      bits_ops::ULessThanOrEqual(huge, std::numeric_limits<int64_t>::max()));
  EXPECT_FALSE(bits_ops::ULessThanOrEqual(huge, 33));
}

TEST(BitsOpsTest, Int64SignedComparisons) {
  Bits b42 = UBits(42, 10);
  Bits minus42 = SBits(-42, 20);
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(b42, 42));
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(minus42, -42));
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(b42, minus42));
  EXPECT_FALSE(bits_ops::UGreaterThanOrEqual(b42, minus42));
  EXPECT_FALSE(bits_ops::SGreaterThanOrEqual(b42, 123));
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(b42, 7));

  EXPECT_FALSE(bits_ops::SGreaterThan(b42, 42));
  EXPECT_FALSE(bits_ops::SGreaterThan(b42, 100));
  EXPECT_TRUE(bits_ops::SGreaterThan(b42, -100));

  EXPECT_TRUE(bits_ops::SLessThanOrEqual(b42, 42));
  EXPECT_TRUE(bits_ops::SLessThanOrEqual(b42, 77));
  EXPECT_FALSE(bits_ops::SLessThanOrEqual(b42, -33));
  EXPECT_TRUE(bits_ops::SLessThanOrEqual(minus42, -33));

  EXPECT_FALSE(bits_ops::SLessThan(b42, 42));
  EXPECT_TRUE(bits_ops::SLessThan(b42, 77));
  EXPECT_FALSE(bits_ops::SLessThan(b42, -10000));
  EXPECT_TRUE(bits_ops::SLessThan(minus42, -10));

  EXPECT_TRUE(
      bits_ops::SLessThan(minus42, std::numeric_limits<int64_t>::max()));
  EXPECT_FALSE(
      bits_ops::SLessThan(minus42, std::numeric_limits<int64_t>::min()));

  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_minus2,
      ParseNumber("0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_fffe"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits tmp,
      ParseNumber("0x2fff_ffff_ffff_ffff_ffff_0000_0000_0000_0000_0000_1234"));
  Bits huge = bits_ops::ZeroExtend(tmp, wide_minus2.bit_count());

  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(huge, 0));
  EXPECT_FALSE(bits_ops::SGreaterThanOrEqual(wide_minus2, 0));
  EXPECT_TRUE(bits_ops::SGreaterThanOrEqual(huge, wide_minus2));
  EXPECT_TRUE(
      bits_ops::SGreaterThanOrEqual(huge, std::numeric_limits<int64_t>::max()));
  EXPECT_TRUE(bits_ops::SGreaterThan(huge, 1234567));
  EXPECT_FALSE(bits_ops::SGreaterThan(wide_minus2, 1234567));
  EXPECT_FALSE(
      bits_ops::SLessThanOrEqual(huge, std::numeric_limits<int64_t>::max()));
  EXPECT_FALSE(bits_ops::SLessThanOrEqual(wide_minus2,
                                          std::numeric_limits<int64_t>::min()));
  EXPECT_FALSE(bits_ops::SLessThanOrEqual(huge, 33));
  EXPECT_TRUE(bits_ops::SLessThanOrEqual(wide_minus2, 33));
}

TEST(BitsOpsTest, ZeroAndSignExtend) {
  Bits empty_bits(0);
  EXPECT_TRUE(bits_ops::ZeroExtend(empty_bits, 47).IsZero());
  EXPECT_TRUE(bits_ops::SignExtend(empty_bits, 123).IsZero());

  Bits b0 = UBits(0, 1);
  EXPECT_TRUE(bits_ops::ZeroExtend(b0, 2).IsZero());
  EXPECT_TRUE(bits_ops::SignExtend(b0, 44).IsZero());

  Bits b1 = UBits(1, 1);
  EXPECT_EQ(bits_ops::ZeroExtend(b1, 32), UBits(1, 32));
  EXPECT_EQ(bits_ops::SignExtend(b1, 47), SBits(-1, 47));

  EXPECT_EQ(bits_ops::ZeroExtend(UBits(0x80, 8), 32), UBits(0x80, 32));
  EXPECT_EQ(bits_ops::SignExtend(UBits(0x80, 8), 32), UBits(0xffffff80UL, 32));
}

TEST(BitsOpsTest, ShiftLeftLogical) {
  Bits b0 = UBits(0, 11);
  EXPECT_EQ(bits_ops::ShiftLeftLogical(b0, 0), b0);

  Bits b1 = UBits(1, 16);
  EXPECT_EQ(bits_ops::ShiftLeftLogical(b1, 1), UBits(2, 16));
  EXPECT_EQ(bits_ops::ShiftLeftLogical(b1, 2), UBits(4, 16));

  Bits b2 = UBits(0x7FFF, 16);
  EXPECT_EQ(bits_ops::ShiftLeftLogical(b2, 1), UBits(0xFFFE, 16));

  Bits b3 = UBits(0x0010, 16);
  EXPECT_EQ(bits_ops::ShiftLeftLogical(b3, 8), UBits(0x1000, 16));

  Bits b4 = UBits(0x0001, 16);
  EXPECT_EQ(bits_ops::ShiftLeftLogical(b4, 15), UBits(0x8000, 16));

  Bits b5 = UBits(0xf, 4);
  EXPECT_EQ(bits_ops::ShiftLeftLogical(b5, 4), UBits(0, 4));
}

TEST(BitsOpsTest, ShiftRightLogical) {
  Bits b0 = UBits(0, 11);
  EXPECT_EQ(bits_ops::ShiftRightLogical(b0, 0), UBits(0, 11));

  Bits b1 = UBits(4, 16);
  EXPECT_EQ(bits_ops::ShiftRightLogical(b1, 1), UBits(2, 16));
  EXPECT_EQ(bits_ops::ShiftRightLogical(b1, 2), UBits(1, 16));

  Bits b2 = UBits(0xFFFE, 16);
  EXPECT_EQ(bits_ops::ShiftRightLogical(b2, 1), UBits(0x7FFF, 16));

  Bits b3 = UBits(0x1000, 16);
  EXPECT_EQ(bits_ops::ShiftRightLogical(b3, 8), UBits(0x0010, 16));

  Bits b4 = UBits(0x8000, 16);
  EXPECT_EQ(bits_ops::ShiftRightLogical(b4, 15), UBits(0x0001, 16));
}

TEST(BitsOpsTest, ShiftRightArith) {
  Bits b1 = UBits(0x8080, 16);
  EXPECT_EQ(bits_ops::ShiftRightArith(b1, 1), UBits(0xc040, 16));

  Bits b2 = UBits(0b10000100, 8);
  EXPECT_EQ(bits_ops::ShiftRightArith(b2, 2), UBits(0b11100001, 8));

  Bits b3 = UBits(0b11111111, 8);
  EXPECT_EQ(bits_ops::ShiftRightArith(b3, 7), UBits(0b11111111, 8));

  Bits b4 = UBits(0xF000000000000000, 64);
  EXPECT_EQ(bits_ops::ShiftRightArith(b4, 4), UBits(0xFF00000000000000, 64));

  Bits b5 = SBits(-1, 64);  // All hexadecimal F's
  EXPECT_EQ(bits_ops::ShiftRightArith(b5, 63), SBits(-1, 64));

  // Shift by the full bit width.
  Bits b6 = SBits(-1, 2);
  Bits b6_shifted = bits_ops::ShiftRightArith(b6, 2);
  EXPECT_EQ(b6_shifted, SBits(-1, 2));
}

TEST(BitsOpsTest, Negate) {
  EXPECT_EQ(bits_ops::Negate(Bits(0)), Bits(0));
  EXPECT_EQ(bits_ops::Negate(UBits(0, 1)), UBits(0, 1));

  // A single-bit 1 as twos-complement is -1.
  EXPECT_EQ(bits_ops::Negate(UBits(1, 1)), UBits(1, 1));

  EXPECT_EQ(bits_ops::Negate(SBits(-4, 3)), UBits(4, 3));
  EXPECT_EQ(bits_ops::Negate(UBits(42, 37)), SBits(-42, 37));
  EXPECT_EQ(bits_ops::Negate(UBits(std::numeric_limits<int64_t>::min(), 64)),
            UBits(0x8000000000000000ULL, 64));
  EXPECT_EQ(bits_ops::Negate(UBits(0, 1234)), UBits(0, 1234));
  EXPECT_EQ(bits_ops::Negate(UBits(1, 1234)), SBits(-1, 1234));
}

TEST(BitsOpsTest, OneHot) {
  EXPECT_EQ(bits_ops::OneHotLsbToMsb(Bits(0)), UBits(1, 1));
  EXPECT_EQ(bits_ops::OneHotMsbToLsb(Bits(0)), UBits(1, 1));

  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(0, 1)), UBits(0b10, 2));
  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(1, 1)), UBits(0b01, 2));
  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(0, 1)), UBits(0b10, 2));
  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(1, 1)), UBits(0b01, 2));

  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(0b00, 2)), UBits(0b100, 3));
  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(0b01, 2)), UBits(0b001, 3));
  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(0b10, 2)), UBits(0b010, 3));
  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(0b11, 2)), UBits(0b001, 3));

  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(0b00, 2)), UBits(0b100, 3));
  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(0b01, 2)), UBits(0b001, 3));
  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(0b10, 2)), UBits(0b010, 3));
  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(0b11, 2)), UBits(0b010, 3));

  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(0b00010000, 8)),
            UBits(0b000010000, 9));
  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(0b00110010, 8)),
            UBits(0b000000010, 9));
  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(0b11111111, 8)),
            UBits(0b000000001, 9));
  EXPECT_EQ(bits_ops::OneHotLsbToMsb(UBits(0b00000000, 8)),
            UBits(0b100000000, 9));

  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(0b00010000, 8)),
            UBits(0b000010000, 9));
  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(0b00110010, 8)),
            UBits(0b000100000, 9));
  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(0b11111111, 8)),
            UBits(0b010000000, 9));
  EXPECT_EQ(bits_ops::OneHotMsbToLsb(UBits(0b00000000, 8)),
            UBits(0b100000000, 9));
}

TEST(BitsOpsTest, Nor3) {
  std::vector<std::pair<std::array<uint8_t, 3>, uint8_t>> cases = {
      {{0, 0, 0}, 1},  //
      {{0, 0, 1}, 0},  //
      {{0, 1, 0}, 0},  //
      {{0, 1, 1}, 0},  //
      {{1, 0, 0}, 0},  //
      {{1, 0, 1}, 0},  //
      {{1, 1, 0}, 0},  //
      {{1, 1, 1}, 0},  //
  };
  for (auto test_case : cases) {
    EXPECT_EQ(bits_ops::NaryNor({UBits(test_case.first[0], 1),
                                 UBits(test_case.first[1], 1),
                                 UBits(test_case.first[2], 1)}),
              UBits(test_case.second, 1));
  }
}

TEST(BitsOpsTest, Nand3) {
  std::vector<std::pair<std::array<uint8_t, 3>, uint8_t>> cases = {
      {{0, 0, 0}, 1},  //
      {{0, 0, 1}, 1},  //
      {{0, 1, 0}, 1},  //
      {{0, 1, 1}, 1},  //
      {{1, 0, 0}, 1},  //
      {{1, 0, 1}, 1},  //
      {{1, 1, 0}, 1},  //
      {{1, 1, 1}, 0},  //
  };
  for (auto test_case : cases) {
    EXPECT_EQ(bits_ops::NaryNand({UBits(test_case.first[0], 1),
                                  UBits(test_case.first[1], 1),
                                  UBits(test_case.first[2], 1)}),
              UBits(test_case.second, 1));
  }
}

TEST(BitsOpsTest, Reverse) {
  EXPECT_EQ(bits_ops::Reverse(Bits()), Bits());
  EXPECT_EQ(bits_ops::Reverse(UBits(0, 1)), UBits(0, 1));
  EXPECT_EQ(bits_ops::Reverse(UBits(1, 1)), UBits(1, 1));
  EXPECT_EQ(bits_ops::Reverse(UBits(1, 100)), Bits::PowerOfTwo(99, 100));
  EXPECT_EQ(bits_ops::Reverse(UBits(0b111001, 6)), UBits(0b100111, 6));
  EXPECT_EQ(bits_ops::Reverse(UBits(0b111001, 10)), UBits(0b1001110000, 10));
}

TEST(BitsOpsTest, ReductionOps) {
  EXPECT_EQ(bits_ops::AndReduce(UBits(0, 1)), UBits(0, 1));
  EXPECT_EQ(bits_ops::AndReduce(UBits(1, 1)), UBits(1, 1));
  EXPECT_EQ(bits_ops::AndReduce(Bits::AllOnes(128)), UBits(1, 1));
  EXPECT_EQ(bits_ops::AndReduce(UBits(128, 128)), UBits(0, 1));

  EXPECT_EQ(bits_ops::OrReduce(UBits(0, 1)), UBits(0, 1));
  EXPECT_EQ(bits_ops::OrReduce(UBits(1, 1)), UBits(1, 1));
  EXPECT_EQ(bits_ops::OrReduce(Bits::AllOnes(128)), UBits(1, 1));
  EXPECT_EQ(bits_ops::OrReduce(UBits(128, 128)), UBits(1, 1));

  EXPECT_EQ(bits_ops::XorReduce(UBits(0, 1)), UBits(0, 1));
  EXPECT_EQ(bits_ops::XorReduce(UBits(1, 1)), UBits(1, 1));
  EXPECT_EQ(bits_ops::XorReduce(Bits::AllOnes(128)), UBits(0, 1));
  EXPECT_EQ(bits_ops::XorReduce(UBits(127, 128)), UBits(1, 1));
}

TEST(BitsOpsTest, DropLeadingZeroes) {
  EXPECT_EQ(bits_ops::DropLeadingZeroes(Bits()), Bits());
  EXPECT_EQ(bits_ops::DropLeadingZeroes(UBits(0, 1)), Bits());
  EXPECT_EQ(bits_ops::DropLeadingZeroes(UBits(1, 1)), UBits(1, 1));
  EXPECT_EQ(bits_ops::DropLeadingZeroes(UBits(0, 2)), Bits());
  EXPECT_EQ(bits_ops::DropLeadingZeroes(UBits(1, 2)), UBits(1, 1));
  EXPECT_EQ(bits_ops::DropLeadingZeroes(UBits(0x5a5a, 16)), UBits(0x5a5a, 15));
  EXPECT_EQ(bits_ops::DropLeadingZeroes(UBits(0x0, 16)), Bits());
  EXPECT_EQ(bits_ops::DropLeadingZeroes(UBits(0xFFF0, 16)), UBits(0xFFF0, 16));
}

TEST(BitsOpsTest, BitSliceUpdate) {
  // Validate a few values.
  EXPECT_EQ(bits_ops::BitSliceUpdate(Bits(), 0, Bits()), Bits());
  EXPECT_EQ(bits_ops::BitSliceUpdate(Bits(), 42, Bits()), Bits());
  EXPECT_EQ(bits_ops::BitSliceUpdate(UBits(0x1234abcd, 32), 0, UBits(7, 4)),
            UBits(0x1234abc7, 32));
  EXPECT_EQ(bits_ops::BitSliceUpdate(UBits(0x1234abcd, 32), 12, UBits(7, 4)),
            UBits(0x12347bcd, 32));
  EXPECT_EQ(
      bits_ops::BitSliceUpdate(UBits(0x1234abcd, 32), 10000000, UBits(7, 4)),
      UBits(0x1234abcd, 32));

  // Exhaustively test all values of subect and update value from 0 to 5 bits
  // with start index ranging from 0 to 6.
  for (int64_t subject_width = 0; subject_width <= 5; ++subject_width) {
    for (int64_t subject_value = 0; subject_value < (1 << subject_width);
         ++subject_value) {
      Bits subject = UBits(subject_value, subject_width);
      for (int64_t update_width = 0; update_width <= 5; ++update_width) {
        for (int64_t update_value = 0; update_value < (1 << update_width);
             ++update_value) {
          Bits update = UBits(update_value, update_width);
          for (int64_t start = 0; start <= 6; ++start) {
            // Create a mask like: 11..1100..0011..11 where the least-significan
            // string of 1's is 'start' long, and the number of zeros is equals
            // to update_width.
            int64_t mask = ~(((1 << update_width) - 1) << start);
            int64_t expected = (mask & subject.ToUint64().value()) |
                               (update.ToUint64().value() << start);
            int64_t expected_trunc = expected & ((1 << subject_width) - 1);
            EXPECT_EQ(UBits(expected_trunc, subject_width),
                      bits_ops::BitSliceUpdate(subject, start, update));
          }
        }
      }
    }
  }
}

}  // namespace
}  // namespace xls
