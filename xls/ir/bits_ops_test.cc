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

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "benchmark/benchmark.h"
#include "gtest/gtest.h"
#include "fuzztest/fuzztest.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/status/matchers.h"
#include "xls/data_structures/inline_bitmap.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_test_utils.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/number_parser.h"

namespace xls {
namespace {

std::string ToPlainString(const std::string& s) {
  return absl::StrJoin(absl::StrSplit(s.substr(2), '_'), "");
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
      BitsToString(bits_ops::Concat({deadbeef2, fofo}), FormatPreference::kHex),
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
  EXPECT_EQ(
      BitsToString(bits_ops::Add(wide_lhs, wide_rhs), FormatPreference::kHex),
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
        BitsToString(bits_ops::Sub(wide_lhs, wide_rhs), FormatPreference::kHex),
        "0xfff_ffff_ffff_ffff_fffd_edcb_bbbb_cccc_ffff_ee0e");
  }
  {
    // Test an underflow case.
    XLS_ASSERT_OK_AND_ASSIGN(
        Bits wide_lhs,
        ParseNumber("0x1000_0000_0000_0000_0000_0000_0000_0000_0000"));
    Bits wide_rhs = UBits(42, wide_lhs.bit_count());
    EXPECT_EQ(
        BitsToString(bits_ops::Sub(wide_lhs, wide_rhs), FormatPreference::kHex),
        "0xfff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffd6");
  }
}

TEST(BitsOpsTest, Increment) {
  EXPECT_EQ(bits_ops::Increment(Bits()), Bits());
  EXPECT_EQ(bits_ops::Increment(UBits(23, 64)), UBits(24, 64));

  // Test overflow conditions.
  EXPECT_EQ(bits_ops::Increment(UBits(15, 4)), UBits(0, 4));
  EXPECT_EQ(
      bits_ops::Increment(UBits(std::numeric_limits<uint64_t>::max(), 64)),
      UBits(0, 64));

  // Test wide values.
  XLS_ASSERT_OK_AND_ASSIGN(
      Bits wide_value,
      ParseNumber("0x1000_0000_0000_0000_0042_1fff_ffff_ffff_ffff_ffff"));
  EXPECT_EQ(
      BitsToString(bits_ops::Increment(wide_value), FormatPreference::kHex),
      "0x1000_0000_0000_0000_0042_2000_0000_0000_0000_0000");
}

void IncrementEqualsAdd1(const Bits& bits) {
  EXPECT_EQ(bits_ops::Increment(bits),
            bits_ops::Add(bits, UBits(1, bits.bit_count())));
}
FUZZ_TEST(BitsOpsFuzzTest, IncrementEqualsAdd1).WithDomains(NonemptyBits());

TEST(BitsOpsTest, Decrement) {
  EXPECT_EQ(bits_ops::Decrement(Bits()), Bits());
  EXPECT_EQ(bits_ops::Decrement(UBits(55, 64)), UBits(54, 64));

  // Test underflow conditions.
  EXPECT_EQ(bits_ops::Decrement(UBits(0, 4)), UBits(15, 4));
  EXPECT_EQ(bits_ops::Decrement(UBits(0, 64)),
            UBits(std::numeric_limits<uint64_t>::max(), 64));
  EXPECT_EQ(bits_ops::Decrement(SBits(-8, 4)), SBits(7, 4));

  // Test wide values.
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        Bits wide_value,
        ParseNumber("0x1234_5678_1234_5678_0000_0000_0000_0000_0000_0000"));
    EXPECT_EQ(
        BitsToString(bits_ops::Decrement(wide_value), FormatPreference::kHex),
        "0x1234_5678_1234_5677_ffff_ffff_ffff_ffff_ffff_ffff");
  }
  {
    // Test an underflow case.
    Bits wide_zero(144);
    EXPECT_EQ(
        BitsToString(bits_ops::Decrement(wide_zero), FormatPreference::kHex),
        "0xffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff_ffff");
  }
}

void DecrementEqualsSub1(const Bits& bits) {
  EXPECT_EQ(bits_ops::Decrement(bits),
            bits_ops::Sub(bits, UBits(1, bits.bit_count())));
}
FUZZ_TEST(BitsOpsFuzzTest, DecrementEqualsSub1).WithDomains(NonemptyBits());

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
  EXPECT_EQ(
      BitsToString(bits_ops::UMul(wide_lhs, wide_rhs), FormatPreference::kHex),
      "0x200_0000_0000_0000_0000_1246_8888_8666_6000_0249_8dcb_bbbb_cccc_"
      "ffff_ee12_b179_9995_3326_0004_b168");

  Bits result = bits_ops::UMul(Bits::AllOnes(65), Bits::AllOnes(65));
  EXPECT_EQ(result.bit_count(), 130);
  EXPECT_EQ(BitsToString(result, FormatPreference::kHex),
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
  EXPECT_EQ(
      BitsToString(bits_ops::SMul(wide_lhs, wide_rhs), FormatPreference::kHex),
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
  EXPECT_EQ(BitsToString(
                bits_ops::UDiv(wide_lhs, bits_ops::ZeroExtend(
                                             wide_rhs, wide_lhs.bit_count())),
                FormatPreference::kHex),
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
  EXPECT_EQ(BitsToString(
                bits_ops::UMod(wide_lhs, bits_ops::ZeroExtend(
                                             wide_rhs, wide_lhs.bit_count())),
                FormatPreference::kHex),
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
  EXPECT_EQ(BitsToString(
                bits_ops::SDiv(wide_lhs, bits_ops::ZeroExtend(
                                             wide_rhs, wide_lhs.bit_count())),
                FormatPreference::kHex),
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
  EXPECT_EQ(BitsToString(
                bits_ops::SMod(wide_lhs, bits_ops::ZeroExtend(
                                             wide_rhs, wide_lhs.bit_count())),
                FormatPreference::kHex),
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

TEST(BitsOpsTest, LongestCommonPrefixLSBTypical) {
  // Simple typical example
  Bits x(absl::InlinedVector<bool, 1>{1, 1, 0, 1, 1});
  Bits y(absl::InlinedVector<bool, 1>{1, 1, 0, 0, 1});
  Bits expected(absl::InlinedVector<bool, 1>{1, 1, 0});
  EXPECT_EQ(bits_ops::LongestCommonPrefixLSB({x, y}), expected);
}

TEST(BitsOpsTest, LongestCommonPrefixLSBEmptyResult) {
  // Differ in the first bit => common prefix is the empty bitstring
  Bits x(absl::InlinedVector<bool, 1>{0, 1, 0, 1, 1});
  Bits y(absl::InlinedVector<bool, 1>{1, 1, 0, 0, 1});
  Bits expected(0);
  EXPECT_EQ(bits_ops::LongestCommonPrefixLSB({x, y}), expected);
}

TEST(BitsOpsTest, LongestCommonPrefixLSBSame) {
  // Everything the same => common prefix is the entire bitstring
  Bits x(absl::InlinedVector<bool, 1>{1, 1, 0, 0, 1});
  Bits y(absl::InlinedVector<bool, 1>{1, 1, 0, 0, 1});
  Bits expected(absl::InlinedVector<bool, 1>{1, 1, 0, 0, 1});
  EXPECT_EQ(bits_ops::LongestCommonPrefixLSB({x, y}), expected);
}

TEST(BitsOpsTest, LongestCommonPrefixLSBMoreThan2) {
  // Example with more than 2 bitstrings
  Bits x(absl::InlinedVector<bool, 1>{0, 1, 1, 1, 1});
  Bits y(absl::InlinedVector<bool, 1>{0, 1, 1, 0, 1});
  Bits z(absl::InlinedVector<bool, 1>{0, 1, 1, 1, 0});
  Bits expected(absl::InlinedVector<bool, 1>{0, 1, 1});
  EXPECT_EQ(bits_ops::LongestCommonPrefixLSB({x, y, z}), expected);
}

void TestBinary(const Bits& b, const std::string& expected) {
  EXPECT_EQ(BitsToString(b, FormatPreference::kBinary), expected);
  EXPECT_EQ(BitsToString(b, FormatPreference::kPlainBinary),
            ToPlainString(expected));
}

void TestHex(const Bits& b, const std::string& expected) {
  EXPECT_EQ(BitsToString(b, FormatPreference::kHex), expected);
  EXPECT_EQ(BitsToString(b, FormatPreference::kPlainHex),
            ToPlainString(expected));
}

void TestDecimal(const Bits& b, const std::string& unsigned_str,
                 const std::string& signed_str) {
  EXPECT_EQ(BitsToString(b, FormatPreference::kUnsignedDecimal), unsigned_str);
  EXPECT_EQ(BitsToString(b, FormatPreference::kSignedDecimal), signed_str);
}

TEST(BitsOpsTest, ToStringEmptyBits) {
  Bits empty_bits(0);
  EXPECT_EQ(BitsToString(empty_bits, FormatPreference::kUnsignedDecimal), "0");
  EXPECT_EQ(BitsToString(empty_bits, FormatPreference::kSignedDecimal), "0");
  TestHex(empty_bits, "0x0");
  TestBinary(empty_bits, "0b0");
  TestDecimal(empty_bits, "0", "0");
  EXPECT_EQ(BitsToString(empty_bits, FormatPreference::kUnsignedDecimal,
                         /*include_bit_count=*/true),
            "0 [0 bits]");
  EXPECT_EQ(BitsToString(empty_bits, FormatPreference::kSignedDecimal,
                         /*include_bit_count=*/true),
            "0 [0 bits]");
}

TEST(BitsOpsTest, U128HighBit) {
  auto b = Bits(128).UpdateWithSet(127, true);
  TestHex(b, "0x8000_0000_0000_0000_0000_0000_0000_0000");
  TestBinary(b,
             "0b1000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_"
             "0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_0000_"
             "0000_0000_0000_0000_0000_0000_0000");
  TestDecimal(b, "170141183460469231731687303715884105728",
              "-170141183460469231731687303715884105728");
}

TEST(BitsOpsTest, ToStringValue1U1) {
  Bits b1 = UBits(1, 1);
  EXPECT_EQ(BitsToString(b1, FormatPreference::kUnsignedDecimal), "1");
  EXPECT_EQ(BitsToString(b1, FormatPreference::kSignedDecimal), "-1");
  TestHex(b1, "0x1");
  TestBinary(b1, "0b1");
  TestDecimal(b1, "1", "-1");
}

TEST(BitsOptTest, ToStringValue1U16) {
  const Bits b = UBits(1, 16);
  TestBinary(b, "0b1");
  TestHex(b, "0x1");
  TestDecimal(b, "1", "1");
}

TEST(BitsOptTest, ToStringValue42U7) {
  Bits b42 = UBits(42, 7);
  EXPECT_EQ(BitsToString(b42, FormatPreference::kUnsignedDecimal), "42");
  EXPECT_EQ(BitsToString(b42, FormatPreference::kSignedDecimal), "42");
  TestHex(b42, "0x2a");
  TestBinary(b42, "0b10_1010");
  TestDecimal(b42, "42", "42");
}

TEST(BitsOptTest, ToStringPrimeBits64) {
  const Bits prime64 = PrimeBits(64);
  EXPECT_EQ(
      prime64.ToDebugString(),
      "0b0010100000100010100010100010000010100010100010100010101010111100");
  EXPECT_EQ(BitsToString(prime64, FormatPreference::kUnsignedDecimal),
            "2892025783495830204");
  EXPECT_EQ(BitsToString(prime64, FormatPreference::kSignedDecimal),
            "2892025783495830204");
  TestHex(prime64, "0x2822_8a20_a28a_2abc");
  TestBinary(prime64,
             "0b10_1000_0010_0010_1000_1010_0010_0000_1010_0010_1000_1010_0010_"
             "1010_1011_1100");
  TestDecimal(prime64, "2892025783495830204", "2892025783495830204");
}

TEST(BitsOptTest, ToStringPrimeBits65) {
  // Test widths wider than 64. Decimal output for wide bit counts is not
  // supported.
  const Bits b = PrimeBits(65);
  EXPECT_EQ(
      b.ToDebugString(),
      "0b00010100000100010100010100010000010100010100010100010101010111100");
  TestHex(b, "0x2822_8a20_a28a_2abc");
  TestBinary(b,
             "0b10_1000_0010_0010_1000_1010_0010_0000_1010_0010_1000_1010_0010_"
             "1010_1011_1100");
  TestDecimal(b, "2892025783495830204", "2892025783495830204");
}

TEST(BitsOptTest, ToStringPrimeBits96) {
  const Bits b = PrimeBits(96);
  EXPECT_EQ(b.ToDebugString(),
            "0b0000001000001000100000101000100000101000001000101000101000100000"
            "10100010100010100010101010111100");
  TestHex(b, "0x208_8288_2822_8a20_a28a_2abc");
  TestBinary(b,
             "0b10_0000_1000_1000_0010_1000_1000_0010_1000_0010_0010_1000_1010_"
             "0010_0000_1010_0010_1000_1010_0010_1010_1011_1100");
  TestDecimal(b, "629257845491600032719841980", "629257845491600032719841980");
}

TEST(BitsOpsTest, ToRawStringEmptyBits) {
  Bits empty_bits(0);
  EXPECT_EQ(empty_bits.ToDebugString(), "0b");
  EXPECT_EQ(BitsToRawDigits(empty_bits, FormatPreference::kUnsignedDecimal),
            "0");
  EXPECT_EQ(BitsToRawDigits(empty_bits, FormatPreference::kSignedDecimal), "0");
  EXPECT_EQ(BitsToRawDigits(empty_bits, FormatPreference::kHex), "0");
  EXPECT_EQ(BitsToRawDigits(empty_bits, FormatPreference::kBinary), "0");
  EXPECT_EQ(BitsToRawDigits(empty_bits, FormatPreference::kHex,
                            /*emit_leading_zeros=*/true),
            "0");
  EXPECT_EQ(BitsToRawDigits(empty_bits, FormatPreference::kBinary,
                            /*emit_leading_zeros=*/true),
            "0");
}

TEST(BitsOpsTest, ToRawDigitsValue1U16) {
  EXPECT_EQ(UBits(1, 16).ToDebugString(), "0b0000000000000001");
  EXPECT_EQ(BitsToRawDigits(UBits(1, 16), FormatPreference::kBinary), "1");
  EXPECT_EQ(BitsToRawDigits(UBits(1, 16), FormatPreference::kHex), "1");
  EXPECT_EQ(BitsToRawDigits(UBits(1, 16), FormatPreference::kBinary,
                            /*emit_leading_zeros=*/true),
            "0000_0000_0000_0001");
  EXPECT_EQ(BitsToRawDigits(UBits(1, 16), FormatPreference::kPlainBinary,
                            /*emit_leading_zeros=*/true),
            "0000000000000001");
  EXPECT_EQ(BitsToRawDigits(UBits(1, 16), FormatPreference::kHex,
                            /*emit_leading_zeros=*/true),
            "0001");
  EXPECT_EQ(BitsToRawDigits(UBits(1, 16), FormatPreference::kPlainHex,
                            /*emit_leading_zeros=*/true),
            "0001");
}

TEST(BitsOpsTest, ToRawDigitsValue0x1bU13) {
  EXPECT_EQ(UBits(0x1b, 13).ToDebugString(), "0b0000000011011");
  EXPECT_EQ(BitsToRawDigits(UBits(0x1b, 13), FormatPreference::kBinary),
            "1_1011");
  EXPECT_EQ(BitsToRawDigits(UBits(0x1b, 13), FormatPreference::kHex), "1b");
  EXPECT_EQ(BitsToRawDigits(UBits(0x1b, 13), FormatPreference::kBinary,
                            /*emit_leading_zeros=*/true),
            "0_0000_0001_1011");
  EXPECT_EQ(BitsToRawDigits(UBits(0x1b, 13), FormatPreference::kPlainBinary,
                            /*emit_leading_zeros=*/true),
            "0000000011011");
  EXPECT_EQ(BitsToRawDigits(UBits(0x1b, 13), FormatPreference::kHex,
                            /*emit_leading_zeros=*/true),
            "001b");
  EXPECT_EQ(BitsToRawDigits(UBits(0x1b, 13), FormatPreference::kPlainHex,
                            /*emit_leading_zeros=*/true),
            "001b");
}

TEST(BitsOpsTest, ToRawDigitsValue0x55U17) {
  EXPECT_EQ(UBits(0x55, 17).ToDebugString(), "0b00000000001010101");
  EXPECT_EQ(BitsToRawDigits(UBits(0x55, 17), FormatPreference::kBinary,
                            /*emit_leading_zeros=*/true),
            "0_0000_0000_0101_0101");
  EXPECT_EQ(BitsToRawDigits(UBits(0x55, 17), FormatPreference::kPlainBinary,
                            /*emit_leading_zeros=*/true),
            "00000000001010101");
  EXPECT_EQ(BitsToRawDigits(UBits(0x55, 17), FormatPreference::kHex,
                            /*emit_leading_zeros=*/true),
            "0_0055");
  EXPECT_EQ(BitsToRawDigits(UBits(0x55, 17), FormatPreference::kPlainHex,
                            /*emit_leading_zeros=*/true),
            "00055");
}

void BM_Increment(benchmark::State& state) {
  Bits f = Bits::AllOnes(state.range(0));
  for (auto _ : state) {
    auto v = bits_ops::Increment(f);
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(BM_Increment)->Range(64, 1 << 20);

void BM_AddNewOne(benchmark::State& state) {
  Bits f = Bits::AllOnes(state.range(0));
  for (auto _ : state) {
    auto b = InlineBitmap::FromWord(1, state.range(0));
    benchmark::DoNotOptimize(b);
    auto one = Bits::FromBitmap(b);
    benchmark::DoNotOptimize(one);
    auto v = bits_ops::Add(f, one);
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(BM_AddNewOne)->Range(64, 1 << 20);

void BM_AddCachedOne(benchmark::State& state) {
  Bits f = Bits::AllOnes(state.range(0));
  Bits one = Bits::FromBitmap(InlineBitmap::FromWord(1, state.range(0)));
  for (auto _ : state) {
    auto v = bits_ops::Add(f, one);
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(BM_AddCachedOne)->Range(64, 1 << 20);

void BM_Decrement(benchmark::State& state) {
  Bits f(state.range(0));
  for (auto _ : state) {
    auto v = bits_ops::Decrement(f);
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(BM_Decrement)->Range(64, 1 << 20);

void BM_SubNewOne(benchmark::State& state) {
  Bits f(state.range(0));
  for (auto _ : state) {
    auto b = InlineBitmap::FromWord(1, state.range(0));
    benchmark::DoNotOptimize(b);
    auto one = Bits::FromBitmap(b);
    benchmark::DoNotOptimize(one);
    auto v = bits_ops::Sub(f, one);
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(BM_SubNewOne)->Range(64, 1 << 20);

void BM_SubCachedOne(benchmark::State& state) {
  Bits f(state.range(0));
  Bits one = Bits::FromBitmap(InlineBitmap::FromWord(1, state.range(0)));
  for (auto _ : state) {
    auto v = bits_ops::Sub(f, one);
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(BM_SubCachedOne)->Range(64, 1 << 20);

void BM_Truncate(benchmark::State& state) {
  for (auto _ : state) {
    Bits f(256);
    auto v = bits_ops::Truncate(f, state.range(0));
    benchmark::DoNotOptimize(v);
    benchmark::DoNotOptimize(f);
  }
}
BENCHMARK(BM_Truncate)->Range(0, 255);

void BM_SignExtend(benchmark::State& state) {
  for (auto _ : state) {
    Bits f = state.range(0) ? Bits(32) : Bits::AllOnes(32);
    auto v = bits_ops::SignExtend(f, state.range(1));
    benchmark::DoNotOptimize(v);
    benchmark::DoNotOptimize(f);
  }
}
BENCHMARK(BM_SignExtend)->RangePair(0, 1, 33, 1 << 20);

void BM_ZeroExtend(benchmark::State& state) {
  for (auto _ : state) {
    Bits f(32);
    auto v = bits_ops::ZeroExtend(f, state.range(0));
    benchmark::DoNotOptimize(v);
    benchmark::DoNotOptimize(f);
  }
}
BENCHMARK(BM_ZeroExtend)->Range(33, 1 << 20);

void BM_TruncateMove(benchmark::State& state) {
  for (auto _ : state) {
    Bits f(256);
    auto v = bits_ops::Truncate(std::move(f), state.range(0));
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(BM_TruncateMove)->Range(0, 255);

void BM_SignExtendMove(benchmark::State& state) {
  for (auto _ : state) {
    Bits f = state.range(0) ? Bits(32) : Bits::AllOnes(32);
    auto v = bits_ops::SignExtend(std::move(f), state.range(1));
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(BM_SignExtendMove)->RangePair(0, 1, 33, 1 << 20);

void BM_ZeroExtendMove(benchmark::State& state) {
  for (auto _ : state) {
    Bits f(32);
    auto v = bits_ops::ZeroExtend(std::move(f), state.range(0));
    benchmark::DoNotOptimize(v);
  }
}
BENCHMARK(BM_ZeroExtendMove)->Range(33, 1 << 20);

}  // namespace
}  // namespace xls
