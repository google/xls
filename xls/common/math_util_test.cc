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

#include "xls/common/math_util.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/base/macros.h"

namespace xls {
namespace {

// Number of arguments for each test of the CeilOrRatio method
const int kNumTestArguments = 4;

template <typename IntegralType>
void TestCeilOfRatio(const IntegralType test_data[][kNumTestArguments],
                     int num_tests) {
  for (int i = 0; i < num_tests; ++i) {
    const IntegralType numerator = test_data[i][0];
    const IntegralType denominator = test_data[i][1];
    const IntegralType expected_floor = test_data[i][2];
    const IntegralType expected_ceil = test_data[i][3];
    // Make sure the two ways to compute the floor return the same thing.
    IntegralType floor_1 = FloorOfRatio(numerator, denominator);
    IntegralType floor_2 =
        CeilOrFloorOfRatio<IntegralType, false>(numerator, denominator);
    EXPECT_EQ(floor_1, floor_2);
    EXPECT_EQ(expected_floor, floor_1)
        << "FloorOfRatio fails with numerator = " << numerator
        << ", denominator = " << denominator
        << (std::numeric_limits<IntegralType>::is_signed ? "signed "
                                                         : "unsigned ")
        << (8 * sizeof(IntegralType)) << " bits";
    IntegralType ceil_1 = CeilOfRatio(numerator, denominator);
    IntegralType ceil_2 =
        CeilOrFloorOfRatio<IntegralType, true>(numerator, denominator);
    EXPECT_EQ(ceil_1, ceil_2);
    EXPECT_EQ(expected_ceil, ceil_1)
        << "CeilOfRatio fails with numerator = " << numerator
        << ", denominator = " << denominator
        << (std::numeric_limits<IntegralType>::is_signed ? "signed "
                                                         : "unsigned ")
        << (8 * sizeof(IntegralType)) << " bits";
  }
}

template <typename UnsignedIntegralType>
void TestCeilOfRatioUnsigned() {
  using Limits = std::numeric_limits<UnsignedIntegralType>;
  EXPECT_TRUE(Limits::is_integer);
  EXPECT_FALSE(Limits::is_signed);
  const UnsignedIntegralType kMax = Limits::max();
  const UnsignedIntegralType kTestData[][kNumTestArguments] = {
      // Numerator  | Denominator | Expected floor of ratio | Expected ceil of
      // ratio |
      // When numerator = 0, the result is always zero
      {0, 1, 0, 0},
      {0, 2, 0, 0},
      {0, kMax, 0, 0},
      // Try some non-extreme cases
      {1, 1, 1, 1},
      {5, 2, 2, 3},
      // Try with huge positive numerator
      {kMax, 1, kMax, kMax},
      {kMax, 2, kMax / 2, kMax / 2 + ((kMax % 2 != 0) ? 1 : 0)},
      {kMax, 3, kMax / 3, kMax / 3 + ((kMax % 3 != 0) ? 1 : 0)},
      // Try with a huge positive denominator
      {1, kMax, 0, 1},
      {2, kMax, 0, 1},
      {3, kMax, 0, 1},
      // Try with a huge numerator and a huge denominator
      {kMax, kMax, 1, 1},
  };
  const int kNumTests = ABSL_ARRAYSIZE(kTestData);
  TestCeilOfRatio<UnsignedIntegralType>(kTestData, kNumTests);
}

template <typename SignedInteger>
void TestCeilOfRatioSigned() {
  using Limits = std::numeric_limits<SignedInteger>;
  EXPECT_TRUE(Limits::is_integer);
  EXPECT_TRUE(Limits::is_signed);
  const SignedInteger kMin = Limits::min();
  const SignedInteger kMax = Limits::max();
  const SignedInteger kTestData[][kNumTestArguments] = {
      // Numerator  | Denominator | Expected floor of ratio | Expected ceil of
      // ratio |
      // When numerator = 0, the result is always zero
      {0, 1, 0, 0},
      {0, -1, 0, 0},
      {0, 2, 0, 0},
      {0, kMin, 0, 0},
      {0, kMax, 0, 0},
      // Try all four combinations of 1 and -1
      {1, 1, 1, 1},
      {-1, 1, -1, -1},
      {1, -1, -1, -1},
      {-1, -1, 1, 1},
      // Try all four combinations of +/-5 divided by +/- 2
      {5, 2, 2, 3},
      {-5, 2, -3, -2},
      {5, -2, -3, -2},
      {-5, -2, 2, 3},
      // Try with huge positive numerator
      {kMax, 1, kMax, kMax},
      {kMax, -1, -kMax, -kMax},
      {kMax, 2, kMax / 2, kMax / 2 + ((kMax % 2 != 0) ? 1 : 0)},
      {kMax, 3, kMax / 3, kMax / 3 + ((kMax % 3 != 0) ? 1 : 0)},
      // Try with huge negative numerator
      {kMin, 1, kMin, kMin},
      {kMin, 2, kMin / 2 - ((kMin % 2 != 0) ? 1 : 0), kMin / 2},
      {kMin, 3, kMin / 3 - ((kMin % 3 != 0) ? 1 : 0), kMin / 3},
      // Try with a huge positive denominator
      {1, kMax, 0, 1},
      {2, kMax, 0, 1},
      {3, kMax, 0, 1},
      // Try with a huge negative denominator
      {1, kMin, -1, 0},
      {2, kMin, -1, 0},
      {3, kMin, -1, 0},
      // Try with a huge numerator and a huge denominator
      {kMin, kMin, 1, 1},
      {kMin, kMax, -2, -1},
      {kMax, kMin, -1, 0},
      {kMax, kMax, 1, 1},
  };
  const int kNumTests = ABSL_ARRAYSIZE(kTestData);
  TestCeilOfRatio<SignedInteger>(kTestData, kNumTests);
}

// An implementation of CeilOfRatio that is correct for small enough values,
// and provided that the numerator and denominator are both positive
template <typename IntegralType>
IntegralType CeilOfRatioDenomMinusOne(IntegralType numerator,
                                      IntegralType denominator) {
  const IntegralType kOne(1);
  return (numerator + denominator - kOne) / denominator;
}

void TestThatCeilOfRatioDenomMinusOneIsIncorrect(int64_t numerator,
                                                 int64_t denominator,
                                                 int64_t expected_error) {
  const int64_t correct_result = CeilOfRatio(numerator, denominator);
  const int64_t result_by_denom_minus_one =
      CeilOfRatioDenomMinusOne(numerator, denominator);
  EXPECT_EQ(result_by_denom_minus_one + expected_error, correct_result)
      << "numerator = " << numerator << " denominator = " << denominator
      << " expected error = " << expected_error
      << " Actual difference: " << (correct_result - result_by_denom_minus_one);
}

TEST(MathUtil, CeilOfRatioUint8) { TestCeilOfRatioUnsigned<uint8_t>(); }

TEST(MathUtil, CeilOfRatioUint16) { TestCeilOfRatioUnsigned<uint16_t>(); }

TEST(MathUtil, CeilOfRatioUint32) { TestCeilOfRatioUnsigned<uint32_t>(); }

TEST(MathUtil, CeilOfRatioUint64) { TestCeilOfRatioUnsigned<uint64_t>(); }

TEST(MathUtil, CeilOfRatioInt8) { TestCeilOfRatioSigned<int8_t>(); }

TEST(MathUtil, CeilOfRatioInt16) { TestCeilOfRatioSigned<int16_t>(); }

TEST(MathUtil, CeilOfRatioInt32) { TestCeilOfRatioSigned<int32_t>(); }

TEST(MathUtil, CeilOfRatioInt64) { TestCeilOfRatioSigned<int64_t>(); }

TEST(MathUtil, CeilOfRatioDenomMinusOneIsIncorrect) {
  // Here we demonstrate why not to use CeilOfRatioDenomMinusOne: It does not
  // work with negative values.
  TestThatCeilOfRatioDenomMinusOneIsIncorrect(int64_t{-1}, int64_t{-2},
                                              int64_t{-1});

  // This would also fail if given kint64max because of signed integer overflow.
}

TEST(MathUtil, CeilOfLog2) {
  EXPECT_EQ(CeilOfLog2(0), 0);
  EXPECT_EQ(CeilOfLog2(1), 0);
  EXPECT_EQ(CeilOfLog2(2), 1);
  EXPECT_EQ(CeilOfLog2(3), 2);
  EXPECT_EQ(CeilOfLog2(4), 2);
  EXPECT_EQ(CeilOfLog2(5), 3);
  EXPECT_EQ(CeilOfLog2((1ULL << 63) - 1ULL), 63);
  EXPECT_EQ(CeilOfLog2(1ULL << 63), 63);
  EXPECT_EQ(CeilOfLog2((1ULL << 63) + 1ULL), 64);
  EXPECT_EQ(CeilOfLog2(std::numeric_limits<uint64_t>::max()), 64);
}

TEST(MathUtil, FloorOfLog2) {
  EXPECT_EQ(FloorOfLog2(0), 0);
  EXPECT_EQ(FloorOfLog2(1), 0);
  EXPECT_EQ(FloorOfLog2(2), 1);
  EXPECT_EQ(FloorOfLog2(3), 1);
  EXPECT_EQ(FloorOfLog2(4), 2);
  EXPECT_EQ(FloorOfLog2(5), 2);
  EXPECT_EQ(FloorOfLog2((1ULL << 63) - 1ULL), 62);
  EXPECT_EQ(FloorOfLog2(1ULL << 63), 63);
  EXPECT_EQ(FloorOfLog2((1ULL << 63) + 1ULL), 63);
  EXPECT_EQ(FloorOfLog2(std::numeric_limits<uint64_t>::max()), 63);
}

TEST(MathUtil, IsEven) {
  EXPECT_EQ(IsEven(0u), true);
  EXPECT_EQ(IsEven(1u), false);
  EXPECT_EQ(IsEven(2u), true);
  EXPECT_EQ(IsEven(3u), false);
  EXPECT_EQ(IsEven((1ULL << 63) - 1ULL), false);
  EXPECT_EQ(IsEven(1ULL << 63), true);
}

TEST(MathUtil, Exp2) {
  EXPECT_EQ(Exp2<uint32_t>(0), 1);
  EXPECT_EQ(Exp2<uint32_t>(1), 2);
  EXPECT_EQ(Exp2<uint32_t>(2), 4);
  EXPECT_EQ(Exp2<uint32_t>(3), 8);
  EXPECT_EQ(Exp2<uint32_t>(31), 2147483648);
  EXPECT_EQ(Exp2<uint64_t>(63), 1ULL << 63);
}

TEST(MathUtil, FactorizePowerOfTwo) {
  {
    auto [odd, exponent] = FactorizePowerOfTwo(0u);
    EXPECT_EQ(odd, 0);
    EXPECT_EQ(exponent, 0);
  }
  {
    auto [odd, exponent] = FactorizePowerOfTwo(1u);
    EXPECT_EQ(odd, 1);
    EXPECT_EQ(exponent, 0);
  }
  {
    auto [odd, exponent] = FactorizePowerOfTwo(2u);
    EXPECT_EQ(odd, 1);
    EXPECT_EQ(exponent, 1);
  }
  {
    auto [odd, exponent] = FactorizePowerOfTwo(3u);
    EXPECT_EQ(odd, 3);
    EXPECT_EQ(exponent, 0);
  }
  {
    auto [odd, exponent] = FactorizePowerOfTwo(4u);
    EXPECT_EQ(odd, 1);
    EXPECT_EQ(exponent, 2);
  }
  {
    auto [odd, exponent] = FactorizePowerOfTwo(6u);
    EXPECT_EQ(odd, 3);
    EXPECT_EQ(exponent, 1);
  }
  {
    auto [odd, exponent] = FactorizePowerOfTwo(40u);
    EXPECT_EQ(odd, 5);
    EXPECT_EQ(exponent, 3);
  }
  {
    auto [odd, exponent] =
        FactorizePowerOfTwo(uint64_t{7} * Exp2<uint64_t>(55));
    EXPECT_EQ(odd, 7);
    EXPECT_EQ(exponent, 55);
  }
}

TEST(MathUtil, SaturatingAdd) {
  {
    auto sa = SaturatingAdd(int8_t{3}, int8_t{4});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 7);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(int8_t{100}, int8_t{27});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 127);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(int8_t{101}, int8_t{27});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 127);
    EXPECT_TRUE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(int8_t{120}, int8_t{120});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 127);
    EXPECT_TRUE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(int8_t{127}, int8_t{127});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 127);
    EXPECT_TRUE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(int8_t{-127}, int8_t{127});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 0);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(int8_t{127}, int8_t{-127});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 0);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(int8_t{0}, int8_t{127});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 127);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(int8_t{127}, int8_t{0});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 127);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(uint8_t{3}, uint8_t{4});
    uint8_t u8 = sa.result;
    EXPECT_EQ(u8, 7);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(uint8_t{255}, uint8_t{4});
    uint8_t u8 = sa.result;
    EXPECT_EQ(u8, 255);
    EXPECT_TRUE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(uint8_t{3}, uint8_t{255});
    uint8_t u8 = sa.result;
    EXPECT_EQ(u8, 255);
    EXPECT_TRUE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(uint8_t{0}, uint8_t{255});
    uint8_t u8 = sa.result;
    EXPECT_EQ(u8, 255);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingAdd(uint8_t{255}, uint8_t{0});
    uint8_t u8 = sa.result;
    EXPECT_EQ(u8, 255);
    EXPECT_FALSE(sa.did_overflow);
  }
}

TEST(MathUtil, SaturatingSub) {
  {
    auto sa = SaturatingSub(int8_t{3}, int8_t{4});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, -1);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingSub(int8_t{0}, int8_t{-127});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 127);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingSub(uint8_t{100}, uint8_t{101});
    int8_t u8 = sa.result;
    EXPECT_EQ(u8, 0);
    EXPECT_TRUE(sa.did_overflow);
  }
  {
    auto sa = SaturatingSub(int8_t{100}, int8_t{-101});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 127);
    EXPECT_TRUE(sa.did_overflow);
  }
}

TEST(MathUtil, SaturatingMul) {
  {
    auto sa = SaturatingMul(int8_t{3}, int8_t{4});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 12);
    EXPECT_FALSE(sa.did_overflow);
  }
  {
    auto sa = SaturatingMul(int8_t{2}, int8_t{-88});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, -128);
    EXPECT_TRUE(sa.did_overflow);
  }
  {
    auto sa = SaturatingMul(uint8_t{100}, uint8_t{101});
    uint8_t u8 = sa.result;
    EXPECT_EQ(u8, 255);
    EXPECT_TRUE(sa.did_overflow);
  }
  {
    auto sa = SaturatingMul(int8_t{-100}, int8_t{-101});
    int8_t i8 = sa.result;
    EXPECT_EQ(i8, 127);
    EXPECT_TRUE(sa.did_overflow);
  }
}
TEST(MathUtil, ExhaustiveSaturatingAdd) {
  for (int16_t i16 = std::numeric_limits<int8_t>::min();
       i16 <= std::numeric_limits<int8_t>::max(); ++i16) {
    int8_t i = static_cast<int8_t>(i16);
    for (int16_t j16 = 0; j16 <= std::numeric_limits<int8_t>::max(); ++j16) {
      int8_t j = static_cast<int8_t>(j16);
      int16_t sum_16 = i16 + j16;
      bool expect_overflow = sum_16 > std::numeric_limits<int8_t>::max();
      auto sa = SaturatingAdd(i, j);
      if (expect_overflow) {
        EXPECT_EQ(std::numeric_limits<int8_t>::max(), sa.result);
        EXPECT_TRUE(sa.did_overflow);
      } else {
        EXPECT_EQ(sum_16, sa.result);
        EXPECT_FALSE(sa.did_overflow);
      }
    }
  }
}

void SaturatingAddFuzz(int16_t l, int16_t r) {
  auto res = SaturatingAdd(l, r);
  int32_t raw = static_cast<int32_t>(l) + static_cast<int32_t>(r);
  int32_t clamped =
      std::clamp<int32_t>(raw, std::numeric_limits<int16_t>::min(),
                          std::numeric_limits<int16_t>::max());
  EXPECT_EQ(static_cast<int32_t>(res.result), clamped);
  if (raw != clamped) {
    // Might have just hit the i16 max/min
    EXPECT_TRUE(res.did_overflow);
  }
}
void SaturatingSubFuzz(int16_t l, int16_t r) {
  auto res = SaturatingSub(l, r);
  int32_t raw = static_cast<int32_t>(l) - static_cast<int32_t>(r);
  int32_t clamped =
      std::clamp<int32_t>(raw, std::numeric_limits<int16_t>::min(),
                          std::numeric_limits<int16_t>::max());
  EXPECT_EQ(static_cast<int32_t>(res.result), clamped);
  if (raw != clamped) {
    // Might have just hit the i16 max/min
    EXPECT_TRUE(res.did_overflow);
  }
}
void SaturatingMulFuzz(int8_t l, int8_t r) {
  auto res = SaturatingMul(l, r);
  int32_t raw = static_cast<int32_t>(l) * static_cast<int32_t>(r);
  int32_t clamped = std::clamp<int32_t>(raw, std::numeric_limits<int8_t>::min(),
                                        std::numeric_limits<int8_t>::max());
  EXPECT_EQ(static_cast<int32_t>(res.result), clamped);
  if (raw != clamped) {
    // Might have just hit the i16 max/min
    EXPECT_TRUE(res.did_overflow);
  }
}

FUZZ_TEST(MathUtil, SaturatingAddFuzz);
FUZZ_TEST(MathUtil, SaturatingSubFuzz);
FUZZ_TEST(MathUtil, SaturatingMulFuzz);

}  // namespace
}  // namespace xls
