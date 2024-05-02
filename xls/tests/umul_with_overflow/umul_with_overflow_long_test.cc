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

#include <cstdint>
#include <memory>
#include <string>
#include <tuple>

#include "gtest/gtest.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/value_view.h"
#include "xls/tests/umul_with_overflow/umul_with_overflow_21_21_18_jit_wrapper.h"
#include "xls/tests/umul_with_overflow/umul_with_overflow_35_32_18_jit_wrapper.h"

namespace xls {
namespace {

class UmulSweepOverflowTest
    : public testing::TestWithParam<std::tuple<int64_t, int64_t>> {
 public:
  int64_t sweep_from_inclusive() { return std::get<0>(GetParam()); }

  int64_t step() { return std::get<1>(GetParam()); }

  int64_t sweep_to_exclusive() { return sweep_from_inclusive() + step(); }

  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ParamType>& info) {
    int64_t from = std::get<0>(info.param);
    int64_t step = std::get<1>(info.param);

    return absl::StrFormat("from_0x%x_to_0x%x", from, from + step);
  }
};

TEST_P(UmulSweepOverflowTest, Umul212118Sweep) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::UmulWithOverflow212118> f,
                           xls::test::UmulWithOverflow212118::Create());

  int64_t y_from_inclusive = sweep_from_inclusive();
  int64_t y_to_exclusive = sweep_to_exclusive();

  int64_t x = 0;
  int64_t y = 0;
  uint8_t result[8] = {0};

  xls::BitsView<21> x_view(reinterpret_cast<uint8_t*>(&x));
  xls::BitsView<18> y_view(reinterpret_cast<uint8_t*>(&y));
  xls::MutableTupleView<xls::MutableBitsView<1>, xls::MutableBitsView<21>>
      result_view(result);

  for (x = 0; x <= 0x1fffff; ++x) {
    for (y = y_from_inclusive; y < y_to_exclusive; ++y) {
      XLS_ASSERT_OK(f->Run(x_view, y_view, result_view));

      bool overflow = result_view.Get<0>().GetValue();
      int64_t product = result_view.Get<1>().GetValue();

      int64_t actual_product = x * y;
      int64_t expected_product = actual_product & 0x1fffff;
      bool expected_overflow = actual_product != expected_product;

      EXPECT_EQ(overflow, expected_overflow) << absl::StrFormat(
          "0x%x * 0x%x = 0x%x, truncated (%x, 0x%x), got (%x 0x%x)", x, y,
          actual_product, expected_overflow, expected_product, overflow,
          product);
      EXPECT_EQ(product, expected_product) << absl::StrFormat(
          "0x%x * 0x%x = 0x%x, truncated (%x, 0x%x), got (%x 0x%x)", x, y,
          actual_product, expected_overflow, expected_product, overflow,
          product);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Umul212118ExhaustiveTest, UmulSweepOverflowTest,
    testing::Combine(testing::Range(0l, 0x40000l,
                                    0x100l),  // From [0x0, 0x3ffff]
                     testing::Values(0x100l)),
    UmulSweepOverflowTest::PrintToStringParamName);

// Expand a 16 bit integer to 32 bits, inserting a zero between each bit.
// If pos = 0, x16[0] == x32[0].
//   ex. 0b1111 will return  0b0101_0101
// If pos = 1, x16[0] == x32[1].
//   ex. 0b1111 will return  0b101_01010
int64_t Generate32From16(int64_t x16, bool pos) {
  int64_t x = 0;
  int64_t i = 0;

  while (x16 > 0) {
    int64_t mask = (x16 & 0x1) << (2 * i + static_cast<int64_t>(pos));
    x |= mask;

    x16 >>= 1;
    ++i;
  }

  return x;
}

TEST_F(UmulSweepOverflowTest, GenerateTest) {
  EXPECT_EQ(Generate32From16(0xffff, 0), 0x5555'5555);
  EXPECT_EQ(Generate32From16(0xffff, 1), 0xaaaa'aaaa);
  EXPECT_EQ(Generate32From16(0x965a, 0), 0x4114'1144);
  EXPECT_EQ(Generate32From16(0x965a, 1), 0x8228'2288);
}

TEST_P(UmulSweepOverflowTest, Umul353218SweepPartial) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::UmulWithOverflow353218> f,
                           xls::test::UmulWithOverflow353218::Create());

  int64_t y_from_inclusive = sweep_from_inclusive();
  int64_t y_to_exclusive = sweep_to_exclusive();

  int64_t x = 0;
  int64_t y = 0;
  uint8_t result[16] = {0};

  xls::BitsView<32> x_view(reinterpret_cast<uint8_t*>(&x));
  xls::BitsView<18> y_view(reinterpret_cast<uint8_t*>(&y));
  xls::MutableTupleView<xls::MutableBitsView<1>, xls::MutableBitsView<35>>
      result_view(result);

  // x is 32 bits so sweep 2 * half of the bit space (17 bits)
  for (int64_t i = 0; i <= 0x1ffff; ++i) {
    x = Generate32From16(i & 0xffff, (i >> 16) != 0);

    for (y = y_from_inclusive; y < y_to_exclusive; ++y) {
      XLS_ASSERT_OK(f->Run(x_view, y_view, result_view));

      bool overflow = result_view.Get<0>().GetValue();
      int64_t product = result_view.Get<1>().GetValue();

      int64_t actual_product = x * y;
      int64_t expected_product = actual_product & 0x7'ffff'ffff;
      bool expected_overflow = actual_product != expected_product;

      EXPECT_EQ(overflow, expected_overflow) << absl::StrFormat(
          "0x%x * 0x%x = 0x%x, truncated (%x, 0x%x), got (%x 0x%x)", x, y,
          actual_product, expected_overflow, expected_product, overflow,
          product);
      EXPECT_EQ(product, expected_product) << absl::StrFormat(
          "0x%x * 0x%x = 0x%x, truncated (%x, 0x%x), got (%x 0x%x)", x, y,
          actual_product, expected_overflow, expected_product, overflow,
          product);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Umul353218SweepPartial, UmulSweepOverflowTest,
    testing::Combine(testing::Range(0l, 0x40000l,
                                    0x100l),  // From [0x0, 0x3ffff]
                     testing::Values(0x100l)),
    UmulSweepOverflowTest::PrintToStringParamName);

}  // namespace
}  // namespace xls
