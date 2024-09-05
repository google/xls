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

#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/stdlib/tests/umul_with_overflow_21_21_18_jit_wrapper.h"
#include "xls/dslx/stdlib/tests/umul_with_overflow_35_32_18_jit_wrapper.h"
#include "xls/ir/value_view.h"

namespace xls {
namespace {

TEST(UmulOverflowTest, Smoke21Test) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::UmulWithOverflow212118> f,
                           xls::test::UmulWithOverflow212118::Create());

  int64_t x = 8;
  int64_t y = 9;
  uint8_t result[8] = {0};

  xls::BitsView<21> x_view(reinterpret_cast<uint8_t*>(&x));
  xls::BitsView<18> y_view(reinterpret_cast<uint8_t*>(&y));

  xls::MutableTupleView<xls::MutableBitsView<1>, xls::MutableBitsView<21>>
      result_view(result);
  XLS_ASSERT_OK(f->Run(x_view, y_view, result_view));

  bool overflow = result_view.Get<0>().GetValue();
  int64_t product = result_view.Get<1>().GetValue();
  EXPECT_EQ(overflow, false);
  EXPECT_EQ(product, 72);
}

TEST(UmulOverflowTest, Smoke35Test) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::UmulWithOverflow353218> f,
                           xls::test::UmulWithOverflow353218::Create());

  int64_t x = 8;
  int64_t y = 9;
  uint8_t result[16] = {0};

  xls::BitsView<32> x_view(reinterpret_cast<uint8_t*>(&x));
  xls::BitsView<18> y_view(reinterpret_cast<uint8_t*>(&y));

  xls::MutableTupleView<xls::MutableBitsView<1>, xls::MutableBitsView<35>>
      result_view(result);
  XLS_ASSERT_OK(f->Run(x_view, y_view, result_view));

  bool overflow = result_view.Get<0>().GetValue();
  int64_t product = result_view.Get<1>().GetValue();
  EXPECT_EQ(overflow, false);
  EXPECT_EQ(product, 72);
}

constexpr int64_t kMaxUint21 = (int64_t{1} << 21) - 1;
constexpr int64_t kMaxUint18 = (int64_t{1} << 18) - 1;

void UmulOverflow212118WorksAsExpected(int64_t x, int64_t y) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::UmulWithOverflow212118> f,
                           xls::test::UmulWithOverflow212118::Create());

  xls::BitsView<21> x_view(reinterpret_cast<uint8_t*>(&x));
  xls::BitsView<18> y_view(reinterpret_cast<uint8_t*>(&y));

  uint8_t result[8] = {0};
  xls::MutableTupleView<xls::MutableBitsView<1>, xls::MutableBitsView<21>>
      result_view(result);
  XLS_ASSERT_OK(f->Run(x_view, y_view, result_view));

  bool overflow = result_view.Get<0>().GetValue();
  int64_t product = result_view.Get<1>().GetValue();

  int64_t actual_product = x * y;
  int64_t expected_product = actual_product & kMaxUint21;
  bool expected_overflow = actual_product != expected_product;

  EXPECT_EQ(overflow, expected_overflow);
  EXPECT_EQ(product, expected_product);
}
FUZZ_TEST(UmulOverflowFuzzTest, UmulOverflow212118WorksAsExpected)
    .WithDomains(fuzztest::InRange<int64_t>(0, kMaxUint21),
                 fuzztest::InRange<int64_t>(0, kMaxUint18));

constexpr int64_t kMaxUint35 = (int64_t{1} << 35) - 1;
constexpr int64_t kMaxUint32 = (int64_t{1} << 32) - 1;

void UmulOverflow353218WorksAsExpected(int64_t x, int64_t y) {
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xls::test::UmulWithOverflow353218> f,
                           xls::test::UmulWithOverflow353218::Create());

  xls::BitsView<32> x_view(reinterpret_cast<uint8_t*>(&x));
  xls::BitsView<18> y_view(reinterpret_cast<uint8_t*>(&y));

  uint8_t result[16] = {0};
  xls::MutableTupleView<xls::MutableBitsView<1>, xls::MutableBitsView<35>>
      result_view(result);
  XLS_ASSERT_OK(f->Run(x_view, y_view, result_view));

  bool overflow = result_view.Get<0>().GetValue();
  int64_t product = result_view.Get<1>().GetValue();

  int64_t actual_product = x * y;
  int64_t expected_product = actual_product & kMaxUint35;
  bool expected_overflow = actual_product != expected_product;

  EXPECT_EQ(overflow, expected_overflow);
  EXPECT_EQ(product, expected_product);
}
FUZZ_TEST(UmulOverflowFuzzTest, UmulOverflow353218WorksAsExpected)
    .WithDomains(fuzztest::InRange<int64_t>(0, kMaxUint32),
                 fuzztest::InRange<int64_t>(0, kMaxUint18));

}  // namespace
}  // namespace xls
