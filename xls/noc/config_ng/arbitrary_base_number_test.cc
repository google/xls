// Copyright 2021 The XLS Authors
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

#include "xls/noc/config_ng/arbitrary_base_number.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "xls/common/status/matchers.h"

namespace xls::noc {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(ArbitraryBaseNumberTest, AccessFields) {
  ArbitraryBaseNumber number(1, 2);
  EXPECT_EQ(number.GetDigitCount(), 1);
  EXPECT_EQ(number.GetNumericalBase(), 2);
}

TEST(ArbitraryBaseNumberTest, AddOne) {
  ArbitraryBaseNumber number(1, 2);
  EXPECT_FALSE(number.AddOne());
  EXPECT_TRUE(number.AddOne());
}

// Increment up to 10 using a four-digit-base-two integer number and retrieve
// value.
TEST(ArbitraryBaseNumberTest, GetValue) {
  ArbitraryBaseNumber number(4, 2);
  for (int64_t count = 0; count < 10; count++) {
    EXPECT_FALSE(number.AddOne());
  }
  EXPECT_EQ(number.GetValue().value(), 10);
}

// Increment up to 10 using a four-digit-base-two integer number and retrieve
// value from index one.
TEST(ArbitraryBaseNumberTest, GetValueIndex) {
  ArbitraryBaseNumber number(4, 2);
  for (int64_t count = 0; count < 10; count++) {
    EXPECT_FALSE(number.AddOne());
  }
  EXPECT_EQ(number.GetValue(1).value(), 5);
}

// Increment up to 10 and reset integer number.
TEST(ArbitraryBaseNumberTest, Reset) {
  ArbitraryBaseNumber number(4, 2);
  for (int64_t count = 0; count < 10; count++) {
    EXPECT_FALSE(number.AddOne());
  }
  EXPECT_THAT(number.GetValue(), 10);
  number.Reset();
  EXPECT_THAT(number.GetValue(), 0);
}

TEST(ArbitraryBaseNumberTest, SwapDigits) {
  ArbitraryBaseNumber number(4, 2);
  for (int64_t count = 0; count < 10; count++) {
    EXPECT_FALSE(number.AddOne());
  }
  // Invalid indices for swap.
  EXPECT_THAT(number.SwapDigits(-1, 2),
              StatusIs(absl::StatusCode::kOutOfRange,
                       HasSubstr("First digit index is out of range.")));
  EXPECT_THAT(number.SwapDigits(6, 2),
              StatusIs(absl::StatusCode::kOutOfRange,
                       HasSubstr("First digit index is out of range.")));
  EXPECT_THAT(number.SwapDigits(1, -1),
              StatusIs(absl::StatusCode::kOutOfRange,
                       HasSubstr("Second digit index is out of range.")));
  EXPECT_THAT(number.SwapDigits(1, 6),
              StatusIs(absl::StatusCode::kOutOfRange,
                       HasSubstr("Second digit index is out of range.")));
  // Swap index 1 and 2 of a four-digit-base-two integer number containing 1010
  // (the value 10 in base ten) yields 1100 (the value 12 in base ten).
  XLS_EXPECT_OK(number.SwapDigits(1, 2));
  EXPECT_EQ(number.GetValue().value(), 12);
}

TEST(ArbitraryBaseNumberTest, Base10) {
  ArbitraryBaseNumber number(2, 10);
  EXPECT_EQ(number.GetDigitCount(), 2);
  EXPECT_EQ(number.GetNumericalBase(), 10);
  for (int64_t count = 0; count < 10; count++) {
    EXPECT_FALSE(number.AddOne());
  }
  EXPECT_EQ(number.GetValue().value(), 10);
  EXPECT_EQ(number.GetValue(1).value(), 1);
}

TEST(ArbitraryBaseNumberTest, Base3) {
  ArbitraryBaseNumber number(4, 3);
  EXPECT_EQ(number.GetDigitCount(), 4);
  EXPECT_EQ(number.GetNumericalBase(), 3);
  for (int64_t count = 0; count < 10; count++) {
    EXPECT_FALSE(number.AddOne());
  }
  EXPECT_EQ(number.GetValue().value(), 10);
  EXPECT_EQ(number.GetValue(1).value(), 3);
  EXPECT_EQ(number.GetValue(2).value(), 1);
  EXPECT_EQ(number.GetValue(3).value(), 0);
}

}  // namespace
}  // namespace xls::noc
