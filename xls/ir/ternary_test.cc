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

#include "xls/ir/ternary.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls {
namespace {

TEST(Ternary, FromKnownBits) {
  // Basic test of functionality
  EXPECT_EQ(*StringToTernaryVector("0b1101X1X001"),
            ternary_ops::FromKnownBits(UBits(0b1111010111, 10),
                                       UBits(0b1101010001, 10)));
  // Empty bitstring should be handled correctly
  EXPECT_EQ(TernaryVector(), ternary_ops::FromKnownBits(Bits(), Bits()));
}

TEST(Ternary, Difference) {
  // Basic test of functionality.
  EXPECT_EQ(*StringToTernaryVector("0b11XXXXXXX1"),
            ternary_ops::Difference(*StringToTernaryVector("0b1101X1X001"),
                                    *StringToTernaryVector("0bXX01X1X00X")));
  // Test that conflict (in the last bit) leads to `absl::nullopt`.
  EXPECT_EQ(absl::nullopt,
            ternary_ops::Difference(*StringToTernaryVector("0b1101X1X001"),
                                    *StringToTernaryVector("0bXX01X1X000")));
  // It's okay if there are unknown bits in lhs that are known in rhs.
  // The point is just to determine what information was gained in lhs that
  // is not already in rhs.
  EXPECT_EQ(*StringToTernaryVector("0b11XXXXXXX1"),
            ternary_ops::Difference(*StringToTernaryVector("0b110XXXX001"),
                                    *StringToTernaryVector("0bXX01X1X00X")));
}

TEST(Ternary, NumberOfKnownBits) {
  // Basic test of functionality
  EXPECT_EQ(
      ternary_ops::NumberOfKnownBits(*StringToTernaryVector("0b1101X1X001")),
      8);
  // Empty ternary vector should be handled correctly
  EXPECT_EQ(ternary_ops::NumberOfKnownBits(TernaryVector()), 0);
}

}  // namespace
}  // namespace xls
