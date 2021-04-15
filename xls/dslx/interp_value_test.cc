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

#include "xls/dslx/interp_value.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace xls::dslx {
namespace {

TEST(InterpValueTest, FormatU8) {
  auto ff = InterpValue::MakeUBits(/*bit_count=*/8, /*value=*/0xff);
  EXPECT_EQ(ff.ToString(), "u8:255");
  EXPECT_EQ(ff.ToString(true, FormatPreference::kHex), "u8:0xff");
  EXPECT_EQ(ff.ToString(true, FormatPreference::kDecimal), "u8:255");
  EXPECT_EQ(ff.ToString(true, FormatPreference::kBinary), "u8:0b1111_1111");
}

TEST(InterpValueTest, FormatS8) {
  auto ff = InterpValue::MakeSBits(/*bit_count=*/8, /*value=*/0xff);
  EXPECT_EQ(ff.ToString(), "s8:-1");
  EXPECT_EQ(ff.ToString(true, FormatPreference::kHex), "s8:0xff");
  EXPECT_EQ(ff.ToString(true, FormatPreference::kDecimal), "s8:-1");
  EXPECT_EQ(ff.ToString(true, FormatPreference::kBinary), "s8:0b1111_1111");
}

}  // namespace
}  // namespace xls::dslx
