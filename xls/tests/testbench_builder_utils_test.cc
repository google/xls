// Copyright 2024 The XLS Authors
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

#include "xls/tests/testbench_builder_utils.h"

#include <bit>
#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

using ::testing::StartsWith;

namespace xls {
namespace internal {
TEST(TestbencUtils, DefaultPrintValuePrintInt) {
  constexpr int32_t a = 42;
  EXPECT_EQ(DefaultPrintValue(a), "42");

  constexpr int64_t b = -12345678901234;
  EXPECT_EQ(DefaultPrintValue(b), "-12345678901234");
}

TEST(TestbencUtils, DefaultPrintValuePrintFloat) {
  constexpr float a = 1.0f;
  EXPECT_EQ(DefaultPrintValue(a), "1.000000000e+00 (0x1p+0)");

  constexpr float b = 2.0f;
  EXPECT_EQ(DefaultPrintValue(b), "2.000000000e+00 (0x1p+1)");

  constexpr double c = 4.0f;
  EXPECT_EQ(DefaultPrintValue(c), "4.0000000000000000e+00 (0x1p+2)");

  constexpr float subnormal = std::bit_cast<float>(0x1cafe);
  EXPECT_EQ(DefaultPrintValue(subnormal), "1.646553722e-40 (0x1.cafep-133)");

  // Unfortunately, %a does not print the bit-pattern for NaN. Maybe refine
  // if ever relevant.
  constexpr float some_nan = std::bit_cast<float>(0xff << 23 | 0xabab);
  EXPECT_EQ(DefaultPrintValue(some_nan), "nan (nan)");
}

TEST(TestbencUtils, DefaultPrintValuePrintUnprintable) {
  struct Foo {
    int value;
  };
  Foo a{42};
  EXPECT_THAT(DefaultPrintValue(a), StartsWith("<unprintable value"));
}

}  // namespace internal
}  // namespace xls
