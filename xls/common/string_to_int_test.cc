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

#include "xls/common/string_to_int.h"

#include <cstdint>
#include <limits>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_replace.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

TEST(StringToInt, TabularTest) {
  struct TestCase {
    std::string input;
    int base;
    int64_t expected;
  };
  auto cases = {
      // The simple string "0" should work in all bases.
      TestCase{"0", /*base=*/0, 0},
      TestCase{"0", /*base=*/2, 0},
      TestCase{"0", /*base=*/8, 0},
      TestCase{"0", /*base=*/10, 0},
      TestCase{"0", /*base=*/16, 0},

      TestCase{"0x0", /*base=*/0, 0},
      TestCase{"0x01", /*base=*/0, 0x01},
      TestCase{"-0x01", /*base=*/0, -0x01},
      TestCase{"0x10", /*base=*/0, 0x10},
      TestCase{"10", /*base=*/16, 0x10},
      TestCase{"-00", /*base=*/16, 0},
      TestCase{"-10", /*base=*/16, -0x10},
      TestCase{"11", /*base=*/16, 0x11},
      TestCase{"abcdef", /*base=*/16, 0xabcdef},
      TestCase{"AaBbCcDdEeFf", /*base=*/16, int64_t{0xaabbccddeeff}},
      TestCase{"ffffffffffffffff", /*base=*/16, int64_t{-1}},
      TestCase{"-ffffffffffffffff", /*base=*/16, int64_t{1}},
      TestCase{"8000000000000000", /*base=*/16,
               std::numeric_limits<int64_t>::min()},
      TestCase{"7fffffffffffffff", /*base=*/16,
               std::numeric_limits<int64_t>::max()},
      TestCase{"-0", /*base=*/16, 0},

      TestCase{"0b0", /*base=*/0, 0},
      TestCase{"10", /*base=*/2, 0b10},
      TestCase{"01", /*base=*/2, 0b01},
      TestCase{"11", /*base=*/2, 0b11},
      TestCase{"-11", /*base=*/2, -0b11},

      TestCase{"-0777", /*base=*/0, -0777},
      TestCase{"777", /*base=*/8, 0777},
      TestCase{"76543210", /*base=*/8, 076543210},
      TestCase{"-0", /*base=*/8, 0},
  };
  for (const auto& test_case : cases) {
    XLS_ASSERT_OK_AND_ASSIGN(int64_t result,
                             StrTo64Base(test_case.input, test_case.base));
    EXPECT_EQ(result, test_case.expected) << absl::StreamFormat(
        "in: %s base: %d want: %#x got: %#x", test_case.input, test_case.base,
        test_case.expected, result);
  }
}

TEST(StringToInt, InvalidChars) {
  EXPECT_THAT(
      StrTo64Base("0xf00t", 0),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Number character 't' is invalid for numeric base 16")));
  EXPECT_THAT(
      StrTo64Base("0xF00T", 0),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Number character 'T' is invalid for numeric base 16")));
  EXPECT_THAT(
      StrTo64Base("0xb1a5g", 0),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Number character 'g' is invalid for numeric base 16")));
  EXPECT_THAT(
      StrTo64Base("0xB1A5G", 0),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Number character 'G' is invalid for numeric base 16")));
  EXPECT_THAT(
      StrTo64Base("0b01012", 0),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Number character '2' is invalid for numeric base 2")));
  EXPECT_THAT(
      StrTo64Base("0778", 0),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Number character '8' is invalid for numeric base 8")));
}

TEST(StringToInt, Overflow64Bits) {
  std::string input = "0x1_ffff_ffff_ffff_ffff";
  input = absl::StrReplaceAll(input, {{"_", ""}});
  EXPECT_THAT(StrTo64Base(input, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Number overflows 64-bit integer")));

  // Even though this has a leading 0 char it should be ok, the data does not
  // overflow int64_t.
  input = "0x0_ffff_ffff_ffff_ffff";
  input = absl::StrReplaceAll(input, {{"_", ""}});
  EXPECT_THAT(StrTo64Base(input, 0), IsOkAndHolds(int64_t{-1}));
}

TEST(StringToInt, JustAMinus) {
  const std::string minus = "-";
  EXPECT_THAT(StrTo64Base(minus, 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Number contained no digits")));
  EXPECT_THAT(StrTo64Base(minus, 10),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Number contained no digits")));
  EXPECT_THAT(StrTo64Base("-0x", 16),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Number contained no digits")));
  EXPECT_THAT(StrTo64Base("-0b", 0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Number contained no digits")));
}

}  // namespace
}  // namespace xls
