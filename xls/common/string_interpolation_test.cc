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

#include "xls/common/string_interpolation.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using ::testing::HasSubstr;
using xls::status_testing::StatusIs;

// concatenate the format spec and the index so we can check both in the
// expected output
absl::StatusOr<std::string> PrintArg(absl::string_view format, int64_t index) {
  return std::to_string(index) + std::string(format);
}

TEST(StringInterpolation, ValidStrings) {
  struct TestCase {
    std::string input;
    std::string expected;
  };
  auto cases = {
      TestCase{"{{}}", "{}"},
      TestCase{"asdf{{", "asdf{"},
      TestCase{"{}", "0"},
      TestCase{"start {a} m {b} n {c} end", "start 0a m 1b n 2c end"},
      TestCase{"{{{x}}}", "{0x}"},
      TestCase{"{x{d}", "0x{d"},
  };
  for (const auto& test_case : cases) {
    XLS_ASSERT_OK_AND_ASSIGN(std::string result,
                             InterpolateArgs(test_case.input, PrintArg));
    EXPECT_EQ(result, test_case.expected);
  }
}

TEST(StringInterpolation, InvalidStrings) {
  EXPECT_THAT(InterpolateArgs("{x}d}", PrintArg),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unmatched '}' at index 4")));
  EXPECT_THAT(InterpolateArgs("{{}}}", PrintArg),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unmatched '}' at index 4")));
  EXPECT_THAT(InterpolateArgs("{{{}}", PrintArg),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unmatched '}' at index 4")));
  EXPECT_THAT(InterpolateArgs("{foo} bar {baz", PrintArg),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("unmatched '{' at index 10")));
  auto at_most_two_args = [](absl::string_view format,
                             int64_t index) -> absl::StatusOr<std::string> {
    if (index >= 2)
      return absl::InvalidArgumentError("not enough data operands");
    return "dontcare";
  };
  EXPECT_THAT(InterpolateArgs("{foo} {bar} {baz}", at_most_two_args),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("not enough data operands")));
}

}  // namespace
}  // namespace xls
