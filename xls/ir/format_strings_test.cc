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

#include "xls/ir/format_strings.h"

#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

// TODO(https://github.com/google/xls/issues/476): Add property-based tests
// using randomly generated strings and formats. Key properties:
// 1. Valid format strings can be parsed and reconstructed accurately.
// 2. Invalid strings only produce errors, no OOB reads or other unsafe
// behavior.

TEST(FormatStringsTest, ParseFormats) {
  std::vector<FormatStep> simple_format = {"x is ", FormatPreference::kDefault,
                                           "."};
  EXPECT_THAT(ParseFormatString("x is {}."), IsOkAndHolds(simple_format));
  EXPECT_EQ(OperandsExpectedByFormat(simple_format), 1);

  std::string complex_format_string =
      R"(x in different formats
{{}} = {}
{{:d}} = {:d}
{{:u}} = {:u}
{{:#x}} = {:#x}
{{:#b}} = {:#b})";

  std::vector<FormatStep> complex_format = {"x in different formats\n{{}} = ",
                                            FormatPreference::kDefault,
                                            "\n{{:d}} = ",
                                            FormatPreference::kSignedDecimal,
                                            "\n{{:u}} = ",
                                            FormatPreference::kUnsignedDecimal,
                                            "\n{{:#x}} = ",
                                            FormatPreference::kHex,
                                            "\n{{:#b}} = ",
                                            FormatPreference::kBinary};

  EXPECT_THAT(ParseFormatString(complex_format_string),
              IsOkAndHolds(complex_format));

  EXPECT_EQ(OperandsExpectedByFormat(complex_format), 5);

  std::string plain_formats_string = "{:x}{:b}";
  std::vector<FormatStep> plain_formats = {FormatPreference::kPlainHex,
                                           FormatPreference::kPlainBinary};

  EXPECT_THAT(ParseFormatString(plain_formats_string),
              IsOkAndHolds(plain_formats));
  EXPECT_EQ(OperandsExpectedByFormat(plain_formats), 2);

  std::string struct_formats_string = "{{st0{{x: {}}}}}";
  std::vector<FormatStep> struct_formats = {
      "{{st0{{x: ", FormatPreference::kDefault, "}}}}"};

  EXPECT_THAT(ParseFormatString(struct_formats_string),
              IsOkAndHolds(struct_formats));
  EXPECT_EQ(OperandsExpectedByFormat(struct_formats), 1);
}

TEST(FormatStringsTest, ErrorTests) {
  EXPECT_THAT(ParseFormatString("{abc}"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("format specifier")));
  EXPECT_THAT(ParseFormatString("asdas { sdsfs"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("matching")));
  EXPECT_THAT(ParseFormatString("ddd } sda"),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("preceding")));
}

}  // namespace
}  // namespace xls
