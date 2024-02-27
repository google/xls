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

#include "xls/ir/format_preference.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

TEST(FormatPreferenceTest, ToString) {
  struct TestCase {
    FormatPreference value;
    std::string stringified;
    bool valid;
  };
  for (const auto& [value, expected, valid] : std::vector<TestCase>{
           {FormatPreference::kDefault, "default", true},
           {FormatPreference::kBinary, "binary", true},
           {FormatPreference::kUnsignedDecimal, "unsigned-decimal", true},
           {FormatPreference::kSignedDecimal, "signed-decimal", true},
           {FormatPreference::kHex, "hex", true},
           {static_cast<FormatPreference>(42), "<invalid format preference>",
            false},
       }) {
    EXPECT_EQ(FormatPreferenceToString(value), expected);
    if (valid) {
      EXPECT_THAT(FormatPreferenceFromString(expected), IsOkAndHolds(value));
    } else {
      EXPECT_THAT(FormatPreferenceFromString(expected),
                  StatusIs(absl::StatusCode::kInvalidArgument,
                           HasSubstr("Invalid format preference: \"")));
    }
  }
}

}  // namespace
}  // namespace xls
