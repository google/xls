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

#include "xls/dslx/frontend/bindings.h"

#include <optional>
#include <string>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/dslx/frontend/ast.h"
#include "xls/dslx/frontend/pos.h"

namespace xls::dslx {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::HasSubstr;

TEST(BindingsTest, ParseErrorGetData) {
  Pos start_pos("fake_file.x", 42, 64);
  Pos limit_pos("fake_file.x", 43, 65);
  Span span(start_pos, limit_pos);
  absl::Status status = ParseErrorStatus(span, "my message");
  XLS_ASSERT_OK_AND_ASSIGN(PositionalErrorData got,
                           GetPositionalErrorData(status));
  EXPECT_EQ(got.span, span);
  EXPECT_EQ(got.message, "my message");
  EXPECT_EQ(got.error_type, "ParseError");
}

TEST(BindingsTest, ParseErrorFakeSpanGetData) {
  Span span = FakeSpan();
  absl::Status status = ParseErrorStatus(span, "my message");
  XLS_ASSERT_OK_AND_ASSIGN(PositionalErrorData got,
                           GetPositionalErrorData(status));
  EXPECT_EQ(got.span, span);
  EXPECT_EQ(got.message, "my message");
  EXPECT_EQ(got.error_type, "ParseError");
}

TEST(BindingsTest, NotPositionalError) {
  auto status = absl::InvalidArgumentError("This has no position data");
  EXPECT_THAT(
      GetPositionalErrorData(status),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Provided status is not in recognized error form")));
}

TEST(BindingsTest, PositionalErrorNotTargetType) {
  auto status = absl::InvalidArgumentError(
      "ParseError: fake_file.x:1:1-2:2 message goes here");
  Span span = {Pos{"fake_file.x", 0, 0}, Pos{"fake_file.x", 1, 1}};
  EXPECT_THAT(GetPositionalErrorData(status),
              IsOkAndHolds(PositionalErrorData{span, "message goes here",
                                               "ParseError"}));
  EXPECT_THAT(
      GetPositionalErrorData(status, /*target_type=*/"ShoobaDoobaError"),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Provided status is not in recognized error form")));
}

TEST(BindingsTest, ResolveNameOrNulloptMissingCase) {
  Bindings bindings;
  std::optional<AnyNameDef> result =
      bindings.ResolveNameOrNullopt("not_present");
  EXPECT_FALSE(result.has_value());
}

TEST(ParseNameErrorTest, ExtractName) {
  const std::string_view kName = "shoobadooba";
  const Span kFakeSpan = FakeSpan();
  const absl::Status name_error = ParseNameErrorStatus(kFakeSpan, kName);
  std::optional<std::string_view> name = MaybeExtractParseNameError(name_error);
  ASSERT_TRUE(name.has_value());
  EXPECT_EQ(name.value(), kName);
}

TEST(ParseNameErrorTest, ExtractFromNonNameError) {
  const absl::Status error = absl::InvalidArgumentError(
      "Cannot find a definition for stuff: \"andthings\"");
  std::optional<std::string_view> name = MaybeExtractParseNameError(error);
  ASSERT_FALSE(name.has_value());
}

}  // namespace
}  // namespace xls::dslx
