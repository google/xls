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

#include "xls/common/subprocess.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::HasSubstr;

TEST(SubprocessTest, EmptyArgvFails) {
  auto result = InvokeSubprocess({}, absl::nullopt);

  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SubprocessTest, FailingCommandFails) {
  auto result = SubprocessErrorAsStatus(InvokeSubprocess(
      {"/bin/sh", "-c", "echo hey && echo hello >&2 && /bin/false"},
      absl::nullopt));

  ASSERT_THAT(result, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(result.status().ToString(), HasSubstr("hey"));
  EXPECT_THAT(result.status().ToString(), HasSubstr("hello"));
}

TEST(SubprocessTest, WorkingCommandWorks) {
  absl::StatusOr<SubprocessResult> result_or_status =
      SubprocessErrorAsStatus(InvokeSubprocess(
          {"/bin/sh", "-c", "echo hey && echo hello >&2"}, absl::nullopt));

  XLS_ASSERT_OK(result_or_status);
  EXPECT_EQ(result_or_status->stdout, "hey\n");
  EXPECT_EQ(result_or_status->stderr, "hello\n");
}

TEST(SubprocessTest, LargeOutputToStdoutFirstWorks) {
  absl::StatusOr<SubprocessResult> result_or_status =
      SubprocessErrorAsStatus(InvokeSubprocess(
          {"/bin/sh", "-c", "/usr/bin/env seq 10000 && echo hello >&2"},
          absl::nullopt));

  XLS_ASSERT_OK(result_or_status);
  EXPECT_THAT(result_or_status->stdout, HasSubstr("\n10000\n"));
  EXPECT_EQ(result_or_status->stderr, "hello\n");
}

TEST(SubprocessTest, LargeOutputToStderrFirstWorks) {
  absl::StatusOr<SubprocessResult> result_or_status =
      SubprocessErrorAsStatus(InvokeSubprocess(
          {"/bin/sh", "-c", "/usr/bin/env seq 10000 >&2 && echo hello"},
          absl::nullopt));

  XLS_ASSERT_OK(result_or_status);
  EXPECT_EQ(result_or_status->stdout, "hello\n");
  EXPECT_THAT(result_or_status->stderr, HasSubstr("\n10000\n"));
}

TEST(SubprocessTest, ResultsUnpackToStringPair) {
  SubprocessResult payload{.stdout = "hello", .stderr = "there"};
  absl::StatusOr<SubprocessResult> result = payload;
  absl::StatusOr<std::pair<std::string, std::string>> transformed =
      SubprocessResultToStrings(result);
  EXPECT_THAT(transformed, IsOkAndHolds(testing::Pair("hello", "there")));

  absl::StatusOr<SubprocessResult> status = absl::InternalError("bad arg");
  absl::StatusOr<std::pair<std::string, std::string>> same_status =
      SubprocessResultToStrings(status);
  EXPECT_THAT(same_status,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("bad arg")));
}

}  // namespace
}  // namespace xls
