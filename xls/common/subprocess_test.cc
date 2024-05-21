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

#include <optional>
#include <string>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::_;
using testing::AllOf;
using testing::FieldsAre;
using testing::HasSubstr;

TEST(SubprocessTest, EmptyArgvFails) {
  auto result = InvokeSubprocess({}, std::nullopt);

  EXPECT_THAT(result, StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(SubprocessTest, NonZeroExitWorks) {
  auto result =
      InvokeSubprocess({"/usr/bin/env", "bash", "-c",
                        "echo -n hey && echo -n hello >&2 && exit 10"},
                       std::nullopt);

  EXPECT_THAT(result, IsOkAndHolds(FieldsAre(
                          /*stdout=*/"hey",
                          /*stderr=*/"hello",
                          /*exit_status=*/10,
                          /*normal_termination=*/true,
                          /*timeout_expired=*/false)));
}

TEST(SubprocessTest, CrashingExitWorks) {
  auto result =
      InvokeSubprocess({"/usr/bin/env", "bash", "-c",
                        "echo -n hey && echo -n hello >&2 && kill -ABRT $$"},
                       std::nullopt);

  EXPECT_THAT(result, IsOkAndHolds(FieldsAre(
                          /*stdout=*/_,
                          /*stderr=*/"hello",
                          /*exit_status=*/_,
                          /*normal_termination=*/false,
                          /*timeout_expired=*/false)));
}

TEST(SubprocessTest, WatchdogFastExitWorks) {
  absl::Time start_time = absl::Now();
  auto result = InvokeSubprocess({"/usr/bin/env", "bash", "-c", "exit 0"},
                                 std::nullopt, absl::Milliseconds(60000));
  absl::Duration duration = absl::Now() - start_time;

  EXPECT_THAT(result, IsOkAndHolds(FieldsAre(
                          /*stdout=*/"",
                          /*stderr=*/"",
                          /*exit_status=*/0,
                          /*normal_termination=*/true,
                          /*timeout_expired=*/false)));
  EXPECT_LT(absl::ToInt64Milliseconds(duration), 10000);
}

TEST(SubprocessTest, WatchdogWorks) {
  auto result = InvokeSubprocess({"/usr/bin/env", "bash", "-c", "sleep 10"},
                                 std::nullopt, absl::Milliseconds(50));

  EXPECT_THAT(result, IsOkAndHolds(FieldsAre(
                          /*stdout=*/"",
                          /*stderr=*/"",
                          /*exit_status=*/_,
                          /*normal_termination=*/false,
                          /*timeout_expired=*/true)));
}

TEST(SubprocessTest, ErrorAsStatusFailingCommand) {
  auto result = SubprocessErrorAsStatus(
      InvokeSubprocess({"/usr/bin/env", "bash", "-c",
                        "echo hey && echo hello >&2 && /bin/false"},
                       std::nullopt));

  ASSERT_THAT(result, StatusIs(absl::StatusCode::kInternal));
  EXPECT_THAT(result.status().ToString(), HasSubstr("hey"));
  EXPECT_THAT(result.status().ToString(), HasSubstr("hello"));
}

TEST(SubprocessTest, WorkingCommandWorks) {
  absl::StatusOr<SubprocessResult> result_or_status =
      SubprocessErrorAsStatus(InvokeSubprocess(
          {"/usr/bin/env", "bash", "-c", "echo hey && echo hello >&2"},
          std::nullopt));

  XLS_ASSERT_OK(result_or_status);
  EXPECT_EQ(result_or_status->stdout, "hey\n");
  EXPECT_EQ(result_or_status->stderr, "hello\n");
}

TEST(SubprocessTest, LargeOutputToStdoutFirstWorks) {
  absl::StatusOr<SubprocessResult> result_or_status = SubprocessErrorAsStatus(
      InvokeSubprocess({"/usr/bin/env", "bash", "-c",
                        "/usr/bin/env seq 10000 && echo hello >&2"},
                       std::nullopt));

  XLS_ASSERT_OK(result_or_status);
  EXPECT_THAT(result_or_status->stdout, HasSubstr("\n10000\n"));
  EXPECT_EQ(result_or_status->stderr, "hello\n");
}

TEST(SubprocessTest, LargeOutputToStderrFirstWorks) {
  absl::StatusOr<SubprocessResult> result_or_status = SubprocessErrorAsStatus(
      InvokeSubprocess({"/usr/bin/env", "bash", "-c",
                        "/usr/bin/env seq 10000 >&2 && echo hello"},
                       std::nullopt));

  XLS_ASSERT_OK(result_or_status);
  EXPECT_EQ(result_or_status->stdout, "hello\n");
  EXPECT_THAT(result_or_status->stderr, HasSubstr("\n10000\n"));
}

TEST(SubprocessTest, ErrorAsStatusWorks) {
  // Translates abnormal termination.
  SubprocessResult bad_exit{.stderr = "word_a", .normal_termination = false};
  absl::StatusOr<SubprocessResult> exit_payload = bad_exit;
  absl::StatusOr<SubprocessResult> exit_result =
      SubprocessErrorAsStatus(exit_payload);
  EXPECT_THAT(exit_result,
              StatusIs(absl::StatusCode::kInternal, HasSubstr("word_a")));

  // Translates non-zero exit code.
  SubprocessResult bad_code{.stdout = "word_c",
                            .stderr = "word_d",
                            .exit_status = 4,
                            .normal_termination = true};
  absl::StatusOr<SubprocessResult> code_payload = bad_code;
  absl::StatusOr<SubprocessResult> code_result =
      SubprocessErrorAsStatus(code_payload);
  EXPECT_THAT(code_result,
              StatusIs(absl::StatusCode::kInternal,
                       AllOf(HasSubstr("word_c"), HasSubstr("word_d"))));

  // Passes status through intact.
  absl::StatusOr<SubprocessResult> status = absl::AbortedError("word_e");
  absl::StatusOr<SubprocessResult> same_status =
      SubprocessErrorAsStatus(status);
  EXPECT_THAT(same_status,
              StatusIs(absl::StatusCode::kAborted, HasSubstr("word_e")));
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
