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

#include "xls/common/logging/logging.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/log_severity.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/capture_stream.h"
#include "xls/common/logging/logging_test_base.h"
#include "xls/common/status/matchers.h"

namespace my_unit_test {

// Define an xls namespace that is not the main xls. This ensures that the
// macros don't accidentally refer to the xls namespace without using the fully
// qualified name ::xls.
namespace xls {
int something_to_put_in_the_xls_namespace = 0;
}  // namespace xls

namespace {

using ::testing::HasSubstr;
using ::testing::Not;
using ::testing::StartsWith;
using ::xls::status_testing::IsOkAndHolds;

class LoggingTest : public ::xls::logging_internal::testing::LoggingTestBase {
 public:
  LoggingTest() {
    absl::SetFlag(&FLAGS_v, 10);  // Enable verbose logging
  }
};

TEST_F(LoggingTest, LogMessagesArePrintedToStderr) {
  absl::StatusOr<std::string> output = ::xls::testing::CaptureStream(
      STDERR_FILENO, [] { XLS_LOG(ERROR) << "test_info_log_message"; });

  EXPECT_THAT(output, IsOkAndHolds(HasSubstr("test_info_log_message")));
  EXPECT_THAT(output, IsOkAndHolds(StartsWith("E")));  // For ERROR.
}

TEST_F(LoggingTest, LogMacroLogsWithInfoSeverity) {
  XLS_LOG(INFO) << "test_info_log_message";

  auto entry = GetSingleEntry();
  EXPECT_THAT(entry.text_message, HasSubstr("test_info_log_message"));
  EXPECT_EQ(entry.log_severity, absl::LogSeverity::kInfo);
}

TEST_F(LoggingTest, LogMacroLogsWithWarningSeverity) {
  XLS_LOG(WARNING) << "test_info_log_message";

  auto entry = GetSingleEntry();
  EXPECT_THAT(entry.text_message, HasSubstr("test_info_log_message"));
  EXPECT_EQ(entry.log_severity, absl::LogSeverity::kWarning);
}

TEST_F(LoggingTest, LogMacroLogsWithRuntimeSpecifiedSeverity) {
  auto severity = absl::LogSeverity::kError;
  XLS_LOG(LEVEL(severity)) << "test_info_log_message";

  auto entry = GetSingleEntry();
  EXPECT_THAT(entry.text_message, HasSubstr("test_info_log_message"));
  EXPECT_EQ(entry.log_severity, absl::LogSeverity::kError);
}

TEST_F(LoggingTest, VlogMacroLogsVerboseMessage) {
  XLS_VLOG(6) << "verbose_message";

  auto entry = GetSingleEntry();
  EXPECT_THAT(entry.text_message, HasSubstr("verbose_message"));
  EXPECT_EQ(entry.log_severity, absl::LogSeverity::kInfo);
  EXPECT_EQ(entry.verbosity, 6);
}

TEST_F(LoggingTest, CheckDoesNothingWhenConditionHolds) {
  CHECK(true);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, CheckCrashesWhenConditionFails) {
  EXPECT_DEATH(
      {
        bool condition_variable = false;
        CHECK(condition_variable);
      },
      HasSubstr("condition_variable"));
}

TEST_F(LoggingTest, CheckPrintsStackTraceWhenConditionFails) {
  EXPECT_DEATH(
      {
        bool condition_variable = false;
        CHECK(condition_variable);
      },
      HasSubstr("CheckPrintsStackTraceWhenConditionFails"));
}

TEST_F(LoggingTest, QcheckDoesNothingWhenConditionHolds) {
  QCHECK(true);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, QcheckCrashesWhenConditionFails) {
  EXPECT_DEATH(
      {
        bool condition_variable = false;
        QCHECK(condition_variable);
      },
      HasSubstr("condition_variable"));
}

TEST_F(LoggingTest, QcheckDoesNotPrintStackTraceWhenConditionFails) {
  EXPECT_DEATH(
      {
        bool condition_variable = false;
        CHECK(condition_variable);
      },
      Not(HasSubstr("CheckPrintsStackTraceWhenConditionFails")));
}

TEST_F(LoggingTest, CheckEqDoesNothingWhenConditionHolds) {
  CHECK_EQ(1, 1);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, CheckEqCrashesWhenConditionFails) {
  EXPECT_DEATH({ CHECK_EQ(1, 2); }, HasSubstr("1 == 2"));
}

TEST_F(LoggingTest, CheckNeDoesNothingWhenConditionHolds) {
  CHECK_NE(1, 2);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, CheckNeCrashesWhenConditionFails) {
  EXPECT_DEATH({ CHECK_NE(1, 1); }, HasSubstr("1 != 1"));
}

TEST_F(LoggingTest, CheckLeDoesNothingWhenConditionHolds) {
  CHECK_LE(1, 1);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, CheckLeCrashesWhenConditionFails) {
  EXPECT_DEATH({ CHECK_LE(2, 1); }, HasSubstr("2 <= 1"));
}

TEST_F(LoggingTest, CheckLtDoesNothingWhenConditionHolds) {
  CHECK_LT(1, 2);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, CheckLtCrashesWhenConditionFails) {
  EXPECT_DEATH({ CHECK_LT(1, 1); }, HasSubstr("1 < 1"));
}

TEST_F(LoggingTest, CheckGeDoesNothingWhenConditionHolds) {
  CHECK_GE(1, 1);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, CheckGeCrashesWhenConditionFails) {
  EXPECT_DEATH({ CHECK_GE(1, 2); }, HasSubstr("1 >= 2"));
}

TEST_F(LoggingTest, CheckGtDoesNothingWhenConditionHolds) {
  CHECK_GT(2, 1);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, CheckGtCrashesWhenConditionFails) {
  EXPECT_DEATH({ CHECK_GT(1, 1); }, HasSubstr("1 > 1"));
}

TEST_F(LoggingTest, QcheckEqDoesNothingWhenConditionHolds) {
  QCHECK_EQ(1, 1);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, QcheckEqCrashesWhenConditionFails) {
  EXPECT_DEATH({ QCHECK_EQ(1, 2); }, HasSubstr("1 == 2"));
}

TEST_F(LoggingTest, QcheckNeDoesNothingWhenConditionHolds) {
  QCHECK_NE(1, 2);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, QcheckNeCrashesWhenConditionFails) {
  EXPECT_DEATH({ QCHECK_NE(1, 1); }, HasSubstr("1 != 1"));
}

TEST_F(LoggingTest, QcheckLeDoesNothingWhenConditionHolds) {
  QCHECK_LE(1, 1);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, QcheckLeCrashesWhenConditionFails) {
  EXPECT_DEATH({ QCHECK_LE(2, 1); }, HasSubstr("2 <= 1"));
}

TEST_F(LoggingTest, QcheckLtDoesNothingWhenConditionHolds) {
  QCHECK_LT(1, 2);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, QcheckLtCrashesWhenConditionFails) {
  EXPECT_DEATH({ QCHECK_LT(1, 1); }, HasSubstr("1 < 1"));
}

TEST_F(LoggingTest, QcheckGeDoesNothingWhenConditionHolds) {
  QCHECK_GE(1, 1);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, QcheckGeCrashesWhenConditionFails) {
  EXPECT_DEATH({ QCHECK_GE(1, 2); }, HasSubstr("1 >= 2"));
}

TEST_F(LoggingTest, QcheckGtDoesNothingWhenConditionHolds) {
  QCHECK_GT(2, 1);

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, QcheckGtCrashesWhenConditionFails) {
  EXPECT_DEATH({ QCHECK_GT(1, 1); }, HasSubstr("1 > 1"));
}

TEST_F(LoggingTest, CheckOkDoesNothingWithOkStatus) {
  CHECK_OK(absl::OkStatus());

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, CheckOkCrashesWithNonOkStatus) {
  EXPECT_DEATH(
      { CHECK_OK(absl::UnknownError("error_msg")); }, HasSubstr("error_msg"));
}

TEST_F(LoggingTest, CheckOkCrashesWithStackTrace) {
  EXPECT_DEATH(
      { CHECK_OK(absl::UnknownError("error_msg")); },
      HasSubstr("CheckOkCrashesWithStackTrace"));
}

TEST_F(LoggingTest, QcheckOkDoesNothingWithOkStatus) {
  QCHECK_OK(absl::OkStatus());

  EXPECT_EQ(entries_.size(), 0);
}

TEST_F(LoggingTest, QcheckOkCrashesWithNonOkStatus) {
  EXPECT_DEATH(
      { QCHECK_OK(absl::UnknownError("error_msg")); }, HasSubstr("error_msg"));
}

TEST_F(LoggingTest, QcheckOkCrashesWithoutStackTrace) {
  EXPECT_DEATH(
      { QCHECK_OK(absl::UnknownError("error_msg")); },
      Not(HasSubstr("CheckOkCrashesWithStackTrace")));
}

}  // namespace
}  // namespace my_unit_test
