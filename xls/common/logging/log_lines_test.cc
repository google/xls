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

#include "xls/common/logging/log_lines.h"

#include <string>
#include <string_view>

#include "absl/base/log_severity.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/logging/scoped_mock_log.h"

namespace {

class LogLinesTest : public ::testing::Test {
 protected:
  LogLinesTest() : mock_log_(::xls::testing::kDoNotCaptureLogsYet) {}
  ~LogLinesTest() override = default;
  void StartCapturingLogs() { mock_log_.StartCapturingLogs(); }
  ::xls::testing::ScopedMockLog mock_log_;
};

#define XLS_EXPECT_LOG(TYPE, MESSAGE)                                        \
  EXPECT_CALL(mock_log_, Log(TYPE, ::testing::_, ::testing::StrEq(MESSAGE))) \
      .Times(1)

TEST_F(LogLinesTest, WorksWithInfo) {
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Some message.");
  StartCapturingLogs();
  XLS_LOG_LINES(INFO, "Some message.");
}

TEST_F(LogLinesTest, WorksWithWarning) {
  XLS_EXPECT_LOG(absl::LogSeverity::kWarning, "Some message.");
  StartCapturingLogs();
  XLS_LOG_LINES(WARNING, "Some message.");
}

TEST_F(LogLinesTest, WorksWithError) {
  XLS_EXPECT_LOG(absl::LogSeverity::kError, "Some message.");
  StartCapturingLogs();
  XLS_LOG_LINES(ERROR, "Some message.");
}

TEST_F(LogLinesTest, WorksWithFatal) {
  EXPECT_DEATH(XLS_LOG_LINES(FATAL, "Some message."),
               "Aborting due to previous errors.");
}

TEST_F(LogLinesTest, WorksWithMultipleLinesEachEndingInNewline) {
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Some message1.");
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Some message2.");
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Some message3.");
  StartCapturingLogs();
  XLS_LOG_LINES(INFO, "Some message1.\nSome message2.\nSome message3.\n");
}

TEST_F(LogLinesTest, WorksWithMultipleLinesLastNotEndingInNewline) {
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Some message1.");
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Some message2.");
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Some message3.");
  StartCapturingLogs();
  XLS_LOG_LINES(INFO, "Some message1.\nSome message2.\nSome message3.");
}

TEST_F(LogLinesTest, RespectsEmptyLines) {
  EXPECT_CALL(mock_log_, Log(absl::LogSeverity::kInfo, ::testing::_, ""))
      .Times(3);
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Some message1.");
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Some message2.");
  StartCapturingLogs();
  XLS_LOG_LINES(INFO, "\nSome message1.\n\nSome message2.\n\n");
}

TEST_F(LogLinesTest, StringTemporaryIsntBrokenByStringPiece) {
  XLS_EXPECT_LOG(absl::LogSeverity::kInfo, "Whatever.");
  StartCapturingLogs();
  XLS_LOG_LINES(INFO, std::string("Whatever."));
}

}  // namespace
