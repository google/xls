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

#include "xls/common/logging/log_flags.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/logging/capture_stream.h"
#include "xls/common/logging/logging_test_base.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"

namespace {

using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::StartsWith;
using xls::status_testing::IsOkAndHolds;

template <typename T>
class ScopedFlagSetter {
 public:
  ScopedFlagSetter(absl::Flag<T>* flag, const T& value)
      : flag_(flag), original_value_(absl::GetFlag(*flag)) {
    absl::SetFlag(flag, value);
  }

  ~ScopedFlagSetter() { absl::SetFlag(flag_, original_value_); }

 private:
  absl::Flag<T>* flag_;
  T original_value_;
};

template <typename T>
ScopedFlagSetter(absl::Flag<T>* b, const T& e) -> ScopedFlagSetter<T>;

class LogFlagsTest : public ::xls::logging_internal::testing::LoggingTestBase {
};

TEST_F(LogFlagsTest, MinloglevelSuppressesLoggingBelowSpecifiedLevel) {
  auto set_flag = ScopedFlagSetter(
      &FLAGS_minloglevel, static_cast<int>(absl::LogSeverity::kWarning));
  XLS_LOG(INFO) << "test_msg";

  EXPECT_THAT(entries_, IsEmpty());
}

TEST_F(LogFlagsTest, MinloglevelAllowsLoggingAtSpecifiedLevel) {
  auto set_flag = ScopedFlagSetter(
      &FLAGS_minloglevel, static_cast<int>(absl::LogSeverity::kWarning));

  XLS_LOG(WARNING) << "test_msg";

  auto entry = GetSingleEntry();
  EXPECT_THAT(entry.text_message, HasSubstr("test_msg"));
}

TEST_F(LogFlagsTest, MinloglevelAllowsLoggingAboveSpecifiedLevel) {
  auto set_flag = ScopedFlagSetter(
      &FLAGS_minloglevel, static_cast<int>(absl::LogSeverity::kWarning));

  XLS_LOG(ERROR) << "test_msg";

  auto entry = GetSingleEntry();
  EXPECT_THAT(entry.text_message, HasSubstr("test_msg"));
}

TEST_F(LogFlagsTest, LogToStderrFalseDoesNotCauseInfoLoggingToStderr) {
  auto set_logtostderr = ScopedFlagSetter(&FLAGS_logtostderr, false);
  auto set_alsologtostderr = ScopedFlagSetter(&FLAGS_alsologtostderr, false);

  absl::StatusOr<std::string> output = ::xls::testing::CaptureStream(
      STDERR_FILENO, [] { XLS_LOG(INFO) << "test_info_log_message"; });

  EXPECT_THAT(output, IsOkAndHolds(Not(HasSubstr("test_info_log_message"))));
}

TEST_F(LogFlagsTest, LogToStderrTrueCausesInfoLoggingToStderr) {
  auto set_logtostderr = ScopedFlagSetter(&FLAGS_logtostderr, true);
  auto set_alsologtostderr = ScopedFlagSetter(&FLAGS_alsologtostderr, false);

  absl::StatusOr<std::string> output = ::xls::testing::CaptureStream(
      STDERR_FILENO, [] { XLS_LOG(INFO) << "test_info_log_message"; });

  EXPECT_THAT(output, IsOkAndHolds(HasSubstr("test_info_log_message")));
}

TEST_F(LogFlagsTest, AlsoLogToStderrTrueCausesInfoLoggingToStderr) {
  auto set_logtostderr = ScopedFlagSetter(&FLAGS_logtostderr, false);
  auto set_alsologtostderr = ScopedFlagSetter(&FLAGS_alsologtostderr, true);

  absl::StatusOr<std::string> output = ::xls::testing::CaptureStream(
      STDERR_FILENO, [] { XLS_LOG(INFO) << "test_info_log_message"; });

  EXPECT_THAT(output, IsOkAndHolds(HasSubstr("test_info_log_message")));
}

TEST_F(LogFlagsTest, StderrThresholdSuppressesLoggingBelowSpecifiedLevel) {
  auto set_logtostderr = ScopedFlagSetter(&FLAGS_logtostderr, false);
  auto set_alsologtostderr = ScopedFlagSetter(&FLAGS_alsologtostderr, false);
  auto set_stderrthreshold = ScopedFlagSetter(
      &FLAGS_stderrthreshold, static_cast<int>(absl::LogSeverity::kWarning));

  absl::StatusOr<std::string> output = ::xls::testing::CaptureStream(
      STDERR_FILENO, [] { XLS_LOG(INFO) << "test_info_log_message"; });

  EXPECT_THAT(output, IsOkAndHolds(Not(HasSubstr("test_info_log_message"))));
}

TEST_F(LogFlagsTest, StderrThresholdAllowsLoggingAtSpecifiedLevel) {
  auto set_flag = ScopedFlagSetter(
      &FLAGS_stderrthreshold, static_cast<int>(absl::LogSeverity::kWarning));

  absl::StatusOr<std::string> output = ::xls::testing::CaptureStream(
      STDERR_FILENO, [] { XLS_LOG(WARNING) << "test_info_log_message"; });

  EXPECT_THAT(output, IsOkAndHolds(HasSubstr("test_info_log_message")));
}

TEST_F(LogFlagsTest, StderrThresholdAllowsLoggingAboveSpecifiedLevel) {
  auto set_flag = ScopedFlagSetter(
      &FLAGS_stderrthreshold, static_cast<int>(absl::LogSeverity::kWarning));

  absl::StatusOr<std::string> output = ::xls::testing::CaptureStream(
      STDERR_FILENO, [] { XLS_LOG(ERROR) << "test_info_log_message"; });

  EXPECT_THAT(output, IsOkAndHolds(HasSubstr("test_info_log_message")));
}

TEST_F(LogFlagsTest, EnabledLogPrefixCausesLoggingToBePrefixed) {
  auto set_flag = ScopedFlagSetter(&FLAGS_log_prefix, true);

  absl::StatusOr<std::string> output = ::xls::testing::CaptureStream(
      STDERR_FILENO, [] { XLS_LOG(ERROR) << "test_info_log_message"; });

  EXPECT_THAT(output, IsOkAndHolds(HasSubstr("test_info_log_message")));
  EXPECT_THAT(output, IsOkAndHolds(StartsWith("E")));  // For ERROR.
}

TEST_F(LogFlagsTest, DisabledLogPrefixCausesLoggingToNotBePrefixed) {
  auto set_flag = ScopedFlagSetter(&FLAGS_log_prefix, false);

  absl::StatusOr<std::string> output = ::xls::testing::CaptureStream(
      STDERR_FILENO, [] { XLS_LOG(ERROR) << "test_info_log_message"; });

  EXPECT_THAT(output, IsOkAndHolds(HasSubstr("test_info_log_message")));
  EXPECT_THAT(output, IsOkAndHolds(Not(StartsWith("E"))));  // For ERROR.
}

}  // namespace
