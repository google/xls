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

// Defines the ScopedMockLog class (using gMock), which is convenient
// for testing code that uses LOG().

#ifndef XLS_COMMON_LOGGING_SCOPED_MOCK_LOG_H_
#define XLS_COMMON_LOGGING_SCOPED_MOCK_LOG_H_

#include <string>

#include "gmock/gmock.h"
#include "absl/base/log_severity.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/log/log_sink_registry.h"

namespace xls {
namespace testing {

// A ScopedMockLog object intercepts LOG() messages issued during its
// lifespan.  Using this together with gMock, it's very easy to test
// how a piece of code calls LOG().  The typical usage, noting the distinction
// between "uninteresting" and "unexpected"
//
//   using ::testing::_;
//   using ::testing::AnyNumber;
//   using ::testing::EndsWith;
//   using ::testing::kDoNotCaptureLogsYet;
//   using ::testing::Lt;
//   using ::testing::ScopedMockLog;
//
//   TEST(FooTest, LogsCorrectly) {
//     // Simple robust setup, ignores unexpected logs.
//     ScopedMockLog log(kDoNotCaptureLogsYet);
//     EXPECT_CALL(log, Log).Times(AnyNumber());
//
//     // We expect the WARNING "Something bad!" exactly twice.
//     EXPECT_CALL(log, Log(absl::LogSeverity::kWarning, _, "Something bad!"))
//         .Times(2);
//
//     // But we want no messages from foo.cc.
//     EXPECT_CALL(log, Log(_, EndsWith("/foo.cc"), _)).Times(0);
//
//     log.StartCapturingLogs();  // Call this after done setting expectations.
//     Foo();  // Exercises the code under test.
//   }
//
//   TEST(BarTest, LogsExactlyCorrectly) {
//     // Strict checking, fails for unexpected logs.
//     ScopedMockLog log(kDoNotCaptureLogsYet);
//
//     // For robust tests, ignore logs in other modules, otherwise your test
//     // may break if their logging changes:
//     EXPECT_CALL(log, Log(_, Not(EndsWith("/foo.cc")), _))
//         .Times(AnyNumber());  // Ignores other modules.
//
//     // We may want to ignore low-severity messages.
//     EXPECT_CALL(log, Log(Lt(absl::LogSeverity::kWarning), _, _))
//         .Times(AnyNumber());
//
//     // We expect the ERROR "Something bad!" exactly once.
//     EXPECT_CALL(log, Log(absl::LogSeverity::kError, EndsWith("/foo.cc"),
//                 "Something bad!"))
//         .Times(1);
//
//     log.StartCapturingLogs();  // Call this after done setting expectations.
//     Bar();  // Exercises the code under test.
//    }
//
// CAVEAT: base/logging does not allow a thread to call LOG() again
// when it's already inside a LOG() call.  Doing so will cause a
// deadlock.  Therefore, it's the user's responsibility to not call
// LOG() in an action triggered by ScopedMockLog::Log().  You may call
// RAW_LOG() instead.

// Used only for invoking the single-argument ScopedMockLog
// constructor.  Do not use the type LogCapturingState_ directly.
// It's OK and expected for a user to use the enum value
// kDoNotCaptureLogsYet.
enum LogCapturingState_ { kDoNotCaptureLogsYet };

class ScopedMockLog : public absl::LogSink {
 public:
  // A user can use the syntax
  //   ScopedMockLog log(kDoNotCaptureLogsYet);
  // to invoke this constructor.  A ScopedMockLog object created this way
  // does not start capturing logs until StartCapturingLogs() is called.
  explicit ScopedMockLog(LogCapturingState_ /* dummy */)
      : is_capturing_logs_(false) {}

  // When the object is destructed, it stops intercepting logs.
  ~ScopedMockLog() override {
    if (is_capturing_logs_) {
      StopCapturingLogs();
    }
  }

  // Starts log capturing if the object isn't already doing so.
  // Otherwise crashes.  Usually this method is called in the same
  // thread that created this object.  It is the user's responsibility
  // to not call this method if another thread may be calling it or
  // StopCapturingLogs() at the same time.
  void StartCapturingLogs() {
    // We don't use CHECK(), which can generate a new LOG message, and
    // thus can confuse ScopedMockLog objects or other registered
    // LogSinks.
    ABSL_RAW_CHECK(
        !is_capturing_logs_,
        "StartCapturingLogs() can be called only when the ScopedMockLog "
        "object is not capturing logs.");

    is_capturing_logs_ = true;
    absl::AddLogSink(this);
  }

  // Stops log capturing if the object is capturing logs.  Otherwise
  // crashes.  Usually this method is called in the same thread that
  // created this object.  It is the user's responsibility to not call
  // this method if another thread may be calling it or
  // StartCapturingLogs() at the same time.
  void StopCapturingLogs() {
    // We don't use CHECK(), which can generate a new LOG message, and
    // thus can confuse ScopedMockLog objects or other registered
    // LogSinks.
    ABSL_RAW_CHECK(
        is_capturing_logs_,
        "StopCapturingLogs() can be called only when the ScopedMockLog "
        "object is capturing logs.");

    is_capturing_logs_ = false;
    absl::RemoveLogSink(this);
  }

  // Implements the mock method:
  //
  //   void Log(LogSeverity severity, const string& file_path,
  //            const string& message);
  //
  // The second argument to Log() is the full path of the source file
  // in which the LOG() was issued.
  //
  // Note, that in a multi-threaded environment, all LOG() messages from a
  // single thread will be handled in sequence, but that cannot be guaranteed
  // for messages from different threads. In fact, if the same or multiple
  // expectations are matched on two threads concurrently, their actions will
  // be executed concurrently as well and may interleave.
  MOCK_METHOD3(Log,
               void(absl::LogSeverity severity, const std::string& file_path,
                    const std::string& message));

 private:
  // Implements the Send() virtual function in class LogSink.
  // Whenever a LOG() statement is executed, this function will be
  // invoked with information presented in the LOG().
  void Send(const absl::LogEntry& entry) override {
    // We are only interested in the log severity, full file name, and
    // log message.
    MessageInfo message_info;
    message_info.severity = entry.log_severity();
    message_info.file_path = std::string(entry.source_filename());
    message_info.message = std::string(entry.text_message());
    Log(message_info.severity, message_info.file_path, message_info.message);
  }

  // All relevant information about a logged message that needs to be passed
  // from Send() to WaitTillSent().
  struct MessageInfo {
    absl::LogSeverity severity;
    std::string file_path;
    std::string message;
  };

  bool is_capturing_logs_;
};

}  // namespace testing
}  // namespace xls

#endif  // XLS_COMMON_LOGGING_SCOPED_MOCK_LOG_H_
