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

#include "xls/common/status/status_builder.h"

#include <memory>
#include <sstream>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/log_severity.h"
#include "absl/log/globals.h"
#include "absl/log/log.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "xls/common/logging/scoped_mock_log.h"
#include "xls/common/source_location.h"

namespace xabsl {
namespace {

using ::absl::LogSeverity;
using ::testing::_;
using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Pointee;
using ::xls::testing::kDoNotCaptureLogsYet;
using ::xls::testing::ScopedMockLog;

// We use `#line` to produce some `source_location` values pointing at various
// different (fake) files to test e.g. `VLog`, but we use it at the end of this
// file so as not to mess up the source location data for the whole file.
// Making them static data members lets us forward-declare them and define them
// at the end.
struct Locs {
  static const xabsl::SourceLocation kSecret;
  static const xabsl::SourceLocation kLevel0;
  static const xabsl::SourceLocation kLevel1;
  static const xabsl::SourceLocation kLevel2;
  static const xabsl::SourceLocation kBar;
};

class StringSink : public absl::LogSink {
 public:
  StringSink() = default;

  void Send(const absl::LogEntry& entry) override {
    absl::StrAppend(&message_, entry.source_basename(), ":",
                    entry.source_line(), " - ", entry.text_message());
  }

  const std::string& ToString() { return message_; }

 private:
  std::string message_;
};

// Converts a StatusBuilder to a Status.
absl::Status ToStatus(const StatusBuilder& s) { return s; }

// Converts a StatusBuilder to a Status and then ignores it.
void ConvertToStatusAndIgnore(const StatusBuilder& s) {
  absl::Status status = s;
  (void)status;
}

// Converts a StatusBuilder to a StatusOr<T>.
template <typename T>
absl::StatusOr<T> ToStatusOr(const StatusBuilder& s) {
  return s;
}

TEST(StatusBuilderTest, Size) {
  EXPECT_LE(sizeof(StatusBuilder), 40)
      << "Relax this test with caution and thorough testing. If StatusBuilder "
         "is too large it can potentially blow stacks, especially in debug "
         "builds. See the comments for StatusBuilder::Rep.";
}

TEST(StatusBuilderTest, Ctors) {
  EXPECT_EQ(ToStatus(StatusBuilder(absl::StatusCode::kUnimplemented) << "nope"),
            absl::Status(absl::StatusCode::kUnimplemented, "nope"));
}

TEST(StatusBuilderTest, ExplicitSourceLocation) {
  const xabsl::SourceLocation kLocation = XABSL_LOC;

  {
    const StatusBuilder builder(absl::OkStatus(), kLocation);
    EXPECT_THAT(builder.source_location().file_name(),
                Eq(kLocation.file_name()));
    EXPECT_THAT(builder.source_location().line(), Eq(kLocation.line()));
  }
}

TEST(StatusBuilderTest, ImplicitSourceLocation) {
  const StatusBuilder builder(absl::OkStatus());
  auto loc = XABSL_LOC;
  EXPECT_THAT(builder.source_location().file_name(),
              AnyOf(Eq(loc.file_name()), Eq("<source_location>")));
  EXPECT_THAT(builder.source_location().line(),
              AnyOf(Eq(1), Eq(loc.line() - 1)));
}

TEST(StatusBuilderTest, StatusCode) {
  // OK
  {
    const StatusBuilder builder(absl::StatusCode::kOk);
    EXPECT_TRUE(builder.ok());
    EXPECT_THAT(builder.code(), Eq(absl::StatusCode::kOk));
  }

  // Non-OK code
  {
    const StatusBuilder builder(absl::StatusCode::kInvalidArgument);
    EXPECT_FALSE(builder.ok());
    EXPECT_THAT(builder.code(), Eq(absl::StatusCode::kInvalidArgument));
  }
}

TEST(StatusBuilderTest, Streaming) {
  EXPECT_THAT(
      ToStatus(StatusBuilder(absl::CancelledError(), xabsl::SourceLocation())
               << "booyah"),
      Eq(absl::CancelledError("booyah")));
  EXPECT_THAT(
      ToStatus(
          StatusBuilder(absl::AbortedError("hello"), xabsl::SourceLocation())
          << "world"),
      Eq(absl::AbortedError("hello; world")));
  EXPECT_THAT(
      ToStatus(StatusBuilder(
                   absl::Status(absl::StatusCode::kUnimplemented, "enosys"),
                   xabsl::SourceLocation())
               << "punk!"),
      Eq(absl::Status(absl::StatusCode::kUnimplemented, "enosys; punk!")));
}

TEST(StatusBuilderTest, PrependLvalue) {
  {
    StatusBuilder builder(absl::CancelledError(), xabsl::SourceLocation());
    EXPECT_THAT(ToStatus(builder.SetPrepend() << "booyah"),
                Eq(absl::CancelledError("booyah")));
  }
  {
    StatusBuilder builder(absl::AbortedError(" hello"),
                          xabsl::SourceLocation());
    EXPECT_THAT(ToStatus(builder.SetPrepend() << "world"),
                Eq(absl::AbortedError("world hello")));
  }
}

TEST(StatusBuilderTest, PrependRvalue) {
  EXPECT_THAT(
      ToStatus(StatusBuilder(absl::CancelledError(), xabsl::SourceLocation())
                   .SetPrepend()
               << "booyah"),
      Eq(absl::CancelledError("booyah")));
  EXPECT_THAT(ToStatus(StatusBuilder(absl::AbortedError(" hello"),
                                     xabsl::SourceLocation())
                           .SetPrepend()
                       << "world"),
              Eq(absl::AbortedError("world hello")));
}

TEST(StatusBuilderTest, AppendLvalue) {
  {
    StatusBuilder builder(absl::CancelledError(), xabsl::SourceLocation());
    EXPECT_THAT(ToStatus(builder.SetAppend() << "booyah"),
                Eq(absl::CancelledError("booyah")));
  }
  {
    StatusBuilder builder(absl::AbortedError("hello"), xabsl::SourceLocation());
    EXPECT_THAT(ToStatus(builder.SetAppend() << " world"),
                Eq(absl::AbortedError("hello world")));
  }
}

TEST(StatusBuilderTest, AppendRvalue) {
  EXPECT_THAT(
      ToStatus(StatusBuilder(absl::CancelledError(), xabsl::SourceLocation())
                   .SetAppend()
               << "booyah"),
      Eq(absl::CancelledError("booyah")));
  EXPECT_THAT(ToStatus(StatusBuilder(absl::AbortedError("hello"),
                                     xabsl::SourceLocation())
                           .SetAppend()
                       << " world"),
              Eq(absl::AbortedError("hello world")));
}

TEST(StatusBuilderTest, LogToMultipleErrorLevelsLvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!"))).Times(1);
  EXPECT_CALL(log, Log(LogSeverity::kError, _, HasSubstr("yes!"))).Times(1);
  EXPECT_CALL(log, Log(LogSeverity::kInfo, _, HasSubstr("Oui!"))).Times(1);
  log.StartCapturingLogs();
  {
    StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
    ConvertToStatusAndIgnore(builder.Log(LogSeverity::kWarning) << "no!");
  }
  {
    StatusBuilder builder(absl::AbortedError(""), Locs::kSecret);

    ConvertToStatusAndIgnore(builder.Log(LogSeverity::kError) << "yes!");

    // This one shouldn't log because vlogging is disabled.
    absl::SetGlobalVLogLevel(0);
    ConvertToStatusAndIgnore(builder.VLog(2) << "Non!");

    absl::SetGlobalVLogLevel(2);
    ConvertToStatusAndIgnore(builder.VLog(2) << "Oui!");
  }
}

TEST(StatusBuilderTest, LogToMultipleErrorLevelsRvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!"))).Times(1);
  EXPECT_CALL(log, Log(LogSeverity::kError, _, HasSubstr("yes!"))).Times(1);
  EXPECT_CALL(log, Log(LogSeverity::kInfo, _, HasSubstr("Oui!"))).Times(1);
  log.StartCapturingLogs();
  ConvertToStatusAndIgnore(StatusBuilder(absl::CancelledError(), Locs::kSecret)
                               .Log(LogSeverity::kWarning)
                           << "no!");
  ConvertToStatusAndIgnore(StatusBuilder(absl::AbortedError(""), Locs::kSecret)
                               .Log(LogSeverity::kError)
                           << "yes!");
  // This one shouldn't log because vlogging is disabled.
  absl::SetGlobalVLogLevel(0);
  ConvertToStatusAndIgnore(
      StatusBuilder(absl::AbortedError(""), Locs::kSecret).VLog(2) << "Non!");
  absl::SetGlobalVLogLevel(2);
  ConvertToStatusAndIgnore(
      StatusBuilder(absl::AbortedError(""), Locs::kSecret).VLog(2) << "Oui!");
}

TEST(StatusBuilderTest, LogEveryNFirstLogs) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!"))).Times(1);
  log.StartCapturingLogs();

  StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
  // Only 1 of the 3 should log.
  for (int i = 0; i < 3; ++i) {
    ConvertToStatusAndIgnore(builder.LogEveryN(LogSeverity::kWarning, 3)
                             << "no!");
  }
}

TEST(StatusBuilderTest, LogEveryN2Lvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!"))).Times(3);
  log.StartCapturingLogs();

  StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
  // Only 3 of the 6 should log.
  for (int i = 0; i < 6; ++i) {
    ConvertToStatusAndIgnore(builder.LogEveryN(LogSeverity::kWarning, 2)
                             << "no!");
  }
}

TEST(StatusBuilderTest, LogEveryN3Lvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!"))).Times(2);
  log.StartCapturingLogs();

  StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
  // Only 2 of the 6 should log.
  for (int i = 0; i < 6; ++i) {
    ConvertToStatusAndIgnore(builder.LogEveryN(LogSeverity::kWarning, 3)
                             << "no!");
  }
}

TEST(StatusBuilderTest, LogEveryN7Lvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!"))).Times(3);
  log.StartCapturingLogs();

  StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
  // Only 3 of the 21 should log.
  for (int i = 0; i < 21; ++i) {
    ConvertToStatusAndIgnore(builder.LogEveryN(LogSeverity::kWarning, 7)
                             << "no!");
  }
}

TEST(StatusBuilderTest, LogEveryNRvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!"))).Times(2);
  log.StartCapturingLogs();

  // Only 2 of the 4 should log.
  for (int i = 0; i < 4; ++i) {
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::CancelledError(), Locs::kSecret)
            .LogEveryN(LogSeverity::kWarning, 2)
        << "no!");
  }
}

TEST(StatusBuilderTest, LogEveryFirstLogs) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!"))).Times(1);
  log.StartCapturingLogs();

  StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
  ConvertToStatusAndIgnore(
      builder.LogEvery(LogSeverity::kWarning, absl::Seconds(2)) << "no!");
}

TEST(StatusBuilderTest, LogEveryLvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!")))
      .Times(testing::AtMost(3));
  log.StartCapturingLogs();

  StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
  for (int i = 0; i < 4; ++i) {
    ConvertToStatusAndIgnore(
        builder.LogEvery(LogSeverity::kWarning, absl::Seconds(2)) << "no!");
    absl::SleepFor(absl::Seconds(1));
  }
}

TEST(StatusBuilderTest, LogEveryRvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!")))
      .Times(testing::AtMost(3));
  log.StartCapturingLogs();

  for (int i = 0; i < 4; ++i) {
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::CancelledError(), Locs::kSecret)
            .LogEvery(LogSeverity::kWarning, absl::Seconds(2))
        << "no!");
    absl::SleepFor(absl::Seconds(1));
  }
}

TEST(StatusBuilderTest, LogEveryZeroDuration) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, _, HasSubstr("no!")))
      .Times(testing::Exactly(4));
  log.StartCapturingLogs();

  StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
  for (int i = 0; i < 4; ++i) {
    ConvertToStatusAndIgnore(
        builder.LogEvery(LogSeverity::kWarning, absl::ZeroDuration()) << "no!");
  }
}

TEST(StatusBuilderTest, VLogModuleLvalue) {
  absl::SetGlobalVLogLevel(0);
  absl::SetVLogLevel("level0", 0);
  absl::SetVLogLevel("level1", 1);
  absl::SetVLogLevel("level2", 2);
  {
    ScopedMockLog log(kDoNotCaptureLogsYet);
    EXPECT_CALL(log, Log(LogSeverity::kInfo, HasSubstr("level0.cc"), _))
        .Times(0);
    EXPECT_CALL(log, Log(LogSeverity::kInfo, HasSubstr("level1.cc"), _))
        .Times(1);
    EXPECT_CALL(log, Log(LogSeverity::kInfo, HasSubstr("level2.cc"), _))
        .Times(2);
    log.StartCapturingLogs();

    {
      StatusBuilder builder(absl::AbortedError(""), Locs::kLevel0);
      ConvertToStatusAndIgnore(builder.VLog(1));
      ConvertToStatusAndIgnore(builder.VLog(2));
    }
    {
      StatusBuilder builder(absl::AbortedError(""), Locs::kLevel1);
      ConvertToStatusAndIgnore(builder.VLog(1));
      ConvertToStatusAndIgnore(builder.VLog(2));
    }
    {
      StatusBuilder builder(absl::AbortedError(""), Locs::kLevel2);
      ConvertToStatusAndIgnore(builder.VLog(1));
      ConvertToStatusAndIgnore(builder.VLog(2));
    }
  }

  absl::SetVLogLevel("level0", 2);
  {
    ScopedMockLog log(kDoNotCaptureLogsYet);
    EXPECT_CALL(log, Log(LogSeverity::kInfo, HasSubstr("level0.cc"), _))
        .Times(2);
    log.StartCapturingLogs();

    StatusBuilder builder(absl::AbortedError(""), Locs::kLevel0);
    ConvertToStatusAndIgnore(builder.VLog(1));
    ConvertToStatusAndIgnore(builder.VLog(2));
  }
}

TEST(StatusBuilderTest, VLogModuleRvalue) {
  absl::SetGlobalVLogLevel(0);
  absl::SetVLogLevel("level0", 0);
  absl::SetVLogLevel("level1", 1);
  absl::SetVLogLevel("level2", 2);
  {
    ScopedMockLog log(kDoNotCaptureLogsYet);
    EXPECT_CALL(log, Log(LogSeverity::kInfo, HasSubstr("level0.cc"), _))
        .Times(0);
    EXPECT_CALL(log, Log(LogSeverity::kInfo, HasSubstr("level1.cc"), _))
        .Times(1);
    EXPECT_CALL(log, Log(LogSeverity::kInfo, HasSubstr("level2.cc"), _))
        .Times(2);
    log.StartCapturingLogs();
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::AbortedError(""), Locs::kLevel0).VLog(1));
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::AbortedError(""), Locs::kLevel0).VLog(2));
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::AbortedError(""), Locs::kLevel1).VLog(1));
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::AbortedError(""), Locs::kLevel1).VLog(2));
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::AbortedError(""), Locs::kLevel2).VLog(1));
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::AbortedError(""), Locs::kLevel2).VLog(2));
  }

  absl::SetVLogLevel("level0", 2);
  {
    ScopedMockLog log(kDoNotCaptureLogsYet);
    EXPECT_CALL(log, Log(LogSeverity::kInfo, HasSubstr("level0.cc"), _))
        .Times(2);
    log.StartCapturingLogs();
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::AbortedError(""), Locs::kLevel0).VLog(1));
    ConvertToStatusAndIgnore(
        StatusBuilder(absl::AbortedError(""), Locs::kLevel0).VLog(2));
  }
}

TEST(StatusBuilderTest, LogIncludesFileAndLine) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kWarning, HasSubstr("/foo/secret.cc"),
                       HasSubstr("maybe?")))
      .Times(1);
  log.StartCapturingLogs();
  ConvertToStatusAndIgnore(StatusBuilder(absl::AbortedError(""), Locs::kSecret)
                               .Log(LogSeverity::kWarning)
                           << "maybe?");
}

TEST(StatusBuilderTest, NoLoggingLvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(_, _, _)).Times(0);
  log.StartCapturingLogs();

  {
    StatusBuilder builder(absl::AbortedError(""), Locs::kSecret);
    EXPECT_THAT(ToStatus(builder << "nope"), Eq(absl::AbortedError("nope")));
  }
  {
    StatusBuilder builder(absl::AbortedError(""), Locs::kSecret);
    // Enable and then disable logging.
    EXPECT_THAT(ToStatus(builder.Log(LogSeverity::kWarning).SetNoLogging()
                         << "not at all"),
                Eq(absl::AbortedError("not at all")));
  }
}

TEST(StatusBuilderTest, NoLoggingRvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(_, _, _)).Times(0);
  log.StartCapturingLogs();
  EXPECT_THAT(
      ToStatus(StatusBuilder(absl::AbortedError(""), Locs::kSecret) << "nope"),
      Eq(absl::AbortedError("nope")));
  // Enable and then disable logging.
  EXPECT_THAT(ToStatus(StatusBuilder(absl::AbortedError(""), Locs::kSecret)
                           .Log(LogSeverity::kWarning)
                           .SetNoLogging()
                       << "not at all"),
              Eq(absl::AbortedError("not at all")));
}

TEST(StatusBuilderTest, EmitStackTracePlusSomethingLikelyUniqueLvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log,
              Log(LogSeverity::kError, HasSubstr("/bar/baz.cc"),
                  // this method shows up in the stack trace
                  HasSubstr("EmitStackTracePlusSomethingLikelyUniqueLvalue")))
      .Times(1);
  log.StartCapturingLogs();
  StatusBuilder builder(absl::AbortedError(""), Locs::kBar);
  ConvertToStatusAndIgnore(builder.LogError().EmitStackTrace() << "maybe?");
}

TEST(StatusBuilderTest, EmitStackTracePlusSomethingLikelyUniqueRvalue) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log,
              Log(LogSeverity::kError, HasSubstr("/bar/baz.cc"),
                  // this method shows up in the stack trace
                  HasSubstr("EmitStackTracePlusSomethingLikelyUniqueRvalue")))
      .Times(1);
  log.StartCapturingLogs();
  ConvertToStatusAndIgnore(StatusBuilder(absl::AbortedError(""), Locs::kBar)
                               .LogError()
                               .EmitStackTrace()
                           << "maybe?");
}

TEST(StatusBuilderTest, AlsoOutputToSinkLvalue) {
  StringSink sink;
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kError, _, HasSubstr("yes!"))).Times(1);
  log.StartCapturingLogs();
  {
    // This should not output anything to sink because logging is not enabled.
    StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
    ConvertToStatusAndIgnore(builder.AlsoOutputToSink(&sink) << "no!");
    EXPECT_TRUE(sink.ToString().empty());
  }
  {
    StatusBuilder builder(absl::CancelledError(), Locs::kSecret);
    ConvertToStatusAndIgnore(
        builder.Log(LogSeverity::kError).AlsoOutputToSink(&sink) << "yes!");
    EXPECT_TRUE(absl::StrContains(sink.ToString(), "yes!"));
  }
}

TEST(StatusBuilderTest, AlsoOutputToSinkRvalue) {
  StringSink sink;
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(LogSeverity::kError, _, HasSubstr("yes!"))).Times(1);
  log.StartCapturingLogs();
  // This should not output anything to sink because logging is not enabled.
  ConvertToStatusAndIgnore(StatusBuilder(absl::CancelledError(), Locs::kSecret)
                               .AlsoOutputToSink(&sink)
                           << "no!");
  EXPECT_TRUE(sink.ToString().empty());
  ConvertToStatusAndIgnore(StatusBuilder(absl::CancelledError(), Locs::kSecret)
                               .Log(LogSeverity::kError)
                               .AlsoOutputToSink(&sink)
                           << "yes!");
  EXPECT_TRUE(absl::StrContains(sink.ToString(), "yes!"));
}

TEST(StatusBuilderTest, WithRvalueRef) {
  auto policy = [](StatusBuilder sb) { return sb << "policy"; };
  EXPECT_THAT(ToStatus(StatusBuilder(absl::AbortedError("hello"),
                                     xabsl::SourceLocation())
                           .With(policy)),
              Eq(absl::AbortedError("hello; policy")));
}

TEST(StatusBuilderTest, WithRef) {
  auto policy = [](StatusBuilder sb) { return sb << "policy"; };
  StatusBuilder sb(absl::AbortedError("zomg"), xabsl::SourceLocation());
  EXPECT_THAT(ToStatus(sb.With(policy)),
              Eq(absl::AbortedError("zomg; policy")));
}

TEST(StatusBuilderTest, WithTypeChange) {
  auto policy = [](StatusBuilder sb) -> std::string {
    return sb.ok() ? "true" : "false";
  };
  EXPECT_EQ(StatusBuilder(absl::CancelledError(), xabsl::SourceLocation())
                .With(policy),
            "false");
  EXPECT_EQ(
      StatusBuilder(absl::OkStatus(), xabsl::SourceLocation()).With(policy),
      "true");
}

TEST(StatusBuilderTest, WithVoidTypeAndSideEffects) {
  absl::StatusCode code = absl::StatusCode::kUnknown;
  auto policy = [&code](absl::Status status) { code = status.code(); };
  StatusBuilder(absl::CancelledError(), xabsl::SourceLocation()).With(policy);
  EXPECT_EQ(absl::StatusCode::kCancelled, code);
  StatusBuilder(absl::OkStatus(), xabsl::SourceLocation()).With(policy);
  EXPECT_EQ(absl::StatusCode::kOk, code);
}

struct MoveOnlyAdaptor {
  std::unique_ptr<int> value;
  std::unique_ptr<int> operator()(const absl::Status&) && {
    return std::move(value);
  }
};

TEST(StatusBuilderTest, WithMoveOnlyAdaptor) {
  StatusBuilder sb(absl::AbortedError("zomg"), xabsl::SourceLocation());
  EXPECT_THAT(sb.With(MoveOnlyAdaptor{std::make_unique<int>(100)}),
              Pointee(100));
  EXPECT_THAT(StatusBuilder(absl::AbortedError("zomg"), xabsl::SourceLocation())
                  .With(MoveOnlyAdaptor{std::make_unique<int>(100)}),
              Pointee(100));
}

template <typename T>
std::string ToStringViaStream(const T& x) {
  std::ostringstream os;
  os << x;
  return os.str();
}

TEST(StatusBuilderTest, StreamInsertionOperator) {
  absl::Status status = absl::AbortedError("zomg");
  StatusBuilder builder(status, xabsl::SourceLocation());
  EXPECT_EQ(ToStringViaStream(status), ToStringViaStream(builder));
  EXPECT_EQ(ToStringViaStream(status),
            ToStringViaStream(StatusBuilder(status, xabsl::SourceLocation())));
}

TEST(StatusBuilderTest, SetCode) {
  StatusBuilder builder(absl::StatusCode::kUnknown, xabsl::SourceLocation());
  builder.SetCode(absl::StatusCode::kResourceExhausted);
  absl::Status status = builder;
  EXPECT_EQ(status, absl::ResourceExhaustedError(""));
}

TEST(CanonicalErrorsTest, CreateAndClassify) {
  struct CanonicalErrorTest {
    absl::StatusCode code;
    StatusBuilder builder;
  };
  xabsl::SourceLocation loc = xabsl::SourceLocation::current();
  CanonicalErrorTest canonical_errors[] = {
      // implicit location
      {absl::StatusCode::kAborted, AbortedErrorBuilder()},
      {absl::StatusCode::kAlreadyExists, AlreadyExistsErrorBuilder()},
      {absl::StatusCode::kCancelled, CancelledErrorBuilder()},
      {absl::StatusCode::kDataLoss, DataLossErrorBuilder()},
      {absl::StatusCode::kDeadlineExceeded, DeadlineExceededErrorBuilder()},
      {absl::StatusCode::kFailedPrecondition, FailedPreconditionErrorBuilder()},
      {absl::StatusCode::kInternal, InternalErrorBuilder()},
      {absl::StatusCode::kInvalidArgument, InvalidArgumentErrorBuilder()},
      {absl::StatusCode::kNotFound, NotFoundErrorBuilder()},
      {absl::StatusCode::kOutOfRange, OutOfRangeErrorBuilder()},
      {absl::StatusCode::kPermissionDenied, PermissionDeniedErrorBuilder()},
      {absl::StatusCode::kUnauthenticated, UnauthenticatedErrorBuilder()},
      {absl::StatusCode::kResourceExhausted, ResourceExhaustedErrorBuilder()},
      {absl::StatusCode::kUnavailable, UnavailableErrorBuilder()},
      {absl::StatusCode::kUnimplemented, UnimplementedErrorBuilder()},
      {absl::StatusCode::kUnknown, UnknownErrorBuilder()},

      // explicit location
      {absl::StatusCode::kAborted, AbortedErrorBuilder(loc)},
      {absl::StatusCode::kAlreadyExists, AlreadyExistsErrorBuilder(loc)},
      {absl::StatusCode::kCancelled, CancelledErrorBuilder(loc)},
      {absl::StatusCode::kDataLoss, DataLossErrorBuilder(loc)},
      {absl::StatusCode::kDeadlineExceeded, DeadlineExceededErrorBuilder(loc)},
      {absl::StatusCode::kFailedPrecondition,
       FailedPreconditionErrorBuilder(loc)},
      {absl::StatusCode::kInternal, InternalErrorBuilder(loc)},
      {absl::StatusCode::kInvalidArgument, InvalidArgumentErrorBuilder(loc)},
      {absl::StatusCode::kNotFound, NotFoundErrorBuilder(loc)},
      {absl::StatusCode::kOutOfRange, OutOfRangeErrorBuilder(loc)},
      {absl::StatusCode::kPermissionDenied, PermissionDeniedErrorBuilder(loc)},
      {absl::StatusCode::kUnauthenticated, UnauthenticatedErrorBuilder(loc)},
      {absl::StatusCode::kResourceExhausted,
       ResourceExhaustedErrorBuilder(loc)},
      {absl::StatusCode::kUnavailable, UnavailableErrorBuilder(loc)},
      {absl::StatusCode::kUnimplemented, UnimplementedErrorBuilder(loc)},
      {absl::StatusCode::kUnknown, UnknownErrorBuilder(loc)},
  };

  for (const auto& test : canonical_errors) {
    SCOPED_TRACE(absl::StrCat("absl::StatusCode::",
                              absl::StatusCodeToString(test.code)));

    // Ensure that the creator does, in fact, create status objects in the
    // canonical space, with the expected error code and message.
    std::string message =
        absl::StrCat("error code ", test.code, " test message");
    absl::Status status = StatusBuilder(test.builder) << message;
    EXPECT_EQ(test.code, status.code());
    EXPECT_EQ(message, status.message());
  }
}

#line 1337 "/foo/secret.cc"
const xabsl::SourceLocation Locs::kSecret = xabsl::SourceLocation::current();
#line 1234 "/tmp/level0.cc"
const xabsl::SourceLocation Locs::kLevel0 = xabsl::SourceLocation::current();
#line 1234 "/tmp/level1.cc"
const xabsl::SourceLocation Locs::kLevel1 = xabsl::SourceLocation::current();
#line 1234 "/tmp/level2.cc"
const xabsl::SourceLocation Locs::kLevel2 = xabsl::SourceLocation::current();
#line 1337 "/bar/baz.cc"
const xabsl::SourceLocation Locs::kBar = xabsl::SourceLocation::current();

}  // namespace
}  // namespace xabsl
