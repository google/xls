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

#include "xls/common/logging/log_message.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/log_severity.h"
#include "absl/flags/flag.h"
#include "absl/strings/str_format.h"
#include "absl/status/status.h"
#include "xls/common/logging/log_flags.h"
#include "xls/common/logging/logging_test_base.h"
#include "xls/common/logging/scoped_mock_log.h"
#include "xls/common/source_location.h"
#include "xls/common/strerror.h"

namespace xls {
namespace logging_internal {
namespace {

using ::testing::_;
using ::testing::AnyOf;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Not;
using ::xls::testing::kDoNotCaptureLogsYet;
using ::xls::testing::ScopedMockLog;

// `#line` is used to produce some `source_location` values pointing at various
// different (fake) files to test e.g. `VLog`, but we use it at the end of this
// file so as not to mess up the source location data for the whole file.
// Making them static data members lets us forward-declare them and define them
// at the end.
struct Locs {
  static const xabsl::SourceLocation kFakeLocation;
};

class LogMessageTest : public testing::LoggingTestBase {
 public:
  LogMessageTest() { absl::SetFlag(&FLAGS_logtostderr, false); }
};

TEST_F(LogMessageTest, FailedCheckPrintsCheckFailed) {
  // This is a test for LogMessage::WithCheckFailureMessage.
  EXPECT_DEATH(
      {
        bool condition_variable = false;
        XLS_CHECK(condition_variable);
      },
      HasSubstr("Check failed: condition_variable"));
}

TEST_F(LogMessageTest, AtLocationWithFileAndLineSetsLocation) {
  XLS_LOG(INFO).AtLocation("dir/a_fake_file.cc", 467) << "msg";

  auto entry = GetSingleEntry();
  EXPECT_EQ(entry.source_filename, "dir/a_fake_file.cc");
  EXPECT_EQ(entry.source_basename, "a_fake_file.cc");
  EXPECT_EQ(entry.source_line, 467);
}

TEST_F(LogMessageTest, AtLocationWithSourceLocationSetsLocation) {
  XLS_LOG(INFO).AtLocation(Locs::kFakeLocation) << "msg";

  auto entry = GetSingleEntry();
  EXPECT_EQ(entry.source_filename, "/foo/fake.cc");
  EXPECT_EQ(entry.source_basename, "fake.cc");
  EXPECT_EQ(entry.source_line, 1337);
}

TEST_F(LogMessageTest, PrefixIsTrueByDefault) {
  XLS_LOG(INFO) << "msg";

  auto entry = GetSingleEntry();
  EXPECT_TRUE(entry.prefix);
}

TEST_F(LogMessageTest, NoPrefixSetsPrefixToFalse) {
  XLS_LOG(INFO).NoPrefix() << "msg";

  auto entry = GetSingleEntry();
  EXPECT_FALSE(entry.prefix);
}

TEST_F(LogMessageTest, WithPerrorAppendsErrnoInfoToMessage) {
  errno = ENOENT;
  XLS_LOG(INFO).WithPerror() << "msg";

  auto entry = GetSingleEntry();
  EXPECT_THAT(entry.text_message, HasSubstr(Strerror(ENOENT)));
}

TEST_F(LogMessageTest, WithVerbositySetsVerbosityLevel) {
  XLS_LOG(INFO).WithVerbosity(1232) << "msg";

  auto entry = GetSingleEntry();
  EXPECT_EQ(entry.verbosity, 1232);
}

TEST_F(LogMessageTest, ToSinkAlsoLogsToProvidedSinkAlso) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _, "msg")).Times(1);

  XLS_LOG(INFO).ToSinkAlso(&log) << "msg";

  auto entry = GetSingleEntry();
  EXPECT_EQ(entry.text_message, "msg");
}

TEST_F(LogMessageTest, ToSinkOnlyLogsToProvidedSinkOnly) {
  ScopedMockLog log(kDoNotCaptureLogsYet);
  EXPECT_CALL(log, Log(absl::LogSeverity::kInfo, _, "msg")).Times(1);

  XLS_LOG(INFO).ToSinkOnly(&log) << "msg";

  EXPECT_THAT(entries_, IsEmpty());
}

TEST_F(LogMessageTest, StreamReturnsSelf) {
  XLS_LOG(INFO).stream().WithVerbosity(1232) << "msg";

  auto entry = GetSingleEntry();
  EXPECT_EQ(entry.verbosity, 1232);
}

TEST_F(LogMessageTest, StreamOperatorAcceptsChars) {
  XLS_LOG(INFO) << static_cast<char>('z');
  EXPECT_EQ(GetSingleEntry().text_message, "z");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsSignedChars) {
  XLS_LOG(INFO) << static_cast<signed char>('z');
  EXPECT_EQ(GetSingleEntry().text_message, "z");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsUnsignedChars) {
  XLS_LOG(INFO) << static_cast<unsigned char>('z');
  EXPECT_EQ(GetSingleEntry().text_message, "z");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsSignedShort) {
  XLS_LOG(INFO) << static_cast<signed short>(-1337);  // NOLINT
  EXPECT_EQ(GetSingleEntry().text_message, "-1337");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsSignedInt) {
  XLS_LOG(INFO) << static_cast<signed int>(-1337);
  EXPECT_EQ(GetSingleEntry().text_message, "-1337");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsSignedLong) {
  XLS_LOG(INFO) << static_cast<signed long>(-1337);  // NOLINT
  EXPECT_EQ(GetSingleEntry().text_message, "-1337");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsSignedLongLong) {
  XLS_LOG(INFO) << static_cast<signed long long>(-1337);  // NOLINT
  EXPECT_EQ(GetSingleEntry().text_message, "-1337");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsUnsignedShort) {
  XLS_LOG(INFO) << static_cast<unsigned short>(1337);  // NOLINT
  EXPECT_EQ(GetSingleEntry().text_message, "1337");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsUnsignedInt) {
  XLS_LOG(INFO) << static_cast<unsigned int>(1337);
  EXPECT_EQ(GetSingleEntry().text_message, "1337");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsUnsignedLong) {
  XLS_LOG(INFO) << static_cast<unsigned long>(1337);  // NOLINT
  EXPECT_EQ(GetSingleEntry().text_message, "1337");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsUnsignedLongLong) {
  XLS_LOG(INFO) << static_cast<unsigned long long>(1337);  // NOLINT
  EXPECT_EQ(GetSingleEntry().text_message, "1337");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsIntPointer) {
  int i = 0xbeef;
  std::string expected = absl::StrFormat("%p", &i);
  XLS_LOG(INFO) << &i;
  EXPECT_EQ(GetSingleEntry().text_message, expected);
}

TEST_F(LogMessageTest, StreamOperatorAcceptsNullIntPointer) {
  int* i_ptr = nullptr;
  XLS_LOG(INFO) << i_ptr;
  // It's undefined behavior to send nullptr to a stream, but our
  // compilers/libraries output either "(nil)" or "0", so this is just a smoke
  // test.
  EXPECT_THAT(GetSingleEntry().text_message, AnyOf(Eq("(nil)"), "0"));
}

TEST_F(LogMessageTest, StreamOperatorAcceptsVoidPointer) {
  XLS_LOG(INFO) << static_cast<void*>(nullptr);
  // See above.
  EXPECT_THAT(GetSingleEntry().text_message, AnyOf(Eq("(nil)"), "0"));
}

TEST_F(LogMessageTest, StreamOperatorAcceptsConstVoidPointer) {
  XLS_LOG(INFO) << static_cast<const void*>(nullptr);
  // See above.
  EXPECT_THAT(GetSingleEntry().text_message, AnyOf(Eq("(nil)"), "0"));
}

TEST_F(LogMessageTest, StreamOperatorAcceptsFloat) {
  XLS_LOG(INFO) << static_cast<float>(0.5);
  EXPECT_EQ(GetSingleEntry().text_message, "0.5");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsDouble) {
  XLS_LOG(INFO) << static_cast<double>(0.5);
  EXPECT_EQ(GetSingleEntry().text_message, "0.5");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsBool) {
  XLS_LOG(INFO) << static_cast<bool>(true);
  EXPECT_EQ(GetSingleEntry().text_message, "true");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsStreamManipulators) {
  XLS_LOG(INFO) << std::endl;
  EXPECT_EQ(GetSingleEntry().text_message, "\n");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsLiteralString) {
  XLS_LOG(INFO) << "hello there";
  EXPECT_EQ(GetSingleEntry().text_message, "hello there");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsNonConstCharArray) {
  char hey_array[] = {'h', 'e', 'y', '\0'};
  XLS_LOG(INFO) << hey_array;
  EXPECT_EQ(GetSingleEntry().text_message, "hey");
}

TEST_F(LogMessageTest, StreamOperatorAcceptsValueWithOstreamOverload) {
  XLS_LOG(INFO) << absl::NotFoundError("test_not_found");
  EXPECT_THAT(GetSingleEntry().text_message, HasSubstr("test_not_found"));
}

TEST_F(LogMessageTest, FailAborts) {
  // The output of abort() isn't standardized, so just make sure we have "main"
  // in the unwound stack (and saw SIGABRT).
  EXPECT_EXIT({ XLS_LOG(INFO).Fail(); },
              ::testing::KilledBySignal(SIGABRT), "main");
}

TEST_F(LogMessageTest, FailWithoutStackTraceAborts) {
  // The output of abort() isn't standardized, so just make sure we die by
  // abort.
  EXPECT_EXIT({ XLS_LOG(INFO).FailWithoutStackTrace(); },
              ::testing::KilledBySignal(SIGABRT), "");
}

TEST_F(LogMessageTest, FailQuietlyExits) {
  EXPECT_DEATH({ XLS_LOG(INFO).FailQuietly(); }, Not(HasSubstr("SIGABRT")));
}

#line 1336 "/foo/fake.cc"
const xabsl::SourceLocation Locs::kFakeLocation =
    xabsl::SourceLocation::current();

}  // namespace
}  // namespace logging_internal
}  // namespace xls
