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

#include "xls/common/logging/log_entry.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/log_severity.h"

namespace xls {
namespace logging_internal {
namespace {

using ::testing::HasSubstr;

LogEntry AnEntry() {
  absl::TimeZone tz;
  absl::CivilSecond cs(2017, 1, 2, 3, 4, 5);
  absl::Time time = absl::FromCivil(cs, tz);
  return LogEntry("abc/full_filename.cc", 42, absl::LogSeverity::kWarning,
                  time);
}

TEST(LogEntryTest, ConstructorInitializesFields) {
  absl::TimeZone tz;
  absl::CivilSecond cs(2017, 1, 2, 3, 4, 5);
  absl::Time time = absl::FromCivil(cs, tz);
  auto entry =
      LogEntry("abc/full_filename.cc", 42, absl::LogSeverity::kWarning, time);

  EXPECT_EQ(entry.source_filename(), "abc/full_filename.cc");
  EXPECT_EQ(entry.source_basename(), "full_filename.cc");
  EXPECT_EQ(entry.source_line(), 42);
  EXPECT_EQ(entry.log_severity(), absl::LogSeverity::kWarning);
  EXPECT_EQ(entry.timestamp(), time);
  EXPECT_TRUE(entry.prefix());  // Should be true by default.
}

TEST(LogEntryTest, SetSourceFilenameUpdatesFilenameAndBasename) {
  auto entry = AnEntry();

  entry.set_source_filename("hey/there.cc");
  EXPECT_EQ(entry.source_filename(), "hey/there.cc");
  EXPECT_EQ(entry.source_basename(), "there.cc");
}

TEST(LogEntryTest, SetSourceLineSetsTheField) {
  auto entry = AnEntry();

  entry.set_source_line(1234);
  EXPECT_EQ(entry.source_line(), 1234);
}

TEST(LogEntryTest, SetPrefixSetsField) {
  auto entry = AnEntry();

  entry.set_prefix(false);
  EXPECT_FALSE(entry.prefix());
  entry.set_prefix(true);
  EXPECT_TRUE(entry.prefix());
}

TEST(LogEntryTest, SetLogSeveritySetsField) {
  auto entry = AnEntry();
  EXPECT_NE(entry.log_severity(), absl::LogSeverity::kError);

  entry.set_log_severity(absl::LogSeverity::kError);

  EXPECT_EQ(entry.log_severity(), absl::LogSeverity::kError);
}

TEST(LogEntryTest, SetVerbositySetsField) {
  auto entry = AnEntry();

  entry.set_verbosity(1243);
  EXPECT_EQ(entry.verbosity(), 1243);
}

TEST(LogEntryTest, SetTimestampSetsField) {
  absl::TimeZone tz;
  absl::CivilSecond cs(2019, 5, 2, 1, 4, 5);
  absl::Time time = absl::FromCivil(cs, tz);

  auto entry = AnEntry();

  entry.set_timestamp(time);
  EXPECT_EQ(entry.timestamp(), time);
}

TEST(LogEntryTest, ToStringIncludesMessage) {
  auto entry = AnEntry();
  entry.set_text_message("the_message");

  std::string string = entry.ToString();

  EXPECT_THAT(string, HasSubstr("the_message"));
}

}  // namespace
}  // namespace logging_internal
}  // namespace xls
