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

#include "xls/common/logging/logging_test_base.h"

#include "gmock/gmock.h"

namespace xls {
namespace logging_internal {
namespace testing {

using ::testing::IsEmpty;
using ::testing::Not;

CapturedLogEntry::CapturedLogEntry() {}

CapturedLogEntry::CapturedLogEntry(const ::xls::LogEntry& entry)
    : text_message(entry.text_message()),
      log_severity(entry.log_severity()),
      verbosity(entry.verbosity()),
      source_filename(entry.source_filename()),
      source_basename(entry.source_basename()),
      source_line(entry.source_line()),
      prefix(entry.prefix()) {}

LoggingTestBase::LoggingTestBase() { ::xls::AddLogSink(this); }

LoggingTestBase::~LoggingTestBase() { ::xls::RemoveLogSink(this); }

void LoggingTestBase::Send(const ::xls::LogEntry& entry) {
  entries_.emplace_back(entry);
}

CapturedLogEntry LoggingTestBase::GetSingleEntry() {
  EXPECT_THAT(entries_, Not(IsEmpty()));
  return entries_.empty() ? CapturedLogEntry() : entries_.front();
}

}  // namespace testing
}  // namespace logging_internal
}  // namespace xls
