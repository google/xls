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

#ifndef XLS_COMMON_LOGGING_LOG_LINES_H_
#define XLS_COMMON_LOGGING_LOG_LINES_H_

#include <string_view>

#include "absl/base/log_severity.h"
#include "absl/log/log.h"

namespace xls {
namespace logging {

// Splits up text into multiple lines and logs it with the given severity.
// The file name and line number provided are used in the logging message.
void LogLines(absl::LogSeverity severity, std::string_view text,
              std::string_view file_name, int line_number);

}  // namespace logging
}  // namespace xls

#define XLS_LOG_INTERNAL_INFO ::absl::LogSeverity::kInfo
#define XLS_LOG_INTERNAL_WARNING ::absl::LogSeverity::kWarning
#define XLS_LOG_INTERNAL_ERROR ::absl::LogSeverity::kError
#define XLS_LOG_INTERNAL_FATAL ::absl::LogSeverity::kFatal
#define XLS_LOG_INTERNAL_DFATAL ::absl::kLogDebugFatal
#define XLS_LOG_INTERNAL_LEVEL(severity) ::absl::NormalizeLogSeverity(severity)

// If you're trying to output something longer than the log buffer size it'll
// get truncated. Here is a macro for logging a long string by breaking it up
// into multiple lines and logging each line separately. Nothing will be
// truncated (unless an individual line is longer than the log buffer size).
//
// WARNING: This macro is *not* intended to output large volumes of
// data into the log -- for that, you should probably write to
// separate files. It is mainly intended for outputting configuration
// information or processing results in more human-readable form
// (e.g., printing multi-line protocol buffers)
//
// Note that STRING is evaluated regardless of whether it will be logged.
#define XLS_LOG_LINES_LOC(SEVERITY, STRING, FILE, LINE) \
  ::xls::logging::LogLines(XLS_LOG_INTERNAL_##SEVERITY, STRING, FILE, LINE)
#define XLS_LOG_LINES(SEVERITY, STRING) \
  XLS_LOG_LINES_LOC(SEVERITY, STRING, __FILE__, __LINE__)

// Like XLS_LOG_LINES, but for VLOG.
// Example:
//   XLS_VLOG_LINES(3, some_proto->DebugString());
#define XLS_VLOG_LINES_LOC(LEVEL, STRING, FILE, LINE)                   \
  do {                                                                  \
    if (VLOG_IS_ON(LEVEL)) XLS_LOG_LINES_LOC(INFO, STRING, FILE, LINE); \
  } while (false)
#define XLS_VLOG_LINES(LEVEL, STRING) \
  XLS_VLOG_LINES_LOC(LEVEL, STRING, __FILE__, __LINE__)

#endif  // XLS_COMMON_LOGGING_LOG_LINES_H_
