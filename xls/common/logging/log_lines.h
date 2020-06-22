// Copyright 2020 Google LLC
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

#include <string>
#include <vector>

#include "absl/base/log_severity.h"
#include "absl/strings/string_view.h"
#include "xls/common/logging/logging_internal.h"
#include "xls/common/logging/vlog_is_on.h"

namespace xls {
namespace logging {

// Splits up text into multiple lines and logs it with the given severity.
// The file name and line number provided are used in the logging message.
void LogLines(absl::LogSeverity severity, absl::string_view text,
              const char* file_name, int line_number);

}  // namespace logging
}  // namespace xls

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
#define XLS_LOG_LINES(SEVERITY, STRING)                                      \
  ::xls::logging::LogLines(XLS_LOGGING_INTERNAL_SEVERITY_##SEVERITY, STRING, \
                           __FILE__, __LINE__)

// Like XLS_LOG_LINES, but for VLOG.
// Example:
//   XLS_VLOG_LINES(3, some_proto->DebugString());
#define XLS_VLOG_LINES(LEVEL, STRING)                       \
  do {                                                      \
    if (XLS_VLOG_IS_ON(LEVEL)) XLS_LOG_LINES(INFO, STRING); \
  } while (false)

#endif  // XLS_COMMON_LOGGING_LOG_LINES_H_
