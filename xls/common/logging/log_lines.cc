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

#include <cstddef>
#include <string_view>

#include "absl/base/log_severity.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"  // NOLINT for ClippedSubstr()

namespace xls {
namespace logging {

void LogLines(absl::LogSeverity severity, std::string_view text,
              std::string_view file_name, int line_number) {
  // Since there are multiple lines, using FATAL will cause only
  // the first line to be printed, so downgrade to ERROR, but
  // remember that it is FATAL so that we abort after logging everything.
  const absl::LogSeverity original_severity = severity;
  if (severity == absl::LogSeverity::kFatal) {
    severity = absl::LogSeverity::kError;
  }

  // Print all consecutive substrings ending in a newline.
  size_t current_position = 0;
  size_t next_newline = text.find_first_of('\n');
  while ((current_position < text.size()) &&
         (next_newline != std::string_view::npos)) {
    size_t length = next_newline - current_position;
    LOG(LEVEL(severity)).AtLocation(file_name, line_number)
        << absl::ClippedSubstr(text, current_position, length);
    current_position = next_newline + 1;
    next_newline = text.find_first_of('\n', current_position);
  }

  // In case the message does not end in a newline, print
  // whatever text is left over.
  if (current_position < text.size()) {
    LOG(LEVEL(severity)).AtLocation(file_name, line_number)
        << absl::ClippedSubstr(text, current_position);
  }

  // Respect FATAL
  if (original_severity == absl::LogSeverity::kFatal) {
    LOG(FATAL) << "Aborting due to previous errors.";
  }
}

}  // namespace logging
}  // namespace xls
