// Copyright 2021 The XLS Authors
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

#include "xls/dslx/error_printer.h"

#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"

namespace xls::dslx {

absl::Status PrintPositionalError(
    const Span& error_span, std::string_view error_message, std::ostream& os,
    std::function<absl::StatusOr<std::string>(std::string_view)>
        get_file_contents,
    PositionalErrorColor color, int64_t error_context_line_count) {
  XLS_RET_CHECK_EQ(error_context_line_count % 2, 1);

  if (get_file_contents == nullptr) {
    get_file_contents = [](std::string_view path) {
      return GetFileContents(path);
    };
  }
  XLS_ASSIGN_OR_RETURN(std::string contents,
                       get_file_contents(error_span.filename()));
  std::vector<std::string_view> lines = absl::StrSplit(contents, '\n');

  int64_t line_count_each_side = error_context_line_count / 2;
  int64_t target_lineno = error_span.start().lineno();
  int64_t limit_lineno = error_span.limit().lineno();
  int64_t low_lineno =
      std::max(target_lineno - line_count_each_side, int64_t{0});
  absl::Span<const std::string_view> lines_before =
      absl::MakeSpan(lines).subspan(low_lineno, target_lineno - low_lineno);
  std::string_view target_line = lines[error_span.start().lineno()];

  int64_t limit_lineno_after = limit_lineno + line_count_each_side;
  absl::Span<const std::string_view> lines_after =
      absl::MakeSpan(lines).subspan(target_lineno + 1,
                                    limit_lineno_after - target_lineno);

  std::string pos_color_leader;
  std::string msg_color_leader;
  std::string color_reset;
  switch (color) {
    case PositionalErrorColor::kNoColor:
      break;
    case PositionalErrorColor::kErrorColor:
      pos_color_leader = "\e[1;33m";  // yellow
      msg_color_leader = "\e[1;31m";  // red
      color_reset = "\e[1;0m";        // reset
      break;
    case PositionalErrorColor::kWarningColor:
      pos_color_leader = "\e[1;33m";            // yellow
      msg_color_leader = "\e[38;2;255;140;0m";  // orange
      color_reset = "\e[1;0m";                  // reset
      break;
  }

  bool is_multiline = error_span.limit().lineno() > error_span.start().lineno();

  auto emit_line = [&](int64_t lineno, std::string_view line) {
    // When emitting multiline errors, we put a leading bar at the start of the
    // line to show which are in the error range. When we're not doing multiline
    // errors, this is just empty.
    std::string multiline_bar;
    if (is_multiline) {
      absl::StrAppend(&multiline_bar, msg_color_leader);
      if (error_span.start().lineno() < lineno &&
          lineno <= error_span.limit().lineno()) {
        absl::StrAppend(&multiline_bar, " |");
      } else {
        absl::StrAppend(&multiline_bar, "  ");
      }
      absl::StrAppend(&multiline_bar, color_reset);
    }

    os << absl::StreamFormat("%s%04d:%s%s %s\n", pos_color_leader, lineno + 1,
                             multiline_bar, color_reset, line);
  };

  // Emit an indicator of what we're displaying.
  os << absl::StreamFormat("%s:%s-%s\n", error_span.filename(),
                           error_span.start().ToStringNoFile(),
                           error_span.limit().ToStringNoFile());

  // Emit the lines that come before.
  for (int64_t i = 0; i < lines_before.size(); ++i) {
    emit_line(low_lineno + i, lines_before[i]);
  }

  // Emit the first "culprit" line.
  emit_line(error_span.start().lineno(), target_line);

  // Emit error indicator.
  if (is_multiline) {
    std::string spaces(std::string_view("0000: |").size(), ' ');
    std::string underscores(error_span.start().colno() + 1, '_');
    os << absl::StreamFormat("%s%s%s^%s\n", msg_color_leader, spaces,
                             underscores, color_reset);
  } else {
    std::string squiggles(error_span.start().colno() + 6, '~');
    int64_t width = std::max(int64_t{1}, error_span.limit().colno() -
                                             error_span.start().colno() - 1);
    std::string dashes(width - 1, '-');
    os << absl::StreamFormat("%s%s^%s^ %s%s\n", msg_color_leader, squiggles,
                             dashes, error_message, color_reset);
  }

  // Emit the lines that come after. In a multiline error these will have
  // leading bars.
  for (int64_t i = 0; i < lines_after.size(); ++i) {
    int64_t lineno = error_span.start().lineno() + 1 + i;
    emit_line(lineno, lines_after[i]);

    // For multiline errors we put an indicator at the "bottom" of the span to
    // say what the error was after the whole relevant region of text has been
    // displayed.
    if (is_multiline && lineno == error_span.limit().lineno()) {
      std::string spaces(std::string_view("0000: ").size(), ' ');
      std::string underscores(error_span.limit().colno(), '_');
      os << absl::StreamFormat("%s%s|%s^ %s%s\n", msg_color_leader, spaces,
                               underscores, error_message, color_reset);
    }
  }

  return absl::OkStatus();
}

void PrintWarnings(const WarningCollector& warnings) {
  PositionalErrorColor color = isatty(fileno(stderr)) == 1
                                   ? PositionalErrorColor::kWarningColor
                                   : PositionalErrorColor::kNoColor;
  for (const WarningCollector::Entry& e : warnings.warnings()) {
    absl::Status print_status = PrintPositionalError(
        e.span, e.message, std::cerr, /*get_file_contents=*/nullptr, color);
    if (!print_status.ok()) {
      XLS_LOG(WARNING) << "Could not print warning: " << print_status;
    }
  }
}

}  // namespace xls::dslx
