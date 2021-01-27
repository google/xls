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

#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"

namespace xls::dslx {

absl::Status PrintPositionalError(const Span& error_span,
                                  absl::string_view error_message,
                                  std::ostream& os, absl::optional<bool> color,
                                  int64 error_context_line_count) {
  XLS_RET_CHECK_EQ(error_context_line_count % 2, 1);

  XLS_ASSIGN_OR_RETURN(std::string contents,
                       GetFileContents(error_span.filename()));
  std::vector<absl::string_view> lines = absl::StrSplit(contents, '\n');

  int64 line_count_each_side = error_context_line_count / 2;
  int64 target_lineno = error_span.start().lineno();
  int64 low_lineno = std::max(target_lineno - line_count_each_side, int64{0});
  absl::Span<const absl::string_view> lines_before =
      absl::MakeSpan(lines).subspan(low_lineno, target_lineno - low_lineno);
  absl::string_view target_line = lines[error_span.start().lineno()];
  // Note: since this is a limit there's a trailing +1.
  absl::Span<const absl::string_view> lines_after =
      absl::MakeSpan(lines).subspan(target_lineno + 1, line_count_each_side);

  // Note: "color" is a tristate, nullopt means no fixed request.
  bool use_color;
  if (color.has_value()) {
    use_color = color.value();
  } else {
    use_color = isatty(fileno(stderr));
  }

  std::string yellow_color_leader;
  std::string red_color_leader;
  std::string color_reset;
  if (use_color) {
    // ANSI color code for yellow.
    yellow_color_leader = "\e[1;33m";
    red_color_leader = "\e[1;31m";
    // ANSI color code for reset.
    color_reset = "\e[1;0m";
  }

  auto emit_line = [&](int64 lineno, absl::string_view line,
                       bool is_culprit = false) {
    // Note: humans generally thing line i=0 is "line 1".
    absl::string_view sigil = is_culprit ? "*" : " ";
    os << absl::StreamFormat("%s%s %04d:%s %s\n", yellow_color_leader, sigil,
                             lineno + 1, color_reset, line);
  };

  // Emit an indicator of what we're displaying.
  os << absl::StreamFormat("%s:%s-%s\n", error_span.filename(),
                           error_span.start().ToStringNoFile(),
                           error_span.limit().ToStringNoFile());

  // Emit the lines that come before.
  for (int64 i = 0; i < lines_before.size(); ++i) {
    emit_line(low_lineno + i, lines_before[i]);
  }

  // Emit the culprit line.
  emit_line(error_span.start().lineno(), target_line, /*is_culprit=*/true);

  // Emit error indicator.
  std::string squiggles(error_span.start().colno() + 6, '~');
  int64 width = std::max(
      int64{1}, error_span.limit().colno() - error_span.start().colno() - 1);
  std::string dashes(width - 1, '-');
  os << absl::StreamFormat("%s  %s^%s^ %s%s\n", red_color_leader, squiggles,
                           dashes, error_message, color_reset);

  // Emit the lines that come after.
  for (int64 i = 0; i < lines_after.size(); ++i) {
    emit_line(error_span.start().lineno() + 1 + i, lines_after[i]);
  }

  return absl::OkStatus();
}

}  // namespace xls::dslx
