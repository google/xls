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

#include <stdio.h>  // NOLINT for fileno
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"
#include "re2/re2.h"

namespace xls::dslx {
namespace {

struct ColorSpec {
  std::string_view pos_color_leader;
  std::string_view msg_color_leader;
  std::string_view color_reset;
};

absl::Status PrintSingleSpanError(const Span& error_span,
                                  std::string_view msg_to_print,
                                  std::ostream& os, const FileTable& file_table,
                                  VirtualizableFilesystem& vfs,
                                  int64_t error_context_line_count,
                                  const ColorSpec& color_spec) {
  XLS_ASSIGN_OR_RETURN(std::string contents,
                       vfs.GetFileContents(error_span.GetFilename(file_table)));
  std::vector<std::string_view> lines = absl::StrSplit(contents, '\n');
  // Lines are \n-terminated, not \n-separated.
  if (lines.size() > 1 && lines.back().empty()) {
    lines.resize(lines.size() - 1);
  }
  XLS_RET_CHECK(!lines.empty());
  const Pos file_start(error_span.fileno(), 0, 0);

  // file_limit as a whole is an exclusive limit (the first not-included
  // character), but its file_limit.lineno() is inclusive (addresses the
  // last line, not the one after).
  const Pos file_limit(error_span.fileno(), lines.size() - 1,
                       lines.back().size());

  // Caps the effective start and limit against the file start and limit,
  // and caps the limit against the start.
  const Pos start =
      std::max(file_start, std::min(error_span.start(), file_limit));
  const Pos limit = std::max(start, std::min(file_limit, error_span.limit()));

  int64_t line_count_each_side = error_context_line_count / 2;
  int64_t first_line_printed =
      std::max(start.lineno() - line_count_each_side, int64_t{0});
  int64_t last_line_printed =
      std::min(limit.lineno() + line_count_each_side, file_limit.lineno());

  bool is_multiline = limit.lineno() > start.lineno();

  // When emitting multiline errors, we put a leading bar at the start of the
  // line to show which are in the error range. When we're not doing multiline
  // errors, this is just empty.
  const std::string bar_on =
      absl::StrCat(color_spec.msg_color_leader, " |", color_spec.color_reset);
  const std::string bar_off = (is_multiline ? "  " : "");
  std::string_view bar = bar_off;

  // Emit an indicator of what we're displaying.
  os << absl::StreamFormat("%s:%s-%s\n", error_span.GetFilename(file_table),
                           error_span.start().ToStringNoFile(),
                           error_span.limit().ToStringNoFile());

  for (int64_t i = first_line_printed; i <= last_line_printed; ++i) {
    os << absl::StreamFormat("%s%04d:%s%s %s\n", color_spec.pos_color_leader,
                             i + 1, color_spec.color_reset, bar, lines[i]);
    if (i == start.lineno()) {
      // Emit arrow pointing to all of, or the start of, the error.
      if (is_multiline) {
        std::string spaces(std::string_view("0000: |").size(), ' ');
        std::string underscores(std::max(int64_t{0}, start.colno()) + 1, '_');
        os << absl::StreamFormat("%s%s%s^%s\n", color_spec.msg_color_leader,
                                 spaces, underscores, color_spec.color_reset);
        // Each line will also draw a little bit of a vertical bar
        // connecting the start and the end of the multi-line error
        // to the left.
        bar = bar_on;
      } else {
        std::string squiggles(std::max(int64_t{0}, start.colno()) + 6, '~');
        int64_t width = std::max(int64_t{0}, limit.colno() - start.colno() - 1);
        std::string dashes_and_arrow;
        if (width > 0) {
          dashes_and_arrow = std::string(width - 1, '-') + "^";
        }
        std::string msg_with_space =
            msg_to_print.empty() ? "" : absl::StrCat(" ", msg_to_print);
        os << absl::StreamFormat("%s%s^%s%s%s\n", color_spec.msg_color_leader,
                                 squiggles, dashes_and_arrow, msg_with_space,
                                 color_spec.color_reset);
      }
    } else if (i == limit.lineno()) {
      // Emit arrow pointing to the end of the multi-line error.
      std::string spaces(std::string_view("0000: ").size(), ' ');
      std::string underscores(std::max(int64_t{0}, limit.colno()), '_');
      os << absl::StreamFormat("%s%s|%s^ %s%s\n", color_spec.msg_color_leader,
                               spaces, underscores, msg_to_print,
                               color_spec.color_reset);
      // We're done drawing the multiline arrows; put down the crayon.
      bar = bar_off;
    }
  }
  return absl::OkStatus();
}
}  // namespace

absl::Status PrintPositionalError(absl::Span<const Span> error_spans,
                                  std::string_view error_message,
                                  std::ostream& os, PositionalErrorColor color,
                                  const FileTable& file_table,
                                  VirtualizableFilesystem& vfs,
                                  int64_t error_context_line_count) {
  XLS_RET_CHECK_EQ(error_context_line_count % 2, 1);

  // Strip ANSI escape codes from the error message opportunistically if we know
  // we shouldn't be emitting colors.
  //
  // This is a bit of a layering violation but makes life easier as it allows
  // "downstream" error message creation to insert ANSI codes without needing to
  // be told explicitly if color error messages are ok.
  std::string msg{error_message};
  if (!isatty(fileno(stderr)) || color == PositionalErrorColor::kNoColor) {
    RE2::GlobalReplace(&msg, "\33\\[\\d+m", "");
  }

  ColorSpec color_spec;
  if (isatty(fileno(stderr))) {
    switch (color) {
      case PositionalErrorColor::kNoColor:
        break;
      case PositionalErrorColor::kErrorColor:
        color_spec.pos_color_leader = "\e[1;33m";  // yellow
        color_spec.msg_color_leader = "\e[1;31m";  // red
        color_spec.color_reset = "\e[1;0m";        // reset
        break;
      case PositionalErrorColor::kWarningColor:
        color_spec.pos_color_leader = "\e[1;33m";            // yellow
        color_spec.msg_color_leader = "\e[38;2;255;140;0m";  // orange
        color_spec.color_reset = "\e[1;0m";                  // reset
        break;
    }
  }

  XLS_RET_CHECK(!error_spans.empty());
  bool first_span = true;
  for (const Span& error_span : error_spans) {
    std::string_view msg_to_print;
    if (first_span) {
      msg_to_print = msg;
      first_span = false;
    }
    XLS_RETURN_IF_ERROR(
        PrintSingleSpanError(error_span, msg_to_print, os, file_table, vfs,
                             error_context_line_count, color_spec));
  }
  return absl::OkStatus();
}

void PrintWarnings(const WarningCollector& warnings,
                   const FileTable& file_table, VirtualizableFilesystem& vfs) {
  for (const WarningCollector::Entry& e : warnings.warnings()) {
    absl::Status print_status = PrintPositionalError(
        {e.span}, e.message, std::cerr, PositionalErrorColor::kWarningColor,
        file_table, vfs);
    if (!print_status.ok()) {
      LOG(WARNING) << "Could not print warning: " << print_status;
    }
  }
}

}  // namespace xls::dslx
