// Copyright 2022 The XLS Authors
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

#include "xls/ir/caret.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

namespace xls {

namespace {

std::string PrintWordWrappedWithLinePrefix(int64_t width,
                                           std::string_view prefix,
                                           std::string_view to_justify) {
  std::vector<std::string> words = absl::StrSplit(to_justify, ' ');
  int64_t i = 0;
  std::string result;
  while (i < words.size()) {
    std::string line(prefix.begin(), prefix.end());
    do {
      absl::StrAppend(&line, words[i], " ");
      ++i;
    } while ((line.size() <= width) && (i < words.size()));
    if (line.back() == ' ') {
      line.pop_back();
    }
    absl::StrAppend(&result, line, "\n");
  }
  return result;
}

absl::StatusOr<std::string> GetLineFromFile(std::string_view path,
                                            int64_t line_number) {
  // It may be good to optimize this further when this gets used more widely.
  XLS_ASSIGN_OR_RETURN(std::string file_contents, GetFileContents(path));
  std::vector<std::string> lines = absl::StrSplit(file_contents, '\n');
  XLS_RET_CHECK_LT(line_number, lines.size());
  return lines[line_number];
}

}  // namespace

std::string PrintCaret(
    std::function<std::optional<std::string>(Fileno)> fileno_to_path,
    const SourceLocation& loc, std::optional<std::string_view> line_contents,
    std::optional<std::string_view> comment, int64_t terminal_width) {
  int32_t line = static_cast<int32_t>(loc.lineno());
  int32_t col = static_cast<int32_t>(loc.colno());

  std::optional<std::string> path_maybe = fileno_to_path(loc.fileno());

  std::string unknown_line_contents = "«unknown line contents»";

  std::string line_contents_owned;
  if (line_contents.has_value()) {
    line_contents_owned = line_contents.value();
  } else if (path_maybe.has_value()) {
    int32_t lineZeroBased = std::max(0, line - 1);
    absl::StatusOr<std::string> line_or_status =
        GetLineFromFile(path_maybe.value(), lineZeroBased);
    line_contents_owned =
        line_or_status.ok() ? *line_or_status : unknown_line_contents;
  } else {
    line_contents_owned = unknown_line_contents;
  }

  int64_t line_number_width = absl::StrFormat("%d", line).size();
  std::string line_number_padding;
  line_number_padding.resize(line_number_width, ' ');
  std::string caret_padding;
  int32_t colZeroBased = std::max(0, col - 1);
  caret_padding.resize(colZeroBased, ' ');

  std::string path = path_maybe.has_value() ? path_maybe.value() : "«unknown»";

  // Remove overhanging text from `line_contents_owned`.
  if (line_contents_owned.size() + line_number_width + 3 >= terminal_width) {
    int64_t slack =
        terminal_width - (line_contents_owned.size() + line_number_width + 3);
    int64_t size =
        std::max(static_cast<int64_t>(0),
                 static_cast<int64_t>(line_contents_owned.size()) + slack);
    line_contents_owned = line_contents_owned.substr(0, size - 1);
    absl::StrAppend(&line_contents_owned, "…");
  }

  std::string result;
  absl::StrAppend(&result, line_number_padding, "--> ", path, ":", line, ":",
                  col, "\n", line_number_padding, " |\n", line, " | ",
                  line_contents_owned, "\n", line_number_padding, " | ",
                  caret_padding, "^\n");

  if (comment.has_value()) {
    std::string prefix =
        absl::StrCat(line_number_padding, " | ", caret_padding);
    absl::StrAppend(&result, prefix, "|\n");
    // This shifts the start of the comment to the left if it comes too close
    // to the width of the terminal.
    int64_t slack = terminal_width - prefix.size();
    if (slack < 30) {
      int64_t size = std::max(static_cast<int64_t>(0),
                              static_cast<int64_t>(prefix.size()) + slack);
      prefix = prefix.substr(0, size);
    }
    absl::StrAppend(&result, PrintWordWrappedWithLinePrefix(
                                 terminal_width, prefix, comment.value()));
  }

  return result;
}

}  // namespace xls
