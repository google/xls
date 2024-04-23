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

#ifndef XLS_IR_CARET_H_
#define XLS_IR_CARET_H_

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

#include "xls/ir/source_location.h"

namespace xls {

// Produces a "caret message", which are used for compiler diagnostics.
// The messages produced by this function look like:
//
// ```
//    --> /foo/bar/baz.ir:461:20
//     |
// 461 | some code that was in this file at line 461
//     |                     ^
//     |                     |
//     |                     a comment on this
// ```
//
// There are a number of subtleties here:
//
// - If no comment is provided, the `|` line after the caret is not displayed.
// - If the caret is too close to the right edge of the terminal, the start of
//   the column will be pushed to the left.
// - If the line text is longer than the terminal width, it will be truncated
//   and the remainder will be replaced with `…`.
// - If the file path couldn't be found in the provided `Package*`, it will be
//   rendered as `«unknown»`.
// - If `line_contents` is `std::nullopt` and the file could not be read or the
//   file path couldn't be found in the provided `Package*` or the line number
//   does not exist in the file, the line contents will be rendered as
//   `«unknown line contents»`.
// - If `comment` is longer than the number of columns between the caret and the
//   right edge of the terminal, it will be split into words and broken up into
//   lines where the words add up to fewer columns than are needed. However, if
//   there is a word that is longer than the available width, it will end up
//   overhanging.
//
// The `std::function` provides a way to translate `Fileno` to paths.
// The `SourceLocation` gives the location of the source.
// `line_contents` should usually be `std::nullopt` except in tests; it provides
// a way to skip the automatic reading of the file path to extract the relevant
// line.
// `comment` allows you to attach a comment to the caret.
// `terminal_width` allows you to control the assumed terminal width for
// justification and overly long line termination.
std::string PrintCaret(
    std::function<std::optional<std::string>(Fileno)> fileno_to_path,
    const SourceLocation& loc,
    std::optional<std::string_view> line_contents = std::nullopt,
    std::optional<std::string_view> comment = std::nullopt,
    int64_t terminal_width = 80);

}  // namespace xls

#endif  // XLS_IR_CARET_H_
