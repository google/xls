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

#ifndef XLS_DSLX_ERROR_PRINTER_H_
#define XLS_DSLX_ERROR_PRINTER_H_

#include <ostream>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/dslx/pos.h"

namespace xls::dslx {

// Prints pretty message to output for a error with a position.
//
// ANSI color escapes are used when stderr appears to be a tty.
//
// Errors:
//   InvalidArgumentError: if the error_context_line_count is not odd (only odd
//    values can be symmetrical around the erroneous line).
//   A filesystem error if the error_filename cannot be opened to retrieve lines
//   from (for printing).
absl::Status PrintPositionalError(
    const Span& error_span, absl::string_view error_message, std::ostream& os,
    std::function<absl::StatusOr<std::string>(absl::string_view)>
        get_file_contents = nullptr,
    std::optional<bool> color = absl::nullopt,
    int64_t error_context_line_count = 5);

}  // namespace xls::dslx

#endif  // XLS_DSLX_ERROR_PRINTER_H_
