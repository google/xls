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

#include <cstdint>
#include <functional>
#include <ostream>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

enum class PositionalErrorColor {
  kNoColor,
  kErrorColor,
  kWarningColor,
};

// Prints pretty message to output for a error with a position.
//
// Errors:
//   InvalidArgumentError: if the error_context_line_count is not odd (only odd
//    values can be symmetrical around the erroneous line).
//   Propagates errors from `get_file_contents` when attempting to retrieve the
//   contents of a file for printing.
absl::Status PrintPositionalError(
    const Span& error_span, std::string_view error_message, std::ostream& os,
    std::function<absl::StatusOr<std::string>(std::string_view)>
        get_file_contents,
    PositionalErrorColor color, int64_t error_context_line_count = 5);

// Prints warnings to stderr.
void PrintWarnings(const WarningCollector& warnings);

}  // namespace xls::dslx

#endif  // XLS_DSLX_ERROR_PRINTER_H_
