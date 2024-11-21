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
#include <ostream>
#include <string_view>

#include "absl/status/status.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/virtualizable_file_system.h"
#include "xls/dslx/warning_collector.h"

namespace xls::dslx {

enum class PositionalErrorColor : uint8_t {
  kNoColor,
  kErrorColor,
  kWarningColor,
};

// Prints pretty message to output for a error with a position.
//
// This primarily prints the context around the error and emits the error
// message with appropriate coloring (as requested by args).
//
// Args:
//  error_span: The span of text that the error message pertains to.
//  error_message: The message to display, attributed to the span.
//  os: The output stream to use in emitting the printed error.
//  color: Whether to print with a particular color; e.g. none/warning/error.
//  file_table: The file table to use for mapping paths to file numbers.
//  vfs: The virtualizable filesystem to use for retrieving file contents.
//  error_context_line_count: Lines to print above & below the error span --
//    this is appropriately clipped if it runs to start-of-file / end-of-file.
//
// Errors:
//   InvalidArgumentError: if the error_context_line_count is not odd (only odd
//    values can be symmetrical around the erroneous line).
//   Propagates errors from `get_file_contents` when attempting to retrieve the
//   contents of a file for printing.
absl::Status PrintPositionalError(const Span& error_span,
                                  std::string_view error_message,
                                  std::ostream& os, PositionalErrorColor color,
                                  const FileTable& file_table,
                                  VirtualizableFilesystem& vfs,
                                  int64_t error_context_line_count = 5);

// Prints warnings to stderr.
void PrintWarnings(const WarningCollector& warnings,
                   const FileTable& file_table, VirtualizableFilesystem& vfs);

}  // namespace xls::dslx

#endif  // XLS_DSLX_ERROR_PRINTER_H_
