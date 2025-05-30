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

#ifndef XLS_DSLX_COMMAND_LINE_UTILS_H_
#define XLS_DSLX_COMMAND_LINE_UTILS_H_

#include <string>
#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/dslx/frontend/pos.h"
#include "xls/dslx/virtualizable_file_system.h"

namespace xls::dslx {

// Attempts to print the status as a positional error to the string with
// surrounding lines as context (a la `xls/dslx/error_printer.h`).
//
// If the error is printed to the screen successfully, true is returned. If it
// is not (i.e. because it does not have position information) false is
// returned, and the status should likely be propagated to the caller instead of
// squashed in some way.
bool TryPrintError(const absl::Status& status, FileTable& file_table,
                   VirtualizableFilesystem& vfs);

// Converts a path to a DSLX module into its corresponding module name; e.g.
//
//    "path/to/foo.x" => "foo"
//
// Returns an error status if a module name cannot be extracted from the given
// path.
absl::StatusOr<std::string> PathToName(std::string_view path);

// PathToName without canonicalization.
absl::StatusOr<std::string> RawNameFromPath(std::string_view path);

// Returns true if the file name would be unparsable as an XLS-ir identifier.
bool NameNeedsCanonicalization(std::string_view path);

}  // namespace xls::dslx

#endif  // XLS_DSLX_COMMAND_LINE_UTILS_H_
