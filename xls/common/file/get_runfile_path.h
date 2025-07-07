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

#ifndef XLS_COMMON_FILE_GET_RUNFILE_PATH_H_
#define XLS_COMMON_FILE_GET_RUNFILE_PATH_H_

#include <filesystem>
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xls {

// Return the path to a file in the Bazel runfiles directory. `path` is
// interpreted as relative to the root XLS source directory. `path` must refer
// to a specific file, looking up directories is not supported.
//
// If the file does not exist as a runfile, this method may return an empty
// path.
absl::StatusOr<std::filesystem::path> GetXlsRunfilePath(
    const std::filesystem::path& path, std::string package = "com_google_xls");

// Called by InitXls; don't call this directly. Sets up global state for the
// other functions in this file.
absl::Status InitRunfilesDir(const std::string& argv0);

}  // namespace xls

#endif  // XLS_COMMON_FILE_GET_RUNFILE_PATH_H_
