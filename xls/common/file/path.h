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

#ifndef XLS_COMMON_FILE_PATH_H_
#define XLS_COMMON_FILE_PATH_H_

#include <filesystem>  // NOLINT

#include "absl/status/statusor.h"

namespace xls {

// Return the path to `path` relative to `reference`.
//
// This is a thin wrapper around std::filesystem::relative with absl::Status
// error handling.
//
// Examples:
// * RelativizePath("/a/d", "/a") == "d"
// * RelativizePath("/a/d", "/a/b/c") == "../../d"
// * RelativizePath("/a/b/c", "/a/d") == "../b/c"
absl::StatusOr<std::filesystem::path> RelativizePath(
    const std::filesystem::path& path, const std::filesystem::path& reference);

}  // namespace xls

#endif  // XLS_COMMON_FILE_PATH_H_
