// Copyright 2024 The XLS Authors
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

#include "xls/dslx/virtualizable_file_system.h"

#include <filesystem>  // NOLINT
#include <string>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/filesystem.h"

namespace xls::dslx {

absl::Status RealFilesystem::FileExists(const std::filesystem::path& path) {
  return xls::FileExists(path);
}

absl::StatusOr<std::string> RealFilesystem::GetFileContents(
    const std::filesystem::path& path) {
  return xls::GetFileContents(path);
}

absl::StatusOr<std::filesystem::path> RealFilesystem::GetCurrentDirectory() {
  return xls::GetCurrentDirectory();
}

}  // namespace xls::dslx
