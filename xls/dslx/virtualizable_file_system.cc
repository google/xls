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

#include <filesystem>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
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

// -- FakeFilesystem

FakeFilesystem::FakeFilesystem(
    absl::flat_hash_map<std::filesystem::path, std::string> files,
    std::filesystem::path cwd)
    : files_(std::move(files)), cwd_(std::move(cwd)) {}

absl::Status FakeFilesystem::FileExists(const std::filesystem::path& path) {
  std::filesystem::path full_path = path.is_absolute() ? path : cwd_ / path;
  if (files_.contains(full_path)) {
    return absl::OkStatus();
  }
  return absl::NotFoundError("FakeFilesystem does not have file: " +
                             full_path.string());
}

absl::StatusOr<std::string> FakeFilesystem::GetFileContents(
    const std::filesystem::path& path) {
  std::filesystem::path full_path = path.is_absolute() ? path : cwd_ / path;
  if (files_.contains(full_path)) {
    return files_.at(full_path);
  }
  return absl::NotFoundError("FakeFilesystem does not have file: " +
                             full_path.string());
}

}  // namespace xls::dslx
