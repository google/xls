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

#include "xls/common/file/temp_directory.h"

#include <stdlib.h>  // NOLINT (needed for mkdtemp())

#include <filesystem>  // NOLINT
#include <string>
#include <system_error>  // NOLINT
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"

namespace xls {

TempDirectory::~TempDirectory() {
  absl::Status cleanup_status = std::move(*this).Cleanup();
  if (!cleanup_status.ok()) {
    LOG(ERROR) << "Failed to clean up temporary directory: " << cleanup_status;
  }
}

std::filesystem::path TempDirectory::Release() && {
  std::filesystem::path result;
  std::swap(result, path_);
  return result;
}

absl::Status TempDirectory::Cleanup() && {
  if (path_.empty()) {
    // This is an empty shell object, maybe it has been moved somewhere else.
    return absl::OkStatus();
  }
  std::error_code ec;
  std::filesystem::remove_all(path_, ec);
  if (ec) {
    return absl::InternalError(absl::StrCat(
        "Failed to recursively delete temporary directory ", path_.c_str()));
  }
  return absl::OkStatus();
}

absl::StatusOr<TempDirectory> TempDirectory::Create() {
  std::error_code ec;
  auto global_temp_dir = std::filesystem::temp_directory_path(ec);
  if (ec) {
    return absl::InternalError("Failed to get temporary directory path.");
  }

  std::string temp_dir = (global_temp_dir / "temp_directory_XXXXXX").string();
  if (mkdtemp(temp_dir.data()) == nullptr) {
    return absl::UnavailableError(
        absl::StrCat("Failed to create temporary directory ", temp_dir));
  }

  return TempDirectory(temp_dir);
}

const std::filesystem::path& TempDirectory::path() const { return path_; }

TempDirectory::TempDirectory(TempDirectory&& other)
    : path_(std::move(other.path_)) {
  other.path_ = std::filesystem::path();
}

TempDirectory& TempDirectory::operator=(TempDirectory&& other) {
  absl::Status cleanup_status = std::move(*this).Cleanup();
  if (!cleanup_status.ok()) {
    LOG(ERROR) << "Failed to clean up temporary directory: " << cleanup_status;
  }
  path_ = std::move(other.path_);
  other.path_ = std::filesystem::path();
  return *this;
}

TempDirectory::TempDirectory(std::filesystem::path&& path)
    : path_(std::move(path)) {}

}  // namespace xls
