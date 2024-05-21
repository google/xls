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

#include "xls/common/file/temp_file.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstdlib>
#include <filesystem>  // NOLINT
#include <string>
#include <string_view>
#include <system_error>  // NOLINT
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/common/status/status_macros.h"
#include "xls/common/strerror.h"

namespace xls {
namespace {

absl::Status WriteContent(int fd, std::string_view content) {
  int bytes_written = 0;
  while (bytes_written < content.size()) {
    int bytes_written_this_time =
        write(fd, static_cast<const void*>(content.data() + bytes_written),
              content.size() - bytes_written);
    if (bytes_written_this_time == -1) {
      if (errno == EINTR) {
        continue;
      }
      return absl::UnavailableError(absl::StrCat(
          "Failed to write content to temporary file: ", Strerror(errno)));
    }
    bytes_written += bytes_written_this_time;
  }
  return ::absl::OkStatus();
}

absl::StatusOr<std::filesystem::path> GetGlobalTemporaryDirectory() {
  std::error_code ec;
  auto temp_dir = std::filesystem::temp_directory_path(ec);
  if (ec) {
    return absl::UnavailableError("Failed to get temporary directory path.");
  }
  return temp_dir;
}

}  // namespace

TempFile::~TempFile() { Cleanup(); }

absl::StatusOr<TempFile> TempFile::Create(std::string_view suffix) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path temp_dir,
                       GetGlobalTemporaryDirectory());
  return CreateInDirectory(temp_dir, suffix);
}

absl::StatusOr<TempFile> TempFile::CreateInDirectory(
    const std::filesystem::path& directory, std::string_view suffix) {
  int fd;
  XLS_ASSIGN_OR_RETURN(TempFile temp_file, Create(directory, suffix, &fd));
  close(fd);
  return temp_file;
}

absl::StatusOr<TempFile> TempFile::CreateWithContent(std::string_view content,
                                                     std::string_view suffix) {
  XLS_ASSIGN_OR_RETURN(std::filesystem::path temp_dir,
                       GetGlobalTemporaryDirectory());
  return CreateWithContentInDirectory(content, temp_dir, suffix);
}

absl::StatusOr<TempFile> TempFile::CreateWithContentInDirectory(
    std::string_view content, const std::filesystem::path& directory,
    std::string_view suffix) {
  int fd;
  XLS_ASSIGN_OR_RETURN(TempFile temp_file, Create(directory, suffix, &fd));
  absl::Status write_status = WriteContent(fd, content);
  close(fd);
  if (!write_status.ok()) {
    return write_status;
  }
  return temp_file;
}

std::filesystem::path TempFile::path() const { return path_; }

std::filesystem::path TempFile::Release() && {
  std::filesystem::path result;
  std::swap(result, path_);
  return result;
}

TempFile::TempFile(TempFile&& other) : path_(std::move(other.path_)) {
  other.path_.clear();
}

TempFile& TempFile::operator=(TempFile&& other) {
  Cleanup();
  path_ = std::move(other.path_);
  other.path_.clear();
  return *this;
}

TempFile::TempFile(const std::filesystem::path& path) : path_(path) {}

void TempFile::Cleanup() {
  if (!path_.empty()) {
    if (unlink(path_.c_str()) != 0) {
      LOG(ERROR) << "Failed to delete temporary file " << path_;
    }
  }
}

absl::StatusOr<TempFile> TempFile::Create(
    const std::filesystem::path& directory, std::string_view suffix,
    int* file_descriptor) {
  std::string path_template =
      (directory / absl::StrCat("xls_tempfile_XXXXXX", suffix)).string();
  *file_descriptor = suffix.empty() ? mkostemp(path_template.data(), O_CLOEXEC)
                                    : mkostemps(path_template.data(),
                                                suffix.size(), O_CLOEXEC);
  if (*file_descriptor == -1) {
    return absl::UnavailableError(
        absl::StrCat("Failed to create temporary file ", path_template, ": ",
                     Strerror(errno)));
  }
  return TempFile(path_template);
}

}  // namespace xls
