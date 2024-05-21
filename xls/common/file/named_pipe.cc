// Copyright 2023 The XLS Authors
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

#include "xls/common/file/named_pipe.h"

#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/common/file/file_descriptor.h"
#include "xls/common/status/error_code_to_status.h"
#include "xls/common/status/status_macros.h"

namespace xls {

/* static */ absl::StatusOr<FileLineReader> FileLineReader::Create(
    const std::filesystem::path& path) {
  XLS_ASSIGN_OR_RETURN(FileStream fs, FileStream::Open(path, "r"));
  return FileLineReader(std::move(fs));
}

absl::StatusOr<std::optional<std::string>> FileLineReader::ReadLine() {
  std::string result;
  int c;
  while ((c = fgetc(file_.get())) != EOF) {
    if (static_cast<char>(c) == '\n') {
      VLOG(1) << "ReadLine: " << result;
      return result;
    }
    result += static_cast<char>(c);
  }
  if (ferror(file_.get())) {
    VLOG(1) << "Error reading from file";
    return absl::InternalError(absl::StrFormat(
        "Error reading line from file %s", file_.path().string()));
  }
  // At end-of-file.
  VLOG(1) << "ReadLine at EOF";
  return std::nullopt;
}

/* static */ absl::StatusOr<FileLineWriter> FileLineWriter::Create(
    const std::filesystem::path& path) {
  XLS_ASSIGN_OR_RETURN(FileStream fs, FileStream::Open(path, "w"));
  return FileLineWriter(std::move(fs));
}

absl::Status FileLineWriter::WriteLine(std::string_view line) {
  VLOG(1) << "WriteLine: " << line;
  if (!line.empty()) {
    size_t written = fwrite(line.data(), line.size(), 1, file_.get());
    if (written == 0) {
      return absl::InternalError("Error writing line");
    }
  }
  size_t newline_written = fwrite("\n", 1, 1, file_.get());
  if (newline_written == 0) {
    return absl::InternalError("Error writing newline");
  }
  return absl::OkStatus();
}

/* static */ absl::StatusOr<NamedPipe> NamedPipe::Create(
    const std::filesystem::path& path) {
  // Create with RW permissions for the user only.
  int ec = mkfifo(path.c_str(), S_IRUSR | S_IWUSR);
  if (ec != 0) {
    return ErrnoToStatus(errno);
  }
  return NamedPipe(path);
}

absl::StatusOr<FileLineWriter> NamedPipe::OpenForWriting() {
  return FileLineWriter::Create(path_);
}

absl::StatusOr<FileLineReader> NamedPipe::OpenForReading() {
  return FileLineReader::Create(path_);
}

std::filesystem::path NamedPipe::Release() && {
  std::filesystem::path result;
  std::swap(result, path_);
  return result;
}

NamedPipe::NamedPipe(NamedPipe&& other) : path_(std::move(other.path_)) {
  other.path_.clear();
}

void NamedPipe::Cleanup() {
  if (!path_.empty()) {
    if (unlink(path_.c_str()) != 0) {
      LOG(ERROR) << "Failed remove named pipe " << path_;
    }
  }
}

}  // namespace xls
