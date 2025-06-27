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

#include "xls/common/file/file_descriptor.h"

#include <cerrno>
#include <cstdio>
#include <filesystem>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "xls/common/status/error_code_to_status.h"

namespace xls {

FileStream::~FileStream() {
  if (file_ != nullptr) {
    fclose(file_);
  }
}

/* static */ absl::StatusOr<FileStream> FileStream::Open(
    const std::filesystem::path& path, const std::string& mode) {
  FILE* f = fopen(path.c_str(), mode.c_str());
  if (f == nullptr) {
    return ErrnoToStatus(errno);
  }
  return FileStream(path, f);
}

FILE* FileStream::Release() && {
  FILE* result = file_;
  path_.clear();
  file_ = nullptr;
  return result;
}

FileStream::FileStream(FileStream&& other)
    : path_(std::move(other.path_)), file_(other.file_) {
  other.path_.clear();
  other.file_ = nullptr;
}

}  // namespace xls
