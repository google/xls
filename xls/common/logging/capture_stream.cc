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

#include "xls/common/logging/capture_stream.h"

#include <fcntl.h>

#include <iostream>
#include <sstream>

#include "absl/status/statusor.h"
#include "xls/common/file/file_descriptor.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/error_code_to_status.h"
#include "xls/common/status/status_macros.h"

namespace xls {
namespace testing {

absl::StatusOr<std::string> CaptureStream(int fd, std::function<void()> fn) {
  XLS_ASSIGN_OR_RETURN(TempFile temp_file, TempFile::Create());

  auto uncaptured_fd = FileDescriptor(dup(fd));
  if (uncaptured_fd.get() == -1) {
    return ErrnoToStatus(errno).SetPrepend()
           << "Failed to duplicate file descriptor: ";
  }

  auto cap_fd = FileDescriptor(
      open(temp_file.path().c_str(), O_WRONLY, S_IRUSR | S_IWUSR));
  if (cap_fd.get() == -1) {
    return ErrnoToStatus(errno).SetPrepend()
           << "Failed to open temporary file for writing: ";
  }
  if (fflush(nullptr) != 0) {
    return ErrnoToStatus(errno).SetPrepend()
           << "Failed to flush stream before capture: ";
  }
  if (dup2(cap_fd.get(), fd) == -1) {
    return ErrnoToStatus(errno).SetPrepend() << "Failed to override stream: ";
  }
  if (close(cap_fd.get()) == -1) {
    return ErrnoToStatus(errno).SetPrepend()
           << "Failed to close temporary file file descriptor: ";
  }

  fn();

  if (fflush(nullptr) != 0) {
    return ErrnoToStatus(errno).SetPrepend()
           << "Failed to flush stream after capture: ";
  }
  if (dup2(uncaptured_fd.get(), fd) == -1) {
    return ErrnoToStatus(errno).SetPrepend()
           << "Failed to override stream back to original: ";
  }

  return GetFileContents(temp_file.path());
}

}  // namespace testing
}  // namespace xls
