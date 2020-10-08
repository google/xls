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

#ifndef XLS_COMMON_FILE_FILE_DESCRIPTOR_H_
#define XLS_COMMON_FILE_FILE_DESCRIPTOR_H_

#include <unistd.h>

namespace xls {

// RAII wrapper for a file descriptor. Users should use the FileDescriptor type
// alias; this class is a template only for testing purposes.
//
// `CloseFn` is a template parameter to avoid unnecessary overhead compared to
// taking a full functor object. It's only marginally more painful to test the
// class this way.
template <int (*CloseFn)(int fd)>
class BasicFileDescriptor {
 public:
  BasicFileDescriptor() = default;
  explicit BasicFileDescriptor(int fd) : fd_(fd) {}

  ~BasicFileDescriptor() { Close(); }

  // File descriptors are inherently not copyable.
  BasicFileDescriptor(const BasicFileDescriptor&) = delete;
  BasicFileDescriptor& operator=(const BasicFileDescriptor&) = delete;

  BasicFileDescriptor(BasicFileDescriptor&& other) : fd_(other.fd_) {
    other.fd_ = -1;
  }
  BasicFileDescriptor& operator=(BasicFileDescriptor&& other) {
    Close();
    fd_ = other.fd_;
    other.fd_ = -1;
    return *this;
  }

  int get() const { return fd_; }

  void Close() {
    if (fd_ != -1) {
      CloseFn(fd_);
      fd_ = -1;
    }
  }

 private:
  int fd_ = -1;
};

using FileDescriptor = BasicFileDescriptor<close>;

}  // namespace xls

#endif  // XLS_COMMON_FILE_FILE_DESCRIPTOR_H_
