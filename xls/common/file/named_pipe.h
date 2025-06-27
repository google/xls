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

#ifndef XLS_COMMON_FILE_NAMED_PIPE_H_
#define XLS_COMMON_FILE_NAMED_PIPE_H_

#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/file/file_descriptor.h"

namespace xls {

// A simple utility classes for reading lines from a file at a particular
// path. Opens the file on object creation and closes the file on destruction.
class FileLineReader {
 public:
  static absl::StatusOr<FileLineReader> Create(
      const std::filesystem::path& path);

  // Reads and returns the line from the file not including the newline. If EOF
  // is reached before encountering '\n' then std::nullopt is returned.
  absl::StatusOr<std::optional<std::string>> ReadLine();

  // FileLineReader is movable but not copyable.
  FileLineReader(FileLineReader&& other) = default;
  FileLineReader& operator=(FileLineReader&& other) = default;

 private:
  explicit FileLineReader(FileStream file) : file_(std::move(file)) {}

  FileStream file_;
};

// A simple utility class for writing lines from a file at a particular
// path. Opens the file on object creation and closes the file on destruction.
class FileLineWriter {
 public:
  static absl::StatusOr<FileLineWriter> Create(
      const std::filesystem::path& path);

  // Writes a line to the file. `line` should not include a newline as a newline
  // is automatically added.
  absl::Status WriteLine(std::string_view line);

  // FileLineWriter is movable but not copyable.
  FileLineWriter(FileLineWriter&& other) = default;
  FileLineWriter& operator=(FileLineWriter&& other) = default;

 private:
  explicit FileLineWriter(FileStream file) : file_(std::move(file)) {}

  FileStream file_;
};

// A wrapper around a named pipe which removes itself when the object goes out
// of scope.
class NamedPipe {
 public:
  ~NamedPipe() { Cleanup(); }

  // Create a named pipe at the given path.
  static absl::StatusOr<NamedPipe> Create(const std::filesystem::path& path);

  // Convenience methods for opening the pipe in text mode for reading and
  // writing.
  absl::StatusOr<FileLineReader> OpenForReading();
  absl::StatusOr<FileLineWriter> OpenForWriting();

  const std::filesystem::path& path() const { return path_; }

  // Leave the named pipe in the file system as is. This causes NamedPipe to not
  // delete itself when it goes out of scope. Returns the path to the named
  // pipe.
  std::filesystem::path Release() &&;

  // NamedPipe is movable but not copyable.
  NamedPipe(NamedPipe&& other);
  NamedPipe& operator=(NamedPipe&& other);
  NamedPipe(const NamedPipe&) = delete;
  NamedPipe& operator=(const NamedPipe&) = delete;

 private:
  explicit NamedPipe(const std::filesystem::path& path) : path_(path) {}

  void Cleanup();

  std::filesystem::path path_;
};

}  // namespace xls

#endif  // XLS_COMMON_FILE_NAMED_PIPE_H_
