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

#ifndef XLS_COMMON_FILE_TEMP_FILE_H_
#define XLS_COMMON_FILE_TEMP_FILE_H_

#include <filesystem>  // NOLINT
#include <string_view>

#include "absl/status/statusor.h"

namespace xls {

// A temporary file which is removed when the object goes out of scope.
class TempFile {
 public:
  ~TempFile();

  // Create an empty temporary file in a temporary directory (not cwd). The name
  // of the temporary file will end with the given suffix.
  static absl::StatusOr<TempFile> Create(std::string_view suffix = "");
  // Create an empty temporary file in a specified directory. The name
  // of the temporary file will end with the given suffix.
  static absl::StatusOr<TempFile> CreateInDirectory(
      const std::filesystem::path& directory, std::string_view suffix = "");
  // Create a temporary file with the given contents in a temporary directory
  // (not cwd). The name of the temporary file will end with the given suffix.
  static absl::StatusOr<TempFile> CreateWithContent(
      std::string_view content, std::string_view suffix = "");
  // Create a temporary file with the given contents in a specified directory.
  // The name of the temporary file will end with the given suffix.
  static absl::StatusOr<TempFile> CreateWithContentInDirectory(
      std::string_view content, const std::filesystem::path& directory,
      std::string_view suffix = "");

  std::filesystem::path path() const;

  // Leave the file on the file system as is. This causes TempFile to not delete
  // the file when it goes out of scope. Returns the path to the temporary file.
  //
  // This method is marked with an rvalue reference qualifier (&&). This
  // indicates that the method consumes the object; after calling `Release` it's
  // not allowed to call other methods such as `path`.
  //
  // Example usage:
  //
  //   // Create a temporary directory.
  //   XLS_ASSIGN_OR_RETURN(TempFile file, TempFile::Create());
  //   // Release it so that it is not deleted when it goes out of scope.
  //   std::move(file).Release();
  std::filesystem::path Release() &&;

  // TempFile is movable but not copyable.
  TempFile(TempFile&& other);
  TempFile& operator=(TempFile&& other);
  TempFile(const TempFile&) = delete;
  TempFile& operator=(const TempFile&) = delete;

 private:
  explicit TempFile(const std::filesystem::path& path);
  void Cleanup();
  void Close();
  // Create an empty temporary file in a specified directory. `file_descriptor`
  // is an output parameter. If the function call succeeds, it is set to a file
  // descriptor that points to the temporary file. The caller is responsible for
  // closing it.
  static absl::StatusOr<TempFile> Create(const std::filesystem::path& directory,
                                         std::string_view suffix,
                                         int* file_descriptor);

  std::filesystem::path path_;
};

}  // namespace xls

#endif  // XLS_COMMON_FILE_TEMP_FILE_H_
