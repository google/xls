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

#ifndef XLS_COMMON_FILE_TEMP_DIRECTORY_H_
#define XLS_COMMON_FILE_TEMP_DIRECTORY_H_

#include <filesystem>  // NOLINT

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xls {

// When a TempDirectory is created, a temporary directory is created where
// temporary files can be added by the owner of the object. When the
// TempDirectory is destroyed it will attempt to delete this directory and all
// of its contents.
class TempDirectory {
 public:
  ~TempDirectory();

  static absl::StatusOr<TempDirectory> Create();

  const std::filesystem::path& path() const;

  // Leave the directory on the file system as is. This causes TempDirectory to
  // not delete the file when it goes out of scope. Returns the path to the
  // temporary directory.
  //
  // This method is marked with an rvalue reference qualifier (&&). This
  // indicates that the method consumes the object; after calling `Release` it's
  // not allowed to call other methods such as `path`.
  //
  // Example usage:
  //
  //   // Create a temporary directory.
  //   XLS_ASSIGN_OR_RETURN(TempDirectory dir, TempDirectory::Create());
  //   // Release it so that it is not deleted when it goes out of scope.
  //   std::move(dir).Release();
  //
  // The rvalue reference qualifier is there to inform the user (and static
  // analysis tools) that the object is not usable after this method is called.
  std::filesystem::path Release() &&;

  // Deletes the temporary directory and its contents. Use this if you care
  // about if the cleanup succeeds or not.
  //
  // This method is marked with an rvalue reference qualifier (&&). This
  // indicates that the method consumes the object. For more info, see the doc
  // comment for `Release`.
  absl::Status Cleanup() &&;

  // TempDirectory is movable but not copyble.
  TempDirectory(TempDirectory&& other);
  TempDirectory& operator=(TempDirectory&& other);
  TempDirectory(const TempDirectory&) = delete;
  TempDirectory& operator=(const TempDirectory&) = delete;

 private:
  explicit TempDirectory(std::filesystem::path&& path);

  std::filesystem::path path_;
};

}  // namespace xls

#endif  // XLS_COMMON_FILE_TEMP_DIRECTORY_H_
