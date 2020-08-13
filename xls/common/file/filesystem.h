// Copyright 2020 Google LLC
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

#ifndef XLS_COMMON_FILE_FILESYSTEM_H_
#define XLS_COMMON_FILE_FILESYSTEM_H_

#include <filesystem>

#include "google/protobuf/message.h"
#include "absl/status/status.h"
#include "xls/common/status/statusor.h"

namespace xls {

// Answers the question, "Does the named path entry exist?"
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk - The path entry definitely exists.
//  * StatusCode::kNotFound - The path entry definitely does not exist.
//  * StatusCode::kPermissionDenied - Insufficient permissions.
//
// Example:
//
//    // Ensures `filename` exists.
//    XLS_CHECK_OK(FileExists(filename));
//
//    // Ensures `filename` doesn't exist (and no other errors occurred).
//    XLS_CHECK(absl::IsNotFound(FileExists(filename)));
//
//    // To handle the possibility of an error:
//    absl::Status status = FileExists(filename);
//    if (status.ok()) {
//      // `filename` definitely exists.
//      ...
//    } else if (absl::IsNotFound(status)) {
//      // `filename` definitely does not exist.
//      ...
//    } else {
//      // `filename` may or may not exist; some error occurred.
//      ...
//    }
absl::Status FileExists(const std::filesystem::path& path);

// Recursively creates the directory path. Returns OK if the directory already
// exists.
absl::Status RecursivelyCreateDir(const std::filesystem::path& path);

// Recursively deletes the given path. Path can be a file or directory. Symlinks
// are not followed.
absl::Status RecursivelyDeletePath(const std::filesystem::path& path);

// Reads and returns the contents of the file `file_name`.
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kPermissionDenied (file not readable)
//  * StatusCode::kNotFound (no such file)
//  * StatusCode::kUnknown (a Read or Open error occurred)
xabsl::StatusOr<std::string> GetFileContents(
    const std::filesystem::path& file_name);

// Writes the data provided in `content` to the file `file_name`, overwriting
// any existing content. Fails if directory does not exist.
//
// NOTE: Will return OK iff all of the data in `content` was written.
// May write some of the data and return an error.
//
// WARNING: The file update is NOT guaranteed to be atomic.
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kPermissionDenied (file not writable)
//  * StatusCode::kUnknown (a Write or Open error occurred)
absl::Status SetFileContents(const std::filesystem::path& file_name,
                             absl::string_view content);

// Writes the contents of data into the file file_name, appending to any
// existing content.
//
// NOTE: Thoroughly consider whether this is the most appropriate API for your
// use.  Calling this multiple times (e.g., in a loop) results in multiple
// open/write/close cycles, often without reading the file-on-disk until the
// appends are complete.  Instead, consider building the data in memory and
// using SetFileContents().
//
// REQUIRES: `file_name` can be opened for writing (or created).
//
// NOTE: Will return OK iff all of the data in `content` was written.
// May write some of the data and return an error.
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kPermissionDenied (file not writable)
//  * StatusCode::kUnknown (a Write error occurred)
//  * StatusCode::kNotFound (an Open error occurred)
absl::Status AppendStringToFile(const std::filesystem::path& file_name,
                                absl::string_view content);

// Reads a single text formatted protobuf from a file.
//
// REQUIRES: `file_name` can be opened for reading.
// REQUIRES: The contents of `file_name` are a single text formatted protobuf.
// Typical return codes (not guaranteed to be exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kPermissionDenied (file not readable)
//  * StatusCode::kNotFound (no such file)
//  * StatusCode::kFailedPrecondition (the file contents couldn't be parsed as
//    text proto)
//
// *proto will hold the result of parsing the text of the file as
// protobuf only if OK is returned.  Regardless of success, the
// contents of that protobuf may be modified.
absl::Status ParseTextProtoFile(const std::filesystem::path& file_name,
                                google::protobuf::Message* proto);

// As above, except returns the proto in a StatusOr rather than an output
// parameter.
template <typename T>
inline xabsl::StatusOr<T> ParseTextProtoFile(
    const std::filesystem::path& file_name) {
  xabsl::StatusOr<T> v_or = T();
  absl::Status status = ParseTextProtoFile(file_name, &v_or.value());
  if (!status.ok()) {
    v_or = status;
  }
  return v_or;
}

// Writes the protobuf provided to the file `filename` in a protobuf text
// format, overwriting any existing content in the file.
//
// NOTE: Will return OK iff the protobuf could be converted to string and all of
// that data in was written.  May write some data and return an error.
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kPermissionDenied (file not writable)
//  * StatusCode::kFailedPrecondition (the proto couldn't be converted to
//    string)
absl::Status SetTextProtoFile(const std::filesystem::path& file_name,
                              const google::protobuf::Message& proto);

// Returns the process's current working directory.
xabsl::StatusOr<std::filesystem::path> GetCurrentDirectory();

// Lists the entries of a directory.
//
// The returned paths are relative to the current working directory if `path` is
// relative, and absolute if `path` is absolute.
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kNotFound (no such directory)
//  * StatusCode::kFailedPrecondition (`path` is not a directory)
//  * StatusCode::kPermissionDenied (directory is not readable)
xabsl::StatusOr<std::vector<std::filesystem::path>> GetDirectoryEntries(
    const std::filesystem::path& path);

// Returns the path pointed to by the file, if it's a link, or the given path
// otherwise.
xabsl::StatusOr<std::filesystem::path> GetRealPath(const std::string& path);

}  // namespace xls

#endif  // XLS_COMMON_FILE_FILESYSTEM_H_
