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

// Note: for changing permissions of files within a filesystem, use
// std::filesystem::permissions directly.

#ifndef XLS_COMMON_FILE_FILESYSTEM_H_
#define XLS_COMMON_FILE_FILESYSTEM_H_

#include <filesystem>  // NOLINT
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "google/protobuf/message.h"
#include "xls/common/status/status_macros.h"

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
//    CHECK_OK(FileExists(filename));
//
//    // Ensures `filename` doesn't exist (and no other errors occurred).
//    CHECK(absl::IsNotFound(FileExists(filename)));
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

// Returns true if any of the executable bits (owner, group, or other) are set
// to true. The may not exactly correspond to whether the file is executable
// because the user may not have sufficient permissions (e.g., not an owner) to
// take advantage of the set executable bit.
absl::StatusOr<bool> FileIsExecutable(const std::filesystem::path& path);

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
absl::StatusOr<std::string> GetFileContents(
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
                             std::string_view content);

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
                                std::string_view content);

// Parses a single text formatted protobuf from the given string which is
// assumed to have come from the given file.
//
// REQUIRES: `contents` is a single text formatted protobuf.
// REQUIRES: The proto must point to a valid object.
// Typical return codes (not guaranteed to be exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kFailedPrecondition (the file contents couldn't be parsed as
//    text proto)
//
// *proto will hold the result of parsing the given string of the file as
// protobuf only if OK is returned.  Regardless of success, the
// contents of that protobuf may be modified.
absl::Status ParseTextProto(std::string_view contents,
                            const std::filesystem::path& file_name,
                            google::protobuf::Message* proto);

// Reads a single text formatted protobuf from a file.
//
// REQUIRES: `file_name` can be opened for reading.
// REQUIRES: The contents of `file_name` are a single text formatted protobuf.
// REQUIRES: The proto must point to a valid object.
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
inline absl::StatusOr<T> ParseTextProtoFile(
    const std::filesystem::path& file_name) {
  T proto_message;
  XLS_RETURN_IF_ERROR(ParseTextProtoFile(file_name, &proto_message));
  return proto_message;
}

// Parses a single binary protobuf from the given string which is assumed to
// have come from the given file.
//
// REQUIRES: `contents` is a single binary protobuf.
// REQUIRES: The proto must point to a valid object.
// Typical return codes (not guaranteed to be exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kFailedPrecondition (the file contents couldn't be parsed as
//    binary protobuf)
//
// *proto will hold the result of parsing the given string of the file as
// protobuf only if OK is returned.  Regardless of success, the
// contents of that protobuf may be modified.
absl::Status ParseProtobin(std::string_view contents,
                           const std::filesystem::path& file_name,
                           google::protobuf::Message* proto);

// Reads a file containing a single protobuf in binary format.
//
// REQUIRES: `file_name` can be opened for reading.
// REQUIRES: The content of `file_name` is a single protobuf in binary format.
// REQUIRES: The proto must point to a valid object.
// Typical return codes (not guaranteed to be exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kPermissionDenied (file not readable)
//  * StatusCode::kNotFound (no such file)
//  * StatusCode::kFailedPrecondition (the file contents couldn't be parsed as
//    a protobuf binary)
//  * StatusCode::kFailedPrecondition (the proto pointer is invalid)
//
// *proto will hold the result of parsing the file as protobuf in binary
// format only if OK is returned.  Regardless of success, the contents of that
// protobuf may be modified.
absl::Status ParseProtobinFile(const std::filesystem::path& file_name,
                               google::protobuf::Message* proto);

// Writes the protobuf provided to the file `filename` in a protobuf binary
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
absl::Status SetProtobinFile(const std::filesystem::path& file_name,
                             const google::protobuf::Message& proto);

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
absl::StatusOr<std::filesystem::path> GetCurrentDirectory();

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
absl::StatusOr<std::vector<std::filesystem::path>> GetDirectoryEntries(
    const std::filesystem::path& path);

// Finds all files under a given path which match a regex pattern.
//
// Follows the same behaviour as `GetDirectoryEntries`, the returned paths are
// relative to the current working directory if `path` is relative, and absolute
// if `path` is absolute.
//
// Typical return codes (not guaranteed exhaustive):
//  * StatusCode::kOk
//  * StatusCode::kNotFound (no such directory)
//  * StatusCode::kFailedPrecondition (`path` is not a directory)
//  * StatusCode::kPermissionDenied (directory is not readable)
absl::StatusOr<std::vector<std::filesystem::path>> FindFilesMatchingRegex(
    const std::filesystem::path& path, const std::string& pattern);

// Returns the path pointed to by the file, if it's a link, or the given path
// otherwise.
absl::StatusOr<std::filesystem::path> GetRealPath(const std::string& path);

}  // namespace xls

#endif  // XLS_COMMON_FILE_FILESYSTEM_H_
