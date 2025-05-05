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

#include "xls/common/file/filesystem.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>  // NOLINT
#include <string>
#include <string_view>
#include <system_error>  // NOLINT
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/cord.h"
#include "absl/strings/str_cat.h"
#include "google/protobuf/descriptor.h"
#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/io/zero_copy_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "xls/common/file/temp_file.h"
#include "xls/common/status/error_code_to_status.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "re2/re2.h"

namespace xls {
namespace {

// Returns a Status error based on the errno value. The error message includes
// the filename.
absl::Status ErrNoToStatusWithFilename(int errno_value,
                                       const std::filesystem::path& file_name) {
  xabsl::StatusBuilder builder = ErrnoToStatus(errno);
  builder << file_name.string();
  return std::move(builder);
}

// For use in reading serialized protos from files.
class ParseTextProtoFileErrorCollector : public google::protobuf::io::ErrorCollector {
 public:
  explicit ParseTextProtoFileErrorCollector(
      const std::filesystem::path& file_name, const google::protobuf::Message& proto)
      : file_name_(file_name), proto_(proto) {}

  void RecordError(int line, google::protobuf::io::ColumnNumber column,
                   std::string_view message) final {
    status_.Update(absl::Status(
        absl::StatusCode::kFailedPrecondition,
        absl::StrCat("Failed to parse ", proto_.GetDescriptor()->name(),
                     " proto from text.  First failure is at line ", line,
                     " column ", column, " in file '", file_name_.string(),
                     "'.  Proto parser error:\n", message)));
  }

  absl::Status status() const { return status_; }

 private:
  absl::Status status_;
  const std::filesystem::path file_name_;
  const google::protobuf::Message& proto_;
};

enum class SetOrAppend { kSet, kAppend };

absl::Status SetFileContentsOrAppend(const std::filesystem::path& file_name,
                                     std::string_view content,
                                     SetOrAppend set_or_append) {
  // Use POSIX C APIs instead of C++ iostreams to avoid exceptions.
  int fd = open(file_name.c_str(),
                O_WRONLY | O_CREAT | O_CLOEXEC |
                    (set_or_append == SetOrAppend::kAppend ? O_APPEND : 0),
                0664);
  if (fd == -1) {
    XLS_RETURN_IF_ERROR(ErrNoToStatusWithFilename(errno, file_name));
    return absl::InternalError("errno returned but no status");
  }

  // Clear existing contents if not appending.
  if (set_or_append == SetOrAppend::kSet) {
    if (ftruncate(fd, 0) == -1) {
      // Continue anyway since some files like /dev/null can't be truncated.
      LOG(WARNING) << "Unable to truncate opened file " << file_name
                   << " due to " << strerror(errno);
    }
  }

  ssize_t written = 0;
  while (written < content.size()) {
    ssize_t n = write(fd, content.data() + written, content.size() - written);
    if (n < 0) {
      if (errno == EAGAIN) {
        continue;
      }
      close(fd);
      XLS_RETURN_IF_ERROR(ErrNoToStatusWithFilename(errno, file_name));
      return absl::InternalError("errno returned but no status");
    }
    written += n;
  }

  if (close(fd) != 0) {
    XLS_RETURN_IF_ERROR(ErrNoToStatusWithFilename(errno, file_name));
    return absl::InternalError("errno returned but no status");
  }
  return absl::OkStatus();
}

}  // anonymous namespace

absl::Status FileExists(const std::filesystem::path& path) {
  std::error_code ec;
  bool result = std::filesystem::exists(path, ec);
  if (ec) {
    return ErrorCodeToStatus(ec);
  }
  return result ? absl::OkStatus()
                : absl::NotFoundError("File does not exist.");
}

absl::StatusOr<bool> FileIsExecutable(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::file_status status = std::filesystem::status(path, ec);
  if (ec) {
    return ErrorCodeToStatus(ec);
  }
  return (status.permissions() & (std::filesystem::perms::owner_exec |
                                  std::filesystem::perms::group_exec |
                                  std::filesystem::perms::others_exec)) !=
         std::filesystem::perms::none;
}

absl::Status RecursivelyCreateDir(const std::filesystem::path& path) {
  std::error_code ec;
  std::filesystem::create_directories(path, ec);
  return ErrorCodeToStatus(ec);
}

absl::Status RecursivelyDeletePath(const std::filesystem::path& path) {
  std::error_code ec;
  if (!std::filesystem::remove_all(path, ec)) {
    return absl::NotFoundError("File or directory does not exist: " +
                               path.string());
  }

  if (ec) {
    return absl::InternalError(
        absl::StrCat("Failed to recursively delete path ", path.c_str()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::string> GetFileContents(
    const std::filesystem::path& file_name) {
  // Use POSIX C APIs instead of C++ iostreams to avoid exceptions.
  std::string result;

  int fd;
  if (file_name == "/dev/stdin" || file_name == "-") {
    fd = dup(STDIN_FILENO);  // dup standard input fd for portability:
                             // - /dev/stdin is not posix
                             // - avoid closing stdin's file descriptor
  } else {
    fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd == -1) {
      return ErrNoToStatusWithFilename(errno, file_name);
    }
  }

  char buf[4096];
  while (ssize_t n = read(fd, buf, sizeof(buf))) {
    if (n < 0) {
      if (errno == EAGAIN) {
        continue;
      }
      close(fd);
      return ErrNoToStatusWithFilename(errno, file_name);
    }
    result.append(buf, n);
  }

  if (close(fd) != 0) {
    return ErrNoToStatusWithFilename(errno, file_name);
  }
  return std::move(result);
}

absl::Status SetFileContents(const std::filesystem::path& file_name,
                             std::string_view content) {
  return SetFileContentsOrAppend(file_name, content, SetOrAppend::kSet);
}

absl::Status SetFileContentsAtomically(const std::filesystem::path& file_name,
                                       std::string_view content) {
  if (file_name == "/dev/null") {
    return absl::OkStatus();
  }
  XLS_ASSIGN_OR_RETURN(TempFile temp_file,
                       TempFile::CreateWithContent(content));
  std::error_code ec;
  std::filesystem::rename(std::move(temp_file).Release(), file_name, ec);
  return ErrorCodeToStatus(ec);
}

absl::Status AppendStringToFile(const std::filesystem::path& file_name,
                                std::string_view content) {
  return SetFileContentsOrAppend(file_name, content, SetOrAppend::kAppend);
}

absl::Status ParseTextProto(std::string_view contents,
                            const std::filesystem::path& file_name,
                            google::protobuf::Message* proto) {
  if (proto == nullptr) {
    return absl::FailedPreconditionError("Invalid pointer value.");
  }
  ParseTextProtoFileErrorCollector collector(file_name, *proto);
  google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&collector);

  const bool success = parser.ParseFromString(contents, proto);
  DCHECK_EQ(success, collector.status().ok());
  return collector.status();
}

absl::Status ParseTextProtoFile(const std::filesystem::path& file_name,
                                google::protobuf::Message* proto) {
  XLS_ASSIGN_OR_RETURN(std::string text_proto, GetFileContents(file_name));
  return ParseTextProto(text_proto, file_name, proto);
}

absl::Status ParseProtobin(std::string_view contents,
                           const std::filesystem::path& file_name,
                           google::protobuf::Message* proto) {
  if (proto == nullptr) {
    return absl::FailedPreconditionError("Invalid pointer value.");
  }
  if (!proto->ParseFromString(contents)) {
    return absl::FailedPreconditionError("Error with parsing file: " +
                                         file_name.string());
  }
  return absl::OkStatus();
}

absl::Status ParseProtobinFile(const std::filesystem::path& file_name,
                               google::protobuf::Message* proto) {
  XLS_ASSIGN_OR_RETURN(std::string protobin, GetFileContents(file_name));
  return ParseProtobin(protobin, file_name, proto);
}

absl::Status SetProtobinFile(const std::filesystem::path& file_name,
                             const google::protobuf::Message& proto) {
  std::string bin_proto;
  if (!proto.IsInitialized()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot serialize proto, missing required field ",
                     proto.InitializationErrorString()));
  }

  if (!proto.SerializeToString(&bin_proto)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Failed to convert proto to protobin for saving to ",
        file_name.string(),
        " (this generally stems from massive protobufs that either exhaust "
        "memory or overflow a 32-bit buffer somewhere)."));
  }
  return SetFileContents(file_name, bin_proto);
}

absl::Status PrintTextProtoToStream(
    const google::protobuf::Message& proto,
    google::protobuf::io::ZeroCopyOutputStream* output_stream) {
  if (!proto.IsInitialized()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot serialize proto, missing required field ",
                     proto.InitializationErrorString()));
  }

  const google::protobuf::Descriptor* const descriptor = proto.GetDescriptor();
  output_stream->WriteCord(absl::Cord(
      absl::StrCat("# proto-file: ", descriptor->file()->name(),
                   "\n"
                   "# proto-message: ",
                   descriptor->containing_type() ? descriptor->full_name()
                                                 : descriptor->name(),
                   "\n\n")));

  google::protobuf::TextFormat::Printer printer = google::protobuf::TextFormat::Printer();
  if (!printer.Print(proto, output_stream)) {
    return absl::FailedPreconditionError(
        absl::StrCat("Failed to convert proto to text & save to stream (this "
                     "generally stems from massive protobufs that either "
                     "exhaust memory or overflow a 32-bit buffer somewhere)."));
  }
  return absl::OkStatus();
}

absl::Status SetTextProtoFile(const std::filesystem::path& file_name,
                              const google::protobuf::Message& proto) {
  // Use POSIX C APIs instead of C++ iostreams to avoid exceptions.
  int fd = open(file_name.c_str(), O_WRONLY | O_CREAT | O_CLOEXEC, 0664);
  if (fd == -1) {
    return ErrNoToStatusWithFilename(errno, file_name);
  }

  // Clear existing contents.
  if (ftruncate(fd, 0) == -1) {
    return ErrNoToStatusWithFilename(errno, file_name);
  }

  google::protobuf::io::FileOutputStream output_stream(fd);
  XLS_RETURN_IF_ERROR(PrintTextProtoToStream(proto, &output_stream));
  if (!output_stream.Close()) {
    return ErrNoToStatusWithFilename(errno, file_name);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::filesystem::path> GetCurrentDirectory() {
  std::error_code ec;
  std::filesystem::path path = std::filesystem::current_path(ec);
  if (ec) {
    return ErrorCodeToStatus(ec);
  }
  return path;
}

absl::StatusOr<std::vector<std::filesystem::path>> GetDirectoryEntries(
    const std::filesystem::path& path) {
  std::vector<std::filesystem::path> result;
  std::error_code ec;
  auto it = std::filesystem::directory_iterator(path, ec);
  auto end = std::filesystem::directory_iterator();
  if (ec) {
    return ErrorCodeToStatus(ec);
  }
  while (it != end) {
    result.push_back(it->path());
    it.increment(ec);
    if (ec) {
      return ErrorCodeToStatus(ec);
    }
  }
  return std::move(result);
}

namespace {
// Internal function which recursively walk the filesystem pushing files which
// matches the given pattern into the output vector.
std::error_code FindFilesInternal(std::vector<std::filesystem::path>& output,
                                  const std::filesystem::path& path,
                                  const std::string& pattern) {
  std::error_code ec;
  auto it = std::filesystem::directory_iterator(path, ec);
  if (ec) {
    return ec;
  }
  auto end = std::filesystem::directory_iterator();
  while (it != end) {
    if (it->is_directory()) {
      ec = FindFilesInternal(output, it->path(), pattern);
    } else if (it->is_regular_file()) {
      if (RE2::FullMatch(it->path().string(), pattern)) {
        output.push_back(it->path());
      }
    }
    it.increment(ec);
    if (ec) {
      return ec;
    }
  }
  return std::error_code();
}
}  // namespace

absl::StatusOr<std::vector<std::filesystem::path>> FindFilesMatchingRegex(
    const std::filesystem::path& path, const std::string& pattern) {
  std::vector<std::filesystem::path> result;
  std::error_code ec = FindFilesInternal(result, path, pattern);
  // Sort the output so we don't depend on filesystem traversal order.
  std::sort(result.begin(), result.end());
  if (ec) {
    return ErrorCodeToStatus(ec);
  }
  return result;
}

absl::StatusOr<std::filesystem::path> GetRealPath(const std::string& path) {
  constexpr int kPathMax = 8192;  // more portable than using PATH_MAX
  struct stat statbuf;
  XLS_RET_CHECK(lstat(path.c_str(), &statbuf) != -1) << strerror(errno);
  // If the file is a link, then dereference it.
  if ((statbuf.st_mode & S_IFMT) == S_IFLNK) {
    char buf[kPathMax];
    ssize_t len = readlink(path.c_str(), buf, sizeof(buf) - 1);
    XLS_RET_CHECK(len != -1) << strerror(errno);
    buf[len] = '\0';
    return buf;
  }

  return path;
}

}  // namespace xls
