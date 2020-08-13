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

#include "xls/common/file/filesystem.h"

#include <errno.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include "google/protobuf/io/tokenizer.h"
#include "google/protobuf/text_format.h"
#include "absl/status/status.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/error_code_to_status.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"

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

  void AddError(int line, int column, const std::string& message) override {
    status_.Update(absl::Status(
        absl::StatusCode::kFailedPrecondition,
        absl::StrCat("Failed to parse ", proto_.GetDescriptor()->name(),
                     " proto from text.  First failure is at line ", line,
                     " column ", column, " in file '", file_name_.string(),
                     "'.  Proto parser error:\n", message)));
  }

  absl::Status status() { return status_; }

 private:
  absl::Status status_;
  const std::filesystem::path file_name_;
  const google::protobuf::Message& proto_;
};

enum class SetOrAppend { kSet, kAppend };

absl::Status SetFileContentsOrAppend(const std::filesystem::path& file_name,
                                     absl::string_view content,
                                     SetOrAppend set_or_append) {
  // Use POSIX C APIs instead of C++ iostreams to avoid exceptions.
  int fd = open(file_name.c_str(),
                O_WRONLY | O_CREAT | O_CLOEXEC |
                    (set_or_append == SetOrAppend::kAppend ? O_APPEND : 0),
                0664);
  if (fd == -1) {
    return ErrNoToStatusWithFilename(errno, file_name);
  }

  // Clear existing contents if not appending.
  if (set_or_append == SetOrAppend::kSet) {
    if (ftruncate(fd, 0) == -1) {
      return ErrNoToStatusWithFilename(errno, file_name);
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
      return ErrNoToStatusWithFilename(errno, file_name);
    }
    written += n;
  }

  if (close(fd) != 0) {
    return ErrNoToStatusWithFilename(errno, file_name);
  }
  return absl::OkStatus();
}

}  // anonymous namespace

absl::Status FileExists(const std::filesystem::path& path) {
  std::error_code ec;
  bool result = std::filesystem::exists(path, ec);
  if (ec) {
    return ErrorCodeToStatus(ec);
  } else {
    return result ? absl::OkStatus()
                  : absl::NotFoundError("File does not exist.");
  }
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

xabsl::StatusOr<std::string> GetFileContents(
    const std::filesystem::path& file_name) {
  // Use POSIX C APIs instead of C++ iostreams to avoid exceptions.
  std::string result;

  int fd = open(file_name.c_str(), O_RDONLY | O_CLOEXEC);
  if (fd == -1) {
    return ErrNoToStatusWithFilename(errno, file_name);
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
                             absl::string_view content) {
  return SetFileContentsOrAppend(file_name, content, SetOrAppend::kSet);
}

absl::Status AppendStringToFile(const std::filesystem::path& file_name,
                                absl::string_view content) {
  return SetFileContentsOrAppend(file_name, content, SetOrAppend::kAppend);
}

absl::Status ParseTextProtoFile(const std::filesystem::path& file_name,
                                google::protobuf::Message* proto) {
  XLS_ASSIGN_OR_RETURN(std::string text_proto, GetFileContents(file_name));

  ParseTextProtoFileErrorCollector collector(file_name, *proto);
  google::protobuf::TextFormat::Parser parser;
  parser.RecordErrorsTo(&collector);

  const bool success = parser.ParseFromString(text_proto, proto);
  XLS_DCHECK_EQ(success, collector.status().ok());
  return collector.status();
}

absl::Status SetTextProtoFile(const std::filesystem::path& file_name,
                              const google::protobuf::Message& proto) {
  std::string text_proto;
  if (!proto.IsInitialized()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Cannot serialize proto, missing required field ",
                     proto.InitializationErrorString()));
  }
  if (!google::protobuf::TextFormat::PrintToString(proto, &text_proto)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Failed to convert proto to text for saving to ", file_name.string(),
        " (this generally stems from massive protobufs that either exhaust "
        "memory or overflow a 32-bit buffer somewhere)."));
  }
  return SetFileContents(file_name, text_proto);
}

xabsl::StatusOr<std::filesystem::path> GetCurrentDirectory() {
  std::error_code ec;
  std::filesystem::path path = std::filesystem::current_path(ec);
  if (ec) {
    return ErrorCodeToStatus(ec);
  }
  return path;
}

xabsl::StatusOr<std::vector<std::filesystem::path>> GetDirectoryEntries(
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

xabsl::StatusOr<std::filesystem::path> GetRealPath(const std::string& path) {
  struct stat statbuf;
  XLS_RET_CHECK(lstat(path.c_str(), &statbuf) != -1) << strerror(errno);
  // If the file is a link, then dereference it.
  if ((statbuf.st_mode & S_IFMT) == S_IFLNK) {
    char buf[PATH_MAX];
    ssize_t len = readlink(path.c_str(), buf, PATH_MAX - 1);
    XLS_RET_CHECK(len != -1) << strerror(errno);
    buf[len] = '\0';
    return buf;
  }

  return path;
}

}  // namespace xls
