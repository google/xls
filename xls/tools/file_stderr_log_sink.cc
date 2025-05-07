// Copyright 2025 The XLS Authors
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

#include "xls/tools/file_stderr_log_sink.h"

#include <filesystem>  // NOLINT
#include <utility>

#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/log_entry.h"
#include "xls/common/file/filesystem.h"

namespace xls {

FileStderrLogSink::FileStderrLogSink(std::filesystem::path path)
    : path_(std::move(path)) {
  CHECK_OK(SetFileContents(path_, ""));
}

void FileStderrLogSink::Send(const absl::LogEntry& entry) {
  if (entry.log_severity() < absl::StderrThreshold()) {
    return;
  }

  if (!entry.stacktrace().empty()) {
    CHECK_OK(AppendStringToFile(path_, entry.stacktrace()));
  } else {
    CHECK_OK(AppendStringToFile(path_,
                                entry.text_message_with_prefix_and_newline()));
  }
}

}  // namespace xls
