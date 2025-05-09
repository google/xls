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

#ifndef XLS_TOOLS_FILE_LOG_SINK_H_
#define XLS_TOOLS_FILE_LOG_SINK_H_

#include <filesystem>  // NOLINT

#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"

namespace xls {

class FileStderrLogSink final : public absl::LogSink {
 public:
  explicit FileStderrLogSink(std::filesystem::path path);

  ~FileStderrLogSink() override = default;

  void Send(const absl::LogEntry& entry) override;

 private:
  const std::filesystem::path path_;
};

}  // namespace xls

#endif  // XLS_TOOLS_FILE_LOG_SINK_H_
