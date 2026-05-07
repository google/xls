// Copyright 2026 The XLS Authors
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

#ifndef XLS_COMMON_LOGGING_COLLECT_LOGS_H_
#define XLS_COMMON_LOGGING_COLLECT_LOGS_H_

#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/log_entry.h"
#include "absl/log/log_sink.h"
#include "absl/synchronization/mutex.h"

namespace xls {

// A utility log-sink that just grabs all log messages from any source and
// stores them in a list for later inspection. This can be used with
// absl::AddLogSink.
class LogCollectorSink : public absl::LogSink {
 public:
  void Send(const absl::LogEntry& entry) override {
    absl::MutexLock lock(mutex_);
    log_lines_.emplace_back(entry.text_message_with_prefix());
  }

  std::vector<std::string> GetLogLines() {
    absl::MutexLock lock(mutex_);
    return log_lines_;
  }

 private:
  absl::Mutex mutex_;
  std::vector<std::string> log_lines_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace xls

#endif  // XLS_COMMON_LOGGING_COLLECT_LOGS_H_
