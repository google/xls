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

#include "xls/common/logging/log_sink_registry.h"

#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/base/thread_annotations.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xls/common/logging/log_entry.h"
#include "xls/common/logging/log_sink.h"

namespace xls {

namespace {

// `global_sinks` holds globally registered `LogSink`s.
ABSL_CONST_INIT absl::Mutex global_sinks_mutex(absl::kConstInit);
ABSL_CONST_INIT std::vector<LogSink*>* global_sinks ABSL_GUARDED_BY(
    global_sinks_mutex) ABSL_PT_GUARDED_BY(global_sinks_mutex) = nullptr;

// `sink_send_mutex` protects against concurrent calls from the logging library
// to any `LogSink::Send()`.
ABSL_CONST_INIT absl::Mutex sink_send_mutex
    ABSL_ACQUIRED_AFTER(global_sinks_mutex)(absl::kConstInit);

}  // namespace

void AddLogSink(LogSink* sink) ABSL_LOCKS_EXCLUDED(global_sinks_mutex) {
  absl::MutexLock global_sinks_lock(&global_sinks_mutex);
  if (!global_sinks) {
    global_sinks = new std::vector<LogSink*>();
  }
  global_sinks->push_back(sink);
}

void RemoveLogSink(LogSink* sink) ABSL_LOCKS_EXCLUDED(global_sinks_mutex) {
  absl::MutexLock global_sinks_lock(&global_sinks_mutex);
  if (!global_sinks) {
    return;
  }
  for (auto iter = global_sinks->begin(); iter != global_sinks->end(); ++iter) {
    if (*iter == sink) {
      global_sinks->erase(iter);
      return;
    }
  }
}

namespace logging_internal {

void LogToSinks(const LogEntry& entry, absl::Span<LogSink*> extra_sinks,
                bool extra_sinks_only)
    ABSL_LOCKS_EXCLUDED(global_sinks_mutex,
                        sink_send_mutex) ABSL_NO_THREAD_SAFETY_ANALYSIS {
  if (!extra_sinks_only) {
    global_sinks_mutex.ReaderLock();
  }
  if (!extra_sinks.empty() ||
      (!extra_sinks_only && global_sinks && !global_sinks->empty())) {
    {
      absl::MutexLock send_sink_lock(&sink_send_mutex);
      for (LogSink* sink : extra_sinks) {
        sink->Send(entry);
      }
      if (!extra_sinks_only && global_sinks) {
        for (LogSink* sink : *global_sinks) {
          sink->Send(entry);
        }
      }
    }
    for (LogSink* sink : extra_sinks) {
      sink->WaitTillSent();
    }
    if (!extra_sinks_only && global_sinks) {
      for (LogSink* sink : *global_sinks) {
        sink->WaitTillSent();
      }
    }
  }
  if (!extra_sinks_only) {
    global_sinks_mutex.ReaderUnlock();
  }
}

}  // namespace logging_internal

}  // namespace xls
