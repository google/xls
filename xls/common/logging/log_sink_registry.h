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

#ifndef XLS_COMMON_LOGGING_LOG_SINK_REGISTRY_H_
#define XLS_COMMON_LOGGING_LOG_SINK_REGISTRY_H_

#include "absl/types/span.h"
#include "xls/common/logging/log_entry.h"
#include "xls/common/logging/log_sink.h"

namespace xls {

// Add or remove a `LogSink` as a consumer of logging data.  Thread-safe.
void AddLogSink(LogSink* sink);
void RemoveLogSink(LogSink* sink);

namespace logging_internal {

// This function may log to two sets of sinks:
//
// * If `extra_sinks_only` is true, it will dispatch only to `extra_sinks`.
//   `LogMessage::ToSinkAlso` and `LogMessage::ToSinkOnly` are used to attach
//    extra sinks to the entry.
// * Otherwise it will also log to the global sinks set. This set is managed
//   by `xls::AddLogSink` and `xls::RemoveLogSink`.
void LogToSinks(const LogEntry& entry, absl::Span<LogSink*> extra_sinks,
                bool extra_sinks_only);

};  // namespace logging_internal

}  // namespace xls

#endif  // XLS_COMMON_LOGGING_LOG_SINK_REGISTRY_H_
