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

#ifndef XLS_COMMON_LOGGING_LOG_SINK_H_
#define XLS_COMMON_LOGGING_LOG_SINK_H_

#include "xls/common/logging/log_entry.h"

// The xabsl namespace has types that are anticipated to become available in
// Abseil reasonably soon, at which point they can be removed. These types are
// not in the xls namespace to make it easier to search/replace migrate usages
// to Abseil in the future.
namespace xls {

// `LogSink` is an interface for processors of log messages. Implementations
// must be thread-safe because a shared instance will be called from whichever
// thread ran the `LOG()` line.
class LogSink {
 public:
  virtual ~LogSink() = default;

  // `Send` is called synchronously during the log statement.  The logging
  // module guarantees not to call `Send` concurrently on the same log sink.
  // Calls to `XLS_LOG` or `XLS_CHECK` inside this call are automatically
  // converted to `ABSL_RAW_LOG` or `ABSL_RAW_CHECK`.  Implementations should be
  // careful not to take any locks that might be held by the `XLS_LOG` caller,
  // to avoid deadlock.
  //
  // `entry` is guaranteed to remain valid until the subsequent call to
  // `WaitTillSent` completes, so implementations may store a pointer to or
  // copy of `entry` (e.g. in a thread local variable) for use in
  // `WaitTillSent`.
  virtual void Send(const LogEntry& entry) = 0;

  // `WaitTillSent` blocks the calling thread (the thread that generated a log
  // message) until the sink has finished processing the log message.
  // `WaitTillSent` is called once per log message, following the call to
  // `Send`.  This may be useful when log messages are buffered or processed
  // asynchronously by an expensive log sink.
  // The default implementation returns immediately.  Like `Send`, calls to
  // `XLS_LOG` or `XLS_CHECK` are automatically converted to `ABSL_RAW_LOG` or
  // `ABSL_RAW_CHECK`. Implementations should be careful not to take any locks
  // that might be held by the `XLS_LOG` caller, to avoid deadlock.
  virtual void WaitTillSent() {}
};

}  // namespace xls

#endif  // XLS_COMMON_LOGGING_LOG_SINK_H_
