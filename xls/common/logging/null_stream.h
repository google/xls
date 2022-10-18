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

#ifndef XLS_COMMON_LOGGING_NULL_STREAM_H_
#define XLS_COMMON_LOGGING_NULL_STREAM_H_

#include <unistd.h>

#include "absl/strings/string_view.h"
#include "xls/common/logging/log_sink.h"
#include "xls/common/source_location.h"

namespace xls {
namespace logging_internal {

// A `NullStream` implements the API of `LogMessage` (a few methods and
// `operator<<`) but does nothing.  All methods are defined inline so the
// compiler can eliminate the whole instance and discard anything that's
// streamed in.
class NullStream {
 public:
  NullStream& stream() { return *this; }
  NullStream& WithCheckFailureMessage(std::string_view) { return *this; }
  NullStream& AtLocation(std::string_view, int) { return *this; }
  NullStream& AtLocation(xabsl::SourceLocation) { return *this; }
  NullStream& NoPrefix() { return *this; }
  NullStream& WithPerror() { return *this; }
  NullStream& WithVerbosity(int) { return *this; }
  NullStream& ToSinkAlso(LogSink*) { return *this; }
  NullStream& ToSinkOnly(LogSink*) { return *this; }
};
template <typename T>
inline NullStream& operator<<(NullStream& str, const T&) {
  return str;
}
inline NullStream& operator<<(NullStream& str,
                              std::ostream& (*)(std::ostream& os)) {
  return str;
}
inline NullStream& operator<<(NullStream& str,
                              std::ios_base& (*)(std::ios_base& os)) {
  return str;
}

// `NullStreamFatal` implements the process termination semantics of
// `LogMessageFatal`, which means it always terminates the process.  `DFATAL`
// and expression-defined severity use `NullStreamMaybeFatal` above.
class NullStreamFatal : public NullStream {
 public:
  NullStreamFatal() {}
  ABSL_ATTRIBUTE_NORETURN ~NullStreamFatal() { _exit(1); }
};

}  // namespace logging_internal
}  // namespace xls

#endif  // XLS_COMMON_LOGGING_NULL_STREAM_H_
