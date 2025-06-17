// Copyright 2021 The XLS Authors
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

#ifndef XLS_IR_EVENTS_H_
#define XLS_IR_EVENTS_H_

#include <compare>
#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"

namespace xls {

// A trace message is a string and a verbosity associated with the message.
struct TraceMessage {
  std::string message;
  int64_t verbosity;

  bool operator==(const TraceMessage& other) const {
    return message == other.message && verbosity == other.verbosity;
  }
  bool operator!=(const TraceMessage& other) const { return !(*this == other); }
  std::strong_ordering operator<=>(const TraceMessage& other) const {
    auto verbosity_cmp = verbosity <=> other.verbosity;
    if (verbosity_cmp != std::strong_ordering::equal) {
      return verbosity_cmp;
    }
    return message <=> other.message;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TraceMessage& t) {
    absl::Format(&sink, "%s [verbosity: %d]", t.message, t.verbosity);
  }
};

// Common structure capturing events that can be produced by any XLS interpreter
// (DSLX, IR, JIT, etc.)
struct InterpreterEvents {
  std::vector<TraceMessage> trace_msgs;
  std::vector<std::string> assert_msgs;

  void Clear() {
    trace_msgs.clear();
    assert_msgs.clear();
  }

  bool operator==(const InterpreterEvents& other) const {
    return trace_msgs == other.trace_msgs && assert_msgs == other.assert_msgs;
  }
  bool operator!=(const InterpreterEvents& other) const {
    return !(*this == other);
  }
};
// Convert an InterpreterEvents structure into a result status, returning
// a failure when an assertion has been raised.
absl::Status InterpreterEventsToStatus(const InterpreterEvents& events);
template <typename ValueT>
struct InterpreterResult {
  ValueT value;
  InterpreterEvents events;
};

// Convert an interpreter result to a status or a value depending on whether
// any assertion has failed.
template <typename ValueT>
absl::StatusOr<ValueT> InterpreterResultToStatusOrValue(
    const InterpreterResult<ValueT>& result) {
  absl::Status status = InterpreterEventsToStatus(result.events);

  if (!status.ok()) {
    return status;
  }

  return result.value;
}

// Convert an interpreter result or error to a value or error by dropping
// interpreter events and including assertion failures as errors.
template <typename ValueT>
absl::StatusOr<ValueT> DropInterpreterEvents(
    const absl::StatusOr<InterpreterResult<ValueT>>& result) {
  if (!result.ok()) {
    return result.status();
  }

  return InterpreterResultToStatusOrValue(result.value());
}

}  // namespace xls

#endif  // XLS_IR_EVENTS_H_
