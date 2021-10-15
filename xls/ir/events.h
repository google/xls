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

#include "absl/status/status.h"
#include "absl/status/statusor.h"

namespace xls {

// Common structure capturing events that can be produced by any XLS interpreter
// (DSLX, IR, JIT, etc.)
struct InterpreterEvents {
  std::vector<std::string> trace_msgs;
  std::vector<std::string> assert_msgs;

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
    InterpreterResult<ValueT> result) {
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
    absl::StatusOr<InterpreterResult<ValueT>> result) {
  if (!result.ok()) {
    return result.status();
  }

  return InterpreterResultToStatusOrValue(result.value());
}

}  // namespace xls

#endif  // XLS_IR_EVENTS_H
