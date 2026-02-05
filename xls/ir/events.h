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

#include <cstdint>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "google/protobuf/repeated_ptr_field.h"
#include "xls/ir/evaluator_result.pb.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace xls {

// Common structure capturing events that can be produced by any XLS interpreter
// (DSLX, IR, JIT, etc.)
class InterpreterEvents {
 public:
  void AddTraceStatementMessage(
      int64_t verbosity, std::string msg,
      std::optional<SourceInfo> source_info = std::nullopt,
      std::optional<std::string> source_filename = std::nullopt);

  void AddTraceCallMessage(
      std::string_view function_name, absl::Span<const Value> args,
      int64_t call_depth, FormatPreference format_preference,
      std::optional<SourceInfo> source_info = std::nullopt,
      std::optional<std::string> source_filename = std::nullopt);
  void AddTraceCallReturnMessage(
      std::string_view function_name, int64_t call_depth,
      FormatPreference format_preference, const Value& return_value,
      std::optional<SourceInfo> source_info = std::nullopt,
      std::optional<std::string> source_filename = std::nullopt);
  void AddAssertMessage(
      std::string_view msg,
      std::optional<SourceInfo> source_info = std::nullopt,
      std::optional<std::string> source_filename = std::nullopt);

  const ::google::protobuf::RepeatedPtrField<TraceMessageProto>&
  GetTraceMessages() const;
  std::vector<std::string> GetTraceMessageStrings() const;
  std::vector<std::string> GetAssertMessages() const;

  void Clear() { proto_.Clear(); }

  bool operator==(const InterpreterEvents& other) const;

  bool operator!=(const InterpreterEvents& other) const {
    return !(*this == other);
  }
  void AppendFrom(const InterpreterEvents& other);

  const EvaluatorEventsProto& AsProto() const { return proto_; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const InterpreterEvents& events) {
    for (const std::string& s : events.GetTraceMessageStrings()) {
      absl::Format(&sink, "%s\n", s);
    }
    for (const std::string& s : events.GetAssertMessages()) {
      absl::Format(&sink, "%s\n", s);
    }
  }

 private:
  EvaluatorEventsProto proto_;
};

inline std::ostream& operator<<(std::ostream& os,
                                const InterpreterEvents& events) {
  return os << absl::StrCat(events);
}

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
