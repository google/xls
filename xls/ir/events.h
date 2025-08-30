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
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xls/ir/evaluator_result.pb.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/value.h"

namespace xls {

// Common structure capturing events that can be produced by any XLS interpreter
// (DSLX, IR, JIT, etc.)
class InterpreterEvents {
 public:
  void AddTraceStatementMessage(int64_t verbosity, std::string msg) {
    TraceMessageProto* tm = proto_.add_trace_msgs();
    tm->set_message(std::move(msg));
    tm->mutable_statement()->set_verbosity(verbosity);
  }

  void AddTraceCallMessage(std::string_view function_name,
                           absl::Span<const Value> args, int64_t call_depth,
                           FormatPreference format_preference) {
    TraceMessageProto* tm = proto_.add_trace_msgs();
    // Build an indented message like: "  foo(1, 2, 3)".
    tm->set_message(absl::StrFormat(
        "%*s%s(%s)", call_depth * 2, "", function_name,
        absl::StrJoin(args, ", ",
                      [format_preference](std::string* out, const Value& v) {
                        out->append(v.ToHumanString(format_preference));
                      })));
    tm->mutable_call()->set_function_name(std::string{function_name});
    tm->mutable_call()->set_call_depth(call_depth);
    for (const Value& v : args) {
      *tm->mutable_call()->add_args() = v.AsProto().value();
    }
  }
  void AddAssertMessage(const std::string& msg) {
    proto_.add_assert_msgs()->set_message(msg);
  }

  const ::google::protobuf::RepeatedPtrField<TraceMessageProto>&
  GetTraceMessages() const {
    return proto_.trace_msgs();
  }
  std::vector<std::string> GetTraceMessageStrings() const {
    std::vector<std::string> messages;
    messages.reserve(proto_.trace_msgs_size());
    for (const TraceMessageProto& t : proto_.trace_msgs()) {
      messages.push_back(t.message());
    }
    return messages;
  }
  std::vector<std::string> GetAssertMessages() const {
    std::vector<std::string> asserts;
    asserts.reserve(proto_.assert_msgs_size());
    for (const AssertMessageProto& a : proto_.assert_msgs()) {
      asserts.push_back(a.message());
    }
    return asserts;
  }

  void Clear() { proto_.Clear(); }

  bool operator==(const InterpreterEvents& other) const {
    return proto_.SerializeAsString() == other.proto_.SerializeAsString();
  }
  bool operator!=(const InterpreterEvents& other) const {
    return !(*this == other);
  }

  void AppendFrom(const InterpreterEvents& other) {
    for (const TraceMessageProto& t : other.proto_.trace_msgs()) {
      *proto_.add_trace_msgs() = t;
    }
    for (const AssertMessageProto& a : other.proto_.assert_msgs()) {
      *proto_.add_assert_msgs() = a;
    }
  }

  const EvaluatorEventsProto& AsProto() const { return proto_; }

 private:
  EvaluatorEventsProto proto_;
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
