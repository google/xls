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

#include "xls/ir/events.h"

#include <string>
#include <string_view>

#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xls/ir/evaluator_result.pb.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"

namespace xls {

bool InterpreterEvents::operator==(const InterpreterEvents& other) const {
  if (proto_.trace_msgs_size() != other.proto_.trace_msgs_size() ||
      proto_.assert_msgs_size() != other.proto_.assert_msgs_size()) {
    return false;
  }
  for (int i = 0; i < proto_.trace_msgs_size(); i++) {
    if (proto_.trace_msgs(i).message() !=
        other.proto_.trace_msgs(i).message()) {
      return false;
    }
  }
  for (int i = 0; i < proto_.assert_msgs_size(); i++) {
    if (proto_.assert_msgs(i).message() !=
        other.proto_.assert_msgs(i).message()) {
      return false;
    }
  }
  return true;
}

void InterpreterEvents::AddTraceStatementMessage(
    int64_t verbosity, std::string msg, std::optional<SourceInfo> source_info,
    std::optional<std::string> source_filename) {
  TraceMessageProto* tm = proto_.add_trace_msgs();
  tm->set_message(std::move(msg));
  tm->mutable_statement()->set_verbosity(verbosity);
  if (source_info.has_value() && !source_info->locations.empty()) {
    const SourceLocation& loc = source_info->locations.front();
    auto* locp = tm->mutable_location();
    if (source_filename.has_value()) {
      locp->set_filename(*source_filename);
    }
    locp->set_line(loc.lineno().value());
    locp->set_column(loc.colno().value());
  }
}

void InterpreterEvents::AddTraceCallMessage(
    std::string_view function_name, absl::Span<const Value> args,
    int64_t call_depth, FormatPreference format_preference,
    std::optional<SourceInfo> source_info,
    std::optional<std::string> source_filename) {
  TraceMessageProto* tm = proto_.add_trace_msgs();
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
  if (source_info.has_value() && !source_info->locations.empty()) {
    const SourceLocation& loc = source_info->locations.front();
    auto* locp = tm->mutable_location();
    if (source_filename.has_value()) {
      locp->set_filename(*source_filename);
    } else {
      locp->set_filename("<unknown>");
    }
    locp->set_line(loc.lineno().value());
    locp->set_column(loc.colno().value());
  }
}

void InterpreterEvents::AddTraceCallReturnMessage(
    std::string_view function_name, int64_t call_depth,
    FormatPreference format_preference, const Value& return_value,
    std::optional<SourceInfo> source_info,
    std::optional<std::string> source_filename) {
  TraceMessageProto* tm = proto_.add_trace_msgs();
  tm->set_message(
      absl::StrFormat("%*s%s(...) => %s", call_depth * 2, "", function_name,
                      return_value.ToHumanString(format_preference)));
  tm->mutable_call_return()->set_function_name(std::string{function_name});
  tm->mutable_call_return()->set_call_depth(call_depth);
  *tm->mutable_call_return()->mutable_return_value() =
      return_value.AsProto().value();
  if (source_info.has_value() && !source_info->locations.empty()) {
    const SourceLocation& loc = source_info->locations.front();
    auto* locp = tm->mutable_location();
    if (source_filename.has_value()) {
      locp->set_filename(*source_filename);
    } else {
      locp->set_filename("<unknown>");
    }
    locp->set_line(loc.lineno().value());
    locp->set_column(loc.colno().value());
  }
}

void InterpreterEvents::AddAssertMessage(
    std::string_view msg, std::optional<SourceInfo> source_info,
    std::optional<std::string> source_filename) {
  AssertMessageProto* am = proto_.add_assert_msgs();
  am->set_message(std::string{msg});
  if (source_info.has_value() && !source_info->locations.empty()) {
    const SourceLocation& loc = source_info->locations.front();
    auto* locp = am->mutable_location();
    if (source_filename.has_value()) {
      locp->set_filename(*source_filename);
    }
    locp->set_line(loc.lineno().value());
    locp->set_column(loc.colno().value());
  }
}

const ::google::protobuf::RepeatedPtrField<TraceMessageProto>&
InterpreterEvents::GetTraceMessages() const {
  return proto_.trace_msgs();
}

std::vector<std::string> InterpreterEvents::GetTraceMessageStrings() const {
  std::vector<std::string> messages;
  messages.reserve(proto_.trace_msgs_size());
  for (const TraceMessageProto& t : proto_.trace_msgs()) {
    messages.push_back(t.message());
  }
  return messages;
}

std::vector<std::string> InterpreterEvents::GetAssertMessages() const {
  std::vector<std::string> asserts;
  asserts.reserve(proto_.assert_msgs_size());
  for (const AssertMessageProto& a : proto_.assert_msgs()) {
    asserts.push_back(a.message());
  }
  return asserts;
}

void InterpreterEvents::AppendFrom(const InterpreterEvents& other) {
  for (const TraceMessageProto& t : other.proto_.trace_msgs()) {
    *proto_.add_trace_msgs() = t;
  }
  for (const AssertMessageProto& a : other.proto_.assert_msgs()) {
    *proto_.add_assert_msgs() = a;
  }
}

absl::Status InterpreterEventsToStatus(const InterpreterEvents& events) {
  if (events.GetAssertMessages().empty()) {
    return absl::OkStatus();
  }

  // If an assertion has been raised, return the message from the first
  // assertion recorded, matching the behavior of short-circuit evaluation.
  return absl::AbortedError(events.GetAssertMessages().front());
}

}  // namespace xls
