// Copyright 2022 The XLS Authors
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

#include "xls/tools/eval_utils.h"

#include <cstdint>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/indent.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/value.h"
#include "xls/ir/xls_value.pb.h"
#include "xls/tools/proc_channel_values.pb.h"
#include "re2/re2.h"

namespace xls {

absl::StatusOr<std::vector<Value>> ParseValuesFile(std::string_view filename,
                                                   int64_t max_lines) {
  if (max_lines == 0) {
    return std::vector<Value>();
  }

  XLS_ASSIGN_OR_RETURN(std::string contents, GetFileContents(filename));
  std::vector<Value> ret;
  int64_t li = 0;
  for (const auto& line :
       absl::StrSplit(contents, '\n', absl::SkipWhitespace())) {
    if (0 == (li % 500)) {
      XLS_VLOG(1) << "Parsing values file at line " << li;
    }
    li++;
    XLS_ASSIGN_OR_RETURN(Value value, Parser::ParseTypedValue(line));
    ret.push_back(std::move(value));
    if (li == max_lines) {
      break;
    }
  }
  return ret;
}

std::string ChannelValuesToString(
    const absl::flat_hash_map<std::string, std::vector<Value>>&
        channel_to_values) {
  std::vector<std::string> lines;
  for (const auto& [channel_name, values] : channel_to_values) {
    lines.push_back(absl::StrCat(channel_name, " : {"));
    for (const Value& value : values) {
      lines.push_back(Indent(value.ToString(FormatPreference::kHex)));
    }
    lines.push_back("}");
  }
  return absl::StrJoin(lines, "\n");
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
ParseChannelValues(std::string_view all_channel_values,
                   std::optional<const int64_t> max_values_count) {
  enum ParseState {
    kExpectStartOfChannel = 0,
    kParsingChannel,
  };
  absl::flat_hash_map<std::string, std::vector<Value>> channel_to_values;
  ParseState state = kExpectStartOfChannel;
  std::string channel_name;
  std::vector<Value> channel_values;
  int64_t line_number = 0, values_per_channel = 0;
  for (const auto& line : absl::StrSplit(all_channel_values, '\n')) {
    if (0 == (line_number % 500)) {
      XLS_VLOG(1) << "Parsing at line " << line_number;
    }
    line_number++;
    if (line.empty() || (line.find_first_not_of(' ') == std::string::npos)) {
      continue;
    }
    switch (state) {
      case kExpectStartOfChannel: {
        if (!RE2::FullMatch(line, "([[:word:]]+)\\s*:\\s*{", &channel_name)) {
          return absl::FailedPreconditionError(
              absl::StrFormat("Expected start of channel declaration with "
                              "format: (\"CHANNEL_NAME : {\", got (\"%s\").",
                              line));
        }
        std::vector<std::string> strings =
            absl::StrSplit(line, ' ', absl::SkipWhitespace());
        if (channel_to_values.contains(channel_name)) {
          return absl::FailedPreconditionError(absl::StrFormat(
              "Channel name '%s' declare more than once.", channel_name));
        }
        XLS_VLOG(1) << "Parsing start of channel " << channel_name;
        state = kParsingChannel;
        break;
      }
      case kParsingChannel: {
        if (line == "}") {
          channel_to_values[channel_name] = channel_values;
          XLS_VLOG(1) << "Adding channel: " << channel_name;
          values_per_channel = 0;
          channel_values.clear();
          state = kExpectStartOfChannel;
          break;
        }
        if (max_values_count.has_value() &&
            values_per_channel == max_values_count.value()) {
          break;
        }
        XLS_ASSIGN_OR_RETURN(Value value, Parser::ParseTypedValue(line));
        channel_values.push_back(value);
        values_per_channel++;
        break;
      }
    }
  }
  return channel_to_values;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
ParseChannelValuesFromFile(std::string_view filename_with_all_channel,
                           std::optional<const int64_t> max_values_count) {
  XLS_ASSIGN_OR_RETURN(std::string content,
                       GetFileContents(filename_with_all_channel));
  return ParseChannelValues(content, max_values_count);
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
ParseChannelValuesFromProto(const ProcChannelValuesProto& values,
                            std::optional<const int64_t> max_values_count) {
  absl::flat_hash_map<std::string, std::vector<Value>> results;
  results.reserve(values.channels_size());
  for (const ProcChannelValuesProto::Channel& c : values.channels()) {
    std::vector<Value>& channel_vec = results[c.name()];
    channel_vec.reserve(c.entry_size());
    int64_t cnt = 0;
    for (const ValueProto& iv : c.entry()) {
      XLS_ASSIGN_OR_RETURN(Value v, Value::FromProto(iv));
      channel_vec.push_back(v);
      if (++cnt == max_values_count) {
        break;
      }
    }
  }
  return results;
}

absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
ParseChannelValuesFromProtoFile(std::string_view filename_with_all_channel,
                                std::optional<const int64_t> max_values_count) {
  ProcChannelValuesProto pcv;
  XLS_ASSIGN_OR_RETURN(std::string content,
                       GetFileContents(filename_with_all_channel));
  XLS_RET_CHECK(pcv.ParseFromString(content));
  return ParseChannelValuesFromProto(pcv, max_values_count);
}

absl::StatusOr<ProcChannelValuesProto> ChannelValuesToProto(
    const absl::flat_hash_map<std::string, std::vector<Value>>& channel_map) {
  ProcChannelValuesProto pcv;
  using ChannelMap = absl::flat_hash_map<std::string, std::vector<Value>>;
  std::vector<ChannelMap::const_pointer> sorted;
  sorted.reserve(channel_map.size());
  absl::c_transform(channel_map, std::back_inserter(sorted),
                    [](const auto& p) { return &p; });
  absl::c_sort(sorted, [](auto* l, auto* r) { return l->first < r->first; });
  for (ChannelMap::const_pointer key_value : sorted) {
    const auto& [name, values] = *key_value;
    ProcChannelValuesProto::Channel* chan = pcv.add_channels();
    chan->set_name(name);
    for (const Value& v : values) {
      XLS_ASSIGN_OR_RETURN(*chan->add_entry(), v.AsProto());
    }
  }
  return pcv;
}
}  // namespace xls
