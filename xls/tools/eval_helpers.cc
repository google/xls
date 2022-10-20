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

#include "xls/tools/eval_helpers.h"

#include <optional>

#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "xls/common/file/filesystem.h"
#include "xls/common/indent.h"
#include "xls/ir/ir_parser.h"
#include "re2/re2.h"

namespace xls {

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
}  // namespace xls
