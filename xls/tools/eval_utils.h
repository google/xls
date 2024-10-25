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

#ifndef XLS_TOOLS_EVAL_UTILS_H_
#define XLS_TOOLS_EVAL_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_join.h"
#include "xls/common/indent.h"
#include "xls/ir/format_preference.h"
#include "xls/ir/value.h"
#include "xls/tests/testvector.pb.h"
#include "xls/tools/proc_channel_values.pb.h"

namespace xls {

// Returns all XLS Values in file.
// If max_lines is <0 then it is ignored.
absl::StatusOr<std::vector<Value>> ParseValuesFile(std::string_view filename,
                                                   int64_t max_lines = -1);

// Returns a string representation of the channels-to-values map. The values are
// represented in Hex format. For example, given the following
// channels-to-values map:
//
//   {
//     in : [bits[32]:0x42, bits[32]:0x64],
//     out : [bits[32]:0x21, bits[32]:0x32],
//   }
//
// the string representation would be:
//
//   in : {
//     bits[32]:0x42
//     bits[32]:0x64
//   }
//   out : {
//     bits[32]:0x21
//     bits[32]:0x32
//   }
// .
// The function translate the channels-to-values map to a human readable string.
// It is the reverse translation of the xls::ParseChannelValues function.
template <typename T>
std::string ChannelValuesToString(const T& channel_to_values) {
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

// Returns a channels-to-values map derived from the input string. For example,
// given the string:
//
//   my_input : {
//     bits[32]:0x4
//     bits[32]:0x2
//   }
//   my_output : {
//     bits[32]:0xBE
//     bits[32]:0xFE
//   }
//
// the channels-to-values map would be:
//
//   {
//     my_input : [bits[32]:0x4, bits[32]:0x2],
//     my_output : [bits[32]:0xBE, bits[32]:0xFE],
//   }
// .
// The max_values_count denotes the maximum number of values for a channel.
// The function translate the human readable string to a channels-to-values map.
// It is the reverse translation of the xls::ChannelValuesToString function.
absl::StatusOr<absl::btree_map<std::string, std::vector<Value>>>
ParseChannelValues(
    std::string_view all_channel_values,
    std::optional<const int64_t> max_values_count = std::nullopt);

// The function returns a channels-to-values map representation of the contents
// of the file. The max_values_count denotes the maximum number of values for a
// channel. The function invokes xls::ParseChannelValues, see
// xls::ParseChannelValues for more detail.
absl::StatusOr<absl::btree_map<std::string, std::vector<Value>>>
ParseChannelValuesFromFile(
    std::string_view filename_with_all_channel,
    std::optional<const int64_t> max_values_count = std::nullopt);

// The functions return a channels-to-values map representation of the contents
// of the proto. The max_values_count denotes the maximum number of values for a
// channel. If more values then 'max_values_count' are included the extra are
// ignored.
// TODO(google/xls#1645) these should be unified.
absl::StatusOr<absl::btree_map<std::string, std::vector<Value>>>
ParseChannelValuesFromProto(
    const ProcChannelValuesProto& values,
    std::optional<const int64_t> max_values_count = std::nullopt);

// Ditto, but input is a ChannelInputsProto
absl::StatusOr<absl::btree_map<std::string, std::vector<Value>>>
ParseChannelValuesFromProto(
    const testvector::ChannelInputsProto& values,
    std::optional<const int64_t> max_values_count = std::nullopt);

// The function returns a channels-to-values map representation of the contents
// of the ProcChannelValuesProto read as textproto from file.
// The max_values_count denotes the maximum number of values for a
// channel. If more values then 'max_values_count' are included the extra are
// ignored.
absl::StatusOr<absl::btree_map<std::string, std::vector<Value>>>
ParseChannelValuesFromProtoFile(
    std::string_view filename_with_all_channel,
    std::optional<const int64_t> max_values_count = std::nullopt);

// Similar to ParseChannelValuesFromProtoFile, but read from a file
// containing a testvector::SampleInputsProto textproto.
absl::StatusOr<absl::btree_map<std::string, std::vector<Value>>>
ParseChannelValuesFromTestVectorFile(
    std::string_view testvector_filename,
    std::optional<const int64_t> max_values_count = std::nullopt);

// Convert a map of channel-name -> channel values into a ProcChannelValuesProto
// proto. This is the inverse of ParseChannelValuesFromProto.
absl::StatusOr<ProcChannelValuesProto> ChannelValuesToProto(
    const absl::btree_map<std::string, std::vector<Value>>& channel_map);

}  // namespace xls

#endif  // XLS_TOOLS_EVAL_UTILS_H_
