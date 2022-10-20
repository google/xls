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

#ifndef XLS_TOOLS_EVAL_HELPERS_H_
#define XLS_TOOLS_EVAL_HELPERS_H_

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "xls/ir/value.h"

namespace xls {

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
std::string ChannelValuesToString(
    const absl::flat_hash_map<std::string, std::vector<Value>>&
        channel_to_values);

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
absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
ParseChannelValues(
    std::string_view all_channel_values,
    std::optional<const int64_t> max_values_count = std::nullopt);

// The function returns a channels-to-values map representation of the contents
// of the file. The max_values_count denotes the maximum number of values for a
// channel. The function invokes xls::ParseChannelValues, see
// xls::ParseChannelValues for more detail.
absl::StatusOr<absl::flat_hash_map<std::string, std::vector<Value>>>
ParseChannelValuesFromFile(
    std::string_view filename_with_all_channel,
    std::optional<const int64_t> max_values_count = std::nullopt);

}  // namespace xls

#endif  // XLS_TOOLS_EVAL_HELPERS_H_
