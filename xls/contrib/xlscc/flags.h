// Copyright 2023 The XLS Authors
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

#ifndef XLS_CONTRIB_XLSCC_FLAGS_H_
#define XLS_CONTRIB_XLSCC_FLAGS_H_

#include <string>
#include <string_view>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "xls/ir/channel.h"

namespace xlscc {

struct ChannelStrictnessMap {
  absl::flat_hash_map<std::string, xls::ChannelStrictness> map;

  auto emplace(const std::string& channel, xls::ChannelStrictness strictness) {
    return map.emplace(channel, strictness);
  }
  auto begin() const { return map.begin(); }
  auto end() const { return map.end(); }
};

inline bool AbslParseFlag(std::string_view text, ChannelStrictnessMap* result,
                          std::string* error) {
  for (std::string_view item : absl::StrSplit(text, ',', absl::SkipEmpty())) {
    std::pair<std::string_view, std::string_view> kv =
        absl::StrSplit(item, absl::MaxSplits(':', 1));
    auto& [channel, strictness_str] = kv;
    if (channel.empty()) {
      return false;
    }
    absl::StatusOr<xls::ChannelStrictness> strictness =
        xls::ChannelStrictnessFromString(strictness_str);
    if (!strictness.ok()) {
      *error = strictness.status().ToString();
      return false;
    }
    result->emplace(std::string(channel), *strictness);
  }
  return true;
}

inline std::string AbslUnparseFlag(
    const ChannelStrictnessMap& channel_strictness_map) {
  std::string result;

  std::string_view item_delim;
  for (const auto& [channel, strictness] : channel_strictness_map) {
    absl::StrAppend(&result, item_delim, channel, ":",
                    xls::ChannelStrictnessToString(strictness));
    item_delim = ",";
  }

  return result;
}

}  // namespace xlscc

#endif  // XLS_CONTRIB_XLSCC_FLAGS_H_
