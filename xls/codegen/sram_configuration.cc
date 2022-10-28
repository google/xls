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

#include "xls/codegen/sram_configuration.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"

namespace xls::verilog {
namespace {

using sram_configuration_parser_t =
    std::function<absl::StatusOr<std::unique_ptr<SramConfiguration>>(
        absl::Span<const std::string_view>)>;

absl::flat_hash_map<std::string, sram_configuration_parser_t>*
GetSramConfigurationParserMap() {
  static auto* singleton =
      new absl::flat_hash_map<std::string, sram_configuration_parser_t>{
          {"1RW", Sram1RWConfiguration::ParseSplitString},
      };
  return singleton;
}

std::optional<sram_configuration_parser_t>
GetSramConfigurationParserForSramKind(std::string_view sram_kind) {
  auto* parser_map = GetSramConfigurationParserMap();
  auto itr = parser_map->find(sram_kind);
  if (itr != parser_map->end()) {
    return itr->second;
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<std::unique_ptr<SramConfiguration>>
SramConfiguration::ParseString(std::string_view text) {
  std::vector<std::string_view> split_str = absl::StrSplit(text, ':');
  if (split_str.size() < 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "SramConfiguration must get at least two ':'-separated elements for "
        "name and "
        "kind (e.g. sram_name:1RW:req_name:resp_name), only got %d elements.",
        split_str.size()));
  }
  // First field is sram name, second field is sram kind. We dispatch further
  // parsing on sram kind.
  std::string_view sram_kind = split_str.at(1);
  auto configuration_parser = GetSramConfigurationParserForSramKind(sram_kind);
  if (!configuration_parser.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "No sram configuration parser found for sram kind %s.", sram_kind));
  }
  return (*configuration_parser)(split_str);
}

absl::StatusOr<std::unique_ptr<Sram1RWConfiguration>>
Sram1RWConfiguration::ParseSplitString(
    absl::Span<const std::string_view> fields) {
  // 1RW SRAM has configuration (name, "1RW", request_channel_name,
  // response_channel_name[, latency]). If not specified, latency is assumed to
  // be 1.
  if (fields.size() < 4 || fields.size() > 5) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected arguments name:1RW:req_name:resp_name[:latency], got %d "
        "fields instead.",
        fields.size()));
  }
  if (fields.at(1) != "1RW") {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected to see SRAM kind 1RW, got %s.", fields.at(1)));
  }
  std::string_view name = fields.at(0);
  std::string_view request_channel_name = fields.at(2);
  std::string_view response_channel_name = fields.at(3);
  int64_t latency = 1;
  if (fields.size() > 4) {
    if (!absl::SimpleAtoi(fields.at(4), &latency)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Latency must be an integer, got %s.", fields.at(4)));
    }
  }
  return std::make_unique<Sram1RWConfiguration>(
      name, latency, request_channel_name, response_channel_name);
}

std::unique_ptr<SramConfiguration> Sram1RWConfiguration::Clone() const {
  return std::make_unique<Sram1RWConfiguration>(*this);
}

}  // namespace xls::verilog
