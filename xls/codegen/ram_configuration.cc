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

#include "xls/codegen/ram_configuration.h"

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"

namespace xls::verilog {
namespace {

using ram_configuration_parser_t =
    std::function<absl::StatusOr<std::unique_ptr<RamConfiguration>>(
        absl::Span<const std::string_view>)>;

absl::flat_hash_map<std::string, ram_configuration_parser_t>*
GetRamConfigurationParserMap() {
  static auto* singleton =
      new absl::flat_hash_map<std::string, ram_configuration_parser_t>{
          {"1RW", Ram1RWConfiguration::ParseSplitString},
      };
  return singleton;
}

std::optional<ram_configuration_parser_t> GetRamConfigurationParserForRamKind(
    std::string_view ram_kind) {
  auto* parser_map = GetRamConfigurationParserMap();
  auto itr = parser_map->find(ram_kind);
  if (itr != parser_map->end()) {
    return itr->second;
  }
  return std::nullopt;
}

}  // namespace

absl::StatusOr<std::unique_ptr<RamConfiguration>> RamConfiguration::ParseString(
    std::string_view text) {
  std::vector<std::string_view> split_str = absl::StrSplit(text, ':');
  if (split_str.size() < 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "RamConfiguration must get at least two ':'-separated elements for "
        "name and "
        "kind (e.g. ram_name:1RW:req_name:resp_name), only got %d elements.",
        split_str.size()));
  }
  // First field is ram name, second field is ram kind. We dispatch further
  // parsing on ram kind.
  std::string_view ram_kind = split_str.at(1);
  auto configuration_parser = GetRamConfigurationParserForRamKind(ram_kind);
  if (!configuration_parser.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "No RAM configuration parser found for kind %s.", ram_kind));
  }
  return (*configuration_parser)(split_str);
}

absl::StatusOr<std::unique_ptr<Ram1RWConfiguration>>
Ram1RWConfiguration::ParseSplitString(
    absl::Span<const std::string_view> fields) {
  // 1RW RAM has configuration (name, "1RW", request_channel_name,
  // response_channel_name[, latency]). If not specified, latency is assumed to
  // be 1.
  if (fields.size() < 4 || fields.size() > 5) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected arguments name:1RW:req_name:resp_name[:latency], got %d "
        "fields instead.",
        fields.size()));
  }
  if (fields.at(1) != "1RW") {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected to see RAM kind 1RW, got %s.", fields.at(1)));
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
  return std::make_unique<Ram1RWConfiguration>(
      name, latency, request_channel_name, response_channel_name);
}

std::unique_ptr<RamConfiguration> Ram1RWConfiguration::Clone() const {
  return std::make_unique<Ram1RWConfiguration>(*this);
}

}  // namespace xls::verilog
