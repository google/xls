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

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_split.h"
#include "absl/types/span.h"
#include "absl/types/variant.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls::verilog {
namespace {
// Parses a split configuration string of the form
// "(ram_name, "1RW", request_channel_name, response_channel_name[,
// latency])".
absl::StatusOr<Ram1RWConfiguration> Ram1RWConfigurationParseSplitString(
    absl::Span<const std::string_view> fields) {
  // 1RW RAM has configuration (name, "1RW", request_channel_name,
  // response_channel_name[, latency]). If not specified, latency is assumed to
  // be 1.
  if (fields.size() < 5 || fields.size() > 6) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected arguments "
        "name:1RW:req_name:resp_name:write_comp_name[:latency], got %d "
        "fields instead.",
        fields.size()));
  }
  if (fields[1] != "1RW") {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected to see RAM kind 1RW, got %s.", fields[1]));
  }
  std::string_view name = fields[0];
  std::string_view request_channel_name = fields[2];
  std::string_view response_channel_name = fields[3];
  std::string_view write_completion_channel_name = fields[4];
  int64_t latency = 1;
  if (fields.size() > 5) {
    if (!absl::SimpleAtoi(fields[5], &latency)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Latency must be an integer, got %s.", fields[5]));
    }
  }
  return Ram1RWConfiguration(
      name, latency, /*request_name=*/request_channel_name,
      /*response_name=*/response_channel_name,
      /*write_completion_name=*/write_completion_channel_name);
}

// Parses a split configuration string of the form
// "(ram_name, "1R1W", read_request_channel_name, read_response_channel_name,
// write_request_channel_name[, latency])".
absl::StatusOr<Ram1R1WConfiguration> Ram1R1WConfigurationParseSplitString(
    absl::Span<const std::string_view> fields) {
  // 1R1W RAM has configuration (name, "1R1W", read_request_channel_name,
  // read_response_channel_name, write_request_channel_name[, latency]). If not
  // specified, latency is assumed to be 1.
  if (fields.size() < 6 || fields.size() > 7) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Expected arguments "
        "name:1R1W:read_req_name:read_resp_name:write_req_name:write_comp_name["
        ":latency], got %d "
        "fields instead.",
        fields.size()));
  }
  if (fields[1] != "1R1W") {
    return absl::InvalidArgumentError(
        absl::StrFormat("Expected to see RAM kind 1R1W, got %s.", fields[1]));
  }
  std::string_view name = fields[0];
  std::string_view read_request_channel_name = fields[2];
  std::string_view read_response_channel_name = fields[3];
  std::string_view write_request_channel_name = fields[4];
  std::string_view write_completion_channel_name = fields[5];
  int64_t latency = 1;
  if (fields.size() > 6) {
    if (!absl::SimpleAtoi(fields[6], &latency)) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Latency must be an integer, got %s.", fields[6]));
    }
  }
  return Ram1R1WConfiguration(
      name, latency, /*read_request_name=*/read_request_channel_name,
      /*read_response_name=*/read_response_channel_name,
      /*write_request_name=*/write_request_channel_name,
      /*write_completion_name=*/write_completion_channel_name);
}

// Ram configurations are in the format
// ram_name:ram_kind[:ram_specific_configuration]. ParseRamConfiguration()
// splits the configuration string on ":" and calls a ram_configuration_parser_t
// function for the desired kind. These functions take a span of string_views
// and produce a RamConfiguration.
using ram_configuration_parser_t =
    std::function<absl::StatusOr<RamConfiguration>(
        absl::Span<const std::string_view>)>;

absl::flat_hash_map<std::string, ram_configuration_parser_t>*
GetRamConfigurationParserMap() {
  static auto* singleton =
      new absl::flat_hash_map<std::string, ram_configuration_parser_t>{
          {"1RW", Ram1RWConfigurationParseSplitString},
          {"1R1W", Ram1R1WConfigurationParseSplitString},
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

absl::StatusOr<RamConfiguration> ParseRamConfiguration(std::string_view text) {
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
  std::string_view ram_kind = split_str[1];
  auto configuration_parser = GetRamConfigurationParserForRamKind(ram_kind);
  if (!configuration_parser.has_value()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "No RAM configuration parser found for kind %s.", ram_kind));
  }
  return (*configuration_parser)(split_str);
}

/* static */ std::array<IOConstraint, 2> Ram1RWConfiguration::MakeIOConstraints(
    const RamRWPortConfiguration& rw_port_configuration, int64_t latency) {
  return {
      IOConstraint(rw_port_configuration.request_channel_name,
                   IODirection::kSend,
                   rw_port_configuration.response_channel_name,
                   IODirection::kReceive, /*minimum_latency=*/latency,
                   /*maximum_latency=*/latency),
      IOConstraint(rw_port_configuration.request_channel_name,
                   IODirection::kSend,
                   rw_port_configuration.write_completion_channel_name,
                   IODirection::kReceive, /*minimum_latency=*/latency,
                   /*maximum_latency=*/latency),
  };
}

/* static */ std::array<IOConstraint, 2>
Ram1R1WConfiguration::MakeIOConstraints(
    const RamRPortConfiguration& r_port_configuration,
    const RamWPortConfiguration& w_port_configuration, int64_t latency) {
  return {
      IOConstraint(
          r_port_configuration.request_channel_name, IODirection::kSend,
          r_port_configuration.response_channel_name, IODirection::kReceive,
          /*minimum_latency=*/latency, /*maximum_latency=*/latency),
      IOConstraint(w_port_configuration.request_channel_name,
                   IODirection::kSend,
                   w_port_configuration.write_completion_channel_name,
                   IODirection::kReceive,
                   /*minimum_latency=*/latency, /*maximum_latency=*/latency),
  };
}

absl::Span<IOConstraint const> GetRamConfigurationIOConstraints(
    const RamConfiguration& ram_configuration) {
  return absl::visit([](const auto& config) { return config.io_constraints(); },
                     ram_configuration);
}

std::string_view RamConfigurationRamName(
    const RamConfiguration& ram_configuration) {
  return absl::visit([](const auto& config) { return config.ram_name(); },
                     ram_configuration);
}

}  // namespace xls::verilog
