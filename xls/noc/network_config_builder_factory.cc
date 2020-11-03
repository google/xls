// Copyright 2020 The XLS Authors
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

#include "xls/noc/network_config_builder_factory.h"

namespace xls::noc {

absl::StatusOr<std::unique_ptr<NetworkConfigBuilder>>
NetworkConfigBuilderFactory::GetNetworkConfigBuilder(
    absl::string_view name) const {
  auto iter = network_config_builders_.find(name);
  if (iter == network_config_builders_.end()) {
    return absl::NotFoundError(
        "Network Config Builder not found in Network Config Factory.");
  }
  return iter->second();
}

absl::Status NetworkConfigBuilderFactory::RegisterNetworkConfigBuilder(
    const std::string& name,
    NetworkConfigBuilderFactory::factory_item network_config_builder) {
  if (network_config_builders_.insert({name, network_config_builder}).second) {
    return absl::OkStatus();
  }
  return absl::AlreadyExistsError(
      "Network Config Builder already in Network Config Factory.");
}

std::vector<absl::string_view>
NetworkConfigBuilderFactory::GetNetworkConfigBuilderNames() const {
  std::vector<absl::string_view> names;
  names.reserve(network_config_builders_.size());
  for (const auto& item : network_config_builders_) {
    names.emplace_back(item.first);
  }
  std::sort(names.begin(), names.end());
  return names;
}

}  // namespace xls::noc
