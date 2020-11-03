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

#ifndef XLS_NOC_NETWORK_CONFIG_BUILDER_H_
#define XLS_NOC_NETWORK_CONFIG_BUILDER_H_

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/status_macros.h"
#include "xls/noc/network_config.pb.h"
#include "xls/noc/network_config_builder_arguments.pb.h"
namespace xls::noc {

// Represents a network config builder.
//
// The network config builder generates a network configuration using network
// configuration builder options.
class NetworkConfigBuilder {
 public:
  absl::StatusOr<NetworkConfigProto> ValidateArgumentsGenerateNetworkConfig(
      const NetworkConfigBuilderOptions& options) const {
    XLS_RETURN_IF_ERROR(ValidateArguments(options));
    return GenerateNetworkConfig(options);
  }
  // Returns a description of the usage for the network config builder.
  virtual std::string GetUsage() const = 0;
  virtual ~NetworkConfigBuilder() = default;

 protected:
  virtual absl::Status ValidateArguments(
      const NetworkConfigBuilderOptions& options) const = 0;
  virtual absl::StatusOr<NetworkConfigProto> GenerateNetworkConfig(
      const NetworkConfigBuilderOptions& options) const = 0;
};

}  // namespace xls::noc

#endif  // XLS_NOC_NETWORK_CONFIG_BUILDER_H_
