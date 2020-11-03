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

#ifndef XLS_NOC_NETWORK_CONFIG_BUILDER_FACTORY_H_
#define XLS_NOC_NETWORK_CONFIG_BUILDER_FACTORY_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/module_initializer.h"
#include "xls/noc/network_config.pb.h"
#include "xls/noc/network_config_builder.h"

namespace xls::noc {

// Factory / registry for network config builders.
class NetworkConfigBuilderFactory {
  typedef std::function<std::unique_ptr<NetworkConfigBuilder>()> factory_item;

 public:
  static NetworkConfigBuilderFactory& GetInstance() {
    static NetworkConfigBuilderFactory* instance =
        new NetworkConfigBuilderFactory;
    return *instance;
  }

  absl::StatusOr<std::unique_ptr<NetworkConfigBuilder>> GetNetworkConfigBuilder(
      absl::string_view name) const;

  absl::Status RegisterNetworkConfigBuilder(
      const std::string& name, factory_item network_config_builder);

  // Returns a list of network config builder names in sorted order from the
  // factory.
  std::vector<absl::string_view> GetNetworkConfigBuilderNames() const;

 private:
  absl::flat_hash_map<std::string, factory_item> network_config_builders_;
};

// MACRO defining a network config builder and registering the network config
// builder into the network config builder factory. The registration occurs when
// a module initialization is performed for a given application.
#define REGISTER_IN_NETWORK_CONFIG_BUILDER_FACTORY(                           \
    __network_config_builder_name, __network_config_builder_class)            \
  class __network_config_builder_class : public NetworkConfigBuilder {        \
   public:                                                                    \
    static const char* GetName() { return #__network_config_builder_name; }   \
    virtual std::string GetUsage() const;                                     \
    virtual ~__network_config_builder_class() = default;                      \
                                                                              \
   protected:                                                                 \
    virtual absl::Status ValidateArguments(                                   \
        const NetworkConfigBuilderOptions& options) const;                    \
    virtual absl::StatusOr<NetworkConfigProto> GenerateNetworkConfig(         \
        const NetworkConfigBuilderOptions& options) const;                    \
  };                                                                          \
  XLS_REGISTER_MODULE_INITIALIZER(__network_config_builder_name##_registry, { \
    absl::Status __network_config_builder_name##_registered_ =                \
        NetworkConfigBuilderFactory::GetInstance()                            \
            .RegisterNetworkConfigBuilder(                                    \
                __network_config_builder_class::GetName(), []() {             \
                  return absl::make_unique<__network_config_builder_class>(); \
                });                                                           \
  });

}  // namespace xls::noc

#endif  // XLS_NOC_NETWORK_CONFIG_BUILDER_FACTORY_H_
