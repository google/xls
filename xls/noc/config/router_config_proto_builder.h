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

#ifndef XLS_NOC_CONFIG_ROUTER_CONFIG_PROTO_BUILDER_H_
#define XLS_NOC_CONFIG_ROUTER_CONFIG_PROTO_BUILDER_H_

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "absl/log/die_if_null.h"
#include "xls/noc/config/arbiter_scheme_config_proto_builder.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/port_config_proto_builder.h"
#include "xls/noc/config/routing_scheme_config_proto_builder.h"

namespace xls::noc {

// A builder for constructing a router configuration proto.
class RouterConfigProtoBuilder {
 public:
  // proto cannot be nullptr.
  explicit RouterConfigProtoBuilder(RouterConfigProto* proto)
      : proto_(ABSL_DIE_IF_NULL(proto)) {}

  // Adds the name of the router.
  RouterConfigProtoBuilder& WithName(std::string_view name);

  // Adds an input port to this object. Returns its builder.
  PortConfigProtoBuilder WithInputPort(std::string_view name);

  // Adds an output port to this object. Returns its builder.
  PortConfigProtoBuilder WithOutputPort(std::string_view name);

  // Sets default Virtual Channels for input ports.
  RouterConfigProtoBuilder& SetDefaultVirtualChannelsForInputPort(
      std::optional<std::vector<std::string>> virtual_channels);

  // Sets default Virtual Channels for output ports.
  RouterConfigProtoBuilder& SetDefaultVirtualChannelsForOutputPort(
      std::optional<std::vector<std::string>> virtual_channels);

  // Returns the routing scheme configuration builder of this builder.
  RoutingSchemeConfigProtoBuilder GetRoutingSchemeConfigProtoBuilder();

  // Returns the arbiter scheme configuration builder of this builder.
  ArbiterSchemeConfigProtoBuilder GetArbiterSchemeConfigProtoBuilder();

 private:
  RouterConfigProto* proto_;
  // The members below are used to preserve the default state for the builder.
  // See corresponding methods above for details.
  std::optional<std::vector<std::string>> virtual_channels_for_input_;
  std::optional<std::vector<std::string>> virtual_channels_for_output_;
};

}  // namespace xls::noc

#endif  // XLS_NOC_CONFIG_ROUTER_CONFIG_PROTO_BUILDER_H_
