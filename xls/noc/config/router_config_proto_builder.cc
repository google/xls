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

#include "xls/noc/config/router_config_proto_builder.h"

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "xls/common/proto_adaptor_utils.h"
#include "xls/noc/config/arbiter_scheme_config_proto_builder.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/port_config_proto_builder.h"
#include "xls/noc/config/routing_scheme_config_proto_builder.h"

namespace xls::noc {

RouterConfigProtoBuilder& RouterConfigProtoBuilder::WithName(
    std::string_view name) {
  proto_->set_name(xls::ToProtoString(name));
  return *this;
}

PortConfigProtoBuilder RouterConfigProtoBuilder::WithInputPort(
    std::string_view name) {
  PortConfigProto* port = proto_->add_ports();
  port->set_name(xls::ToProtoString(name));
  port->set_direction(PortConfigProto::INPUT);
  if (virtual_channels_for_input_.has_value()) {
    for (const std::string& virtual_channel :
         virtual_channels_for_input_.value()) {
      port->add_virtual_channels(virtual_channel);
    }
  }
  return PortConfigProtoBuilder(port);
}

PortConfigProtoBuilder RouterConfigProtoBuilder::WithOutputPort(
    std::string_view name) {
  PortConfigProto* port = proto_->add_ports();
  port->set_name(xls::ToProtoString(name));
  port->set_direction(PortConfigProto::OUTPUT);
  if (virtual_channels_for_output_.has_value()) {
    for (const std::string& virtual_channel :
         virtual_channels_for_output_.value()) {
      port->add_virtual_channels(virtual_channel);
    }
  }
  return PortConfigProtoBuilder(port);
}

RouterConfigProtoBuilder&
RouterConfigProtoBuilder::SetDefaultVirtualChannelsForInputPort(
    std::optional<std::vector<std::string>> virtual_channels) {
  virtual_channels_for_input_ = virtual_channels;
  return *this;
}

RouterConfigProtoBuilder&
RouterConfigProtoBuilder::SetDefaultVirtualChannelsForOutputPort(
    std::optional<std::vector<std::string>> virtual_channels) {
  virtual_channels_for_output_ = virtual_channels;
  return *this;
}

RoutingSchemeConfigProtoBuilder
RouterConfigProtoBuilder::GetRoutingSchemeConfigProtoBuilder() {
  return RoutingSchemeConfigProtoBuilder(proto_->mutable_routing_scheme());
}

ArbiterSchemeConfigProtoBuilder
RouterConfigProtoBuilder::GetArbiterSchemeConfigProtoBuilder() {
  return ArbiterSchemeConfigProtoBuilder(proto_->mutable_arbiter_scheme());
}

}  // namespace xls::noc
