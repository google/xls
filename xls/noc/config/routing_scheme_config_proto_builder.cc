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

#include "xls/noc/config/routing_scheme_config_proto_builder.h"

#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder_utils.h"

namespace xls::noc {

RoutingSchemeConfigProtoBuilder&
RoutingSchemeConfigProtoBuilder::WithDistributedRoutingEntry(
    PortVirtualChannelTuple network_receive_port,
    PortVirtualChannelTuple router_output_port) {
  RouterConfigProto::RoutingTableEntryConfig* routing_table_entry_config =
      proto_->mutable_routing_table()->add_entries();
  routing_table_entry_config->mutable_network_receive_port()->set_port_name(
      network_receive_port.port_name);
  routing_table_entry_config->mutable_network_receive_port()
      ->set_virtual_channel_name(network_receive_port.virtual_channel_name);
  routing_table_entry_config->mutable_router_output_port()->set_port_name(
      router_output_port.port_name);
  routing_table_entry_config->mutable_router_output_port()
      ->set_virtual_channel_name(router_output_port.virtual_channel_name);
  return *this;
}

}  // namespace xls::noc
