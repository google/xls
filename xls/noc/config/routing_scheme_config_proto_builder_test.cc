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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/noc/config/network_config.pb.h"

namespace xls::noc {

// Test field values for distributed routing.
TEST(RoutingSchemeConfigBuilderTest, DistributedRoutingFieldValues) {
  const char* kNetworkPortName = "NetworkPort";
  const char* kNetworkVirtualChannelName = "VC0";
  const char* kRouterPortName = "RouterPort";
  const char* kRouterVirtualChannelName = "VC1";
  RouterConfigProto::RoutingSchemeConfigProto proto;
  RoutingSchemeConfigProtoBuilder builder(&proto);
  builder.WithDistributedRoutingEntry(
      {kNetworkPortName, kNetworkVirtualChannelName},
      {kRouterPortName, kRouterVirtualChannelName});

  EXPECT_TRUE(proto.has_routing_table());
  EXPECT_EQ(proto.routing_table().entries_size(), 1);
  const RouterConfigProto::RoutingTableEntryConfig& routing_table_entry_config =
      proto.routing_table().entries(0);
  EXPECT_THAT(routing_table_entry_config.network_receive_port().port_name(),
              kNetworkPortName);
  EXPECT_THAT(
      routing_table_entry_config.network_receive_port().virtual_channel_name(),
      kNetworkVirtualChannelName);
  EXPECT_THAT(routing_table_entry_config.router_output_port().port_name(),
              kRouterPortName);
  EXPECT_THAT(
      routing_table_entry_config.router_output_port().virtual_channel_name(),
      kRouterVirtualChannelName);
}

}  // namespace xls::noc
