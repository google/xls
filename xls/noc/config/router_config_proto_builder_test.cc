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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/noc/config/arbiter_scheme_config_proto_builder.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_proto_builder_utils.h"
#include "xls/noc/config/port_config_proto_builder.h"
#include "xls/noc/config/routing_scheme_config_proto_builder.h"

namespace xls::noc {

// Test the field values of a router with an input port.
TEST(RouterConfigBuilderTest, InputPortFieldValues) {
  constexpr std::string_view kName = "Test";
  constexpr std::string_view kRouterPortName = "RouterPort";
  RouterConfigProto proto;
  RouterConfigProtoBuilder builder(&proto);
  builder.WithName(kName);
  builder.WithInputPort(kRouterPortName);
  RoutingSchemeConfigProtoBuilder routing_scheme_config_builder =
      builder.GetRoutingSchemeConfigProtoBuilder();
  routing_scheme_config_builder.WithDistributedRoutingEntry(
      {"NetworkPort", "VC0"}, {"RouterPort", "VC0"});
  ArbiterSchemeConfigProtoBuilder arbiter_scheme_config_builder =
      builder.GetArbiterSchemeConfigProtoBuilder();
  std::vector<PortVirtualChannelTuple> priority_list;
  priority_list.push_back({"RouterInputPort", "VC0"});
  arbiter_scheme_config_builder.WithPriorityEntry(kRouterPortName,
                                                  priority_list);

  EXPECT_TRUE(proto.has_name());
  EXPECT_EQ(proto.ports_size(), 1);
  EXPECT_THAT(proto.ports(0).name(), kRouterPortName);
  EXPECT_TRUE(proto.has_routing_scheme());
  EXPECT_TRUE(proto.has_arbiter_scheme());
  EXPECT_THAT(proto.name(), kName);
}

// Test the field values of a router with an output port.
TEST(RouterConfigBuilderTest, OutputPortFieldValues) {
  constexpr std::string_view kName = "Test";
  constexpr std::string_view kRouterPortName = "RouterPort";
  RouterConfigProto proto;
  RouterConfigProtoBuilder builder(&proto);
  builder.WithName(kName);
  builder.WithOutputPort(kRouterPortName);
  RoutingSchemeConfigProtoBuilder routing_scheme_config_builder =
      builder.GetRoutingSchemeConfigProtoBuilder();
  routing_scheme_config_builder.WithDistributedRoutingEntry(
      {"NetworkPort", "VC0"}, {"RouterPort", "VC0"});
  ArbiterSchemeConfigProtoBuilder arbiter_scheme_config_builder =
      builder.GetArbiterSchemeConfigProtoBuilder();
  std::vector<PortVirtualChannelTuple> priority_list;
  priority_list.push_back({"RouterInputPort", "VC0"});
  arbiter_scheme_config_builder.WithPriorityEntry(kRouterPortName,
                                                  priority_list);

  EXPECT_TRUE(proto.has_name());
  EXPECT_EQ(proto.ports_size(), 1);
  EXPECT_THAT(proto.ports(0).name(), kRouterPortName);
  EXPECT_TRUE(proto.has_routing_scheme());
  EXPECT_TRUE(proto.has_arbiter_scheme());
  EXPECT_THAT(proto.name(), kName);
}

// Test the default virtual channels for an input port. The test creates an
// input port with non-default virtual channels, default virtual channels, then
// non-default virtual channels (in that order).
TEST(RouterConfigBuilderTest, VirtualChannelsInputPortDefaultFields) {
  constexpr std::string_view kNonDefault = "NonDefault";
  constexpr std::string_view kDefault = "Default";
  const std::vector<std::string> kVirtualChannels = {std::string(kDefault)};
  RouterConfigProto proto;
  RouterConfigProtoBuilder builder(&proto);
  PortConfigProtoBuilder port_builder = builder.WithInputPort("Test");
  port_builder.WithVirtualChannel(kNonDefault);
  builder.SetDefaultVirtualChannelsForInputPort(kVirtualChannels);
  builder.WithInputPort("Test");
  builder.SetDefaultVirtualChannelsForInputPort(std::nullopt);
  port_builder = builder.WithInputPort("Test");
  port_builder.WithVirtualChannel(kNonDefault);

  EXPECT_EQ(proto.ports_size(), 3);
  EXPECT_EQ(proto.ports(0).virtual_channels_size(), 1);
  EXPECT_EQ(proto.ports(0).virtual_channels(0), kNonDefault);
  EXPECT_EQ(proto.ports(1).virtual_channels_size(), 1);
  EXPECT_EQ(proto.ports(1).virtual_channels(0), kDefault);
  EXPECT_EQ(proto.ports(2).virtual_channels_size(), 1);
  EXPECT_EQ(proto.ports(2).virtual_channels(0), kNonDefault);
}

// Test the default virtual channels for an output port. The test creates an
// output port with non-default virtual channels, default virtual channels, then
// non-default virtual channels (in that order).
TEST(RouterConfigBuilderTest, VirtualChannelsOutputPortDefaultFields) {
  constexpr std::string_view kNonDefault = "NonDefault";
  constexpr std::string_view kDefault = "Default";
  const std::vector<std::string> kVirtualChannels = {std::string(kDefault)};
  RouterConfigProto proto;
  RouterConfigProtoBuilder builder(&proto);
  PortConfigProtoBuilder port_builder = builder.WithOutputPort("Test");
  port_builder.WithVirtualChannel(kNonDefault);
  builder.SetDefaultVirtualChannelsForOutputPort(kVirtualChannels);
  builder.WithOutputPort("Test");
  builder.SetDefaultVirtualChannelsForOutputPort(std::nullopt);
  port_builder = builder.WithOutputPort("Test");
  port_builder.WithVirtualChannel(kNonDefault);

  EXPECT_EQ(proto.ports_size(), 3);
  EXPECT_EQ(proto.ports(0).virtual_channels_size(), 1);
  EXPECT_EQ(proto.ports(0).virtual_channels(0), kNonDefault);
  EXPECT_EQ(proto.ports(1).virtual_channels_size(), 1);
  EXPECT_EQ(proto.ports(1).virtual_channels(0), kDefault);
  EXPECT_EQ(proto.ports(2).virtual_channels_size(), 1);
  EXPECT_EQ(proto.ports(2).virtual_channels(0), kNonDefault);
}

}  // namespace xls::noc
