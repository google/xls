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

#include "xls/noc/config/network_config_proto_builder.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/noc/config/link_config_proto_builder.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/router_config_proto_builder.h"
#include "xls/noc/config/virtual_channel_config_proto_builder.h"

namespace xls::noc {

// Test field values for an empty network.
TEST(NetworkConfigBuilderTest, FieldValuesEmptyNetwork) {
  NetworkConfigProtoBuilder builder;

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  EXPECT_FALSE(network.has_name());
  EXPECT_FALSE(network.has_description());
  EXPECT_EQ(network.ports_size(), 0);
  EXPECT_EQ(network.routers_size(), 0);
  EXPECT_EQ(network.links_size(), 0);
  EXPECT_EQ(network.virtual_channels_size(), 0);
}

// Test the field values of a network.
TEST(NetworkConfigBuilderTest, FieldValues) {
  constexpr std::string_view kName = "Test";
  NetworkConfigProtoBuilder builder(kName);
  builder.WithDescription(kName);
  builder.WithPort(kName);
  builder.WithRouter(kName);
  builder.WithLink(kName);
  builder.WithVirtualChannel(kName);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  EXPECT_TRUE(network.has_name());
  EXPECT_TRUE(network.has_description());
  EXPECT_THAT(network.name(), kName);
  EXPECT_THAT(network.description(), kName);
  EXPECT_EQ(network.ports_size(), 1);
  EXPECT_THAT(network.ports(0).name(), kName);
  EXPECT_EQ(network.routers_size(), 1);
  EXPECT_THAT(network.routers(0).name(), kName);
  EXPECT_EQ(network.links_size(), 1);
  EXPECT_THAT(network.links(0).name(), kName);
  EXPECT_EQ(network.virtual_channels_size(), 1);
  EXPECT_THAT(network.virtual_channels(0).name(), kName);
}

// Test the default phit bit width mode for links. The test creates a link in
// non-default mode, default mode, then non-default mode (in that order).
TEST(NetworkConfigBuilderTest, LinkDefaultPhitBitWidthField) {
  const int64_t kNonDefault = 2;
  const int64_t kDefault = 42;
  NetworkConfigProtoBuilder builder("Test");
  LinkConfigProtoBuilder link_builder0 = builder.WithLink("Link0");
  link_builder0.WithPhitBitWidth(kNonDefault);
  builder.SetDefaultLinkPhitBitWidth(kDefault);
  builder.WithLink("Link1");
  builder.SetDefaultLinkPhitBitWidth(std::nullopt);
  LinkConfigProtoBuilder link_builder2 = builder.WithLink("Link2");
  link_builder2.WithPhitBitWidth(kNonDefault);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  EXPECT_EQ(network.links_size(), 3);
  EXPECT_EQ(network.links(0).phit_bit_width(), kNonDefault);
  EXPECT_EQ(network.links(1).phit_bit_width(), kDefault);
  EXPECT_EQ(network.links(2).phit_bit_width(), kNonDefault);
}

// Test the default source-sink pipeline stage mode for links. The test creates
// a link in non-default mode, default mode, then non-default mode (in that
// order).
TEST(NetworkConfigBuilderTest, LinkDefaultSourceSinkPipelineStageField) {
  const int64_t kNonDefault = 2;
  const int64_t kDefault = 42;
  NetworkConfigProtoBuilder builder("Test");
  LinkConfigProtoBuilder link_builder0 = builder.WithLink("Link0");
  link_builder0.WithSourceSinkPipelineStage(kNonDefault);
  builder.SetDefaultLinkSourceSinkPipelineStage(kDefault);
  builder.WithLink("Link1");
  builder.SetDefaultLinkPhitBitWidth(std::nullopt);
  LinkConfigProtoBuilder link_builder2 = builder.WithLink("Link2");
  link_builder2.WithSourceSinkPipelineStage(kNonDefault);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  EXPECT_EQ(network.links_size(), 3);
  EXPECT_EQ(network.links(0).source_sink_pipeline_stage(), kNonDefault);
  EXPECT_EQ(network.links(1).source_sink_pipeline_stage(), kDefault);
  EXPECT_EQ(network.links(2).source_sink_pipeline_stage(), kNonDefault);
}

// Test the default sink-source pipeline stage mode for links. The test creates
// a link in non-default mode, default mode, then non-default mode (in that
// order).
TEST(NetworkConfigBuilderTest, LinkDefaultSinkSourcePipelineStageField) {
  const int64_t kNonDefault = 2;
  const int64_t kDefault = 42;
  NetworkConfigProtoBuilder builder("Test");
  LinkConfigProtoBuilder link_builder0 = builder.WithLink("Link0");
  link_builder0.WithSinkSourcePipelineStage(kNonDefault);
  builder.SetDefaultLinkSinkSourcePipelineStage(kDefault);
  builder.WithLink("Link1");
  builder.SetDefaultLinkPhitBitWidth(std::nullopt);
  LinkConfigProtoBuilder link_builder2 = builder.WithLink("Link2");
  link_builder2.WithSinkSourcePipelineStage(kNonDefault);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  EXPECT_EQ(network.links_size(), 3);
  EXPECT_EQ(network.links(0).sink_source_pipeline_stage(), kNonDefault);
  EXPECT_EQ(network.links(1).sink_source_pipeline_stage(), kDefault);
  EXPECT_EQ(network.links(2).sink_source_pipeline_stage(), kNonDefault);
}

// Test the default flit bit width mode for virtual channels. The test creates a
// virtual channel in non-default mode, default mode, then non-default mode (in
// that order).
TEST(NetworkConfigBuilderTest, VirtualChannelFlitBitWidthDefaultFields) {
  const int64_t kNonDefault = 2;
  const int64_t kDefault = 42;
  NetworkConfigProtoBuilder builder("Test");
  VirtualChannelConfigProtoBuilder virtual_channel_builder0 =
      builder.WithVirtualChannel("VC0");
  virtual_channel_builder0.WithFlitBitWidth(kNonDefault);
  builder.SetDefaultVirtualChannelFlitBitWidth(kDefault);
  builder.WithVirtualChannel("VC1");
  builder.SetDefaultVirtualChannelFlitBitWidth(std::nullopt);
  VirtualChannelConfigProtoBuilder virtual_channel_builder2 =
      builder.WithVirtualChannel("VC2");
  virtual_channel_builder2.WithFlitBitWidth(kNonDefault);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  EXPECT_EQ(network.virtual_channels_size(), 3);
  EXPECT_EQ(network.virtual_channels(0).flit_bit_width(), kNonDefault);
  EXPECT_EQ(network.virtual_channels(1).flit_bit_width(), kDefault);
  EXPECT_EQ(network.virtual_channels(2).flit_bit_width(), kNonDefault);
}

// Test the default depth mode for virtual channels. The test creates a virtual
// channel in non-default mode, default mode, then non-default mode (in that
// order).
TEST(NetworkConfigBuilderTest, VirtualChannelDepthDefaultFields) {
  const int64_t kNonDefault = 2;
  const int64_t kDefault = 42;
  NetworkConfigProtoBuilder builder("Test");
  VirtualChannelConfigProtoBuilder virtual_channel_builder0 =
      builder.WithVirtualChannel("VC0");
  virtual_channel_builder0.WithDepth(kNonDefault);
  builder.SetDefaultVirtualChannelDepth(kDefault);
  builder.WithVirtualChannel("VC1");
  builder.SetDefaultVirtualChannelDepth(std::nullopt);
  VirtualChannelConfigProtoBuilder virtual_channel_builder2 =
      builder.WithVirtualChannel("VC2");
  virtual_channel_builder2.WithDepth(kNonDefault);

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  EXPECT_EQ(network.virtual_channels_size(), 3);
  EXPECT_EQ(network.virtual_channels(0).depth(), kNonDefault);
  EXPECT_EQ(network.virtual_channels(1).depth(), kDefault);
  EXPECT_EQ(network.virtual_channels(2).depth(), kNonDefault);
}

// Test validates the creation of a simple network with two endpoint ports, a
// router and two links.
TEST(NetworkConfigBuilderTest, SimpleNetwork) {
  NetworkConfigProtoBuilder builder("Test");
  builder.WithPort("SendPort");
  builder.WithPort("RecvPort");
  builder.WithRouter("Router0");
  builder.WithLink("Link0");
  builder.WithLink("Link1");

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  ASSERT_EQ(network.ports_size(), 2);
  ASSERT_EQ(network.routers_size(), 1);
  ASSERT_EQ(network.links_size(), 2);
}

// Test the default virtual channels for an input port.
TEST(NetworkConfigBuilderTest, VirtualChannelsInputPortDefaultFields) {
  constexpr std::string_view kDefault = "Default";
  const std::vector<std::string> kVirtualChannels = {std::string(kDefault)};
  NetworkConfigProtoBuilder builder("Test");
  RouterConfigProtoBuilder router_builder = builder.WithRouter("Test");
  router_builder.WithInputPort("Test");
  builder.SetDefaultVirtualChannelsForRouterInputPort(kVirtualChannels);
  router_builder = builder.WithRouter("Test");
  router_builder.WithInputPort("Test");

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  EXPECT_EQ(network.routers_size(), 2);
  EXPECT_EQ(network.routers(0).ports_size(), 1);
  EXPECT_EQ(network.routers(0).ports(0).virtual_channels_size(), 0);
  EXPECT_EQ(network.routers(1).ports_size(), 1);
  EXPECT_EQ(network.routers(1).ports_size(), 1);
  EXPECT_EQ(network.routers(1).ports(0).virtual_channels_size(), 1);
  EXPECT_EQ(network.routers(1).ports(0).virtual_channels(0), kDefault);
}

// Test the default virtual channels for an output port.
TEST(NetworkConfigBuilderTest, VirtualChannelsOutputPortDefaultFields) {
  constexpr std::string_view kDefault = "Default";
  const std::vector<std::string> kVirtualChannels = {std::string(kDefault)};
  NetworkConfigProtoBuilder builder("Test");
  RouterConfigProtoBuilder router_builder = builder.WithRouter("Test");
  router_builder.WithOutputPort("Test");
  builder.SetDefaultVirtualChannelsForRouterOutputPort(kVirtualChannels);
  router_builder = builder.WithRouter("Test");
  router_builder.WithOutputPort("Test");

  XLS_ASSERT_OK_AND_ASSIGN(NetworkConfigProto network, builder.Build());
  EXPECT_EQ(network.routers_size(), 2);
  EXPECT_EQ(network.routers(0).ports_size(), 1);
  EXPECT_EQ(network.routers(0).ports(0).virtual_channels_size(), 0);
  EXPECT_EQ(network.routers(1).ports_size(), 1);
  EXPECT_EQ(network.routers(1).ports_size(), 1);
  EXPECT_EQ(network.routers(1).ports(0).virtual_channels_size(), 1);
  EXPECT_EQ(network.routers(1).ports(0).virtual_channels(0), kDefault);
}

}  // namespace xls::noc
