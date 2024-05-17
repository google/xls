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

#include "xls/noc/config/common_network_config_builder_options_proto_builder.h"

#include <cstdint>

#include "gtest/gtest.h"
#include "xls/noc/config/network_config.pb.h"
#include "xls/noc/config/network_config_builder_options.pb.h"

namespace xls::noc {
namespace {

// Test field values for an arbiter scheme option.
TEST(ArbiterSchemeOptionsProtoBuilderTest, EnablePriority) {
  RouterOptionsProto::ArbiterSchemeOptionsProto proto;
  ArbiterSchemeOptionsProtoBuilder arbiter_scheme_options_proto_builder(&proto);
  EXPECT_FALSE(proto.has_priority());
  arbiter_scheme_options_proto_builder.EnablePriority();
  EXPECT_TRUE(proto.has_priority());
}

// Test field values for a endpoint options.
TEST(EndpointOptionsProtoBuilderTest, FieldValues) {
  const int64_t kNumSendPorts = 42;
  const int64_t kNumRecvPorts = 1337;
  EndpointOptionsProto proto;
  EndpointOptionsProtoBuilder builder(&proto);
  builder.WithNumSendPorts(kNumSendPorts);
  builder.WithNumRecvPorts(kNumRecvPorts);

  EXPECT_TRUE(proto.has_num_send_ports());
  EXPECT_TRUE(proto.has_num_recv_ports());
  EXPECT_EQ(proto.num_send_ports(), kNumSendPorts);
  EXPECT_EQ(proto.num_recv_ports(), kNumRecvPorts);
}

// Test field values for a data options.
TEST(DataOptionsProtoBuilderTest, FieldValues) {
  const int64_t kDataBitWidth = 42;
  DataOptionsProto proto;
  DataOptionsProtoBuilder builder(&proto);
  builder.WithDataBitWidth(kDataBitWidth);

  EXPECT_TRUE(proto.has_data_bit_width());
  EXPECT_EQ(proto.data_bit_width(), kDataBitWidth);
}

// Test field values for a peek flow control option.
TEST(FlowControlOptionsProtoBuilderTest, PeekFlowControlFieldValues) {
  LinkConfigProto::FlowControlConfigProto proto;
  FlowControlOptionsProtoBuilder builder(&proto);
  builder.EnablePeekFlowControl();

  EXPECT_TRUE(proto.has_peek());
  EXPECT_FALSE(proto.has_token_credit_based());
  EXPECT_FALSE(proto.has_total_credit_based());
}

// Test field values for a token credit flow control option.
TEST(FlowControlOptionsProtoBuilderTest, TokenCreditFlowControlFieldValues) {
  LinkConfigProto::FlowControlConfigProto proto;
  FlowControlOptionsProtoBuilder builder(&proto);
  builder.EnableTokenCreditBasedFlowControl();

  EXPECT_FALSE(proto.has_peek());
  EXPECT_TRUE(proto.has_token_credit_based());
  EXPECT_FALSE(proto.has_total_credit_based());
}

// Test field values for a total credit flow control option.
TEST(FlowControlOptionsProtoBuilderTest, TotalCreditFlowControlFieldValues) {
  const int64_t kCreditBitWidth = 42;
  LinkConfigProto::FlowControlConfigProto proto;
  FlowControlOptionsProtoBuilder builder(&proto);
  builder.EnableTotalCreditBasedFlowControl(kCreditBitWidth);

  EXPECT_FALSE(proto.has_peek());
  EXPECT_FALSE(proto.has_token_credit_based());
  EXPECT_TRUE(proto.has_total_credit_based());
  EXPECT_EQ(proto.total_credit_based().credit_bit_width(), kCreditBitWidth);
}

// Test last enabled flow control option.
TEST(FlowControlOptionsProtoBuilderTest, LastEnabledFlowControl) {
  LinkConfigProto::FlowControlConfigProto proto;
  FlowControlOptionsProtoBuilder builder(&proto);
  EXPECT_FALSE(proto.has_peek());
  EXPECT_FALSE(proto.has_token_credit_based());
  EXPECT_FALSE(proto.has_total_credit_based());
  builder.EnablePeekFlowControl();
  EXPECT_TRUE(proto.has_peek());
  builder.EnableTokenCreditBasedFlowControl();
  EXPECT_TRUE(proto.has_token_credit_based());
  builder.EnableTotalCreditBasedFlowControl(1337);
  EXPECT_TRUE(proto.has_total_credit_based());
}

// Test field values for a link option.
TEST(LinkOptionsProtoBuilderTest, FieldValues) {
  const int64_t kSourceSinkPipelineStage = 4;
  const int64_t kSinkSourcePipelineStage = 2;
  LinkOptionsProto proto;
  LinkOptionsProtoBuilder link_options_proto_builder(&proto);
  link_options_proto_builder.WithSourceSinkPipelineStage(
      kSourceSinkPipelineStage);
  link_options_proto_builder.WithSinkSourcePipelineStage(
      kSinkSourcePipelineStage);
  link_options_proto_builder.GetFlowControlOptionsProtoBuilder();

  EXPECT_TRUE(proto.has_source_sink_pipeline_stage());
  EXPECT_TRUE(proto.has_sink_source_pipeline_stage());
  EXPECT_TRUE(proto.has_flow_control());
}

// Test field values for a virtual channel option.
TEST(VirtualChannelOptionsProtoBuilderTest, FieldValues) {
  const int64_t kDepth0 = 42;
  const int64_t kDepth1 = 4;
  const int64_t kDepth2 = 2;
  VirtualChannelOptionsProto proto;
  VirtualChannelOptionsProtoBuilder virtual_channel_options_proto_builder(
      &proto);
  virtual_channel_options_proto_builder.WithVirtualChannelDepth(kDepth0);
  virtual_channel_options_proto_builder.WithVirtualChannelDepth(kDepth1);
  virtual_channel_options_proto_builder.WithVirtualChannelDepth(kDepth2);

  EXPECT_EQ(proto.virtual_channel_depth_size(), 3);
  EXPECT_EQ(proto.virtual_channel_depth(0), kDepth0);
  EXPECT_EQ(proto.virtual_channel_depth(1), kDepth1);
  EXPECT_EQ(proto.virtual_channel_depth(2), kDepth2);
}

// Test field values for a routing scheme option.
TEST(RoutingSchemeOptionsProtoBuilderTest, EnableDistributedRouting) {
  RouterOptionsProto::RoutingSchemeOptionsProto proto;
  RoutingSchemeOptionsProtoBuilder routing_scheme_options_proto_builder(&proto);
  EXPECT_FALSE(proto.has_routing_table());
  routing_scheme_options_proto_builder.EnableDistributedRouting();
  EXPECT_TRUE(proto.has_routing_table());
}

// Test field values for a router option.
TEST(RouterOptionsProtoBuilderTest, FieldValues) {
  RouterOptionsProto proto;
  RouterOptionsProtoBuilder router_options_proto_builder(&proto);
  EXPECT_FALSE(proto.has_routing_scheme());
  EXPECT_FALSE(proto.has_arbiter_scheme());
  router_options_proto_builder.GetRoutingSchemeOptionsProtoBuilder();
  router_options_proto_builder.GetArbiterSchemeOptionsProtoBuilder();
  EXPECT_TRUE(proto.has_routing_scheme());
  EXPECT_TRUE(proto.has_arbiter_scheme());
}

}  // namespace
}  // namespace xls::noc
