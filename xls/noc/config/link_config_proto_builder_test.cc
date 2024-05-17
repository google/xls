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

#include "xls/noc/config/link_config_proto_builder.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/noc/config/network_config.pb.h"

namespace xls::noc {

// Helper function to populate a link with arbitrary values for the source port
// name, the sink port name, the phit bit width, the number of pipeline stage
// between the source and the sink, and, the number of pipeline stage between
// the sink and the source.
static LinkConfigProtoBuilder& PopulateFields(LinkConfigProtoBuilder& builder) {
  builder.WithSourcePort("Source");
  builder.WithSinkPort("Sink");
  builder.WithPhitBitWidth(42);
  builder.WithSourceSinkPipelineStage(4);
  builder.WithSinkSourcePipelineStage(2);
  return builder;
}

// Test the populate fields helper function
TEST(LinkConfigBuilderTest, PopulateFields) {
  LinkConfigProto proto;
  LinkConfigProtoBuilder builder(&proto);
  builder = PopulateFields(builder);

  EXPECT_FALSE(proto.has_name());
  EXPECT_TRUE(proto.has_source_port_name());
  EXPECT_TRUE(proto.has_sink_port_name());
  EXPECT_TRUE(proto.has_phit_bit_width());
  EXPECT_TRUE(proto.has_source_sink_pipeline_stage());
  EXPECT_TRUE(proto.has_sink_source_pipeline_stage());
  EXPECT_FALSE(proto.has_flow_control());
}

// Test the field values of a link.
TEST(LinkConfigBuilderTest, FieldValues) {
  const char* kName = "Test";
  const char* kSourceName = "Source";
  const char* kSinkName = "Sink";
  const int64_t kPhitBitWidth = 42;
  const int64_t kSourceSinkPipelineStage = 4;
  const int64_t kSinkSourcePipelineStage = 2;
  LinkConfigProto proto;
  LinkConfigProtoBuilder builder(&proto);
  builder.WithName(kName);
  builder.WithSourcePort(kSourceName);
  builder.WithSinkPort(kSinkName);
  builder.WithPhitBitWidth(kPhitBitWidth);
  builder.WithSourceSinkPipelineStage(kSourceSinkPipelineStage);
  builder.WithSinkSourcePipelineStage(kSinkSourcePipelineStage);

  EXPECT_TRUE(proto.has_name());
  EXPECT_TRUE(proto.has_source_port_name());
  EXPECT_TRUE(proto.has_sink_port_name());
  EXPECT_TRUE(proto.has_phit_bit_width());
  EXPECT_TRUE(proto.has_source_sink_pipeline_stage());
  EXPECT_TRUE(proto.has_sink_source_pipeline_stage());
  EXPECT_FALSE(proto.has_flow_control());

  EXPECT_THAT(proto.name(), kName);
  EXPECT_THAT(proto.source_port_name(), kSourceName);
  EXPECT_THAT(proto.sink_port_name(), kSinkName);
  EXPECT_EQ(proto.phit_bit_width(), kPhitBitWidth);
  EXPECT_EQ(proto.source_sink_pipeline_stage(), kSourceSinkPipelineStage);
  EXPECT_EQ(proto.sink_source_pipeline_stage(), kSinkSourcePipelineStage);
}

// Test enabling peek flow control of a link.
TEST(LinkConfigBuilderTest, PeekFlowControl) {
  LinkConfigProto proto;
  LinkConfigProtoBuilder builder(&proto);
  builder.WithName("Test");
  builder = PopulateFields(builder);
  builder.WithPeekFlowControl();

  EXPECT_TRUE(proto.has_flow_control());

  EXPECT_TRUE(proto.flow_control().has_peek());
  EXPECT_FALSE(proto.flow_control().has_token_credit_based());
  EXPECT_FALSE(proto.flow_control().has_total_credit_based());
}

// Test enabling token credit flow control of a link.
TEST(LinkConfigBuilderTest, TokenCreditFlowControl) {
  LinkConfigProto proto;
  LinkConfigProtoBuilder builder(&proto);
  builder.WithName("Test");
  builder = PopulateFields(builder);
  builder.WithTokenCreditBasedFlowControl();

  EXPECT_TRUE(proto.has_flow_control());

  EXPECT_FALSE(proto.flow_control().has_peek());
  EXPECT_TRUE(proto.flow_control().has_token_credit_based());
  EXPECT_FALSE(proto.flow_control().has_total_credit_based());
}

// Test enabling total credit flow control of a link.
TEST(LinkConfigBuilderTest, TotalCreditFlowControl) {
  const int64_t kCreditBitWidth = 42;
  LinkConfigProto proto;
  LinkConfigProtoBuilder builder(&proto);
  builder.WithName("Test");
  builder = PopulateFields(builder);
  builder.WithTotalCreditBasedFlowControl(kCreditBitWidth);

  EXPECT_TRUE(proto.has_flow_control());
  const LinkConfigProto::FlowControlConfigProto& flow_control_config_proto =
      proto.flow_control();

  EXPECT_FALSE(flow_control_config_proto.has_peek());
  EXPECT_FALSE(flow_control_config_proto.has_token_credit_based());
  EXPECT_TRUE(flow_control_config_proto.has_total_credit_based());
  EXPECT_TRUE(
      flow_control_config_proto.total_credit_based().has_credit_bit_width());
  EXPECT_EQ(flow_control_config_proto.total_credit_based().credit_bit_width(),
            kCreditBitWidth);
}

// Test last enabled flow control of a link. Tests valid flow control method
// after enabling each flow contorl method.
TEST(LinkConfigBuilderTest, LastEnabledFlowControl) {
  LinkConfigProto proto;
  LinkConfigProtoBuilder builder(&proto);

  EXPECT_FALSE(proto.has_flow_control());
  builder.WithPeekFlowControl();
  EXPECT_TRUE(proto.has_flow_control());
  const LinkConfigProto::FlowControlConfigProto& flow_control_config_proto =
      proto.flow_control();
  EXPECT_TRUE(flow_control_config_proto.has_peek());
  builder.WithTokenCreditBasedFlowControl();
  EXPECT_TRUE(flow_control_config_proto.has_token_credit_based());
  builder.WithTotalCreditBasedFlowControl(1337);
  EXPECT_TRUE(flow_control_config_proto.has_total_credit_based());
}

}  // namespace xls::noc
