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

#include "xls/codegen/module_signature.h"

#include <optional>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/types/span.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::Property;

namespace m = ::xls::op_matchers;

std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

TEST(ModuleSignatureTest, SimpledFixedLatencyInterface) {
  ModuleSignatureBuilder b(TestName());

  b.AddDataInputAsBits("x", 42)
      .AddDataOutputAsBits("y", 2)
      .WithFixedLatencyInterface(123);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());
  ASSERT_EQ(signature.data_inputs().size(), 1);
  EXPECT_EQ(signature.data_inputs().front().width(), 42);
  EXPECT_EQ(signature.data_inputs().front().name(), "x");
  EXPECT_EQ(signature.TotalDataInputBits(), 42);

  ASSERT_EQ(signature.data_outputs().size(), 1);
  EXPECT_EQ(signature.data_outputs().front().width(), 2);
  EXPECT_EQ(signature.data_outputs().front().name(), "y");
  EXPECT_EQ(signature.TotalDataOutputBits(), 2);

  ASSERT_TRUE(signature.proto().has_fixed_latency());
  EXPECT_EQ(signature.proto().fixed_latency().latency(), 123);
}

TEST(ModuleSignatureTest, RemoveData) {
  ModuleSignatureBuilder b(TestName());

  b.AddDataInputAsBits("x", 42).AddDataInputAsBits("y", 32);
  XLS_ASSERT_OK(b.RemoveData("y"));
  b.AddDataOutputAsBits("y", 2).WithFixedLatencyInterface(123);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());
  ASSERT_EQ(signature.data_inputs().size(), 1);
  EXPECT_EQ(signature.data_inputs().front().width(), 42);
  EXPECT_EQ(signature.data_inputs().front().name(), "x");
  EXPECT_EQ(signature.TotalDataInputBits(), 42);

  ASSERT_EQ(signature.data_outputs().size(), 1);
  EXPECT_EQ(signature.data_outputs().front().width(), 2);
  EXPECT_EQ(signature.data_outputs().front().name(), "y");
  EXPECT_EQ(signature.TotalDataOutputBits(), 2);

  ASSERT_TRUE(signature.proto().has_fixed_latency());
  EXPECT_EQ(signature.proto().fixed_latency().latency(), 123);
}

TEST(ModuleSignatureTest, PipelineInterface) {
  ModuleSignatureBuilder b(TestName());

  b.WithPipelineInterface(/*latency=*/2, /*initiation_interval=*/3)
      .WithClock("clk")
      .AddDataInputAsBits("in", 4)
      .AddDataOutputAsBits("out", 5);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());
  ASSERT_TRUE(signature.proto().has_pipeline());
  EXPECT_EQ(signature.proto().pipeline().latency(), 2);
  EXPECT_EQ(signature.proto().pipeline().initiation_interval(), 3);
}

TEST(ModuleSignatureTest, PipelineInterfaceMissingClock) {
  ModuleSignatureBuilder b(TestName());

  b.WithPipelineInterface(/*latency=*/2, /*initiation_interval=*/3)
      .AddDataInputAsBits("in", 4)
      .AddDataOutputAsBits("out", 5);

  EXPECT_THAT(b.Build(), StatusIs(absl::StatusCode::kInvalidArgument,
                                  HasSubstr("Missing clock")));
}

TEST(ModuleSignatureTest, ToKwargs) {
  ModuleSignatureBuilder b(TestName());
  b.AddDataInputAsBits("x", 42)
      .AddDataInputAsBits("y", 2)
      .AddDataOutputAsBits("z", 32)
      .WithFixedLatencyInterface(123);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  absl::flat_hash_map<std::string, Value> kwargs;
  XLS_ASSERT_OK_AND_ASSIGN(
      kwargs, signature.ToKwargs({Value(UBits(7, 42)), Value(UBits(0, 2))}));
  EXPECT_THAT(kwargs, testing::UnorderedElementsAre(
                          testing::Pair("x", Value(UBits(7, 42))),
                          testing::Pair("y", Value(UBits(0, 2)))));
}

TEST(ModuleSignatureTest, SingleValueChannelsInterface) {
  ModuleSignatureBuilder b(TestName());

  b.AddDataInputAsBits("single_val_in_port", 32);
  b.AddDataOutputAsBits("single_val_out_port", 64);

  ChannelMetadataProto single_val_in_metadata;
  BlockPortMappingProto* block_port_mapping =
      single_val_in_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("single_val_in_port");
  b.AddSingleValueChannel("single_val_in", ChannelOps::kReceiveOnly,
                          single_val_in_metadata);
  ChannelMetadataProto single_val_out_metadata;
  block_port_mapping = single_val_out_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("single_val_out_port");
  b.AddSingleValueChannel("single_val_out", ChannelOps::kSendOnly,
                          single_val_out_metadata);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  ASSERT_EQ(signature.single_value_channels().size(), 2);
  ASSERT_EQ(signature.streaming_channels().size(), 0);

  EXPECT_EQ(signature.single_value_channels().at(0).name(), "single_val_in");
  EXPECT_EQ(signature.single_value_channels().at(0).kind(),
            CHANNEL_KIND_SINGLE_VALUE);
  EXPECT_EQ(signature.single_value_channels().at(0).supported_ops(),
            CHANNEL_OPS_RECEIVE_ONLY);
  EXPECT_EQ(signature.single_value_channels().at(0).flow_control(),
            CHANNEL_FLOW_CONTROL_NONE);
  EXPECT_THAT(signature.single_value_channels().at(0).metadata().block_ports(),
              ElementsAre(Property(&BlockPortMappingProto::data_port_name,
                                   "single_val_in_port")));

  EXPECT_EQ(signature.single_value_channels().at(1).name(), "single_val_out");
  EXPECT_EQ(signature.single_value_channels().at(1).kind(),
            CHANNEL_KIND_SINGLE_VALUE);
  EXPECT_EQ(signature.single_value_channels().at(1).supported_ops(),
            CHANNEL_OPS_SEND_ONLY);
  EXPECT_EQ(signature.single_value_channels().at(1).flow_control(),
            CHANNEL_FLOW_CONTROL_NONE);
  EXPECT_THAT(signature.single_value_channels().at(1).metadata().block_ports(),
              ElementsAre(Property(&BlockPortMappingProto::data_port_name,
                                   "single_val_out_port")));
}

TEST(ModuleSignatureTest, StreamingChannelsInterface) {
  Package p(TestName());
  ModuleSignatureBuilder b(TestName());

  // Add ports for streaming channels.
  b.AddDataInputAsBits("streaming_in_data", 24);
  b.AddDataInputAsBits("streaming_in_valid", 1);
  b.AddDataOutputAsBits("streaming_in_ready", 1);

  b.AddDataOutputAsBits("streaming_out_data", 16);

  ChannelMetadataProto streaming_in_metadata;
  BlockPortMappingProto* block_port_mapping =
      streaming_in_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("streaming_in_data");
  block_port_mapping->set_valid_port_name("streaming_in_valid");
  block_port_mapping->set_ready_port_name("streaming_in_ready");
  b.AddStreamingChannel(
      "streaming_in", ChannelOps::kReceiveOnly, FlowControl::kReadyValid,
      p.GetTupleType({p.GetBitsType(32)}),
      /*fifo_config=*/
      FifoConfig(/*depth=*/42, /*bypass=*/true, /*register_push_outputs=*/true,
                 /*register_pop_outputs=*/false),
      streaming_in_metadata);

  ChannelMetadataProto streaming_out_metadata;
  block_port_mapping = streaming_out_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("streaming_out_data");
  b.AddStreamingChannel("streaming_out", ChannelOps::kSendOnly,
                        FlowControl::kNone, p.GetBitsType(32),
                        /*fifo_config=*/std::nullopt, streaming_out_metadata);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  ASSERT_EQ(signature.single_value_channels().size(), 0);
  ASSERT_EQ(signature.streaming_channels().size(), 2);

  EXPECT_EQ(signature.streaming_channels().at(0).name(), "streaming_in");
  EXPECT_EQ(signature.streaming_channels().at(0).kind(),
            CHANNEL_KIND_STREAMING);
  EXPECT_EQ(signature.streaming_channels().at(0).supported_ops(),
            CHANNEL_OPS_RECEIVE_ONLY);
  EXPECT_EQ(signature.streaming_channels().at(0).flow_control(),
            CHANNEL_FLOW_CONTROL_READY_VALID);
  EXPECT_THAT(p.GetTypeFromProto(signature.streaming_channels().at(0).type()),
              IsOkAndHolds(p.GetTupleType({p.GetBitsType(32)})));
  EXPECT_EQ(signature.streaming_channels().at(0).fifo_config().depth(), 42);
  EXPECT_THAT(
      signature.streaming_channels().at(0).metadata().block_ports(),
      ElementsAre(AllOf(
          Property(&BlockPortMappingProto::data_port_name, "streaming_in_data"),
          Property(&BlockPortMappingProto::valid_port_name,
                   "streaming_in_valid"),
          Property(&BlockPortMappingProto::ready_port_name,
                   "streaming_in_ready"))));

  EXPECT_EQ(signature.streaming_channels().at(1).name(), "streaming_out");
  EXPECT_EQ(signature.streaming_channels().at(1).kind(),
            CHANNEL_KIND_STREAMING);
  EXPECT_EQ(signature.streaming_channels().at(1).supported_ops(),
            CHANNEL_OPS_SEND_ONLY);
  EXPECT_EQ(signature.streaming_channels().at(1).flow_control(),
            CHANNEL_FLOW_CONTROL_NONE);
  EXPECT_THAT(p.GetTypeFromProto(signature.streaming_channels().at(1).type()),
              IsOkAndHolds(p.GetBitsType(32)));
  EXPECT_FALSE(signature.streaming_channels().at(1).has_fifo_config());
  EXPECT_THAT(
      signature.streaming_channels().at(1).metadata().block_ports(),
      ElementsAre(
          AllOf(Property(&BlockPortMappingProto::data_port_name,
                         "streaming_out_data"),
                Property(&BlockPortMappingProto::has_valid_port_name, false),
                Property(&BlockPortMappingProto::has_ready_port_name, false))));
}

TEST(ModuleSignatureTest, GetByName) {
  Package p(TestName());
  ModuleSignatureBuilder b(TestName());

  // Add ports for streaming channels.
  b.AddDataInputAsBits("streaming_in_data", 24);
  b.AddDataInputAsBits("streaming_in_valid", 1);
  b.AddDataOutputAsBits("streaming_in_ready", 1);

  ChannelMetadataProto streaming_in_metadata;
  BlockPortMappingProto* block_port_mapping =
      streaming_in_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("streaming_in_data");
  block_port_mapping->set_valid_port_name("streaming_in_valid");
  block_port_mapping->set_ready_port_name("streaming_in_ready");
  b.AddStreamingChannel(
      "streaming_in", ChannelOps::kReceiveOnly, FlowControl::kReadyValid,
      p.GetBitsType(24),
      /*fifo_config=*/
      FifoConfig(/*depth=*/42, /*bypass=*/true, /*register_push_outputs=*/true,
                 /*register_pop_outputs=*/false),
      streaming_in_metadata);

  b.AddDataOutputAsBits("single_val_out_port", 64);

  ChannelMetadataProto single_val_out_metadata;
  block_port_mapping = single_val_out_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("single_val_out_port");
  b.AddSingleValueChannel("single_val_out", ChannelOps::kSendOnly,
                          single_val_out_metadata);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  XLS_EXPECT_OK(signature.GetInputPortProtoByName("streaming_in_data"));
  XLS_EXPECT_OK(signature.GetInputPortProtoByName("streaming_in_valid"));
  XLS_EXPECT_OK(signature.GetOutputPortProtoByName("streaming_in_ready"));

  XLS_EXPECT_OK(signature.GetInputChannelProtoByName("streaming_in"));

  XLS_EXPECT_OK(signature.GetOutputPortProtoByName("single_val_out_port"));
  XLS_EXPECT_OK(signature.GetOutputChannelProtoByName("single_val_out"));

  // Test that a port/channel that is an input is not an output, and vice versa.
  EXPECT_THAT(
      signature.GetOutputPortProtoByName("streaming_in_data"),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("is not an output")));
  EXPECT_THAT(
      signature.GetOutputPortProtoByName("streaming_in_valid"),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("is not an output")));
  EXPECT_THAT(
      signature.GetInputPortProtoByName("streaming_in_ready"),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("is not an input")));

  EXPECT_THAT(
      signature.GetOutputChannelProtoByName("streaming_in"),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("is not an output")));

  EXPECT_THAT(
      signature.GetInputPortProtoByName("single_val_out_port"),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("is not an input")));
  EXPECT_THAT(
      signature.GetInputChannelProtoByName("single_val_out"),
      StatusIs(absl::StatusCode::kNotFound, HasSubstr("is not an input")));
}

TEST(ModuleSignatureTest, GetChannels) {
  Package p(TestName());
  ModuleSignatureBuilder b(TestName());

  // Add ports for streaming channels.
  b.AddDataInputAsBits("streaming_in_data", 24);
  b.AddDataInputAsBits("streaming_in_valid", 1);
  b.AddDataOutputAsBits("streaming_in_ready", 1);

  ChannelMetadataProto streaming_in_metadata;
  BlockPortMappingProto* block_port_mapping =
      streaming_in_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("streaming_in_data");
  block_port_mapping->set_valid_port_name("streaming_in_valid");
  block_port_mapping->set_ready_port_name("streaming_in_ready");
  b.AddStreamingChannel(
      "streaming_in", ChannelOps::kReceiveOnly, FlowControl::kReadyValid,
      p.GetBitsType(24),
      /*fifo_config=*/
      FifoConfig(/*depth=*/42, /*bypass=*/true, /*register_push_outputs=*/true,
                 /*register_pop_outputs=*/false),
      streaming_in_metadata);

  b.AddDataOutputAsBits("single_val_out_port", 64);

  ChannelMetadataProto single_val_out_metadata;
  block_port_mapping = single_val_out_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("single_val_out_port");
  b.AddSingleValueChannel("single_val_out", ChannelOps::kSendOnly,
                          single_val_out_metadata);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  absl::Span<const ChannelProto> input_channels = signature.GetInputChannels();
  EXPECT_EQ(input_channels.size(), 1);
  EXPECT_EQ(input_channels.at(0).name(), "streaming_in");

  absl::Span<const ChannelProto> output_channels =
      signature.GetOutputChannels();
  EXPECT_EQ(output_channels.size(), 1);
  EXPECT_EQ(output_channels.at(0).name(), "single_val_out");
}

TEST(ModuleSignatureTest, GetChannelNameWith) {
  Package p(TestName());
  ModuleSignatureBuilder b(TestName());

  // Add ports for streaming channels.
  b.AddDataInputAsBits("streaming_in_data", 24);
  b.AddDataInputAsBits("streaming_in_valid", 1);
  b.AddDataOutputAsBits("streaming_in_ready", 1);

  ChannelMetadataProto streaming_in_metadata;
  BlockPortMappingProto* block_port_mapping =
      streaming_in_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("streaming_in_data");
  block_port_mapping->set_valid_port_name("streaming_in_valid");
  block_port_mapping->set_ready_port_name("streaming_in_ready");
  b.AddStreamingChannel(
      "streaming_in", ChannelOps::kReceiveOnly, FlowControl::kReadyValid,
      p.GetBitsType(24),
      /*fifo_config=*/
      FifoConfig(/*depth=*/42, /*bypass=*/true, /*register_push_outputs=*/true,
                 /*register_pop_outputs=*/false),
      streaming_in_metadata);

  b.AddDataOutputAsBits("single_val_out_port", 64);

  ChannelMetadataProto single_val_out_metadata;
  block_port_mapping = single_val_out_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("single_val_out_port");
  b.AddSingleValueChannel("single_val_out", ChannelOps::kSendOnly,
                          single_val_out_metadata);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  EXPECT_THAT(signature.GetChannelNameWith("streaming_in_data"),
              IsOkAndHolds("streaming_in"));
  EXPECT_THAT(signature.GetChannelNameWith("streaming_in_valid"),
              IsOkAndHolds("streaming_in"));
  EXPECT_THAT(signature.GetChannelNameWith("streaming_in_ready"),
              IsOkAndHolds("streaming_in"));

  EXPECT_THAT(signature.GetChannelNameWith("single_val_out_port"),
              IsOkAndHolds("single_val_out"));

  EXPECT_THAT(signature.GetChannelNameWith("does not exist"),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("is not referenced by a channel")));
}

TEST(ModuleSignatureTest, RemoveChannel) {
  Package p(TestName());
  ModuleSignatureBuilder b(TestName());

  // Add ports for streaming channels.
  b.AddDataInputAsBits("streaming_in_data", 24);
  b.AddDataInputAsBits("streaming_in_valid", 1);
  b.AddDataOutputAsBits("streaming_in_ready", 1);

  b.AddDataOutputAsBits("streaming_out_data", 16);

  ChannelMetadataProto streaming_in_metadata;
  BlockPortMappingProto* block_port_mapping =
      streaming_in_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("streaming_in_data");
  block_port_mapping->set_valid_port_name("streaming_in_valid");
  block_port_mapping->set_ready_port_name("streaming_in_ready");

  b.AddStreamingChannel(
      "streaming_in", ChannelOps::kReceiveOnly, FlowControl::kReadyValid,
      p.GetBitsType(24),
      /*fifo_config=*/
      FifoConfig(/*depth=*/42, /*bypass=*/true, /*register_push_outputs=*/true,
                 /*register_pop_outputs=*/false),
      streaming_in_metadata);

  ChannelMetadataProto streaming_out_metadata;
  block_port_mapping = streaming_out_metadata.add_block_ports();
  block_port_mapping->set_block_name(TestName());
  block_port_mapping->set_data_port_name("streaming_out_data");
  b.AddStreamingChannel("streaming_out", ChannelOps::kSendOnly,
                        FlowControl::kNone, p.GetBitsType(24),
                        /*fifo_config=*/std::nullopt, streaming_out_metadata);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  EXPECT_EQ(signature.streaming_channels().size(), 2);

  XLS_EXPECT_OK(b.RemoveChannel("streaming_in"));
  XLS_ASSERT_OK_AND_ASSIGN(signature, b.Build());
  EXPECT_EQ(signature.streaming_channels().size(), 1);
  EXPECT_EQ(signature.streaming_channels()[0].name(), "streaming_out");

  XLS_EXPECT_OK(b.RemoveChannel("streaming_out"));
  XLS_ASSERT_OK_AND_ASSIGN(signature, b.Build());
  EXPECT_EQ(signature.streaming_channels().size(), 0);
}

TEST(ModuleSignatureTest, FromProtoAndAddOutputPort) {
  ModuleSignatureBuilder b(TestName());

  // Add ports
  b.AddDataInputAsBits("a", 24);
  b.AddDataInputAsBits("b", 1);
  b.AddDataOutputAsBits("c", 1);
  b.AddDataOutputAsBits("d", 16);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());
  ModuleSignatureBuilder from_proto_b =
      ModuleSignatureBuilder::FromProto(signature.proto());

  from_proto_b.AddDataOutputAsBits("e", 10);
  XLS_ASSERT_OK_AND_ASSIGN(signature, from_proto_b.Build());

  EXPECT_EQ(signature.data_inputs().size(), 2);
  EXPECT_EQ(signature.data_inputs()[0].name(), "a");
  EXPECT_EQ(signature.data_inputs()[0].width(), 24);
  EXPECT_EQ(signature.data_inputs()[1].name(), "b");
  EXPECT_EQ(signature.data_inputs()[1].width(), 1);

  EXPECT_EQ(signature.data_outputs().size(), 3);
  EXPECT_EQ(signature.data_outputs()[0].name(), "c");
  EXPECT_EQ(signature.data_outputs()[0].width(), 1);
  EXPECT_EQ(signature.data_outputs()[1].name(), "d");
  EXPECT_EQ(signature.data_outputs()[1].width(), 16);
  EXPECT_EQ(signature.data_outputs()[2].name(), "e");
  EXPECT_EQ(signature.data_outputs()[2].width(), 10);
}

TEST(ModuleSignatureTest, RamPortInterface1RW) {
  Package p(TestName());
  ModuleSignatureBuilder b(TestName());

  // Add ports for streaming channels.
  b.AddDataInputAsBits("ram_resp_rd_data", 32);
  b.AddDataOutputAsBits("ram_req_addr", 24);
  b.AddDataOutputAsBits("ram_req_wr_data", 32);
  b.AddDataOutputAsBits("ram_req_re", 1);
  b.AddDataOutputAsBits("ram_req_we", 1);

  b.AddRam1RW({
      .package = &p,
      .data_type = p.GetTupleType({p.GetBitsType(32)}),
      .ram_name = "ram",
      .req_name = "ram_req",
      .resp_name = "ram_resp",
      .address_width = 24,
      .address_name = "ram_req_addr",
      .read_enable_name = "ram_req_re",
      .write_enable_name = "ram_req_we",
      .read_data_name = "ram_resp_rd_data",
      .write_data_name = "ram_req_wr_data",
  });

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  EXPECT_EQ(signature.rams().size(), 1);
  EXPECT_EQ(signature.rams().at(0).name(), "ram");
  EXPECT_EQ(signature.rams().at(0).ram_oneof_case(),
            RamProto::RamOneofCase::kRam1Rw);
  const Ram1RWProto& ram_1rw = signature.rams().at(0).ram_1rw();
  EXPECT_EQ(ram_1rw.rw_port().request().name(), "ram_req");
  EXPECT_EQ(ram_1rw.rw_port().response().name(), "ram_resp");

  EXPECT_EQ(ram_1rw.rw_port().request().address().name(), "ram_req_addr");
  EXPECT_EQ(ram_1rw.rw_port().request().address().width(), 24);
  EXPECT_THAT(p.GetTypeFromProto(ram_1rw.rw_port().request().address().type()),
              IsOkAndHolds(m::Type("bits[24]")));
  EXPECT_EQ(ram_1rw.rw_port().request().address().direction(),
            DirectionProto::DIRECTION_OUTPUT);
  EXPECT_EQ(ram_1rw.rw_port().request().read_enable().name(), "ram_req_re");
  EXPECT_EQ(ram_1rw.rw_port().request().read_enable().width(), 1);
  EXPECT_THAT(
      p.GetTypeFromProto(ram_1rw.rw_port().request().read_enable().type()),
      IsOkAndHolds(m::Type("bits[1]")));
  EXPECT_EQ(ram_1rw.rw_port().request().read_enable().direction(),
            DirectionProto::DIRECTION_OUTPUT);
  EXPECT_EQ(ram_1rw.rw_port().request().write_enable().name(), "ram_req_we");
  EXPECT_EQ(ram_1rw.rw_port().request().write_enable().width(), 1);
  EXPECT_THAT(
      p.GetTypeFromProto(ram_1rw.rw_port().request().write_enable().type()),
      IsOkAndHolds(m::Type("bits[1]")));
  EXPECT_EQ(ram_1rw.rw_port().request().write_enable().direction(),
            DirectionProto::DIRECTION_OUTPUT);
  EXPECT_EQ(ram_1rw.rw_port().response().read_data().name(),
            "ram_resp_rd_data");
  EXPECT_EQ(ram_1rw.rw_port().response().read_data().width(), 32);
  EXPECT_THAT(
      p.GetTypeFromProto(ram_1rw.rw_port().response().read_data().type()),
      IsOkAndHolds(m::Type("(bits[32])")));
  EXPECT_EQ(ram_1rw.rw_port().response().read_data().direction(),
            DirectionProto::DIRECTION_INPUT);
  EXPECT_EQ(ram_1rw.rw_port().request().write_data().name(), "ram_req_wr_data");
  EXPECT_EQ(ram_1rw.rw_port().request().write_data().width(), 32);
  EXPECT_THAT(
      p.GetTypeFromProto(ram_1rw.rw_port().request().write_data().type()),
      IsOkAndHolds(m::Type("(bits[32])")));
  EXPECT_EQ(ram_1rw.rw_port().request().write_data().direction(),
            DirectionProto::DIRECTION_OUTPUT);
}

TEST(ModuleSignatureTest, RamPortInterface1R1W) {
  Package p(TestName());
  ModuleSignatureBuilder b(TestName());

  // Add ports for streaming channels.
  b.AddDataOutputAsBits("ram_req_rd_addr", 24);
  b.AddDataInputAsBits("ram_resp_rd_data", 32);
  b.AddDataOutputAsBits("ram_req_rd_en", 1);
  b.AddDataOutputAsBits("ram_req_wr_addr", 24);
  b.AddDataOutputAsBits("ram_req_wr_data", 32);
  b.AddDataOutputAsBits("ram_req_wr_en", 1);

  b.AddRam1R1W({
      .package = &p,
      .data_type = p.GetTupleType({p.GetBitsType(32)}),
      .ram_name = "ram",
      .rd_req_name = "ram_rd_req",
      .rd_resp_name = "ram_rd_resp",
      .wr_req_name = "ram_wr_req",
      .address_width = 24,
      .read_address_name = "ram_rd_addr",
      .read_data_name = "ram_rd_data",
      .read_enable_name = "ram_rd_en",
      .write_address_name = "ram_wr_addr",
      .write_data_name = "ram_wr_data",
      .write_enable_name = "ram_wr_en",
  });

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  EXPECT_EQ(signature.rams().size(), 1);
  EXPECT_EQ(signature.rams().at(0).name(), "ram");
  EXPECT_EQ(signature.rams().at(0).ram_oneof_case(),
            RamProto::RamOneofCase::kRam1R1W);
  const Ram1R1WProto& ram_1r1w = signature.rams().at(0).ram_1r1w();
  EXPECT_EQ(ram_1r1w.r_port().request().name(), "ram_rd_req");
  EXPECT_EQ(ram_1r1w.r_port().response().name(), "ram_rd_resp");
  EXPECT_EQ(ram_1r1w.w_port().request().name(), "ram_wr_req");

  EXPECT_EQ(ram_1r1w.r_port().request().address().name(), "ram_rd_addr");
  EXPECT_EQ(ram_1r1w.r_port().request().address().width(), 24);
  EXPECT_THAT(p.GetTypeFromProto(ram_1r1w.r_port().request().address().type()),
              IsOkAndHolds(m::Type("bits[24]")));
  EXPECT_EQ(ram_1r1w.r_port().request().address().direction(),
            DirectionProto::DIRECTION_OUTPUT);
  EXPECT_EQ(ram_1r1w.r_port().request().enable().name(), "ram_rd_en");
  EXPECT_EQ(ram_1r1w.r_port().request().enable().width(), 1);
  EXPECT_THAT(p.GetTypeFromProto(ram_1r1w.r_port().request().enable().type()),
              IsOkAndHolds(m::Type("bits[1]")));
  EXPECT_EQ(ram_1r1w.r_port().request().enable().direction(),
            DirectionProto::DIRECTION_OUTPUT);
  EXPECT_EQ(ram_1r1w.w_port().request().enable().name(), "ram_wr_en");
  EXPECT_EQ(ram_1r1w.w_port().request().enable().width(), 1);
  EXPECT_THAT(p.GetTypeFromProto(ram_1r1w.w_port().request().enable().type()),
              IsOkAndHolds(m::Type("bits[1]")));
  EXPECT_EQ(ram_1r1w.w_port().request().enable().direction(),
            DirectionProto::DIRECTION_OUTPUT);
  EXPECT_EQ(ram_1r1w.r_port().response().data().name(), "ram_rd_data");
  EXPECT_EQ(ram_1r1w.r_port().response().data().width(), 32);
  EXPECT_THAT(p.GetTypeFromProto(ram_1r1w.r_port().response().data().type()),
              IsOkAndHolds(m::Type("(bits[32])")));
  EXPECT_EQ(ram_1r1w.r_port().response().data().direction(),
            DirectionProto::DIRECTION_INPUT);

  EXPECT_EQ(ram_1r1w.w_port().request().address().name(), "ram_wr_addr");
  EXPECT_EQ(ram_1r1w.w_port().request().address().width(), 24);
  EXPECT_THAT(p.GetTypeFromProto(ram_1r1w.w_port().request().address().type()),
              IsOkAndHolds(m::Type("bits[24]")));
  EXPECT_EQ(ram_1r1w.w_port().request().address().direction(),
            DirectionProto::DIRECTION_OUTPUT);
  EXPECT_EQ(ram_1r1w.w_port().request().data().name(), "ram_wr_data");
  EXPECT_EQ(ram_1r1w.w_port().request().data().width(), 32);
  EXPECT_THAT(p.GetTypeFromProto(ram_1r1w.w_port().request().data().type()),
              IsOkAndHolds(m::Type("(bits[32])")));
  EXPECT_EQ(ram_1r1w.w_port().request().data().direction(),
            DirectionProto::DIRECTION_OUTPUT);
}

TEST(ModuleSignatureTest, FifoInstantiation) {
  Package p(TestName());

  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * loopback_channel,
      p.CreateStreamingChannel("loopback_channel", ChannelOps::kSendReceive,
                               p.GetTupleType({p.GetBitsType(32)}),
                               /*initial_values=*/{},
                               /*fifo_config=*/
                               FifoConfig(/*depth=*/10, /*bypass=*/false,
                                          /*register_push_outputs=*/true,
                                          /*register_pop_outputs=*/false),
                               /*flow_control=*/FlowControl::kReadyValid));
  ModuleSignatureBuilder b(TestName());

  // Add ports for streaming channels.
  b.AddFifoInstantiation(&p, "fifo_inst", "loopback_channel",
                         loopback_channel->type(),
                         *loopback_channel->fifo_config());

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  EXPECT_EQ(signature.instantiations().size(), 1);
  EXPECT_TRUE(signature.instantiations().at(0).has_fifo_instantiation());
  const FifoInstantiationProto& inst =
      signature.instantiations().at(0).fifo_instantiation();
  EXPECT_EQ(inst.instance_name(), "fifo_inst");
  EXPECT_EQ(inst.channel_name(), "loopback_channel");
  EXPECT_EQ(inst.fifo_config().width(), 32);
  EXPECT_EQ(inst.fifo_config().depth(), 10);
  EXPECT_EQ(inst.fifo_config().bypass(), false);
  EXPECT_THAT(p.GetTypeFromProto(inst.type()),
              IsOkAndHolds(m::Type("(bits[32])")));
}

}  // namespace
}  // namespace verilog
}  // namespace xls
