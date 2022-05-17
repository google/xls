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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/value.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

TEST(ModuleSignatureTest, SimpledFixedLatencyInterface) {
  ModuleSignatureBuilder b(TestName());

  b.AddDataInput("x", 42).AddDataOutput("y", 2).WithFixedLatencyInterface(123);

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

TEST(ModuleSignatureTest, ReadyValidInterface) {
  ModuleSignatureBuilder b(TestName());

  b.WithReadyValidInterface("input_rdy", "input_vld", "output_rdy",
                            "output_vld")
      .WithClock("the_clk")
      .WithReset("reset_me", /*asynchronous=*/true, /*active_low=*/false)
      .AddDataInput("x", 42)
      .AddDataInput("y", 2)
      .AddDataInput("z", 44444)
      .AddDataOutput("o1", 1)
      .AddDataOutput("o2", 3);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());
  ASSERT_TRUE(signature.proto().has_ready_valid());
  EXPECT_EQ(signature.proto().ready_valid().input_ready(), "input_rdy");
  EXPECT_EQ(signature.proto().ready_valid().input_valid(), "input_vld");
  EXPECT_EQ(signature.proto().ready_valid().output_ready(), "output_rdy");
  EXPECT_EQ(signature.proto().ready_valid().output_valid(), "output_vld");

  EXPECT_EQ(signature.TotalDataInputBits(), 44488);
  EXPECT_EQ(signature.TotalDataOutputBits(), 4);

  EXPECT_EQ(signature.proto().clock_name(), "the_clk");
  EXPECT_TRUE(signature.proto().has_reset());
  EXPECT_EQ(signature.proto().reset().name(), "reset_me");
  EXPECT_TRUE(signature.proto().reset().asynchronous());
  EXPECT_FALSE(signature.proto().reset().active_low());

  EXPECT_EQ(signature.data_inputs().size(), 3);
  EXPECT_EQ(signature.data_outputs().size(), 2);
}

TEST(ModuleSignatureTest, PipelineInterface) {
  ModuleSignatureBuilder b(TestName());

  b.WithPipelineInterface(/*latency=*/2, /*initiation_interval=*/3)
      .WithClock("clk")
      .AddDataInput("in", 4)
      .AddDataOutput("out", 5);

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());
  ASSERT_TRUE(signature.proto().has_pipeline());
  EXPECT_EQ(signature.proto().pipeline().latency(), 2);
  EXPECT_EQ(signature.proto().pipeline().initiation_interval(), 3);
}

TEST(ModuleSignatureTest, PipelineInterfaceMissingClock) {
  ModuleSignatureBuilder b(TestName());

  b.WithPipelineInterface(/*latency=*/2, /*initiation_interval=*/3)
      .AddDataInput("in", 4)
      .AddDataOutput("out", 5);

  EXPECT_THAT(b.Build(), StatusIs(absl::StatusCode::kInvalidArgument,
                                  HasSubstr("Missing clock")));
}

TEST(ModuleSignatureTest, ToKwargs) {
  ModuleSignatureBuilder b(TestName());
  b.AddDataInput("x", 42)
      .AddDataInput("y", 2)
      .AddDataOutput("z", 32)
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

  b.AddDataInput("single_val_in_port", 32);
  b.AddDataOutput("single_val_out_port", 64);

  b.AddSingleValueChannel("single_val_in", ChannelOps::kReceiveOnly,
                          "single_val_in_port");
  b.AddSingleValueChannel("single_val_out", ChannelOps::kSendOnly,
                          "single_val_out_port");

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
  EXPECT_EQ(signature.single_value_channels().at(0).data_port_name(),
            "single_val_in_port");

  EXPECT_EQ(signature.single_value_channels().at(1).name(), "single_val_out");
  EXPECT_EQ(signature.single_value_channels().at(1).kind(),
            CHANNEL_KIND_SINGLE_VALUE);
  EXPECT_EQ(signature.single_value_channels().at(1).supported_ops(),
            CHANNEL_OPS_SEND_ONLY);
  EXPECT_EQ(signature.single_value_channels().at(1).flow_control(),
            CHANNEL_FLOW_CONTROL_NONE);
  EXPECT_EQ(signature.single_value_channels().at(1).data_port_name(),
            "single_val_out_port");
}

TEST(ModuleSignatureTest, StreamingChannelsInterface) {
  ModuleSignatureBuilder b(TestName());

  // Add ports for streaming channels.
  b.AddDataInput("streaming_in_data", 24);
  b.AddDataInput("streaming_in_valid", 1);
  b.AddDataOutput("streaming_in_ready", 1);

  b.AddDataOutput("streaming_out_data", 16);

  b.AddStreamingChannel("streaming_in", ChannelOps::kReceiveOnly,
                        FlowControl::kReadyValid, /*fifo_depth=*/42,
                        "streaming_in_data", "streaming_in_valid",
                        "streaming_in_ready");

  b.AddStreamingChannel("streaming_out", ChannelOps::kSendOnly,
                        FlowControl::kNone, /*fifo_depth=*/std::nullopt,
                        "streaming_out_data");

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
  EXPECT_EQ(signature.streaming_channels().at(0).fifo_depth(), 42);
  EXPECT_EQ(signature.streaming_channels().at(0).data_port_name(),
            "streaming_in_data");
  EXPECT_EQ(signature.streaming_channels().at(0).valid_port_name(),
            "streaming_in_valid");
  EXPECT_EQ(signature.streaming_channels().at(0).ready_port_name(),
            "streaming_in_ready");

  EXPECT_EQ(signature.streaming_channels().at(1).name(), "streaming_out");
  EXPECT_EQ(signature.streaming_channels().at(1).kind(),
            CHANNEL_KIND_STREAMING);
  EXPECT_EQ(signature.streaming_channels().at(1).supported_ops(),
            CHANNEL_OPS_SEND_ONLY);
  EXPECT_EQ(signature.streaming_channels().at(1).flow_control(),
            CHANNEL_FLOW_CONTROL_NONE);
  EXPECT_FALSE(signature.streaming_channels().at(1).has_fifo_depth());
  EXPECT_EQ(signature.streaming_channels().at(1).data_port_name(),
            "streaming_out_data");
  EXPECT_FALSE(signature.streaming_channels().at(1).has_valid_port_name());
  EXPECT_FALSE(signature.streaming_channels().at(1).has_ready_port_name());
}

}  // namespace
}  // namespace verilog
}  // namespace xls
