// Copyright 2021 The XLS Authors
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

#include "xls/codegen/signature_generator.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/scheduling/pipeline_schedule.h"

namespace xls {
namespace verilog {
namespace {

TEST(SignatureGeneratorTest, CombinationalBlock) {
  Package package("test");
  FunctionBuilder fb("test", &package);
  auto a = fb.Param("a", package.GetBitsType(8));
  auto b = fb.Param("b", package.GetBitsType(32));
  fb.Param("c", package.GetBitsType(0));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({a, b})));

  // Default options.
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(CodegenOptions(), f));

  ASSERT_EQ(sig.data_inputs().size(), 3);

  EXPECT_EQ(sig.data_inputs()[0].direction(), DIRECTION_INPUT);
  EXPECT_EQ(sig.data_inputs()[0].name(), "a");
  EXPECT_EQ(sig.data_inputs()[0].width(), 8);

  EXPECT_EQ(sig.data_inputs()[1].direction(), DIRECTION_INPUT);
  EXPECT_EQ(sig.data_inputs()[1].name(), "b");
  EXPECT_EQ(sig.data_inputs()[1].width(), 32);

  EXPECT_EQ(sig.data_inputs()[2].direction(), DIRECTION_INPUT);
  EXPECT_EQ(sig.data_inputs()[2].name(), "c");
  EXPECT_EQ(sig.data_inputs()[2].width(), 0);

  ASSERT_EQ(sig.data_outputs().size(), 1);

  EXPECT_EQ(sig.data_outputs()[0].direction(), DIRECTION_OUTPUT);
  EXPECT_EQ(sig.data_outputs()[0].name(), "out");
  EXPECT_EQ(sig.data_outputs()[0].width(), 40);

  ASSERT_TRUE(sig.proto().has_combinational());
}

TEST(SignatureGeneratorTest, PipelinedFunction) {
  Package package("test");
  FunctionBuilder fb("test", &package);
  auto a = fb.Param("a", package.GetBitsType(32));
  auto b = fb.Param("b", package.GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Not(fb.Negate(fb.Add(a, b)))));

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(f, *estimator,
                            SchedulingOptions().pipeline_stages(4)));

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(
            CodegenOptions()
                .module_name("foobar")
                .reset("rst_n", /*asynchronous=*/false, /*active_low=*/true,
                       /*reset_data_path=*/false)
                .clock_name("the_clock"),
            f, schedule));

    ASSERT_EQ(sig.data_inputs().size(), 2);

    EXPECT_EQ(sig.data_inputs()[0].direction(), DIRECTION_INPUT);
    EXPECT_EQ(sig.data_inputs()[0].name(), "a");
    EXPECT_EQ(sig.data_inputs()[0].width(), 32);

    EXPECT_EQ(sig.data_inputs()[1].direction(), DIRECTION_INPUT);
    EXPECT_EQ(sig.data_inputs()[1].name(), "b");
    EXPECT_EQ(sig.data_inputs()[1].width(), 32);

    ASSERT_EQ(sig.data_outputs().size(), 1);

    EXPECT_EQ(sig.data_outputs()[0].direction(), DIRECTION_OUTPUT);
    EXPECT_EQ(sig.data_outputs()[0].name(), "out");
    EXPECT_EQ(sig.data_outputs()[0].width(), 32);

    EXPECT_EQ(sig.proto().reset().name(), "rst_n");
    EXPECT_FALSE(sig.proto().reset().asynchronous());
    EXPECT_TRUE(sig.proto().reset().active_low());
    EXPECT_EQ(sig.proto().clock_name(), "the_clock");

    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 3);
  }

  {
    // Adding flopping of the inputs should increase latency by one.
    XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                             GenerateSignature(CodegenOptions()
                                                   .module_name("foobar")
                                                   .clock_name("the_clock")
                                                   .flop_inputs(true),
                                               f, schedule));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    // Adding flopping of the outputs should increase latency by one.
    XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                             GenerateSignature(CodegenOptions()
                                                   .module_name("foobar")
                                                   .clock_name("the_clock")
                                                   .flop_outputs(true),
                                               f, schedule));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    // Adding flopping of both inputs and outputs should increase latency by
    // two.
    XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                             GenerateSignature(CodegenOptions()
                                                   .module_name("foobar")
                                                   .clock_name("the_clock")
                                                   .flop_inputs(true)
                                                   .flop_outputs(true),
                                               f, schedule));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 5);
  }

  {
    // Switching input to a zero latency buffer should reduce latency by one.
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(
            CodegenOptions()
                .module_name("foobar")
                .clock_name("the_clock")
                .flop_inputs(true)
                .flop_inputs_kind(CodegenOptions::IOKind::kZeroLatencyBuffer)
                .flop_outputs(true),
            f, schedule));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    // Switching output to a zero latency buffer should reduce latency by one.
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(
            CodegenOptions()
                .module_name("foobar")
                .clock_name("the_clock")
                .flop_inputs(true)
                .flop_inputs_kind(CodegenOptions::IOKind::kZeroLatencyBuffer)
                .flop_outputs(true)
                .flop_outputs_kind(CodegenOptions::IOKind::kZeroLatencyBuffer),
            f, schedule));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 3);
  }
}

TEST(SignatureGeneratorTest, IOSignatureProcToPipelinedBLock) {
  Package package("test");
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(Channel * in_single_val,
                           package.CreateSingleValueChannel(
                               "in_single_val", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_streaming_rv,
      package.CreateStreamingChannel(
          "in_streaming", ChannelOps::kReceiveOnly, u32,
          /*initial_values=*/{}, /*fifo_depth=*/std::nullopt,
          FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * out_single_val,
                           package.CreateSingleValueChannel(
                               "out_single_val", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_streaming_rv,
      package.CreateStreamingChannel(
          "out_streaming", ChannelOps::kSendOnly, u32,
          /*initial_values=*/{}, /*fifo_depth=*/std::nullopt,
          FlowControl::kReadyValid));

  TokenlessProcBuilder pb("test", /*token_name=*/"tkn", &package);
  BValue in0 = pb.Receive(in_single_val);
  BValue in1 = pb.Receive(in_streaming_rv);
  pb.Send(out_single_val, in0);
  pb.Send(out_streaming_rv, in1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  EXPECT_FALSE(in_single_val->HasCompletedBlockPortNames());
  EXPECT_FALSE(out_single_val->HasCompletedBlockPortNames());
  EXPECT_FALSE(in_streaming_rv->HasCompletedBlockPortNames());
  EXPECT_FALSE(out_streaming_rv->HasCompletedBlockPortNames());

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(proc, *estimator,
                            SchedulingOptions().pipeline_stages(1)));
  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);
  options.streaming_channel_data_suffix("_data");
  options.streaming_channel_valid_suffix("_valid");
  options.streaming_channel_ready_suffix("_ready");
  options.module_name("pipelined_proc");

  XLS_ASSERT_OK_AND_ASSIGN(Block * block,
                           ProcToPipelinedBlock(schedule, options, proc));
  XLS_VLOG_LINES(2, block->DumpIr());

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(options, block, schedule));

  EXPECT_EQ(sig.proto().data_channels_size(), 4);
  {
    ChannelProto ch = sig.proto().data_channels(0);
    EXPECT_EQ(ch.name(), "in_single_val");
    EXPECT_EQ(ch.kind(), CHANNEL_KIND_SINGLE_VALUE);
    EXPECT_EQ(ch.supported_ops(), CHANNEL_OPS_RECEIVE_ONLY);
    EXPECT_EQ(ch.flow_control(), CHANNEL_FLOW_CONTROL_NONE);
    EXPECT_EQ(ch.data_port_name(), "in_single_val");
    EXPECT_FALSE(ch.has_ready_port_name());
    EXPECT_FALSE(ch.has_valid_port_name());
  }

  {
    ChannelProto ch = sig.proto().data_channels(1);
    EXPECT_EQ(ch.name(), "in_streaming");
    EXPECT_EQ(ch.kind(), CHANNEL_KIND_STREAMING);
    EXPECT_EQ(ch.supported_ops(), CHANNEL_OPS_RECEIVE_ONLY);
    EXPECT_EQ(ch.flow_control(), CHANNEL_FLOW_CONTROL_READY_VALID);
    EXPECT_EQ(ch.data_port_name(), "in_streaming_data");
    EXPECT_EQ(ch.ready_port_name(), "in_streaming_ready");
    EXPECT_EQ(ch.valid_port_name(), "in_streaming_valid");
  }

  {
    ChannelProto ch = sig.proto().data_channels(2);
    EXPECT_EQ(ch.name(), "out_single_val");
    EXPECT_EQ(ch.kind(), CHANNEL_KIND_SINGLE_VALUE);
    EXPECT_EQ(ch.supported_ops(), CHANNEL_OPS_SEND_ONLY);
    EXPECT_EQ(ch.flow_control(), CHANNEL_FLOW_CONTROL_NONE);
    EXPECT_EQ(ch.data_port_name(), "out_single_val");
    EXPECT_FALSE(ch.has_ready_port_name());
    EXPECT_FALSE(ch.has_valid_port_name());
  }

  {
    ChannelProto ch = sig.proto().data_channels(3);
    EXPECT_EQ(ch.name(), "out_streaming");
    EXPECT_EQ(ch.kind(), CHANNEL_KIND_STREAMING);
    EXPECT_EQ(ch.supported_ops(), CHANNEL_OPS_SEND_ONLY);
    EXPECT_EQ(ch.flow_control(), CHANNEL_FLOW_CONTROL_READY_VALID);
    EXPECT_EQ(ch.data_port_name(), "out_streaming_data");
    EXPECT_EQ(ch.ready_port_name(), "out_streaming_ready");
    EXPECT_EQ(ch.valid_port_name(), "out_streaming_valid");
  }
}

}  // namespace
}  // namespace verilog
}  // namespace xls
