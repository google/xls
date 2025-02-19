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

#include <memory>
#include <optional>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace verilog {
namespace {

using ::absl_testing::IsOkAndHolds;

TEST(SignatureGeneratorTest, CombinationalBlock) {
  Package package("test");
  FunctionBuilder fb("test", &package);
  auto a = fb.Param("a", package.GetBitsType(8));
  auto b = fb.Param("b", package.GetBitsType(32));
  fb.Param("c", package.GetBitsType(0));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Concat({a, b})));
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenPassUnit unit,
      FunctionBaseToCombinationalBlock(f, CodegenOptions()));

  // Default options.
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(CodegenOptions(), unit.top_block));

  ASSERT_EQ(sig.data_inputs().size(), 3);

  EXPECT_EQ(sig.data_inputs()[0].direction(), PORT_DIRECTION_INPUT);
  EXPECT_EQ(sig.data_inputs()[0].name(), "a");
  EXPECT_EQ(sig.data_inputs()[0].width(), 8);

  EXPECT_EQ(sig.data_inputs()[1].direction(), PORT_DIRECTION_INPUT);
  EXPECT_EQ(sig.data_inputs()[1].name(), "b");
  EXPECT_EQ(sig.data_inputs()[1].width(), 32);

  EXPECT_EQ(sig.data_inputs()[2].direction(), PORT_DIRECTION_INPUT);
  EXPECT_EQ(sig.data_inputs()[2].name(), "c");
  EXPECT_EQ(sig.data_inputs()[2].width(), 0);

  ASSERT_EQ(sig.data_outputs().size(), 1);

  EXPECT_EQ(sig.data_outputs()[0].direction(), PORT_DIRECTION_OUTPUT);
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
      RunPipelineSchedule(f, *estimator,
                          SchedulingOptions().pipeline_stages(4)));

  {
    auto codegen_options =
        CodegenOptions()
            .module_name("foobar")
            .reset("rst_n", /*asynchronous=*/false, /*active_low=*/true,
                   /*reset_data_path=*/false)
            .clock_name("the_clock");
    XLS_ASSERT_OK_AND_ASSIGN(
        CodegenPassUnit unit,
        FunctionBaseToPipelinedBlock(schedule, codegen_options, f));
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(codegen_options, unit.top_block,
                          unit.metadata[unit.top_block]
                              .streaming_io_and_pipeline.node_to_stage_map));

    ASSERT_EQ(sig.data_inputs().size(), 2);

    EXPECT_EQ(sig.data_inputs()[0].direction(), PORT_DIRECTION_INPUT);
    EXPECT_EQ(sig.data_inputs()[0].name(), "a");
    EXPECT_EQ(sig.data_inputs()[0].width(), 32);

    EXPECT_EQ(sig.data_inputs()[1].direction(), PORT_DIRECTION_INPUT);
    EXPECT_EQ(sig.data_inputs()[1].name(), "b");
    EXPECT_EQ(sig.data_inputs()[1].width(), 32);

    ASSERT_EQ(sig.data_outputs().size(), 1);

    EXPECT_EQ(sig.data_outputs()[0].direction(), PORT_DIRECTION_OUTPUT);
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

    auto codegen_options = CodegenOptions()
                               .module_name("foobar")
                               .clock_name("the_clock")
                               .flop_inputs(true);
    XLS_ASSERT_OK_AND_ASSIGN(
        CodegenPassUnit unit,
        FunctionBaseToPipelinedBlock(schedule, codegen_options, f));

    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(codegen_options, unit.top_block,
                          unit.metadata[unit.top_block]
                              .streaming_io_and_pipeline.node_to_stage_map));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    auto codegen_options = CodegenOptions()
                               .module_name("foobar")
                               .clock_name("the_clock")
                               .flop_outputs(true);
    XLS_ASSERT_OK_AND_ASSIGN(
        CodegenPassUnit unit,
        FunctionBaseToPipelinedBlock(schedule, codegen_options, f));

    // Adding flopping of the outputs should increase latency by one.
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(codegen_options, unit.top_block,
                          unit.metadata[unit.top_block]
                              .streaming_io_and_pipeline.node_to_stage_map));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    auto codegen_options = CodegenOptions()
                               .module_name("foobar")
                               .clock_name("the_clock")
                               .flop_inputs(true)
                               .flop_outputs(true);

    XLS_ASSERT_OK_AND_ASSIGN(
        CodegenPassUnit unit,
        FunctionBaseToPipelinedBlock(schedule, codegen_options, f));

    // Adding flopping of both inputs and outputs should increase latency by
    // two.
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(codegen_options, unit.top_block,
                          unit.metadata[unit.top_block]
                              .streaming_io_and_pipeline.node_to_stage_map));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 5);
  }

  {
    // Switching input to a zero latency buffer should reduce latency by one.
    auto codegen_options =
        CodegenOptions()
            .module_name("foobar")
            .clock_name("the_clock")
            .flop_inputs(true)
            .flop_inputs_kind(CodegenOptions::IOKind::kZeroLatencyBuffer)
            .flop_outputs(true);
    XLS_ASSERT_OK_AND_ASSIGN(
        CodegenPassUnit unit,
        FunctionBaseToPipelinedBlock(schedule, codegen_options, f));
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(codegen_options, unit.top_block,
                          unit.metadata[unit.top_block]
                              .streaming_io_and_pipeline.node_to_stage_map));
    ASSERT_TRUE(sig.proto().has_pipeline());
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    // Switching output to a zero latency buffer should further reduce latency
    // by one.
    auto codegen_options =
        CodegenOptions()
            .module_name("foobar")
            .clock_name("the_clock")
            .flop_inputs(true)
            .flop_inputs_kind(CodegenOptions::IOKind::kZeroLatencyBuffer)
            .flop_outputs(true)
            .flop_outputs_kind(CodegenOptions::IOKind::kZeroLatencyBuffer);
    XLS_ASSERT_OK_AND_ASSIGN(
        CodegenPassUnit unit,
        FunctionBaseToPipelinedBlock(schedule, codegen_options, f));
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleSignature sig,
        GenerateSignature(codegen_options, unit.top_block,
                          unit.metadata[unit.top_block]
                              .streaming_io_and_pipeline.node_to_stage_map));
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
          /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
          FlowControl::kReadyValid));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * out_single_val,
                           package.CreateSingleValueChannel(
                               "out_single_val", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_streaming_rv,
      package.CreateStreamingChannel(
          "out_streaming", ChannelOps::kSendOnly, u32,
          /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
          FlowControl::kReadyValid));

  TokenlessProcBuilder pb("test", /*token_name=*/"tkn", &package);
  BValue in0 = pb.Receive(in_single_val);
  BValue in1 = pb.Receive(in_streaming_rv);
  pb.Send(out_single_val, in0);
  pb.Send(out_streaming_rv, in1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(DelayEstimator * estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *estimator,
                          SchedulingOptions().pipeline_stages(1)));
  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);
  options.streaming_channel_data_suffix("_data");
  options.streaming_channel_valid_suffix("_valid");
  options.streaming_channel_ready_suffix("_ready");
  options.module_name("pipelined_proc");

  XLS_ASSERT_OK_AND_ASSIGN(CodegenPassUnit unit, FunctionBaseToPipelinedBlock(
                                                     schedule, options, proc));
  Block* block = unit.top_block;
  XLS_VLOG_LINES(2, block->DumpIr());

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(options, block, schedule.GetCycleMap()));

  EXPECT_EQ(sig.GetInputChannelInterfaces().size(), 2);
  EXPECT_EQ(sig.GetOutputChannelInterfaces().size(), 2);

  {
    XLS_ASSERT_OK_AND_ASSIGN(ChannelInterfaceProto ch,
                             sig.GetChannelInterfaceByName("in_single_val"));
    EXPECT_EQ(ch.channel_name(), "in_single_val");
    EXPECT_EQ(ch.direction(), CHANNEL_DIRECTION_RECEIVE);
    EXPECT_THAT(package.GetTypeFromProto(ch.type()),
                IsOkAndHolds(m::Type("bits[32]")));
    EXPECT_EQ(ch.kind(), CHANNEL_KIND_SINGLE_VALUE);
    EXPECT_EQ(ch.flow_control(), CHANNEL_FLOW_CONTROL_NONE);
    EXPECT_EQ(ch.data_port_name(), "in_single_val");
    EXPECT_FALSE(ch.has_ready_port_name());
    EXPECT_FALSE(ch.has_valid_port_name());
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(ChannelInterfaceProto ch,
                             sig.GetChannelInterfaceByName("in_streaming"));
    EXPECT_EQ(ch.channel_name(), "in_streaming");
    EXPECT_EQ(ch.direction(), CHANNEL_DIRECTION_RECEIVE);
    EXPECT_THAT(package.GetTypeFromProto(ch.type()),
                IsOkAndHolds(m::Type("bits[32]")));
    EXPECT_EQ(ch.kind(), CHANNEL_KIND_STREAMING);
    EXPECT_EQ(ch.flow_control(), CHANNEL_FLOW_CONTROL_READY_VALID);
    EXPECT_EQ(ch.data_port_name(), "in_streaming_data");
    EXPECT_EQ(ch.ready_port_name(), "in_streaming_ready");
    EXPECT_EQ(ch.valid_port_name(), "in_streaming_valid");
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(ChannelInterfaceProto ch,
                             sig.GetChannelInterfaceByName("out_single_val"));
    EXPECT_EQ(ch.channel_name(), "out_single_val");
    EXPECT_EQ(ch.direction(), CHANNEL_DIRECTION_SEND);
    EXPECT_THAT(package.GetTypeFromProto(ch.type()),
                IsOkAndHolds(m::Type("bits[32]")));
    EXPECT_EQ(ch.kind(), CHANNEL_KIND_SINGLE_VALUE);
    EXPECT_EQ(ch.flow_control(), CHANNEL_FLOW_CONTROL_NONE);
    EXPECT_EQ(ch.data_port_name(), "out_single_val");
    EXPECT_FALSE(ch.has_ready_port_name());
    EXPECT_FALSE(ch.has_valid_port_name());
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(ChannelInterfaceProto ch,
                             sig.GetChannelInterfaceByName("out_streaming"));
    EXPECT_EQ(ch.channel_name(), "out_streaming");
    EXPECT_EQ(ch.direction(), CHANNEL_DIRECTION_SEND);
    EXPECT_THAT(package.GetTypeFromProto(ch.type()),
                IsOkAndHolds(m::Type("bits[32]")));
    EXPECT_EQ(ch.kind(), CHANNEL_KIND_STREAMING);
    EXPECT_EQ(ch.flow_control(), CHANNEL_FLOW_CONTROL_READY_VALID);
    EXPECT_EQ(ch.data_port_name(), "out_streaming_data");
    EXPECT_EQ(ch.ready_port_name(), "out_streaming_ready");
    EXPECT_EQ(ch.valid_port_name(), "out_streaming_valid");
  }
}

TEST(SignatureGeneratorTest, BlockWithFifoInstantiationNoChannel) {
  constexpr std::string_view ir_text = R"(package test

block my_block(in: bits[32], out: (bits[32])) {
  in: bits[32] = input_port(name=in)
  instantiation my_inst(data_type=(bits[32]), depth=3, bypass=false, register_push_outputs=false, register_pop_outputs=false, kind=fifo)
  in_inst_input: () = instantiation_input(in, instantiation=my_inst, port_name=push_data)
  pop_data_inst_output: (bits[32]) = instantiation_output(instantiation=my_inst, port_name=pop_data)
  out_output_port: () = output_port(pop_data_inst_output, name=out)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Block * my_block, p->GetBlock("my_block"));

  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);
  options.streaming_channel_data_suffix("_data");
  options.streaming_channel_valid_suffix("_valid");
  options.streaming_channel_ready_suffix("_ready");
  options.module_name("pipelined_proc");
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(options, my_block, /*stage_map=*/{}));

  ASSERT_EQ(sig.instantiations().size(), 1);
  ASSERT_TRUE(sig.instantiations()[0].has_fifo_instantiation());
  const FifoInstantiationProto& instantiation =
      sig.instantiations()[0].fifo_instantiation();
  EXPECT_EQ(instantiation.instance_name(), "my_inst");
  EXPECT_FALSE(instantiation.has_channel_name());
  EXPECT_THAT(p->GetTypeFromProto(instantiation.type()),
              IsOkAndHolds(m::Type("(bits[32])")));
  EXPECT_EQ(instantiation.fifo_config().depth(), 3);
  EXPECT_FALSE(instantiation.fifo_config().bypass());
}

TEST(SignatureGeneratorTest, BlockWithFifoInstantiationWithChannel) {
  constexpr std::string_view ir_text = R"(package test
chan a(bits[32], id=0, ops=send_only, fifo_depth=3, bypass=false, register_push_outputs=false, register_pop_outputs=false, kind=streaming, flow_control=ready_valid)

proc needed_to_verify(state: (), init={()}) {
  tok: token = literal(value=token)
  literal0: bits[32] = literal(value=32)
  send_tok: token = send(tok, literal0, channel=a)
  next (state)
}

block my_block(in: bits[32], out: (bits[32])) {
  in: bits[32] = input_port(name=in)
  instantiation my_inst(data_type=(bits[32]), depth=3, bypass=false, register_push_outputs=false, register_pop_outputs=false, channel=a, kind=fifo)
  in_inst_input: () = instantiation_input(in, instantiation=my_inst, port_name=push_data)
  pop_data_inst_output: (bits[32]) = instantiation_output(instantiation=my_inst, port_name=pop_data)
  out_output_port: () = output_port(pop_data_inst_output, name=out)
}
)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Block * my_block, p->GetBlock("my_block"));

  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);
  options.streaming_channel_data_suffix("_data");
  options.streaming_channel_valid_suffix("_valid");
  options.streaming_channel_ready_suffix("_ready");
  options.module_name("pipelined_proc");
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(options, my_block, /*stage_map=*/{}));
  ASSERT_EQ(sig.instantiations().size(), 1);
  ASSERT_TRUE(sig.instantiations()[0].has_fifo_instantiation());
  const FifoInstantiationProto& instantiation =
      sig.instantiations()[0].fifo_instantiation();
  EXPECT_EQ(instantiation.instance_name(), "my_inst");
  EXPECT_TRUE(instantiation.has_channel_name());
  EXPECT_EQ(instantiation.channel_name(), "a");
  EXPECT_THAT(p->GetTypeFromProto(instantiation.type()),
              IsOkAndHolds(m::Type("(bits[32])")));
  EXPECT_EQ(instantiation.fifo_config().depth(), 3);
  EXPECT_FALSE(instantiation.fifo_config().bypass());
}

TEST(SignatureGeneratorTest, ProcNetwork) {
  constexpr std::string_view ir_text = R"(package test

block generator(out: bits[32], out_valid: bits[1], out_ready: bits[1]) {
  #![channel_ports(name=out, type=bits[32], direction=send, kind=streaming, data_port=out, ready_port=out_ready, valid_port=out_valid)]

  out_ready: bits[1] = input_port(name=out_ready)

  forty_two: bits[32] = literal(value=42)
  one: bits[1] = literal(value=1)

  out: () = output_port(forty_two, name=out)
  out_valid: () = output_port(one, name=out_valid)
}

block consumer(in: bits[32], in_valid: bits[1], in_ready: bits[1]) {
  #![channel_ports(name=in, type=bits[32], direction=receive, kind=streaming, data_port=in, ready_port=in_ready, valid_port=in_valid)]

  in: bits[32] = input_port(name=in)
  in_valid: bits[1] = input_port(name=in_valid)

  one: bits[1] = literal(value=1)

  in_ready: () = output_port(one, name=in_ready)
}

top block my_block() {
  instantiation generator_inst(block=generator, kind=block)
  instantiation consumer_inst(block=consumer, kind=block)
  instantiation fifo_inst(data_type=bits[32], depth=1, bypass=false, register_push_outputs=false, register_pop_outputs=false, channel=my_channel, kind=fifo)

  // Plumb data signal forward.
  generator_out: bits[32] = instantiation_output(instantiation=generator_inst, port_name=out)
  fifo_in_data: () = instantiation_input(generator_out, instantiation=fifo_inst, port_name=push_data)
  fifo_out_data: bits[32] = instantiation_output(instantiation=fifo_inst, port_name=pop_data)
  consumer_in: () = instantiation_input(fifo_out_data, instantiation=consumer_inst, port_name=in)

  // Plumb valid signal forward.
  generator_out_valid: bits[1] = instantiation_output(instantiation=generator_inst, port_name=out_valid)
  fifo_in_valid: () = instantiation_input(generator_out_valid, instantiation=fifo_inst, port_name=push_valid)
  fifo_out_valid: bits[1] = instantiation_output(instantiation=fifo_inst, port_name=pop_valid)
  consumer_in_valid: () = instantiation_input(fifo_out_valid, instantiation=consumer_inst, port_name=in_valid)

  // Plumb ready signal backwards.
  consumer_in_ready: bits[1] = instantiation_output(instantiation=consumer_inst, port_name=in_ready)
  fifo_out_ready: () = instantiation_input(consumer_in_ready, instantiation=fifo_inst, port_name=pop_ready)
  fifo_in_ready: bits[1] = instantiation_output(instantiation=fifo_inst, port_name=push_ready)
  generator_out_ready: () = instantiation_input(fifo_in_ready, instantiation=generator_inst, port_name=out_ready)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Block * my_block, p->GetBlock("my_block"));

  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);
  options.streaming_channel_data_suffix("_data");
  options.streaming_channel_valid_suffix("_valid");
  options.streaming_channel_ready_suffix("_ready");
  options.module_name("pipelined_proc");
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(options, my_block, /*stage_map=*/{}));

  ASSERT_EQ(sig.instantiations().size(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(FifoInstantiationProto fifo,
                           sig.GetFifoInstantiation("fifo_inst"));
  EXPECT_EQ(fifo.instance_name(), "fifo_inst");
  EXPECT_TRUE(fifo.has_channel_name());
  EXPECT_EQ(fifo.channel_name(), "my_channel");
  EXPECT_THAT(p->GetTypeFromProto(fifo.type()),
              IsOkAndHolds(m::Type("bits[32]")));
  EXPECT_EQ(fifo.fifo_config().depth(), 1);
  EXPECT_FALSE(fifo.fifo_config().bypass());

  XLS_ASSERT_OK_AND_ASSIGN(BlockInstantiationProto generator,
                           sig.GetBlockInstantiation("generator_inst"));
  EXPECT_EQ(generator.instance_name(), "generator_inst");
  EXPECT_EQ(generator.block_name(), "generator");

  XLS_ASSERT_OK_AND_ASSIGN(BlockInstantiationProto consumer,
                           sig.GetBlockInstantiation("consumer_inst"));
  EXPECT_EQ(consumer.instance_name(), "consumer_inst");
  EXPECT_EQ(consumer.block_name(), "consumer");
}

TEST(SignatureGeneratorTest, InstantiatedSubblockWithPassThroughChannel) {
  constexpr std::string_view ir_text = R"(package test

block subblock(in: bits[32], in_valid: bits[1], in_ready: bits[1],
               out: bits[32], out_valid: bits[1], out_ready: bits[1]) {
  #![channel_ports(name=in, type=bits[32], direction=receive, kind=streaming, data_port=in, ready_port=in_ready, valid_port=in_valid)]
  #![channel_ports(name=out, type=bits[32], direction=send, kind=streaming, data_port=out, ready_port=out_ready, valid_port=out_valid)]

  in: bits[32] = input_port(name=in)
  in_valid: bits[1] = input_port(name=in_valid)
  out_ready: bits[1] = input_port(name=out_ready)

  in_ready: () = output_port(out_ready, name=in_ready)
  out: () = output_port(in, name=out)
  out_valid: () = output_port(in_valid, name=out_valid)
}

top block my_block(in: bits[32], in_valid: bits[1], in_ready: bits[1],
                 out: bits[32], out_valid: bits[1], out_ready: bits[1]) {
  #![channel_ports(name=in, type=bits[32], direction=receive, kind=streaming, data_port=in, ready_port=in_ready, valid_port=in_valid)]
  #![channel_ports(name=out, type=bits[32], direction=send, kind=streaming, data_port=out, ready_port=out_ready, valid_port=out_valid)]

  instantiation subblock_inst(block=subblock, kind=block)

  in: bits[32] = input_port(name=in)
  in_valid: bits[1] = input_port(name=in_valid)
  out_ready: bits[1] = input_port(name=out_ready)

  inst_in: () = instantiation_input(in, instantiation=subblock_inst, port_name=in)
  inst_out: bits[32] = instantiation_output(instantiation=subblock_inst, port_name=out)

  inst_in_valid: () = instantiation_input(in_valid, instantiation=subblock_inst, port_name=in_valid)
  inst_out_valid: bits[1] = instantiation_output(instantiation=subblock_inst, port_name=out_valid)

  inst_out_ready: () = instantiation_input(out_ready, instantiation=subblock_inst, port_name=out_ready)
  inst_in_ready: bits[1] = instantiation_output(instantiation=subblock_inst, port_name=in_ready)

  out: () = output_port(inst_out, name=out)
  out_valid: () = output_port(inst_out_valid, name=out_valid)
  in_ready: () = output_port(inst_in_ready, name=in_ready)
}

)";
  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                           Parser::ParsePackage(ir_text));
  XLS_ASSERT_OK_AND_ASSIGN(Block * my_block, p->GetBlock("my_block"));

  CodegenOptions options;
  options.flop_inputs(false).flop_outputs(false).clock_name("clk");
  options.valid_control("input_valid", "output_valid");
  options.reset("rst", false, false, false);
  options.streaming_channel_data_suffix("_data");
  options.streaming_channel_valid_suffix("_valid");
  options.streaming_channel_ready_suffix("_ready");
  options.module_name("pipelined_proc");
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(options, my_block, /*stage_map=*/{}));

  ASSERT_EQ(sig.instantiations().size(), 1);

  XLS_ASSERT_OK_AND_ASSIGN(BlockInstantiationProto generator,
                           sig.GetBlockInstantiation("subblock_inst"));
  EXPECT_EQ(generator.instance_name(), "subblock_inst");
  EXPECT_EQ(generator.block_name(), "subblock");

  XLS_ASSERT_OK_AND_ASSIGN(ChannelInterfaceProto top_in_channel,
                           sig.GetChannelInterfaceByName("in"));
  EXPECT_EQ(top_in_channel.channel_name(), "in");
  EXPECT_EQ(top_in_channel.direction(), CHANNEL_DIRECTION_RECEIVE);
  EXPECT_THAT(p->GetTypeFromProto(top_in_channel.type()),
              IsOkAndHolds(p->GetBitsType(32)));
  EXPECT_EQ(top_in_channel.kind(), CHANNEL_KIND_STREAMING);
  EXPECT_EQ(top_in_channel.flow_control(), CHANNEL_FLOW_CONTROL_READY_VALID);
  EXPECT_EQ(top_in_channel.data_port_name(), "in");
  EXPECT_EQ(top_in_channel.ready_port_name(), "in_ready");
  EXPECT_EQ(top_in_channel.valid_port_name(), "in_valid");

  XLS_ASSERT_OK_AND_ASSIGN(ChannelInterfaceProto top_out_channel,
                           sig.GetChannelInterfaceByName("out"));
  EXPECT_EQ(top_out_channel.channel_name(), "out");
  EXPECT_EQ(top_out_channel.direction(), CHANNEL_DIRECTION_SEND);
  EXPECT_THAT(p->GetTypeFromProto(top_out_channel.type()),
              IsOkAndHolds(p->GetBitsType(32)));
  EXPECT_EQ(top_out_channel.kind(), CHANNEL_KIND_STREAMING);
  EXPECT_EQ(top_out_channel.flow_control(), CHANNEL_FLOW_CONTROL_READY_VALID);
  EXPECT_EQ(top_out_channel.data_port_name(), "out");
  EXPECT_EQ(top_out_channel.ready_port_name(), "out_ready");
  EXPECT_EQ(top_out_channel.valid_port_name(), "out_valid");

  XLS_ASSERT_OK_AND_ASSIGN(Block * subblock, p->GetBlock("subblock"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature subblock_sig,
      GenerateSignature(options, subblock, /*stage_map=*/{}));

  XLS_ASSERT_OK_AND_ASSIGN(ChannelInterfaceProto subblock_in_channel,
                           subblock_sig.GetChannelInterfaceByName("in"));
  EXPECT_EQ(subblock_in_channel.channel_name(), "in");
  EXPECT_EQ(subblock_in_channel.direction(), CHANNEL_DIRECTION_RECEIVE);
  EXPECT_THAT(p->GetTypeFromProto(subblock_in_channel.type()),
              IsOkAndHolds(p->GetBitsType(32)));
  EXPECT_EQ(subblock_in_channel.kind(), CHANNEL_KIND_STREAMING);
  EXPECT_EQ(subblock_in_channel.flow_control(),
            CHANNEL_FLOW_CONTROL_READY_VALID);
  EXPECT_EQ(subblock_in_channel.data_port_name(), "in");
  EXPECT_EQ(subblock_in_channel.ready_port_name(), "in_ready");
  EXPECT_EQ(subblock_in_channel.valid_port_name(), "in_valid");

  XLS_ASSERT_OK_AND_ASSIGN(ChannelInterfaceProto subblock_out_channel,
                           subblock_sig.GetChannelInterfaceByName("out"));
  EXPECT_EQ(subblock_out_channel.channel_name(), "out");
  EXPECT_EQ(subblock_out_channel.direction(), CHANNEL_DIRECTION_SEND);
  EXPECT_THAT(p->GetTypeFromProto(subblock_out_channel.type()),
              IsOkAndHolds(p->GetBitsType(32)));
  EXPECT_EQ(subblock_out_channel.kind(), CHANNEL_KIND_STREAMING);
  EXPECT_EQ(subblock_out_channel.flow_control(),
            CHANNEL_FLOW_CONTROL_READY_VALID);
  EXPECT_EQ(subblock_out_channel.data_port_name(), "out");
  EXPECT_EQ(subblock_out_channel.ready_port_name(), "out_ready");
  EXPECT_EQ(subblock_out_channel.valid_port_name(), "out_valid");
}

}  // namespace
}  // namespace verilog
}  // namespace xls
