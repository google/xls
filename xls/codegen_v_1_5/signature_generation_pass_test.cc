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

#include "xls/codegen_v_1_5/signature_generation_pass.h"

#include <memory>
#include <optional>
#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/channel_to_port_io_lowering_pass.h"
#include "xls/codegen_v_1_5/function_io_lowering_pass.h"
#include "xls/codegen_v_1_5/scheduled_block_conversion_pass.h"
#include "xls/common/proto_test_utils.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/clone_package.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/ir/type.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;

using ::testing::ElementsAre;
using ::testing::UnorderedElementsAre;
using ::xls::proto_testing::EqualsProto;

class SignatureGenerationPassTest : public IrTestBase {
 protected:
  // Run the signature generation pass after scheduled-block conversion and I/O
  // lowering.
  absl::StatusOr<bool> Run(Package* p, BlockConversionPassOptions options =
                                           BlockConversionPassOptions()) {
    PassResults results;

    // Set default options if not provided
    if (!options.codegen_options.clock_name().has_value()) {
      options.codegen_options.clock_name("clk");
    }
    if (!options.codegen_options.reset().has_value()) {
      options.codegen_options.reset("rst", false, false, false);
    }

    XLS_RETURN_IF_ERROR(
        ScheduledBlockConversionPass().Run(p, options, &results).status());
    XLS_RETURN_IF_ERROR(
        ChannelToPortIoLoweringPass().Run(p, options, &results).status());
    XLS_RETURN_IF_ERROR(
        FunctionIOLoweringPass().Run(p, options, &results).status());

    return SignatureGenerationPass().Run(p, options, &results);
  }
};

TEST_F(SignatureGenerationPassTest, CombinationalBlock) {
  Package package("test");
  ScheduledFunctionBuilder fb("test", &package);
  auto a = fb.Param("a", package.GetBitsType(8));
  auto b = fb.Param("b", package.GetBitsType(32));
  fb.Param("c", package.GetBitsType(0));
  XLS_ASSERT_OK(fb.BuildWithReturnValue(fb.Concat({a, b})).status());

  BlockConversionPassOptions options = {
      .codegen_options = verilog::CodegenOptions().generate_combinational(true),
  };
  ASSERT_THAT(Run(&package, options), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("test"));
  ASSERT_NE(block->GetSignature(), std::nullopt);

  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::ModuleSignature sig,
      verilog::ModuleSignature::FromProto(*block->GetSignature()));

  EXPECT_THAT(sig.data_inputs(),
              ElementsAre(EqualsProto(R"pb(
                            direction: PORT_DIRECTION_INPUT
                            name: "a"
                            width: 8
                            type { type_enum: BITS bit_count: 8 }
                          )pb"),
                          EqualsProto(R"pb(
                            direction: PORT_DIRECTION_INPUT
                            name: "b"
                            width: 32
                            type { type_enum: BITS bit_count: 32 }
                          )pb"),
                          EqualsProto(R"pb(
                            direction: PORT_DIRECTION_INPUT
                            name: "c"
                            width: 0
                            type { type_enum: BITS bit_count: 0 }
                          )pb")));

  EXPECT_THAT(sig.data_outputs(), ElementsAre(EqualsProto(R"pb(
                direction: PORT_DIRECTION_OUTPUT
                name: "out"
                width: 40
                type { type_enum: BITS bit_count: 40 }
              )pb")));

  EXPECT_TRUE(sig.proto().has_combinational()) << sig.proto().DebugString();
}

TEST_F(SignatureGenerationPassTest, PipelinedFunction) {
  Package package("test");
  ScheduledFunctionBuilder fb("test", &package);
  auto a = fb.Param("a", package.GetBitsType(32));
  auto b = fb.Param("b", package.GetBitsType(32));
  fb.EndStage();
  auto sum = fb.Add(a, b);
  fb.EndStage();
  auto negated_sum = fb.Negate(sum);
  fb.EndStage();
  XLS_ASSERT_OK(fb.BuildWithReturnValue(fb.Not(negated_sum)).status());

  auto data_inputs_expectations =
      ElementsAre(EqualsProto(R"pb(
                    direction: PORT_DIRECTION_INPUT
                    name: "a"
                    width: 32
                    type { type_enum: BITS bit_count: 32 }
                  )pb"),
                  EqualsProto(R"pb(
                    direction: PORT_DIRECTION_INPUT
                    name: "b"
                    width: 32
                    type { type_enum: BITS bit_count: 32 }
                  )pb"));
  auto data_outputs_expectations = ElementsAre(EqualsProto(R"pb(
    direction: PORT_DIRECTION_OUTPUT
    name: "out"
    width: 32
    type { type_enum: BITS bit_count: 32 }
  )pb"));
  auto reset_expectations =
      EqualsProto(R"pb(
        name: "rst" asynchronous: false active_low: false
      )pb");

  {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                             ClonePackage(&package));
    // Default options.
    ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));

    XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test"));
    ASSERT_NE(block->GetSignature(), std::nullopt);
    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::ModuleSignature sig,
        verilog::ModuleSignature::FromProto(*block->GetSignature()));

    EXPECT_THAT(sig.data_inputs(), data_inputs_expectations);
    EXPECT_THAT(sig.data_outputs(), data_outputs_expectations);
    EXPECT_THAT(sig.proto().reset(), reset_expectations);
    EXPECT_EQ(sig.proto().clock_name(), "clk");

    EXPECT_EQ(sig.proto().pipeline().latency(), 3);
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                             ClonePackage(&package));

    BlockConversionPassOptions options = {
        .codegen_options = verilog::CodegenOptions()
                               .module_name("foobar")
                               .clock_name("the_clock")
                               .flop_inputs(true),
    };
    ASSERT_THAT(Run(p.get(), options), IsOkAndHolds(true));

    XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test"));
    ASSERT_NE(block->GetSignature(), std::nullopt);
    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::ModuleSignature sig,
        verilog::ModuleSignature::FromProto(*block->GetSignature()));

    EXPECT_THAT(sig.data_inputs(), data_inputs_expectations);
    EXPECT_THAT(sig.data_outputs(), data_outputs_expectations);
    EXPECT_THAT(sig.proto().reset(), reset_expectations);
    EXPECT_EQ(sig.proto().clock_name(), "the_clock");

    // Flopping the inputs should increase latency by one.
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                             ClonePackage(&package));

    BlockConversionPassOptions options = {
        .codegen_options = verilog::CodegenOptions()
                               .module_name("foobar")
                               .clock_name("the_clock")
                               .flop_outputs(true),
    };
    ASSERT_THAT(Run(p.get(), options), IsOkAndHolds(true));

    XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test"));
    ASSERT_NE(block->GetSignature(), std::nullopt);
    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::ModuleSignature sig,
        verilog::ModuleSignature::FromProto(*block->GetSignature()));

    EXPECT_THAT(sig.data_inputs(), data_inputs_expectations);
    EXPECT_THAT(sig.data_outputs(), data_outputs_expectations);
    EXPECT_THAT(sig.proto().reset(), reset_expectations);
    EXPECT_EQ(sig.proto().clock_name(), "the_clock");

    // Flopping the outputs should also increase latency by one.
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                             ClonePackage(&package));

    BlockConversionPassOptions options = {
        .codegen_options = verilog::CodegenOptions()
                               .module_name("foobar")
                               .clock_name("the_clock")
                               .flop_inputs(true)
                               .flop_outputs(true),
    };
    ASSERT_THAT(Run(p.get(), options), IsOkAndHolds(true));

    XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test"));
    ASSERT_NE(block->GetSignature(), std::nullopt);
    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::ModuleSignature sig,
        verilog::ModuleSignature::FromProto(*block->GetSignature()));

    EXPECT_THAT(sig.data_inputs(), data_inputs_expectations);
    EXPECT_THAT(sig.data_outputs(), data_outputs_expectations);
    EXPECT_THAT(sig.proto().reset(), reset_expectations);
    EXPECT_EQ(sig.proto().clock_name(), "the_clock");

    // Flopping both inputs and outputs should increase latency by two.
    EXPECT_EQ(sig.proto().pipeline().latency(), 5);
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                             ClonePackage(&package));

    BlockConversionPassOptions options = {
        .codegen_options =
            verilog::CodegenOptions()
                .module_name("foobar")
                .clock_name("the_clock")
                .flop_inputs(true)
                .flop_inputs_kind(
                    verilog::CodegenOptions::IOKind::kZeroLatencyBuffer)
                .flop_outputs(true),
    };
    ASSERT_THAT(Run(p.get(), options), IsOkAndHolds(true));

    XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test"));
    ASSERT_NE(block->GetSignature(), std::nullopt);
    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::ModuleSignature sig,
        verilog::ModuleSignature::FromProto(*block->GetSignature()));

    EXPECT_THAT(sig.data_inputs(), data_inputs_expectations);
    EXPECT_THAT(sig.data_outputs(), data_outputs_expectations);
    EXPECT_THAT(sig.proto().reset(), reset_expectations);
    EXPECT_EQ(sig.proto().clock_name(), "the_clock");

    // Using zero-latency buffering on the inputs should reduce latency by one.
    EXPECT_EQ(sig.proto().pipeline().latency(), 4);
  }

  {
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p,
                             ClonePackage(&package));

    BlockConversionPassOptions options = {
        .codegen_options =
            verilog::CodegenOptions()
                .module_name("foobar")
                .clock_name("the_clock")
                .flop_inputs(true)
                .flop_inputs_kind(
                    verilog::CodegenOptions::IOKind::kZeroLatencyBuffer)
                .flop_outputs(true)
                .flop_outputs_kind(
                    verilog::CodegenOptions::IOKind::kZeroLatencyBuffer),
    };
    ASSERT_THAT(Run(p.get(), options), IsOkAndHolds(true));

    XLS_ASSERT_OK_AND_ASSIGN(Block * block, p->GetBlock("test"));
    ASSERT_NE(block->GetSignature(), std::nullopt);
    XLS_ASSERT_OK_AND_ASSIGN(
        verilog::ModuleSignature sig,
        verilog::ModuleSignature::FromProto(*block->GetSignature()));

    EXPECT_THAT(sig.data_inputs(), data_inputs_expectations);
    EXPECT_THAT(sig.data_outputs(), data_outputs_expectations);
    EXPECT_THAT(sig.proto().reset(), reset_expectations);
    EXPECT_EQ(sig.proto().clock_name(), "the_clock");

    // Using zero-latency buffering on inputs and outputs should reduce latency
    // by two.
    EXPECT_EQ(sig.proto().pipeline().latency(), 3);
  }
}

TEST_F(SignatureGenerationPassTest, IOSignatureProcToPipelinedBLock) {
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

  ScheduledProcBuilder pb("test", &package);
  BValue tkn = pb.AfterAll({});
  BValue recv0 = pb.Receive(in_single_val, tkn);
  tkn = pb.TupleIndex(recv0, 0);
  BValue in0 = pb.TupleIndex(recv0, 1);
  BValue recv1 = pb.Receive(in_streaming_rv, tkn);
  tkn = pb.TupleIndex(recv1, 0);
  BValue in1 = pb.TupleIndex(recv1, 1);
  tkn = pb.Send(out_single_val, tkn, in0);
  tkn = pb.Send(out_streaming_rv, tkn, in1);
  XLS_ASSERT_OK(pb.Build().status());

  BlockConversionPassOptions options = {
      .codegen_options = verilog::CodegenOptions()
                             .flop_inputs(false)
                             .flop_outputs(false)
                             .clock_name("clk")
                             .valid_control("input_valid", "output_valid")
                             .reset("rst", false, false, false)
                             .streaming_channel_data_suffix("_data")
                             .streaming_channel_valid_suffix("_valid")
                             .streaming_channel_ready_suffix("_ready")
                             .module_name("pipelined_proc"),
  };

  ASSERT_THAT(Run(&package, options), IsOkAndHolds(true));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, package.GetBlock("test"));
  ASSERT_NE(block->GetSignature(), std::nullopt);
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::ModuleSignature sig,
      verilog::ModuleSignature::FromProto(*block->GetSignature()));

  EXPECT_THAT(
      sig.GetInputChannelInterfaces(),
      UnorderedElementsAre(EqualsProto(R"pb(
                             channel_name: "in_single_val"
                             direction: CHANNEL_DIRECTION_RECEIVE
                             type { type_enum: BITS bit_count: 32 }
                             kind: CHANNEL_KIND_SINGLE_VALUE
                             single_value { data_port_name: "in_single_val" }
                             flop_kind: FLOP_KIND_NONE
                           )pb"),
                           EqualsProto(R"pb(
                             channel_name: "in_streaming"
                             direction: CHANNEL_DIRECTION_RECEIVE
                             type { type_enum: BITS bit_count: 32 }
                             kind: CHANNEL_KIND_STREAMING
                             streaming {
                               flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
                               data_port_name: "in_streaming_data"
                               ready_port_name: "in_streaming_ready"
                               valid_port_name: "in_streaming_valid"
                             }
                             flop_kind: FLOP_KIND_NONE
                           )pb")));

  EXPECT_THAT(
      sig.GetOutputChannelInterfaces(),
      UnorderedElementsAre(EqualsProto(R"pb(
                             channel_name: "out_single_val"
                             direction: CHANNEL_DIRECTION_SEND
                             type { type_enum: BITS bit_count: 32 }
                             kind: CHANNEL_KIND_SINGLE_VALUE
                             single_value { data_port_name: "out_single_val" }
                             flop_kind: FLOP_KIND_NONE
                           )pb"),
                           EqualsProto(R"pb(
                             channel_name: "out_streaming"
                             direction: CHANNEL_DIRECTION_SEND
                             type { type_enum: BITS bit_count: 32 }
                             kind: CHANNEL_KIND_STREAMING
                             streaming {
                               flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
                               data_port_name: "out_streaming_data"
                               ready_port_name: "out_streaming_ready"
                               valid_port_name: "out_streaming_valid"
                             }
                             flop_kind: FLOP_KIND_NONE
                           )pb")));
}

TEST_F(SignatureGenerationPassTest, BlockWithFifoInstantiationNoChannel) {
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

  BlockConversionPassOptions options = {
      .codegen_options = verilog::CodegenOptions()
                             .flop_inputs(false)
                             .flop_outputs(false)
                             .clock_name("clk")
                             .valid_control("input_valid", "output_valid")
                             .reset("rst", false, false, false)
                             .streaming_channel_data_suffix("_data")
                             .streaming_channel_valid_suffix("_valid")
                             .streaming_channel_ready_suffix("_ready")
                             .module_name("pipelined_proc"),
  };
  ASSERT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  ASSERT_NE(my_block->GetSignature(), std::nullopt);
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::ModuleSignature sig,
      verilog::ModuleSignature::FromProto(*my_block->GetSignature()));

  EXPECT_THAT(sig.instantiations(), ElementsAre(EqualsProto(R"pb(
                fifo_instantiation {
                  instance_name: "my_inst"
                  type {
                    type_enum: TUPLE
                    tuple_elements { type_enum: BITS bit_count: 32 }
                  }
                  fifo_config {
                    width: 32
                    depth: 3
                    bypass: false
                    register_push_outputs: false
                    register_pop_outputs: false
                  }
                }
              )pb")));
}

TEST_F(SignatureGenerationPassTest, BlockWithFifoInstantiationWithChannel) {
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

  BlockConversionPassOptions options = {
      .codegen_options = verilog::CodegenOptions()
                             .flop_inputs(false)
                             .flop_outputs(false)
                             .clock_name("clk")
                             .valid_control("input_valid", "output_valid")
                             .reset("rst", false, false, false)
                             .streaming_channel_data_suffix("_data")
                             .streaming_channel_valid_suffix("_valid")
                             .streaming_channel_ready_suffix("_ready")
                             .module_name("pipelined_proc"),
  };
  ASSERT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  ASSERT_NE(my_block->GetSignature(), std::nullopt);
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::ModuleSignature sig,
      verilog::ModuleSignature::FromProto(*my_block->GetSignature()));

  EXPECT_THAT(sig.instantiations(), ElementsAre(EqualsProto(R"pb(
                fifo_instantiation {
                  instance_name: "my_inst"
                  type {
                    type_enum: TUPLE
                    tuple_elements { type_enum: BITS bit_count: 32 }
                  }
                  channel_name: "a"
                  fifo_config {
                    width: 32
                    depth: 3
                    bypass: false
                    register_push_outputs: false
                    register_pop_outputs: false
                  }
                }
              )pb")));
}

TEST_F(SignatureGenerationPassTest, ProcNetwork) {
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

  BlockConversionPassOptions options = {
      .codegen_options = verilog::CodegenOptions()
                             .flop_inputs(false)
                             .flop_outputs(false)
                             .clock_name("clk")
                             .valid_control("input_valid", "output_valid")
                             .reset("rst", false, false, false)
                             .streaming_channel_data_suffix("_data")
                             .streaming_channel_valid_suffix("_valid")
                             .streaming_channel_ready_suffix("_ready")
                             .module_name("pipelined_proc"),
  };
  ASSERT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  ASSERT_NE(my_block->GetSignature(), std::nullopt);
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::ModuleSignature sig,
      verilog::ModuleSignature::FromProto(*my_block->GetSignature()));

  EXPECT_THAT(sig.instantiations(),
              UnorderedElementsAre(EqualsProto(R"pb(
                                     fifo_instantiation {
                                       instance_name: "fifo_inst"
                                       type { type_enum: BITS bit_count: 32 }
                                       channel_name: "my_channel"
                                       fifo_config {
                                         width: 32
                                         depth: 1
                                         bypass: false
                                         register_push_outputs: false
                                         register_pop_outputs: false
                                       }
                                     }
                                   )pb"),
                                   EqualsProto(R"pb(
                                     block_instantiation {
                                       instance_name: "generator_inst"
                                       block_name: "generator"
                                     }
                                   )pb"),
                                   EqualsProto(R"pb(
                                     block_instantiation {
                                       instance_name: "consumer_inst"
                                       block_name: "consumer"
                                     }
                                   )pb")));
}

TEST_F(SignatureGenerationPassTest,
       InstantiatedSubblockWithPassThroughChannel) {
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

  BlockConversionPassOptions options = {
      .codegen_options = verilog::CodegenOptions()
                             .flop_inputs(false)
                             .flop_outputs(false)
                             .clock_name("clk")
                             .valid_control("input_valid", "output_valid")
                             .reset("rst", false, false, false)
                             .streaming_channel_data_suffix("_data")
                             .streaming_channel_valid_suffix("_valid")
                             .streaming_channel_ready_suffix("_ready")
                             .module_name("pipelined_proc"),
  };
  ASSERT_THAT(Run(p.get(), options), IsOkAndHolds(true));
  ASSERT_NE(my_block->GetSignature(), std::nullopt);
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::ModuleSignature sig,
      verilog::ModuleSignature::FromProto(*my_block->GetSignature()));

  EXPECT_THAT(sig.instantiations(), ElementsAre(EqualsProto(R"pb(
                block_instantiation {
                  instance_name: "subblock_inst"
                  block_name: "subblock"
                }
              )pb")));

  EXPECT_THAT(sig.GetChannelInterfaceByName("in"),
              IsOkAndHolds(EqualsProto(R"pb(
                channel_name: "in"
                direction: CHANNEL_DIRECTION_RECEIVE
                type { type_enum: BITS bit_count: 32 }
                kind: CHANNEL_KIND_STREAMING
                streaming {
                  flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
                  data_port_name: "in"
                  ready_port_name: "in_ready"
                  valid_port_name: "in_valid"
                }
                flop_kind: FLOP_KIND_NONE
              )pb")));

  EXPECT_THAT(sig.GetChannelInterfaceByName("out"),
              IsOkAndHolds(EqualsProto(R"pb(
                channel_name: "out"
                direction: CHANNEL_DIRECTION_SEND
                type { type_enum: BITS bit_count: 32 }
                kind: CHANNEL_KIND_STREAMING
                streaming {
                  flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
                  data_port_name: "out"
                  ready_port_name: "out_ready"
                  valid_port_name: "out_valid"
                }
                flop_kind: FLOP_KIND_NONE
              )pb")));

  XLS_ASSERT_OK_AND_ASSIGN(Block * subblock, p->GetBlock("subblock"));
  ASSERT_NE(subblock->GetSignature(), std::nullopt);
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::ModuleSignature subblock_sig,
      verilog::ModuleSignature::FromProto(*subblock->GetSignature()));

  EXPECT_THAT(subblock_sig.GetChannelInterfaceByName("in"),
              IsOkAndHolds(EqualsProto(R"pb(
                channel_name: "in"
                direction: CHANNEL_DIRECTION_RECEIVE
                type { type_enum: BITS bit_count: 32 }
                kind: CHANNEL_KIND_STREAMING
                streaming {
                  flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
                  data_port_name: "in"
                  ready_port_name: "in_ready"
                  valid_port_name: "in_valid"
                }
                flop_kind: FLOP_KIND_NONE
              )pb")));

  EXPECT_THAT(subblock_sig.GetChannelInterfaceByName("out"),
              IsOkAndHolds(EqualsProto(R"pb(
                channel_name: "out"
                direction: CHANNEL_DIRECTION_SEND
                type { type_enum: BITS bit_count: 32 }
                kind: CHANNEL_KIND_STREAMING
                streaming {
                  flow_control: CHANNEL_FLOW_CONTROL_READY_VALID
                  data_port_name: "out"
                  ready_port_name: "out_ready"
                  valid_port_name: "out_valid"
                }
                flop_kind: FLOP_KIND_NONE
              )pb")));
}

}  // namespace
}  // namespace xls::codegen
