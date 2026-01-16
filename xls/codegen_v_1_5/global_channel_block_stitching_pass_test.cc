// Copyright 2026 The XLS Authors
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

#include "xls/codegen_v_1_5/global_channel_block_stitching_pass.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_conversion_utils.h"
#include "xls/codegen_v_1_5/channel_to_port_io_lowering_pass.h"
#include "xls/codegen_v_1_5/scheduled_block_conversion_pass.h"
#include "xls/codegen_v_1_5/scheduling_pass.h"
#include "xls/codegen_v_1_5/state_to_register_io_lowering_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/schedule.h"

namespace xls::codegen {
namespace {

namespace m = xls::op_matchers;

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

class GlobalChannelBlockStitchingPassTest : public IrTestBase {
 protected:
  void SetUp() override { p_ = CreatePackage(); }

  absl::StatusOr<BlockConversionPassOptions> CreateOptions(
      Package* p, int64_t pipeline_stages,
      ::xls::verilog::CodegenOptions codegen_options =
          ::xls::verilog::CodegenOptions().clock_name("clk").reset("rst", false,
                                                                   false,
                                                                   false)) {
    TestDelayEstimator delay_estimator;
    XLS_ASSIGN_OR_RETURN(SchedulingResult scheduling_result,
                         Schedule(p,
                                  SchedulingOptions()
                                      .opt_level(0)
                                      .pipeline_stages(pipeline_stages)
                                      .schedule_all_procs(true),
                                  &delay_estimator));

    return BlockConversionPassOptions{
        .codegen_options = std::move(codegen_options),
        .package_schedule = std::move(scheduling_result.package_schedule),
    };
  }

  absl::StatusOr<bool> Run(Package* p, int64_t pipeline_stages) {
    PassResults results;

    XLS_ASSIGN_OR_RETURN(BlockConversionPassOptions options,
                         CreateOptions(p, pipeline_stages));
    XLS_RETURN_IF_ERROR(SchedulingPass().Run(p, options, &results).status());

    XLS_RETURN_IF_ERROR(
        ScheduledBlockConversionPass().Run(p, options, &results).status());

    XLS_RETURN_IF_ERROR(
        StateToRegisterIoLoweringPass().Run(p, options, &results).status());

    XLS_RETURN_IF_ERROR(
        ChannelToPortIoLoweringPass().Run(p, options, &results).status());

    XLS_ASSIGN_OR_RETURN(bool result, GlobalChannelBlockStitchingPass().Run(
                                          p, options, &results));

    VLOG(5) << "IR after instantiation lowering: " << p_->DumpIr();
    return result;
  }

  std::unique_ptr<VerifiedPackage> p_;
};

TEST_F(GlobalChannelBlockStitchingPassTest, NoProcs) {
  FunctionBuilder fb("f", p_.get());
  BValue a = fb.Param("a", p_->GetBitsType(32));
  BValue b = fb.Param("b", p_->GetBitsType(32));
  BValue c = fb.Param("c", p_->GetBitsType(32));
  BValue res = fb.UMul(c, fb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(res));
  XLS_ASSERT_OK(p_->SetTop(f));

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_FALSE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());
  EXPECT_THAT(block->GetInstantiations(), IsEmpty());
}

TEST_F(GlobalChannelBlockStitchingPassTest, ProcWithProcScopedChannels) {
  Type* u32 = p_->GetBitsType(32);
  TokenlessProcBuilder pb(NewStyleProc{}, TestName(), /*token_name=*/"tkn",
                          p_.get());
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in_ch,
                           pb.AddInputChannel("in_ch", u32));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out_ch,
                           pb.AddOutputChannel("out_ch", u32));
  BValue value = pb.Receive(in_ch);
  pb.Send(out_ch, value);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK(p_->SetTop(proc));

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_FALSE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());
  EXPECT_THAT(block->GetInstantiations(), IsEmpty());
}

TEST_F(GlobalChannelBlockStitchingPassTest, OneInstantiation) {
  ChannelConfig channel_config = ChannelConfig()
                                     .WithFifoConfig(FifoConfig(
                                         /*depth=*/1,
                                         /*bypass=*/false,
                                         /*register_push_outputs=*/true,
                                         /*register_pop_outputs=*/false))
                                     .WithInputFlopKind(FlopKind::kNone)
                                     .WithOutputFlopKind(FlopKind::kNone);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p_->CreateStreamingChannel("in_ch", ChannelOps::kReceiveOnly,
                                 p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p_->CreateStreamingChannel("out_ch", ChannelOps::kSendOnly,
                                 p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_to_leaf,
      p_->CreateStreamingChannel("internal_ch_to_leaf",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_from_leaf,
      p_->CreateStreamingChannel("internal_ch_from_leaf",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));

  TokenlessProcBuilder pb_leaf(absl::StrCat(TestName(), "_leaf"),
                               /*token_name=*/"tkn", p_.get());
  pb_leaf.Send(internal_ch_from_leaf, pb_leaf.Receive(internal_ch_to_leaf));
  XLS_ASSERT_OK(pb_leaf.Build().status());

  TokenlessProcBuilder pb_main(TestName(), /*token_name=*/"tkn", p_.get());
  pb_main.Send(internal_ch_to_leaf, pb_main.Receive(in_ch));
  pb_main.Send(out_ch, pb_main.Receive(internal_ch_from_leaf));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());

  XLS_ASSERT_OK(p_->SetTop(main));
  XLS_ASSERT_OK(VerifyPackage(p_.get()));
  p_->AcceptInvalid();

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());

  EXPECT_THAT(
      block->GetInstantiations(),
      UnorderedElementsAre(
          m::Instantiation(HasSubstr("fifo_internal_ch_to_leaf"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr("fifo_internal_ch_from_leaf"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "__1_inst0")),
                           InstantiationKind::kBlock),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf_inst1")),
                           InstantiationKind::kBlock)));

  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * fifo_internal_ch_to_leaf,
                           block->GetInstantiation("fifo_internal_ch_to_leaf"));
  EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_to_leaf),
              UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                   m::InstantiationInput(_, "push_data"),
                                   m::InstantiationInput(_, "push_valid"),
                                   m::InstantiationInput(_, "pop_ready")));
  EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_to_leaf),
              UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                   m::InstantiationOutput("pop_data"),
                                   m::InstantiationOutput("pop_valid")));

  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * fifo_internal_ch_from_leaf,
      block->GetInstantiation("fifo_internal_ch_from_leaf"));
  EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_from_leaf),
              UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                   m::InstantiationInput(_, "push_data"),
                                   m::InstantiationInput(_, "push_valid"),
                                   m::InstantiationInput(_, "pop_ready")));
  EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_from_leaf),
              UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                   m::InstantiationOutput("pop_data"),
                                   m::InstantiationOutput("pop_valid")));
}

TEST_F(GlobalChannelBlockStitchingPassTest, MultiInstantiation) {
  ChannelConfig channel_config = ChannelConfig()
                                     .WithFifoConfig(FifoConfig(
                                         /*depth=*/1,
                                         /*bypass=*/false,
                                         /*register_push_outputs=*/true,
                                         /*register_pop_outputs=*/false))
                                     .WithInputFlopKind(FlopKind::kNone)
                                     .WithOutputFlopKind(FlopKind::kNone);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p_->CreateStreamingChannel("in_ch", ChannelOps::kReceiveOnly,
                                 p_->GetBitsType(32), /*initial_values=*/{},
                                 channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p_->CreateStreamingChannel("out_ch", ChannelOps::kSendOnly,
                                 p_->GetBitsType(32), /*initial_values=*/{},
                                 channel_config));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_to_leaf1,
      p_->CreateStreamingChannel("internal_ch_to_leaf1",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_from_leaf1,
      p_->CreateStreamingChannel("internal_ch_from_leaf1",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_to_leaf2,
      p_->CreateStreamingChannel("internal_ch_to_leaf2",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_from_leaf2,
      p_->CreateStreamingChannel("internal_ch_from_leaf2",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));

  TokenlessProcBuilder pb_leaf1(absl::StrCat(TestName(), "_leaf1"),
                                /*token_name=*/"tkn", p_.get());
  pb_leaf1.Send(internal_ch_from_leaf1, pb_leaf1.Receive(internal_ch_to_leaf1));
  XLS_ASSERT_OK(pb_leaf1.Build().status());

  TokenlessProcBuilder pb_leaf2(absl::StrCat(TestName(), "_leaf2"),
                                /*token_name=*/"tkn", p_.get());
  pb_leaf2.Send(internal_ch_from_leaf2, pb_leaf2.Receive(internal_ch_to_leaf2));
  XLS_ASSERT_OK(pb_leaf2.Build().status());

  TokenlessProcBuilder pb_main(TestName(), /*token_name=*/"tkn", p_.get());
  pb_main.Send(internal_ch_to_leaf1, pb_main.Receive(in_ch));
  pb_main.Send(internal_ch_to_leaf2, pb_main.Receive(internal_ch_from_leaf1));
  pb_main.Send(out_ch, pb_main.Receive(internal_ch_from_leaf2));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());

  XLS_ASSERT_OK(p_->SetTop(main));
  XLS_ASSERT_OK(VerifyPackage(p_.get()));
  p_->AcceptInvalid();

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/3));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());
  EXPECT_THAT(
      block->GetInstantiations(),
      UnorderedElementsAre(
          m::Instantiation(HasSubstr("fifo_internal_ch_to_leaf1"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr("fifo_internal_ch_from_leaf1"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr("fifo_internal_ch_to_leaf2"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr("fifo_internal_ch_from_leaf2"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "__1_inst0")),
                           InstantiationKind::kBlock),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf1_inst1")),
                           InstantiationKind::kBlock),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf2_inst2")),
                           InstantiationKind::kBlock)));

  for (int i = 1; i <= 2; i++) {
    XLS_ASSERT_OK_AND_ASSIGN(
        Instantiation * fifo_internal_ch_to_leaf,
        block->GetInstantiation(absl::StrCat("fifo_internal_ch_to_leaf", i)));
    EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_to_leaf),
                UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                     m::InstantiationInput(_, "push_data"),
                                     m::InstantiationInput(_, "push_valid"),
                                     m::InstantiationInput(_, "pop_ready")));
    EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_to_leaf),
                UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                     m::InstantiationOutput("pop_data"),
                                     m::InstantiationOutput("pop_valid")));

    XLS_ASSERT_OK_AND_ASSIGN(
        Instantiation * fifo_internal_ch_from_leaf,
        block->GetInstantiation(absl::StrCat("fifo_internal_ch_from_leaf", i)));
    EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_from_leaf),
                UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                     m::InstantiationInput(_, "push_data"),
                                     m::InstantiationInput(_, "push_valid"),
                                     m::InstantiationInput(_, "pop_ready")));
    EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_from_leaf),
                UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                     m::InstantiationOutput("pop_data"),
                                     m::InstantiationOutput("pop_valid")));
  }
}

TEST_F(GlobalChannelBlockStitchingPassTest, SingleValueOutputChannelInLeaf) {
  ChannelConfig channel_config = ChannelConfig()
                                     .WithFifoConfig(FifoConfig(
                                         /*depth=*/1,
                                         /*bypass=*/false,
                                         /*register_push_outputs=*/true,
                                         /*register_pop_outputs=*/false))
                                     .WithInputFlopKind(FlopKind::kNone)
                                     .WithOutputFlopKind(FlopKind::kNone);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p_->CreateStreamingChannel("in_ch", ChannelOps::kReceiveOnly,
                                 p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p_->CreateStreamingChannel("out_ch", ChannelOps::kSendOnly,
                                 p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_to_leaf,
      p_->CreateStreamingChannel("internal_ch_to_leaf",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * internal_ch_from_leaf,
                           p_->CreateSingleValueChannel(
                               "internal_ch_from_leaf",
                               ChannelOps::kSendReceive, p_->GetBitsType(32),
                               /*initial_values=*/{}));

  TokenlessProcBuilder pb_leaf(absl::StrCat(TestName(), "_leaf"),
                               /*token_name=*/"tkn", p_.get());
  pb_leaf.Send(internal_ch_from_leaf, pb_leaf.Receive(internal_ch_to_leaf));
  XLS_ASSERT_OK(pb_leaf.Build().status());

  TokenlessProcBuilder pb_main(TestName(), /*token_name=*/"tkn", p_.get());
  pb_main.Send(internal_ch_to_leaf, pb_main.Receive(in_ch));
  pb_main.Send(out_ch, pb_main.Receive(internal_ch_from_leaf));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());

  XLS_ASSERT_OK(p_->SetTop(main));
  XLS_ASSERT_OK(VerifyPackage(p_.get()));
  p_->AcceptInvalid();

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());

  EXPECT_THAT(
      block->GetInstantiations(),
      UnorderedElementsAre(
          m::Instantiation(HasSubstr("fifo_internal_ch_to_leaf"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "__1_inst0")),
                           InstantiationKind::kBlock),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf_inst1")),
                           InstantiationKind::kBlock)));

  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * fifo_internal_ch_to_leaf,
                           block->GetInstantiation("fifo_internal_ch_to_leaf"));
  EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_to_leaf),
              UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                   m::InstantiationInput(_, "push_data"),
                                   m::InstantiationInput(_, "push_valid"),
                                   m::InstantiationInput(_, "pop_ready")));
  EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_to_leaf),
              UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                   m::InstantiationOutput("pop_data"),
                                   m::InstantiationOutput("pop_valid")));
}

TEST_F(GlobalChannelBlockStitchingPassTest, SingleValueInputChannelInLeaf) {
  ChannelConfig channel_config = ChannelConfig()
                                     .WithFifoConfig(FifoConfig(
                                         /*depth=*/1,
                                         /*bypass=*/false,
                                         /*register_push_outputs=*/true,
                                         /*register_pop_outputs=*/false))
                                     .WithInputFlopKind(FlopKind::kNone)
                                     .WithOutputFlopKind(FlopKind::kNone);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p_->CreateStreamingChannel("in_ch", ChannelOps::kReceiveOnly,
                                 p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p_->CreateStreamingChannel("out_ch", ChannelOps::kSendOnly,
                                 p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * internal_ch_to_leaf,
                           p_->CreateSingleValueChannel(
                               "internal_ch_to_leaf", ChannelOps::kSendReceive,
                               p_->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_from_leaf,
      p_->CreateStreamingChannel("internal_ch_from_leaf",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));

  TokenlessProcBuilder pb_leaf(absl::StrCat(TestName(), "_leaf"),
                               /*token_name=*/"tkn", p_.get());
  pb_leaf.Send(internal_ch_from_leaf, pb_leaf.Receive(internal_ch_to_leaf));
  XLS_ASSERT_OK(pb_leaf.Build().status());

  TokenlessProcBuilder pb_main(TestName(), /*token_name=*/"tkn", p_.get());
  pb_main.Send(internal_ch_to_leaf, pb_main.Receive(in_ch));
  pb_main.Send(out_ch, pb_main.Receive(internal_ch_from_leaf));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());

  XLS_ASSERT_OK(p_->SetTop(main));
  XLS_ASSERT_OK(VerifyPackage(p_.get()));
  p_->AcceptInvalid();

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());

  EXPECT_THAT(
      block->GetInstantiations(),
      UnorderedElementsAre(
          m::Instantiation(HasSubstr("fifo_internal_ch_from_leaf"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "__1_inst0")),
                           InstantiationKind::kBlock),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf_inst1")),
                           InstantiationKind::kBlock)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * fifo_internal_ch_from_leaf,
      block->GetInstantiation("fifo_internal_ch_from_leaf"));
  EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_from_leaf),
              UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                   m::InstantiationInput(_, "push_data"),
                                   m::InstantiationInput(_, "push_valid"),
                                   m::InstantiationInput(_, "pop_ready")));
  EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_from_leaf),
              UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                   m::InstantiationOutput("pop_data"),
                                   m::InstantiationOutput("pop_valid")));
}

TEST_F(GlobalChannelBlockStitchingPassTest, ExposedSingleValueOutput) {
  ChannelConfig channel_config = ChannelConfig()
                                     .WithFifoConfig(FifoConfig(
                                         /*depth=*/1,
                                         /*bypass=*/false,
                                         /*register_push_outputs=*/true,
                                         /*register_pop_outputs=*/false))
                                     .WithInputFlopKind(FlopKind::kNone)
                                     .WithOutputFlopKind(FlopKind::kNone);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p_->CreateStreamingChannel("in_ch", ChannelOps::kReceiveOnly,
                                 p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p_->CreateSingleValueChannel("out_ch", ChannelOps::kSendOnly,
                                   p_->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_to_leaf,
      p_->CreateStreamingChannel("internal_ch_to_leaf",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_from_leaf,
      p_->CreateStreamingChannel("internal_ch_from_leaf",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));

  TokenlessProcBuilder pb_leaf(absl::StrCat(TestName(), "_leaf"),
                               /*token_name=*/"tkn", p_.get());
  pb_leaf.Send(internal_ch_from_leaf, pb_leaf.Receive(internal_ch_to_leaf));
  XLS_ASSERT_OK(pb_leaf.Build().status());

  TokenlessProcBuilder pb_main(TestName(), /*token_name=*/"tkn", p_.get());
  pb_main.Send(internal_ch_to_leaf, pb_main.Receive(in_ch));
  pb_main.Send(out_ch, pb_main.Receive(internal_ch_from_leaf));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());

  XLS_ASSERT_OK(p_->SetTop(main));
  XLS_ASSERT_OK(VerifyPackage(p_.get()));
  p_->AcceptInvalid();

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());

  std::string main_block_name = absl::StrCat(TestName(), "__1_inst0");
  EXPECT_THAT(
      block->GetInstantiations(),
      UnorderedElementsAre(
          m::Instantiation(HasSubstr("fifo_internal_ch_to_leaf"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr("fifo_internal_ch_from_leaf"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr(main_block_name),
                           InstantiationKind::kBlock),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf_inst1")),
                           InstantiationKind::kBlock)));

  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * main_block_inst,
                           block->GetInstantiation(main_block_name));
  EXPECT_THAT(block->GetInstantiationInputs(main_block_inst),
              Not(Contains(m::InstantiationInput(_, "out_ch_rdy"))));
  EXPECT_THAT(block->GetInstantiationOutputs(main_block_inst),
              Not(Contains(m::InstantiationOutput("out_ch_vld"))));

  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * fifo_internal_ch_to_leaf,
                           block->GetInstantiation("fifo_internal_ch_to_leaf"));
  EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_to_leaf),
              UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                   m::InstantiationInput(_, "push_data"),
                                   m::InstantiationInput(_, "push_valid"),
                                   m::InstantiationInput(_, "pop_ready")));
  EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_to_leaf),
              UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                   m::InstantiationOutput("pop_data"),
                                   m::InstantiationOutput("pop_valid")));

  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * fifo_internal_ch_from_leaf,
      block->GetInstantiation("fifo_internal_ch_from_leaf"));
  EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_from_leaf),
              UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                   m::InstantiationInput(_, "push_data"),
                                   m::InstantiationInput(_, "push_valid"),
                                   m::InstantiationInput(_, "pop_ready")));
  EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_from_leaf),
              UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                   m::InstantiationOutput("pop_data"),
                                   m::InstantiationOutput("pop_valid")));
}

TEST_F(GlobalChannelBlockStitchingPassTest, ExposedSingleValueInput) {
  ChannelConfig channel_config = ChannelConfig()
                                     .WithFifoConfig(FifoConfig(
                                         /*depth=*/1,
                                         /*bypass=*/false,
                                         /*register_push_outputs=*/true,
                                         /*register_pop_outputs=*/false))
                                     .WithInputFlopKind(FlopKind::kNone)
                                     .WithOutputFlopKind(FlopKind::kNone);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p_->CreateSingleValueChannel("in_ch", ChannelOps::kReceiveOnly,
                                   p_->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p_->CreateStreamingChannel("out_ch", ChannelOps::kSendOnly,
                                 p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_to_leaf,
      p_->CreateStreamingChannel("internal_ch_to_leaf",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * internal_ch_from_leaf,
      p_->CreateStreamingChannel("internal_ch_from_leaf",
                                 ChannelOps::kSendReceive, p_->GetBitsType(32),
                                 /*initial_values=*/{}, channel_config));

  TokenlessProcBuilder pb_leaf(absl::StrCat(TestName(), "_leaf"),
                               /*token_name=*/"tkn", p_.get());
  pb_leaf.Send(internal_ch_from_leaf, pb_leaf.Receive(internal_ch_to_leaf));
  XLS_ASSERT_OK(pb_leaf.Build().status());

  TokenlessProcBuilder pb_main(TestName(), /*token_name=*/"tkn", p_.get());
  pb_main.Send(internal_ch_to_leaf, pb_main.Receive(in_ch));
  pb_main.Send(out_ch, pb_main.Receive(internal_ch_from_leaf));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());

  XLS_ASSERT_OK(p_->SetTop(main));
  XLS_ASSERT_OK(VerifyPackage(p_.get()));
  p_->AcceptInvalid();

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());

  std::string main_block_name = absl::StrCat(TestName(), "__1_inst0");
  EXPECT_THAT(
      block->GetInstantiations(),
      UnorderedElementsAre(
          m::Instantiation(HasSubstr("fifo_internal_ch_to_leaf"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr("fifo_internal_ch_from_leaf"),
                           InstantiationKind::kFifo),
          m::Instantiation(HasSubstr(main_block_name),
                           InstantiationKind::kBlock),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf_inst1")),
                           InstantiationKind::kBlock)));

  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * main_block_inst,
                           block->GetInstantiation(main_block_name));
  EXPECT_THAT(block->GetInstantiationInputs(main_block_inst),
              Not(Contains(m::InstantiationInput(_, "in_ch_vld"))));
  EXPECT_THAT(block->GetInstantiationOutputs(main_block_inst),
              Not(Contains(m::InstantiationOutput("in_ch_rdy"))));

  XLS_ASSERT_OK_AND_ASSIGN(Instantiation * fifo_internal_ch_to_leaf,
                           block->GetInstantiation("fifo_internal_ch_to_leaf"));
  EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_to_leaf),
              UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                   m::InstantiationInput(_, "push_data"),
                                   m::InstantiationInput(_, "push_valid"),
                                   m::InstantiationInput(_, "pop_ready")));
  EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_to_leaf),
              UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                   m::InstantiationOutput("pop_data"),
                                   m::InstantiationOutput("pop_valid")));

  XLS_ASSERT_OK_AND_ASSIGN(
      Instantiation * fifo_internal_ch_from_leaf,
      block->GetInstantiation("fifo_internal_ch_from_leaf"));
  EXPECT_THAT(block->GetInstantiationInputs(fifo_internal_ch_from_leaf),
              UnorderedElementsAre(m::InstantiationInput(_, "rst"),
                                   m::InstantiationInput(_, "push_data"),
                                   m::InstantiationInput(_, "push_valid"),
                                   m::InstantiationInput(_, "pop_ready")));
  EXPECT_THAT(block->GetInstantiationOutputs(fifo_internal_ch_from_leaf),
              UnorderedElementsAre(m::InstantiationOutput("push_ready"),
                                   m::InstantiationOutput("pop_data"),
                                   m::InstantiationOutput("pop_valid")));
}

}  // namespace
}  // namespace xls::codegen
