// Copyright 2025 The XLS Authors
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

#include "xls/codegen_v_1_5/proc_instantiation_lowering_pass.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

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
#include "xls/ir/proc_elaboration.h"
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

class ProcInstantiationLoweringPassTest : public IrTestBase {
 protected:
  void SetUp() override {
    p_ = CreatePackage();
    ran_pass_ = false;
  }

  void TearDown() override {
    if (!ran_pass_) {
      return;
    }

    for (const auto& [_, proc] :
         GetScheduledBlocksWithProcSources(p_.get(), /*new_style_only=*/true)) {
      EXPECT_THAT(proc->channels(), IsEmpty());
      EXPECT_THAT(proc->interface(), IsEmpty());
      EXPECT_THAT(proc->proc_instantiations(), IsEmpty());
    }
  }

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

    XLS_ASSIGN_OR_RETURN(
        bool result, ProcInstantiationLoweringPass().Run(p, options, &results));
    ran_pass_ = true;

    VLOG(5) << "IR after instantiation lowering: " << p_->DumpIr();
    return result;
  }

  std::unique_ptr<VerifiedPackage> p_;
  bool ran_pass_;
};

TEST_F(ProcInstantiationLoweringPassTest, NoProcs) {
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

TEST_F(ProcInstantiationLoweringPassTest, ProcWithGlobalChannels) {
  Type* u32 = p_->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p_->CreateStreamingChannel("in_ch", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p_->CreateStreamingChannel("out_ch", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestName(), /*token_name=*/"tkn", p_.get());
  BValue value = pb.Receive(in_ch);
  pb.Send(out_ch, value);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK(p_->SetTop(proc));

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_FALSE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());
  EXPECT_THAT(block->GetInstantiations(), IsEmpty());
}

TEST_F(ProcInstantiationLoweringPassTest, NoInstantiations) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc(), TestName(), /*token_name=*/"tkn",
                          p_.get());
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in_ch,
                           pb.AddInputChannel("in_ch", p_->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out_ch,
                           pb.AddOutputChannel("out_ch", p_->GetBitsType(32)));
  BValue value = pb.Receive(in_ch);
  pb.Send(out_ch, value);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK(p_->SetTop(proc));

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  // In this scenario, all that changes is the removal of the channels from the
  // source proc.
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());
  EXPECT_THAT(block->GetInstantiations(), IsEmpty());
}

TEST_F(ProcInstantiationLoweringPassTest, OneInstantiation) {
  TokenlessProcBuilder pb_leaf(NewStyleProc(),
                               absl::StrCat(TestName(), "_leaf"),
                               /*token_name=*/"tkn", p_.get());
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * in_ch,
        pb_leaf.AddInputChannel("in_ch", p_->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * out_ch,
        pb_leaf.AddOutputChannel("out_ch", p_->GetBitsType(32)));
    pb_leaf.Send(out_ch, pb_leaf.Receive(in_ch));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Proc * leaf, pb_leaf.Build());

  TokenlessProcBuilder pb_main(NewStyleProc(), TestName(), /*token_name=*/"tkn",
                               p_.get());
  {
    ChannelConfig channel_config = ChannelConfig()
                                       .WithFifoConfig(FifoConfig(
                                           /*depth=*/1,
                                           /*bypass=*/false,
                                           /*register_push_outputs=*/true,
                                           /*register_pop_outputs=*/false))
                                       .WithInputFlopKind(FlopKind::kNone)
                                       .WithOutputFlopKind(FlopKind::kNone);

    XLS_ASSERT_OK_AND_ASSIGN(
        ChannelWithInterfaces internal_ch_to_leaf,
        pb_main.AddChannel("internal_ch_to_leaf", p_->GetBitsType(32),
                           ChannelKind::kStreaming, /*initial_values=*/{},
                           channel_config));
    XLS_ASSERT_OK_AND_ASSIGN(
        ChannelWithInterfaces internal_ch_from_leaf,
        pb_main.AddChannel("internal_ch_from_leaf", p_->GetBitsType(32),
                           ChannelKind::kStreaming, /*initial_values=*/{},
                           channel_config));
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * in_ch,
        pb_main.AddInputChannel("in_ch", p_->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * out_ch,
        pb_main.AddOutputChannel("out_ch", p_->GetBitsType(32)));
    pb_main.Send(internal_ch_to_leaf.send_interface, pb_main.Receive(in_ch));
    pb_main.Send(out_ch,
                 pb_main.Receive(internal_ch_from_leaf.receive_interface));
    XLS_ASSERT_OK(pb_main.InstantiateProc(
        leaf->name(), leaf,
        std::vector<ChannelInterface*>{internal_ch_to_leaf.receive_interface,
                                       internal_ch_from_leaf.send_interface}));
  }

  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab_main,
                           ProcElaboration::Elaborate(main));

  XLS_ASSERT_OK(p_->SetTop(main));
  XLS_ASSERT_OK(VerifyPackage(p_.get()));
  p_->AcceptInvalid();

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());
  EXPECT_THAT(block->GetInstantiations(),
              UnorderedElementsAre(
                  m::Instantiation(HasSubstr("fifo_internal_ch_to_leaf"),
                                   InstantiationKind::kFifo),
                  m::Instantiation(HasSubstr("fifo_internal_ch_from_leaf"),
                                   InstantiationKind::kFifo),
                  m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf")),
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

TEST_F(ProcInstantiationLoweringPassTest, MultiInstantiation) {
  TokenlessProcBuilder pb_leaf(NewStyleProc(),
                               absl::StrCat(TestName(), "_leaf"),
                               /*token_name=*/"tkn", p_.get());
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * in_ch,
        pb_leaf.AddInputChannel("in_ch", p_->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * out_ch,
        pb_leaf.AddOutputChannel("out_ch", p_->GetBitsType(32)));
    pb_leaf.Send(out_ch, pb_leaf.Receive(in_ch));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Proc * leaf, pb_leaf.Build());

  TokenlessProcBuilder pb_main(NewStyleProc(), TestName(), /*token_name=*/"tkn",
                               p_.get());
  {
    ChannelConfig channel_config = ChannelConfig()
                                       .WithFifoConfig(FifoConfig(
                                           /*depth=*/1,
                                           /*bypass=*/false,
                                           /*register_push_outputs=*/true,
                                           /*register_pop_outputs=*/false))
                                       .WithInputFlopKind(FlopKind::kNone)
                                       .WithOutputFlopKind(FlopKind::kNone);

    XLS_ASSERT_OK_AND_ASSIGN(
        ChannelWithInterfaces internal_ch_to_leaf1,
        pb_main.AddChannel("internal_ch_to_leaf1", p_->GetBitsType(32),
                           ChannelKind::kStreaming, /*initial_values=*/{},
                           channel_config));
    XLS_ASSERT_OK_AND_ASSIGN(
        ChannelWithInterfaces internal_ch_from_leaf1,
        pb_main.AddChannel("internal_ch_from_leaf1", p_->GetBitsType(32),
                           ChannelKind::kStreaming, /*initial_values=*/{},
                           channel_config));

    XLS_ASSERT_OK_AND_ASSIGN(
        ChannelWithInterfaces internal_ch_to_leaf2,
        pb_main.AddChannel("internal_ch_to_leaf2", p_->GetBitsType(32),
                           ChannelKind::kStreaming, /*initial_values=*/{},
                           channel_config));
    XLS_ASSERT_OK_AND_ASSIGN(
        ChannelWithInterfaces internal_ch_from_leaf2,
        pb_main.AddChannel("internal_ch_from_leaf2", p_->GetBitsType(32),
                           ChannelKind::kStreaming, /*initial_values=*/{},
                           channel_config));

    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * in_ch,
        pb_main.AddInputChannel("in_ch", p_->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * out_ch,
        pb_main.AddOutputChannel("out_ch", p_->GetBitsType(32)));
    pb_main.Send(internal_ch_to_leaf1.send_interface, pb_main.Receive(in_ch));
    pb_main.Send(internal_ch_to_leaf2.send_interface,
                 pb_main.Receive(internal_ch_from_leaf1.receive_interface));
    pb_main.Send(out_ch,
                 pb_main.Receive(internal_ch_from_leaf2.receive_interface));
    XLS_ASSERT_OK(pb_main.InstantiateProc(
        absl::StrCat(leaf->name(), "1"), leaf,
        std::vector<ChannelInterface*>{internal_ch_to_leaf1.receive_interface,
                                       internal_ch_from_leaf1.send_interface}));
    XLS_ASSERT_OK(pb_main.InstantiateProc(
        absl::StrCat(leaf->name(), "2"), leaf,
        std::vector<ChannelInterface*>{internal_ch_to_leaf2.receive_interface,
                                       internal_ch_from_leaf2.send_interface}));
  }

  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab_main,
                           ProcElaboration::Elaborate(main));

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
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf1")),
                           InstantiationKind::kBlock),
          m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf2")),
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

TEST_F(ProcInstantiationLoweringPassTest,
       SingleValueOutputChannelInferfaceInLeaf) {
  TokenlessProcBuilder pb_leaf(NewStyleProc(),
                               absl::StrCat(TestName(), "_leaf"),
                               /*token_name=*/"tkn", p_.get());
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * in_ch,
        pb_leaf.AddInputChannel("in_ch", p_->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * out_ch,
        pb_leaf.AddOutputChannel("out_ch", p_->GetBitsType(32),
                                 ChannelKind::kSingleValue));
    pb_leaf.Send(out_ch, pb_leaf.Receive(in_ch));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Proc * leaf, pb_leaf.Build());

  TokenlessProcBuilder pb_main(NewStyleProc(), TestName(), /*token_name=*/"tkn",
                               p_.get());
  {
    ChannelConfig channel_config = ChannelConfig()
                                       .WithFifoConfig(FifoConfig(
                                           /*depth=*/1,
                                           /*bypass=*/false,
                                           /*register_push_outputs=*/true,
                                           /*register_pop_outputs=*/false))
                                       .WithInputFlopKind(FlopKind::kNone)
                                       .WithOutputFlopKind(FlopKind::kNone);

    XLS_ASSERT_OK_AND_ASSIGN(
        ChannelWithInterfaces internal_ch_to_leaf,
        pb_main.AddChannel("internal_ch_to_leaf", p_->GetBitsType(32),
                           ChannelKind::kStreaming, /*initial_values=*/{},
                           channel_config));
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * in_ch,
        pb_main.AddInputChannel("in_ch", p_->GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * out_ch,
        pb_main.AddOutputChannel("out_ch", p_->GetBitsType(32),
                                 ChannelKind::kSingleValue));
    pb_main.Send(internal_ch_to_leaf.send_interface, pb_main.Receive(in_ch));
    XLS_ASSERT_OK(pb_main.InstantiateProc(
        leaf->name(), leaf,
        std::vector<ChannelInterface*>{internal_ch_to_leaf.receive_interface,
                                       out_ch}));
  }

  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab_main,
                           ProcElaboration::Elaborate(main));

  XLS_ASSERT_OK(p_->SetTop(main));
  XLS_ASSERT_OK(VerifyPackage(p_.get()));
  p_->AcceptInvalid();

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());
  EXPECT_THAT(block->GetInstantiations(),
              UnorderedElementsAre(
                  m::Instantiation(HasSubstr("fifo_internal_ch_to_leaf"),
                                   InstantiationKind::kFifo),
                  m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf")),
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

TEST_F(ProcInstantiationLoweringPassTest,
       SingleValueInputChannelInferfaceInLeaf) {
  TokenlessProcBuilder pb_leaf(NewStyleProc(),
                               absl::StrCat(TestName(), "_leaf"),
                               /*token_name=*/"tkn", p_.get());
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * in_ch,
        pb_leaf.AddInputChannel("in_ch", p_->GetBitsType(32),
                                ChannelKind::kSingleValue));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * out_ch,
        pb_leaf.AddOutputChannel("out_ch", p_->GetBitsType(32)));
    pb_leaf.Send(out_ch, pb_leaf.Receive(in_ch));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Proc * leaf, pb_leaf.Build());

  TokenlessProcBuilder pb_main(NewStyleProc(), TestName(), /*token_name=*/"tkn",
                               p_.get());
  {
    ChannelConfig channel_config = ChannelConfig()
                                       .WithFifoConfig(FifoConfig(
                                           /*depth=*/1,
                                           /*bypass=*/false,
                                           /*register_push_outputs=*/true,
                                           /*register_pop_outputs=*/false))
                                       .WithInputFlopKind(FlopKind::kNone)
                                       .WithOutputFlopKind(FlopKind::kNone);

    XLS_ASSERT_OK_AND_ASSIGN(
        ChannelWithInterfaces internal_ch_from_leaf,
        pb_main.AddChannel("internal_ch_from_leaf", p_->GetBitsType(32),
                           ChannelKind::kStreaming, /*initial_values=*/{},
                           channel_config));
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * in_ch,
        pb_main.AddInputChannel("in_ch", p_->GetBitsType(32),
                                ChannelKind::kSingleValue));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * out_ch,
        pb_main.AddOutputChannel("out_ch", p_->GetBitsType(32)));
    pb_main.Send(out_ch,
                 pb_main.Receive(internal_ch_from_leaf.receive_interface));
    XLS_ASSERT_OK(pb_main.InstantiateProc(
        leaf->name(), leaf,
        std::vector<ChannelInterface*>{in_ch,
                                       internal_ch_from_leaf.send_interface}));
  }

  XLS_ASSERT_OK_AND_ASSIGN(Proc * main, pb_main.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab_main,
                           ProcElaboration::Elaborate(main));

  XLS_ASSERT_OK(p_->SetTop(main));
  XLS_ASSERT_OK(VerifyPackage(p_.get()));
  p_->AcceptInvalid();

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, p_->GetTopAsBlock());
  EXPECT_THAT(block->GetInstantiations(),
              UnorderedElementsAre(
                  m::Instantiation(HasSubstr("fifo_internal_ch_from_leaf"),
                                   InstantiationKind::kFifo),
                  m::Instantiation(HasSubstr(absl::StrCat(TestName(), "_leaf")),
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

}  // namespace
}  // namespace xls::codegen
