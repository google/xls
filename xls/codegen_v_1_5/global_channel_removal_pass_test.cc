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

#include "xls/codegen_v_1_5/global_channel_removal_pass.h"

#include <cstdint>
#include <memory>
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
#include "xls/codegen_v_1_5/global_channel_block_stitching_pass.h"
#include "xls/codegen_v_1_5/scheduled_block_conversion_pass.h"
#include "xls/codegen_v_1_5/scheduling_pass.h"
#include "xls/codegen_v_1_5/state_to_register_io_lowering_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/verifier.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_result.h"
#include "xls/tools/schedule.h"

namespace xls::codegen {
namespace {

using ::testing::_;
using ::testing::IsEmpty;

class GlobalChannelRemovalPassTest : public IrTestBase {
 protected:
  void SetUp() override {
    p_ = CreatePackage();
    ran_pass_ = false;
  }

  void TearDown() override {
    if (!ran_pass_) {
      return;
    }

    for (const auto& [_, proc] : GetScheduledBlocksWithProcSources(p_.get())) {
      if (!proc->is_new_style_proc()) {
        EXPECT_THAT(p_->channels(), IsEmpty());
      }
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

    XLS_RETURN_IF_ERROR(
        GlobalChannelBlockStitchingPass().Run(p, options, &results).status());

    XLS_ASSIGN_OR_RETURN(bool result,
                         GlobalChannelRemovalPass().Run(p, options, &results));
    ran_pass_ = true;

    VLOG(5) << "IR after removal pass: " << p_->DumpIr();
    return result;
  }

  std::unique_ptr<VerifiedPackage> p_;
  bool ran_pass_;
};

TEST_F(GlobalChannelRemovalPassTest, NoProcs) {
  FunctionBuilder fb("f", p_.get());
  BValue a = fb.Param("a", p_->GetBitsType(32));
  BValue b = fb.Param("b", p_->GetBitsType(32));
  BValue c = fb.Param("c", p_->GetBitsType(32));
  BValue res = fb.UMul(c, fb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(res));
  XLS_ASSERT_OK(p_->SetTop(f));

  XLS_ASSERT_OK_AND_ASSIGN(bool changed, Run(p_.get(), /*pipeline_stages=*/2));
  EXPECT_FALSE(changed);
}

TEST_F(GlobalChannelRemovalPassTest, ProcWithProcScopedChannels) {
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
}

TEST_F(GlobalChannelRemovalPassTest, OneInstantiation) {
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
}

}  // namespace
}  // namespace xls::codegen
