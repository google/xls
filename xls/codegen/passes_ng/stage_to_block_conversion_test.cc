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

#include "xls/codegen/passes_ng/stage_to_block_conversion.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/passes_ng/block_channel_adapter.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/evaluator_options.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/ir/verifier.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/codegen.h"
#include "xls/tools/codegen_flags.pb.h"
#include "xls/tools/scheduling_options_flags.pb.h"

namespace xls::verilog {
namespace {

namespace m = ::xls::op_matchers;
using ::absl_testing::IsOkAndHolds;
using ::testing::Field;
using ::testing::Optional;

class BlockConversionTestBase : public IrTestBase {
 protected:
  // Creates a runtime for the given proc and the proc hierarchy beneath it.
  std::unique_ptr<ProcRuntime> CreateRuntime(
      Proc* top, const EvaluatorOptions& options = EvaluatorOptions()) const {
    return CreateInterpreterSerialProcRuntime(top, options).value();
  }

  virtual CodegenOptions codegen_options() {
    return CodegenOptions().module_name(TestName());
  }
};

// Fixture to sweep pipeline stages.
class SweepPipelineStagesFixture : public BlockConversionTestBase,
                                   public testing::WithParamInterface<int64_t> {
 public:
  static std::string PrintToStringParamName(
      const testing::TestParamInfo<ParamType>& info) {
    return absl::StrFormat("stage_count_%d", info.param);
  }

  int64_t GetStageCount() const { return GetParam(); }
};

// Simple pipelined function test fixture.
class SweepTrivialPipelinedFunctionFixture : public SweepPipelineStagesFixture {
 public:
  CodegenOptions codegen_options() override {
    return SweepPipelineStagesFixture::codegen_options()
        .flop_inputs(false)
        .flop_outputs(true)
        .clock_name("clk")
        .reset("rst", /*asynchronous=*/false, /*active_low=*/false,
               /*reset_data_path=*/true);
  }

  absl::Status CreateStageProcInPackage() {
    package_ = CreatePackage();

    FunctionBuilder fb(TestName(), package_.get());
    BValue x = fb.Param("x", package_->GetBitsType(32));
    BValue y = fb.Param("y", package_->GetBitsType(32));

    XLS_ASSIGN_OR_RETURN(
        Function * f,
        fb.BuildWithReturnValue(fb.Add(fb.Negate(fb.Not(fb.Add(x, y))), x)));
    XLS_RETURN_IF_ERROR(package_->SetTop(f));

    XLS_ASSIGN_OR_RETURN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            f, TestDelayEstimator(),
            SchedulingOptions().pipeline_stages(GetStageCount())));

    XLS_RET_CHECK_OK(SingleFunctionBaseToPipelinedStages(
        "top", schedule, codegen_options(), stage_conversion_metadata_));

    return absl::OkStatus();
  }

 protected:
  std::unique_ptr<Package> package_;
  StageConversionMetadata stage_conversion_metadata_;
  BlockConversionMetadata block_conversion_metadata_;
};

TEST_P(SweepTrivialPipelinedFunctionFixture, TestBlockAndClocksCreation) {
  XLS_ASSERT_OK(CreateStageProcInPackage());

  XLS_ASSERT_OK_AND_ASSIGN(ProcMetadata * top_metadata,
                           stage_conversion_metadata_.GetTopProcMetadata(
                               package_->GetTop().value()));

  XLS_ASSERT_OK(CreateBlocksForProcHierarchy(codegen_options(), *top_metadata,
                                             stage_conversion_metadata_,
                                             block_conversion_metadata_)
                    .status());

  XLS_ASSERT_OK(AddResetAndClockPortsToBlockHierarchy(
                    codegen_options(), *top_metadata,
                    stage_conversion_metadata_, block_conversion_metadata_)
                    .status());

  XLS_VLOG_LINES(2, package_->DumpIr());

  // There should be the same number of blocks, procs, and metadata entries.
  EXPECT_EQ(package_->blocks().size(), package_->procs().size());

  // Each block should have a clock/reset and a corresponding metadata entry.
  for (std::unique_ptr<Block>& block : package_->blocks()) {
    EXPECT_THAT(block->GetClockPort(),
                Optional(Field(&Block::ClockPort::name, "clk")));
    EXPECT_THAT(block->GetResetPort(), Optional(m::InputPort("rst")));

    XLS_ASSERT_OK_AND_ASSIGN(
        BlockMetadata * block_metadata,
        block_conversion_metadata_.GetBlockMetadata(block.get()));
    EXPECT_EQ(block_metadata->block(), block.get());

    // It should be possible to get the same block metadata from the proc
    // metadata created by stage conversion.
    ProcMetadata* proc_metadata = block_metadata->stage_metadata();
    XLS_ASSERT_OK_AND_ASSIGN(
        BlockMetadata * block_metadata_from_conversion_metadata,
        block_conversion_metadata_.GetBlockMetadata(proc_metadata));
    EXPECT_EQ(block_metadata, block_metadata_from_conversion_metadata);
  }
}

INSTANTIATE_TEST_SUITE_P(
    TestBlockAndClocksCreation, SweepTrivialPipelinedFunctionFixture,
    testing::Values(1, 2, 3, 4),
    SweepTrivialPipelinedFunctionFixture::PrintToStringParamName);

TEST_P(SweepTrivialPipelinedFunctionFixture, TestBlockChannelMetadata) {
  XLS_ASSERT_OK(CreateStageProcInPackage());

  XLS_ASSERT_OK_AND_ASSIGN(ProcMetadata * top_metadata,
                           stage_conversion_metadata_.GetTopProcMetadata(
                               package_->GetTop().value()));

  XLS_ASSERT_OK_AND_ASSIGN(
      Block * top_block,
      CreateBlocksForProcHierarchy(codegen_options(), *top_metadata,
                                   stage_conversion_metadata_,
                                   block_conversion_metadata_));

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockMetadata * top_block_metadata,
      block_conversion_metadata_.GetBlockMetadata(top_block));

  // Add Ports to the block
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * literal_1,
      top_block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(1, 1))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * literal_32,
      top_block->MakeNode<xls::Literal>(SourceInfo(), Value(UBits(0, 32))));

  XLS_ASSERT_OK_AND_ASSIGN(
      InputPort * x, top_block->AddInputPort("x", package_->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      InputPort * x_valid,
      top_block->AddInputPort("x_valid", package_->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(OutputPort * x_rdy,
                           top_block->AddOutputPort("x_rdy", literal_1));

  XLS_ASSERT_OK_AND_ASSIGN(
      InputPort * y, top_block->AddInputPort("y", package_->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      InputPort * y_valid,
      top_block->AddInputPort("y_valid", package_->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(OutputPort * y_rdy,
                           top_block->AddOutputPort("y_rdy", literal_1));

  XLS_ASSERT_OK_AND_ASSIGN(OutputPort * out,
                           top_block->AddOutputPort("out", literal_32));
  XLS_ASSERT_OK_AND_ASSIGN(OutputPort * out_valid,
                           top_block->AddOutputPort("out_valid", literal_32));
  XLS_ASSERT_OK_AND_ASSIGN(
      InputPort * out_rdy,
      top_block->AddInputPort("out_rdy", package_->GetBitsType(1)));

  // Add slots/adapters and metadata for the channels
  Proc* top_proc = top_block_metadata->stage_metadata()->proc();
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * recv_x,
                           top_proc->GetReceiveChannelInterface("x"));
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * recv_y,
                           top_proc->GetReceiveChannelInterface("y"));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * send_out,
                           top_proc->GetSendChannelInterface("out"));

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockRDVSlot recv_x_slot,
      BlockRDVSlot::CreateReceiveSlot("recv_x", RDVNodeGroup{x_rdy, x, x_valid},
                                      top_block));
  XLS_ASSERT_OK_AND_ASSIGN(
      BlockRDVSlot recv_y_slot,
      BlockRDVSlot::CreateReceiveSlot("recv_y", RDVNodeGroup{y_rdy, y, y_valid},
                                      top_block));
  XLS_ASSERT_OK_AND_ASSIGN(
      BlockRDVSlot send_out_slot,
      BlockRDVSlot::CreateSendSlot(
          "send_out", RDVNodeGroup{out_rdy, out, out_valid}, top_block));

  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter recv_x_adapter,
      RDVAdapter::CreateInterfaceReceiveAdapter(recv_x_slot, top_block));
  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter recv_y_adapter,
      RDVAdapter::CreateInterfaceReceiveAdapter(recv_y_slot, top_block));
  XLS_ASSERT_OK_AND_ASSIGN(
      RDVAdapter send_out_adapter,
      RDVAdapter::CreateInterfaceSendAdapter(send_out_slot, top_block));

  XLS_VLOG_LINES(2, top_block->DumpIr());
  XLS_VLOG_LINES(2, top_proc->DumpIr());

  BlockChannelMetadata input_x_metadata(recv_x);
  BlockChannelMetadata input_y_metadata(recv_y);
  BlockChannelMetadata output_metadata(send_out);

  // Test adding an adapter after the fact.
  XLS_ASSERT_OK(
      input_x_metadata.AddSlot(recv_x_slot).AddAdapter(recv_x_adapter));
  XLS_ASSERT_OK(
      output_metadata.AddSlot(send_out_slot).AddAdapter(send_out_adapter));

  EXPECT_FALSE(input_y_metadata.adapter().has_value());
  XLS_ASSERT_OK(
      input_y_metadata.AddSlot(recv_y_slot).AddAdapter(recv_y_adapter));
  EXPECT_TRUE(input_y_metadata.adapter().has_value());

  // Test accessors.
  EXPECT_EQ(input_x_metadata.channel_interface(), recv_x);
  EXPECT_EQ(input_y_metadata.channel_interface(), recv_y);
  EXPECT_EQ(output_metadata.channel_interface(), send_out);
  EXPECT_EQ(input_x_metadata.direction(), ChannelDirection::kReceive);
  EXPECT_EQ(input_y_metadata.direction(), ChannelDirection::kReceive);
  EXPECT_EQ(output_metadata.direction(), ChannelDirection::kSend);

  EXPECT_FALSE(input_x_metadata.IsInternalChannel());
  EXPECT_FALSE(input_y_metadata.IsInternalChannel());
  EXPECT_FALSE(output_metadata.IsInternalChannel());

  // Add to Block Metadata
  top_block_metadata->AddChannelMetadata(std::move(input_x_metadata));
  top_block_metadata->AddChannelMetadata(std::move(input_y_metadata));
  top_block_metadata->AddChannelMetadata(std::move(output_metadata));

  EXPECT_EQ(top_block_metadata->inputs().size(), 2);
  EXPECT_EQ(top_block_metadata->outputs().size(), 1);

  EXPECT_THAT(top_block_metadata->GetChannelMetadata(recv_x),
              IsOkAndHolds(top_block_metadata->inputs()[0].get()));
  EXPECT_THAT(top_block_metadata->GetChannelMetadata(recv_y),
              IsOkAndHolds(top_block_metadata->inputs()[1].get()));
  EXPECT_THAT(top_block_metadata->GetChannelMetadata(send_out),
              IsOkAndHolds(top_block_metadata->outputs()[0].get()));
}

INSTANTIATE_TEST_SUITE_P(
    TestBlockChannelMetadata, SweepTrivialPipelinedFunctionFixture,
    testing::Values(1, 2, 3, 4),
    SweepTrivialPipelinedFunctionFixture::PrintToStringParamName);

}  // namespace
}  // namespace xls::verilog
