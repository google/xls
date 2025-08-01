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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/codegen/block_conversion_test_fixture.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/passes_ng/block_channel_adapter.h"
#include "xls/codegen/passes_ng/block_channel_slot.h"
#include "xls/codegen/passes_ng/block_pipeline_inserter.h"
#include "xls/codegen/passes_ng/stage_conversion.h"
#include "xls/common/logging/log_lines.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
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

namespace xls::verilog {
namespace {

namespace m = ::xls::op_matchers;
using ::absl_testing::IsOkAndHolds;
using ::testing::ElementsAreArray;
using ::testing::Field;
using ::testing::Optional;

// Fixture to sweep pipeline stages.
class SweepPipelineStagesFixture : public BlockConversionTestFixture,
                                   public testing::WithParamInterface<int64_t> {
 public:
  virtual CodegenOptions codegen_options() {
    return CodegenOptions().module_name(TestName());
  }

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
        Function * f, fb.BuildWithReturnValue(fb.Identity(
                          fb.Add(fb.Add(fb.Identity(x), fb.Identity(y)), x))));
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

  // Simulates the block with the given name and returns the output sequence.
  absl::StatusOr<std::vector<uint64_t>> SimulateBlock(
      std::string_view block_name, absl::Span<const uint64_t> x,
      absl::Span<const uint64_t> y, uint64_t cycle_count) {
    XLS_RET_CHECK_EQ(x.size(), y.size());

    // Interpret the stage block itself.
    XLS_ASSIGN_OR_RETURN(Block * block, package_->GetBlock(block_name));

    std::vector<ChannelSource> sources{
        ChannelSource("x", "x_vld", "x_rdy", 1.0, block),
        ChannelSource("y", "y_vld", "y_rdy", 1.0, block)};

    XLS_RETURN_IF_ERROR(sources[0].SetDataSequence(x));
    XLS_RETURN_IF_ERROR(sources[1].SetDataSequence(y));

    std::vector<ChannelSink> sinks{
        ChannelSink("out", "out_vld", "out_rdy", 0.25, block),
    };
    std::vector<absl::flat_hash_map<std::string, uint64_t>> inputs(
        100, {{"rst", 0}});
    inputs[0]["rst"] = 1;

    XLS_ASSIGN_OR_RETURN(
        BlockIOResultsAsUint64 results,
        InterpretChannelizedSequentialBlockWithUint64(
            block, absl::MakeSpan(sources), absl::MakeSpan(sinks), inputs,
            codegen_options().reset()));

    // Add a cycle count for easier comparison with simulation results.
    XLS_RETURN_IF_ERROR(
        SetIncrementingSignalOverCycles(0, results.outputs.size() - 1, "cycle",
                                        0, results.outputs)
            .status());

    VLOG(1) << "Signal Trace";
    XLS_RETURN_IF_ERROR(VLogTestPipelinedIO(
        std::vector<SignalSpec>{{"cycle", SignalType::kOutput},
                                {"rst", SignalType::kInput},
                                {"x", SignalType::kInput},
                                {"x_vld", SignalType::kInput},
                                {"x_rdy", SignalType::kOutput},
                                {"y", SignalType::kInput},
                                {"y_vld", SignalType::kInput},
                                {"y_rdy", SignalType::kOutput},
                                {"out", SignalType::kOutput},
                                {"out_vld", SignalType::kOutput},
                                {"out_rdy", SignalType::kInput}},
        /*column_width=*/10, results.inputs, results.outputs));

    return sinks[0].GetOutputSequenceAsUint64();
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

// Test the conversion to stage blocks by creating a multi-stage and/or
// combinational pipeline, sensitizing the block and testing the I/O behavior.
TEST_P(SweepTrivialPipelinedFunctionFixture, TestConvertProcHierarchyCreation) {
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

  XLS_ASSERT_OK(ConvertProcHierarchyToBlocks(codegen_options(), *top_metadata,
                                             stage_conversion_metadata_,
                                             block_conversion_metadata_)
                    .status());

  XLS_VLOG_LINES(2, package_->DumpIr());

  // Simulate the pipeline
  // out = 2*x + y
  std::vector<uint64_t> x = {0x1, 0x10, 0x30};
  std::vector<uint64_t> y = {0x2, 0x20, 0x30};

  std::vector<uint64_t> out_expected(x.size());
  for (int64_t i = 0; i < out_expected.size(); ++i) {
    out_expected[i] = x[i] * 2 + y[i];
  }

  EXPECT_THAT(SimulateBlock(top_metadata->proc()->name(), absl::MakeSpan(x),
                            absl::MakeSpan(y), /*cycle_count=*/100),
              IsOkAndHolds(ElementsAreArray(out_expected)));
}

// Test the conversion to stage blocks by creating a multi-stage pipeline,
// sensitizing the block and testing the I/O behavior.
TEST_P(SweepTrivialPipelinedFunctionFixture, TestPipelineCreation) {
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

  XLS_ASSERT_OK(ConvertProcHierarchyToBlocks(codegen_options(), *top_metadata,
                                             stage_conversion_metadata_,
                                             block_conversion_metadata_)
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN(
      BlockMetadata * top_block_metadata,
      block_conversion_metadata_.GetBlockMetadata(top_metadata));
  XLS_ASSERT_OK(
      InsertPipelineIntoBlock(codegen_options(), *top_block_metadata));

  XLS_VLOG_LINES(2, package_->DumpIr());

  // Simulate the pipeline
  // out = 2*x + y
  std::vector<uint64_t> x = {0x1, 0x10, 0x30};
  std::vector<uint64_t> y = {0x2, 0x20, 0x30};

  std::vector<uint64_t> out_expected(x.size());
  for (int64_t i = 0; i < out_expected.size(); ++i) {
    out_expected[i] = x[i] * 2 + y[i];
  }

  EXPECT_THAT(SimulateBlock(top_metadata->proc()->name(), absl::MakeSpan(x),
                            absl::MakeSpan(y), /*cycle_count=*/100),
              IsOkAndHolds(ElementsAreArray(out_expected)));
}

INSTANTIATE_TEST_SUITE_P(
    TestProcHierarchyCreationAndSimulation,
    SweepTrivialPipelinedFunctionFixture, testing::Values(1, 2, 3, 4, 5),
    SweepTrivialPipelinedFunctionFixture::PrintToStringParamName);

}  // namespace
}  // namespace xls::verilog
