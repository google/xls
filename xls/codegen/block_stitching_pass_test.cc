// Copyright 2024 The XLS Authors
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

#include "xls/codegen/block_stitching_pass.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/random/distributions.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/substitute.h"
#include "absl/types/span.h"
#include "xls/codegen/block_conversion.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/side_effect_condition_pass.h"
#include "xls/codegen/signature_generation_pass.h"
#include "xls/common/source_location.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
#include "xls/interpreter/block_evaluator.h"
#include "xls/interpreter/block_interpreter.h"
#include "xls/interpreter/channel_queue.h"
#include "xls/interpreter/interpreter_proc_runtime.h"
#include "xls/interpreter/serial_proc_runtime.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/events.h"
#include "xls/ir/foreign_function.h"
#include "xls/ir/foreign_function_data.pb.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/instantiation.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/ir/value_utils.h"
#include "xls/passes/channel_legalization_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/tools/codegen.h"

namespace xls::verilog {
namespace {

namespace m = ::xls::op_matchers;

using ::testing::_;
using ::testing::AllOf;
using ::testing::AnyOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::Eq;
using ::testing::FieldsAre;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;
using ::testing::Pair;
using ::testing::UnorderedElementsAre;
using ::xls::status_testing::IsOkAndHolds;
using ::xls::status_testing::StatusIs;

using BlockStitchingPassTest = IrTestBase;

// Convenience wrapper for creating a fifo config.
// If `depth` is 0, the fifo will be configured to bypass, otherwise it will be
// no-bypass.
FifoConfig FifoConfigWithDepth(int64_t depth) {
  return FifoConfig(/*depth=*/depth, /*bypass=*/depth == 0,
                    /*register_push_outputs=*/false,
                    /*register_pop_outputs=*/false);
}

CodegenOptions DefaultCodegenOptions() {
  return CodegenOptions()
      .flop_inputs(false)
      .flop_outputs(false)
      .clock_name("clk")
      .valid_control("input_valid", "output_valid")
      .reset("rst", false, false, true)
      .streaming_channel_data_suffix("_data")
      .streaming_channel_valid_suffix("_valid")
      .streaming_channel_ready_suffix("_ready");
}

// Run channel legalization, multi-proc scheduling, block conversion,
// side-effect condition pass, and ultimately the  block stitching pass on the
// given package. If `unit_out` is non-null, the codegen unit will be returned
// in it.
absl::StatusOr<std::pair<bool, CodegenPassUnit>> RunBlockStitchingPass(
    Package* p, std::string_view top_name = "top_proc",
    CodegenOptions options = DefaultCodegenOptions(),
    bool generate_signature = true) {
  options.module_name(top_name);
  if (!p->GetTop().has_value()) {
    XLS_RETURN_IF_ERROR(p->SetTop(p->GetFunctionBases().front()));
  }
  // Run channel legalization pass to test that multiple send/recv on the same
  // channel works.
  PassResults opt_results;
  XLS_ASSIGN_OR_RETURN(
      bool changed, ChannelLegalizationPass().Run(p, OptimizationPassOptions(),
                                                  &opt_results));
  TestDelayEstimator delay_estimator;
  XLS_ASSIGN_OR_RETURN(PipelineScheduleOrGroup schedule,
                       Schedule(p,
                                SchedulingOptions()
                                    .clock_period_ps(100)
                                    .ffi_fallback_delay_ps(100)
                                    .pipeline_stages(10)
                                    // Multi-proc scheduling
                                    .schedule_all_procs(true),
                                &delay_estimator));
  XLS_RET_CHECK(std::holds_alternative<PackagePipelineSchedules>(schedule));

  XLS_ASSIGN_OR_RETURN(
      CodegenPassUnit unit,
      PackageToPipelinedBlocks(std::get<PackagePipelineSchedules>(schedule),
                               options, p));
  CodegenCompoundPass ccp("block_stitching_and_friends",
                          "Block stitching and friends.");
  // Some tests rely on the side effect condition pass to update the predicates
  // of side-effecting ops.
  ccp.Add<SideEffectConditionPass>();
  ccp.Add<BlockStitchingPass>();
  if (generate_signature) {
    ccp.Add<SignatureGenerationPass>();
  }
  CodegenPassResults results;
  XLS_ASSIGN_OR_RETURN(
      bool ccp_changed,
      ccp.Run(&unit, CodegenPassOptions{.codegen_options = options}, &results));
  changed = changed || ccp_changed;
  return std::make_pair(changed, unit);
}

TEST_F(BlockStitchingPassTest, SingleBlockIsNoop) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch0,
      p->CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch1,
      p->CreateStreamingChannel("ch1", ChannelOps::kSendOnly, u32));
  ProcBuilder pb(TestName(), p.get());
  BValue rcv = pb.Receive(ch0, pb.AfterAll({}));
  pb.Send(ch1, pb.TupleIndex(rcv, 0), pb.TupleIndex(rcv, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK(p->SetTop(proc));

  EXPECT_THAT(
      RunBlockStitchingPass(
          p.get(), /*top_name=*/"top_proc", /*options=*/DefaultCodegenOptions(),
          /*generate_signature=*/false),  // Don't generate signature, otherwise
                                          // we'll always get changed=true.
      IsOkAndHolds(Pair(false, _)));
}

TEST_F(BlockStitchingPassTest, StitchNetworkWithFifos) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch0,
      p->CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch1,
      p->CreateStreamingChannel("ch1", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/
                                FifoConfig(/*depth=*/10, /*bypass=*/false,
                                           /*register_push_outputs=*/false,
                                           /*register_pop_outputs=*/false)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch2,
      p->CreateStreamingChannel("ch2", ChannelOps::kSendOnly, u32));

  ProcBuilder pb0(absl::StrCat(TestName(), 0), p.get());
  BValue rcv0 = pb0.Receive(ch0, pb0.AfterAll({}));
  pb0.Send(ch1, pb0.TupleIndex(rcv0, 0), pb0.TupleIndex(rcv0, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc0, pb0.Build());

  ProcBuilder pb1(absl::StrCat(TestName(), 1), p.get());
  BValue rcv1 = pb1.Receive(ch1, pb1.AfterAll({}));
  pb1.Send(ch2, pb1.TupleIndex(rcv1, 0), pb1.TupleIndex(rcv1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, pb1.Build());
  XLS_ASSERT_OK(p->SetTop(proc0));

  EXPECT_THAT(RunBlockStitchingPass(p.get()), IsOkAndHolds(Pair(true, _)));
  EXPECT_THAT(p->blocks(), UnorderedElementsAre(m::Block("top_proc__1"),
                                                m::Block("top_proc"),
                                                m::Block(proc1->name())));
  EXPECT_THAT(
      p->GetBlock("top_proc").value()->GetInstantiations(),
      UnorderedElementsAre(
          m::Instantiation(proc1->name() + "_inst0", InstantiationKind::kBlock),
          m::Instantiation("top_proc__1_inst1", InstantiationKind::kBlock),
          m::Instantiation(HasSubstr("fifo"), InstantiationKind::kFifo)));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * inst1,
      p->GetBlock("top_proc").value()->GetInstantiation("top_proc__1_inst1"));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Instantiation * inst0,
                           p->GetBlock("top_proc")
                               .value()
                               ->GetInstantiation(proc1->name() + "_inst0"));
  EXPECT_THAT(
      p->GetBlock("top_proc").value()->nodes(),
      IsSupersetOf({
          m::InstantiationInput(m::InputPort("ch0_valid"), "ch0_valid",
                                Eq(inst1)),
          m::InstantiationInput(m::InputPort("ch0_data"), "ch0_data",
                                Eq(inst1)),
          m::InstantiationInput(
              m::InstantiationOutput(
                  "push_ready", m::Instantiation(HasSubstr("fifo"),
                                                 InstantiationKind::kFifo)),
              "ch1_ready", Eq(inst1)),
          m::InstantiationOutput("ch1_data", Eq(inst1)),
          m::InstantiationOutput("ch1_valid", Eq(inst1)),
          m::InstantiationOutput("ch0_ready", Eq(inst1)),
          m::InstantiationInput(
              m::InstantiationOutput(
                  "pop_valid", m::Instantiation(HasSubstr("fifo"),
                                                InstantiationKind::kFifo)),
              "ch1_valid", Eq(inst0)),
          m::InstantiationInput(
              m::InstantiationOutput(
                  "pop_data", m::Instantiation(HasSubstr("fifo"),
                                               InstantiationKind::kFifo)),
              "ch1_data", Eq(inst0)),
          m::InstantiationInput(m::InputPort("ch2_ready"), "ch2_ready",
                                Eq(inst0)),
          m::InstantiationOutput("ch2_data", Eq(inst0)),
          m::InstantiationOutput("ch2_valid", Eq(inst0)),
          m::InstantiationOutput("ch1_ready", Eq(inst0)),
      }));
}

TEST_F(BlockStitchingPassTest, StitchNetworkWithDirectConnections) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch0,
      p->CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch1,
      p->CreateStreamingChannel("ch1", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/
                                FifoConfig(/*depth=*/0, /*bypass=*/true,
                                           /*register_push_outputs=*/false,
                                           /*register_pop_outputs=*/false)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch2,
      p->CreateStreamingChannel("ch2", ChannelOps::kSendOnly, u32));

  ProcBuilder pb0(absl::StrCat(TestName(), 0), p.get());
  BValue rcv0 = pb0.Receive(ch0, pb0.AfterAll({}));
  pb0.Send(ch1, pb0.TupleIndex(rcv0, 0), pb0.TupleIndex(rcv0, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc0, pb0.Build());

  ProcBuilder pb1(absl::StrCat(TestName(), 1), p.get());
  BValue rcv1 = pb1.Receive(ch1, pb1.AfterAll({}));
  pb1.Send(ch2, pb1.TupleIndex(rcv1, 0), pb1.TupleIndex(rcv1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, pb1.Build());

  XLS_ASSERT_OK(p->SetTop(proc0));
  EXPECT_THAT(RunBlockStitchingPass(p.get()), IsOkAndHolds(Pair(true, _)));
  EXPECT_THAT(p->blocks(), UnorderedElementsAre(m::Block("top_proc__1"),
                                                m::Block("top_proc"),
                                                m::Block(proc1->name())));
  EXPECT_THAT(
      p->GetBlock("top_proc").value()->GetInstantiations(),
      UnorderedElementsAre(
          m::Instantiation(proc1->name() + "_inst0", InstantiationKind::kBlock),
          m::Instantiation("top_proc__1_inst1", InstantiationKind::kBlock)));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Instantiation * inst0,
                           p->GetBlock("top_proc")
                               .value()
                               ->GetInstantiation(proc1->name() + "_inst0"));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * inst1,
      p->GetBlock("top_proc").value()->GetInstantiation("top_proc__1_inst1"));
  EXPECT_THAT(
      p->GetBlock("top_proc").value()->nodes(),
      IsSupersetOf({
          m::InstantiationInput(m::InputPort("ch0_valid"), "ch0_valid",
                                Eq(inst1)),
          m::InstantiationInput(m::InputPort("ch0_data"), "ch0_data",
                                Eq(inst1)),
          m::InstantiationInput(
              m::InstantiationOutput(
                  "ch1_ready", m::Instantiation(proc1->name() + "_inst0")),
              "ch1_ready", Eq(inst1)),
          m::InstantiationOutput("ch1_data", Eq(inst1)),
          m::InstantiationOutput("ch1_valid", Eq(inst1)),
          m::InstantiationOutput("ch0_ready", Eq(inst1)),
          m::InstantiationInput(
              m::InstantiationOutput("ch1_valid",
                                     m::Instantiation("top_proc__1_inst1")),
              "ch1_valid", Eq(inst0)),
          m::InstantiationInput(
              m::InstantiationOutput("ch1_data",
                                     m::Instantiation("top_proc__1_inst1")),
              "ch1_data", Eq(inst0)),
          m::InstantiationInput(m::InputPort("ch2_ready"), "ch2_ready",
                                Eq(inst0)),
          m::InstantiationOutput("ch2_data", Eq(inst0)),
          m::InstantiationOutput("ch2_valid", Eq(inst0)),
          m::InstantiationOutput("ch1_ready", Eq(inst0)),
      }));
}

TEST_F(BlockStitchingPassTest, BlocksWithHardToUniquifyNames) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch0,
      p->CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch1,
      p->CreateStreamingChannel("ch1", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/
                                FifoConfig(/*depth=*/0, /*bypass=*/true,
                                           /*register_push_outputs=*/false,
                                           /*register_pop_outputs=*/false)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch2,
      p->CreateStreamingChannel("ch2", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/
                                FifoConfig(/*depth=*/0, /*bypass=*/true,
                                           /*register_push_outputs=*/false,
                                           /*register_pop_outputs=*/false)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch3,
      p->CreateStreamingChannel("ch3", ChannelOps::kSendOnly, u32));

  ProcBuilder pb0(TestName(), p.get());
  BValue rcv0 = pb0.Receive(ch0, pb0.AfterAll({}));
  pb0.Send(ch1, pb0.TupleIndex(rcv0, 0), pb0.TupleIndex(rcv0, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc0, pb0.Build());

  ProcBuilder pb1(absl::StrCat(TestName(), "__1"), p.get());
  BValue rcv1 = pb1.Receive(ch1, pb1.AfterAll({}));
  pb1.Send(ch2, pb1.TupleIndex(rcv1, 0), pb1.TupleIndex(rcv1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, pb1.Build());

  ProcBuilder pb2(absl::StrCat(TestName(), "__2"), p.get());
  BValue rcv2 = pb2.Receive(ch2, pb2.AfterAll({}));
  pb2.Send(ch3, pb2.TupleIndex(rcv2, 0), pb2.TupleIndex(rcv2, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc2, pb2.Build());

  XLS_ASSERT_OK(p->SetTop(proc0));
  EXPECT_THAT(RunBlockStitchingPass(p.get(), /*top_name=*/TestName()),
              IsOkAndHolds(Pair(true, _)));
  EXPECT_THAT(p->blocks(),
              UnorderedElementsAre(
                  m::Block(proc0->name()), m::Block(proc1->name()),
                  m::Block(proc2->name()),
                  // proc0 should be renamed to next available suffix to make
                  // room for top, which has the name proc0 originally had.
                  m::Block(absl::StrCat(TestName(), "__3"))));
}

TEST_F(BlockStitchingPassTest, StitchBlockWithLoopback) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch0,
      p->CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch1,
      p->CreateStreamingChannel("ch1", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/
                                FifoConfigWithDepth(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch2,
      p->CreateStreamingChannel("ch2", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/
                                FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch3,
      p->CreateStreamingChannel("ch3", ChannelOps::kSendOnly, u32));

  Proc* proc0;
  {
    ProcBuilder pb0(absl::StrCat(TestName(), 0), p.get());
    BValue rcv0 = pb0.Receive(ch0, pb0.AfterAll({}));
    BValue send0 =
        pb0.Send(ch1, pb0.TupleIndex(rcv0, 0), pb0.TupleIndex(rcv0, 1));
    BValue rcv1 = pb0.Receive(ch1, send0);
    pb0.Send(ch2, pb0.TupleIndex(rcv1, 0), pb0.TupleIndex(rcv1, 1));
    XLS_ASSERT_OK_AND_ASSIGN(proc0, pb0.Build());
  }

  Proc* proc1;
  {
    ProcBuilder pb1(absl::StrCat(TestName(), 1), p.get());
    BValue rcv0 = pb1.Receive(ch2, pb1.AfterAll({}));
    pb1.Send(ch3, pb1.TupleIndex(rcv0, 0), pb1.TupleIndex(rcv0, 1));
    XLS_ASSERT_OK_AND_ASSIGN(proc1, pb1.Build());
  }

  XLS_ASSERT_OK(p->SetTop(proc0));
  EXPECT_THAT(RunBlockStitchingPass(p.get()), IsOkAndHolds(Pair(true, _)));
  EXPECT_THAT(p->blocks(), UnorderedElementsAre(m::Block("top_proc__1"),
                                                m::Block("top_proc"),
                                                m::Block(proc1->name())));
  EXPECT_THAT(
      p->GetBlock("top_proc").value()->GetInstantiations(),
      UnorderedElementsAre(
          m::Instantiation(proc1->name() + "_inst0", InstantiationKind::kBlock),
          m::Instantiation("top_proc__1_inst1", InstantiationKind::kBlock)));
  XLS_ASSERT_OK_AND_ASSIGN(xls::Instantiation * inst0,
                           p->GetBlock("top_proc")
                               .value()
                               ->GetInstantiation(proc1->name() + "_inst0"));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * inst1,
      p->GetBlock("top_proc").value()->GetInstantiation("top_proc__1_inst1"));
  EXPECT_THAT(
      p->GetBlock("top_proc").value()->nodes(),
      IsSupersetOf({
          m::InstantiationInput(m::InputPort("ch0_valid"), "ch0_valid",
                                Eq(inst1)),
          m::InstantiationInput(m::InputPort("ch0_data"), "ch0_data",
                                Eq(inst1)),
          m::InstantiationInput(
              m::InstantiationOutput(
                  "ch2_ready", m::Instantiation(proc1->name() + "_inst0")),
              "ch2_ready", Eq(inst1)),
          m::InstantiationOutput("ch2_data", Eq(inst1)),
          m::InstantiationOutput("ch2_valid", Eq(inst1)),
          m::InstantiationOutput("ch0_ready", Eq(inst1)),
          m::InstantiationInput(
              m::InstantiationOutput("ch2_valid",
                                     m::Instantiation("top_proc__1_inst1")),
              "ch2_valid", Eq(inst0)),
          m::InstantiationInput(
              m::InstantiationOutput("ch2_data",
                                     m::Instantiation("top_proc__1_inst1")),
              "ch2_data", Eq(inst0)),
          m::InstantiationInput(m::InputPort("ch3_ready"), "ch3_ready",
                                Eq(inst0)),
          m::InstantiationOutput("ch3_data", Eq(inst0)),
          m::InstantiationOutput("ch3_valid", Eq(inst0)),
          m::InstantiationOutput("ch2_ready", Eq(inst0)),
      }));
  EXPECT_THAT(p->GetBlock("top_proc__1").value()->GetInstantiations(),
              UnorderedElementsAre(
                  m::Instantiation("fifo_ch1", InstantiationKind::kFifo)));
}

TEST_F(BlockStitchingPassTest, StitchBlockWithFfi) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch0,
      p->CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch1,
      p->CreateStreamingChannel("ch1", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/
                                FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch2,
      p->CreateStreamingChannel("ch2", ChannelOps::kSendOnly, u32));

  Function* ffi_fun;
  {
    FunctionBuilder fb("ffi_func", p.get());
    const BValue param_a = fb.Param("a", u32);
    const BValue param_b = fb.Param("b", u32);
    const BValue add = fb.Add(param_a, param_b);
    XLS_ASSERT_OK_AND_ASSIGN(ForeignFunctionData ffd,
                             ForeignFunctionDataCreateFromTemplate(
                                 "foo {fn} (.ma({a}), .mb{b}) .out({return})"));
    fb.SetForeignFunctionData(ffd);
    XLS_ASSERT_OK_AND_ASSIGN(ffi_fun, fb.BuildWithReturnValue(add));
  }

  Proc* proc0;
  {
    ProcBuilder pb0(absl::StrCat(TestName(), 0), p.get());
    BValue rcv0 = pb0.Receive(ch0, pb0.AfterAll({}));
    BValue rcv0_data = pb0.TupleIndex(rcv0, 1);
    BValue invocation = pb0.Invoke({rcv0_data, rcv0_data}, ffi_fun);
    pb0.Send(ch1, pb0.TupleIndex(rcv0, 0), invocation);
    XLS_ASSERT_OK_AND_ASSIGN(proc0, pb0.Build());
  }

  Proc* proc1;
  {
    ProcBuilder pb1(absl::StrCat(TestName(), 1), p.get());
    BValue rcv0 = pb1.Receive(ch1, pb1.AfterAll({}));
    pb1.Send(ch2, pb1.TupleIndex(rcv0, 0), pb1.TupleIndex(rcv0, 1));
    XLS_ASSERT_OK_AND_ASSIGN(proc1, pb1.Build());
  }

  XLS_ASSERT_OK(p->SetTop(proc0));
  EXPECT_THAT(RunBlockStitchingPass(p.get()), IsOkAndHolds(Pair(true, _)));
  EXPECT_THAT(p->blocks(), UnorderedElementsAre(m::Block("top_proc__1"),
                                                m::Block("top_proc"),
                                                m::Block(proc1->name())));
}

TEST_F(BlockStitchingPassTest, StitchBlockWithIdleOutput) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch0,
      p->CreateStreamingChannel("ch0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch1,
      p->CreateStreamingChannel("ch1", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/
                                FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * ch2,
      p->CreateStreamingChannel("ch2", ChannelOps::kSendOnly, u32));

  ProcBuilder pb0(absl::StrCat(TestName(), 0), p.get());
  BValue rcv0 = pb0.Receive(ch0, pb0.AfterAll({}));
  pb0.Send(ch1, pb0.TupleIndex(rcv0, 0), pb0.TupleIndex(rcv0, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc0, pb0.Build());

  ProcBuilder pb1(absl::StrCat(TestName(), 1), p.get());
  BValue rcv1 = pb1.Receive(ch1, pb1.AfterAll({}));
  pb1.Send(ch2, pb1.TupleIndex(rcv1, 0), pb1.TupleIndex(rcv1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, pb1.Build());
  XLS_ASSERT_OK(p->SetTop(proc0));

  EXPECT_THAT(RunBlockStitchingPass(
                  p.get(), /*top_name=*/"top_proc",
                  /*options=*/DefaultCodegenOptions().add_idle_output(true)),
              IsOkAndHolds(Pair(true, _)));
  EXPECT_THAT(p->blocks(), UnorderedElementsAre(m::Block("top_proc__1"),
                                                m::Block("top_proc"),
                                                m::Block(proc1->name())));

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("top_proc"));
  ASSERT_EQ(top_block->GetInstantiations().size(), 2);
  EXPECT_THAT(
      top_block->nodes(),
      Contains(m::OutputPort(
          "idle", m::And(m::InstantiationOutput(
                             "idle", top_block->GetInstantiations().at(0)),
                         m::InstantiationOutput(
                             "idle", top_block->GetInstantiations().at(1))))));
}

std::string ValueMapToString(
    const absl::flat_hash_map<std::string, std::vector<uint64_t>>& map) {
  std::string output;
  for (const auto& [k, v] : map) {
    absl::StrAppendFormat(&output, "\t%s: [%s],\n", k, absl::StrJoin(v, ", "));
  }
  return output;
}

struct BlockEvaluationResults {
  absl::flat_hash_map<std::string, std::vector<uint64_t>> actual_outputs;
  InterpreterEvents interpreter_events;
};

// Matcher to check outputs from block evaluation.
struct BlockEvaluationOutputsEqMatcher {
 public:
  using is_gtest_matcher = void;

  explicit BlockEvaluationOutputsEqMatcher(
      absl::flat_hash_map<std::string, std::vector<uint64_t>> expected_outputs)
      : expected_outputs(std::move(expected_outputs)) {}

  bool MatchAndExplain(const BlockEvaluationResults& results,
                       ::testing::MatchResultListener* listener) const {
    if (!results.interpreter_events.assert_msgs.empty()) {
      *listener << absl::StreamFormat(
          "Unexpected assertion failures: %s.",
          absl::StrJoin(results.interpreter_events.assert_msgs, ", "));
      return false;
    }
    if (results.actual_outputs != expected_outputs) {
      *listener << absl::StrCat("\nActual outputs: {\n",
                                ValueMapToString(results.actual_outputs),
                                "\n}\n", " Expected outputs: {\n",
                                ValueMapToString(expected_outputs), "\n}");
      return false;
    }
    return true;
  }
  void DescribeTo(::std::ostream* os) const {
    *os << "outputs = " << ValueMapToString(expected_outputs);
  }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "outputs != " << ValueMapToString(expected_outputs);
  }

  absl::flat_hash_map<std::string, std::vector<uint64_t>> expected_outputs;
};

inline BlockEvaluationOutputsEqMatcher BlockOutputsEq(
    absl::flat_hash_map<std::string, std::vector<uint64_t>> expected_outputs) {
  return BlockEvaluationOutputsEqMatcher(std::move(expected_outputs));
}

template <typename T>
inline ::testing::Matcher<BlockEvaluationResults> BlockOutputsMatch(T matcher) {
  return ::testing::Field("actual_outputs",
                          &BlockEvaluationResults::actual_outputs, matcher);
}

// Evaluates a block with the given inputs and returns the outputs and
// interpreter events.
// Similar to EvalAndExpect, but only evaluates a block. To
// replicate EvalAndExpect, EXPECT_THAT() with BlockOutputsEq().
absl::StatusOr<BlockEvaluationResults> EvalBlock(
    Block* block, const CodegenPassUnit::MetadataMap& metadata_map,
    const absl::flat_hash_map<std::string, std::vector<int64_t>>& inputs,
    std::optional<int64_t> num_cycles = std::nullopt) {
  int64_t cycles = num_cycles.value_or(999) + 1;  // + 1 for the reset cycle
  BlockEvaluationResults evaluation_results;
  std::vector<ChannelSource> sources;
  std::vector<ChannelSink> sinks;
  absl::flat_hash_map<std::string, std::string_view> sink_names_by_data_name;

  std::vector<absl::flat_hash_map<std::string, Value>> fixed_values;
  fixed_values.reserve(cycles);
  constexpr int64_t kResetCycles = 1;
  for (int i = 0; i < cycles; ++i) {
    fixed_values.push_back(
        {{"rst", Value(UBits((i < kResetCycles) ? 1 : 0, 1))}});
  }

  for (const auto& [_block, metadata] : metadata_map) {
    for (const std::vector<StreamingInput>& metadata_inputs :
         metadata.streaming_io_and_pipeline.inputs) {
      for (const StreamingInput& input : metadata_inputs) {
        if (input.channel->supported_ops() == ChannelOps::kSendReceive) {
          continue;
        }
        ChannelSource source(input.port.value()->GetName(),
                             input.port_valid->GetName(),
                             input.port_ready->GetName(), 0.5, block,
                             ChannelSource::BehaviorDuringReset::kIgnoreReady);
        VLOG(3) << absl::StreamFormat(
            "Adding source for channel %s with ports %s %s %s",
            input.channel->name(), input.port.value()->GetName(),
            input.port_valid->GetName(), input.port_ready->GetName());
        std::vector<uint64_t> cast_data_sequence;
        cast_data_sequence.reserve(inputs.at(input.channel->name()).size());
        for (int64_t in : inputs.at(input.channel->name())) {
          cast_data_sequence.push_back(in);
        }
        XLS_RETURN_IF_ERROR(source.SetDataSequence(cast_data_sequence));
        sources.push_back(std::move(source));
      }
    }
    for (const SingleValueInput& metadata_input :
         metadata.streaming_io_and_pipeline.single_value_inputs) {
      if (metadata_input.channel->supported_ops() == ChannelOps::kSendReceive) {
        continue;
      }
      for (absl::flat_hash_map<std::string, Value>& cycle_values :
           fixed_values) {
        auto input_iter = inputs.find(metadata_input.port->GetName());
        if (input_iter == inputs.end()) {
          return absl::InvalidArgumentError(
              absl::StrFormat("No input provided for channel %s",
                              metadata_input.port->GetName()));
        }
        XLS_RET_CHECK_EQ(input_iter->second.size(), 1)
            << "Single value channels may only have a single input";
        XLS_RET_CHECK(metadata_input.port->GetType()->IsBits());
        cycle_values[metadata_input.port->GetName()] =
            Value(UBits(input_iter->second.front(),
                        metadata_input.port->GetType()->GetFlatBitCount()));
      }
    }

    for (const std::vector<StreamingOutput>& outputs :
         metadata.streaming_io_and_pipeline.outputs) {
      for (const StreamingOutput& output : outputs) {
        if (output.channel->supported_ops() == ChannelOps::kSendReceive) {
          continue;
        }
        sinks.push_back(ChannelSink(
            output.port.value()->GetName(), output.port_valid->GetName(),
            output.port_ready->GetName(), 0.5, block,
            ChannelSink::BehaviorDuringReset::kIgnoreValid));
        sink_names_by_data_name[sinks.back().data_name()] =
            output.channel->name();
      }
    }
  }

  InterpreterBlockEvaluator evaluator;
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<BlockContinuation> continuation,
                       evaluator.NewContinuation(block));
  ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);
  reset_proto.set_reset_data_path(false);

  XLS_ASSIGN_OR_RETURN(BlockIOResults results,
                       evaluator.EvaluateChannelizedSequentialBlock(
                           block, absl::MakeSpan(sources),
                           absl::MakeSpan(sinks), fixed_values, reset_proto));

  absl::flat_hash_map<std::string, std::vector<uint64_t>> actual_outputs;
  for (const ChannelSink& sink : sinks) {
    XLS_ASSIGN_OR_RETURN(
        actual_outputs[sink_names_by_data_name.at(sink.data_name())],
        sink.GetOutputSequenceAsUint64());
  }
  for (const auto& [_block, metadata] : metadata_map) {
    for (const SingleValueOutput& output :
         metadata.streaming_io_and_pipeline.single_value_outputs) {
      if (output.channel->supported_ops() == ChannelOps::kSendReceive) {
        continue;
      }
      std::vector<uint64_t>& channel_int_outputs =
          actual_outputs[output.port->name()];
      channel_int_outputs.reserve(results.outputs.size());
      for (const absl::flat_hash_map<std::string, Value>& value_map :
           results.outputs) {
        XLS_RET_CHECK(value_map.contains(output.port->name()));
        const Value& value = value_map.at(output.port->name());
        XLS_RET_CHECK(value.IsBits());
        XLS_ASSIGN_OR_RETURN(int64_t value_int, value.bits().ToUint64());
        channel_int_outputs.push_back(value_int);
      }
    }
  }

  return BlockEvaluationResults{
      .actual_outputs = std::move(actual_outputs),
      .interpreter_events = std::move(results.interpreter_events),
  };
}

// Tests from proc inlining.
// We port the test suite from proc inlining to check that block stitching
// supports the same features. Note that some changes need to be made to allow
// blocks to codegen without cycles or deadlocks, and pipelining can change the
// order of some outputs (especially when using non-blocking receives).
// TODO: google/xls#1508 - Update these tests to not reference proc inlining
// once proc inlining is removed.
absl::StatusOr<std::unique_ptr<SerialProcRuntime>> CreateInterpreter(
    Package* p,
    const absl::flat_hash_map<std::string, std::vector<int64_t>>& inputs) {
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<SerialProcRuntime> interpreter,
                       CreateInterpreterSerialProcRuntime(p));

  for (const auto& [ch_name, ch_inputs] : inputs) {
    XLS_ASSIGN_OR_RETURN(Channel * ch, p->GetChannel(ch_name));
    XLS_RET_CHECK(ch->type()->IsBits());

    std::vector<Value> input_values;
    for (int64_t in : ch_inputs) {
      input_values.push_back(Value(UBits(in, ch->type()->GetFlatBitCount())));
    }

    ChannelQueue& queue = interpreter->queue_manager().GetQueue(ch);
    if (ch->kind() == ChannelKind::kStreaming) {
      XLS_RETURN_IF_ERROR(
          queue.AttachGenerator(FixedValueGenerator(input_values)));
    } else {
      XLS_RET_CHECK(ch->kind() == ChannelKind::kSingleValue);
      XLS_RET_CHECK_EQ(input_values.size(), 1)
          << "Single value channels may only have a single input";
      XLS_RETURN_IF_ERROR(queue.Write(input_values.front()));
    }
  }
  return std::move(interpreter);
}

// Evaluate the proc with the given inputs and expect the given
// outputs. Inputs and outputs are given as a map from channel name to
// sequence of values. `expected_ticks` if given is the expected number of
// ticks to run to generate the expected outputs. If the expected number of
// outputs is not generated by this number of ticks an error is raised.
using ProcInliningPassTest = IrTestBase;

absl::StatusOr<std::unique_ptr<SerialProcRuntime>> EvalAndExpect(
    Package* p,
    const absl::flat_hash_map<std::string, std::vector<int64_t>>& inputs,
    const absl::flat_hash_map<std::string, std::vector<int64_t>>&
        expected_outputs,
    std::optional<int64_t> expected_ticks = std::nullopt,
    xabsl::SourceLocation loc = xabsl::SourceLocation::current()) {
  testing::ScopedTrace trace(loc.file_name(), loc.line(),
                             "EvalAndExpect failed");
  XLS_ASSIGN_OR_RETURN(std::unique_ptr<SerialProcRuntime> interpreter,
                       CreateInterpreter(p, inputs));

  std::vector<Channel*> output_channels;
  absl::flat_hash_map<Channel*, int64_t> expected_output_count;
  for (const auto& [ch_name, expected_values] : expected_outputs) {
    XLS_ASSIGN_OR_RETURN(Channel * ch, p->GetChannel(ch_name));
    XLS_RET_CHECK(ch->type()->IsBits());
    output_channels.push_back(ch);
    expected_output_count[ch] = expected_values.size();
  }
  // Sort output channels the output expectations are done in a deterministic
  // order.
  absl::c_sort(output_channels, Channel::NameLessThan);

  const int64_t kMaxTicks = 1000;
  XLS_ASSIGN_OR_RETURN(int64_t ticks, interpreter->TickUntilOutput(
                                          expected_output_count, kMaxTicks));
  if (expected_ticks.has_value()) {
    XLS_RET_CHECK_EQ(expected_ticks.value(), ticks);
  }

  for (Channel* ch : output_channels) {
    std::vector<int64_t> outputs;
    ChannelQueue& queue = interpreter->queue_manager().GetQueue(ch);
    while (outputs.size() < expected_outputs.at(ch->name()).size()) {
      Value output = queue.Read().value();
      outputs.push_back(output.bits().ToUint64().value());
    }
    EXPECT_THAT(outputs, ElementsAreArray(expected_outputs.at(ch->name())))
        << "Output of channel: " << ch->name();
  }

  return interpreter;
}

// Make a proc which receives data on channel `in` and immediate sends back
// the data on channel `out`.
absl::StatusOr<Proc*> MakeLoopbackProc(std::string_view name, Channel* in,
                                       Channel* out, Package* p) {
  XLS_RET_CHECK(in->type() == out->type());
  ProcBuilder b(name, p);
  BValue rcv = b.Receive(in, b.AfterAll({}));
  b.Send(out, b.TupleIndex(rcv, 0), b.TupleIndex(rcv, 1));
  return b.Build();
}

// Make a proc which receives data on channel `in` and sends back the data on
// channel `out` after `delay` ticks.
absl::StatusOr<Proc*> MakeDelayedLoopbackProc(std::string_view name,
                                              int64_t delay, Channel* in,
                                              Channel* out, Package* p) {
  XLS_RET_CHECK(in->type() == out->type());
  ProcBuilder b(name, p);
  BValue cnt = b.StateElement("cnt", Value(UBits(0, 32)));
  BValue data = b.StateElement("cnt", Value(UBits(0, 32)));

  BValue cnt_eq_0 = b.Eq(cnt, b.Literal(UBits(0, 32)));
  BValue cnt_last = b.Eq(cnt, b.Literal(UBits(delay - 1, 32)));

  BValue rcv = b.ReceiveIf(in, b.AfterAll({}), cnt_eq_0);

  b.SendIf(out, b.TupleIndex(rcv, 0), cnt_last, data);

  BValue next_cnt = b.Select(cnt_last, b.Literal(UBits(0, 32)),
                             b.Add(cnt, b.Literal(UBits(1, 32))));
  BValue next_data = b.Select(cnt_eq_0, b.TupleIndex(rcv, 1), data);
  b.Next(cnt, next_cnt);
  b.Next(data, next_data);

  return b.Build();
}

// Make a proc which receives data on channel `in` and sends back twice the
// value of data on channel `out`.
absl::StatusOr<Proc*> MakeDoublerProc(std::string_view name, Channel* in,
                                      Channel* out, Package* p) {
  XLS_RET_CHECK(in->type() == out->type());
  ProcBuilder b(name, p);
  BValue rcv = b.Receive(in, b.AfterAll({}));
  BValue data = b.TupleIndex(rcv, 1);
  b.Send(out, b.TupleIndex(rcv, 0), b.Add(data, data));
  return b.Build();
}

// Make a proc which receives data on channel `a_in` and sends the data on
// `a_out`, then receives data on channel `b_in` and sends that data on
// `b_out`.
absl::StatusOr<Proc*> MakePassThroughProc(std::string_view name, Channel* a_in,
                                          Channel* a_out, Channel* b_in,
                                          Channel* b_out, Package* p) {
  XLS_RET_CHECK(a_in->type() == a_out->type());
  XLS_RET_CHECK(b_in->type() == b_out->type());

  ProcBuilder b(name, p);
  BValue rcv_a = b.Receive(a_in, b.AfterAll({}));
  BValue send_a = b.Send(a_out, b.TupleIndex(rcv_a, 0), b.TupleIndex(rcv_a, 1));
  BValue rcv_b = b.Receive(b_in, send_a);
  b.Send(b_out, b.TupleIndex(rcv_b, 0), b.TupleIndex(rcv_b, 1));
  return b.Build();
}

// Make a proc which loops `iteration` times. It receives on the first
// iteration. The state accumulates the received data with the iteration
// count. The accumulated value is sent on the last iteration.
//
//   i = 0
//   accum = 0
//   while(true):
//    if i == 0:
//      x = rcv(in)
//    else:
//      x = accum + i
//    if i == ITERATIONS:
//      i = 0
//      accum = 0
//      send(out, x)
//    else:
//      i = i + 1
//      accum = x
absl::StatusOr<Proc*> MakeLoopingAccumulatorProc(std::string_view name,
                                                 Channel* input_ch,
                                                 Channel* output_ch,
                                                 int64_t iterations,
                                                 Package* p) {
  XLS_RET_CHECK(input_ch->type() == output_ch->type());
  XLS_RET_CHECK(input_ch->type()->IsBits());
  int64_t bit_count = input_ch->type()->AsBitsOrDie()->bit_count();

  SourceInfo loc;

  ProcBuilder b(name, p);
  BValue i = b.StateElement("i", ZeroOfType(input_ch->type()));
  BValue accum = b.StateElement("accum", ZeroOfType(input_ch->type()));

  BValue zero = b.Literal(UBits(0, bit_count), loc, "zero");
  BValue one = b.Literal(UBits(1, bit_count), loc, "one");

  BValue is_first_iteration = b.Eq(i, zero, loc, "is_first_iteration");
  BValue is_last_iteration = b.Eq(
      i, b.Literal(UBits(iterations - 1, bit_count)), loc, "is_last_iteration");

  BValue rcv = b.ReceiveIf(input_ch, b.AfterAll({}), is_first_iteration);
  BValue rcv_token = b.TupleIndex(rcv, 0);
  BValue data = b.TupleIndex(rcv, 1, loc, "data");

  BValue next_i =
      b.Select(is_last_iteration, zero, b.Add(i, one), loc, "next_i");
  BValue updated_accum =
      b.Select(is_first_iteration, data, b.Add(accum, i), loc, "updated_accum");
  BValue next_accum =
      b.Select(is_last_iteration, zero, updated_accum, loc, "next_accum");

  b.SendIf(output_ch, rcv_token, is_last_iteration, updated_accum);

  b.Next(i, next_i);
  b.Next(accum, next_accum);

  return b.Build();
}

// Make a proc which receives data values `x` and `y` and sends the sum and
// difference.
absl::StatusOr<Proc*> MakeSumAndDifferenceProc(std::string_view name,
                                               Channel* x_in, Channel* y_in,
                                               Channel* x_plus_y_out,
                                               Channel* x_minus_y_out,
                                               Package* p) {
  XLS_RET_CHECK(x_in->type() == y_in->type());
  XLS_RET_CHECK(x_plus_y_out->type() == x_minus_y_out->type());

  ProcBuilder b(name, p);
  BValue x_rcv = b.Receive(x_in, b.AfterAll({}));
  BValue y_rcv = b.Receive(y_in, b.TupleIndex(x_rcv, 0));
  BValue x = b.TupleIndex(x_rcv, 1);
  BValue y = b.TupleIndex(y_rcv, 1);

  BValue send_x_plus_y =
      b.Send(x_plus_y_out, b.TupleIndex(y_rcv, 0), b.Add(x, y));
  b.Send(x_minus_y_out, send_x_plus_y, b.Subtract(x, y));

  return b.Build();
}

// Make a proc which receives data values `x` and `y` and sends out the sum.
absl::StatusOr<Proc*> MakeSumProc(std::string_view name, Channel* x_in,
                                  Channel* y_in, Channel* out, Package* p) {
  XLS_RET_CHECK(x_in->type() == y_in->type());
  XLS_RET_CHECK(x_in->type() == out->type());

  ProcBuilder b(name, p);
  BValue x_rcv = b.Receive(x_in, b.Literal(Value::Token()));
  BValue y_rcv = b.Receive(y_in, b.TupleIndex(x_rcv, 0));
  BValue x = b.TupleIndex(x_rcv, 1);
  BValue y = b.TupleIndex(y_rcv, 1);
  b.Send(out, b.TupleIndex(y_rcv, 0), b.Add(x, y));

  return b.Build();
}

// Make a proc which receives a tuple of data and loops twice accumulating the
// element values before sending out the accumulation:
//
// u1: cnt = 0
// x_accum = 0
// y_accum = 0
// while (true):
//   (x, y) = rcv(in)
//   x_accum += x
//   y_accum += y
//   send_if(out, cnt, (x_accum, y_accum))
//   cnt = !cnt
absl::StatusOr<Proc*> MakeTupleAccumulator(std::string_view name, Channel* in,
                                           Channel* out, Package* p) {
  XLS_RET_CHECK(in->type()->IsTuple());
  int64_t x_bit_count =
      in->type()->AsTupleOrDie()->element_type(0)->AsBitsOrDie()->bit_count();
  int64_t y_bit_count =
      in->type()->AsTupleOrDie()->element_type(1)->AsBitsOrDie()->bit_count();
  ProcBuilder b(name, p);

  BValue cnt = b.StateElement("cnt", Value(UBits(0, 1)));
  BValue x_accum = b.StateElement("x_accum", Value(UBits(0, x_bit_count)));
  BValue y_accum = b.StateElement("y_accum", Value(UBits(0, y_bit_count)));

  BValue rcv_x_y = b.Receive(in, b.Literal(Value::Token()));
  BValue rcv_x_y_data = b.TupleIndex(rcv_x_y, 1);
  BValue x = b.TupleIndex(rcv_x_y_data, 0);
  BValue y = b.TupleIndex(rcv_x_y_data, 1);

  BValue x_plus_x_accum = b.Add(x, x_accum);
  BValue y_plus_y_accum = b.Add(y, y_accum);

  b.SendIf(out, b.TupleIndex(rcv_x_y, 0), cnt,
           b.Tuple({x_plus_x_accum, y_plus_y_accum}));

  b.Next(cnt, b.Not(cnt));
  b.Next(x_accum, x_plus_x_accum);
  b.Next(y_accum, y_plus_y_accum);

  return b.Build();
}

// Simple proc that arbitrates between inputs, with lower-index inputs being
// higher priority than higher-index inputs.
absl::StatusOr<Proc*> MakeArbiterProc(std::string_view name,
                                      absl::Span<Channel* const> inputs,
                                      Channel* out, Package* p) {
  XLS_RET_CHECK(!inputs.empty());
  XLS_RETURN_IF_ERROR(std::accumulate(
      inputs.begin(), inputs.end(), absl::OkStatus(),
      [inputs](const absl::Status& status_in,
               Channel* channel) -> absl::Status {
        XLS_RETURN_IF_ERROR(status_in);
        if (!inputs[0]->type()->IsEqualTo(channel->type())) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Inputs must be same type, got %s != %s for channel %s.",
              inputs[0]->type()->ToString(), channel->type()->ToString(),
              channel->name()));
        }
        if (!channel->CanReceive()) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Must be able to receive on channel %s.", channel->name()));
        }
        return absl::OkStatus();
      }));

  ProcBuilder b(name, p);

  BValue token = b.Literal(Value::Token());
  std::vector<BValue> recv_data;
  recv_data.reserve(inputs.size());
  std::vector<BValue> recv_data_valids;
  recv_data_valids.reserve(inputs.size());
  for (Channel* in : inputs) {
    BValue recv_pred;
    if (recv_data_valids.empty()) {
      recv_pred = b.Literal(UBits(1, /*bit_count=*/1));
    } else {
      recv_pred = b.Not(b.Or(recv_data_valids));
    }
    BValue recv = b.ReceiveIfNonBlocking(in, token, recv_pred);
    token = b.TupleIndex(recv, 0);
    recv_data.push_back(b.TupleIndex(recv, 1));
    BValue recv_data_valid = b.And({recv_pred, b.TupleIndex(recv, 2)});
    recv_data_valids.push_back(recv_data_valid);
  }
  BValue send_pred = b.Or(recv_data_valids);
  // First element should be LSB, last should be MSB.
  absl::c_reverse(recv_data_valids);
  BValue send_data = b.OneHotSelect(b.Concat(recv_data_valids), recv_data);
  token = b.SendIf(out, token, send_pred, send_data);

  return b.Build();
}

TEST_F(ProcInliningPassTest, SingleProc) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder b(TestName(), p.get());
  BValue rcv = b.Receive(ch_in, b.Literal(Value::Token()));
  b.Send(ch_out, b.TupleIndex(rcv, 0), b.TupleIndex(rcv, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, b.Build());

  ASSERT_THAT(
      RunBlockStitchingPass(
          p.get(), proc->name(),
          /*options=*/DefaultCodegenOptions(),
          /*generate_signature=*/false),  // Don't generate signature, otherwise
                                          // we'll always get changed=true.
      IsOkAndHolds(Pair(false, _)));
}

TEST_F(ProcInliningPassTest, NestedProcs) {
  // Nested procs where the inner proc does a trivial arithmetic operation.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  XLS_ASSERT_OK(MakeDoublerProc("B", a_to_b, b_to_a, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {2, 4, 6}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);  // 2 leaf + 1 container
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}},
                        /*num_cycles=*/10),
              IsOkAndHolds(BlockOutputsEq({{"out", {2, 4, 6}}})));
}

TEST_F(ProcInliningPassTest, NestedProcsFifoDepth1) {
  // Nested procs where the inner proc does a trivial arithmetic operation.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  XLS_ASSERT_OK(MakeDoublerProc("B", a_to_b, b_to_a, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {2, 4, 6}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), p->procs().size() + 1);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}},
                        /*num_cycles=*/12),
              IsOkAndHolds(BlockOutputsEq({{"out", {2, 4, 6}}})));
}

TEST_F(ProcInliningPassTest, NestedProcsWithUnspecifiedFifoDepth) {
  // Nested procs where the inner proc does a trivial arithmetic operation.
  // Don't use CreatePackage() because that returns a
  // VerifiedPackage and this purposely will fail to verify.
  Package p(TestName());
  Type* u32 = p.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p.CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p.CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                               /*initial_values=*/{},
                               /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK(
      MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, &p).status());
  XLS_ASSERT_OK(MakeDoublerProc("B", a_to_b, b_to_a, &p).status());

  EXPECT_EQ(p.procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(&p, {{"in", {1, 2, 3}}}, {{"out", {2, 4, 6}}}).status());

  EXPECT_THAT(RunBlockStitchingPass(&p, /*top_name=*/"A"),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Channel a_to_b has no fifo config.")));
}

TEST_F(ProcInliningPassTest, NestedProcsWithNonzeroFifoDepth) {
  // Nested procs where the inner proc does a trivial arithmetic operation.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(42)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  XLS_ASSERT_OK(MakeDoublerProc("B", a_to_b, b_to_a, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {2, 4, 6}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {2, 4, 6}}})));
}

TEST_F(ProcInliningPassTest, NestedProcsWithSingleValue) {
  // Nested procs where the inner proc does a trivial arithmetic operation.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateSingleValueChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateSingleValueChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateSingleValueChannel("a_to_b", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateSingleValueChannel("b_to_a", ChannelOps::kSendReceive, u32));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  XLS_ASSERT_OK(MakeDoublerProc("B", a_to_b, b_to_a, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1}}}, {{"out", {2}}}).status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"in", {1}}}),
      // Single value outputs show up for every cycle.
      IsOkAndHolds(BlockOutputsMatch(ElementsAre(Pair("out", Each(Eq(2)))))));
}

TEST_F(ProcInliningPassTest, NestedProcsWithConditionalSingleValueSend) {
  // Nested procs where the outer proc conditionally sends on a single value
  // channel to the inner proc.
  std::unique_ptr<Package> p = std::make_unique<Package>(TestName());
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateSingleValueChannel("a_to_b", ChannelOps::kSendReceive, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  ProcBuilder ab("A", p.get());
  BValue rcv_in = ab.Receive(ch_in, ab.Literal(Value::Token()));
  BValue data_in = ab.TupleIndex(rcv_in, 1);
  BValue pred_data_is_odd = ab.BitSlice(data_in, /*start=*/0, /*width=*/1);
  BValue send_to_b = ab.SendIf(a_to_b, ab.TupleIndex(rcv_in, 0),
                               /*pred=*/pred_data_is_odd, /*data=*/data_in);
  BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);
  ab.Send(ch_out, ab.TupleIndex(rcv_from_b, 0), ab.TupleIndex(rcv_from_b, 1));
  XLS_ASSERT_OK(ab.Build());

  XLS_ASSERT_OK(MakeDoublerProc("B", a_to_b, b_to_a, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3, 4, 5}}},
                              {{"out", {2, 2, 6, 6, 10}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3, 4, 5}}}),
              IsOkAndHolds(BlockOutputsMatch(ElementsAre(
                  // send_if is not specially codegen'd for single value
                  // channels, nor is the value retained. It's just a direct
                  // connection to the sometimes-invalid streaming channel's
                  // data, so the value is garbage.
                  Pair("out", _)))));
}

TEST_F(ProcInliningPassTest, NestedProcPassThrough) {
  // Nested procs where the inner proc passes through the value.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  XLS_ASSERT_OK(MakeLoopbackProc("B", a_to_b, b_to_a, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {123, 22, 42}}}, {{"out", {123, 22, 42}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {123, 22, 42}}},
                        /*num_cycles=*/12),
              IsOkAndHolds(BlockOutputsEq({{"out", {123, 22, 42}}})));
}

TEST_F(ProcInliningPassTest, NestedProcDelayedPassThrough) {
  // Nested procs where the inner proc passes through the value after a delay.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  ProcBuilder ab("A", p.get());
  BValue rcv_in = ab.Receive(ch_in, ab.Literal(Value::Token()));
  BValue in_data = ab.TupleIndex(rcv_in, 1);
  BValue send_to_b = ab.Send(a_to_b, ab.TupleIndex(rcv_in, 0), in_data);
  BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);
  ab.Send(ch_out, ab.TupleIndex(rcv_from_b, 0),
          ab.Add(in_data, ab.TupleIndex(rcv_from_b, 1)));
  XLS_ASSERT_OK(ab.Build());

  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("B", /*delay=*/3, a_to_b, b_to_a, p.get())
          .status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {123, 22, 42}}}, {{"out", {246, 44, 84}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {123, 22, 42}}},
                        /*num_cycles=*/12),
              IsOkAndHolds(BlockOutputsEq({{"out", {246, 44, 84}}})));
}

TEST_F(ProcInliningPassTest, InputPlusDelayedInput) {
  // Proc where a value is added to a delayed version of itself. The value is
  // delayed by sending it through another proc. This tests the saving of inputs
  // from external channels.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("B", /*delay=*/42, a_to_b, b_to_a, p.get())
          .status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {123, 22, 42}}}, {{"out", {123, 22, 42}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {123, 22, 42}}},
                        /*num_cycles=*/168),
              IsOkAndHolds(BlockOutputsEq({{"out", {123, 22, 42}}})));
}

TEST_F(ProcInliningPassTest, NestedProcsTrivialInnerLoop) {
  // Nested procs where the inner proc loops more than once for each received
  // input.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());

  {
    // Inner proc performs the following:
    //
    //   st = 1
    //   while(true):
    //    if(st): rcv()
    //    if(!st): send(42)
    //    st = !st
    ProcBuilder bb("B", p.get());
    BValue st = bb.StateElement("st", Value(UBits(1, 1)));
    BValue rcv_from_a = bb.ReceiveIf(a_to_b, bb.Literal(Value::Token()), st);
    bb.SendIf(b_to_a, bb.TupleIndex(rcv_from_a, 0), bb.Not(st),
              bb.Literal(UBits(42, 32)));
    bb.Next(st, bb.Not(st));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {42, 42, 42}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}},
                        /*num_cycles=*/12),
              IsOkAndHolds(BlockOutputsEq({{"out", {42, 42, 42}}})));
}

TEST_F(ProcInliningPassTest, NestedProcsIota) {
  // Output only nested procs where the inner proc just implements iota
  // starting at 42.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_from_b = ab.Receive(b_to_a, ab.Literal(Value::Token()));
    ab.Send(ch_out, ab.TupleIndex(rcv_from_b, 0), ab.TupleIndex(rcv_from_b, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  {
    ProcBuilder bb("B", p.get());
    BValue st = bb.StateElement("st", Value(UBits(42, 32)));
    bb.Send(b_to_a, bb.Literal(Value::Token()), bb.GetStateParam(0));
    bb.Next(st, bb.Add(st, bb.Literal(UBits(1, 32))));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {}, {{"out", {42, 43, 44}}}).status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {},
                        /*num_cycles=*/3),
              IsOkAndHolds(BlockOutputsEq({{"out", {42, 43, 44}}})));
}

TEST_F(ProcInliningPassTest, NestedProcsOddIota) {
  // Output only nested procs where the inner proc just implements iota
  // starting at 42 but only sends the odd values.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_from_b = ab.Receive(b_to_a, ab.Literal(Value::Token()));
    ab.Send(ch_out, ab.TupleIndex(rcv_from_b, 0), ab.TupleIndex(rcv_from_b, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  {
    ProcBuilder bb("B", p.get());
    BValue st = bb.StateElement("st", Value(UBits(42, 32)));
    bb.SendIf(b_to_a, bb.Literal(Value::Token()),
              bb.BitSlice(bb.GetStateParam(0), /*start=*/0, /*width=*/1),
              bb.GetStateParam(0));
    bb.Next(st, bb.Add(st, bb.Literal(UBits(1, 32))));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {}, {{"out", {43, 45, 47, 49}}}).status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {},
                        /*num_cycles=*/12),
              IsOkAndHolds(BlockOutputsEq({{"out", {43, 45, 47, 49}}})));
}

TEST_F(ProcInliningPassTest, SynchronizedNestedProcs) {
  // Nested procs where every other iteration each proc does nothing (send and
  // receive predicates off). The procs are synchronized in that they are active
  // on the same ticks.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    // Outer proc performs the following:
    //
    //   st = 0
    //   while(true):
    //    if(st):
    //      x = rcv(in)
    //      send(a_to_b, x)
    //      y = rcv(b_t_a)
    //      send(out, y)
    //    st = !st
    ProcBuilder ab("A", p.get());
    BValue st = ab.StateElement("st", Value(UBits(0, 1)));
    BValue rcv_in = ab.ReceiveIf(ch_in, ab.Literal(Value::Token()), st);
    BValue send_to_b = ab.SendIf(a_to_b, ab.TupleIndex(rcv_in, 0), st,
                                 ab.TupleIndex(rcv_in, 1));
    BValue rcv_from_b = ab.ReceiveIf(b_to_a, send_to_b, ab.GetStateParam(0));
    ab.SendIf(ch_out, ab.TupleIndex(rcv_from_b, 0), ab.GetStateParam(0),
              ab.TupleIndex(rcv_from_b, 1));
    ab.Next(st, ab.Not(st));
    XLS_ASSERT_OK(ab.Build());
  }

  {
    // Inner proc performs the following:
    //
    //   st = 0
    //   while(true):
    //    if(st):
    //       x = rcv(a_to_b)
    //       send(b_to_a, x + 42)
    //    st = !st
    ProcBuilder bb("B", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 1)));
    BValue rcv_from_a = bb.ReceiveIf(a_to_b, bb.Literal(Value::Token()), st);
    bb.SendIf(b_to_a, bb.TupleIndex(rcv_from_a, 0), st,
              bb.Add(bb.TupleIndex(rcv_from_a, 1), bb.Literal(UBits(42, 32))));
    bb.Next(st, bb.Not(st));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {43, 44, 45}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}},
                        /*num_cycles=*/12),
              IsOkAndHolds(BlockOutputsEq({{"out", {43, 44, 45}}})));
}

TEST_F(ProcInliningPassTest, NestedProcsNontrivialInnerLoop) {
  // Nested procs where the inner proc loops more than once for each received
  // input and does something interesting.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  {
    // Inner proc has a counter which loops from 0, 1, 2, 3. It essentially
    // performs the following:
    //
    //   x = rcv()
    //   for i in 0..3:
    //     x += i
    //   snd(x)
    ProcBuilder bb("B", p.get());
    BValue cnt = bb.StateElement("cnt", Value(UBits(0, 2)));
    BValue accum = bb.StateElement("accum", Value(UBits(0, 32)));

    BValue cnt_eq_0 = bb.Eq(cnt, bb.Literal(UBits(0, 2)));
    BValue cnt_eq_3 = bb.Eq(cnt, bb.Literal(UBits(3, 2)));

    BValue rcv_from_a =
        bb.ReceiveIf(a_to_b, bb.Literal(Value::Token()), cnt_eq_0);

    BValue data = bb.Select(cnt_eq_0, bb.TupleIndex(rcv_from_a, 1), accum);
    BValue data_plus_cnt = bb.Add(data, bb.ZeroExtend(cnt, 32));

    bb.SendIf(b_to_a, bb.TupleIndex(rcv_from_a, 0), cnt_eq_3, data_plus_cnt);

    bb.Next(cnt, bb.Add(cnt, bb.Literal(UBits(1, 2))));
    bb.Next(accum,
            bb.Select(cnt_eq_3, bb.Literal(UBits(0, 32)), data_plus_cnt));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {7, 8, 9}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {7, 8, 9}}})));
}

TEST_F(ProcInliningPassTest, DoubleNestedProcsPassThrough) {
  // Nested procs where the inner proc passes through the value.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_c,
      p->CreateStreamingChannel("b_to_c", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c_to_b,
      p->CreateStreamingChannel("c_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  XLS_ASSERT_OK(
      MakePassThroughProc("B", a_to_b, b_to_c, c_to_b, b_to_a, p.get())
          .status());
  XLS_ASSERT_OK(MakeLoopbackProc("C", b_to_c, c_to_b, p.get()).status());

  EXPECT_EQ(p->procs().size(), 3);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {123, 22, 42}}}, {{"out", {123, 22, 42}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {123, 22, 42}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {123, 22, 42}}})));
}

TEST_F(ProcInliningPassTest, SequentialNestedProcsPassThrough) {
  // Sequential procs where each inner proc passes through the value.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_c,
      p->CreateStreamingChannel("a_to_c", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c_to_a,
      p->CreateStreamingChannel("c_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_in = ab.Receive(ch_in, ab.Literal(Value::Token()));

    BValue send_to_b =
        ab.Send(a_to_b, ab.TupleIndex(rcv_in, 0), ab.TupleIndex(rcv_in, 1));
    BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);

    BValue send_to_c = ab.Send(a_to_c, ab.TupleIndex(rcv_from_b, 0),
                               ab.TupleIndex(rcv_from_b, 1));
    BValue rcv_from_c = ab.Receive(c_to_a, send_to_c);

    ab.Send(ch_out, ab.TupleIndex(rcv_from_c, 0), ab.TupleIndex(rcv_from_c, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("B", /*delay=*/3, a_to_b, b_to_a, p.get())
          .status());
  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("C", /*delay=*/2, a_to_c, c_to_a, p.get())
          .status());

  EXPECT_EQ(p->procs().size(), 3);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {123, 22, 42}}}, {{"out", {123, 22, 42}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {123, 22, 42}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {123, 22, 42}}})));
}

TEST_F(ProcInliningPassTest, SequentialNestedLoopingProcsWithState) {
  // Sequential procs where each inner proc loops several times.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_c,
      p->CreateStreamingChannel("a_to_c", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c_to_a,
      p->CreateStreamingChannel("c_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_d,
      p->CreateStreamingChannel("a_to_d", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * d_to_a,
      p->CreateStreamingChannel("d_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_in = ab.Receive(ch_in, ab.Literal(Value::Token()));

    BValue send_to_b =
        ab.Send(a_to_b, ab.TupleIndex(rcv_in, 0), ab.TupleIndex(rcv_in, 1));
    BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);

    BValue send_to_c = ab.Send(a_to_c, ab.TupleIndex(rcv_from_b, 0),
                               ab.TupleIndex(rcv_from_b, 1));
    BValue rcv_from_c = ab.Receive(c_to_a, send_to_c);

    BValue send_to_d = ab.Send(a_to_d, ab.TupleIndex(rcv_from_c, 0),
                               ab.TupleIndex(rcv_from_c, 1));
    BValue rcv_from_d = ab.Receive(d_to_a, send_to_d);

    ab.Send(ch_out, ab.TupleIndex(rcv_from_d, 0), ab.TupleIndex(rcv_from_d, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(
      MakeLoopingAccumulatorProc("B", a_to_b, b_to_a, /*iterations=*/3, p.get())
          .status());
  XLS_ASSERT_OK(
      MakeLoopingAccumulatorProc("C", a_to_c, c_to_a, /*iterations=*/1, p.get())
          .status());
  XLS_ASSERT_OK(
      MakeLoopingAccumulatorProc("D", a_to_d, d_to_a, /*iterations=*/5, p.get())
          .status());

  EXPECT_EQ(p->procs().size(), 4);
  // Result should be:
  //   x + (0 + 1 + 2) + (0) + (0 + 1 + 2 + 3 + 4) = x + 13
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {0, 1, 2}}}, {{"out", {13, 14, 15}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 5);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {0, 1, 2}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {13, 14, 15}}})));
}

TEST_F(ProcInliningPassTest, SequentialNestedProcsWithLoops) {
  // Sequential procs where the inner procs loop and do interesting things.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_c,
      p->CreateStreamingChannel("a_to_c", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c_to_a,
      p->CreateStreamingChannel("c_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_in = ab.Receive(ch_in, ab.Literal(Value::Token()));

    BValue send_to_b =
        ab.Send(a_to_b, ab.TupleIndex(rcv_in, 0), ab.TupleIndex(rcv_in, 1));
    BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);

    BValue send_to_c = ab.Send(a_to_c, ab.TupleIndex(rcv_from_b, 0),
                               ab.TupleIndex(rcv_from_b, 1));
    BValue rcv_from_c = ab.Receive(c_to_a, send_to_c);

    ab.Send(ch_out, ab.TupleIndex(rcv_from_c, 0), ab.TupleIndex(rcv_from_c, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("B", /*delay=*/4, a_to_b, b_to_a, p.get())
          .status());
  XLS_ASSERT_OK(MakeDoublerProc("C", a_to_c, c_to_a, p.get()).status());

  EXPECT_EQ(p->procs().size(), 3);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {123, 22, 42}}}, {{"out", {246, 44, 84}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {123, 22, 42}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {246, 44, 84}}})));
}

TEST_F(ProcInliningPassTest, DoubleNestedLoops) {
  // Nested procs where the nested procs loop. The innermost proc loops 4 times,
  // the middle proc loops 2 times.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_c,
      p->CreateStreamingChannel("b_to_c", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c_to_b,
      p->CreateStreamingChannel("c_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  {
    // Middle proc "B" accumulates the input and passes on the accumulation
    // value:
    //
    //  cnt = 1
    //  accum = 0
    //  while true:
    //    if cnt == 1:
    //      x = rcv(a_to_b)
    //      accum += x
    //      send(b_to_c, accum)
    //    else:
    //      z = rcv(c_to_b)
    //      send(b_to_a, z)
    //    cnt = !cnt
    ProcBuilder bb("B", p.get());
    BValue cnt = bb.StateElement("cnt", Value(UBits(1, 1)));
    BValue accum = bb.StateElement("accum", Value(UBits(0, 32)));
    BValue rcv_from_a = bb.ReceiveIf(a_to_b, bb.Literal(Value::Token()), cnt);
    BValue next_accum = bb.Add(accum, bb.TupleIndex(rcv_from_a, 1),
                               SourceInfo(), "B_accum_next");
    BValue send_to_c =
        bb.SendIf(b_to_c, bb.TupleIndex(rcv_from_a, 0), cnt, next_accum);

    BValue rcv_from_c = bb.ReceiveIf(c_to_b, send_to_c, bb.Not(cnt));
    bb.SendIf(b_to_a, bb.TupleIndex(rcv_from_c, 0), bb.Not(cnt),
              bb.TupleIndex(rcv_from_c, 1));
    bb.Next(cnt, bb.Not(cnt));
    bb.Next(accum, next_accum);
    XLS_ASSERT_OK(bb.Build());
  }

  {
    // Innermost proc "C" adds 15 to the received value over four iterations and
    // sends back the result:
    //
    // u4 cnt = 0
    // u32 accum = 0
    // while true:
    //   if cnt == 0:
    //     accum = rcv(b_to_c)
    //   else:
    //     accum += 5
    //   if cnt == 3
    //     send(c_to_b, accum)
    //   cnt += 1
    ProcBuilder cb("C", p.get());
    BValue cnt = cb.StateElement("cnt", Value(UBits(0, 2)));
    BValue accum = cb.StateElement("accum", Value(UBits(0, 32)));
    BValue cnt_eq_0 = cb.Eq(cnt, cb.Literal(UBits(0, 2)));
    BValue cnt_eq_3 = cb.Eq(cnt, cb.Literal(UBits(3, 2)));

    BValue rcv_from_b =
        cb.ReceiveIf(b_to_c, cb.Literal(Value::Token()), cnt_eq_0);

    BValue next_accum = cb.Select(
        cnt_eq_0, cb.TupleIndex(rcv_from_b, 1),
        cb.Add(accum, cb.Literal(UBits(5, 32)), SourceInfo(), "C_accum_next"));

    cb.SendIf(c_to_b, cb.TupleIndex(rcv_from_b, 0), cnt_eq_3, next_accum);

    BValue next_cnt = cb.Add(cnt, cb.Literal(UBits(1, 2)));

    cb.Next(cnt, next_cnt);
    cb.Next(accum, next_accum);
    XLS_ASSERT_OK(cb.Build());
  }

  // Output is sum of all inputs so far plus 15.
  EXPECT_EQ(p->procs().size(), 3);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 100, 100000}}},
                              {{"out", {16, 116, 100116}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 100, 100000}}},
                        /*num_cycles=*/16),
              IsOkAndHolds(BlockOutputsEq({{"out", {16, 116, 100116}}})));
}

TEST_F(ProcInliningPassTest, MultiIO) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_in,
      p->CreateStreamingChannel("x", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * y_in,
      p->CreateStreamingChannel("y", ChannelOps::kReceiveOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_plus_y_out,
      p->CreateStreamingChannel("x_plus_y_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_minus_y_out,
      p->CreateStreamingChannel("x_minus_y_out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_x,
      p->CreateStreamingChannel("pass_x", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_y,
      p->CreateStreamingChannel("pass_y", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_plus_y,
      p->CreateStreamingChannel("x_plus_y", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_minus_y,
      p->CreateStreamingChannel("x_minus_y", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_x = ab.Receive(x_in, ab.Literal(Value::Token()));
    BValue rcv_y = ab.Receive(y_in, ab.TupleIndex(rcv_x, 0));

    BValue send_x =
        ab.Send(pass_x, ab.TupleIndex(rcv_y, 0), ab.TupleIndex(rcv_x, 1));
    BValue send_y = ab.Send(pass_y, send_x, ab.TupleIndex(rcv_y, 1));

    BValue rcv_sum = ab.Receive(x_plus_y, send_y);
    BValue rcv_diff = ab.Receive(x_minus_y, ab.TupleIndex(rcv_sum, 0));

    BValue send_sum = ab.Send(x_plus_y_out, ab.TupleIndex(rcv_diff, 0),
                              ab.TupleIndex(rcv_sum, 1));
    ab.Send(x_minus_y_out, send_sum, ab.TupleIndex(rcv_diff, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(MakeSumAndDifferenceProc("B", pass_x, pass_y, x_plus_y,
                                         x_minus_y, p.get()));

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(
          p.get(), {{"x", {123, 22, 42}}, {"y", {10, 20, 30}}},
          {{"x_plus_y_out", {133, 42, 72}}, {"x_minus_y_out", {113, 2, 12}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata,
                        {{"x", {123, 22, 42}}, {"y", {10, 20, 30}}}),
              IsOkAndHolds(BlockOutputsEq({{"x_plus_y_out", {133, 42, 72}},
                                           {"x_minus_y_out", {113, 2, 12}}})));
}

TEST_F(ProcInliningPassTest, NonTopProcsWithExternalStreamingIO) {
  // The inlined proc has streaming IO (external channels).
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_in,
      p->CreateStreamingChannel("x", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * y_in,
      p->CreateStreamingChannel("y", ChannelOps::kReceiveOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_plus_y_out,
      p->CreateStreamingChannel("x_plus_y_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_minus_y_out,
      p->CreateStreamingChannel("x_minus_y_out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_x,
      p->CreateStreamingChannel("pass_x", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_plus_y,
      p->CreateStreamingChannel("pass_x_plus_y", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_x = ab.Receive(x_in, ab.Literal(Value::Token()));
    BValue send_x =
        ab.Send(pass_x, ab.TupleIndex(rcv_x, 0), ab.TupleIndex(rcv_x, 1));

    BValue rcv_sum = ab.Receive(x_plus_y, send_x);
    ab.Send(x_plus_y_out, ab.TupleIndex(rcv_sum, 0), ab.TupleIndex(rcv_sum, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  // Proc "B" will be inlined and has internal communication with "A" (pass_x
  // and pass_x_plus_y channels) as well as external IO (y and x_minus_y_out).
  XLS_ASSERT_OK(MakeSumAndDifferenceProc("B", pass_x, y_in, x_plus_y,
                                         x_minus_y_out, p.get()));

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(
          p.get(), {{"x", {123, 22, 42}}, {"y", {10, 20, 30}}},
          {{"x_plus_y_out", {133, 42, 72}}, {"x_minus_y_out", {113, 2, 12}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata,
                        {{"x", {123, 22, 42}}, {"y", {10, 20, 30}}},
                        /*num_cycles=*/15),
              IsOkAndHolds(BlockOutputsEq({{"x_plus_y_out", {133, 42, 72}},
                                           {"x_minus_y_out", {113, 2, 12}}})));
}

TEST_F(ProcInliningPassTest, NonTopProcsWithExternalSingleValueIO) {
  // The inlined proc has single-value input on an external channel.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_in,
      p->CreateStreamingChannel("x", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * y_in,
      p->CreateSingleValueChannel("y_sv", ChannelOps::kReceiveOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_plus_y_out,
      p->CreateStreamingChannel("x_plus_y_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_minus_y_out,
      p->CreateStreamingChannel("x_minus_y_out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_x,
      p->CreateStreamingChannel("pass_x", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_plus_y,
      p->CreateStreamingChannel("pass_x_plus_y", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_x = ab.Receive(x_in, ab.Literal(Value::Token()));
    BValue send_x =
        ab.Send(pass_x, ab.TupleIndex(rcv_x, 0), ab.TupleIndex(rcv_x, 1));

    BValue rcv_sum = ab.Receive(x_plus_y, send_x);
    ab.Send(x_plus_y_out, ab.TupleIndex(rcv_sum, 0), ab.TupleIndex(rcv_sum, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  // Proc "B" will be inlined and has internal communication with "A" (pass_x
  // and pass_x_plus_y channels) as well as external IO (y and x_minus_y_out).
  XLS_ASSERT_OK(MakeSumAndDifferenceProc("B", pass_x, y_in, x_plus_y,
                                         x_minus_y_out, p.get()));

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"x", {123, 22, 42}}, {"y_sv", {10}}},
                              {{"x_plus_y_out", {133, 32, 52}},
                               {"x_minus_y_out", {113, 12, 32}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata,
                        {{"x", {123, 22, 42}}, {"y_sv", {10}}},
                        /*num_cycles=*/15),
              IsOkAndHolds(BlockOutputsEq({{"x_plus_y_out", {133, 32, 52}},
                                           {"x_minus_y_out", {113, 12, 32}}})));
}

TEST_F(ProcInliningPassTest, SingleValueAndStreamingChannels) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_in,
      p->CreateStreamingChannel("x", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * sv_in,
      p->CreateSingleValueChannel("sv", ChannelOps::kReceiveOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_x,
      p->CreateStreamingChannel("pass_x", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_sv,
      p->CreateSingleValueChannel("pass_sv", ChannelOps::kSendReceive, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_sum,
      p->CreateStreamingChannel("pass_sum", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * sum_out,
      p->CreateStreamingChannel("sum", ChannelOps::kSendOnly, u32));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_x = ab.Receive(x_in, ab.Literal(Value::Token()));
    BValue rcv_sv = ab.Receive(sv_in, ab.TupleIndex(rcv_x, 0));

    BValue send_x =
        ab.Send(pass_x, ab.TupleIndex(rcv_sv, 0), ab.TupleIndex(rcv_x, 1));
    BValue send_sv = ab.Send(pass_sv, send_x, ab.TupleIndex(rcv_sv, 1));

    BValue rcv_sum = ab.Receive(pass_sum, send_sv);
    ab.Send(sum_out, ab.TupleIndex(rcv_sum, 0), ab.TupleIndex(rcv_sum, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(MakeSumProc("B", pass_x, pass_sv, pass_sum, p.get()));

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"x", {123, 22, 42}}, {"sv", {10}}},
                              {{"sum", {133, 32, 52}}})
                    .status());
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"x", {123, 22, 42}}, {"sv", {25}}},
                              {{"sum", {148, 47, 67}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"x", {123, 22, 42}}, {"sv", {10}}}),
      IsOkAndHolds(BlockOutputsEq({{"sum", {133, 32, 52}}})));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"x", {123, 22, 42}}, {"sv", {25}}}),
      IsOkAndHolds(BlockOutputsEq({{"sum", {148, 47, 67}}})));
}

TEST_F(ProcInliningPassTest, TriangleProcNetwork) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_c,
      p->CreateStreamingChannel("a_to_c", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c_to_a,
      p->CreateStreamingChannel("c_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_c,
      p->CreateStreamingChannel("b_to_c", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    ProcBuilder ab("A", p.get());
    BValue rcv_in = ab.Receive(ch_in, ab.Literal(Value::Token()));

    BValue send_to_b =
        ab.Send(a_to_b, ab.TupleIndex(rcv_in, 0), ab.TupleIndex(rcv_in, 1));
    BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);

    BValue send_to_c = ab.Send(a_to_c, ab.TupleIndex(rcv_from_b, 0),
                               ab.TupleIndex(rcv_from_b, 1));
    BValue rcv_from_c = ab.Receive(c_to_a, send_to_c);

    ab.Send(ch_out, ab.TupleIndex(rcv_from_c, 0), ab.TupleIndex(rcv_from_c, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  {
    ProcBuilder bb("B", p.get());
    BValue rcv_a = bb.Receive(a_to_b, bb.Literal(Value::Token()));
    BValue rcv_data = bb.TupleIndex(rcv_a, 1);

    BValue send_to_b = bb.Send(b_to_a, bb.TupleIndex(rcv_a, 0), rcv_data);
    bb.Send(b_to_c, send_to_b, bb.Shll(rcv_data, bb.Literal(UBits(1, 32))));
    XLS_ASSERT_OK(bb.Build());
  }

  {
    ProcBuilder cb("C", p.get());
    BValue rcv_a = cb.Receive(a_to_c, cb.Literal(Value::Token()));
    BValue rcv_a_data = cb.TupleIndex(rcv_a, 1);

    BValue rcv_b = cb.Receive(b_to_c, cb.TupleIndex(rcv_a, 0));
    BValue rcv_b_data = cb.TupleIndex(rcv_b, 1);

    cb.Send(c_to_a, cb.TupleIndex(rcv_b, 0), cb.Add(rcv_a_data, rcv_b_data));
    XLS_ASSERT_OK(cb.Build());
  }

  EXPECT_EQ(p->procs().size(), 3);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {123, 22, 42}}}, {{"out", {369, 66, 126}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {123, 22, 42}}},
                        /*num_cycles=*/30),
              IsOkAndHolds(BlockOutputsEq({{"out", {369, 66, 126}}})));
}

TEST_F(ProcInliningPassTest, DelayedReceiveWithDataLossFifoDepth0) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  {
    // Inner proc does nothing on the first tick (i.e., doesn't receive
    // data). This causes data loss because the channel is FIFO depth 0.
    //
    //   st = 0
    //   while(true):
    //    if(st):
    //       x = rcv(a_to_b)
    //       send(b_to_a, x + 42)
    //    st = 1
    ProcBuilder bb("B", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 1)));
    BValue rcv_from_a = bb.ReceiveIf(a_to_b, bb.Literal(Value::Token()), st);
    bb.SendIf(b_to_a, bb.TupleIndex(rcv_from_a, 0), st,
              bb.Add(bb.TupleIndex(rcv_from_a, 1), bb.Literal(UBits(42, 32))));
    bb.Next(st, bb.Literal(UBits(1, 1)));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {43, 44, 45}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_THAT(p->blocks().size(), 3);

  EXPECT_THAT(
      EvalBlock(p->GetBlock("A").value(), unit.metadata, {{"in", {1, 2, 3}}}),
      IsOkAndHolds(BlockOutputsEq({{"out", {43, 44, 45}}})));
}

TEST_F(ProcInliningPassTest, DelayedReceiveWithNoDataLossFifoDepth1Variant0) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  {
    // Inner proc does nothing on the first tick (i.e., doesn't receive
    // data). Because the FIFO depth is 1, data is stored for a cycle which
    // avoids data loss.
    //
    //   st = 0
    //   while(true):
    //    if(st):
    //       x = rcv(a_to_b)
    //       send(b_to_a, x + 42)
    //    st = 1
    ProcBuilder bb("B", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 1)));
    BValue rcv_from_a = bb.ReceiveIf(a_to_b, bb.Literal(Value::Token()), st);
    bb.SendIf(b_to_a, bb.TupleIndex(rcv_from_a, 0), st,
              bb.Add(bb.TupleIndex(rcv_from_a, 1), bb.Literal(UBits(42, 32))));
    bb.Next(st, bb.Literal(UBits(1, 1)));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {43, 44, 45}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);
  EXPECT_EQ(p->blocks().size(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {43, 44, 45}}})));
}

TEST_F(ProcInliningPassTest, DelayedReceiveWithNoDataLossFifoDepth1Variant1) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(5)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(5)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  {
    // Inner proc does not receive on the first tick, but does send. Because the
    // FIFO depth is 1 no data should be lost. On the second tick of the inner
    // proc, the existing channel state should be received and updated in the
    // same tick.
    //
    //   u32: st = 0
    //   while(true):
    //    if st >= 1:
    //       x = rcv(a_to_b)
    //    else:
    //       x = 0
    //    send(b_to_a, x + 42)
    //    st = st + 1
    ProcBuilder bb("B", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 32)));
    BValue rcv_from_a = bb.ReceiveIf(a_to_b, bb.Literal(Value::Token()),
                                     bb.UGe(st, bb.Literal(UBits(1, 32))));
    bb.Send(b_to_a, bb.TupleIndex(rcv_from_a, 0),
            bb.Add(bb.TupleIndex(rcv_from_a, 1), bb.Literal(UBits(42, 32))));
    bb.Next(st, bb.Add(st, bb.Literal(UBits(1, 32))));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {42, 43, 44}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {42, 43, 44}}})));
}

TEST_F(ProcInliningPassTest, DelayedReceiveWithDataLossFifoDepth1) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK(MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get())
                    .status());
  {
    // Inner proc does not receive on the first *two* ticks (i.e., doesn't
    // receive data). The FIFO depth is 1 but that is insufficient to prevent
    // data loss.
    //
    //   u32: st = 0
    //   while(true):
    //    if st >= 2:
    //       x = rcv(a_to_b)
    //    else:
    //       x = 0
    //    send(b_to_a, x + 42)
    //    st = st + 1
    ProcBuilder bb("B", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 32)));
    BValue rcv_from_a = bb.ReceiveIf(a_to_b, bb.Literal(Value::Token()),
                                     bb.UGe(st, bb.Literal(UBits(2, 32))));
    bb.Send(b_to_a, bb.TupleIndex(rcv_from_a, 0),
            bb.Add(bb.TupleIndex(rcv_from_a, 1), bb.Literal(UBits(42, 32))));
    bb.Next(st, bb.Add(st, bb.Literal(UBits(1, 32))));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {42, 42, 43}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  // For proc inlining, inlined channels could not backpressure and this would
  // be an assertion failure. With block stitching, the FIFO will backpressure
  // and you get the same behavior as the proc network.
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {42, 42, 43}}})));
}

TEST_F(ProcInliningPassTest, DataLoss) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(2)));

  // Outer proc "A" sends every tick to inner proc "B", but only receives from
  // "B" every other tick.
  //
  //   st = 1
  //   while(true):
  //    x = rcv(in)
  //    send(a_to_b, x)
  //    if st:
  //      y = rcv(a_to_b)
  //      send(out, y)
  //    st = !st
  ProcBuilder ab("A", p.get());
  BValue st = ab.StateElement("st", Value(UBits(1, 1)));
  BValue rcv_in = ab.Receive(ch_in, ab.Literal(Value::Token()));
  BValue in_data = ab.TupleIndex(rcv_in, 1);
  BValue send_to_b = ab.Send(a_to_b, ab.TupleIndex(rcv_in, 0), in_data);

  BValue rcv_from_b = ab.ReceiveIf(b_to_a, send_to_b, st);
  ab.SendIf(ch_out, ab.TupleIndex(rcv_from_b, 0), st,
            ab.TupleIndex(rcv_from_b, 1));
  ab.Next(st, ab.Not(st));
  XLS_ASSERT_OK(ab.Build());

  XLS_ASSERT_OK(MakeLoopbackProc("B", a_to_b, b_to_a, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3, 4, 5}}}, {{"out", {1, 2, 3}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3, 4, 5}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {1, 2, 3}}})));
}

TEST_F(ProcInliningPassTest, BlockingReceiveBlocksSendsForDepth0Fifos) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b0,
      p->CreateStreamingChannel("a_to_b0", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b1,
      p->CreateStreamingChannel("a_to_b1", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));

  {
    // Top proc doesn't send on a_to_b0 on the first cycle which stalls the
    // inner proc.
    //
    //   st = 0
    //   while(true):
    //    if(st):
    //       snd(a_to_b0)
    //    snd(a_to_b1)
    //    st = 1
    ProcBuilder ab("A", p.get());
    BValue st = ab.StateElement("st", Value(UBits(0, 1)));
    BValue rcv = ab.Receive(ch_in, ab.Literal(Value::Token()));
    BValue send0 =
        ab.SendIf(a_to_b0, ab.TupleIndex(rcv, 0), st, ab.Literal(UBits(0, 32)));
    ab.Send(a_to_b1, send0, ab.Literal(UBits(0, 32)));
    ab.Next(st, ab.Literal(UBits(1, 1)));
    XLS_ASSERT_OK(ab.Build().status());
  }

  {
    // Inner proc is stalled on the first receive which results in the second
    // receive dropping data.
    //
    //   while(true):
    //    rcv(a_to_b0)
    //    x = rcv(a_to_b1)
    //    snd(out, x)
    ProcBuilder bb("B", p.get());
    BValue rcv0 = bb.Receive(a_to_b0, bb.Literal(Value::Token()));
    BValue rcv1 = bb.Receive(a_to_b1, bb.TupleIndex(rcv0, 0));
    bb.Send(ch_out, bb.TupleIndex(rcv1, 0), bb.TupleIndex(rcv1, 1));

    XLS_ASSERT_OK(bb.Build());
  }

  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3}}},
                              {{"out", {}}})  // produces no outputs
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  // For proc inlining, inlined channels could not backpressure and this would
  // be an assertion failure. With block stitching, the depth-0 FIFO will
  // backpressure and you get the same behavior as the proc network.
  EXPECT_EQ(p->blocks().size(), 3);
  EXPECT_THAT(
      EvalBlock(p->GetBlock("A").value(), unit.metadata, {{"in", {1, 2, 3}}}),
      IsOkAndHolds(BlockOutputsEq({{"out", {}}})));
}

// In the original proc inlining test, these
// 'SingleValueChannelWithVariantElements' tests had a single value channel for
// `pass_inputs`. This was fine in proc inlining because the inter-proc channel
// became a wire within the inlined proc, and proc codegen ensured the input
// from `x` was valid when sending on `pass_inputs`. With multi-proc, the
// situation is different because the the two procs tick truly independently, so
// we need this channel to be streaming to synchronize the two procs.
//
// Note that the original test also relied on the behavior of single value
// channels holding their output. The tuple accumulator proc received twice for
// each send on `pass_inputs`. This results in deadlock on a streaming channel,
// so now we send the same data twice. In order for *that* to work, we need to
// set a channel strictness that is not kProvenMutuallyExclusive so we get an
// adapter. So... this test doesn't really look like the original test. Still,
// we end up getting the same outputs as the original test and `y` is still a
// single value channel.
TEST_F(ProcInliningPassTest, SingleValueChannelWithVariantElements1) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  Type* u64 = p->GetBitsType(64);
  Type* u32_u64 = p->GetTupleType({u32, u64});
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_in,
      p->CreateStreamingChannel("x", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * y_in,
      p->CreateSingleValueChannel("y", ChannelOps::kReceiveOnly, u64));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_inputs,
      p->CreateStreamingChannel("pass_inputs", ChannelOps::kSendReceive,
                                u32_u64, /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0),
                                /*flow_control=*/FlowControl::kReadyValid,
                                /*strictness=*/ChannelStrictness::kTotalOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_result,
      p->CreateStreamingChannel(
          "pass_result", ChannelOps::kSendReceive, u32_u64,
          /*initial_values=*/{}, /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * result0_out,
      p->CreateStreamingChannel("result0_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * result1_out,
      p->CreateStreamingChannel("result1_out", ChannelOps::kSendOnly, u64));

  {
    // Proc "A". Element 1 sent on pass_inputs is variant:
    //
    // while true:
    //   x = rcv(x_in)  // streaming
    //   y = rcv(y_in)  // single-value
    //   send(pass_inputs, (x, y))
    //   (result0, result1) = rcv(pass_result)
    //   send(result0_out, result0)
    //   send(result1_out, result1)
    ProcBuilder ab("A", p.get());
    BValue rcv_x = ab.Receive(x_in, ab.Literal(Value::Token()));
    BValue rcv_y = ab.Receive(y_in, ab.TupleIndex(rcv_x, 0));
    BValue send_inputs =
        ab.Send(pass_inputs, ab.TupleIndex(rcv_y, 0),
                ab.Tuple({ab.TupleIndex(rcv_x, 1), ab.TupleIndex(rcv_y, 1)}));
    send_inputs =
        ab.Send(pass_inputs, send_inputs,
                ab.Tuple({ab.TupleIndex(rcv_x, 1), ab.TupleIndex(rcv_y, 1)}));

    BValue rcv_result = ab.Receive(pass_result, send_inputs);
    BValue rcv_result_data = ab.TupleIndex(rcv_result, 1);
    BValue send_result0 = ab.Send(result0_out, ab.TupleIndex(rcv_result, 0),
                                  ab.TupleIndex(rcv_result_data, 0));
    ab.Send(result1_out, send_result0, ab.TupleIndex(rcv_result_data, 1));

    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(
      MakeTupleAccumulator("B", pass_inputs, pass_result, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"x", {2, 5, 7}}, {"y", {10}}},
                              {{"result0_out", {4, 14, 28}},
                               {"result1_out", {20, 40, 60}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  // A, B, adapter, and the top block.
  EXPECT_EQ(p->blocks().size(), 4);
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"x", {2, 5, 7}}, {"y", {10}}}),
      IsOkAndHolds(BlockOutputsEq(
          {{"result0_out", {4, 14, 28}}, {"result1_out", {20, 40, 60}}})));
}

TEST_F(ProcInliningPassTest, SingleValueChannelWithVariantElements2) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  Type* u64 = p->GetBitsType(64);
  Type* u32_u64 = p->GetTupleType({u32, u64});
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_in,
      p->CreateStreamingChannel("x", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * y_in,
      p->CreateSingleValueChannel("y", ChannelOps::kReceiveOnly, u64));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_inputs,
      p->CreateStreamingChannel("pass_inputs", ChannelOps::kSendReceive,
                                u32_u64, /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0),
                                /*flow_control=*/FlowControl::kReadyValid,
                                /*strictness=*/ChannelStrictness::kTotalOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_result,
      p->CreateStreamingChannel(
          "pass_result", ChannelOps::kSendReceive, u32_u64,
          /*initial_values=*/{}, /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * result0_out,
      p->CreateStreamingChannel("result0_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * result1_out,
      p->CreateStreamingChannel("result1_out", ChannelOps::kSendOnly, u64));

  {
    // Proc "A". Element 1 sent on pass_inputs is variant:
    //
    // while true:
    //   x = rcv(x_in)  // streaming
    //   y = rcv(y_in)  // single-value
    //   send(pass_inputs, (x+1, y+1))
    //   (result0, result1) = rcv(pass_result)
    //   send(result0_out, result0)
    //   send(result1_out, result1)
    ProcBuilder ab("A", p.get());
    BValue rcv_x = ab.Receive(x_in, ab.Literal(Value::Token()));
    BValue rcv_y = ab.Receive(y_in, ab.TupleIndex(rcv_x, 0));
    BValue x_plus_1 = ab.Add(ab.TupleIndex(rcv_x, 1), ab.Literal(UBits(1, 32)));
    BValue y_plus_1 = ab.Add(ab.TupleIndex(rcv_y, 1), ab.Literal(UBits(1, 64)));
    BValue send_inputs = ab.Send(pass_inputs, ab.TupleIndex(rcv_y, 0),
                                 ab.Tuple({x_plus_1, y_plus_1}));
    send_inputs =
        ab.Send(pass_inputs, send_inputs, ab.Tuple({x_plus_1, y_plus_1}));

    BValue rcv_result = ab.Receive(pass_result, send_inputs);
    BValue rcv_result_data = ab.TupleIndex(rcv_result, 1);
    BValue send_result0 = ab.Send(result0_out, ab.TupleIndex(rcv_result, 0),
                                  ab.TupleIndex(rcv_result_data, 0));
    ab.Send(result1_out, send_result0, ab.TupleIndex(rcv_result_data, 1));

    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(
      MakeTupleAccumulator("B", pass_inputs, pass_result, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"x", {2, 5, 7}}, {"y", {10}}},
                              {{"result0_out", {6, 18, 34}},
                               {"result1_out", {22, 44, 66}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_EQ(p->blocks().size(), 4);
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"x", {2, 5, 7}}, {"y", {10}}}),
      IsOkAndHolds(BlockOutputsEq(
          {{"result0_out", {6, 18, 34}}, {"result1_out", {22, 44, 66}}})));
}

TEST_F(ProcInliningPassTest, SingleValueChannelWithVariantElements3) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  Type* u32_u32 = p->GetTupleType({u32, u32});
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_in,
      p->CreateStreamingChannel("x", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * y_in,
      p->CreateSingleValueChannel("y", ChannelOps::kReceiveOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_inputs,
      p->CreateStreamingChannel("pass_inputs", ChannelOps::kSendReceive,
                                u32_u32, /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0),
                                /*flow_control=*/FlowControl::kReadyValid,
                                /*strictness=*/ChannelStrictness::kTotalOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_result,
      p->CreateStreamingChannel(
          "pass_result", ChannelOps::kSendReceive, u32_u32,
          /*initial_values=*/{}, /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * result0_out,
      p->CreateStreamingChannel("result0_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * result1_out,
      p->CreateStreamingChannel("result1_out", ChannelOps::kSendOnly, u32));

  {
    // Proc "A". Element 1 sent on pass_inputs is variant:
    //
    // while true:
    //   x = rcv(x_in)  // streaming
    //   y = rcv(y_in)  // single-value
    //   send(pass_inputs, (y, x+y))
    //   (result0, result1) = rcv(pass_result)
    //   send(result0_out, result0)
    //   send(result1_out, result1)
    ProcBuilder ab("A", p.get());
    BValue rcv_x = ab.Receive(x_in, ab.Literal(Value::Token()));
    BValue rcv_y = ab.Receive(y_in, ab.TupleIndex(rcv_x, 0));
    BValue x = ab.TupleIndex(rcv_x, 1);
    BValue y = ab.TupleIndex(rcv_y, 1);
    BValue send_inputs = ab.Send(pass_inputs, ab.TupleIndex(rcv_y, 0),
                                 ab.Tuple({y, ab.Add(x, y)}));
    send_inputs =
        ab.Send(pass_inputs, send_inputs, ab.Tuple({y, ab.Add(x, y)}));

    BValue rcv_result = ab.Receive(pass_result, send_inputs);
    BValue rcv_result_data = ab.TupleIndex(rcv_result, 1);
    BValue send_result0 = ab.Send(result0_out, ab.TupleIndex(rcv_result, 0),
                                  ab.TupleIndex(rcv_result_data, 0));
    ab.Send(result1_out, send_result0, ab.TupleIndex(rcv_result_data, 1));

    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(
      MakeTupleAccumulator("B", pass_inputs, pass_result, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"x", {2, 5, 7}}, {"y", {10}}},
                              {{"result0_out", {20, 40, 60}},
                               {"result1_out", {24, 54, 88}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);
  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"x", {2, 5, 7}}, {"y", {10}}}),
      IsOkAndHolds(BlockOutputsEq(
          {{"result0_out", {20, 40, 60}}, {"result1_out", {24, 54, 88}}})));
}

TEST_F(ProcInliningPassTest, SingleValueChannelWithVariantElements4) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  Type* u32_u32 = p->GetTupleType({u32, u32});
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * x_in,
      p->CreateStreamingChannel("x", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * y_in,
      p->CreateSingleValueChannel("y", ChannelOps::kReceiveOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_inputs,
      p->CreateStreamingChannel("pass_inputs", ChannelOps::kSendReceive,
                                u32_u32, /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0),
                                /*flow_control=*/FlowControl::kReadyValid,
                                /*strictness=*/ChannelStrictness::kTotalOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pass_result,
      p->CreateStreamingChannel(
          "pass_result", ChannelOps::kSendReceive, u32_u32,
          /*initial_values=*/{}, /*fifo_config=*/FifoConfigWithDepth(0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * result0_out,
      p->CreateStreamingChannel("result0_out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * result1_out,
      p->CreateStreamingChannel("result1_out", ChannelOps::kSendOnly, u32));

  {
    // Proc "A". Both elements sent on pass_inputs are variant:
    //
    // while true:
    //   x = rcv(x_in)  // streaming
    //   y = rcv(y_in)  // single-value
    //   send(pass_inputs, (x, x+y))  // single-value
    //   (result0, result1) = rcv(pass_result)
    //   send(result0_out, result0)
    //   send(result1_out, result1)
    ProcBuilder ab("A", p.get());
    BValue rcv_x = ab.Receive(x_in, ab.Literal(Value::Token()));
    BValue rcv_y = ab.Receive(y_in, ab.TupleIndex(rcv_x, 0));
    BValue x = ab.TupleIndex(rcv_x, 1);
    BValue y = ab.TupleIndex(rcv_y, 1);
    BValue send_inputs = ab.Send(pass_inputs, ab.TupleIndex(rcv_y, 0),
                                 ab.Tuple({x, ab.Add(x, y)}));
    send_inputs =
        ab.Send(pass_inputs, send_inputs, ab.Tuple({x, ab.Add(x, y)}));

    BValue rcv_result = ab.Receive(pass_result, send_inputs);
    BValue rcv_result_data = ab.TupleIndex(rcv_result, 1);
    BValue send_result0 = ab.Send(result0_out, ab.TupleIndex(rcv_result, 0),
                                  ab.TupleIndex(rcv_result_data, 0));
    ab.Send(result1_out, send_result0, ab.TupleIndex(rcv_result_data, 1));

    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(
      MakeTupleAccumulator("B", pass_inputs, pass_result, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"x", {2, 5, 7}}, {"y", {10}}},
                              {{"result0_out", {4, 14, 28}},
                               {"result1_out", {24, 54, 88}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);
  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"x", {2, 5, 7}}, {"y", {10}}}),
      IsOkAndHolds(BlockOutputsEq(
          {{"result0_out", {4, 14, 28}}, {"result1_out", {24, 54, 88}}})));
}

TEST_F(ProcInliningPassTest, TokenFanIn) {
  // Receive from two inputs, join the tokens then send the sum through another
  // proc.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in0,
      p->CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in1,
      p->CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    ProcBuilder ab("A", p.get());
    BValue tok = ab.Literal(Value::Token());
    BValue rcv_in0 = ab.Receive(ch_in0, tok);
    BValue rcv_in1 = ab.Receive(ch_in1, tok);
    BValue tkn_join =
        ab.AfterAll({ab.TupleIndex(rcv_in0, 0), ab.TupleIndex(rcv_in1, 0)});
    BValue send_to_b =
        ab.Send(a_to_b, tkn_join,
                ab.Add(ab.TupleIndex(rcv_in0, 1), ab.TupleIndex(rcv_in1, 1)));
    BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);
    ab.Send(ch_out, ab.TupleIndex(rcv_from_b, 0), ab.TupleIndex(rcv_from_b, 1));
    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(MakeLoopbackProc("B", a_to_b, b_to_a, p.get()).status());

  XLS_EXPECT_OK(EvalAndExpect(p.get(),
                              {{"in0", {2, 5, 7}}, {"in1", {10, 20, 30}}},
                              {{"out", {12, 25, 37}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);
  EXPECT_EQ(p->blocks().size(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata,
                        {{"in0", {2, 5, 7}}, {"in1", {10, 20, 30}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {12, 25, 37}}})));
}

TEST_F(ProcInliningPassTest, TokenFanOut) {
  // Send an input to two different procs, receive from them, join the tokens
  // and send the sum of the results.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_c,
      p->CreateStreamingChannel("a_to_c", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * c_to_a,
      p->CreateStreamingChannel("c_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    // Proc "A":
    //
    // while true:
    //   x = rcv(in)
    //   send(a_to_b, x)
    //   send(a_to_c, 2*x)
    //   y = rcv(b_to_a)
    //   z = rcv(c_to_a)
    //   send(out, y + z)
    ProcBuilder ab("A", p.get());
    BValue rcv_in = ab.Receive(ch_in, ab.Literal(Value::Token()));
    BValue rcv_tkn = ab.TupleIndex(rcv_in, 0);
    BValue rcv_data = ab.TupleIndex(rcv_in, 1);

    BValue send_to_b = ab.Send(a_to_b, rcv_tkn, rcv_data);
    BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);

    BValue send_to_c =
        ab.Send(a_to_c, rcv_tkn, ab.UMul(rcv_data, ab.Literal(UBits(2, 32))));
    BValue rcv_from_c = ab.Receive(c_to_a, send_to_c);

    BValue tkn_join = ab.AfterAll(
        {ab.TupleIndex(rcv_from_b, 0), ab.TupleIndex(rcv_from_c, 0)});
    ab.Send(ch_out, tkn_join,
            ab.Add(ab.TupleIndex(rcv_from_b, 1), ab.TupleIndex(rcv_from_c, 1)));
    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(
      MakeLoopingAccumulatorProc("B", a_to_b, b_to_a, /*iterations=*/2, p.get())
          .status());
  XLS_ASSERT_OK(
      MakeLoopingAccumulatorProc("C", a_to_c, c_to_a, /*iterations=*/3, p.get())
          .status());

  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {2, 5, 7}}}, {{"out", {10, 19, 25}}})
          .status());

  LOG(INFO) << "================= BEFORE";
  LOG(INFO) << p->DumpIr();

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 4);

  LOG(INFO) << "================= AFTER";
  LOG(INFO) << p->DumpIr();

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {2, 5, 7}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {10, 19, 25}}})));
}

TEST_F(ProcInliningPassTest, RandomProcNetworks) {
  // Create random proc networks and verify the results are the same before and
  // after proc inlining.

  // Number of proc networks to generate.
  const int kNumberSamples = 100;

  // Maximum number of iterations of any proc after receiving but before sending
  // data.
  const int kMaxIterationCount = 10;

  // Maximum number of procs in the network.
  const int kMaxProcCount = 10;

  std::minstd_rand bit_gen;

  for (int sample = 0; sample < kNumberSamples; ++sample) {
    auto p = CreatePackage();
    Type* u32 = p->GetBitsType(32);

    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * ch_in,
        p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
    XLS_ASSERT_OK_AND_ASSIGN(
        Channel * ch_out,
        p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

    // Top level builder.
    ProcBuilder b("top_proc", p.get());
    BValue receive_in = b.Receive(ch_in, b.Literal(Value::Token()));

    // Vector of all the receives in the top level proc. When data is needed to
    // send to another proc, one of these values is randomly chosen.
    std::vector<BValue> receives = {receive_in};

    // Vector of all the tokens coming form send/receive nodes. When a
    // send/receive is generated a random subset of these is chosen as
    // predecessors in the token network.
    std::vector<BValue> tokens = {b.TupleIndex(receive_in, 0)};

    // Pair of channels for communicated with each nested proc and whether or
    // not data has been sent/received to/from the proc.
    struct ChannelPair {
      Channel* send_channel;
      bool sent;
      BValue send;
      Channel* receive_channel;
      bool received;
    };
    std::vector<ChannelPair> channel_pairs;

    // Construct set of non-top-level procs
    int proc_count =
        absl::Uniform(absl::IntervalClosed, bit_gen, 1, kMaxProcCount);
    std::vector<Proc*> procs;
    for (int proc_number = 0; proc_number < proc_count; ++proc_number) {
      // Generate channels to talk with proc.
      XLS_ASSERT_OK_AND_ASSIGN(
          Channel * from_top,
          p->CreateStreamingChannel(
              absl::StrFormat("top_to_proc%d", proc_number),
              ChannelOps::kSendReceive, u32,
              /*initial_values=*/{},
              /*fifo_config=*/
              FifoConfigWithDepth(absl::Uniform<int64_t>(
                  // TODO: google/xls#1509 - use depth-0 here when it is better
                  // supported (i.e. when this does not lead to cycles).
                  absl::IntervalClosed, bit_gen, 1, 10))));
      XLS_ASSERT_OK_AND_ASSIGN(
          Channel * to_top, p->CreateStreamingChannel(
                                absl::StrFormat("proc%d_to_top", proc_number),
                                ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/
                                FifoConfigWithDepth(absl::Uniform<int64_t>(
                                    absl::IntervalClosed, bit_gen, 0, 1))));

      channel_pairs.push_back(ChannelPair{.send_channel = from_top,
                                          .sent = false,
                                          .send = BValue(),
                                          .receive_channel = to_top,
                                          .received = false});

      // Generate proc with random loop iteration count.
      int64_t iteration_count =
          absl::Uniform(absl::IntervalClosed, bit_gen, 1, kMaxIterationCount);
      XLS_ASSERT_OK(
          MakeLoopingAccumulatorProc(absl::StrFormat("proc%d", proc_number),
                                     from_top, to_top,
                                     /*iterations=*/iteration_count, p.get())
              .status());
    }

    auto join_tokens = [&](absl::Span<const BValue> tkns) {
      if (tkns.size() == 1) {
        CHECK(tkns.front().node()->GetType()->IsToken())
            << tkns.front().node()->GetName();
        return tkns.front();
      }
      return b.AfterAll(tkns);
    };

    // While there still exists channels which haven't yet been send/received
    // on, pick a random channel and add a corresponding send/receive node.
    while (!channel_pairs.empty()) {
      // Choose a random channel to send/receive on.
      std::shuffle(channel_pairs.begin(), channel_pairs.end(), bit_gen);
      ChannelPair& pair = channel_pairs.back();

      bool send_on_channel;
      Channel* channel;
      if (!pair.sent) {
        pair.sent = true;
        send_on_channel = true;
        channel = pair.send_channel;
      } else {
        CHECK(!pair.received);
        pair.received = true;
        send_on_channel = false;
        channel = pair.receive_channel;
      }

      if (send_on_channel) {
        // Choose random token predecessors.
        std::shuffle(tokens.begin(), tokens.end(), bit_gen);
        size_t token_count = absl::Uniform<size_t>(absl::IntervalClosed,
                                                   bit_gen, 1, tokens.size());
        std::vector<BValue> token_predecessors;
        token_predecessors.reserve(token_count);
        for (size_t i = 0; i < token_count; i++) {
          token_predecessors.push_back(tokens[i]);
        }

        // Pick a random receive to get data from.
        std::shuffle(receives.begin(), receives.end(), bit_gen);
        BValue receive = receives.front();
        BValue data = b.TupleIndex(receive, 1);

        // The send must be a token successor of the data source (receive).
        token_predecessors.push_back(b.TupleIndex(receive, 0));

        BValue send = b.Send(channel, join_tokens(token_predecessors), data);
        pair.send = send;
        tokens.push_back(send);
      } else {
        // The receive must be a token successor of the corresponding send.
        BValue receive = b.Receive(channel, pair.send);
        receives.push_back(receive);
        tokens.push_back(b.TupleIndex(receive, 0));

        // Done with this channel pair.
        channel_pairs.pop_back();
      }
    }

    // Sum all data from all receives together.
    BValue sum = b.TupleIndex(receives[0], 1);
    for (int64_t i = 1; i < receives.size(); ++i) {
      sum = b.Add(sum, b.TupleIndex(receives[i], 1));
    }

    b.Send(ch_out, join_tokens(tokens), sum);
    XLS_ASSERT_OK_AND_ASSIGN(Proc * top, b.Build());
    XLS_ASSERT_OK(p->SetTop(top));

    VLOG(1) << "Sample " << sample << " (before inlining):\n" << p->DumpIr();

    // Run the proc network interpreter on the proc network before inlining
    // using a few prechosen inputs. After inlining, the generated results
    // should be the same.
    absl::flat_hash_map<std::string, std::vector<int64_t>> inputs = {
        {"in", {2, 5, 7}}};
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<SerialProcRuntime> interpreter,
                             CreateInterpreter(p.get(), inputs));
    ChannelQueue& output_queue = interpreter->queue_manager().GetQueue(ch_out);
    while (output_queue.GetSize() < inputs.at("in").size()) {
      XLS_ASSERT_OK(interpreter->Tick());
    }
    absl::flat_hash_map<std::string, std::vector<uint64_t>> expected_outputs;
    while (!output_queue.IsEmpty()) {
      Value output = output_queue.Read().value();
      expected_outputs[ch_out->name()].push_back(
          output.bits().ToUint64().value());
    }

    XLS_ASSERT_OK_AND_ASSIGN(
        (auto [changed, unit]),
        RunBlockStitchingPass(p.get(), /*top_name=*/"top_proc"));
    EXPECT_TRUE(changed);

    VLOG(1) << "Sample " << sample << " (after inlining):\n" << p->DumpIr();
    XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("top_proc"));
    EXPECT_THAT(EvalBlock(top_block, unit.metadata, inputs, /*num_cycles=*/100),
                IsOkAndHolds(BlockOutputsEq(expected_outputs)));
  }
}

TEST_F(ProcInliningPassTest, DataDependencyWithoutTokenDependency) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  ProcBuilder ab("A", p.get());
  BValue tok = ab.Literal(Value::Token());
  BValue rcv_in = ab.Receive(ch_in, tok);
  BValue in_data = ab.TupleIndex(rcv_in, 1);
  BValue send_to_b = ab.Send(a_to_b, ab.TupleIndex(rcv_in, 0), in_data);
  BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);

  // Send gets its token from the token param, so no token dependency from the
  // (data-dependent) receive from b.
  ab.Send(ch_out, tok, ab.Add(in_data, ab.TupleIndex(rcv_from_b, 1)));

  XLS_ASSERT_OK(ab.Build());

  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("B", /*delay=*/3, a_to_b, b_to_a, p.get())
          .status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {123, 22, 42}}}, {{"out", {246, 44, 84}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {123, 22, 42}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {246, 44, 84}}})));
}

TEST_F(ProcInliningPassTest, ReceivedValueSentAndNext) {
  // Receive a value and pass to a send and a next-state value. This tests
  // whether the received value is properly saved due to being passed to the
  // next-state value.
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));

  {
    // Proc "A":
    //
    // st = 0
    // while true:
    //   in = rcv(in)
    //   send(a_to_b, in)
    //   x = rcv(b_to_a)
    //   send(out, st + x)
    //   st = in
    ProcBuilder ab("A", p.get());
    BValue st = ab.StateElement("st", Value(UBits(0, 32)));
    BValue rcv_in = ab.Receive(ch_in, ab.Literal(Value::Token()));
    BValue rcv_tkn = ab.TupleIndex(rcv_in, 0);
    BValue rcv_data = ab.TupleIndex(rcv_in, 1);

    BValue send_to_b = ab.Send(a_to_b, rcv_tkn, rcv_data);
    BValue rcv_from_b = ab.Receive(b_to_a, send_to_b);

    ab.Send(ch_out, ab.TupleIndex(rcv_from_b, 0),
            ab.Add(ab.TupleIndex(rcv_from_b, 1), st));
    ab.Next(st, ab.Identity(rcv_data));
    XLS_ASSERT_OK(ab.Build().status());
  }

  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("B", /*delay=*/2, a_to_b, b_to_a, p.get())
          .status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {5, 7, 13}}}, {{"out", {5, 12, 20}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {5, 7, 13}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {5, 12, 20}}})));
}

TEST_F(ProcInliningPassTest, OffsetSendAndReceive) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  {
    // Proc A sends every 16 ticks starting at tick 0:
    //
    // u32 st = 0
    // while true:
    //   if st & 0xf == 0:
    //     send(a_to_b, st)
    //   st = st + 1
    ProcBuilder ab("A", p.get());
    BValue st = ab.StateElement("st", Value(UBits(0, 32)));
    BValue send_cond =
        ab.Eq(ab.And(st, ab.Literal(UBits(0xf, 32))), ab.Literal(UBits(0, 32)));
    ab.SendIf(a_to_b, ab.Literal(Value::Token()), send_cond, st);
    ab.Next(st, ab.Add(st, ab.Literal(UBits(1, 32))));
    XLS_ASSERT_OK(ab.Build());
  }

  {
    // Proc B receives every 8 ticks starting at tick 7.
    //
    // u32 st = 0:
    // while true:
    //   if st & 0b111 == 0b111:
    //     data = rcv(a_to_b)
    //     send(out, data)
    //   st = st + 1
    ProcBuilder bb("B", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 32)));
    BValue cond =
        bb.Eq(bb.And(st, bb.Literal(UBits(7, 32))), bb.Literal(UBits(7, 32)));
    BValue rcv = bb.ReceiveIf(a_to_b, bb.Literal(Value::Token()), cond);
    BValue rcv_token = bb.TupleIndex(rcv, 0);
    BValue rcv_data = bb.TupleIndex(rcv, 1);
    bb.SendIf(ch_out, rcv_token, cond, rcv_data);
    bb.Next(st, bb.Add(st, bb.Literal(UBits(1, 32))));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {}, {{"out", {0, 16, 32, 48, 64}}}).status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {}, /*num_cycles=*/80),
              IsOkAndHolds(BlockOutputsEq({{"out", {0, 16, 32, 48, 64}}})));
}

TEST_F(ProcInliningPassTest, InliningProducesCycle) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel(
          "b_to_a", ChannelOps::kSendReceive, u32,
          /*initial_values=*/{},
          /*fifo_config=*/
          FifoConfig(/*depth=*/10, /*bypass=*/false,
                     // We register push outputs to break the cycle.
                     /*register_push_outputs=*/true,
                     /*register_pop_outputs=*/false)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  {
    // On tick 0, outer proc only sends to inner proc (and out). Subsequent
    // ticks receive from inner proc and then send the value back to inner proc
    // and to out channel.
    //
    // u1 st = 0
    // while true:
    //   if st:
    //     data = rcv(b_to_a)
    //   else:
    //     data = 0
    //   send(a_to_b, data)
    //   send(out, data)
    //   st = 1
    //
    ProcBuilder ab("A", p.get());
    BValue st = ab.StateElement("st", Value(UBits(0, 1)));
    BValue rcv = ab.ReceiveIf(b_to_a, ab.Literal(Value::Token()), st);
    BValue rcv_token = ab.TupleIndex(rcv, 0);
    BValue rcv_data = ab.TupleIndex(rcv, 1);
    BValue send_to_b = ab.Send(a_to_b, rcv_token, rcv_data);
    ab.Send(ch_out, send_to_b, rcv_data);
    ab.Next(st, ab.Literal(UBits(1, 1)));
    XLS_ASSERT_OK(ab.Build());
  }

  {
    // Inner proc receives every 8 ticks starting at tick 7:
    //
    // while true:
    //   data = rcv(a_to_b)
    //   data = data + 1
    //   send(b_to_a, data)
    ProcBuilder bb("B", p.get());
    BValue rcv = bb.Receive(a_to_b, bb.Literal(Value::Token()));
    BValue rcv_token = bb.TupleIndex(rcv, 0);
    BValue rcv_data = bb.TupleIndex(rcv, 1);
    bb.Send(b_to_a, rcv_token, bb.Add(rcv_data, bb.Literal(UBits(1, 32))));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {}, {{"out", {0, 1, 2, 3}}}).status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {}, /*num_cycles=*/6),
              IsOkAndHolds(BlockOutputsEq({{"out", {0, 1, 2, 3}}})));
}

TEST_F(ProcInliningPassTest, MultipleSends) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1),
                                /*flow_control=*/FlowControl::kReadyValid,
                                /*strictness=*/ChannelStrictness::kTotalOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  {
    // u1 st = 0
    // while true:
    //   input = rcv(in)
    //   if st
    //     send(a_to_b, input + 10)
    //   else:
    //     send(a_to_b, input)
    //   st = !st
    TokenlessProcBuilder ab("A", "tkn", p.get());
    BValue st = ab.StateElement("st", Value(UBits(0, 1)));
    BValue input = ab.Receive(ch_in);
    ab.SendIf(a_to_b, st, ab.Add(input, ab.Literal(UBits(10, 32))));
    ab.SendIf(a_to_b, ab.Not(st), input);
    ab.Next(st, ab.Not(st));
    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(MakeLoopbackProc("B", a_to_b, ch_out, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3, 42, 123}}},
                              {{"out", {1, 12, 3, 52, 123}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  // top, A, B, and adapter
  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(
      // Seems to hang on last output unless you give another input.
      EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3, 42, 123, 200}}}),
      IsOkAndHolds(BlockOutputsEq({{"out", {1, 12, 3, 52, 123, 210}}})));
}

TEST_F(ProcInliningPassTest, MultipleSendsInDifferentOrder) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1),
                                /*flow_control=*/FlowControl::kReadyValid,
                                /*strictness=*/ChannelStrictness::kTotalOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  {
    // u1 st = 0
    // while true:
    //   input = rcv(in)
    //   if !st
    //     send(a_to_b, input)
    //   else:
    //     send(a_to_b, input + 10)
    //   st = !st
    TokenlessProcBuilder ab("A", "tkn", p.get());
    BValue st = ab.StateElement("st", Value(UBits(0, 1)));
    BValue input = ab.Receive(ch_in);
    ab.SendIf(a_to_b, ab.Not(st), input);
    ab.SendIf(a_to_b, st, ab.Add(input, ab.Literal(UBits(10, 32))));
    ab.Next(st, ab.Not(st));
    XLS_ASSERT_OK(ab.Build());
  }

  XLS_ASSERT_OK(MakeLoopbackProc("B", a_to_b, ch_out, p.get()).status());

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3, 42, 123}}},
                              {{"out", {1, 12, 3, 52, 123}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  // top, A, B, and adapter
  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3, 42, 123}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {1, 12, 3, 52, 123}}})));
}

TEST_F(ProcInliningPassTest, MultipleReceivesFifoDepth0) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0),
                                /*flow_control=*/FlowControl::kReadyValid,
                                /*strictness=*/ChannelStrictness::kTotalOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK(MakeLoopbackProc("A", ch_in, a_to_b, p.get()).status());

  {
    // u1 st = 0
    // while true:
    //   if st
    //     x = rcv(a_to_b) + 10
    //   else:
    //     x = rcv(a_to_b)
    //   send(out, x)
    //   st = !st
    TokenlessProcBuilder bb("B", "tkn", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 1)));
    BValue tmp0 = bb.Add(bb.ReceiveIf(a_to_b, st), bb.Literal(UBits(10, 32)));
    BValue tmp1 = bb.ReceiveIf(a_to_b, bb.Not(st));
    BValue x = bb.Select(st, tmp0, tmp1);
    bb.Send(ch_out, x);
    bb.Next(st, bb.Not(st));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3, 42, 123}}},
                              {{"out", {1, 12, 3, 52, 123}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  // top, A, B, and adapter
  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3, 42, 123}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {1, 12, 3, 52, 123}}})));
}

TEST_F(ProcInliningPassTest, MultipleReceivesFifoDepth1) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1),
                                /*flow_control=*/FlowControl::kReadyValid,
                                /*strictness=*/ChannelStrictness::kTotalOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK(MakeLoopbackProc("A", ch_in, a_to_b, p.get()).status());

  {
    // u1 st = 0
    // while true:
    //   if st
    //     x = rcv(a_to_b) + 10
    //   else:
    //     x = rcv(a_to_b)
    //   send(out, x)
    //   st = !st
    TokenlessProcBuilder bb("B", "tkn", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 1)));
    BValue tmp0 = bb.Add(bb.ReceiveIf(a_to_b, st), bb.Literal(UBits(10, 32)));
    BValue tmp1 = bb.ReceiveIf(a_to_b, bb.Not(st));
    BValue x = bb.Select(st, tmp0, tmp1);
    bb.Send(ch_out, x);
    bb.Next(st, bb.Not(st));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3, 42, 123}}},
                              {{"out", {1, 12, 3, 52, 123}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  // top, A, B, and adapter
  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3, 42, 123}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {1, 12, 3, 52, 123}}})));
}

TEST_F(ProcInliningPassTest, MultipleReceivesDoesNotFireEveryTick) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel(
          "a_to_b", ChannelOps::kSendReceive, u32,
          /*initial_values=*/{},
          /*fifo_config=*/FifoConfigWithDepth(0),
          /*flow_control=*/FlowControl::kReadyValid,
          /*strictness=*/ChannelStrictness::kRuntimeMutuallyExclusive));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK(MakeLoopbackProc("A", ch_in, a_to_b, p.get()).status());

  {
    // u2 st = 0
    // while true:
    //   if st == 0:
    //     x = rcv(a_to_b)
    //   else if st == 1:
    //     x = rcv(a_to_b) + 1
    //   else if st == 2:
    //     x = rcv(a_to_b) + 2
    //   send(out, x)
    //   st = st + 1
    TokenlessProcBuilder bb("B", "tkn", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 2)));
    BValue st_eq_0 = bb.Eq(st, bb.Literal(UBits(0, 2)));
    BValue st_eq_1 = bb.Eq(st, bb.Literal(UBits(1, 2)));
    BValue st_eq_2 = bb.Eq(st, bb.Literal(UBits(2, 2)));
    BValue tmp0 = bb.ReceiveIf(a_to_b, st_eq_0);
    BValue tmp1 =
        bb.Add(bb.ReceiveIf(a_to_b, st_eq_1), bb.Literal(UBits(1, 32)));
    BValue tmp2 =
        bb.Add(bb.ReceiveIf(a_to_b, st_eq_2), bb.Literal(UBits(2, 32)));
    BValue x = bb.Select(st, {tmp0, tmp1, tmp2, bb.Literal(UBits(0, 32))});
    bb.Trace(bb.AfterAll({}), bb.Literal(Value(UBits(1, 1))), {x}, "x = {}");
    bb.Send(ch_out, x);
    bb.Next(st, bb.Add(st, bb.Literal(UBits(1, 2))));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3, 42, 123, 333}}},
                              {{"out", {1, 3, 5, 0, 42, 124}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  // top, A, B, and adapter
  EXPECT_EQ(p->blocks().size(), 4);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3, 42, 123, 333}}}),
      // EvalBlock() runs further and produces all outputs.
      IsOkAndHolds(BlockOutputsEq({{"out", {1, 3, 5, 0, 42, 124, 335, 0}}})));
}

TEST_F(ProcInliningPassTest, MultipleReceivesDoesNotFireEveryTickFifoDepth0) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel(
          "a_to_b", ChannelOps::kSendReceive, u32,
          /*initial_values=*/{},
          /*fifo_config=*/FifoConfigWithDepth(0),
          /*flow_control=*/FlowControl::kReadyValid,
          /*strictness=*/ChannelStrictness::kRuntimeMutuallyExclusive));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK(MakeLoopbackProc("A", ch_in, a_to_b, p.get()).status());

  {
    // u2 st = 0
    // while true:
    //   if st == 0:
    //     x = rcv(a_to_b)
    //   else if st == 1:
    //     x = rcv(a_to_b) + 1
    //   else if st == 2:
    //     x = rcv(a_to_b) + 2
    //   send(out, x)
    //   st = st + 1
    TokenlessProcBuilder bb("B", "tkn", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 2)));
    BValue st_eq_0 = bb.Eq(st, bb.Literal(UBits(0, 2)));
    BValue st_eq_1 = bb.Eq(st, bb.Literal(UBits(1, 2)));
    BValue st_eq_2 = bb.Eq(st, bb.Literal(UBits(2, 2)));
    // Use after_all() as token to prevent serial dependence between receives
    // that can lead to deadlock with the adapter + pipelining.
    BValue tmp0 =
        bb.TupleIndex(bb.ReceiveIf(a_to_b, bb.AfterAll({}), st_eq_0), 1);
    BValue tmp1 =
        bb.Add(bb.TupleIndex(bb.ReceiveIf(a_to_b, bb.AfterAll({}), st_eq_1), 1),
               bb.Literal(UBits(1, 32)));
    BValue tmp2 =
        bb.Add(bb.TupleIndex(bb.ReceiveIf(a_to_b, bb.AfterAll({}), st_eq_2), 1),
               bb.Literal(UBits(2, 32)));
    BValue x = bb.Select(st, {tmp0, tmp1, tmp2, bb.Literal(UBits(0, 32))});
    bb.Send(ch_out, x);
    bb.Next(st, bb.Add(st, bb.Literal(UBits(1, 2))));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3, 42, 123, 333}}},
                              {{"out", {1, 3, 5, 0, 42, 124}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  // top, A, B, and adapter
  EXPECT_EQ(p->blocks().size(), 4);

  EXPECT_THAT(
      EvalBlock(p->GetBlock("A").value(), unit.metadata,
                {{"in", {1, 2, 3, 42, 123, 333}}}),
      // EvalBlock() runs further and produces all outputs.
      IsOkAndHolds(BlockOutputsEq({{"out", {1, 3, 5, 0, 42, 124, 335, 0}}})));
}

TEST_F(ProcInliningPassTest, MultipleSendsAndReceives) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1),
                                /*flow_control=*/FlowControl::kReadyValid,
                                /*strictness=*/ChannelStrictness::kTotalOrder));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  {
    // u1 st = 1
    // while true:
    //   input = rcv(in)
    //   if st
    //     send(a_to_b, input + 10)
    //   else:
    //     send(a_to_b, input)
    //   st = !st
    TokenlessProcBuilder ab("A", "tkn", p.get());
    BValue st = ab.StateElement("st", Value(UBits(1, 1)));
    BValue input = ab.Receive(ch_in);
    ab.SendIf(a_to_b, st, ab.Add(input, ab.Literal(UBits(10, 32))));
    ab.SendIf(a_to_b, ab.Not(st), input);
    ab.Next(st, ab.Not(st));
    XLS_ASSERT_OK(ab.Build());
  }

  {
    // u1 st = 0
    // while true:
    //   if st
    //     x = rcv(a_to_b) + 100
    //   else:
    //     x = rcv(a_to_b)
    //   send(out, x)
    //   st = !st
    TokenlessProcBuilder bb("B", "tkn", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 1)));
    BValue tmp0 = bb.Add(bb.ReceiveIf(a_to_b, st), bb.Literal(UBits(100, 32)));
    BValue tmp1 = bb.ReceiveIf(a_to_b, bb.Not(st));
    BValue x = bb.Select(st, tmp0, tmp1);
    bb.Send(ch_out, x);
    bb.Next(st, bb.Not(st));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3, 42, 123}}},
                              {{"out", {11, 102, 13, 142, 133}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  // top, A, B, send adapter, and receive adapter
  EXPECT_EQ(p->blocks().size(), 5);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3, 42, 123}}},
                        /*num_cycles=*/2000),
              IsOkAndHolds(BlockOutputsEq({{"out", {11, 102, 13, 142, 133}}})));
}

TEST_F(ProcInliningPassTest, ReceiveIfsWithFalseCondition) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out0,
      p->CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out1,
      p->CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK(MakeLoopbackProc("A", ch_in, a_to_b, p.get()).status());

  {
    // u1 st = 0
    // while true:
    //   x = 0
    //   y = 0
    //   if st
    //     x = rcv(a_to_b) + 100
    //   else:
    //     y = rcv(a_to_b)
    //   send(out0, x)
    //   send(out1, y)
    //   st = !st
    TokenlessProcBuilder bb("B", "tkn", p.get());
    BValue st = bb.StateElement("st", Value(UBits(0, 1)));
    BValue x = bb.Add(bb.ReceiveIf(a_to_b, st), bb.Literal(UBits(100, 32)));
    BValue y = bb.ReceiveIf(a_to_b, bb.Not(st));
    bb.Send(ch_out0, x);
    bb.Send(ch_out1, y);
    bb.Next(st, bb.Not(st));
    XLS_ASSERT_OK(bb.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {1, 2, 3, 42, 123}}},
                              {{"out0", {100, 102, 100, 142, 100}},
                               {"out1", {1, 0, 3, 0, 123}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3, 42, 123}}}),
              IsOkAndHolds(BlockOutputsEq({{"out0", {100, 102, 100, 142, 100}},
                                           {"out1", {1, 0, 3, 0, 123}}})));
}

TEST_F(ProcInliningPassTest, ProcsWithDifferentII) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_to_b,
      p->CreateStreamingChannel("a_to_b", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_to_a,
      p->CreateStreamingChannel("b_to_a", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(42)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * top,
      MakePassThroughProc("A", ch_in, a_to_b, b_to_a, ch_out, p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * inner,
                           MakeDoublerProc("B", a_to_b, b_to_a, p.get()));

  top->SetInitiationInterval(5);
  inner->SetInitiationInterval(7);

  XLS_EXPECT_OK(
      EvalAndExpect(p.get(), {{"in", {1, 2, 3}}}, {{"out", {2, 4, 6}}})
          .status());
  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"A"));
  EXPECT_TRUE(changed);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("A"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {1, 2, 3}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {2, 4, 6}}})));
}

TEST_F(ProcInliningPassTest, ProcsWithNonblockingReceivesWithDroppingProc) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in0,
      p->CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in1,
      p->CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * drop_to_arb,
      p->CreateStreamingChannel("drop_to_arb", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  {
    ProcBuilder b("drop", p.get());
    BValue done = b.StateElement("done", Value(UBits(0, 1)));
    BValue recv = b.Receive(in0, b.Literal(Value::Token()));
    BValue recv_token = b.TupleIndex(recv, 0);
    BValue data = b.TupleIndex(recv, 1);
    b.SendIf(drop_to_arb, recv_token, b.Not(done), data);

    b.Next(done, b.Literal(UBits(1, /*bit_count=*/1)));
    XLS_ASSERT_OK(b.Build());
  }

  XLS_ASSERT_OK(
      MakeArbiterProc("arb", {drop_to_arb, in1}, out, p.get()).status());

  EXPECT_THAT(p->GetFunctionBases(),
              UnorderedElementsAre(m::Proc("drop"), m::Proc("arb")));
  XLS_EXPECT_OK(EvalAndExpect(p.get(),
                              {{"in0", {100, 3, 5, 7, 9, 11}},
                               {"in1", {0, 2, 4, 6, 8, 10}}},
                              {{"out", {100, 0, 2, 4, 6, 8, 10}}})
                    .status());
  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"arb"));
  EXPECT_TRUE(changed);

  EXPECT_THAT(p->blocks(),
              UnorderedElementsAre(m::Block("arb"), m::Block("arb__1"),
                                   m::Block("arb__2")));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("arb"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata,
                {{"in0", {100, 3, 5, 7, 9, 11}}, {"in1", {0, 2, 4, 6, 8, 10}}}),
      IsOkAndHolds(BlockOutputsEq({{"out", {100, 0, 2, 4, 6, 8, 10}}})));
}

TEST_F(ProcInliningPassTest,
       ProcsWithNonblockingReceivesWithLoopingAccumulator) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in0,
      p->CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in1,
      p->CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * accum_to_arb,
      p->CreateStreamingChannel("accum_to_arb", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK(
      MakeLoopingAccumulatorProc("accum", in0, accum_to_arb, 3, p.get()));

  XLS_ASSERT_OK(
      MakeArbiterProc("arb", {accum_to_arb, in1}, out, p.get()).status());

  EXPECT_THAT(p->GetFunctionBases(),
              UnorderedElementsAre(m::Proc("accum"), m::Proc("arb")));
  XLS_EXPECT_OK(
      EvalAndExpect(p.get(),
                    {{"in0", {100, 200, 300}}, {"in1", {0, 2, 4, 6, 8, 10}}},
                    {{"out", {0, 2, 103, 4, 6, 203, 8, 10, 303}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(
      (auto [changed, unit]),
      RunBlockStitchingPass(p.get(), /*top_name=*/"accum"));
  EXPECT_TRUE(changed);
  EXPECT_THAT(p->blocks(),
              UnorderedElementsAre(m::Block("accum"), m::Block("accum__1"),
                                   m::Block("arb")));
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("accum"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata,
                {{"in0", {100, 200, 300}}, {"in1", {0, 2, 4, 6, 8, 10}}}),
      IsOkAndHolds(
          // Arbiter is latency sensitive, so we get the same outputs in a
          // different (still valid) order.
          BlockOutputsEq({{"out", {0, 103, 2, 4, 203, 6, 303, 8, 10}}})));
}

TEST_F(ProcInliningPassTest, ProcWithAssert) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * middle,
      p->CreateStreamingChannel("middle", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("loopback", 4, in, middle, p.get()).status());
  {
    ProcBuilder b("assert_not_zero", p.get());
    BValue recv = b.Receive(middle, b.Literal(Value::Token()));
    BValue recv_token = b.TupleIndex(recv, 0);
    BValue recv_data = b.TupleIndex(recv, 1);
    BValue assert_token =
        b.Assert(recv_token, b.Ne(recv_data, b.Literal(UBits(0, 32))),
                 "input must not be zero");
    b.Send(out, assert_token, recv_data);

    XLS_ASSERT_OK(b.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  // Eval without tripping the assertion
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {100, 200, 300}}},
                              {{"out", {100, 200, 300}}})
                    .status());

  // Eval with a zero input, which should trip the assertion.
  EXPECT_THAT(EvalAndExpect(p.get(), {{"in", {100, 200, 300, 0}}},
                            {{"out", {100, 200, 300, 0}}}),
              StatusIs(absl::StatusCode::kAborted,
                       HasSubstr("input must not be zero")));

  XLS_ASSERT_OK_AND_ASSIGN(
      (auto [changed, unit]),
      RunBlockStitchingPass(p.get(), /*top_name=*/"loopback"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);

  // Eval without tripping the assertion
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("loopback"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {100, 200, 300}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {100, 200, 300}}})));

  // Eval with a zero input, which should trip the assertion.
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"in", {100, 200, 300, 0}}}),
      IsOkAndHolds(Field(
          "interpreter_events", &BlockEvaluationResults::interpreter_events,
          Field("assert_msgs", &InterpreterEvents::assert_msgs,
                ElementsAre(HasSubstr("input must not be zero"))))));
}

TEST_F(ProcInliningPassTest, ProcWithCover) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * middle,
      p->CreateStreamingChannel("middle", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("loopback", 4, in, middle, p.get()).status());
  {
    ProcBuilder b("cover_not_zero", p.get());
    BValue recv = b.Receive(middle, b.Literal(Value::Token()));
    BValue recv_token = b.TupleIndex(recv, 0);
    BValue recv_data = b.TupleIndex(recv, 1);
    b.Cover(b.Ne(recv_data, b.Literal(UBits(0, 32))), "cover_data_ne_0");
    b.Send(out, recv_token, recv_data);

    XLS_ASSERT_OK(b.Build());
  }

  // TODO(google/xls#499): Currently, covers are a no-op in the interpreter, so
  // there's not a great way to test that inlining has done the correct thing.
  // When cover is better supported, check that the cover actually works
  // correctly.
  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(), {{"in", {100, 200, 300, 0}}},
                              {{"out", {100, 200, 300, 0}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN(
      (auto [changed, unit]),
      RunBlockStitchingPass(p.get(), /*top_name=*/"loopback"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("loopback"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"in", {100, 200, 300, 0}}}),
              IsOkAndHolds(BlockOutputsEq({{"out", {100, 200, 300, 0}}})));
}

TEST_F(ProcInliningPassTest, ProcWithGate) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * middle,
      p->CreateStreamingChannel("middle", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("loopback", 4, in, middle, p.get()).status());
  {
    ProcBuilder b("gate_proc", p.get());
    BValue recv = b.Receive(middle, b.Literal(Value::Token()));
    BValue recv_token = b.TupleIndex(recv, 0);
    BValue recv_data = b.TupleIndex(recv, 1);
    // Gate will pass values <= 100, otherwise output zero
    BValue cmp = b.AddCompareOp(Op::kULe, recv_data, b.Literal(UBits(100, 32)));
    BValue gate_data = b.Gate(cmp, recv_data);
    b.Send(out, recv_token, gate_data);

    XLS_ASSERT_OK(b.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  XLS_EXPECT_OK(EvalAndExpect(p.get(),
                              {{"in", {0, 1, 2, 99, 100, 200, 300, 1000}}},
                              {{"out", {0, 1, 2, 99, 100, 0, 0, 0}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN(
      (auto [changed, unit]),
      RunBlockStitchingPass(p.get(), /*top_name=*/"loopback"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("loopback"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata,
                {{"in", {0, 1, 2, 99, 100, 200, 300, 1000}}}),
      IsOkAndHolds(BlockOutputsEq({{"out", {0, 1, 2, 99, 100, 0, 0, 0}}})));
}

TEST_F(ProcInliningPassTest, ProcWithTrace) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in,
      p->CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * middle,
      p->CreateStreamingChannel("middle", ChannelOps::kSendReceive, u32,
                                /*initial_values=*/{},
                                /*fifo_config=*/FifoConfigWithDepth(0)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK(
      MakeDelayedLoopbackProc("loopback", 4, in, middle, p.get()).status());
  {
    ProcBuilder b("trace_not_zero", p.get());
    BValue recv = b.Receive(middle, b.Literal(Value::Token()));
    BValue recv_token = b.TupleIndex(recv, 0);
    BValue recv_data = b.TupleIndex(recv, 1);
    // Trace when input data != 0
    BValue trace_token =
        b.Trace(recv_token, b.Ne(recv_data, b.Literal(UBits(0, 32))),
                {recv_data}, "non-zero data: {}");
    b.Send(out, trace_token, recv_data);

    XLS_ASSERT_OK(b.Build());
  }

  EXPECT_EQ(p->procs().size(), 2);
  // Every number except for the zeros should appear in the trace output.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto interpreter,
      EvalAndExpect(p.get(), {{"in", {100, 200, 300, 0, 400, 0, 500}}},
                    {{"out", {100, 200, 300, 0, 400, 0, 500}}}));

  EXPECT_THAT(
      interpreter->GetInterpreterEvents(p->GetProc("trace_not_zero").value())
          .trace_msgs,
      ElementsAre(FieldsAre(HasSubstr("data: 100"), 0),
                  FieldsAre(HasSubstr("data: 200"), 0),
                  FieldsAre(HasSubstr("data: 300"), 0),
                  FieldsAre(HasSubstr("data: 400"), 0),
                  FieldsAre(HasSubstr("data: 500"), 0)));

  XLS_ASSERT_OK_AND_ASSIGN(
      (auto [changed, unit]),
      RunBlockStitchingPass(p.get(), /*top_name=*/"trace_not_zero"));
  EXPECT_TRUE(changed);

  EXPECT_EQ(p->blocks().size(), 3);

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("trace_not_zero"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata,
                {{"in", {100, 200, 300, 0, 400, 0, 500}}},
                /*num_cycles=*/45),
      IsOkAndHolds(AllOf(
          BlockOutputsEq({{"out", {100, 200, 300, 0, 400, 0, 500}}}),
          Field("interpreter_events",
                &BlockEvaluationResults::interpreter_events,
                Field("trace_msgs", &InterpreterEvents::trace_msgs,
                      ElementsAre(FieldsAre(HasSubstr("data: 100"), 0),
                                  FieldsAre(HasSubstr("data: 200"), 0),
                                  FieldsAre(HasSubstr("data: 300"), 0),
                                  FieldsAre(HasSubstr("data: 400"), 0),
                                  FieldsAre(HasSubstr("data: 500"), 0)))))));
}

TEST_F(ProcInliningPassTest, ProcWithNonblockingReceivesWithPassthrough) {
  // Constructs two procs:
  //  foo: a counter that counts down to zero and then receives on 'in',
  //   sending the value to 'internal' and counting down from that value.
  //  output_passthrough: a passthrough block that non-blocking receives on
  //   'internal'. If a value is present, it passes it through; otherwise, it
  //   sends a literal 1000.
  // Together, the procs function such that if you send a value n to 'in', you
  // will see n as the output followed by n repetitions of 1000.
  // Note the '$0' in the declaration of chan internal. This lets us substitute
  // different values for fifo_depth to be sure that does impact proc
  // inlinining's correctness.
  constexpr std::string_view ir_text = R"(package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="")
chan internal(bits[32], id=1, kind=streaming, ops=send_receive, flow_control=ready_valid, fifo_depth=$0, bypass=$1, register_push_outputs=$2, register_pop_outputs=$2, metadata="")
chan out(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

top proc foo(count: bits[32], init={0}) {
  tkn: token = literal(value=token)
  lit0: bits[32] = literal(value=0)
  lit1: bits[32] = literal(value=1)
  pred: bits[1] = eq(count, lit0)
  recv: (token, bits[32]) = receive(tkn, predicate=pred, channel=in)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  count_minus_one: bits[32] = sub(count, lit1)
  next_count: bits[32] = sel(pred, cases=[count_minus_one, recv_data])
  send_token: token = send(recv_token, recv_data, predicate=pred, channel=internal)
  next_value_for_count: () = next_value(param=count, value=next_count)
}

proc output_passthrough(state:(), init={()}) {
  tkn: token = literal(value=token)
  recv: (token, bits[32], bits[1]) = receive(tkn, blocking=false, channel=internal)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  recv_valid: bits[1] = tuple_index(recv, index=2)
  literal1000: bits[32] = literal(value=1000)
  send_data: bits[32] = sel(recv_valid, cases=[literal1000, recv_data])
  send_token: token = send(recv_token, send_data, channel=out)
  next_value_for_state: () = next_value(param=state, value=state)
}
  )";

  for (int64_t fifo_depth : {0, 1}) {
    LOG(INFO) << "fifo_depth: " << fifo_depth << "\n";
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Package> p,
        ParsePackage(absl::Substitute(ir_text, fifo_depth, fifo_depth == 0,
                                      fifo_depth != 0)));
    EXPECT_THAT(
        p->GetFunctionBases(),
        UnorderedElementsAre(m::Proc("foo"), m::Proc("output_passthrough")));

    XLS_ASSERT_OK(
        EvalAndExpect(p.get(), {{"in", {5, 10}}},
                      {{"out",
                        {5, 1000, 1000, 1000, 1000, 1000, 10, 1000, 1000, 1000,
                         1000, 1000, 1000, 1000, 1000, 1000, 1000}}})
            .status());

    XLS_ASSERT_OK_AND_ASSIGN(
        (auto [changed, unit]),
        RunBlockStitchingPass(p.get(), /*top_name=*/"foo"));
    EXPECT_TRUE(changed);
    EXPECT_THAT(p->blocks(),
                UnorderedElementsAre(m::Block("foo"), m::Block("foo__1"),
                                     m::Block("output_passthrough")));
    XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("foo"));
    EXPECT_THAT(
        EvalBlock(top_block, unit.metadata, {{"in", {5, 10}}}),
        IsOkAndHolds(BlockOutputsMatch(UnorderedElementsAre(
            // Latency sensitivity means we don't necessarily get the same
            // output order, but we should see 5 and 10 once each and all other
            // elements should be 1000.
            Pair(Eq("out"), AllOf(Contains(5).Times(1), Contains(10).Times(1),
                                  Each(AnyOf(Eq(5), Eq(10), Eq(1000)))))))));
  }
}

TEST_F(ProcInliningPassTest, ProcWithConditionalNonblockingReceives) {
  // Similar to ProcWithNonblockingReceivesWithPassthrough, except the
  // passthrough proc has a state bit which alternates between 0 and 1. When the
  // state is 0, it does not perform the non-blocking receive and instead sends
  // 500.
  // Note the '$0' in the declaration of chan internal. This lets us substitute
  // different values for fifo_depth to be sure that does impact proc
  // inlinining's correctness.
  constexpr std::string_view ir_text = R"(package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="")
chan internal(bits[32], id=1, kind=streaming, ops=send_receive, flow_control=ready_valid, fifo_depth=$0, bypass=false, metadata="")
chan out(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

top proc foo(count: bits[32], init={0}) {
  tkn: token = literal(value=token)
  lit0: bits[32] = literal(value=0)
  lit1: bits[32] = literal(value=1)
  pred: bits[1] = eq(count, lit0)
  recv: (token, bits[32]) = receive(tkn, predicate=pred, channel=in)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  count_minus_one: bits[32] = sub(count, lit1)
  next_count: bits[32] = sel(pred, cases=[count_minus_one, recv_data])
  send_token: token = send(recv_token, recv_data, predicate=pred, channel=internal)
  next_value_for_count: () = next_value(param=count, value=next_count)
}

proc output_passthrough(state: bits[1], init={1}) {
  tkn: token = literal(value=token)
  recv: (token, bits[32], bits[1]) = receive(tkn, blocking=false, predicate=state, channel=internal)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  recv_valid: bits[1] = tuple_index(recv, index=2)
  literal500: bits[32] = literal(value=500)
  literal1000: bits[32] = literal(value=1000)
  recv_data_or_literal: bits[32] = sel(recv_valid, cases=[literal1000, recv_data])
  send_data: bits[32] = sel(state, cases=[literal500, recv_data_or_literal])
  send_token: token = send(recv_token, send_data, channel=out)
  next_state: bits[1] = not(state)
  next_value_for_state: () = next_value(param=state, value=next_state)
}
  )";

  for (int64_t fifo_depth : {0, 1, 10}) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<Package> p,
        ParsePackage(absl::Substitute(ir_text, fifo_depth)));
    EXPECT_THAT(
        p->GetFunctionBases(),
        UnorderedElementsAre(m::Proc("foo"), m::Proc("output_passthrough")));

    // No backpressure.
    XLS_ASSERT_OK(
        EvalAndExpect(p.get(), {{"in", {5, 10}}},
                      {{"out",
                        {5, 500, 1000, 500, 1000, 500, 10, 500, 1000, 500, 1000,
                         500, 1000, 500, 1000, 500, 1000}}})
            .status());

    // Yes backpressure.
    XLS_ASSERT_OK(EvalAndExpect(p.get(), {{"in", {4, 10}}},
                                {{"out",
                                  {4, 500, 1000, 500, 1000, 500, 10, 500, 1000,
                                   500, 1000, 500, 1000, 500, 1000, 500}}})
                      .status());

    XLS_ASSERT_OK_AND_ASSIGN(
        (auto [changed, unit]),
        RunBlockStitchingPass(p.get(), /*top_name=*/"foo"));
    EXPECT_TRUE(changed);
    EXPECT_THAT(p->blocks(),
                UnorderedElementsAre(m::Block("foo"), m::Block("foo__1"),
                                     m::Block("output_passthrough")));

    XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("foo"));
    // This test is latency sensitive, so the output can (and does) differ from
    // the proc version. Also, we randomly send/recv on top-level IOs, so these
    // results aren't very stable. The original proc-inlining test only tested
    // at the proc level, so things were more stable (if not more meaningful).
    // Here, at the block level, we simply check that 4/5 and 10 show up in the
    // output.
    int64_t no_backpressure_cycles = 29;
    int64_t yes_backpressure_cycles = 26;
    if (fifo_depth == 0) {
      no_backpressure_cycles += 1;
      yes_backpressure_cycles += 4;
    }
    // No backpressure.
    EXPECT_THAT(
        EvalBlock(top_block, unit.metadata, {{"in", {5, 10}}},
                  /*num_cycles=*/no_backpressure_cycles),
        IsOkAndHolds(BlockOutputsMatch(UnorderedElementsAre(
            // Latency sensitivity means we don't necessarily get the same
            // output order, but we should see 5 and 10 once each and all other
            // elements should be 500 or 1000.
            Pair(Eq("out"),
                 AllOf(Contains(5).Times(1), Contains(10).Times(1),
                       Each(AnyOf(Eq(5), Eq(10), Eq(500), Eq(1000)))))))));

    // Yes backpressure.
    EXPECT_THAT(
        EvalBlock(top_block, unit.metadata, {{"in", {4, 10}}},
                  /*num_cycles=*/yes_backpressure_cycles),
        IsOkAndHolds(BlockOutputsMatch(UnorderedElementsAre(
            // Latency sensitivity means we don't necessarily get the same
            // output order, but we should see 4 and 10 once each and all other
            // elements should be 500 or 1000.
            Pair(Eq("out"),
                 AllOf(Contains(4).Times(1), Contains(10).Times(1),
                       Each(AnyOf(Eq(4), Eq(10), Eq(500), Eq(1000)))))))));
  }
}

TEST_F(ProcInliningPassTest, ProcWithExternalConditionalNonblockingReceives) {
  constexpr std::string_view ir_text = R"(package test

chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, metadata="")
chan internal(bits[32], id=1, kind=streaming, ops=send_receive, flow_control=ready_valid, fifo_depth=2, bypass=false, metadata="")
chan out0(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")
chan out1(bits[32], id=3, kind=streaming, ops=send_only, flow_control=ready_valid, metadata="")

top proc foo(count: bits[2], init={0}) {
  tkn: token = literal(value=token)
  lit0: bits[2] = literal(value=0)
  lit1: bits[2] = literal(value=1)
  pred: bits[1] = eq(count, lit0)
  recv: (token, bits[32], bits[1]) = receive(tkn, blocking=false, predicate=pred, channel=in)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  recv_valid: bits[1] = tuple_index(recv, index=2)
  send0_token: token = send(recv_token, recv_data, predicate=recv_valid, channel=internal)
  send1_token: token = send(send0_token, recv_data, predicate=recv_valid, channel=out1)
  next_count: bits[2] = add(count, lit1)
  next_value_for_count: () = next_value(param=count, value=next_count)
}

proc output_passthrough(state:bits[1], init={0}) {
  tkn: token = literal(value=token)
  recv: (token, bits[32]) = receive(tkn, predicate=state, channel=internal)
  recv_token: token = tuple_index(recv, index=0)
  recv_data: bits[32] = tuple_index(recv, index=1)
  send_token: token = send(recv_token, recv_data, predicate=state, channel=out0)
  next_state: bits[1] = not(state)
  next_value_for_state: () = next_value(param=state, value=next_state)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(ir_text));
  EXPECT_THAT(
      p->GetFunctionBases(),
      UnorderedElementsAre(m::Proc("foo"), m::Proc("output_passthrough")));

  XLS_ASSERT_OK(EvalAndExpect(p.get(), {{"in", {5, 10}}},
                              {{"out0", {5, 10}}, {"out1", {5, 10}}})
                    .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"foo"));
  EXPECT_TRUE(changed);
  EXPECT_THAT(p->blocks(),
              UnorderedElementsAre(m::Block("foo"), m::Block("foo__1"),
                                   m::Block("output_passthrough")));

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("foo"));
  EXPECT_THAT(
      EvalBlock(top_block, unit.metadata, {{"in", {5, 10}}}),
      IsOkAndHolds(BlockOutputsEq({{"out0", {5, 10}}, {"out1", {5, 10}}})));
}
TEST_F(ProcInliningPassTest, NestedProcsUsingEmptyAfterAll) {
  // Uses tokens created using empty after_all.
  constexpr std::string_view ir_text = R"(package test

chan data_in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid, strictness=proven_mutually_exclusive, metadata="""""")
chan data_out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid, strictness=proven_mutually_exclusive, metadata="""""")
chan from_inner_proc(bits[32], id=2, kind=streaming, ops=send_receive, flow_control=ready_valid, strictness=proven_mutually_exclusive, fifo_depth=1, bypass=true, register_push_outputs=false, register_pop_outputs=true, metadata="""""")

top proc foo(__state: (), init={()}) {
  tok: token = after_all(id=5)
  literal.4: bits[1] = literal(value=1, id=4)
  receive.6: (token, bits[32]) = receive(tok, predicate=literal.4, channel=from_inner_proc, id=6)
  tok__1: token = tuple_index(receive.6, index=0, id=8)
  data: bits[32] = tuple_index(receive.6, index=1, id=9)
  tuple_index.7: token = tuple_index(receive.6, index=0, id=7)
  tok__2: token = send(tok__1, data, predicate=literal.4, channel=data_out, id=10)
  tuple.11: () = tuple(id=11)
  next (tuple.11)
}

proc input_passthrough(__state: (), init={()}) {
  after_all.15: token = after_all(id=15)
  literal.14: bits[1] = literal(value=1, id=14)
  receive.16: (token, bits[32]) = receive(after_all.15, predicate=literal.14, channel=data_in, id=16)
  tok: token = tuple_index(receive.16, index=0, id=18)
  data: bits[32] = tuple_index(receive.16, index=1, id=19)
  tuple_index.17: token = tuple_index(receive.16, index=0, id=17)
  tok__1: token = send(tok, data, predicate=literal.14, channel=from_inner_proc, id=20)
  tuple.21: () = tuple(id=21)
  next (tuple.21)
}
  )";

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> p, ParsePackage(ir_text));
  EXPECT_THAT(
      p->GetFunctionBases(),
      UnorderedElementsAre(m::Proc("input_passthrough"), m::Proc("foo")));

  XLS_ASSERT_OK(
      EvalAndExpect(p.get(), {{"data_in", {5, 10}}}, {{"data_out", {5, 10}}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN((auto [changed, unit]),
                           RunBlockStitchingPass(p.get(), /*top_name=*/"foo"));
  EXPECT_TRUE(changed);
  EXPECT_THAT(p->blocks(),
              UnorderedElementsAre(m::Block("foo"), m::Block("foo__1"),
                                   m::Block("input_passthrough")));

  XLS_ASSERT_OK_AND_ASSIGN(Block * top_block, p->GetBlock("foo"));
  EXPECT_THAT(EvalBlock(top_block, unit.metadata, {{"data_in", {5, 10}}}),
              IsOkAndHolds(BlockOutputsEq({{"data_out", {5, 10}}})));
}

}  // namespace
}  // namespace xls::verilog
