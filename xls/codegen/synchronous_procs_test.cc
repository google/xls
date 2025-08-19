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

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/source_location.h"
#include "xls/ir/verifier.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

constexpr char kTestName[] = "synchronous_procs_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class TestDelayEstimator : public DelayEstimator {
 public:
  TestDelayEstimator() : DelayEstimator("test") {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kParam:
      case Op::kStateRead:
      case Op::kLiteral:
      case Op::kBitSlice:
      case Op::kTupleIndex:
      case Op::kSend:
      case Op::kReceive:
        return 0;
      default:
        return 1;
    }
  }
};

class SynchronousProcsTest : public VerilogTestBase {};

TEST_P(SynchronousProcsTest, NestedProc) {
  Package package(TestBaseName());

  // Top proc passes input/output transparently to instantiationed subproc.
  TokenlessProcBuilder pb(NewStyleProc(), "my_proc", "tkn", &package);
  XLS_ASSERT_OK_AND_ASSIGN(
      ReceiveChannelInterface * in,
      pb.AddInputChannel("top_in", package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      SendChannelInterface * out,
      pb.AddOutputChannel("top_out", package.GetBitsType(32)));
  in->SetFlowControl(FlowControl::kNone);
  out->SetFlowControl(FlowControl::kNone);

  Proc* subproc;
  {
    // Subproc adds one to its input.
    TokenlessProcBuilder sub_pb(NewStyleProc(), "subproc", "tkn", &package);
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * sub_in,
        sub_pb.AddInputChannel("sub_in", package.GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * sub_out,
        sub_pb.AddOutputChannel("sub_out", package.GetBitsType(32)));
    sub_pb.Send(sub_out, sub_pb.Add(sub_pb.Receive(sub_in),
                                    sub_pb.Literal(UBits(1, 32))));

    sub_in->SetFlowControl(FlowControl::kNone);
    sub_out->SetFlowControl(FlowControl::kNone);

    XLS_ASSERT_OK_AND_ASSIGN(subproc, sub_pb.Build());
  }

  XLS_ASSERT_OK(pb.InstantiateProc("inst1", subproc, {in, out}));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));
  XLS_ASSERT_OK(package.SetTop(top));

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(top));

  XLS_ASSERT_OK_AND_ASSIGN(PackageSchedule package_schedule,
                           RunSynchronousPipelineSchedule(
                               &package, TestDelayEstimator(),
                               SchedulingOptions().clock_period_ps(3), elab));

  ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenResult result,
      ToPipelineModuleText(
          package_schedule, &package,
          BuildPipelineOptions()
              .reset(reset_proto.name(), reset_proto.asynchronous(),
                     reset_proto.active_low(), reset_proto.reset_data_path())
              .use_system_verilog(UseSystemVerilog())
              .emit_as_pipeline(false)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text, /*macro_definitions=*/{},
                                 {});

  // The proc adds one to it's input.
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature, {});
  absl::flat_hash_map<std::string, std::vector<Bits>> inputs = {
      {"top_in", {UBits(3, 32), UBits(10, 32), UBits(42, 32)}}};
  absl::flat_hash_map<std::string, std::vector<Bits>> outputs = {
      {"top_out", {UBits(4, 32), UBits(11, 32), UBits(43, 32)}}};
  EXPECT_THAT(simulator.RunInputSeriesProc(inputs, {{"top_out", 3}}),
              absl_testing::IsOkAndHolds(outputs));
}

TEST_P(SynchronousProcsTest, ChainedProc) {
  Package package(TestBaseName());

  // Top proc instantiates two proc sequentially passing a single input through
  // the chain. The proc also includes a parallel data path within the top proc
  // itself.
  TokenlessProcBuilder pb(NewStyleProc(), "my_proc", "tkn", &package);
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in,
                           pb.AddInputChannel("top_in", u32));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out,
                           pb.AddOutputChannel("top_out", u32));
  in->SetFlowControl(FlowControl::kNone);
  out->SetFlowControl(FlowControl::kNone);

  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces tmp1_ch,
                           pb.AddChannel("tmp1", u32));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces tmp2_ch,
                           pb.AddChannel("tmp2", u32));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces tmp3_ch,
                           pb.AddChannel("tmp3", u32));
  auto setup_channel = [](ChannelWithInterfaces c) {
    down_cast<StreamingChannel*>(c.channel)->SetFlowControl(FlowControl::kNone);
    c.send_interface->SetFlowControl(FlowControl::kNone);
    c.receive_interface->SetFlowControl(FlowControl::kNone);
  };
  setup_channel(tmp1_ch);
  setup_channel(tmp2_ch);
  setup_channel(tmp3_ch);

  Proc* subproc1;
  {
    TokenlessProcBuilder sub_pb(NewStyleProc(), "subproc1", "tkn", &package);
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * sub_in,
        sub_pb.AddInputChannel("sub1_in", package.GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * sub_out,
        sub_pb.AddOutputChannel("sub1_out", package.GetBitsType(32)));
    sub_pb.Send(
        sub_out,
        sub_pb.Add(sub_pb.Receive(sub_in, SourceInfo(), "receive_sub1_in"),
                   sub_pb.Literal(UBits(1, 32))),
        SourceInfo(), "send_sub1_out");

    sub_in->SetFlowControl(FlowControl::kNone);
    sub_out->SetFlowControl(FlowControl::kNone);

    XLS_ASSERT_OK_AND_ASSIGN(subproc1, sub_pb.Build());
  }

  Proc* subproc2;
  {
    TokenlessProcBuilder sub_pb(NewStyleProc(), "subproc2", "tkn", &package);
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * sub_in,
        sub_pb.AddInputChannel("sub2_in", package.GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * sub_out,
        sub_pb.AddOutputChannel("sub2_out", package.GetBitsType(32)));

    sub_pb.Send(
        sub_out,
        sub_pb.Add(sub_pb.Receive(sub_in), sub_pb.Literal(UBits(10, 32)),
                   SourceInfo(), "receive_sub2_in"),
        SourceInfo(), "send_sub2_out");

    sub_in->SetFlowControl(FlowControl::kNone);
    sub_out->SetFlowControl(FlowControl::kNone);

    XLS_ASSERT_OK_AND_ASSIGN(subproc2, sub_pb.Build());
  }

  BValue input = pb.Receive(in, SourceInfo(), "receive_in");
  pb.Send(tmp1_ch.send_interface, input, SourceInfo(), "send_tmp1");

  XLS_ASSERT_OK(pb.InstantiateProc(
      "inst1", subproc1, {tmp1_ch.receive_interface, tmp2_ch.send_interface}));
  XLS_ASSERT_OK(pb.InstantiateProc(
      "inst2", subproc2, {tmp2_ch.receive_interface, tmp3_ch.send_interface}));

  pb.Send(out,
          pb.Add(input, pb.Receive(tmp3_ch.receive_interface, SourceInfo(),
                                   "receive_tmp3")),
          SourceInfo(), "send_out");

  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));
  XLS_ASSERT_OK(package.SetTop(top));

  in->SetFlowControl(FlowControl::kNone);
  out->SetFlowControl(FlowControl::kNone);

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(top));

  XLS_ASSERT_OK_AND_ASSIGN(PackageSchedule package_schedule,
                           RunSynchronousPipelineSchedule(
                               &package, TestDelayEstimator(),
                               SchedulingOptions().clock_period_ps(2), elab));

  ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenResult result,
      ToPipelineModuleText(
          package_schedule, &package,
          BuildPipelineOptions()
              .reset(reset_proto.name(), reset_proto.asynchronous(),
                     reset_proto.active_low(), reset_proto.reset_data_path())
              .use_system_verilog(UseSystemVerilog())
              .emit_as_pipeline(false)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text, /*macro_definitions=*/{},
                                 {});

  // The proc computes ((x + 1) + 10) + x = 2x + 11
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature, {});
  absl::flat_hash_map<std::string, std::vector<Bits>> inputs = {
      {"top_in", {UBits(3, 32), UBits(10, 32), UBits(42, 32)}}};
  absl::flat_hash_map<std::string, std::vector<Bits>> outputs = {
      {"top_out", {UBits(17, 32), UBits(31, 32), UBits(95, 32)}}};
  EXPECT_THAT(simulator.RunInputSeriesProc(inputs, {{"top_out", 3}}),
              absl_testing::IsOkAndHolds(outputs));
}

TEST_P(SynchronousProcsTest, SingleProcWithChannelsWithNoFlowControl) {
  Package package(TestBaseName());

  // Top proc passes input/output transparently to instantiationed subproc.
  TokenlessProcBuilder pb(NewStyleProc(), "my_proc", "tkn", &package);
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in,
                           pb.AddInputChannel("in", package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out,
                           pb.AddOutputChannel("out", package.GetBitsType(32)));

  pb.Send(out, pb.Add(pb.Receive(in), pb.Literal(UBits(42, 32))));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));
  XLS_ASSERT_OK(package.SetTop(top));

  in->SetFlowControl(FlowControl::kNone);
  out->SetFlowControl(FlowControl::kNone);

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(top));
  XLS_ASSERT_OK_AND_ASSIGN(PackageSchedule package_schedule,
                           RunSynchronousPipelineSchedule(
                               &package, TestDelayEstimator(),
                               SchedulingOptions().clock_period_ps(1), elab));

  ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenResult result,
      ToPipelineModuleText(
          package_schedule, &package,
          BuildPipelineOptions()
              .reset(reset_proto.name(), reset_proto.asynchronous(),
                     reset_proto.active_low(), reset_proto.reset_data_path())
              .use_system_verilog(UseSystemVerilog())
              .emit_as_pipeline(false)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text, /*macro_definitions=*/{},
                                 {});

  // The proc computes x + 42.
  ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature, {});
  absl::flat_hash_map<std::string, std::vector<Bits>> inputs = {
      {"in", {UBits(3, 32), UBits(10, 32), UBits(42, 32)}}};
  absl::flat_hash_map<std::string, std::vector<Bits>> outputs = {
      {"out", {UBits(45, 32), UBits(52, 32), UBits(84, 32)}}};
  EXPECT_THAT(simulator.RunInputSeriesProc(inputs, {{"out", 3}}),
              absl_testing::IsOkAndHolds(outputs));
}

INSTANTIATE_TEST_SUITE_P(SynchronousProcsTestInstantiation,
                         SynchronousProcsTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<SynchronousProcsTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
