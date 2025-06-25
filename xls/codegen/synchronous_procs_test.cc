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
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/pipeline_generator.h"
#include "xls/codegen/test_fifos.h"
#include "xls/common/casts.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/scheduling/pipeline_schedule.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_include.h"
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
      case Op::kConcat:
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
                               SchedulingOptions().clock_period_ps(1), elab));

  ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  // TODO(https://github.com/google/xls/issues/2175): For now use an asynchonous
  // implementation for testing (but use the synchronous schedule).
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

  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces tmp1_ch,
                           pb.AddChannel("tmp1", u32));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces tmp2_ch,
                           pb.AddChannel("tmp2", u32));
  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces tmp3_ch,
                           pb.AddChannel("tmp3", u32));

  down_cast<StreamingChannel*>(tmp1_ch.channel)
      ->channel_config(ChannelConfig(kDepth1Fifo.config));
  down_cast<StreamingChannel*>(tmp2_ch.channel)
      ->channel_config(ChannelConfig(kDepth1Fifo.config));
  down_cast<StreamingChannel*>(tmp3_ch.channel)
      ->channel_config(ChannelConfig(kDepth1Fifo.config));

  Proc* subproc1;
  {
    TokenlessProcBuilder sub_pb(NewStyleProc(), "subproc1", "tkn", &package);
    XLS_ASSERT_OK_AND_ASSIGN(
        ReceiveChannelInterface * sub_in,
        sub_pb.AddInputChannel("sub1_in", package.GetBitsType(32)));
    XLS_ASSERT_OK_AND_ASSIGN(
        SendChannelInterface * sub_out,
        sub_pb.AddOutputChannel("sub1_out", package.GetBitsType(32)));
    sub_pb.Send(sub_out,
                sub_pb.Negate(sub_pb.Add(sub_pb.Receive(sub_in),
                                         sub_pb.Literal(UBits(1, 32)))));
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
    sub_pb.Send(sub_out, sub_pb.UMul(sub_pb.Receive(sub_in),
                                     sub_pb.Literal(UBits(2, 32))));
    XLS_ASSERT_OK_AND_ASSIGN(subproc2, sub_pb.Build());
  }

  BValue negated_input = pb.Negate(pb.Receive(in));
  pb.Send(tmp1_ch.send_interface, negated_input);

  XLS_ASSERT_OK(pb.InstantiateProc(
      "inst1", subproc1, {tmp1_ch.receive_interface, tmp2_ch.send_interface}));
  XLS_ASSERT_OK(pb.InstantiateProc(
      "inst2", subproc2, {tmp2_ch.receive_interface, tmp3_ch.send_interface}));

  pb.Send(out, pb.Add(negated_input, pb.Receive(tmp3_ch.receive_interface)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));
  XLS_ASSERT_OK(package.SetTop(top));

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

  // TODO(https://github.com/google/xls/issues/2175): For now use an
  // asynchronous implementation for testing (but use the synchronous schedule).
  XLS_ASSERT_OK_AND_ASSIGN(
      CodegenResult result,
      ToPipelineModuleText(
          package_schedule, &package,
          BuildPipelineOptions()
              .reset(reset_proto.name(), reset_proto.asynchronous(),
                     reset_proto.active_low(), reset_proto.reset_data_path())
              .use_system_verilog(UseSystemVerilog())
              .emit_as_pipeline(false)));

  VerilogInclude fifo_definition{.relative_path = "fifo.v",
                                 .verilog_text = kDepth1Fifo.rtl};
  std::vector<VerilogInclude> include_definitions = {fifo_definition};
  std::string verilog =
      absl::StrCat("`include \"fifo.v\"\n\n", result.verilog_text);

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog, /*macro_definitions=*/{},
                                 include_definitions);

  // The proc computes (-(-x + 1)) * 2 + -x = x - 2
  ModuleSimulator simulator =
      NewModuleSimulator(verilog, result.signature, include_definitions);
  absl::flat_hash_map<std::string, std::vector<Bits>> inputs = {
      {"top_in", {UBits(3, 32), UBits(10, 32), UBits(42, 32)}}};
  absl::flat_hash_map<std::string, std::vector<Bits>> outputs = {
      {"top_out", {UBits(1, 32), UBits(8, 32), UBits(40, 32)}}};
  EXPECT_THAT(simulator.RunInputSeriesProc(inputs, {{"top_out", 3}}),
              absl_testing::IsOkAndHolds(outputs));
}

INSTANTIATE_TEST_SUITE_P(PipelineGeneratorTestInstantiation,
                         SynchronousProcsTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<SynchronousProcsTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
