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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen/codegen_result.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/test_fifos.h"
#include "xls/codegen_v_1_5/codegen.h"
#include "xls/common/golden_files.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_include.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;

constexpr char kTestName[] = "codegen_proc_test";
constexpr char kTestdataPath[] = "xls/codegen_v_1_5/testdata";

class TestDelayEstimator : public DelayEstimator {
 public:
  TestDelayEstimator() : DelayEstimator("test") {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kParam:
      case Op::kStateRead:
      case Op::kLiteral:
      case Op::kBitSlice:
      case Op::kConcat:
        return 0;
      default:
        return 1;
    }
  }
};

class CodegenProcTest : public verilog::VerilogTestBase {};

TEST_P(CodegenProcTest, IIGreaterThanOne) {
  const std::string ir_text = absl::Substitute(R"(package $0
chan in(bits[32], id=0, kind=streaming, ops=receive_only, flow_control=ready_valid)
chan out(bits[32], id=1, kind=streaming, ops=send_only, flow_control=ready_valid)
chan in_out(bits[32], id=2, kind=streaming, ops=send_only, flow_control=ready_valid)

#[initiation_interval(2)]
proc ii_greater_than_one(st: bits[32], init={0}) {
  tkn: token = literal(value=token, id=1000)
  send.1: token = send(tkn, st, channel=out, id=1)
  min_delay.2: token = min_delay(send.1, delay=1, id=2)
  receive.3: (token, bits[32]) = receive(min_delay.2, channel=in, id=3)
  tuple_index.4: token = tuple_index(receive.3, index=0, id=4)
  tuple_index.5: bits[32] = tuple_index(receive.3, index=1, id=5)
  send.6: token = send(tuple_index.4, tuple_index.5, channel=in_out, id=6)
  next_st: () = next_value(param=st, value=tuple_index.5)
}
)",
                                               TestBaseName());

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Package> package,
                           xls::Parser::ParsePackage(ir_text));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           package->GetProc("ii_greater_than_one"));
  XLS_ASSERT_OK(package->SetTop(proc));

  verilog::ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::CodegenResult result,
      Codegen(
          package.get(),
          verilog::CodegenOptions()
              .flop_inputs(true)
              .flop_outputs(true)
              .clock_name("clk")
              .emit_as_pipeline(true)
              .reset(reset_proto.name(), reset_proto.asynchronous(),
                     reset_proto.active_low(), reset_proto.reset_data_path())
              .use_system_verilog(UseSystemVerilog()),
          SchedulingOptions().clock_period_ps(50).pipeline_stages(2),
          &delay_estimator));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CodegenProcTest, SingleProcWithProcScopedChannels) {
  Package package(TestBaseName());

  TokenlessProcBuilder pb(NewStyleProc(), "myleaf", "tkn", &package);
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in,
                           pb.AddInputChannel("in", package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out,
                           pb.AddOutputChannel("out", package.GetBitsType(32)));

  pb.Send(out, pb.Add(pb.Receive(in), pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK(package.SetTop(proc));

  verilog::ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::CodegenResult result,
      Codegen(
          &package,
          verilog::CodegenOptions()
              .flop_inputs(true)
              .flop_outputs(true)
              .clock_name("clk")
              .emit_as_pipeline(true)
              .reset(reset_proto.name(), reset_proto.asynchronous(),
                     reset_proto.active_low(), reset_proto.reset_data_path())
              .use_system_verilog(UseSystemVerilog()),
          SchedulingOptions().clock_period_ps(50).pipeline_stages(2),
          &delay_estimator));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

TEST_P(CodegenProcTest, ProcScopedChannelsWithLoopbackChannel) {
  Package package(TestBaseName());

  TokenlessProcBuilder pb(NewStyleProc(), "myproc", "tkn", &package);
  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces loopback,
                           pb.AddChannel("loopback", package.GetBitsType(32)));
  dynamic_cast<StreamingChannel*>(loopback.channel)
      ->SetChannelConfig(
          ChannelConfig(FifoConfig(/*depth=*/2, /*bypass=*/false,
                                   /*register_push_outputs=*/true,
                                   /*register_pop_outputs=*/false)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out,
                           pb.AddOutputChannel("out", package.GetBitsType(32)));

  BValue myvalue =
      pb.Add(pb.Receive(loopback.receive_interface), pb.Literal(UBits(1, 32)));
  pb.Send(loopback.send_interface, myvalue);
  pb.Send(out, myvalue);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK(package.SetTop(proc));

  verilog::ResetProto reset_proto;
  reset_proto.set_name("rst");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::CodegenResult result,
      Codegen(
          &package,
          verilog::CodegenOptions()
              .flop_inputs(true)
              .flop_outputs(true)
              .clock_name("clk")
              .emit_as_pipeline(true)
              .reset(reset_proto.name(), reset_proto.asynchronous(),
                     reset_proto.active_low(), reset_proto.reset_data_path())
              .use_system_verilog(UseSystemVerilog()),
          SchedulingOptions().pipeline_stages(3).add_constraint(IOConstraint(
              "loopback", IODirection::kReceive, "loopback", IODirection::kSend,
              /*minimum_latency=*/1, /*maximum_latency=*/1)),
          &delay_estimator));

  // Don't use the verilog variant of ExpectEqual... because that parses the
  // verilog but there is no fifo implementation.
  ExpectEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                          result.verilog_text);
}

absl::StatusOr<Proc*> CreateNewStyleAccumProc(std::string_view proc_name,
                                              Package* package) {
  TokenlessProcBuilder pb(NewStyleProc(), proc_name, "tkn", package);
  BValue accum = pb.StateElement("accum", Value(UBits(0, 32)));
  XLS_ASSIGN_OR_RETURN(
      ReceiveChannelInterface * in_channel,
      pb.AddInputChannel("accum_in", package->GetBitsType(32)));
  BValue input = pb.Receive(in_channel);
  BValue next_accum = pb.Add(accum, input);
  XLS_ASSIGN_OR_RETURN(
      SendChannelInterface * out_channel,
      pb.AddOutputChannel("accum_out", package->GetBitsType(32)));
  pb.Send(out_channel, next_accum);
  return pb.Build({next_accum});
}

TEST_P(CodegenProcTest, TrivialProcHierarchyWithProcScopedChannels) {
  // Construct a proc which instantiates two accumulator procs tied in series.
  Package p(TestBaseName());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * leaf_proc,
                           CreateNewStyleAccumProc("leaf_proc", &p));

  TokenlessProcBuilder pb(NewStyleProc(), "a_top_proc", "tkn", &p);
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in_channel,
                           pb.AddInputChannel("in_ch", p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out_channel,
                           pb.AddOutputChannel("out_ch", p.GetBitsType(32)));

  XLS_ASSERT_OK(pb.InstantiateProc(
      "inst", leaf_proc,
      std::vector<ChannelInterface*>{in_channel, out_channel}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));
  XLS_ASSERT_OK(p.SetTop(top));

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::CodegenResult result,
      Codegen(&p,
              verilog::CodegenOptions()
                  .flop_inputs(true)
                  .flop_outputs(true)
                  .clock_name("clk")
                  .emit_as_pipeline(true)
                  .reset("rst", false, false, false)
                  .use_system_verilog(UseSystemVerilog()),
              SchedulingOptions().pipeline_stages(2).schedule_all_procs(true),
              &delay_estimator));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  verilog::ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, std::vector<Bits>> inputs = {
      {"in_ch", {UBits(0, 32), UBits(10, 32), UBits(42, 32)}}};
  absl::flat_hash_map<std::string, std::vector<Bits>> outputs = {
      {"out_ch", {UBits(0, 32), UBits(10, 32), UBits(52, 32)}}};
  EXPECT_THAT(simulator.RunInputSeriesProc(inputs, {{"out_ch", 3}}),
              IsOkAndHolds(outputs));
}

TEST_P(CodegenProcTest, MultiplyInstantiatedProc) {
  // Construct a proc which instantiates a proc twice which accumulates its
  // inputs.
  Package p(TestBaseName());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * leaf_proc,
                           CreateNewStyleAccumProc("leaf_proc", &p));

  TokenlessProcBuilder pb(NewStyleProc(), "a_top_proc", "tkn", &p);
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in0_channel,
                           pb.AddInputChannel("in0_ch", p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in1_channel,
                           pb.AddInputChannel("in1_ch", p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out0_channel,
                           pb.AddOutputChannel("out0_ch", p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out1_channel,
                           pb.AddOutputChannel("out1_ch", p.GetBitsType(32)));

  XLS_ASSERT_OK(pb.InstantiateProc(
      "inst0", leaf_proc,
      std::vector<ChannelInterface*>{in0_channel, out0_channel}));
  XLS_ASSERT_OK(pb.InstantiateProc(
      "inst1", leaf_proc,
      std::vector<ChannelInterface*>{in1_channel, out1_channel}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));
  XLS_ASSERT_OK(p.SetTop(top));

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::CodegenResult result,
      Codegen(&p,
              verilog::CodegenOptions()
                  .flop_inputs(true)
                  .flop_outputs(true)
                  .clock_name("clk")
                  .emit_as_pipeline(true)
                  .reset("rst", false, false, false)
                  .use_system_verilog(UseSystemVerilog()),
              SchedulingOptions().pipeline_stages(2).schedule_all_procs(true),
              &delay_estimator));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  verilog::ModuleSimulator simulator =
      NewModuleSimulator(result.verilog_text, result.signature);
  absl::flat_hash_map<std::string, std::vector<Bits>> inputs = {
      {"in0_ch", {UBits(0, 32), UBits(10, 32), UBits(42, 32)}},
      {"in1_ch", {UBits(1, 32), UBits(2, 32), UBits(5, 32)}}};
  absl::flat_hash_map<std::string, std::vector<Bits>> outputs = {
      {"out0_ch", {UBits(0, 32), UBits(10, 32), UBits(52, 32)}},
      {"out1_ch", {UBits(1, 32), UBits(3, 32), UBits(8, 32)}}};
  EXPECT_THAT(
      simulator.RunInputSeriesProc(inputs, {{"out0_ch", 3}, {"out1_ch", 3}}),
      IsOkAndHolds(outputs));
}

TEST_P(CodegenProcTest, DeclaredChannelInProc) {
  Package p(TestBaseName());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * leaf_proc,
                           CreateNewStyleAccumProc("leaf_proc", &p));

  TokenlessProcBuilder pb(NewStyleProc(), "a_top_proc", "tkn", &p);
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in_channel,
                           pb.AddInputChannel("in_ch", p.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out_channel,
                           pb.AddOutputChannel("out_ch", p.GetBitsType(32)));

  XLS_ASSERT_OK_AND_ASSIGN(ChannelWithInterfaces middle_channel,
                           pb.AddChannel("middle_ch", p.GetBitsType(32)));

  XLS_ASSERT_OK(
      pb.InstantiateProc("inst0", leaf_proc,
                         std::vector<ChannelInterface*>{
                             in_channel, middle_channel.send_interface}));
  XLS_ASSERT_OK(
      pb.InstantiateProc("inst1", leaf_proc,
                         std::vector<ChannelInterface*>{
                             middle_channel.receive_interface, out_channel}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, pb.Build({}));
  XLS_ASSERT_OK(p.SetTop(top));

  dynamic_cast<StreamingChannel*>(middle_channel.channel)
      ->SetChannelConfig(ChannelConfig(verilog::kDepth1Fifo.config));

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      verilog::CodegenResult result,
      Codegen(&p,
              verilog::CodegenOptions()
                  .flop_inputs(true)
                  .flop_outputs(true)
                  .clock_name("clk")
                  .emit_as_pipeline(true)
                  .reset("rst", false, false, false)
                  .use_system_verilog(UseSystemVerilog()),
              SchedulingOptions().pipeline_stages(2).schedule_all_procs(true),
              &delay_estimator));

  verilog::VerilogInclude fifo_definition{
      .relative_path = "fifo.v", .verilog_text = verilog::kDepth1Fifo.rtl};
  std::vector<verilog::VerilogInclude> include_definitions = {fifo_definition};
  std::string verilog =
      absl::StrCat("`include \"fifo.v\"\n\n", result.verilog_text);

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog, /*macro_definitions=*/{},
                                 include_definitions);

  verilog::ModuleSimulator simulator =
      NewModuleSimulator(verilog, result.signature, include_definitions);
  absl::flat_hash_map<std::string, std::vector<Bits>> inputs = {
      {"in_ch", {UBits(3, 32), UBits(10, 32), UBits(42, 32)}}};
  absl::flat_hash_map<std::string, std::vector<Bits>> outputs = {
      {"out_ch", {UBits(3, 32), UBits(16, 32), UBits(71, 32)}}};
  EXPECT_THAT(simulator.RunInputSeriesProc(inputs, {{"out_ch", 3}}),
              IsOkAndHolds(outputs));
}

TEST_P(CodegenProcTest, CombinationalSingleProcWithProcScopedChannels) {
  Package package(TestBaseName());

  TokenlessProcBuilder pb(NewStyleProc(), "myleaf", "tkn", &package);
  XLS_ASSERT_OK_AND_ASSIGN(ReceiveChannelInterface * in,
                           pb.AddInputChannel("in", package.GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(SendChannelInterface * out,
                           pb.AddOutputChannel("out", package.GetBitsType(32)));

  pb.Send(out, pb.Add(pb.Receive(in), pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK(package.SetTop(proc));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      Codegen(&package, codegen_options().generate_combinational(true),
              SchedulingOptions(),
              /*delay_estimator=*/nullptr));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);
}

INSTANTIATE_TEST_SUITE_P(CodegenProcTestInstantiation, CodegenProcTest,
                         testing::ValuesIn(verilog::kDefaultSimulationTargets),
                         verilog::ParameterizedTestName<CodegenProcTest>);

}  // namespace
}  // namespace xls::codegen
