// Copyright 2020 The XLS Authors
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

#include "xls/codegen/proc_generator.h"

#include "xls/common/status/matchers.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/package.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/simulation/module_simulator.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;
using ::testing::HasSubstr;

constexpr char kTestName[] = "proc_generator_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class ProcGeneratorTest : public VerilogTestBase {
 protected:
  GeneratorOptions options() {
    return GeneratorOptions().use_system_verilog(UseSystemVerilog());
  }
};

TEST_P(ProcGeneratorTest, APlusB) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch,
      package.CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch,
      package.CreatePortChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("sum", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue a = pb.Receive(a_ch);
  BValue b = pb.Receive(b_ch);
  pb.Send(output_ch, pb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  XLS_ASSERT_OK_AND_ASSIGN(auto result, GenerateModule(proc, options()));

  // The block should not have a clock or reset signal.
  EXPECT_FALSE(result.signature.proto().has_reset());
  EXPECT_FALSE(result.signature.proto().has_clock_name());

  ModuleTestbench tb(result.verilog_text, result.signature, GetSimulator());
  tb.ExpectX("sum");
  // The combinational module doesn't a connected clock, but the clock can still
  // be used to sequence events in time.
  tb.NextCycle().Set("a", 0).Set("b", 0).ExpectEq("sum", 0);
  tb.NextCycle().Set("a", 100).Set("b", 42).ExpectEq("sum", 142);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ProcGeneratorTest, PipelinedAPlusB) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * a_ch,
      package.CreatePortChannel("a", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * b_ch,
      package.CreatePortChannel("b", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * p0_a_ch,
      package.CreateRegisterChannel("p0_a", u32,
                                    /*reset_value=*/Value(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * p0_b_ch,
      package.CreateRegisterChannel("p0_b", u32,
                                    /*reset_value=*/Value(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * p1_sum_ch,
      package.CreateRegisterChannel("p1_sum", u32,
                                    /*reset_value=*/Value(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("sum", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);

  BValue a = pb.Receive(a_ch);
  BValue b = pb.Receive(b_ch);

  // Pipeline register 0.
  pb.Send(p0_a_ch, a);
  BValue p0_a = pb.Receive(p0_a_ch);
  pb.Send(p0_b_ch, b);
  BValue p0_b = pb.Receive(p0_b_ch);

  // Pipeline register 1.
  pb.Send(p1_sum_ch, pb.Add(p0_a, p0_b));
  BValue p1_sum = pb.Receive(p1_sum_ch);

  pb.Send(output_ch, p1_sum);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto result,
      GenerateModule(proc, options()
                               .clock_name("the_clock")
                               .reset("the_reset", /*asynchronous=*/false,
                                      /*active_low=*/false)));

  // The block should have a clock or reset signal.
  EXPECT_EQ(result.signature.proto().reset().name(), "the_reset");
  EXPECT_EQ(result.signature.proto().clock_name(), "the_clock");

  ModuleTestbench tb(result.verilog_text, result.signature, GetSimulator());

  tb.ExpectX("sum");
  tb.Set("a", 0).Set("b", 0);
  tb.AdvanceNCycles(2).ExpectEq("sum", 0);

  tb.Set("a", 100).Set("b", 42);
  tb.AdvanceNCycles(2).ExpectEq("sum", 142);

  tb.Set("the_reset", 1).NextCycle();
  tb.ExpectEq("sum", 0);

  tb.Set("the_reset", 0);
  tb.AdvanceNCycles(2).ExpectEq("sum", 142);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ProcGeneratorTest, RegisteredInputNoReset) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * input_ch,
      package.CreatePortChannel("foo", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * reg_ch,
                           package.CreateRegisterChannel("foo_reg", u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue data = pb.Receive(input_ch);
  pb.Send(reg_ch, data);
  BValue reg_data = pb.Receive(reg_ch);
  pb.Send(output_ch, reg_data);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateModule(proc, options().clock_name("foo_clock")));

  ModuleTestbench tb(result.verilog_text, result.signature, GetSimulator());

  // The block should have a clock, but no reset.
  EXPECT_FALSE(result.signature.proto().has_reset());
  EXPECT_EQ(result.signature.proto().clock_name(), "foo_clock");

  tb.ExpectX("out");
  tb.Set("foo", 42).NextCycle().ExpectEq("out", 42);
  tb.Set("foo", 100).ExpectEq("out", 42).NextCycle().ExpectEq("out", 100);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ProcGeneratorTest, Accumulator) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * input_ch,
      package.CreatePortChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * accum_ch,
      package.CreateRegisterChannel("accum", u32,
                                    /*reset_value=*/Value(UBits(10, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue input = pb.Receive(input_ch);
  BValue accum = pb.Receive(accum_ch);
  BValue next_accum = pb.Add(input, accum);
  pb.Send(accum_ch, next_accum);
  pb.Send(output_ch, accum);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  XLS_ASSERT_OK_AND_ASSIGN(
      auto result, GenerateModule(proc, options().clock_name("clk").reset(
                                            "rst_n", /*asynchronous=*/false,
                                            /*active_low=*/true)));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 result.verilog_text);

  ModuleTestbench tb(result.verilog_text, result.signature, GetSimulator());

  tb.Set("in", 0).Set("rst_n", 0).NextCycle().Set("rst_n", 1);

  tb.ExpectEq("out", 10);
  tb.Set("in", 42).NextCycle().ExpectEq("out", 52);
  tb.Set("in", 100).NextCycle().ExpectEq("out", 152);

  tb.Set("in", 0).Set("rst_n", 0).NextCycle().Set("rst_n", 1);
  tb.ExpectEq("out", 10);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ProcGeneratorTest, SendIfRegister) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * pred_ch,
      package.CreatePortChannel("pred", ChannelOps::kReceiveOnly,
                                package.GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * data_ch,
      package.CreatePortChannel("data", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * reg_ch,
                           package.CreateRegisterChannel("reg", u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * output_ch,
      package.CreatePortChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(),
                          /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue pred = pb.Receive(pred_ch);
  BValue data = pb.Receive(data_ch);
  pb.SendIf(reg_ch, pred, data);
  BValue reg_data = pb.Receive(reg_ch);
  pb.Send(output_ch, reg_data);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  ASSERT_THAT(
      GenerateModule(proc, options().clock_name("clk")).status(),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("SendIf to register channels not supported yet")));
}

TEST_P(ProcGeneratorTest, ProcWithNonNilState) {
  Package package(TestBaseName());
  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value(UBits(42, 32)),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  EXPECT_THAT(
      GenerateModule(proc, options()).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("The proc state must be an empty tuple for codegen")));
}

TEST_P(ProcGeneratorTest, ProcWithStreamingChannel) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      package.CreateStreamingChannel("a", ChannelOps::kReceiveOnly, u32));

  TokenlessProcBuilder pb(TestBaseName(), /*init_value=*/Value(UBits(42, 32)),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue sum = pb.Add(pb.GetStateParam(), pb.Receive(ch));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(sum));

  EXPECT_THAT(
      GenerateModule(proc, options()).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr(
                   "Only register and port channel are supported in codegen")));
}

TEST_P(ProcGeneratorTest, ResetValueWithoutResetSignal) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      package.CreatePortChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * reg_ch, package.CreateRegisterChannel(
                            "reg", u32, /*reset_value=*/Value(UBits(123, 32))));
  TokenlessProcBuilder pb(TestBaseName(),
                          /*init_value=*/Value::Tuple({}),
                          /*token_name=*/"tkn", /*state_name=*/"st", &package);
  BValue in = pb.Receive(in_ch);
  BValue reg_d = pb.Receive(reg_ch);
  pb.Send(reg_ch, pb.Add(in, reg_d));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetStateParam()));

  EXPECT_THAT(
      GenerateModule(proc, options().clock_name("clk")).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr(
              "Must specify a reset signal if registers have a reset value")));
}

INSTANTIATE_TEST_SUITE_P(ProcGeneratorTestInstantiation, ProcGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<ProcGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
