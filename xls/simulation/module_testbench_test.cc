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

#include "xls/simulation/module_testbench.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/source_location.h"
#include "xls/simulation/module_testbench_thread.h"
#include "xls/simulation/testbench_signal_capture.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;
using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr char kTestName[] = "module_testbench_test";
constexpr char kTestdataPath[] = "xls/simulation/testdata";

class ModuleTestbenchTest : public VerilogTestBase {
 protected:
  // Creates and returns a module which simply flops its input twice.
  Module* MakeTwoStageIdentityPipeline(VerilogFile* f, int64_t width = 16) {
    Module* m = f->AddModule("test_module", SourceInfo());
    LogicRef* clk =
        m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());
    LogicRef* in =
        m->AddInput("in", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* out = m->AddOutput("out", f->BitVectorType(width, SourceInfo()),
                                 SourceInfo());

    LogicRef* p0 =
        m->AddReg("p0", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* p1 =
        m->AddReg("p1", f->BitVectorType(width, SourceInfo()), SourceInfo());

    auto af = m->Add<AlwaysFlop>(SourceInfo(), clk);
    af->AddRegister(p0, in, SourceInfo());
    af->AddRegister(p1, p0, SourceInfo());

    m->Add<ContinuousAssignment>(SourceInfo(), out, p1);
    return m;
  }

  // Creates and returns a module which simply flops its input twice.
  Module* MakeTwoStageIdentityPipelineWithReset(VerilogFile* f,
                                                int64_t width = 16) {
    Module* m = f->AddModule("test_module", SourceInfo());
    LogicRef* clk =
        m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());
    LogicRef* reset =
        m->AddInput("reset", f->ScalarType(SourceInfo()), SourceInfo());
    LogicRef* in =
        m->AddInput("in", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* out = m->AddOutput("out", f->BitVectorType(width, SourceInfo()),
                                 SourceInfo());

    LogicRef* p0 =
        m->AddReg("p0", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* p1 =
        m->AddReg("p1", f->BitVectorType(width, SourceInfo()), SourceInfo());

    AlwaysFlop* af = m->Add<AlwaysFlop>(
        SourceInfo(), clk,
        Reset{.signal = reset, .asynchronous = false, .active_low = false});
    Expression* reset_expr = f->Literal(UBits(0, width), SourceInfo());
    af->AddRegister(p0, in, SourceInfo(), reset_expr);
    af->AddRegister(p1, p0, SourceInfo(), reset_expr);

    m->Add<ContinuousAssignment>(SourceInfo(), out, p1);
    return m;
  }
  // Creates and returns a adder module which flops its input twice.
  // The flip-flops use a synchronous active-high reset.
  Module* MakeTwoStageAdderPipeline(VerilogFile* f, int64_t width = 16) {
    Module* m = f->AddModule("test_module", SourceInfo());
    LogicRef* clk =
        m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());
    LogicRef* reset =
        m->AddInput("reset", f->ScalarType(SourceInfo()), SourceInfo());
    LogicRef* operand_0 = m->AddInput(
        "operand_0", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* operand_1 = m->AddInput(
        "operand_1", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* result = m->AddWire(
        "result", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* out = m->AddOutput("out", f->BitVectorType(width, SourceInfo()),
                                 SourceInfo());

    LogicRef* p0 =
        m->AddReg("p0", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* p1 =
        m->AddReg("p1", f->BitVectorType(width, SourceInfo()), SourceInfo());

    auto af = m->Add<AlwaysFlop>(
        SourceInfo(), clk, Reset{reset, /*async*/ false, /*active_low*/ false});
    af->AddRegister(p0, result, SourceInfo());
    af->AddRegister(p1, p0, SourceInfo());

    m->Add<ContinuousAssignment>(SourceInfo(), result,
                                 f->Add(operand_0, operand_1, SourceInfo()));
    m->Add<ContinuousAssignment>(SourceInfo(), out, p1);
    return m;
  }
  // Creates and returns a module which simply prints two messages.
  Module* MakeTwoMessageModule(VerilogFile* f) {
    Module* m = f->AddModule("test_module", SourceInfo());
    m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());
    Initial* initial = m->Add<Initial>(SourceInfo());
    initial->statements()->Add<Display>(
        SourceInfo(), std::vector<Expression*>{f->Make<QuotedString>(
                          SourceInfo(), "This is the first message.")});
    initial->statements()->Add<Display>(
        SourceInfo(), std::vector<Expression*>{f->Make<QuotedString>(
                          SourceInfo(), "This is the second message.")});
    return m;
  }
  // Creates a module which concatenates two values together.
  Module* MakeConcatModule(VerilogFile* f, int64_t width = 16) {
    Module* m = f->AddModule("test_module", SourceInfo());
    LogicRef* clk =
        m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo());
    LogicRef* a =
        m->AddInput("a", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* b =
        m->AddInput("b", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* out = m->AddOutput(
        "out", f->BitVectorType(2 * width, SourceInfo()), SourceInfo());

    LogicRef* a_d =
        m->AddReg("a_d", f->BitVectorType(width, SourceInfo()), SourceInfo());
    LogicRef* b_d =
        m->AddReg("b_d", f->BitVectorType(width, SourceInfo()), SourceInfo());

    auto af = m->Add<AlwaysFlop>(SourceInfo(), clk);
    af->AddRegister(a_d, a, SourceInfo());
    af->AddRegister(b_d, b, SourceInfo());

    m->Add<ContinuousAssignment>(SourceInfo(), out,
                                 f->Concat({a_d, b_d}, SourceInfo()));
    return m;
  }
};

TEST_P(ModuleTestbenchTest, TwoStagePipelineZeroThreads) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest, TwoStageAdderPipelineThreeThreadsWithDoneSignal) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageAdderPipeline(&f);

  ResetProto reset_proto;
  reset_proto.set_name("reset");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleTestbench> tb,
                           ModuleTestbench::CreateFromVastModule(
                               m, GetSimulator(), "clk", reset_proto));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * operand_0,
      tb->CreateThread("operand_0 driver", {DutInput{.port_name = "operand_0",
                                                     .initial_value = IsX()}}));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * operand_1,
      tb->CreateThread("operand_1 driver", {DutInput{.port_name = "operand_1",
                                                     .initial_value = IsX()}}));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * result,
      tb->CreateThread("result capture", /*dut_inputs=*/{}));
  operand_0->MainBlock().Set("operand_0", 0x21);
  operand_1->MainBlock().Set("operand_1", 0x21);
  operand_0->MainBlock().NextCycle().Set("operand_0", 0x32);
  operand_1->MainBlock().NextCycle().Set("operand_1", 0x32);
  operand_0->MainBlock().NextCycle().Set("operand_0", 0x80);
  operand_1->MainBlock().NextCycle().Set("operand_1", 0x2a);
  operand_0->MainBlock().NextCycle().SetX("operand_0");
  operand_1->MainBlock().NextCycle().SetX("operand_1");

  // Pipeline has two stages, data path is not reset, and inputs are X out of
  // reset.
  result->MainBlock().AtEndOfCycle().ExpectX("out");
  result->MainBlock().AtEndOfCycle().ExpectX("out");
  result->MainBlock().AtEndOfCycle().ExpectEq("out", 0x42);
  result->MainBlock().AtEndOfCycle().ExpectEq("out", 0x64);
  result->MainBlock().AtEndOfCycle().ExpectEq("out", 0xaa);

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 tb->GenerateVerilog());

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest,
       TwoStageAdderPipelineThreeThreadsWithDoneSignalAtOutputOnly) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageAdderPipeline(&f);

  ResetProto reset_proto;
  reset_proto.set_name("reset");
  reset_proto.set_asynchronous(false);
  reset_proto.set_active_low(false);

  XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<ModuleTestbench> tb,
                           ModuleTestbench::CreateFromVastModule(
                               m, GetSimulator(), "clk", reset_proto));

  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * operand_0,
                           tb->CreateThread("operand_0 driver",
                                            {DutInput{.port_name = "operand_0",
                                                      .initial_value = IsX()}},
                                            /*wait_until_done=*/false));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * operand_1,
                           tb->CreateThread("operand_1 driver",
                                            {DutInput{.port_name = "operand_1",
                                                      .initial_value = IsX()}},
                                            /*wait_until_done=*/false));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * result,
      tb->CreateThread("result capture", /*dut_inputs=*/{}));
  operand_0->MainBlock().Set("operand_0", 0x21);
  operand_1->MainBlock().Set("operand_1", 0x21);
  operand_0->MainBlock().NextCycle().Set("operand_0", 0x32);
  operand_1->MainBlock().NextCycle().Set("operand_1", 0x32);
  operand_0->MainBlock().NextCycle().Set("operand_0", 0x80);
  operand_1->MainBlock().NextCycle().Set("operand_1", 0x2a);
  operand_0->MainBlock().NextCycle().SetX("operand_0");
  operand_1->MainBlock().NextCycle().SetX("operand_1");

  // Pipeline has two stages, data path is not reset, and inputs are X out of
  // reset.
  result->MainBlock().AtEndOfCycle().ExpectX("out");
  result->MainBlock().AtEndOfCycle().ExpectX("out");
  result->MainBlock().AtEndOfCycle().ExpectEq("out", 0x42);
  result->MainBlock().AtEndOfCycle().ExpectEq("out", 0x64);
  result->MainBlock().AtEndOfCycle().ExpectEq("out", 0xaa);

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 tb->GenerateVerilog());

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest, TwoStagePipeline) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThread("input driver",
                       /*dut_inputs=*/{DutInput{.port_name = "in",
                                                .initial_value = IsX()}}));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("in", 0xabcd);
  seq.AtEndOfCycle().ExpectX("out");
  seq.Set("in", 0x1122);
  seq.AtEndOfCycle().ExpectX("out");
  seq.AtEndOfCycle().ExpectEq("out", 0xabcd);
  seq.SetX("in");
  seq.AtEndOfCycle().ExpectEq("out", 0x1122);

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest, WhenXAndWhenNotX) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeConcatModule(&f, /*width=*/8);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThread(
          "input driver",
          /*dut_inputs=*/{DutInput{.port_name = "a", .initial_value = IsX()},
                          DutInput{.port_name = "b", .initial_value = IsX()}}));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("a", 0x12);
  seq.Set("b", 0x34);
  seq.AtEndOfCycleWhenNotX("out").ExpectEq("out", 0x1234);
  seq.SetX("a").SetX("b");
  seq.AtEndOfCycleWhenX("out").ExpectX("out");
  seq.Set("b", 0);
  seq.NextCycle();
  // Only half of the bits of the output should be X, but that should still
  // trigger WhenX.
  seq.AtEndOfCycleWhenX("out").ExpectX("out");

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest, TwoStagePipelineWithWideInput) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f, 128);

  Bits input1 = bits_ops::Concat(
      {UBits(0x1234567887654321ULL, 64), UBits(0xababababababababULL, 64)});
  Bits input2 = bits_ops::Concat(
      {UBits(0xffffff000aaaaabbULL, 64), UBits(0x1122334455667788ULL, 64)});

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThread("input driver",
                       /*dut_inputs=*/{DutInput{.port_name = "in",
                                                .initial_value = IsX()}}));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("in", input1);
  seq.NextCycle().Set("in", input2);
  seq.AtEndOfCycle().ExpectX("out");
  seq.AtEndOfCycle().ExpectEq("out", input1);
  seq.SetX("in");
  seq.AtEndOfCycle().ExpectEq("out", input2);

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest, TwoStagePipelineWithX) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  // Drive the pipeline with a valid value, then X, then another valid
  // value.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThread("input driver",
                       /*dut_inputs=*/{DutInput{.port_name = "in",
                                                .initial_value = IsX()}}));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("in", 42);
  seq.NextCycle().SetX("in");
  seq.AtEndOfCycle().ExpectX("out");
  seq.AtEndOfCycle().ExpectEq("out", 42);
  seq.Set("in", 1234);
  seq.AdvanceNCycles(3);
  seq.AtEndOfCycle().ExpectEq("out", 1234);

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest, TwoStageWithExpectationFailure) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThread("input driver",
                       /*dut_inputs=*/{DutInput{.port_name = "in",
                                                .initial_value = IsX()}}));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("in", 42);
  seq.NextCycle().Set("in", 1234);
  seq.AtEndOfCycle().ExpectX("out");
  seq.AtEndOfCycle().ExpectEq("out", 42);
  seq.SetX("in");
  seq.AtEndOfCycle().ExpectEq("out", 7);

  EXPECT_THAT(
      tb->Run(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               ContainsRegex("module_testbench_test.cc@[0-9]+: expected "
                             "output `out`, instance #2, recurrence 0 to have "
                             "value: 7, actual: 1234")));
}

TEST_P(ModuleTestbenchTest, MultipleOutputsWithCapture) {
  VerilogFile f = NewVerilogFile();
  Module* m = f.AddModule("test_module", SourceInfo());
  LogicRef* clk = m->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* x =
      m->AddInput("x", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* y =
      m->AddInput("y", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* out0 =
      m->AddOutput("out0", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* out1 =
      m->AddOutput("out1", f.BitVectorType(8, SourceInfo()), SourceInfo());

  LogicRef* not_x =
      m->AddReg("not_x", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* sum =
      m->AddReg("sum", f.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* sum_plus_1 =
      m->AddReg("sum_plus_1", f.BitVectorType(8, SourceInfo()), SourceInfo());

  // Logic is as follows:
  //
  //  out0 <= ~x        // one-cycle latency.
  //  sum  <= x + y
  //  out1 <= sum + 1  // two-cycle latency.
  auto af = m->Add<AlwaysFlop>(SourceInfo(), clk);
  af->AddRegister(not_x, f.BitwiseNot(x, SourceInfo()), SourceInfo());
  af->AddRegister(sum, f.Add(x, y, SourceInfo()), SourceInfo());
  af->AddRegister(sum_plus_1,
                  f.Add(sum, f.PlainLiteral(1, SourceInfo()), SourceInfo()),
                  SourceInfo());

  m->Add<ContinuousAssignment>(SourceInfo(), out0, not_x);
  m->Add<ContinuousAssignment>(SourceInfo(), out1, sum_plus_1);

  Bits out0_captured;
  Bits out1_captured;

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * tbt,
                           tb->CreateThreadDrivingAllInputs(
                               "input driver", /*initial_value=*/ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("x", 10);
  seq.Set("y", 123);
  seq.NextCycle();
  seq.AtEndOfCycle().Capture("out0", &out0_captured);
  seq.AtEndOfCycle()
      .ExpectEq("out0", 245)
      .Capture("out1", &out1_captured)
      .ExpectEq("out1", 134);
  seq.Set("x", 0);
  seq.NextCycle();
  seq.AtEndOfCycle().ExpectEq("out0", 0xff);
  XLS_ASSERT_OK(tb->Run());

  EXPECT_EQ(out0_captured, UBits(245, 8));
  EXPECT_EQ(out1_captured, UBits(134, 8));
}

TEST_P(ModuleTestbenchTest, TestTimeout) {
  VerilogFile f = NewVerilogFile();
  Module* m = f.AddModule("test_module", SourceInfo());
  m->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* out = m->AddOutput("out", f.ScalarType(SourceInfo()), SourceInfo());
  m->Add<ContinuousAssignment>(SourceInfo(), out,
                               f.PlainLiteral(0, SourceInfo()));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * tbt,
                           tb->CreateThread("main",
                                            /*dut_inputs=*/{}));
  SequentialBlock& seq = tbt->MainBlock();
  seq.WaitForCycleAfter("out");

  EXPECT_THAT(tb->Run(),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("Simulation exceeded maximum length")));
}

TEST_P(ModuleTestbenchTest, TracesFound) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoMessageModule(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * tbt,
                           tb->CreateThread("main",
                                            /*dut_inputs=*/{}));
  SequentialBlock& seq = tbt->MainBlock();
  tbt->ExpectTrace("This is the first message.");
  tbt->ExpectTrace("This is the second message.");
  seq.NextCycle();

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest, TracesNotFound) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoMessageModule(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * tbt,
                           tb->CreateThread("main",
                                            /*dut_inputs=*/{}));
  SequentialBlock& seq = tbt->MainBlock();
  tbt->ExpectTrace("This is the missing message.");
  seq.NextCycle();

  EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kNotFound,
                                  HasSubstr("This is the missing message.")));
}

TEST_P(ModuleTestbenchTest, TracesOutOfOrder) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoMessageModule(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * tbt,
                           tb->CreateThread("main",
                                            /*dut_inputs=*/{}));
  SequentialBlock& seq = tbt->MainBlock();
  tbt->ExpectTrace("This is the second message.");
  tbt->ExpectTrace("This is the first message.");
  seq.NextCycle();

  EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kNotFound,
                                  HasSubstr("This is the first message.")));
}

TEST_P(ModuleTestbenchTest, AssertTest) {
  if (!UseSystemVerilog()) {
    // Asserts are a SV only feature.
    return;
  }

  VerilogFile f = NewVerilogFile();
  Module* m = f.AddModule("test_module", SourceInfo());
  LogicRef* clk = m->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo());
  LogicRef* a =
      m->AddInput("a", f.BitVectorType(32, SourceInfo()), SourceInfo());
  LogicRef* b =
      m->AddInput("b", f.BitVectorType(32, SourceInfo()), SourceInfo());
  auto is_unknown = [&](Expression* e) {
    return f.Make<SystemFunctionCall>(SourceInfo(), "isunknown",
                                      std::vector<Expression*>({e}));
  };
  m->Add<ConcurrentAssertion>(
      SourceInfo(), f.LessThan(a, b, SourceInfo()),
      f.Make<PosEdge>(SourceInfo(), clk),
      /*disable_iff=*/f.LogicalOr(is_unknown(a), is_unknown(b), SourceInfo()),
      "my_label", "`a` must be less than `b`!");

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<ModuleTestbench> tb,
        ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleTestbenchThread * tbt,
        tb->CreateThreadDrivingAllInputs("input driver",
                                         /*initial_value=*/ZeroOrX::kX));
    SequentialBlock& seq = tbt->MainBlock();
    seq.Set("a", 42);
    seq.Set("b", 100);
    seq.NextCycle();
    seq.Set("a", 200);
    seq.Set("b", 300);
    seq.NextCycle();

    ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                   tb->GenerateVerilog());

    XLS_ASSERT_OK(tb->Run());
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<ModuleTestbench> tb,
        ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleTestbenchThread * tbt,
        tb->CreateThreadDrivingAllInputs("input driver",
                                         /*initial_value=*/ZeroOrX::kX));
    SequentialBlock& seq = tbt->MainBlock();
    seq.Set("a", 100);
    seq.Set("b", 10);
    EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kAborted,
                                    HasSubstr("`a` must be less than `b`")));
  }
}

TEST_P(ModuleTestbenchTest, IdentityPipelineRepeatN) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * input_thread,
      tb->CreateThread(
          "input driver",
          {DutInput{.port_name = "reset", .initial_value = UBits(1, 1)},
           DutInput{.port_name = "in", .initial_value = UBits(0, 16)}}));
  {
    SequentialBlock& seq = input_thread->MainBlock();
    seq.Set("reset", 0).Set("in", 1).NextCycle();
    seq.Set("in", 2).NextCycle();
    seq.Set("in", 3).NextCycle();
    seq.Set("in", 4).NextCycle();
    seq.Set("in", 5).NextCycle();
  }

  std::vector<Bits> output;
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * output_thread,
                           tb->CreateThread("output capture",
                                            /*dut_inputs=*/{}));
  {
    SequentialBlock& seq = output_thread->MainBlock();
    SequentialBlock& loop = seq.Repeat(10);
    loop.AtEndOfCycle().CaptureMultiple("out", &output);
  }

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 tb->GenerateVerilog());

  XLS_ASSERT_OK(tb->Run());

  EXPECT_THAT(output, ElementsAre(UBits(0, 16), UBits(0, 16), UBits(1, 16),
                                  UBits(2, 16), UBits(3, 16), UBits(4, 16),
                                  UBits(5, 16), UBits(5, 16), UBits(5, 16),
                                  UBits(5, 16)));
}

TEST_P(ModuleTestbenchTest, IdentityPipelineRepeatZeroTimes) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * input_thread,
      tb->CreateThread(
          "input driver",
          {DutInput{.port_name = "reset", .initial_value = UBits(1, 1)},
           DutInput{.port_name = "in", .initial_value = UBits(0, 16)}}));
  {
    SequentialBlock& seq = input_thread->MainBlock();
    seq.Set("reset", 0).Set("in", 1).NextCycle();
    seq.Set("in", 2).NextCycle();
    seq.Set("in", 3).NextCycle();
    seq.Set("in", 4).NextCycle();
    seq.Set("in", 5).NextCycle();
  }

  std::vector<Bits> output;
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * output_thread,
                           tb->CreateThread("output capture",
                                            /*dut_inputs=*/{}));
  {
    SequentialBlock& seq = output_thread->MainBlock();
    SequentialBlock& loop = seq.Repeat(0);
    loop.AtEndOfCycle().CaptureMultiple("out", &output);
  }

  XLS_ASSERT_OK(tb->Run());

  EXPECT_TRUE(output.empty());
}

TEST_P(ModuleTestbenchTest, IdentityPipelineRepeatWithExpectations) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * input_thread,
      tb->CreateThreadDrivingAllInputs("input driver",
                                       /*initial_value=*/ZeroOrX::kZero));
  {
    SequentialBlock& seq = input_thread->MainBlock();
    seq.Set("reset", 0).Set("in", 42).NextCycle();
  }

  std::vector<Bits> output;
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * output_thread,
                           tb->CreateThread("output capture",
                                            /*dut_inputs=*/{}));
  {
    SequentialBlock& seq = output_thread->MainBlock();
    // Advance a couple cycles to wait for the initial values to drain.
    seq.AdvanceNCycles(2);
    SequentialBlock& loop = seq.Repeat(10);
    loop.AtEndOfCycle().ExpectEq("out", UBits(42, 16));
  }

  XLS_ASSERT_OK(tb->Run());

  EXPECT_TRUE(output.empty());
}

TEST_P(ModuleTestbenchTest, IdentityPipelineRepeatWithFailedExpectations) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * input_thread,
      tb->CreateThreadDrivingAllInputs("input driver",
                                       /*initial_value=*/ZeroOrX::kZero));
  {
    SequentialBlock& seq = input_thread->MainBlock();
    seq.Set("reset", 0).Set("in", 42).AdvanceNCycles(5);
    seq.Set("in", 123);
  }

  std::vector<Bits> output;
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * output_thread,
                           tb->CreateThread("output capture",
                                            /*dut_inputs=*/{}));
  {
    SequentialBlock& seq = output_thread->MainBlock();
    // Advance a couple cycles to wait for the initial values to drain.
    seq.AdvanceNCycles(2);
    SequentialBlock& loop = seq.Repeat(10);
    loop.AtEndOfCycle().ExpectEq("out", UBits(42, 16));
  }

  EXPECT_THAT(
      tb->Run(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               ContainsRegex("module_testbench_test.cc@[0-9]+: expected "
                             "output `out`, instance #0, recurrence 5 to have "
                             "value: 42, actual: 123")));

  EXPECT_TRUE(output.empty());
}

TEST_P(ModuleTestbenchTest, IdentityPipelineRepeatForever) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * input_thread,
      tb->CreateThread(
          "input driver",
          {DutInput{.port_name = "reset", .initial_value = UBits(1, 1)},
           DutInput{.port_name = "in", .initial_value = UBits(0, 16)}},
          /*wait_until_done=*/false));
  {
    SequentialBlock& seq = input_thread->MainBlock();
    seq.Set("reset", 0).NextCycle();
    SequentialBlock& loop = seq.RepeatForever();
    loop.Set("in", 2).NextCycle();
    loop.Set("in", 3).NextCycle();
  }

  std::vector<Bits> output;
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * output_thread,
                           tb->CreateThread("output capture",
                                            /*dut_inputs=*/{}));
  {
    SequentialBlock& seq = output_thread->MainBlock();
    SequentialBlock& loop = seq.Repeat(10);
    loop.AtEndOfCycle().CaptureMultiple("out", &output);
  }

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 tb->GenerateVerilog());

  XLS_ASSERT_OK(tb->Run());

  EXPECT_THAT(output, ElementsAre(UBits(0, 16), UBits(0, 16), UBits(0, 16),
                                  UBits(2, 16), UBits(3, 16), UBits(2, 16),
                                  UBits(3, 16), UBits(2, 16), UBits(3, 16),
                                  UBits(2, 16)));
}

TEST_P(ModuleTestbenchTest, DuplicateThreadNames) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));

  XLS_ASSERT_OK(tb->CreateThread("same name", {}).status());
  EXPECT_THAT(tb->CreateThread("same name", {}).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Already a thread named `same name`")));
}

TEST_P(ModuleTestbenchTest, InputPortsDrivenByTwoThreads) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));

  XLS_ASSERT_OK(tb->CreateThread("thread1", {DutInput{.port_name = "in",
                                                      .initial_value = IsX()}})
                    .status());
  EXPECT_THAT(
      tb->CreateThread("thread2",
                       {DutInput{.port_name = "in", .initial_value = IsX()}})
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("`in` is already being drive by thread `thread1`; "
                         "cannot also be drive by thread `thread2`")));
}

TEST_P(ModuleTestbenchTest, InvalidInputPortName) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  EXPECT_THAT(
      tb->CreateThread("thread", {DutInput{.port_name = "not_an_input",
                                           .initial_value = IsX()}})
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("`not_an_input` is not a input port on the DUT")));
}

TEST_P(ModuleTestbenchTest, DrivingInvalidInputPort) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", /*initial_value=*/ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  EXPECT_DEATH(seq.Set("not_a_port", 10),
               HasSubstr("'not_a_port' is not a signal that thread `main is "
                         "designated to drive"));
}

INSTANTIATE_TEST_SUITE_P(ModuleTestbenchTestInstantiation, ModuleTestbenchTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<ModuleTestbenchTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
