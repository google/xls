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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/simulation/verilog_simulators.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using status_testing::StatusIs;
using ::testing::ContainsRegex;
using ::testing::HasSubstr;

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
};

TEST_P(ModuleTestbenchTest, TwoStagePipeline) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.Set("in", 0xabcd).ExpectX("out");
  tb.NextCycle().Set("in", 0x1122).ExpectX("out");
  tb.NextCycle().ExpectEq("out", 0xabcd).SetX("in");
  tb.NextCycle().ExpectEq("out", 0x1122);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ModuleTestbenchTest, WaitForXAndNotX) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.Set("in", 0xabcd);
  tb.WaitForNotX("out").ExpectEq("out", 0xabcd);
  tb.SetX("in");
  tb.WaitForX("out").ExpectX("out");

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ModuleTestbenchTest, TwoStagePipelineWithWideInput) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f, 128);

  Bits input1 = bits_ops::Concat(
      {UBits(0x1234567887654321ULL, 64), UBits(0xababababababababULL, 64)});
  Bits input2 = bits_ops::Concat(
      {UBits(0xffffff000aaaaabbULL, 64), UBits(0x1122334455667788ULL, 64)});

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.Set("in", input1);
  tb.NextCycle().Set("in", input2);
  tb.NextCycle().ExpectEq("out", input1).SetX("in");
  tb.NextCycle().ExpectEq("out", input2);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ModuleTestbenchTest, TwoStagePipelineWithX) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  // Drive the pipeline with a valid value, then X, then another valid
  // value.
  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.Set("in", 42);
  tb.NextCycle().SetX("in");
  tb.NextCycle().ExpectEq("out", 42).Set("in", 1234);
  tb.AdvanceNCycles(2).ExpectEq("out", 1234);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ModuleTestbenchTest, TwoStageWithExpectationFailure) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoStageIdentityPipeline(&f);

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.Set("in", 42);
  tb.NextCycle().Set("in", 1234);
  tb.NextCycle().ExpectEq("out", 42).SetX("in");
  tb.NextCycle().ExpectEq("out", 7);

  EXPECT_THAT(
      tb.Run(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               ContainsRegex("module_testbench_test.cc@[0-9]+: expected "
                             "output 'out', instance #1 to have "
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
  //  out0 <= x        // one-cycle latency.
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

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.Set("x", 10);
  tb.Set("y", 123);
  tb.NextCycle().Capture("out0", &out0_captured);
  tb.NextCycle()
      .ExpectEq("out0", 245)
      .Capture("out1", &out1_captured)
      .ExpectEq("out1", 134)
      .Set("x", 0);
  tb.NextCycle().ExpectEq("out0", 0xff);
  XLS_ASSERT_OK(tb.Run());

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

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.WaitFor("out");

  EXPECT_THAT(tb.Run(),
              StatusIs(absl::StatusCode::kDeadlineExceeded,
                       HasSubstr("Simulation exceeded maximum length")));
}

TEST_P(ModuleTestbenchTest, TracesFound) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoMessageModule(&f);

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.ExpectTrace("This is the first message.");
  tb.ExpectTrace("This is the second message.");
  tb.NextCycle();

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(ModuleTestbenchTest, TracesNotFound) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoMessageModule(&f);

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.ExpectTrace("This is the missing message.");
  tb.NextCycle();

  EXPECT_THAT(tb.Run(), StatusIs(absl::StatusCode::kNotFound,
                                 HasSubstr("This is the missing message.")));
}

TEST_P(ModuleTestbenchTest, TracesOutOfOrder) {
  VerilogFile f = NewVerilogFile();
  Module* m = MakeTwoMessageModule(&f);

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.ExpectTrace("This is the second message.");
  tb.ExpectTrace("This is the first message.");
  tb.NextCycle();

  EXPECT_THAT(tb.Run(), StatusIs(absl::StatusCode::kNotFound,
                                 HasSubstr("This is the first message.")));
}

INSTANTIATE_TEST_SUITE_P(ModuleTestbenchTestInstantiation, ModuleTestbenchTest,
                         testing::ValuesIn(kVerilogOnlySimulationTargets),
                         ParameterizedTestName<ModuleTestbenchTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
