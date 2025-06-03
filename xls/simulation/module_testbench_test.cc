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
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "xls/codegen/module_signature.pb.h"
#include "xls/codegen/vast/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/bits_ops.h"
#include "xls/ir/source_location.h"
#include "xls/simulation/module_testbench_thread.h"
#include "xls/simulation/testbench_signal_capture.h"
#include "xls/simulation/testbench_stream.h"
#include "xls/simulation/verilog_simulator.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ContainsRegex;
using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr char kTestName[] = "module_testbench_test";
constexpr char kTestdataPath[] = "xls/simulation/testdata";

// Functor for generating sequential values starting an an optional offset for
// use with testbench streams.
class SequentialProducer {
 public:
  SequentialProducer(int64_t width, int64_t input_count,
                     int64_t starting_offset = 0)
      : width_(width),
        input_count_(input_count),
        starting_offset_(starting_offset),
        count_(0) {}

  std::optional<Bits> operator()() const {
    if (count_ >= input_count_) {
      return std::nullopt;
    }
    Bits result = UBits(starting_offset_ + count_, width_);
    ++count_;
    return result;
  }

 private:
  int64_t width_;
  int64_t input_count_;
  int64_t starting_offset_;
  // The testbench stream API takes a absl::FunctionRef which is a const
  // reference so make this field mutable.
  mutable int64_t count_;
};

// Functor for consuming and EXPECT_EQ'ing values read from a testbench stream.
class SequentialConsumer {
 public:
  explicit SequentialConsumer(int64_t starting_offset = 0)
      : starting_offset_(starting_offset), count_(0) {}

  absl::Status operator()(const Bits& bits) const {
    EXPECT_EQ(bits.ToUint64().value(), starting_offset_ + count_);
    ++count_;
    return absl::OkStatus();
  }

 private:
  int64_t starting_offset_;
  // The testbench stream API takes a absl::FunctionRef which is a const
  // reference so make this field mutable.
  mutable int64_t count_;
};

class ModuleTestbenchTest : public VerilogTestBase {
 protected:
  // Creates and returns a module which simply flops its input twice.
  absl::StatusOr<Module*> MakeTwoStageIdentityPipeline(VerilogFile* f,
                                                       int64_t width = 16) {
    Module* m = f->AddModule("test_module", SourceInfo());
    XLS_ASSIGN_OR_RETURN(
        LogicRef * clk,
        m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * in,
        m->AddInput("in", f->BitVectorType(width, SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * out,
        m->AddOutput("out", f->BitVectorType(width, SourceInfo()),
                     SourceInfo()));

    XLS_ASSIGN_OR_RETURN(
        LogicRef * p0,
        m->AddReg("p0", f->BitVectorType(width, SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * p1,
        m->AddReg("p1", f->BitVectorType(width, SourceInfo()), SourceInfo()));

    auto af = m->Add<AlwaysFlop>(SourceInfo(), clk);
    af->AddRegister(p0, in, SourceInfo());
    af->AddRegister(p1, p0, SourceInfo());

    m->Add<ContinuousAssignment>(SourceInfo(), out, p1);
    return m;
  }

  // Creates and returns a module which simply flops its input twice.
  absl::StatusOr<Module*> MakeTwoStageIdentityPipelineWithReset(
      VerilogFile* f, int64_t width = 16) {
    Module* m = f->AddModule("test_module", SourceInfo());
    XLS_ASSIGN_OR_RETURN(
        LogicRef * clk,
        m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * reset,
        m->AddInput("reset", f->ScalarType(SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * in,
        m->AddInput("in", f->BitVectorType(width, SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * out,
        m->AddOutput("out", f->BitVectorType(width, SourceInfo()),
                     SourceInfo()));

    XLS_ASSIGN_OR_RETURN(
        LogicRef * p0,
        m->AddReg("p0", f->BitVectorType(width, SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * p1,
        m->AddReg("p1", f->BitVectorType(width, SourceInfo()), SourceInfo()));

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
  absl::StatusOr<Module*> MakeTwoStageAdderPipeline(VerilogFile* f,
                                                    int64_t width = 16) {
    Module* m = f->AddModule("test_module", SourceInfo());
    XLS_ASSIGN_OR_RETURN(
        LogicRef * clk,
        m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * reset,
        m->AddInput("reset", f->ScalarType(SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * operand_0,
        m->AddInput("operand_0", f->BitVectorType(width, SourceInfo()),
                    SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * operand_1,
        m->AddInput("operand_1", f->BitVectorType(width, SourceInfo()),
                    SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * result,
        m->AddWire("result", f->BitVectorType(width, SourceInfo()),
                   SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * out,
        m->AddOutput("out", f->BitVectorType(width, SourceInfo()),
                     SourceInfo()));

    XLS_ASSIGN_OR_RETURN(
        LogicRef * p0,
        m->AddReg("p0", f->BitVectorType(width, SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * p1,
        m->AddReg("p1", f->BitVectorType(width, SourceInfo()), SourceInfo()));

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
  absl::StatusOr<Module*> MakeTwoMessageModule(VerilogFile* f) {
    Module* m = f->AddModule("test_module", SourceInfo());
    XLS_ASSIGN_OR_RETURN(
        LogicRef * clk,
        m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo()));
    (void)clk;  // unused

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
  absl::StatusOr<Module*> MakeConcatModule(VerilogFile* f, int64_t width = 16) {
    Module* m = f->AddModule("test_module", SourceInfo());
    XLS_ASSIGN_OR_RETURN(
        LogicRef * clk,
        m->AddInput("clk", f->ScalarType(SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * a,
        m->AddInput("a", f->BitVectorType(width, SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * b,
        m->AddInput("b", f->BitVectorType(width, SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * out,
        m->AddOutput("out", f->BitVectorType(2 * width, SourceInfo()),
                     SourceInfo()));

    XLS_ASSIGN_OR_RETURN(
        LogicRef * a_d,
        m->AddReg("a_d", f->BitVectorType(width, SourceInfo()), SourceInfo()));
    XLS_ASSIGN_OR_RETURN(
        LogicRef * b_d,
        m->AddReg("b_d", f->BitVectorType(width, SourceInfo()), SourceInfo()));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest, TwoStageAdderPipelineThreeThreadsWithDoneSignal) {
  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageAdderPipeline(&f));

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

  XLS_ASSERT_OK_AND_ASSIGN(std::string generated, tb->GenerateVerilog());
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 generated);
  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest,
       TwoStageAdderPipelineThreeThreadsWithDoneSignalAtOutputOnly) {
  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageAdderPipeline(&f));

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

  XLS_ASSERT_OK_AND_ASSIGN(std::string generated, tb->GenerateVerilog());
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 generated);

  XLS_ASSERT_OK(tb->Run());
}

TEST_P(ModuleTestbenchTest, TwoStagePipeline) {
  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeConcatModule(&f, /*width=*/8));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f, 128));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f));

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
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * clk,
      m->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo()));
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * x,
      m->AddInput("x", f.BitVectorType(8, SourceInfo()), SourceInfo()));
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * y,
      m->AddInput("y", f.BitVectorType(8, SourceInfo()), SourceInfo()));
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * out0,
      m->AddOutput("out0", f.BitVectorType(8, SourceInfo()), SourceInfo()));
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * out1,
      m->AddOutput("out1", f.BitVectorType(8, SourceInfo()), SourceInfo()));

  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * not_x,
      m->AddReg("not_x", f.BitVectorType(8, SourceInfo()), SourceInfo()));
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * sum,
      m->AddReg("sum", f.BitVectorType(8, SourceInfo()), SourceInfo()));
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * sum_plus_1,
      m->AddReg("sum_plus_1", f.BitVectorType(8, SourceInfo()), SourceInfo()));

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
                               "input driver", /*default_value=*/ZeroOrX::kX));
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
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * clk,
      m->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo()));
  (void)clk;  // unused
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * out,
      m->AddOutput("out", f.ScalarType(SourceInfo()), SourceInfo()));
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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoMessageModule(&f));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoMessageModule(&f));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoMessageModule(&f));

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
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * clk,
      m->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo()));
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * a,
      m->AddInput("a", f.BitVectorType(32, SourceInfo()), SourceInfo()));
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * b,
      m->AddInput("b", f.BitVectorType(32, SourceInfo()), SourceInfo()));
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
                                         /*default_value=*/ZeroOrX::kX));
    SequentialBlock& seq = tbt->MainBlock();
    seq.Set("a", 42);
    seq.Set("b", 100);
    seq.NextCycle();
    seq.Set("a", 200);
    seq.Set("b", 300);
    seq.NextCycle();

    XLS_ASSERT_OK_AND_ASSIGN(std::string generated, tb->GenerateVerilog());
    ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                   generated);

    XLS_ASSERT_OK(tb->Run());
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<ModuleTestbench> tb,
        ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleTestbenchThread * tbt,
        tb->CreateThreadDrivingAllInputs("input driver",
                                         /*default_value=*/ZeroOrX::kX));
    SequentialBlock& seq = tbt->MainBlock();
    seq.Set("a", 100);
    seq.Set("b", 10);
    EXPECT_THAT(tb->Run(), StatusIs(absl::StatusCode::kAborted,
                                    HasSubstr("`a` must be less than `b`")));
  }
}

TEST_P(ModuleTestbenchTest, IdentityPipelineRepeatN) {
  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(
      Module * m, MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16));
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

  XLS_ASSERT_OK_AND_ASSIGN(std::string generated, tb->GenerateVerilog());
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 generated);

  XLS_ASSERT_OK(tb->Run());

  EXPECT_THAT(output, ElementsAre(UBits(0, 16), UBits(0, 16), UBits(1, 16),
                                  UBits(2, 16), UBits(3, 16), UBits(4, 16),
                                  UBits(5, 16), UBits(5, 16), UBits(5, 16),
                                  UBits(5, 16)));
}

TEST_P(ModuleTestbenchTest, IdentityPipelineRepeatZeroTimes) {
  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(
      Module * m, MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16));
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
  XLS_ASSERT_OK_AND_ASSIGN(
      Module * m, MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * input_thread,
      tb->CreateThreadDrivingAllInputs("input driver",
                                       /*default_value=*/ZeroOrX::kZero));
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
  XLS_ASSERT_OK_AND_ASSIGN(
      Module * m, MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * input_thread,
      tb->CreateThreadDrivingAllInputs("input driver",
                                       /*default_value=*/ZeroOrX::kZero));
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
  XLS_ASSERT_OK_AND_ASSIGN(
      Module * m, MakeTwoStageIdentityPipelineWithReset(&f, /*width=*/16));
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

  XLS_ASSERT_OK_AND_ASSIGN(std::string generated, tb->GenerateVerilog());
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 generated);

  XLS_ASSERT_OK(tb->Run());

  EXPECT_THAT(output, ElementsAre(UBits(0, 16), UBits(0, 16), UBits(0, 16),
                                  UBits(2, 16), UBits(3, 16), UBits(2, 16),
                                  UBits(3, 16), UBits(2, 16), UBits(3, 16),
                                  UBits(2, 16)));
}

TEST_P(ModuleTestbenchTest, DuplicateThreadNames) {
  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f));

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
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", /*default_value=*/ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  EXPECT_DEATH(seq.Set("not_a_port", 10),
               HasSubstr("'not_a_port' is not a signal that thread `main is "
                         "designated to drive"));
}

TEST_P(ModuleTestbenchTest, StreamingIo) {
  constexpr int64_t kInputCount = 100000;
  constexpr int64_t kWidth = 32;

  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           MakeTwoStageIdentityPipeline(&f, kWidth));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(
          m, GetSimulator(), "clk", /*reset=*/std::nullopt,
          /*includes=*/{}, /*simulation_cycle_limit=*/kInputCount + 10));

  XLS_ASSERT_OK_AND_ASSIGN(const TestbenchStream* input_stream,
                           tb->CreateInputStream("my_input", kWidth));
  XLS_ASSERT_OK_AND_ASSIGN(const TestbenchStream* output_stream,
                           tb->CreateOutputStream("my_output", kWidth));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * input_thread,
      tb->CreateThreadDrivingAllInputs("input", /*default_value=*/ZeroOrX::kX));
  {
    SequentialBlock& seq = input_thread->MainBlock();
    SequentialBlock& loop = seq.Repeat(kInputCount);
    loop.ReadFromStreamAndSet("in", input_stream).NextCycle();
  }
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * output_thread,
                           tb->CreateThread("output",
                                            /*dut_inputs=*/{}));
  {
    SequentialBlock& seq = output_thread->MainBlock();
    seq.NextCycle().NextCycle();
    SequentialBlock& loop = seq.Repeat(kInputCount);
    loop.AtEndOfCycle().CaptureAndWriteToStream("out", output_stream);
  }

  XLS_ASSERT_OK_AND_ASSIGN(std::string generated, tb->GenerateVerilog());
  ExpectVerilogEqualToGoldenFile(
      GoldenFilePath(kTestName, kTestdataPath), generated,
      /*macro_definitions=*/
      {
          VerilogSimulator::MacroDefinition{input_stream->path_macro_name,
                                            "\"/tmp/my_input\""},
          VerilogSimulator::MacroDefinition{output_stream->path_macro_name,
                                            "\"/tmp/my_output\""},
      });

  XLS_ASSERT_OK(tb->RunWithStreamingIo(
      {{input_stream->name, SequentialProducer(kWidth, kInputCount)}},
      {{output_stream->name, SequentialConsumer()}}));
}

TEST_P(ModuleTestbenchTest, CycleLimit) {
  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f));

  {
    // Default cycle limit.
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
    seq.AdvanceNCycles(kDefaultSimulationCycleLimit);

    EXPECT_THAT(tb->Run(),
                StatusIs(absl::StatusCode::kDeadlineExceeded,
                         HasSubstr("Simulation exceeded maximum length")));
  }

  {
    // Custom cycle limit.
    constexpr int64_t kCycleLimit = 100;
    XLS_ASSERT_OK_AND_ASSIGN(
        std::unique_ptr<ModuleTestbench> tb,
        ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk",
                                              /*reset=*/std::nullopt,
                                              /*includes=*/{}, kCycleLimit));
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleTestbenchThread * tbt,
        tb->CreateThread("input driver",
                         /*dut_inputs=*/{DutInput{.port_name = "in",
                                                  .initial_value = IsX()}}));
    SequentialBlock& seq = tbt->MainBlock();
    seq.Set("in", 42);
    seq.AdvanceNCycles(kCycleLimit);

    EXPECT_THAT(tb->Run(),
                StatusIs(absl::StatusCode::kDeadlineExceeded,
                         HasSubstr("Simulation exceeded maximum length")));
  }
}

TEST_P(ModuleTestbenchTest, StreamingIoWithError) {
  constexpr int64_t kInputCount = 10;

  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f, 32));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(
          m, GetSimulator(), "clk", /*reset=*/std::nullopt,
          /*includes=*/{}, /*simulation_cycle_limit=*/kInputCount + 10));

  XLS_ASSERT_OK_AND_ASSIGN(const TestbenchStream* input_stream,
                           tb->CreateInputStream("my_input", 32));
  XLS_ASSERT_OK_AND_ASSIGN(const TestbenchStream* output_stream,
                           tb->CreateOutputStream("my_output", 32));

  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * input_thread,
      tb->CreateThreadDrivingAllInputs("input", /*default_value=*/ZeroOrX::kX));
  {
    SequentialBlock& seq = input_thread->MainBlock();
    SequentialBlock& loop = seq.Repeat(kInputCount);
    loop.ReadFromStreamAndSet("in", input_stream).NextCycle();
  }
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * output_thread,
                           tb->CreateThread("output",
                                            /*dut_inputs=*/{}));
  {
    SequentialBlock& seq = output_thread->MainBlock();
    seq.NextCycle().NextCycle();
    SequentialBlock& loop = seq.Repeat(kInputCount);
    loop.AtEndOfCycle().CaptureAndWriteToStream("out", output_stream);
  }

  int64_t i = 0;
  auto producer = [&]() -> std::optional<Bits> {
    if (i == kInputCount) {
      return std::nullopt;
    }
    Bits bits = UBits(i, 32);
    ++i;
    return bits;
  };

  int64_t out_i = 0;
  auto consumer = [&](const Bits& bits) -> absl::Status {
    absl::Status status =
        absl::InternalError(absl::StrFormat("My failure %d.", out_i));
    ++out_i;
    return status;
  };

  EXPECT_THAT(
      tb->RunWithStreamingIo({{input_stream->name, producer}},
                             {{output_stream->name, consumer}}),
      StatusIs(absl::StatusCode::kInternal, HasSubstr("My failure 0.")));
}

TEST_P(ModuleTestbenchTest, StreamingIoProducesX) {
  constexpr int64_t kInputCount = 10;

  VerilogFile f = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, MakeTwoStageIdentityPipeline(&f, 32));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(
          m, GetSimulator(), "clk", /*reset=*/std::nullopt,
          /*includes=*/{}, /*simulation_cycle_limit=*/kInputCount + 10));

  XLS_ASSERT_OK(
      tb->CreateThread("input driver",
                       /*dut_inputs=*/{DutInput{.port_name = "in",
                                                .initial_value = IsX()}})
          .status());

  XLS_ASSERT_OK_AND_ASSIGN(const TestbenchStream* output_stream,
                           tb->CreateOutputStream("my_output", 32));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * output_thread,
                           tb->CreateThread("output",
                                            /*dut_inputs=*/{}));
  {
    SequentialBlock& seq = output_thread->MainBlock();
    seq.NextCycle().NextCycle();
    SequentialBlock& loop = seq.Repeat(kInputCount);
    loop.AtEndOfCycle().CaptureAndWriteToStream("out", output_stream);
  }

  auto consumer = [&](const Bits& bits) -> absl::Status {
    ADD_FAILURE() << "The consumer function should not be called because "
                     "all values are X";
    return absl::OkStatus();
  };

  EXPECT_THAT(tb->RunWithStreamingIo({}, {{output_stream->name, consumer}}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Stream `my_output` produced an X value")));
}

TEST_P(ModuleTestbenchTest, StreamingIoMultipleInputOutput) {
  constexpr int64_t kInputCount = 10;
  constexpr int64_t kWidth = 16;
  constexpr int64_t kNumPorts = 3;

  VerilogFile f = NewVerilogFile();

  // Build a module with kNumPorts input and output ports. Each input port
  // passes its value through to the corresponding output port.
  Module* m = f.AddModule("test_module", SourceInfo());
  XLS_ASSERT_OK_AND_ASSIGN(
      LogicRef * clk,
      m->AddInput("clk", f.ScalarType(SourceInfo()), SourceInfo()));
  (void)clk;  // unused
  std::vector<LogicRef*> input_ports;
  std::vector<LogicRef*> output_ports;
  for (int64_t i = 0; i < kNumPorts; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(
        LogicRef * input_port,
        m->AddInput(absl::StrCat("in", i),
                    f.BitVectorType(kWidth, SourceInfo()), SourceInfo()));
    input_ports.push_back(input_port);

    XLS_ASSERT_OK_AND_ASSIGN(
        LogicRef * output_port,
        m->AddOutput(absl::StrCat("out", i),
                     f.BitVectorType(kWidth, SourceInfo()), SourceInfo()));
    output_ports.push_back(output_port);
    m->Add<ContinuousAssignment>(SourceInfo(), output_ports.back(),
                                 input_ports.back());
  }

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));

  std::vector<const TestbenchStream*> input_streams;
  for (int64_t i = 0; i < kNumPorts; ++i) {
    std::string port = absl::StrCat("in", i);
    XLS_ASSERT_OK_AND_ASSIGN(const TestbenchStream* input_stream,
                             tb->CreateInputStream(port, kWidth));
    input_streams.push_back(input_stream);
    XLS_ASSERT_OK_AND_ASSIGN(
        ModuleTestbenchThread * input_thread,
        tb->CreateThread(port, {DutInput{port, UBits(0, kWidth)}}));
    SequentialBlock& seq = input_thread->MainBlock();
    SequentialBlock& loop = seq.Repeat(kInputCount);
    loop.ReadFromStreamAndSet(port, input_stream).NextCycle();
  }

  std::vector<const TestbenchStream*> output_streams;
  for (int64_t i = 0; i < kNumPorts; ++i) {
    std::string port = absl::StrCat("out", i);
    XLS_ASSERT_OK_AND_ASSIGN(const TestbenchStream* output_stream,
                             tb->CreateOutputStream(port, kWidth));
    output_streams.push_back(output_stream);
    XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * output_thread,
                             tb->CreateThread(port, {}));
    SequentialBlock& seq = output_thread->MainBlock();
    SequentialBlock& loop = seq.Repeat(kInputCount);
    loop.AtEndOfCycle().CaptureAndWriteToStream(port, output_stream);
  }

  // Create stream producers and consumers.
  absl::flat_hash_map<std::string, TestbenchStreamThread::Producer>
      producer_map;
  absl::flat_hash_map<std::string, TestbenchStreamThread::Consumer>
      consumer_map;
  // The maps only hold a view of the functors. Store the actual functors here
  // in vectors with stable pointers.
  std::vector<std::unique_ptr<SequentialProducer>> producers;
  std::vector<std::unique_ptr<SequentialConsumer>> consumers;
  for (int64_t i = 0; i < kNumPorts; ++i) {
    // Give each producer/consumer pair a different starting offset for
    // generating/checking sequences.
    int64_t starting_offset = 123 * i;
    producers.push_back(std::make_unique<SequentialProducer>(
        kWidth, kInputCount, starting_offset));
    producer_map.insert({input_streams[i]->name, *producers.back()});
    consumers.push_back(std::make_unique<SequentialConsumer>(starting_offset));
    consumer_map.insert({output_streams[i]->name, *consumers.back()});
  }

  XLS_ASSERT_OK(tb->RunWithStreamingIo(producer_map, consumer_map));
}

INSTANTIATE_TEST_SUITE_P(ModuleTestbenchTestInstantiation, ModuleTestbenchTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<ModuleTestbenchTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
