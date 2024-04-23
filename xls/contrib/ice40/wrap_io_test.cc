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

#include "xls/contrib/ice40/wrap_io.h"

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"
#include "xls/codegen/module_signature.h"
#include "xls/codegen/vast.h"
#include "xls/common/status/matchers.h"
#include "xls/contrib/ice40/ice40_io_strategy.h"
#include "xls/contrib/ice40/null_io_strategy.h"
#include "xls/ir/source_location.h"
#include "xls/simulation/module_testbench.h"
#include "xls/simulation/module_testbench_thread.h"
#include "xls/simulation/verilog_test_base.h"

namespace xls {
namespace verilog {
namespace {

constexpr char kTestName[] = "wrap_io_test";
constexpr char kTestdataPath[] = "xls/contrib/ice40/testdata";

class WrapIOTest : public VerilogTestBase {};

TEST_P(WrapIOTest, Ice40WrapIOIdentity32b) {
  VerilogFile file = NewVerilogFile();

  const std::string kWrappedModuleName = "device_to_wrap";
  Module* wrapped_m = file.AddModule("device_to_wrap", SourceInfo());
  LogicRef* m_input = wrapped_m->AddInput(
      "in", file.BitVectorType(32, SourceInfo()), SourceInfo());
  LogicRef* m_output = wrapped_m->AddOutput(
      "out", file.BitVectorType(32, SourceInfo()), SourceInfo());
  wrapped_m->Add<ContinuousAssignment>(SourceInfo(), m_output, m_input);

  ModuleSignatureBuilder b(kWrappedModuleName);
  b.AddDataInputAsBits(m_input->GetName(), 32);
  b.AddDataOutputAsBits(m_output->GetName(), 32);
  b.WithFixedLatencyInterface(1);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  Ice40IoStrategy io_strategy(&file);
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, WrapIO(kWrappedModuleName, "dtw",
                                              signature, &io_strategy, &file));
  EXPECT_NE(m, nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<VerilogInclude> includes,
                           io_strategy.GetIncludes());

  VLOG(1) << file.Emit();
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit(), /*macro_definitions=*/{},
                                 includes);
}

TEST_P(WrapIOTest, WrapIOIncrement8b) {
  VerilogFile file = NewVerilogFile();

  const std::string kWrappedModuleName = TestBaseName();
  Module* wrapped_m = file.AddModule(kWrappedModuleName, SourceInfo());
  LogicRef* m_input = wrapped_m->AddInput(
      "in", file.BitVectorType(8, SourceInfo()), SourceInfo());
  LogicRef* m_output = wrapped_m->AddOutput(
      "out", file.BitVectorType(8, SourceInfo()), SourceInfo());
  wrapped_m->Add<ContinuousAssignment>(
      SourceInfo(), m_output,
      file.Add(m_input, file.PlainLiteral(1, SourceInfo()), SourceInfo()));

  ModuleSignatureBuilder b(kWrappedModuleName);
  b.AddDataInputAsBits(m_input->GetName(), 8);
  b.AddDataOutputAsBits(m_output->GetName(), 8);
  b.WithFixedLatencyInterface(1);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  NullIOStrategy io_strategy;
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, WrapIO(kWrappedModuleName, "dtw",
                                              signature, &io_strategy, &file));
  EXPECT_NE(m, nullptr);
  VLOG(1) << file.Emit();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleTestbenchThread * tbt,
                           tb->CreateThreadDrivingAllInputs(
                               "main", /*initial_value=*/ZeroOrX::kZero));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("byte_out_ready", 0).Set("byte_in_valid", 1).Set("byte_in", 42);
  seq.WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in_valid", 0).Set("byte_out_ready", 1);
  seq.AtEndOfCycleWhen("byte_out_valid").ExpectEq("byte_out", 43);
  seq.Set("byte_out_ready", 0);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, WrapIONot16b) {
  VerilogFile file = NewVerilogFile();

  const std::string kWrappedModuleName = TestBaseName();
  Module* wrapped_m = file.AddModule(kWrappedModuleName, SourceInfo());
  LogicRef* m_input = wrapped_m->AddInput(
      "in", file.BitVectorType(16, SourceInfo()), SourceInfo());
  LogicRef* m_output = wrapped_m->AddOutput(
      "out", file.BitVectorType(16, SourceInfo()), SourceInfo());
  wrapped_m->Add<ContinuousAssignment>(SourceInfo(), m_output,
                                       file.BitwiseNot(m_input, SourceInfo()));

  ModuleSignatureBuilder b(kWrappedModuleName);
  b.AddDataInputAsBits(m_input->GetName(), 16);
  b.AddDataOutputAsBits(m_output->GetName(), 16);
  b.WithFixedLatencyInterface(1);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  NullIOStrategy io_strategy;
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, WrapIO(kWrappedModuleName, "dtw",
                                              signature, &io_strategy, &file));
  EXPECT_NE(m, nullptr);
  VLOG(1) << file.Emit();

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("byte_out_ready", 0).Set("byte_in_valid", 1);
  seq.Set("byte_in", 0x12).WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in", 0x34).WaitForCycleAfter("byte_in_ready");
  seq.SetX("byte_in").Set("byte_in_valid", 0);

  // The output controller is not exactly ready/valid signaling. Pulse ready a
  // cycle after valid to consume the output value.
  seq.AtEndOfCycleWhen("byte_out_valid").ExpectEq("byte_out", 0xcb);
  seq.Set("byte_out_ready", 1).NextCycle();
  seq.Set("byte_out_ready", 0).NextCycle();
  seq.AtEndOfCycleWhen("byte_out_valid").ExpectEq("byte_out", 0xed);
  seq.Set("byte_out_ready", 0);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, InputShiftRegisterTest) {
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/16, &file));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("clear", 1);

  // Hold write_en for two cycles and drive in the two bytes.
  seq.NextCycle();
  seq.Set("clear", 0).Set("byte_in", 0xab).Set("write_en", 1);

  seq.AtEndOfCycle().ExpectEq("done", 0);
  seq.Set("byte_in", 0xcd);
  seq.NextCycle();
  seq.Set("write_en", 0).SetX("byte_in");
  seq.AtEndOfCycle().ExpectEq("done", 1).ExpectEq("data_out", 0xabcd);

  // Done and data_out should be held until clear is asserted.
  seq.AtEndOfCycle().ExpectEq("done", 1).ExpectEq("data_out", 0xabcd);
  seq.AtEndOfCycle().ExpectEq("done", 1).ExpectEq("data_out", 0xabcd);

  seq.Set("clear", 1);
  seq.NextCycle();
  seq.AtEndOfCycle().ExpectEq("done", 0);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, OneByteShiftRegisterTest) {
  // Verify the input shift register works when it is only a byte wide.
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/8, &file));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("clear", 1);

  seq.NextCycle().Set("clear", 0);
  seq.AtEndOfCycle().ExpectEq("done", 0);
  seq.Set("byte_in", 42).Set("write_en", 1);

  seq.NextCycle();
  seq.AtEndOfCycle().ExpectEq("done", 1).ExpectEq("data_out", 42);
  seq.Set("clear", 1);
  seq.Set("write_en", 0).SetX("byte_in");

  seq.NextCycle();
  seq.Set("clear", 0);
  seq.AtEndOfCycle().ExpectEq("done", 0);
  seq.Set("byte_in", 123).Set("write_en", 1);

  seq.NextCycle().Set("write_en", 0);
  seq.AtEndOfCycle().ExpectEq("done", 1).ExpectEq("data_out", 123);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, ThreeBitShiftRegisterTest) {
  // Verify the input shift register can handle small inputs (3 bits).
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/3, &file));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("clear", 1);

  seq.NextCycle().Set("clear", 0);
  seq.AtEndOfCycle().ExpectEq("done", 0);
  seq.Set("byte_in", 2).Set("write_en", 1);

  seq.NextCycle();
  seq.AtEndOfCycle().ExpectEq("done", 1).ExpectEq("data_out", 2);
  seq.Set("clear", 1);
  seq.NextCycle();

  seq.AtEndOfCycle().ExpectEq("done", 0);
  seq.NextCycle().Set("clear", 0);
  seq.Set("byte_in", 3).Set("write_en", 1);

  seq.NextCycle().Set("write_en", 0);
  seq.AtEndOfCycle().ExpectEq("done", 1).ExpectEq("data_out", 3);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, OddBitWidthShiftRegisterTest) {
  VerilogFile file = NewVerilogFile();
  // 57 bits is 7 bytes with a bit left over. The left over bit (MSb) is written
  // in first.
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/57, &file));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("clear", 1);

  seq.NextCycle().Set("clear", 0);
  seq.AtEndOfCycle().ExpectEq("done", 0);
  seq.Set("byte_in", 0x01).Set("write_en", 1);
  seq.NextCycle().Set("byte_in", 0x23);
  seq.NextCycle().Set("byte_in", 0x45);
  seq.NextCycle().Set("byte_in", 0x67);
  seq.NextCycle().Set("byte_in", 0x89);
  seq.NextCycle().Set("byte_in", 0x0a);
  seq.NextCycle().Set("byte_in", 0xbc);
  seq.NextCycle().Set("byte_in", 0xde);

  seq.NextCycle().Set("write_en", 0);
  seq.AtEndOfCycle().ExpectEq("done", 1).ExpectEq("data_out",
                                                  0x1234567890abcdeULL);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, InputResetModuleTest) {
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputResetModule(&file));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("rst_n_in", 0);
  seq.AdvanceNCycles(5);
  seq.Set("rst_n_in", 1);
  seq.Set("byte_in", 0);
  seq.Set("byte_in_valid", 0);
  seq.AtEndOfCycle().ExpectEq("rst_n_out", 1).ExpectEq("byte_in_ready", 0);

  // Drive random byte, verify that there is no reset.
  seq.Set("byte_in", 0xab).Set("byte_in_valid", 1);

  seq.AtEndOfCycle().ExpectEq("rst_n_out", 1).ExpectEq("byte_in_ready", 0);
  // Set input to reset character.
  seq.Set("byte_in", IOControlCode::kReset).Set("byte_in_valid", 0);

  // Though the reset character was passed in, byte_in_valid was not asserted so
  // rst_n_out is not asserted. Now assert byte_in_valid.
  seq.AtEndOfCycle().ExpectEq("rst_n_out", 1).ExpectEq("byte_in_ready", 0);
  seq.Set("byte_in_valid", 1);
  seq.NextCycle();

  // Reset and byte_in_ready should be asserted.
  seq.AtEndOfCycle().ExpectEq("rst_n_out", 0).ExpectEq("byte_in_ready", 1);
  seq.Set("byte_in_valid", 0);

  // Next cycle, everything shoud be back to normal.
  seq.AtEndOfCycle().ExpectEq("rst_n_out", 1).ExpectEq("byte_in_ready", 0);

  // Asserting rst_in should assert rst_out.
  seq.NextCycle().Set("rst_n_in", 0);
  seq.AtEndOfCycle().ExpectEq("rst_n_out", 0);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, InputControllerForSimpleComputation) {
  ModuleSignatureBuilder mb("x_plus_y");
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInputAsBits("x", 8);
  mb.AddDataInputAsBits("y", 8);
  mb.AddDataOutputAsBits("sum", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputControllerModule(signature, &file));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("rst_n_in", 0);
  seq.AdvanceNCycles(5);
  seq.Set("rst_n_in", 1).Set("data_out_ready", 0).Set("byte_in_valid", 1);
  seq.Set("byte_in", 0x12).WaitForCycleAfter("byte_in_ready").NextCycle();
  seq.Set("byte_in", 0x34).WaitForCycleAfter("byte_in_ready").NextCycle();
  seq.Set("byte_in_valid", 0).Set("data_out_ready", 1);
  seq.AtEndOfCycleWhen("data_out_valid").ExpectEq("data_out", 0x1234);
  seq.AtEndOfCycle().ExpectEq("data_out_valid", 0);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, InputControllerResetControlCode) {
  // Verify that passing in IOControlCode::kReset asserts rst_n and resets the
  // input shift register.
  ModuleSignatureBuilder mb("x_plus_y");
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInputAsBits("x", 8);
  mb.AddDataInputAsBits("y", 8);
  mb.AddDataOutputAsBits("sum", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputControllerModule(signature, &file));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("rst_n_in", 0);
  seq.AdvanceNCycles(5);
  seq.Set("rst_n_in", 1).Set("data_out_ready", 0).Set("byte_in_valid", 1);
  seq.Set("byte_in", 0x42).WaitForCycleAfter("byte_in_ready").NextCycle();
  seq.Set("byte_in", IOControlCode::kReset);
  seq.NextCycle();
  seq.AtEndOfCycle().ExpectEq("rst_n_out", 0).ExpectEq("byte_in_ready", 1);

  seq.Set("byte_in_valid", 0);

  // Asserting reset should have discarded the previously passed in byte (0x42).
  seq.WaitForCycleAfter("rst_n_out").NextCycle();
  seq.Set("byte_in_valid", 1);
  seq.Set("byte_in", 0x12).WaitForCycleAfter("byte_in_ready").NextCycle();
  seq.Set("byte_in", 0x34).WaitForCycleAfter("byte_in_ready").NextCycle();
  seq.Set("byte_in_valid", 0).Set("data_out_ready", 1);
  seq.AtEndOfCycleWhen("data_out_valid").ExpectEq("data_out", 0x1234);
  seq.AtEndOfCycle().ExpectEq("data_out_valid", 0);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, InputControllerEscapedCharacters) {
  // Verify that the two escape codes work as expected for passing in the reset
  // and escape control code values as data bytes.
  ModuleSignatureBuilder mb("x_plus_y");
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInputAsBits("x", 16);
  mb.AddDataOutputAsBits("sum", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputControllerModule(signature, &file));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("rst_n_in", 0);
  seq.AdvanceNCycles(5);
  seq.Set("rst_n_in", 1).Set("data_out_ready", 0).Set("byte_in_valid", 1);
  seq.Set("byte_in", IOControlCode::kEscape)
      .WaitForCycleAfter("byte_in_ready")
      .NextCycle();
  seq.Set("byte_in", IOEscapeCode::kEscapeByte)
      .WaitForCycleAfter("byte_in_ready")
      .NextCycle();
  seq.Set("byte_in", IOControlCode::kEscape)
      .WaitForCycleAfter("byte_in_ready")
      .NextCycle();
  seq.Set("byte_in", IOEscapeCode::kResetByte)
      .WaitForCycleAfter("byte_in_ready")
      .NextCycle();
  seq.Set("byte_in_valid", 0).Set("data_out_ready", 1);
  seq.AtEndOfCycleWhen("data_out_valid")
      .ExpectEq("data_out",
                (static_cast<uint16_t>(IOControlCode::kEscape) << 8) |
                    IOControlCode::kReset);

  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, InputControllerWideInput) {
  ModuleSignatureBuilder mb("wide_x");
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInputAsBits("x", 64);
  mb.AddDataOutputAsBits("out", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputControllerModule(signature, &file));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("data_out_ready", 0).Set("byte_in_valid", 0);
  seq.Set("rst_n_in", 0);
  seq.AdvanceNCycles(5);
  seq.Set("rst_n_in", 1);

  // Should be able to wait arbitrarly long between passing in input bytes.
  seq.AdvanceNCycles(42);

  seq.Set("byte_in_valid", 1);
  seq.Set("byte_in", 0x12).WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in", 0x34).WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in", 0x56).WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in_valid", 0);

  seq.AdvanceNCycles(123);

  seq.Set("byte_in_valid", 1);
  seq.Set("byte_in", 0x78).WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in", 0x90).WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in", 0xab).WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in", 0xcd).WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in", 0xef).WaitForCycleAfter("byte_in_ready");
  seq.Set("byte_in_valid", 0);

  seq.Set("data_out_ready", 0);
  seq.AtEndOfCycleWhen("data_out_valid")
      .ExpectEq("data_out", 0x1234567890abcdefULL);
  XLS_EXPECT_OK(tb->Run());
}

TEST_P(WrapIOTest, OutputControllerForSimpleComputation) {
  ModuleSignatureBuilder mb(TestBaseName());
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInputAsBits("in", 8);
  mb.AddDataOutputAsBits("out", 32);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           OutputControllerModule(signature, &file));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<ModuleTestbench> tb,
      ModuleTestbench::CreateFromVastModule(m, GetSimulator(), "clk"));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleTestbenchThread * tbt,
      tb->CreateThreadDrivingAllInputs("main", ZeroOrX::kX));
  SequentialBlock& seq = tbt->MainBlock();
  seq.Set("byte_out_ready", 0).Set("data_in_valid", 1);
  seq.Set("rst_n", 0);
  seq.AdvanceNCycles(5);
  seq.Set("rst_n", 1);

  seq.Set("data_in_valid", 1);
  seq.Set("data_in", 0x12345678ULL).WaitForCycleAfter("data_in_ready");
  seq.SetX("data_in").Set("data_in_valid", 0);

  // The output controller is not exactly ready/valid signaling. Pulse ready a
  // cycle after valid to consume the output value.
  seq.AtEndOfCycleWhen("byte_out_valid").ExpectEq("byte_out", 0x78);
  seq.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0);
  seq.AtEndOfCycleWhen("byte_out_valid").ExpectEq("byte_out", 0x56);
  seq.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0);
  seq.AtEndOfCycleWhen("byte_out_valid").ExpectEq("byte_out", 0x34);
  seq.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0);
  seq.AtEndOfCycleWhen("byte_out_valid").ExpectEq("byte_out", 0x12);
  seq.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0);

  XLS_EXPECT_OK(tb->Run());
}

// Iverilog hangs when simulating some of these tests.
// TODO(meheff): Add iverilog to the simulator list.
INSTANTIATE_TEST_SUITE_P(WrapIOTestInstantiation, WrapIOTest,
                         testing::ValuesIn(kVerilogOnlySimulationTargets),
                         ParameterizedTestName<WrapIOTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
