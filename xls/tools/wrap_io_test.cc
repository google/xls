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

#include "xls/tools/wrap_io.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/module_signature.h"
#include "xls/common/logging/logging.h"
#include "xls/common/status/matchers.h"
#include "xls/simulation/module_testbench.h"
#include "xls/simulation/verilog_simulators.h"
#include "xls/simulation/verilog_test_base.h"
#include "xls/tools/ice40_io_strategy.h"
#include "xls/tools/null_io_strategy.h"

namespace xls {
namespace verilog {
namespace {

constexpr char kTestName[] = "wrap_io_test";
constexpr char kTestdataPath[] = "xls/tools/testdata";

class WrapIoTest : public VerilogTestBase {};

TEST_P(WrapIoTest, Ice40WrapIoIdentity32b) {
  VerilogFile file = NewVerilogFile();

  const std::string kWrappedModuleName = "device_to_wrap";
  Module* wrapped_m = file.AddModule("device_to_wrap", SourceInfo());
  LogicRef* m_input = wrapped_m->AddInput(
      "in", file.BitVectorType(32, SourceInfo()), SourceInfo());
  LogicRef* m_output = wrapped_m->AddOutput(
      "out", file.BitVectorType(32, SourceInfo()), SourceInfo());
  wrapped_m->Add<ContinuousAssignment>(SourceInfo(), m_output, m_input);

  ModuleSignatureBuilder b(kWrappedModuleName);
  b.AddDataInput(m_input->GetName(), 32);
  b.AddDataOutput(m_output->GetName(), 32);
  b.WithFixedLatencyInterface(1);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  Ice40IoStrategy io_strategy(&file);
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, WrapIo(kWrappedModuleName, "dtw",
                                              signature, &io_strategy, &file));
  EXPECT_NE(m, nullptr);
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<VerilogInclude> includes,
                           io_strategy.GetIncludes());

  XLS_VLOG(1) << file.Emit();
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit(), includes);
}

TEST_P(WrapIoTest, WrapIoIncrement8b) {
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
  b.AddDataInput(m_input->GetName(), 8);
  b.AddDataOutput(m_output->GetName(), 8);
  b.WithFixedLatencyInterface(1);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  NullIoStrategy io_strategy;
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, WrapIo(kWrappedModuleName, "dtw",
                                              signature, &io_strategy, &file));
  EXPECT_NE(m, nullptr);
  XLS_VLOG(1) << file.Emit();

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.Set("byte_out_ready", 0).Set("byte_in_valid", 1).Set("byte_in", 42);
  tb.WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in_valid", 0).Set("byte_out_ready", 1);
  tb.WaitFor("byte_out_valid").ExpectEq("byte_out", 43);
  tb.NextCycle();
  tb.Set("byte_out_ready", 0);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, WrapIoNot16b) {
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
  b.AddDataInput(m_input->GetName(), 16);
  b.AddDataOutput(m_output->GetName(), 16);
  b.WithFixedLatencyInterface(1);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, b.Build());

  NullIoStrategy io_strategy;
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, WrapIo(kWrappedModuleName, "dtw",
                                              signature, &io_strategy, &file));
  EXPECT_NE(m, nullptr);
  XLS_VLOG(1) << file.Emit();

  ModuleTestbench tb(m, GetSimulator(), "clk");
  tb.Set("byte_out_ready", 0).Set("byte_in_valid", 1);
  tb.Set("byte_in", 0x12).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", 0x34).WaitFor("byte_in_ready").NextCycle();
  tb.SetX("byte_in").Set("byte_in_valid", 0);

  // The output controller is not exactly ready/valid signaling. Pulse ready a
  // cycle after valid to consume the output value.
  tb.WaitFor("byte_out_valid").ExpectEq("byte_out", 0xcb).NextCycle();
  tb.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();
  tb.WaitFor("byte_out_valid").ExpectEq("byte_out", 0xed).NextCycle();
  tb.Set("byte_out_ready", 0);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, InputShiftRegisterTest) {
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/16, &file));
  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("clear", 1);

  // Hold write_en for two cycles and drive in the two bytes.
  tb.NextCycle()
      .ExpectEq("done", 0)
      .Set("clear", 0)
      .Set("byte_in", 0xab)
      .Set("write_en", 1);

  tb.NextCycle().ExpectEq("done", 0).Set("byte_in", 0xcd);

  tb.NextCycle()
      .ExpectEq("done", 1)
      .ExpectEq("data_out", 0xabcd)
      .Set("write_en", 0)
      .SetX("byte_in");

  // Done and data_out should be held until clear is asserted.
  tb.NextCycle().ExpectEq("done", 1).ExpectEq("data_out", 0xabcd);
  tb.NextCycle()
      .ExpectEq("done", 1)
      .ExpectEq("data_out", 0xabcd)
      .Set("clear", 1);

  tb.NextCycle().ExpectEq("done", 0);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, OneByteShiftRegisterTest) {
  // Verify the input shift register works when it is only a byte wide.
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/8, &file));
  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("clear", 1);

  tb.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 42)
      .Set("write_en", 1);

  tb.NextCycle()
      .Set("clear", 1)
      .ExpectEq("done", 1)
      .ExpectEq("data_out", 42)
      .Set("write_en", 0)
      .SetX("byte_in");

  tb.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 123)
      .Set("write_en", 1);

  tb.NextCycle()
      .Set("write_en", 0)
      .ExpectEq("done", 1)
      .ExpectEq("data_out", 123);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, ThreeBitShiftRegisterTest) {
  // Verify the input shift register can handle small inputs (3 bits).
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/3, &file));
  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("clear", 1);

  tb.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 2)
      .Set("write_en", 1);

  tb.NextCycle().Set("clear", 1).ExpectEq("done", 1).ExpectEq("data_out", 2);

  tb.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 3)
      .Set("write_en", 1);

  tb.NextCycle().Set("write_en", 0).ExpectEq("done", 1).ExpectEq("data_out", 3);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, OddBitWidthShiftRegisterTest) {
  VerilogFile file = NewVerilogFile();
  // 57 bits is 7 bytes with a bit left over. The left over bit (MSb) is written
  // in first.
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/57, &file));
  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("clear", 1);

  tb.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 0x01)
      .Set("write_en", 1);
  tb.NextCycle().Set("byte_in", 0x23);
  tb.NextCycle().Set("byte_in", 0x45);
  tb.NextCycle().Set("byte_in", 0x67);
  tb.NextCycle().Set("byte_in", 0x89);
  tb.NextCycle().Set("byte_in", 0x0a);
  tb.NextCycle().Set("byte_in", 0xbc);
  tb.NextCycle().Set("byte_in", 0xde);

  tb.NextCycle().Set("write_en", 0);
  tb.NextCycle().ExpectEq("done", 1).ExpectEq("data_out", 0x1234567890abcdeULL);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, InputResetModuleTest) {
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputResetModule(&file));

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("rst_n_in", 0);
  tb.AdvanceNCycles(5);
  tb.Set("rst_n_in", 1);
  tb.Set("byte_in", 0);
  tb.Set("byte_in_valid", 0);
  tb.NextCycle().ExpectEq("rst_n_out", 1).ExpectEq("byte_in_ready", 0);

  // Drive random byte, verify that there is no reset.
  tb.Set("byte_in", 0xab).Set("byte_in_valid", 1);

  tb.NextCycle()
      .ExpectEq("rst_n_out", 1)
      .ExpectEq("byte_in_ready", 0)
      // Set input to reset character.
      .Set("byte_in", IoControlCode::kReset)
      .Set("byte_in_valid", 0);

  // Though the reset character was passed in, byte_in_valid was not asserted so
  // rst_n_out is not asserted. Now assert byte_in_valid.
  tb.NextCycle()
      .ExpectEq("rst_n_out", 1)
      .ExpectEq("byte_in_ready", 0)
      .Set("byte_in_valid", 1);

  // Reset and byte_in_ready should be asserted.
  tb.NextCycle()
      .ExpectEq("rst_n_out", 0)
      .ExpectEq("byte_in_ready", 1)
      .Set("byte_in_valid", 0);

  // Next cycle, everything shoud be back to normal.
  tb.NextCycle().ExpectEq("rst_n_out", 1).ExpectEq("byte_in_ready", 0);

  // Asserting rst_in should assert rst_out.
  tb.NextCycle().Set("rst_n_in", 0);
  tb.NextCycle().ExpectEq("rst_n_out", 0);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, InputControllerForSimpleComputation) {
  ModuleSignatureBuilder mb("x_plus_y");
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInput("x", 8);
  mb.AddDataInput("y", 8);
  mb.AddDataOutput("sum", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputControllerModule(signature, &file));

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("rst_n_in", 0);
  tb.AdvanceNCycles(5);
  tb.Set("rst_n_in", 1).Set("data_out_ready", 0).Set("byte_in_valid", 1);
  tb.Set("byte_in", 0x12).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", 0x34).WaitFor("byte_in_ready").NextCycle();
  tb.ExpectEq("byte_in_ready", 0);
  tb.SetX("byte_in").Set("byte_in_valid", 0).Set("data_out_ready", 1);
  tb.WaitFor("data_out_valid").ExpectEq("data_out", 0x1234).NextCycle();
  tb.ExpectEq("data_out_valid", 0);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, InputControllerResetControlCode) {
  // Verify that passing in IoControlCode::kReset asserts rst_n and resets the
  // input shift register.
  ModuleSignatureBuilder mb("x_plus_y");
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInput("x", 8);
  mb.AddDataInput("y", 8);
  mb.AddDataOutput("sum", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputControllerModule(signature, &file));

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("rst_n_in", 0);
  tb.AdvanceNCycles(5);
  tb.Set("rst_n_in", 1).Set("data_out_ready", 0).Set("byte_in_valid", 1);
  tb.Set("byte_in", 0x42).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", IoControlCode::kReset).NextCycle();
  tb.ExpectEq("rst_n_out", 0).ExpectEq("byte_in_ready", 1);

  tb.Set("byte_in_valid", 0);

  // Asserting reset should have discarded the previously passed in byte (0x42).
  tb.WaitFor("rst_n_out").NextCycle();
  tb.Set("byte_in_valid", 1);
  tb.Set("byte_in", 0x12).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", 0x34).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in_valid", 0).Set("data_out_ready", 1);
  tb.WaitFor("data_out_valid").ExpectEq("data_out", 0x1234).NextCycle();
  tb.ExpectEq("data_out_valid", 0);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, InputControllerEscapedCharacters) {
  // Verify that the two escape codes work as expected for passing in the reset
  // and escape control code values as data bytes.
  ModuleSignatureBuilder mb("x_plus_y");
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInput("x", 16);
  mb.AddDataOutput("sum", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputControllerModule(signature, &file));

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("rst_n_in", 0);
  tb.AdvanceNCycles(5);
  tb.Set("rst_n_in", 1).Set("data_out_ready", 0).Set("byte_in_valid", 1);
  tb.Set("byte_in", IoControlCode::kEscape)
      .WaitFor("byte_in_ready")
      .NextCycle();
  tb.Set("byte_in", IoEscapeCode::kEscapeByte)
      .WaitFor("byte_in_ready")
      .NextCycle();
  tb.Set("byte_in", IoControlCode::kEscape)
      .WaitFor("byte_in_ready")
      .NextCycle();
  tb.Set("byte_in", IoEscapeCode::kResetByte)
      .WaitFor("byte_in_ready")
      .NextCycle();
  tb.Set("byte_in_valid", 0).Set("data_out_ready", 1);
  tb.WaitFor("data_out_valid")
      .ExpectEq("data_out",
                (static_cast<uint16_t>(IoControlCode::kEscape) << 8) |
                    IoControlCode::kReset)
      .NextCycle();

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, InputControllerWideInput) {
  ModuleSignatureBuilder mb("wide_x");
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInput("x", 64);
  mb.AddDataOutput("out", 8);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputControllerModule(signature, &file));

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("data_out_ready", 0).Set("byte_in_valid", 0);
  tb.Set("rst_n_in", 0);
  tb.AdvanceNCycles(5);
  tb.Set("rst_n_in", 1);

  // Should be able to wait arbitrarly long between passing in input bytes.
  tb.AdvanceNCycles(42);

  tb.Set("byte_in_valid", 1);
  tb.Set("byte_in", 0x12).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", 0x34).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", 0x56).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in_valid", 0);

  tb.AdvanceNCycles(123);

  tb.Set("byte_in_valid", 1);
  tb.Set("byte_in", 0x78).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", 0x90).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", 0xab).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", 0xcd).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in", 0xef).WaitFor("byte_in_ready").NextCycle();
  tb.Set("byte_in_valid", 0);

  tb.Set("data_out_ready", 0);
  tb.WaitFor("data_out_valid").ExpectEq("data_out", 0x1234567890abcdefULL);
  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIoTest, OutputControllerForSimpleComputation) {
  ModuleSignatureBuilder mb(TestBaseName());
  mb.WithClock("clk");
  mb.WithFixedLatencyInterface(42);
  mb.AddDataInput("in", 8);
  mb.AddDataOutput("out", 32);
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature signature, mb.Build());

  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           OutputControllerModule(signature, &file));

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  tb.Set("byte_out_ready", 0).Set("data_in_valid", 1);
  tb.Set("rst_n", 0);
  tb.AdvanceNCycles(5);
  tb.Set("rst_n", 1);

  tb.Set("data_in_valid", 1);
  tb.Set("data_in", 0x12345678ULL).WaitFor("data_in_ready").NextCycle();
  tb.SetX("data_in").Set("data_in_valid", 0);

  // The output controller is not exactly ready/valid signaling. Pulse ready a
  // cycle after valid to consume the output value.
  tb.WaitFor("byte_out_valid").ExpectEq("byte_out", 0x78).NextCycle();
  tb.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();
  tb.WaitFor("byte_out_valid").ExpectEq("byte_out", 0x56).NextCycle();
  tb.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();
  tb.WaitFor("byte_out_valid").ExpectEq("byte_out", 0x34).NextCycle();
  tb.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();
  tb.WaitFor("byte_out_valid").ExpectEq("byte_out", 0x12).NextCycle();
  tb.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();

  XLS_EXPECT_OK(tb.Run());
}

// Iverilog hangs when simulating some of these tests.
// TODO(meheff): Add iverilog to the simulator list.
INSTANTIATE_TEST_SUITE_P(WrapIoTestInstantiation, WrapIoTest,
                         testing::ValuesIn(kVerilogOnlySimulationTargets),
                         ParameterizedTestName<WrapIoTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
