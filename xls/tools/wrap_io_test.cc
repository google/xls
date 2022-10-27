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

  XLS_VLOG(1) << file.Emit();
  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 file.Emit(), includes);
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
  XLS_VLOG(1) << file.Emit();

  ModuleTestbench tb(m, GetSimulator(), "clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("byte_out_ready", 0).Set("byte_in_valid", 1).Set("byte_in", 42);
  tbt.WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in_valid", 0).Set("byte_out_ready", 1);
  tbt.WaitFor("byte_out_valid").ExpectEq("byte_out", 43);
  tbt.NextCycle();
  tbt.Set("byte_out_ready", 0);

  XLS_EXPECT_OK(tb.Run());
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
  XLS_VLOG(1) << file.Emit();

  ModuleTestbench tb(m, GetSimulator(), "clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("byte_out_ready", 0).Set("byte_in_valid", 1);
  tbt.Set("byte_in", 0x12).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", 0x34).WaitFor("byte_in_ready").NextCycle();
  tbt.SetX("byte_in").Set("byte_in_valid", 0);

  // The output controller is not exactly ready/valid signaling. Pulse ready a
  // cycle after valid to consume the output value.
  tbt.WaitFor("byte_out_valid").ExpectEq("byte_out", 0xcb).NextCycle();
  tbt.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();
  tbt.WaitFor("byte_out_valid").ExpectEq("byte_out", 0xed).NextCycle();
  tbt.Set("byte_out_ready", 0);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIOTest, InputShiftRegisterTest) {
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/16, &file));
  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("clear", 1);

  // Hold write_en for two cycles and drive in the two bytes.
  tbt.NextCycle()
      .ExpectEq("done", 0)
      .Set("clear", 0)
      .Set("byte_in", 0xab)
      .Set("write_en", 1);

  tbt.NextCycle().ExpectEq("done", 0).Set("byte_in", 0xcd);

  tbt.NextCycle()
      .ExpectEq("done", 1)
      .ExpectEq("data_out", 0xabcd)
      .Set("write_en", 0)
      .SetX("byte_in");

  // Done and data_out should be held until clear is asserted.
  tbt.NextCycle().ExpectEq("done", 1).ExpectEq("data_out", 0xabcd);
  tbt.NextCycle()
      .ExpectEq("done", 1)
      .ExpectEq("data_out", 0xabcd)
      .Set("clear", 1);

  tbt.NextCycle().ExpectEq("done", 0);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIOTest, OneByteShiftRegisterTest) {
  // Verify the input shift register works when it is only a byte wide.
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/8, &file));
  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("clear", 1);

  tbt.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 42)
      .Set("write_en", 1);

  tbt.NextCycle()
      .Set("clear", 1)
      .ExpectEq("done", 1)
      .ExpectEq("data_out", 42)
      .Set("write_en", 0)
      .SetX("byte_in");

  tbt.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 123)
      .Set("write_en", 1);

  tbt.NextCycle()
      .Set("write_en", 0)
      .ExpectEq("done", 1)
      .ExpectEq("data_out", 123);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIOTest, ThreeBitShiftRegisterTest) {
  // Verify the input shift register can handle small inputs (3 bits).
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/3, &file));
  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("clear", 1);

  tbt.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 2)
      .Set("write_en", 1);

  tbt.NextCycle().Set("clear", 1).ExpectEq("done", 1).ExpectEq("data_out", 2);

  tbt.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 3)
      .Set("write_en", 1);

  tbt.NextCycle()
      .Set("write_en", 0)
      .ExpectEq("done", 1)
      .ExpectEq("data_out", 3);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIOTest, OddBitWidthShiftRegisterTest) {
  VerilogFile file = NewVerilogFile();
  // 57 bits is 7 bytes with a bit left over. The left over bit (MSb) is written
  // in first.
  XLS_ASSERT_OK_AND_ASSIGN(Module * m,
                           InputShiftRegisterModule(/*bit_count=*/57, &file));
  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("clear", 1);

  tbt.NextCycle()
      .Set("clear", 0)
      .ExpectEq("done", 0)
      .Set("byte_in", 0x01)
      .Set("write_en", 1);
  tbt.NextCycle().Set("byte_in", 0x23);
  tbt.NextCycle().Set("byte_in", 0x45);
  tbt.NextCycle().Set("byte_in", 0x67);
  tbt.NextCycle().Set("byte_in", 0x89);
  tbt.NextCycle().Set("byte_in", 0x0a);
  tbt.NextCycle().Set("byte_in", 0xbc);
  tbt.NextCycle().Set("byte_in", 0xde);

  tbt.NextCycle().Set("write_en", 0);
  tbt.NextCycle().ExpectEq("done", 1).ExpectEq("data_out",
                                               0x1234567890abcdeULL);

  XLS_EXPECT_OK(tb.Run());
}

TEST_P(WrapIOTest, InputResetModuleTest) {
  VerilogFile file = NewVerilogFile();
  XLS_ASSERT_OK_AND_ASSIGN(Module * m, InputResetModule(&file));

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("rst_n_in", 0);
  tbt.AdvanceNCycles(5);
  tbt.Set("rst_n_in", 1);
  tbt.Set("byte_in", 0);
  tbt.Set("byte_in_valid", 0);
  tbt.NextCycle().ExpectEq("rst_n_out", 1).ExpectEq("byte_in_ready", 0);

  // Drive random byte, verify that there is no reset.
  tbt.Set("byte_in", 0xab).Set("byte_in_valid", 1);

  tbt.NextCycle()
      .ExpectEq("rst_n_out", 1)
      .ExpectEq("byte_in_ready", 0)
      // Set input to reset character.
      .Set("byte_in", IOControlCode::kReset)
      .Set("byte_in_valid", 0);

  // Though the reset character was passed in, byte_in_valid was not asserted so
  // rst_n_out is not asserted. Now assert byte_in_valid.
  tbt.NextCycle()
      .ExpectEq("rst_n_out", 1)
      .ExpectEq("byte_in_ready", 0)
      .Set("byte_in_valid", 1);

  // Reset and byte_in_ready should be asserted.
  tbt.NextCycle()
      .ExpectEq("rst_n_out", 0)
      .ExpectEq("byte_in_ready", 1)
      .Set("byte_in_valid", 0);

  // Next cycle, everything shoud be back to normal.
  tbt.NextCycle().ExpectEq("rst_n_out", 1).ExpectEq("byte_in_ready", 0);

  // Asserting rst_in should assert rst_out.
  tbt.NextCycle().Set("rst_n_in", 0);
  tbt.NextCycle().ExpectEq("rst_n_out", 0);

  XLS_EXPECT_OK(tb.Run());
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

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("rst_n_in", 0);
  tbt.AdvanceNCycles(5);
  tbt.Set("rst_n_in", 1).Set("data_out_ready", 0).Set("byte_in_valid", 1);
  tbt.Set("byte_in", 0x12).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", 0x34).WaitFor("byte_in_ready").NextCycle();
  tbt.ExpectEq("byte_in_ready", 0);
  tbt.SetX("byte_in").Set("byte_in_valid", 0).Set("data_out_ready", 1);
  tbt.WaitFor("data_out_valid").ExpectEq("data_out", 0x1234).NextCycle();
  tbt.ExpectEq("data_out_valid", 0);

  XLS_EXPECT_OK(tb.Run());
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

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("rst_n_in", 0);
  tbt.AdvanceNCycles(5);
  tbt.Set("rst_n_in", 1).Set("data_out_ready", 0).Set("byte_in_valid", 1);
  tbt.Set("byte_in", 0x42).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", IOControlCode::kReset).NextCycle();
  tbt.ExpectEq("rst_n_out", 0).ExpectEq("byte_in_ready", 1);

  tbt.Set("byte_in_valid", 0);

  // Asserting reset should have discarded the previously passed in byte (0x42).
  tbt.WaitFor("rst_n_out").NextCycle();
  tbt.Set("byte_in_valid", 1);
  tbt.Set("byte_in", 0x12).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", 0x34).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in_valid", 0).Set("data_out_ready", 1);
  tbt.WaitFor("data_out_valid").ExpectEq("data_out", 0x1234).NextCycle();
  tbt.ExpectEq("data_out_valid", 0);

  XLS_EXPECT_OK(tb.Run());
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

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("rst_n_in", 0);
  tbt.AdvanceNCycles(5);
  tbt.Set("rst_n_in", 1).Set("data_out_ready", 0).Set("byte_in_valid", 1);
  tbt.Set("byte_in", IOControlCode::kEscape)
      .WaitFor("byte_in_ready")
      .NextCycle();
  tbt.Set("byte_in", IOEscapeCode::kEscapeByte)
      .WaitFor("byte_in_ready")
      .NextCycle();
  tbt.Set("byte_in", IOControlCode::kEscape)
      .WaitFor("byte_in_ready")
      .NextCycle();
  tbt.Set("byte_in", IOEscapeCode::kResetByte)
      .WaitFor("byte_in_ready")
      .NextCycle();
  tbt.Set("byte_in_valid", 0).Set("data_out_ready", 1);
  tbt.WaitFor("data_out_valid")
      .ExpectEq("data_out",
                (static_cast<uint16_t>(IOControlCode::kEscape) << 8) |
                    IOControlCode::kReset)
      .NextCycle();

  XLS_EXPECT_OK(tb.Run());
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

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("data_out_ready", 0).Set("byte_in_valid", 0);
  tbt.Set("rst_n_in", 0);
  tbt.AdvanceNCycles(5);
  tbt.Set("rst_n_in", 1);

  // Should be able to wait arbitrarly long between passing in input bytes.
  tbt.AdvanceNCycles(42);

  tbt.Set("byte_in_valid", 1);
  tbt.Set("byte_in", 0x12).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", 0x34).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", 0x56).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in_valid", 0);

  tbt.AdvanceNCycles(123);

  tbt.Set("byte_in_valid", 1);
  tbt.Set("byte_in", 0x78).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", 0x90).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", 0xab).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", 0xcd).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in", 0xef).WaitFor("byte_in_ready").NextCycle();
  tbt.Set("byte_in_valid", 0);

  tbt.Set("data_out_ready", 0);
  tbt.WaitFor("data_out_valid").ExpectEq("data_out", 0x1234567890abcdefULL);
  XLS_EXPECT_OK(tb.Run());
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

  ModuleTestbench tb(m, GetSimulator(), /*clk_name=*/"clk");
  ModuleTestbenchThread& tbt = tb.CreateThread();
  tbt.Set("byte_out_ready", 0).Set("data_in_valid", 1);
  tbt.Set("rst_n", 0);
  tbt.AdvanceNCycles(5);
  tbt.Set("rst_n", 1);

  tbt.Set("data_in_valid", 1);
  tbt.Set("data_in", 0x12345678ULL).WaitFor("data_in_ready").NextCycle();
  tbt.SetX("data_in").Set("data_in_valid", 0);

  // The output controller is not exactly ready/valid signaling. Pulse ready a
  // cycle after valid to consume the output value.
  tbt.WaitFor("byte_out_valid").ExpectEq("byte_out", 0x78).NextCycle();
  tbt.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();
  tbt.WaitFor("byte_out_valid").ExpectEq("byte_out", 0x56).NextCycle();
  tbt.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();
  tbt.WaitFor("byte_out_valid").ExpectEq("byte_out", 0x34).NextCycle();
  tbt.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();
  tbt.WaitFor("byte_out_valid").ExpectEq("byte_out", 0x12).NextCycle();
  tbt.Set("byte_out_ready", 1).NextCycle().Set("byte_out_ready", 0).NextCycle();

  XLS_EXPECT_OK(tb.Run());
}

// Iverilog hangs when simulating some of these tests.
// TODO(meheff): Add iverilog to the simulator list.
INSTANTIATE_TEST_SUITE_P(WrapIOTestInstantiation, WrapIOTest,
                         testing::ValuesIn(kVerilogOnlySimulationTargets),
                         ParameterizedTestName<WrapIOTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
