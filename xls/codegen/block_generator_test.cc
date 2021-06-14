// Copyright 2021 The XLS Authors
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

#include "xls/codegen/block_generator.h"

#include "xls/codegen/signature_generator.h"
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

constexpr char kTestName[] = "block_generator_test";
constexpr char kTestdataPath[] = "xls/codegen/testdata";

class BlockGeneratorTest : public VerilogTestBase {
 protected:
  CodegenOptions codegen_options(
      absl::optional<std::string> clock_name = absl::nullopt) {
    CodegenOptions options;
    options.use_system_verilog(UseSystemVerilog());
    if (clock_name.has_value()) {
      options.clock_name(clock_name.value());
    }
    return options;
  }
};

TEST_P(BlockGeneratorTest, APlusB) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  BlockBuilder bb(TestBaseName(), &package);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("sum", bb.Add(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, UseSystemVerilog()));

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options(), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb(verilog, sig, GetSimulator());

  tb.ExpectX("sum");
  // The combinational module doesn't a connected clock, but the clock can still
  // be used to sequence events in time.
  tb.NextCycle().Set("a", 0).Set("b", 0).ExpectEq("sum", 0);
  tb.NextCycle().Set("a", 100).Set("b", 42).ExpectEq("sum", 142);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, PipelinedAPlusB) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  BlockBuilder bb(TestBaseName(), &package);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  BValue rst = bb.InputPort("the_reset", package.GetBitsType(1));

  // Pipeline register 0.
  BValue p0_a = bb.InsertRegister("p0_a", a, rst,
                                  xls::Reset{.reset_value = Value(UBits(0, 32)),
                                             .asynchronous = false,
                                             .active_low = false});
  BValue p0_b = bb.InsertRegister("p0_b", b, rst,
                                  xls::Reset{.reset_value = Value(UBits(0, 32)),
                                             .asynchronous = false,
                                             .active_low = false});

  // Pipeline register 1.
  BValue p1_sum =
      bb.InsertRegister("p1_sum", bb.Add(p0_a, p0_b), rst,
                        xls::Reset{.reset_value = Value(UBits(0, 32)),
                                   .asynchronous = false,
                                   .active_low = false});

  bb.OutputPort("sum", p1_sum);
  XLS_ASSERT_OK(bb.block()->AddClockPort("the_clock"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, UseSystemVerilog()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(codegen_options("the_clock"), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb(verilog, sig, GetSimulator());

  tb.ExpectX("sum");
  tb.Set("a", 0).Set("b", 0);
  tb.AdvanceNCycles(2).ExpectEq("sum", 0);

  tb.Set("a", 100).Set("b", 42);
  tb.AdvanceNCycles(2).ExpectEq("sum", 142);

  tb.Set("the_reset", 1).NextCycle();
  tb.ExpectEq("sum", 0);

  tb.Set("the_reset", 0).NextCycle();
  tb.ExpectEq("sum", 0).NextCycle();
  tb.ExpectEq("sum", 142);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, PipelinedAPlusBNoReset) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  BlockBuilder bb(TestBaseName(), &package);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);

  // Pipeline register 0.
  BValue p0_a = bb.InsertRegister("p0_a", a);
  BValue p0_b = bb.InsertRegister("p0_b", b);

  // Pipeline register 1.
  BValue p1_sum = bb.InsertRegister("p1_sum", bb.Add(p0_a, p0_b));

  bb.OutputPort("sum", p1_sum);
  XLS_ASSERT_OK(bb.block()->AddClockPort("the_clock"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, UseSystemVerilog()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(codegen_options("the_clock"), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb(verilog, sig, GetSimulator());

  tb.ExpectX("sum");
  tb.Set("a", 0).Set("b", 0);
  tb.AdvanceNCycles(2).ExpectEq("sum", 0);

  tb.Set("a", 100).Set("b", 42);
  tb.AdvanceNCycles(2).ExpectEq("sum", 142);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, Accumulator) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  BlockBuilder bb(TestBaseName(), &package);
  BValue in = bb.InputPort("in", u32);
  BValue rst_n = bb.InputPort("rst_n", package.GetBitsType(1));

  XLS_ASSERT_OK_AND_ASSIGN(
      Register * accum_reg,
      bb.block()->AddRegister("accum", u32,
                              xls::Reset{.reset_value = Value(UBits(10, 32)),
                                         .asynchronous = false,
                                         .active_low = true}));
  BValue accum = bb.RegisterRead(accum_reg);
  bb.RegisterWrite(accum_reg, bb.Add(in, accum), /*load_enable=*/absl::nullopt,
                   rst_n);
  bb.OutputPort("out", accum);
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, UseSystemVerilog()));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options("clk"), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb(verilog, sig, GetSimulator());

  tb.Set("in", 0).Set("rst_n", 0).NextCycle().Set("rst_n", 1);

  tb.ExpectEq("out", 10);
  tb.Set("in", 42).NextCycle().ExpectEq("out", 52);
  tb.Set("in", 100).NextCycle().ExpectEq("out", 152);

  tb.Set("in", 0).Set("rst_n", 0).NextCycle().Set("rst_n", 1);
  tb.ExpectEq("out", 10);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, RegisterWithoutClockPort) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  BlockBuilder bb(TestBaseName(), &package);
  BValue a = bb.InputPort("a", u32);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Register * reg,
                           block->AddRegister("reg", a.node()->GetType()));
  XLS_ASSERT_OK(block
                    ->MakeNode<RegisterWrite>(absl::nullopt, a.node(),
                                              /*load_enable=*/absl::nullopt,
                                              /*reset=*/absl::nullopt,
                                              reg->name())
                    .status());
  XLS_ASSERT_OK(
      block->MakeNode<RegisterRead>(absl::nullopt, reg->name()).status());

  EXPECT_THAT(GenerateVerilog(block, UseSystemVerilog()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Block has registers but no clock port")));
}

TEST_P(BlockGeneratorTest, RegisterWithDifferentResetBehavior) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  BlockBuilder bb(TestBaseName(), &package);
  BValue a = bb.InputPort("a", u32);
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue rst = bb.InputPort("the_reset", package.GetBitsType(1));
  BValue a_d = bb.InsertRegister("a_d", a, rst,
                                 xls::Reset{.reset_value = Value(UBits(0, 32)),
                                            .asynchronous = false,
                                            .active_low = true});
  bb.InsertRegister("a_d_d", a_d, rst,
                    xls::Reset{.reset_value = Value(UBits(0, 32)),
                               .asynchronous = false,
                               .active_low = false});
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(
      GenerateVerilog(block, UseSystemVerilog()).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Block has active low and active high reset signals")));
}

TEST_P(BlockGeneratorTest, UnrepresentableType) {
  Package package(TestBaseName());

  BlockBuilder bb(TestBaseName(), &package);
  // An empty tuple is unrepresentable in Verilog.
  bb.Literal(Value::Tuple({}));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(GenerateVerilog(block, UseSystemVerilog()).status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("is unrepresentable type")));
}

TEST_P(BlockGeneratorTest, PortOrderTest) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  BlockBuilder bb(TestBaseName(), &package);
  BValue a = bb.InputPort("a", u32);
  bb.OutputPort("b", a);
  BValue c = bb.InputPort("c", u32);
  bb.OutputPort("d", c);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, UseSystemVerilog()));
  EXPECT_THAT(verilog,
              HasSubstr("input wire [31:0] a,\n  input wire [31:0] c,\n  "
                        "output wire [31:0] b,\n  output wire [31:0] d"));
}

TEST_P(BlockGeneratorTest, LoadEnables) {
  // Construct a block with two parallel data paths: "a" and "b". Each consists
  // of a single register with a load enable. Verify that the two load enables
  // work as expected.
  Package package(TestBaseName());

  Type* u1 = package.GetBitsType(1);
  Type* u32 = package.GetBitsType(32);
  BlockBuilder bb(TestBaseName(), &package);
  BValue a = bb.InputPort("a", u32);
  BValue a_le = bb.InputPort("a_le", u1);
  BValue b = bb.InputPort("b", u32);
  BValue b_le = bb.InputPort("b_le", u1);
  BValue rst = bb.InputPort("rst", u1);

  BValue a_reg =
      bb.InsertRegister("a_reg", a, rst,
                        xls::Reset{.reset_value = Value(UBits(42, 32)),
                                   .asynchronous = false,
                                   .active_low = false},
                        a_le);
  BValue b_reg =
      bb.InsertRegister("b_reg", b, rst,
                        xls::Reset{.reset_value = Value(UBits(43, 32)),
                                   .asynchronous = false,
                                   .active_low = false},
                        b_le);

  bb.OutputPort("a_out", a_reg);
  bb.OutputPort("b_out", b_reg);

  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, UseSystemVerilog()));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options("clk"), block));
  ModuleTestbench tb(verilog, sig, GetSimulator());

  // Set inputs to zero and disable load-enables.
  tb.Set("a", 100).Set("b", 200).Set("a_le", 0).Set("b_le", 0).Set("rst", 1);
  tb.NextCycle();
  tb.Set("rst", 0);
  tb.NextCycle();

  // Outputs should be at the reset value.
  tb.ExpectEq("a_out", 42).ExpectEq("b_out", 43);

  // Outputs should remain at reset values after clocking because load enables
  // are unasserted.
  tb.NextCycle();
  tb.ExpectEq("a_out", 42).ExpectEq("b_out", 43);

  // Assert load enable of 'a'. Load enable of 'b' remains unasserted.
  tb.Set("a_le", 1);
  tb.NextCycle();
  tb.ExpectEq("a_out", 100).ExpectEq("b_out", 43);

  // Assert load enable of 'b'. Deassert load enable of 'a' and change a's
  // input. New input of 'a' should not propagate.
  tb.Set("a", 101).Set("a_le", 0).Set("b_le", 1);
  tb.NextCycle();
  tb.ExpectEq("a_out", 100).ExpectEq("b_out", 200);

  // Assert both load enables.
  tb.Set("b", 201).Set("a_le", 1).Set("b_le", 1);
  tb.NextCycle();
  tb.ExpectEq("a_out", 101).ExpectEq("b_out", 201);

  XLS_ASSERT_OK(tb.Run());
}

INSTANTIATE_TEST_SUITE_P(BlockGeneratorTestInstantiation, BlockGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<BlockGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
