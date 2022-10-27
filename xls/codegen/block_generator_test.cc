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

#include "xls/codegen/op_override_impls.h"
#include "xls/codegen/signature_generator.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel.pb.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_parser.h"
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
      std::optional<std::string> clock_name = absl::nullopt) {
    CodegenOptions options;
    options.use_system_verilog(UseSystemVerilog());
    if (clock_name.has_value()) {
      options.clock_name(clock_name.value());
    }
    return options;
  }

  // Make and return a block which adds two u32 numbers.
  absl::StatusOr<Block*> MakeSubtractBlock(std::string_view name,
                                           Package* package) {
    Type* u32 = package->GetBitsType(32);
    BlockBuilder bb(name, package);
    BValue a = bb.InputPort("a", u32);
    BValue b = bb.InputPort("b", u32);
    bb.OutputPort("result", bb.Subtract(a, b));
    return bb.Build();
  }

  // Make and return a register block.
  absl::StatusOr<Block*> MakeRegisterBlock(std::string_view name,
                                           std::string_view clock_name,
                                           Package* package) {
    Type* u32 = package->GetBitsType(32);
    BlockBuilder bb(name, package);
    BValue a = bb.InputPort("a", u32);
    BValue reg_a = bb.InsertRegister(name, a);
    bb.OutputPort("result", reg_a);
    XLS_RETURN_IF_ERROR(bb.block()->AddClockPort(clock_name));
    return bb.Build();
  }

  // Make and return a block which instantiates the given block. Given block
  // should take two u32s (`a` and `b`) and return a u32 (`result`).
  absl::StatusOr<Block*> MakeDelegatingBlock(std::string_view name,
                                             Block* sub_block,
                                             Package* package) {
    Type* u32 = package->GetBitsType(32);
    BlockBuilder bb(name, package);
    BValue x = bb.InputPort("x", u32);
    BValue y = bb.InputPort("y", u32);
    XLS_ASSIGN_OR_RETURN(
        xls::Instantiation * instantiation,
        bb.block()->AddBlockInstantiation(
            absl::StrFormat("%s_instantiation", sub_block->name()), sub_block));
    bb.InstantiationInput(instantiation, "a", x);
    bb.InstantiationInput(instantiation, "b", y);
    BValue result = bb.InstantiationOutput(instantiation, "result");
    bb.OutputPort("z", result);
    return bb.Build();
  }
};

TEST_P(BlockGeneratorTest, AandB) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  BlockBuilder bb(TestBaseName(), &package);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);
  bb.OutputPort("sum", bb.And(a, b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, codegen_options()));

  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options(), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb = NewModuleTestbench(verilog, sig);
  ModuleTestbenchThread& tbt = tb.CreateThread();

  tbt.ExpectX("sum");
  // The combinational module doesn't a connected clock, but the clock can still
  // be used to sequence events in time.
  tbt.NextCycle().Set("a", 0).Set("b", 0).ExpectEq("sum", 0);
  tbt.NextCycle().Set("a", 0x11ff).Set("b", 0x77bb).ExpectEq("sum", 0x11bb);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, PipelinedAandB) {
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
      bb.InsertRegister("p1_sum", bb.And(p0_a, p0_b), rst,
                        xls::Reset{.reset_value = Value(UBits(0, 32)),
                                   .asynchronous = false,
                                   .active_low = false});

  bb.OutputPort("sum", p1_sum);
  XLS_ASSERT_OK(bb.block()->AddClockPort("the_clock"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::string verilog,
      GenerateVerilog(block, codegen_options().emit_as_pipeline(true)));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(codegen_options("the_clock"), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb = NewModuleTestbench(verilog, sig);
  ModuleTestbenchThread& tbt = tb.CreateThread();

  tbt.ExpectX("sum");
  tbt.Set("a", 0).Set("b", 0);
  tbt.AdvanceNCycles(2).ExpectEq("sum", 0);

  tbt.Set("a", 0x11ff).Set("b", 0x77bb);
  tbt.AdvanceNCycles(2).ExpectEq("sum", 0x11bb);

  tbt.Set("the_reset", 1).NextCycle();
  tbt.ExpectEq("sum", 0);

  tbt.Set("the_reset", 0).NextCycle();
  tbt.ExpectEq("sum", 0).NextCycle();
  tbt.ExpectEq("sum", 0x11bb);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, PipelinedAandBNoReset) {
  Package package(TestBaseName());

  Type* u32 = package.GetBitsType(32);
  BlockBuilder bb(TestBaseName(), &package);
  BValue a = bb.InputPort("a", u32);
  BValue b = bb.InputPort("b", u32);

  // Pipeline register 0.
  BValue p0_a = bb.InsertRegister("p0_a", a);
  BValue p0_b = bb.InsertRegister("p0_b", b);

  // Pipeline register 1.
  BValue p1_sum = bb.InsertRegister("p1_sum", bb.And(p0_a, p0_b));

  bb.OutputPort("sum", p1_sum);
  XLS_ASSERT_OK(bb.block()->AddClockPort("the_clock"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(codegen_options("the_clock"), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb = NewModuleTestbench(verilog, sig);
  ModuleTestbenchThread& tbt = tb.CreateThread();

  tbt.ExpectX("sum");
  tbt.Set("a", 0).Set("b", 0);
  tbt.AdvanceNCycles(2).ExpectEq("sum", 0);

  tbt.Set("a", 0x11ff).Set("b", 0x77bb);
  tbt.AdvanceNCycles(2).ExpectEq("sum", 0x11bb);

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
                           GenerateVerilog(block, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options("clk"), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb = NewModuleTestbench(verilog, sig);
  ModuleTestbenchThread& tbt = tb.CreateThread();

  tbt.Set("in", 0).Set("rst_n", 0).NextCycle().Set("rst_n", 1);

  tbt.ExpectEq("out", 10);
  tbt.Set("in", 42).NextCycle().ExpectEq("out", 52);
  tbt.Set("in", 100).NextCycle().ExpectEq("out", 152);

  tbt.Set("in", 0).Set("rst_n", 0).NextCycle().Set("rst_n", 1);
  tbt.ExpectEq("out", 10);

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
                    ->MakeNode<RegisterWrite>(SourceInfo(), a.node(),
                                              /*load_enable=*/absl::nullopt,
                                              /*reset=*/absl::nullopt, reg)
                    .status());
  XLS_ASSERT_OK(block->MakeNode<RegisterRead>(SourceInfo(), reg).status());

  EXPECT_THAT(GenerateVerilog(block, codegen_options()).status(),
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
      GenerateVerilog(block, codegen_options()).status(),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Block has active low and active high reset signals")));
}

TEST_P(BlockGeneratorTest, BlockWithAssertNoLabel) {
  Package package(TestBaseName());
  BlockBuilder b(TestBaseName(), &package);
  BValue rst = b.InputPort("my_rst", package.GetBitsType(1));
  BValue a = b.InputPort("a", package.GetBitsType(32));
  BValue a_d = b.InsertRegister(
      "a_d", a, rst,
      xls::Reset{/*reset_value=*/Value(UBits(123, 32)),
                 /*asynchronous=*/false, /*active_low=*/false});
  b.Assert(b.AfterAll({}), b.ULt(a_d, b.Literal(UBits(42, 32))),
           "a is not greater than 42");
  XLS_ASSERT_OK(b.block()->AddClockPort("my_clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  {
    // No format string.
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                             GenerateVerilog(block, codegen_options()));
    if (UseSystemVerilog()) {
      EXPECT_THAT(
          verilog,
          HasSubstr(
              R"(assert #0 ($isunknown(a_d < 32'h0000_002a) || a_d < 32'h0000_002a) else $fatal(0, "a is not greater than 42"))"));
    } else {
      EXPECT_THAT(verilog, Not(HasSubstr("assert")));
    }
  }

  {
    // With format string, no label.
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string verilog,
        GenerateVerilog(
            block,
            codegen_options().SetOpOverride(
                Op::kAssert,
                std::make_unique<OpOverrideAssertion>(
                    R"(`MY_ASSERT({condition}, "{message}", {clk}, {rst}))"))));
    if (UseSystemVerilog()) {
      EXPECT_THAT(
          verilog,
          HasSubstr(
              R"(`MY_ASSERT(a_d < 32'h0000_002a, "a is not greater than 42", my_clk, my_rst))"));
    } else {
      EXPECT_THAT(verilog, Not(HasSubstr("assert")));
    }
  }

  // Format string with label but assert doesn't have label.
  EXPECT_THAT(
      GenerateVerilog(block,
                      codegen_options().SetOpOverride(
                          Op::kAssert, std::make_unique<OpOverrideAssertion>(
                                           R"({label} foobar)"))),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Assert format string has {label} placeholder, "
                         "but assert operation has no label")));

  // Format string with invalid placeholder.
  EXPECT_THAT(
      GenerateVerilog(block,
                      codegen_options().SetOpOverride(
                          Op::kAssert, std::make_unique<OpOverrideAssertion>(
                                           R"({foobar} blargfoobar)"))),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid placeholder {foobar} in format string. "
                         "Valid placeholders: {clk}, {condition}, {label}, "
                         "{message}, {rst}")));
}

TEST_P(BlockGeneratorTest, BlockWithAssertWithLabel) {
  Package package(TestBaseName());
  BlockBuilder b(TestBaseName(), &package);
  BValue a = b.InputPort("a", package.GetBitsType(32));
  b.Assert(b.AfterAll({}), b.ULt(a, b.Literal(UBits(42, 32))),
           "a is not greater than 42", "the_label");
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  {
    // No format string.
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                             GenerateVerilog(block, codegen_options()));
    if (UseSystemVerilog()) {
      EXPECT_THAT(
          verilog,
          HasSubstr(
              R"(assert #0 ($isunknown(a < 32'h0000_002a) || a < 32'h0000_002a) else $fatal(0, "a is not greater than 42"))"));
    } else {
      EXPECT_THAT(verilog, Not(HasSubstr("assert")));
    }
  }

  {
    // With format string.
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string verilog,
        GenerateVerilog(
            block,
            codegen_options().SetOpOverride(
                Op::kAssert,
                std::make_unique<OpOverrideAssertion>(
                    R"({label}: `MY_ASSERT({condition}, "{message}") // {label})"))));
    if (UseSystemVerilog()) {
      EXPECT_THAT(
          verilog,
          HasSubstr(
              R"(the_label: `MY_ASSERT(a < 32'h0000_002a, "a is not greater than 42") // the_label)"));
    } else {
      EXPECT_THAT(verilog, Not(HasSubstr("assert")));
    }
  }

  // Format string with reset but block doesn't have reset.
  EXPECT_THAT(GenerateVerilog(
                  block, codegen_options().SetOpOverride(
                             Op::kAssert, std::make_unique<OpOverrideAssertion>(
                                              R"({rst} foobar)"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Assert format string has {rst} placeholder, "
                                 "but block has no reset signal")));

  // Format string with clock but block doesn't have clock.
  EXPECT_THAT(GenerateVerilog(
                  block, codegen_options().SetOpOverride(
                             Op::kAssert, std::make_unique<OpOverrideAssertion>(
                                              R"({clk} foobar)"))),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Assert format string has {clk} placeholder, "
                                 "but block has no clock signal")));
}

TEST_P(BlockGeneratorTest, BlockWithTrace) {
  Package package(TestBaseName());
  BlockBuilder b(TestBaseName(), &package);
  BValue a = b.InputPort("a", package.GetBitsType(32));
  b.Trace(b.AfterAll({}), b.ULt(a, b.Literal(UBits(42, 32))), {a},
          "a ({}) is not greater than 42");
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  {
    // No format string.
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                             GenerateVerilog(block, codegen_options()));
    EXPECT_THAT(verilog,
                HasSubstr(R"($display("a (%d) is not greater than 42", a)"));
  }
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
                           GenerateVerilog(block, codegen_options()));
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
                           GenerateVerilog(block, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options("clk"), block));
  ModuleTestbench tb = NewModuleTestbench(verilog, sig);
  ModuleTestbenchThread& tbt = tb.CreateThread();

  // Set inputs to zero and disable load-enables.
  tbt.Set("a", 100).Set("b", 200).Set("a_le", 0).Set("b_le", 0).Set("rst", 1);
  tbt.NextCycle();
  tbt.Set("rst", 0);
  tbt.NextCycle();

  // Outputs should be at the reset value.
  tbt.ExpectEq("a_out", 42).ExpectEq("b_out", 43);

  // Outputs should remain at reset values after clocking because load enables
  // are unasserted.
  tbt.NextCycle();
  tbt.ExpectEq("a_out", 42).ExpectEq("b_out", 43);

  // Assert load enable of 'a'. Load enable of 'b' remains unasserted.
  tbt.Set("a_le", 1);
  tbt.NextCycle();
  tbt.ExpectEq("a_out", 100).ExpectEq("b_out", 43);

  // Assert load enable of 'b'. Deassert load enable of 'a' and change a's
  // input. New input of 'a' should not propagate.
  tbt.Set("a", 101).Set("a_le", 0).Set("b_le", 1);
  tbt.NextCycle();
  tbt.ExpectEq("a_out", 100).ExpectEq("b_out", 200);

  // Assert both load enables.
  tbt.Set("b", 201).Set("a_le", 1).Set("b_le", 1);
  tbt.NextCycle();
  tbt.ExpectEq("a_out", 101).ExpectEq("b_out", 201);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, GatedBitsType) {
  Package package(TestBaseName());
  BlockBuilder b(TestBaseName(), &package);
  BValue cond = b.InputPort("cond", package.GetBitsType(1));
  BValue x = b.InputPort("x", package.GetBitsType(32));
  BValue y = b.InputPort("y", package.GetBitsType(32));
  b.Add(b.Gate(cond, x, SourceInfo(), "gated_x"), y);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  {
    // No format string.
    XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                             GenerateVerilog(block, codegen_options()));
    EXPECT_THAT(verilog, HasSubstr(R"(wire [31:0] gated_x;)"));
    EXPECT_THAT(verilog, HasSubstr(R"(assign gated_x = {32{cond}} & x;)"));
  }

  {
    // With format string.
    XLS_ASSERT_OK_AND_ASSIGN(
        std::string verilog,
        GenerateVerilog(
            block,
            codegen_options().SetOpOverride(
                Op::kGate,
                std::make_unique<OpOverrideGateAssignment>(
                    R"(my_and {output} [{width}-1:0] = my_and({condition}, {input}))"))));
    EXPECT_THAT(verilog, Not(HasSubstr(R"(wire gated_x [31:0];)")));
    EXPECT_THAT(verilog,
                HasSubstr(R"(my_and gated_x [32-1:0] = my_and(cond, x);)"));
  }
}

TEST_P(BlockGeneratorTest, SmulpWithFormat) {
  Package package(TestBaseName());
  BlockBuilder b(TestBaseName(), &package);
  Type* u32 = package.GetBitsType(32);
  BValue x = b.InputPort("x", u32);
  BValue y = b.InputPort("y", u32);
  BValue x_smulp_y = b.SMulp(x, y, SourceInfo(), "x_smulp_y");
  BValue z = b.InputPort("z", u32);
  BValue z_smulp_z = b.SMulp(z, z, SourceInfo(), "z_smulp_z");
  b.OutputPort("out", b.Tuple({x_smulp_y, z_smulp_z}));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  CodegenOptions options = codegen_options().SetOpOverride(
      Op::kSMulp, std::make_unique<OpOverrideInstantiation>(
                      R"(HardMultp #(
  .lhs_width({input0_width}),
  .rhs_width({input1_width}),
  .output_width({output_width})
) {output}_inst (
  .lhs({input0}),
  .rhs({input1}),
  .do_signed(1'b1),
  .output0({output}[({output_width}>>1)-1:0]),
  .output1({output}[({output_width}>>1)*2-1:({output_width}>>1)])
);)"));

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, options));
  verilog = absl::StrCat("`include \"hardmultp.v\"\n\n", verilog);

  VerilogInclude hardmultp_definition;
  hardmultp_definition.relative_path = "hardmultp.v";
  hardmultp_definition.verilog_text =
      R"(module HardMultp (lhs, rhs, do_signed, output0, output1);
  parameter lhs_width = 32,
    rhs_width = 32,
    output_width = 32;
  input wire [lhs_width-1:0] lhs;
  input wire [rhs_width-1:0] rhs;
  input wire do_signed;
  output wire [output_width-1:0] output0;
  output wire [output_width-1:0] output1;

  assign output0 = 1'b0;
  assign output1 = lhs * rhs;
endmodule
)";

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog, {hardmultp_definition});
}

TEST_P(BlockGeneratorTest, GatedSingleBitType) {
  Package package(TestBaseName());
  BlockBuilder b(TestBaseName(), &package);
  BValue cond = b.InputPort("cond", package.GetBitsType(1));
  BValue x = b.InputPort("x", package.GetBitsType(1));
  b.Gate(cond, x, SourceInfo(), "gated_x");
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, codegen_options()));
  EXPECT_THAT(verilog, HasSubstr(R"(assign gated_x = cond & x;)"));
}

TEST_P(BlockGeneratorTest, GatedTupleType) {
  Package package(TestBaseName());
  BlockBuilder b(TestBaseName(), &package);
  BValue cond = b.InputPort("cond", package.GetBitsType(1));
  BValue x = b.InputPort("x", package.GetTupleType({package.GetBitsType(32),
                                                    package.GetBitsType(8)}));
  b.Gate(cond, x, SourceInfo(), "gated_x");
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, codegen_options()));
  EXPECT_THAT(verilog, HasSubstr(R"(wire [39:0] gated_x;)"));
  EXPECT_THAT(verilog, HasSubstr(R"(assign gated_x = {40{cond}} & x;)"));
}

TEST_P(BlockGeneratorTest, GatedArrayType) {
  Package package(TestBaseName());
  BlockBuilder b(TestBaseName(), &package);
  BValue cond = b.InputPort("cond", package.GetBitsType(1));
  BValue x = b.InputPort("x", package.GetArrayType(7, package.GetBitsType(32)));
  b.Gate(cond, x, SourceInfo(), "gated_x");
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  EXPECT_THAT(GenerateVerilog(block, codegen_options()),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("Gate operation only supported for bits and "
                                 "tuple types, has type: bits[32][7]")));
}

TEST_P(BlockGeneratorTest, InstantiatedBlock) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(Block * sub_block,
                           MakeSubtractBlock("subtractor", &package));

  BlockBuilder bb("my_block", &package);
  XLS_ASSERT_OK_AND_ASSIGN(xls::Instantiation * subtractor,
                           bb.block()->AddBlockInstantiation("sub", sub_block));
  BValue x = bb.InputPort("x", u32);
  BValue y = bb.InputPort("y", u32);
  BValue one = bb.Literal(UBits(1, 32));
  bb.InstantiationInput(subtractor, "a", bb.Add(x, one));
  bb.InstantiationInput(subtractor, "b", bb.Subtract(y, one));
  BValue sum = bb.InstantiationOutput(subtractor, "result");
  bb.OutputPort("out", bb.Shll(sum, one));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options(), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb = NewModuleTestbench(verilog, sig);
  ModuleTestbenchThread& tbt = tb.CreateThread();

  tbt.ExpectX("out");
  // The module doesn't a connected clock, but the clock can still
  // be used to sequence events in time.
  // `out` should be: ((x + 1) - (y - 1)) << 1
  tbt.NextCycle().Set("x", 0).Set("y", 0).ExpectEq("out", 4);
  tbt.NextCycle().Set("x", 100).Set("y", 42).ExpectEq("out", 120);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, InstantiatedBlockWithClockButNoClock) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(Block * sub_block,
                           MakeRegisterBlock("my_register", "clk", &package));

  BlockBuilder bb("my_block", &package);
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * my_reg,
      bb.block()->AddBlockInstantiation("my_reg", sub_block));
  BValue x = bb.InputPort("x", u32);
  bb.InstantiationInput(my_reg, "a", x);
  BValue result = bb.InstantiationOutput(my_reg, "result");
  bb.OutputPort("out", result);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(GenerateVerilog(block, codegen_options()).status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("The instantiated block requires a clock but "
                                 "the instantiating block has no clock.")));
}

TEST_P(BlockGeneratorTest, InstantiatedBlockWithClock) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(Block * sub_block,
                           MakeRegisterBlock("my_register", "clk", &package));

  BlockBuilder bb("my_block", &package);
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * my_reg,
      bb.block()->AddBlockInstantiation("my_reg", sub_block));
  BValue x = bb.InputPort("x", u32);
  bb.InstantiationInput(my_reg, "a", x);
  BValue result = bb.InstantiationOutput(my_reg, "result");
  bb.OutputPort("out", result);
  XLS_ASSERT_OK(bb.block()->AddClockPort("the_clock"));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(
      ModuleSignature sig,
      GenerateSignature(codegen_options("the_clock"), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb = NewModuleTestbench(verilog, sig);
  ModuleTestbenchThread& tbt = tb.CreateThread();

  tbt.Set("x", 100).ExpectX("out");
  tbt.NextCycle().Set("x", 101).ExpectEq("out", 100);
  tbt.NextCycle().Set("x", 102).ExpectEq("out", 101);
  tbt.NextCycle().Set("x", 0).ExpectEq("out", 102);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, MultiplyInstantiatedBlock) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(Block * sub_block,
                           MakeSubtractBlock("subtractor", &package));

  BlockBuilder bb("my_block", &package);
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * subtractor0,
      bb.block()->AddBlockInstantiation("sub0", sub_block));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * subtractor1,
      bb.block()->AddBlockInstantiation("sub1", sub_block));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * subtractor2,
      bb.block()->AddBlockInstantiation("sub2", sub_block));
  BValue x = bb.InputPort("x", u32);
  BValue y = bb.InputPort("y", u32);

  bb.InstantiationInput(subtractor0, "a", x);
  bb.InstantiationInput(subtractor0, "b", y);
  BValue x_minus_y = bb.InstantiationOutput(subtractor0, "result");

  bb.InstantiationInput(subtractor1, "a", y);
  bb.InstantiationInput(subtractor1, "b", x);
  BValue y_minus_x = bb.InstantiationOutput(subtractor1, "result");

  bb.InstantiationInput(subtractor2, "a", x);
  bb.InstantiationInput(subtractor2, "b", x);
  BValue x_minus_x = bb.InstantiationOutput(subtractor2, "result");

  bb.OutputPort("x_minus_y", x_minus_y);
  bb.OutputPort("y_minus_x", y_minus_x);
  bb.OutputPort("x_minus_x", x_minus_x);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options(), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb = NewModuleTestbench(verilog, sig);
  ModuleTestbenchThread& tbt = tb.CreateThread();

  tbt.ExpectX("x_minus_y").ExpectX("y_minus_x").ExpectX("x_minus_x");

  // The module doesn't a connected clock, but the clock can still
  // be used to sequence events in time.
  tbt.NextCycle()
      .Set("x", 0)
      .Set("y", 0)
      .ExpectEq("x_minus_y", 0)
      .ExpectEq("y_minus_x", 0)
      .ExpectEq("x_minus_x", 0);

  tbt.NextCycle()
      .Set("x", 0xabcd)
      .Set("y", 0x4242)
      .ExpectEq("x_minus_y", 0x698b)
      .ExpectEq("y_minus_x", 0xffff9675)
      .ExpectEq("x_minus_x", 0);

  XLS_ASSERT_OK(tb.Run());
}

TEST_P(BlockGeneratorTest, DiamondDependencyInstantiations) {
  Package package(TestBaseName());
  Type* u32 = package.GetBitsType(32);

  XLS_ASSERT_OK_AND_ASSIGN(Block * sub_block,
                           MakeSubtractBlock("subtractor", &package));
  XLS_ASSERT_OK_AND_ASSIGN(
      Block * delegator0,
      MakeDelegatingBlock("delegator0", sub_block, &package));
  XLS_ASSERT_OK_AND_ASSIGN(
      Block * delegator1,
      MakeDelegatingBlock("delegator1", sub_block, &package));

  BlockBuilder bb("my_block", &package);
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * instantiation0,
      bb.block()->AddBlockInstantiation("deleg0", delegator0));
  XLS_ASSERT_OK_AND_ASSIGN(
      xls::Instantiation * instantiation1,
      bb.block()->AddBlockInstantiation("deleg1", delegator1));

  BValue j = bb.InputPort("j", u32);
  BValue k = bb.InputPort("k", u32);

  bb.InstantiationInput(instantiation0, "x", j);
  bb.InstantiationInput(instantiation0, "y", k);
  BValue j_minus_k = bb.InstantiationOutput(instantiation0, "z");

  bb.InstantiationInput(instantiation1, "x", k);
  bb.InstantiationInput(instantiation1, "y", j);
  BValue k_minus_j = bb.InstantiationOutput(instantiation1, "z");

  bb.OutputPort("j_minus_k", j_minus_k);
  bb.OutputPort("k_minus_j", k_minus_j);

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::string verilog,
                           GenerateVerilog(block, codegen_options()));
  XLS_ASSERT_OK_AND_ASSIGN(ModuleSignature sig,
                           GenerateSignature(codegen_options(), block));

  ExpectVerilogEqualToGoldenFile(GoldenFilePath(kTestName, kTestdataPath),
                                 verilog);

  ModuleTestbench tb = NewModuleTestbench(verilog, sig);
  ModuleTestbenchThread& tbt = tb.CreateThread();

  tbt.ExpectX("j_minus_k").ExpectX("k_minus_j");

  // The module doesn't a connected clock, but the clock can still
  // be used to sequence events in time.
  tbt.NextCycle()
      .Set("j", 0)
      .Set("k", 0)
      .ExpectEq("j_minus_k", 0)
      .ExpectEq("k_minus_j", 0);

  tbt.NextCycle()
      .Set("j", 0xabcd)
      .Set("k", 0x4242)
      .ExpectEq("j_minus_k", 0x698b)
      .ExpectEq("k_minus_j", 0xffff9675);

  XLS_ASSERT_OK(tb.Run());
}

INSTANTIATE_TEST_SUITE_P(BlockGeneratorTestInstantiation, BlockGeneratorTest,
                         testing::ValuesIn(kDefaultSimulationTargets),
                         ParameterizedTestName<BlockGeneratorTest>);

}  // namespace
}  // namespace verilog
}  // namespace xls
