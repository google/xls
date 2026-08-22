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

#include "xls/codegen_v_1_5/merge_registers_pass.h"

#include <memory>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_options.h"
#include "xls/codegen_v_1_5/block_conversion_pass.h"
#include "xls/codegen_v_1_5/block_conversion_wrapper_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/ir/value.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/pass_base.h"

namespace xls::codegen {
namespace {

using ::absl_testing::IsOkAndHolds;

class MergeRegistersPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(
      Package* p,
      verilog::CodegenOptions::RegisterMergeStrategy strategy =
          verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly) {
    verilog::CodegenOptions codegen_options;
    codegen_options.register_merge_strategy(strategy);
    BlockConversionPassOptions options{
        .codegen_options = codegen_options,
    };
    PassResults results;
    BlockConversionContext context;
    return MergeRegistersPass().Run(p, options, &results, context);
  }
};

TEST_F(MergeRegistersPassTest, DisabledByStrategy) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_a, bb.block()->AddRegister("reg_a", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_b, bb.block()->AddRegister("reg_b", p->GetBitsType(32)));
  bb.RegisterWrite(reg_a, in);
  bb.RegisterWrite(reg_b, in);
  bb.OutputPort("out_a", bb.RegisterRead(reg_a));
  bb.OutputPort("out_b", bb.RegisterRead(reg_b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(
      Run(p.get(), verilog::CodegenOptions::RegisterMergeStrategy::kDontMerge),
      IsOkAndHolds(false));
  EXPECT_EQ(block->GetRegisters().size(), 2);
}

TEST_F(MergeRegistersPassTest, MergeSimpleRegisters) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_a, bb.block()->AddRegister("reg_a", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_b, bb.block()->AddRegister("reg_b", p->GetBitsType(32)));
  bb.RegisterWrite(reg_a, in);
  bb.RegisterWrite(reg_b, in);
  bb.OutputPort("out_a", bb.RegisterRead(reg_a));
  bb.OutputPort("out_b", bb.RegisterRead(reg_b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetRegisters().size(), 1);

  // Both output ports should now read from the same single register.
  OutputPort* out_a = block->GetOutputPorts()[0];
  OutputPort* out_b = block->GetOutputPorts()[1];
  EXPECT_EQ(out_a->operand(0), out_b->operand(0));
}

TEST_F(MergeRegistersPassTest, MergeRegistersWithLoadEnableAndReset) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  bb.AddClockPort("clk");
  BValue rst = bb.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false});
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  BValue load_en = bb.InputPort("load_en", p->GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg_a,
                           bb.block()->AddRegister("reg_a", p->GetBitsType(32),
                                                   Value(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg_b,
                           bb.block()->AddRegister("reg_b", p->GetBitsType(32),
                                                   Value(UBits(0, 32))));
  bb.RegisterWrite(reg_a, in, load_en, rst);
  bb.RegisterWrite(reg_b, in, load_en, rst);
  bb.OutputPort("out_a", bb.RegisterRead(reg_a));
  bb.OutputPort("out_b", bb.RegisterRead(reg_b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(block->GetRegisters().size(), 1);
  EXPECT_EQ(block->GetOutputPorts()[0]->operand(0),
            block->GetOutputPorts()[1]->operand(0));
}

TEST_F(MergeRegistersPassTest, DistinctDataInputsNotMerged) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  bb.AddClockPort("clk");
  BValue in_a = bb.InputPort("in_a", p->GetBitsType(32));
  BValue in_b = bb.InputPort("in_b", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_a, bb.block()->AddRegister("reg_a", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_b, bb.block()->AddRegister("reg_b", p->GetBitsType(32)));
  bb.RegisterWrite(reg_a, in_a);
  bb.RegisterWrite(reg_b, in_b);
  bb.OutputPort("out_a", bb.RegisterRead(reg_a));
  bb.OutputPort("out_b", bb.RegisterRead(reg_b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(block->GetRegisters().size(), 2);
}

TEST_F(MergeRegistersPassTest, DistinctLoadEnablesNotMerged) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  bb.AddClockPort("clk");
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  BValue le_a = bb.InputPort("le_a", p->GetBitsType(1));
  BValue le_b = bb.InputPort("le_b", p->GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_a, bb.block()->AddRegister("reg_a", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_b, bb.block()->AddRegister("reg_b", p->GetBitsType(32)));
  bb.RegisterWrite(reg_a, in, le_a);
  bb.RegisterWrite(reg_b, in, le_b);
  bb.OutputPort("out_a", bb.RegisterRead(reg_a));
  bb.OutputPort("out_b", bb.RegisterRead(reg_b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(block->GetRegisters().size(), 2);
}

TEST_F(MergeRegistersPassTest, DistinctResetValuesNotMerged) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  bb.AddClockPort("clk");
  BValue rst = bb.ResetPort(
      "rst", ResetBehavior{.asynchronous = false, .active_low = false});
  BValue in = bb.InputPort("in", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg_a,
                           bb.block()->AddRegister("reg_a", p->GetBitsType(32),
                                                   Value(UBits(0, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Register * reg_b,
                           bb.block()->AddRegister("reg_b", p->GetBitsType(32),
                                                   Value(UBits(1, 32))));
  bb.RegisterWrite(reg_a, in, /*load_enable=*/std::nullopt, rst);
  bb.RegisterWrite(reg_b, in, /*load_enable=*/std::nullopt, rst);
  bb.OutputPort("out_a", bb.RegisterRead(reg_a));
  bb.OutputPort("out_b", bb.RegisterRead(reg_b));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(block->GetRegisters().size(), 2);
}

TEST_F(MergeRegistersPassTest,
       CascadingMultiStageRegisterMergeInFixedPointPipeline) {
  auto p = CreatePackage();
  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue in = bb.InputPort("in", p->GetBitsType(32));

  // Stage 1 registers (initially identical D-inputs)
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_a1, bb.block()->AddRegister("reg_a1", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_b1, bb.block()->AddRegister("reg_b1", p->GetBitsType(32)));
  bb.RegisterWrite(reg_a1, in);
  bb.RegisterWrite(reg_b1, in);
  BValue read_a1 = bb.RegisterRead(reg_a1);
  BValue read_b1 = bb.RegisterRead(reg_b1);

  // Compute downstream of Stage 1
  BValue add_a = bb.Add(read_a1, bb.Literal(UBits(42, 32)));
  BValue add_b = bb.Add(read_b1, bb.Literal(UBits(42, 32)));

  // Stage 2 registers (initially distinct D-inputs: add_a vs add_b)
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_a2, bb.block()->AddRegister("reg_a2", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Register * reg_b2, bb.block()->AddRegister("reg_b2", p->GetBitsType(32)));
  bb.RegisterWrite(reg_a2, add_a);
  bb.RegisterWrite(reg_b2, add_b);
  bb.OutputPort("out_a", bb.RegisterRead(reg_a2));
  bb.OutputPort("out_b", bb.RegisterRead(reg_b2));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_EQ(block->GetRegisters().size(), 4);

  verilog::CodegenOptions codegen_options;
  codegen_options.register_merge_strategy(
      verilog::CodegenOptions::RegisterMergeStrategy::kIdentityOnly);
  BlockConversionPassOptions options{
      .codegen_options = codegen_options,
  };
  PassResults results;
  BlockConversionContext context;

  BlockConversionFixedPointCompoundPass pipeline(
      "test_pipeline",
      "Simple fixed-point pipeline: [merge_registers dce cse dce]");
  pipeline.Add<MergeRegistersPass>();
  pipeline.Add<BlockConversionWrapperPass>(
      std::make_unique<DeadCodeEliminationPass>());
  pipeline.Add<BlockConversionWrapperPass>(std::make_unique<CsePass>());
  pipeline.Add<BlockConversionWrapperPass>(
      std::make_unique<DeadCodeEliminationPass>());

  EXPECT_THAT(pipeline.Run(p.get(), options, &results, context),
              IsOkAndHolds(true));

  // Both Stage 1 (a1/b1) and Stage 2 (a2/b2) should have merged down to 2 total
  // registers.
  EXPECT_EQ(block->GetRegisters().size(), 2);
  EXPECT_EQ(block->GetOutputPorts()[0]->operand(0),
            block->GetOutputPorts()[1]->operand(0));
}

}  // namespace
}  // namespace xls::codegen
