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

#include "xls/codegen/register_legalization_pass.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/register.h"
#include "xls/passes/pass_base.h"

namespace xls::verilog {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::testing::UnorderedElementsAre;

class RegisterLegalizationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Package* p, CodegenContext& context) {
    PassResults results;
    return RegisterLegalizationPass().Run(p, CodegenPassOptions(), &results,
                                          context);
  }
  absl::StatusOr<bool> Run(Block* block) {
    CodegenContext context(block);
    return Run(block->package(), context);
  }
};

TEST_F(RegisterLegalizationPassTest, RegistersOfDifferentSizes) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(0));
  BValue c = bb.InputPort("c", p->GetTupleType({}));
  BValue d = bb.InputPort("d", p->GetTupleType({p->GetTupleType({})}));
  BValue e = bb.InputPort("e", p->GetTupleType({p->GetBitsType(32)}));
  bb.InsertRegister("a_reg", a);
  bb.InsertRegister("b_reg", b);
  bb.InsertRegister("c_reg", c);
  bb.InsertRegister("d_reg", d);
  bb.InsertRegister("e_reg", e);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  // Registers b, c, and d should be removed.
  EXPECT_EQ(block->GetRegisters().size(), 5);
  EXPECT_THAT(Run(block), IsOkAndHolds(true));
  EXPECT_EQ(block->GetRegisters().size(), 2);

  EXPECT_THAT(block->GetRegisters(),
              UnorderedElementsAre(block->GetRegister("a_reg").value(),
                                   block->GetRegister("e_reg").value()));

  // Pass should be idempotent.
  EXPECT_THAT(Run(block), IsOkAndHolds(false));
  EXPECT_EQ(block->GetRegisters().size(), 2);
}

namespace m {
MATCHER_P3(PipelineRegister, reg, write, read, "") {
  const ::xls::verilog::PipelineRegister& pr = arg;
  return testing::ExplainMatchResult(reg, pr.reg, result_listener) &&
         testing::ExplainMatchResult(write, pr.reg_write, result_listener) &&
         testing::ExplainMatchResult(read, pr.reg_read, result_listener);
}
}  // namespace m

TEST_F(RegisterLegalizationPassTest, KeepsUnitListsValid) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  auto read32 = bb.InsertRegister("reg32", bb.Literal(UBits(32, 32)));
  auto read0 = bb.InsertRegister("reg0", bb.Tuple({}));
  XLS_ASSERT_OK_AND_ASSIGN(Block * blk, bb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(auto* reg32, blk->GetRegister("reg32"));
  XLS_ASSERT_OK_AND_ASSIGN(auto* write32, blk->GetUniqueRegisterWrite(reg32));
  XLS_ASSERT_OK_AND_ASSIGN(auto* reg0, blk->GetRegister("reg0"));
  XLS_ASSERT_OK_AND_ASSIGN(auto* write0, blk->GetUniqueRegisterWrite(reg0));
  CodegenContext context(blk);
  context.GetMetadataForBlock(blk)
      .streaming_io_and_pipeline.pipeline_registers.push_back(
          {PipelineRegister{.reg = reg32,
                            .reg_write = write32->As<RegisterWrite>(),
                            .reg_read = read32.node()->As<RegisterRead>()},

           PipelineRegister{.reg = reg0,
                            .reg_write = write0->As<RegisterWrite>(),
                            .reg_read = read0.node()->As<RegisterRead>()}});

  XLS_ASSERT_OK(Run(p.get(), context));
  EXPECT_THAT(context.GetMetadataForBlock(blk)
                  .streaming_io_and_pipeline.pipeline_registers,
              testing::ElementsAre(testing::ElementsAre(
                  m::PipelineRegister(reg32, write32, read32.node()))));
}

TEST_F(RegisterLegalizationPassTest, MultipleRegisterWrites) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  XLS_ASSERT_OK_AND_ASSIGN(Register * r,
                           bb.block()->AddRegister("r", p->GetBitsType(32)));
  BValue en1 = bb.InputPort("en1", p->GetBitsType(1));
  BValue en2 = bb.InputPort("en2", p->GetBitsType(1));
  BValue en3 = bb.InputPort("en3", p->GetBitsType(1));
  bb.RegisterRead(r);
  bb.RegisterWrite(r, bb.Literal(UBits(10, 32)), en1);
  bb.RegisterWrite(r, bb.Literal(UBits(20, 32)), en2);
  bb.RegisterWrite(r, bb.Literal(UBits(30, 32)), en3);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(block->GetRegisterWrites(r), IsOkAndHolds(testing::SizeIs(3)));
  EXPECT_THAT(Run(block), IsOkAndHolds(true));
  EXPECT_THAT(block->GetRegisterWrites(r), IsOkAndHolds(testing::SizeIs(1)));

  XLS_ASSERT_OK_AND_ASSIGN(RegisterWrite * rw,
                           block->GetUniqueRegisterWrite(r));
  EXPECT_THAT(rw->data(),
              op_matchers::OneHotSelect(
                  op_matchers::Concat(),
                  {op_matchers::Literal(30), op_matchers::Literal(20),
                   op_matchers::Literal(10)}));
  EXPECT_THAT(rw->load_enable(),
              Optional(op_matchers::Or(op_matchers::Name("en1"),
                                       op_matchers::Name("en2"),
                                       op_matchers::Name("en3"))));

  // Pass should be idempotent.
  EXPECT_THAT(Run(block), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls::verilog
