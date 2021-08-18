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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"

namespace xls::verilog {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::UnorderedElementsAre;

class RegisterLegalizationPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Block* block) {
    PassResults results;
    CodegenPassUnit unit(block->package(), block);
    return RegisterLegalizationPass().Run(&unit, CodegenPassOptions(),
                                          &results);
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

}  // namespace
}  // namespace xls::verilog
