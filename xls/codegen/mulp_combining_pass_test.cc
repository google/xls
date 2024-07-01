// Copyright 2022 The XLS Authors
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

#include "xls/codegen/mulp_combining_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/codegen/codegen_pass.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/block.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"

namespace m = xls::op_matchers;
namespace xls::verilog {
namespace {

using status_testing::IsOkAndHolds;
using testing::AllOf;

class MulpCombiningPassTest : public IrTestBase {
 protected:
  absl::StatusOr<bool> Run(Block* block) {
    CodegenPassResults results;
    CodegenPassUnit unit(block->package(), block);
    return MulpCombiningPass().Run(&unit, CodegenPassOptions(), &results);
  }
};

TEST_F(MulpCombiningPassTest, SimpleMulpThenAdd) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue c = bb.InputPort("c", p->GetBitsType(32));
  BValue d = bb.InputPort("d", p->GetBitsType(32));
  BValue umulp = bb.UMulp(a, b);
  BValue smulp = bb.SMulp(c, d);
  BValue x = bb.OutputPort(
      "x", bb.Add(bb.TupleIndex(umulp, 0), bb.TupleIndex(umulp, 1)));
  BValue y = bb.OutputPort(
      "y", bb.Add(bb.TupleIndex(smulp, 1), bb.TupleIndex(smulp, 0)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(block), IsOkAndHolds(true));

  EXPECT_THAT(x.node(),
              m::OutputPort(m::UMul(m::InputPort("a"), m::InputPort("b"))));
  EXPECT_THAT(y.node(),
              m::OutputPort(m::SMul(m::InputPort("c"), m::InputPort("d"))));
}

TEST_F(MulpCombiningPassTest, MulpThenAddButWrongIndices) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue c = bb.InputPort("c", p->GetBitsType(32));
  BValue d = bb.InputPort("d", p->GetBitsType(32));
  BValue umulp = bb.UMulp(a, b);
  BValue smulp = bb.SMulp(c, d);
  // Output ports `x` and `y` are the sum the same tuple element from the mulp
  // op and thus aren't actually products.
  bb.OutputPort("x", bb.Add(bb.TupleIndex(umulp, 0), bb.TupleIndex(umulp, 0)));
  bb.OutputPort("y", bb.Add(bb.TupleIndex(smulp, 1), bb.TupleIndex(smulp, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(block), IsOkAndHolds(false));
}

TEST_F(MulpCombiningPassTest, MulpWithMultipleUsesOfMulp) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue umulp = bb.UMulp(a, b);
  bb.OutputPort("x", bb.Add(bb.TupleIndex(umulp, 0), bb.TupleIndex(umulp, 1)));
  bb.OutputPort("y", bb.Add(bb.TupleIndex(umulp, 0), bb.TupleIndex(umulp, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(block), IsOkAndHolds(false));
}

TEST_F(MulpCombiningPassTest, MulpWithOutputUseOfMulp) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue umulp = bb.UMulp(a, b);
  bb.OutputPort("x", bb.Add(bb.TupleIndex(umulp, 0), bb.TupleIndex(umulp, 1)));
  bb.OutputPort("y", umulp);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(block), IsOkAndHolds(false));
}

TEST_F(MulpCombiningPassTest, MulpWithMultipleUsesOfTupleIndex) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue umulp = bb.UMulp(a, b);
  BValue elem0 = bb.TupleIndex(umulp, 0);
  BValue elem1 = bb.TupleIndex(umulp, 1);
  bb.OutputPort("x", bb.Add(elem0, elem1));
  bb.OutputPort("y", bb.Subtract(elem0, elem1));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(block), IsOkAndHolds(false));
}

TEST_F(MulpCombiningPassTest, BitSlicedMulp) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue umulp = bb.UMulp(a, b);
  BValue elem0 = bb.TupleIndex(umulp, 0);
  BValue elem1 = bb.TupleIndex(umulp, 1);
  BValue x = bb.OutputPort(
      "x", bb.Add(bb.BitSlice(elem0, 0, 17), bb.BitSlice(elem1, 0, 17)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(block), IsOkAndHolds(true));

  EXPECT_THAT(x.node(), m::OutputPort(AllOf(
                            m::Type("bits[17]"),
                            m::UMul(m::InputPort("a"), m::InputPort("b")))));
}

TEST_F(MulpCombiningPassTest, BitSlicedMulpWithNonzeroOffset) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue umulp = bb.UMulp(a, b);
  BValue elem0 = bb.TupleIndex(umulp, 0);
  BValue elem1 = bb.TupleIndex(umulp, 1);
  bb.OutputPort("x",
                bb.Add(bb.BitSlice(elem0, 1, 17), bb.BitSlice(elem1, 1, 17)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(block), IsOkAndHolds(false));
}

TEST_F(MulpCombiningPassTest, BitSlicedMulpWithMultipleUsesOfBitslice) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue umulp = bb.UMulp(a, b);
  BValue lhs = bb.BitSlice(bb.TupleIndex(umulp, 0), 0, 17);
  BValue rhs = bb.BitSlice(bb.TupleIndex(umulp, 1), 0, 17);
  bb.OutputPort("x", bb.Add(lhs, rhs));
  bb.OutputPort("y", lhs);
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(block), IsOkAndHolds(false));
}

TEST_F(MulpCombiningPassTest, MulpThenAddWithRegisterInBetween) {
  auto p = CreatePackage();

  BlockBuilder bb(TestName(), p.get());
  XLS_ASSERT_OK(bb.block()->AddClockPort("clk"));
  BValue a = bb.InputPort("a", p->GetBitsType(32));
  BValue b = bb.InputPort("b", p->GetBitsType(32));
  BValue umulp = bb.UMulp(a, b);
  bb.InsertRegister("umulp_reg", umulp);
  bb.OutputPort("x", bb.Add(bb.TupleIndex(umulp, 0), bb.TupleIndex(umulp, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Block * block, bb.Build());

  EXPECT_THAT(Run(block), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls::verilog
