// Copyright 2025 The XLS Authors
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

#include "xls/passes/expression_builder.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/passes/bdd_query_engine.h"

namespace xls {
namespace {

namespace m = ::xls::op_matchers;

class ExpressionBuilderTest : public IrTestBase {};

TEST_F(ExpressionBuilderTest, DropRedundantOpsInAnd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue ab = fb.And(a, b);
  BValue ab2 = fb.And(a, b);
  BValue a_or_b = fb.Or(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f, fb.BuildWithReturnValue(fb.Tuple({ab, ab2, a_or_b})));

  auto bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  ExpressionBuilder builder(f, bdd_engine.get());

  EXPECT_TRUE(builder.Equivalent(ab.node(), ab2.node()));
  EXPECT_TRUE(builder.Implies(ab.node(), a_or_b.node()));
  std::vector<Node*> operands = {a_or_b.node(), ab.node(), a.node(), ab.node()};
  XLS_ASSERT_OK_AND_ASSIGN(Node * new_and, builder.AndOperands(operands));
  EXPECT_THAT(new_and, m::And(m::Param("a"), m::Param("b")));
  operands = {ab.node(), a_or_b.node()};
  XLS_ASSERT_OK_AND_ASSIGN(Node * reordered_and, builder.AndOperands(operands));
  EXPECT_THAT(reordered_and, m::And(m::Param("a"), m::Param("b")));
}

TEST_F(ExpressionBuilderTest, DropRedundantOpsInOr) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  BValue or_ab = fb.Or(a, b);
  BValue or_abc = fb.Or(c, or_ab);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({or_abc})));

  auto bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  ExpressionBuilder builder(f, bdd_engine.get());

  std::vector<Node*> operands = {b.node(), or_ab.node()};
  XLS_ASSERT_OK_AND_ASSIGN(Node * new_or, builder.OrOperands(operands));
  EXPECT_THAT(new_or, m::Or(m::Param("a"), m::Param("b")));
  operands = {or_ab.node(), b.node()};
  XLS_ASSERT_OK_AND_ASSIGN(Node * reordered_or, builder.OrOperands(operands));
  EXPECT_THAT(reordered_or, m::Or(m::Param("a"), m::Param("b")));

  operands = {b.node(), or_ab.node(), a.node(), or_abc.node(), c.node()};
  XLS_ASSERT_OK_AND_ASSIGN(new_or, builder.OrOperands(operands));
  EXPECT_THAT(new_or,
              m::Or(m::Param("c"), m::Or(m::Param("a"), m::Param("b"))));
}

TEST_F(ExpressionBuilderTest, UseCachedNaryOperations) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(1));
  BValue b = fb.Param("b", p->GetBitsType(1));
  BValue c = fb.Param("c", p->GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({a, b, c})));

  auto bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  ExpressionBuilder builder(f, bdd_engine.get());

  XLS_ASSERT_OK_AND_ASSIGN(Node * not_a,
                           builder.FindOrMakeUnaryNode(Op::kNot, a.node()));
  XLS_ASSERT_OK_AND_ASSIGN(Node * not_a_again,
                           builder.FindOrMakeUnaryNode(Op::kNot, a.node()));
  EXPECT_EQ(not_a, not_a_again);

  std::vector<Node*> ops_ab = {a.node(), b.node()};
  XLS_ASSERT_OK_AND_ASSIGN(Node * ab, builder.AndOperands(ops_ab));
  EXPECT_THAT(ab, m::And(m::Param("a"), m::Param("b")));
  XLS_ASSERT_OK_AND_ASSIGN(Node * ab_again, builder.AndOperands(ops_ab));
  EXPECT_EQ(ab, ab_again);

  std::vector<Node*> ops_abc = {a.node(), c.node(), b.node()};
  XLS_ASSERT_OK_AND_ASSIGN(Node * abc, builder.AndOperands(ops_abc));
  EXPECT_THAT(abc, m::And(m::And(m::Param("a"), m::Param("b")), m::Param("c")));
  EXPECT_EQ(abc->operand(0), ab);
}

TEST_F(ExpressionBuilderTest, UseCachedBitsliceOperations) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(3));
  BValue bit0 = fb.BitSlice(x, 0, 1);
  BValue bits12 = fb.BitSlice(x, 1, 2);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Tuple({bit0, bits12})));

  auto bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  ExpressionBuilder builder(f, bdd_engine.get());

  XLS_ASSERT_OK_AND_ASSIGN(Node * bit1,
                           builder.FindOrMakeBitSlice(x.node(), 1, 1));
  EXPECT_THAT(bit1, m::BitSlice(m::Param("x"), 1, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Node * bit1_again,
                           builder.FindOrMakeBitSlice(x.node(), 1, 1));
  EXPECT_EQ(bit1, bit1_again);

  // Reuses existing bit slices in function because they are easily located.
  XLS_ASSERT_OK_AND_ASSIGN(Node * bit0_again,
                           builder.FindOrMakeBitSlice(x.node(), 0, 1));
  EXPECT_THAT(bit0_again, m::BitSlice(m::Param("x"), 0, 1));
  EXPECT_EQ(bit0.node(), bit0_again);
  XLS_ASSERT_OK_AND_ASSIGN(Node * bits12_again,
                           builder.FindOrMakeBitSlice(x.node(), 1, 2));
  EXPECT_THAT(bits12_again, m::BitSlice(m::Param("x"), 1, 2));
  EXPECT_EQ(bits12.node(), bits12_again);
}

TEST_F(ExpressionBuilderTest, CachesEvenWithTempFunctionNodes) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue y = fb.Param("y", p->GetBitsType(1));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(fb.Or(x, y)));

  auto bdd_engine = BddQueryEngine::MakeDefault();
  XLS_ASSERT_OK(bdd_engine->Populate(f).status());
  ExpressionBuilder builder(f, bdd_engine.get());

  XLS_ASSERT_OK_AND_ASSIGN(Node * tmp_x, builder.TmpFuncNodeOrParam(x.node()));
  XLS_ASSERT_OK_AND_ASSIGN(Node * tmp_y, builder.TmpFuncNodeOrParam(y.node()));
  EXPECT_THAT(tmp_x, m::Param("param.1"));
  EXPECT_THAT(tmp_y, m::Param("param.2"));
  EXPECT_EQ(builder.TmpFunc(), tmp_x->function_base());

  std::vector<Node*> operands = {tmp_x, tmp_y};
  XLS_ASSERT_OK_AND_ASSIGN(Node * tmp_xy, builder.AndOperands(operands));
  EXPECT_THAT(tmp_xy, m::And(m::Param("param.1"), m::Param("param.2")));
  XLS_ASSERT_OK_AND_ASSIGN(Node * tmp_xy_again, builder.AndOperands(operands));
  EXPECT_EQ(tmp_xy, tmp_xy_again);
}

}  // namespace
}  // namespace xls
