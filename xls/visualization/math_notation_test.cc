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

#include "xls/visualization/math_notation.h"

#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/source_location.h"

namespace xls {
namespace {

class MathNotationTest : public IrTestBase {
 protected:
  MathNotationTest() = default;
};

TEST_F(MathNotationTest, NullExpression) {
  EXPECT_EQ(ToMathNotation(nullptr), "");
}

TEST_F(MathNotationTest, BooleanExpression) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32_type = p->GetBitsType(32);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue k = fb.Param("k", u32_type);
  BValue not_i = fb.Not(i, SourceInfo(), "not_i");
  BValue and_i_j_k = fb.And({i, j, k});
  BValue or_i_j_k = fb.Or({i, j, k});
  BValue xor_i_j = fb.Xor(i, j);
  BValue nand_i_k = fb.Nand(i, k);
  BValue nor_i_k = fb.Nor(i, k);
  BValue or_and_or = fb.Or({and_i_j_k, or_i_j_k});

  XLS_ASSERT_OK(fb.BuildWithReturnValue(or_and_or));
  EXPECT_EQ(ToMathNotation(not_i.node()), "!i");
  EXPECT_EQ(ToMathNotation(and_i_j_k.node()), "(i & j & k)");
  EXPECT_EQ(ToMathNotation(or_i_j_k.node()), "(i | j | k)");
  EXPECT_EQ(ToMathNotation(xor_i_j.node()), "(i ⊕ j)");
  EXPECT_EQ(ToMathNotation(nand_i_k.node()), "!(i & k)");
  EXPECT_EQ(ToMathNotation(nor_i_k.node()), "!(i | k)");
  EXPECT_EQ(ToMathNotation(or_and_or.node()), "((i & j & k) | (i | j | k))");
}

TEST_F(MathNotationTest, ArithmeticExpression) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32_type = p->GetBitsType(32);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue neg_i = fb.Negate(i);
  BValue add_i_j = fb.Add(i, j);
  BValue sub_i_j = fb.Subtract(i, j);
  BValue mul_i_j = fb.UMul(i, j);
  BValue div_i_j = fb.UDiv(i, j);
  BValue mod_i_j = fb.UMod(i, j);
  BValue smul_i_j = fb.SMul(i, j);
  BValue sdiv_i_j = fb.SDiv(i, j);
  BValue smod_i_j = fb.SMod(i, j);
  BValue shll_i_j = fb.Shll(i, j);
  BValue shrl_i_j = fb.Shrl(i, j);
  BValue shra_i_j = fb.Shra(i, j);

  XLS_ASSERT_OK(fb.BuildWithReturnValue(i));
  EXPECT_EQ(ToMathNotation(neg_i.node()), "-i");
  EXPECT_EQ(ToMathNotation(add_i_j.node()), "(i + j)");
  EXPECT_EQ(ToMathNotation(sub_i_j.node()), "(i - j)");
  EXPECT_EQ(ToMathNotation(mul_i_j.node()), "(i * j)");
  EXPECT_EQ(ToMathNotation(div_i_j.node()), "(i / j)");
  EXPECT_EQ(ToMathNotation(mod_i_j.node()), "(i % j)");
  EXPECT_EQ(ToMathNotation(smul_i_j.node()), "(i * j)");
  EXPECT_EQ(ToMathNotation(sdiv_i_j.node()), "(i / j)");
  EXPECT_EQ(ToMathNotation(smod_i_j.node()), "(i % j)");
  EXPECT_EQ(ToMathNotation(shll_i_j.node()), "(i << j)");
  EXPECT_EQ(ToMathNotation(shrl_i_j.node()), "(i >> j)");
  EXPECT_EQ(ToMathNotation(shra_i_j.node()), "(i >>> j)");
}

TEST_F(MathNotationTest, EqualityExpression) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32_type = p->GetBitsType(32);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue eq_i_j = fb.Eq(i, j);
  BValue ne_i_j = fb.Ne(i, j);
  BValue lt_i_j = fb.ULt(i, j);
  BValue le_i_j = fb.ULe(i, j);
  BValue gt_i_j = fb.UGt(i, j);
  BValue ge_i_j = fb.UGe(i, j);
  BValue slt_i_j = fb.SLt(i, j);
  BValue sle_i_j = fb.SLe(i, j);
  BValue sgt_i_j = fb.SGt(i, j);
  BValue sge_i_j = fb.SGe(i, j);

  XLS_ASSERT_OK(fb.BuildWithReturnValue(i));
  EXPECT_EQ(ToMathNotation(eq_i_j.node()), "(i == j)");
  EXPECT_EQ(ToMathNotation(ne_i_j.node()), "(i != j)");
  EXPECT_EQ(ToMathNotation(lt_i_j.node()), "(i < j)");
  EXPECT_EQ(ToMathNotation(le_i_j.node()), "(i <= j)");
  EXPECT_EQ(ToMathNotation(gt_i_j.node()), "(i > j)");
  EXPECT_EQ(ToMathNotation(ge_i_j.node()), "(i >= j)");
  EXPECT_EQ(ToMathNotation(slt_i_j.node()), "(i < j)");
  EXPECT_EQ(ToMathNotation(sle_i_j.node()), "(i <= j)");
  EXPECT_EQ(ToMathNotation(sgt_i_j.node()), "(i > j)");
  EXPECT_EQ(ToMathNotation(sge_i_j.node()), "(i >= j)");
}

TEST_F(MathNotationTest, ConcatAndBitSliceExpression) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32_type = p->GetBitsType(32);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue literal_42 = fb.Literal(UBits(42, 32));
  BValue concat_i_j = fb.Concat({i, j, literal_42});
  BValue bitslice_i = fb.BitSlice(concat_i_j, 2, 1);

  XLS_ASSERT_OK(fb.BuildWithReturnValue(concat_i_j));
  EXPECT_EQ(ToMathNotation(concat_i_j.node()), "(i ++ j ++ bits[32]:42)");
  EXPECT_EQ(ToMathNotation(bitslice_i.node()),
            "((i ++ j ++ bits[32]:42)[2:3])");
}

TEST_F(MathNotationTest, ReferenceComplexOperandsByName) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32_type = p->GetBitsType(32);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue add_i_j = fb.Add(i, j);
  BValue sub_i_j = fb.Subtract(i, j);
  BValue select = fb.Select(i, {add_i_j, sub_i_j}, i);
  BValue add_select_i = fb.Add(select, i);

  XLS_ASSERT_OK(fb.BuildWithReturnValue(add_select_i));
  EXPECT_EQ(ToMathNotation(add_select_i.node()),
            "(" + select.node()->GetName() + " + i)");
}

TEST_F(MathNotationTest, ReferenceAtomsByName) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32_type = p->GetBitsType(32);
  BValue i = fb.Param("i", u32_type);
  BValue j = fb.Param("j", u32_type);
  BValue literal_0 = fb.Literal(UBits(0, 32));
  BValue add_i = fb.Add(i, literal_0, SourceInfo(), "add_i");
  BValue add_j = fb.Add(j, add_i);

  XLS_ASSERT_OK(fb.BuildWithReturnValue(add_j));
  EXPECT_EQ(ToMathNotation(
                add_j.node(),
                [](const Node* node) { return node->GetName() == "add_i"; }),
            "(j + add_i)");
}

}  // namespace
}  // namespace xls
