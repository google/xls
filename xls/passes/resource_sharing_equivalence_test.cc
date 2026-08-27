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

#include "xls/passes/resource_sharing_equivalence.h"

#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/area_model/area_estimator.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/solvers/ir_equivalence_testutils.h"

namespace xls {
namespace {

using ::testing::ElementsAre;
using ::xls::solvers::ScopedVerifyEquivalence;
namespace m = ::xls::op_matchers;

class ResourceSharingEquivalenceTest : public IrTestBase {};

class FakeAreaEstimator : public AreaEstimator {
 public:
  FakeAreaEstimator() : AreaEstimator("fake") {}

  absl::StatusOr<double> GetOneBitRegisterAreaInSquareMicrons() const override {
    return 1.0;
  }

  absl::StatusOr<double> GetOperationAreaInSquareMicrons(
      Node* node) const override {
    if (node->op() == Op::kNeg) {
      return 10.0 * node->BitCountOrDie();
    }
    if (node->op() == Op::kXor) {
      return 5.0 * node->BitCountOrDie();
    }
    return 1.0;
  }
};

TEST_F(ResourceSharingEquivalenceTest, IdentityEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue add0 = fb.Add(a, b);
  BValue add1 = fb.Add(a, b);
  BValue mul0 = fb.UMul(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add0));

  // Same op and bitwidths -> Identity mapping succeeds
  auto mapping =
      GetNodeEquivalenceMapper().ComputeMapping(add0.node(), add1.node());
  ASSERT_TRUE(mapping.has_value());
  EXPECT_FALSE((*mapping)->RequiresOperandTransformation());
  EXPECT_FALSE((*mapping)->RequiresOutputTransformation());

  ScopedVerifyEquivalence sve(f);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<Node*> coerced,
      (*mapping)->ApplyToOperands(f, add0.node()->operands()));
  EXPECT_THAT(coerced, ElementsAre(a.node(), b.node()));

  XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                           (*mapping)->ApplyToOutput(f, add1.node()));
  EXPECT_EQ(output, add1.node());

  FakeAreaEstimator area_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      double overhead,
      (*mapping)->EstimateAreaOverhead(area_estimator, add0.node()->operands(),
                                       add0.node()));
  EXPECT_EQ(overhead, 0.0);

  // Different ops -> Not Identity
  auto mul_add_mapping =
      GetNodeEquivalenceMapper().ComputeMapping(mul0.node(), add0.node());
  EXPECT_FALSE(mul_add_mapping.has_value());
}

TEST_F(ResourceSharingEquivalenceTest, BitwidthExtendingEquivalenceMapping) {
  // umul: narrow to wide
  {
    auto p = CreatePackage();
    FunctionBuilder fb("umul_test", p.get());
    BValue a = fb.Param("a", p->GetBitsType(16));
    BValue b = fb.Param("b", p->GetBitsType(16));
    BValue mul_narrow = fb.UMul(a, b);
    BValue a32 = fb.ZeroExtend(a, 32);
    BValue b32 = fb.ZeroExtend(b, 32);
    BValue mul_wide = fb.UMul(a32, b32);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(mul_narrow));

    auto mapping = GetNodeEquivalenceMapper().ComputeMapping(mul_narrow.node(),
                                                             mul_wide.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->RequiresOperandTransformation());
    EXPECT_TRUE((*mapping)->RequiresOutputTransformation());

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, mul_narrow.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Param("a")),
                                     m::ZeroExt(m::Param("b"))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * new_mul,
        f->MakeNode<ArithOp>(mul_wide.node()->loc(), coerced[0], coerced[1],
                             /*width=*/32, Op::kUMul));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_mul));
    EXPECT_THAT(output, m::BitSlice(m::UMul(m::ZeroExt(), m::ZeroExt()),
                                    /*start=*/0, /*width=*/16));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // smul: narrow to wide
  {
    auto p = CreatePackage();
    FunctionBuilder fb("smul_test", p.get());
    BValue a = fb.Param("a", p->GetBitsType(16));
    BValue b = fb.Param("b", p->GetBitsType(16));
    BValue smul_narrow = fb.SMul(a, b);
    BValue a32 = fb.SignExtend(a, 32);
    BValue b32 = fb.SignExtend(b, 32);
    BValue smul_wide = fb.SMul(a32, b32);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                             fb.BuildWithReturnValue(smul_narrow));

    auto mapping = GetNodeEquivalenceMapper().ComputeMapping(smul_narrow.node(),
                                                             smul_wide.node());
    ASSERT_TRUE(mapping.has_value());

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, smul_narrow.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::SignExt(m::Param("a")),
                                     m::SignExt(m::Param("b"))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * new_smul,
        f->MakeNode<ArithOp>(smul_wide.node()->loc(), coerced[0], coerced[1],
                             /*width=*/32, Op::kSMul));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_smul));
    EXPECT_THAT(output, m::BitSlice(m::SMul(m::SignExt(), m::SignExt()),
                                    /*start=*/0, /*width=*/16));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, AddSubEquivalenceMapping) {
  // add -> sub: same bitwidth, different ops
  {
    auto p = CreatePackage();
    FunctionBuilder fb("add_to_sub", p.get());
    BValue a = fb.Param("a", p->GetBitsType(32));
    BValue b = fb.Param("b", p->GetBitsType(32));
    BValue add32 = fb.Add(a, b);
    BValue sub32 = fb.Subtract(a, b);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add32));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(add32.node(), sub32.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->RequiresOperandTransformation());
    EXPECT_FALSE((*mapping)->RequiresOutputTransformation());

    // Confirm area estimation
    FakeAreaEstimator area_estimator;
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        (*mapping)->EstimateAreaOverhead(
            area_estimator, add32.node()->operands(), add32.node()));
    EXPECT_EQ(overhead, 320.0);  // 10.0 * 32

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, add32.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::Param("a"), m::Neg(m::Param("b"))));
    // sub(a, neg(b)) == add(a, b)
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_sub,
                             f->MakeNode<BinOp>(sub32.node()->loc(), coerced[0],
                                                coerced[1], Op::kSub));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_sub));
    EXPECT_THAT(output, m::Sub(m::Param("a"), m::Neg(m::Param("b"))));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // add -> sub: operand 1 is already negated
  {
    auto p = CreatePackage();
    FunctionBuilder fb("add_neg_to_sub", p.get());
    BValue a = fb.Param("a", p->GetBitsType(32));
    BValue b = fb.Param("b", p->GetBitsType(32));
    BValue neg_b = fb.Negate(b);
    BValue add_neg = fb.Add(a, neg_b);
    BValue sub32 = fb.Subtract(a, b);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add_neg));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(add_neg.node(), sub32.node());
    ASSERT_TRUE(mapping.has_value());

    // Confirm area estimation
    FakeAreaEstimator area_estimator;
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        (*mapping)->EstimateAreaOverhead(
            area_estimator, add_neg.node()->operands(), add_neg.node()));
    EXPECT_EQ(overhead, 0.0);

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, add_neg.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::Param("a"), m::Param("b")));
    // sub(a, b) == add(a, neg(b))
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_sub,
                             f->MakeNode<BinOp>(sub32.node()->loc(), coerced[0],
                                                coerced[1], Op::kSub));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_sub));
    EXPECT_THAT(output, m::Sub(m::Param("a"), m::Param("b")));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // add -> sub: different ops AND different bit widths
  {
    auto p = CreatePackage();
    FunctionBuilder fb("add16_to_sub32", p.get());
    BValue a = fb.Param("a", p->GetBitsType(16));
    BValue b = fb.Param("b", p->GetBitsType(16));
    BValue add16 = fb.Add(a, b);
    BValue a32 = fb.ZeroExtend(a, 32);
    BValue b32 = fb.ZeroExtend(b, 32);
    BValue sub32 = fb.Subtract(a32, b32);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add16));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(add16.node(), sub32.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->RequiresOperandTransformation());
    EXPECT_TRUE((*mapping)->RequiresOutputTransformation());

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, add16.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Param("a")),
                                     m::ZeroExt(m::Neg(m::Param("b")))));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_sub,
                             f->MakeNode<BinOp>(sub32.node()->loc(), coerced[0],
                                                coerced[1], Op::kSub));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_sub));
    EXPECT_THAT(output,
                m::BitSlice(m::Sub(m::ZeroExt(), m::ZeroExt(m::Neg())), 0, 16));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ShiftEquivalenceMapping) {
  // different ops / unsupported mappings
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shift_unsupported_ops", p.get());
    BValue x32 = fb.Param("x32", p->GetBitsType(32));
    BValue x16 = fb.Param("x16", p->GetBitsType(16));
    BValue shll = fb.Shll(x32, x32);
    BValue shrl = fb.Shrl(x32, x32);
    BValue shra = fb.Shra(x32, x32);
    BValue shrl16 = fb.Shrl(x16, x16);
    BValue mul = fb.UMul(x32, x32);
    XLS_ASSERT_OK(fb.Build());
    EXPECT_FALSE(GetNodeEquivalenceMapper()
                     .ComputeMapping(mul.node(), shll.node())
                     .has_value());
    EXPECT_FALSE(GetNodeEquivalenceMapper()
                     .ComputeMapping(shrl.node(), mul.node())
                     .has_value());
    // shra (32-bit) -> shrl (16-bit) unsupported because narrowing is invalid
    EXPECT_FALSE(GetNodeEquivalenceMapper()
                     .ComputeMapping(shra.node(), shrl16.node())
                     .has_value());
  }

  // shll -> shrl
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shll_to_shrl", p.get());
    BValue x = fb.Param("x", p->GetBitsType(32));
    BValue s = fb.Param("s", p->GetBitsType(32));
    BValue shll = fb.Shll(x, s);
    BValue shrl = fb.Shrl(x, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shll));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shll.node(), shrl.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->RequiresOperandTransformation());
    EXPECT_TRUE((*mapping)->RequiresOutputTransformation());

    // Confirm area estimation
    FakeAreaEstimator area_estimator;
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        (*mapping)->EstimateAreaOverhead(area_estimator,
                                         shll.node()->operands(), shll.node()));
    EXPECT_EQ(overhead, 0.0);

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shll.node()->operands()));
    auto rev_x = m::Reverse(m::Param("x"));
    EXPECT_THAT(coerced, ElementsAre(rev_x, m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shrl,
                             f->MakeNode<BinOp>(shrl.node()->loc(), coerced[0],
                                                coerced[1], Op::kShrl));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shrl));
    EXPECT_THAT(output, m::Reverse(m::Shrl(rev_x, m::Param("s"))));

    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shrl -> shll
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shrl_to_shll", p.get());
    BValue x = fb.Param("x", p->GetBitsType(32));
    BValue s = fb.Param("s", p->GetBitsType(32));
    BValue shrl = fb.Shrl(x, s);
    BValue shll = fb.Shll(x, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shrl));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shrl.node(), shll.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->RequiresOperandTransformation());
    EXPECT_TRUE((*mapping)->RequiresOutputTransformation());

    // Confirm area estimation
    FakeAreaEstimator area_estimator;
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        (*mapping)->EstimateAreaOverhead(area_estimator,
                                         shrl.node()->operands(), shrl.node()));
    EXPECT_EQ(overhead, 0.0);

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shrl.node()->operands()));
    auto rev_x = m::Reverse(m::Param("x"));
    EXPECT_THAT(coerced, ElementsAre(rev_x, m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shll,
                             f->MakeNode<BinOp>(shll.node()->loc(), coerced[0],
                                                coerced[1], Op::kShll));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shll));
    EXPECT_THAT(output, m::Reverse(m::Shll(rev_x, m::Param("s"))));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ArithShiftEquivalenceMapping) {
  // shra -> shrl with same width
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shra_to_shrl_same_width", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue t = fb.Param("t", p->GetBitsType(4));
    BValue shra = fb.Shra(x, s);
    BValue shrl = fb.Shrl(x, t);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shra));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shra.node(), shrl.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->RequiresOperandTransformation());
    EXPECT_TRUE((*mapping)->RequiresOutputTransformation());

    // Confirm area estimation (2 * 5.0 * 4 = 40.0)
    FakeAreaEstimator area_estimator;
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        (*mapping)->EstimateAreaOverhead(area_estimator,
                                         shra.node()->operands(), shra.node()));
    EXPECT_EQ(overhead, 40.0);

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shra.node()->operands()));
    auto msb_mask =
        m::SignExt(m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1));
    auto x_xor_msb = m::Xor(m::Param("x"), msb_mask);
    EXPECT_THAT(coerced, ElementsAre(x_xor_msb, m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shrl,
                             f->MakeNode<BinOp>(shrl.node()->loc(), coerced[0],
                                                coerced[1], Op::kShrl));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shrl));
    EXPECT_THAT(output, m::Xor(m::Shrl(x_xor_msb, m::Param("s")), msb_mask));

    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shra -> shll with same width
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shra_to_shll_same_width", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue t = fb.Param("t", p->GetBitsType(4));
    BValue shra = fb.Shra(x, s);
    BValue shll = fb.Shll(x, t);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shra));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shra.node(), shll.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->RequiresOperandTransformation());
    EXPECT_TRUE((*mapping)->RequiresOutputTransformation());

    // Confirm area estimation (2 * 5.0 * 4 = 40.0)
    FakeAreaEstimator area_estimator;
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        (*mapping)->EstimateAreaOverhead(area_estimator,
                                         shra.node()->operands(), shra.node()));
    EXPECT_EQ(overhead, 40.0);

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shra.node()->operands()));
    auto msb_mask =
        m::SignExt(m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1));
    auto x_xor_msb = m::Xor(m::Param("x"), msb_mask);
    EXPECT_THAT(coerced, ElementsAre(m::Reverse(x_xor_msb), m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shll,
                             f->MakeNode<BinOp>(shll.node()->loc(), coerced[0],
                                                coerced[1], Op::kShll));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shll));
    EXPECT_THAT(
        output,
        m::Xor(m::Reverse(m::Shll(m::Reverse(x_xor_msb), m::Param("s"))),
               msb_mask));

    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ArithShiftEquivalenceMappingWidening) {
  // shra (4-bit) -> shrl (8-bit) widening
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shra_to_shrl_widening", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue shra = fb.Shra(x, s);
    BValue dummy = fb.Param("dummy", p->GetBitsType(8));
    BValue shrl = fb.Shrl(dummy, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shra));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shra.node(), shrl.node());
    ASSERT_TRUE(mapping.has_value());

    // Confirm area estimation (2 * 5.0 * 4 = 40.0)
    FakeAreaEstimator area_estimator;
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        (*mapping)->EstimateAreaOverhead(area_estimator,
                                         shra.node()->operands(), shra.node()));
    EXPECT_EQ(overhead, 40.0);

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shra.node()->operands()));
    auto msb_mask =
        m::SignExt(m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1));
    auto x_xor_msb = m::Xor(m::Param("x"), msb_mask);
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(x_xor_msb), m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shrl,
                             f->MakeNode<BinOp>(shrl.node()->loc(), coerced[0],
                                                coerced[1], Op::kShrl));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shrl));
    EXPECT_THAT(output, m::Xor(m::BitSlice(m::Shrl(m::ZeroExt(), m::Param("s")),
                                           /*start=*/0, /*width=*/4),
                               msb_mask));

    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shra (4-bit) -> shll (8-bit) widening
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shra_to_shll_widening", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue shra = fb.Shra(x, s);
    BValue dummy = fb.Param("dummy", p->GetBitsType(8));
    BValue shll = fb.Shll(dummy, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shra));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shra.node(), shll.node());
    ASSERT_TRUE(mapping.has_value());

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shra.node()->operands()));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shll,
                             f->MakeNode<BinOp>(shll.node()->loc(), coerced[0],
                                                coerced[1], Op::kShll));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shll));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shll (4-bit) -> shrl (8-bit) widening
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shll_to_shrl_widening", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue shll = fb.Shll(x, s);
    BValue dummy = fb.Param("dummy", p->GetBitsType(8));
    BValue shrl = fb.Shrl(dummy, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shll));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shll.node(), shrl.node());
    ASSERT_TRUE(mapping.has_value());

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shll.node()->operands()));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shrl,
                             f->MakeNode<BinOp>(shrl.node()->loc(), coerced[0],
                                                coerced[1], Op::kShrl));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shrl));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shrl (4-bit) -> shll (8-bit) widening
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shrl_to_shll_widening", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue shrl = fb.Shrl(x, s);
    BValue dummy = fb.Param("dummy", p->GetBitsType(8));
    BValue shll = fb.Shll(dummy, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shrl));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shrl.node(), shll.node());
    ASSERT_TRUE(mapping.has_value());

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shrl.node()->operands()));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shll,
                             f->MakeNode<BinOp>(shll.node()->loc(), coerced[0],
                                                coerced[1], Op::kShll));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shll));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shrl (4-bit) -> shra (8-bit) widening
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shrl_to_shra_widening", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue shrl = fb.Shrl(x, s);
    BValue dummy = fb.Param("dummy", p->GetBitsType(8));
    BValue shra = fb.Shra(dummy, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shrl));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shrl.node(), shra.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->RequiresOperandTransformation());
    EXPECT_TRUE((*mapping)->RequiresOutputTransformation());

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shrl.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Param("x")), m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shra,
                             f->MakeNode<BinOp>(shra.node()->loc(), coerced[0],
                                                coerced[1], Op::kShra));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shra));
    EXPECT_THAT(output, m::BitSlice(m::Shra(m::ZeroExt(), m::Param("s")),
                                    /*start=*/0, /*width=*/4));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shll (4-bit) -> shra (8-bit) widening
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shll_to_shra_widening", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue shll = fb.Shll(x, s);
    BValue dummy = fb.Param("dummy", p->GetBitsType(8));
    BValue shra = fb.Shra(dummy, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shll));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shll.node(), shra.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->RequiresOperandTransformation());
    EXPECT_TRUE((*mapping)->RequiresOutputTransformation());

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        (*mapping)->ApplyToOperands(f, shll.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Reverse(m::Param("x"))),
                                     m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shra,
                             f->MakeNode<BinOp>(shra.node()->loc(), coerced[0],
                                                coerced[1], Op::kShra));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shra));
    EXPECT_THAT(output, m::Reverse(m::BitSlice(
                            m::Shra(m::ZeroExt(m::Reverse()), m::Param("s")),
                            /*start=*/0, /*width=*/4)));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ArithShiftEquivalenceMappingMSBPadding) {
  // shrl (4-bit) -> shra (4-bit) with dst widening (to 5-bit)
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shrl_to_shra_same_width", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue shrl = fb.Shrl(x, s);
    BValue shra = fb.Shra(x, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shrl));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shrl.node(), shra.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->ModifiesDestinationNode());

    Node* unified_node = nullptr;
    // Verify equivalence for src (shrl)
    {
      ScopedVerifyEquivalence sve(f);
      XLS_ASSERT_OK_AND_ASSIGN(
          std::vector<Node*> coerced_from,
          (*mapping)->ApplyToOperands(f, shrl.node()->operands()));
      auto x_zeroext = m::ZeroExt(m::Param("x"));
      EXPECT_THAT(coerced_from, ElementsAre(x_zeroext, m::Param("s")));

      XLS_ASSERT_OK_AND_ASSIGN(unified_node,
                               (*mapping)->CreateUnifiedNode(f, coerced_from));
      EXPECT_THAT(unified_node, m::Shra(x_zeroext, m::Param("s")));
      EXPECT_EQ(unified_node->BitCountOrDie(), 5);

      XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                               (*mapping)->ApplyToOutput(f, unified_node));
      EXPECT_THAT(output, m::BitSlice(m::Shra(x_zeroext, m::Param("s")),
                                      /*start=*/0, /*width=*/4));
      XLS_ASSERT_OK(f->set_return_value(output));
    }

    // Verify equivalence for dst (shra mapped into unified 5-bit shra)
    {
      XLS_ASSERT_OK(f->set_return_value(shra.node()));
      ScopedVerifyEquivalence sve_var(f);

      std::unique_ptr<EquivalenceMapping> to_mapping =
          (*mapping)->GetDestinationMapping(unified_node);
      ASSERT_NE(to_mapping, nullptr);

      XLS_ASSERT_OK_AND_ASSIGN(
          std::vector<Node*> coerced_var,
          to_mapping->ApplyToOperands(f, shra.node()->operands()));
      EXPECT_THAT(coerced_var,
                  ElementsAre(m::SignExt(m::Param("x")), m::Param("s")));

      XLS_ASSERT_OK_AND_ASSIGN(
          Node * unified_var_node,
          f->MakeNode<BinOp>(shra.node()->loc(), coerced_var[0], coerced_var[1],
                             Op::kShra));
      EXPECT_THAT(unified_var_node, m::Shra(m::SignExt(), m::Param("s")));
      EXPECT_EQ(unified_var_node->BitCountOrDie(), 5);

      XLS_ASSERT_OK_AND_ASSIGN(Node * var_output,
                               to_mapping->ApplyToOutput(f, unified_var_node));
      EXPECT_THAT(var_output, m::BitSlice(m::Shra(m::SignExt(), m::Param("s")),
                                          /*start=*/0, /*width=*/4));
      XLS_ASSERT_OK(f->set_return_value(var_output));
    }
  }

  // shll (4-bit) -> shra (4-bit) with dst widening (to 5-bit)
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shll_to_shra_same_width", p.get());
    BValue x = fb.Param("x", p->GetBitsType(4));
    BValue s = fb.Param("s", p->GetBitsType(4));
    BValue shll = fb.Shll(x, s);
    BValue shra = fb.Shra(x, s);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shll));

    auto mapping =
        GetNodeEquivalenceMapper().ComputeMapping(shll.node(), shra.node());
    ASSERT_TRUE(mapping.has_value());
    EXPECT_TRUE((*mapping)->ModifiesDestinationNode());

    // Verify equivalence for src (shll)
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced_from,
        (*mapping)->ApplyToOperands(f, shll.node()->operands()));
    auto x_rev_zeroext = m::ZeroExt(m::Reverse(m::Param("x")));
    EXPECT_THAT(coerced_from, ElementsAre(x_rev_zeroext, m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * unified_node,
                             (*mapping)->CreateUnifiedNode(f, coerced_from));
    EXPECT_THAT(unified_node, m::Shra(x_rev_zeroext, m::Param("s")));
    EXPECT_EQ(unified_node->BitCountOrDie(), 5);

    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, unified_node));
    EXPECT_THAT(output,
                m::Reverse(m::BitSlice(m::Shra(x_rev_zeroext, m::Param("s")),
                                       /*start=*/0, /*width=*/4)));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

}  // namespace
}  // namespace xls
