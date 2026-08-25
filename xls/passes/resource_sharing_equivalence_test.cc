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
  // different ops
  {
    auto p = CreatePackage();
    FunctionBuilder fb("shll_to_shrl", p.get());
    BValue x = fb.Param("x", p->GetBitsType(32));
    BValue shll = fb.Shll(x, x);
    BValue shrl = fb.Shrl(x, x);
    BValue mul = fb.UMul(x, x);
    XLS_ASSERT_OK(fb.Build());
    EXPECT_FALSE(GetNodeEquivalenceMapper()
                     .ComputeMapping(mul.node(), shll.node())
                     .has_value());
    EXPECT_FALSE(GetNodeEquivalenceMapper()
                     .ComputeMapping(shrl.node(), mul.node())
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
    EXPECT_THAT(coerced, ElementsAre(m::Reverse(m::Param("x")), m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shrl,
                             f->MakeNode<BinOp>(shrl.node()->loc(), coerced[0],
                                                coerced[1], Op::kShrl));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shrl));
    EXPECT_THAT(output,
                m::Reverse(m::Shrl(m::Reverse(m::Param("x")), m::Param("s"))));

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
    EXPECT_THAT(coerced, ElementsAre(m::Reverse(m::Param("x")), m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shll,
                             f->MakeNode<BinOp>(shll.node()->loc(), coerced[0],
                                                coerced[1], Op::kShll));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             (*mapping)->ApplyToOutput(f, new_shll));
    EXPECT_THAT(output,
                m::Reverse(m::Shll(m::Reverse(m::Param("x")), m::Param("s"))));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

}  // namespace
}  // namespace xls
