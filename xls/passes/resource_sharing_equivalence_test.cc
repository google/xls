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
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
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

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::xls::solvers::ScopedVerifyEquivalence;
namespace m = ::xls::op_matchers;

using NodeToMappings =
    absl::flat_hash_map<Node*, std::unique_ptr<EquivalenceMapping>>;

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
  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<NodeToMappings> mappings,
      GetNodeEquivalenceMapper().ComputeMappings({add0.node()}, add1.node()));
  ASSERT_TRUE(mappings.has_value());
  const std::unique_ptr<EquivalenceMapping>& mapping =
      mappings->at(add0.node());
  EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(false));
  EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(false));

  ScopedVerifyEquivalence sve(f);
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<Node*> coerced,
      mapping->ApplyToOperands(f, add0.node()->operands()));
  EXPECT_THAT(coerced, ElementsAre(a.node(), b.node()));

  XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                           mapping->ApplyToOutput(f, add1.node()));
  EXPECT_EQ(output, add1.node());

  FakeAreaEstimator area_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      double overhead,
      mapping->EstimateAreaOverhead(area_estimator, add0.node()->operands(),
                                    add0.node()));
  EXPECT_EQ(overhead, 0.0);

  // Different ops -> Not Identity
  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<NodeToMappings> mul_add_mapping,
      GetNodeEquivalenceMapper().ComputeMappings({mul0.node()}, add0.node()));
  EXPECT_FALSE(mul_add_mapping.has_value());
}

TEST_F(ResourceSharingEquivalenceTest, BitwidthExtendingEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(16));
  BValue b = fb.Param("b", p->GetBitsType(16));
  BValue mul_narrow = fb.UMul(a, b);
  BValue mul_wide = fb.UMul(fb.ZeroExtend(a, 32), fb.ZeroExtend(b, 32));
  BValue smul_narrow = fb.SMul(a, b);
  BValue smul_wide = fb.SMul(fb.SignExtend(a, 32), fb.SignExtend(b, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(mul_narrow));

  // umul: narrow to wide
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {mul_narrow.node()}, mul_wide.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(mul_narrow.node());
    EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(true));
    EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(true));

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, mul_narrow.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Param("a")),
                                     m::ZeroExt(m::Param("b"))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * new_mul,
        f->MakeNode<ArithOp>(mul_wide.node()->loc(), coerced[0], coerced[1],
                             /*width=*/32, Op::kUMul));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output, mapping->ApplyToOutput(f, new_mul));
    EXPECT_THAT(output, m::BitSlice(m::UMul(m::ZeroExt(), m::ZeroExt()),
                                    /*start=*/0, /*width=*/16));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // smul: narrow to wide
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {smul_narrow.node()}, smul_wide.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(smul_narrow.node());

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(smul_narrow.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, smul_narrow.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::SignExt(m::Param("a")),
                                     m::SignExt(m::Param("b"))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * new_smul,
        f->MakeNode<ArithOp>(smul_wide.node()->loc(), coerced[0], coerced[1],
                             /*width=*/32, Op::kSMul));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_smul));
    EXPECT_THAT(output, m::BitSlice(m::SMul(m::SignExt(), m::SignExt()),
                                    /*start=*/0, /*width=*/16));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, AddSubEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue add32 = fb.Add(a, b);
  BValue sub32 = fb.Subtract(a, b);
  BValue add_neg = fb.Add(a, fb.Negate(b));
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));
  BValue add16 = fb.Add(a16, b16);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add32));
  FakeAreaEstimator area_estimator;

  // add -> sub: same bitwidth, different ops
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {add32.node()}, sub32.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(add32.node());
    EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(true));
    EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(false));

    // Confirm area estimation
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        mapping->EstimateAreaOverhead(area_estimator, add32.node()->operands(),
                                      add32.node()));
    EXPECT_EQ(overhead, 320.0);  // 10.0 * 32

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, add32.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::Param("a"), m::Neg(m::Param("b"))));
    // sub(a, neg(b)) == add(a, b)
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_sub,
                             f->MakeNode<BinOp>(sub32.node()->loc(), coerced[0],
                                                coerced[1], Op::kSub));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output, mapping->ApplyToOutput(f, new_sub));
    EXPECT_THAT(output, m::Sub(m::Param("a"), m::Neg(m::Param("b"))));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // add -> sub: operand 1 is already negated
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {add_neg.node()}, sub32.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(add_neg.node());

    // Confirm area estimation
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        mapping->EstimateAreaOverhead(
            area_estimator, add_neg.node()->operands(), add_neg.node()));
    EXPECT_EQ(overhead, 0.0);

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(add_neg.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, add_neg.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::Param("a"), m::Param("b")));
    // sub(a, b) == add(a, neg(b))
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_sub,
                             f->MakeNode<BinOp>(sub32.node()->loc(), coerced[0],
                                                coerced[1], Op::kSub));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output, mapping->ApplyToOutput(f, new_sub));
    EXPECT_THAT(output, m::Sub(m::Param("a"), m::Param("b")));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // add -> sub: different ops AND different bit widths
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {add16.node()}, sub32.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(add16.node());
    EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(true));
    EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(true));

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(add16.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, add16.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Param("a16")),
                                     m::ZeroExt(m::Neg(m::Param("b16")))));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_sub,
                             f->MakeNode<BinOp>(sub32.node()->loc(), coerced[0],
                                                coerced[1], Op::kSub));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output, mapping->ApplyToOutput(f, new_sub));
    EXPECT_THAT(output,
                m::BitSlice(m::Sub(m::ZeroExt(), m::ZeroExt(m::Neg())), 0, 16));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ShiftEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue s = fb.Param("s", p->GetBitsType(32));
  BValue x16 = fb.Param("x16", p->GetBitsType(16));
  BValue shll = fb.Shll(x, s);
  BValue shrl = fb.Shrl(x, s);
  BValue shra = fb.Shra(x, s);
  BValue shrl16 = fb.Shrl(x16, x16);
  BValue mul = fb.UMul(x, s);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shll));
  FakeAreaEstimator area_estimator;
  auto rev_x = m::Reverse(m::Param("x"));

  // different ops / unsupported mappings
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mul_shll,
        GetNodeEquivalenceMapper().ComputeMappings({mul.node()}, shll.node()));
    EXPECT_FALSE(mul_shll.has_value());
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> shrl_mul,
        GetNodeEquivalenceMapper().ComputeMappings({shrl.node()}, mul.node()));
    EXPECT_FALSE(shrl_mul.has_value());
    // shra (32-bit) -> shrl (16-bit) unsupported because narrowing is invalid
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> shra_shrl16,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shra.node()}, shrl16.node()));
    EXPECT_FALSE(shra_shrl16.has_value());
  }

  // shll -> shrl
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({shll.node()}, shrl.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shll.node());
    EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(true));
    EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(true));

    // Confirm area estimation
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        mapping->EstimateAreaOverhead(area_estimator, shll.node()->operands(),
                                      shll.node()));
    EXPECT_EQ(overhead, 0.0);

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shll.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(rev_x, m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shrl,
                             f->MakeNode<BinOp>(shrl.node()->loc(), coerced[0],
                                                coerced[1], Op::kShrl));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shrl));
    EXPECT_THAT(output, m::Reverse(m::Shrl(rev_x, m::Param("s"))));

    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shrl -> shll
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({shrl.node()}, shll.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shrl.node());
    EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(true));
    EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(true));

    // Confirm area estimation
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        mapping->EstimateAreaOverhead(area_estimator, shrl.node()->operands(),
                                      shrl.node()));
    EXPECT_EQ(overhead, 0.0);

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(shrl.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shrl.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(rev_x, m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shll,
                             f->MakeNode<BinOp>(shll.node()->loc(), coerced[0],
                                                coerced[1], Op::kShll));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shll));
    EXPECT_THAT(output, m::Reverse(m::Shll(rev_x, m::Param("s"))));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ArithShiftEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(4));
  BValue shra = fb.Shra(x, s);
  BValue shrl = fb.Shrl(x, s);
  BValue shll = fb.Shll(x, s);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shra));
  FakeAreaEstimator area_estimator;
  auto msb_mask =
      m::SignExt(m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1));
  auto x_xor_msb = m::Xor(m::Param("x"), msb_mask);
  auto rev_x_xor_msb = m::Reverse(x_xor_msb);

  // shra -> shrl with same width
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({shra.node()}, shrl.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shra.node());
    EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(true));
    EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(true));

    // Confirm area estimation (2 * 5.0 * 4 = 40.0)
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        mapping->EstimateAreaOverhead(area_estimator, shra.node()->operands(),
                                      shra.node()));
    EXPECT_EQ(overhead, 40.0);

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shra.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(x_xor_msb, m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shrl,
                             f->MakeNode<BinOp>(shrl.node()->loc(), coerced[0],
                                                coerced[1], Op::kShrl));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shrl));
    EXPECT_THAT(output, m::Xor(m::Shrl(x_xor_msb, m::Param("s")), msb_mask));

    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shra -> shll with same width
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({shra.node()}, shll.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shra.node());
    EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(true));
    EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(true));

    // Confirm area estimation (2 * 5.0 * 4 = 40.0)
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        mapping->EstimateAreaOverhead(area_estimator, shra.node()->operands(),
                                      shra.node()));
    EXPECT_EQ(overhead, 40.0);

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(shra.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shra.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(rev_x_xor_msb, m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shll,
                             f->MakeNode<BinOp>(shll.node()->loc(), coerced[0],
                                                coerced[1], Op::kShll));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shll));
    EXPECT_THAT(
        output,
        m::Xor(m::Reverse(m::Shll(rev_x_xor_msb, m::Param("s"))), msb_mask));

    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ArithShiftEquivalenceMappingWidening) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(4));
  BValue y = fb.Param("y", p->GetBitsType(8));
  BValue shra4 = fb.Shra(x, s);
  BValue shrl4 = fb.Shrl(x, s);
  BValue shll4 = fb.Shll(x, s);
  BValue shra8 = fb.Shra(y, s);
  BValue shrl8 = fb.Shrl(y, s);
  BValue shll8 = fb.Shll(y, s);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shra4));
  FakeAreaEstimator area_estimator;
  auto msb_mask =
      m::SignExt(m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1));
  auto x_xor_msb = m::Xor(m::Param("x"), msb_mask);
  auto rev_x = m::Reverse(m::Param("x"));

  // shra (4-bit) -> shrl (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shra4.node()}, shrl8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shra4.node());

    // Confirm area estimation (2 * 5.0 * 4 = 40.0)
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        mapping->EstimateAreaOverhead(area_estimator, shra4.node()->operands(),
                                      shra4.node()));
    EXPECT_EQ(overhead, 40.0);

    // Create mapped node to confirm equivalence
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shra4.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(x_xor_msb), m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shrl,
                             f->MakeNode<BinOp>(shrl8.node()->loc(), coerced[0],
                                                coerced[1], Op::kShrl));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shrl));
    EXPECT_THAT(output, m::Xor(m::BitSlice(m::Shrl(m::ZeroExt(), m::Param("s")),
                                           /*start=*/0, /*width=*/4),
                               msb_mask));

    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shra (4-bit) -> shll (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shra4.node()}, shll8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shra4.node());

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(shra4.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shra4.node()->operands()));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shll,
                             f->MakeNode<BinOp>(shll8.node()->loc(), coerced[0],
                                                coerced[1], Op::kShll));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shll));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shll (4-bit) -> shrl (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shll4.node()}, shrl8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shll4.node());

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(shll4.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shll4.node()->operands()));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shrl,
                             f->MakeNode<BinOp>(shrl8.node()->loc(), coerced[0],
                                                coerced[1], Op::kShrl));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shrl));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shrl (4-bit) -> shll (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shrl4.node()}, shll8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shrl4.node());

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(shrl4.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shrl4.node()->operands()));
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shll,
                             f->MakeNode<BinOp>(shll8.node()->loc(), coerced[0],
                                                coerced[1], Op::kShll));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shll));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shrl (4-bit) -> shra (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shrl4.node()}, shra8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shrl4.node());
    EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(true));
    EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(true));

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(shrl4.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shrl4.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Param("x")), m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shra,
                             f->MakeNode<BinOp>(shra8.node()->loc(), coerced[0],
                                                coerced[1], Op::kShra));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shra));
    EXPECT_THAT(output, m::BitSlice(m::Shra(m::ZeroExt(), m::Param("s")),
                                    /*start=*/0, /*width=*/4));
    XLS_ASSERT_OK(f->set_return_value(output));
  }

  // shll (4-bit) -> shra (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shll4.node()}, shra8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shll4.node());
    EXPECT_THAT(mapping->RequiresOperandTransformation(), IsOkAndHolds(true));
    EXPECT_THAT(mapping->RequiresOutputTransformation(), IsOkAndHolds(true));

    // Create mapped node to confirm equivalence
    XLS_ASSERT_OK(f->set_return_value(shll4.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shll4.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(rev_x), m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(Node * new_shra,
                             f->MakeNode<BinOp>(shra8.node()->loc(), coerced[0],
                                                coerced[1], Op::kShra));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping->ApplyToOutput(f, new_shra));
    EXPECT_THAT(output, m::Reverse(m::BitSlice(
                            m::Shra(m::ZeroExt(rev_x), m::Param("s")),
                            /*start=*/0, /*width=*/4)));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ArithShiftEquivalenceMappingMSBPadding) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue s = fb.Param("s", p->GetBitsType(4));
  BValue shrl = fb.Shrl(x, s);
  BValue shll = fb.Shll(x, s);
  BValue shra = fb.Shra(x, s);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shrl));
  auto x_zeroext = m::ZeroExt(m::Param("x"));
  auto x_signext = m::SignExt(m::Param("x"));
  auto x_rev_zeroext = m::ZeroExt(m::Reverse(m::Param("x")));

  // shrl (4-bit) -> shra (4-bit) with dst widening (to 5-bit)
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({shrl.node()}, shra.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& from_mapping =
        mappings->at(shrl.node());
    const std::unique_ptr<EquivalenceMapping>& to_mapping =
        mappings->at(shra.node());

    // Verify equivalence for src (shrl)
    {
      ScopedVerifyEquivalence sve(f);
      XLS_ASSERT_OK_AND_ASSIGN(
          std::vector<Node*> coerced_from,
          from_mapping->ApplyToOperands(f, shrl.node()->operands()));
      EXPECT_THAT(coerced_from, ElementsAre(x_zeroext, m::Param("s")));

      XLS_ASSERT_OK_AND_ASSIGN(
          Node * unified_node,
          from_mapping->dst()->CloneInNewFunction(coerced_from, f));
      EXPECT_THAT(unified_node, m::Shra(x_zeroext, m::Param("s")));
      EXPECT_EQ(unified_node->BitCountOrDie(), 5);

      XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                               from_mapping->ApplyToOutput(f, unified_node));
      EXPECT_THAT(output, m::BitSlice(m::Shra(x_zeroext, m::Param("s")),
                                      /*start=*/0, /*width=*/4));
      XLS_ASSERT_OK(f->set_return_value(output));
    }

    // Verify equivalence for dst (shra mapped into 5-bit shra)
    {
      XLS_ASSERT_OK(f->set_return_value(shra.node()));
      ScopedVerifyEquivalence sve_var(f);

      XLS_ASSERT_OK_AND_ASSIGN(
          std::vector<Node*> coerced_var,
          to_mapping->ApplyToOperands(f, shra.node()->operands()));
      EXPECT_THAT(coerced_var, ElementsAre(x_signext, m::Param("s")));

      XLS_ASSERT_OK_AND_ASSIGN(
          Node * unified_var_node,
          to_mapping->dst()->CloneInNewFunction(coerced_var, f));
      EXPECT_THAT(unified_var_node, m::Shra(x_signext, m::Param("s")));
      EXPECT_EQ(unified_var_node->BitCountOrDie(), 5);

      XLS_ASSERT_OK_AND_ASSIGN(Node * var_output,
                               to_mapping->ApplyToOutput(f, unified_var_node));
      EXPECT_THAT(var_output, m::BitSlice(m::Shra(x_signext, m::Param("s")),
                                          /*start=*/0, /*width=*/4));
      XLS_ASSERT_OK(f->set_return_value(var_output));
    }
  }

  // shll (4-bit) -> shra (4-bit) with dst widening (to 5-bit)
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({shll.node()}, shra.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& from_mapping =
        mappings->at(shll.node());

    // Verify equivalence for src (shll)
    XLS_ASSERT_OK(f->set_return_value(shll.node()));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced_from,
        from_mapping->ApplyToOperands(f, shll.node()->operands()));
    EXPECT_THAT(coerced_from, ElementsAre(x_rev_zeroext, m::Param("s")));

    XLS_ASSERT_OK_AND_ASSIGN(
        Node * unified_node,
        from_mapping->dst()->CloneInNewFunction(coerced_from, f));
    EXPECT_THAT(unified_node, m::Shra(x_rev_zeroext, m::Param("s")));
    EXPECT_EQ(unified_node->BitCountOrDie(), 5);

    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             from_mapping->ApplyToOutput(f, unified_node));
    EXPECT_THAT(output,
                m::Reverse(m::BitSlice(m::Shra(x_rev_zeroext, m::Param("s")),
                                       /*start=*/0, /*width=*/4)));
    XLS_ASSERT_OK(f->set_return_value(output));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ComparatorEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));

  // Unsigned inequalities
  BValue ult = fb.ULt(a, b);
  BValue ugt = fb.UGt(a, b);
  BValue ule = fb.ULe(a, b);
  BValue uge = fb.UGe(a, b);
  BValue ult16 = fb.ULt(a16, b16);

  // Signed inequalities
  BValue slt = fb.SLt(a, b);
  BValue sgt = fb.SGt(a, b);
  BValue sle = fb.SLe(a, b);
  BValue sge = fb.SGe(a, b);
  BValue slt16 = fb.SLt(a16, b16);

  // Equalities
  BValue eq = fb.Eq(a, b);
  BValue ne = fb.Ne(a, b);
  BValue eq16 = fb.Eq(a16, b16);

  // Default return value to be overwritten by each test case.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(eq));
  FakeAreaEstimator area_estimator;

  // Helper to test swapping either the equality or inequality comparison.
  auto test_swap = [&](Node* src, Node* dst, int tuple_index,
                       double expected_overhead) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({src}, dst));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping = mappings->at(src);

    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        mapping->EstimateAreaOverhead(area_estimator, src->operands(), src));
    EXPECT_EQ(overhead, expected_overhead);

    XLS_ASSERT_OK(f->set_return_value(src));
    ScopedVerifyEquivalence sve(f);
    absl::Span<Node* const> src_ops = src->operands();
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> coerced,
                             mapping->ApplyToOperands(f, src_ops));
    // Confirm the mapping knows when it has to transform operands.
    ASSERT_EQ(coerced.size(), src_ops.size());
    for (int i = 0; i < coerced.size(); ++i) {
      XLS_ASSERT_OK_AND_ASSIGN(bool requires_transform,
                               mapping->RequiresOperandTransformation());
      EXPECT_EQ(coerced[i] != src_ops[i], requires_transform);
    }
    XLS_ASSERT_OK_AND_ASSIGN(Node * new_dst,
                             dst->CloneInNewFunction(coerced, f));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output, mapping->ApplyToOutput(f, new_dst));
    // Confirm the mapping knows when it has to transform the output.
    XLS_ASSERT_OK_AND_ASSIGN(bool requires_output_transform,
                             mapping->RequiresOutputTransformation());
    EXPECT_EQ(output != new_dst, requires_output_transform);
    XLS_ASSERT_OK(f->set_return_value(output));
  };

  // Test unsigned inequality swaps (tuple element 0):
  // ult -> ugt: swap operands (overhead 0.0)
  test_swap(ult.node(), ugt.node(), /*tuple_index=*/0, 0.0);
  // ult -> ule: swap operands + invert output (overhead 1.0)
  test_swap(ult.node(), ule.node(), /*tuple_index=*/0, 1.0);
  // ult -> uge: invert output (overhead 1.0)
  test_swap(ult.node(), uge.node(), /*tuple_index=*/0, 1.0);
  // ult16 -> ugt (with zero extension): swap operands + extend (overhead 0.0)
  test_swap(ult16.node(), ugt.node(), /*tuple_index=*/0, 0.0);
  // ult16 -> uge (with zero extension): extend + invert output (overhead 1.0)
  test_swap(ult16.node(), uge.node(), /*tuple_index=*/0, 1.0);

  // Test signed inequality swaps (tuple element 0):
  // slt -> sgt: swap operands (overhead 0.0)
  test_swap(slt.node(), sgt.node(), /*tuple_index=*/0, 0.0);
  // slt -> sle: swap operands + invert output (overhead 1.0)
  test_swap(slt.node(), sle.node(), /*tuple_index=*/0, 1.0);
  // slt -> sge: invert output (overhead 1.0)
  test_swap(slt.node(), sge.node(), /*tuple_index=*/0, 1.0);
  // slt16 -> sgt (with sign extension): swap operands + extend (overhead 0.0)
  test_swap(slt16.node(), sgt.node(), /*tuple_index=*/0, 0.0);
  // slt16 -> sge (with sign extension): extend + invert output (overhead 1.0)
  test_swap(slt16.node(), sge.node(), /*tuple_index=*/0, 1.0);

  // Test equality swaps (tuple element 1):
  // eq -> ne: invert output (overhead 1.0)
  test_swap(eq.node(), ne.node(), /*tuple_index=*/1, 1.0);
  // ne -> eq: invert output (overhead 1.0)
  test_swap(ne.node(), eq.node(), /*tuple_index=*/1, 1.0);
  // eq16 -> ne (with zero extension): extend + invert output (overhead 1.0)
  test_swap(eq16.node(), ne.node(), /*tuple_index=*/1, 1.0);

  // Incompatible comparator mappings
  {
    // Equality -> Inequality
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> eq_ult,
        GetNodeEquivalenceMapper().ComputeMappings({eq.node()}, ult.node()));
    EXPECT_FALSE(eq_ult.has_value());

    // Inequality -> Equality
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> ult_eq,
        GetNodeEquivalenceMapper().ComputeMappings({ult.node()}, eq.node()));
    EXPECT_FALSE(ult_eq.has_value());

    // Signed -> Unsigned
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> slt_ult,
        GetNodeEquivalenceMapper().ComputeMappings({slt.node()}, ult.node()));
    EXPECT_FALSE(slt_ult.has_value());

    // Unsigned -> Signed
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> ult_slt,
        GetNodeEquivalenceMapper().ComputeMappings({ult.node()}, slt.node()));
    EXPECT_FALSE(ult_slt.has_value());

    // Narrowing unsupported (32-bit -> 16-bit)
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> ult_ult16,
        GetNodeEquivalenceMapper().ComputeMappings({ult.node()}, ult16.node()));
    EXPECT_FALSE(ult_ult16.has_value());
  }
}

TEST_F(ResourceSharingEquivalenceTest, CloneEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue add0 = fb.Add(a, b);
  BValue add1 = fb.Add(a, b);
  BValue sub0 = fb.Subtract(a, b);
  BValue shra0 = fb.Shra(a, b);
  BValue shrl0 = fb.Shrl(a, b);
  BValue eq0 = fb.Eq(a, b);
  BValue ne0 = fb.Ne(a, b);
  BValue ult0 = fb.ULt(a, b);
  BValue uge0 = fb.UGe(a, b);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(add0).status());

  // confirm cloning does node remapping and complains when original to clone
  // node pairs are incomplete.
  auto eq_mapping_clone = [&](Node* src, Node* dst) {
    XLS_ASSERT_OK_AND_ASSIGN(
        auto mappings, GetNodeEquivalenceMapper().ComputeMappings({src}, dst));
    ASSERT_TRUE(mappings.has_value());
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<EquivalenceMapping> clone_same,
                             mappings->at(src)->Clone(std::nullopt));
    EXPECT_EQ(clone_same->src(), src);
    EXPECT_EQ(clone_same->dst(), dst);

    // Illogical remapping is accepted; we trust the user's node pairings.
    absl::flat_hash_map<Node*, Node*> remapping = {
        {src, dst},
        {dst, src},
    };
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<EquivalenceMapping> clone_remapped,
                             mappings->at(src)->Clone(&remapping));
    EXPECT_EQ(clone_remapped->src(), dst);
    EXPECT_EQ(clone_remapped->dst(), src);

    // Missing node in non-empty remapping returns an error.
    absl::flat_hash_map<Node*, Node*> incomplete_remapping = {
        {src, dst},
    };
    EXPECT_THAT(mappings->at(src)->Clone(&incomplete_remapping),
                StatusIs(absl::StatusCode::kInternal));
  };

  // IdentityEquivalenceMapping
  eq_mapping_clone(add0.node(), add1.node());
  // AddSubEquivalenceMapping
  eq_mapping_clone(add0.node(), sub0.node());
  // ShiftEquivalenceMapping
  eq_mapping_clone(shra0.node(), shrl0.node());
  // ComparatorEquivalenceMapping
  eq_mapping_clone(eq0.node(), ne0.node());
  eq_mapping_clone(ult0.node(), uge0.node());
}

}  // namespace
}  // namespace xls
