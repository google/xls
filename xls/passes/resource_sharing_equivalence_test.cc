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

#include <cstdint>
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
#include "xls/common/status/ret_check.h"
#include "xls/common/status/status_macros.h"
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
using ::testing::Key;
using ::testing::UnorderedElementsAre;

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

// Applies `mapping` to `src` and uses ScopedVerifyEquivalence to confirm the
// mapped node is equivalent to `src`.
absl::StatusOr<Node*> VerifyEquivalence(Function* f, Node* src, Node* dst,
                                        const EquivalenceMapping& mapping) {
  XLS_RETURN_IF_ERROR(f->set_return_value(src));
  ScopedVerifyEquivalence sve(f);
  XLS_ASSIGN_OR_RETURN(std::vector<Node*> coerced,
                       mapping.ApplyToOperands(f, src->operands()));
  XLS_ASSIGN_OR_RETURN(Node * unified_node,
                       dst->CloneInNewFunction(coerced, f));
  XLS_ASSIGN_OR_RETURN(Node * output, mapping.ApplyToOutput(f, unified_node));
  XLS_RETURN_IF_ERROR(f->set_return_value(output));
  return output;
}

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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, mul_narrow.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Param("a")),
                                     m::ZeroExt(m::Param("b"))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, mul_narrow.node(), mul_wide.node(), *mapping));
    EXPECT_THAT(output, m::BitSlice(m::UMul(m::ZeroExt(), m::ZeroExt()),
                                    /*start=*/0, /*width=*/16));
  }

  // smul: narrow to wide
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {smul_narrow.node()}, smul_wide.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(smul_narrow.node());

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, smul_narrow.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::SignExt(m::Param("a")),
                                     m::SignExt(m::Param("b"))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, smul_narrow.node(), smul_wide.node(), *mapping));
    EXPECT_THAT(output, m::BitSlice(m::SMul(m::SignExt(), m::SignExt()),
                                    /*start=*/0, /*width=*/16));
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, add32.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::Param("a"), m::Neg(m::Param("b"))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, add32.node(), sub32.node(), *mapping));
    EXPECT_THAT(output, m::Sub(m::Param("a"), m::Neg(m::Param("b"))));
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, add_neg.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::Param("a"), m::Param("b")));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, add_neg.node(), sub32.node(), *mapping));
    EXPECT_THAT(output, m::Sub(m::Param("a"), m::Param("b")));
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, add16.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Param("a16")),
                                     m::ZeroExt(m::Neg(m::Param("b16")))));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, add16.node(), sub32.node(), *mapping));
    EXPECT_THAT(output,
                m::BitSlice(m::Sub(m::ZeroExt(), m::ZeroExt(m::Neg())), 0, 16));
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shll.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(rev_x, m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, shll.node(), shrl.node(), *mapping));
    EXPECT_THAT(output, m::Reverse(m::Shrl(rev_x, m::Param("s"))));
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shrl.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(rev_x, m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, shrl.node(), shll.node(), *mapping));
    EXPECT_THAT(output, m::Reverse(m::Shll(rev_x, m::Param("s"))));
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shra.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(x_xor_msb, m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, shra.node(), shrl.node(), *mapping));
    EXPECT_THAT(output, m::Xor(m::Shrl(x_xor_msb, m::Param("s")), msb_mask));
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shra.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(rev_x_xor_msb, m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, shra.node(), shll.node(), *mapping));
    EXPECT_THAT(
        output,
        m::Xor(m::Reverse(m::Shll(rev_x_xor_msb, m::Param("s"))), msb_mask));
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shra4.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(x_xor_msb), m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, shra4.node(), shrl8.node(), *mapping));
    EXPECT_THAT(output, m::Xor(m::BitSlice(m::Shrl(m::ZeroExt(), m::Param("s")),
                                           /*start=*/0, /*width=*/4),
                               msb_mask));
  }

  // shra (4-bit) -> shll (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shra4.node()}, shll8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shra4.node());

    XLS_ASSERT_OK(
        VerifyEquivalence(f, shra4.node(), shll8.node(), *mapping).status());
  }

  // shll (4-bit) -> shrl (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shll4.node()}, shrl8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shll4.node());

    XLS_ASSERT_OK(
        VerifyEquivalence(f, shll4.node(), shrl8.node(), *mapping).status());
  }

  // shrl (4-bit) -> shll (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shrl4.node()}, shll8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shrl4.node());

    XLS_ASSERT_OK(
        VerifyEquivalence(f, shrl4.node(), shll8.node(), *mapping).status());
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shrl4.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(m::Param("x")), m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, shrl4.node(), shra8.node(), *mapping));
    EXPECT_THAT(output, m::BitSlice(m::Shra(m::ZeroExt(), m::Param("s")),
                                    /*start=*/0, /*width=*/4));
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

    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping->ApplyToOperands(f, shll4.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::ZeroExt(rev_x), m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        VerifyEquivalence(f, shll4.node(), shra8.node(), *mapping));
    EXPECT_THAT(output, m::Reverse(m::BitSlice(
                            m::Shra(m::ZeroExt(rev_x), m::Param("s")),
                            /*start=*/0, /*width=*/4)));
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

  auto verify_equivalence =
      [&](Node* node,
          const EquivalenceMapping& mapping) -> absl::StatusOr<Node*> {
    XLS_RETURN_IF_ERROR(f->set_return_value(node));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> coerced,
                         mapping.ApplyToOperands(f, node->operands()));
    XLS_ASSIGN_OR_RETURN(Node * unified_node,
                         mapping.dst()->CloneInNewFunction(coerced, f));
    if (unified_node->BitCountOrDie() != 5) {
      return absl::InternalError("Unexpected bit count");
    }
    XLS_ASSIGN_OR_RETURN(Node * output, mapping.ApplyToOutput(f, unified_node));
    XLS_RETURN_IF_ERROR(f->set_return_value(output));
    return output;
  };

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
      XLS_ASSERT_OK_AND_ASSIGN(
          std::vector<Node*> coerced_from,
          from_mapping->ApplyToOperands(f, shrl.node()->operands()));
      EXPECT_THAT(coerced_from, ElementsAre(x_zeroext, m::Param("s")));
      XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                               verify_equivalence(shrl.node(), *from_mapping));
      EXPECT_THAT(output, m::BitSlice(m::Shra(x_zeroext, m::Param("s")),
                                      /*start=*/0, /*width=*/4));
    }

    // Verify equivalence for dst (shra mapped into 5-bit shra)
    {
      XLS_ASSERT_OK_AND_ASSIGN(
          std::vector<Node*> coerced_var,
          to_mapping->ApplyToOperands(f, shra.node()->operands()));
      EXPECT_THAT(coerced_var, ElementsAre(x_signext, m::Param("s")));
      XLS_ASSERT_OK_AND_ASSIGN(Node * var_output,
                               verify_equivalence(shra.node(), *to_mapping));
      EXPECT_THAT(var_output, m::BitSlice(m::Shra(x_signext, m::Param("s")),
                                          /*start=*/0, /*width=*/4));
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
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced_from,
        from_mapping->ApplyToOperands(f, shll.node()->operands()));
    EXPECT_THAT(coerced_from, ElementsAre(x_rev_zeroext, m::Param("s")));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             verify_equivalence(shll.node(), *from_mapping));
    EXPECT_THAT(output,
                m::Reverse(m::BitSlice(m::Shra(x_rev_zeroext, m::Param("s")),
                                       /*start=*/0, /*width=*/4)));
  }
}

TEST_F(ResourceSharingEquivalenceTest,
       ShiftEquivalenceMappingShiftAmountExtension) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(4));
  BValue s4 = fb.Param("s4", p->GetBitsType(4));
  BValue s6 = fb.Param("s6", p->GetBitsType(6));
  BValue s8 = fb.Param("s8", p->GetBitsType(8));
  BValue shll_s6 = fb.Shll(x, s6);
  BValue shrl_s4 = fb.Shrl(x, s4);
  BValue shrl_s8 = fb.Shrl(x, s8);
  BValue shra_s4 = fb.Shra(x, s4);
  BValue shra_s8 = fb.Shra(x, s8);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(shll_s6));

  auto verify_equivalence =
      [&](Node* node,
          const EquivalenceMapping& mapping) -> absl::StatusOr<Node*> {
    XLS_RETURN_IF_ERROR(f->set_return_value(node));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> coerced,
                         mapping.ApplyToOperands(f, node->operands()));
    XLS_ASSIGN_OR_RETURN(Node * unified_node,
                         mapping.dst()->CloneInNewFunction(coerced, f));
    XLS_ASSIGN_OR_RETURN(Node * output, mapping.ApplyToOutput(f, unified_node));
    XLS_RETURN_IF_ERROR(f->set_return_value(output));
    return output;
  };

  // shrl_s8 -> shra_s4: both MSB padding to 5-bit and shift extend to 8-bit
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shrl_s8.node()}, shra_s4.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& from_mapping =
        mappings->at(shrl_s8.node());
    const std::unique_ptr<EquivalenceMapping>& to_mapping =
        mappings->at(shra_s4.node());
    EXPECT_EQ(from_mapping->dst()->operand(0)->BitCountOrDie(), 5);
    EXPECT_EQ(from_mapping->dst()->operand(1)->BitCountOrDie(), 8);

    // Verify src (shrl_s8)
    auto x_zeroext5 = m::ZeroExt(m::Param("x"));
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced_from,
        from_mapping->ApplyToOperands(f, shrl_s8.node()->operands()));
    EXPECT_THAT(coerced_from, ElementsAre(x_zeroext5, m::Param("s8")));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output_from,
                             verify_equivalence(shrl_s8.node(), *from_mapping));
    EXPECT_THAT(output_from, m::BitSlice(m::Shra(x_zeroext5, m::Param("s8")),
                                         /*start=*/0, /*width=*/4));

    // Verify dst (shra_s4)
    auto x_signext5 = m::SignExt(m::Param("x"));
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced_to,
        to_mapping->ApplyToOperands(f, shra_s4.node()->operands()));
    EXPECT_THAT(coerced_to,
                ElementsAre(x_signext5, m::ZeroExt(m::Param("s4"))));
    XLS_ASSERT_OK_AND_ASSIGN(Node * output_to,
                             verify_equivalence(shra_s4.node(), *to_mapping));
    EXPECT_THAT(output_to,
                m::BitSlice(m::Shra(x_signext5, m::ZeroExt(m::Param("s4"))),
                            /*start=*/0, /*width=*/4));
  }

  // Multi-source: {shrl_s4, shra_s8} -> shll_s6
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings(
            {shrl_s4.node(), shra_s8.node()}, shll_s6.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& shrl_mapping =
        mappings->at(shrl_s4.node());
    const std::unique_ptr<EquivalenceMapping>& shra_mapping =
        mappings->at(shra_s8.node());
    const std::unique_ptr<EquivalenceMapping>& dst_mapping =
        mappings->at(shll_s6.node());

    EXPECT_EQ(shrl_mapping->dst()->operand(1)->BitCountOrDie(), 8);

    // Verify shrl_s4
    auto rev_x = m::Reverse(m::Param("x"));
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced_shrl,
        shrl_mapping->ApplyToOperands(f, shrl_s4.node()->operands()));
    EXPECT_THAT(coerced_shrl, ElementsAre(rev_x, m::ZeroExt(m::Param("s4"))));
    XLS_ASSERT_OK(verify_equivalence(shrl_s4.node(), *shrl_mapping).status());

    // Verify shra_s8
    auto msb_mask =
        m::SignExt(m::BitSlice(m::Param("x"), /*start=*/3, /*width=*/1));
    auto x_xor_msb = m::Xor(m::Param("x"), msb_mask);
    auto rev_x_xor_msb = m::Reverse(x_xor_msb);
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced_shra,
        shra_mapping->ApplyToOperands(f, shra_s8.node()->operands()));
    EXPECT_THAT(coerced_shra, ElementsAre(rev_x_xor_msb, m::Param("s8")));
    XLS_ASSERT_OK(verify_equivalence(shra_s8.node(), *shra_mapping).status());

    // Verify shll_s6 (dst)
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced_dst,
        dst_mapping->ApplyToOperands(f, shll_s6.node()->operands()));
    EXPECT_THAT(coerced_dst,
                ElementsAre(m::Param("x"), m::ZeroExt(m::Param("s6"))));
    XLS_ASSERT_OK(verify_equivalence(shll_s6.node(), *dst_mapping).status());
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
  auto test_swap = [&](Node* src, Node* dst,
                       double expected_overhead) -> absl::StatusOr<Node*> {
    XLS_ASSIGN_OR_RETURN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({src}, dst));
    XLS_RET_CHECK(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping = mappings->at(src);

    XLS_ASSIGN_OR_RETURN(
        double overhead,
        mapping->EstimateAreaOverhead(area_estimator, src->operands(), src));
    EXPECT_EQ(overhead, expected_overhead);

    XLS_RETURN_IF_ERROR(f->set_return_value(src));
    ScopedVerifyEquivalence sve(f);
    absl::Span<Node* const> src_ops = src->operands();
    XLS_ASSIGN_OR_RETURN(std::vector<Node*> coerced,
                         mapping->ApplyToOperands(f, src_ops));
    // Confirm the mapping knows when it has to transform operands.
    XLS_RET_CHECK_EQ(coerced.size(), src_ops.size());
    for (int i = 0; i < coerced.size(); ++i) {
      XLS_ASSIGN_OR_RETURN(bool requires_transform,
                           mapping->RequiresOperandTransformation());
      EXPECT_EQ(coerced[i] != src_ops[i], requires_transform);
    }
    XLS_ASSIGN_OR_RETURN(Node * new_dst, dst->CloneInNewFunction(coerced, f));
    XLS_ASSIGN_OR_RETURN(Node * output, mapping->ApplyToOutput(f, new_dst));
    // Confirm the mapping knows when it has to transform the output.
    XLS_ASSIGN_OR_RETURN(bool requires_output_transform,
                         mapping->RequiresOutputTransformation());
    EXPECT_EQ(output != new_dst, requires_output_transform);
    XLS_RETURN_IF_ERROR(f->set_return_value(output));
    return output;
  };

  auto swap_msb = [](auto lsb_param, auto msb_param, int64_t width) {
    return m::Concat(m::BitSlice(msb_param, width - 1, 1),
                     m::BitSlice(lsb_param, 0, width - 1));
  };
  auto a_msbswap = swap_msb(m::Param("a"), m::Param("b"), 32);
  auto b_msbswap = swap_msb(m::Param("b"), m::Param("a"), 32);
  auto a16_msbswap = m::ZeroExt(swap_msb(m::Param("a16"), m::Param("b16"), 16));
  auto b16_msbswap = m::ZeroExt(swap_msb(m::Param("b16"), m::Param("a16"), 16));
  Node* output = nullptr;

  // Test unsigned inequality swaps:
  // ult -> ugt: swap operands (overhead 0.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult.node(), ugt.node(), 0.0));
  EXPECT_THAT(output, m::UGt(m::Param("b"), m::Param("a")));
  // ult -> ule: swap operands + invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult.node(), ule.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::ULe(m::Param("b"), m::Param("a"))));
  // ult -> uge: invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult.node(), uge.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::UGe(m::Param("a"), m::Param("b"))));
  // ult16 -> ugt (with zero extension): swap operands + extend (overhead 0.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult16.node(), ugt.node(), 0.0));
  EXPECT_THAT(output,
              m::UGt(m::ZeroExt(m::Param("b16")), m::ZeroExt(m::Param("a16"))));
  // ult16 -> uge (with zero extension): extend + invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult16.node(), uge.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::UGe(m::ZeroExt(m::Param("a16")),
                                    m::ZeroExt(m::Param("b16")))));

  // Test signed inequality swaps:
  // slt -> sgt: swap operands (overhead 0.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt.node(), sgt.node(), 0.0));
  EXPECT_THAT(output, m::SGt(m::Param("b"), m::Param("a")));
  // slt -> sle: swap operands + invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt.node(), sle.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::SLe(m::Param("b"), m::Param("a"))));
  // slt -> sge: invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt.node(), sge.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::SGe(m::Param("a"), m::Param("b"))));
  // slt16 -> sgt (with sign extension): swap operands + extend (overhead 0.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt16.node(), sgt.node(), 0.0));
  EXPECT_THAT(output,
              m::SGt(m::SignExt(m::Param("b16")), m::SignExt(m::Param("a16"))));
  // slt16 -> sge (with sign extension): extend + invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt16.node(), sge.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::SGe(m::SignExt(m::Param("a16")),
                                    m::SignExt(m::Param("b16")))));

  // Test signed <-> unsigned inequality swaps:
  // ult -> slt: swap MSBs (overhead 0.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult.node(), slt.node(), 0.0));
  EXPECT_THAT(output, m::SLt(a_msbswap, b_msbswap));
  // ult -> sle: swap operands + swap MSBs + invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult.node(), sle.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::SLe(b_msbswap, a_msbswap)));
  // ult16 -> sgt (with zero extension): swap operands + extend (overhead 0.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult16.node(), sgt.node(), 0.0));
  EXPECT_THAT(output,
              m::SGt(m::ZeroExt(m::Param("b16")), m::ZeroExt(m::Param("a16"))));
  // ult16 -> sge (with zero extension): extend + invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult16.node(), sge.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::SGe(m::ZeroExt(m::Param("a16")),
                                    m::ZeroExt(m::Param("b16")))));
  // slt -> ult: swap MSBs (overhead 0.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt.node(), ult.node(), 0.0));
  EXPECT_THAT(output, m::ULt(a_msbswap, b_msbswap));
  // slt -> uge: swap MSBs + invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt.node(), uge.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::UGe(a_msbswap, b_msbswap)));
  // slt16 -> ugt: swap operands + swap MSBs + extend (overhead 0.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt16.node(), ugt.node(), 0.0));
  EXPECT_THAT(output, m::UGt(b16_msbswap, a16_msbswap));
  // slt16 -> ule: swap operands + swap MSBs + extend + invert output (1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt16.node(), ule.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::ULe(b16_msbswap, a16_msbswap)));

  // Test equality swaps:
  // eq -> ne: invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(eq.node(), ne.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::Ne(m::Param("a"), m::Param("b"))));
  // ne -> eq: invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ne.node(), eq.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::Eq(m::Param("a"), m::Param("b"))));
  // eq16 -> ne (with zero extension): extend + invert output (overhead 1.0)
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(eq16.node(), ne.node(), 1.0));
  EXPECT_THAT(output, m::Not(m::Ne(m::ZeroExt(m::Param("a16")),
                                   m::ZeroExt(m::Param("b16")))));

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

    // Narrowing unsupported (32-bit -> 16-bit)
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> ult_ult16,
        GetNodeEquivalenceMapper().ComputeMappings({ult.node()}, ult16.node()));
    EXPECT_FALSE(ult_ult16.has_value());
  }
}

TEST_F(ResourceSharingEquivalenceTest, CompareToArithEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));

  // Subtraction destinations
  BValue sub32 = fb.Subtract(a, b);
  BValue add32 = fb.Add(a, b);

  // 16-bit unsigned inequalities
  BValue ult16 = fb.ULt(a16, b16);
  BValue ugt16 = fb.UGt(a16, b16);
  BValue ule16 = fb.ULe(a16, b16);
  BValue uge16 = fb.UGe(a16, b16);

  // 16-bit signed inequalities
  BValue slt16 = fb.SLt(a16, b16);
  BValue sgt16 = fb.SGt(a16, b16);
  BValue sle16 = fb.SLe(a16, b16);
  BValue sge16 = fb.SGe(a16, b16);

  // 16-bit equalities
  BValue eq16 = fb.Eq(a16, b16);
  BValue ne16 = fb.Ne(a16, b16);

  // Default return value to be overwritten by each test case.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(eq16));
  FakeAreaEstimator area_estimator;

  // Helper to test swapping comparator nodes mapped to subtraction in the tuple
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

  // Test unsigned inequality mappings to subtraction (tuple element 0):
  // ult16 -> sub32 (with zero extension): extend (overhead 0.0)
  test_swap(ult16.node(), sub32.node(), /*tuple_index=*/0, 0.0);
  // ugt16 -> sub32 (with zero extension): swap operands + extend (overhead 0.0)
  test_swap(ugt16.node(), sub32.node(), /*tuple_index=*/0, 0.0);
  // ule16 -> sub32 (with zero extension): swap operands + extend + invert
  // output (overhead 1.0)
  test_swap(ule16.node(), sub32.node(), /*tuple_index=*/0, 1.0);
  // uge16 -> sub32 (with zero extension): extend + invert output (overhead 1.0)
  test_swap(uge16.node(), sub32.node(), /*tuple_index=*/0, 1.0);

  // Test signed inequality mappings to subtraction (tuple element 1):
  // slt16 -> sub32 (with sign extension): extend (overhead 0.0)
  test_swap(slt16.node(), sub32.node(), /*tuple_index=*/1, 0.0);
  // sgt16 -> sub32 (with sign extension): swap operands + extend (overhead 0.0)
  test_swap(sgt16.node(), sub32.node(), /*tuple_index=*/1, 0.0);
  // sle16 -> sub32 (with sign extension): swap operands + extend + invert
  // output (overhead 1.0)
  test_swap(sle16.node(), sub32.node(), /*tuple_index=*/1, 1.0);
  // sge16 -> sub32 (with sign extension): extend + invert output (overhead 1.0)
  test_swap(sge16.node(), sub32.node(), /*tuple_index=*/1, 1.0);

  // Incompatible compare-to-arith mappings
  {
    // Equality -> Subtraction
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> eq_sub,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {eq16.node()}, sub32.node()));
    EXPECT_FALSE(eq_sub.has_value());

    // Inequality (!=) -> Subtraction
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> ne_sub,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {ne16.node()}, sub32.node()));
    EXPECT_FALSE(ne_sub.has_value());

    // Comparison -> Addition
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> ult_add,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {ult16.node()}, add32.node()));
    EXPECT_FALSE(ult_add.has_value());
  }
}

TEST_F(ResourceSharingEquivalenceTest,
       CompareToArithEquivalenceMappingWidening) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));

  BValue sub32 = fb.Subtract(a, b);
  BValue sub16 = fb.Subtract(a16, b16);

  // 32-bit unsigned inequalities
  BValue ult32 = fb.ULt(a, b);
  BValue ugt32 = fb.UGt(a, b);
  BValue ule32 = fb.ULe(a, b);
  BValue uge32 = fb.UGe(a, b);

  // 32-bit signed inequalities
  BValue slt32 = fb.SLt(a, b);
  BValue sgt32 = fb.SGt(a, b);
  BValue sle32 = fb.SLe(a, b);
  BValue sge32 = fb.SGe(a, b);

  // 16-bit unsigned inequality
  BValue ult16 = fb.ULt(a16, b16);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sub32));
  FakeAreaEstimator area_estimator;

  auto verify_mapping = [&](Node* node, const EquivalenceMapping& mapping,
                            int64_t expected_target_width) {
    XLS_ASSERT_OK(f->set_return_value(node));
    ScopedVerifyEquivalence sve(f);
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> coerced,
                             mapping.ApplyToOperands(f, node->operands()));
    XLS_ASSERT_OK_AND_ASSIGN(Node * unified_node,
                             mapping.dst()->CloneInNewFunction(coerced, f));
    EXPECT_EQ(unified_node->BitCountOrDie(), expected_target_width);
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             mapping.ApplyToOutput(f, unified_node));
    XLS_ASSERT_OK(f->set_return_value(output));
  };

  auto test_widened_mapping = [&](Node* src, Node* dst,
                                  int64_t expected_target_width,
                                  double expected_overhead) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({src}, dst));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& from_mapping = mappings->at(src);
    const std::unique_ptr<EquivalenceMapping>& to_mapping = mappings->at(dst);

    XLS_ASSERT_OK_AND_ASSIGN(double overhead,
                             from_mapping->EstimateAreaOverhead(
                                 area_estimator, src->operands(), src));
    EXPECT_EQ(overhead, expected_overhead);

    // Verify equivalence for src and dst
    verify_mapping(src, *from_mapping, expected_target_width);
    verify_mapping(dst, *to_mapping, expected_target_width);
  };

  // 32-bit unsigned compares mapped into 32-bit sub (widened to 33 bits):
  test_widened_mapping(ult32.node(), sub32.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/0.0);
  test_widened_mapping(ugt32.node(), sub32.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/0.0);
  test_widened_mapping(ule32.node(), sub32.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/1.0);
  test_widened_mapping(uge32.node(), sub32.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/1.0);

  // 32-bit signed compares mapped into 32-bit sub (widened to 33 bits):
  test_widened_mapping(slt32.node(), sub32.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/0.0);
  test_widened_mapping(sgt32.node(), sub32.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/0.0);
  test_widened_mapping(sle32.node(), sub32.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/1.0);
  test_widened_mapping(sge32.node(), sub32.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/1.0);

  // 32-bit compare mapped into 16-bit sub (widened to 33 bits):
  test_widened_mapping(ult32.node(), sub16.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/0.0);
  test_widened_mapping(slt32.node(), sub16.node(), /*expected_target_width=*/33,
                       /*expected_overhead=*/0.0);

  // 16-bit compare mapped into 16-bit sub (widened to 17 bits):
  test_widened_mapping(ult16.node(), sub16.node(), /*expected_target_width=*/17,
                       /*expected_overhead=*/0.0);

  // Multi-source mapping with widening: {ult16, slt32} -> sub16 (widened to 33
  // bits)
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {ult16.node(), slt32.node()}, sub16.node()));
    ASSERT_TRUE(mappings.has_value());
    EXPECT_THAT(*mappings,
                UnorderedElementsAre(Key(ult16.node()), Key(slt32.node()),
                                     Key(sub16.node())));

    // Verify equivalence for all three
    for (Node* node : {ult16.node(), slt32.node(), sub16.node()}) {
      verify_mapping(node, *mappings->at(node), /*expected_target_width=*/33);
    }
  }
}

TEST_F(ResourceSharingEquivalenceTest, CloneEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));
  BValue add0 = fb.Add(a, b);
  BValue add1 = fb.Add(a, b);
  BValue sub0 = fb.Subtract(a, b);
  BValue shra0 = fb.Shra(a, b);
  BValue shrl0 = fb.Shrl(a, b);
  BValue shrl_b16 = fb.Shrl(a, b16);
  BValue eq0 = fb.Eq(a, b);
  BValue ne0 = fb.Ne(a, b);
  BValue ult0 = fb.ULt(a, b);
  BValue uge0 = fb.UGe(a, b);
  BValue ult16 = fb.ULt(a16, b16);
  BValue sle16 = fb.SLe(a16, b16);
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
    EXPECT_EQ(clone_same->dst(), mappings->at(src)->dst());

    // Illogical remapping is accepted; we trust the user's node pairings.
    absl::flat_hash_map<Node*, Node*> remapping = {
        {src, dst},
        {dst, src},
    };
    XLS_ASSERT_OK_AND_ASSIGN(std::unique_ptr<EquivalenceMapping> clone_remapped,
                             mappings->at(src)->Clone(&remapping));
    EXPECT_EQ(clone_remapped->src(), dst);
    Node* expected_remapped_dst =
        (mappings->at(src)->dst() == dst) ? src : mappings->at(src)->dst();
    EXPECT_EQ(clone_remapped->dst(), expected_remapped_dst);

    // Missing node in non-empty remapping returns an error.
    absl::flat_hash_map<Node*, Node*> incomplete_remapping = {
        {dst, src},
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
  eq_mapping_clone(shrl_b16.node(), shra0.node());
  // ComparatorEquivalenceMapping
  eq_mapping_clone(eq0.node(), ne0.node());
  eq_mapping_clone(ult0.node(), uge0.node());
  // CompareToArithEquivalenceMapping
  eq_mapping_clone(ult16.node(), sub0.node());
  eq_mapping_clone(sle16.node(), sub0.node());
  eq_mapping_clone(ult0.node(), sub0.node());
}

}  // namespace
}  // namespace xls
