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
#include <string>
#include <utility>
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
#include "xls/estimators/area_model/area_estimators.h"
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
using ::testing::AllOf;
using ::testing::ElementsAre;
using ::testing::Key;
using ::testing::Pointee;
using ::testing::Property;
using ::testing::UnorderedElementsAre;
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
    if (node->op() == Op::kSub) {
      return 100.0 * node->BitCountOrDie();
    }
    if (node->OpIn({Op::kUMul, Op::kSMul})) {
      return node->operand(0)->BitCountOrDie() *
             node->operand(1)->BitCountOrDie();
    }
    return 1.0;
  }
};

// Applies `mapping` to `src` and uses ScopedVerifyEquivalence to confirm the
// mapped node is equivalent to `src`.
absl::StatusOr<Node*> VerifyEquivalence(Function* f, Node* src,
                                        const EquivalenceMapping& mapping) {
  XLS_RETURN_IF_ERROR(f->set_return_value(src));
  ScopedVerifyEquivalence sve(f);
  XLS_ASSIGN_OR_RETURN(Node * mapped, CloneWithMapping(src, mapping));
  XLS_RETURN_IF_ERROR(f->set_return_value(mapped));
  return mapped;
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, mul_narrow.node(), *mapping));
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
        Node * output, VerifyEquivalence(f, smul_narrow.node(), *mapping));
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, add32.node(), *mapping));
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, add_neg.node(), *mapping));
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, add16.node(), *mapping));
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, shll.node(), *mapping));
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, shrl.node(), *mapping));
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, shra.node(), *mapping));
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, shra.node(), *mapping));
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, shra4.node(), *mapping));
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

    XLS_ASSERT_OK(VerifyEquivalence(f, shra4.node(), *mapping).status());
  }

  // shll (4-bit) -> shrl (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shll4.node()}, shrl8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shll4.node());

    XLS_ASSERT_OK(VerifyEquivalence(f, shll4.node(), *mapping).status());
  }

  // shrl (4-bit) -> shll (8-bit) widening
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {shrl4.node()}, shll8.node()));
    ASSERT_TRUE(mappings.has_value());
    const std::unique_ptr<EquivalenceMapping>& mapping =
        mappings->at(shrl4.node());

    XLS_ASSERT_OK(VerifyEquivalence(f, shrl4.node(), *mapping).status());
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, shrl4.node(), *mapping));
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
    XLS_ASSERT_OK_AND_ASSIGN(Node * output,
                             VerifyEquivalence(f, shll4.node(), *mapping));
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

  // 32-bit inequalities
  BValue ult32 = fb.ULt(a, b);
  BValue ugt32 = fb.UGt(a, b);
  BValue ule32 = fb.ULe(a, b);
  BValue uge32 = fb.UGe(a, b);
  BValue slt32 = fb.SLt(a, b);
  BValue sgt32 = fb.SGt(a, b);
  BValue sle32 = fb.SLe(a, b);
  BValue sge32 = fb.SGe(a, b);

  // 16-bit inequalities
  BValue ult16 = fb.ULt(a16, b16);
  BValue ugt16 = fb.UGt(a16, b16);
  BValue ule16 = fb.ULe(a16, b16);
  BValue uge16 = fb.UGe(a16, b16);
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

  // Helper to test swapping comparator nodes mapped to subtraction.
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

  auto msb = [](auto node_matcher, int64_t width = 32) {
    return m::BitSlice(node_matcher, width - 1, 1);
  };
  auto msb_a = msb(m::Param("a"));
  auto msb_b = msb(m::Param("b"));
  auto msb_a_sub_b = msb(m::Sub(m::Param("a"), m::Param("b")));
  auto msb_b_sub_a = msb(m::Sub(m::Param("b"), m::Param("a")));
  Node* output = nullptr;

  // 32-bit unsigned compares mapped into 32-bit sub
  // overhead: 5.0 xor + 1.0 sel = 6.0, + 1.0 if inverted = 7.0
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult32.node(), sub32.node(), 6.0));
  EXPECT_THAT(output,
              m::Select(m::Xor(msb_a, msb_b), /*cases=*/{msb_a_sub_b, msb_b}));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ugt32.node(), sub32.node(), 6.0));
  EXPECT_THAT(output,
              m::Select(m::Xor(msb_b, msb_a), /*cases=*/{msb_b_sub_a, msb_a}));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ule32.node(), sub32.node(), 7.0));
  EXPECT_THAT(output, m::Not(m::Select(m::Xor(msb_b, msb_a),
                                       /*cases=*/{msb_b_sub_a, msb_a})));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(uge32.node(), sub32.node(), 7.0));
  EXPECT_THAT(output, m::Not(m::Select(m::Xor(msb_a, msb_b),
                                       /*cases=*/{msb_a_sub_b, msb_b})));

  // 32-bit signed compares mapped into 32-bit sub
  // overhead: 6.0, or 7.0 if inverted
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt32.node(), sub32.node(), 6.0));
  EXPECT_THAT(output,
              m::Select(m::Xor(msb_a, msb_b), /*cases=*/{msb_a_sub_b, msb_a}));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(sgt32.node(), sub32.node(), 6.0));
  EXPECT_THAT(output,
              m::Select(m::Xor(msb_b, msb_a), /*cases=*/{msb_b_sub_a, msb_b}));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(sle32.node(), sub32.node(), 7.0));
  EXPECT_THAT(output, m::Not(m::Select(m::Xor(msb_b, msb_a),
                                       /*cases=*/{msb_b_sub_a, msb_b})));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(sge32.node(), sub32.node(), 7.0));
  EXPECT_THAT(output, m::Not(m::Select(m::Xor(msb_a, msb_b),
                                       /*cases=*/{msb_a_sub_b, msb_a})));

  auto a16_m = m::Param("a16");
  auto b16_m = m::Param("b16");

  // 16-bit unsigned inequality mappings to 32-bit subtraction (with zeroext):
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ult16.node(), sub32.node(), 0.0));
  EXPECT_THAT(output, msb(m::Sub(m::ZeroExt(a16_m), m::ZeroExt(b16_m))));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ugt16.node(), sub32.node(), 0.0));
  EXPECT_THAT(output, msb(m::Sub(m::ZeroExt(b16_m), m::ZeroExt(a16_m))));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(ule16.node(), sub32.node(), 1.0));
  EXPECT_THAT(output,
              m::Not(msb(m::Sub(m::ZeroExt(b16_m), m::ZeroExt(a16_m)))));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(uge16.node(), sub32.node(), 1.0));
  EXPECT_THAT(output,
              m::Not(msb(m::Sub(m::ZeroExt(a16_m), m::ZeroExt(b16_m)))));

  // 16-bit signed inequality mappings to 32-bit subtraction (with signext):
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(slt16.node(), sub32.node(), 0.0));
  EXPECT_THAT(output, msb(m::Sub(m::SignExt(a16_m), m::SignExt(b16_m))));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(sgt16.node(), sub32.node(), 0.0));
  EXPECT_THAT(output, msb(m::Sub(m::SignExt(b16_m), m::SignExt(a16_m))));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(sle16.node(), sub32.node(), 1.0));
  EXPECT_THAT(output,
              m::Not(msb(m::Sub(m::SignExt(b16_m), m::SignExt(a16_m)))));
  XLS_ASSERT_OK_AND_ASSIGN(output, test_swap(sge16.node(), sub32.node(), 1.0));
  EXPECT_THAT(output,
              m::Not(msb(m::Sub(m::SignExt(a16_m), m::SignExt(b16_m)))));

  // Multi-source mapping without widening: {ult16, slt32} -> sub32
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {ult16.node(), slt32.node()}, sub32.node()));
    ASSERT_TRUE(mappings.has_value());
    EXPECT_THAT(*mappings,
                UnorderedElementsAre(Key(ult16.node()), Key(slt32.node())));
    XLS_ASSERT_OK(
        VerifyEquivalence(f, ult16.node(), *mappings->at(ult16.node()))
            .status());
    XLS_ASSERT_OK(
        VerifyEquivalence(f, slt32.node(), *mappings->at(slt32.node()))
            .status());
  }

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

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sub16));
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

  // 32-bit unsigned compares mapped into 16-bit sub (widened to 32 bits):
  // overhead: (32 - 16) * 100.0 sub diff + 5.0 xor + 1.0 sel, +1 if inverted
  test_widened_mapping(ult32.node(), sub16.node(), /*expected_target_width=*/32,
                       /*expected_overhead=*/1606.0);
  test_widened_mapping(ugt32.node(), sub16.node(), /*expected_target_width=*/32,
                       /*expected_overhead=*/1606.0);
  test_widened_mapping(ule32.node(), sub16.node(), /*expected_target_width=*/32,
                       /*expected_overhead=*/1607.0);
  test_widened_mapping(uge32.node(), sub16.node(), /*expected_target_width=*/32,
                       /*expected_overhead=*/1607.0);

  // 32-bit signed compares mapped into 16-bit sub (widened to 32 bits):
  test_widened_mapping(slt32.node(), sub16.node(), /*expected_target_width=*/32,
                       /*expected_overhead=*/1606.0);
  test_widened_mapping(sgt32.node(), sub16.node(), /*expected_target_width=*/32,
                       /*expected_overhead=*/1606.0);
  test_widened_mapping(sle32.node(), sub16.node(), /*expected_target_width=*/32,
                       /*expected_overhead=*/1607.0);
  test_widened_mapping(sge32.node(), sub16.node(), /*expected_target_width=*/32,
                       /*expected_overhead=*/1607.0);

  // Multi-source mapping with widening: {ult16, slt32} -> sub16 (widened to 32
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
      verify_mapping(node, *mappings->at(node), /*expected_target_width=*/32);
    }
  }
}

TEST_F(ResourceSharingEquivalenceTest, MultiplySplitterEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a4 = fb.Param("a4", p->GetBitsType(4));
  BValue b4 = fb.Param("b4", p->GetBitsType(4));
  BValue a3 = fb.Param("a3", p->GetBitsType(3));
  BValue b2 = fb.Param("b2", p->GetBitsType(2));
  BValue umul8 = fb.UMul(a3, b2, /*result_width=*/8);
  BValue smul8 = fb.SMul(a4, b4, /*result_width=*/8);
  BValue umul4 = fb.UMul(a4, b4, /*result_width=*/4);
  BValue smul4 = fb.SMul(a3, b2, /*result_width=*/4);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(umul8));

  // Too small width to decompose:
  EXPECT_THAT(GetNodeEquivalenceMapper().ComputeDecompositions(umul4.node()),
              IsOkAndHolds(testing::IsEmpty()));
  EXPECT_THAT(GetNodeEquivalenceMapper().ComputeDecompositions(smul4.node()),
              IsOkAndHolds(testing::IsEmpty()));

  // 8 bit umul decomposes into four 4 bit umul parts.
  XLS_ASSERT_OK_AND_ASSIGN(AreaEstimator * area_estimator,
                           GetAreaEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<EquivalenceMapping>> umul_decompositions,
      GetNodeEquivalenceMapper().ComputeDecompositions(umul8.node()));
  ASSERT_EQ(umul_decompositions.size(), 1);
  EXPECT_THAT(
      umul_decompositions[0],
      Pointee(AllOf(Property(&EquivalenceMapping::RequiresOperandTransformation,
                             IsOkAndHolds(true)),
                    Property(&EquivalenceMapping::RequiresOutputTransformation,
                             IsOkAndHolds(false)))));
  // 4 umuls + 3 zero_exts + 3 concats + 3 adds (13) - 1 orig = 12.0.
  EXPECT_THAT(umul_decompositions[0]->EstimateAreaOverhead(
                  *area_estimator, umul8.node()->operands(), umul8.node()),
              IsOkAndHolds(12.0));
  EXPECT_THAT(umul_decompositions[0]->EstimateAreaOverhead(*area_estimator, {},
                                                           nullptr),
              IsOkAndHolds(12.0));
  XLS_ASSERT_OK(VerifyEquivalence(f, umul8.node(), *umul_decompositions[0]));

  // 8 bit smul decomposes into four 4 bit smul parts.
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<std::shared_ptr<EquivalenceMapping>> smul_decompositions,
      GetNodeEquivalenceMapper().ComputeDecompositions(smul8.node()));
  ASSERT_EQ(smul_decompositions.size(), 1);
  // 4 smuls + 5 sign_exts + 8 concats + 2 bit_slices + 2 selects +
  // 1 and + 1 zero_ext + 5 adds + 1 sub (29) - 1 orig = 28.0.
  EXPECT_THAT(smul_decompositions[0]->EstimateAreaOverhead(
                  *area_estimator, smul8.node()->operands(), smul8.node()),
              IsOkAndHolds(28.0));
  XLS_ASSERT_OK(VerifyEquivalence(f, smul8.node(), *smul_decompositions[0]));
}

TEST_F(ResourceSharingEquivalenceTest, CloneEquivalenceMapping) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));
  BValue a17 = fb.Param("a17", p->GetBitsType(17));
  BValue add0 = fb.Add(a, b);
  BValue add1 = fb.Add(a, b);
  BValue sub0 = fb.Subtract(a, b);
  BValue sub16 = fb.Subtract(a16, b16);
  BValue shra0 = fb.Shra(a, b);
  BValue shrl0 = fb.Shrl(a, b);
  BValue shrl_b16 = fb.Shrl(a, b16);
  BValue eq0 = fb.Eq(a, b);
  BValue ne0 = fb.Ne(a, b);
  BValue ult0 = fb.ULt(a, b);
  BValue uge0 = fb.UGe(a, b);
  BValue ult16 = fb.ULt(a16, b16);
  BValue sle16 = fb.SLe(a16, b16);
  BValue umul16 = fb.UMul(a16, b16, 32);
  BValue smul16 = fb.SMul(a16, b16, 32);
  BValue smul_16_17 = fb.SMul(a16, a17, 33);
  BValue smul_17_16 = fb.SMul(a17, a16, 33);
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
                             mappings->at(src)->Clone(remapping));
    EXPECT_EQ(clone_remapped->src(), dst);
    Node* expected_remapped_dst =
        (mappings->at(src)->dst() == dst) ? src : mappings->at(src)->dst();
    EXPECT_EQ(clone_remapped->dst(), expected_remapped_dst);

    // Missing node in non-empty remapping returns an error.
    absl::flat_hash_map<Node*, Node*> incomplete_remapping = {
        {dst, src},
    };
    EXPECT_THAT(mappings->at(src)->Clone(incomplete_remapping),
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
  eq_mapping_clone(ult0.node(), sub16.node());
  // MulEquivalenceMapping
  eq_mapping_clone(umul16.node(), smul16.node());
  eq_mapping_clone(smul_17_16.node(), smul_16_17.node());
}

TEST_F(ResourceSharingEquivalenceTest, CloneMultiplySplitter) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));
  BValue umul32 = fb.UMul(a16, b16, /*result_width=*/32);
  BValue smul32 = fb.SMul(a16, b16, /*result_width=*/32);
  XLS_ASSERT_OK(fb.BuildWithReturnValue(umul32).status());

  for (Node* node : {umul32.node(), smul32.node()}) {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<std::shared_ptr<EquivalenceMapping>> decomp_mappings,
        GetNodeEquivalenceMapper().ComputeDecompositions(node));
    ASSERT_EQ(decomp_mappings.size(), 1);
    EXPECT_THAT(
        decomp_mappings[0]->Clone(std::nullopt),
        IsOkAndHolds(Pointee(AllOf(
            Property(&EquivalenceMapping::src, node),
            Property(&EquivalenceMapping::dst, decomp_mappings[0]->dst())))));
  }
}

TEST_F(ResourceSharingEquivalenceTest, ApplyToOperandsReusesExistingNodes) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));
  BValue a32 = fb.Param("a32", p->GetBitsType(32));
  BValue b32 = fb.Param("b32", p->GetBitsType(32));

  BValue add16 = fb.Add(a16, b16);
  BValue sub32 = fb.Subtract(a32, b32);
  BValue shll16 = fb.Shll(a16, b16);
  BValue shra32 = fb.Shra(a32, b32);
  BValue shrl32 = fb.Shrl(a32, b32);
  BValue ult32 = fb.ULt(a32, b32);
  BValue slt32 = fb.SLt(a32, b32);
  BValue smul16 = fb.SMul(a16, b16);
  BValue smul32 = fb.SMul(a32, b32);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(sub32));

  // Applies the `src` -> `dst` mapping twice. The first application must
  // coerce at least one operand; the second must return the same nodes without
  // adding any to `f`.
  auto expect_reuse = [&](Node* src, Node* dst) {
    SCOPED_TRACE(src->GetName() + " -> " + dst->GetName());
    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({src}, dst));
    ASSERT_TRUE(mappings.has_value());
    const EquivalenceMapping& mapping = *mappings->at(src);
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> first,
                             mapping.ApplyToOperands(f, src->operands()));
    EXPECT_NE(first, std::vector<Node*>(src->operands().begin(),
                                        src->operands().end()));
    int64_t node_count = f->node_count();
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<Node*> second,
                             mapping.ApplyToOperands(f, src->operands()));
    EXPECT_EQ(first, second);
    EXPECT_EQ(f->node_count(), node_count);
  };

  // neg, zero_ext
  expect_reuse(add16.node(), sub32.node());
  // reverse, zero_ext
  expect_reuse(shll16.node(), shra32.node());
  // bit_slice, sign_ext mask, xor
  expect_reuse(shra32.node(), shrl32.node());
  // bit_slice, concat (MSB swap)
  expect_reuse(ult32.node(), slt32.node());
  // sign_ext
  expect_reuse(smul16.node(), smul32.node());
}

TEST_F(ResourceSharingEquivalenceTest,
       ApplyToOperandsReusesNodesAlreadyInGraph) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));
  BValue a32 = fb.Param("a32", p->GetBitsType(32));
  BValue b32 = fb.Param("b32", p->GetBitsType(32));

  BValue sub16 = fb.Subtract(a16, b16);
  BValue add32 = fb.Add(a32, b32);
  BValue smul16 = fb.SMul(a16, b16);
  BValue smul32 = fb.SMul(a32, b32);
  // Coercions already present in the graph before any mapping is applied.
  BValue zext_a16 = fb.ZeroExtend(a16, 32);
  BValue zext_neg_b16 = fb.ZeroExtend(fb.Negate(b16), 32);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(add32));

  // sub16 -> add32 needs zero_ext(a16) and zero_ext(neg(b16)); both exist.
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {sub16.node()}, add32.node()));
    ASSERT_TRUE(mappings.has_value());
    const EquivalenceMapping& mapping = *mappings->at(sub16.node());
    int64_t node_count = f->node_count();
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping.ApplyToOperands(f, sub16.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(zext_a16.node(), zext_neg_b16.node()));
    EXPECT_EQ(f->node_count(), node_count);
  }

  // smul16 -> smul32 needs sign_ext; the existing zero_ext(a16) must not be
  // reused.
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {smul16.node()}, smul32.node()));
    ASSERT_TRUE(mappings.has_value());
    const EquivalenceMapping& mapping = *mappings->at(smul16.node());
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<Node*> coerced,
        mapping.ApplyToOperands(f, smul16.node()->operands()));
    EXPECT_THAT(coerced, ElementsAre(m::SignExt(m::Param("a16")),
                                     m::SignExt(m::Param("b16"))));
  }
}

TEST_F(ResourceSharingEquivalenceTest,
       CloneWithMappingPopulatesOriginalNodeToClone) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));
  BValue sub16 = fb.Subtract(a16, b16);
  BValue add16 = fb.Add(a16, b16);
  BValue umul32 = fb.UMul(a16, b16, /*result_width=*/32);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(umul32));

  // Mapping whose destination lives in the same function (dst_fn == f)
  {
    XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                             GetNodeEquivalenceMapper().ComputeMappings(
                                 {sub16.node()}, add16.node()));
    ASSERT_TRUE(mappings.has_value());
    absl::flat_hash_map<Node*, Node*> original_node_to_clone;
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        CloneWithMapping(sub16.node(), *mappings->at(sub16.node()),
                         original_node_to_clone));
    ASSERT_TRUE(original_node_to_clone.contains(add16.node()));
    EXPECT_EQ(original_node_to_clone.at(add16.node()), output);
    EXPECT_THAT(output, m::Add(a16.node(), m::Neg(b16.node())));
  }

  // Decomposition mapping whose destination lives in a temporary function
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        std::vector<std::shared_ptr<EquivalenceMapping>> decomps,
        GetNodeEquivalenceMapper().ComputeDecompositions(umul32.node()));
    ASSERT_EQ(decomps.size(), 1);

    absl::flat_hash_map<Node*, Node*> original_node_to_clone;
    XLS_ASSERT_OK_AND_ASSIGN(
        Node * output,
        CloneWithMapping(umul32.node(), *decomps[0], original_node_to_clone));
    FunctionBase* dst_fn = decomps[0]->dst()->function_base();
    std::vector<Node*> cloned_params;
    for (Param* param : dst_fn->params()) {
      cloned_params.push_back(original_node_to_clone.at(param));
    }
    EXPECT_THAT(cloned_params,
                ElementsAre(m::BitSlice(a16.node(), /*start=*/8, /*width=*/8),
                            m::BitSlice(a16.node(), /*start=*/0, /*width=*/8),
                            m::BitSlice(b16.node(), /*start=*/8, /*width=*/8),
                            m::BitSlice(b16.node(), /*start=*/0, /*width=*/8)));
    for (Node* part : dst_fn->nodes()) {
      if (part->Is<Param>()) {
        continue;
      }
      ASSERT_TRUE(original_node_to_clone.contains(part));
      Node* cloned_part = original_node_to_clone.at(part);
      EXPECT_EQ(cloned_part->function_base(), f);
      EXPECT_TRUE(cloned_part->IsDefinitelyEqualTo(part));
    }
    EXPECT_EQ(original_node_to_clone.at(decomps[0]->dst()), output);
  }
}

struct MulSpec {
  Op op;
  int64_t lhs_width;
  int64_t rhs_width;
  int64_t out_width;
};

struct MulPositiveCase {
  std::string test_name;
  std::vector<MulSpec> sources;
  MulSpec dst;
  MulSpec expected_dst;
  bool expect_dst_modified;
  bool expect_src_op_transform;
  bool expect_src_out_transform;
};

class MulEquivalenceMappingPositiveTest
    : public ResourceSharingEquivalenceTest,
      public ::testing::WithParamInterface<MulPositiveCase> {};

TEST_P(MulEquivalenceMappingPositiveTest, FoldsAndPreservesEquivalence) {
  const MulPositiveCase& tc = GetParam();
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  int64_t param_idx = 0;
  auto build_mul = [&](const MulSpec& spec) -> Node* {
    BValue lhs = fb.Param("lhs_" + std::to_string(param_idx),
                          p->GetBitsType(spec.lhs_width));
    BValue rhs = fb.Param("rhs_" + std::to_string(param_idx),
                          p->GetBitsType(spec.rhs_width));
    ++param_idx;
    BValue mul = spec.op == Op::kSMul ? fb.SMul(lhs, rhs, spec.out_width)
                                      : fb.UMul(lhs, rhs, spec.out_width);
    return mul.node();
  };

  std::vector<Node*> sources;
  sources.reserve(tc.sources.size());
  for (const MulSpec& src_spec : tc.sources) {
    sources.push_back(build_mul(src_spec));
  }
  Node* dst = build_mul(tc.dst);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  FakeAreaEstimator area_estimator;

  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<NodeToMappings> mappings,
      GetNodeEquivalenceMapper().ComputeMappings(sources, dst));
  ASSERT_TRUE(mappings.has_value());
  EXPECT_EQ(mappings->contains(dst), tc.expect_dst_modified);
  const EquivalenceMapping& first_src_map = *mappings->at(sources.front());
  EXPECT_EQ(first_src_map.dst()->op(), tc.expected_dst.op);
  EXPECT_EQ(first_src_map.dst()->operand(0)->BitCountOrDie(),
            tc.expected_dst.lhs_width);
  EXPECT_EQ(first_src_map.dst()->operand(1)->BitCountOrDie(),
            tc.expected_dst.rhs_width);
  EXPECT_EQ(first_src_map.dst()->BitCountOrDie(), tc.expected_dst.out_width);
  EXPECT_THAT(first_src_map.RequiresOperandTransformation(),
              IsOkAndHolds(tc.expect_src_op_transform));
  EXPECT_THAT(first_src_map.RequiresOutputTransformation(),
              IsOkAndHolds(tc.expect_src_out_transform));

  for (const auto& [node, map] : *mappings) {
    XLS_ASSERT_OK_AND_ASSIGN(
        double overhead,
        map->EstimateAreaOverhead(area_estimator, node->operands(), node));
    if (tc.expect_dst_modified) {
      EXPECT_GT(overhead, 0.0);
    } else {
      EXPECT_EQ(overhead, 0.0);
    }
    XLS_ASSERT_OK(VerifyEquivalence(f, node, *map).status());
  }
}

INSTANTIATE_TEST_SUITE_P(
    MulEquivalenceMappingPositiveTests, MulEquivalenceMappingPositiveTest,
    ::testing::ValuesIn<MulPositiveCase>({
        // umul(16, 16)->32 to smul(16, 16)->32 widens dst to smul(17, 17).
        {
            .test_name = "UMulToSMulWidensDst",
            .sources = {{Op::kUMul, 16, 16, 32}},
            .dst = {Op::kSMul, 16, 16, 32},
            .expected_dst = {Op::kSMul, 17, 17, 32},
            .expect_dst_modified = true,
            .expect_src_op_transform = true,
            .expect_src_out_transform = false,
        },
        // smul(16, 16)->32 to umul(16, 16)->32 promotes dst to smul(17, 17).
        {
            .test_name = "SMulToUMulPromotesAndWidensDst",
            .sources = {{Op::kSMul, 16, 16, 32}},
            .dst = {Op::kUMul, 16, 16, 32},
            .expected_dst = {Op::kSMul, 17, 17, 32},
            .expect_dst_modified = true,
            .expect_src_op_transform = true,
            .expect_src_out_transform = false,
        },
        // Low-half umul(16, 16)->16 to smul(16, 16)->16 needs no widening.
        {
            .test_name = "LowHalfUMulToSMulNoWidening",
            .sources = {{Op::kUMul, 16, 16, 16}},
            .dst = {Op::kSMul, 16, 16, 16},
            .expected_dst = {Op::kSMul, 16, 16, 16},
            .expect_dst_modified = false,
            .expect_src_op_transform = false,
            .expect_src_out_transform = false,
        },
        // Commutative operand swap: smul(17, 16) -> smul(16, 17) without
        // widening.
        {
            .test_name = "CommutativeOperandSwapNoWidening",
            .sources = {{Op::kSMul, 17, 16, 32}},
            .dst = {Op::kSMul, 16, 17, 32},
            .expected_dst = {Op::kSMul, 16, 17, 32},
            .expect_dst_modified = false,
            .expect_src_op_transform = true,
            .expect_src_out_transform = false,
        },
        // N-ary commutative alignment is order-independent onto smul(16, 17).
        {
            .test_name = "NaryCommutativeAlignmentNarrowFirst",
            .sources = {{Op::kUMul, 15, 16, 32}, {Op::kUMul, 16, 15, 32}},
            .dst = {Op::kSMul, 16, 16, 32},
            .expected_dst = {Op::kSMul, 16, 17, 32},
            .expect_dst_modified = true,
            .expect_src_op_transform = true,
            .expect_src_out_transform = false,
        },
        {
            .test_name = "NaryCommutativeAlignmentWideFirst",
            .sources = {{Op::kUMul, 16, 15, 32}, {Op::kUMul, 15, 16, 32}},
            .dst = {Op::kSMul, 16, 16, 32},
            .expected_dst = {Op::kSMul, 16, 17, 32},
            .expect_dst_modified = true,
            .expect_src_op_transform = true,
            .expect_src_out_transform = false,
        },
    }),
    [](const ::testing::TestParamInfo<MulPositiveCase>& info) {
      return info.param.test_name;
    });

TEST_F(ResourceSharingEquivalenceTest,
       MulEquivalenceMappingSymmetricWidthOperandTieBreaker) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a16 = fb.Param("a16", p->GetBitsType(16));
  BValue b16 = fb.Param("b16", p->GetBitsType(16));
  BValue smul_swapped = fb.SMul(b16, a16, 32);
  BValue umul = fb.UMul(a16, b16, 32);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(std::optional<NodeToMappings> mappings,
                           GetNodeEquivalenceMapper().ComputeMappings(
                               {smul_swapped.node()}, umul.node()));
  ASSERT_TRUE(mappings.has_value());
  const EquivalenceMapping& src_map = *mappings->at(smul_swapped.node());
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<Node*> coerced,
      src_map.ApplyToOperands(f, smul_swapped.node()->operands()));
  EXPECT_THAT(coerced, ElementsAre(m::SignExt(m::Param("a16")),
                                   m::SignExt(m::Param("b16"))));
  XLS_ASSERT_OK(VerifyEquivalence(f, smul_swapped.node(), src_map).status());
}

struct MulNegativeCase {
  std::string test_name;
  MulSpec src;
  MulSpec dst;
};

class MulEquivalenceMappingNegativeTest
    : public ResourceSharingEquivalenceTest,
      public ::testing::WithParamInterface<MulNegativeCase> {};

TEST_P(MulEquivalenceMappingNegativeTest, RejectsIncompatibleWidths) {
  const MulNegativeCase& tc = GetParam();
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto build_mul = [&](const std::string& prefix,
                       const MulSpec& spec) -> Node* {
    BValue lhs = fb.Param(prefix + "_lhs", p->GetBitsType(spec.lhs_width));
    BValue rhs = fb.Param(prefix + "_rhs", p->GetBitsType(spec.rhs_width));
    BValue mul = spec.op == Op::kSMul ? fb.SMul(lhs, rhs, spec.out_width)
                                      : fb.UMul(lhs, rhs, spec.out_width);
    return mul.node();
  };
  Node* src = build_mul("src", tc.src);
  Node* dst = build_mul("dst", tc.dst);
  XLS_ASSERT_OK(fb.Build().status());

  XLS_ASSERT_OK_AND_ASSIGN(
      std::optional<NodeToMappings> mappings,
      GetNodeEquivalenceMapper().ComputeMappings({src}, dst));
  EXPECT_FALSE(mappings.has_value());
}

INSTANTIATE_TEST_SUITE_P(
    MulEquivalenceMappingNegativeTests, MulEquivalenceMappingNegativeTest,
    ::testing::ValuesIn<MulNegativeCase>({
        {
            .test_name = "WiderOperandsAndOutputThanDst",
            .src = {Op::kUMul, 16, 16, 32},
            .dst = {Op::kSMul, 8, 8, 16},
        },
        {
            .test_name = "WiderOutputThanDst",
            .src = {Op::kUMul, 16, 16, 32},
            .dst = {Op::kSMul, 16, 16, 16},
        },
        {
            .test_name = "NarrowerOperandsButWiderOutputThanDst",
            .src = {Op::kUMul, 8, 8, 16},
            .dst = {Op::kSMul, 16, 16, 8},
        },
    }),
    [](const ::testing::TestParamInfo<MulNegativeCase>& info) {
      return info.param.test_name;
    });

class MulEquivalenceMappingWidthSweepTestBase
    : public ResourceSharingEquivalenceTest {
 protected:
  // Builds a function with `src_op(src_lhs_w, src_rhs_w) -> out_w` and
  // `dst_op(dst_lhs_w, dst_rhs_w) -> out_w`, then verifies mapping `src` to
  // `dst` preserves equivalence.
  void VerifyMulFolding(Op src_op, int64_t src_lhs_w, int64_t src_rhs_w,
                        Op dst_op, int64_t dst_lhs_w, int64_t dst_rhs_w,
                        int64_t out_w) {
    auto p = CreatePackage();
    FunctionBuilder fb("mul_sweep_fn", p.get());
    BValue src_lhs = fb.Param("src_lhs", p->GetBitsType(src_lhs_w));
    BValue src_rhs = fb.Param("src_rhs", p->GetBitsType(src_rhs_w));
    BValue dst_lhs = fb.Param("dst_lhs", p->GetBitsType(dst_lhs_w));
    BValue dst_rhs = fb.Param("dst_rhs", p->GetBitsType(dst_rhs_w));
    BValue src = fb.AddArithOp(src_op, src_lhs, src_rhs, out_w);
    BValue dst = fb.AddArithOp(dst_op, dst_lhs, dst_rhs, out_w);
    XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(src));

    XLS_ASSERT_OK_AND_ASSIGN(
        std::optional<NodeToMappings> mappings,
        GetNodeEquivalenceMapper().ComputeMappings({src.node()}, dst.node()));
    ASSERT_TRUE(mappings.has_value());
    for (const auto& [node, mapping] : *mappings) {
      XLS_ASSERT_OK(VerifyEquivalence(f, node, *mapping).status());
    }
  }
};

class MulEquivalenceMappingSymmetricWidthSweepTest
    : public MulEquivalenceMappingWidthSweepTestBase,
      public ::testing::WithParamInterface<int64_t> {};

TEST_P(MulEquivalenceMappingSymmetricWidthSweepTest,
       FoldsUMulIntoSMulAcrossOutputWidthRegimes) {
  const int64_t w = GetParam();
  // Truncated to operand width (out_w == W): signed and unsigned products
  // agree in the low W bits, so no operand widening is needed.
  VerifyMulFolding(Op::kUMul, w, w, Op::kSMul, w, w, /*out_w=*/w);
  // Partial high product (out_w == 2*W - 1): out_w > W when W > 1, so
  // unsigned operands must be zero-extended to W + 1 bits.
  if (w > 1) {
    VerifyMulFolding(Op::kUMul, w, w, Op::kSMul, w, w, /*out_w=*/2 * w - 1);
  }
  // Full product (out_w == 2*W): standard W x W -> 2*W multiply.
  VerifyMulFolding(Op::kUMul, w, w, Op::kSMul, w, w, /*out_w=*/2 * w);
  // Over-wide output (out_w > 2*W): output is sign/zero-extended past 2*W.
  VerifyMulFolding(Op::kUMul, w, w, Op::kSMul, w, w, /*out_w=*/2 * w + 2);
}

INSTANTIATE_TEST_SUITE_P(MulEquivalenceMappingSymmetricWidthSweepTests,
                         MulEquivalenceMappingSymmetricWidthSweepTest,
                         ::testing::Values(int64_t{1}, int64_t{2}, int64_t{3},
                                           int64_t{8}));

class MulEquivalenceMappingAsymmetricWidthSweepTest
    : public MulEquivalenceMappingWidthSweepTestBase,
      public ::testing::WithParamInterface<std::pair<int64_t, int64_t>> {};

TEST_P(MulEquivalenceMappingAsymmetricWidthSweepTest,
       FoldsAcrossOutputWidthThresholdsAndOperandOrders) {
  const auto [w_narrow, w_wide] = GetParam();
  // out_w == w_narrow (neither operand extended), out_w == w_wide (only
  // narrow operand extended), and out_w == w_narrow + w_wide (both extended).
  for (int64_t out_w : {w_narrow, w_wide, w_narrow + w_wide}) {
    VerifyMulFolding(Op::kUMul, w_narrow, w_wide, Op::kSMul, w_narrow, w_wide,
                     out_w);
    VerifyMulFolding(Op::kUMul, w_wide, w_narrow, Op::kSMul, w_narrow, w_wide,
                     out_w);
    VerifyMulFolding(Op::kUMul, w_narrow, w_wide, Op::kSMul, w_wide, w_narrow,
                     out_w);
    VerifyMulFolding(Op::kSMul, w_wide, w_narrow, Op::kUMul, w_narrow, w_wide,
                     out_w);
    VerifyMulFolding(Op::kUMul, w_wide, w_narrow, Op::kUMul, w_narrow, w_wide,
                     out_w);
    VerifyMulFolding(Op::kSMul, w_wide, w_narrow, Op::kSMul, w_narrow, w_wide,
                     out_w);
  }
}

INSTANTIATE_TEST_SUITE_P(MulEquivalenceMappingAsymmetricWidthSweepTests,
                         MulEquivalenceMappingAsymmetricWidthSweepTest,
                         ::testing::Values(std::pair<int64_t, int64_t>{1, 2},
                                           std::pair<int64_t, int64_t>{2, 3},
                                           std::pair<int64_t, int64_t>{3, 5}));

}  // namespace
}  // namespace xls
