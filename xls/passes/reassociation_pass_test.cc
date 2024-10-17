// Copyright 2020 The XLS Authors
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

#include "xls/passes/reassociation_pass.h"

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/time/time.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

constexpr absl::Duration kProverTimeout = absl::Seconds(10);

using ::absl_testing::IsOkAndHolds;
using ::testing::AllOf;

using ::xls::solvers::z3::ScopedVerifyEquivalence;

class ReassociationPassTest : public IrTestBase {
 protected:
  ReassociationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return ReassociationPass().Run(p, OptimizationPassOptions(), &results);
  }
};

TEST_F(ReassociationPassTest, SingleAdd) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Param("a", u32), fb.Param("b", u32));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, TwoAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Param("a", u32), fb.Add(fb.Param("b", u32), fb.Param("c", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, ChainOfThreeAddsRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Param("a", u32),
         fb.Add(fb.Param("b", u32),
                fb.Add(fb.Param("c", u32), fb.Param("d", u32))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Add(m::Param("a"), m::Param("b")),
                                        m::Add(m::Param("c"), m::Param("d"))));
}

TEST_F(ReassociationPassTest, NearToFinalBitWidth) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue v6 = fb.Literal(UBits(0, 6));
  BValue v16 = fb.Literal(UBits(0, 16));
  BValue ext_9 = fb.SignExtend(v6, 9);
  BValue add_9 = fb.Add(ext_9, ext_9);
  BValue ext_17_add_9 = fb.SignExtend(add_9, 17);
  BValue ext_17 = fb.SignExtend(v16, 17);
  // NB Use the 'add' second so the bit width is already 17 by the time the
  // second addition is processed.
  fb.Add(ext_17, ext_17_add_9);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  RecordProperty("ir", p->DumpIr());
  EXPECT_THAT(f->return_value(),
              m::Add(m::SignExt(m::Literal(UBits(0, 6))),
                     m::SignExt(m::Add(m::SignExt(m::Literal(UBits(0, 16))),
                                       m::SignExt(m::Literal(UBits(0, 6)))))));
}

TEST_F(ReassociationPassTest, ChainOfThreeFullWidthUnsignedAddsRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.ZeroExtend(fb.Param("a", u32), 35),
         fb.ZeroExtend(
             fb.Add(fb.ZeroExtend(fb.Param("b", u32), 34),
                    fb.ZeroExtend(fb.Add(fb.ZeroExtend(fb.Param("c", u32), 33),
                                         fb.ZeroExtend(fb.Param("d", u32), 33)),
                                  34)),
             35));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[35]")));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::ZeroExt(AllOf(m::Add(m::ZeroExt(m::Add(m::ZeroExt(m::Param("a")),
                                                m::ZeroExt(m::Param("b")))),
                              m::ZeroExt(m::Add(m::ZeroExt(m::Param("c")),
                                                m::ZeroExt(m::Param("d"))))),
                       m::Type("bits[34]"))));
}

TEST_F(ReassociationPassTest, ChainOfThreeFullWidthSignedAddsRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.SignExtend(fb.Param("a", u32), 35),
         fb.SignExtend(
             fb.Add(fb.SignExtend(fb.Param("b", u32), 34),
                    fb.SignExtend(fb.Add(fb.SignExtend(fb.Param("c", u32), 33),
                                         fb.SignExtend(fb.Param("d", u32), 33)),
                                  34)),
             35));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[35]")));

  ScopedVerifyEquivalence stays_equivalent(f, kProverTimeout);
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::SignExt(AllOf(m::Add(m::SignExt(m::Add(m::SignExt(m::Param("a")),
                                                m::SignExt(m::Param("b")))),
                              m::SignExt(m::Add(m::SignExt(m::Param("c")),
                                                m::SignExt(m::Param("d"))))),
                       m::Type("bits[34]"))));
}

TEST_F(ReassociationPassTest, ChainOfThreeFullWidthMixedAddsRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.SignExtend(fb.Param("a", u32), 35),
         fb.SignExtend(
             fb.Add(fb.ZeroExtend(fb.Param("b", u32), 34),
                    fb.ZeroExtend(fb.Add(fb.SignExtend(fb.Param("c", u32), 33),
                                         fb.SignExtend(fb.Param("d", u32), 33)),
                                  34)),
             35));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[35]")));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, ChainOfThreeUMulRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.UMul(fb.Param("a", u32),
          fb.UMul(fb.Param("b", u32),
                  fb.UMul(fb.Param("c", u32), fb.Param("d", u32))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::UMul(m::UMul(m::Param("a"), m::Param("b")),
                      m::UMul(m::Param("c"), m::Param("d"))));
}

TEST_F(ReassociationPassTest, ChainOfFourUMulRight) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.UMul(fb.Param("a", u32),
          fb.UMul(fb.Param("b", u32),
                  fb.UMul(fb.Param("c", u32),
                          fb.UMul(fb.Param("d", u32), fb.Param("e", u32)))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::UMul(m::UMul(m::UMul(m::Param("a"), m::Param("b")), m::Param("c")),
              m::UMul(m::Param("d"), m::Param("e"))));
}

TEST_F(ReassociationPassTest, ChainOfMixedOperations) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.UMul(fb.Param("a", u32),
          fb.UMul(fb.Param("b", u32),
                  fb.Add(fb.Param("c", u32),
                         fb.Add(fb.Param("d", u32), fb.Param("e", u32)))));
  XLS_ASSERT_OK(fb.Build().status());
}

TEST_F(ReassociationPassTest, ChainOfThreeAddsLeft) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Add(fb.Param("a", u32), fb.Param("b", u32)),
                fb.Param("c", u32)),
         fb.Param("d", u32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Add(m::Add(m::Param("a"), m::Param("b")),
                                        m::Add(m::Param("c"), m::Param("d"))));
}

TEST_F(ReassociationPassTest, DeepChain) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  BValue lhs = fb.Param("p0", u32);
  for (int64_t i = 1; i < 41; ++i) {
    lhs = fb.Add(lhs, fb.Param(absl::StrFormat("p%d", i), u32));
  }
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
}

TEST_F(ReassociationPassTest, DeepChainOfFullWidthUnsignedAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  BValue lhs = fb.Param("p0", u8);
  for (int64_t i = 1; i < 10; ++i) {
    lhs = fb.Add(fb.ZeroExtend(lhs, 8 + i),
                 fb.ZeroExtend(fb.Param(absl::StrFormat("p%d", i), u8), 8 + i));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[17]")));

  ScopedVerifyEquivalence stays_equivalent(f, /*timeout=*/absl::Seconds(30));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::ZeroExt(AllOf(m::Add(), m::Type("bits[12]"))));
}

TEST_F(ReassociationPassTest, DeepChainOfFullWidthSignedAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u8 = p->GetBitsType(8);
  BValue lhs = fb.Param("p0", u8);
  for (int64_t i = 1; i < 10; ++i) {
    lhs = fb.Add(fb.SignExtend(lhs, 8 + i),
                 fb.SignExtend(fb.Param(absl::StrFormat("p%d", i), u8), 8 + i));
  }
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(f->return_value(), AllOf(m::Add(), m::Type("bits[17]")));

  ScopedVerifyEquivalence stays_equivalent(f, /*timeout=*/absl::Seconds(30));
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::SignExt(AllOf(m::Add(), m::Type("bits[12]"))));
}

TEST_F(ReassociationPassTest, BalancedTreeOfThreeAdds) {
  // An already balanced tree should not be transformed.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Param("b", u32)), fb.Param("c", u32));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, BalancedTreeOfFourAdds) {
  // An already balanced tree should not be transformed.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Param("b", u32)),
         fb.Add(fb.Param("c", u32), fb.Param("d", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, BalancedTreeOfFiveAdds) {
  // An already balanced tree should not be transformed.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Param("b", u32)),
         fb.Add(fb.Param("c", u32),
                fb.Add(fb.Param("d", u32), fb.Param("e", u32))));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, ReassociateMultipleLiterals) {
  // Multiple Literals should be reassociated to the right even if the tree is
  // balanced.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Literal(UBits(42, 32))),
         fb.Add(fb.Literal(UBits(123, 32)), fb.Param("b", u32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Add(m::Param("a"), m::Param("b")),
                     m::Add(m::Literal(42), m::Literal(123))));
}

TEST_F(ReassociationPassTest, SingleLiteralNoReassociate) {
  // If there is a single literal in the expression and the tree is balanced
  // then no reassociation should happen.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Add(fb.Param("a", u32), fb.Literal(UBits(42, 32))),
         fb.Add(fb.Param("b", u32), fb.Param("c", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SingleSubtract) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Param("a", u32), fb.Param("b", u32));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SingleSubtractOfLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Param("a", u32), fb.Literal(Value(UBits(42, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Add(m::Param("a"), m::Literal(SBits(-42, 32))));
}

TEST_F(ReassociationPassTest, TreeOfSubtracts) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Subtract(fb.Param("a", u32), fb.Param("b", u32)),
              fb.Subtract(fb.Param("c", u32), fb.Param("d", u32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Sub(m::Add(m::Param("a"), m::Param("d")),
                                        m::Add(m::Param("b"), m::Param("c"))));
}

TEST_F(ReassociationPassTest, AddOfSubtracts) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Add(fb.Subtract(fb.Param("a", u32), fb.Param("b", u32)),
         fb.Subtract(fb.Param("c", u32), fb.Param("d", u32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(), m::Sub(m::Add(m::Param("a"), m::Param("c")),
                                        m::Add(m::Param("b"), m::Param("d"))));
}

TEST_F(ReassociationPassTest, SubtractOfAdds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Add(fb.Param("a", u32), fb.Param("b", u32)),
              fb.Add(fb.Param("c", u32), fb.Param("d", u32)));
  XLS_ASSERT_OK(fb.Build().status());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ReassociationPassTest, SubtractOfAddsWithLiterals) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  fb.Subtract(fb.Add(fb.Param("a", u32), fb.Literal(Value(UBits(100, 32)))),
              fb.Add(fb.Literal(Value(UBits(42, 32))), fb.Param("b", u32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Sub(m::Add(m::Param("a"), m::Add(m::Literal(UBits(100, 32)),
                                                  m::Literal(SBits(-42, 32)))),
                     m::Param("b")));
}

TEST_F(ReassociationPassTest, SubOfSub) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Subtract(fb.Param("x", p->GetBitsType(32)),
              fb.Subtract(fb.Param("y", p->GetBitsType(32)),
                          fb.Param("z", p->GetBitsType(32))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::Sub(m::Add(m::Param("x"), m::Param("z")), m::Param("y")));
}

}  // namespace
}  // namespace xls
