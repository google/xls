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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class ReassociationPassTest : public IrTestBase {
 protected:
  ReassociationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return ReassociationPass().Run(p, PassOptions(), &results);
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
