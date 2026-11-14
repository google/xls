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

#include "xls/passes/name_promotion_pass.h"

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class NamePromotionPassTest : public IrTestBase {
 protected:
  NamePromotionPassTest() = default;

  absl::StatusOr<bool> Run(FunctionBase* f) {
    PassResults results;
    OptimizationContext context;
    return NamePromotionPass().RunOnFunctionBase(f, OptimizationPassOptions(),
                                                 &results, context);
  }

  absl::StatusOr<bool> RunDceAndNamePromotion(FunctionBase* f) {
    PassResults results;
    OptimizationContext context;
    OptimizationPassOptions options;
    XLS_ASSIGN_OR_RETURN(bool dce_changed,
                         DeadCodeEliminationPass().RunOnFunctionBase(
                             f, options, &results, context));
    XLS_ASSIGN_OR_RETURN(
        bool promo_changed,
        NamePromotionPass().RunOnFunctionBase(f, options, &results, context));
    return dce_changed || promo_changed;
  }
};

TEST_F(NamePromotionPassTest, NoPromotionWhenOriginalNameStillInUse) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue first = fb.Add(x, y, SourceInfo(), "val");
  BValue second = fb.Subtract(x, y, SourceInfo(), "val");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Xor(first, second)));

  EXPECT_EQ(first.node()->GetName(), "val");
  EXPECT_EQ(second.node()->GetName(), "val__1");

  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(first.node()->GetName(), "val");
  EXPECT_EQ(second.node()->GetName(), "val__1");
}

TEST_F(NamePromotionPassTest, PromotesNodeWhenCollidingNodeRemovedByDce) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  // Dead node takes the name "result".
  BValue dead = fb.Add(x, y, SourceInfo(), "result");
  // Live return node requests "result" and gets "result__1".
  BValue live = fb.Subtract(x, y, SourceInfo(), "result");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(live));

  EXPECT_EQ(dead.node()->GetName(), "result");
  EXPECT_EQ(live.node()->GetName(), "result__1");

  PassResults results;
  OptimizationContext context;
  ASSERT_THAT(DeadCodeEliminationPass().RunOnFunctionBase(
                  f, OptimizationPassOptions(), &results, context),
              IsOkAndHolds(true));
  ASSERT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(live.node()->GetName(), "result");

  // Running a second time is a no-op.
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(live.node()->GetName(), "result");
}

TEST_F(NamePromotionPassTest, MultipleContendersForSameOriginalName) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  BValue n0 = fb.Add(x, y, SourceInfo(), "foo");
  BValue n1 = fb.Subtract(x, y, SourceInfo(), "foo");
  BValue n2 = fb.UMul(x, y, SourceInfo(), "foo");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           fb.BuildWithReturnValue(fb.Or(n1, n2)));

  EXPECT_EQ(n0.node()->GetName(), "foo");
  EXPECT_EQ(n1.node()->GetName(), "foo__1");
  EXPECT_EQ(n2.node()->GetName(), "foo__2");

  // Remove n0 ("foo") and promote: n1 should become "foo" and n2 stays
  // "foo__2".
  ASSERT_THAT(RunDceAndNamePromotion(f), IsOkAndHolds(true));
  EXPECT_EQ(n1.node()->GetName(), "foo");
  EXPECT_EQ(n2.node()->GetName(), "foo__2");

  // Now make n1 dead by returning only n2, run DCE and promote again: n2 should
  // become "foo".
  XLS_ASSERT_OK(f->set_return_value(n2.node()));
  ASSERT_THAT(RunDceAndNamePromotion(f), IsOkAndHolds(true));
  EXPECT_EQ(n2.node()->GetName(), "foo");
}

TEST_F(NamePromotionPassTest, CascadingPromotionAcrossReverseTopoOrder) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  // n0 takes "foo".
  BValue n0 = fb.Add(x, y, SourceInfo(), "foo");
  // n1 requests "foo", gets "foo__1" (orig "foo").
  BValue n1 = fb.Subtract(x, y, SourceInfo(), "foo");
  // n2 explicitly requests "foo__1", gets "foo__2" (orig "foo__1").
  BValue n2 = fb.UMul(x, y, SourceInfo(), "foo__1");
  // Make n1 depend on n2 so that n2 ("foo__2", orig "foo__1") appears BEFORE
  // n1_replacement ("foo__1", orig "foo") in TopoSort order.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(n1));
  EXPECT_EQ(n0.node()->GetName(), "foo");

  XLS_ASSERT_OK_AND_ASSIGN(
      Node * n1_replacement,
      n1.node()->ReplaceUsesWithNew<BinOp>(n2.node(), y.node(), Op::kAdd));
  EXPECT_EQ(n1_replacement->GetName(), "foo__1");
  EXPECT_EQ(n2.node()->GetName(), "foo__2");

  // Remove dead nodes (n0 and old n1) and run NamePromotionPass.
  // The initial TopoSort scan visits n2 (records n2 as waiting for "foo__1"),
  // then visits n1_replacement and enqueues it onto the worklist. Promoting
  // n1_replacement ("foo__1" -> "foo") frees "foo__1", which wakes n2 and
  // promotes it ("foo__2" -> "foo__1").
  ASSERT_THAT(RunDceAndNamePromotion(f), IsOkAndHolds(true));
  EXPECT_EQ(n1_replacement->GetName(), "foo");
  EXPECT_EQ(n2.node()->GetName(), "foo__1");
}

TEST_F(NamePromotionPassTest, EarlierTopoSortNodeGetsFirstDibs) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue y = fb.Param("y", p->GetBitsType(32));
  // n0 takes "foo" (and is dead).
  BValue n0 = fb.Add(x, y, SourceInfo(), "foo");
  // user_node is created before operand_node in f->nodes(), requesting "foo" ->
  // "foo__1".
  BValue placeholder = fb.Literal(UBits(0, 32));
  BValue user_node = fb.Add(placeholder, y, SourceInfo(), "foo");
  // operand_node is created after user_node in f->nodes(), requesting "foo" ->
  // "foo__2", and then wired as an operand of user_node so operand_node comes
  // BEFORE user_node in TopoSort order.
  BValue operand_node = fb.Subtract(x, y, SourceInfo(), "foo");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(user_node));
  XLS_ASSERT_OK(user_node.node()->ReplaceOperandNumber(0, operand_node.node()));

  EXPECT_EQ(n0.node()->GetName(), "foo");
  EXPECT_EQ(user_node.node()->GetName(), "foo__1");
  EXPECT_EQ(operand_node.node()->GetName(), "foo__2");

  ASSERT_THAT(RunDceAndNamePromotion(f), IsOkAndHolds(true));
  // operand_node is earlier in TopoSort order, so it gets "foo".
  EXPECT_EQ(operand_node.node()->GetName(), "foo");
  EXPECT_EQ(user_node.node()->GetName(), "foo__1");
}

TEST_F(NamePromotionPassTest,
       DoesNotPromoteExplicitlySuffixedNonCollidingName) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue n = fb.Not(x, SourceInfo(), "bar__1");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(n));

  EXPECT_EQ(n.node()->GetName(), "bar__1");
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(n.node()->GetName(), "bar__1");
}

TEST_F(NamePromotionPassTest, DoesNotPromoteParamName) {
  auto p = CreatePackage();
  Function* f = p->AddFunction(std::make_unique<Function>(TestName(), p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(
      Node * lit,
      f->MakeNodeWithName<Literal>(SourceInfo(), Value(UBits(1, 32)), "x"));
  XLS_ASSERT_OK_AND_ASSIGN(
      Param * param,
      f->MakeNodeWithName<Param>(SourceInfo(), p->GetBitsType(32), "x"));
  XLS_ASSERT_OK(f->set_return_value(param));
  XLS_ASSERT_OK(f->RemoveNode(lit));

  EXPECT_EQ(param->GetName(), "x__1");
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(param->GetName(), "x__1");
}

TEST_F(NamePromotionPassTest, PromotesNodesInProc) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(TestName(), "tkn", p.get());
  BValue st = pb.ReadStateElement("st", Value(UBits(0, 32)));
  BValue dead = pb.Add(st, pb.Literal(UBits(1, 32)), SourceInfo(), "next_st");
  BValue live = pb.Add(st, pb.Literal(UBits(2, 32)), SourceInfo(), "next_st");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({live}));

  EXPECT_EQ(dead.node()->GetName(), "next_st");
  EXPECT_EQ(live.node()->GetName(), "next_st__1");

  ASSERT_THAT(RunDceAndNamePromotion(proc), IsOkAndHolds(true));
  EXPECT_EQ(live.node()->GetName(), "next_st");
}

void IrFuzzNamePromotion(FuzzPackageWithArgs fuzz_package_with_args) {
  NamePromotionPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzNamePromotion)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
