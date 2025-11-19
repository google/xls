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

#include "xls/ir/scheduled_builder.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using absl_testing::IsOkAndHolds;

class ScheduledBuilderTest : public IrTestBase {};

TEST_F(ScheduledBuilderTest, BuildFunctionWithStages) {
  auto p = CreatePackage();
  ScheduledFunctionBuilder sfb(TestName(), p.get());
  BValue x = sfb.Param("x", p->GetBitsType(32));
  BValue y = sfb.Param("y", p->GetBitsType(32));
  BValue lit = sfb.Literal(UBits(10, 32));

  sfb.SetCurrentStage(1);
  BValue not_y = sfb.Not(y);

  sfb.SetCurrentStage(0);
  BValue neg_x = sfb.Negate(x);

  sfb.SetCurrentStage(2);
  BValue add = sfb.Add(neg_x, not_y);
  BValue result = sfb.Add(add, lit);

  XLS_ASSERT_OK_AND_ASSIGN(ScheduledFunction * f,
                           sfb.BuildWithReturnValue(result));

  EXPECT_TRUE(f->IsScheduled());
  EXPECT_THAT(f->GetStageIndex(x.node()), IsOkAndHolds(0));
  EXPECT_THAT(f->GetStageIndex(y.node()), IsOkAndHolds(0));
  EXPECT_THAT(f->GetStageIndex(lit.node()), IsOkAndHolds(0));
  EXPECT_THAT(f->GetStageIndex(neg_x.node()), IsOkAndHolds(0));
  EXPECT_THAT(f->GetStageIndex(not_y.node()), IsOkAndHolds(1));
  EXPECT_THAT(f->GetStageIndex(add.node()), IsOkAndHolds(2));
  EXPECT_THAT(f->GetStageIndex(result.node()), IsOkAndHolds(2));
}

TEST_F(ScheduledBuilderTest, FunctionAssignNodeToStage) {
  auto p = CreatePackage();
  ScheduledFunctionBuilder sfb(TestName(), p.get());
  BValue x = sfb.Param("x", p->GetBitsType(32));
  BValue y = sfb.Param("y", p->GetBitsType(32));

  sfb.SetCurrentStage(1);
  BValue neg_x = sfb.Negate(x);
  BValue not_y = sfb.AssignNodeToStage(sfb.Not(y), 0);
  BValue add = sfb.Add(neg_x, not_y);

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, sfb.BuildWithReturnValue(add));

  EXPECT_TRUE(f->IsScheduled());
  EXPECT_THAT(f->GetStageIndex(neg_x.node()), IsOkAndHolds(1));
  EXPECT_THAT(f->GetStageIndex(not_y.node()), IsOkAndHolds(0));
  EXPECT_THAT(f->GetStageIndex(add.node()), IsOkAndHolds(1));
}

TEST_F(ScheduledBuilderTest, BuildProcWithStages) {
  auto p = CreatePackage();
  ScheduledProcBuilder spb(TestName(), p.get());
  BValue state = spb.StateElement("s", Value(UBits(0, 32)));
  BValue lit1 = spb.Literal(UBits(1, 32));

  spb.SetCurrentStage(0);
  BValue neg_state = spb.Negate(state);

  spb.EndStage();
  BValue add = spb.Add(neg_state, lit1);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, spb.Build({add}));

  EXPECT_TRUE(proc->IsScheduled());
  EXPECT_THAT(proc->GetStageIndex(state.node()), IsOkAndHolds(0));
  EXPECT_THAT(proc->GetStageIndex(lit1.node()), IsOkAndHolds(0));
  EXPECT_THAT(proc->GetStageIndex(neg_state.node()), IsOkAndHolds(0));
  EXPECT_THAT(proc->GetStageIndex(add.node()), IsOkAndHolds(1));
}

TEST_F(ScheduledBuilderTest, ProcAssignNodeToStage) {
  auto p = CreatePackage();
  ScheduledProcBuilder spb(TestName(), p.get());
  BValue state = spb.StateElement("s", Value(UBits(0, 32)));

  spb.SetCurrentStage(1);
  BValue neg_state = spb.Negate(state);
  BValue not_state = spb.AssignNodeToStage(spb.Not(state), 0);
  BValue add = spb.Add(neg_state, not_state);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, spb.Build({add}));

  EXPECT_TRUE(proc->IsScheduled());
  EXPECT_THAT(proc->GetStageIndex(neg_state.node()), IsOkAndHolds(1));
  EXPECT_THAT(proc->GetStageIndex(not_state.node()), IsOkAndHolds(0));
  EXPECT_THAT(proc->GetStageIndex(add.node()), IsOkAndHolds(1));
}

}  // namespace
}  // namespace xls
