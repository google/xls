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

#include "xls/passes/dfe_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class DeadFunctionEliminationPassTest : public IrTestBase {
 protected:
  DeadFunctionEliminationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return DeadFunctionEliminationPass().Run(p, PassOptions(), &results);
  }

  absl::StatusOr<Function*> MakeFunction(absl::string_view name, Package* p) {
    FunctionBuilder fb(name, p);
    fb.Param("arg", p->GetBitsType(32));
    return fb.Build();
  }
};

TEST_F(DeadFunctionEliminationPassTest, NoDeadFunctions) {
  auto p = std::make_unique<Package>(TestName(), /*entry=*/"the_entry");
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  XLS_ASSERT_OK_AND_ASSIGN(Function * b, MakeFunction("b", p.get()));
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Invoke({x}, a), fb.Invoke({x}, b));
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

TEST_F(DeadFunctionEliminationPassTest, OneDeadFunction) {
  auto p = std::make_unique<Package>(TestName(), /*entry=*/"the_entry");
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  XLS_ASSERT_OK(MakeFunction("dead", p.get()).status());
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Invoke({x}, a), fb.Invoke({x}, a));
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(p->functions().size(), 2);
}

TEST_F(DeadFunctionEliminationPassTest, OneDeadFunctionButNoEntry) {
  // If no entry function is specified, then DFS cannot happen as all functions
  // are live.
  auto p = std::make_unique<Package>(TestName());
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  XLS_ASSERT_OK(MakeFunction("dead", p.get()).status());
  FunctionBuilder fb("blah", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Invoke({x}, a), fb.Invoke({x}, a));
  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

TEST_F(DeadFunctionEliminationPassTest, ProcCallingFunction) {
  auto p = std::make_unique<Package>(TestName(), "entry");
  XLS_ASSERT_OK(MakeFunction("entry", p.get()).status());
  XLS_ASSERT_OK_AND_ASSIGN(Function * f,
                           MakeFunction("called_by_proc", p.get()));

  TokenlessProcBuilder b(TestName(), Value(UBits(0, 32)), "tkn", "st", p.get());
  BValue invoke = b.Invoke({b.GetStateParam()}, f);
  XLS_ASSERT_OK(b.Build(invoke));

  EXPECT_EQ(p->functions().size(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 2);
}

TEST_F(DeadFunctionEliminationPassTest, MapAndCountedFor) {
  // If no entry function is specified, then DFS cannot happen as all functions
  // are live.
  auto p = std::make_unique<Package>(TestName(), /*entry=*/"the_entry");
  XLS_ASSERT_OK_AND_ASSIGN(Function * a, MakeFunction("a", p.get()));
  Function* body;
  {
    FunctionBuilder fb("jesse_the_loop_body", p.get());
    fb.Param("i", p->GetBitsType(32));
    fb.Param("arg", p->GetBitsType(32));
    fb.Literal(UBits(123, 32));
    XLS_ASSERT_OK_AND_ASSIGN(body, fb.Build());
  }
  FunctionBuilder fb("the_entry", p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue ar = fb.Param("ar", p->GetArrayType(42, p->GetBitsType(32)));
  BValue mapped_ar = fb.Map(ar, a);
  BValue for_loop = fb.CountedFor(x, /*trip_count=*/42, /*stride=*/1, body);
  fb.Tuple({mapped_ar, for_loop});

  XLS_ASSERT_OK(fb.Build().status());

  EXPECT_EQ(p->functions().size(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(p->functions().size(), 3);
}

}  // namespace
}  // namespace xls
