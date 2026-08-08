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

#include "xls/passes/non_synth_removal_pass.h"

#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;

class NonSynthRemovalPassTest : public IrTestBase {
 protected:
  NonSynthRemovalPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    OptimizationContext context;
    NonSynthRemovalPass pass;
    return pass.Run(p, OptimizationPassOptions(), &results, context);
  }
};

TEST_F(NonSynthRemovalPassTest, RemovesAssertAndDependencies) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  auto tok = fb.Literal(Value::Token());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto zero = fb.Literal(UBits(0, 32));
  auto eq = fb.Eq(x, zero);
  fb.Assert(tok, eq, "x is zero");
  fb.Add(x, fb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK(p->SetTop(f));

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(p->GetFunctionBases().size(), 1);
  for (Node* n : p->GetFunctionBases().front()->nodes()) {
    EXPECT_FALSE(n->Is<Assert>());
    EXPECT_FALSE(n->Is<Invoke>());
    EXPECT_NE(n->GetName(), "eq");
  }
}

TEST_F(NonSynthRemovalPassTest, RemovesTraceAndCoverAndDependencies) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  auto tok = fb.Literal(Value::Token());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto cond = fb.Eq(x, fb.Literal(UBits(10, 32)));
  fb.Trace(tok, cond, {}, "x is 10");
  fb.Cover(cond, "cover label");
  fb.Add(x, fb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK(p->SetTop(f));

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(p->GetFunctionBases().size(), 1);
  for (Node* n : p->GetFunctionBases().front()->nodes()) {
    EXPECT_FALSE(n->Is<Trace>());
    EXPECT_FALSE(n->Is<Cover>());
    EXPECT_FALSE(n->Is<Invoke>());
    EXPECT_NE(n->GetName(), "cond");
  }
}

TEST_F(NonSynthRemovalPassTest, FunctionWithNoNonSynthNodesUnchanged) {
  auto p = CreatePackage();
  FunctionBuilder fb("f", p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  fb.Add(x, fb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  XLS_ASSERT_OK(p->SetTop(f));

  ASSERT_THAT(Run(p.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
