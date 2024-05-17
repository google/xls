// Copyright 2021 The XLS Authors
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

#include "xls/passes/useless_assert_removal_pass.h"

#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {

namespace {

class AssertCleanupPassTest : public IrTestBase {
 protected:
  AssertCleanupPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         UselessAssertRemovalPass().RunOnFunctionBase(
                             f, OptimizationPassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(f, OptimizationPassOptions(), &results)
            .status());
    // Return whether useless assert removal changed anything.
    return changed;
  }
};

TEST_F(AssertCleanupPassTest, RemoveSingleAssertNoTokenThreading) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue token = fb.AfterAll({});
  BValue l1 = fb.Literal(UBits(1, 1));
  BValue l2 = fb.Literal(UBits(0, 31));
  fb.Assert(token, l1, "Assert");
  fb.Concat({x, l2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_EQ(f->node_count(), 6);
  EXPECT_THAT(Run(f), status_testing::IsOkAndHolds(true));
  // We expect there to be three nodes left.
  // The assert with literal 1 condition should be removed, along with the dead
  // literal 1 and after all.
  EXPECT_EQ(f->node_count(), 3);
}

TEST_F(AssertCleanupPassTest, DontRemoveSingleAssertLiteral0) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue token = fb.AfterAll({});
  BValue l1 = fb.Literal(UBits(0, 1));
  BValue l2 = fb.Literal(UBits(0, 31));
  fb.Assert(token, l1, "Assert");
  fb.Concat({x, l2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_EQ(f->node_count(), 6);
  EXPECT_THAT(Run(f), status_testing::IsOkAndHolds(false));
  // We expect there to be six nodes left.
  // Assert with literal 0 kept, nothing changed.
  EXPECT_EQ(f->node_count(), 6);
}

TEST_F(AssertCleanupPassTest, DontRemoveSingleAssertNotLiteral) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(2));
  BValue token = fb.AfterAll({});
  BValue l1 = fb.Literal(UBits(1, 2));
  BValue comp = fb.UGe(x, l1);
  BValue l2 = fb.Literal(UBits(0, 31));
  fb.Assert(token, comp, "Assert");
  fb.Concat({x, l2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  EXPECT_EQ(f->node_count(), 7);
  EXPECT_THAT(Run(f), status_testing::IsOkAndHolds(false));
  // We expect there to be seven nodes left.
  // Assert with non-literal condition kept, nothing changed.
  EXPECT_EQ(f->node_count(), 7);
}

TEST_F(AssertCleanupPassTest, RemoveSingleAssertTokenThreading) {
  auto p = CreatePackage();
  // Single assert that is never triggered (literal 1 as condition).
  // Assert has a user, so it is necessary to thread the token to it.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn fail_test_redundant__main(x: bits[1]) -> bits[2] {
      after_all.1: token = after_all(id=1)
      literal.2: bits[1] = literal(value=0, id=2)
      literal.3: bits[1] = literal(value=1, id=3)
      assert.4: token = assert(after_all.1, literal.3, message="Assert to be removed", id=4)
      assert.5: token = assert(assert.4, x, message="Assert to be kept", id=5)
      ret concat.6: bits[2] = concat(literal.2, x, id=6)
    }
  )",
                                                       p.get()));
  // Should be seven nodes to start.
  // Input operand x and six nodes defined in function.
  EXPECT_EQ(f->node_count(), 7);
  EXPECT_THAT(Run(f), status_testing::IsOkAndHolds(true));
  // We expect there to be five nodes left.
  // Assert and the literal 1 condition removed, token rewired.
  EXPECT_EQ(f->node_count(), 5);
}

TEST_F(AssertCleanupPassTest, RemoveCascadedAssertTokenThreading_2Levels) {
  auto p = CreatePackage();
  // Two cascaded useless asserts followed by non-useless assert.
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn cascaded_assert_removal_test1__main(x: bits[1]) -> bits[2] {
      after_all.1: token = after_all(id=1)
      literal.2: bits[1] = literal(value=1, id=2)
      assert.3: token = assert(after_all.1, literal.2, message="X", id=3)
      literal.4: bits[1] = literal(value=1, id=4)
      assert.5: token = assert(assert.3, literal.4, message="X", id=5)
      assert.6: token = assert(assert.5, x, message="O", id=6)
      literal.7: bits[1] = literal(value=0, id=7)
      ret concat.8: bits[2] = concat(literal.7, x, id=8)
    }
  )",
                                                       p.get()));
  // Should be 9 nodes to start.
  // Input operand x, 8 nodes defined in function.
  EXPECT_EQ(f->node_count(), 9);
  EXPECT_THAT(Run(f), status_testing::IsOkAndHolds(true));
  // We expect there to be five nodes left.
  // Asserts and literal 1 conditions removed, token rewired.
  EXPECT_EQ(f->node_count(), 5);
}

TEST_F(AssertCleanupPassTest, RemoveCascadedAssertTokenThreading_20Levels) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(1));
  BValue token = fb.AfterAll({});
  for (int idx = 0; idx < 20; idx++) {
    BValue l1 = fb.Literal(UBits(1, 1));
    token = fb.Assert(token, l1, "Assert To Be Removed");
  }
  fb.Assert(token, x, "Assert To Be Kept");
  BValue l2 = fb.Literal(UBits(0, 32));
  fb.Concat({x, l2});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  // Should be 45 nodes to start.
  EXPECT_EQ(f->node_count(), 45);
  EXPECT_THAT(Run(f), status_testing::IsOkAndHolds(true));
  // We expect there to be five nodes left, same as above.
  // Asserts and literal 1 conditions removed, token rewired.
  EXPECT_EQ(f->node_count(), 5);
}

}  // namespace

}  // namespace xls
