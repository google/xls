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

#include "xls/passes/dce_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/ir_test_base.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class DeadCodeEliminationPassTest : public IrTestBase {
 protected:
  DeadCodeEliminationPassTest() = default;

  absl::StatusOr<bool> Run(Function* f) {
    PassResults results;
    return DeadCodeEliminationPass().RunOnFunctionBase(f, PassOptions(),
                                                       &results);
  }
};

TEST_F(DeadCodeEliminationPassTest, NoDeadCode) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[42], y: bits[42]) -> bits[42] {
       ret neg.1: bits[42] = neg(x)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 3);
  EXPECT_THAT(Run(f), IsOkAndHolds(false));
  EXPECT_EQ(f->node_count(), 3);
}

TEST_F(DeadCodeEliminationPassTest, SomeDeadCode) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[42], y: bits[42]) -> bits[42] {
       neg.1: bits[42] = neg(x)
       add.2: bits[42] = add(x, y)
       neg.3: bits[42] = neg(add.2)
       ret sub.4: bits[42] = sub(neg.1, y)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 6);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 4);
}

TEST_F(DeadCodeEliminationPassTest, RepeatedOperand) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
     fn func(x: bits[42], y: bits[42]) -> bits[42] {
       neg.1: bits[42] = neg(x)
       add.2: bits[42] = add(neg.1, neg.1)
       ret sub.3: bits[42] = sub(x, y)
     }
  )",
                                                       p.get()));
  EXPECT_EQ(f->node_count(), 5);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 3);
}

// Verifies that the DCE pass doesn't remove side-effecting ops.
TEST_F(DeadCodeEliminationPassTest, AvoidsSideEffecting) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, ParseFunction(R"(
    fn has_token(x: bits[32], y: bits[32]) -> bits[32] {
      after_all.1: token = after_all()
      add.2: bits[32] = add(x, y)
      literal.3: bits[1] = literal(value=1)
      cover.4: token = cover(after_all.1, literal.3, label="my_coverpoint")
      tuple.5: (token, bits[32]) = tuple(cover.4, add.2)
      tuple_index.6: token = tuple_index(tuple.5, index=0)
      tuple_index.7: bits[32] = tuple_index(tuple.5, index=1)
      sub.8: bits[32] = sub(x, y)
      ret sub.9: bits[32] = sub(tuple_index.7, y)
    }
  )",
                                                       p.get()));

  EXPECT_EQ(f->node_count(), 11);
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_EQ(f->node_count(), 9);
}

}  // namespace
}  // namespace xls
