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

#include <string_view>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/block.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace xls {
namespace {

namespace m = xls::op_matchers;
using status_testing::IsOk;
using status_testing::IsOkAndHolds;
using status_testing::StatusIs;
using testing::Not;

class DeadCodeEliminationPassTest : public IrTestBase {
 protected:
  DeadCodeEliminationPassTest() = default;

  absl::StatusOr<bool> Run(FunctionBase* f) {
    PassResults results;
    return DeadCodeEliminationPass().RunOnFunctionBase(
        f, OptimizationPassOptions(), &results);
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
      after_all_1: token = after_all()
      add_2: bits[32] = add(x, y)
      literal_3: bits[1] = literal(value=1)
      my_cover: token = cover(after_all_1, literal_3, label="my_coverpoint")
      tuple_5: (token, bits[32]) = tuple(my_cover, add_2)
      tuple_index_6: token = tuple_index(tuple_5, index=0)
      tuple_index_7: bits[32] = tuple_index(tuple_5, index=1)
      dead_afterall: token = after_all()
      dead_sub: bits[32] = sub(x, y)
      ret sub_9: bits[32] = sub(tuple_index_7, y)
    }
  )",
                                                       p.get()));

  EXPECT_EQ(f->node_count(), 12);
  XLS_EXPECT_OK(f->GetNode("my_cover").status());
  XLS_EXPECT_OK(f->GetNode("dead_afterall").status());
  XLS_EXPECT_OK(f->GetNode("dead_afterall").status());

  EXPECT_THAT(Run(f), IsOkAndHolds(true));

  EXPECT_EQ(f->node_count(), 9);
  XLS_EXPECT_OK(f->GetNode("my_cover").status());
  EXPECT_THAT(f->GetNode("dead_afterall"),
              StatusIs(absl::StatusCode::kNotFound));
  EXPECT_THAT(f->GetNode("dead_sub"), StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(DeadCodeEliminationPassTest, Block) {
  auto p = CreatePackage();
  BlockBuilder b(TestName(), p.get());
  BValue x = b.InputPort("x", p->GetBitsType(32));
  BValue y = b.InputPort("y", p->GetBitsType(32));
  b.OutputPort("out", b.Add(x, y));

  // Create a dead literal.
  b.Literal(Value(UBits(123, 32)));

  // Create a dead input port (should not be removed).
  b.InputPort("z", p->GetBitsType(32));

  XLS_ASSERT_OK_AND_ASSIGN(Block * block, b.Build());

  EXPECT_EQ(block->node_count(), 6);
  EXPECT_THAT(Run(block), IsOkAndHolds(true));
  EXPECT_EQ(block->node_count(), 5);
}

TEST_F(DeadCodeEliminationPassTest, GateRemoved) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue one_bit = fb.Param("one_bit", p->GetBitsType(1));
  BValue eight_bit = fb.Param("eight_bit", p->GetBitsType(8));
  // Dead value, presumably strength reduction or something got rid of the need
  // for it, possibly something proved that one_bit is always true.
  BValue dead = fb.Gate(one_bit, eight_bit);
  constexpr std::string_view kDeadNodeName = "DeadGate";
  dead.SetName(kDeadNodeName);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(eight_bit));
  EXPECT_THAT(f->GetNode(kDeadNodeName), IsOkAndHolds(m::Gate()));
  EXPECT_THAT(Run(f), IsOkAndHolds(true));
  EXPECT_THAT(f->GetNode(kDeadNodeName), Not(IsOk()));
}

}  // namespace
}  // namespace xls
