// Copyright 2022 The XLS Authors
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

#include "xls/passes/proc_state_flattening_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/tuple_simplification_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;

class ProcStateFlatteningPassTest : public IrTestBase {
 protected:
  ProcStateFlatteningPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed, ProcStateFlatteningPass().Run(
                                           p, PassOptions(), &results));
    // Run tuple_simplification and dce to clean things up.
    XLS_RETURN_IF_ERROR(
        TupleSimplificationPass().Run(p, PassOptions(), &results).status());
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass().Run(p, PassOptions(), &results).status());
    return changed;
  }
};

TEST_F(ProcStateFlatteningPassTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam(), std::vector<BValue>()).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ProcStateFlatteningPassTest, NontupleStateProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 64)));
  BValue a = pb.StateElement("a", Value::UBitsArray({1, 2, 3}, 16).value());
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam(), {x, y, a}));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ProcStateFlatteningPassTest, EmptyTupleState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), {x}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  // An empty tuple decomposes into zero elements, so the proc should be
  // stateless now.
  EXPECT_EQ(proc->GetStateElementCount(), 0);
}

TEST_F(ProcStateFlatteningPassTest, MultipleEmptyTupleState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({}));
  BValue y = pb.StateElement("y", Value::Tuple({}));
  BValue z =
      pb.StateElement("z", Value::Tuple({Value::Tuple({}), Value::Tuple({})}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.GetTokenParam(), {x, y, z}));

  EXPECT_EQ(proc->GetStateElementCount(), 3);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  // All the empty tuples decompose into zero elements, so the proc should be
  // stateless now.
  EXPECT_EQ(proc->GetStateElementCount(), 0);
}

TEST_F(ProcStateFlatteningPassTest, EmptyTupleAndBitsState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({}));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  BValue z = pb.StateElement("z", Value::Tuple({}));
  BValue q = pb.StateElement("q", Value(UBits(0, 64)));

  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, pb.Build(pb.GetTokenParam(), {x, y, z, pb.Add(q, q)}));

  EXPECT_EQ(proc->GetStateElementCount(), 4);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateElementCount(), 2);

  // The name uniquer thinks the names "y" and "q" are already taken (as they
  // were the names of previously deleted nodes). So the new state params get
  // suffixes.
  // TODO(meheff): 2022/4/7 Figure out how to preserve the names.
  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "y__1");
  EXPECT_EQ(proc->GetInitValueElement(0), Value(UBits(0, 32)));
  EXPECT_THAT(proc->GetNextStateElement(0), m::Param("y__1"));

  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "q__1");
  EXPECT_EQ(proc->GetInitValueElement(1), Value(UBits(0, 64)));
  EXPECT_THAT(proc->GetNextStateElement(1),
              m::Add(m::Param("q__1"), m::Param("q__1")));
}

TEST_F(ProcStateFlatteningPassTest, TrivialTupleState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({Value(UBits(42, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), {x}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "x__1");
  EXPECT_EQ(proc->GetInitValueElement(0), Value(UBits(42, 32)));
  EXPECT_THAT(proc->GetNextStateElement(0), m::Param("x__1"));
}

TEST_F(ProcStateFlatteningPassTest, TrivialTupleStateWithNextExpression) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value::Tuple({Value(UBits(42, 32))}));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc,
      pb.Build(pb.GetTokenParam(), {pb.Tuple({pb.Not(pb.TupleIndex(x, 0))})}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "x__1");
  EXPECT_EQ(proc->GetInitValueElement(0), Value(UBits(42, 32)));
  EXPECT_THAT(proc->GetNextStateElement(0), m::Not(m::Param("x__1")));
}

TEST_F(ProcStateFlatteningPassTest, ComplicatedState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue a = pb.StateElement(
      "a",
      Value::Tuple({Value(UBits(1, 32)),
                    Value::Tuple({Value(UBits(2, 32)), Value(UBits(3, 32))})}));
  BValue b = pb.StateElement("b", Value(UBits(4, 32)));
  BValue c = pb.StateElement(
      "c", Value::Tuple({Value(UBits(5, 32)), Value(UBits(6, 32))}));

  BValue next_a = pb.Tuple({b, c});
  BValue next_b = pb.TupleIndex(a, 0);
  BValue next_c = pb.TupleIndex(a, 1);
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, pb.Build(pb.GetTokenParam(), {next_a, next_b, next_c}));

  EXPECT_EQ(proc->GetStateElementCount(), 3);

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateElementCount(), 6);

  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "a_0");
  EXPECT_THAT(proc->GetNextStateElement(0), m::Param("b__1"));

  EXPECT_EQ(proc->GetStateParam(1)->GetName(), "a_1");
  EXPECT_THAT(proc->GetNextStateElement(1), m::Param("c_0"));

  EXPECT_EQ(proc->GetStateParam(2)->GetName(), "a_2");
  EXPECT_THAT(proc->GetNextStateElement(2), m::Param("c_1"));

  EXPECT_EQ(proc->GetStateParam(3)->GetName(), "b__1");
  EXPECT_THAT(proc->GetNextStateElement(3), m::Param("a_0"));

  EXPECT_EQ(proc->GetStateParam(4)->GetName(), "c_0");
  EXPECT_THAT(proc->GetNextStateElement(4), m::Param("a_1"));

  EXPECT_EQ(proc->GetStateParam(5)->GetName(), "c_1");
  EXPECT_THAT(proc->GetNextStateElement(5), m::Param("a_2"));
}

}  // namespace
}  // namespace xls
