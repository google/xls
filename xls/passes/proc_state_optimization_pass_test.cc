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

#include "xls/passes/proc_state_optimization_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using testing::ElementsAre;

class ProcStateOptimizationPassTest : public IrTestBase {
 protected:
  ProcStateOptimizationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    return ProcStateOptimizationPass().Run(p, PassOptions(), &results);
  }
};

TEST_F(ProcStateOptimizationPassTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam(), std::vector<BValue>()).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ProcStateOptimizationPassTest, SimpleNonoptimizableStateProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  pb.Send(out, pb.Add(x, y));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Not(x), pb.Not(y)}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
  EXPECT_EQ(proc->GetStateElementCount(), 2);
}

TEST_F(ProcStateOptimizationPassTest, ProcWithDeadElements) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  BValue z = pb.StateElement("z", Value(UBits(0, 32)));
  pb.Send(out, x);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Not(x), y, pb.Not(z)}));

  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "x");
}

TEST_F(ProcStateOptimizationPassTest, CrissCrossDeadElements) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({y, x}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 0);
}

TEST_F(ProcStateOptimizationPassTest, CrissCrossDeadAndLiveElements) {
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue a = pb.StateElement("a", Value(UBits(0, 32)));
  BValue b = pb.StateElement("b", Value(UBits(0, 32)));
  BValue c = pb.StateElement("c", Value(UBits(0, 32)));
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));

  pb.Send(out, c);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({b, c, a, y, x}));

  EXPECT_EQ(proc->GetStateElementCount(), 5);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_THAT(proc->StateParams(), ElementsAre(a.node(), b.node(), c.node()));
}

TEST_F(ProcStateOptimizationPassTest, ProcWithZeroWidthElement) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 0)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  BValue send = pb.Send(out, pb.Concat({x, y}));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Not(x), pb.Not(y)}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "y");
  EXPECT_THAT(send.node(),
              m::Send(m::Param("tkn"),
                      m::Concat(m::Literal(UBits(0, 0)), m::Param("y"))));
}

TEST_F(ProcStateOptimizationPassTest, StateElementsIntoTuplesAndOut) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  BValue z = pb.StateElement("z", Value(UBits(0, 32)));

  BValue xy = pb.Tuple({x, y});
  BValue xy_z = pb.Tuple({xy, z});

  // Send element `y` from the tuple.
  pb.Send(out, pb.TupleIndex(xy, 1));

  BValue next_x = y;
  BValue next_y = pb.TupleIndex(pb.TupleIndex(xy_z, 0), 0);
  BValue next_z = pb.TupleIndex(xy_z, 1);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({next_x, next_y, next_z}));

  EXPECT_EQ(proc->GetStateElementCount(), 3);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(proc->StateParams(), ElementsAre(x.node(), y.node()));
}

TEST_F(ProcStateOptimizationPassTest, ProcWithPartiallyDeadStateElement) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  Value zero(UBits(0, 32));
  BValue dead_state = pb.StateElement("dead", Value::Tuple({zero, zero}));
  BValue not_dead_state =
      pb.StateElement("not_dead", Value::Tuple({zero, zero}));
  // Send only one tuple element of the `not_dead` state.
  pb.Send(out, pb.TupleIndex(not_dead_state, 0));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({dead_state, not_dead_state}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "not_dead");
}

TEST_F(ProcStateOptimizationPassTest, LiteralChainOfSize1) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(100, 32)));
  BValue lit = pb.Literal(Value(UBits(200, 32)));
  BValue send = pb.Send(out, x);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({lit}));

  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);
  EXPECT_EQ(proc->GetStateParam(0)->GetType()->GetFlatBitCount(), 1);

  EXPECT_THAT(
      send.node(),
      m::Send(m::Param("tkn"), m::Select(m::Param("state_machine_x"),
                                         /*cases=*/{m::Literal(100)},
                                         /*default_value=*/m::Literal(200))));
}

}  // namespace
}  // namespace xls
