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

TEST_F(ProcStateOptimizationPassTest, ProcWithOneDeadElement) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                               p->GetBitsType(32)));

  TokenlessProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  pb.Send(out, x);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Not(x), y}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "x");
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

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Not(x), y}));

  EXPECT_EQ(proc->GetStateElementCount(), 2);
  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_EQ(proc->GetStateElementCount(), 1);

  EXPECT_EQ(proc->GetStateParam(0)->GetName(), "y");
  EXPECT_THAT(send.node(),
              m::Send(m::Param("tkn"),
                      m::Concat(m::Literal(UBits(0, 0)), m::Param("y"))));
}

}  // namespace
}  // namespace xls
