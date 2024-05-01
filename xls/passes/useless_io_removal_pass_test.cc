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

#include "xls/passes/useless_io_removal_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {

namespace {

class UselessIORemovalPassTest : public IrTestBase {
 protected:
  UselessIORemovalPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(
        bool changed,
        UselessIORemovalPass().Run(p, OptimizationPassOptions(), &results));
    // Run dce to clean things up.
    for (FunctionBase* f : p->GetFunctionBases()) {
      XLS_RETURN_IF_ERROR(
          DeadCodeEliminationPass()
              .RunOnFunctionBase(f, OptimizationPassOptions(), &results)
              .status());
    }
    // Return whether useless IO removal changed anything.
    return changed;
  }
};

TEST_F(UselessIORemovalPassTest, DontRemoveOnlySend) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * channel,
      p->CreateStreamingChannel("test_channel", ChannelOps::kSendOnly,
                                p->GetBitsType(32)));
  ProcBuilder pb(TestName(), "token", p.get());
  pb.StateElement("state", Value(UBits(0, 0)));
  BValue token = pb.SendIf(channel, pb.GetTokenParam(), pb.Literal(UBits(0, 1)),
                           pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(token, {pb.Literal(UBits(0, 0))}));
  EXPECT_EQ(proc->node_count(), 6);
  EXPECT_THAT(Run(p.get()), status_testing::IsOkAndHolds(false));
  EXPECT_EQ(proc->node_count(), 6);
}

TEST_F(UselessIORemovalPassTest, DontRemoveOnlySendNewStyle) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc(), TestName(), "token", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      SendChannelReference * channel,
      pb.AddOutputChannel("test_channel", p->GetBitsType(32)));
  pb.StateElement("state", Value(UBits(0, 0)));
  pb.SendIf(channel, pb.Literal(UBits(0, 1)), pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Literal(UBits(0, 0))}));
  EXPECT_EQ(proc->node_count(), 6);
  EXPECT_THAT(Run(p.get()), status_testing::IsOkAndHolds(false));
  EXPECT_EQ(proc->node_count(), 6);
}

TEST_F(UselessIORemovalPassTest, RemoveSendIfLiteralFalse) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * channel,
      p->CreateStreamingChannel("test_channel", ChannelOps::kSendOnly,
                                p->GetBitsType(32)));
  ProcBuilder pb(TestName(), "token", p.get());
  pb.StateElement("state", Value(UBits(0, 0)));
  BValue token = pb.SendIf(channel, pb.GetTokenParam(), pb.Literal(UBits(0, 1)),
                           pb.Literal(UBits(1, 32)));
  // Extra send so that this does something
  token = pb.Send(channel, token, pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(token, {pb.Literal(UBits(0, 0))}));
  EXPECT_EQ(proc->node_count(), 8);
  EXPECT_THAT(Run(p.get()), status_testing::IsOkAndHolds(true));
  EXPECT_EQ(proc->node_count(), 5);
  EXPECT_THAT(proc->NextToken(), m::Send(proc->TokenParam(), m::Literal(1)));
}

TEST_F(UselessIORemovalPassTest, RemoveSendIfLiteralFalseNewStyle) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc(), TestName(), "token", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      SendChannelReference * channel,
      pb.AddOutputChannel("test_channel", p->GetBitsType(32)));
  pb.StateElement("state", Value(UBits(0, 0)));
  pb.SendIf(channel, pb.Literal(UBits(0, 1)), pb.Literal(UBits(1, 32)));
  // Extra send so that this does something
  pb.Send(channel, pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.Literal(UBits(0, 0))}));

  EXPECT_EQ(proc->node_count(), 8);
  EXPECT_THAT(Run(p.get()), status_testing::IsOkAndHolds(true));
  EXPECT_EQ(proc->node_count(), 5);
  EXPECT_THAT(proc->NextToken(), m::Send(proc->TokenParam(), m::Literal(1)));
}

TEST_F(UselessIORemovalPassTest, DontRemoveOnlyReceive) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * channel,
      p->CreateStreamingChannel("test_channel", ChannelOps::kReceiveOnly,
                                p->GetBitsType(32)));
  ProcBuilder pb(TestName(), "token", p.get());
  pb.StateElement("state", Value(UBits(0, 32)));
  BValue token_and_result =
      pb.ReceiveIf(channel, pb.GetTokenParam(), pb.Literal(UBits(0, 1)));
  BValue token = pb.TupleIndex(token_and_result, 0);
  BValue result = pb.TupleIndex(token_and_result, 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(token, {result}));
  EXPECT_EQ(proc->node_count(), 6);
  EXPECT_THAT(Run(p.get()), status_testing::IsOkAndHolds(false));
  EXPECT_EQ(proc->node_count(), 6);
}

TEST_F(UselessIORemovalPassTest, RemoveReceiveIfLiteralFalse) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * channel,
      p->CreateStreamingChannel("test_channel", ChannelOps::kReceiveOnly,
                                p->GetBitsType(32)));
  ProcBuilder pb(TestName(), "token", p.get());
  pb.StateElement("state", Value(UBits(0, 32)));
  BValue token = pb.TupleIndex(pb.Receive(channel, pb.GetTokenParam()), 0);
  BValue token_and_result =
      pb.ReceiveIf(channel, token, pb.Literal(UBits(0, 1)));
  token = pb.TupleIndex(token_and_result, 0);
  BValue result = pb.TupleIndex(token_and_result, 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(token, {result}));
  EXPECT_EQ(proc->node_count(), 8);
  EXPECT_THAT(Run(p.get()), status_testing::IsOkAndHolds(true));
  EXPECT_EQ(proc->node_count(), 8);
  auto tuple = m::Tuple(
      m::TupleIndex(m::Receive(proc->TokenParam(), channel), 0), m::Literal(0));
  EXPECT_THAT(proc->NextToken(), m::TupleIndex(tuple, 0));
  EXPECT_THAT(proc->GetNextStateElement(0), m::TupleIndex(tuple, 1));
}

TEST_F(UselessIORemovalPassTest, RemoveSendPredIfLiteralTrue) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * channel,
      p->CreateStreamingChannel("test_channel", ChannelOps::kSendOnly,
                                p->GetBitsType(32)));
  ProcBuilder pb(TestName(), "token", p.get());
  pb.StateElement("state", Value(UBits(0, 0)));
  BValue token = pb.SendIf(channel, pb.GetTokenParam(), pb.Literal(UBits(1, 1)),
                           pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(token, {pb.Literal(UBits(0, 0))}));
  EXPECT_EQ(proc->node_count(), 6);
  EXPECT_THAT(Run(p.get()), status_testing::IsOkAndHolds(true));
  EXPECT_EQ(proc->node_count(), 5);
  EXPECT_THAT(proc->NextToken(), m::Send(proc->TokenParam(), m::Literal(1)));
  EXPECT_THAT(proc->GetNextStateElement(0), m::Literal(0));
}

TEST_F(UselessIORemovalPassTest, RemoveReceivePredIfLiteralTrue) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      StreamingChannel * channel,
      p->CreateStreamingChannel("test_channel", ChannelOps::kReceiveOnly,
                                p->GetBitsType(32)));
  ProcBuilder pb(TestName(), "token", p.get());
  pb.StateElement("state", Value(UBits(0, 32)));
  BValue token_and_result =
      pb.ReceiveIf(channel, pb.GetTokenParam(), pb.Literal(UBits(1, 1)));
  BValue token = pb.TupleIndex(token_and_result, 0);
  BValue result = pb.TupleIndex(token_and_result, 1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(token, {result}));
  EXPECT_EQ(proc->node_count(), 6);
  EXPECT_THAT(Run(p.get()), status_testing::IsOkAndHolds(true));
  EXPECT_EQ(proc->node_count(), 5);
  auto tuple = m::Receive(proc->TokenParam(), channel);
  EXPECT_THAT(proc->NextToken(), m::TupleIndex(tuple, 0));
  EXPECT_THAT(proc->GetNextStateElement(0), m::TupleIndex(tuple, 1));
}

}  // namespace

}  // namespace xls
