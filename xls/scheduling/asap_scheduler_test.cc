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

#include "xls/scheduling/asap_scheduler.h"

#include <cstdint>
#include <optional>

#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {

class ASAPSchedulerTest : public IrTestBase {};

TEST_F(ASAPSchedulerTest, SimpleFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  auto add = fb.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(f, /*dead_after_synthesis=*/{}));
  ASAPScheduler scheduler(graph, delay_estimator);
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler.Schedule(/*pipeline_stages=*/std::nullopt,
                         /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_EQ(cycle_map.at(x.node()), 0);
  EXPECT_EQ(cycle_map.at(y.node()), 0);
  EXPECT_EQ(cycle_map.at(add.node()), 0);
}

TEST_F(ASAPSchedulerTest, SimpleProc) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  pb.StateElement("x", Value(UBits(0, 32)));
  pb.Next(pb.GetStateParam(0),
          pb.Add(pb.GetStateParam(0), pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(proc, /*dead_after_synthesis=*/{}));
  ASAPScheduler scheduler(graph, delay_estimator);
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler.Schedule(/*pipeline_stages=*/std::nullopt,
                         /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_EQ(cycle_map.at(pb.GetStateParam(0).node()), 0);
}

TEST_F(ASAPSchedulerTest, WithConstraint) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  auto add = fb.Add(x, y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(f, /*dead_after_synthesis=*/{}));
  TestDelayEstimator delay_estimator;
  ASAPScheduler scheduler(graph, delay_estimator);
  XLS_ASSERT_OK(
      scheduler.AddConstraints({NodeInCycleConstraint(add.node(), 2)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler.Schedule(/*pipeline_stages=*/std::nullopt,
                         /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_EQ(cycle_map.at(x.node()), 0);
  EXPECT_EQ(cycle_map.at(y.node()), 0);
  EXPECT_EQ(cycle_map.at(add.node()), 2);
}

TEST_F(ASAPSchedulerTest, WithIOConstraint) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_a, p->CreateStreamingChannel("a", ChannelOps::kReceiveOnly,
                                                p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_b,
                           p->CreateStreamingChannel("b", ChannelOps::kSendOnly,
                                                     p->GetBitsType(32)));

  ProcBuilder pb(TestName(), p.get());
  BValue tkn = pb.Literal(Value::Token());
  BValue rcv = pb.Receive(ch_a, tkn);
  BValue send = pb.Send(ch_b, pb.TupleIndex(rcv, 0), pb.TupleIndex(rcv, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  TestDelayEstimator delay_estimator;
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(proc, /*dead_after_synthesis=*/{}));
  ASAPScheduler scheduler(graph, delay_estimator);
  XLS_ASSERT_OK(scheduler.AddConstraints({IOConstraint(
      "a", IODirection::kReceive, "b", IODirection::kSend, 1, 1)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler.Schedule(/*pipeline_stages=*/std::nullopt,
                         /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_EQ(cycle_map.at(rcv.node()), 0);
  EXPECT_EQ(cycle_map.at(send.node()), 1);
}

}  // namespace
}  // namespace xls
