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

#include "xls/scheduling/sdc_scheduler.h"

#include <cstdint>
#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
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

class TestDelayEstimator : public DelayEstimator {
 public:
  TestDelayEstimator() : DelayEstimator("test") {}
  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    return 1;
  }
};

class SDCSchedulerTest : public IrTestBase {};

TEST_F(SDCSchedulerTest, SimpleFunction) {
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
  SchedulingOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto scheduler, SDCScheduler::Create(graph, delay_estimator, options));
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler->Schedule(/*pipeline_stages=*/std::nullopt,
                          /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_EQ(cycle_map.at(x.node()), 0);
  EXPECT_EQ(cycle_map.at(y.node()), 0);
  EXPECT_EQ(cycle_map.at(add.node()), 0);
}

TEST_F(SDCSchedulerTest, SimpleProc) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  pb.StateElement("x", Value(UBits(0, 32)));
  pb.Next(pb.GetStateParam(0),
          pb.Add(pb.GetStateParam(0), pb.Literal(UBits(1, 32))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(proc, /*dead_after_synthesis=*/{}));
  TestDelayEstimator delay_estimator;
  SchedulingOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto scheduler, SDCScheduler::Create(graph, delay_estimator, options));
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler->Schedule(/*pipeline_stages=*/std::nullopt,
                          /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_EQ(cycle_map.at(pb.GetStateParam(0).node()), 0);
}

TEST_F(SDCSchedulerTest, WithConstraint) {
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
  SchedulingOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto scheduler, SDCScheduler::Create(graph, delay_estimator, options));
  XLS_ASSERT_OK(
      scheduler->AddConstraints({NodeInCycleConstraint(add.node(), 2)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler->Schedule(/*pipeline_stages=*/std::nullopt,
                          /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_EQ(cycle_map.at(x.node()), 0);
  EXPECT_EQ(cycle_map.at(y.node()), 0);
  EXPECT_EQ(cycle_map.at(add.node()), 2);
}

TEST_F(SDCSchedulerTest, WithIOConstraint) {
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

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(proc, /*dead_after_synthesis=*/{}));
  TestDelayEstimator delay_estimator;
  SchedulingOptions options;
  XLS_ASSERT_OK_AND_ASSIGN(
      auto scheduler, SDCScheduler::Create(graph, delay_estimator, options));
  XLS_ASSERT_OK(scheduler->AddConstraints({IOConstraint(
      "a", IODirection::kReceive, "b", IODirection::kSend, 1, 1)}));
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler->Schedule(/*pipeline_stages=*/std::nullopt,
                          /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_EQ(cycle_map.at(rcv.node()), 0);
  EXPECT_EQ(cycle_map.at(send.node()), 1);
}

TEST_F(SDCSchedulerTest, DecoupledThroughputConstraintsEnforced) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * x_element,
                           pb.UnreadStateElement("x", Value(UBits(0, 32)),
                                                 /*non_synthesizable=*/false));
  BValue x_read = pb.StateRead(x_element);
  BValue add1 = pb.Add(x_read, pb.Literal(UBits(1, 32)));
  BValue add2 = pb.Add(add1, pb.Literal(UBits(1, 32)));
  BValue add3 = pb.Add(add2, pb.Literal(UBits(1, 32)));
  pb.Next(x_element, add3);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(ScheduleGraph graph,
                           ScheduleGraph::Create(proc, {}));
  TestDelayEstimator delay_estimator;
  SchedulingOptions options;

  // 1. Infeasible check: With unbounded stages std::nullopt and tight
  // target worst_case_throughput = 1, AddThroughputConstraint forces
  // cycle(Next) - cycle(StateRead) <= worst_case_throughput.
  // 3 cycles (> 1), which returns infeasible.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto infeasible_scheduler,
      SDCScheduler::Create(graph, delay_estimator, options));
  auto infeasible_result = infeasible_scheduler->Schedule(
      /*pipeline_stages=*/std::nullopt, /*clock_period_ps=*/1,
      SchedulingFailureBehavior{.explain_infeasibility = false},
      /*worst_case_throughput=*/1);
  EXPECT_THAT(infeasible_result,
              absl_testing::StatusIs(
                  absl::StatusCode::kInternal,
                  testing::HasSubstr("does not have an optimal solution")));

  // 2. Feasible check: Relax worst_case_throughput = 4, unbounded stages,
  // and clock_period_ps = 2. With 2 ps per cycle, the 5 1-ps nodes
  // (StateRead -> add1 -> add2 -> add3 -> Next) complete in 2 clock cycles
  // (cycle(Next) - cycle(read) == 2), with cycle(Next) = 2 and
  // cycle(read) = 0.
  XLS_ASSERT_OK_AND_ASSIGN(
      auto feasible_scheduler,
      SDCScheduler::Create(graph, delay_estimator, options));
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      feasible_scheduler->Schedule(
          /*pipeline_stages=*/std::nullopt, /*clock_period_ps=*/2,
          SchedulingFailureBehavior{.explain_infeasibility = false},
          /*worst_case_throughput=*/4));
  EXPECT_EQ(
      cycle_map.at(*proc->next_values().begin()) - cycle_map.at(x_read.node()),
      2);
}

}  // namespace
}  // namespace xls
