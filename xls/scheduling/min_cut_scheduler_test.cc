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

#include "xls/scheduling/min_cut_scheduler.h"

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"
#include "xls/scheduling/schedule_graph.h"
#include "xls/scheduling/scheduling_options.h"

namespace xls {
namespace {

using ::testing::ElementsAre;

class MinCutSchedulerTest : public IrTestBase {};

TEST_F(MinCutSchedulerTest, MinCutCycleOrders) {
  EXPECT_THAT(GetMinCutCycleOrders(0), ElementsAre(std::vector<int64_t>()));
  EXPECT_THAT(GetMinCutCycleOrders(1), ElementsAre(std::vector<int64_t>({0})));
  EXPECT_THAT(
      GetMinCutCycleOrders(2),
      ElementsAre(std::vector<int64_t>({0, 1}), std::vector<int64_t>({1, 0})));
  EXPECT_THAT(GetMinCutCycleOrders(3),
              ElementsAre(std::vector<int64_t>({0, 1, 2}),
                          std::vector<int64_t>({2, 1, 0}),
                          std::vector<int64_t>({1, 0, 2})));
  EXPECT_THAT(GetMinCutCycleOrders(4),
              ElementsAre(std::vector<int64_t>({0, 1, 2, 3}),
                          std::vector<int64_t>({3, 2, 1, 0}),
                          std::vector<int64_t>({1, 0, 2, 3})));
  EXPECT_THAT(GetMinCutCycleOrders(5),
              ElementsAre(std::vector<int64_t>({0, 1, 2, 3, 4}),
                          std::vector<int64_t>({4, 3, 2, 1, 0}),
                          std::vector<int64_t>({2, 0, 1, 3, 4})));
  EXPECT_THAT(GetMinCutCycleOrders(8),
              ElementsAre(std::vector<int64_t>({0, 1, 2, 3, 4, 5, 6, 7}),
                          std::vector<int64_t>({7, 6, 5, 4, 3, 2, 1, 0}),
                          std::vector<int64_t>({3, 1, 0, 2, 5, 4, 6, 7})));
}

TEST_F(MinCutSchedulerTest, SimpleFunction) {
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
  MinCutScheduler scheduler(graph, delay_estimator);
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler.Schedule(/*pipeline_stages=*/1,
                         /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_TRUE(cycle_map.contains(x.node()));
  EXPECT_TRUE(cycle_map.contains(y.node()));
  EXPECT_TRUE(cycle_map.contains(add.node()));
}

TEST_F(MinCutSchedulerTest, SimpleProc) {
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
  MinCutScheduler scheduler(graph, delay_estimator);
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleCycleMap cycle_map,
      scheduler.Schedule(/*pipeline_stages=*/1,
                         /*clock_period_ps=*/2, SchedulingFailureBehavior{}));
  EXPECT_TRUE(cycle_map.contains(pb.GetStateParam(0).node()));
}

}  // namespace
}  // namespace xls
