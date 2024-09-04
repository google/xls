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

#include "xls/scheduling/schedule_bounds.h"

#include <cstdint>
#include <limits>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/op.h"

namespace xls {
namespace sched {
namespace {

using testing::Pair;

class TestDelayEstimator : public DelayEstimator {
 public:
  TestDelayEstimator() : DelayEstimator("test") {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kParam:
      case Op::kLiteral:
      case Op::kBitSlice:
      case Op::kConcat:
      case Op::kMinDelay:
        return 0;
      default:
        return 1;
    }
  }
};

class ScheduleBoundsTest : public IrTestBase {
 protected:
  TestDelayEstimator delay_estimator_;
};

TEST_F(ScheduleBoundsTest, ParameterOnlyFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(ScheduleBounds bounds,
                           ScheduleBounds::ComputeAsapAndAlapBounds(
                               f,
                               /*clock_period_ps=*/1, delay_estimator_));
  EXPECT_EQ(bounds.lb(x.node()), 0);
  EXPECT_EQ(bounds.ub(x.node()), 0);
  EXPECT_EQ(bounds.lb(y.node()), 0);
  EXPECT_EQ(bounds.ub(y.node()), 0);
}

TEST_F(ScheduleBoundsTest, SimpleExpressionAsapAndAlap) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  auto not_x = fb.Not(x);
  auto x_plus_y = fb.Add(x, y);
  auto not_x_plus_y = fb.Not(x_plus_y);
  auto result = fb.Add(not_x, not_x_plus_y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(ScheduleBounds bounds,
                           ScheduleBounds::ComputeAsapAndAlapBounds(
                               f,
                               /*clock_period_ps=*/1, delay_estimator_));
  EXPECT_THAT(bounds.bounds(x.node()), Pair(0, 0));
  EXPECT_THAT(bounds.bounds(y.node()), Pair(0, 0));
  EXPECT_THAT(bounds.bounds(not_x.node()), Pair(0, 1));
  EXPECT_THAT(bounds.bounds(x_plus_y.node()), Pair(0, 0));
  EXPECT_THAT(bounds.bounds(not_x_plus_y.node()), Pair(1, 1));
  EXPECT_THAT(bounds.bounds(result.node()), Pair(2, 2));
}

TEST_F(ScheduleBoundsTest, SimpleExpressionTightenBounds) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  auto not_x = fb.Not(x);
  auto x_plus_y = fb.Add(x, y);
  auto not_x_plus_y = fb.Not(x_plus_y);
  auto result = fb.Add(not_x, not_x_plus_y);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ScheduleBounds bounds(f, /*clock_period_ps=*/1, delay_estimator_);
  EXPECT_EQ(bounds.lb(x.node()), 0);
  EXPECT_EQ(bounds.lb(y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x.node()), 0);
  EXPECT_EQ(bounds.lb(x_plus_y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x_plus_y.node()), 0);
  EXPECT_EQ(bounds.lb(result.node()), 0);

  XLS_ASSERT_OK(bounds.PropagateLowerBounds());

  // The initial call to PropagateLowerBounds should make all the lower bounds
  // satisfy dependency and clock period constraints.
  EXPECT_EQ(bounds.lb(x.node()), 0);
  EXPECT_EQ(bounds.lb(y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x.node()), 0);
  EXPECT_EQ(bounds.lb(x_plus_y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x_plus_y.node()), 1);
  EXPECT_EQ(bounds.lb(result.node()), 2);

  // Upper bounds should still be all at their limit.
  const int64_t kMax = std::numeric_limits<int64_t>::max();
  EXPECT_EQ(bounds.ub(x.node()), kMax);
  EXPECT_EQ(bounds.ub(y.node()), kMax);
  EXPECT_EQ(bounds.ub(not_x.node()), kMax);
  EXPECT_EQ(bounds.ub(x_plus_y.node()), kMax);
  EXPECT_EQ(bounds.ub(not_x_plus_y.node()), kMax);
  EXPECT_EQ(bounds.ub(result.node()), kMax);

  // Tightening the ub of one node in the graph and propagating should tighten
  // the upper bounds of the predecessors of that node.
  XLS_ASSERT_OK(bounds.TightenNodeUb(not_x_plus_y.node(), 42));
  XLS_ASSERT_OK(bounds.PropagateUpperBounds());

  EXPECT_EQ(bounds.ub(not_x.node()), kMax);
  EXPECT_EQ(bounds.ub(x_plus_y.node()), 41);
  EXPECT_EQ(bounds.ub(not_x_plus_y.node()), 42);
  EXPECT_EQ(bounds.ub(result.node()), kMax);

  // Tightening the ub of the root node should give every node a non-INT_MAX
  // upper bound.
  XLS_ASSERT_OK(bounds.TightenNodeUb(result.node(), 100));
  XLS_ASSERT_OK(bounds.PropagateUpperBounds());

  EXPECT_EQ(bounds.ub(not_x.node()), 99);
  EXPECT_EQ(bounds.ub(x_plus_y.node()), 41);
  EXPECT_EQ(bounds.ub(not_x_plus_y.node()), 42);
  EXPECT_EQ(bounds.ub(result.node()), 100);

  // Setting one node's lb and propagating should result in further tightening.
  XLS_ASSERT_OK(bounds.TightenNodeLb(not_x.node(), 22));
  XLS_ASSERT_OK(bounds.PropagateLowerBounds());

  EXPECT_EQ(bounds.lb(x.node()), 0);
  EXPECT_EQ(bounds.lb(y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x.node()), 22);
  EXPECT_EQ(bounds.lb(x_plus_y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x_plus_y.node()), 1);
  EXPECT_EQ(bounds.lb(result.node()), 23);
}

}  // namespace
}  // namespace sched
}  // namespace xls
