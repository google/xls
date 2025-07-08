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

#include "xls/scheduling/pipeline_schedule.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/fdo/delay_manager.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/type.h"
#include "xls/ir/value.h"
#include "xls/scheduling/pipeline_schedule.pb.h"
#include "xls/scheduling/run_pipeline_schedule.h"
#include "xls/scheduling/scheduling_options.h"

namespace m = ::xls::op_matchers;

// Matcher for use with UnorderedPointwise where you get a tuple<pair<key_t,
// value_v>, key_t>, useful for checking that an array's elements are the same
// as the keys of a map.
MATCHER(KeyEqElement, "") { return std::get<0>(arg).first == std::get<1>(arg); }

// Matcher to check that all nodes in FunctionBase `arg` are scheduled in the
// matcher param.
MATCHER_P(AllNodesScheduled, schedules, "") {
  const ::xls::PipelineSchedule& schedule = schedules.at(arg);
  for (::xls::Node* node : arg->nodes()) {
    if (!schedule.IsScheduled(node)) {
      *result_listener << absl::StreamFormat("%v  is not scheduled.", *node);
      return false;
    }
  }
  return true;
}

// Matcher to check that lhs and rhs PackageSchedule both have the same
// schedule.
MATCHER_P2(CyclesMatch, lhs, rhs, "") {
  const ::xls::PipelineSchedule& lhs_schedule = lhs.at(arg);
  const ::xls::PipelineSchedule& rhs_schedule = rhs.at(arg);
  for (::xls::Node* node : arg->nodes()) {
    if (lhs_schedule.cycle(node) != rhs_schedule.cycle(node)) {
      *result_listener << absl::StreamFormat(
          "%v (%d) is not scheduled in the same cycle as clone (%d).", *node,
          lhs_schedule.cycle(node), rhs_schedule.cycle(node));
      return false;
    }
  }
  return true;
}

MATCHER_P(OperationDelayInPs, matcher,
          absl::StrCat("operation delay ",
                       ::testing::DescribeMatcher<absl::StatusOr<int64_t>>(
                           matcher, negation))) {
  xls::TestDelayEstimator estimator;
  absl::StatusOr<int64_t> delay = estimator.GetOperationDelayInPs(arg);
  if (delay.ok()) {
    *result_listener << absl::StreamFormat("operation delay is %d", *delay);
  } else {
    *result_listener << "operation delay is not ok: " << delay.status();
  }
  return ::testing::ExplainMatchResult(matcher, delay, result_listener);
}

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedPointwise;

class PipelineScheduleTest : public IrTestBase {};

TEST_F(PipelineScheduleTest, SelectsEntry) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  SchedulingOptions options(SchedulingStrategy::ASAP);
  options.clock_period_ps(2);

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(f, TestDelayEstimator(), options));
  EXPECT_EQ(schedule.length(), 1);
  EXPECT_THAT(schedule.nodes_in_cycle(0), UnorderedElementsAre(m::Param()));

  FunctionBuilder fb_2("other_fn", p.get());
  fb_2.Literal(Value(UBits(16, 16)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f_2, fb_2.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, RunPipelineSchedule(f_2, TestDelayEstimator(), options));
  EXPECT_EQ(schedule.length(), 1);
  EXPECT_THAT(schedule.nodes_in_cycle(0), UnorderedElementsAre(m::Literal()));
}

TEST_F(PipelineScheduleTest, AsapScheduleTrivial) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Param("x", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          f, TestDelayEstimator(),
          SchedulingOptions(SchedulingStrategy::ASAP).clock_period_ps(2)));

  EXPECT_EQ(schedule.length(), 1);
  EXPECT_THAT(schedule.nodes_in_cycle(0), UnorderedElementsAre(m::Param()));
}

TEST_F(PipelineScheduleTest, OutrightInfeasibleSchedule) {
  // Create a schedule in which the critical path doesn't even fit in the
  // requested clock_period * stages.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Not(fb.Not(fb.Not(fb.Not(x))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(
      RunPipelineSchedule(f, TestDelayEstimator(),
                          SchedulingOptions(SchedulingStrategy::MIN_CUT)
                              .clock_period_ps(1)
                              .pipeline_stages(2))
          .status(),
      StatusIs(
          absl::StatusCode::kResourceExhausted,
          HasSubstr(
              "Cannot be scheduled in 2 stages. Computed lower bound is 4.")));
}

TEST_F(PipelineScheduleTest, InfeasibleScheduleWithBinPacking) {
  // Create a schedule in which the critical path fits in the requested
  // clock_period * stages, but there is no way to bin pack the instructions
  // into the stages such that the schedule is met.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Not(fb.UDiv(fb.Not(x), fb.Not(x)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(
      RunPipelineSchedule(f, TestDelayEstimator(),
                          SchedulingOptions(SchedulingStrategy::MIN_CUT)
                              .clock_period_ps(2)
                              .pipeline_stages(2))
          .status(),
      StatusIs(
          absl::StatusCode::kResourceExhausted,
          HasSubstr(
              "Cannot be scheduled in 2 stages. Computed lower bound is 3.")));
}

TEST_F(PipelineScheduleTest, InfeasibleScheduleWithReturnValueUsers) {
  // Create function which has users of the return value node such that the
  // return value cannot be scheduled in the final cycle.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  BValue ret_value = fb.Not(x, SourceInfo(), "ret_value");
  fb.Gate(fb.Literal(UBits(1, 1)), ret_value);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret_value));

  ASSERT_THAT(
      RunPipelineSchedule(f, TestDelayEstimator(),
                          SchedulingOptions(SchedulingStrategy::MIN_CUT)
                              .clock_period_ps(1)
                              .pipeline_stages(2))
          .status(),
      StatusIs(
          absl::StatusCode::kResourceExhausted,
          HasSubstr(
              "the following node(s) must be scheduled in the final cycle but "
              "that is impossible due to users of these node(s): ret_value")));
}

TEST_F(PipelineScheduleTest, AsapScheduleNoParameters) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Negate(fb.Add(fb.Literal(UBits(42, 8)), fb.Literal(UBits(100, 8))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          f, TestDelayEstimator(),
          SchedulingOptions(SchedulingStrategy::ASAP).clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(m::Add(), m::Literal(42), m::Literal(100)));
  EXPECT_THAT(schedule.nodes_in_cycle(1), UnorderedElementsAre(m::Neg()));
}

TEST_F(PipelineScheduleTest, AsapScheduleIncrementChain) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Add(fb.Add(x, fb.Literal(UBits(1, 32))), fb.Literal(UBits(1, 32))),
         fb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          f, TestDelayEstimator(),
          SchedulingOptions(SchedulingStrategy::ASAP).clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(m::Param("x"), m::Add(), m::Literal(1),
                                   m::Literal(1), m::Literal(1)));
  EXPECT_THAT(schedule.nodes_in_cycle(1), UnorderedElementsAre(m::Add()));
  EXPECT_THAT(schedule.nodes_in_cycle(2), UnorderedElementsAre(m::Add()));
}

TEST_F(PipelineScheduleTest, MinimizeRegisterBitslices) {
  // When minimizing registers, bit-slices should be hoisted in the schedule if
  // their operand is not otherwise live, and sunk in the schedule if their
  // operand *is* otherwise live.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  auto x_slice = fb.BitSlice(x, /*start=*/8, /*width=*/8);
  auto y_slice = fb.BitSlice(y, /*start=*/8, /*width=*/8);
  auto neg_neg_y = fb.Negate(fb.Negate(y));
  // 'x' is live throughout the function, 'y' is not.
  fb.Concat({x, x_slice, y_slice, neg_neg_y});

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(f, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(m::Param("x"), m::Param("y"),
                                   m::BitSlice(m::Param("y")), m::Neg()));
  EXPECT_THAT(
      schedule.nodes_in_cycle(1),
      UnorderedElementsAre(m::BitSlice(m::Param("x")), m::Neg(), m::Concat()));
}

TEST_F(PipelineScheduleTest, AsapScheduleComplex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  fb.Negate(fb.Concat({(fb.Not(x | y) - z) * x, z + z}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          f, TestDelayEstimator(),
          SchedulingOptions(SchedulingStrategy::ASAP).clock_period_ps(2)));

  EXPECT_EQ(schedule.length(), 3);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(m::Param("x"), m::Param("y"), m::Param("z"),
                                   m::Or(), m::Not(), m::Add()));
  EXPECT_THAT(schedule.nodes_in_cycle(1),
              UnorderedElementsAre(m::Concat(), m::Sub(), m::UMul()));
  EXPECT_THAT(schedule.nodes_in_cycle(2), UnorderedElementsAre(m::Neg()));
}

TEST_F(PipelineScheduleTest, JustClockPeriodGiven) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  fb.Negate(fb.Concat({(fb.Not(x | y) - z) * x, z + z}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(2)));

  // Returns the unique scheduled Ops in the given cycle.
  auto scheduled_ops = [&](int64_t cycle) {
    absl::flat_hash_set<Op> ops;
    for (const auto& node : schedule.nodes_in_cycle(cycle)) {
      ops.insert(node->op());
    }
    return ops;
  };

  EXPECT_EQ(schedule.length(), 3);
  EXPECT_THAT(scheduled_ops(0),
              UnorderedElementsAre(Op::kParam, Op::kOr, Op::kNot));
  EXPECT_THAT(scheduled_ops(1),
              UnorderedElementsAre(Op::kAdd, Op::kConcat, Op::kUMul, Op::kSub));
  EXPECT_THAT(scheduled_ops(2), UnorderedElementsAre(Op::kNeg));
  EXPECT_THAT(scheduled_ops(3), UnorderedElementsAre());
}

TEST_F(PipelineScheduleTest, TestVerifyTiming) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto x_plus_y = fb.Add(x, y);
  fb.Subtract(x_plus_y, fb.Negate(x_plus_y));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(5)));

  EXPECT_EQ(schedule.length(), 1);
  XLS_EXPECT_OK(
      schedule.VerifyTiming(/*clock_period_ps=*/5, TestDelayEstimator()));
  EXPECT_THAT(
      schedule.VerifyTiming(/*clock_period_ps=*/1, TestDelayEstimator()),
      absl_testing::StatusIs(
          absl::StatusCode::kInternal,
          ::testing::HasSubstr(
              "Schedule does not meet timing (1ps). Longest failing path "
              "(3ps): add.3 (1ps) -> neg.4 (1ps) -> sub.5 (1ps)")));

  DelayManager delay_manager(func, TestDelayEstimator());
  XLS_EXPECT_OK(schedule.VerifyTiming(/*clock_period_ps=*/5, delay_manager));
  EXPECT_THAT(
      schedule.VerifyTiming(/*clock_period_ps=*/1, delay_manager),
      absl_testing::StatusIs(
          absl::StatusCode::kInternal,
          ::testing::HasSubstr(
              "Schedule does not meet timing (1ps). Longest failing path "
              "(3ps): add.3 (1ps) -> neg.4 (1ps) -> sub.5 (1ps)")));
}

TEST_F(PipelineScheduleTest, ClockPeriodAndPipelineLengthGiven) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  fb.Negate(fb.Concat({(fb.Not(fb.Negate(x | y)) - z) * x, z + z}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          func, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(2).pipeline_stages(4)));

  // Returns the unique scheduled Ops in the given cycle.
  auto scheduled_ops = [&](int64_t cycle) {
    absl::flat_hash_set<Op> ops;
    for (const auto& node : schedule.nodes_in_cycle(cycle)) {
      ops.insert(node->op());
    }
    return ops;
  };

  EXPECT_EQ(schedule.length(), 4);
  EXPECT_THAT(scheduled_ops(0),
              UnorderedElementsAre(Op::kParam, Op::kOr, Op::kNeg));
  EXPECT_THAT(scheduled_ops(1),
              UnorderedElementsAre(Op::kAdd, Op::kNot, Op::kSub));
  EXPECT_THAT(scheduled_ops(2), UnorderedElementsAre(Op::kConcat, Op::kUMul));
  EXPECT_THAT(scheduled_ops(3), UnorderedElementsAre(Op::kNeg));
}

TEST_F(PipelineScheduleTest, JustPipelineLengthGiven) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  fb.Negate(fb.Concat({(fb.Not(fb.Negate(x | y)) - z) * x, z + z}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(6)));

  // Returns the unique scheduled Ops in the given cycle.
  auto scheduled_ops = [&](int64_t cycle) {
    absl::flat_hash_set<Op> ops;
    for (const auto& node : schedule.nodes_in_cycle(cycle)) {
      ops.insert(node->op());
    }
    return ops;
  };

  EXPECT_EQ(schedule.length(), 6);

  // The maximum delay of any stage should be minimum feasible value. In this
  // case it is 1ps, which means there should be no dependent instructions in
  // single cycle.
  EXPECT_THAT(scheduled_ops(0), UnorderedElementsAre(Op::kParam, Op::kOr));
  EXPECT_THAT(scheduled_ops(1), UnorderedElementsAre(Op::kNeg));
  EXPECT_THAT(scheduled_ops(2), UnorderedElementsAre(Op::kNot));
  EXPECT_THAT(scheduled_ops(3), UnorderedElementsAre(Op::kAdd, Op::kSub));
  EXPECT_THAT(scheduled_ops(4), UnorderedElementsAre(Op::kConcat, Op::kUMul));
  EXPECT_THAT(scheduled_ops(5), UnorderedElementsAre(Op::kNeg));
}

TEST_F(PipelineScheduleTest, LongPipelineLength) {
  // Generate an absurdly long pipeline schedule. Most stages are empty, but it
  // should not crash.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto bitslice = fb.BitSlice(x, /*start=*/7, /*width=*/20);
  auto zero_ext = fb.ZeroExtend(bitslice, /*new_bit_count=*/32);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(100)));

  EXPECT_EQ(schedule.length(), 100);
  // Most stages should be empty.
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(x.node(), bitslice.node()));
  // The bitslice is the narrowest among the chain of operations so it should
  // precede the long chain of empty stages.
  for (int64_t i = 1; i < 99; ++i) {
    EXPECT_THAT(schedule.nodes_in_cycle(i), UnorderedElementsAre());
  }
  EXPECT_THAT(schedule.nodes_in_cycle(99),
              UnorderedElementsAre(zero_ext.node()));
}

TEST_F(PipelineScheduleTest, ClockPeriodMargin) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  // Chain of six negates.
  fb.Negate(fb.Negate(fb.Negate(fb.Negate(fb.Negate(fb.Negate(x))))));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(3)));
  EXPECT_EQ(schedule.length(), 2);

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func, TestDelayEstimator(),
            SchedulingOptions().clock_period_ps(3).clock_margin_percent(0)));
    EXPECT_EQ(schedule.length(), 2);
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func, TestDelayEstimator(),
            SchedulingOptions().clock_period_ps(3).clock_margin_percent(33)));
    EXPECT_EQ(schedule.length(), 3);
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func, TestDelayEstimator(),
            SchedulingOptions().clock_period_ps(3).clock_margin_percent(66)));
    EXPECT_EQ(schedule.length(), 6);
  }
  EXPECT_THAT(
      RunPipelineSchedule(
          func, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(3).clock_margin_percent(200))
          .status(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr(
              "Clock period non-positive (-3ps) after adjusting for margin. "
              "Original clock period: 3ps, clock margin: 200%")));
}

TEST_F(PipelineScheduleTest, PeriodRelaxation) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);

  // Fanout
  auto x1 = fb.Negate(x);
  auto x11 = fb.Negate(x1);
  auto x21 = fb.Negate(x1);
  auto x111 = fb.Negate(x11);
  auto x211 = fb.Negate(x11);
  auto x121 = fb.Negate(x21);
  auto x221 = fb.Negate(x21);

  // Fanin
  auto y11 = fb.Or(x111, x211);
  auto y21 = fb.Or(x121, x221);
  auto y1 = fb.Or(y11, y21);
  fb.Negate(y1);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(2)));
  EXPECT_EQ(schedule.length(), 2);
  int64_t reg_count_default = schedule.CountFinalInteriorPipelineRegisters();

  for (int64_t relax_percent : std::vector{50, 100}) {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func, TestDelayEstimator(),
            SchedulingOptions().pipeline_stages(2).period_relaxation_percent(
                relax_percent)));
    EXPECT_EQ(schedule.length(), 2);
    int64_t reg_count_relaxed = schedule.CountFinalInteriorPipelineRegisters();
    EXPECT_LT(reg_count_relaxed, reg_count_default);
  }
}

TEST_F(PipelineScheduleTest, SerializeAndDeserialize) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  fb.Negate(fb.Concat({(fb.Not(fb.Negate(x | y)) - z) * x, z + z}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(3)));

  ASSERT_TRUE(schedule.min_clock_period_ps().has_value());
  PipelineScheduleProto proto = schedule.ToProto(TestDelayEstimator());
  PackageScheduleProto package_schedule_proto;
  package_schedule_proto.mutable_schedules()->emplace(func->name(),
                                                      std::move(proto));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule clone,
      PipelineSchedule::FromProto(func, package_schedule_proto));
  for (const Node* node : func->nodes()) {
    EXPECT_EQ(schedule.cycle(node), clone.cycle(node));
  }
  ASSERT_TRUE(clone.min_clock_period_ps().has_value());
  EXPECT_EQ(*clone.min_clock_period_ps(), *schedule.min_clock_period_ps());
  EXPECT_EQ(clone.length(), schedule.length());
}

TEST_F(PipelineScheduleTest, NodeDelayInScheduleProto) {
  // Tests that node and path delays are serialized in the schedule proto
  // using trivial pipeline: 3 stages of 2 x 1-bit inverters.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(1));
  fb.Not(fb.Not(fb.Not(fb.Not(fb.Not(fb.Not(x))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(func, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(3)));

  PipelineScheduleProto proto = schedule.ToProto(TestDelayEstimator());
  for (const auto& stage : proto.stages()) {
    int64_t path_delay = 0;
    for (const TimedNodeProto& node : stage.timed_nodes()) {
      path_delay += node.node_delay_ps();
      EXPECT_EQ(node.path_delay_ps(), path_delay);
    }
  }
}

TEST_F(PipelineScheduleTest, ProcSchedule) {
  Package p("p");
  Type* u16 = p.GetBitsType(16);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));
  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  BValue st = pb.StateElement("st", Value(UBits(42, 16)));
  BValue rcv = pb.Receive(in_ch);
  BValue out = pb.Negate(pb.Not(pb.Negate(rcv)));
  BValue send = pb.Send(out_ch, out);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({st}));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);

  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 2);
}

TEST_F(PipelineScheduleTest, StatelessProcSchedule) {
  Package p("p");
  Type* u16 = p.GetBitsType(16);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));
  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  BValue rcv = pb.Receive(in_ch);
  BValue out = pb.Negate(pb.Not(pb.Negate(rcv)));
  BValue send = pb.Send(out_ch, out);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);

  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 2);
}

TEST_F(PipelineScheduleTest, MultistateProcSchedule) {
  Package p("p");
  Type* u16 = p.GetBitsType(16);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));
  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  BValue st0 = pb.StateElement("st0", Value(UBits(0, 16)));
  BValue st1 = pb.StateElement("st1", Value(UBits(0, 16)));
  BValue rcv = pb.Receive(in_ch);
  BValue out = pb.Negate(pb.Not(pb.Negate(rcv)));
  BValue send = pb.Send(out_ch, out);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build({pb.Add(st0, rcv), pb.Subtract(st1, rcv)}));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);

  EXPECT_EQ(schedule.cycle(st0.node()), 0);
  EXPECT_EQ(schedule.cycle(st1.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 2);
}

TEST_F(PipelineScheduleTest, ProcWithConditionalReceive) {
  // Test a proc with a conditional receive.
  Package p("p");
  Type* u16 = p.GetBitsType(16);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));
  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  BValue st = pb.StateElement("st", Value(UBits(42, 16)));
  BValue cond = pb.Literal(UBits(0, 1));
  BValue rcv = pb.ReceiveIf(in_ch, cond);
  BValue out = pb.Negate(pb.Not(pb.Negate(rcv)));
  BValue send = pb.Send(out_ch, out);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({st}));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);

  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(cond.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 2);
}

TEST_F(PipelineScheduleTest, ProcWithConditionalReceiveLongCondition) {
  // Test a proc with a conditional receive. The receive condition takes too
  // long to compute in the same cycle as the receive so the receive is pushed
  // to stage 1.
  Package p("p");
  Type* u16 = p.GetBitsType(16);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));
  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  BValue st = pb.StateElement("st", Value(UBits(42, 16)));
  BValue cond = pb.Not(pb.Not(pb.Literal(UBits(0, 1))));
  BValue rcv = pb.ReceiveIf(in_ch, cond, SourceInfo(), "rcv");
  BValue out = pb.Negate(pb.Not(pb.Negate(rcv)));
  pb.Send(out_ch, out);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({st}));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 5);

  EXPECT_EQ(schedule.cycle(cond.node()), 1);
  EXPECT_EQ(schedule.cycle(rcv.node()), 2);
}

TEST_F(PipelineScheduleTest, ReceiveFollowedBySend) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());

  BValue rcv = pb.Receive(ch_in, tkn);
  BValue send = pb.Send(ch_out, /*token=*/pb.TupleIndex(rcv, 0),
                        /*data=*/pb.TupleIndex(rcv, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().pipeline_stages(5)));
  EXPECT_EQ(schedule.length(), 5);
  EXPECT_GE(schedule.cycle(send.node()), schedule.cycle(rcv.node()));
}

TEST_F(PipelineScheduleTest, SendFollowedByReceiveCannotBeInSameCycle) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());

  BValue send = pb.Send(ch_out, tkn, pb.Literal(Bits(32)));
  BValue rcv = pb.Receive(ch_in, /*token=*/send);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  ASSERT_THAT(RunPipelineSchedule(proc, *delay_estimator,
                                  SchedulingOptions().pipeline_stages(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 1);
}

TEST_F(PipelineScheduleTest, SendFollowedByReceiveIfCannotBeInSameCycle) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());

  BValue send = pb.Send(ch_out, tkn, pb.Literal(Bits(32)));
  BValue rcv_if = pb.ReceiveIf(ch_in, /*token=*/send, pb.Literal(UBits(0, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  ASSERT_THAT(RunPipelineSchedule(proc, *delay_estimator,
                                  SchedulingOptions().pipeline_stages(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv_if.node()), 1);
}

TEST_F(PipelineScheduleTest,
       SendFollowedByNonblockingReceiveCannotBeInSameCycle) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());

  BValue send = pb.Send(ch_out, tkn, pb.Literal(Bits(32)));
  BValue rcv = pb.ReceiveNonBlocking(ch_in, /*token=*/send);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  ASSERT_THAT(RunPipelineSchedule(proc, *delay_estimator,
                                  SchedulingOptions().pipeline_stages(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 1);
}

TEST_F(PipelineScheduleTest,
       SendFollowedByNonblockingReceiveIfCannotBeInSameCycle) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());

  BValue send = pb.Send(ch_out, tkn, pb.Literal(Bits(32)));
  BValue rcv_if =
      pb.ReceiveIfNonBlocking(ch_in, /*token=*/send, pb.Literal(UBits(0, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  ASSERT_THAT(RunPipelineSchedule(proc, *delay_estimator,
                                  SchedulingOptions().pipeline_stages(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv_if.node()), 1);
}

TEST_F(PipelineScheduleTest,
       SendFollowedIndirectlyByReceiveCannotBeInSameCycle) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());

  BValue rcv0 = pb.Receive(ch_in, tkn);
  BValue rcv1 = pb.Receive(ch_in, tkn);
  BValue send = pb.Send(ch_out, pb.TupleIndex(rcv1, 0), pb.Literal(Bits(32)));
  BValue joined_token = pb.AfterAll({pb.TupleIndex(rcv0, 0), send});
  BValue rcv = pb.Receive(ch_in, joined_token);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  ASSERT_THAT(RunPipelineSchedule(proc, *delay_estimator,
                                  SchedulingOptions().pipeline_stages(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 1);
}

TEST_F(PipelineScheduleTest, SendFollowedByDelayedReceive) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());

  BValue send = pb.Send(ch_out, tkn, pb.Literal(Bits(32)));
  BValue delay = pb.MinDelay(send, /*delay=*/3);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().pipeline_stages(5)));
  EXPECT_EQ(schedule.length(), 5);
  EXPECT_EQ(schedule.cycle(rcv.node()) - schedule.cycle(send.node()), 3);
}

TEST_F(PipelineScheduleTest, SendFollowedByDelayedReceiveWithState) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));
  pb.proc()->SetInitiationInterval(2);

  BValue send = pb.Send(ch_out, tkn, state);
  BValue delay = pb.MinDelay(send, /*delay=*/1);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.TupleIndex(rcv, 1)}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().pipeline_stages(5)));
  EXPECT_EQ(schedule.length(), 5);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 1);
}

TEST_F(PipelineScheduleTest, SuggestIncreasedPipelineLengthWhenNeeded) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  BValue send = pb.Send(ch_out, tkn, state);
  BValue delay = pb.MinDelay(send, /*delay=*/2);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.TupleIndex(rcv, 1)}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  EXPECT_THAT(
      RunPipelineSchedule(
          proc, *delay_estimator,
          SchedulingOptions().pipeline_stages(1).worst_case_throughput(3)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("--pipeline_stages=3"),
                     Not(HasSubstr("--worst_case_throughput")))));
}

TEST_F(PipelineScheduleTest, SuggestReducedThroughputWhenFullThroughputFails) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  BValue send = pb.Send(ch_out, tkn, state);
  BValue delay = pb.MinDelay(send, /*delay=*/2);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.TupleIndex(rcv, 1)}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  EXPECT_THAT(RunPipelineSchedule(proc, *delay_estimator,
                                  SchedulingOptions().pipeline_stages(5)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--worst_case_throughput=3"),
                             Not(HasSubstr("--pipeline_stages")))));
}

TEST_F(PipelineScheduleTest, UnboundedThroughputWorks) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  BValue send = pb.Send(ch_out, tkn, state);
  BValue delay = pb.MinDelay(send, /*delay=*/2);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.TupleIndex(rcv, 1)}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          proc, *delay_estimator,
          SchedulingOptions().pipeline_stages(5).worst_case_throughput(0)));
  EXPECT_EQ(schedule.length(), 5);
  EXPECT_EQ(schedule.cycle(rcv.node()) - schedule.cycle(send.node()), 2);
}

TEST_F(PipelineScheduleTest, MinimizedThroughputWorksWithGivenPipelineLength) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  BValue send = pb.Send(ch_out, tkn, state);
  BValue delay = pb.MinDelay(send, /*delay=*/2);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.TupleIndex(rcv, 1)}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions()
                              .pipeline_stages(5)
                              .worst_case_throughput(0)
                              .minimize_worst_case_throughput(true)));
  EXPECT_EQ(schedule.length(), 5);
  EXPECT_EQ(schedule.cycle(rcv.node()) - schedule.cycle(send.node()), 2);
  EXPECT_EQ(proc->GetInitiationInterval().value_or(1), 4);
}

TEST_F(PipelineScheduleTest, MinimizedThroughputWorksWithGivenClockPeriod) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  BValue send = pb.Send(ch_out, tkn, state);
  BValue delay = pb.MinDelay(send, /*delay=*/2);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.TupleIndex(rcv, 1)}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions()
                              .clock_period_ps(2)
                              .worst_case_throughput(0)
                              .minimize_worst_case_throughput(true)));
  EXPECT_EQ(schedule.length(), 3);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 2);
  EXPECT_EQ(proc->GetInitiationInterval().value_or(1), 3);
}

TEST_F(PipelineScheduleTest,
       SuggestReducedThroughputWhenFullThroughputFailsWithClockGiven) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  BValue send = pb.Send(ch_out, tkn, state);
  BValue delay = pb.MinDelay(send, /*delay=*/2);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.TupleIndex(rcv, 1)}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  EXPECT_THAT(RunPipelineSchedule(proc, *delay_estimator,
                                  SchedulingOptions().clock_period_ps(1000)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--worst_case_throughput=3"),
                             Not(HasSubstr("--pipeline_stages")))));
}

TEST_F(PipelineScheduleTest,
       SuggestIncreasedPipelineLengthAndReducedThroughputWhenNeeded) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  BValue send = pb.Send(ch_out, tkn, state);
  BValue delay = pb.MinDelay(send, /*delay=*/2);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({pb.TupleIndex(rcv, 1)}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  EXPECT_THAT(RunPipelineSchedule(proc, *delay_estimator,
                                  SchedulingOptions().pipeline_stages(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--pipeline_stages=3"),
                             HasSubstr("--worst_case_throughput=3"))));
}

TEST_F(PipelineScheduleTest, SuggestIncreasedPipelineLengthAndIndividualSlack) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  BValue send = pb.Send(ch_out, tkn, state);
  BValue delay = pb.MinDelay(send, /*delay=*/2);
  BValue rcv = pb.Receive(ch_in, /*token=*/delay);
  pb.Next(state, pb.TupleIndex(rcv, 1), /*pred=*/std::nullopt, SourceInfo(),
          "next_state");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  EXPECT_THAT(
      RunPipelineSchedule(
          proc, *delay_estimator,
          SchedulingOptions().pipeline_stages(1).failure_behavior(
              SchedulingFailureBehavior{
                  .explain_infeasibility = true,
                  .infeasible_per_state_backedge_slack_pool = 2.0})),
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("--pipeline_stages=3"),
                     HasSubstr("looking at paths between state and next_state "
                               "(needs 2 additional slack)"))));
}

TEST_F(
    PipelineScheduleTest,
    SuggestIncreasedPipelineLengthWorstCaseThroughtputAndIndividualSlackPool2) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out0,
      package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out1,
      package.CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state0 = pb.StateElement("state0", Value(Bits(32)));
  BValue state1 = pb.StateElement("state1", Value(Bits(32)));

  BValue send0 = pb.Send(ch_out0, tkn, state0);
  BValue send1 = pb.Send(ch_out1, tkn, state1);
  BValue delay0 = pb.MinDelay(send0, /*delay=*/2);
  BValue delay1 = pb.MinDelay(send1, /*delay=*/1);
  BValue rcv = pb.Receive(ch_in, /*token=*/pb.AfterAll({delay0, delay1}));
  BValue rcv_data = pb.TupleIndex(rcv, 1, SourceInfo(), "rcv_data");
  pb.Next(state0, rcv_data, /*pred=*/std::nullopt, SourceInfo(), "next_state0");
  pb.Next(state1, pb.Add(rcv_data, state1), /*pred=*/std::nullopt, SourceInfo(),
          "next_state1");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  EXPECT_THAT(
      RunPipelineSchedule(
          proc, *delay_estimator,
          SchedulingOptions().pipeline_stages(1).failure_behavior(
              SchedulingFailureBehavior{
                  .explain_infeasibility = true,
                  .infeasible_per_state_backedge_slack_pool = 2.0})),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstr("--pipeline_stages=3"),
                HasSubstr("--worst_case_throughput=2"),
                HasSubstr("looking at paths between state0 and next_state0 "
                          "(needs 1 additional slack)"))));
}

TEST_F(
    PipelineScheduleTest,
    SuggestIncreasedPipelineLengthWorstCaseThroughtputAndIndividualSlackPool3) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out0,
      package.CreateStreamingChannel("out0", ChannelOps::kSendOnly, u32));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out1,
      package.CreateStreamingChannel("out1", ChannelOps::kSendOnly, u32));

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out2,
      package.CreateStreamingChannel("out2", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state0 = pb.StateElement("state0", Value(Bits(32)));
  BValue state1 = pb.StateElement("state1", Value(Bits(32)));
  BValue state2 = pb.StateElement("state2", Value(Bits(32)));

  BValue send0 = pb.Send(ch_out0, tkn, state0);
  BValue send1 = pb.Send(ch_out1, tkn, state1);
  BValue send2 = pb.Send(ch_out2, tkn, state2);
  BValue delay0 = pb.MinDelay(send0, /*delay=*/3);
  BValue delay1 = pb.MinDelay(send1, /*delay=*/2);
  BValue delay2 = pb.MinDelay(send2, /*delay=*/2);
  BValue rcv =
      pb.Receive(ch_in, /*token=*/pb.AfterAll({delay0, delay1, delay2}));
  BValue rcv_data = pb.TupleIndex(rcv, 1, SourceInfo(), "rcv_data");
  pb.Next(state0, rcv_data, /*pred=*/std::nullopt, SourceInfo(), "next_state0");
  pb.Next(state1, pb.Add(rcv_data, state1), /*pred=*/std::nullopt, SourceInfo(),
          "next_state1");
  pb.Next(state2, pb.Add(rcv_data, state2), /*pred=*/std::nullopt, SourceInfo(),
          "next_state2");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  EXPECT_THAT(
      RunPipelineSchedule(
          proc, *delay_estimator,
          SchedulingOptions().pipeline_stages(1).failure_behavior(
              SchedulingFailureBehavior{
                  .explain_infeasibility = true,
                  .infeasible_per_state_backedge_slack_pool =
                      // Add epsilon to confirm that small errors in the pool
                      // don't cause us to incorrectly prefer per-node slack
                      // over shared slack.
                  3.0 + std::numeric_limits<double>::epsilon()})),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          AllOf(HasSubstr("--pipeline_stages=4"),
                HasSubstr("--worst_case_throughput=3"),
                HasSubstr("looking at paths between state0 and next_state0 "
                          "(needs 1 additional slack)"))));
}

TEST_F(PipelineScheduleTest, SuggestIncreasedClockPeriodWhenNecessary) {
  Package package = Package(TestName());

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  pb.Send(ch_out, tkn, state);
  BValue add2 = pb.Add(state, pb.Literal(UBits(2, 32)));
  BValue mul3 = pb.UMul(add2, pb.Literal(UBits(3, 32)));
  BValue add1 = pb.Add(mul3, pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({add1}));

  // Each operation takes 500ps, so (with no pipeline depth restrictions), 500ps
  // is the fastest clock we can support.
  EXPECT_THAT(RunPipelineSchedule(proc, TestDelayEstimator(500),
                                  SchedulingOptions().clock_period_ps(100)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--clock_period_ps=500")));

  // Each operation takes 500ps, but we have a chain of three operations; in two
  // stages, the best we can do is a 1000ps clock.
  EXPECT_THAT(RunPipelineSchedule(
                  proc, TestDelayEstimator(500),
                  SchedulingOptions().clock_period_ps(100).pipeline_stages(2)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--clock_period_ps=1000")));

  // Each operation takes 500ps, and our schedule fits nicely into 3 stages; we
  // can get down to a 500ps clock at 3 or more pipeline stages.
  EXPECT_THAT(RunPipelineSchedule(
                  proc, TestDelayEstimator(500),
                  SchedulingOptions().clock_period_ps(100).pipeline_stages(3)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--clock_period_ps=500")));
  EXPECT_THAT(RunPipelineSchedule(
                  proc, TestDelayEstimator(500),
                  SchedulingOptions().clock_period_ps(100).pipeline_stages(20)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--clock_period_ps=500")));

  // But... if told not to search for the smallest possible clock period, the
  // best we can do is signal that a longer clock period might help.
  EXPECT_THAT(RunPipelineSchedule(proc, TestDelayEstimator(500),
                                  SchedulingOptions()
                                      .clock_period_ps(100)
                                      .pipeline_stages(4)
                                      .minimize_clock_on_failure(false)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Try increasing `--clock_period_ps`")));
}

TEST_F(PipelineScheduleTest, OptimizeForDynamicThroughput) {
  Package package = Package(TestName());
  Type* u1 = package.GetBitsType(1);
  Type* u2 = package.GetBitsType(2);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_in,
      package.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u2));

  ProcBuilder pb(TestName(), &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(1)));
  BValue recv = pb.Receive(ch_in, tkn);
  BValue recv_tkn = pb.TupleIndex(recv, 0);
  BValue change = pb.TupleIndex(recv, 1);
  BValue extended_value = pb.SignExtend(pb.Xor(state, change), 2);
  BValue next_state = pb.OrReduce(extended_value);
  pb.Next(state, next_state);
  pb.Send(ch_out, pb.MinDelay(recv_tkn, 1), extended_value);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));

  // If we don't optimize for dynamic throughput, we end up with throughput = 2,
  // since that lets the scheduler reduce the area very slightly.
  //
  // NOTE: This is a VERY contrived example, but there are real examples of this
  // happening in the wild.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          proc, *delay_estimator,
          SchedulingOptions().clock_period_ps(10).worst_case_throughput(2)));
  EXPECT_EQ(schedule.cycle(next_state.node()) - schedule.cycle(state.node()),
            1);

  // On the other hand, if we do optimize for dynamic throughput, we end up with
  // full throughput.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule2,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions()
                              .clock_period_ps(10)
                              .worst_case_throughput(2)
                              .dynamic_throughput_objective_weight(1024.0)));
  EXPECT_EQ(schedule2.cycle(next_state.node()) - schedule2.cycle(state.node()),
            0);
}

// Proc next state does not depend on param; next state can now be scheduled in
// an earlier stage than the param node's use, so (all else being equal), the
// scheduler prefers to schedule the param node ASAP, in the same stage as the
// next-state node. The schedule is forced to multiple stages by having two
// receive nodes where the second receive node depends on the first node, and
// the first receive node produces the next state node and the param is used by
// the second receive node.
TEST_F(PipelineScheduleTest, ProcParamScheduledEarlyWithNextState) {
  Package p("p");
  Type* u1 = p.GetBitsType(1);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in0,
      p.CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in1,
      p.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u1));
  ProcBuilder pb(TestName(), &p);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(UBits(1, 1)));
  BValue nb_rcv = pb.ReceiveNonBlocking(in0, tkn);
  BValue nb_rcv_tkn = pb.TupleIndex(nb_rcv, 0);
  BValue nb_rcv_data = pb.TupleIndex(nb_rcv, 1);
  BValue nb_rcv_valid = pb.TupleIndex(nb_rcv, 2);
  BValue after_all = pb.AfterAll({tkn, nb_rcv_tkn});
  // The statement explicitly shows the use of the state node after the next
  // state node.
  BValue use_state = pb.And(nb_rcv_data, state);
  pb.ReceiveIf(in1, after_all, use_state);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({nb_rcv_valid}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().pipeline_stages(3)));
  EXPECT_EQ(schedule.length(), 3);
  // The state's param node should be scheduled ASAP (i.e., the next-state
  // node's stage), while still leaving its user in the later stage.
  EXPECT_EQ(schedule.cycle(state.node()), schedule.cycle(nb_rcv_valid.node()));
  EXPECT_LT(schedule.cycle(state.node()), schedule.cycle(use_state.node()));
}

// Proc next state does not depend on param. Force schedule of a param node's
// user in a later stage than the next state is computed. The schedule can be
// forced by having two receive nodes where the second receive node depends on
// the first node, and the first receive node produces the next state node and
// the param is used by the second receive node. We make the scheduler prefer to
// schedule the next-state computation earlier by making it narrower than the
// param value, then widening later.
TEST_F(PipelineScheduleTest, ProcParamScheduledAfterNextState) {
  Package p("p");
  Type* u1 = p.GetBitsType(1);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in0,
      p.CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in1,
      p.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u1));
  ProcBuilder pb(TestName(), &p);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(UBits(1, 8)));
  BValue nb_rcv = pb.ReceiveNonBlocking(in0, tkn);
  BValue nb_rcv_tkn = pb.TupleIndex(nb_rcv, 0);
  BValue nb_rcv_data = pb.TupleIndex(nb_rcv, 1);
  BValue nb_rcv_valid = pb.TupleIndex(nb_rcv, 2);
  BValue after_all = pb.AfterAll({tkn, nb_rcv_tkn});
  // The statement explicitly shows the use of the state node after the next
  // state's information is available.
  BValue extended_nb_rcv_data = pb.ZeroExtend(nb_rcv_data, 8);
  BValue use_state = pb.UGe(extended_nb_rcv_data, state);
  pb.ReceiveIf(in1, after_all, use_state);
  BValue extended_nb_rcv_valid = pb.ZeroExtend(nb_rcv_valid, 8);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({extended_nb_rcv_valid}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().pipeline_stages(3)));
  EXPECT_EQ(schedule.length(), 3);
  EXPECT_GT(schedule.cycle(state.node()), schedule.cycle(nb_rcv_valid.node()));
}

// If two param nodes are mutually dependent, they (and their next state nodes)
// all need to be scheduled in the same stage.
TEST_F(PipelineScheduleTest, ProcParamsScheduledInSameStage) {
  Package p("p");
  Type* u1 = p.GetBitsType(1);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in0,
      p.CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in1,
      p.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u1));
  ProcBuilder pb(TestName(), &p);
  BValue tkn = pb.Literal(Value::Token());
  BValue a = pb.StateElement("a", Value(UBits(0, 1)));
  BValue b = pb.StateElement("b", Value(UBits(1, 1)));
  BValue nb_rcv = pb.ReceiveNonBlocking(in0, tkn);
  BValue nb_rcv_tkn = pb.TupleIndex(nb_rcv, 0);
  BValue nb_rcv_data = pb.TupleIndex(nb_rcv, 1);
  BValue nb_rcv_valid = pb.TupleIndex(nb_rcv, 2);
  BValue after_all = pb.AfterAll({tkn, nb_rcv_tkn});
  BValue use_state = pb.And(nb_rcv_data, a);
  pb.ReceiveIf(in1, after_all, use_state);
  BValue next_a = pb.Xor(b, nb_rcv_valid);
  BValue next_b = pb.Xor(a, nb_rcv_valid);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({next_a, next_b}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          SchedulingOptions().pipeline_stages(3)));
  EXPECT_EQ(schedule.length(), 3);
  EXPECT_EQ(schedule.cycle(a.node()), schedule.cycle(b.node()));
  EXPECT_EQ(schedule.cycle(a.node()), schedule.cycle(next_a.node()));
  EXPECT_EQ(schedule.cycle(a.node()), schedule.cycle(next_b.node()));
}

TEST_F(PipelineScheduleTest, FunctionScheduleWithInputAndOutputDelay) {
  Package p("p");

  Type* u16 = p.GetBitsType(16);

  FunctionBuilder fb("f", &p);

  BValue x = fb.Param("x", u16);
  BValue y = fb.Param("y", u16);
  BValue prod = fb.UMul(x, y);
  BValue negate = fb.Negate(prod);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(negate));

  // No additional input/output delay, we get [{x, y, prod, negate}]
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(f, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(2)));
  ASSERT_EQ(schedule.length(), 1);
  EXPECT_THAT(
      schedule.nodes_in_cycle(0),
      UnorderedElementsAre(x.node(), y.node(), prod.node(), negate.node()));

  // Additional input delay bumps prod to stage 2, we get
  // [{x,y}, {prod, negate}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule,
      RunPipelineSchedule(
          f, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(2).additional_input_delay_ps(2)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(x.node(), y.node()));
  EXPECT_THAT(schedule.nodes_in_cycle(1),
              UnorderedElementsAre(prod.node(), negate.node()));

  // Additional output delay bumps negate to stage 3, we get
  // [{x,y}, {prod}, {negate}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, RunPipelineSchedule(f, TestDelayEstimator(),
                                    SchedulingOptions()
                                        .clock_period_ps(2)
                                        .additional_input_delay_ps(2)
                                        .additional_output_delay_ps(1)));
  ASSERT_EQ(schedule.length(), 3);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(x.node(), y.node()));
  EXPECT_THAT(schedule.nodes_in_cycle(1), UnorderedElementsAre(prod.node()));
  EXPECT_THAT(schedule.nodes_in_cycle(2), UnorderedElementsAre(negate.node()));
}

TEST_F(PipelineScheduleTest, ProcScheduleWithInputDelay) {
  Package p("p");

  Type* u16 = p.GetBitsType(16);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));

  TokenlessProcBuilder pb("the_proc", "tkn", &p);

  BValue rcv = pb.Receive(in_ch);
  BValue out = pb.Negate(pb.Not(pb.Negate(pb.Not(pb.Negate(rcv)))));
  BValue send = pb.Send(out_ch, out);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(4)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 1);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              Contains(OperationDelayInPs(IsOkAndHolds(Gt(0)))));

  // Input delay of 1 is not large enough to bump all non-zero-latency nodes to
  // later stages.
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule,
      RunPipelineSchedule(
          proc, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(4).additional_input_delay_ps(1)));
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              Contains(OperationDelayInPs(IsOkAndHolds(Gt(0)))));

  // With a large enough input delay the only things that will
  // be scheduled in the first cycle is the receive, token, and state, as
  // well as potentially some zero-latency nodes.
  //
  // tkn: token = param(tkn, id=1)
  // receive.3: (token, bits[16]) = receive(tkn, channel_id=0, id=3)
  // st: () = param(st, id=2)
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule,
      RunPipelineSchedule(
          proc, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(4).additional_input_delay_ps(4)));
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              Each(OperationDelayInPs(IsOkAndHolds(0))));
}

TEST_F(PipelineScheduleTest, ProcScheduleWithInputAndOutputDelay) {
  Package p("p");

  Type* u16 = p.GetBitsType(16);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));

  TokenlessProcBuilder pb("the_proc", "tkn", &p);

  BValue rcv = pb.Receive(in_ch);
  BValue negate = pb.Negate(rcv);
  BValue send = pb.Send(out_ch, negate);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  // No input delay, we get [{rcv, negate, send}]
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(2)));
  ASSERT_EQ(schedule.length(), 1);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv.node(), negate.node(), send.node()}));

  // Input delay bumps send to stage 1, we get [{rcv, negate}, {send}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule,
      RunPipelineSchedule(
          proc, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(2).additional_input_delay_ps(1)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv.node(), negate.node()}));
  EXPECT_THAT(schedule.nodes_in_cycle(1), IsSupersetOf({send.node()}));

  // Output delay also bumps send to stage 1, we get [{rcv, negate}, {send}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule,
      RunPipelineSchedule(
          proc, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(2).additional_output_delay_ps(
              1)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv.node(), negate.node()}));
  EXPECT_THAT(schedule.nodes_in_cycle(1), IsSupersetOf({send.node()}));

  // Specifying both input and output delay doesn't change anything.
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, RunPipelineSchedule(proc, TestDelayEstimator(),
                                    SchedulingOptions()
                                        .clock_period_ps(2)
                                        .additional_input_delay_ps(1)
                                        .additional_output_delay_ps(1)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv.node(), negate.node()}));
  EXPECT_THAT(schedule.nodes_in_cycle(1), IsSupersetOf({send.node()}));
}

TEST_F(PipelineScheduleTest, ProcScheduleWithChannelSpecificDelay) {
  Package p("p");

  Type* u16 = p.GetBitsType(16);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in1_ch,
      p.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in2_ch,
      p.CreateStreamingChannel("in2", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));

  TokenlessProcBuilder pb("the_proc", "tkn", &p);

  BValue rcv1 = pb.Receive(in1_ch);
  BValue neg_rcv1 = pb.Negate(rcv1);
  BValue rcv2 = pb.Receive(in2_ch);
  BValue neg_rcv2 = pb.Negate(rcv2);
  BValue sum = pb.Add(neg_rcv1, neg_rcv2);
  BValue send = pb.Send(out_ch, sum);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  // No input delay, we get [{rcv1, neg_rcv1, rcv2, neg_rcv2, sum, send}]
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(2)));
  ASSERT_EQ(schedule.length(), 1);
  EXPECT_THAT(
      schedule.nodes_in_cycle(0),
      IsSupersetOf({rcv1.node(), rcv2.node(), sum.node(), send.node()}));

  // Small ch2 delay bumps just the sum to stage 1, we get:
  // [{rcv1, neg_rcv1, rcv2, neg_rcv2}, {sum, send}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions()
                              .clock_period_ps(2)
                              .add_additional_channel_delay_ps("in2", 1)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv1.node(), neg_rcv1.node(), rcv2.node(),
                            neg_rcv2.node()}));
  EXPECT_THAT(schedule.nodes_in_cycle(1),
              IsSupersetOf({sum.node(), send.node()}));

  // Larger ch2 delay bumps neg_rcv2 to stage 1, we get:
  // [{rcv1, neg_rcv1, rcv2}, {neg_rcv2, sum, send}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(),
                          SchedulingOptions()
                              .clock_period_ps(2)
                              .add_additional_channel_delay_ps("in2", 2)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv1.node(), neg_rcv1.node(), rcv2.node()}));
  EXPECT_THAT(schedule.nodes_in_cycle(1),
              IsSupersetOf({neg_rcv2.node(), sum.node(), send.node()}));
}

TEST_F(PipelineScheduleTest, ProcScheduleWithConstraints) {
  Package p("p");
  Type* u16 = p.GetBitsType(16);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p.CreateStreamingChannel("in", ChannelOps::kReceiveOnly, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));
  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  BValue st = pb.StateElement("st", Value(UBits(42, 16)));
  BValue rcv = pb.Receive(in_ch);
  BValue out = pb.Negate(pb.Not(pb.Negate(rcv)));
  BValue send = pb.Send(out_ch, out);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({st}));

  for (int64_t i = 3; i <= 9; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            proc, TestDelayEstimator(),
            SchedulingOptions().pipeline_stages(10).add_constraint(
                IOConstraint("in", IODirection::kReceive, "out",
                             IODirection::kSend, i, i))));

    EXPECT_EQ(schedule.length(), 10);
    EXPECT_EQ(schedule.cycle(send.node()) - schedule.cycle(rcv.node()), i);
  }
}

TEST_F(PipelineScheduleTest, RandomSchedule) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  for (int64_t i = 0; i < 20; ++i) {
    x = fb.Negate(x);
  }

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  for (int32_t i = 0; i < 1000; ++i) {
    // Running the scheduler will call `VerifyTiming`.
    XLS_ASSERT_OK(
        RunPipelineSchedule(func, TestDelayEstimator(),
                            SchedulingOptions(SchedulingStrategy::RANDOM)
                                .seed(i)
                                .pipeline_stages(50))
            .status());
  }
}

TEST_F(PipelineScheduleTest, SingleStageSchedule) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  for (int64_t i = 0; i < 20; ++i) {
    x = fb.Negate(x);
  }

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  // This simply puts all of func's nodes into a single pipeline stage.
  XLS_ASSERT_OK_AND_ASSIGN(absl::StatusOr<PipelineSchedule> schedule,
                           PipelineSchedule::SingleStage(func));

  EXPECT_EQ(schedule.value().length(), 1);

  // 20 negates plus the input param
  EXPECT_EQ(schedule.value().nodes_in_cycle(0).size(), 21);
}

TEST_F(PipelineScheduleTest, LoopbackChannelWithConstraint) {
  auto p = CreatePackage();
  Type* u16 = p->GetBitsType(16);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * loopback_ch,
      p->CreateStreamingChannel("loopback", ChannelOps::kSendReceive, u16));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p->CreateStreamingChannel("out", ChannelOps::kSendOnly, u16));
  TokenlessProcBuilder pb("the_proc", "tkn", p.get());
  BValue st = pb.StateElement("st", Value(UBits(42, 16)));
  BValue rcv = pb.Receive(loopback_ch);
  BValue out = pb.Negate(pb.Not(pb.Negate(rcv)));
  pb.Send(out_ch, out);
  BValue loopback_send = pb.Send(loopback_ch, out);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({st}));

  for (int64_t i = 3; i <= 9; ++i) {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            proc, TestDelayEstimator(),
            SchedulingOptions().pipeline_stages(10).add_constraint(
                IOConstraint("loopback", IODirection::kReceive, "loopback",
                             IODirection::kSend, i, i))));

    EXPECT_EQ(schedule.length(), 10);
    EXPECT_EQ(schedule.cycle(loopback_send.node()) - schedule.cycle(rcv.node()),
              i);
  }
}

TEST_F(PipelineScheduleTest, PackageScheduleProtoSerializeAndDeserialize) {
  auto p = CreatePackage();
  auto make_test_fn = [](Package* p, std::string_view name) {
    FunctionBuilder fb(name, p);
    Type* u32 = p->GetBitsType(32);
    auto x = fb.Param("x", u32);
    // Perform several additions to populate the schedule with some nodes.
    return fb.BuildWithReturnValue(x + x + x + x + x + x);
  };
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * func0, make_test_fn(p.get(), absl::StrCat(TestName(), "0")));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * func1, make_test_fn(p.get(), absl::StrCat(TestName(), "1")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule0,
      RunPipelineSchedule(func0, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(3)));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule1,
      RunPipelineSchedule(func1, TestDelayEstimator(),
                          SchedulingOptions().pipeline_stages(3)));

  PackageSchedule package_schedule(p.get(),
                                   {{func0, schedule0}, {func1, schedule1}});

  PackageScheduleProto proto = package_schedule.ToProto(TestDelayEstimator());
  XLS_ASSERT_OK_AND_ASSIGN(PackageSchedule clone,
                           PackageSchedule::FromProto(p.get(), proto));
  ASSERT_THAT(package_schedule.GetSchedules(),
              UnorderedPointwise(KeyEqElement(), p->GetFunctionBases()));
  ASSERT_THAT(clone.GetSchedules(),
              UnorderedPointwise(KeyEqElement(), p->GetFunctionBases()));
  ASSERT_THAT(p->GetFunctionBases(),
              Each(AllNodesScheduled(package_schedule.GetSchedules())));
  ASSERT_THAT(p->GetFunctionBases(),
              Each(AllNodesScheduled(clone.GetSchedules())));
  EXPECT_THAT(
      p->GetFunctionBases(),
      Each(CyclesMatch(package_schedule.GetSchedules(), clone.GetSchedules())));
}

TEST_F(PipelineScheduleTest, SerializeAndDeserializeWithSynchronousSchedule) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb1("proc1", "tkn", p.get());
  auto literal1 = pb1.Literal(UBits(1, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, pb1.Build());

  TokenlessProcBuilder pb2("proc2", "tkn", p.get());
  auto literal2 = pb2.Literal(UBits(2, 32));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc2, pb2.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule1,
      RunPipelineSchedule(proc1, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule2,
      RunPipelineSchedule(proc2, TestDelayEstimator(),
                          SchedulingOptions().clock_period_ps(1)));

  absl::flat_hash_map<FunctionBase*, int64_t> synchronous_offsets(
      {{proc1, 0}, {proc2, 42}});
  PackageSchedule package_schedule(
      p.get(), {{proc1, schedule1}, {proc2, schedule2}}, synchronous_offsets);

  EXPECT_TRUE(package_schedule.IsSynchronousSchedule());
  EXPECT_EQ(package_schedule.GetSynchronousCycle(literal1.node()), 0);
  EXPECT_EQ(package_schedule.GetSynchronousCycle(literal2.node()), 42);

  XLS_ASSERT_OK_AND_ASSIGN(
      PackageSchedule clone,
      PackageSchedule::FromProto(
          p.get(), package_schedule.ToProto(TestDelayEstimator())));

  EXPECT_TRUE(clone.IsSynchronousSchedule());
  EXPECT_EQ(clone.GetSynchronousCycle(literal1.node()), 0);
  EXPECT_EQ(clone.GetSynchronousCycle(literal2.node()), 42);
}

}  // namespace
}  // namespace xls
