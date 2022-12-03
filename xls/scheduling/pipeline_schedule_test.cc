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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/delay_model/delay_estimator.h"
#include "xls/delay_model/delay_estimators.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/op.h"
#include "xls/ir/type.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;
using xls::status_testing::StatusIs;

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
      PipelineSchedule::Run(f, TestDelayEstimator(), options));
  EXPECT_EQ(schedule.length(), 1);
  EXPECT_THAT(schedule.nodes_in_cycle(0), UnorderedElementsAre(m::Param()));

  FunctionBuilder fb_2("other_fn", p.get());
  fb_2.Literal(Value(UBits(16, 16)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f_2, fb_2.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, PipelineSchedule::Run(f_2, TestDelayEstimator(), options));
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
      PipelineSchedule::Run(
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
      PipelineSchedule::Run(f, TestDelayEstimator(),
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
      PipelineSchedule::Run(f, TestDelayEstimator(),
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
  fb.Negate(ret_value);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(ret_value));

  ASSERT_THAT(
      PipelineSchedule::Run(f, TestDelayEstimator(),
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
      PipelineSchedule::Run(
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
      PipelineSchedule::Run(
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
      PipelineSchedule::Run(f, TestDelayEstimator(),
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
      PipelineSchedule::Run(
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
      PipelineSchedule::Run(func, TestDelayEstimator(),
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
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(5)));

  EXPECT_EQ(schedule.length(), 1);
  XLS_EXPECT_OK(
      schedule.VerifyTiming(/*clock_period_ps=*/5, TestDelayEstimator()));
  EXPECT_THAT(
      schedule.VerifyTiming(/*clock_period_ps=*/1, TestDelayEstimator()),
      status_testing::StatusIs(
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
      PipelineSchedule::Run(
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
      PipelineSchedule::Run(func, TestDelayEstimator(),
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
  auto zext = fb.ZeroExtend(bitslice, /*new_bit_count=*/32);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(func, TestDelayEstimator(),
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
  EXPECT_THAT(schedule.nodes_in_cycle(99), UnorderedElementsAre(zext.node()));
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
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().clock_period_ps(3)));
  EXPECT_EQ(schedule.length(), 2);

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(
            func, TestDelayEstimator(),
            SchedulingOptions().clock_period_ps(3).clock_margin_percent(0)));
    EXPECT_EQ(schedule.length(), 2);
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(
            func, TestDelayEstimator(),
            SchedulingOptions().clock_period_ps(3).clock_margin_percent(33)));
    EXPECT_EQ(schedule.length(), 3);
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(
            func, TestDelayEstimator(),
            SchedulingOptions().clock_period_ps(3).clock_margin_percent(66)));
    EXPECT_EQ(schedule.length(), 6);
  }
  EXPECT_THAT(
      PipelineSchedule::Run(
          func, TestDelayEstimator(),
          SchedulingOptions().clock_period_ps(3).clock_margin_percent(200))
          .status(),
      status_testing::StatusIs(
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
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(2)));
  EXPECT_EQ(schedule.length(), 2);
  int64_t reg_count_default = schedule.CountFinalInteriorPipelineRegisters();

  for (int64_t relax_percent : std::vector{50, 100}) {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        PipelineSchedule::Run(
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
      PipelineSchedule::Run(func, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(3)));

  PipelineScheduleProto proto = schedule.ToProto();
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule clone,
                           PipelineSchedule::FromProto(func, proto));
  for (const Node* node : func->nodes()) {
    EXPECT_EQ(schedule.cycle(node), clone.cycle(node));
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
      PipelineSchedule::Run(proc, TestDelayEstimator(),
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
      PipelineSchedule::Run(proc, TestDelayEstimator(),
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
      PipelineSchedule::Run(proc, TestDelayEstimator(),
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
      PipelineSchedule::Run(proc, TestDelayEstimator(),
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
      PipelineSchedule::Run(proc, TestDelayEstimator(),
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

  ProcBuilder pb(TestName(), /*token_name=*/"tkn", &package);

  BValue rcv = pb.Receive(ch_in, pb.GetTokenParam());
  BValue send = pb.Send(ch_out, /*token=*/pb.TupleIndex(rcv, 0),
                        /*data=*/pb.TupleIndex(rcv, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(send, {}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(proc, *delay_estimator,
                            SchedulingOptions().pipeline_stages(5)));
  EXPECT_EQ(schedule.length(), 5);
  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 2);
}

// Proc next state does not depend on param. Force schedule of a param node in a
// later stage than the next state node. The schedule can be forced by having
// two receive nodes where the second receive node depends on the first node,
// and the first receive node produces the next state node and the param is used
// by the second receive node.
TEST_F(PipelineScheduleTest, ProcParamAndNextStateAreInSameCycle) {
  Package p("p");
  Type* u1 = p.GetBitsType(1);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in0,
      p.CreateStreamingChannel("in0", ChannelOps::kReceiveOnly, u1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in1,
      p.CreateStreamingChannel("in1", ChannelOps::kReceiveOnly, u1));
  ProcBuilder pb(TestName(), /*token_name=*/"tkn", &p);
  BValue state = pb.StateElement("state", Value(UBits(1, 1)));
  BValue nb_rcv = pb.ReceiveNonBlocking(in0, pb.GetTokenParam());
  BValue nb_rcv_tkn = pb.TupleIndex(nb_rcv, 0);
  BValue nb_rcv_data = pb.TupleIndex(nb_rcv, 1);
  BValue nb_rcv_valid = pb.TupleIndex(nb_rcv, 2);
  BValue after_all = pb.AfterAll({pb.GetTokenParam(), nb_rcv_tkn});
  // The statement explicitly shows the use of the state node after the next
  // state node.
  BValue use_state = pb.And(nb_rcv_data, state);
  BValue rcv = pb.ReceiveIf(in1, after_all, use_state);
  BValue rcv_tkn = pb.TupleIndex(rcv, 0);
  BValue after_all_final = pb.AfterAll({after_all, rcv_tkn});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(after_all_final, {nb_rcv_valid}));

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      PipelineSchedule::Run(proc, *delay_estimator,
                            SchedulingOptions().pipeline_stages(3)));
  EXPECT_EQ(schedule.length(), 3);
  EXPECT_EQ(schedule.cycle(state.node()), schedule.cycle(nb_rcv_valid.node()));
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
      PipelineSchedule::Run(proc, TestDelayEstimator(),
                            SchedulingOptions().pipeline_stages(2)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 1);

  for (int64_t input_delay : std::vector{2, 5, 10}) {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule_with_input_delay,
        PipelineSchedule::Run(
            proc, TestDelayEstimator(),
            SchedulingOptions().pipeline_stages(2).additional_input_delay_ps(
                input_delay)));

    absl::Span<Node* const> nodes_in_first_cycle =
        schedule_with_input_delay.nodes_in_cycle(0);

    if (input_delay >= 5) {
      // With a large enough input delay the only things that will
      // be scheduled in the first cycle is the receive, token, and state, as
      // well as potentially some zero-latency nodes.
      //
      // tkn: token = param(tkn, id=1)
      // receive.3: (token, bits[16]) = receive(tkn, channel_id=0, id=3)
      // st: () = param(st, id=2)
      EXPECT_GE(nodes_in_first_cycle.size(), 3);
      EXPECT_EQ(nodes_in_first_cycle.size(), 4);  // adjust if scheduler changes
      EXPECT_TRUE(
          std::all_of(nodes_in_first_cycle.begin(), nodes_in_first_cycle.end(),
                      [](Node* node) -> bool {
                        TestDelayEstimator estimator;
                        absl::StatusOr<int64_t> zero = 0;
                        return estimator.GetOperationDelayInPs(node) == zero;
                      }));
    }
  }
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
        PipelineSchedule::Run(
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
        PipelineSchedule::Run(func, TestDelayEstimator(),
                              SchedulingOptions(SchedulingStrategy::RANDOM)
                                  .seed(i)
                                  .pipeline_stages(50))
            .status());
  }
}

}  // namespace
}  // namespace xls
