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
#include <memory>
#include <optional>
#include <string>
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
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/estimators/delay_model/delay_estimators.h"
#include "xls/fdo/delay_manager.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/node.h"
#include "xls/ir/nodes.h"
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
using ::testing::AllOf;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::IsSupersetOf;
using ::testing::Key;
using ::testing::Not;
using ::testing::UnorderedElementsAre;
using ::testing::UnorderedPointwise;

struct Strategies {
  SchedulingStrategy primary;
  SchedulingStrategy bounds;
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Strategies& s) {
    absl::Format(&sink, "primary_%v_bounds_%v", s.primary, s.bounds);
  }
};
class PipelineScheduleTestBase
    : public IrTestBase,
      public testing::WithParamInterface<Strategies> {
 public:
  PipelineScheduleTestBase() = default;
  PipelineScheduleTestBase(PipelineScheduleTestBase&& other) = delete;
  PipelineScheduleTestBase(const PipelineScheduleTestBase& other) = delete;
  PipelineScheduleTestBase& operator=(PipelineScheduleTestBase&& other) =
      delete;
  PipelineScheduleTestBase& operator=(const PipelineScheduleTestBase& other) =
      delete;
  SchedulingOptions options() const {
    return SchedulingOptions(GetParam().primary, GetParam().bounds);
  }
  bool is_sdc() const { return GetParam().primary == SchedulingStrategy::SDC; }
  bool is_asap() const {
    return GetParam().primary == SchedulingStrategy::ASAP;
  }
  bool is_random() const {
    return GetParam().primary == SchedulingStrategy::RANDOM;
  }
  bool is_min_cut() const {
    return GetParam().primary == SchedulingStrategy::MIN_CUT;
  }
};

// ASAP, SDC, and Random primary, ASAP and SDC bounds in all combos
class PipelineScheduleTest : public PipelineScheduleTestBase {};

// Tests for error messages from scheduling. Currently does SDC ASAP in all
// combos since those 2 specifically have similar error formats.
//
// TODO(allight): Really the error messages should be created outside of the
// scheduler themselves so that we can have nice test error messages without
// having to depend on a particular implementation of scheduling.
class PipelineScheduleErrorTest : public PipelineScheduleTestBase {};
// Error message tests where only SDC can give all the information we are
// looking for. Most of these are checking for throughput change suggestions
// that SDC can extract from the model but ASAP cannot.
class SdcOnlyPipelineScheduleErrorTest : public PipelineScheduleTestBase {};
// Pure asap
class AsapPipelineScheduleTest : public PipelineScheduleTestBase {};
// Pure SDC
class SdcPipelineScheduleTest : public PipelineScheduleTestBase {};
class SdcPrimaryPipelineScheduleTest : public PipelineScheduleTestBase {};
// Random scheduler only.
class RandomPipelineScheduleTest : public PipelineScheduleTestBase {};

struct LabeledFeedbackArcProc {
  std::unique_ptr<Package> package;
  Proc* proc;
  BValue read;
  BValue next;
};

absl::StatusOr<LabeledFeedbackArcProc> BuildLabeledFeedbackArcProc(
    std::optional<std::string> write_label,
    std::optional<std::string> read_label) {
  auto package = std::make_unique<Package>("test_package");
  Type* u32 = package->GetBitsType(32);
  XLS_ASSIGN_OR_RETURN(auto out_ch, package->CreateStreamingChannel(
                                        "out_ch", ChannelOps::kSendOnly, u32));

  ProcBuilder pb("the_proc", package.get());
  BValue tkn = pb.Literal(Value::Token());
  XLS_ASSIGN_OR_RETURN(auto se,
                       pb.UnreadStateElement("state", Value(UBits(0, 32)),
                                             /*non_synthesizable=*/false));
  BValue read = pb.StateRead(se, /*predicate=*/std::nullopt, read_label);
  BValue add_val = pb.Add(read, pb.Literal(UBits(1, 32)));
  pb.Send(out_ch, tkn, add_val);
  BValue next = pb.Next(se, add_val, /*pred=*/std::nullopt, write_label);
  XLS_ASSIGN_OR_RETURN(auto proc, pb.Build());

  return LabeledFeedbackArcProc{
      .package = std::move(package),
      .proc = proc,
      .read = read,
      .next = next,
  };
}

TEST_P(PipelineScheduleTest, SelectsEntry) {
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

TEST_P(AsapPipelineScheduleTest, AsapScheduleTrivial) {
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

TEST_P(PipelineScheduleErrorTest, OutrightInfeasibleSchedule) {
  // Create a schedule in which the critical path doesn't even fit in the
  // requested clock_period * stages.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue x = fb.Param("x", p->GetBitsType(32));
  fb.Not(fb.Not(fb.Not(fb.Not(x))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ASSERT_THAT(
      RunPipelineSchedule(f, TestDelayEstimator(),
                          options().clock_period_ps(1).pipeline_stages(2))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("--pipeline_stages=4")));
}

TEST_P(PipelineScheduleErrorTest, InfeasibleScheduleWithBinPacking) {
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
                          options().clock_period_ps(2).pipeline_stages(2))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("--pipeline_stages=3")));
}

TEST_P(PipelineScheduleErrorTest, InfeasibleScheduleWithReturnValueUsers) {
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
                          options().clock_period_ps(1).pipeline_stages(2))
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("cannot achieve the specified clock period. Try "
                         "`--clock_period_ps=2`")));
}

TEST_P(AsapPipelineScheduleTest, AsapScheduleNoParameters) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Negate(fb.Add(fb.Literal(UBits(42, 8)), fb.Literal(UBits(100, 8))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(f, TestDelayEstimator(),
                                               options().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(m::Add(), m::Literal(42), m::Literal(100)));
  EXPECT_THAT(schedule.nodes_in_cycle(1), UnorderedElementsAre(m::Neg()));
}

TEST_P(AsapPipelineScheduleTest, AsapScheduleIncrementChain) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  fb.Add(fb.Add(fb.Add(x, fb.Literal(UBits(1, 32))), fb.Literal(UBits(1, 32))),
         fb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(f, TestDelayEstimator(),
                                               options().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(m::Param("x"), m::Add(), m::Literal(1),
                                   m::Literal(1), m::Literal(1)));
  EXPECT_THAT(schedule.nodes_in_cycle(1), UnorderedElementsAre(m::Add()));
  EXPECT_THAT(schedule.nodes_in_cycle(2), UnorderedElementsAre(m::Add()));
}

TEST_P(SdcPrimaryPipelineScheduleTest, MinimizeRegisterBitslices) {
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

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(f, TestDelayEstimator(),
                                               options().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(m::Param("x"), m::Param("y"),
                                   m::BitSlice(m::Param("y")), m::Neg()));
  EXPECT_THAT(
      schedule.nodes_in_cycle(1),
      UnorderedElementsAre(m::BitSlice(m::Param("x")), m::Neg(), m::Concat()));
}

TEST_P(AsapPipelineScheduleTest, AsapScheduleComplex) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  fb.Negate(fb.Concat({(fb.Not(x | y) - z) * x, z + z}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(f, TestDelayEstimator(),
                                               options().clock_period_ps(2)));

  EXPECT_EQ(schedule.length(), 3);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(m::Param("x"), m::Param("y"), m::Param("z"),
                                   m::Or(), m::Not(), m::Add()));
  EXPECT_THAT(schedule.nodes_in_cycle(1),
              UnorderedElementsAre(m::Concat(), m::Sub(), m::UMul()));
  EXPECT_THAT(schedule.nodes_in_cycle(2), UnorderedElementsAre(m::Neg()));
}

TEST_P(PipelineScheduleTest, JustClockPeriodGiven) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  fb.Negate(fb.Concat({(fb.Not(x | y) - z) * x, z + z}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(func, TestDelayEstimator(),
                                               options().clock_period_ps(2)));

  // Returns the unique scheduled Ops in the given cycle.
  auto scheduled_ops = [&](int64_t cycle) {
    absl::flat_hash_set<Op> ops;
    for (const auto& node : schedule.nodes_in_cycle(cycle)) {
      ops.insert(node->op());
    }
    return ops;
  };

  EXPECT_EQ(schedule.length(), 3);
  if (is_sdc()) {
    EXPECT_THAT(scheduled_ops(0),
                UnorderedElementsAre(Op::kParam, Op::kOr, Op::kNot));
    EXPECT_THAT(scheduled_ops(1), UnorderedElementsAre(Op::kAdd, Op::kConcat,
                                                       Op::kUMul, Op::kSub));
    EXPECT_THAT(scheduled_ops(2), UnorderedElementsAre(Op::kNeg));
    EXPECT_THAT(scheduled_ops(3), UnorderedElementsAre());
  } else if (is_asap()) {
    EXPECT_THAT(scheduled_ops(0),
                UnorderedElementsAre(Op::kParam, Op::kOr, Op::kNot, Op::kAdd));
    EXPECT_THAT(scheduled_ops(1),
                UnorderedElementsAre(Op::kConcat, Op::kUMul, Op::kSub));
    EXPECT_THAT(scheduled_ops(2), UnorderedElementsAre(Op::kNeg));
    EXPECT_THAT(scheduled_ops(3), IsEmpty());
  } else {
    EXPECT_TRUE(is_random());
  }
}

TEST_P(PipelineScheduleTest, TestVerifyTiming) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto x_plus_y = fb.Add(x, y);
  fb.Subtract(x_plus_y, fb.Negate(x_plus_y));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(func, TestDelayEstimator(),
                                               options().clock_period_ps(5)));

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

  XLS_ASSERT_OK_AND_ASSIGN(DelayManager delay_manager,
                           DelayManager::Create(func, TestDelayEstimator()));
  XLS_EXPECT_OK(schedule.VerifyTiming(/*clock_period_ps=*/5, delay_manager));
  EXPECT_THAT(
      schedule.VerifyTiming(/*clock_period_ps=*/1, delay_manager),
      absl_testing::StatusIs(
          absl::StatusCode::kInternal,
          ::testing::HasSubstr(
              "Schedule does not meet timing (1ps). Longest failing path "
              "(3ps): add.3 (1ps) -> neg.4 (1ps) -> sub.5 (1ps)")));
}

TEST_P(PipelineScheduleTest, ClockPeriodAndPipelineLengthGiven) {
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
                          options().clock_period_ps(2).pipeline_stages(4)));

  // Returns the unique scheduled Ops in the given cycle.
  auto scheduled_ops = [&](int64_t cycle) {
    absl::flat_hash_set<Op> ops;
    for (const auto& node : schedule.nodes_in_cycle(cycle)) {
      ops.insert(node->op());
    }
    return ops;
  };

  EXPECT_EQ(schedule.length(), 4);
  if (is_sdc()) {
    EXPECT_THAT(scheduled_ops(0),
                UnorderedElementsAre(Op::kParam, Op::kOr, Op::kNeg));
    EXPECT_THAT(scheduled_ops(1),
                UnorderedElementsAre(Op::kAdd, Op::kNot, Op::kSub));
    EXPECT_THAT(scheduled_ops(2), UnorderedElementsAre(Op::kConcat, Op::kUMul));
    EXPECT_THAT(scheduled_ops(3), UnorderedElementsAre(Op::kNeg));
  } else if (is_asap()) {
    // Since ASAP doesn't minimize regs the assignments are slightly different.
    EXPECT_THAT(scheduled_ops(0),
                UnorderedElementsAre(Op::kParam, Op::kOr, Op::kNeg, Op::kAdd));
    EXPECT_THAT(scheduled_ops(1), UnorderedElementsAre(Op::kNot, Op::kSub));
    EXPECT_THAT(scheduled_ops(2),
                UnorderedElementsAre(Op::kConcat, Op::kUMul, Op::kNeg));
    EXPECT_THAT(scheduled_ops(3), IsEmpty());
  } else {
    EXPECT_TRUE(is_random());
  }
}

TEST_P(PipelineScheduleTest, JustPipelineLengthGiven) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  fb.Negate(fb.Concat({(fb.Not(fb.Negate(x | y)) - z) * x, z + z}));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(func, TestDelayEstimator(),
                                               options().pipeline_stages(6)));

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
  if (is_random()) {
    return;
  }
  EXPECT_TRUE(is_sdc() || is_asap());
  if (is_sdc()) {
    EXPECT_THAT(scheduled_ops(0), UnorderedElementsAre(Op::kParam, Op::kOr));
  } else {
    EXPECT_THAT(scheduled_ops(0),
                UnorderedElementsAre(Op::kParam, Op::kOr, Op::kAdd));
  }
  EXPECT_THAT(scheduled_ops(1), UnorderedElementsAre(Op::kNeg));
  EXPECT_THAT(scheduled_ops(2), UnorderedElementsAre(Op::kNot));
  if (is_sdc()) {
    EXPECT_THAT(scheduled_ops(3), UnorderedElementsAre(Op::kAdd, Op::kSub));
  } else {
    EXPECT_THAT(scheduled_ops(3), UnorderedElementsAre(Op::kSub));
  }
  EXPECT_THAT(scheduled_ops(4), UnorderedElementsAre(Op::kConcat, Op::kUMul));
  EXPECT_THAT(scheduled_ops(5), UnorderedElementsAre(Op::kNeg));
}

TEST_P(PipelineScheduleTest, LongPipelineLength) {
  // Generate an absurdly long pipeline schedule. Most stages are empty, but it
  // should not crash.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto bitslice = fb.BitSlice(x, /*start=*/7, /*width=*/20);
  auto zero_ext = fb.ZeroExtend(bitslice, /*new_bit_count=*/32);

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(func, TestDelayEstimator(),
                                               options().pipeline_stages(100)));

  EXPECT_EQ(schedule.length(), 100);
  // Most stages should be empty.
  if (is_random()) {
    return;
  }
  EXPECT_TRUE(is_sdc() || is_asap());
  if (is_sdc()) {
    EXPECT_THAT(schedule.nodes_in_cycle(0),
                UnorderedElementsAre(x.node(), bitslice.node()));
  } else {
    EXPECT_THAT(
        schedule.nodes_in_cycle(0),
        UnorderedElementsAre(x.node(), bitslice.node(), zero_ext.node()));
  }
  // The bitslice is the narrowest among the chain of operations so it should
  // precede the long chain of empty stages if we are SDC. It can fit in the
  // first stage so ASAP will place it there.
  for (int64_t i = 1; i < 99; ++i) {
    EXPECT_THAT(schedule.nodes_in_cycle(i), UnorderedElementsAre());
  }
  if (is_sdc()) {
    EXPECT_THAT(schedule.nodes_in_cycle(99),
                UnorderedElementsAre(zero_ext.node()));
  } else {
    EXPECT_THAT(schedule.nodes_in_cycle(99), IsEmpty());
  }
}

TEST_P(PipelineScheduleTest, ClockPeriodMargin) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  // Chain of six negates.
  fb.Negate(fb.Negate(fb.Negate(fb.Negate(fb.Negate(fb.Negate(x))))));

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(func, TestDelayEstimator(),
                                               options().clock_period_ps(3)));
  EXPECT_EQ(schedule.length(), 2);

  {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func, TestDelayEstimator(),
            options().clock_period_ps(3).clock_margin_percent(0)));
    EXPECT_EQ(schedule.length(), 2);
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func, TestDelayEstimator(),
            options().clock_period_ps(3).clock_margin_percent(33)));
    EXPECT_EQ(schedule.length(), 3);
  }
  {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func, TestDelayEstimator(),
            options().clock_period_ps(3).clock_margin_percent(66)));
    EXPECT_EQ(schedule.length(), 6);
  }
  EXPECT_THAT(
      RunPipelineSchedule(
          func, TestDelayEstimator(),
          options().clock_period_ps(3).clock_margin_percent(200))
          .status(),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr(
              "Clock period non-positive (-3ps) after adjusting for margin. "
              "Original clock period: 3ps, clock margin: 200%")));
}

TEST_P(PipelineScheduleTest, PeriodRelaxation) {
  if (is_random()) {
    GTEST_SKIP() << "Relaxation not guarneteed to improve schedule quality "
                    "with random scheduler.";
  }
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

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(func, TestDelayEstimator(),
                                               options().pipeline_stages(2)));
  EXPECT_EQ(schedule.length(), 2);
  int64_t reg_count_default = schedule.CountFinalInteriorPipelineRegisters();

  for (int64_t relax_percent : std::vector{50, 100}) {
    XLS_ASSERT_OK_AND_ASSIGN(
        PipelineSchedule schedule,
        RunPipelineSchedule(
            func, TestDelayEstimator(),
            options().pipeline_stages(2).period_relaxation_percent(
                relax_percent)));
    EXPECT_EQ(schedule.length(), 2);
    int64_t reg_count_relaxed = schedule.CountFinalInteriorPipelineRegisters();
    EXPECT_LT(reg_count_relaxed, reg_count_default);
  }
}

TEST_P(PipelineScheduleTest, SerializeAndDeserialize) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  Type* u32 = p->GetBitsType(32);
  auto x = fb.Param("x", u32);
  auto y = fb.Param("y", u32);
  auto z = fb.Param("z", u32);
  fb.Negate(fb.Concat({(fb.Not(fb.Negate(x | y)) - z) * x, z + z}));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(func, TestDelayEstimator(),
                                               options().pipeline_stages(3)));

  ASSERT_TRUE(schedule.min_clock_period_ps().has_value());
  XLS_ASSERT_OK_AND_ASSIGN(PipelineScheduleProto proto,
                           schedule.ToProto(TestDelayEstimator()));
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

TEST_P(PipelineScheduleTest, NodeDelayInScheduleProto) {
  // Tests that node and path delays are serialized in the schedule proto
  // using trivial pipeline: 3 stages of 2 x 1-bit inverters.
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(1));
  fb.Not(fb.Not(fb.Not(fb.Not(fb.Not(fb.Not(x))))));
  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(func, TestDelayEstimator(),
                                               options().pipeline_stages(3)));

  XLS_ASSERT_OK_AND_ASSIGN(PipelineScheduleProto proto,
                           schedule.ToProto(TestDelayEstimator()));
  for (const auto& stage : proto.stages()) {
    int64_t path_delay = 0;
    for (const TimedNodeProto& node : stage.timed_nodes()) {
      path_delay += node.node_delay_ps();
      EXPECT_EQ(node.path_delay_ps(), path_delay);
    }
  }
}

TEST_P(PipelineScheduleTest, ProcSchedule) {
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

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, TestDelayEstimator(),
                                               options().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);

  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 2);
}

TEST_P(PipelineScheduleTest, StatelessProcSchedule) {
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

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, TestDelayEstimator(),
                                               options().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);

  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 2);
}

TEST_P(PipelineScheduleTest, MultistateProcSchedule) {
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

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, TestDelayEstimator(),
                                               options().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);

  if (is_random()) {
    return;
  }
  EXPECT_EQ(schedule.cycle(st0.node()), 0);
  EXPECT_EQ(schedule.cycle(st1.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 2);
}

TEST_P(PipelineScheduleTest, ProcWithConditionalReceive) {
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

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, TestDelayEstimator(),
                                               options().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 3);

  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  // Literals are "untimed" and not in the schedule for SDC.
  // TODO(allight): We might want to match this with ASAP and remove this check.
  if (is_sdc()) {
    EXPECT_FALSE(schedule.IsScheduled(cond.node()));
  }
  EXPECT_EQ(schedule.cycle(send.node()), 2);
}

TEST_P(PipelineScheduleTest, ProcWithConditionalReceiveLongCondition) {
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
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, TestDelayEstimator(),
                                               options().clock_period_ps(1)));

  EXPECT_EQ(schedule.length(), 5);

  EXPECT_EQ(schedule.cycle(cond.node()), 1);
  // NB ASAP location is cycle 1 but since the condition is 1 bit and the recv'd
  // value is 16 SDC pushes it back one cycle to save register bits.
  EXPECT_GE(schedule.cycle(rcv.node()), 1);
}

TEST_P(PipelineScheduleTest, ReceiveFollowedBySend) {
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
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, *delay_estimator,
                                               options().pipeline_stages(5)));
  EXPECT_EQ(schedule.length(), 5);
  EXPECT_GE(schedule.cycle(send.node()), schedule.cycle(rcv.node()));
}

TEST_P(PipelineScheduleErrorTest, SendFollowedByReceiveCannotBeInSameCycle) {
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
  ASSERT_THAT(
      RunPipelineSchedule(proc, *delay_estimator, options().pipeline_stages(1)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          options().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 1);
}

TEST_P(PipelineScheduleErrorTest, SendFollowedByReceiveIfCannotBeInSameCycle) {
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
  ASSERT_THAT(
      RunPipelineSchedule(proc, *delay_estimator, options().pipeline_stages(1)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          options().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv_if.node()), 1);
}

TEST_P(PipelineScheduleErrorTest,
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
  ASSERT_THAT(
      RunPipelineSchedule(proc, *delay_estimator, options().pipeline_stages(1)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          options().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 1);
}

TEST_P(PipelineScheduleErrorTest,
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
  ASSERT_THAT(
      RunPipelineSchedule(proc, *delay_estimator, options().pipeline_stages(1)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          options().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv_if.node()), 1);
}

TEST_P(PipelineScheduleErrorTest,
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
  ASSERT_THAT(
      RunPipelineSchedule(proc, *delay_estimator, options().pipeline_stages(1)),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("--pipeline_stages=2")));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, *delay_estimator,
                          options().clock_period_ps(10'000)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 1);
}

TEST_P(PipelineScheduleTest, SendFollowedByDelayedReceive) {
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
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, *delay_estimator,
                                               options().pipeline_stages(5)));
  EXPECT_EQ(schedule.length(), 5);
  if (is_random()) {
    EXPECT_GE(schedule.cycle(rcv.node()) - schedule.cycle(send.node()), 3);
  } else {
    EXPECT_EQ(schedule.cycle(rcv.node()) - schedule.cycle(send.node()), 3);
  }
}

TEST_P(PipelineScheduleTest, SendFollowedByDelayedReceiveWithState) {
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
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, *delay_estimator,
                                               options().pipeline_stages(5)));
  EXPECT_EQ(schedule.length(), 5);
  EXPECT_EQ(schedule.cycle(send.node()) + 1, schedule.cycle(rcv.node()));
}

TEST_P(PipelineScheduleErrorTest, SuggestIncreasedPipelineLengthWhenNeeded) {
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
  EXPECT_THAT(RunPipelineSchedule(
                  proc, *delay_estimator,
                  options().pipeline_stages(1).worst_case_throughput(3)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--pipeline_stages=3"),
                             Not(HasSubstr("--worst_case_throughput")))));
}

TEST_P(SdcOnlyPipelineScheduleErrorTest,
       SuggestReducedThroughputWhenFullThroughputFails) {
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
  EXPECT_THAT(RunPipelineSchedule(
                  proc, *delay_estimator,
                  options().pipeline_stages(5).worst_case_throughput(1)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--worst_case_throughput=3"),
                             Not(HasSubstr("--pipeline_stages")))));
}

TEST_P(PipelineScheduleTest, UnboundedThroughputWorks) {
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
          options().pipeline_stages(5).worst_case_throughput(0)));
  EXPECT_EQ(schedule.length(), 5);
  if (is_random()) {
    EXPECT_GE(schedule.cycle(rcv.node()) - schedule.cycle(send.node()), 2);
  } else {
    EXPECT_EQ(schedule.cycle(rcv.node()) - schedule.cycle(send.node()), 2);
  }
}

TEST_P(PipelineScheduleTest, MinimizedThroughputWorksWithGivenPipelineLength) {
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
                          options()
                              .pipeline_stages(5)
                              .worst_case_throughput(0)
                              .minimize_worst_case_throughput(true)));
  EXPECT_EQ(schedule.length(), 5);
  EXPECT_EQ(schedule.cycle(rcv.node()) - schedule.cycle(send.node()), 2);
  EXPECT_EQ(proc->GetInitiationInterval().value_or(1), 4);
}

TEST_P(PipelineScheduleTest, MinimizedThroughputWorksWithGivenClockPeriod) {
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
                          options()
                              .clock_period_ps(2)
                              .worst_case_throughput(0)
                              .minimize_worst_case_throughput(true)));
  EXPECT_EQ(schedule.length(), 3);
  EXPECT_EQ(schedule.cycle(send.node()), 0);
  EXPECT_EQ(schedule.cycle(rcv.node()), 2);
  EXPECT_EQ(proc->GetInitiationInterval().value_or(1), 3);
}

TEST_P(SdcOnlyPipelineScheduleErrorTest,
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
                                  options()
                                      .clock_period_ps(1000)
                                      .worst_case_throughput(1)
                                      .minimize_clock_on_failure(false)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--worst_case_throughput=3"),
                             Not(HasSubstr("--pipeline_stages")))));
}

TEST_P(SdcOnlyPipelineScheduleErrorTest,
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
                                  options()
                                      .pipeline_stages(1)
                                      .worst_case_throughput(1)
                                      .minimize_worst_case_throughput(true)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       AllOf(HasSubstr("--pipeline_stages=3"),
                             HasSubstr("--worst_case_throughput=3"))));
}

TEST_P(SdcOnlyPipelineScheduleErrorTest,
       SuggestIncreasedPipelineLengthAndIndividualSlack) {
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
  pb.Next(state, pb.TupleIndex(rcv, 1), /*pred=*/std::nullopt,
          /*label=*/std::nullopt, SourceInfo(), "next_state");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  auto sched = RunPipelineSchedule(
      proc, *delay_estimator,
      options().pipeline_stages(1).failure_behavior(SchedulingFailureBehavior{
          .explain_infeasibility = true,
          .infeasible_per_state_backedge_slack_pool = 2.0}));
  if (is_sdc()) {
    EXPECT_THAT(
        sched,
        StatusIs(
            absl::StatusCode::kInvalidArgument,
            AllOf(HasSubstr("--pipeline_stages=3"),
                  HasSubstr("looking at paths between state and next_state "
                            "(needs 2 additional slack)"))));
  } else {
    EXPECT_THAT(sched, StatusIs(absl::StatusCode::kInvalidArgument,
                                HasSubstr("--pipeline_stages=3")));
  }
  EXPECT_THAT(
      sched,
      StatusIs(absl::StatusCode::kInvalidArgument,
               AllOf(HasSubstr("--pipeline_stages=3"),
                     HasSubstr("looking at paths between state and next_state "
                               "(needs 2 additional slack)"))));
}

TEST_P(
    SdcOnlyPipelineScheduleErrorTest,
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
  pb.Next(state0, rcv_data, /*pred=*/std::nullopt, /*label=*/std::nullopt,
          SourceInfo(), "next_state0");
  pb.Next(state1, pb.Add(rcv_data, state1), /*pred=*/std::nullopt,
          /*label=*/std::nullopt, SourceInfo(), "next_state1");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  EXPECT_THAT(
      RunPipelineSchedule(
          proc, *delay_estimator,
          options().pipeline_stages(1).failure_behavior(
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

TEST_P(
    SdcOnlyPipelineScheduleErrorTest,
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
  pb.Next(state0, rcv_data, /*pred=*/std::nullopt, /*label=*/std::nullopt,
          SourceInfo(), "next_state0");
  pb.Next(state1, pb.Add(rcv_data, state1), /*pred=*/std::nullopt,
          /*label=*/std::nullopt, SourceInfo(), "next_state1");
  pb.Next(state2, pb.Add(rcv_data, state2), /*pred=*/std::nullopt,
          /*label=*/std::nullopt, SourceInfo(), "next_state2");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(const DelayEstimator* delay_estimator,
                           GetDelayEstimator("unit"));
  EXPECT_THAT(
      RunPipelineSchedule(
          proc, *delay_estimator,
          options().pipeline_stages(1).failure_behavior(
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

void SuggestIncreasedClockPeriodWhenNecessaryCommon(
    std::string test_name, const DelayEstimator& delay_estimator,
    SchedulingOptions options, std::string expected) {
  Package package = Package(test_name);

  Type* u32 = package.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_out,
      package.CreateStreamingChannel("out", ChannelOps::kSendOnly, u32));

  ProcBuilder pb(test_name, &package);
  BValue tkn = pb.Literal(Value::Token());
  BValue state = pb.StateElement("state", Value(Bits(32)));

  pb.Send(ch_out, tkn, state);
  BValue add2 = pb.Add(state, pb.Literal(UBits(2, 32)));
  BValue mul3 = pb.UMul(add2, pb.Literal(UBits(3, 32)));
  BValue add1 = pb.Add(mul3, pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({add1}));

  // Each operation takes 500ps, so (with no pipeline depth restrictions), 500ps
  // is the fastest clock we can support.
  EXPECT_THAT(
      RunPipelineSchedule(proc, delay_estimator, options),
      StatusIs(absl::StatusCode::kInvalidArgument, HasSubstr(expected)));
}
TEST_P(PipelineScheduleErrorTest, SuggestIncreasedClockPeriodWhenNecessary500) {
  // Each operation takes 500ps, so (with no pipeline depth restrictions), 500ps
  // is the fastest clock we can support.
  SuggestIncreasedClockPeriodWhenNecessaryCommon(
      TestName(), TestDelayEstimator(500),
      options().clock_period_ps(100).worst_case_throughput(0),
      "--clock_period_ps=500");
}
TEST_P(PipelineScheduleErrorTest,
       SuggestIncreasedClockPeriodWhenNecessary1000) {
  // Each operation takes 500ps, but we have a chain of three operations; in two
  // stages, the best we can do is a 1000ps clock.
  SuggestIncreasedClockPeriodWhenNecessaryCommon(
      TestName(), TestDelayEstimator(500),
      options().clock_period_ps(100).pipeline_stages(2).worst_case_throughput(
          0),
      "--clock_period_ps=1000");
}
TEST_P(PipelineScheduleErrorTest,
       SuggestIncreasedClockPeriodWhenNecessary3Stage) {
  // Each operation takes 500ps, and our schedule fits nicely into 3 stages; we
  // can get down to a 500ps clock at 3 or more pipeline stages.
  SuggestIncreasedClockPeriodWhenNecessaryCommon(
      TestName(), TestDelayEstimator(500),
      options().clock_period_ps(100).pipeline_stages(3).worst_case_throughput(
          0),
      "--clock_period_ps=500");
}
TEST_P(PipelineScheduleErrorTest,
       SuggestIncreasedClockPeriodWhenNecessary20Stage) {
  // Each operation takes 500ps, and our schedule fits nicely into 3 stages; we
  // can get down to a 500ps clock at 3 or more pipeline stages.
  SuggestIncreasedClockPeriodWhenNecessaryCommon(
      TestName(), TestDelayEstimator(500),
      options().clock_period_ps(100).pipeline_stages(20).worst_case_throughput(
          0),
      "--clock_period_ps=500");
}
TEST_P(PipelineScheduleErrorTest,
       SuggestIncreasedClockPeriodWhenNecessary4StageNoMinimize) {
  // But... if told not to search for the smallest possible clock period, the
  // best we can do is signal that a longer clock period might help.
  SuggestIncreasedClockPeriodWhenNecessaryCommon(
      TestName(), TestDelayEstimator(500),
      options()
          .clock_period_ps(100)
          .pipeline_stages(4)
          .minimize_clock_on_failure(false),
      "Try increasing `--clock_period_ps`");
}

TEST_P(SdcPrimaryPipelineScheduleTest, OptimizeForDynamicThroughput) {
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
  //
  // NB This is only true for SDC scheduling. ASAP will put it in stage 0
  // regardless of the penalty.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(
          proc, *delay_estimator,
          options().clock_period_ps(10).worst_case_throughput(2)));
  EXPECT_EQ(schedule.cycle(next_state.node()) - schedule.cycle(state.node()),
            1);

  // On the other hand, if we do optimize for dynamic throughput, we end up with
  // full throughput.
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule2,
      RunPipelineSchedule(proc, *delay_estimator,
                          options()
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
TEST_P(PipelineScheduleTest, ProcParamScheduledEarlyWithNextState) {
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
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, *delay_estimator,
                                               options().pipeline_stages(3)));
  EXPECT_EQ(schedule.length(), 3);
  if (!is_random()) {
    // The state's param node should be scheduled ASAP (i.e., the next-state
    // node's stage), while still leaving its user in the later stage.
    EXPECT_EQ(schedule.cycle(state.node()),
              schedule.cycle(nb_rcv_valid.node()));
    EXPECT_LT(schedule.cycle(state.node()), schedule.cycle(use_state.node()));
  }
}

// Proc next state does not depend on param. Force schedule of a param node's
// user in a later stage than the next state is computed. The schedule can be
// forced by having two receive nodes where the second receive node depends on
// the first node, and the first receive node produces the next state node and
// the param is used by the second receive node. We make the scheduler prefer to
// schedule the next-state computation earlier by making it narrower than the
// param value, then widening later.
//
// We also ask to minimize WCT to force the next state and state to be close
// even on schedulers incapable of minimizing cross cycle edges such as ASAP.
TEST_P(PipelineScheduleTest, ProcParamScheduledAfterNextState) {
  if (is_random()) {
    // TODO(allight): Fix this.
    GTEST_SKIP() << "Random scheduler can fail this for unclear reasons.";
  }
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
      RunPipelineSchedule(
          proc, *delay_estimator,
          options().pipeline_stages(3).minimize_worst_case_throughput(true)));
  EXPECT_EQ(schedule.length(), 3);
  EXPECT_GT(schedule.cycle(state.node()), schedule.cycle(nb_rcv_valid.node()));
}

// If two param nodes are mutually dependent, they (and their next state nodes)
// all need to be scheduled in the same stage.
//
// TODO(allight): This is only true for SDC I think.
TEST_P(PipelineScheduleTest, ProcParamsScheduledInSameStage) {
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
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, *delay_estimator,
                                               options().pipeline_stages(3)));
  EXPECT_EQ(schedule.length(), 3);
  if (!is_random()) {
    EXPECT_EQ(schedule.cycle(a.node()), schedule.cycle(b.node()));
  }
  EXPECT_EQ(schedule.cycle(next_a.node()), schedule.cycle(next_a.node()));
}

TEST_P(PipelineScheduleTest, FunctionScheduleWithInputAndOutputDelay) {
  Package p("p");

  Type* u16 = p.GetBitsType(16);

  FunctionBuilder fb("f", &p);

  BValue x = fb.Param("x", u16);
  BValue y = fb.Param("y", u16);
  BValue prod = fb.UMul(x, y);
  BValue negate = fb.Negate(prod);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(negate));

  // No additional input/output delay, we get [{x, y, prod, negate}]
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(f, TestDelayEstimator(),
                                               options().clock_period_ps(2)));
  ASSERT_EQ(schedule.length(), 1);
  EXPECT_THAT(
      schedule.nodes_in_cycle(0),
      UnorderedElementsAre(x.node(), y.node(), prod.node(), negate.node()));

  // Additional input delay bumps prod to stage 2, we get
  // [{x,y}, {prod, negate}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, RunPipelineSchedule(
                    f, TestDelayEstimator(),
                    options().clock_period_ps(2).additional_input_delay_ps(2)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(x.node(), y.node()));
  EXPECT_THAT(schedule.nodes_in_cycle(1),
              UnorderedElementsAre(prod.node(), negate.node()));

  // Additional output delay bumps negate to stage 3, we get
  // [{x,y}, {prod}, {negate}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, RunPipelineSchedule(f, TestDelayEstimator(),
                                    options()
                                        .clock_period_ps(2)
                                        .additional_input_delay_ps(2)
                                        .additional_output_delay_ps(1)));
  ASSERT_EQ(schedule.length(), 3);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              UnorderedElementsAre(x.node(), y.node()));
  EXPECT_THAT(schedule.nodes_in_cycle(1), UnorderedElementsAre(prod.node()));
  EXPECT_THAT(schedule.nodes_in_cycle(2), UnorderedElementsAre(negate.node()));
}

TEST_P(PipelineScheduleTest, ProcScheduleWithInputDelay) {
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

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, TestDelayEstimator(),
                                               options().clock_period_ps(4)));
  EXPECT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 1);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              Contains(OperationDelayInPs(IsOkAndHolds(Gt(0)))));

  // Input delay of 1 is not large enough to bump all non-zero-latency nodes to
  // later stages.
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, RunPipelineSchedule(
                    proc, TestDelayEstimator(),
                    options().clock_period_ps(4).additional_input_delay_ps(1)));
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
      schedule, RunPipelineSchedule(
                    proc, TestDelayEstimator(),
                    options().clock_period_ps(4).additional_input_delay_ps(4)));
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              Each(OperationDelayInPs(IsOkAndHolds(0))));
}

TEST_P(PipelineScheduleTest, ProcScheduleWithInputAndOutputDelay) {
  if (is_random()) {
    GTEST_SKIP() << "Skipping test for random scheduler due to being unable to "
                    "realistically check output.";
  }
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
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, TestDelayEstimator(),
                                               options().clock_period_ps(2)));
  ASSERT_EQ(schedule.length(), 1);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv.node(), negate.node(), send.node()}));

  // Input delay bumps send to stage 1, we get [{rcv, negate}, {send}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, RunPipelineSchedule(
                    proc, TestDelayEstimator(),
                    options().clock_period_ps(2).additional_input_delay_ps(1)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv.node(), negate.node()}));
  EXPECT_THAT(schedule.nodes_in_cycle(1), IsSupersetOf({send.node()}));

  // Output delay also bumps send to stage 1, we get [{rcv, negate}, {send}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule,
      RunPipelineSchedule(
          proc, TestDelayEstimator(),
          options().clock_period_ps(2).additional_output_delay_ps(1)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv.node(), negate.node()}));
  EXPECT_THAT(schedule.nodes_in_cycle(1), IsSupersetOf({send.node()}));

  // Specifying both input and output delay doesn't change anything.
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule, RunPipelineSchedule(proc, TestDelayEstimator(),
                                    options()
                                        .clock_period_ps(2)
                                        .additional_input_delay_ps(1)
                                        .additional_output_delay_ps(1)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv.node(), negate.node()}));
  EXPECT_THAT(schedule.nodes_in_cycle(1), IsSupersetOf({send.node()}));
}

TEST_P(PipelineScheduleTest, ProcScheduleWithChannelSpecificDelay) {
  if (is_random()) {
    GTEST_SKIP() << "Unable to verify output for random scheduler.";
  }
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
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           RunPipelineSchedule(proc, TestDelayEstimator(),
                                               options().clock_period_ps(2)));
  ASSERT_EQ(schedule.length(), 1);
  EXPECT_THAT(
      schedule.nodes_in_cycle(0),
      IsSupersetOf({rcv1.node(), rcv2.node(), sum.node(), send.node()}));

  // Small ch2 delay bumps just the sum to stage 1, we get:
  // [{rcv1, neg_rcv1, rcv2, neg_rcv2}, {sum, send}]
  XLS_ASSERT_OK_AND_ASSIGN(
      schedule,
      RunPipelineSchedule(
          proc, TestDelayEstimator(),
          options().clock_period_ps(2).add_additional_channel_delay_ps("in2",
                                                                       1)));
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
      RunPipelineSchedule(
          proc, TestDelayEstimator(),
          options().clock_period_ps(2).add_additional_channel_delay_ps("in2",
                                                                       2)));
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_THAT(schedule.nodes_in_cycle(0),
              IsSupersetOf({rcv1.node(), neg_rcv1.node(), rcv2.node()}));
  EXPECT_THAT(schedule.nodes_in_cycle(1),
              IsSupersetOf({neg_rcv2.node(), sum.node(), send.node()}));
}

TEST_P(PipelineScheduleTest, ProcScheduleWithChannelDirectionSpecificDelay) {
  Package p("p");

  Type* u16 = p.GetBitsType(16);

  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch,
      p.CreateStreamingChannel("ch", ChannelOps::kSendReceive, u16));

  TokenlessProcBuilder pb("the_proc", "tkn", &p);

  BValue rcv = pb.Receive(ch);
  BValue neg_rcv = pb.Negate(rcv);
  BValue send = pb.Send(ch, neg_rcv);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({}));

  // Set different delays for send and receive on channel "ch".
  SchedulingOptions options =
      this->options()
          .clock_period_ps(5)
          .add_additional_channel_delay_ps("ch:recv", 1)
          .add_additional_channel_delay_ps("ch:send", 5);

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(), options));

  // rcv: Delay 1. Fits in cycle 0.
  // neg_rcv: Starts no sooner than t=1, with delay 1. Fits in cycle 0.
  // send: Starts no sooner than t=2, with delay 5. Does not fit in cycle 0.
  ASSERT_EQ(schedule.length(), 2);
  EXPECT_EQ(schedule.cycle(rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(neg_rcv.node()), 0);
  EXPECT_EQ(schedule.cycle(send.node()), 1);
}

TEST_P(PipelineScheduleTest, ProcScheduleWithConstraints) {
  if (is_random()) {
    GTEST_SKIP() << "Random scheduler does not fully respect this constraint "
                    "in all circumstances";
  }
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
        RunPipelineSchedule(proc, TestDelayEstimator(),
                            options().pipeline_stages(10).add_constraint(
                                IOConstraint("in", IODirection::kReceive, "out",
                                             IODirection::kSend, i, i))));

    EXPECT_EQ(schedule.length(), 10);
    EXPECT_EQ(schedule.cycle(send.node()) - schedule.cycle(rcv.node()), i);
  }
}

TEST_P(RandomPipelineScheduleTest, RandomScheduleRuns) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  for (int64_t i = 0; i < 20; ++i) {
    x = fb.Negate(x);
  }

  XLS_ASSERT_OK_AND_ASSIGN(Function * func, fb.Build());

  for (int32_t i = 0; i < 100; ++i) {
    // Running the scheduler will call `VerifyTiming`.
    XLS_ASSERT_OK(RunPipelineSchedule(func, TestDelayEstimator(),
                                      options().seed(i).pipeline_stages(50))
                      .status());
  }
}

TEST_P(SdcPipelineScheduleTest, SingleStageSchedule) {
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

TEST_P(PipelineScheduleTest, LoopbackChannelWithConstraint) {
  if (is_random()) {
    GTEST_SKIP() << "Random scheduler does not fully respect this constraint "
                    "in all circumstances";
  }
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
            options().pipeline_stages(10).add_constraint(
                IOConstraint("loopback", IODirection::kReceive, "loopback",
                             IODirection::kSend, i, i))));

    EXPECT_EQ(schedule.length(), 10);
    EXPECT_EQ(schedule.cycle(loopback_send.node()) - schedule.cycle(rcv.node()),
              i);
  }
}

TEST_P(PipelineScheduleTest, PackageScheduleProtoSerializeAndDeserialize) {
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
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule0,
                           RunPipelineSchedule(func0, TestDelayEstimator(),
                                               options().pipeline_stages(3)));
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule1,
                           RunPipelineSchedule(func1, TestDelayEstimator(),
                                               options().pipeline_stages(3)));

  PackageSchedule package_schedule(p.get(),
                                   {{func0, schedule0}, {func1, schedule1}});

  XLS_ASSERT_OK_AND_ASSIGN(PackageScheduleProto proto,
                           package_schedule.ToProto(TestDelayEstimator()));
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

// TODO(allight): We should rewrite the tests to allow for running with ASAP
// scheduler too.
TEST_P(SdcPipelineScheduleTest,
       SerializeAndDeserializeWithSynchronousSchedule) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb1("proc1", "tkn", p.get());
  auto literal1 = pb1.Literal(UBits(1, 32));
  auto add1 = pb1.Add(literal1, literal1);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, pb1.Build());

  TokenlessProcBuilder pb2("proc2", "tkn", p.get());
  auto literal2 = pb2.Literal(UBits(2, 32));
  auto add2 = pb2.Add(literal2, literal2);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc2, pb2.Build());

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule1,
                           RunPipelineSchedule(proc1, TestDelayEstimator(),
                                               options().clock_period_ps(1)));
  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule2,
                           RunPipelineSchedule(proc2, TestDelayEstimator(),
                                               options().clock_period_ps(1)));

  absl::flat_hash_map<FunctionBase*, int64_t> synchronous_offsets(
      {{proc1, 0}, {proc2, 42}});
  PackageSchedule package_schedule(
      p.get(), {{proc1, schedule1}, {proc2, schedule2}}, synchronous_offsets);

  EXPECT_TRUE(package_schedule.IsSynchronousSchedule());
  EXPECT_THAT(package_schedule.GetSchedules().at(proc1).GetCycleMap(),
              Not(Contains(Key(literal1.node()))));
  EXPECT_THAT(package_schedule.GetSchedules().at(proc2).GetCycleMap(),
              Not(Contains(Key(literal2.node()))));
  EXPECT_EQ(package_schedule.GetSynchronousCycle(add1.node()), 0);
  EXPECT_EQ(package_schedule.GetSynchronousCycle(add2.node()), 42);

  XLS_ASSERT_OK_AND_ASSIGN(PackageScheduleProto proto,
                           package_schedule.ToProto(TestDelayEstimator()));
  XLS_ASSERT_OK_AND_ASSIGN(PackageSchedule clone,
                           PackageSchedule::FromProto(p.get(), proto));

  EXPECT_TRUE(clone.IsSynchronousSchedule());
  EXPECT_THAT(clone.GetSchedules().at(proc1).GetCycleMap(),
              Not(Contains(Key(literal1.node()))));
  EXPECT_THAT(clone.GetSchedules().at(proc2).GetCycleMap(),
              Not(Contains(Key(literal2.node()))));
  EXPECT_EQ(clone.GetSynchronousCycle(add1.node()), 0);
  EXPECT_EQ(clone.GetSynchronousCycle(add2.node()), 42);
}

TEST_P(PipelineScheduleTest, ProcWithExplicitStateAccess) {
  Package p(TestName());
  Type* u32 = p.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out_ch", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * se,
                           pb.UnreadStateElement("state", Value(UBits(0, 32)),
                                                 /*non_synthesizable=*/false));
  BValue current = pb.StateRead(se);
  BValue add_val = pb.Add(current, pb.Literal(UBits(1, 32)));

  pb.Send(out_ch, add_val);

  pb.Next(se, add_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  SchedulingOptions options = this->options();
  options.clock_period_ps(2);

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(), options));

  EXPECT_EQ(schedule.length(), 1);
}

TEST_P(PipelineScheduleTest, ProcWithMultipleStateReads) {
  Package p(TestName());
  Type* u32 = p.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out_ch", ChannelOps::kSendOnly, u32));

  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * se,
                           pb.UnreadStateElement("state", Value(UBits(0, 32)),
                                                 /*non_synthesizable=*/false));
  BValue read1 = pb.StateRead(se);

  // Create a second read with a predicate
  BValue cond = pb.Literal(UBits(1, 1));
  BValue read2 = pb.StateRead(se, cond);

  BValue add_val = pb.Add(read1, read2);
  pb.Send(out_ch, add_val);

  pb.Next(se, add_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  SchedulingOptions options = this->options();
  options.clock_period_ps(2);

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(), options));

  EXPECT_EQ(schedule.length(), 1);
}

TEST_P(PipelineScheduleErrorTest, ProcWithZeroReadsErrors) {
  Package p(TestName());
  TokenlessProcBuilder pb("the_proc", "tkn", &p);
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * se,
                           pb.UnreadStateElement("state", Value(UBits(0, 32)),
                                                 /*non_synthesizable=*/false));

  BValue add_val = pb.Literal(UBits(42, 32));
  pb.Next(se, add_val);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  SchedulingOptions options = this->options();
  options.clock_period_ps(2);

  EXPECT_THAT(RunPipelineSchedule(proc, TestDelayEstimator(), options),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("has no reads")));
}

INSTANTIATE_TEST_SUITE_P(
    PipelineScheduleTest, PipelineScheduleTest,
    testing::Values(
        Strategies{SchedulingStrategy::ASAP, SchedulingStrategy::ASAP},
        Strategies{SchedulingStrategy::ASAP, SchedulingStrategy::SDC},
        Strategies{SchedulingStrategy::SDC, SchedulingStrategy::ASAP},
        Strategies{SchedulingStrategy::SDC, SchedulingStrategy::SDC},
        // TODO(allight): Min cut doesn't respect all constraints yet.
        // Strategies{SchedulingStrategy::MIN_CUT, SchedulingStrategy::ASAP},
        // Strategies{SchedulingStrategy::MIN_CUT, SchedulingStrategy::SDC},
        Strategies{SchedulingStrategy::RANDOM, SchedulingStrategy::ASAP},
        Strategies{SchedulingStrategy::RANDOM, SchedulingStrategy::SDC}),
    testing::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(
    PipelineScheduleErrorTest, PipelineScheduleErrorTest,
    testing::Values(
        Strategies{SchedulingStrategy::ASAP, SchedulingStrategy::ASAP},
        Strategies{SchedulingStrategy::ASAP, SchedulingStrategy::SDC},
        Strategies{SchedulingStrategy::SDC, SchedulingStrategy::ASAP},
        Strategies{SchedulingStrategy::SDC, SchedulingStrategy::SDC}),
    testing::PrintToStringParamName());
// TODO(allight): Ideally this suite wouldn't need to exist and all error
// messages would be handled generically.
INSTANTIATE_TEST_SUITE_P(SdcOnlyPipelineScheduleErrorTest,
                         SdcOnlyPipelineScheduleErrorTest,
                         testing::Values(Strategies{SchedulingStrategy::SDC,
                                                    SchedulingStrategy::SDC}),
                         testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(AsapPipelineScheduleTest, AsapPipelineScheduleTest,
                         testing::Values(Strategies{SchedulingStrategy::ASAP,
                                                    SchedulingStrategy::ASAP}),
                         testing::PrintToStringParamName());

INSTANTIATE_TEST_SUITE_P(SdcPipelineScheduleTest, SdcPipelineScheduleTest,
                         testing::Values(Strategies{SchedulingStrategy::SDC,
                                                    SchedulingStrategy::SDC}),
                         testing::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(RandomPipelineScheduleTest, RandomPipelineScheduleTest,
                         testing::Values(Strategies{SchedulingStrategy::RANDOM,
                                                    SchedulingStrategy::ASAP},
                                         Strategies{SchedulingStrategy::RANDOM,
                                                    SchedulingStrategy::SDC}),
                         testing::PrintToStringParamName());
INSTANTIATE_TEST_SUITE_P(SdcPrimaryPipelineScheduleTest,
                         SdcPrimaryPipelineScheduleTest,
                         testing::Values(Strategies{SchedulingStrategy::SDC,
                                                    SchedulingStrategy::SDC},
                                         Strategies{SchedulingStrategy::SDC,
                                                    SchedulingStrategy::ASAP}),
                         testing::PrintToStringParamName());

TEST_F(PipelineScheduleTest,
       ProcFeedbackArcThroughputLabeledArcClampedByWorstCase) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("W1", "R1"));
  SchedulingOptions options;
  options.clock_period_ps(2);
  // Clamps to 2 (Next - Read <= 1)
  options.worst_case_throughput(2);
  options.arc_worst_case_throughput({{{"W1", "R1"}, 3}});

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));

  EXPECT_LE(
      schedule.cycle(setup.next.node()) - schedule.cycle(setup.read.node()), 1);
}

TEST_F(PipelineScheduleTest,
       ProcFeedbackArcThroughputDefaultClampedByWorstCase) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("W1", "R1"));
  SchedulingOptions options;
  options.clock_period_ps(2);
  // Clamps to 2 (Next - Read <= 1)
  options.worst_case_throughput(2);
  options.default_arc_worst_case_throughput(4);

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));

  EXPECT_LE(
      schedule.cycle(setup.next.node()) - schedule.cycle(setup.read.node()), 1);
}

TEST_F(PipelineScheduleTest,
       ProcFeedbackArcThroughputUsesDefaultWhenNoWorstCaseSpecified) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("W1", "R1"));
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.worst_case_throughput(0);
  // Fallback (Next - Read <= 1)
  options.default_arc_worst_case_throughput(2);

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));

  EXPECT_LE(
      schedule.cycle(setup.next.node()) - schedule.cycle(setup.read.node()), 1);
}

TEST_F(PipelineScheduleTest,
       ProcFeedbackArcThroughputUnlabeledWinsOverWildcardPattern) {
  XLS_ASSERT_OK_AND_ASSIGN(
      LabeledFeedbackArcProc setup,
      BuildLabeledFeedbackArcProc(std::nullopt, std::nullopt));
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.worst_case_throughput(0);
  // Unlabeled (2) wins over wildcard (5) due to specificity score
  options.arc_worst_case_throughput({{{"_", "_"}, 2}, {{"*", "*"}, 5}});

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));

  EXPECT_LE(
      schedule.cycle(setup.next.node()) - schedule.cycle(setup.read.node()), 1);
}

TEST_F(PipelineScheduleTest,
       ProcFeedbackArcThroughputSpecificWinsOverWorstCase) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("W1", "R1"));
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.worst_case_throughput(4);
  // Specific (2) wins over Worst Case (4)
  options.arc_worst_case_throughput({{{"W1", "R1"}, 2}});

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));

  EXPECT_LE(
      schedule.cycle(setup.next.node()) - schedule.cycle(setup.read.node()), 1);
}

TEST_F(PipelineScheduleTest, ProcFeedbackArcThroughputAmbiguousMatch) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("L_W", "L_R"));

  // Tie between (L_W, * = 4) and (*, L_R = 2). Both have score 2 but different
  // values. Should fail with ambiguous match error.
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.arc_worst_case_throughput({{{"L_W", "*"}, 4}, {{"*", "L_R"}, 2}});

  EXPECT_THAT(RunPipelineSchedule(setup.proc, TestDelayEstimator(), options),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Ambiguous throughput configuration")));
}

TEST_F(PipelineScheduleTest,
       ProcFeedbackArcThroughputAmbiguousMatchButSameValueSucceeds) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("L_W", "L_R"));
  // Overlap between ("L_W", "*") and ("*", "L_R"). Both have the same
  // specificity score (2) and the same throughput value (3).
  // This is a harmless tie and should succeed without errors.
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.arc_worst_case_throughput({{{"L_W", "*"}, 3}, {{"*", "L_R"}, 3}});
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));
  // Throughput limit 3 allows backedge length <= 2.
  EXPECT_LE(
      schedule.cycle(setup.next.node()) - schedule.cycle(setup.read.node()), 2);
}

TEST_F(PipelineScheduleTest,
       ProcFeedbackArcThroughputAmbiguousMatchButSpecificityWins) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("L_W", "L_R"));

  // Rules ("L_W", "*") and ("*", "L_R") overlap at ("L_W", "L_R").
  // Because they have the same specificity score (2) but different values (4
  // vs 3), they would normally conflict. However, the more specific rule
  // ("L_W", "L_R") (score 4, value 2) successfully masks/resolves the tie.
  // SDC check must pass, and the solver clamps to 2 (Next - Read <= 1).
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.arc_worst_case_throughput(
      {{{"L_W", "*"}, 4}, {{"*", "L_R"}, 3}, {{"L_W", "L_R"}, 2}});

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));

  EXPECT_LE(
      schedule.cycle(setup.next.node()) - schedule.cycle(setup.read.node()), 1);
}

TEST_F(PipelineScheduleTest,
       ProcFeedbackArcThroughputWorstCaseZeroNotEnforced) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("L_W", "L_R"));

  // worst_case = 0 (not enforced), specific = 2 (enforced).
  // Effective limit should be 2 (Next - Read <= 1).
  SchedulingOptions options;
  options.clock_period_ps(2);
  // Not enforced
  options.worst_case_throughput(0);
  options.arc_worst_case_throughput({{{"L_W", "L_R"}, 2}});

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));

  EXPECT_LE(
      schedule.cycle(setup.next.node()) - schedule.cycle(setup.read.node()), 1);
}

TEST_F(PipelineScheduleTest, ProcFeedbackArcThroughputUnusedPattern) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("W1", "R1"));
  SchedulingOptions options;
  options.clock_period_ps(2);
  // Typo in pattern: "W2" doesn't exist
  options.arc_worst_case_throughput({{{"W2", "R1"}, 2}});

  EXPECT_THAT(RunPipelineSchedule(setup.proc, TestDelayEstimator(), options),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "Throughput override pattern \"W2,R1\" did not match any "
                      "feedback arc in the package test_package")));
}

TEST_F(PipelineScheduleTest, VerifyConstraintsSucceedsWithUnlabeledArc) {
  XLS_ASSERT_OK_AND_ASSIGN(
      LabeledFeedbackArcProc setup,
      BuildLabeledFeedbackArcProc(std::nullopt, std::nullopt));
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.arc_worst_case_throughput({{{"_", "_"}, 2}});
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));
  XLS_EXPECT_OK(schedule.VerifyConstraints(options.constraints(), options));
}

TEST_F(PipelineScheduleTest, VerifyConstraintsSucceedsWithCustomArcThroughput) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("W1", "R1"));
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.arc_worst_case_throughput({{{"W1", "R1"}, 2}, {{"*", "*"}, 5}});

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));

  XLS_EXPECT_OK(schedule.VerifyConstraints(options.constraints(), options));
}

TEST_F(PipelineScheduleTest, ProcFeedbackArcThroughputMultiProc) {
  auto package = CreatePackage();

  // Proc 1: has ("W1", "R1")
  ProcBuilder pb1("proc_1", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto se1,
                           pb1.UnreadStateElement("st1", Value(UBits(0, 32)),
                                                  /*non_synthesizable=*/false));
  BValue read1 = pb1.StateRead(se1, /*predicate=*/std::nullopt, "R1");
  BValue add1 = pb1.Add(read1, pb1.Literal(UBits(1, 32)));
  pb1.Next(se1, add1, /*pred=*/std::nullopt, "W1");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, pb1.Build());

  // Proc 2: has ("W2", "R2")
  ProcBuilder pb2("proc_2", package.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto se2,
                           pb2.UnreadStateElement("st2", Value(UBits(0, 32)),
                                                  /*non_synthesizable=*/false));
  BValue read2 = pb2.StateRead(se2, /*predicate=*/std::nullopt, "R2");
  BValue add2 = pb2.Add(read2, pb2.Literal(UBits(1, 32)));
  pb2.Next(se2, add2, /*pred=*/std::nullopt, "W2");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc2, pb2.Build());

  // Pattern ("W1", "*") matches in proc_1.
  // Pattern ("W2", "*") matches in proc_2.
  // Pattern ("*", "R1") matches in proc_1 (ambiguous pattern but throughput
  // is the same).
  // Pattern ("*", "*") matches in both procs but should be masked by more
  // specific rules
  // In package-wide checking, all of these are used across the package.
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.arc_worst_case_throughput(
      {{{"W1", "*"}, 2}, {{"W2", "*"}, 2}, {{"*", "R1"}, 2}, {{"*", "*"}, 4}});

  // Scheduling both should succeed because package-wide validation passes
  XLS_EXPECT_OK(RunPipelineSchedule(proc1, TestDelayEstimator(), options));
  XLS_EXPECT_OK(RunPipelineSchedule(proc2, TestDelayEstimator(), options));
}

TEST_F(PipelineScheduleTest, VerifyConstraintsFailsWithViolatedArcThroughput) {
  XLS_ASSERT_OK_AND_ASSIGN(LabeledFeedbackArcProc setup,
                           BuildLabeledFeedbackArcProc("W1", "R1"));
  SchedulingOptions options;
  options.clock_period_ps(2);
  options.arc_worst_case_throughput({{{"W1", "R1"}, 2}});
  // Force read to cycle 0 and next to cycle 1 to ensure backedge length is
  // exactly 1.
  options.add_constraint(NodeInCycleConstraint(setup.read.node(), 0));
  options.add_constraint(NodeInCycleConstraint(setup.next.node(), 1));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(setup.proc, TestDelayEstimator(), options));

  // Create a tighter constraint manually that is violated by the schedule
  SchedulingOptions tighter_options;
  tighter_options.clock_period_ps(2);
  tighter_options.arc_worst_case_throughput({{{"W1", "R1"}, 1}});

  EXPECT_THAT(schedule.VerifyConstraints(tighter_options.constraints(),
                                         tighter_options),
              absl_testing::StatusIs(
                  absl::StatusCode::kResourceExhausted,
                  testing::HasSubstr("Scheduling constraint violated")));
}

TEST_F(PipelineScheduleTest, ProcStateReadBeforeWriteSucceeds) {
  Package p(TestName());
  Type* u32 = p.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out_ch", ChannelOps::kSendOnly, u32));

  ProcBuilder pb("the_proc", &p);
  BValue tkn = pb.Literal(Value::Token());
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * se,
                           pb.UnreadStateElement("state", Value(UBits(0, 32)),
                                                 /*non_synthesizable=*/false));

  // Read state value, add logic to force it late in the pipeline
  BValue read = pb.StateRead(se);
  BValue neg1 = pb.Negate(read);
  BValue neg2 = pb.Negate(neg1);
  BValue neg3 = pb.Negate(neg2);
  pb.Send(out_ch, tkn, neg3);

  // Next-state update is just a constant, completely independent!
  // This would normally be scheduled at cycle 0 if not constrained.
  BValue next = pb.Next(se, pb.Literal(UBits(42, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  // Configure SDC with NodeInCycleConstraint to push StateRead to stage 2.
  // SDC must be forced to schedule Next at stage >= 2 to satisfy
  // read-before-write constraint.
  SchedulingOptions options;
  options.clock_period_ps(1);
  options.pipeline_stages(5);
  options.add_constraint(NodeInCycleConstraint(read.node(), 2));

  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(), options));

  EXPECT_GE(schedule.cycle(next.node()), schedule.cycle(read.node()));
}

TEST_F(PipelineScheduleTest,
       ProcStateReadBeforeWriteConstraintMutualExclusion) {
  Package p(TestName());
  Type* u1 = p.GetBitsType(1);
  Type* u32 = p.GetBitsType(32);
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * in_ch,
      p.CreateStreamingChannel("in_ch", ChannelOps::kReceiveOnly, u1));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * out_ch,
      p.CreateStreamingChannel("out_ch", ChannelOps::kSendOnly, u32));
  ProcBuilder pb("the_proc", &p);
  BValue tkn = pb.Literal(Value::Token());
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * se,
                           pb.UnreadStateElement("state", Value(UBits(0, 32)),
                                                 /*non_synthesizable=*/false));
  BValue rcv = pb.Receive(in_ch, tkn);
  BValue rcv_tkn = pb.TupleIndex(rcv, 0);
  BValue cond = pb.TupleIndex(rcv, 1);
  BValue cond_not = pb.Not(cond);
  // StateRead triggers only when cond is TRUE
  BValue read_mutually_exclusive = pb.StateRead(se, cond);
  pb.Send(out_ch, rcv_tkn, read_mutually_exclusive);

  // ADDED: StateRead triggers when cond is FALSE (cond_not is TRUE)
  // This satisfies the "read before write" check for the write path.
  BValue read_for_write = pb.StateRead(se, cond_not);
  // Next-state update triggers only when cond is FALSE (cond_not is TRUE)
  BValue next = pb.Next(se, pb.Literal(UBits(42, 32)), cond_not);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  // Constraints:
  // - Force the main read (when cond is TRUE) to stage 2.
  // - Force the dummy read (when cond is FALSE) to stage 0.
  // - Force the write (when cond is FALSE) to stage 1.
  SchedulingOptions options;
  options.clock_period_ps(1);
  options.pipeline_stages(5);
  options.worst_case_throughput(2);
  options.add_constraint(
      NodeInCycleConstraint(read_mutually_exclusive.node(), 2));
  options.add_constraint(NodeInCycleConstraint(read_for_write.node(), 0));
  options.add_constraint(NodeInCycleConstraint(next.node(), 1));
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(), options));
  // Verify schedule:
  // - next (1) is scheduled after read_for_write (0) -> OK (same activation)
  // - next (1) is scheduled before read (2) -> OK (mutually exclusive)
  EXPECT_EQ(schedule.cycle(read_for_write.node()), 0);
  EXPECT_EQ(schedule.cycle(next.node()), 1);
  EXPECT_EQ(schedule.cycle(read_mutually_exclusive.node()), 2);
}

TEST_F(PipelineScheduleTest, ProcWriteBeforeReadFailsVerification) {
  Package p(TestName());
  ProcBuilder pb("the_proc", &p);
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * se,
                           pb.UnreadStateElement("state", Value(UBits(0, 32)),
                                                 /*non_synthesizable=*/false));
  BValue read = pb.StateRead(se);
  BValue next = pb.Next(se, pb.Literal(UBits(42, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  // Run scheduler to get a valid base schedule first.
  SchedulingOptions options;
  options.clock_period_ps(1);
  options.pipeline_stages(3);
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule valid_schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(), options));

  // Modify the valid cycle map to violate sequential safety.
  ScheduleCycleMap cycle_map = valid_schedule.GetCycleMap();
  cycle_map[read.node()] = 2;
  cycle_map[next.node()] = 0;

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           PipelineSchedule::Create(proc, cycle_map));

  // Verification should fail because next (0) is scheduled before read (2).
  EXPECT_THAT(schedule.Verify(),
              StatusIs(absl::StatusCode::kInternal,
                       testing::HasSubstr("scheduled before state read")));
}

TEST_F(PipelineScheduleTest, ProcFeedbackArcTooLongFailsVerification) {
  Package p(TestName());
  ProcBuilder pb("the_proc", &p);
  XLS_ASSERT_OK_AND_ASSIGN(StateElement * se,
                           pb.UnreadStateElement("state", Value(UBits(0, 32)),
                                                 /*non_synthesizable=*/false));
  pb.StateRead(se);
  BValue next = pb.Next(se, pb.Literal(UBits(42, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  // Run scheduler to get a valid base schedule first.
  SchedulingOptions options;
  options.clock_period_ps(1);
  options.pipeline_stages(3);
  XLS_ASSERT_OK_AND_ASSIGN(
      PipelineSchedule valid_schedule,
      RunPipelineSchedule(proc, TestDelayEstimator(), options));

  // Set worst-case throughput to 2.
  // So we need: cycle(next) - cycle(read) < 2.
  proc->SetInitiationInterval(2);

  // Modify the valid cycle map.
  ScheduleCycleMap cycle_map = valid_schedule.GetCycleMap();
  cycle_map[next.node()] = 2;

  XLS_ASSERT_OK_AND_ASSIGN(PipelineSchedule schedule,
                           PipelineSchedule::Create(proc, cycle_map));

  // Verification should fail because next (2) - read (0) = 2
  EXPECT_THAT(schedule.Verify(),
              StatusIs(absl::StatusCode::kInternal,
                       testing::HasSubstr("scheduled too late after")));
}
}  // namespace
}  // namespace xls
