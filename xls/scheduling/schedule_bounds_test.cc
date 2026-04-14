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
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/estimators/delay_model/delay_estimator.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_annotator.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/op.h"
#include "xls/ir/source_location.h"
#include "xls/ir/value.h"
#include "xls/scheduling/schedule_graph.h"

namespace m = xls::op_matchers;
namespace xls {
namespace sched {
namespace {

using absl_testing::StatusIs;
using testing::ContainsRegex;
using ::testing::Pair;

using NodeDifferenceConstraint =
    ScheduleBounds::NodeSchedulingConstraint::NodeDifferenceConstraint;
using LastStageConstraint =
    ScheduleBounds::NodeSchedulingConstraint::LastStageConstraint;

class TestDelayEstimator : public DelayEstimator {
 public:
  TestDelayEstimator() : DelayEstimator("test") {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    switch (node->op()) {
      case Op::kParam:
      case Op::kStateRead:
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

class ControllableDelayEstimator : public DelayEstimator {
 public:
  ControllableDelayEstimator() : DelayEstimator("test") {}

  absl::StatusOr<int64_t> GetOperationDelayInPs(Node* node) const override {
    if (delays_.contains(node)) {
      return delays_.at(node);
    }
    testing::StringMatchResultListener listener;
    for (const auto& [matcher, delay] : match_delay_) {
      if (matcher.MatchAndExplain(node, &listener)) {
        return delay;
      }
      *listener.stream() << "\n\n";
    }
    return absl::InvalidArgumentError(
        absl::StrFormat("No delay set for node: %s. Could not use any of: %s",
                        node->GetName(), listener.str()));
  }

  void ClearMatchers() { match_delay_.clear(); }
  // Set delay for this exact node to the given value. These are checked before
  // any matchers are checked.
  ControllableDelayEstimator& SetDelay(Node* node, int64_t delay) {
    delays_[node] = delay;
    return *this;
  }
  // Set delay for this exact node to the given value. These are checked before
  // any matchers are checked.
  ControllableDelayEstimator& SetDelay(const BValue& val, int64_t delay) {
    delays_[val.node()] = delay;
    return *this;
  }
  // Set delay to 'delay' for any node which matches. The matchers are checked
  // in the order they're added. These are checked only if there isn't an exact
  // node match.
  ControllableDelayEstimator& SetDelay(testing::Matcher<const Node*> matcher,
                                       int64_t delay) {
    match_delay_.push_back({matcher, delay});
    return *this;
  }

  // Set delay to 'delay' for any node which matches the given operation. The
  // matchers are checked in the order they're added. These are checked only if
  // there isn't an exact node match.
  ControllableDelayEstimator& SetDelay(Op op, int64_t delay) {
    return SetDelay(testing::Property("op", &Node::op, testing::Eq(op)), delay);
  }

 private:
  absl::flat_hash_map<Node*, int64_t> delays_;
  std::vector<std::pair<testing::Matcher<const Node*>, int64_t>> match_delay_;
};

class ScheduleBoundsTest : public IrTestBase {
 protected:
  TestDelayEstimator unit_delay_estimator_;

  absl::StatusOr<ScheduleBounds> CreateBasic(
      Function* f, int64_t clock_period_ps,
      const DelayEstimator& delay_estimator) {
    XLS_ASSIGN_OR_RETURN(ScheduleGraph schedule_graph,
                         ScheduleGraph::Create(f, /*dead_after_synthesis=*/{}));
    XLS_ASSIGN_OR_RETURN(ScheduleBounds bounds,
                         ScheduleBounds::Create(schedule_graph, clock_period_ps,
                                                delay_estimator));
    return bounds;
  }

  absl::StatusOr<ScheduleBounds> ComputeAsapAndAlapBoundsDirect(
      FunctionBase* f, int64_t clock_period_ps, const DelayEstimator& delay,
      absl::Span<
          ScheduleBounds::NodeSchedulingConstraint::InnerConstraints const>
          constraints) {
    std::vector<ScheduleBounds::NodeSchedulingConstraint> node_constraints;
    for (const auto& constraint : constraints) {
      node_constraints.emplace_back(constraint);
    }
    return ScheduleBounds::ComputeAsapAndAlapBounds(f, clock_period_ps, delay,
                                                    node_constraints);
  }
};

// Helper that matches any generally free operation.
auto FreeOperations() {
  return testing::AnyOf(m::Param(), m::Literal(), m::Send(), m::Receive(),
                        m::Literal(), m::Next(), m::ZeroExt(), m::SignExt(),
                        m::StateRead(), m::BitSlice(), m::TupleIndex(),
                        m::Tuple(), m::Array(), m::Concat());
}

MATCHER_P2(Bounds, lb, ub,
           absl::StrFormat("Lower bound is %s %s upper bound is %s",
                           testing::DescribeMatcher<int64_t>(lb, negation),
                           negation ? "or" : "and",
                           testing::DescribeMatcher<int64_t>(ub, negation))) {
  const auto& [low, high] = arg;
  return testing::ExplainMatchResult(lb, low, result_listener) &&
         testing::ExplainMatchResult(ub, high, result_listener);
}

TEST_F(ScheduleBoundsTest, AsapBasic) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Literal(UBits(12, 32), SourceInfo(), "d");
  auto e = fb.Add(a, b, SourceInfo(), "e");
  auto g = fb.UMul(c, d, SourceInfo(), "g");
  auto h = fb.Add(e, g, SourceInfo(), "h");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ControllableDelayEstimator delay;
  delay.SetDelay(m::Add(), 1);
  delay.SetDelay(m::UMul(), 3);
  delay.SetDelay(FreeOperations(), 0);

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleBounds sched,
      ScheduleBounds::ComputeAsapAndAlapBounds(f,
                                               /*clock_period_ps=*/3, delay));
  XLS_ASSERT_OK_AND_ASSIGN(auto delay_anno, DelayAnnotator::Create(f, delay));
  RecordProperty("ir",
                 f->DumpIr(IrAnnotatorJoiner(std::move(delay_anno),
                                             ScheduleBoundsAnnotator(sched))));
  EXPECT_THAT(sched.bounds(a.node()), Bounds(0, 1));
  EXPECT_THAT(sched.bounds(b.node()), Bounds(0, 1));
  EXPECT_THAT(sched.bounds(c.node()), Bounds(0, 0));
  EXPECT_THAT(sched.bounds(d.node()), Bounds(0, 0));
  EXPECT_THAT(sched.bounds(e.node()), Bounds(0, 1));
  EXPECT_THAT(sched.bounds(g.node()), Bounds(0, 0));
  EXPECT_THAT(sched.bounds(h.node()), Bounds(1, 1));
}

TEST_F(ScheduleBoundsTest, AsapInCycleConstraints) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Literal(UBits(12, 32), SourceInfo(), "d");
  auto e = fb.Add(a, b, SourceInfo(), "e");
  auto g = fb.UMul(c, d, SourceInfo(), "g");
  auto h = fb.Add(e, g, SourceInfo(), "h");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ControllableDelayEstimator delay;
  delay.SetDelay(m::Add(), 1);
  delay.SetDelay(h, 2);
  delay.SetDelay(m::UMul(), 3);
  delay.SetDelay(FreeOperations(), 0);

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleBounds sched,
      ScheduleBounds::ComputeAsapAndAlapBounds(
          f,
          /*clock_period_ps=*/3, delay, /*ii=*/1, /*constraints=*/
          {NodeInCycleConstraint(e.node(), 2),
           NodeInCycleConstraint(g.node(), 3)}));
  XLS_ASSERT_OK_AND_ASSIGN(auto delay_anno, DelayAnnotator::Create(f, delay));
  RecordProperty("ir",
                 f->DumpIr(IrAnnotatorJoiner(std::move(delay_anno),
                                             ScheduleBoundsAnnotator(sched))));
  EXPECT_THAT(sched.bounds(a.node()), Bounds(0, 2));
  EXPECT_THAT(sched.bounds(b.node()), Bounds(0, 2));
  EXPECT_THAT(sched.bounds(c.node()), Bounds(0, 3));
  EXPECT_THAT(sched.bounds(d.node()), Bounds(0, 3));
  EXPECT_THAT(sched.bounds(e.node()), Bounds(2, 2));
  EXPECT_THAT(sched.bounds(g.node()), Bounds(3, 3));
  EXPECT_THAT(sched.bounds(h.node()), Bounds(4, 4));
}

TEST_F(ScheduleBoundsTest, BinPacking) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto not_a = fb.Not(a, SourceInfo(), "not_a");
  auto mul = fb.UMul(not_a, not_a, SourceInfo(), "mul_nd");
  auto res = fb.Not(mul, SourceInfo(), "res");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ControllableDelayEstimator delay;
  delay.SetDelay(m::UMul(), 2);
  delay.SetDelay(m::Not(), 1);
  delay.SetDelay(FreeOperations(), 0);
  XLS_ASSERT_OK_AND_ASSIGN(auto bounds,
                           ComputeAsapAndAlapBoundsDirect(
                               f,
                               /*clock_period_ps=*/2, delay, /*constraints=*/
                               {LastStageConstraint(res.node())}));
  XLS_ASSERT_OK_AND_ASSIGN(auto delay_anno, DelayAnnotator::Create(f, delay));
  RecordProperty("ir",
                 f->DumpIr(IrAnnotatorJoiner(std::move(delay_anno),
                                             ScheduleBoundsAnnotator(bounds))));
  EXPECT_THAT(bounds.bounds(a.node()), Bounds(0, 0));
  EXPECT_THAT(bounds.bounds(not_a.node()), Bounds(0, 0));
  EXPECT_THAT(bounds.bounds(mul.node()), Bounds(1, 1));
  EXPECT_THAT(bounds.bounds(res.node()), Bounds(2, 2));
}

// 2 nodes dependent must both be in same cycle but too long.
TEST_F(ScheduleBoundsTest, FailToScheduleLastCycleConstraint) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Add(a, b, SourceInfo(), "d");
  fb.Add(d, c, SourceInfo(), "e");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ControllableDelayEstimator delay;
  delay.SetDelay(m::Add(), 1);
  delay.SetDelay(FreeOperations(), 0);
  EXPECT_THAT(
      ComputeAsapAndAlapBoundsDirect(f,
                                     /*clock_period_ps=*/1,
                                     delay, /*constraints=*/
                                     {LastStageConstraint(d.node())}),
      StatusIs(absl::StatusCode::kInternal,
               ContainsRegex(".*Lower bound \\([0-9]+\\) of .* incompatible "
                             "with constraint .*:last_stage due to being "
                             "greater than max lower bound of [0-9]+.*")));
}

// 2 nodes dependent must both be in same cycle but too long.
TEST_F(ScheduleBoundsTest, FailToScheduleInCycleConstraints) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Add(a, b, SourceInfo(), "d");
  auto e = fb.Add(d, c, SourceInfo(), "e");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ControllableDelayEstimator delay;
  delay.SetDelay(m::Add(), 2);
  delay.SetDelay(FreeOperations(), 0);
  EXPECT_THAT(ComputeAsapAndAlapBoundsDirect(
                  f,
                  /*clock_period_ps=*/3, delay, /*constraints=*/
                  {NodeInCycleConstraint(d.node(), 2),
                   NodeInCycleConstraint(e.node(), 2)}),
              StatusIs(absl::StatusCode::kInternal,
                       ContainsRegex(
                           ".*Constraint [de]@2 is not compatible with .* "
                           "due to delay requiring a later cycle \\(3\\).*")));
}

// 2 nodes dependent must both be in same cycle but too long.
TEST_F(ScheduleBoundsTest, FailToScheduleConstraintLoop) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Param("d", p->GetBitsType(32));
  auto e = fb.Add(a, b, SourceInfo(), "e");
  auto g = fb.Add(c, d, SourceInfo(), "g");
  auto h = fb.Add(e, g, SourceInfo(), "h");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ControllableDelayEstimator delay;
  delay.SetDelay(m::Add(), 2);
  delay.SetDelay(FreeOperations(), 0);
  EXPECT_THAT(ComputeAsapAndAlapBoundsDirect(
                  f,
                  /*clock_period_ps=*/3, delay, /*constraints=*/
                  {
                      NodeDifferenceConstraint{.anchor = g.node(),
                                               .subject = h.node(),
                                               .min_after = 1,
                                               .max_after = 1},
                      NodeDifferenceConstraint{.anchor = e.node(),
                                               .subject = g.node(),
                                               .min_after = 1,
                                               .max_after = 1},
                      NodeDifferenceConstraint{.anchor = e.node(),
                                               .subject = h.node(),
                                               .min_after = 1,
                                               .max_after = 1},
                  }),
              StatusIs(absl::StatusCode::kInternal,
                       ContainsRegex(".*failed to converge.*potentially "
                                     "incompatible with constraint.* ")));
}

TEST_F(ScheduleBoundsTest, FailToSchedulePastMaxDifference) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a0 = fb.Param("a0", p->GetBitsType(32));
  auto a1 = fb.Not(a0, SourceInfo(), "a1");
  auto a2 = fb.Not(a1, SourceInfo(), "a2");
  auto a3 = fb.Not(a2, SourceInfo(), "a3");
  auto a4 = fb.Not(a3, SourceInfo(), "a4");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ControllableDelayEstimator delay;
  delay.SetDelay(m::Not(), 1);
  delay.SetDelay(FreeOperations(), 0);
  EXPECT_THAT(ComputeAsapAndAlapBoundsDirect(
                  f,
                  /*clock_period_ps=*/1, delay, /*constraints=*/
                  {NodeDifferenceConstraint{.anchor = a1.node(),
                                            .subject = a4.node(),
                                            .min_after = 0,
                                            .max_after = 2}}),
              StatusIs(absl::StatusCode::kInternal,
                       ContainsRegex(".*failed to converge.*")));
}

// node is max 2 cycles after another node and forced to be in exactly stage 4.
TEST_F(ScheduleBoundsTest, PullNodeLater) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Add(a, b, SourceInfo(), "d");
  auto e = fb.Add(d, c, SourceInfo(), "e");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ControllableDelayEstimator delay;
  delay.SetDelay(m::Add(), 2);
  delay.SetDelay(FreeOperations(), 0);
  XLS_ASSERT_OK_AND_ASSIGN(ScheduleBounds sched,
                           ComputeAsapAndAlapBoundsDirect(
                               f,
                               /*clock_period_ps=*/3, delay, /*constraints=*/
                               {NodeDifferenceConstraint{.anchor = d.node(),
                                                         .subject = e.node(),
                                                         .min_after = 0,
                                                         .max_after = 2},
                                NodeInCycleConstraint(e.node(), 4)}));
  XLS_ASSERT_OK_AND_ASSIGN(auto delay_anno, DelayAnnotator::Create(f, delay));
  RecordProperty("ir",
                 f->DumpIr(IrAnnotatorJoiner(std::move(delay_anno),
                                             ScheduleBoundsAnnotator(sched))));
  EXPECT_THAT(sched.bounds(a.node()), Bounds(0, 3));
  EXPECT_THAT(sched.bounds(b.node()), Bounds(0, 3));
  EXPECT_THAT(sched.bounds(c.node()), Bounds(0, 4));
  EXPECT_THAT(sched.bounds(d.node()), Bounds(2, 3));
  EXPECT_THAT(sched.bounds(e.node()), Bounds(4, 4));
}

TEST_F(ScheduleBoundsTest, PullNodeLaterChain) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Param("d", p->GetBitsType(32));
  auto e = fb.Add(a, b, SourceInfo(), "e");
  auto g = fb.Add(e, c, SourceInfo(), "g");
  auto h = fb.Add(g, d, SourceInfo(), "h");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ControllableDelayEstimator delay;
  delay.SetDelay(m::Add(), 2);
  delay.SetDelay(FreeOperations(), 0);
  XLS_ASSERT_OK_AND_ASSIGN(ScheduleBounds sched,
                           ComputeAsapAndAlapBoundsDirect(
                               f,
                               /*clock_period_ps=*/3, delay, /*constraints=*/
                               {NodeDifferenceConstraint{.anchor = e.node(),
                                                         .subject = g.node(),
                                                         .min_after = 2,
                                                         .max_after = 2},
                                NodeDifferenceConstraint{.anchor = g.node(),
                                                         .subject = h.node(),
                                                         .min_after = 2,
                                                         .max_after = 2},
                                NodeInCycleConstraint(h.node(), 8)}));
  XLS_ASSERT_OK_AND_ASSIGN(auto delay_anno, DelayAnnotator::Create(f, delay));
  RecordProperty("ir",
                 f->DumpIr(IrAnnotatorJoiner(std::move(delay_anno),
                                             ScheduleBoundsAnnotator(sched))));
  EXPECT_THAT(sched.bounds(a.node()), Bounds(0, 4));
  EXPECT_THAT(sched.bounds(b.node()), Bounds(0, 4));
  EXPECT_THAT(sched.bounds(c.node()), Bounds(0, 6));
  EXPECT_THAT(sched.bounds(d.node()), Bounds(0, 8));
  EXPECT_THAT(sched.bounds(e.node()), Bounds(4, 4));
  EXPECT_THAT(sched.bounds(g.node()), Bounds(6, 6));
  EXPECT_THAT(sched.bounds(h.node()), Bounds(8, 8));
}
TEST_F(ScheduleBoundsTest, PushNodeLaterChain) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto c = fb.Param("c", p->GetBitsType(32));
  auto d = fb.Param("d", p->GetBitsType(32));
  auto e = fb.Add(a, b, SourceInfo(), "e");
  auto g = fb.Add(e, c, SourceInfo(), "g");
  auto h = fb.Add(g, d, SourceInfo(), "h");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  ControllableDelayEstimator delay;
  delay.SetDelay(m::Add(), 2);
  delay.SetDelay(FreeOperations(), 0);
  XLS_ASSERT_OK_AND_ASSIGN(ScheduleBounds sched,
                           ComputeAsapAndAlapBoundsDirect(
                               f,
                               /*clock_period_ps=*/3, delay, /*constraints=*/
                               {NodeDifferenceConstraint{.anchor = e.node(),
                                                         .subject = g.node(),
                                                         .min_after = 2,
                                                         .max_after = 2},
                                NodeDifferenceConstraint{.anchor = g.node(),
                                                         .subject = h.node(),
                                                         .min_after = 2,
                                                         .max_after = 2},
                                NodeInCycleConstraint(b.node(), 3)}));
  XLS_ASSERT_OK_AND_ASSIGN(auto delay_anno, DelayAnnotator::Create(f, delay));
  RecordProperty("ir",
                 f->DumpIr(IrAnnotatorJoiner(std::move(delay_anno),
                                             ScheduleBoundsAnnotator(sched))));
  EXPECT_THAT(sched.bounds(a.node()), Bounds(0, 3));
  EXPECT_THAT(sched.bounds(b.node()), Bounds(3, 3));
  EXPECT_THAT(sched.bounds(c.node()), Bounds(0, 5));
  EXPECT_THAT(sched.bounds(d.node()), Bounds(0, 7));
  EXPECT_THAT(sched.bounds(e.node()), Bounds(3, 3));
  EXPECT_THAT(sched.bounds(g.node()), Bounds(5, 5));
  EXPECT_THAT(sched.bounds(h.node()), Bounds(7, 7));
}

TEST_F(ScheduleBoundsTest, StringifyConstraints) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  XLS_ASSERT_OK(fb.Build().status());

  using LSC = ScheduleBounds::NodeSchedulingConstraint::LastStageConstraint;
  using NDC =
      ScheduleBounds::NodeSchedulingConstraint::NodeDifferenceConstraint;
  using NSC = ScheduleBounds::NodeSchedulingConstraint;

  LSC lsc{.node = a.node()};
  EXPECT_EQ(absl::StrFormat("%v", lsc), "a:last_stage");

  NDC ndc{
      .anchor = a.node(), .subject = b.node(), .min_after = 2, .max_after = 5};
  EXPECT_EQ(absl::StrFormat("%v", ndc), "b - a ∈ [2, 5]");

  NodeInCycleConstraint nicc(a.node(), 42);
  EXPECT_EQ(absl::StrFormat("%v", nicc), "a@42");

  EXPECT_EQ(absl::StrFormat("%v", NSC(lsc)), "a:last_stage");
  EXPECT_EQ(absl::StrFormat("%v", NSC(ndc)), "b - a ∈ [2, 5]");
  EXPECT_EQ(absl::StrFormat("%v", NSC(nicc)), "a@42");
}

TEST_F(ScheduleBoundsTest, ParameterOnlyFunction) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(ScheduleBounds bounds,
                           ScheduleBounds::ComputeAsapAndAlapBounds(
                               f,
                               /*clock_period_ps=*/1, unit_delay_estimator_));
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
                               /*clock_period_ps=*/1, unit_delay_estimator_));
  EXPECT_THAT(bounds.bounds(x.node()), Pair(0, 0));
  EXPECT_THAT(bounds.bounds(y.node()), Pair(0, 0));
  EXPECT_THAT(bounds.bounds(not_x.node()), Pair(0, 1));
  EXPECT_THAT(bounds.bounds(x_plus_y.node()), Pair(0, 0));
  EXPECT_THAT(bounds.bounds(not_x_plus_y.node()), Pair(1, 1));
  EXPECT_THAT(bounds.bounds(result.node()), Pair(2, 2));
}

TEST_F(ScheduleBoundsTest, SimpleExpressionAndAssertAsapAndAlap) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto x = fb.Param("x", p->GetBitsType(32));
  auto y = fb.Param("y", p->GetBitsType(32));
  auto not_x = fb.Not(x);
  auto x_plus_y = fb.Add(x, y);
  auto not_x_plus_y = fb.Not(x_plus_y);
  auto result = fb.Add(not_x, not_x_plus_y);
  auto assert = fb.Assert(fb.Literal(Value::Token()), fb.UGe(result, y),
                          "expensive assert");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(result));

  XLS_ASSERT_OK_AND_ASSIGN(ScheduleBounds bounds,
                           ScheduleBounds::ComputeAsapAndAlapBounds(
                               f,
                               /*clock_period_ps=*/1, unit_delay_estimator_));
  EXPECT_THAT(bounds.bounds(x.node()), Pair(0, 0));
  EXPECT_THAT(bounds.bounds(y.node()), Pair(0, 0));
  EXPECT_THAT(bounds.bounds(not_x.node()), Pair(0, 1));
  EXPECT_THAT(bounds.bounds(x_plus_y.node()), Pair(0, 0));
  EXPECT_THAT(bounds.bounds(not_x_plus_y.node()), Pair(1, 1));
  EXPECT_THAT(bounds.bounds(result.node()), Pair(2, 2));
  EXPECT_THAT(bounds.bounds(assert.node()), Pair(2, 2));
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

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleBounds bounds,
      CreateBasic(f, /*clock_period_ps=*/1, unit_delay_estimator_));
  EXPECT_EQ(bounds.lb(x.node()), 0);
  EXPECT_EQ(bounds.lb(y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x.node()), 0);
  EXPECT_EQ(bounds.lb(x_plus_y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x_plus_y.node()), 0);
  EXPECT_EQ(bounds.lb(result.node()), 0);

  const int64_t kMax = ScheduleBounds::kDefaultMaxUpperBound;
  EXPECT_EQ(bounds.ub(x.node()), kMax);
  EXPECT_EQ(bounds.ub(y.node()), kMax);
  EXPECT_EQ(bounds.ub(not_x.node()), kMax);
  EXPECT_EQ(bounds.ub(x_plus_y.node()), kMax);
  EXPECT_EQ(bounds.ub(not_x_plus_y.node()), kMax);
  EXPECT_EQ(bounds.ub(result.node()), kMax);

  XLS_ASSERT_OK(bounds.PropagateBounds());

  // The initial call to PropagateLowerBounds should make all the lower bounds
  // satisfy dependency and clock period constraints.
  EXPECT_EQ(bounds.lb(x.node()), 0);
  EXPECT_EQ(bounds.lb(y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x.node()), 0);
  EXPECT_EQ(bounds.lb(x_plus_y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x_plus_y.node()), 1);
  EXPECT_EQ(bounds.lb(result.node()), 2);

  // Upper bounds should be updated too.
  EXPECT_EQ(bounds.ub(x.node()), kMax - 2);
  EXPECT_EQ(bounds.ub(y.node()), kMax - 2);
  EXPECT_EQ(bounds.ub(not_x.node()), kMax - 1);
  EXPECT_EQ(bounds.ub(x_plus_y.node()), kMax - 2);
  EXPECT_EQ(bounds.ub(not_x_plus_y.node()), kMax - 1);
  EXPECT_EQ(bounds.ub(result.node()), kMax);

  // Tightening the ub of one node in the graph and propagating should tighten
  // the upper bounds of the predecessors of that node.
  XLS_ASSERT_OK(bounds.TightenNodeUb(not_x_plus_y.node(), 42));
  XLS_ASSERT_OK(bounds.PropagateBounds());

  EXPECT_EQ(bounds.ub(not_x.node()), kMax - 1);
  EXPECT_EQ(bounds.ub(x_plus_y.node()), 41);
  EXPECT_EQ(bounds.ub(not_x_plus_y.node()), 42);
  EXPECT_EQ(bounds.ub(result.node()), kMax);

  // Tightening the ub of the root node should give every node a non-INT_MAX
  // upper bound.
  XLS_ASSERT_OK(bounds.TightenNodeUb(result.node(), 100));
  XLS_ASSERT_OK(bounds.PropagateBounds());

  EXPECT_EQ(bounds.ub(not_x.node()), 99);
  EXPECT_EQ(bounds.ub(x_plus_y.node()), 41);
  EXPECT_EQ(bounds.ub(not_x_plus_y.node()), 42);
  EXPECT_EQ(bounds.ub(result.node()), 100);

  // Setting one node's lb and propagating should result in further tightening.
  XLS_ASSERT_OK(bounds.TightenNodeLb(not_x.node(), 22));
  XLS_ASSERT_OK(bounds.PropagateBounds());

  EXPECT_EQ(bounds.lb(x.node()), 0);
  EXPECT_EQ(bounds.lb(y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x.node()), 22);
  EXPECT_EQ(bounds.lb(x_plus_y.node()), 0);
  EXPECT_EQ(bounds.lb(not_x_plus_y.node()), 1);
  EXPECT_EQ(bounds.lb(result.node()), 23);
}

TEST_F(ScheduleBoundsTest, ConvertNodeInCycleConstraint) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  auto add = fb.Add(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(f, /*dead_after_synthesis=*/{}));
  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ScheduleBounds::NodeSchedulingConstraint> converted,
      ScheduleBounds::ConvertSchedulingConstraints(
          graph, {NodeInCycleConstraint(add.node(), 5)}, /*ii=*/std::nullopt,
          /*max_upper_bound=*/100));

  ASSERT_EQ(converted.size(), 1);
  EXPECT_TRUE(converted[0].Is<NodeInCycleConstraint>());
  EXPECT_EQ(converted[0].As<NodeInCycleConstraint>().GetNode(), add.node());
  EXPECT_EQ(converted[0].As<NodeInCycleConstraint>().GetCycle(), 5);
}

TEST_F(ScheduleBoundsTest, ConvertIOConstraint) {
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
  IOConstraint io_const("a", IODirection::kReceive, "b", IODirection::kSend,
                        /*minimum_latency=*/2, /*maximum_latency=*/4);

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ScheduleBounds::NodeSchedulingConstraint> converted,
      ScheduleBounds::ConvertSchedulingConstraints(
          graph, {io_const}, /*ii=*/std::nullopt, /*max_upper_bound=*/100));

  ASSERT_EQ(converted.size(), 1);
  EXPECT_TRUE(converted[0].Is<NodeDifferenceConstraint>());
  auto ndc = converted[0].As<NodeDifferenceConstraint>();
  EXPECT_EQ(ndc.anchor, rcv.node());
  EXPECT_EQ(ndc.subject, send.node());
  EXPECT_EQ(ndc.min_after, 2);
  EXPECT_EQ(ndc.max_after, 4);
}

TEST_F(ScheduleBoundsTest, ConvertRecvsFirstSendsLastConstraint) {
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

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ScheduleBounds::NodeSchedulingConstraint> converted,
      ScheduleBounds::ConvertSchedulingConstraints(
          graph, {RecvsFirstSendsLastConstraint()}, /*ii=*/std::nullopt,
          /*max_upper_bound=*/100));

  // Should result in NodeInCycleConstraint(rcv, 0) and
  // LastStageConstraint(send).
  ASSERT_EQ(converted.size(), 2);
  bool found_rcv = false;
  bool found_send = false;
  for (const auto& c : converted) {
    if (c.Is<NodeInCycleConstraint>()) {
      auto nicc = c.As<NodeInCycleConstraint>();
      if (nicc.GetNode() == rcv.node()) {
        EXPECT_EQ(nicc.GetCycle(), 0);
        found_rcv = true;
      }
    } else if (c.Is<LastStageConstraint>()) {
      auto lsc = c.As<LastStageConstraint>();
      if (lsc.node == send.node()) {
        found_send = true;
      }
    }
  }
  EXPECT_TRUE(found_rcv);
  EXPECT_TRUE(found_send);
}

TEST_F(ScheduleBoundsTest, ConvertSendThenRecvConstraint) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Channel * ch_a,
                           p->CreateStreamingChannel("a", ChannelOps::kSendOnly,
                                                     p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_b, p->CreateStreamingChannel("b", ChannelOps::kReceiveOnly,
                                                p->GetBitsType(32)));

  ProcBuilder pb(TestName(), p.get());
  BValue tkn = pb.Literal(Value::Token());
  BValue send = pb.Send(ch_a, tkn, pb.Literal(UBits(42, 32)));
  BValue rcv = pb.Receive(ch_b, send);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(proc, /*dead_after_synthesis=*/{}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ScheduleBounds::NodeSchedulingConstraint> converted,
      ScheduleBounds::ConvertSchedulingConstraints(
          graph, {SendThenRecvConstraint(/*latency=*/3)}, /*ii=*/std::nullopt,
          /*max_upper_bound=*/100));

  ASSERT_EQ(converted.size(), 1);
  EXPECT_TRUE(converted[0].Is<NodeDifferenceConstraint>());
  auto ndc = converted[0].As<NodeDifferenceConstraint>();
  EXPECT_EQ(ndc.anchor, send.node());
  EXPECT_EQ(ndc.subject, rcv.node());
  EXPECT_EQ(ndc.min_after, 3);
  EXPECT_EQ(ndc.max_after, 100);
}

TEST_F(ScheduleBoundsTest, ConvertSameChannelConstraint) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * ch_a,
      p->CreateStreamingChannel(
          "a", ChannelOps::kSendOnly, p->GetBitsType(32),
          /*initial_values=*/{}, /*fifo_config=*/std::nullopt,
          FlowControl::kReadyValid, ChannelStrictness::kArbitraryStaticOrder));

  ProcBuilder pb(TestName(), p.get());
  BValue tkn = pb.Literal(Value::Token());
  BValue send0 = pb.Send(ch_a, tkn, pb.Literal(UBits(1, 32)));
  BValue send1 = pb.Send(ch_a, send0, pb.Literal(UBits(2, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(proc, /*dead_after_synthesis=*/{}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ScheduleBounds::NodeSchedulingConstraint> converted,
      ScheduleBounds::ConvertSchedulingConstraints(
          graph, {SameChannelConstraint(/*latency=*/2)}, /*ii=*/std::nullopt,
          /*max_upper_bound=*/100));

  ASSERT_EQ(converted.size(), 1);
  EXPECT_TRUE(converted[0].Is<NodeDifferenceConstraint>());
  auto ndc = converted[0].As<NodeDifferenceConstraint>();
  EXPECT_EQ(ndc.anchor, send0.node());
  EXPECT_EQ(ndc.subject, send1.node());
  EXPECT_EQ(ndc.min_after, 2);
  EXPECT_EQ(ndc.max_after, 100);
}

TEST_F(ScheduleBoundsTest, ConvertBackedgeConstraint) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue st = pb.StateElement("st", Value(UBits(0, 32)));
  BValue next_val = pb.Add(st, pb.Literal(UBits(1, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({next_val}));

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(proc, /*dead_after_synthesis=*/{}));

  XLS_ASSERT_OK_AND_ASSIGN(
      std::vector<ScheduleBounds::NodeSchedulingConstraint> converted,
      ScheduleBounds::ConvertSchedulingConstraints(
          graph, {BackedgeConstraint()}, /*ii=*/5,
          /*max_upper_bound=*/100));

  // Backedge constraint results in NodeDifferenceConstraint between state_read
  // and next_value.
  ASSERT_EQ(converted.size(), 1);
  EXPECT_TRUE(converted[0].Is<NodeDifferenceConstraint>());
  auto ndc = converted[0].As<NodeDifferenceConstraint>();
  // Find the StateRead and Next nodes.
  Node* state_read = nullptr;
  Node* next_node = nullptr;
  for (Node* node : proc->nodes()) {
    if (node->Is<StateRead>()) {
      state_read = node;
    }
    if (node->Is<Next>()) {
      next_node = node;
    }
  }
  EXPECT_EQ(ndc.anchor, state_read);
  EXPECT_EQ(ndc.subject, next_node);
  EXPECT_EQ(ndc.min_after, 0);
  EXPECT_EQ(ndc.max_after, 4);  // ii - 1
}

TEST_F(ScheduleBoundsTest, ConvertDifferenceConstraintUnimplemented) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  auto a = fb.Param("a", p->GetBitsType(32));
  auto b = fb.Param("b", p->GetBitsType(32));
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(f, /*dead_after_synthesis=*/{}));
  EXPECT_THAT(ScheduleBounds::ConvertSchedulingConstraints(
                  graph, {DifferenceConstraint(a.node(), b.node(), 5)},
                  /*ii=*/std::nullopt, /*max_upper_bound=*/100),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace sched
}  // namespace xls
