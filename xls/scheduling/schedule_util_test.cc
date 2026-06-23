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

#include "xls/scheduling/schedule_util.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "xls/common/logging/scoped_vlog_level.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

using absl_testing::StatusIs;
using testing::HasSubstr;
using testing::UnorderedElementsAre;

class ScheduleUtilTest : public IrTestBase {};

TEST_F(ScheduleUtilTest, GetDeadAfterSynthesisNodesNormal) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue a = fb.Param("a", p->GetBitsType(32));
  BValue b = fb.Param("b", p->GetBitsType(32));
  BValue c = fb.Param("c", p->GetBitsType(32));
  BValue ab = fb.Add(a, b);
  BValue bc = fb.Add(b, c);
  BValue tok = fb.Literal(Value::Token());
  BValue cond = fb.Eq(ab, bc);
  BValue assert = fb.Assert(tok, cond, "assert");
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.BuildWithReturnValue(bc));
  XLS_ASSERT_OK_AND_ASSIGN(absl::flat_hash_set<Node*> dead_after_synthesis,
                           GetDeadAfterSynthesisNodes(f));
  EXPECT_THAT(
      dead_after_synthesis,
      UnorderedElementsAre(ab.node(), tok.node(), cond.node(), assert.node()));
}

TEST_F(ScheduleUtilTest, GetDeadAfterSynthesisNodesState) {
  auto p = CreatePackage();
  ScopedSetVlogLevel ssvl("schedule_util", 2);
  TokenlessProcBuilder pb(NewStyleProc(), TestName(), "tok", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto chan,
                           pb.AddInputChannel("foo", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(auto chan_out,
                           pb.AddOutputChannel("bar", p->GetBitsType(32)));
  BValue state = pb.StateElement("state_real", UBits(1, 32), std::nullopt,
                                 /*non_synthesizable=*/false);
  pb.Send(chan_out, state);
  BValue non_synth_state =
      pb.StateElement("nonsynth", UBits(0, 32), std::nullopt,
                      /*non_synthesizable=*/false);
  BValue cond = pb.UGe(state, non_synth_state);
  BValue assert = pb.Assert(pb.InitialToken(), cond, "assert");
  BValue recv = pb.Receive(chan);
  BValue add_state = pb.Add(state, recv);
  BValue add_non_synth = pb.Add(non_synth_state, recv);
  pb.Next(state, add_state);
  BValue non_synth_next = pb.Next(non_synth_state, add_non_synth);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(absl::flat_hash_set<Node*> dead_after_synthesis,
                           GetDeadAfterSynthesisNodes(proc));
  EXPECT_THAT(
      dead_after_synthesis,
      UnorderedElementsAre(assert.node(), cond.node(), non_synth_state.node(),
                           non_synth_next.node(), add_non_synth.node(),
                           m::TupleIndex()));
}

TEST_F(ScheduleUtilTest, GetDeadAfterSynthesisStateChasing) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc(), TestName(), "tok", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto chan,
                           pb.AddInputChannel("foo", p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(auto chan_out,
                           pb.AddOutputChannel("bar", p->GetBitsType(32)));
  BValue state = pb.StateElement("state_real", UBits(1, 32), std::nullopt,
                                 /*non_synthesizable=*/false);
  pb.Send(chan_out, state);
  // State is kept synth due to being the source of next cycles 'state_real'.
  BValue synth_state = pb.StateElement("nonsynth", UBits(0, 32), std::nullopt,
                                       /*non_synthesizable=*/false);
  BValue cond = pb.UGe(state, synth_state);
  BValue assert = pb.Assert(pb.InitialToken(), cond, "assert");
  BValue recv = pb.Receive(chan);
  BValue add_synth = pb.Add(synth_state, recv);
  pb.Next(state, synth_state);
  pb.Next(synth_state, add_synth);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(absl::flat_hash_set<Node*> dead_after_synthesis,
                           GetDeadAfterSynthesisNodes(proc));
  EXPECT_THAT(
      dead_after_synthesis,
      UnorderedElementsAre(assert.node(), cond.node(), m::TupleIndex()));
}

TEST_F(ScheduleUtilTest, GetLabelMatchScoreTest) {
  // Exact match (score 2)
  EXPECT_EQ(GetLabelMatchScore("my_label", "my_label"), 2);
  EXPECT_EQ(GetLabelMatchScore("my_label", "other_label"), std::nullopt);

  // Unlabeled match "_" (score 1)
  EXPECT_EQ(GetLabelMatchScore("_", std::nullopt), 1);
  EXPECT_EQ(GetLabelMatchScore("_", ""), 1);
  EXPECT_EQ(GetLabelMatchScore("_", "some_label"), std::nullopt);

  // Wildcard match "*" (score 0)
  EXPECT_EQ(GetLabelMatchScore("*", "any_label"), 0);
  EXPECT_EQ(GetLabelMatchScore("*", std::nullopt), 0);
  EXPECT_EQ(GetLabelMatchScore("*", ""), 0);
}

TEST_F(ScheduleUtilTest, GetArcMatchScoreTest) {
  // Exact-Exact match (score 4)
  EXPECT_EQ(GetArcMatchScore({"W1", "R1"}, "W1", "R1"), 4);
  EXPECT_EQ(GetArcMatchScore({"W1", "R1"}, "W1", "R2"), std::nullopt);

  // Unlabeled-Exact match (score 3)
  EXPECT_EQ(GetArcMatchScore({"_", "R1"}, std::nullopt, "R1"), 3);
  EXPECT_EQ(GetArcMatchScore({"_", "R1"}, "some_write", "R1"), std::nullopt);

  // Wildcard-Exact match (score 2)
  EXPECT_EQ(GetArcMatchScore({"*", "R1"}, "any_write", "R1"), 2);
  EXPECT_EQ(GetArcMatchScore({"*", "R1"}, std::nullopt, "R1"), 2);

  // Wildcard-Wildcard match (score 0)
  EXPECT_EQ(GetArcMatchScore({"*", "*"}, "W", "R"), 0);
  EXPECT_EQ(GetArcMatchScore({"*", "*"}, std::nullopt, std::nullopt), 0);
}

TEST_F(ScheduleUtilTest, GetResolvedThroughputLimitTest) {
  absl::flat_hash_map<std::pair<std::string, std::string>, int64_t> overrides =
      {
          {{"W1", "R1"}, 4},
          {{"W1", "*"}, 3},
          {{"*", "*"}, 2},
      };

  // 1. Exact-Exact match wins (score 4, limit 4)
  {
    ResolvedThroughput resolved =
        GetResolvedThroughputLimit("W1", "R1", overrides, /*default_limit=*/5,
                                   /*worst_case_throughput=*/std::nullopt);
    EXPECT_EQ(resolved.limit, 4);
    EXPECT_EQ(*resolved.matched_pattern, std::make_pair("W1", "R1"));
  }

  // 2. Exact-Wildcard match wins (score 2, limit 3)
  {
    ResolvedThroughput resolved =
        GetResolvedThroughputLimit("W1", "R2", overrides, /*default_limit=*/5,
                                   /*worst_case_throughput=*/std::nullopt);
    EXPECT_EQ(resolved.limit, 3);
    EXPECT_EQ(*resolved.matched_pattern, std::make_pair("W1", "*"));
  }

  // 3. Wildcard-Wildcard fallback (score 0, limit 2)
  {
    ResolvedThroughput resolved =
        GetResolvedThroughputLimit("W2", "R2", overrides, /*default_limit=*/5,
                                   /*worst_case_throughput=*/std::nullopt);
    EXPECT_EQ(resolved.limit, 2);
    EXPECT_EQ(*resolved.matched_pattern, std::make_pair("*", "*"));
  }

  // 4. Default limit fallback
  {
    ResolvedThroughput resolved = GetResolvedThroughputLimit(
        "W2", "R2", /*arc_worst_case_throughput=*/{}, /*default_limit=*/5,
        /*worst_case_throughput=*/std::nullopt);
    EXPECT_EQ(resolved.limit, 5);
    EXPECT_FALSE(resolved.matched_pattern.has_value());
  }

  // 5. Worst-case throughput fallback
  {
    ResolvedThroughput resolved =
        GetResolvedThroughputLimit("W2", "R2", /*arc_worst_case_throughput=*/{},
                                   /*default_limit=*/std::nullopt,
                                   /*worst_case_throughput=*/6);
    EXPECT_EQ(resolved.limit, 6);
    EXPECT_FALSE(resolved.matched_pattern.has_value());
  }

  // 6. Clamping against worst-case throughput
  {
    // Resolved limit = 4, worst-case throughput = 2 -> Clamped to 2
    ResolvedThroughput resolved =
        GetResolvedThroughputLimit("W1", "R1", overrides, /*default_limit=*/5,
                                   /*worst_case_throughput=*/2);
    EXPECT_EQ(resolved.limit, 2);
  }

  // 7. Specific override of 0 (unconstrained) keeps it unconstrained despite
  // default limit
  {
    absl::flat_hash_map<std::pair<std::string, std::string>, int64_t>
        overrides_with_unconstrained = {
            {{"W1", "R1"}, 0},
        };
    ResolvedThroughput resolved = GetResolvedThroughputLimit(
        "W1", "R1", overrides_with_unconstrained, /*default_limit=*/5,
        /*worst_case_throughput=*/std::nullopt);
    EXPECT_FALSE(resolved.limit.has_value());
    EXPECT_EQ(*resolved.matched_pattern, std::make_pair("W1", "R1"));
  }
}

TEST_F(ScheduleUtilTest, GetFeedbackArcsTest) {
  auto p = CreatePackage();

  // Proc 1
  ProcBuilder pb1(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      StateElement * se1, pb1.UnreadStateElement("state1", Value(UBits(42, 32)),
                                                 /*non_synthesizable=*/false));
  BValue read1 = pb1.StateRead(se1, /*predicate=*/std::nullopt, "my_read1");
  BValue add_val1 = pb1.Add(read1, pb1.Literal(UBits(1, 32)));
  pb1.Next(se1, add_val1, /*predicate=*/std::nullopt, "my_write1");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1, pb1.Build());

  // Proc 2
  ProcBuilder pb2("proc_2", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      StateElement * se2,
      pb2.UnreadStateElement("state2", Value(UBits(100, 32)),
                             /*non_synthesizable=*/false));
  BValue read2 = pb2.StateRead(se2, /*predicate=*/std::nullopt, "my_read2");
  BValue add_val2 = pb2.Add(read2, pb2.Literal(UBits(5, 32)));
  pb2.Next(se2, add_val2, /*predicate=*/std::nullopt, "my_write2");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc2, pb2.Build());

  // Verify single proc 1 feedback arc discovery
  std::vector<FeedbackArc> arcs1 = GetFeedbackArcsProc(proc1);
  ASSERT_EQ(arcs1.size(), 1);
  EXPECT_EQ(arcs1[0].write_node->label(), "my_write1");
  EXPECT_EQ(arcs1[0].read_node->label(), "my_read1");

  // Verify single proc 2 feedback arc discovery
  std::vector<FeedbackArc> arcs2 = GetFeedbackArcsProc(proc2);
  ASSERT_EQ(arcs2.size(), 1);
  EXPECT_EQ(arcs2[0].write_node->label(), "my_write2");
  EXPECT_EQ(arcs2[0].read_node->label(), "my_read2");

  // Verify package-wide feedback arc discovery (finds all arcs from both procs)
  std::vector<FeedbackArc> pkg_arcs = GetFeedbackArcsPackage(p.get());
  ASSERT_EQ(pkg_arcs.size(), 2);
  EXPECT_EQ(pkg_arcs[0].write_node->label(), "my_write1");
  EXPECT_EQ(pkg_arcs[0].read_node->label(), "my_read1");
  EXPECT_EQ(pkg_arcs[1].write_node->label(), "my_write2");
  EXPECT_EQ(pkg_arcs[1].read_node->label(), "my_read2");
}

TEST_F(ScheduleUtilTest, GetSpecificityScoreTest) {
  EXPECT_EQ(GetSpecificityScore({"*", "*"}), 0);
  EXPECT_EQ(GetSpecificityScore({"_", "*"}), 1);
  EXPECT_EQ(GetSpecificityScore({"*", "_"}), 1);
  EXPECT_EQ(GetSpecificityScore({"W1", "*"}), 2);
  EXPECT_EQ(GetSpecificityScore({"*", "R1"}), 2);
  EXPECT_EQ(GetSpecificityScore({"_", "_"}), 2);
  EXPECT_EQ(GetSpecificityScore({"W1", "_"}), 3);
  EXPECT_EQ(GetSpecificityScore({"_", "R1"}), 3);
  EXPECT_EQ(GetSpecificityScore({"W1", "R1"}), 4);
}

TEST_F(ScheduleUtilTest, GetPatternIntersectionComponentTest) {
  EXPECT_EQ(GetPatternIntersectionComponent("W1", "W1"), "W1");
  EXPECT_EQ(GetPatternIntersectionComponent("*", "W1"), "W1");
  EXPECT_EQ(GetPatternIntersectionComponent("W1", "*"), "W1");
  EXPECT_EQ(GetPatternIntersectionComponent("W1", "W2"), std::nullopt);
}

TEST_F(ScheduleUtilTest, GetPatternIntersectionTest) {
  // Disjoint
  EXPECT_EQ(GetPatternIntersection({"W1", "R1"}, {"W2", "R2"}), std::nullopt);

  // Partial overlap
  auto inter1 = GetPatternIntersection({"W1", "*"}, {"*", "R1"});
  ASSERT_TRUE(inter1.has_value());
  EXPECT_EQ(*inter1, std::make_pair("W1", "R1"));

  // Exact same
  auto inter2 = GetPatternIntersection({"W1", "R1"}, {"W1", "R1"});
  ASSERT_TRUE(inter2.has_value());
  EXPECT_EQ(*inter2, std::make_pair("W1", "R1"));
}

TEST_F(ScheduleUtilTest, CheckAmbiguousArcWorstCaseThroughputTest) {
  // 1. Empty / Single rule -> Success
  EXPECT_TRUE(CheckAmbiguousArcWorstCaseThroughput({}).ok());
  EXPECT_TRUE(CheckAmbiguousArcWorstCaseThroughput({{{"W1", "R1"}, 4}}).ok());

  // 2. Disjoint rules -> Success
  EXPECT_TRUE(CheckAmbiguousArcWorstCaseThroughput(
                  {{{"W1", "R1"}, 4}, {{"W2", "R2"}, 2}})
                  .ok());

  // 3. Overlap but identical values (harmless tie) -> Success
  EXPECT_TRUE(
      CheckAmbiguousArcWorstCaseThroughput({{{"W1", "*"}, 3}, {{"*", "R1"}, 3}})
          .ok());

  // 4. Overlap but different specificity scores (higher wins) -> Success
  EXPECT_TRUE(CheckAmbiguousArcWorstCaseThroughput(
                  {{{"W1", "*"}, 3}, {{"W1", "R1"}, 4}})
                  .ok());

  // 5. Unresolvable tie (equal specificity score 2, disjoint rules overlapping,
  // different values) -> Failure
  EXPECT_THAT(
      CheckAmbiguousArcWorstCaseThroughput(
          {{{"W1", "*"}, 4}, {{"*", "R1"}, 2}}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               testing::AllOf(
                   HasSubstr("Ambiguous throughput configuration"),
                   HasSubstr("W1,*"), HasSubstr("*,R1"),
                   testing::AnyOf(HasSubstr("4 vs 2"), HasSubstr("2 vs 4")))));

  // 6. Ambiguity tie is successfully overridden by a more specific rule
  // (specificity 4 wins over specificity 2 tie) -> Success
  EXPECT_TRUE(CheckAmbiguousArcWorstCaseThroughput(
                  {{{"W1", "*"}, 4}, {{"*", "R1"}, 2}, {{"W1", "R1"}, 3}})
                  .ok());
}

}  // namespace
}  // namespace xls
