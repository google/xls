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

#include "xls/scheduling/schedule_graph.h"

#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/proc_elaboration.h"
#include "xls/ir/value.h"

namespace xls {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Contains;
using ::testing::HasSubstr;
using ::testing::UnorderedElementsAre;

MATCHER_P2(EqScheduleBackedge, source, destination,
           absl::StrFormat("is backedge (%s -> %s)",
                           testing::DescribeMatcher<Node*>(source, negation),
                           testing::DescribeMatcher<Node*>(destination,
                                                           negation))) {
  return testing::ExplainMatchResult(source, arg.source, result_listener) &&
         testing::ExplainMatchResult(destination, arg.destination,
                                     result_listener) &&
         arg.distance.has_value() &&
         std::holds_alternative<LessThanInitiationInterval>(*arg.distance);
}

class ScheduleGraphTest : public IrTestBase {};

TEST_F(ScheduleGraphTest, ProcWithMultipleStateReadsAndNexts) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb("the_proc", "tkn", p.get());
  BStateElement se = pb.StateElement("x", Value(UBits(0, 32)));

  BValue cond = pb.Literal(UBits(1, 1));
  BValue not_cond = pb.Not(cond);
  BValue read1 = pb.StateRead(se, cond);
  BValue read2 = pb.StateRead(se, not_cond);

  BValue add1 = pb.Add(read1, pb.Literal(UBits(1, 32)));
  BValue add2 = pb.Add(read2, pb.Literal(UBits(2, 32)));

  BValue next1 = pb.Next(se, add1, cond);
  BValue next2 = pb.Next(se, add2, not_cond);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::Create(proc, /*dead_after_synthesis=*/{}));

  // 2 Next nodes * 2 StateRead nodes = 4 backedges.
  EXPECT_THAT(
      graph.backedges(),
      UnorderedElementsAre(EqScheduleBackedge(next1.node(), read1.node()),
                           EqScheduleBackedge(next1.node(), read2.node()),
                           EqScheduleBackedge(next2.node(), read1.node()),
                           EqScheduleBackedge(next2.node(), read2.node())));

  // Verify ScheduleNode structure.
  EXPECT_TRUE(graph.contains(read1.node()));
  EXPECT_TRUE(graph.contains(read2.node()));
  EXPECT_THAT(graph.GetScheduleNode(read1.node()).predecessors,
              Contains(cond.node()));
  EXPECT_THAT(graph.GetScheduleNode(read2.node()).predecessors,
              Contains(not_cond.node()));
  EXPECT_THAT(graph.GetScheduleNode(read1.node()).successors,
              Contains(add1.node()));
  EXPECT_THAT(graph.GetScheduleNode(read2.node()).successors,
              Contains(add2.node()));
}

TEST_F(ScheduleGraphTest, SynchronousGraphMultipleProcsWithMultipleStateReads) {
  auto p = CreatePackage();
  Type* u32 = p->GetBitsType(32);

  Proc* counter_proc;
  BValue read_a1;
  BValue read_a2;
  BValue next_a1;
  BValue next_a2;
  BValue recv_a;
  BValue send_a;
  {
    TokenlessProcBuilder pb(NewStyleProc(), "counter_proc", "tkn", p.get());
    BReceiveChannel in = pb.AddInputChannel("counter_in", u32);
    BSendChannel out = pb.AddOutputChannel("counter_out", u32);
    BStateElement se = pb.StateElement("state_a", Value(UBits(0, 32)));
    BValue cond = pb.Literal(UBits(1, 1));
    BValue not_cond = pb.Not(cond);
    read_a1 = pb.StateRead(se, cond);
    read_a2 = pb.StateRead(se, not_cond);
    recv_a = pb.Receive(in);
    BValue sum = pb.Add(pb.Select(cond, {read_a2, read_a1}), recv_a);
    send_a = pb.Send(out, sum);
    next_a1 = pb.Next(se, pb.Add(read_a1, pb.Literal(UBits(1, 32))), cond);
    next_a2 = pb.Next(se, pb.Add(read_a2, pb.Literal(UBits(2, 32))), not_cond);
    XLS_ASSERT_OK_AND_ASSIGN(counter_proc, pb.Build());
  }

  Proc* accumulator_proc;
  BValue read_b1;
  BValue read_b2;
  BValue next_b;
  BValue recv_b;
  BValue send_b;
  {
    TokenlessProcBuilder pb(NewStyleProc(), "accumulator_proc", "tkn", p.get());
    BReceiveChannel in = pb.AddInputChannel("acc_in", u32);
    BSendChannel out = pb.AddOutputChannel("acc_out", u32);
    BStateElement se = pb.StateElement("state_b", Value(UBits(10, 32)));
    BValue cond = pb.Literal(UBits(1, 1));
    BValue not_cond = pb.Not(cond);
    read_b1 = pb.StateRead(se, cond);
    read_b2 = pb.StateRead(se, not_cond);
    recv_b = pb.Receive(in);
    BValue sum = pb.Add(pb.Select(cond, {read_b2, read_b1}), recv_b);
    send_b = pb.Send(out, sum);
    next_b = pb.Next(se, sum);
    XLS_ASSERT_OK_AND_ASSIGN(accumulator_proc, pb.Build());
  }

  TokenlessProcBuilder top_pb(NewStyleProc(), "top_proc", "tkn", p.get());
  BReceiveChannel top_in = top_pb.AddInputChannel("top_in", u32);
  BSendChannel top_out = top_pb.AddOutputChannel("top_out", u32);
  BChannelWithInterfaces tmp_ch = top_pb.AddChannel("tmp_ch", u32);

  top_pb.InstantiateProc("inst1", counter_proc,
                         {top_in, tmp_ch.send_interface});
  top_pb.InstantiateProc("inst2", accumulator_proc,
                         {tmp_ch.receive_interface, top_out});

  XLS_ASSERT_OK_AND_ASSIGN(Proc * top, top_pb.Build({}));
  XLS_ASSERT_OK(p->SetTop(top));

  XLS_ASSERT_OK_AND_ASSIGN(ProcElaboration elab,
                           ProcElaboration::Elaborate(top));

  XLS_ASSERT_OK_AND_ASSIGN(
      ScheduleGraph graph,
      ScheduleGraph::CreateSynchronousGraph(p.get(), /*loopback_channels=*/{},
                                            elab, /*dead_after_synthesis=*/{}));

  // Total backedges = 4 (Proc A) + 2 (Proc B) = 6.
  EXPECT_THAT(
      graph.backedges(),
      UnorderedElementsAre(EqScheduleBackedge(next_a1.node(), read_a1.node()),
                           EqScheduleBackedge(next_a1.node(), read_a2.node()),
                           EqScheduleBackedge(next_a2.node(), read_a1.node()),
                           EqScheduleBackedge(next_a2.node(), read_a2.node()),
                           EqScheduleBackedge(next_b.node(), read_b1.node()),
                           EqScheduleBackedge(next_b.node(), read_b2.node())));

  // Verify cross-proc channel dataflow edge (Send -> Receive).
  Node* recv_a_node = recv_a.node()->operand(0);
  Node* recv_b_node = recv_b.node()->operand(0);
  EXPECT_THAT(graph.GetScheduleNode(recv_b_node).predecessors,
              Contains(send_a.node()));

  EXPECT_THAT(graph.GetScheduleNode(send_a.node()).successors,
              Contains(recv_b_node));

  // Verify top-level interface live in/out flags.
  EXPECT_TRUE(graph.GetScheduleNode(recv_a_node).is_live_in);
  EXPECT_FALSE(graph.GetScheduleNode(recv_b_node).is_live_in);
  EXPECT_TRUE(graph.GetScheduleNode(send_b.node()).is_live_out);
  EXPECT_FALSE(graph.GetScheduleNode(send_a.node()).is_live_out);
}

TEST_F(ScheduleGraphTest, StateElementWithZeroReadsFails) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb("proc_no_reads", "tkn", p.get());
  BStateElement se = pb.StateElement("unused_state", Value(UBits(0, 32)));
  pb.Next(se, pb.Literal(UBits(42, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());
  EXPECT_THAT(ScheduleGraph::Create(proc, /*dead_after_synthesis=*/{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("State element `unused_state` has no reads. "
                                 "This is not allowed.")));
}

}  // namespace
}  // namespace xls
