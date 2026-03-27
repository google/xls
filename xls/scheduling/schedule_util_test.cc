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

#include <optional>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"
#include "xls/common/logging/scoped_vlog_level.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/value.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

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
  BValue state = pb.StateElement("state_real", UBits(1, 32), std::nullopt);
  pb.Send(chan_out, state);
  BValue non_synth_state =
      pb.StateElement("nonsynth", UBits(0, 32), std::nullopt);
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
  BValue state = pb.StateElement("state_real", UBits(1, 32), std::nullopt);
  pb.Send(chan_out, state);
  // State is kept synth due to being the source of next cycles 'state_real'.
  BValue synth_state = pb.StateElement("nonsynth", UBits(0, 32), std::nullopt);
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

}  // namespace
}  // namespace xls
