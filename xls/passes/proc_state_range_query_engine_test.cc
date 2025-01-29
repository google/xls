// Copyright 2024 The XLS Authors
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

#include "xls/passes/proc_state_range_query_engine.h"

#include "gtest/gtest.h"
#include "absl/log/check.h"
#include "absl/strings/str_format.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/passes/range_query_engine.h"

namespace xls {
namespace {

class ProcStateRangeQueryEngineTest : public IrTestBase {};

TEST_F(ProcStateRangeQueryEngineTest, BasicNarrow) {
  auto p = CreatePackage();
  ProcBuilder fb(TestName(), p.get());
  BValue st = fb.StateElement("foo", UBits(0, 32));
  BValue res = fb.Add(st, fb.Literal(UBits(12, 32)));
  fb.Next(st, fb.ZeroExtend(
                  fb.Add(fb.Literal(UBits(1, 3)), fb.BitSlice(st, 0, 3)), 32));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, fb.Build());
  ProcStateRangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(proc).status());
  EXPECT_EQ(IntervalSetTreeToString(qe.GetIntervals(res.node())), "[[12, 19]]");
  EXPECT_EQ(IntervalSetTreeToString(qe.GetIntervals(st.node())), "[[0, 7]]");
  EXPECT_EQ(qe.MinUnsignedValue(st.node()), UBits(0, 32));
  EXPECT_EQ(qe.MaxUnsignedValue(st.node()), UBits(7, 32));
  EXPECT_EQ(qe.MinUnsignedValue(res.node()), UBits(12, 32));
  EXPECT_EQ(qe.MaxUnsignedValue(res.node()), UBits(19, 32));
}

TEST_F(ProcStateRangeQueryEngineTest, Negatives) {
  auto p = CreatePackage();
  ProcBuilder fb(TestName(), p.get());
  BValue st = fb.StateElement("foo", SBits(0, 32));
  BValue res = fb.Add(st, fb.Literal(UBits(3, 32)));
  BValue res_pos = fb.Add(st, fb.Literal(UBits(7, 32)));
  // Count -7 to 7
  auto in_loop = fb.SLt(st, fb.Literal(UBits(7, 32)));
  fb.Next(st, fb.Add(st, fb.Literal(UBits(1, 32))), in_loop);
  // If we aren't looping the value goes to -7
  fb.Next(st, fb.Literal(SBits(-7, 32)), fb.Not(in_loop));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, fb.Build());

  ProcStateRangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(proc).status());
  EXPECT_EQ(
      IntervalSetTreeToString(qe.GetIntervals(st.node())),
      absl::StrFormat("[[0, 7], [%v, %v]]", SBits(-7, 32), SBits(-1, 32)));
  EXPECT_EQ(
      IntervalSetTreeToString(qe.GetIntervals(res.node())),
      absl::StrFormat("[[0, 10], [%v, %v]]", SBits(-7 + 3, 32), SBits(-1, 32)));
  EXPECT_EQ(IntervalSetTreeToString(qe.GetIntervals(res_pos.node())),
            "[[0, 14]]");
}

TEST_F(ProcStateRangeQueryEngineTest, NegativesWithEq) {
  auto p = CreatePackage();
  ProcBuilder fb(TestName(), p.get());
  BValue st = fb.StateElement("foo", SBits(0, 32));
  BValue res = fb.Add(st, fb.Literal(UBits(3, 32)));
  BValue res_pos = fb.Add(st, fb.Literal(UBits(7, 32)));
  // Count -7 to 7
  auto in_loop = fb.Ne(st, fb.Literal(UBits(7, 32)));
  fb.Next(st, fb.Add(st, fb.Literal(UBits(1, 32))), in_loop);
  // If we aren't looping the value goes to -7
  fb.Next(st, fb.Literal(SBits(-7, 32)), fb.Not(in_loop));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, fb.Build());

  ProcStateRangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(proc).status());
  EXPECT_EQ(
      IntervalSetTreeToString(qe.GetIntervals(st.node())),
      absl::StrFormat("[[0, 7], [%v, %v]]", SBits(-7, 32), SBits(-1, 32)));
  EXPECT_EQ(
      IntervalSetTreeToString(qe.GetIntervals(res.node())),
      absl::StrFormat("[[0, 10], [%v, %v]]", SBits(-7 + 3, 32), SBits(-1, 32)));
  EXPECT_EQ(IntervalSetTreeToString(qe.GetIntervals(res_pos.node())),
            "[[0, 14]]");
}
TEST_F(ProcStateRangeQueryEngineTest, DecrementToZeroSigned) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue state = pb.StateElement("the_state", UBits(7, 32));
  BValue cont = pb.SGt(state, pb.Literal(UBits(0, 32)));
  pb.Next(state, pb.Literal(UBits(7, 32)), pb.Not(cont));
  pb.Next(state, pb.Subtract(state, pb.Literal(UBits(1, 32))), cont);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ProcStateRangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(proc).status());
  EXPECT_EQ(IntervalSetTreeToString(qe.GetIntervals(state.node())), "[[0, 7]]");
}

TEST_F(ProcStateRangeQueryEngineTest, MaskReset) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * chan, p->CreateStreamingChannel(
                          "chan", ChannelOps::kReceiveOnly, p->GetBitsType(1)));
  ProcBuilder pb(TestName(), p.get());
  BValue state = pb.StateElement("the_state", UBits(0, 32));
  BValue reset = pb.TupleIndex(pb.Receive(chan, pb.Literal(Value::Token())), 1);
  BValue range_reset = pb.UGe(state, pb.Literal(UBits(8, 32)));
  BValue nxt_val =
      pb.Add(pb.And(pb.SignExtend(reset, 32), state), pb.Literal(UBits(1, 32)));
  pb.Next(state, nxt_val, pb.Not(range_reset));
  pb.Next(state, pb.Literal(UBits(0, 32)), range_reset);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ProcStateRangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(proc).status());
  EXPECT_EQ(IntervalSetTreeToString(qe.GetIntervals(state.node())), "[[0, 8]]");
}

TEST_F(ProcStateRangeQueryEngineTest, SelectReset) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * chan, p->CreateStreamingChannel(
                          "chan", ChannelOps::kReceiveOnly, p->GetBitsType(1)));
  ProcBuilder pb(TestName(), p.get());
  BValue state = pb.StateElement("the_state", UBits(0, 32));
  BValue reset = pb.TupleIndex(pb.Receive(chan, pb.Literal(Value::Token())), 1);
  BValue range_reset = pb.UGe(state, pb.Literal(UBits(8, 32)));
  BValue nxt_val = pb.Add(pb.Select(reset, pb.Literal(UBits(0, 32)), state),
                          pb.Literal(UBits(1, 32)));
  pb.Next(state, nxt_val, pb.Not(range_reset));
  pb.Next(state, pb.Literal(UBits(0, 32)), range_reset);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ProcStateRangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(proc).status());
  EXPECT_EQ(IntervalSetTreeToString(qe.GetIntervals(state.node())), "[[0, 8]]");
}

TEST_F(ProcStateRangeQueryEngineTest, BitSliceCompare) {
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  BValue state = pb.StateElement("the_state", UBits(0, 32));
  BValue nxt_val = pb.Add(state, pb.Literal(UBits(1, 32)));
  BValue range_reset =
      pb.Ne(pb.BitSlice(nxt_val, 3, 29), pb.Literal(UBits(0, 29)));
  pb.Next(state, nxt_val, pb.Not(range_reset));
  pb.Next(state, pb.Literal(UBits(0, 32)), range_reset);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ProcStateRangeQueryEngine qe;
  XLS_ASSERT_OK(qe.Populate(proc).status());
  EXPECT_EQ(IntervalSetTreeToString(qe.GetIntervals(state.node())), "[[0, 7]]");
}

}  // namespace
}  // namespace xls
