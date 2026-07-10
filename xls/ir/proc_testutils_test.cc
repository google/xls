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

#include "xls/ir/proc_testutils.h"

#include <cstdint>
#include <string_view>
#include <variant>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/solvers/ir_equivalence.h"
#include "xls/solvers/ir_equivalence_testutils.h"
#include "xls/solvers/prover_matchers.h"
#include "xls/solvers/solver.h"

namespace xls {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::xls::solvers::IsProvenFalse;
using ::xls::solvers::IsProvenTrue;
using ::xls::solvers::ScopedVerifyProcEquivalence;
using ::xls::solvers::TryProveEquivalence;

class UnrollProcTest : public IrTestBase {};
class UnrollProcUntimedTest : public IrTestBase {};
TEST_F(UnrollProcTest, BasicProcEquivalence) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  ProcBuilder pb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto foo_ch, p->CreateStreamingChannel("foo_ch", ChannelOps::kReceiveOnly,
                                             p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ret_ch, p->CreateStreamingChannel("ret_ch", ChannelOps::kSendOnly,
                                             p->GetBitsType(4)));
  auto tok = pb.StateElement("tok", Value::Token());
  auto state = pb.StateElement("cnt", UBits(1, 4));
  auto recv = pb.Receive(foo_ch, tok);
  auto nxt_val = pb.Add(state, pb.TupleIndex(recv, 1));
  auto final_tok = pb.Send(ret_ch, pb.TupleIndex(recv, 0), nxt_val);
  pb.Next(state, nxt_val);
  pb.Next(tok, final_tok);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  auto read_1 = fb.Param("foo_ch_act0_read", p->GetBitsType(4));
  auto read_2 = fb.Param("foo_ch_act1_read", p->GetBitsType(4));
  auto st_1 = fb.Literal(UBits(1, 4));
  auto st_2 = fb.Add(st_1, read_1);
  auto st_3 = fb.Add(st_2, read_2);

  fb.Tuple({fb.Tuple({fb.Tuple({fb.Literal(UBits(1, 1)), st_2})}),
            fb.Tuple({fb.Tuple({fb.Literal(UBits(1, 1)), st_3})})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollProcToFunction(proc, 2, /*include_state=*/false));

  RecordProperty("func", f->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  EXPECT_THAT(TryProveEquivalence(f, converted), IsOkAndHolds(IsProvenTrue()));
}

TEST_F(UnrollProcTest, BasicProcEquivalenceWithAsserts) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  ProcBuilder pb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto foo_ch, p->CreateStreamingChannel("foo_ch", ChannelOps::kReceiveOnly,
                                             p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ret_ch, p->CreateStreamingChannel("ret_ch", ChannelOps::kSendOnly,
                                             p->GetBitsType(4)));
  auto tok = pb.StateElement("tok", Value::Token());
  auto state = pb.StateElement("cnt", UBits(1, 4));
  auto recv = pb.Receive(foo_ch, tok);
  auto nxt_val = pb.Add(state, pb.TupleIndex(recv, 1));
  auto final_tok = pb.Send(ret_ch, pb.TupleIndex(recv, 0), nxt_val);
  pb.Assert(pb.Literal(Value::Token()), pb.ULt(state, pb.Literal(UBits(4, 4))),
            "assert_msg", /*label=*/"my_label");
  pb.Next(state, nxt_val);
  pb.Next(tok, final_tok);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  auto read_1 = fb.Param("foo_ch_act0_read", p->GetBitsType(4));
  auto read_2 = fb.Param("foo_ch_act1_read", p->GetBitsType(4));
  auto st_1 = fb.Literal(UBits(1, 4));
  auto ftok = fb.Literal(Value::Token());
  auto limit = fb.Literal(UBits(4, 4));
  fb.Assert(ftok, fb.ULt(st_1, limit), "assert_msg",
            /*label=*/"my_label_act0_assert");
  auto st_2 = fb.Add(st_1, read_1);
  fb.Assert(ftok, fb.ULt(st_2, limit), "assert_msg",
            /*label=*/"my_label_act1_assert");
  auto st_3 = fb.Add(st_2, read_2);

  fb.Tuple({fb.Tuple({fb.Tuple({fb.Literal(UBits(1, 1)), st_2})}),
            fb.Tuple({fb.Tuple({fb.Literal(UBits(1, 1)), st_3})})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollProcToFunction(proc, 2, /*include_state=*/false));

  RecordProperty("func", f->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  EXPECT_THAT(TryProveEquivalence(f, converted), IsOkAndHolds(IsProvenTrue()));
}

TEST_F(UnrollProcTest, StateOnlyProcs) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch1, p->CreateStreamingChannel("in_chan1", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));

  ProcBuilder pb(absl::StrCat(TestName(), "_add_left"), p.get());
  auto tok = pb.StateElement("Tok", Value::Token());
  auto st = pb.StateElement("foo", UBits(0, 10));
  auto rd = pb.Receive(ch1, tok);
  pb.Next(st, pb.Add(st, pb.ZeroExtend(pb.TupleIndex(rd, 1), 10)));
  pb.Next(tok, pb.TupleIndex(rd, 0));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * left, pb.Build());

  ProcBuilder pb2(absl::StrCat(TestName(), "_add_right"), p.get());
  auto tok2 = pb2.StateElement("Tok", Value::Token());
  auto st2 = pb2.StateElement("foo", UBits(0, 10));
  auto rd2 = pb2.Receive(ch1, tok2);
  pb2.Next(st2, pb2.Add(pb2.ZeroExtend(pb2.TupleIndex(rd2, 1), 10), st2));
  pb2.Next(tok2, pb2.TupleIndex(rd2, 0));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * right, pb2.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Function * l,
                           UnrollProcToFunction(left, /*activation_count=*/5,
                                                /*include_state=*/true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * r,
                           UnrollProcToFunction(right, /*activation_count=*/5,
                                                /*include_state=*/true));

  EXPECT_THAT(TryProveEquivalence(l, r),
              absl_testing::IsOkAndHolds(IsProvenTrue()));
}

TEST_F(UnrollProcTest, DetectChangesProcs) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch1, p->CreateStreamingChannel("in_chan1", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));

  ProcBuilder pb(absl::StrCat(TestName(), "_add_left"), p.get());
  auto tok = pb.StateElement("Tok", Value::Token());
  auto st = pb.StateElement("foo", UBits(0, 10));
  auto rd = pb.Receive(ch1, tok);
  pb.Next(st, pb.Add(st, pb.ZeroExtend(pb.TupleIndex(rd, 1), 10)));
  pb.Next(tok, pb.TupleIndex(rd, 0));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * left, pb.Build());

  ProcBuilder pb2(absl::StrCat(TestName(), "_add_right"), p.get());
  auto tok2 = pb2.StateElement("Tok", Value::Token());
  auto st2 = pb2.StateElement("foo", UBits(0, 10));
  auto rd2 = pb2.Receive(ch1, tok2);
  pb2.Next(st2,
           pb2.Add(pb2.Literal(UBits(1, 10)),
                   pb2.Add(pb2.ZeroExtend(pb2.TupleIndex(rd2, 1), 10), st2)));
  pb2.Next(tok2, pb2.TupleIndex(rd2, 0));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * right, pb2.Build());

  XLS_ASSERT_OK_AND_ASSIGN(Function * l,
                           UnrollProcToFunction(left, /*activation_count=*/5,
                                                /*include_state=*/true));
  XLS_ASSERT_OK_AND_ASSIGN(Function * r,
                           UnrollProcToFunction(right, /*activation_count=*/5,
                                                /*include_state=*/true));

  EXPECT_THAT(TryProveEquivalence(l, r),
              absl_testing::IsOkAndHolds(IsProvenFalse()));
}

TEST_F(UnrollProcTest, UnrollDetectsImpossibleProcs) {
  // Has no sends and we say we don't want to check state so unroll can't return
  // anything.
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch1, p->CreateStreamingChannel("in_chan1", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch2, p->CreateStreamingChannel("in_chan2", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));

  ProcBuilder pb(absl::StrCat(TestName(), "_add_left"), p.get());
  auto tok = pb.StateElement("Tok", Value::Token());
  auto st = pb.StateElement("st", UBits(0, 4));
  auto rd1 = pb.Receive(ch1, tok);
  auto rd2 = pb.Receive(ch2, tok);
  pb.Next(st, pb.Add(pb.TupleIndex(rd1, 1), pb.TupleIndex(rd2, 1)));
  pb.Next(tok, pb.AfterAll({pb.TupleIndex(rd1, 0), pb.TupleIndex(rd2, 0)}));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * left, pb.Build());

  EXPECT_THAT(
      UnrollProcToFunction(left, /*activation_count=*/4,
                           /*include_state=*/false),
      absl_testing::StatusIs(
          absl::StatusCode::kInternal,
          testing::ContainsRegex(".*No sends means returned function would "
                                 "return a single constant value.*")));
}

TEST_F(UnrollProcTest, MultiProcs) {
  // A proc that does stuff and does bit-slice-zero-extend in a dumb way.
  auto p = CreatePackage();
  ProcBuilder pb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch1, p->CreateStreamingChannel("in_chan1", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch2, p->CreateStreamingChannel("in_chan2", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto out1, p->CreateStreamingChannel("out_chan1", ChannelOps::kSendOnly,
                                           p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto out2, p->CreateStreamingChannel("out_chan2", ChannelOps::kSendOnly,
                                           p->GetBitsType(4)));
  auto tok = pb.StateElement("tok", Value::Token());
  auto st1 = pb.StateElement("st1", UBits(0, 4));
  auto st2 = pb.StateElement("st2", UBits(1, 4));
  auto off = pb.Literal(UBits(0b11, 4));
  auto fancy_bitslice = [&](BValue in, int64_t st) -> BValue {
    // Just do some random stuff before the bit-slice
    return pb.BitSlice(pb.Subtract(pb.Add(in, off), off), st, 2);
  };
  auto ch1_read = pb.Receive(ch1, tok);
  auto ch2_read = pb.Receive(ch2, tok);
  auto ch1_val = pb.TupleIndex(ch1_read, 1);
  auto ch2_val = pb.TupleIndex(ch2_read, 1);
  auto ch1_top = fancy_bitslice(ch1_val, 0);
  auto st1_bot = fancy_bitslice(st1, 2);
  auto st2_top = fancy_bitslice(st2, 0);
  auto ch2_bot = fancy_bitslice(ch2_val, 2);

  auto s1 =
      pb.Send(out1, pb.TupleIndex(ch1_read, 0), pb.Concat({ch1_top, st1_bot}));
  auto s2 =
      pb.Send(out2, pb.TupleIndex(ch2_read, 0), pb.Concat({st2_top, ch2_bot}));
  auto out_tok = pb.AfterAll({s1, s2});
  pb.Next(st1, ch1_val, pb.BitSlice(st1, 0, 1));
  pb.Next(st1, ch2_val, pb.BitSlice(st1, 1, 1));
  pb.Next(st1, pb.Add(st1, pb.Literal(UBits(1, 4))), pb.BitSlice(st1, 2, 1));
  pb.Next(st1, st2, pb.BitSlice(st1, 3, 1));
  pb.Next(st2, ch1_val, pb.BitSlice(st2, 0, 1));
  pb.Next(st2, ch2_val, pb.BitSlice(st2, 1, 1));
  pb.Next(st2, pb.Add(st2, pb.Literal(UBits(1, 4))), pb.BitSlice(st2, 2, 1));
  pb.Next(st2, st1, pb.BitSlice(st2, 3, 1));
  pb.Next(tok, out_tok);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * orig_proc, pb.Build());

  ScopedVerifyProcEquivalence svpe(orig_proc, /*activation_count=*/3,
                                   /*include_state_=*/true);
  // Replace add-sub bitslice extend with bitslice-extend
  XLS_ASSERT_OK(
      ch1_top.node()->ReplaceUsesWithNew<BitSlice>(ch1_val.node(), 0, 2));
  XLS_ASSERT_OK(st2_top.node()->ReplaceUsesWithNew<BitSlice>(st2.node(), 0, 2));
  XLS_ASSERT_OK(
      ch2_bot.node()->ReplaceUsesWithNew<BitSlice>(ch2_val.node(), 2, 2));
  XLS_ASSERT_OK(st1_bot.node()->ReplaceUsesWithNew<BitSlice>(st1.node(), 2, 2));

  // Check the modification was good.
}

TEST_F(UnrollProcTest, StatelessProc) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch1, p->CreateStreamingChannel("in_chan1", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch2, p->CreateStreamingChannel("in_chan2", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto out1, p->CreateStreamingChannel("out_chan1", ChannelOps::kSendOnly,
                                           p->GetBitsType(4)));
  ProcBuilder pb(TestName(), p.get());
  BValue recv1 = pb.Receive(ch1, pb.Literal(Value::Token()));
  BValue recv2 = pb.Receive(ch2, pb.TupleIndex(recv1, 0));
  pb.Send(out1, pb.TupleIndex(recv2, 0),
          pb.Add(pb.TupleIndex(recv1, 1), pb.TupleIndex(recv2, 1)));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * orig_proc, pb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * unrolled,
      UnrollProcToFunction(orig_proc, /*activation_count=*/2,
                           /*include_state=*/false));
  RecordProperty("unrolled", unrolled->DumpIr());
  FunctionBuilder fb(absl::StrCat(TestName(), "_manual"), p.get());
  fb.Tuple({fb.Tuple({fb.Tuple({fb.Literal(UBits(1, 1)),
                                fb.Add(fb.Param("a", p->GetBitsType(4)),
                                       fb.Param("b", p->GetBitsType(4)))})}),
            fb.Tuple({fb.Tuple({fb.Literal(UBits(1, 1)),
                                fb.Add(fb.Param("c", p->GetBitsType(4)),
                                       fb.Param("d", p->GetBitsType(4)))})})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * manual, fb.Build());
  RecordProperty("manual", manual->DumpIr());
  EXPECT_THAT(TryProveEquivalence(unrolled, manual),
              IsOkAndHolds(IsProvenTrue()));
}

TEST_F(UnrollProcTest, MultiProcsDifferentSizedState) {
  // A proc that does stuff and does bit-slice-zero-extend in a dumb way.
  auto p = CreatePackage();

  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch1, p->CreateStreamingChannel("in_chan1", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ch2, p->CreateStreamingChannel("in_chan2", ChannelOps::kReceiveOnly,
                                          p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto out1, p->CreateStreamingChannel("out_chan1", ChannelOps::kSendOnly,
                                           p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto out2, p->CreateStreamingChannel("out_chan2", ChannelOps::kSendOnly,
                                           p->GetBitsType(4)));
  auto make_test_proc_with_state_bits =
      [&](std::string_view name, int64_t bits) -> absl::StatusOr<Proc*> {
    ProcBuilder pb(absl::StrCat(TestName(), "_", name), p.get());
    auto tok = pb.StateElement("tok", Value::Token());
    auto st1 = pb.StateElement("st1", UBits(0, bits));
    auto st2 = pb.StateElement("st2", UBits(1, bits));
    auto ch1_read = pb.Receive(ch1, tok);
    auto ch2_read = pb.Receive(ch2, tok);
    auto ch1_val = pb.TupleIndex(ch1_read, 1);
    auto ch2_val = pb.TupleIndex(ch2_read, 1);
    auto ch1_top = pb.BitSlice(ch1_val, 0, 2);
    auto st1_bot = pb.BitSlice(st1, 2, 2);
    auto st2_top = pb.BitSlice(st2, 0, 2);
    auto ch2_bot = pb.BitSlice(ch2_val, 2, 2);

    auto s1 = pb.Send(out1, pb.TupleIndex(ch1_read, 0),
                      pb.Concat({ch1_top, st1_bot}));
    auto s2 = pb.Send(out2, pb.TupleIndex(ch2_read, 0),
                      pb.Concat({st2_top, ch2_bot}));
    auto out_tok = pb.AfterAll({s1, s2});
    pb.Next(st1, pb.ZeroExtend(ch1_val, bits), pb.BitSlice(st1, 0, 1));
    pb.Next(st1, pb.ZeroExtend(ch2_val, bits), pb.BitSlice(st1, 1, 1));
    pb.Next(st1,
            pb.ZeroExtend(
                pb.Add(pb.BitSlice(st1, 0, 4), pb.Literal(UBits(1, 4))), bits),
            pb.BitSlice(st1, 2, 1));
    pb.Next(st1, st2, pb.BitSlice(st1, 3, 1));
    pb.Next(st2, pb.ZeroExtend(ch1_val, bits), pb.BitSlice(st2, 0, 1));
    pb.Next(st2, pb.ZeroExtend(ch2_val, bits), pb.BitSlice(st2, 1, 1));
    pb.Next(st2,
            pb.ZeroExtend(
                pb.Add(pb.BitSlice(st2, 0, 4), pb.Literal(UBits(1, 4))), bits),
            pb.BitSlice(st2, 2, 1));
    pb.Next(st2, st1, pb.BitSlice(st2, 3, 1));
    pb.Next(tok, out_tok);
    return pb.Build();
  };
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc1,
                           make_test_proc_with_state_bits("big", 8));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc2,
                           make_test_proc_with_state_bits("small", 5));

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f1, UnrollProcToFunction(proc1, 3, /*include_state=*/false));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * f2, UnrollProcToFunction(proc2, 3, /*include_state=*/false));

  EXPECT_THAT(TryProveEquivalence(f1, f2), IsOkAndHolds(IsProvenTrue()));
}

TEST_F(UnrollProcTest, PredicatedReceives) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  ProcBuilder pb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto read_ch,
      p->CreateStreamingChannel("do_read", ChannelOps::kReceiveOnly,
                                p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto write_ch,
      p->CreateStreamingChannel("do_write", ChannelOps::kReceiveOnly,
                                p->GetBitsType(1)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto bar_ch, p->CreateStreamingChannel("bar_ch", ChannelOps::kReceiveOnly,
                                             p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ret_ch, p->CreateStreamingChannel("ret_ch", ChannelOps::kSendOnly,
                                             p->GetBitsType(4)));
  BValue tok = pb.StateElement("tok", Value::Token());
  BValue state = pb.StateElement("cnt", UBits(1, 4));
  BValue cont = pb.Receive(read_ch, tok);
  BValue recv =
      pb.ReceiveIf(bar_ch, pb.TupleIndex(cont, 0), pb.TupleIndex(cont, 1));
  BValue nxt_val = pb.Add(state, pb.TupleIndex(recv, 1));
  BValue do_write = pb.Receive(write_ch, pb.TupleIndex(recv, 0));
  BValue final_tok = pb.SendIf(ret_ch, pb.TupleIndex(do_write, 0),
                               pb.TupleIndex(do_write, 1), nxt_val);
  pb.Next(state, nxt_val);
  pb.Next(tok, final_tok);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  BValue act_read_1 = fb.Param("do_read_ch_act0_read", p->GetBitsType(1));
  BValue read_1 = fb.Param("bar_ch_act0_read", p->GetBitsType(4));
  BValue act_write_1 = fb.Param("do_write_ch_act0_read", p->GetBitsType(1));
  BValue act_read_2 = fb.Param("do_read_ch_act1_read", p->GetBitsType(1));
  BValue read_2 = fb.Param("bar_ch_act1_read", p->GetBitsType(4));
  BValue act_write_2 = fb.Param("do_write_ch_act1_read", p->GetBitsType(1));
  BValue act_read_3 = fb.Param("do_read_ch_act2_read", p->GetBitsType(1));
  BValue read_3 = fb.Param("bar_ch_act2_read", p->GetBitsType(4));
  BValue act_write_3 = fb.Param("do_write_ch_act2_read", p->GetBitsType(1));
  BValue act_read_4 = fb.Param("do_read_ch_act3_read", p->GetBitsType(1));
  BValue read_4 = fb.Param("bar_ch_act3_read", p->GetBitsType(4));
  BValue act_write_4 = fb.Param("do_write_ch_act3_read", p->GetBitsType(1));
  BValue lit_zero = fb.Literal(UBits(0, 4));
  BValue st_1 = fb.Literal(UBits(1, 4));
  BValue st_2 = fb.Add(st_1, fb.Select(act_read_1, read_1, lit_zero));
  BValue st_3 = fb.Add(st_2, fb.Select(act_read_2, read_2, lit_zero));
  BValue st_4 = fb.Add(st_3, fb.Select(act_read_3, read_3, lit_zero));
  BValue st_5 = fb.Add(st_4, fb.Select(act_read_4, read_4, lit_zero));

  auto single_activation = [&](auto act_write, auto next_st) {
    return fb.Tuple(
        {fb.Tuple({act_write, fb.Select(act_write, next_st, lit_zero)})});
  };

  fb.Tuple({
      single_activation(act_write_1, st_2),
      single_activation(act_write_2, st_3),
      single_activation(act_write_3, st_4),
      single_activation(act_write_4, st_5),
  });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollProcToFunction(proc, 4, /*include_state=*/false));

  RecordProperty("func", f->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  EXPECT_THAT(TryProveEquivalence(f, converted), IsOkAndHolds(IsProvenTrue()));
}

TEST_F(UnrollProcUntimedTest, Simple) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  ProcBuilder pb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto foo_ch, p->CreateStreamingChannel("foo_ch", ChannelOps::kReceiveOnly,
                                             p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ret_ch, p->CreateStreamingChannel("ret_ch", ChannelOps::kSendOnly,
                                             p->GetBitsType(4)));
  auto tok = pb.StateElement("tok", Value::Token());
  auto state = pb.StateElement("cnt", UBits(1, 4));
  auto recv = pb.Receive(foo_ch, tok);
  auto nxt_val = pb.Add(state, pb.TupleIndex(recv, 1));
  auto final_tok = pb.Send(ret_ch, pb.TupleIndex(recv, 0), nxt_val);
  pb.Next(state, nxt_val);
  pb.Next(tok, final_tok);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  auto read_1 = fb.Param("input_values", p->GetArrayType(4, p->GetBitsType(4)));
  auto st_1 = fb.Literal(UBits(1, 4));
  auto st_2 = fb.Add(st_1, fb.ArrayIndex(read_1, {fb.Literal(UBits(0, 2))}));
  auto st_3 = fb.Add(st_2, fb.ArrayIndex(read_1, {fb.Literal(UBits(1, 2))}));
  auto st_4 = fb.Add(st_3, fb.ArrayIndex(read_1, {fb.Literal(UBits(2, 2))}));
  auto st_5 = fb.Add(st_4, fb.ArrayIndex(read_1, {fb.Literal(UBits(3, 2))}));

  fb.Tuple({fb.Tuple({/* chan send cnt= */ fb.Literal(UBits(4, 8))}),
            fb.Tuple({fb.Tuple(
                {fb.Array({st_2, st_3, st_4, st_5, fb.Literal(UBits(0, 4)),
                           fb.Literal(UBits(0, 4))},
                          p->GetBitsType(4)),
                 fb.Literal(UBits(4, 8))})})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollProcToUntimedFunction(proc, /*activation_count=*/6,
                                  /*input_value_count=*/4,
                                  /*output_value_count=*/6));

  RecordProperty("func", f->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto result, TryProveEquivalence(f, converted));
  EXPECT_THAT(result, IsProvenTrue());
}

TEST_F(UnrollProcUntimedTest, SimpleSingleIteration) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  ProcBuilder pb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto foo_ch, p->CreateStreamingChannel("foo_ch", ChannelOps::kReceiveOnly,
                                             p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto ret_ch, p->CreateStreamingChannel("ret_ch", ChannelOps::kSendOnly,
                                             p->GetBitsType(4)));
  auto tok = pb.Literal(Value::Token());
  auto recv = pb.Receive(foo_ch, tok);
  pb.Send(ret_ch, pb.TupleIndex(recv, 0), pb.TupleIndex(recv, 1));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  auto read_1 = fb.Param("input_values", p->GetArrayType(1, p->GetBitsType(4)));

  fb.Tuple({fb.Tuple({fb.Literal(UBits(1, 8))}),
            fb.Tuple({fb.Tuple(
                {fb.Array({fb.ArrayIndex(read_1, {fb.Literal(UBits(0, 1))}),
                           fb.Literal(UBits(0, 4)), fb.Literal(UBits(0, 4))},
                          p->GetBitsType(4)),
                 fb.Literal(UBits(1, 8))})})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollProcToUntimedFunction(proc, /*activation_count=*/6,
                                  /*input_value_count=*/1,
                                  /*output_value_count=*/3));

  RecordProperty("func", f->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto result, TryProveEquivalence(f, converted));
  EXPECT_THAT(result, IsProvenTrue());
}
TEST_F(UnrollProcUntimedTest, SimpleCondRecv) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  ProcBuilder pb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto foo_ch, p->CreateStreamingChannel("foo_ch", ChannelOps::kReceiveOnly,
                                             p->GetBitsType(4)));
  auto first = pb.StateElement("First", UBits(1, 1));
  pb.ReceiveIf(foo_ch, pb.Literal(Value::Token()), pb.Not(first));
  pb.Next(first, pb.Literal(UBits(0, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  fb.Param("input_values", p->GetArrayType(2, p->GetBitsType(4)));
  fb.Tuple({fb.Tuple({fb.Literal(UBits(2, 8))}), fb.Tuple({})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollProcToUntimedFunction(proc, /*activation_count=*/6,
                                  /*input_value_count=*/2,
                                  /*output_value_count=*/0));

  RecordProperty("func", f->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto result, TryProveEquivalence(f, converted));
  EXPECT_THAT(result, IsProvenTrue());
  if (!std::holds_alternative<solvers::ProvenTrue>(result)) {
    RecordProperty("initial", f->DumpIr(solvers::CounterExampleAnnotator(
                                  std::get<solvers::ProvenFalse>(result))));
    RecordProperty("proc", converted->DumpIr(solvers::CounterExampleAnnotator(
                               std::get<solvers::ProvenFalse>(result))));
  }
}
TEST_F(UnrollProcUntimedTest, RecvSome) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  ProcBuilder pb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto foo_ch, p->CreateStreamingChannel("foo_ch", ChannelOps::kReceiveOnly,
                                             p->GetBitsType(4)));
  auto cnt = pb.StateElement("cnt", UBits(0, 4));
  pb.ReceiveIf(foo_ch, pb.Literal(Value::Token()),
               pb.Or({pb.Eq(cnt, pb.Literal(UBits(2, 4))),
                      pb.Eq(cnt, pb.Literal(UBits(3, 4)))}));
  pb.Next(cnt, pb.Add(cnt, pb.Literal(UBits(1, 4))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  fb.Param("input_values", p->GetArrayType(4, p->GetBitsType(4)));
  fb.Tuple({fb.Tuple({fb.Literal(UBits(2, 8))}), fb.Tuple({})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollProcToUntimedFunction(proc, /*activation_count=*/6,
                                  /*input_value_count=*/4,
                                  /*output_value_count=*/0));

  RecordProperty("func", f->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto result, TryProveEquivalence(f, converted));
  EXPECT_THAT(result, IsProvenTrue());
  if (!std::holds_alternative<solvers::ProvenTrue>(result)) {
    RecordProperty("initial", f->DumpIr(solvers::CounterExampleAnnotator(
                                  std::get<solvers::ProvenFalse>(result))));
    RecordProperty("proc", converted->DumpIr(solvers::CounterExampleAnnotator(
                               std::get<solvers::ProvenFalse>(result))));
  }
}

TEST_F(UnrollProcUntimedTest, SimpleCondSend) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  ProcBuilder pb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto foo_ch, p->CreateStreamingChannel("foo_ch", ChannelOps::kSendOnly,
                                             p->GetBitsType(4)));
  auto not_first = pb.StateElement("not_first", UBits(0, 1));
  auto cnt = pb.StateElement("cnt", UBits(0, 4));
  pb.SendIf(foo_ch, pb.Literal(Value::Token()), not_first, cnt);
  pb.Next(not_first, pb.Literal(UBits(1, 1)));
  pb.Next(cnt, pb.Add(cnt, pb.Literal(UBits(1, 4))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  fb.Tuple({fb.Tuple({}),
            fb.Tuple({fb.Tuple(
                {fb.Array({fb.Literal(UBits(1, 4)), fb.Literal(UBits(2, 4))},
                          p->GetBitsType(4)),
                 fb.Literal(UBits(2, 8))})})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollProcToUntimedFunction(proc, /*activation_count=*/4,
                                  /*input_value_count=*/0,
                                  /*output_value_count=*/2));

  RecordProperty("func", f->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto result, TryProveEquivalence(f, converted));
  EXPECT_THAT(result, IsProvenTrue());
  if (!std::holds_alternative<solvers::ProvenTrue>(result)) {
    RecordProperty("counter_func",
                   f->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
    RecordProperty("counter_proc",
                   converted->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
  }
}
TEST_F(UnrollProcUntimedTest, SendSome) {
  auto p = CreatePackage();
  FunctionBuilder fb(absl::StrCat(TestName(), "_func"), p.get());
  ProcBuilder pb(absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto foo_ch, p->CreateStreamingChannel("foo_ch", ChannelOps::kSendOnly,
                                             p->GetBitsType(4)));
  auto cnt = pb.StateElement("cnt", UBits(0, 4));
  pb.SendIf(
      foo_ch, pb.Literal(Value::Token()),
      pb.And({pb.ULt(cnt, pb.Literal(UBits(4, 4))), pb.BitSlice(cnt, 0, 1)}),
      cnt);
  pb.Next(cnt, pb.Add(cnt, pb.Literal(UBits(1, 4))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  fb.Tuple({fb.Tuple({}),
            fb.Tuple({fb.Tuple(
                {fb.Array({fb.Literal(UBits(1, 4)), fb.Literal(UBits(3, 4)),
                           fb.Literal(UBits(0, 4)), fb.Literal(UBits(0, 4))},
                          p->GetBitsType(4)),
                 fb.Literal(UBits(2, 8))})})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(
      Function * converted,
      UnrollProcToUntimedFunction(proc, /*activation_count=*/8,
                                  /*input_value_count=*/0,
                                  /*output_value_count=*/4));

  RecordProperty("func", f->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("converted", converted->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto result, TryProveEquivalence(f, converted));
  EXPECT_THAT(result, IsProvenTrue());
  if (!std::holds_alternative<solvers::ProvenTrue>(result)) {
    RecordProperty("counter_func",
                   f->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
    RecordProperty("counter_proc",
                   converted->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
  }
}

TEST_F(UnrollProcUntimedTest, SkipSendAndRecv) {
  auto p = CreatePackage();
  Function* golden;
  Proc* slowed;
  {
    FunctionBuilder fb("golden", p.get());
    auto inputs = fb.Param("inputs", p->GetArrayType(4, p->GetBitsType(4)));
    fb.Tuple({fb.Tuple({fb.Literal(UBits(4, 8))}),
              fb.Tuple({fb.Tuple({inputs, fb.Literal(UBits(4, 8))})})});
    XLS_ASSERT_OK_AND_ASSIGN(golden, fb.Build());
  }
  {
    ProcBuilder pb(NewStyleProc{}, absl::StrCat(TestName(), "_slowed_proc"),
                   p.get());
    XLS_ASSERT_OK_AND_ASSIGN(auto input_ch,
                             pb.AddInputChannel("input", p->GetBitsType(4)));
    XLS_ASSERT_OK_AND_ASSIGN(auto output_ch,
                             pb.AddOutputChannel("output", p->GetBitsType(4)));
    auto delay = pb.StateElement("delay", UBits(0, 1));
    auto tok = pb.Literal(Value::Token());
    auto recv = pb.ReceiveIf(input_ch, tok, delay);
    pb.SendIf(output_ch, pb.TupleIndex(recv, 0), delay, pb.TupleIndex(recv, 1));
    pb.Next(delay, pb.Not(delay));
    XLS_ASSERT_OK_AND_ASSIGN(slowed, pb.Build());
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * slowed_unroll,
      UnrollProcToUntimedFunction(slowed, /*activation_count=*/10,
                                  /*input_value_count=*/4,
                                  /*output_value_count=*/4));
  RecordProperty("golden", golden->DumpIr());
  RecordProperty("slowed_unroll", slowed_unroll->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           TryProveEquivalence(golden, slowed_unroll));
  EXPECT_THAT(result, IsProvenTrue());
  if (!std::holds_alternative<solvers::ProvenTrue>(result)) {
    RecordProperty("initial", golden->DumpIr(solvers::CounterExampleAnnotator(
                                  std::get<solvers::ProvenFalse>(result))));
    RecordProperty("rotated",
                   slowed_unroll->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
  }
}

TEST_F(UnrollProcUntimedTest, MultipleChans) {
  auto p = CreatePackage();
  ProcBuilder pb(NewStyleProc{}, absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto chan_a,
                           pb.AddInputChannel("chan_a", p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(auto ret_a,
                           pb.AddOutputChannel("ret_a", p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(auto chan_b,
                           pb.AddInputChannel("chan_b", p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(auto ret_b,
                           pb.AddOutputChannel("ret_b", p->GetBitsType(4)));
  auto tok = pb.Literal(Value::Token());
  auto idx = pb.StateElement("idx", UBits(0, 3));
  auto send_a = pb.ULt(idx, pb.Literal(UBits(2, 3)));
  auto send_b = pb.UGt(idx, pb.Literal(UBits(0, 3)));
  auto recv_a = pb.ReceiveIf(chan_a, tok, send_a);
  auto recv_b = pb.ReceiveIf(chan_b, tok, send_b);
  pb.SendIf(ret_a, pb.TupleIndex(recv_a, 0), send_a, pb.TupleIndex(recv_a, 1));
  pb.SendIf(ret_b, pb.TupleIndex(recv_b, 0), send_b, pb.TupleIndex(recv_b, 1));
  pb.Next(idx, pb.Add(idx, pb.Literal(UBits(1, 3))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  FunctionBuilder fb("golden", p.get());
  auto inputs_a = fb.Param("chan_a", p->GetArrayType(4, p->GetBitsType(4)));
  auto inputs_b = fb.Param("chan_b", p->GetArrayType(4, p->GetBitsType(4)));
  fb.Tuple(
      {fb.Tuple({/* chan_a read amnt= */ fb.Literal(UBits(2, 8)),
                 /* chan_b read amnt= */ fb.Literal(UBits(3, 8))}),
       fb.Tuple(
           // ret_a value [a0, a1, <null>, <null>]
           {fb.Tuple({fb.ArrayConcat(
                          {fb.ArraySlice(inputs_a, fb.Literal(UBits(0, 4)), 2),
                           fb.Array({fb.Literal(UBits(0, 4)),
                                     fb.Literal(UBits(0, 4))},
                                    p->GetBitsType(4))}),
                      fb.Literal(UBits(2, 8))}),
            // ret_b value [b0, b1, b2, <null>]
            fb.Tuple(
                {fb.ArrayConcat(
                     {fb.ArraySlice(inputs_b, fb.Literal(UBits(0, 4)), 3),
                      fb.Array({fb.Literal(UBits(0, 4))}, p->GetBitsType(4))}),
                 fb.Literal(UBits(3, 8))})})});
  XLS_ASSERT_OK_AND_ASSIGN(Function * golden, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * unroll, UnrollProcToUntimedFunction(
                                                  proc, /*activation_count=*/4,
                                                  /*input_value_count=*/4,
                                                  /*output_value_count=*/4));
  RecordProperty("golden", golden->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("unroll", unroll->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto result, TryProveEquivalence(golden, unroll));
  EXPECT_THAT(result, IsProvenTrue());
  if (!std::holds_alternative<solvers::ProvenTrue>(result)) {
    RecordProperty("golden_ann",
                   golden->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
    RecordProperty("unroll_ann",
                   unroll->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
  }
}

TEST_F(UnrollProcUntimedTest, PartialActivation) {
  // Make sure the token edge does prevent a recv/send from occurring.
  auto p = CreatePackage();
  ProcBuilder pb(NewStyleProc{}, absl::StrCat(TestName(), "_proc"), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(auto chan_a,
                           pb.AddInputChannel("chan_a", p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(auto ret_a,
                           pb.AddOutputChannel("ret_a", p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(auto chan_b,
                           pb.AddInputChannel("chan_b", p->GetBitsType(4)));
  XLS_ASSERT_OK_AND_ASSIGN(auto ret_b,
                           pb.AddOutputChannel("ret_b", p->GetBitsType(4)));
  auto tok = pb.Literal(Value::Token());
  auto idx = pb.StateElement("idx", UBits(0, 4));
  auto do_it = pb.StateElement("send_it", UBits(0, 1));
  auto state = pb.Or(do_it, pb.UGe(idx, pb.Literal(UBits(3, 4))));
  auto recv_a = pb.Receive(chan_a, tok);
  auto send_b = pb.Send(ret_b, tok, idx);
  pb.SendIf(ret_a, pb.TupleIndex(recv_a, 0), state, pb.TupleIndex(recv_a, 1));
  pb.ReceiveIf(chan_b, send_b, state);
  pb.Next(do_it, pb.Not(do_it));
  pb.Next(idx, pb.Add(idx, pb.Literal(UBits(1, 4))));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  FunctionBuilder fb("golden", p.get());
  auto inputs_a = fb.Param("chan_a", p->GetArrayType(4, p->GetBitsType(4)));
  fb.Param("chan_b", p->GetArrayType(4, p->GetBitsType(4)));
  fb.Tuple({
      fb.Tuple({
          // chan_a read amnt=
          fb.Literal(UBits(4, 8)),
          // chan_b read amnt=
          fb.Literal(UBits(2, 8)),
      }),
      fb.Tuple({
          // ret_a value [a0, a1, <null>, <null>]
          fb.Tuple({
              fb.Array({fb.ArrayIndex(inputs_a, {fb.Literal(UBits(1, 4))}),
                        fb.ArrayIndex(inputs_a, {fb.Literal(UBits(3, 4))}),
                        fb.Literal(UBits(0, 4)), fb.Literal(UBits(0, 4))},
                       p->GetBitsType(4)),
              fb.Literal(UBits(2, 8)),
          }),
          // ret_b value [0, 1, 2, 3]
          fb.Tuple({
              fb.Array(
                  {
                      fb.Literal(UBits(0, 4)),
                      fb.Literal(UBits(1, 4)),
                      fb.Literal(UBits(2, 4)),
                      fb.Literal(UBits(3, 4)),
                  },
                  p->GetBitsType(4)),
              fb.Literal(UBits(4, 8)),
          }),
      }),
  });
  XLS_ASSERT_OK_AND_ASSIGN(Function * golden, fb.Build());
  XLS_ASSERT_OK_AND_ASSIGN(Function * unroll, UnrollProcToUntimedFunction(
                                                  proc, /*activation_count=*/6,
                                                  /*input_value_count=*/4,
                                                  /*output_value_count=*/4));
  RecordProperty("golden", golden->DumpIr());
  RecordProperty("proc", proc->DumpIr());
  RecordProperty("unroll", unroll->DumpIr());
  XLS_ASSERT_OK_AND_ASSIGN(auto result, TryProveEquivalence(golden, unroll));
  EXPECT_THAT(result, IsProvenTrue());
  if (!std::holds_alternative<solvers::ProvenTrue>(result)) {
    RecordProperty("golden_ann",
                   golden->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
    RecordProperty("unroll_ann",
                   unroll->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
  }
}

TEST_F(UnrollProcUntimedTest, Rotated) {
  auto p = CreatePackage();
  Proc* initial;
  Proc* rotated;
  {
    ProcBuilder pb(NewStyleProc{}, absl::StrCat(TestName(), "_normal_proc"),
                   p.get());

    XLS_ASSERT_OK_AND_ASSIGN(auto input_ch,
                             pb.AddInputChannel("input", p->GetBitsType(4)));
    XLS_ASSERT_OK_AND_ASSIGN(auto output_ch,
                             pb.AddOutputChannel("output", p->GetBitsType(4)));
    auto st = pb.StateElement("state", UBits(0, 4));
    auto tok = pb.Send(output_ch, pb.Literal(Value::Token()), st);
    auto recv = pb.TupleIndex(pb.Receive(input_ch, tok), 1);
    pb.Next(st, recv);
    XLS_ASSERT_OK_AND_ASSIGN(initial, pb.Build());
  }
  {
    ProcBuilder pb(NewStyleProc{}, absl::StrCat(TestName(), "_rotated_proc"),
                   p.get());
    XLS_ASSERT_OK_AND_ASSIGN(auto input_ch,
                             pb.AddInputChannel("input", p->GetBitsType(4)));
    XLS_ASSERT_OK_AND_ASSIGN(auto output_ch,
                             pb.AddOutputChannel("output", p->GetBitsType(4)));
    auto tok = pb.StateElement("tok", Value::Token());
    auto first = pb.StateElement("first", UBits(1, 1));
    auto recv = pb.ReceiveIf(input_ch, tok, pb.Not(first));
    auto send = pb.Send(
        output_ch, pb.Literal(Value::Token()),
        pb.Select(first, pb.Literal(UBits(0, 4)), pb.TupleIndex(recv, 1)));
    pb.Next(first, pb.Literal(UBits(0, 1)));
    pb.Next(tok, send);

    XLS_ASSERT_OK_AND_ASSIGN(rotated, pb.Build());
  }
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * initial_unroll,
      UnrollProcToUntimedFunction(initial, /*activation_count=*/4,
                                  /*input_value_count=*/4,
                                  /*output_value_count=*/4));
  XLS_ASSERT_OK_AND_ASSIGN(
      Function * rotated_unroll,
      UnrollProcToUntimedFunction(rotated, /*activation_count=*/5,
                                  /*input_value_count=*/4,
                                  /*output_value_count=*/4));
  XLS_ASSERT_OK_AND_ASSIGN(auto result,
                           TryProveEquivalence(initial_unroll, rotated_unroll));
  EXPECT_THAT(result, IsProvenTrue());
  if (!std::holds_alternative<solvers::ProvenTrue>(result)) {
    RecordProperty("initial",
                   initial_unroll->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
    RecordProperty("rotated",
                   rotated_unroll->DumpIr(solvers::CounterExampleAnnotator(
                       std::get<solvers::ProvenFalse>(result))));
  }
}

}  // namespace
}  // namespace xls
