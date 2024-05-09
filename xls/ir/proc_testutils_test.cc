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

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
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
#include "xls/solvers/z3_ir_equivalence.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"
#include "xls/solvers/z3_ir_translator_matchers.h"

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using xls::solvers::z3::IsProvenFalse;
using xls::solvers::z3::IsProvenTrue;
using xls::solvers::z3::ScopedVerifyProcEquivalence;
using xls::solvers::z3::TryProveEquivalence;

class UnrollProcTest : public IrTestBase {};
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
              status_testing::IsOkAndHolds(IsProvenTrue()));
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
              status_testing::IsOkAndHolds(IsProvenFalse()));
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
      status_testing::StatusIs(
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

}  // namespace
}  // namespace xls
