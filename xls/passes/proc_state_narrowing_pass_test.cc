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

#include "xls/passes/proc_state_narrowing_pass.h"

#include <string_view>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/proc_state_optimization_pass.h"

namespace m = ::xls::op_matchers;
namespace xls {
namespace {
using status_testing::IsOkAndHolds;
using testing::AllOf;
using testing::UnorderedElementsAre;

class ProcStateNarrowingPassTest : public IrTestBase {
 public:
  absl::StatusOr<bool> RunPass(Proc* p) {
    ProcStateNarrowingPass pass;
    PassResults r;
    return pass.Run(p->package(), {}, &r);
  }

  absl::StatusOr<bool> RunProcStateCleanup(Proc* p) {
    ProcStateOptimizationPass psop;
    PassResults r;
    return psop.Run(p->package(), {}, &r);
  }
};

TEST_F(ProcStateNarrowingPassTest, ZeroExtend) {
  auto p = CreatePackage();
  ProcBuilder fb(TestName(), "tok", p.get());
  auto st = fb.StateElement("foo", UBits(0, 32));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto chan , p->CreateStreamingChannel(
          "side_effect", ChannelOps::kSendOnly, p->GetBitsType(32)));
  auto tok = fb.Send(chan, fb.GetTokenParam(), st);
  fb.Next(st, fb.ZeroExtend(
                  fb.Add(fb.Literal(UBits(1, 3)), fb.BitSlice(st, 0, 3)), 32));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, fb.Build(tok));

  EXPECT_THAT(RunPass(proc), IsOkAndHolds(true));
  EXPECT_THAT(RunProcStateCleanup(proc), IsOkAndHolds(true));

  EXPECT_THAT(
      proc->StateParams(),
      UnorderedElementsAre(AllOf(m::Type(p->GetBitsType(3)), m::Param("foo"))));
}

TEST_F(ProcStateNarrowingPassTest, ZeroExtendMultiple) {
  auto p = CreatePackage();
  ProcBuilder fb(TestName(), "tok", p.get());
  auto st = fb.StateElement("foo", UBits(0, 32));
  auto onehot = fb.OneHot(st, LsbOrMsb::kLsb);
  XLS_ASSERT_OK_AND_ASSIGN(
      auto chan , p->CreateStreamingChannel(
          "side_effect", ChannelOps::kSendOnly, p->GetBitsType(32)));
  auto tok = fb.Send(chan, fb.GetTokenParam(), st);
  fb.Next(
      st,
      fb.ZeroExtend(fb.Add(fb.Literal(UBits(1, 3)), fb.BitSlice(st, 0, 3)), 32),
      /*pred=*/fb.BitSlice(onehot, 0, 1));
  fb.Next(
      st,
      fb.ZeroExtend(fb.Add(fb.Literal(UBits(2, 3)), fb.BitSlice(st, 0, 3)), 32),
      /*pred=*/fb.BitSlice(onehot, 1, 1));
  fb.Next(
      st,
      fb.ZeroExtend(fb.Add(fb.Literal(UBits(3, 3)), fb.BitSlice(st, 0, 3)), 32),
      /*pred=*/fb.BitSlice(onehot, 2, 1));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, fb.Build(tok));

  EXPECT_THAT(RunPass(proc), IsOkAndHolds(true));
  EXPECT_THAT(RunProcStateCleanup(proc), IsOkAndHolds(true));

  EXPECT_THAT(
      proc->StateParams(),
      UnorderedElementsAre(AllOf(m::Type(p->GetBitsType(3)), m::Param("foo"))));
}

TEST_F(ProcStateNarrowingPassTest, ZeroExtendWithBigInitial) {
  auto p = CreatePackage();
  ProcBuilder fb(TestName(), "tok", p.get());
  auto st = fb.StateElement("foo", UBits(0xFF, 32));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto chan , p->CreateStreamingChannel(
          "side_effect", ChannelOps::kSendOnly, p->GetBitsType(32)));
  auto tok = fb.Send(chan, fb.GetTokenParam(), st);
  fb.Next(st, fb.ZeroExtend(
                  fb.Add(fb.Literal(UBits(1, 3)), fb.BitSlice(st, 0, 3)), 32));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, fb.Build(tok));

  EXPECT_THAT(RunPass(proc), IsOkAndHolds(true));
  EXPECT_THAT(RunProcStateCleanup(proc), IsOkAndHolds(true));

  EXPECT_THAT(
      proc->StateParams(),
      UnorderedElementsAre(AllOf(m::Type(p->GetBitsType(8)), m::Param("foo"))));
}

}  // namespace
}  // namespace xls
