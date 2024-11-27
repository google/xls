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

#include "xls/passes/proc_state_provenance_narrowing_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/globals.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_parser.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/passes/proc_state_optimization_pass.h"
#include "xls/solvers/z3_ir_equivalence_testutils.h"

namespace m = ::xls::op_matchers;
namespace xls {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::testing::UnorderedElementsAre;

class ProcStateProvenanceNarrowingPassTest : public IrTestBase {
 public:
  absl::StatusOr<bool> RunPass(Proc* p) {
    PassResults res;
    ProcStateProvenanceNarrowingPass pass;
    ScopedRecordIr sri(p->package());
    return pass.Run(p->package(), {}, &res);
  }
  absl::StatusOr<bool> RunProcStateCleanup(Proc* p) {
    ScopedRecordIr sri(p->package(), "cleanup", /*with_initial=*/false);
    OptimizationCompoundPass pipeline("cleanup", "cleanup");
    pipeline.Add<ProcStateOptimizationPass>();
    pipeline.Add<DeadCodeEliminationPass>();
    PassResults r;
    return pipeline.Run(p->package(), {}, &r);
  }
};

TEST_F(ProcStateProvenanceNarrowingPassTest, BasicJoin) {
  absl::SetVLogLevel("proc_state_provenance_narrowing_pass", 3);
  auto p = CreatePackage();
  ProcBuilder fb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(
      auto choice_chan,
      p->CreateStreamingChannel("choice", ChannelOps::kReceiveOnly,
                                p->GetBitsType(2)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto value_chan,
      p->CreateStreamingChannel("value", ChannelOps::kReceiveOnly,
                                p->GetBitsType(32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      auto chan, p->CreateStreamingChannel("out", ChannelOps::kSendOnly,
                                           p->GetBitsType(128)));

  // state is secretly a 4 element 32-bit array
  auto st = fb.StateElement("foo", UBits(0, 128));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value near_end,
      Parser::ParseValue("0x0000_0000_FFFF_0000_0000_0000_0000_0000",
                         p->GetBitsType(128)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Value near_end2,
      Parser::ParseValue("0x0000_0000_0000_FFFF_0000_0000_0000_0000",
                         p->GetBitsType(128)));
  auto choice =
      fb.TupleIndex(fb.Receive(choice_chan, fb.Literal(Value::Token())), 1);
  auto value =
      fb.TupleIndex(fb.Receive(value_chan, fb.Literal(Value::Token())), 1);
  auto top_64 = fb.BitSlice(st, 64, 64);
  auto bottom_32 = fb.BitSlice(st, 0, 32);
  auto new_st = fb.Concat({top_64, value, bottom_32});
  fb.Send(chan, fb.Literal(Value::Token()), st);
  fb.Next(st, new_st, fb.Eq(choice, fb.Literal(UBits(0, 2))));
  fb.Next(st, fb.Literal(UBits(0, 128)),
          fb.Eq(choice, fb.Literal(UBits(1, 2))));
  fb.Next(st, fb.Literal(near_end), fb.Eq(choice, fb.Literal(UBits(2, 2))));
  fb.Next(st, fb.Literal(near_end2), fb.Eq(choice, fb.Literal(UBits(3, 2))));

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, fb.Build());
  solvers::z3::ScopedVerifyProcEquivalence svpe(proc, /*activation_count=*/6,
                                                /*include_state=*/false);
  EXPECT_THAT(RunPass(proc), IsOkAndHolds(true));
  EXPECT_THAT(RunProcStateCleanup(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->StateElements(),
              UnorderedElementsAre(m::StateElement("foo", p->GetBitsType(64))));
}

}  // namespace
}  // namespace xls
