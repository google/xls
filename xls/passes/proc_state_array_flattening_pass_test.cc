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

#include "xls/passes/proc_state_array_flattening_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/channel.h"
#include "xls/ir/channel_ops.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/proc.h"
#include "xls/ir/value.h"
#include "xls/passes/constant_folding_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using ProcStateArrayFlatteningPassTest = IrTestBase;

using ::xls::status_testing::IsOkAndHolds;

absl::StatusOr<bool> RunArrayFlattening(Proc* p) {
  PassResults results;
  bool changed = false;
  bool changed_this_iteration = true;
  while (changed_this_iteration) {
    XLS_ASSIGN_OR_RETURN(changed_this_iteration,
                         ProcStateArrayFlatteningPass().RunOnProc(
                             p, OptimizationPassOptions(), &results));
    // Run dce and constant folding to clean things up.
    XLS_RETURN_IF_ERROR(
        ConstantFoldingPass()
            .RunOnFunctionBase(p, OptimizationPassOptions(), &results)
            .status());
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(p, OptimizationPassOptions(), &results)
            .status());
    changed = changed || changed_this_iteration;
  }

  return changed;
}

TEST_F(ProcStateArrayFlatteningPassTest, FlattenSize1ArrayParams) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel, p->CreateStreamingChannel("ch0", ChannelOps::kSendOnly,
                                                   p->GetBitsType(8)));
  ProcBuilder pb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Value state_init, Value::UBitsArray({1}, 8));
  BValue state = pb.StateElement("state", state_init);
  BValue state0 = pb.ArrayIndex(state, {pb.Literal(Value(UBits(0, 32)))});
  pb.Send(channel, pb.AfterAll({}), state0);
  BValue next_state = pb.ArrayUpdate(state, state0, {state0});
  pb.Next(state, next_state);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  EXPECT_THAT(RunArrayFlattening(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->params(),
              ElementsAre(AllOf(m::Param("state"),
                                // After flattening, state should be a 1-tuple
                                m::Type("(bits[8])"))));
}

TEST_F(ProcStateArrayFlatteningPassTest, FlattenSize2ArrayParams) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(
      Channel * channel, p->CreateStreamingChannel("ch0", ChannelOps::kSendOnly,
                                                   p->GetBitsType(8)));
  ProcBuilder pb(TestName(), p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Value state_init, Value::UBitsArray({1, 2}, 8));
  BValue state = pb.StateElement("state", state_init);
  BValue state0 = pb.ArrayIndex(state, {pb.Literal(Value(UBits(0, 32)))});
  pb.Send(channel, pb.AfterAll({}), state0);
  BValue next_state = pb.ArrayUpdate(
      state, pb.Add(state0, pb.Literal(UBits(1, 8))),
      {pb.UMod(pb.ZeroExtend(state0, 32), pb.Literal(UBits(1, 32)))});
  pb.Next(state, next_state);

  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  EXPECT_THAT(RunArrayFlattening(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->params(),
              ElementsAre(AllOf(m::Param("state"),
                                // After flattening, state should be a 2-tuple
                                m::Type("(bits[8], bits[8])"))));
}

}  // namespace
}  // namespace xls
