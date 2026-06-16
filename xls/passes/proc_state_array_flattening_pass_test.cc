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

#include <optional>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/fuzzing/fuzztest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_domain.h"
#include "xls/fuzzer/ir_fuzzer/ir_fuzz_test_library.h"
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

using ::absl_testing::IsOkAndHolds;
using ::testing::_;
using ::testing::Contains;
using ::testing::ElementsAre;

absl::StatusOr<bool> RunArrayFlattening(Proc* p) {
  PassResults results;
  bool changed = false;
  bool changed_this_iteration = true;
  OptimizationContext context;
  while (changed_this_iteration) {
    XLS_ASSIGN_OR_RETURN(changed_this_iteration,
                         ProcStateArrayFlatteningPass().RunOnProc(
                             p, OptimizationPassOptions(), &results, context));
    // Run dce and constant folding to clean things up.
    XLS_RETURN_IF_ERROR(
        ConstantFoldingPass()
            .RunOnFunctionBase(p, OptimizationPassOptions(), &results, context)
            .status());
    XLS_RETURN_IF_ERROR(
        DeadCodeEliminationPass()
            .RunOnFunctionBase(p, OptimizationPassOptions(), &results, context)
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
  EXPECT_THAT(
      proc->StateElements(),
      ElementsAre(m::StateElement("state",
                                  // After flattening, state should be a 1-tuple
                                  "(bits[8])")));
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
  EXPECT_THAT(
      proc->StateElements(),
      ElementsAre(m::StateElement("state",
                                  // After flattening, state should be a 2-tuple
                                  "(bits[8], bits[8])")));
}

TEST_F(ProcStateArrayFlatteningPassTest,
       ExplicitStateAccessLabelPropagationThroughArrayFlattening) {
  auto p = CreatePackage();
  TokenlessProcBuilder pb(NewStyleProc(), "simple_proc", "tkn", p.get());

  XLS_ASSERT_OK_AND_ASSIGN(
      StateElement * state,
      pb.UnreadStateElement(
          "state",
          Value::ArrayOrDie({Value(UBits(10, 32)), Value(UBits(20, 32))})));
  BValue read = pb.StateRead(state, /*predicate=*/std::nullopt,
                             /*label=*/"my_read_label");

  BValue zero = pb.Literal(UBits(0, 32));
  BValue one = pb.Literal(UBits(1, 32));
  BValue elem0 = pb.ArrayIndex(read, {zero});
  BValue elem1 = pb.ArrayIndex(read, {one});
  BValue next0 = pb.Add(elem0, pb.Literal(UBits(1, 32)));
  BValue next1 = pb.Add(elem1, pb.Literal(UBits(2, 32)));
  BValue next_val = pb.Array({next0, next1}, p->GetBitsType(32));

  pb.Next(read, next_val, /*predicate=*/std::nullopt,
          /*label=*/"my_write_label");
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  EXPECT_THAT(RunArrayFlattening(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->nodes(),
              Contains(m::StateRead(
                  "state", std::optional<std::string>("my_read_label"))));
  EXPECT_THAT(
      proc->nodes(),
      Contains(m::NextWithLabel(m::StateRead(), _,
                                std::optional<std::string>("my_write_label"))));
}

void IrFuzzProcStateArrayFlattening(
    FuzzPackageWithArgs fuzz_package_with_args) {
  ProcStateArrayFlatteningPass pass;
  OptimizationPassChangesOutputs(std::move(fuzz_package_with_args), pass);
}
FUZZ_TEST(IrFuzzTest, IrFuzzProcStateArrayFlattening)
    .WithDomains(IrFuzzDomainWithArgs(/*arg_set_count=*/10));

}  // namespace
}  // namespace xls
