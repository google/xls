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

#include "xls/scheduling/proc_state_legalization_pass.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/cse_pass.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/scheduling/scheduling_options.h"
#include "xls/scheduling/scheduling_pass.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::UnorderedElementsAre;

class SimplificationPass : public OptimizationCompoundPass {
 public:
  explicit SimplificationPass()
      : OptimizationCompoundPass("simp", "Simplification") {
    Add<DeadCodeEliminationPass>();
    Add<CsePass>();
  }
};

class ProcStateLegalizationPassTest : public IrTestBase {
 protected:
  ProcStateLegalizationPassTest() = default;

  absl::StatusOr<bool> Run(Proc* f) { return Run(f, SchedulingPassOptions()); }
  absl::StatusOr<bool> Run(Proc* f, const SchedulingPassOptions& options) {
    PassResults results;
    SchedulingUnit unit = SchedulingUnit::CreateForSingleFunction(f);
    SchedulingPassResults scheduling_results;
    return ProcStateLegalizationPass().RunOnFunctionBase(f, &unit, options,
                                                         &scheduling_results);
  }
};

TEST_F(ProcStateLegalizationPassTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.GetTokenParam(), std::vector<BValue>()));

  ASSERT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithUnchangingState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), {x, y}));

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(m::Next(x.node(), x.node()),
                                   m::Next(y.node(), y.node())));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithChangingState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam(), {y, x}));

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  ASSERT_THAT(proc->next_values(),
              UnorderedElementsAre(m::Next(x.node(), y.node()),
                                   m::Next(y.node(), x.node())));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithUnconditionalNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  pb.Next(x, incremented);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  ASSERT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithPredicatedNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue predicate = pb.Eq(x, pb.Literal(UBits(0, 32)));
  pb.Next(x, incremented, predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(x.node(), incremented.node(), predicate.node()),
                  m::Next(x.node(), x.node(), m::Not(predicate.node()))));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithPredicatedNextValueAndDefault) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue predicate = pb.Eq(x, pb.Literal(UBits(0, 32)));
  pb.Next(x, incremented, predicate);
  pb.Next(x, x, pb.Not(predicate));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  ASSERT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithMultiplePredicatedNextValues) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue decremented = pb.Subtract(x, pb.Literal(UBits(1, 32)));
  BValue predicate1 = pb.Eq(x, pb.Literal(UBits(0, 32)));
  BValue predicate2 = pb.Eq(x, pb.Literal(UBits(1, 32)));
  pb.Next(x, incremented, predicate1);
  pb.Next(x, decremented, predicate2);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(x.node(), incremented.node(), predicate1.node()),
                  m::Next(x.node(), decremented.node(), predicate2.node()),
                  m::Next(x.node(), x.node(),
                          m::Nor(predicate1.node(), predicate2.node()))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithMultiplePredicatedNextValuesAndDefault) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue decremented = pb.Subtract(x, pb.Literal(UBits(1, 32)));
  BValue predicate1 = pb.Eq(x, pb.Literal(UBits(0, 32)));
  BValue predicate2 = pb.Eq(x, pb.Literal(UBits(1, 32)));
  pb.Next(x, x, pb.Nor({predicate2, predicate1, predicate2}));
  pb.Next(x, incremented, predicate1);
  pb.Next(x, decremented, predicate2);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  ASSERT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithNoExplicitDefaultNeededAndZ3Enabled) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue positive_predicate = pb.Eq(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, x, positive_predicate);
  BValue negative_predicate = pb.Ne(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, incremented, negative_predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  ASSERT_THAT(
      Run(proc, {.scheduling_options =
                     SchedulingOptions().default_next_value_z3_rlimit(5000)}),
      IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithNoExplicitDefaultNeededButZ3Disabled) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue positive_predicate = pb.Eq(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, x, positive_predicate);
  BValue negative_predicate = pb.Ne(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, incremented, negative_predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(
      proc->next_values(),
      UnorderedElementsAre(
          m::Next(x.node(), x.node(), positive_predicate.node()),
          m::Next(x.node(), incremented.node(), negative_predicate.node()),
          m::Next(
              x.node(), x.node(),
              m::Nor(positive_predicate.node(), negative_predicate.node()))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithPredicatedNextValueAndSmallRlimit) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue predicate = pb.Eq(x, pb.Literal(UBits(0, 32)));
  pb.Next(x, incremented, predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  ASSERT_THAT(
      Run(proc, {.scheduling_options =
                     SchedulingOptions().default_next_value_z3_rlimit(1)}),
      IsOkAndHolds(true));

  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(x.node(), incremented.node(), predicate.node()),
                  m::Next(x.node(), x.node(), m::Not(predicate.node()))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithNoExplicitDefaultNeededButSmallRlimit) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue positive_predicate = pb.Eq(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, x, positive_predicate);
  BValue negative_predicate = pb.Ne(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, incremented, negative_predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  ASSERT_THAT(
      Run(proc, {.scheduling_options =
                     SchedulingOptions().default_next_value_z3_rlimit(1)}),
      IsOkAndHolds(true));

  EXPECT_THAT(
      proc->next_values(),
      UnorderedElementsAre(
          m::Next(x.node(), x.node(), positive_predicate.node()),
          m::Next(x.node(), incremented.node(), negative_predicate.node()),
          m::Next(
              x.node(), x.node(),
              m::Nor(positive_predicate.node(), negative_predicate.node()))));
}

}  // namespace
}  // namespace xls
