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

#include "xls/passes/proc_state_legalization_pass.h"

#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/common/status/status_macros.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::UnorderedElementsAre;

class ProcStateLegalizationPassTest : public IrTestBase {
 protected:
  ProcStateLegalizationPassTest() = default;

  absl::StatusOr<bool> Run(Package* p) {
    PassResults results;
    XLS_ASSIGN_OR_RETURN(bool changed,
                         ProcStateLegalizationPass().Run(
                             p, OptimizationPassOptions(), &results));
    // Run dce to clean things up.
    XLS_RETURN_IF_ERROR(DeadCodeEliminationPass()
                            .Run(p, OptimizationPassOptions(), &results)
                            .status());
    return changed;
  }
};

TEST_F(ProcStateLegalizationPassTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam(), std::vector<BValue>()).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithUnchangingState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam(), {x, y}).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithChangingState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam(), {y, x}).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithUnconditionalNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  pb.Next(x, incremented);
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam()).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithPredicatedNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue predicate = pb.Eq(x, pb.Literal(UBits(0, 32)));
  pb.Next(x, incremented, predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

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
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam()).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
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

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));

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
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam()).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

}  // namespace
}  // namespace xls
