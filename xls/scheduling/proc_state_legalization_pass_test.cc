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

#include <iterator>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/nodes.h"
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

using ::absl_testing::IsOkAndHolds;
using ::testing::_;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Not;
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
  ProcBuilder pb("p", p.get());
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ASSERT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithUnchangingState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({x, y}));

  ASSERT_THAT(Run(proc), IsOkAndHolds(false));
  EXPECT_THAT(proc->nodes(), Each(Not(m::Assert())));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithChangingState) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build({y, x}));

  ASSERT_THAT(Run(proc), IsOkAndHolds(false));
  EXPECT_THAT(proc->nodes(), Each(Not(m::Assert())));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithUnconditionalNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  pb.Next(x, incremented);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ASSERT_THAT(Run(proc), IsOkAndHolds(false));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithPredicatedNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue predicate = pb.Eq(x, pb.Literal(UBits(0, 32)));
  pb.Next(x, incremented, predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(x.node(), incremented.node(), predicate.node()),
                  m::Next(x.node(), x.node(), m::Not(predicate.node()))));
  EXPECT_THAT(proc->nodes(), Not(Contains(m::Assert())));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithPredicatedNextValueAndDefault) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue predicate = pb.Eq(x, pb.Literal(UBits(0, 32)));
  pb.Next(x, incremented, predicate);
  pb.Next(x, x, pb.Not(predicate));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(x.node(), incremented.node(), predicate.node()),
                  m::Next(x.node(), x.node(), m::Not(predicate.node()))));

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(
      asserts,
      UnorderedElementsAre(m::Assert(
          _, m::Eq(m::Concat(predicate.node(), m::Not(predicate.node())),
                   m::BitSlice(m::OneHot(m::Concat()))))));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithMultiplePredicatedNextValues) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue decremented = pb.Subtract(x, pb.Literal(UBits(1, 32)));
  BValue predicate1 = pb.Eq(x, pb.Literal(UBits(0, 32)));
  BValue predicate2 = pb.Eq(x, pb.Literal(UBits(1, 32)));
  pb.Next(x, incremented, predicate1);
  pb.Next(x, decremented, predicate2);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(x.node(), incremented.node(), predicate1.node()),
                  m::Next(x.node(), decremented.node(), predicate2.node()),
                  m::Next(x.node(), x.node(),
                          m::Nor(predicate1.node(), predicate2.node()))));

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(asserts,
              UnorderedElementsAre(m::Assert(
                  _, m::Eq(m::Concat(predicate1.node(), predicate2.node()),
                           m::BitSlice(m::OneHot(m::Concat()))))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithMultiplePredicatedNextValuesAndDefault) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue decremented = pb.Subtract(x, pb.Literal(UBits(1, 32)));
  BValue predicate1 = pb.Eq(x, pb.Literal(UBits(0, 32)));
  BValue predicate2 = pb.Eq(x, pb.Literal(UBits(1, 32)));
  pb.Next(x, x, pb.Nor({predicate2, predicate1, predicate2}));
  pb.Next(x, incremented, predicate1);
  pb.Next(x, decremented, predicate2);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(x.node(), incremented.node(), predicate1.node()),
                  m::Next(x.node(), decremented.node(), predicate2.node()),
                  m::Next(x.node(), x.node(),
                          m::Nor(predicate2.node(), predicate1.node(),
                                 predicate2.node()))));

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(
      asserts,
      UnorderedElementsAre(m::Assert(
          _, m::Eq(m::Concat(m::Nor(predicate2.node(), predicate1.node(),
                                    predicate2.node()),
                             predicate1.node(), predicate2.node()),
                   m::BitSlice(m::OneHot(m::Concat()))))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithNoExplicitDefaultNeededAndZ3Enabled) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue positive_predicate = pb.Eq(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, x, positive_predicate);
  BValue negative_predicate = pb.Ne(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, incremented, negative_predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ASSERT_THAT(
      Run(proc, {.scheduling_options =
                     SchedulingOptions().default_next_value_z3_rlimit(5000)}),
      IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(),
      UnorderedElementsAre(
          m::Next(x.node(), x.node(), positive_predicate.node()),
          m::Next(x.node(), incremented.node(), negative_predicate.node())));

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(asserts, UnorderedElementsAre(m::Assert(
                           _, m::Eq(m::Concat(positive_predicate.node(),
                                              negative_predicate.node()),
                                    m::BitSlice(m::OneHot(m::Concat()))))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithNoExplicitDefaultNeededButZ3Disabled) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue positive_predicate = pb.Eq(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, x, positive_predicate);
  BValue negative_predicate = pb.Ne(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, incremented, negative_predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_THAT(
      proc->next_values(),
      UnorderedElementsAre(
          m::Next(x.node(), x.node(), positive_predicate.node()),
          m::Next(x.node(), incremented.node(), negative_predicate.node()),
          m::Next(
              x.node(), x.node(),
              m::Nor(positive_predicate.node(), negative_predicate.node()))));

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(asserts, UnorderedElementsAre(m::Assert(
                           _, m::Eq(m::Concat(positive_predicate.node(),
                                              negative_predicate.node()),
                                    m::BitSlice(m::OneHot(m::Concat()))))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithPredicatedNextValueAndSmallRlimit) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue predicate = pb.Eq(x, pb.Literal(UBits(0, 32)));
  pb.Next(x, incremented, predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ASSERT_THAT(
      Run(proc, {.scheduling_options =
                     SchedulingOptions().default_next_value_z3_rlimit(1)}),
      IsOkAndHolds(true));

  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(x.node(), incremented.node(), predicate.node()),
                  m::Next(x.node(), x.node(), m::Not(predicate.node()))));
  EXPECT_THAT(proc->nodes(), Not(Contains(m::Assert())));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithNoExplicitDefaultNeededButSmallRlimit) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue incremented = pb.Add(x, pb.Literal(UBits(1, 32)));
  BValue positive_predicate = pb.Eq(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, x, positive_predicate);
  BValue negative_predicate = pb.Ne(x, pb.Literal(UBits(5, 32)));
  pb.Next(x, incremented, negative_predicate);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

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

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(asserts, UnorderedElementsAre(m::Assert(
                           _, m::Eq(m::Concat(positive_predicate.node(),
                                              negative_predicate.node()),
                                    m::BitSlice(m::OneHot(m::Concat()))))));
}

TEST_F(ProcStateLegalizationPassTest, ProcWithPredicatedStateRead) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue x_even =
      pb.Eq(pb.UMod(x, pb.Literal(UBits(2, 32))), pb.Literal(UBits(0, 32)));
  BValue x_multiple_of_3 =
      pb.Eq(pb.UMod(x, pb.Literal(UBits(3, 32))), pb.Literal(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)),
                             /*read_predicate=*/x_even);
  pb.Next(x, pb.Add(x, pb.Literal(UBits(1, 32))));
  pb.Next(y, pb.Add(y, pb.Literal(UBits(1, 32))), x_multiple_of_3);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateRead(*proc->GetStateElement("x"))->predicate(),
            std::nullopt);
  EXPECT_THAT(
      proc->GetStateRead(*proc->GetStateElement("y"))->predicate(),
      Optional(m::Or(
          m::Eq(m::UMod(m::StateRead("x"), m::Literal(2)), m::Literal(0)),
          m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)), m::Literal(0)))));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(*proc->GetStateElement("y"))),
      UnorderedElementsAre(
          m::Next(
              m::StateRead("y"), m::Add(m::StateRead("y"), m::Literal(1)),
              m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)), m::Literal(0))),
          m::Next(m::StateRead("y"), m::StateRead("y"),
                  m::And(m::Or(m::Eq(m::UMod(m::StateRead("x"), m::Literal(2)),
                                     m::Literal(0)),
                               m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                                     m::Literal(0))),
                         m::Not(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                                      m::Literal(0)))))));

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(
      asserts,
      UnorderedElementsAre(m::Assert(
          _, m::Or(m::Or(m::Eq(m::UMod(m::StateRead("x"), m::Literal(2)),
                               m::Literal(0)),
                         m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                               m::Literal(0))),
                   m::Not(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                                m::Literal(0)))))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithPredicatedStateReadAndPotentialCycle) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue x_even =
      pb.Eq(pb.UMod(x, pb.Literal(UBits(2, 32))), pb.Literal(UBits(0, 32)));
  BValue x_multiple_of_3 =
      pb.Eq(pb.UMod(x, pb.Literal(UBits(3, 32))), pb.Literal(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)),
                             /*read_predicate=*/x_even);
  BValue y_even =
      pb.Eq(pb.UMod(y, pb.Literal(UBits(2, 32))), pb.Literal(UBits(0, 32)));
  pb.Next(x, pb.Add(x, pb.Literal(UBits(1, 32))));
  pb.Next(y, pb.Add(y, pb.Literal(UBits(1, 32))),
          pb.And(x_multiple_of_3, y_even));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  EXPECT_EQ(proc->GetStateRead(*proc->GetStateElement("x"))->predicate(),
            std::nullopt);
  EXPECT_THAT(
      proc->GetStateRead(*proc->GetStateElement("y"))->predicate(),
      Optional(m::Or(
          m::Eq(m::UMod(m::StateRead("x"), m::Literal(2)), m::Literal(0)),
          m::And(
              m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)), m::Literal(0)),
              m::Literal(1)))));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(*proc->GetStateElement("y"))),
      UnorderedElementsAre(
          m::Next(m::StateRead("y"), m::Add(m::StateRead("y"), m::Literal(1)),
                  m::And(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                               m::Literal(0)),
                         m::Eq(m::UMod(m::StateRead("y"), m::Literal(2)),
                               m::Literal(0)))),
          m::Next(
              m::StateRead("y"), m::StateRead("y"),
              m::And(
                  m::Or(m::Eq(m::UMod(m::StateRead("x"), m::Literal(2)),
                              m::Literal(0)),
                        m::And(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                                     m::Literal(0)),
                               m::Literal(1))),
                  m::Not(m::And(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                                      m::Literal(0)),
                                m::Eq(m::UMod(m::StateRead("y"), m::Literal(2)),
                                      m::Literal(0))))))));

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(
      asserts,
      UnorderedElementsAre(m::Assert(
          _,
          m::Or(m::Or(m::Eq(m::UMod(m::StateRead("x"), m::Literal(2)),
                            m::Literal(0)),
                      m::And(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                                   m::Literal(0)),
                             m::Literal(1))),
                m::Not(m::And(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                                    m::Literal(0)),
                              m::Eq(m::UMod(m::StateRead("y"), m::Literal(2)),
                                    m::Literal(0))))))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithCorrectlyPredicatedStateReadAndNoDefaultNextNeeded) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue x_multiple_of_3 =
      pb.Eq(pb.UMod(x, pb.Literal(UBits(3, 32))), pb.Literal(UBits(0, 32)));
  BValue x_not_multiple_of_3 = pb.Not(x_multiple_of_3);
  BValue disjunction = pb.Or(x_multiple_of_3, x_not_multiple_of_3);
  BValue y = pb.StateElement("y", Value(UBits(0, 32)),
                             /*read_predicate=*/disjunction);
  pb.Next(x, pb.Add(x, pb.Literal(UBits(1, 32))));
  pb.Next(y, pb.Add(y, pb.Literal(UBits(1, 32))), x_multiple_of_3);
  pb.Next(y, y, x_not_multiple_of_3);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  const testing::Matcher<const Node*> expected_read_predicate = m::Or(
      m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)), m::Literal(0)),
      m::Not(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)), m::Literal(0))));
  EXPECT_THAT(proc->GetStateRead(*proc->GetStateElement("y"))->predicate(),
              Optional(expected_read_predicate));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(*proc->GetStateElement("y"))),
      UnorderedElementsAre(
          m::Next(
              m::StateRead("y"), m::Add(m::StateRead("y"), m::Literal(1)),
              m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)), m::Literal(0))),
          m::Next(m::StateRead("y"), m::StateRead("y"),
                  m::Not(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                               m::Literal(0))))));

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(
      asserts,
      UnorderedElementsAre(
          m::Assert(_, m::Eq(m::Concat(x_multiple_of_3.node(),
                                       x_not_multiple_of_3.node()),
                             m::BitSlice(m::OneHot(m::Concat())))),
          m::Assert(
              _, m::Or(expected_read_predicate,
                       m::Not(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                                    m::Literal(0))))),
          m::Assert(_, m::Or(expected_read_predicate,
                             m::Not(m::Not(m::Eq(
                                 m::UMod(m::StateRead("x"), m::Literal(3)),
                                 m::Literal(0))))))));
}

TEST_F(ProcStateLegalizationPassTest,
       ProcWithPredicatedStateReadAndNoDefaultNextNeeded) {
  auto p = CreatePackage();
  ProcBuilder pb("p", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  BValue x_even =
      pb.Eq(pb.UMod(x, pb.Literal(UBits(2, 32))), pb.Literal(UBits(0, 32)));
  BValue y = pb.StateElement("y", Value(UBits(0, 32)),
                             /*read_predicate=*/x_even);
  BValue x_multiple_of_3 =
      pb.Eq(pb.UMod(x, pb.Literal(UBits(3, 32))), pb.Literal(UBits(0, 32)));
  BValue x_not_multiple_of_3 = pb.Not(x_multiple_of_3);
  pb.Next(x, pb.Add(x, pb.Literal(UBits(1, 32))));
  pb.Next(y, pb.Add(y, pb.Literal(UBits(1, 32))), x_multiple_of_3);
  pb.Next(y, y, x_not_multiple_of_3);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build());

  ScopedRecordIr sri(p.get());
  ASSERT_THAT(Run(proc), IsOkAndHolds(true));

  const ::testing::Matcher<const Node*> match_x_even_or_threeven =
      m::Or(m::Eq(m::UMod(m::StateRead("x"), m::Literal(2)), m::Literal(0)),
            m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)), m::Literal(0)));
  EXPECT_THAT(proc->GetStateRead(*proc->GetStateElement("y"))->predicate(),
              Optional(match_x_even_or_threeven));

  const ::testing::Matcher<const Node*>
      match_x_even_or_threeven_and_not_threeven =
          m::And(match_x_even_or_threeven,
                 m::Not(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                              m::Literal(0))));
  EXPECT_THAT(
      proc->next_values(proc->GetStateRead(*proc->GetStateElement("y"))),
      UnorderedElementsAre(
          m::Next(
              m::StateRead("y"), m::Add(m::StateRead("y"), m::Literal(1)),
              m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)), m::Literal(0))),
          m::Next(m::StateRead("y"), m::StateRead("y"),
                  match_x_even_or_threeven_and_not_threeven)));

  std::vector<Node*> asserts;
  absl::c_copy_if(proc->nodes(), std::back_inserter(asserts),
                  [](Node* node) { return node->Is<Assert>(); });
  EXPECT_THAT(
      asserts,
      UnorderedElementsAre(
          m::Assert(_,
                    m::Eq(m::Concat(x_multiple_of_3.node(),
                                    match_x_even_or_threeven_and_not_threeven),
                          m::BitSlice(m::OneHot(m::Concat())))),
          m::Assert(
              _, m::Or(match_x_even_or_threeven,
                       m::Not(m::Eq(m::UMod(m::StateRead("x"), m::Literal(3)),
                                    m::Literal(0))))),
          m::Assert(_,
                    m::Or(match_x_even_or_threeven,
                          m::Not(match_x_even_or_threeven_and_not_threeven)))));
}

}  // namespace
}  // namespace xls
