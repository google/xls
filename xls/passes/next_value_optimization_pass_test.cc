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

#include "xls/passes/next_value_optimization_pass.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/lsb_or_msb.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"

namespace m = ::xls::op_matchers;

namespace xls {
namespace {

using status_testing::IsOkAndHolds;
using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::UnorderedElementsAre;

class NextValueOptimizationPassTest : public IrTestBase {
 protected:
  NextValueOptimizationPassTest() = default;

  absl::StatusOr<bool> Run(
      Package* p,
      std::optional<int64_t> split_next_value_selects = std::nullopt,
      std::optional<int64_t> split_depth_limit = std::nullopt) {
    PassResults results;
    OptimizationPassOptions options;
    options.split_next_value_selects = split_next_value_selects;
    return NextValueOptimizationPass(
               kMaxOptLevel,
               split_depth_limit.value_or(
                   NextValueOptimizationPass::kDefaultMaxSplitDepth))
        .Run(p, options, &results);
  }
};

TEST_F(NextValueOptimizationPassTest, StatelessProc) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam()).status());

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(false));
}

TEST_F(NextValueOptimizationPassTest, LegacyNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  pb.StateElement("x", Value(UBits(0, 32)));
  XLS_ASSERT_OK_AND_ASSIGN(
      Proc * proc, pb.Build(pb.GetTokenParam(), {pb.Literal(UBits(5, 32))}));
  ASSERT_THAT(proc->GetStateParam(0), m::Param("x"));
  ASSERT_THAT(proc->GetNextStateElement(0), m::Literal(5));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(proc->GetNextStateElement(0), m::Param("x"));
  EXPECT_THAT(proc->next_values(),
              ElementsAre(m::Next(m::Param(), m::Literal(5))));
}

TEST_F(NextValueOptimizationPassTest, DeadNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  pb.Next(/*param=*/x, /*value=*/pb.Literal(UBits(5, 32)),
          /*pred=*/pb.Literal(UBits(0, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(), IsEmpty());
}

TEST_F(NextValueOptimizationPassTest, NextValuesWithLiteralPredicates) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 32)));
  pb.Next(/*param=*/x, /*value=*/pb.Literal(UBits(5, 32)),
          /*pred=*/pb.Literal(UBits(0, 1)));
  pb.Next(/*param=*/x, /*value=*/pb.Literal(UBits(3, 32)),
          /*pred=*/pb.Literal(UBits(1, 1)));
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(),
              ElementsAre(m::Next(m::Param(), m::Literal(3))));
}

TEST_F(NextValueOptimizationPassTest, PrioritySelectNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 3)));
  BValue priority_select = pb.PrioritySelect(
      x, std::vector{pb.Literal(UBits(2, 3)), pb.Literal(UBits(1, 3)),
                     pb.Literal(UBits(2, 3))});
  pb.Next(/*param=*/x, /*value=*/priority_select);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(),
      UnorderedElementsAre(
          m::Next(m::Param(), m::Literal(2), m::BitSlice(m::Param(), 0, 1)),
          m::Next(m::Param(), m::Literal(1),
                  m::And(m::BitSlice(m::Param(), 1, 1),
                         m::Not(m::BitSlice(m::Param(), 0, 1)))),
          m::Next(m::Param(), m::Literal(2),
                  m::And(m::BitSlice(m::Param(), 2, 1),
                         m::Not(m::OrReduce(m::BitSlice(m::Param(), 0, 2))))),
          m::Next(m::Param(), m::Literal(0),
                  m::Eq(m::Param(), m::Literal(0)))));
}

TEST_F(NextValueOptimizationPassTest, PrioritySelectLegacyNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 3)));
  BValue priority_select = pb.PrioritySelect(
      x, std::vector{pb.Literal(UBits(2, 3)), pb.Literal(UBits(1, 3)),
                     pb.Literal(UBits(2, 3))});
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc,
                           pb.Build(pb.GetTokenParam(), {priority_select}));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(),
      UnorderedElementsAre(
          m::Next(m::Param(), m::Literal(2), m::BitSlice(m::Param(), 0, 1)),
          m::Next(m::Param(), m::Literal(1),
                  m::And(m::BitSlice(m::Param(), 1, 1),
                         m::Not(m::BitSlice(m::Param(), 0, 1)))),
          m::Next(m::Param(), m::Literal(2),
                  m::And(m::BitSlice(m::Param(), 2, 1),
                         m::Not(m::OrReduce(m::BitSlice(m::Param(), 0, 2))))),
          m::Next(m::Param(), m::Literal(0),
                  m::Eq(m::Param(), m::Literal(0)))));
}

TEST_F(NextValueOptimizationPassTest, OneHotSelectNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 3)));
  BValue one_hot_x = pb.OneHot(x, LsbOrMsb::kMsb);
  BValue one_hot_select = pb.OneHotSelect(
      one_hot_x, std::vector{pb.Literal(UBits(2, 3)), pb.Literal(UBits(1, 3)),
                             pb.Literal(UBits(2, 3)), pb.Literal(UBits(3, 3))});
  pb.Next(/*param=*/x, /*value=*/one_hot_select);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  EXPECT_THAT(Run(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(m::Param(), m::Literal(2),
                          m::BitSlice(m::OneHot(LsbOrMsb::kMsb), 0, 1)),
                  m::Next(m::Param(), m::Literal(1),
                          m::BitSlice(m::OneHot(LsbOrMsb::kMsb), 1, 1)),
                  m::Next(m::Param(), m::Literal(2),
                          m::BitSlice(m::OneHot(LsbOrMsb::kMsb), 2, 1)),
                  m::Next(m::Param(), m::Literal(3),
                          m::BitSlice(m::OneHot(LsbOrMsb::kMsb), 3, 1))));
}

TEST_F(NextValueOptimizationPassTest, SmallSelectNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 2)));
  BValue select = pb.Select(
      x, std::vector{pb.Literal(UBits(2, 2)), pb.Literal(UBits(1, 2)),
                     pb.Literal(UBits(2, 2)), pb.Literal(UBits(3, 2))});
  pb.Next(/*param=*/x, /*value=*/select);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  EXPECT_THAT(Run(p.get(), /*split_next_value_selects=*/4), IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(),
      UnorderedElementsAre(
          m::Next(m::Param(), m::Literal(2), m::Eq(m::Param(), m::Literal(0))),
          m::Next(m::Param(), m::Literal(1), m::Eq(m::Param(), m::Literal(1))),
          m::Next(m::Param(), m::Literal(2), m::Eq(m::Param(), m::Literal(2))),
          m::Next(m::Param(), m::Literal(3),
                  m::Eq(m::Param(), m::Literal(3)))));
}

TEST_F(NextValueOptimizationPassTest, SmallSelectNextValueWithDefault) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 2)));
  BValue select =
      pb.Select(x,
                std::vector{pb.Literal(UBits(2, 2)), pb.Literal(UBits(1, 2)),
                            pb.Literal(UBits(2, 2))},
                /*default_value=*/pb.Literal(UBits(3, 2)));
  pb.Next(/*param=*/x, /*value=*/select);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  EXPECT_THAT(Run(p.get(), /*split_next_value_selects=*/4), IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(),
      UnorderedElementsAre(
          m::Next(m::Param(), m::Literal(2), m::Eq(m::Param(), m::Literal(0))),
          m::Next(m::Param(), m::Literal(1), m::Eq(m::Param(), m::Literal(1))),
          m::Next(m::Param(), m::Literal(2), m::Eq(m::Param(), m::Literal(2))),
          m::Next(m::Param(), m::Literal(3),
                  m::UGt(m::Param(), m::Literal(2)))));
}

TEST_F(NextValueOptimizationPassTest, BigSelectNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 2)));
  BValue select = pb.Select(
      x, std::vector{pb.Literal(UBits(2, 2)), pb.Literal(UBits(1, 2)),
                     pb.Literal(UBits(2, 2)), pb.Literal(UBits(3, 2))});
  pb.Next(/*param=*/x, /*value=*/select);
  XLS_ASSERT_OK(pb.Build(pb.GetTokenParam()).status());

  EXPECT_THAT(Run(p.get(), /*split_next_value_selects=*/3),
              IsOkAndHolds(false));
}

TEST_F(NextValueOptimizationPassTest, CascadingSmallSelectsNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 2)));
  BValue a = pb.StateElement("a", Value(UBits(0, 1)));
  BValue b = pb.StateElement("b", Value(UBits(0, 1)));
  BValue select_b_1 = pb.Select(
      b, std::vector{pb.Literal(UBits(2, 2)), pb.Literal(UBits(1, 2))});
  BValue select_b_2 = pb.Select(
      b, std::vector{pb.Literal(UBits(2, 2)), pb.Literal(UBits(3, 2))});
  BValue select_a = pb.Select(a, std::vector{select_b_1, select_b_2});
  pb.Next(/*param=*/x, /*value=*/select_a);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  EXPECT_THAT(Run(p.get(), /*split_next_value_selects=*/2), IsOkAndHolds(true));
  EXPECT_THAT(proc->next_values(),
              UnorderedElementsAre(
                  m::Next(m::Param("x"), m::Literal(2),
                          m::And(m::Eq(m::Param("a"), m::Literal(0)),
                                 m::Eq(m::Param("b"), m::Literal(0)))),
                  m::Next(m::Param("x"), m::Literal(1),
                          m::And(m::Eq(m::Param("a"), m::Literal(0)),
                                 m::Eq(m::Param("b"), m::Literal(1)))),
                  m::Next(m::Param("x"), m::Literal(2),
                          m::And(m::Eq(m::Param("a"), m::Literal(1)),
                                 m::Eq(m::Param("b"), m::Literal(0)))),
                  m::Next(m::Param("x"), m::Literal(3),
                          m::And(m::Eq(m::Param("a"), m::Literal(1)),
                                 m::Eq(m::Param("b"), m::Literal(1))))));
}

TEST_F(NextValueOptimizationPassTest,
       DepthLimitedCascadingSmallSelectsNextValue) {
  auto p = CreatePackage();
  ProcBuilder pb("p", "tkn", p.get());
  BValue x = pb.StateElement("x", Value(UBits(0, 2)));
  BValue a = pb.StateElement("a", Value(UBits(0, 1)));
  BValue b = pb.StateElement("b", Value(UBits(0, 1)));
  BValue select_b_1 = pb.Select(
      b, std::vector{pb.Literal(UBits(2, 2)), pb.Literal(UBits(1, 2))});
  BValue select_b_2 = pb.Select(
      b, std::vector{pb.Literal(UBits(2, 2)), pb.Literal(UBits(3, 2))});
  BValue select_a = pb.Select(a, std::vector{select_b_1, select_b_2});
  pb.Next(/*param=*/x, /*value=*/select_a);
  XLS_ASSERT_OK_AND_ASSIGN(Proc * proc, pb.Build(pb.GetTokenParam()));

  EXPECT_THAT(
      Run(p.get(), /*split_next_value_selects=*/2, /*split_depth_limit=*/1),
      IsOkAndHolds(true));
  EXPECT_THAT(
      proc->next_values(),
      UnorderedElementsAre(
          m::Next(m::Param("x"),
                  m::Select(m::Param("b"), {m::Literal(2), m::Literal(1)}),
                  m::Eq(m::Param("a"), m::Literal(0))),
          m::Next(m::Param("x"),
                  m::Select(m::Param("b"), {m::Literal(2), m::Literal(3)}),
                  m::Eq(m::Param("a"), m::Literal(1)))));
}

}  // namespace
}  // namespace xls
