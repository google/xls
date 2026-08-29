// Copyright 2026 The XLS Authors
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

#include "xls/passes/concat_select_removal_pass.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_matcher.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/scheduled_builder.h"
#include "xls/passes/dce_pass.h"
#include "xls/passes/optimization_pass.h"
#include "xls/passes/pass_base.h"
#include "xls/solvers/ir_equivalence_testutils.h"

namespace m = xls::op_matchers;

namespace xls {
namespace {

using absl_testing::IsOkAndHolds;
using solvers::ScopedVerifyEquivalence;

class ConcatSelectRemovalPassTest : public IrTestBase {
 public:
  absl::StatusOr<bool> RunPass(Package* p) {
    OptimizationCompoundPass pass("pass_and_cleanup", "Pass and Cleanup");
    pass.Add<ConcatSelectRemovalPass>();
    pass.Add<DeadCodeEliminationPass>();

    PassResults results;
    OptimizationContext context;
    return pass.Run(p, OptimizationPassOptions(), &results, context);
  }
};

TEST_F(ConcatSelectRemovalPassTest, SimpleConcatSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("selector", p->GetBitsType(3));
  fb.Concat({fb.PrioritySelect(sel,
                               {fb.Param("a", p->GetBitsType(4)),
                                fb.Param("b", p->GetBitsType(4)),
                                fb.Param("c", p->GetBitsType(4))},
                               fb.Param("d", p->GetBitsType(4))),
             fb.PrioritySelect(sel,
                               {fb.Param("e", p->GetBitsType(4)),
                                fb.Param("f", p->GetBitsType(4)),
                                fb.Param("g", p->GetBitsType(4))},
                               fb.Param("h", p->GetBitsType(4))),
             fb.PrioritySelect(sel,
                               {fb.Param("i", p->GetBitsType(4)),
                                fb.Param("j", p->GetBitsType(4)),
                                fb.Param("k", p->GetBitsType(4))},
                               fb.Param("l", p->GetBitsType(4)))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(RunPass(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(f->return_value(),
              m::PrioritySelect(
                  m::Param("selector"),
                  {m::Concat(m::Param("a"), m::Param("e"), m::Param("i")),
                   m::Concat(m::Param("b"), m::Param("f"), m::Param("j")),
                   m::Concat(m::Param("c"), m::Param("g"), m::Param("k"))},
                  m::Concat(m::Param("d"), m::Param("h"), m::Param("l"))));
}

TEST_F(ConcatSelectRemovalPassTest, DifferentCaseCounts) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("selector", p->GetBitsType(3));
  fb.Concat({fb.Select(sel,
                       {fb.Param("a", p->GetBitsType(4)),
                        fb.Param("b", p->GetBitsType(4))},
                       fb.Param("c", p->GetBitsType(4))),
             fb.Select(sel,
                       {fb.Param("d", p->GetBitsType(4)),
                        fb.Param("e", p->GetBitsType(4)),
                        fb.Param("f", p->GetBitsType(4))},
                       fb.Param("g", p->GetBitsType(4))),
             fb.Select(sel, {
                                fb.Param("h", p->GetBitsType(4)),
                                fb.Param("i", p->GetBitsType(4)),
                                fb.Param("j", p->GetBitsType(4)),
                                fb.Param("k", p->GetBitsType(4)),
                                fb.Param("l", p->GetBitsType(4)),
                                fb.Param("m", p->GetBitsType(4)),
                                fb.Param("n", p->GetBitsType(4)),
                                fb.Param("o", p->GetBitsType(4)),
                            })});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(RunPass(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Select(m::Param("selector"),
                {m::Concat(m::Param("a"), m::Param("d"), m::Param("h")),
                 m::Concat(m::Param("b"), m::Param("e"), m::Param("i")),
                 m::Concat(m::Param("c"), m::Param("f"), m::Param("j")),
                 m::Concat(m::Param("c"), m::Param("g"), m::Param("k")),
                 m::Concat(m::Param("c"), m::Param("g"), m::Param("l")),
                 m::Concat(m::Param("c"), m::Param("g"), m::Param("m")),
                 m::Concat(m::Param("c"), m::Param("g"), m::Param("n")),
                 m::Concat(m::Param("c"), m::Param("g"), m::Param("o"))}));
}

TEST_F(ConcatSelectRemovalPassTest, NoSelects) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  fb.Concat({fb.Param("a", p->GetBitsType(4)), fb.Param("b", p->GetBitsType(4)),
             fb.Param("c", p->GetBitsType(4))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(RunPass(p.get()), IsOkAndHolds(false));
}

TEST_F(ConcatSelectRemovalPassTest, SingleSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("selector", p->GetBitsType(3));
  fb.Concat({fb.Select(sel,
                       {fb.Param("a", p->GetBitsType(4)),
                        fb.Param("b", p->GetBitsType(4))},
                       fb.Param("c", p->GetBitsType(4))),
             fb.Param("d", p->GetBitsType(4))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(RunPass(p.get()), IsOkAndHolds(false));
}
TEST_F(ConcatSelectRemovalPassTest, SeparatedSelect) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("selector", p->GetBitsType(3));
  fb.Concat({fb.Select(sel,
                       {fb.Param("a", p->GetBitsType(4)),
                        fb.Param("b", p->GetBitsType(4))},
                       fb.Param("c", p->GetBitsType(4))),
             fb.Param("d", p->GetBitsType(4)),
             fb.Select(sel,
                       {fb.Param("e", p->GetBitsType(4)),
                        fb.Param("f", p->GetBitsType(4))},
                       fb.Param("g", p->GetBitsType(4)))});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(RunPass(p.get()), IsOkAndHolds(false));
}

TEST_F(ConcatSelectRemovalPassTest, TwoPieces) {
  auto p = CreatePackage();
  FunctionBuilder fb(TestName(), p.get());
  BValue sel_a = fb.Param("sel_a", p->GetBitsType(2));
  BValue sel_b = fb.Param("sel_b", p->GetBitsType(2));
  fb.Concat({
      fb.PrioritySelect(sel_a,
                        {
                            fb.Param("a", p->GetBitsType(4)),
                            fb.Param("b", p->GetBitsType(4)),
                        },
                        fb.Param("c", p->GetBitsType(4))),
      fb.PrioritySelect(sel_a,
                        {
                            fb.Param("d", p->GetBitsType(4)),
                            fb.Param("e", p->GetBitsType(4)),
                        },
                        fb.Param("f", p->GetBitsType(4))),
      fb.PrioritySelect(sel_b,
                        {
                            fb.Param("h", p->GetBitsType(4)),
                            fb.Param("i", p->GetBitsType(4)),
                        },
                        fb.Param("j", p->GetBitsType(4))),
      fb.PrioritySelect(sel_b,
                        {
                            fb.Param("k", p->GetBitsType(4)),
                            fb.Param("l", p->GetBitsType(4)),
                        },
                        fb.Param("m", p->GetBitsType(4))),
  });
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  EXPECT_THAT(RunPass(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::PrioritySelect(m::Param("sel_a"),
                                  {m::Concat(m::Param("a"), m::Param("d")),
                                   m::Concat(m::Param("b"), m::Param("e"))},
                                  m::Concat(m::Param("c"), m::Param("f"))),
                m::PrioritySelect(m::Param("sel_b"),
                                  {m::Concat(m::Param("h"), m::Param("k")),
                                   m::Concat(m::Param("i"), m::Param("l"))},
                                  m::Concat(m::Param("j"), m::Param("m")))));
}

TEST_F(ConcatSelectRemovalPassTest, OnlyMergeSameStage) {
  auto p = CreatePackage();
  ScheduledFunctionBuilder fb(TestName(), p.get());
  BValue sel = fb.Param("sel", p->GetBitsType(2));
  fb.SetCurrentStage(0);
  BValue a = fb.PrioritySelect(sel,
                               {
                                   fb.Param("a", p->GetBitsType(4)),
                                   fb.Param("b", p->GetBitsType(4)),
                               },
                               fb.Param("c", p->GetBitsType(4)));
  BValue b = fb.PrioritySelect(sel,
                               {
                                   fb.Param("d", p->GetBitsType(4)),
                                   fb.Param("e", p->GetBitsType(4)),
                               },
                               fb.Param("f", p->GetBitsType(4)));
  fb.SetCurrentStage(1);
  BValue c = fb.PrioritySelect(sel,
                               {
                                   fb.Param("h", p->GetBitsType(4)),
                                   fb.Param("i", p->GetBitsType(4)),
                               },
                               fb.Param("j", p->GetBitsType(4)));
  BValue d = fb.PrioritySelect(sel,
                               {
                                   fb.Param("k", p->GetBitsType(4)),
                                   fb.Param("l", p->GetBitsType(4)),
                               },
                               fb.Param("m", p->GetBitsType(4)));
  fb.Concat({a, b, c, d});
  XLS_ASSERT_OK_AND_ASSIGN(Function * f, fb.Build());
  ScopedVerifyEquivalence sve(f);
  ScopedRecordIr sri(p.get());
  EXPECT_THAT(RunPass(p.get()), IsOkAndHolds(true));
  EXPECT_THAT(
      f->return_value(),
      m::Concat(m::PrioritySelect(m::Param("sel"),
                                  {m::Concat(m::Param("a"), m::Param("d")),
                                   m::Concat(m::Param("b"), m::Param("e"))},
                                  m::Concat(m::Param("c"), m::Param("f"))),
                m::PrioritySelect(m::Param("sel"),
                                  {m::Concat(m::Param("h"), m::Param("k")),
                                   m::Concat(m::Param("i"), m::Param("l"))},
                                  m::Concat(m::Param("j"), m::Param("m")))));
}

}  // namespace
}  // namespace xls
