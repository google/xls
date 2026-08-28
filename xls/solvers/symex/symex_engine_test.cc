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

#include "xls/solvers/symex/symex_engine.h"

#include <cstddef>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/solvers/symex/symbolic_path.h"
#include "z3/src/api/z3_api.h"

namespace xls::solvers::symex {
namespace {

using ::testing::ElementsAre;
using ::testing::IsEmpty;
using ::testing::SizeIs;

MATCHER_P2(BranchDecisionIs, arm_index, is_default,
           absl::StrCat("has arm_index ", arm_index, " and is_default ",
                        is_default ? "true" : "false")) {
  return arg.arm_index == arm_index && arg.is_default() == is_default;
}

MATCHER_P2(SymbolicPathIs, arm_index, is_default,
           absl::StrCat("has first decision arm_index ", arm_index,
                        " and is_default ", is_default ? "true" : "false")) {
  if (arg.branch_decisions.empty()) {
    *result_listener << "has empty branch_decisions";
    return false;
  }
  const BranchDecision& decision = arg.branch_decisions[0];
  return decision.arm_index == arm_index && decision.is_default() == is_default;
}

class SymExEngineTest : public IrTestBase {
 protected:
  void SetUp() override {
    IrTestBase::SetUp();
    config_ = Z3_mk_config();
    ctx_ = Z3_mk_context(config_);
  }

  void TearDown() override {
    if (ctx_ != nullptr) {
      Z3_del_context(ctx_);
    }
    if (config_ != nullptr) {
      Z3_del_config(config_);
    }
    IrTestBase::TearDown();
  }

  bool AreMutuallyExclusive(Z3_ast cond1, Z3_ast cond2) {
    Z3_solver solver = Z3_mk_solver(ctx_);
    Z3_solver_inc_ref(ctx_, solver);
    Z3_solver_assert(ctx_, solver, cond1);
    Z3_solver_assert(ctx_, solver, cond2);
    Z3_lbool result = Z3_solver_check(ctx_, solver);
    Z3_solver_dec_ref(ctx_, solver);
    return result == Z3_L_FALSE;
  }

  Z3_config config_ = nullptr;
  Z3_context ctx_ = nullptr;
};

TEST_F(SymExEngineTest, InitializesWithDefaultOptions) {
  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  EXPECT_EQ(engine.options().max_paths, 1000);
}

TEST_F(SymExEngineTest, RespectsMaxPathsLimit) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue sel = fb.Param("sel", p.GetBitsType(3));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  BValue c = fb.Param("c", p.GetBitsType(8));
  BValue def = fb.Param("def", p.GetBitsType(8));
  fb.PrioritySelect(sel, {a, b, c}, def);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  SymExOptions options;
  options.max_paths = 2;
  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  EXPECT_THAT(paths, SizeIs(2));
}

TEST_F(SymExEngineTest, NoSelectsSinglePath) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue a = fb.Param("a", p.GetBitsType(32));
  BValue b = fb.Param("b", p.GetBitsType(32));
  fb.Add(a, b);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(1));
  EXPECT_THAT(paths[0].branch_decisions, IsEmpty());
}

TEST_F(SymExEngineTest, SingleSelectTwoPaths) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue sel = fb.Param("sel", p.GetBitsType(1));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  fb.Select(sel, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(2));
  EXPECT_THAT(paths,
              ElementsAre(SymbolicPathIs(0, false), SymbolicPathIs(1, false)));
  EXPECT_TRUE(
      AreMutuallyExclusive(paths[0].path_condition, paths[1].path_condition));
}

TEST_F(SymExEngineTest, PrioritySelectFourPaths) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue sel = fb.Param("sel", p.GetBitsType(3));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  BValue c = fb.Param("c", p.GetBitsType(8));
  BValue def = fb.Param("def", p.GetBitsType(8));
  fb.PrioritySelect(sel, {a, b, c}, def);
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(4));
  EXPECT_THAT(paths[0].branch_decisions,
              ElementsAre(BranchDecisionIs(0, false)));
  EXPECT_THAT(paths[1].branch_decisions,
              ElementsAre(BranchDecisionIs(1, false)));
  EXPECT_THAT(paths[2].branch_decisions,
              ElementsAre(BranchDecisionIs(2, false)));
  EXPECT_THAT(paths[3].branch_decisions,
              ElementsAre(BranchDecisionIs(3, true)));
}

TEST_F(SymExEngineTest, CascadedMuxesFourPaths) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue s0 = fb.Param("s0", p.GetBitsType(1));
  BValue s1 = fb.Param("s1", p.GetBitsType(1));
  BValue v0 = fb.Literal(UBits(0, 8));
  BValue v1 = fb.Literal(UBits(1, 8));
  BValue v2 = fb.Literal(UBits(2, 8));
  BValue v3 = fb.Literal(UBits(3, 8));
  BValue m0 = fb.Select(s0, {v0, v1});
  BValue m1 = fb.Select(s0, {v2, v3});
  fb.Select(s1, {m0, m1});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  ASSERT_THAT(paths, SizeIs(4));
  for (size_t i = 0; i < paths.size(); ++i) {
    for (size_t j = i + 1; j < paths.size(); ++j) {
      EXPECT_TRUE(AreMutuallyExclusive(paths[i].path_condition,
                                       paths[j].path_condition));
    }
  }
}

TEST_F(SymExEngineTest, PrunesInfeasibleBranchDecisions) {
  Package p(TestName());
  FunctionBuilder fb(TestName(), &p);
  BValue sel = fb.Param("sel", p.GetBitsType(1));
  BValue a = fb.Param("a", p.GetBitsType(8));
  BValue b = fb.Param("b", p.GetBitsType(8));
  // Create an unsatisfiable branch condition: constrained_sel is forced to 0,
  // so branch 1 (constrained_sel == 1) is unsatisfiable.
  BValue zero = fb.Literal(UBits(0, 1));
  BValue constrained_sel = fb.And(sel, zero);
  fb.Select(constrained_sel, {a, b});
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, fb.Build());

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine, SymExEngine::Create(ctx_));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  // Branch 1 is pruned by incremental solver push/pop, leaving 1 path.
  ASSERT_THAT(paths, SizeIs(1));
  EXPECT_THAT(paths[0].branch_decisions,
              ElementsAre(BranchDecisionIs(0, false)));
}

}  // namespace
}  // namespace xls::solvers::symex
