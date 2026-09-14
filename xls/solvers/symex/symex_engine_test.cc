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
#include "absl/status/status.h"
#include "xls/common/status/matchers.h"
#include "xls/ir/bits.h"
#include "xls/ir/function.h"
#include "xls/ir/function_builder.h"
#include "xls/ir/ir_test_base.h"
#include "xls/ir/package.h"
#include "xls/ir/value.h"
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

TEST_F(SymExEngineTest, ConcolicExecutionPrunesIncompatibleBranches) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn test_fn(sel: bits[2], a: bits[32], b: bits[32]) -> bits[32] {
      c0: bits[32] = literal(value=10)
      c1: bits[32] = literal(value=20)
      c2: bits[32] = literal(value=30)
      ret mux: bits[32] = sel(sel, cases=[c0, c1, c2], default=a)
    }
  )",
                                                        p.get()));

  SymExOptions options;
  options.concrete_inputs.BindParam("sel", Value(UBits(1, 2)));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  // Only the path corresponding to sel == 1 should be feasible.
  EXPECT_EQ(paths.size(), 1);
  EXPECT_EQ(paths[0].branch_decisions.size(), 1);
  EXPECT_EQ(paths[0].branch_decisions[0].arm_index, 1);
}

TEST_F(SymExEngineTest, ConcolicExecutionRejectsMismatchedBitwidth) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn test_fn(sel: bits[2], a: bits[32]) -> bits[32] {
      c0: bits[32] = literal(value=10)
      ret mux: bits[32] = sel(sel, cases=[c0], default=a)
    }
  )",
                                                        p.get()));

  SymExOptions options;
  // sel is bits[2], but binding bits[4] should return InvalidArgument.
  options.concrete_inputs.BindParam("sel", Value(UBits(1, 4)));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  EXPECT_THAT(
      engine.ExplorePaths(fn).status(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             testing::HasSubstr("bitwidth mismatch")));
}

TEST_F(SymExEngineTest,
       ConcolicExecutionPrunesViaIntermediateSelectorComputation) {
  auto p = CreatePackage();
  XLS_ASSERT_OK_AND_ASSIGN(Function * fn, ParseFunction(R"(
    fn test_fn(x: bits[32], y: bits[32]) -> bits[32] {
      two: bits[32] = literal(value=2)
      zero: bits[32] = literal(value=0)
      five: bits[32] = literal(value=5)
      eight: bits[32] = literal(value=8)
      mod_x: bits[32] = umod(x, two)
      cond_x: bits[1] = eq(mod_x, zero)
      real_x: bits[32] = sel(cond_x, cases=[five, x])
      cond_y: bits[1] = eq(y, real_x)
      real_y: bits[32] = sel(cond_y, cases=[y, eight])
      ret sum: bits[32] = add(real_x, real_y)
    }
  )",
                                                        p.get()));

  // Without concolic constraint, there are 4 paths (2 branches at each of 2
  // muxes).
  {
    XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                             SymExEngine::Create(ctx_, SymExOptions()));
    XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> all_paths,
                             engine.ExplorePaths(fn));
    EXPECT_THAT(all_paths, SizeIs(4));
  }

  // Restricting x = 3 forces mod_x = 1, pruning the cond_x == 1 branch.
  // Exactly 2 feasible paths remain for y.
  SymExOptions options;
  options.concrete_inputs.BindParam("x", Value(UBits(3, 32)));

  XLS_ASSERT_OK_AND_ASSIGN(SymExEngine engine,
                           SymExEngine::Create(ctx_, options));
  XLS_ASSERT_OK_AND_ASSIGN(std::vector<SymbolicPath> paths,
                           engine.ExplorePaths(fn));

  EXPECT_THAT(paths, SizeIs(2));
  for (const SymbolicPath& path : paths) {
    ASSERT_THAT(path.branch_decisions, SizeIs(2));
    // The first decision (mux for real_x) must be arm 0 (five).
    EXPECT_EQ(path.branch_decisions[0].arm_index, 0);
  }
}

}  // namespace
}  // namespace xls::solvers::symex
